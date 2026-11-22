import Darwin
import Foundation
import OnnxRuntimeBindings
import os

public enum MemoryMonitor {
    private static let bytesPerMB: Double = 1024.0 * 1024.0

    /// Exact physical memory footprint in MB (matches Xcode's Memory Gauge & iPadOS Jetsam).
    public static func physicalFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let kerr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / bytesPerMB
    }

    /// Remaining app RAM headroom in MB before iPadOS Jetsam terminates the process.
    public static func availableHeadroomMB() -> Double {
        Double(os_proc_available_memory()) / bytesPerMB
    }

    public static func fileSizeMB(_ url: URL) -> Double {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        let bytes = (attrs?[.size] as? NSNumber)?.doubleValue ?? 0
        return bytes / bytesPerMB
    }
}

@MainActor
public final class InkTeXEngine: ObservableObject {
    public enum Status: Equatable {
        case loading
        case ready
        case missingModel
        case failed(String)
    }

    @Published public private(set) var status: Status = .loading
    @Published public private(set) var availableBundles: [LocatedModelBundle] = []
    @Published public private(set) var loadedBundle: LocatedModelBundle?
    @Published public private(set) var hasR2LScorer: Bool = false
    @Published public private(set) var supportsV2FastDecode: Bool = false
    @Published public private(set) var isRecognizing: Bool = false
    @Published public private(set) var memorySnapshot: MemorySnapshot = .empty
    @Published public var useCoreMLForVision: Bool = true
    @Published public var liveTwoPhase: Bool = true
    @Published public var showLiveRamInBar: Bool = true
    @Published public var beamWidth: Int = 4
    @Published public var r2lWeight: Double = 0.5

    private var preferredBundleId: String? = nil
    private var peakFootprintMB: Double = 0
    private let workerQueue = DispatchQueue(label: "com.wosca.inktex.inference", qos: .userInitiated)
    private var runtime: LoadedRuntime?

    private final class EncodedMemoryCache {
        let fingerprint: UInt64
        let modality: ModalityMode
        let strip: StripRasterizer.Output
        let strokeCount: Int
        let pointCount: Int
        let strokeMemLen: Int
        let visionMemLen: Int
        let totalMemLen: Int
        let memTensor: ORTValue
        let memMaskVal: ORTValue
        let mkVal: ORTValue
        let mvVal: ORTValue

        init(
            fingerprint: UInt64,
            modality: ModalityMode,
            strip: StripRasterizer.Output,
            strokeCount: Int,
            pointCount: Int,
            strokeMemLen: Int,
            visionMemLen: Int,
            totalMemLen: Int,
            memTensor: ORTValue,
            memMaskVal: ORTValue,
            mkVal: ORTValue,
            mvVal: ORTValue
        ) {
            self.fingerprint = fingerprint
            self.modality = modality
            self.strip = strip
            self.strokeCount = strokeCount
            self.pointCount = pointCount
            self.strokeMemLen = strokeMemLen
            self.visionMemLen = visionMemLen
            self.totalMemLen = totalMemLen
            self.memTensor = memTensor
            self.memMaskVal = memMaskVal
            self.mkVal = mkVal
            self.mvVal = mvVal
        }
    }

    private final class LoadedRuntime {
        let env: ORTEnv
        let bundle: LocatedModelBundle
        let tokenizer: LatexTokenizer
        let visionSession: ORTSession
        let strokeSession: ORTSession
        let memoryProjSession: ORTSession
        let decoderStepSession: ORTSession
        let r2lScorerSession: ORTSession?
        let isV2Decoder: Bool
        let baselineMB: Double
        let memoryComponents: [MemoryComponent]

        // Incremental caches (accessed strictly on workerQueue)
        var lastPrefixBodyIds: [Int] = []
        var lastEncoded: EncodedMemoryCache? = nil

        init(bundle: LocatedModelBundle, useCoreMLForVision: Bool) throws {
            self.bundle = bundle
            self.tokenizer = LatexTokenizer(manifest: bundle.manifest)
            self.isV2Decoder = (bundle.manifest.supportsPrefixVerify == true) || (bundle.manifest.supportsBatchedDecoder == true)
            self.env = try ORTEnv(loggingLevel: .warning)

            let visionOpts = try Self.makeOptions(useCoreML: useCoreMLForVision, threads: 4)
            let cpuOpts = try Self.makeOptions(useCoreML: false, threads: 4)

            var components: [MemoryComponent] = []
            let base = MemoryMonitor.physicalFootprintMB()
            self.baselineMB = base

            var before = base
            self.visionSession = try ORTSession(env: env, modelPath: bundle.visionEncoderURL.path, sessionOptions: visionOpts)
            var after = MemoryMonitor.physicalFootprintMB()
            components.append(MemoryComponent(
                id: "vision",
                name: useCoreMLForVision ? "Vision Encoder (CoreML)" : "Vision Encoder (CPU)",
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.visionEncoderURL),
                colorRGB: (0.22, 0.74, 0.97)
            ))

            before = after
            self.strokeSession = try ORTSession(env: env, modelPath: bundle.strokeEncoderURL.path, sessionOptions: cpuOpts)
            after = MemoryMonitor.physicalFootprintMB()
            components.append(MemoryComponent(
                id: "stroke",
                name: "Stroke Encoder (8L)",
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.strokeEncoderURL),
                colorRGB: (0.20, 0.83, 0.60)
            ))

            before = after
            self.memoryProjSession = try ORTSession(env: env, modelPath: bundle.memoryProjURL.path, sessionOptions: cpuOpts)
            after = MemoryMonitor.physicalFootprintMB()
            components.append(MemoryComponent(
                id: "memproj",
                name: "Memory Projection",
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.memoryProjURL),
                colorRGB: (0.98, 0.75, 0.14)
            ))

            before = after
            self.decoderStepSession = try ORTSession(env: env, modelPath: bundle.decoderStepURL.path, sessionOptions: cpuOpts)
            after = MemoryMonitor.physicalFootprintMB()
            components.append(MemoryComponent(
                id: "decoder",
                name: "L2R Decoder (6L KV-Cache)",
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.decoderStepURL),
                colorRGB: (0.75, 0.52, 0.99)
            ))

            if let r2lURL = bundle.r2lScorerURL {
                before = after
                self.r2lScorerSession = try ORTSession(env: env, modelPath: r2lURL.path, sessionOptions: cpuOpts)
                after = MemoryMonitor.physicalFootprintMB()
                components.append(MemoryComponent(
                    id: "r2l",
                    name: "R2L Reranker (3L)",
                    ramMB: max(0, after - before),
                    diskMB: MemoryMonitor.fileSizeMB(r2lURL),
                    colorRGB: (0.96, 0.45, 0.71)
                ))
            } else {
                self.r2lScorerSession = nil
            }
            self.memoryComponents = components
        }

        private static func makeOptions(useCoreML: Bool, threads: Int32) throws -> ORTSessionOptions {
            let opts = try ORTSessionOptions()
            try opts.setGraphOptimizationLevel(.all)
            try opts.setIntraOpNumThreads(threads)
            if useCoreML && ORTIsCoreMLExecutionProviderAvailable() {
                let coreML = ORTCoreMLExecutionProviderOptions()
                coreML.enableOnSubgraphs = true
                try? opts.appendCoreMLExecutionProvider(with: coreML)
            }
            return opts
        }
    }

    private struct BatchedBeam {
        var bodyIds: [Int]
        var decodedTokens: [DecodedToken]
        var sumLogProb: Double
        var stepCount: Int
        var isFinished: Bool
        var lastTokenId: Int64
        // Per-beam state buffers for batched stacking ([L * H * T * D], [H * M])
        var kvLen: Int
        var kData: [Float]
        var vData: [Float]
        var coverageData: [Float]
        var prevWeightsData: [Float]

        var rankingScore: Double {
            sumLogProb / pow(Double(max(1, stepCount)), 0.7)
        }

        var avgLogProb: Double {
            sumLogProb / Double(max(1, stepCount))
        }
    }

    private struct LegacyBeam {
        var bodyIds: [Int]
        var decodedTokens: [DecodedToken]
        var sumLogProb: Double
        var stepCount: Int
        var isFinished: Bool
        var lastTokenId: Int64
        var pastK: ORTValue
        var pastV: ORTValue
        var coverage: ORTValue
        var prevWeights: ORTValue

        var rankingScore: Double {
            sumLogProb / pow(Double(max(1, stepCount)), 0.7)
        }

        var avgLogProb: Double {
            sumLogProb / Double(max(1, stepCount))
        }
    }

    public init() {
        reloadModel()
    }

    public func clearCaches() {
        let rt = runtime
        workerQueue.async {
            rt?.lastPrefixBodyIds = []
            rt?.lastEncoded = nil
        }
    }

    public func selectBundle(_ bundle: LocatedModelBundle) {
        preferredBundleId = bundle.id
        reloadModel()
    }

    public func refreshMemory(lastInferenceDeltaMB: Double? = nil) {
        let current = MemoryMonitor.physicalFootprintMB()
        peakFootprintMB = max(peakFootprintMB, current)
        let headroom = MemoryMonitor.availableHeadroomMB()
        let delta = lastInferenceDeltaMB ?? memorySnapshot.lastInferenceDeltaMB
        if let rt = runtime {
            memorySnapshot = MemorySnapshot(
                baselineMB: rt.baselineMB,
                components: rt.memoryComponents,
                currentFootprintMB: current,
                peakFootprintMB: peakFootprintMB,
                lastInferenceDeltaMB: delta,
                availableHeadroomMB: headroom
            )
        } else {
            memorySnapshot = MemorySnapshot(
                baselineMB: current,
                components: [],
                currentFootprintMB: current,
                peakFootprintMB: peakFootprintMB,
                lastInferenceDeltaMB: 0,
                availableHeadroomMB: headroom
            )
        }
    }

    public func reloadModel() {
        status = .loading
        let useCoreML = useCoreMLForVision
        let preferred = preferredBundleId
        // Release old runtime first so baseline RAM reflects the clean app footprint
        runtime = nil
        workerQueue.async { [weak self] in
            guard let self else { return }
            let allBundles = ModelBundleLocator.locateAll()
            let chosen = allBundles.first(where: { $0.id == preferred }) ?? allBundles.first
            guard let bundle = chosen else {
                let current = MemoryMonitor.physicalFootprintMB()
                let headroom = MemoryMonitor.availableHeadroomMB()
                Task { @MainActor in
                    self.availableBundles = allBundles
                    self.runtime = nil
                    self.loadedBundle = nil
                    self.hasR2LScorer = false
                    self.supportsV2FastDecode = false
                    self.peakFootprintMB = max(self.peakFootprintMB, current)
                    self.memorySnapshot = MemorySnapshot(
                        baselineMB: current,
                        components: [],
                        currentFootprintMB: current,
                        peakFootprintMB: self.peakFootprintMB,
                        lastInferenceDeltaMB: 0,
                        availableHeadroomMB: headroom
                    )
                    self.status = .missingModel
                }
                return
            }
            do {
                let loaded = try LoadedRuntime(bundle: bundle, useCoreMLForVision: useCoreML)
                let hasR2L = loaded.r2lScorerSession != nil
                let isV2 = loaded.isV2Decoder
                let afterLoad = MemoryMonitor.physicalFootprintMB()
                let headroom = MemoryMonitor.availableHeadroomMB()
                Task { @MainActor in
                    self.availableBundles = allBundles
                    self.runtime = loaded
                    self.loadedBundle = bundle
                    self.preferredBundleId = bundle.id
                    self.hasR2LScorer = hasR2L
                    self.supportsV2FastDecode = isV2
                    self.peakFootprintMB = max(self.peakFootprintMB, afterLoad)
                    self.memorySnapshot = MemorySnapshot(
                        baselineMB: loaded.baselineMB,
                        components: loaded.memoryComponents,
                        currentFootprintMB: afterLoad,
                        peakFootprintMB: self.peakFootprintMB,
                        lastInferenceDeltaMB: 0,
                        availableHeadroomMB: headroom
                    )
                    self.status = .ready
                }
            } catch {
                Task { @MainActor in
                    self.availableBundles = allBundles
                    self.runtime = nil
                    self.loadedBundle = bundle
                    self.hasR2LScorer = false
                    self.supportsV2FastDecode = false
                    self.status = .failed(error.localizedDescription)
                }
            }
        }
    }

    public func recognize(
        strokes: [RawStroke],
        modality: ModalityMode,
        decodingMode: DecodingMode = .beamR2L,
        maxTokens: Int = 200
    ) async throws -> RecognitionResult {
        guard let runtime else {
            throw NSError(
                domain: "InkTeX.Engine",
                code: 100,
                userInfo: [NSLocalizedDescriptionKey: "Model is not loaded yet."]
            )
        }
        isRecognizing = true
        defer { isRecognizing = false }

        let width = beamWidth
        let weight = r2lWeight

        let (result, inferDeltaMB): (RecognitionResult, Double) = try await withCheckedThrowingContinuation { continuation in
            workerQueue.async {
                let beforeMB = MemoryMonitor.physicalFootprintMB()
                do {
                    let res = try Self.runInference(
                        runtime: runtime,
                        strokes: strokes,
                        modality: modality,
                        decodingMode: decodingMode,
                        beamWidth: width,
                        r2lWeight: weight,
                        maxTokens: maxTokens
                    )
                    let afterMB = MemoryMonitor.physicalFootprintMB()
                    continuation.resume(returning: (res, max(0, afterMB - beforeMB)))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
        refreshMemory(lastInferenceDeltaMB: inferDeltaMB)
        return result
    }

    // MARK: - Core Inference Pipeline

    private static func runInference(
        runtime: LoadedRuntime,
        strokes: [RawStroke],
        modality: ModalityMode,
        decodingMode: DecodingMode,
        beamWidth: Int,
        r2lWeight: Double,
        maxTokens: Int
    ) throws -> RecognitionResult {
        let totalStart = CFAbsoluteTimeGetCurrent()
        let manifest = runtime.bundle.manifest

        // 1. Encode canvas (or reuse exact cached memory between Phase 1 Greedy & Phase 2 Beam+R2L!)
        let fp = fingerprintStrokes(strokes)
        let encoded: EncodedMemoryCache
        let encodeMs: Double
        let encoderCacheHit: Bool

        if let cached = runtime.lastEncoded, cached.fingerprint == fp, cached.modality == modality {
            encoded = cached
            encodeMs = 0
            encoderCacheHit = true
        } else {
            let encodeStart = CFAbsoluteTimeGetCurrent()
            encoded = try encodeCanvas(runtime: runtime, strokes: strokes, modality: modality, fingerprint: fp)
            encodeMs = (CFAbsoluteTimeGetCurrent() - encodeStart) * 1000.0
            runtime.lastEncoded = encoded
            encoderCacheHit = false
        }

        // 2. Decode (Prefix-Verified Greedy, Batched Beam Search, or v1 fallback)
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let tokenLimit = min(maxTokens, manifest.maxTokens - 1)
        let specialSet: Set<Int> = [manifest.padId, manifest.bosId, manifest.eosId, manifest.unkId]

        struct CandidateSeq {
            var bodyIds: [Int]
            var decodedTokens: [DecodedToken]
            var avgLogProb: Double
            var rankingScore: Double
            var isFinished: Bool
        }

        var candidates: [CandidateSeq] = []
        var prefixVerifiedCount = 0

        if decodingMode == .greedy {
            if runtime.isV2Decoder && !runtime.lastPrefixBodyIds.isEmpty {
                let (bodyIds, toks, avgLP, finished, verified) = try decodePrefixVerifiedGreedy(
                    runtime: runtime,
                    encoded: encoded,
                    prevGuess: runtime.lastPrefixBodyIds,
                    tokenLimit: tokenLimit,
                    specialSet: specialSet
                )
                prefixVerifiedCount = verified
                candidates = [CandidateSeq(bodyIds: bodyIds, decodedTokens: toks, avgLogProb: avgLP, rankingScore: avgLP, isFinished: finished)]
            } else {
                let (bodyIds, toks, avgLP, finished) = try decodeSingleGreedy(
                    runtime: runtime,
                    encoded: encoded,
                    tokenLimit: tokenLimit,
                    specialSet: specialSet
                )
                candidates = [CandidateSeq(bodyIds: bodyIds, decodedTokens: toks, avgLogProb: avgLP, rankingScore: avgLP, isFinished: finished)]
            }
        } else {
            let effectiveBeam = max(2, min(8, beamWidth))
            if runtime.isV2Decoder {
                let beams = try decodeBatchedBeam(
                    runtime: runtime,
                    encoded: encoded,
                    beamWidth: effectiveBeam,
                    tokenLimit: tokenLimit,
                    specialSet: specialSet
                )
                candidates = beams.map {
                    CandidateSeq(bodyIds: $0.bodyIds, decodedTokens: $0.decodedTokens, avgLogProb: $0.avgLogProb, rankingScore: $0.rankingScore, isFinished: $0.isFinished)
                }
            } else {
                let beams = try decodeLegacyBeam(
                    runtime: runtime,
                    encoded: encoded,
                    beamWidth: effectiveBeam,
                    tokenLimit: tokenLimit,
                    specialSet: specialSet
                )
                candidates = beams.map {
                    CandidateSeq(bodyIds: $0.bodyIds, decodedTokens: $0.decodedTokens, avgLogProb: $0.avgLogProb, rankingScore: $0.rankingScore, isFinished: $0.isFinished)
                }
            }
        }

        // Deduplicate by decoded LaTeX keeping best L2R score
        candidates.sort { $0.rankingScore > $1.rankingScore }
        var uniqueCandidates: [CandidateSeq] = []
        var seenLatex: Set<String> = []
        for c in candidates {
            let tex = LatexTokenizer.detokenize(c.decodedTokens.map(\.text))
            if seenLatex.insert(tex).inserted {
                uniqueCandidates.append(c)
            }
        }
        if uniqueCandidates.isEmpty, let first = candidates.first {
            uniqueCandidates = [first]
        }
        let decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000.0

        // 3. Optional R2L Bidirectional Reranking (1 batched pass)
        var r2lMs: Double = 0
        var r2lScores: [Double]? = nil
        var r2lUsed = false

        if decodingMode == .beamR2L, let r2lSession = runtime.r2lScorerSession, !uniqueCandidates.isEmpty {
            let r2lStart = CFAbsoluteTimeGetCurrent()
            let r2lSeqs: [[Int64]] = uniqueCandidates.map { c in
                var seq: [Int64] = [Int64(manifest.bosId)]
                seq.reserveCapacity(c.bodyIds.count + 2)
                for id in c.bodyIds.reversed() {
                    seq.append(Int64(id))
                }
                seq.append(Int64(manifest.eosId))
                return seq
            }
            let maxLen = max(2, r2lSeqs.map(\.count).max() ?? 2)
            let batchCount = r2lSeqs.count
            var tokFlat = [Int64](repeating: Int64(manifest.padId), count: batchCount * maxLen)
            var padFlat = [UInt8](repeating: 1, count: batchCount * maxLen)
            for (row, seq) in r2lSeqs.enumerated() {
                let base = row * maxLen
                for (col, id) in seq.enumerated() {
                    tokFlat[base + col] = id
                    padFlat[base + col] = 0
                }
            }
            let tokTensor = try makeInt64Tensor(tokFlat, shape: [batchCount, maxLen])
            let padTensor = try makeUInt8Tensor(padFlat, shape: [batchCount, maxLen])
            let r2lOut = try r2lSession.run(
                withInputs: [
                    "r2l_tokens": tokTensor,
                    "r2l_padding_mask": padTensor,
                    "memory": encoded.memTensor,
                    "memory_mask": encoded.memMaskVal,
                ],
                outputNames: ["seq_log_prob", "avg_log_prob"],
                runOptions: nil
            )
            if let avgVal = r2lOut["avg_log_prob"] {
                r2lScores = try extractFloats(avgVal).map { Double($0) }
                r2lUsed = true
            }
            r2lMs = (CFAbsoluteTimeGetCurrent() - r2lStart) * 1000.0
        }

        // 4. Rank & pick winner
        var indexedScores: [(index: Int, l2r: Double, r2l: Double?, combined: Double)] = uniqueCandidates.enumerated().map { idx, c in
            let l2r = c.avgLogProb
            let r2l = r2lScores?[idx]
            let combined = r2l != nil ? (l2r + r2lWeight * r2l!) : l2r
            return (idx, l2r, r2l, combined)
        }
        indexedScores.sort { $0.combined > $1.combined }

        let winnerIdx = indexedScores.first?.index ?? 0
        let r2lChangedWinner = r2lUsed && (winnerIdx != 0)
        let winner = uniqueCandidates[winnerIdx]

        // Save winning tokens for next stroke's parallel prefix verification!
        runtime.lastPrefixBodyIds = winner.bodyIds

        let beamSummaries: [BeamCandidate] = indexedScores.enumerated().map { rank, item in
            let c = uniqueCandidates[item.index]
            let tex = LatexTokenizer.detokenize(c.decodedTokens.map(\.text))
            return BeamCandidate(
                id: rank,
                latex: tex,
                l2rScore: item.l2r,
                r2lScore: item.r2l,
                combinedScore: item.combined,
                isWinner: rank == 0,
                wasPromotedByR2L: rank == 0 && r2lChangedWinner
            )
        }

        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000.0
        let latex = LatexTokenizer.detokenize(winner.decodedTokens.map(\.text))
        let overallConf: Float
        if winner.decodedTokens.isEmpty {
            overallConf = 0
        } else {
            let meanLog = winner.decodedTokens.reduce(0.0) { $0 + log(Double(max($1.confidence, 1e-6))) } / Double(winner.decodedTokens.count)
            overallConf = Float(exp(meanLog))
        }

        let gridH = encoded.visionMemLen > 0 ? max(1, encoded.strip.height / 16) : 0
        let gridW = gridH > 0 ? (encoded.visionMemLen / gridH) : 0

        return RecognitionResult(
            latex: latex,
            decodedTokens: winner.decodedTokens,
            overallConfidence: overallConf,
            totalMs: totalMs,
            encodeMs: encodeMs,
            decodeMs: decodeMs,
            r2lMs: r2lMs,
            strokeCount: encoded.strokeCount,
            pointCount: encoded.pointCount,
            stripHeight: encoded.strip.height,
            stripWidth: encoded.strip.width,
            visionGridW: gridW,
            visionGridH: gridH,
            stripLumBytes: encoded.strip.lumBytes,
            stripOrderBytes: encoded.strip.orderBytes,
            modality: modality,
            decodingMode: decodingMode,
            r2lUsed: r2lUsed,
            r2lChangedWinner: r2lChangedWinner,
            prefixVerifiedCount: prefixVerifiedCount,
            encoderCacheHit: encoderCacheHit,
            beams: beamSummaries,
            truncated: !winner.isFinished
        )
    }

    // MARK: - Canvas Encoding

    private static func encodeCanvas(
        runtime: LoadedRuntime,
        strokes: [RawStroke],
        modality: ModalityMode,
        fingerprint: UInt64
    ) throws -> EncodedMemoryCache {
        let manifest = runtime.bundle.manifest
        let strip = StripRasterizer.render(strokes: strokes, config: manifest.vision)
        let strokeInput = StrokePreprocessor.preprocess(strokes: strokes, config: manifest.preprocessing)

        var strokeMemFloats: [Float]? = nil
        var strokeMaskBytes: [UInt8]? = nil
        var strokeMemLen = 0

        if (modality == .bimodal || modality == .stroke), let si = strokeInput {
            let featVal = try makeFloatTensor(si.features, shape: [1, si.pointCount, manifest.inputDim])
            let maskVal = try makeUInt8Tensor(si.paddingMask, shape: [1, si.pointCount])
            let outputs = try runtime.strokeSession.run(
                withInputs: [
                    "stroke_features": featVal,
                    "stroke_padding_mask": maskVal,
                ],
                outputNames: ["stroke_memory", "stroke_mask"],
                runOptions: nil
            )
            guard let sm = outputs["stroke_memory"], let smask = outputs["stroke_mask"] else {
                throw inferenceError("Missing outputs from stroke_encoder.onnx")
            }
            let smShape = try sm.tensorTypeAndShapeInfo().shape.map(\.intValue)
            strokeMemLen = smShape[1]
            strokeMemFloats = try extractFloats(sm)
            strokeMaskBytes = try extractUInt8s(smask)
        }

        var visionMemFloats: [Float]? = nil
        var visionMaskBytes: [UInt8]? = nil
        var visionMemLen = 0

        if modality == .bimodal || modality == .vision {
            let imgVal = try makeFloatTensor(strip.tensor, shape: [1, manifest.vision.channels, strip.height, strip.width])
            let widthVal = try makeInt64Tensor([Int64(strip.usedWidth)], shape: [1])
            let outputs = try runtime.visionSession.run(
                withInputs: [
                    "images": imgVal,
                    "image_widths": widthVal,
                ],
                outputNames: ["vision_memory", "vision_mask"],
                runOptions: nil
            )
            guard let vm = outputs["vision_memory"], let vmask = outputs["vision_mask"] else {
                throw inferenceError("Missing outputs from vision_encoder.onnx")
            }
            let vmShape = try vm.tensorTypeAndShapeInfo().shape.map(\.intValue)
            visionMemLen = vmShape[1]
            visionMemFloats = try extractFloats(vm)
            visionMaskBytes = try extractUInt8s(vmask)
        }

        var combinedMem: [Float] = []
        var combinedMask: [UInt8] = []
        let totalMemLen = strokeMemLen + visionMemLen
        guard totalMemLen > 0 else {
            throw inferenceError("No valid encoder memory produced for input strokes.")
        }
        combinedMem.reserveCapacity(totalMemLen * manifest.dModel)
        combinedMask.reserveCapacity(totalMemLen)

        if let sm = strokeMemFloats, let smask = strokeMaskBytes {
            combinedMem.append(contentsOf: sm)
            combinedMask.append(contentsOf: smask)
        }
        if let vm = visionMemFloats, let vmask = visionMaskBytes {
            combinedMem.append(contentsOf: vm)
            combinedMask.append(contentsOf: vmask)
        }

        let memTensor = try makeFloatTensor(combinedMem, shape: [1, totalMemLen, manifest.dModel])
        let memMaskVal = try makeUInt8Tensor(combinedMask, shape: [1, totalMemLen])
        let projOut = try runtime.memoryProjSession.run(
            withInputs: ["memory": memTensor],
            outputNames: ["mk", "mv"],
            runOptions: nil
        )
        guard let mkVal = projOut["mk"], let mvVal = projOut["mv"] else {
            throw inferenceError("Missing outputs from memory_proj.onnx")
        }

        return EncodedMemoryCache(
            fingerprint: fingerprint,
            modality: modality,
            strip: strip,
            strokeCount: strokeInput?.strokeCount ?? strokes.count,
            pointCount: strokeInput?.pointCount ?? 0,
            strokeMemLen: strokeMemLen,
            visionMemLen: visionMemLen,
            totalMemLen: totalMemLen,
            memTensor: memTensor,
            memMaskVal: memMaskVal,
            mkVal: mkVal,
            mvVal: mvVal
        )
    }

    // MARK: - 1. Prefix-Verified Parallel Greedy Decoding (100% Exact Parity!)

    private static func decodePrefixVerifiedGreedy(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        prevGuess: [Int],
        tokenLimit: Int,
        specialSet: Set<Int>
    ) throws -> (bodyIds: [Int], tokens: [DecodedToken], avgLogProb: Double, isFinished: Bool, verifiedCount: Int) {
        let manifest = runtime.bundle.manifest
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let M = encoded.totalMemLen
        let V = manifest.vocabSize

        let cappedGuess = Array(prevGuess.prefix(max(1, tokenLimit - 1)))
        let S = cappedGuess.count + 1
        var inputIds: [Int64] = [Int64(manifest.bosId)]
        inputIds.reserveCapacity(S)
        for id in cappedGuess { inputIds.append(Int64(id)) }

        let initK = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        let initV = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        let initCov = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])
        let initPrev = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])

        let out = try runtime.decoderStepSession.run(
            withInputs: [
                "token": try makeInt64Tensor(inputIds, shape: [1, S]),
                "past_k": initK,
                "past_v": initV,
                "mk": encoded.mkVal,
                "mv": encoded.mvVal,
                "memory_mask": encoded.memMaskVal,
                "coverage": initCov,
                "prev_weights": initPrev,
            ],
            outputNames: ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
            runOptions: nil
        )
        guard let logitsVal = out["logits"],
              let newKVal = out["new_k"],
              let newVVal = out["new_v"],
              let newCovVal = out["new_coverage"],
              let newPrevVal = out["new_prev_weights"] else {
            throw inferenceError("Missing outputs from decoder_step.onnx during prefix verification")
        }

        let allLogits = try extractFloats(logitsVal)   // [1, S, V]
        let allPrevW = try extractFloats(newPrevVal)   // [1, H, S, M]

        var bodyIds: [Int] = []
        var decodedTokens: [DecodedToken] = []
        var sumLogProb: Double = 0
        var stepCount = 0
        var divergeIndex: Int? = nil
        var hitEOS = false
        var lastPredictedId: Int = manifest.bosId

        for i in 0..<S {
            let sliceStart = i * V
            let choice = argmaxLogSoftmax(allLogits, offset: sliceStart, count: V)
            sumLogProb += choice.logProb
            stepCount += 1
            lastPredictedId = choice.id

            if choice.id == manifest.eosId {
                hitEOS = true
                divergeIndex = nil
                break
            }

            if !specialSet.contains(choice.id) {
                let attn = encoded.visionMemLen > 0
                    ? extractVisionAttentionAtStep(allPrevW, numHeads: H, stepLen: S, stepIdx: i, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
                    : nil
                bodyIds.append(choice.id)
                decodedTokens.append(makeToken(stepIdx: i, tokenId: choice.id, prob: choice.prob, visionAttn: attn, manifest: manifest))
            }

            if i < cappedGuess.count && choice.id == cappedGuess[i] {
                continue
            } else {
                divergeIndex = i
                break
            }
        }

        let verifiedCount = max(0, stepCount - 1)

        // If we haven't hit EOS yet, slice KV-cache + coverage at divergeIndex and decode only the tail!
        if !hitEOS, let divI = divergeIndex, stepCount < tokenLimit {
            let fullK = try extractFloats(newKVal)     // [L, 1, H, 1 + S, D]
            let fullV = try extractFloats(newVVal)     // [L, 1, H, 1 + S, D]
            let fullCov = try extractFloats(newCovVal) // [1, H, S, M]

            let fullT = 1 + S
            let keepT = divI + 2 // dummy(0) + BOS(1) + verified tokens(2 ..< divI+2)
            var slicedK = [Float](repeating: 0, count: L * H * keepT * D)
            var slicedV = [Float](repeating: 0, count: L * H * keepT * D)
            let copyFloats = keepT * D

            for l in 0..<L {
                for h in 0..<H {
                    let srcBase = ((l * H + h) * fullT) * D
                    let dstBase = ((l * H + h) * keepT) * D
                    for j in 0..<copyFloats {
                        slicedK[dstBase + j] = fullK[srcBase + j]
                        slicedV[dstBase + j] = fullV[srcBase + j]
                    }
                }
            }

            var slicedCov = [Float](repeating: 0, count: H * M)
            var slicedPrev = [Float](repeating: 0, count: H * M)
            for h in 0..<H {
                let srcBase = (h * S + divI) * M
                let dstBase = h * M
                for m in 0..<M {
                    slicedCov[dstBase + m] = fullCov[srcBase + m]
                    slicedPrev[dstBase + m] = allPrevW[srcBase + m]
                }
            }

            var pastK = try makeFloatTensor(slicedK, shape: [L, 1, H, keepT, D])
            var pastV = try makeFloatTensor(slicedV, shape: [L, 1, H, keepT, D])
            var coverage = try makeFloatTensor(slicedCov, shape: [1, H, 1, M])
            var prevWeights = try makeFloatTensor(slicedPrev, shape: [1, H, 1, M])
            var curTok = Int64(lastPredictedId)

            while stepCount < tokenLimit {
                let stepOut = try runtime.decoderStepSession.run(
                    withInputs: [
                        "token": try makeInt64Tensor([curTok], shape: [1, 1]),
                        "past_k": pastK,
                        "past_v": pastV,
                        "mk": encoded.mkVal,
                        "mv": encoded.mvVal,
                        "memory_mask": encoded.memMaskVal,
                        "coverage": coverage,
                        "prev_weights": prevWeights,
                    ],
                    outputNames: ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
                    runOptions: nil
                )
                guard let lVal = stepOut["logits"],
                      let nk = stepOut["new_k"],
                      let nv = stepOut["new_v"],
                      let nc = stepOut["new_coverage"],
                      let np = stepOut["new_prev_weights"] else {
                    throw inferenceError("Missing outputs from decoder_step.onnx")
                }
                let logits = try extractFloats(lVal)
                let choice = argmaxLogSoftmax(logits, offset: 0, count: V)
                sumLogProb += choice.logProb
                let idx = stepCount
                stepCount += 1

                if choice.id == manifest.eosId {
                    hitEOS = true
                    break
                }
                if !specialSet.contains(choice.id) {
                    let attn = encoded.visionMemLen > 0
                        ? averageVisionAttention(weights: try extractFloats(np), numHeads: H, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
                        : nil
                    bodyIds.append(choice.id)
                    decodedTokens.append(makeToken(stepIdx: idx, tokenId: choice.id, prob: choice.prob, visionAttn: attn, manifest: manifest))
                }
                curTok = Int64(choice.id)
                pastK = nk
                pastV = nv
                coverage = nc
                prevWeights = np
            }
        }

        let avgLP = sumLogProb / Double(max(1, stepCount))
        return (bodyIds, decodedTokens, avgLP, hitEOS, verifiedCount)
    }

    // MARK: - 2. Standard Single-Step Greedy (first stroke or v1 bundle)

    private static func decodeSingleGreedy(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        tokenLimit: Int,
        specialSet: Set<Int>
    ) throws -> (bodyIds: [Int], tokens: [DecodedToken], avgLogProb: Double, isFinished: Bool) {
        let manifest = runtime.bundle.manifest
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let M = encoded.totalMemLen
        let V = manifest.vocabSize

        var pastK = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        var pastV = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        var coverage = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])
        var prevWeights = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])

        var curTok = Int64(manifest.bosId)
        var bodyIds: [Int] = []
        var decodedTokens: [DecodedToken] = []
        var sumLogProb: Double = 0
        var stepCount = 0
        var hitEOS = false

        for stepIdx in 0..<tokenLimit {
            let out = try runtime.decoderStepSession.run(
                withInputs: [
                    "token": try makeInt64Tensor([curTok], shape: [1, 1]),
                    "past_k": pastK,
                    "past_v": pastV,
                    "mk": encoded.mkVal,
                    "mv": encoded.mvVal,
                    "memory_mask": encoded.memMaskVal,
                    "coverage": coverage,
                    "prev_weights": prevWeights,
                ],
                outputNames: ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
                runOptions: nil
            )
            guard let lVal = out["logits"],
                  let nk = out["new_k"],
                  let nv = out["new_v"],
                  let nc = out["new_coverage"],
                  let np = out["new_prev_weights"] else {
                throw inferenceError("Missing outputs from decoder_step.onnx")
            }
            let logits = try extractFloats(lVal)
            let choice = argmaxLogSoftmax(logits, offset: 0, count: V)
            sumLogProb += choice.logProb
            stepCount += 1

            if choice.id == manifest.eosId {
                hitEOS = true
                break
            }
            if !specialSet.contains(choice.id) {
                let attn = encoded.visionMemLen > 0
                    ? averageVisionAttention(weights: try extractFloats(np), numHeads: H, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
                    : nil
                bodyIds.append(choice.id)
                decodedTokens.append(makeToken(stepIdx: stepIdx, tokenId: choice.id, prob: choice.prob, visionAttn: attn, manifest: manifest))
            }
            curTok = Int64(choice.id)
            pastK = nk
            pastV = nv
            coverage = nc
            prevWeights = np
        }

        return (bodyIds, decodedTokens, sumLogProb / Double(max(1, stepCount)), hitEOS)
    }

    // MARK: - 3. Batched Beam Search (All active beams in 1 ONNX call per step!)

    private static func decodeBatchedBeam(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        beamWidth: Int,
        tokenLimit: Int,
        specialSet: Set<Int>
    ) throws -> [BatchedBeam] {
        let manifest = runtime.bundle.manifest
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let M = encoded.totalMemLen
        let V = manifest.vocabSize

        // Step 0: batch=1 from BOS
        let initK = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        let initV = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        let initCov = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])
        let initPrev = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])

        let out0 = try runtime.decoderStepSession.run(
            withInputs: [
                "token": try makeInt64Tensor([Int64(manifest.bosId)], shape: [1, 1]),
                "past_k": initK,
                "past_v": initV,
                "mk": encoded.mkVal,
                "mv": encoded.mvVal,
                "memory_mask": encoded.memMaskVal,
                "coverage": initCov,
                "prev_weights": initPrev,
            ],
            outputNames: ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
            runOptions: nil
        )
        guard let l0Val = out0["logits"],
              let k0Val = out0["new_k"],
              let v0Val = out0["new_v"],
              let c0Val = out0["new_coverage"],
              let p0Val = out0["new_prev_weights"] else {
            throw inferenceError("Missing outputs from decoder_step.onnx")
        }

        let logits0 = try extractFloats(l0Val)
        let k0 = try extractFloats(k0Val) // [L, 1, H, 2, D]
        let v0 = try extractFloats(v0Val)
        let c0 = try extractFloats(c0Val) // [1, H, 1, M]
        let p0 = try extractFloats(p0Val)
        let attn0 = encoded.visionMemLen > 0
            ? averageVisionAttention(weights: p0, numHeads: H, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
            : nil

        let choices0 = topKLogSoftmax(logits0, offset: 0, count: V, k: beamWidth)
        var beams: [BatchedBeam] = choices0.map { ch in
            let fin = (ch.id == manifest.eosId)
            var body: [Int] = []
            var toks: [DecodedToken] = []
            if !fin && !specialSet.contains(ch.id) {
                body.append(ch.id)
                toks.append(makeToken(stepIdx: 0, tokenId: ch.id, prob: ch.prob, visionAttn: attn0, manifest: manifest))
            }
            return BatchedBeam(
                bodyIds: body,
                decodedTokens: toks,
                sumLogProb: ch.logProb,
                stepCount: 1,
                isFinished: fin,
                lastTokenId: Int64(ch.id),
                kvLen: 2,
                kData: k0,
                vData: v0,
                coverageData: c0,
                prevWeightsData: p0
            )
        }

        // Steps 1 ..< tokenLimit: stack all unfinished beams into 1 batched ONNX call!
        if tokenLimit > 1 {
            for stepIdx in 1..<tokenLimit {
                let active = beams.filter { !$0.isFinished }
                if active.isEmpty { break }

                let B = active.count
                let T = active[0].kvLen
                let perLayerBeamKV = H * T * D

                var tokBatch = [Int64](repeating: 0, count: B)
                var kBatch = [Float](repeating: 0, count: L * B * perLayerBeamKV)
                var vBatch = [Float](repeating: 0, count: L * B * perLayerBeamKV)
                var covBatch = [Float](repeating: 0, count: B * H * M)
                var prevBatch = [Float](repeating: 0, count: B * H * M)

                for (b, beam) in active.enumerated() {
                    tokBatch[b] = beam.lastTokenId
                    for l in 0..<L {
                        let srcOff = l * perLayerBeamKV
                        let dstOff = (l * B + b) * perLayerBeamKV
                        for j in 0..<perLayerBeamKV {
                            kBatch[dstOff + j] = beam.kData[srcOff + j]
                            vBatch[dstOff + j] = beam.vData[srcOff + j]
                        }
                    }
                    let covOff = b * H * M
                    for j in 0..<(H * M) {
                        covBatch[covOff + j] = beam.coverageData[j]
                        prevBatch[covOff + j] = beam.prevWeightsData[j]
                    }
                }

                let stepOut = try runtime.decoderStepSession.run(
                    withInputs: [
                        "token": try makeInt64Tensor(tokBatch, shape: [B, 1]),
                        "past_k": try makeFloatTensor(kBatch, shape: [L, B, H, T, D]),
                        "past_v": try makeFloatTensor(vBatch, shape: [L, B, H, T, D]),
                        "mk": encoded.mkVal,
                        "mv": encoded.mvVal,
                        "memory_mask": encoded.memMaskVal,
                        "coverage": try makeFloatTensor(covBatch, shape: [B, H, 1, M]),
                        "prev_weights": try makeFloatTensor(prevBatch, shape: [B, H, 1, M]),
                    ],
                    outputNames: ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
                    runOptions: nil
                )
                guard let lVal = stepOut["logits"],
                      let nkVal = stepOut["new_k"],
                      let nvVal = stepOut["new_v"],
                      let ncVal = stepOut["new_coverage"],
                      let npVal = stepOut["new_prev_weights"] else {
                    throw inferenceError("Missing outputs from batched decoder_step.onnx")
                }

                let allLogits = try extractFloats(lVal)   // [B, 1, V]
                let allK = try extractFloats(nkVal)       // [L, B, H, T+1, D]
                let allV = try extractFloats(nvVal)
                let allCov = try extractFloats(ncVal)     // [B, H, 1, M]
                let allPrev = try extractFloats(npVal)    // [B, H, 1, M]

                let nextT = T + 1
                let nextPerLayerBeamKV = H * nextT * D

                // First rank candidates without copying KV buffers for pruned beams!
                struct PendingCandidate {
                    let isAlreadyFinished: Bool
                    let existingFinishedBeam: BatchedBeam?
                    let activeBeamIdx: Int
                    let tokenId: Int
                    let prob: Float
                    let sumLogProb: Double
                    let stepCount: Int
                    var rankingScore: Double { sumLogProb / pow(Double(max(1, stepCount)), 0.7) }
                }

                var pending: [PendingCandidate] = []
                pending.reserveCapacity(beamWidth * beamWidth + beamWidth)
                for b in beams where b.isFinished {
                    pending.append(PendingCandidate(
                        isAlreadyFinished: true, existingFinishedBeam: b, activeBeamIdx: -1,
                        tokenId: manifest.eosId, prob: 1, sumLogProb: b.sumLogProb, stepCount: b.stepCount
                    ))
                }
                for (bIdx, beam) in active.enumerated() {
                    let choices = topKLogSoftmax(allLogits, offset: bIdx * V, count: V, k: beamWidth)
                    for ch in choices {
                        pending.append(PendingCandidate(
                            isAlreadyFinished: false, existingFinishedBeam: nil, activeBeamIdx: bIdx,
                            tokenId: ch.id, prob: ch.prob, sumLogProb: beam.sumLogProb + ch.logProb, stepCount: beam.stepCount + 1
                        ))
                    }
                }
                pending.sort { $0.rankingScore > $1.rankingScore }
                let survivors = pending.prefix(beamWidth)

                // Materialize KV buffers only for surviving active beams (at most beamWidth copies)
                var extractedK: [Int: [Float]] = [:]
                var extractedV: [Int: [Float]] = [:]
                var extractedCov: [Int: [Float]] = [:]
                var extractedPrev: [Int: [Float]] = [:]
                var extractedAttn: [Int: [Float]?] = [:]

                var nextBeams: [BatchedBeam] = []
                nextBeams.reserveCapacity(beamWidth)

                for item in survivors {
                    if item.isAlreadyFinished, let fin = item.existingFinishedBeam {
                        nextBeams.append(fin)
                        continue
                    }
                    let bIdx = item.activeBeamIdx
                    let parent = active[bIdx]
                    let fin = (item.tokenId == manifest.eosId)

                    if extractedAttn[bIdx] == nil {
                        let covOff = bIdx * H * M
                        let prevSlice = Array(allPrev[covOff ..< (covOff + H * M)])
                        extractedPrev[bIdx] = prevSlice
                        extractedCov[bIdx] = Array(allCov[covOff ..< (covOff + H * M)])
                        extractedAttn[bIdx] = encoded.visionMemLen > 0
                            ? averageVisionAttention(weights: prevSlice, numHeads: H, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
                            : nil
                    }
                    if !fin && extractedK[bIdx] == nil {
                        var bk = [Float](repeating: 0, count: L * nextPerLayerBeamKV)
                        var bv = [Float](repeating: 0, count: L * nextPerLayerBeamKV)
                        for l in 0..<L {
                            let srcOff = (l * B + bIdx) * nextPerLayerBeamKV
                            let dstOff = l * nextPerLayerBeamKV
                            for j in 0..<nextPerLayerBeamKV {
                                bk[dstOff + j] = allK[srcOff + j]
                                bv[dstOff + j] = allV[srcOff + j]
                            }
                        }
                        extractedK[bIdx] = bk
                        extractedV[bIdx] = bv
                    }

                    var body = parent.bodyIds
                    var toks = parent.decodedTokens
                    if !fin && !specialSet.contains(item.tokenId) {
                        body.append(item.tokenId)
                        toks.append(makeToken(stepIdx: stepIdx, tokenId: item.tokenId, prob: item.prob, visionAttn: extractedAttn[bIdx] ?? nil, manifest: manifest))
                    }

                    nextBeams.append(BatchedBeam(
                        bodyIds: body,
                        decodedTokens: toks,
                        sumLogProb: item.sumLogProb,
                        stepCount: item.stepCount,
                        isFinished: fin,
                        lastTokenId: Int64(item.tokenId),
                        kvLen: nextT,
                        kData: fin ? [] : (extractedK[bIdx] ?? []),
                        vData: fin ? [] : (extractedV[bIdx] ?? []),
                        coverageData: fin ? [] : (extractedCov[bIdx] ?? []),
                        prevWeightsData: fin ? [] : (extractedPrev[bIdx] ?? [])
                    ))
                }
                beams = nextBeams
            }
        }
        return beams
    }

    // MARK: - 4. Legacy v1 Bundle Beam Search (serial fallback if user hasn't updated model-bundle yet)

    private static func decodeLegacyBeam(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        beamWidth: Int,
        tokenLimit: Int,
        specialSet: Set<Int>
    ) throws -> [LegacyBeam] {
        let manifest = runtime.bundle.manifest
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let M = encoded.totalMemLen
        let V = manifest.vocabSize

        func runStep(_ tokId: Int64, _ k: ORTValue, _ v: ORTValue, _ c: ORTValue, _ p: ORTValue) throws -> ([Float], ORTValue, ORTValue, ORTValue, ORTValue) {
            let out = try runtime.decoderStepSession.run(
                withInputs: [
                    "token": try makeInt64Tensor([tokId], shape: [1, 1]),
                    "past_k": k, "past_v": v, "mk": encoded.mkVal, "mv": encoded.mvVal,
                    "memory_mask": encoded.memMaskVal, "coverage": c, "prev_weights": p,
                ],
                outputNames: ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
                runOptions: nil
            )
            guard let lv = out["logits"], let nk = out["new_k"], let nv = out["new_v"], let nc = out["new_coverage"], let np = out["new_prev_weights"] else {
                throw inferenceError("Missing outputs from legacy decoder_step.onnx")
            }
            return (try extractFloats(lv), nk, nv, nc, np)
        }

        let initK = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        let initV = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * 1 * D), shape: [L, 1, H, 1, D])
        let initCov = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])
        let initPrev = try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])

        let (l0, k0, v0, c0, p0) = try runStep(Int64(manifest.bosId), initK, initV, initCov, initPrev)
        let attn0 = encoded.visionMemLen > 0
            ? averageVisionAttention(weights: try extractFloats(p0), numHeads: H, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
            : nil
        var beams: [LegacyBeam] = topKLogSoftmax(l0, offset: 0, count: V, k: beamWidth).map { ch in
            let fin = ch.id == manifest.eosId
            var body: [Int] = []
            var toks: [DecodedToken] = []
            if !fin && !specialSet.contains(ch.id) {
                body.append(ch.id)
                toks.append(makeToken(stepIdx: 0, tokenId: ch.id, prob: ch.prob, visionAttn: attn0, manifest: manifest))
            }
            return LegacyBeam(bodyIds: body, decodedTokens: toks, sumLogProb: ch.logProb, stepCount: 1, isFinished: fin, lastTokenId: Int64(ch.id), pastK: k0, pastV: v0, coverage: c0, prevWeights: p0)
        }

        for stepIdx in 1..<tokenLimit {
            if beams.allSatisfy(\.isFinished) { break }
            var pool: [LegacyBeam] = []
            for b in beams {
                if b.isFinished { pool.append(b); continue }
                let (ls, nk, nv, nc, np) = try runStep(b.lastTokenId, b.pastK, b.pastV, b.coverage, b.prevWeights)
                let attn = encoded.visionMemLen > 0
                    ? averageVisionAttention(weights: try extractFloats(np), numHeads: H, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
                    : nil
                for ch in topKLogSoftmax(ls, offset: 0, count: V, k: beamWidth) {
                    let fin = ch.id == manifest.eosId
                    var body = b.bodyIds
                    var toks = b.decodedTokens
                    if !fin && !specialSet.contains(ch.id) {
                        body.append(ch.id)
                        toks.append(makeToken(stepIdx: stepIdx, tokenId: ch.id, prob: ch.prob, visionAttn: attn, manifest: manifest))
                    }
                    pool.append(LegacyBeam(bodyIds: body, decodedTokens: toks, sumLogProb: b.sumLogProb + ch.logProb, stepCount: b.stepCount + 1, isFinished: fin, lastTokenId: Int64(ch.id), pastK: nk, pastV: nv, coverage: nc, prevWeights: np))
                }
            }
            pool.sort { $0.rankingScore > $1.rankingScore }
            beams = Array(pool.prefix(beamWidth))
        }
        return beams
    }

    // MARK: - Math & Tensor Helpers

    private static func makeToken(stepIdx: Int, tokenId: Int, prob: Float, visionAttn: [Float]?, manifest: ModelManifest) -> DecodedToken {
        let text = (tokenId >= 0 && tokenId < manifest.tokens.count) ? manifest.tokens[tokenId] : "<UNK>"
        return DecodedToken(
            id: stepIdx,
            tokenId: tokenId,
            text: text,
            confidence: prob,
            category: TokenCategory.classify(text),
            visionAttention: visionAttn
        )
    }

    private static func fingerprintStrokes(_ strokes: [RawStroke]) -> UInt64 {
        var h: UInt64 = 14695981039346656037
        let prime: UInt64 = 1099511628211
        func mix(_ v: UInt64) {
            h ^= v
            h = h &* prime
        }
        mix(UInt64(strokes.count))
        for s in strokes {
            mix(UInt64(s.points.count))
            for pt in s.points {
                mix(pt.x.bitPattern)
                mix(pt.y.bitPattern)
                mix((pt.timeMs ?? -1).bitPattern)
            }
        }
        return h
    }

    private static func argmaxLogSoftmax(_ logits: [Float], offset: Int, count: Int) -> (id: Int, logProb: Double, prob: Float) {
        var bestIdx = 0
        var bestVal = -Float.greatestFiniteMagnitude
        for i in 0..<count {
            let v = logits[offset + i]
            if v > bestVal {
                bestVal = v
                bestIdx = i
            }
        }
        var sumExp: Double = 0
        for i in 0..<count {
            sumExp += exp(Double(logits[offset + i] - bestVal))
        }
        let logProb = -log(max(sumExp, 1e-30))
        return (bestIdx, logProb, Float(exp(logProb)))
    }

    private static func topKLogSoftmax(_ logits: [Float], offset: Int, count: Int, k: Int) -> [(id: Int, logProb: Double, prob: Float)] {
        guard count > 0 else { return [] }
        var maxVal = -Float.greatestFiniteMagnitude
        for i in 0..<count {
            let v = logits[offset + i]
            if v > maxVal { maxVal = v }
        }
        var sumExp: Double = 0
        for i in 0..<count {
            sumExp += exp(Double(logits[offset + i] - maxVal))
        }
        let logZ = Double(maxVal) + log(max(sumExp, 1e-30))

        var top: [(id: Int, val: Float)] = []
        top.reserveCapacity(k + 1)
        for i in 0..<count {
            let v = logits[offset + i]
            if top.count < k || v > top[top.count - 1].val {
                var pos = top.count
                while pos > 0 && v > top[pos - 1].val {
                    pos -= 1
                }
                top.insert((i, v), at: pos)
                if top.count > k {
                    top.removeLast()
                }
            }
        }
        return top.map { item in
            let lp = Double(item.val) - logZ
            return (item.id, lp, Float(exp(lp)))
        }
    }

    private static func averageVisionAttention(
        weights: [Float],
        numHeads: Int,
        totalMemLen: Int,
        visionOffset: Int,
        visionLen: Int
    ) -> [Float] {
        guard weights.count >= numHeads * totalMemLen, visionLen > 0 else { return [] }
        var avg = [Float](repeating: 0, count: visionLen)
        let invHeads = 1.0 / Float(max(numHeads, 1))
        for h in 0..<numHeads {
            let base = h * totalMemLen + visionOffset
            for i in 0..<visionLen {
                avg[i] += weights[base + i] * invHeads
            }
        }
        return avg
    }

    private static func extractVisionAttentionAtStep(
        _ allPrevW: [Float],
        numHeads: Int,
        stepLen: Int,
        stepIdx: Int,
        totalMemLen: Int,
        visionOffset: Int,
        visionLen: Int
    ) -> [Float] {
        guard allPrevW.count >= numHeads * stepLen * totalMemLen, visionLen > 0 else { return [] }
        var avg = [Float](repeating: 0, count: visionLen)
        let invHeads = 1.0 / Float(max(numHeads, 1))
        for h in 0..<numHeads {
            let base = (h * stepLen + stepIdx) * totalMemLen + visionOffset
            for i in 0..<visionLen {
                avg[i] += allPrevW[base + i] * invHeads
            }
        }
        return avg
    }

    private static func makeFloatTensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buf -> NSMutableData in
            NSMutableData(bytes: buf.baseAddress, length: buf.count * MemoryLayout<Float>.stride)
        }
        return try ORTValue(tensorData: data, elementType: .float, shape: shape.map { NSNumber(value: $0) })
    }

    private static func makeInt64Tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buf -> NSMutableData in
            NSMutableData(bytes: buf.baseAddress, length: buf.count * MemoryLayout<Int64>.stride)
        }
        return try ORTValue(tensorData: data, elementType: .int64, shape: shape.map { NSNumber(value: $0) })
    }

    private static func makeUInt8Tensor(_ values: [UInt8], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buf -> NSMutableData in
            NSMutableData(bytes: buf.baseAddress, length: buf.count * MemoryLayout<UInt8>.stride)
        }
        return try ORTValue(tensorData: data, elementType: .uInt8, shape: shape.map { NSNumber(value: $0) })
    }

    private static func extractFloats(_ value: ORTValue) throws -> [Float] {
        let data = try value.tensorData() as Data
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private static func extractUInt8s(_ value: ORTValue) throws -> [UInt8] {
        let data = try value.tensorData() as Data
        return Array(data)
    }

    private static func inferenceError(_ message: String) -> NSError {
        NSError(domain: "InkTeX.Engine", code: 200, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
