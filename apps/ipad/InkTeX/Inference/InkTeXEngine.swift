import Foundation
import OnnxRuntimeBindings

@MainActor
public final class InkTeXEngine: ObservableObject {
    public enum Status: Equatable {
        case loading
        case ready
        case missingModel
        case failed(String)
    }

    @Published public private(set) var status: Status = .loading
    @Published public private(set) var loadedBundle: LocatedModelBundle?
    @Published public private(set) var hasR2LScorer: Bool = false
    @Published public private(set) var isRecognizing: Bool = false
    @Published public var useCoreMLForVision: Bool = true
    @Published public var beamWidth: Int = 4
    @Published public var r2lWeight: Double = 0.5

    private let workerQueue = DispatchQueue(label: "com.wosca.inktex.inference", qos: .userInitiated)
    private var runtime: LoadedRuntime?

    private final class LoadedRuntime {
        let env: ORTEnv
        let bundle: LocatedModelBundle
        let tokenizer: LatexTokenizer
        let visionSession: ORTSession
        let strokeSession: ORTSession
        let memoryProjSession: ORTSession
        let decoderStepSession: ORTSession
        let r2lScorerSession: ORTSession?

        init(bundle: LocatedModelBundle, useCoreMLForVision: Bool) throws {
            self.bundle = bundle
            self.tokenizer = LatexTokenizer(manifest: bundle.manifest)
            self.env = try ORTEnv(loggingLevel: .warning)

            let visionOpts = try Self.makeOptions(useCoreML: useCoreMLForVision, threads: 4)
            let cpuOpts = try Self.makeOptions(useCoreML: false, threads: 4)

            self.visionSession = try ORTSession(env: env, modelPath: bundle.visionEncoderURL.path, sessionOptions: visionOpts)
            self.strokeSession = try ORTSession(env: env, modelPath: bundle.strokeEncoderURL.path, sessionOptions: cpuOpts)
            self.memoryProjSession = try ORTSession(env: env, modelPath: bundle.memoryProjURL.path, sessionOptions: cpuOpts)
            self.decoderStepSession = try ORTSession(env: env, modelPath: bundle.decoderStepURL.path, sessionOptions: cpuOpts)
            if let r2lURL = bundle.r2lScorerURL {
                self.r2lScorerSession = try ORTSession(env: env, modelPath: r2lURL.path, sessionOptions: cpuOpts)
            } else {
                self.r2lScorerSession = nil
            }
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

    private struct BeamHypothesis {
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

    public func reloadModel() {
        status = .loading
        let useCoreML = useCoreMLForVision
        workerQueue.async { [weak self] in
            guard let self else { return }
            guard let bundle = ModelBundleLocator.locate() else {
                Task { @MainActor in
                    self.runtime = nil
                    self.loadedBundle = nil
                    self.hasR2LScorer = false
                    self.status = .missingModel
                }
                return
            }
            do {
                let loaded = try LoadedRuntime(bundle: bundle, useCoreMLForVision: useCoreML)
                let hasR2L = loaded.r2lScorerSession != nil
                Task { @MainActor in
                    self.runtime = loaded
                    self.loadedBundle = bundle
                    self.hasR2LScorer = hasR2L
                    self.status = .ready
                }
            } catch {
                Task { @MainActor in
                    self.runtime = nil
                    self.loadedBundle = bundle
                    self.hasR2LScorer = false
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

        return try await withCheckedThrowingContinuation { continuation in
            workerQueue.async {
                do {
                    let result = try Self.runInference(
                        runtime: runtime,
                        strokes: strokes,
                        modality: modality,
                        decodingMode: decodingMode,
                        beamWidth: width,
                        r2lWeight: weight,
                        maxTokens: maxTokens
                    )
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

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

        let strip = StripRasterizer.render(strokes: strokes, config: manifest.vision)
        let strokeInput = StrokePreprocessor.preprocess(strokes: strokes, config: manifest.preprocessing)

        let encodeStart = CFAbsoluteTimeGetCurrent()
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
        let projOut = try runtime.memoryProjSession.run(
            withInputs: ["memory": memTensor],
            outputNames: ["mk", "mv"],
            runOptions: nil
        )
        guard let mkVal = projOut["mk"], let mvVal = projOut["mv"] else {
            throw inferenceError("Missing outputs from memory_proj.onnx")
        }
        let encodeMs = (CFAbsoluteTimeGetCurrent() - encodeStart) * 1000.0

        // KV-cached L2R decode (Greedy when effectiveBeam == 1, Beam Search when > 1)
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let numLayers = manifest.decoderLayers
        let numHeads = manifest.attentionHeads
        let headDim = manifest.headDim
        let memMaskVal = try makeUInt8Tensor(combinedMask, shape: [1, totalMemLen])

        let initKVCount = numLayers * 1 * numHeads * 1 * headDim
        let initK = try makeFloatTensor([Float](repeating: 0, count: initKVCount), shape: [numLayers, 1, numHeads, 1, headDim])
        let initV = try makeFloatTensor([Float](repeating: 0, count: initKVCount), shape: [numLayers, 1, numHeads, 1, headDim])
        let covCount = 1 * numHeads * 1 * totalMemLen
        let initCov = try makeFloatTensor([Float](repeating: 0, count: covCount), shape: [1, numHeads, 1, totalMemLen])
        let initPrev = try makeFloatTensor([Float](repeating: 0, count: covCount), shape: [1, numHeads, 1, totalMemLen])

        let effectiveBeam = decodingMode == .greedy ? 1 : max(2, min(8, beamWidth))
        let tokenLimit = min(maxTokens, manifest.maxTokens - 1)
        let specialSet: Set<Int> = [manifest.padId, manifest.bosId, manifest.eosId, manifest.unkId]

        func runDecoderStep(
            tokenId: Int64,
            pastK: ORTValue,
            pastV: ORTValue,
            coverage: ORTValue,
            prevWeights: ORTValue
        ) throws -> (logits: [Float], nextK: ORTValue, nextV: ORTValue, nextCov: ORTValue, nextPrev: ORTValue) {
            let tokenVal = try makeInt64Tensor([tokenId], shape: [1, 1])
            let out = try runtime.decoderStepSession.run(
                withInputs: [
                    "token": tokenVal,
                    "past_k": pastK,
                    "past_v": pastV,
                    "mk": mkVal,
                    "mv": mvVal,
                    "memory_mask": memMaskVal,
                    "coverage": coverage,
                    "prev_weights": prevWeights,
                ],
                outputNames: ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
                runOptions: nil
            )
            guard let logitsVal = out["logits"],
                  let nk = out["new_k"],
                  let nv = out["new_v"],
                  let nc = out["new_coverage"],
                  let np = out["new_prev_weights"] else {
                throw inferenceError("Missing outputs from decoder_step.onnx")
            }
            return (try extractFloats(logitsVal), nk, nv, nc, np)
        }

        func makeDecodedToken(stepIdx: Int, tokenId: Int, prob: Float, visionAttn: [Float]?) -> DecodedToken? {
            guard !specialSet.contains(tokenId) else { return nil }
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

        // Step 0: single pass from BOS
        let step0 = try runDecoderStep(
            tokenId: Int64(manifest.bosId),
            pastK: initK,
            pastV: initV,
            coverage: initCov,
            prevWeights: initPrev
        )
        let step0Attn: [Float]? = visionMemLen > 0
            ? averageVisionAttention(
                weights: try extractFloats(step0.nextPrev),
                numHeads: numHeads,
                totalMemLen: totalMemLen,
                visionOffset: strokeMemLen,
                visionLen: visionMemLen
            )
            : nil

        let firstChoices = topKLogSoftmax(step0.logits, k: effectiveBeam)
        var beams: [BeamHypothesis] = firstChoices.map { choice in
            let finished = (choice.id == manifest.eosId)
            var body: [Int] = []
            var toks: [DecodedToken] = []
            if !finished, let dt = makeDecodedToken(stepIdx: 0, tokenId: choice.id, prob: choice.prob, visionAttn: step0Attn) {
                body.append(choice.id)
                toks.append(dt)
            }
            return BeamHypothesis(
                bodyIds: body,
                decodedTokens: toks,
                sumLogProb: choice.logProb,
                stepCount: 1,
                isFinished: finished,
                lastTokenId: Int64(choice.id),
                pastK: step0.nextK,
                pastV: step0.nextV,
                coverage: step0.nextCov,
                prevWeights: step0.nextPrev
            )
        }

        // Steps 1 ..< tokenLimit
        if tokenLimit > 1 {
            for stepIdx in 1..<tokenLimit {
                if beams.allSatisfy(\.isFinished) { break }
                var nextPool: [BeamHypothesis] = []
                nextPool.reserveCapacity(effectiveBeam * effectiveBeam)

                for beam in beams {
                    if beam.isFinished {
                        nextPool.append(beam)
                        continue
                    }
                    let step = try runDecoderStep(
                        tokenId: beam.lastTokenId,
                        pastK: beam.pastK,
                        pastV: beam.pastV,
                        coverage: beam.coverage,
                        prevWeights: beam.prevWeights
                    )
                    let stepAttn: [Float]? = visionMemLen > 0
                        ? averageVisionAttention(
                            weights: try extractFloats(step.nextPrev),
                            numHeads: numHeads,
                            totalMemLen: totalMemLen,
                            visionOffset: strokeMemLen,
                            visionLen: visionMemLen
                        )
                        : nil

                    let choices = topKLogSoftmax(step.logits, k: effectiveBeam)
                    for choice in choices {
                        let finished = (choice.id == manifest.eosId)
                        var body = beam.bodyIds
                        var toks = beam.decodedTokens
                        if !finished, let dt = makeDecodedToken(stepIdx: stepIdx, tokenId: choice.id, prob: choice.prob, visionAttn: stepAttn) {
                            body.append(choice.id)
                            toks.append(dt)
                        }
                        nextPool.append(
                            BeamHypothesis(
                                bodyIds: body,
                                decodedTokens: toks,
                                sumLogProb: beam.sumLogProb + choice.logProb,
                                stepCount: beam.stepCount + 1,
                                isFinished: finished,
                                lastTokenId: Int64(choice.id),
                                pastK: step.nextK,
                                pastV: step.nextV,
                                coverage: step.nextCov,
                                prevWeights: step.nextPrev
                            )
                        )
                    }
                }

                nextPool.sort { $0.rankingScore > $1.rankingScore }
                beams = Array(nextPool.prefix(effectiveBeam))
            }
        }

        // Deduplicate final beams by decoded LaTeX while keeping highest L2R score
        var uniqueBeams: [BeamHypothesis] = []
        var seenLatex: Set<String> = []
        beams.sort { $0.rankingScore > $1.rankingScore }
        for b in beams {
            let tex = LatexTokenizer.detokenize(b.decodedTokens.map(\.text))
            if seenLatex.insert(tex).inserted {
                uniqueBeams.append(b)
            }
        }
        if uniqueBeams.isEmpty, let first = beams.first {
            uniqueBeams = [first]
        }
        let decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000.0

        // Optional R2L Bidirectional Reranking (single batched forward pass!)
        var r2lMs: Double = 0
        var r2lScores: [Double]? = nil
        var r2lUsed = false

        if decodingMode == .beamR2L, let r2lSession = runtime.r2lScorerSession, !uniqueBeams.isEmpty {
            let r2lStart = CFAbsoluteTimeGetCurrent()
            let r2lSeqs: [[Int64]] = uniqueBeams.map { b in
                var seq: [Int64] = [Int64(manifest.bosId)]
                seq.reserveCapacity(b.bodyIds.count + 2)
                for id in b.bodyIds.reversed() {
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
                    "memory": memTensor,
                    "memory_mask": memMaskVal,
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

        // Rank candidates and pick winner
        let l2rTopIndex = 0
        var indexedScores: [(index: Int, l2r: Double, r2l: Double?, combined: Double)] = uniqueBeams.enumerated().map { idx, b in
            let l2r = b.avgLogProb
            let r2l = r2lScores?[idx]
            let combined = r2l != nil ? (l2r + r2lWeight * r2l!) : l2r
            return (idx, l2r, r2l, combined)
        }
        indexedScores.sort { $0.combined > $1.combined }

        let winnerIdx = indexedScores.first?.index ?? 0
        let r2lChangedWinner = r2lUsed && (winnerIdx != l2rTopIndex)
        let winner = uniqueBeams[winnerIdx]

        let beamSummaries: [BeamCandidate] = indexedScores.enumerated().map { rank, item in
            let b = uniqueBeams[item.index]
            let tex = LatexTokenizer.detokenize(b.decodedTokens.map(\.text))
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

        let gridH = visionMemLen > 0 ? max(1, strip.height / 16) : 0
        let gridW = gridH > 0 ? (visionMemLen / gridH) : 0

        return RecognitionResult(
            latex: latex,
            decodedTokens: winner.decodedTokens,
            overallConfidence: overallConf,
            totalMs: totalMs,
            encodeMs: encodeMs,
            decodeMs: decodeMs,
            r2lMs: r2lMs,
            strokeCount: strokeInput?.strokeCount ?? strokes.count,
            pointCount: strokeInput?.pointCount ?? 0,
            stripHeight: strip.height,
            stripWidth: strip.width,
            visionGridW: gridW,
            visionGridH: gridH,
            stripLumBytes: strip.lumBytes,
            stripOrderBytes: strip.orderBytes,
            modality: modality,
            decodingMode: decodingMode,
            r2lUsed: r2lUsed,
            r2lChangedWinner: r2lChangedWinner,
            beams: beamSummaries,
            truncated: !winner.isFinished
        )
    }

    // MARK: - Math & Tensor Helpers

    private static func topKLogSoftmax(_ logits: [Float], k: Int) -> [(id: Int, logProb: Double, prob: Float)] {
        guard !logits.isEmpty else { return [] }
        var maxVal = -Float.greatestFiniteMagnitude
        for v in logits where v > maxVal { maxVal = v }

        var sumExp: Double = 0
        for v in logits {
            sumExp += exp(Double(v - maxVal))
        }
        let logZ = Double(maxVal) + log(max(sumExp, 1e-30))

        // Fast partial top-K selection (vocab = 907, k <= 8)
        var top: [(id: Int, val: Float)] = []
        top.reserveCapacity(k + 1)
        for i in 0..<logits.count {
            let v = logits[i]
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
