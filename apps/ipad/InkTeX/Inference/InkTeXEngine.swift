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
    @Published public private(set) var isV2Model: Bool = false
    @Published public private(set) var hasR2LScorer: Bool = false
    @Published public private(set) var hasSharedR2L: Bool = false
    @Published public private(set) var supportsDynamicLoRA: Bool = false
    @Published public private(set) var activeStyleModelSha: String? = nil
    @Published public private(set) var supportsV2FastDecode: Bool = false
    @Published public private(set) var staticKvLen: Int? = nil
    @Published public private(set) var isRecognizing: Bool = false
    @Published public private(set) var memorySnapshot: MemorySnapshot = .empty
    @Published public var coreMLProfile: CoreMLProfile = .encodersAndR2L
    @Published public var liveTwoPhase: Bool = true
    @Published public var showLiveRamInBar: Bool = true
    @Published public var beamWidth: Int = 4
    @Published public var r2lWeight: Double = 0.5

    private var preferredBundleId: String? = nil
    private var peakFootprintMB: Double = 0
    private var latestPersonalization: PersonalizationState = .none
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
        let chunkBucket: Int?
        let usedWriterStyle: Bool
        let usedLoRAAdapter: Bool
        let activeProfileName: String?
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
            chunkBucket: Int? = nil,
            usedWriterStyle: Bool = false,
            usedLoRAAdapter: Bool = false,
            activeProfileName: String? = nil,
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
            self.chunkBucket = chunkBucket
            self.usedWriterStyle = usedWriterStyle
            self.usedLoRAAdapter = usedLoRAAdapter
            self.activeProfileName = activeProfileName
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
        let isV2: Bool
        let writerStyleSession: ORTSession?
        let writerStyleSha: String
        let visionSession: ORTSession
        let strokeSession: ORTSession
        let memoryProjSession: ORTSession
        let decoderStepSession: ORTSession
        let r2lScorerSession: ORTSession?
        let isV2Decoder: Bool
        let hasSharedR2L: Bool
        let staticKvLen: Int?
        let staticShapes: Bool
        let supportsDynamicLoRA: Bool
        let styleDim: Int
        let loraShapeA: [Int]
        let loraShapeB: [Int]
        let zeroStyleTensor: ORTValue?
        let zeroLoraATensor: ORTValue?
        let zeroLoraBTensor: ORTValue?
        let baselineMB: Double
        let memoryComponents: [MemoryComponent]

        // Incremental & personalization caches (accessed strictly on workerQueue)
        var lastPrefixBodyIds: [Int] = []
        var lastEncoded: EncodedMemoryCache? = nil
        var personalization: PersonalizationState = .none
        var activeProfileDigest: UInt64 = 0
        var activeAdapterDigest: UInt64 = 0
        var cachedStyleDigest: UInt64 = 0
        var cachedStyleTensor: ORTValue? = nil
        var cachedAdapterDigest: UInt64 = 0
        var cachedLoraATensor: ORTValue? = nil
        var cachedLoraBTensor: ORTValue? = nil

        init(bundle: LocatedModelBundle, profile: CoreMLProfile) throws {
            self.bundle = bundle
            self.isV2 = bundle.isV2
            self.tokenizer = LatexTokenizer(manifest: bundle.manifest)
            self.staticKvLen = bundle.manifest.staticKvLen ?? (bundle.isV2 ? 256 : nil)
            self.staticShapes = bundle.manifest.staticShapes == true
            self.isV2Decoder = bundle.isV2
                || (bundle.manifest.supportsPrefixVerify == true)
                || (bundle.manifest.supportsBatchedDecoder == true)
                || (bundle.manifest.staticKvLen != nil)
            self.hasSharedR2L = bundle.isV2 && (bundle.manifest.resolvedBosR2LId != nil)
            self.env = try ORTEnv(loggingLevel: .warning)

            let pers = bundle.manifest.personalization
            self.styleDim = pers?.styleDim ?? 64
            let loraLayersCount = pers?.loraLayers.count ?? 0
            let loraRank = pers?.loraRank ?? 0
            let strokeDim = pers?.strokeDim ?? bundle.manifest.strokeDim ?? 576
            self.supportsDynamicLoRA = bundle.isV2 && (pers?.dynamicLora == true) && loraLayersCount > 0 && loraRank > 0
            self.loraShapeA = [loraLayersCount, 3, loraRank, strokeDim]
            self.loraShapeB = [loraLayersCount, 3, strokeDim, loraRank]

            let bundleTag = "\(bundle.manifest.modelName)-\(bundle.manifest.epoch ?? 0)-\(bundle.manifest.quantization ?? "fp32")-\(profile.rawValue)"
                .replacingOccurrences(of: " ", with: "_")
                .replacingOccurrences(of: "/", with: "_")

            func fileFallbackTag(_ url: URL) -> String {
                let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
                let size = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
                let mtime = Int64(((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0).rounded())
                return "sz\(size)-t\(mtime)"
            }

            func graphCacheKey(_ name: String, fileKey: String, url: URL) -> String {
                let sha = bundle.manifest.files?[fileKey]?.sha256.prefix(12).description ?? fileFallbackTag(url)
                return "\(bundleTag)/\(name)-\(sha)"
            }

            let wantVisionCoreML = (profile != .off)
            let wantEncodersCoreML = (profile == .encodersAndR2L || profile == .allGraphs)
            let wantDecoderCoreML = (profile == .allGraphs)
            let useStaticCoreML = self.isV2 && self.staticShapes

            var components: [MemoryComponent] = []
            let base = MemoryMonitor.physicalFootprintMB()
            self.baselineMB = base

            var before = base
            var after = base

            if bundle.isV2, let styleURL = bundle.writerStyleURL {
                // WriterStyleEncoder is tiny (~50K params) and runs ONCE per calibration:
                // M1 CPU executes it in <0.5 ms with zero CoreML compilation overhead.
                self.writerStyleSha = bundle.manifest.files?["writer_style"]?.sha256 ?? fileFallbackTag(styleURL)
                let (styleSess, styleCoreML) = try Self.loadSession(
                    env: env,
                    url: styleURL,
                    tryCoreML: false,
                    requireStaticShapes: true,
                    cacheKey: graphCacheKey("writer_style", fileKey: "writer_style", url: styleURL),
                    threads: 2
                )
                self.writerStyleSession = styleSess
                after = MemoryMonitor.physicalFootprintMB()
                components.append(MemoryComponent(
                    id: "writer_style",
                    name: "Writer Style Encoder (FiLM)",
                    ramMB: max(0, after - before),
                    diskMB: MemoryMonitor.fileSizeMB(styleURL),
                    coreMLEnabled: styleCoreML,
                    colorRGB: (0.96, 0.45, 0.71)
                ))
                before = after
            } else {
                self.writerStyleSession = nil
                self.writerStyleSha = ""
            }

            let (visSess, visCoreML) = try Self.loadSession(
                env: env,
                url: bundle.visionEncoderURL,
                tryCoreML: wantVisionCoreML,
                requireStaticShapes: useStaticCoreML,
                cacheKey: graphCacheKey("vision", fileKey: "vision_encoder", url: bundle.visionEncoderURL),
                threads: 4
            )
            self.visionSession = visSess
            after = MemoryMonitor.physicalFootprintMB()
            components.append(MemoryComponent(
                id: "vision",
                name: bundle.isV2 ? "Vision Encoder (ConvNeXt-V2 + FPN)" : "Vision Encoder (ConvNeXt)",
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.visionEncoderURL),
                coreMLEnabled: visCoreML,
                colorRGB: (0.22, 0.74, 0.97)
            ))

            before = after
            let (strSess, strCoreML) = try Self.loadSession(
                env: env,
                url: bundle.strokeEncoderURL,
                tryCoreML: wantEncodersCoreML,
                requireStaticShapes: useStaticCoreML,
                cacheKey: graphCacheKey("stroke", fileKey: "stroke_encoder", url: bundle.strokeEncoderURL),
                threads: 4
            )
            self.strokeSession = strSess
            after = MemoryMonitor.physicalFootprintMB()
            let strokeLayerLabel = bundle.manifest.strokeLayers.map { "\($0)L" } ?? (bundle.isV2 ? "13L" : "8L")
            components.append(MemoryComponent(
                id: "stroke",
                name: bundle.isV2
                    ? "Stroke Encoder (\(strokeLayerLabel)\(self.supportsDynamicLoRA ? " + LoRA" : ""))"
                    : "Stroke Encoder (\(strokeLayerLabel))",
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.strokeEncoderURL),
                coreMLEnabled: strCoreML,
                colorRGB: (0.20, 0.83, 0.60)
            ))

            before = after
            let memFileKey = bundle.isV2 ? "fusion_memproj" : "memory_proj"
            let (projSess, projCoreML) = try Self.loadSession(
                env: env,
                url: bundle.memoryProjURL,
                tryCoreML: wantEncodersCoreML,
                requireStaticShapes: useStaticCoreML,
                cacheKey: graphCacheKey(memFileKey, fileKey: memFileKey, url: bundle.memoryProjURL),
                threads: 4
            )
            self.memoryProjSession = projSess
            after = MemoryMonitor.physicalFootprintMB()
            components.append(MemoryComponent(
                id: "memproj",
                name: bundle.isV2 ? "Bimodal Fusion + Memory Proj" : "Memory Projection",
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.memoryProjURL),
                coreMLEnabled: projCoreML,
                colorRGB: (0.98, 0.75, 0.14)
            ))

            before = after
            let (decSess, decCoreML) = try Self.loadSession(
                env: env,
                url: bundle.decoderStepURL,
                tryCoreML: wantDecoderCoreML,
                requireStaticShapes: false,
                cacheKey: graphCacheKey("decoder", fileKey: "decoder_step", url: bundle.decoderStepURL),
                threads: 4
            )
            self.decoderStepSession = decSess
            after = MemoryMonitor.physicalFootprintMB()
            let decTitle: String
            if bundle.isV2 {
                decTitle = "Shared Bidi Decoder (L2R+R2L, KV \(self.staticKvLen ?? 256))"
            } else if let kv = self.staticKvLen {
                decTitle = "L2R Decoder (Static KV \(kv))"
            } else {
                decTitle = "L2R Decoder (6L KV-Cache)"
            }
            components.append(MemoryComponent(
                id: "decoder",
                name: decTitle,
                ramMB: max(0, after - before),
                diskMB: MemoryMonitor.fileSizeMB(bundle.decoderStepURL),
                coreMLEnabled: decCoreML,
                colorRGB: (0.75, 0.52, 0.99)
            ))

            if let r2lURL = bundle.r2lScorerURL {
                before = after
                let (r2lSess, r2lCoreML) = try Self.loadSession(
                    env: env,
                    url: r2lURL,
                    tryCoreML: wantEncodersCoreML,
                    requireStaticShapes: false,
                    cacheKey: graphCacheKey("r2l", fileKey: "r2l_scorer", url: r2lURL),
                    threads: 4
                )
                self.r2lScorerSession = r2lSess
                after = MemoryMonitor.physicalFootprintMB()
                components.append(MemoryComponent(
                    id: "r2l",
                    name: "R2L Reranker (3L)",
                    ramMB: max(0, after - before),
                    diskMB: MemoryMonitor.fileSizeMB(r2lURL),
                    coreMLEnabled: r2lCoreML,
                    colorRGB: (0.96, 0.45, 0.71)
                ))
            } else {
                self.r2lScorerSession = nil
            }

            // Pre-allocate reusable zero personalization tensors ONCE at load time (0 allocation per stroke!)
            if bundle.isV2 {
                self.zeroStyleTensor = try InkTeXEngine.makeFloatTensor(
                    [Float](repeating: 0, count: self.styleDim),
                    shape: [1, self.styleDim]
                )
                if self.supportsDynamicLoRA {
                    let countA = self.loraShapeA.reduce(1, *)
                    let countB = self.loraShapeB.reduce(1, *)
                    self.zeroLoraATensor = try InkTeXEngine.makeFloatTensor(
                        [Float](repeating: 0, count: countA),
                        shape: self.loraShapeA
                    )
                    self.zeroLoraBTensor = try InkTeXEngine.makeFloatTensor(
                        [Float](repeating: 0, count: countB),
                        shape: self.loraShapeB
                    )
                } else {
                    self.zeroLoraATensor = nil
                    self.zeroLoraBTensor = nil
                }
            } else {
                self.zeroStyleTensor = nil
                self.zeroLoraATensor = nil
                self.zeroLoraBTensor = nil
            }

            self.memoryComponents = components
        }

        private static func coreMLCacheDir(for key: String) -> URL {
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            let dir = caches.appendingPathComponent("InkTeXCoreMLCache", isDirectory: true)
                .appendingPathComponent(key, isDirectory: true)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir
        }

        /// Loads an ONNX session using CoreML MLProgram + on-disk cache when requested,
        /// automatically falling back to M1 CPU if a graph contains an unsupported CoreML op.
        private static func loadSession(
            env: ORTEnv,
            url: URL,
            tryCoreML: Bool,
            requireStaticShapes: Bool,
            cacheKey: String,
            threads: Int32
        ) throws -> (ORTSession, Bool) {
            if tryCoreML && ORTIsCoreMLExecutionProviderAvailable() {
                do {
                    let opts = try ORTSessionOptions()
                    try opts.setGraphOptimizationLevel(.all)
                    try opts.setIntraOpNumThreads(threads)

                    let cacheDir = coreMLCacheDir(for: cacheKey)
                    let v2Options: [String: String] = [
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "All",
                        "EnableOnSubgraphs": "1",
                        "RequireStaticInputShapes": requireStaticShapes ? "1" : "0",
                        "ModelCacheDirectory": cacheDir.path,
                    ]
                    do {
                        try opts.appendCoreMLExecutionProvider(withOptionsV2: v2Options)
                    } catch {
                        let legacy = ORTCoreMLExecutionProviderOptions()
                        legacy.createMLProgram = true
                        legacy.enableOnSubgraphs = true
                        try opts.appendCoreMLExecutionProvider(with: legacy)
                    }
                    let sess = try ORTSession(env: env, modelPath: url.path, sessionOptions: opts)
                    return (sess, true)
                } catch {
                    // Graceful fallback to CPU if CoreML EP rejects a subgraph
                }
            }

            let cpuOpts = try ORTSessionOptions()
            try cpuOpts.setGraphOptimizationLevel(.all)
            try cpuOpts.setIntraOpNumThreads(threads)
            let sess = try ORTSession(env: env, modelPath: url.path, sessionOptions: cpuOpts)
            return (sess, false)
        }
    }

    private struct BatchedBeam {
        var bodyIds: [Int]
        var decodedTokens: [DecodedToken]
        var sumLogProb: Double
        var stepCount: Int
        var isFinished: Bool
        var lastTokenId: Int64
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

    // MARK: - Personalization Wiring

    /// Updates active writer profile / LoRA adapter state and immediately invalidates
    /// cached encoder memory & prefix guesses when either digest changes.
    public func applyPersonalization(_ state: PersonalizationState) {
        latestPersonalization = state
        let rt = runtime
        workerQueue.async {
            guard let rt else { return }
            Self.installPersonalization(state, on: rt)
        }
    }

    private static func installPersonalization(_ state: PersonalizationState, on rt: LoadedRuntime) {
        let profileChanged = rt.activeProfileDigest != state.profileDigest
        let adapterChanged = rt.activeAdapterDigest != state.adapterDigest
        rt.personalization = state
        rt.activeProfileDigest = state.profileDigest
        rt.activeAdapterDigest = state.adapterDigest
        if profileChanged {
            rt.cachedStyleDigest = 0
            rt.cachedStyleTensor = nil
        }
        if adapterChanged {
            rt.cachedAdapterDigest = 0
            rt.cachedLoraATensor = nil
            rt.cachedLoraBTensor = nil
        }
        if profileChanged || adapterChanged {
            rt.lastEncoded = nil
            rt.lastPrefixBodyIds = []
        }
    }

    /// Runs `writer_style_v2.onnx` once on calibration strokes so the UI can persist
    /// and display the enrolled 64-d writer vector immediately.
    public func computeWriterStyle(strokes: [RawStroke]) async throws -> (vector: [Float], modelSha: String, elapsedMs: Double) {
        guard let runtime else {
            throw Self.inferenceError("Model is not loaded yet.")
        }
        guard runtime.isV2, let styleSess = runtime.writerStyleSession else {
            throw Self.inferenceError("Active model is v1 and does not include writer_style_v2.onnx.")
        }
        return try await withCheckedThrowingContinuation { continuation in
            workerQueue.async {
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let pers = runtime.bundle.manifest.personalization
                    let maxStrokes = pers?.styleMaxStrokes ?? InkPreprocessorV2.styleMaxStrokes
                    let ptsPerStroke = pers?.stylePoints ?? InkPreprocessorV2.stylePoints
                    let styleIn = InkPreprocessorV2.extractStyleStrokes(
                        strokes: strokes,
                        maxStrokes: maxStrokes,
                        pointsPerStroke: ptsPerStroke
                    )
                    let ptsVal = try Self.makeFloatTensor(
                        styleIn.stylePoints,
                        shape: [1, maxStrokes, ptsPerStroke, InkPreprocessorV2.stylePointDim]
                    )
                    let aspVal = try Self.makeFloatTensor(styleIn.styleAspect, shape: [1, maxStrokes, 4])
                    let maskVal = try Self.makeUInt8Tensor(styleIn.styleMask, shape: [1, maxStrokes])
                    let out = try styleSess.run(
                        withInputs: [
                            "style_points": ptsVal,
                            "style_aspect": aspVal,
                            "style_mask": maskVal,
                        ],
                        outputNames: ["writer_style"],
                        runOptions: nil
                    )
                    guard let wsVal = out["writer_style"] else {
                        throw Self.inferenceError("Missing 'writer_style' output from writer_style_v2.onnx")
                    }
                    let vec = try Self.extractFloats(wsVal)
                    let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
                    continuation.resume(returning: (vec, runtime.writerStyleSha, ms))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Cache & Model Lifecycle

    public func clearCaches() {
        let rt = runtime
        workerQueue.async {
            rt?.lastPrefixBodyIds = []
            rt?.lastEncoded = nil
        }
    }

    public func clearCoreMLDiskCache() {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let dir = caches.appendingPathComponent("InkTeXCoreMLCache", isDirectory: true)
        try? FileManager.default.removeItem(at: dir)
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
        let profile = coreMLProfile
        let preferred = preferredBundleId
        let currentPersonalization = latestPersonalization
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
                    self.isV2Model = false
                    self.hasR2LScorer = false
                    self.hasSharedR2L = false
                    self.supportsDynamicLoRA = false
                    self.activeStyleModelSha = nil
                    self.supportsV2FastDecode = false
                    self.staticKvLen = nil
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
                let loaded = try LoadedRuntime(bundle: bundle, profile: profile)
                Self.installPersonalization(currentPersonalization, on: loaded)
                let isV2 = loaded.isV2
                let hasShared = loaded.hasSharedR2L
                let hasR2L = hasShared || (loaded.r2lScorerSession != nil)
                let dynLora = loaded.supportsDynamicLoRA
                let styleSha = loaded.writerStyleSha.isEmpty ? nil : loaded.writerStyleSha
                let isFastDec = loaded.isV2Decoder
                let staticKV = loaded.staticKvLen
                let afterLoad = MemoryMonitor.physicalFootprintMB()
                let headroom = MemoryMonitor.availableHeadroomMB()
                Task { @MainActor in
                    self.availableBundles = allBundles
                    self.runtime = loaded
                    self.loadedBundle = bundle
                    self.preferredBundleId = bundle.id
                    self.isV2Model = isV2
                    self.hasR2LScorer = hasR2L
                    self.hasSharedR2L = hasShared
                    self.supportsDynamicLoRA = dynLora
                    self.activeStyleModelSha = styleSha
                    self.supportsV2FastDecode = isFastDec
                    self.staticKvLen = staticKV
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
                    self.isV2Model = false
                    self.hasR2LScorer = false
                    self.hasSharedR2L = false
                    self.supportsDynamicLoRA = false
                    self.activeStyleModelSha = nil
                    self.supportsV2FastDecode = false
                    self.staticKvLen = nil
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

        // 1. Encode canvas (or reuse cached memory between Phase 1 Greedy & Phase 2 Beam+R2L)
        let fp = fingerprintStrokes(
            strokes,
            profileDigest: runtime.activeProfileDigest,
            adapterDigest: runtime.activeAdapterDigest
        )
        let encoded: EncodedMemoryCache
        let encodeMs: Double
        let encoderCacheHit: Bool

        if let cached = runtime.lastEncoded, cached.fingerprint == fp, cached.modality == modality {
            encoded = cached
            encodeMs = 0
            encoderCacheHit = true
        } else {
            let encodeStart = CFAbsoluteTimeGetCurrent()
            if runtime.isV2 {
                encoded = try encodeCanvasV2(runtime: runtime, strokes: strokes, modality: modality, fingerprint: fp)
            } else {
                encoded = try encodeCanvasV1(runtime: runtime, strokes: strokes, modality: modality, fingerprint: fp)
            }
            encodeMs = (CFAbsoluteTimeGetCurrent() - encodeStart) * 1000.0
            runtime.lastEncoded = encoded
            encoderCacheHit = false
        }

        // 2. Decode (InkTeX v2 Shared Static-KV / v1 Static-KV / Legacy)
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let kvCap = (runtime.staticKvLen ?? manifest.maxTokens) - 1
        let tokenLimit = min(maxTokens, kvCap)
        var specialSet: Set<Int> = [manifest.padId, manifest.resolvedBosL2RId, manifest.eosId, manifest.unkId]
        if let r2lBos = manifest.resolvedBosR2LId {
            specialSet.insert(r2lBos)
        }

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

        // 3. Optional R2L Bidirectional Reranking (1 batched parallel pass!)
        var r2lMs: Double = 0
        var r2lScores: [Double]? = nil
        var r2lUsed = false

        if decodingMode == .beamR2L, !uniqueCandidates.isEmpty {
            if runtime.hasSharedR2L, let bosR2L = manifest.resolvedBosR2LId {
                // InkTeX v2 unified shared bidirectional decoder: scores ALL beams in ONE forward pass with zero extra RAM!
                let r2lStart = CFAbsoluteTimeGetCurrent()
                r2lScores = try scoreR2LSharedDecoderV2(
                    runtime: runtime,
                    encoded: encoded,
                    candidates: uniqueCandidates.map(\.bodyIds),
                    bosR2LId: bosR2L,
                    eosId: manifest.eosId,
                    padId: manifest.padId,
                    maxStepLen: kvCap
                )
                r2lUsed = (r2lScores != nil)
                r2lMs = (CFAbsoluteTimeGetCurrent() - r2lStart) * 1000.0
            } else if let r2lSession = runtime.r2lScorerSession {
                let r2lStart = CFAbsoluteTimeGetCurrent()
                let r2lSeqs: [[Int64]] = uniqueCandidates.map { c in
                    var seq: [Int64] = [Int64(manifest.resolvedBosL2RId)]
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

        let vStride = manifest.vision.verticalStride ?? (runtime.isV2 ? 8 : 16)
        let gridH = encoded.visionMemLen > 0 ? max(1, encoded.strip.height / vStride) : 0
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
            truncated: !winner.isFinished,
            isV2Model: runtime.isV2,
            chunkBucket: encoded.chunkBucket,
            activeProfileName: encoded.activeProfileName,
            usedWriterStyle: encoded.usedWriterStyle,
            usedLoRAAdapter: encoded.usedLoRAAdapter,
            matchedCorrectionCache: false
        )
    }

    // MARK: - InkTeX v2 Canvas Encoding + Cached Personalization

    private static func resolveWriterStyleTensor(runtime: LoadedRuntime) throws -> (ORTValue, Bool) {
        let p = runtime.personalization
        guard p.styleEnabled, !p.calibrationStrokes.isEmpty else {
            guard let zero = runtime.zeroStyleTensor else {
                throw inferenceError("Missing pre-allocated zero writer_style tensor.")
            }
            return (zero, false)
        }
        if runtime.cachedStyleDigest == p.profileDigest, let cached = runtime.cachedStyleTensor {
            return (cached, true)
        }
        if let vec = p.cachedStyleVector,
           vec.count == runtime.styleDim,
           p.cachedStyleModelSha == runtime.writerStyleSha {
            let tensor = try makeFloatTensor(vec, shape: [1, runtime.styleDim])
            runtime.cachedStyleDigest = p.profileDigest
            runtime.cachedStyleTensor = tensor
            return (tensor, true)
        }
        guard let styleSess = runtime.writerStyleSession else {
            guard let zero = runtime.zeroStyleTensor else {
                throw inferenceError("Missing writer_style session and zero fallback.")
            }
            return (zero, false)
        }
        let pers = runtime.bundle.manifest.personalization
        let maxStrokes = pers?.styleMaxStrokes ?? InkPreprocessorV2.styleMaxStrokes
        let ptsPerStroke = pers?.stylePoints ?? InkPreprocessorV2.stylePoints
        let styleIn = InkPreprocessorV2.extractStyleStrokes(
            strokes: p.calibrationStrokes,
            maxStrokes: maxStrokes,
            pointsPerStroke: ptsPerStroke
        )
        let ptsVal = try makeFloatTensor(
            styleIn.stylePoints,
            shape: [1, maxStrokes, ptsPerStroke, InkPreprocessorV2.stylePointDim]
        )
        let aspVal = try makeFloatTensor(styleIn.styleAspect, shape: [1, maxStrokes, 4])
        let maskVal = try makeUInt8Tensor(styleIn.styleMask, shape: [1, maxStrokes])
        let out = try styleSess.run(
            withInputs: [
                "style_points": ptsVal,
                "style_aspect": aspVal,
                "style_mask": maskVal,
            ],
            outputNames: ["writer_style"],
            runOptions: nil
        )
        guard let wsVal = out["writer_style"] else {
            throw inferenceError("Missing 'writer_style' output from writer_style_v2.onnx")
        }
        runtime.cachedStyleDigest = p.profileDigest
        runtime.cachedStyleTensor = wsVal
        return (wsVal, true)
    }

    private static func resolveLoRATensors(runtime: LoadedRuntime) throws -> (ORTValue, ORTValue, Bool) {
        guard let zeroA = runtime.zeroLoraATensor, let zeroB = runtime.zeroLoraBTensor else {
            throw inferenceError("Missing pre-allocated zero LoRA tensors.")
        }
        let p = runtime.personalization
        guard p.loraEnabled, let weights = p.loraWeights else {
            return (zeroA, zeroB, false)
        }
        guard weights.metadata.loraAShape == runtime.loraShapeA,
              weights.metadata.loraBShape == runtime.loraShapeB else {
            // Incompatible adapter dimensions for this model preset -> fail safely to base weights
            return (zeroA, zeroB, false)
        }
        if runtime.cachedAdapterDigest == p.adapterDigest,
           let ca = runtime.cachedLoraATensor,
           let cb = runtime.cachedLoraBTensor {
            return (ca, cb, true)
        }
        let aVal = try makeFloatTensor(weights.loraA, shape: runtime.loraShapeA)
        let bVal = try makeFloatTensor(weights.loraB, shape: runtime.loraShapeB)
        runtime.cachedAdapterDigest = p.adapterDigest
        runtime.cachedLoraATensor = aVal
        runtime.cachedLoraBTensor = bVal
        return (aVal, bVal, true)
    }

    private static func encodeCanvasV2(
        runtime: LoadedRuntime,
        strokes: [RawStroke],
        modality: ModalityMode,
        fingerprint: UInt64
    ) throws -> EncodedMemoryCache {
        let manifest = runtime.bundle.manifest
        let staticChunkBucket: Int? = runtime.staticShapes ? manifest.preprocessing.resolvedChunkBuckets.first : nil
        let staticWidthBucket: Int? = runtime.staticShapes ? manifest.vision.resolvedWidthBuckets.first : nil

        let strip = StripRasterizer.renderV2(
            strokes: strokes,
            config: manifest.vision,
            bucketOverride: staticWidthBucket
        )

        var strokeMemVal: ORTValue
        var strokeMaskVal: ORTValue
        var strokeMemLen = 0
        var strokeCount = strokes.count
        var pointCount = 0
        var chunkBucketUsed: Int? = nil
        var usedWriterStyle = false
        var usedLoRA = false

        if modality == .bimodal || modality == .stroke {
            let chunked = InkPreprocessorV2.chunkInk(
                strokes: strokes,
                config: manifest.preprocessing,
                bucketOverride: staticChunkBucket
            )
            strokeCount = chunked.strokeCount
            pointCount = chunked.pointCount
            chunkBucketUsed = chunked.bucket
            strokeMemLen = chunked.bucket

            let (styleVal, styleActive) = try resolveWriterStyleTensor(runtime: runtime)
            usedWriterStyle = styleActive

            let cpVal = try makeFloatTensor(
                chunked.chunkPoints,
                shape: [1, chunked.bucket, chunked.chunkSize, InkPreprocessorV2.pointFeatureDim]
            )
            let cpmVal = try makeUInt8Tensor(
                chunked.chunkPointMask,
                shape: [1, chunked.bucket, chunked.chunkSize]
            )
            let cmVal = try makeFloatTensor(
                chunked.chunkMeta,
                shape: [1, chunked.bucket, InkPreprocessorV2.chunkMetaDim]
            )
            let ckVal = try makeUInt8Tensor(chunked.chunkMask, shape: [1, chunked.bucket])

            var inputs: [String: ORTValue] = [
                "chunk_points": cpVal,
                "chunk_point_mask": cpmVal,
                "chunk_meta": cmVal,
                "chunk_mask": ckVal,
                "writer_style": styleVal,
            ]
            if runtime.supportsDynamicLoRA {
                let (la, lb, loraActive) = try resolveLoRATensors(runtime: runtime)
                inputs["lora_a"] = la
                inputs["lora_b"] = lb
                usedLoRA = loraActive
            }
            let stOut = try runtime.strokeSession.run(
                withInputs: inputs,
                outputNames: ["stroke_memory"],
                runOptions: nil
            )
            guard let sm = stOut["stroke_memory"] else {
                throw inferenceError("Missing 'stroke_memory' output from stroke_encoder_v2.onnx")
            }
            strokeMemVal = sm
            strokeMaskVal = ckVal
        } else {
            // Vision-only: skip stroke encoder completely and pass minimal all-masked placeholder to fusion
            let dummyChunks = staticChunkBucket ?? 1
            strokeMemLen = dummyChunks
            strokeMemVal = try makeFloatTensor(
                [Float](repeating: 0, count: dummyChunks * manifest.dModel),
                shape: [1, dummyChunks, manifest.dModel]
            )
            strokeMaskVal = try makeUInt8Tensor([UInt8](repeating: 1, count: dummyChunks), shape: [1, dummyChunks])
        }

        var visionMemVal: ORTValue
        var visionMaskVal: ORTValue
        var visionMemLen = 0

        if modality == .bimodal || modality == .vision {
            let imgVal = try makeFloatTensor(
                strip.tensor,
                shape: [1, manifest.vision.channels, strip.height, strip.width]
            )
            let widthVal = try makeInt64Tensor([Int64(strip.usedWidth)], shape: [1])
            let visOut = try runtime.visionSession.run(
                withInputs: [
                    "images": imgVal,
                    "image_widths": widthVal,
                ],
                outputNames: ["vision_memory", "vision_mask"],
                runOptions: nil
            )
            guard let vm = visOut["vision_memory"], let vmask = visOut["vision_mask"] else {
                throw inferenceError("Missing outputs from vision_encoder_v2.onnx")
            }
            let vmShape = try vm.tensorTypeAndShapeInfo().shape.map(\.intValue)
            visionMemLen = vmShape[1]
            visionMemVal = vm
            visionMaskVal = vmask
        } else {
            // Stroke-only: skip vision encoder completely and pass minimal all-masked placeholder (vision_avail == 0 -> exact no-op in fusion!)
            let hRows = max(1, manifest.vision.height / (manifest.vision.verticalStride ?? 8))
            let dummyVisTokens: Int
            if let sw = staticWidthBucket {
                dummyVisTokens = hRows * max(1, sw / (manifest.vision.horizontalStride ?? 16))
            } else {
                dummyVisTokens = hRows
            }
            visionMemLen = 0
            visionMemVal = try makeFloatTensor(
                [Float](repeating: 0, count: dummyVisTokens * manifest.dModel),
                shape: [1, dummyVisTokens, manifest.dModel]
            )
            visionMaskVal = try makeUInt8Tensor([UInt8](repeating: 1, count: dummyVisTokens), shape: [1, dummyVisTokens])
        }

        let fusionOut = try runtime.memoryProjSession.run(
            withInputs: [
                "stroke_memory": strokeMemVal,
                "stroke_mask": strokeMaskVal,
                "vision_memory": visionMemVal,
                "vision_mask": visionMaskVal,
            ],
            outputNames: ["memory", "memory_mask", "mk", "mv"],
            runOptions: nil
        )
        guard let memVal = fusionOut["memory"],
              let memMaskVal = fusionOut["memory_mask"],
              let mkVal = fusionOut["mk"],
              let mvVal = fusionOut["mv"] else {
            throw inferenceError("Missing outputs from fusion_memproj_v2.onnx")
        }
        let memShape = try memVal.tensorTypeAndShapeInfo().shape.map(\.intValue)
        let totalMemLen = memShape[1]

        return EncodedMemoryCache(
            fingerprint: fingerprint,
            modality: modality,
            strip: strip,
            strokeCount: strokeCount,
            pointCount: pointCount,
            strokeMemLen: strokeMemLen,
            visionMemLen: visionMemLen,
            totalMemLen: totalMemLen,
            chunkBucket: chunkBucketUsed,
            usedWriterStyle: usedWriterStyle,
            usedLoRAAdapter: usedLoRA,
            activeProfileName: runtime.personalization.profileName,
            memTensor: memVal,
            memMaskVal: memMaskVal,
            mkVal: mkVal,
            mvVal: mvVal
        )
    }

    // MARK: - Legacy InkTeX v1 Canvas Encoding

    private static func encodeCanvasV1(
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

    // MARK: - Decoder Feed Helpers (Unified v2 + v1)

    private static func decoderOutputNames(isV2: Bool) -> Set<String> {
        isV2 ? ["logits", "new_k", "new_v"] : ["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"]
    }

    private static func makeDecoderInputs(
        isV2: Bool,
        token: ORTValue,
        cachePos: Int?,
        pastK: ORTValue,
        pastV: ORTValue,
        encoded: EncodedMemoryCache,
        coverage: ORTValue?,
        prevWeights: ORTValue?
    ) throws -> [String: ORTValue] {
        var feed: [String: ORTValue] = [
            "token": token,
            "past_k": pastK,
            "past_v": pastV,
            "mk": encoded.mkVal,
            "mv": encoded.mvVal,
            "memory_mask": encoded.memMaskVal,
        ]
        if let pos = cachePos {
            feed["cache_pos"] = try makeInt64Tensor([Int64(pos)], shape: [1])
        }
        if !isV2 {
            if let coverage { feed["coverage"] = coverage }
            if let prevWeights { feed["prev_weights"] = prevWeights }
        }
        return feed
    }

    // MARK: - Shared Bidirectional R2L Reranking (InkTeX v2 — Single Batched Pass!)

    private static func scoreR2LSharedDecoderV2(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        candidates: [[Int]],
        bosR2LId: Int,
        eosId: Int,
        padId: Int,
        maxStepLen: Int
    ) throws -> [Double]? {
        guard !candidates.isEmpty, let staticKV = runtime.staticKvLen else { return nil }
        let manifest = runtime.bundle.manifest
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let V = manifest.vocabSize
        let B = candidates.count

        var inSeqs: [[Int64]] = []
        var targetSeqs: [[Int]] = []
        inSeqs.reserveCapacity(B)
        targetSeqs.reserveCapacity(B)

        for body in candidates {
            let rev = Array(body.reversed().prefix(max(0, maxStepLen - 1)))
            var inp: [Int64] = [Int64(bosR2LId)]
            inp.reserveCapacity(rev.count + 1)
            var tgt: [Int] = []
            tgt.reserveCapacity(rev.count + 1)
            for id in rev {
                inp.append(Int64(id))
                tgt.append(id)
            }
            tgt.append(eosId)
            inSeqs.append(inp)
            targetSeqs.append(tgt)
        }

        let maxS = max(1, inSeqs.map(\.count).max() ?? 1)
        var tokFlat = [Int64](repeating: Int64(padId), count: B * maxS)
        for (row, seq) in inSeqs.enumerated() {
            let base = row * maxS
            for (col, id) in seq.enumerated() {
                tokFlat[base + col] = id
            }
        }

        let zeroKVCount = L * B * H * staticKV * D
        let initK = try makeFloatTensor([Float](repeating: 0, count: zeroKVCount), shape: [L, B, H, staticKV, D])
        let initV = try makeFloatTensor([Float](repeating: 0, count: zeroKVCount), shape: [L, B, H, staticKV, D])
        let feed = try makeDecoderInputs(
            isV2: true,
            token: try makeInt64Tensor(tokFlat, shape: [B, maxS]),
            cachePos: 0,
            pastK: initK,
            pastV: initV,
            encoded: encoded,
            coverage: nil,
            prevWeights: nil
        )
        let out = try runtime.decoderStepSession.run(
            withInputs: feed,
            outputNames: ["logits"],
            runOptions: nil
        )
        guard let logitsVal = out["logits"] else {
            throw inferenceError("Missing 'logits' output during shared v2 R2L scoring")
        }
        let allLogits = try extractFloats(logitsVal)

        var scores = [Double](repeating: 0, count: B)
        for b in 0..<B {
            let tgt = targetSeqs[b]
            var sumLP = 0.0
            for t in 0..<tgt.count {
                let offset = (b * maxS + t) * V
                sumLP += logSoftmaxAt(allLogits, offset: offset, count: V, targetId: tgt[t])
            }
            scores[b] = sumLP / Double(max(1, tgt.count))
        }
        return scores
    }

    // MARK: - 1. Prefix-Verified Parallel Greedy Decoding (Zero-Slice Static KV!)

    private static func decodePrefixVerifiedGreedy(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        prevGuess: [Int],
        tokenLimit: Int,
        specialSet: Set<Int>
    ) throws -> (bodyIds: [Int], tokens: [DecodedToken], avgLogProb: Double, isFinished: Bool, verifiedCount: Int) {
        let manifest = runtime.bundle.manifest
        let isV2 = runtime.isV2
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let M = encoded.totalMemLen
        let V = manifest.vocabSize
        let bosId = manifest.resolvedBosL2RId
        let staticKV = runtime.staticKvLen
        let initKVLen = staticKV ?? 1

        let cappedGuess = Array(prevGuess.prefix(max(1, tokenLimit - 1)))
        let S = cappedGuess.count + 1
        var inputIds: [Int64] = [Int64(bosId)]
        inputIds.reserveCapacity(S)
        for id in cappedGuess { inputIds.append(Int64(id)) }

        let initK = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * initKVLen * D), shape: [L, 1, H, initKVLen, D])
        let initV = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * initKVLen * D), shape: [L, 1, H, initKVLen, D])
        let initCov: ORTValue? = isV2 ? nil : try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])
        let initPrev: ORTValue? = isV2 ? nil : try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])

        let feed = try makeDecoderInputs(
            isV2: isV2,
            token: try makeInt64Tensor(inputIds, shape: [1, S]),
            cachePos: staticKV != nil ? 0 : nil,
            pastK: initK,
            pastV: initV,
            encoded: encoded,
            coverage: initCov,
            prevWeights: initPrev
        )
        let out = try runtime.decoderStepSession.run(
            withInputs: feed,
            outputNames: decoderOutputNames(isV2: isV2),
            runOptions: nil
        )
        guard let logitsVal = out["logits"],
              let newKVal = out["new_k"],
              let newVVal = out["new_v"] else {
            throw inferenceError("Missing outputs from decoder_step during prefix verification")
        }

        let allLogits = try extractFloats(logitsVal)
        let allPrevW: [Float]? = isV2 ? nil : try out["new_prev_weights"].map { try extractFloats($0) }

        var bodyIds: [Int] = []
        var decodedTokens: [DecodedToken] = []
        var sumLogProb: Double = 0
        var stepCount = 0
        var divergeIndex: Int? = nil
        var hitEOS = false
        var lastPredictedId: Int = bosId

        for i in 0..<S {
            let choice = argmaxLogSoftmax(allLogits, offset: i * V, count: V)
            sumLogProb += choice.logProb
            stepCount += 1
            lastPredictedId = choice.id

            if choice.id == manifest.eosId {
                hitEOS = true
                divergeIndex = nil
                break
            }

            if !specialSet.contains(choice.id) {
                let attn: [Float]? = (!isV2 && encoded.visionMemLen > 0 && allPrevW != nil)
                    ? extractVisionAttentionAtStep(allPrevW!, numHeads: H, stepLen: S, stepIdx: i, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
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

        if !hitEOS, let divI = divergeIndex, stepCount < tokenLimit {
            var pastK: ORTValue
            var pastV: ORTValue

            if staticKV != nil {
                // Zero-copy! Fixed-size KV buffer already holds slots 0...divI; overwrite from divI + 1!
                pastK = newKVal
                pastV = newVVal
            } else {
                let fullK = try extractFloats(newKVal)
                let fullV = try extractFloats(newVVal)
                let fullT = 1 + S
                let keepT = divI + 2
                var slicedK = [Float](repeating: 0, count: L * H * keepT * D)
                var slicedV = [Float](repeating: 0, count: L * H * keepT * D)
                let copyFloats = keepT * D
                for l in 0..<L {
                    for h in 0..<H {
                        let srcBase = ((l * H + h) * fullT) * D
                        let dstBase = ((l * H + h) * keepT) * D
                        copyFloatSlice(src: fullK, srcOffset: srcBase, dst: &slicedK, dstOffset: dstBase, count: copyFloats)
                        copyFloatSlice(src: fullV, srcOffset: srcBase, dst: &slicedV, dstOffset: dstBase, count: copyFloats)
                    }
                }
                pastK = try makeFloatTensor(slicedK, shape: [L, 1, H, keepT, D])
                pastV = try makeFloatTensor(slicedV, shape: [L, 1, H, keepT, D])
            }

            var coverage: ORTValue? = nil
            var prevWeights: ORTValue? = nil
            if !isV2, let newCovVal = out["new_coverage"], let prevW = allPrevW {
                let fullCov = try extractFloats(newCovVal)
                var slicedCov = [Float](repeating: 0, count: H * M)
                var slicedPrev = [Float](repeating: 0, count: H * M)
                for h in 0..<H {
                    let srcBase = (h * S + divI) * M
                    let dstBase = h * M
                    copyFloatSlice(src: fullCov, srcOffset: srcBase, dst: &slicedCov, dstOffset: dstBase, count: M)
                    copyFloatSlice(src: prevW, srcOffset: srcBase, dst: &slicedPrev, dstOffset: dstBase, count: M)
                }
                coverage = try makeFloatTensor(slicedCov, shape: [1, H, 1, M])
                prevWeights = try makeFloatTensor(slicedPrev, shape: [1, H, 1, M])
            }

            var curTok = Int64(lastPredictedId)
            var nextPos = divI + 1

            while stepCount < tokenLimit {
                let stFeed = try makeDecoderInputs(
                    isV2: isV2,
                    token: try makeInt64Tensor([curTok], shape: [1, 1]),
                    cachePos: staticKV != nil ? nextPos : nil,
                    pastK: pastK,
                    pastV: pastV,
                    encoded: encoded,
                    coverage: coverage,
                    prevWeights: prevWeights
                )
                let stepOut = try runtime.decoderStepSession.run(
                    withInputs: stFeed,
                    outputNames: decoderOutputNames(isV2: isV2),
                    runOptions: nil
                )
                guard let lVal = stepOut["logits"],
                      let nk = stepOut["new_k"],
                      let nv = stepOut["new_v"] else {
                    throw inferenceError("Missing outputs from decoder_step")
                }
                let logits = try extractFloats(lVal)
                let choice = argmaxLogSoftmax(logits, offset: 0, count: V)
                sumLogProb += choice.logProb
                let idx = stepCount
                stepCount += 1
                nextPos += 1

                if choice.id == manifest.eosId {
                    hitEOS = true
                    break
                }
                if !specialSet.contains(choice.id) {
                    let attn: [Float]?
                    if !isV2, encoded.visionMemLen > 0, let np = stepOut["new_prev_weights"] {
                        attn = averageVisionAttention(
                            weights: try extractFloats(np),
                            numHeads: H,
                            totalMemLen: M,
                            visionOffset: encoded.strokeMemLen,
                            visionLen: encoded.visionMemLen
                        )
                    } else {
                        attn = nil
                    }
                    bodyIds.append(choice.id)
                    decodedTokens.append(makeToken(stepIdx: idx, tokenId: choice.id, prob: choice.prob, visionAttn: attn, manifest: manifest))
                }
                curTok = Int64(choice.id)
                pastK = nk
                pastV = nv
                if !isV2 {
                    coverage = stepOut["new_coverage"]
                    prevWeights = stepOut["new_prev_weights"]
                }
            }
        }

        return (bodyIds, decodedTokens, sumLogProb / Double(max(1, stepCount)), hitEOS, verifiedCount)
    }

    // MARK: - 2. Standard Single-Step Greedy

    private static func decodeSingleGreedy(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        tokenLimit: Int,
        specialSet: Set<Int>
    ) throws -> (bodyIds: [Int], tokens: [DecodedToken], avgLogProb: Double, isFinished: Bool) {
        let manifest = runtime.bundle.manifest
        let isV2 = runtime.isV2
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let M = encoded.totalMemLen
        let V = manifest.vocabSize
        let bosId = manifest.resolvedBosL2RId
        let staticKV = runtime.staticKvLen
        let initKVLen = staticKV ?? 1

        var pastK = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * initKVLen * D), shape: [L, 1, H, initKVLen, D])
        var pastV = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * initKVLen * D), shape: [L, 1, H, initKVLen, D])
        var coverage: ORTValue? = isV2 ? nil : try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])
        var prevWeights: ORTValue? = isV2 ? nil : try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])

        var curTok = Int64(bosId)
        var bodyIds: [Int] = []
        var decodedTokens: [DecodedToken] = []
        var sumLogProb: Double = 0
        var stepCount = 0
        var hitEOS = false

        for stepIdx in 0..<tokenLimit {
            let feed = try makeDecoderInputs(
                isV2: isV2,
                token: try makeInt64Tensor([curTok], shape: [1, 1]),
                cachePos: staticKV != nil ? stepIdx : nil,
                pastK: pastK,
                pastV: pastV,
                encoded: encoded,
                coverage: coverage,
                prevWeights: prevWeights
            )
            let out = try runtime.decoderStepSession.run(
                withInputs: feed,
                outputNames: decoderOutputNames(isV2: isV2),
                runOptions: nil
            )
            guard let lVal = out["logits"],
                  let nk = out["new_k"],
                  let nv = out["new_v"] else {
                throw inferenceError("Missing outputs from decoder_step")
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
                let attn: [Float]?
                if !isV2, encoded.visionMemLen > 0, let np = out["new_prev_weights"] {
                    attn = averageVisionAttention(
                        weights: try extractFloats(np),
                        numHeads: H,
                        totalMemLen: M,
                        visionOffset: encoded.strokeMemLen,
                        visionLen: encoded.visionMemLen
                    )
                } else {
                    attn = nil
                }
                bodyIds.append(choice.id)
                decodedTokens.append(makeToken(stepIdx: stepIdx, tokenId: choice.id, prob: choice.prob, visionAttn: attn, manifest: manifest))
            }
            curTok = Int64(choice.id)
            pastK = nk
            pastV = nv
            if !isV2 {
                coverage = out["new_coverage"]
                prevWeights = out["new_prev_weights"]
            }
        }

        return (bodyIds, decodedTokens, sumLogProb / Double(max(1, stepCount)), hitEOS)
    }

    // MARK: - 3. Optimized Batched Beam Search (Active-Slot memcpy + Reusable Scratch Buffers)

    private static func decodeBatchedBeam(
        runtime: LoadedRuntime,
        encoded: EncodedMemoryCache,
        beamWidth: Int,
        tokenLimit: Int,
        specialSet: Set<Int>
    ) throws -> [BatchedBeam] {
        let manifest = runtime.bundle.manifest
        let isV2 = runtime.isV2
        let L = manifest.decoderLayers
        let H = manifest.attentionHeads
        let D = manifest.headDim
        let M = encoded.totalMemLen
        let V = manifest.vocabSize
        let bosId = manifest.resolvedBosL2RId
        let staticKV = runtime.staticKvLen
        let initKVLen = staticKV ?? 1

        let initK = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * initKVLen * D), shape: [L, 1, H, initKVLen, D])
        let initV = try makeFloatTensor([Float](repeating: 0, count: L * 1 * H * initKVLen * D), shape: [L, 1, H, initKVLen, D])
        let initCov: ORTValue? = isV2 ? nil : try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])
        let initPrev: ORTValue? = isV2 ? nil : try makeFloatTensor([Float](repeating: 0, count: 1 * H * 1 * M), shape: [1, H, 1, M])

        let feed0 = try makeDecoderInputs(
            isV2: isV2,
            token: try makeInt64Tensor([Int64(bosId)], shape: [1, 1]),
            cachePos: staticKV != nil ? 0 : nil,
            pastK: initK,
            pastV: initV,
            encoded: encoded,
            coverage: initCov,
            prevWeights: initPrev
        )
        let out0 = try runtime.decoderStepSession.run(
            withInputs: feed0,
            outputNames: decoderOutputNames(isV2: isV2),
            runOptions: nil
        )
        guard let l0Val = out0["logits"],
              let k0Val = out0["new_k"],
              let v0Val = out0["new_v"] else {
            throw inferenceError("Missing outputs from decoder_step")
        }

        let logits0 = try extractFloats(l0Val)
        let k0 = try extractFloats(k0Val)
        let v0 = try extractFloats(v0Val)
        let c0: [Float] = isV2 ? [] : (try out0["new_coverage"].map { try extractFloats($0) } ?? [])
        let p0: [Float] = isV2 ? [] : (try out0["new_prev_weights"].map { try extractFloats($0) } ?? [])
        let attn0: [Float]? = (!isV2 && encoded.visionMemLen > 0)
            ? averageVisionAttention(weights: p0, numHeads: H, totalMemLen: M, visionOffset: encoded.strokeMemLen, visionLen: encoded.visionMemLen)
            : nil

        let step0KVLen = staticKV ?? 2
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
                kvLen: step0KVLen,
                kData: k0,
                vData: v0,
                coverageData: c0,
                prevWeightsData: p0
            )
        }

        // Reusable scratch buffers across beam steps (eliminates ~19 MB/step malloc churn on static KV!)
        var kBatchScratch: [Float] = []
        var vBatchScratch: [Float] = []
        var covBatchScratch: [Float] = []
        var prevBatchScratch: [Float] = []

        if tokenLimit > 1 {
            for stepIdx in 1..<tokenLimit {
                let active = beams.filter { !$0.isFinished }
                if active.isEmpty { break }

                let B = active.count
                let T = active[0].kvLen
                let perHeadKV = T * D
                let perLayerBeamKV = H * perHeadKV
                let neededKV = L * B * perLayerBeamKV
                if kBatchScratch.count != neededKV {
                    kBatchScratch = [Float](repeating: 0, count: neededKV)
                    vBatchScratch = [Float](repeating: 0, count: neededKV)
                }

                var tokBatch = [Int64](repeating: 0, count: B)
                // With static KV, only slots 0..<activeSlots have valid keys/values; causal mask ignores > stepIdx!
                let activeSlots = staticKV != nil ? min(T, stepIdx) : T
                let copyFloatsPerHead = activeSlots * D

                kBatchScratch.withUnsafeMutableBufferPointer { kDstBuf in
                    vBatchScratch.withUnsafeMutableBufferPointer { vDstBuf in
                        guard let kDst = kDstBuf.baseAddress, let vDst = vDstBuf.baseAddress else { return }
                        for (b, beam) in active.enumerated() {
                            tokBatch[b] = beam.lastTokenId
                            beam.kData.withUnsafeBufferPointer { bkBuf in
                                beam.vData.withUnsafeBufferPointer { bvBuf in
                                    guard let kSrc = bkBuf.baseAddress, let vSrc = bvBuf.baseAddress else { return }
                                    if activeSlots == T {
                                        let bytes = perLayerBeamKV * MemoryLayout<Float>.stride
                                        for l in 0..<L {
                                            let srcOff = l * perLayerBeamKV
                                            let dstOff = (l * B + b) * perLayerBeamKV
                                            memcpy(kDst.advanced(by: dstOff), kSrc.advanced(by: srcOff), bytes)
                                            memcpy(vDst.advanced(by: dstOff), vSrc.advanced(by: srcOff), bytes)
                                        }
                                    } else {
                                        let headBytes = copyFloatsPerHead * MemoryLayout<Float>.stride
                                        for l in 0..<L {
                                            let srcLayer = l * perLayerBeamKV
                                            let dstLayer = (l * B + b) * perLayerBeamKV
                                            for h in 0..<H {
                                                let hOff = h * perHeadKV
                                                memcpy(kDst.advanced(by: dstLayer + hOff), kSrc.advanced(by: srcLayer + hOff), headBytes)
                                                memcpy(vDst.advanced(by: dstLayer + hOff), vSrc.advanced(by: srcLayer + hOff), headBytes)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                var covVal: ORTValue? = nil
                var prevVal: ORTValue? = nil
                if !isV2 {
                    let neededCov = B * H * M
                    if covBatchScratch.count != neededCov {
                        covBatchScratch = [Float](repeating: 0, count: neededCov)
                        prevBatchScratch = [Float](repeating: 0, count: neededCov)
                    }
                    let beamCovCount = H * M
                    for (b, beam) in active.enumerated() {
                        let covOff = b * beamCovCount
                        copyFloatSlice(src: beam.coverageData, srcOffset: 0, dst: &covBatchScratch, dstOffset: covOff, count: beamCovCount)
                        copyFloatSlice(src: beam.prevWeightsData, srcOffset: 0, dst: &prevBatchScratch, dstOffset: covOff, count: beamCovCount)
                    }
                    covVal = try makeFloatTensor(covBatchScratch, shape: [B, H, 1, M])
                    prevVal = try makeFloatTensor(prevBatchScratch, shape: [B, H, 1, M])
                }

                let feed = try makeDecoderInputs(
                    isV2: isV2,
                    token: try makeInt64Tensor(tokBatch, shape: [B, 1]),
                    cachePos: staticKV != nil ? stepIdx : nil,
                    pastK: try makeFloatTensor(kBatchScratch, shape: [L, B, H, T, D]),
                    pastV: try makeFloatTensor(vBatchScratch, shape: [L, B, H, T, D]),
                    encoded: encoded,
                    coverage: covVal,
                    prevWeights: prevVal
                )
                let stepOut = try runtime.decoderStepSession.run(
                    withInputs: feed,
                    outputNames: decoderOutputNames(isV2: isV2),
                    runOptions: nil
                )
                guard let lVal = stepOut["logits"],
                      let nkVal = stepOut["new_k"],
                      let nvVal = stepOut["new_v"] else {
                    throw inferenceError("Missing outputs from batched decoder_step")
                }

                let allLogits = try extractFloats(lVal)
                let allK = try extractFloats(nkVal)
                let allV = try extractFloats(nvVal)
                let allCov: [Float] = isV2 ? [] : (try stepOut["new_coverage"].map { try extractFloats($0) } ?? [])
                let allPrev: [Float] = isV2 ? [] : (try stepOut["new_prev_weights"].map { try extractFloats($0) } ?? [])

                let nextT = staticKV ?? (T + 1)
                let nextPerHeadKV = nextT * D
                let nextPerLayerBeamKV = H * nextPerHeadKV
                let nextActiveSlots = staticKV != nil ? min(nextT, stepIdx + 1) : nextT

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

                    if !isV2 && extractedAttn[bIdx] == nil {
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
                        let copyHeadBytes = nextActiveSlots * D * MemoryLayout<Float>.stride
                        allK.withUnsafeBufferPointer { akBuf in
                            allV.withUnsafeBufferPointer { avBuf in
                                bk.withUnsafeMutableBufferPointer { bkBuf in
                                    bv.withUnsafeMutableBufferPointer { bvBuf in
                                        guard let ak = akBuf.baseAddress, let av = avBuf.baseAddress,
                                              let bkP = bkBuf.baseAddress, let bvP = bvBuf.baseAddress else { return }
                                        if nextActiveSlots == nextT {
                                            let layerBytes = nextPerLayerBeamKV * MemoryLayout<Float>.stride
                                            for l in 0..<L {
                                                let srcOff = (l * B + bIdx) * nextPerLayerBeamKV
                                                let dstOff = l * nextPerLayerBeamKV
                                                memcpy(bkP.advanced(by: dstOff), ak.advanced(by: srcOff), layerBytes)
                                                memcpy(bvP.advanced(by: dstOff), av.advanced(by: srcOff), layerBytes)
                                            }
                                        } else {
                                            for l in 0..<L {
                                                let srcLayer = (l * B + bIdx) * nextPerLayerBeamKV
                                                let dstLayer = l * nextPerLayerBeamKV
                                                for h in 0..<H {
                                                    let hOff = h * nextPerHeadKV
                                                    memcpy(bkP.advanced(by: dstLayer + hOff), ak.advanced(by: srcLayer + hOff), copyHeadBytes)
                                                    memcpy(bvP.advanced(by: dstLayer + hOff), av.advanced(by: srcLayer + hOff), copyHeadBytes)
                                                }
                                            }
                                        }
                                    }
                                }
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
                        coverageData: (fin || isV2) ? [] : (extractedCov[bIdx] ?? []),
                        prevWeightsData: (fin || isV2) ? [] : (extractedPrev[bIdx] ?? [])
                    ))
                }
                beams = nextBeams
            }
        }
        return beams
    }

    // MARK: - 4. Legacy v1 Bundle Beam Search

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

        let (l0, k0, v0, c0, p0) = try runStep(Int64(manifest.resolvedBosL2RId), initK, initV, initCov, initPrev)
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

    private static func copyFloatSlice(src: [Float], srcOffset: Int, dst: inout [Float], dstOffset: Int, count: Int) {
        guard count > 0 else { return }
        src.withUnsafeBufferPointer { sBuf in
            dst.withUnsafeMutableBufferPointer { dBuf in
                guard let s = sBuf.baseAddress, let d = dBuf.baseAddress else { return }
                memcpy(d.advanced(by: dstOffset), s.advanced(by: srcOffset), count * MemoryLayout<Float>.stride)
            }
        }
    }

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

    private static func fingerprintStrokes(
        _ strokes: [RawStroke],
        profileDigest: UInt64 = 0,
        adapterDigest: UInt64 = 0
    ) -> UInt64 {
        var h: UInt64 = 14695981039346656037
        let prime: UInt64 = 1099511628211
        func mix(_ v: UInt64) {
            h ^= v
            h = h &* prime
        }
        mix(profileDigest)
        mix(adapterDigest)
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

    private static func logSoftmaxAt(_ logits: [Float], offset: Int, count: Int, targetId: Int) -> Double {
        guard count > 0, targetId >= 0, targetId < count else { return -20.0 }
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
        return Double(logits[offset + targetId]) - logZ
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
