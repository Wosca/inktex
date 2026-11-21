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
    @Published public private(set) var isRecognizing: Bool = false
    @Published public var useCoreMLForVision: Bool = true

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
                    self.status = .missingModel
                }
                return
            }
            do {
                let loaded = try LoadedRuntime(bundle: bundle, useCoreMLForVision: useCoreML)
                Task { @MainActor in
                    self.runtime = loaded
                    self.loadedBundle = bundle
                    self.status = .ready
                }
            } catch {
                Task { @MainActor in
                    self.runtime = nil
                    self.loadedBundle = bundle
                    self.status = .failed(error.localizedDescription)
                }
            }
        }
    }

    public func recognize(
        strokes: [RawStroke],
        modality: ModalityMode,
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

        return try await withCheckedThrowingContinuation { continuation in
            workerQueue.async {
                do {
                    let result = try Self.runInference(
                        runtime: runtime,
                        strokes: strokes,
                        modality: modality,
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

        // Fuse [stroke_memory, vision_memory] along sequence dim 1
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

        // Pre-project cross-attention K/V once
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

        // KV-cached autoregressive greedy decode loop
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let numLayers = manifest.decoderLayers
        let numHeads = manifest.attentionHeads
        let headDim = manifest.headDim

        let memMaskVal = try makeUInt8Tensor(combinedMask, shape: [1, totalMemLen])
        let initKVCount = numLayers * 1 * numHeads * 1 * headDim
        var pastK = try makeFloatTensor([Float](repeating: 0, count: initKVCount), shape: [numLayers, 1, numHeads, 1, headDim])
        var pastV = try makeFloatTensor([Float](repeating: 0, count: initKVCount), shape: [numLayers, 1, numHeads, 1, headDim])

        let covCount = 1 * numHeads * 1 * totalMemLen
        var coverage = try makeFloatTensor([Float](repeating: 0, count: covCount), shape: [1, numHeads, 1, totalMemLen])
        var prevWeights = try makeFloatTensor([Float](repeating: 0, count: covCount), shape: [1, numHeads, 1, totalMemLen])

        let tokenLimit = min(maxTokens, manifest.maxTokens - 1)
        var currentTokenId = Int64(manifest.bosId)
        var decodedTokens: [DecodedToken] = []
        decodedTokens.reserveCapacity(64)
        var hitEOS = false

        let specialSet: Set<Int> = [manifest.padId, manifest.bosId, manifest.eosId, manifest.unkId]

        for stepIdx in 0..<tokenLimit {
            let tokenVal = try makeInt64Tensor([currentTokenId], shape: [1, 1])
            let stepOut = try runtime.decoderStepSession.run(
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
            guard let logitsVal = stepOut["logits"],
                  let nextK = stepOut["new_k"],
                  let nextV = stepOut["new_v"],
                  let nextCov = stepOut["new_coverage"],
                  let nextPrev = stepOut["new_prev_weights"] else {
                throw inferenceError("Missing outputs from decoder_step.onnx")
            }

            let logits = try extractFloats(logitsVal)
            let (nextId, confidence) = argmaxWithConfidence(logits)
            if nextId == manifest.eosId {
                hitEOS = true
                break
            }

            if !specialSet.contains(nextId) {
                let text = (nextId >= 0 && nextId < manifest.tokens.count) ? manifest.tokens[nextId] : "<UNK>"
                var visionAttn: [Float]? = nil
                if visionMemLen > 0 {
                    let attnAll = try extractFloats(nextPrev) // [1, numHeads, 1, totalMemLen]
                    visionAttn = averageVisionAttention(
                        weights: attnAll,
                        numHeads: numHeads,
                        totalMemLen: totalMemLen,
                        visionOffset: strokeMemLen,
                        visionLen: visionMemLen
                    )
                }
                decodedTokens.append(
                    DecodedToken(
                        id: stepIdx,
                        tokenId: nextId,
                        text: text,
                        confidence: confidence,
                        category: TokenCategory.classify(text),
                        visionAttention: visionAttn
                    )
                )
            }

            currentTokenId = Int64(nextId)
            pastK = nextK
            pastV = nextV
            coverage = nextCov
            prevWeights = nextPrev
        }

        let decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000.0
        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000.0

        let latex = LatexTokenizer.detokenize(decodedTokens.map(\.text))
        let overallConf: Float
        if decodedTokens.isEmpty {
            overallConf = 0
        } else {
            let meanLog = decodedTokens.reduce(0.0) { $0 + log(Double(max($1.confidence, 1e-6))) } / Double(decodedTokens.count)
            overallConf = Float(exp(meanLog))
        }

        let gridH = visionMemLen > 0 ? max(1, strip.height / 16) : 0
        let gridW = gridH > 0 ? (visionMemLen / gridH) : 0

        return RecognitionResult(
            latex: latex,
            decodedTokens: decodedTokens,
            overallConfidence: overallConf,
            totalMs: totalMs,
            encodeMs: encodeMs,
            decodeMs: decodeMs,
            strokeCount: strokeInput?.strokeCount ?? strokes.count,
            pointCount: strokeInput?.pointCount ?? 0,
            stripHeight: strip.height,
            stripWidth: strip.width,
            visionGridW: gridW,
            visionGridH: gridH,
            stripLumBytes: strip.lumBytes,
            stripOrderBytes: strip.orderBytes,
            modality: modality,
            truncated: !hitEOS
        )
    }

    // MARK: - Math & Tensor Helpers

    private static func argmaxWithConfidence(_ logits: [Float]) -> (Int, Float) {
        var bestIdx = 0
        var bestVal = -Float.greatestFiniteMagnitude
        for i in 0..<logits.count {
            let v = logits[i]
            if v > bestVal {
                bestVal = v
                bestIdx = i
            }
        }
        var sumExp: Double = 0
        for v in logits {
            sumExp += exp(Double(v - bestVal))
        }
        let prob = sumExp > 0 ? Float(1.0 / sumExp) : 0
        return (bestIdx, prob)
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
