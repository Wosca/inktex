import Foundation
import SwiftUI
import UIKit

public struct RawStrokePoint: Codable, Sendable, Equatable {
    public let x: Double
    public let y: Double
    public let timeMs: Double?

    public init(x: Double, y: Double, timeMs: Double?) {
        self.x = x
        self.y = y
        self.timeMs = timeMs
    }
}

public struct RawStroke: Codable, Sendable, Equatable {
    public let points: [RawStrokePoint]

    public init(points: [RawStrokePoint]) {
        self.points = points
    }
}

public enum ModalityMode: String, CaseIterable, Identifiable, Sendable {
    case bimodal
    case vision
    case stroke

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .bimodal: return "Bimodal"
        case .vision: return "Vision"
        case .stroke: return "Stroke"
        }
    }

    public var subtitle: String {
        switch self {
        case .bimodal: return "Ink + Strip"
        case .vision: return "Strip Only"
        case .stroke: return "Ink Only"
        }
    }
}

public enum CoreMLProfile: String, CaseIterable, Identifiable, Sendable {
    case off
    case visionOnly
    case encodersAndR2L
    case allGraphs

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .off: return "Off (M1 CPU)"
        case .visionOnly: return "Vision Only"
        case .encodersAndR2L: return "Encoders + R2L"
        case .allGraphs: return "All Graphs (Incl. Decoder)"
        }
    }

    public var description: String {
        switch self {
        case .off:
            return "Runs all 5 graphs on M1 performance cores + AMX."
        case .visionOnly:
            return "CoreML MLProgram (ANE/GPU) on ConvNeXt Vision Encoder; M1 CPU on transformers."
        case .encodersAndR2L:
            return "CoreML MLProgram (ANE/GPU) on Vision, Stroke, Memory Proj & R2L; M1 CPU on step decoder. Best all-round for FP16."
        case .allGraphs:
            return "CoreML MLProgram (ANE/GPU) on all 5 graphs including the fixed-shape static KV-cache decoder."
        }
    }
}

public enum DecodingMode: String, CaseIterable, Identifiable, Sendable {
    case greedy
    case beam
    case beamR2L

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .greedy: return "Greedy"
        case .beam: return "Beam"
        case .beamR2L: return "Beam + R2L"
        }
    }
}

public struct BeamCandidate: Sendable, Identifiable {
    public let id: Int
    public let latex: String
    public let l2rScore: Double
    public let r2lScore: Double?
    public let combinedScore: Double
    public let isWinner: Bool
    public let wasPromotedByR2L: Bool
}

public enum StripColorMode: String, CaseIterable, Identifiable, Sendable {
    case strokeOrder
    case confidence
    case symbolType

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .strokeOrder: return "Order"
        case .confidence: return "Confidence"
        case .symbolType: return "Type"
        }
    }
}

public enum TokenCategory: String, CaseIterable, Identifiable, Sendable {
    case number
    case variable
    case operatorSymbol
    case structure
    case bracket

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .number: return "Number"
        case .variable: return "Variable"
        case .operatorSymbol: return "Operator"
        case .structure: return "Structure"
        case .bracket: return "Bracket"
        }
    }

    public var rgb: (Float, Float, Float) {
        switch self {
        case .number: return (56, 189, 248)        // Sky blue
        case .variable: return (52, 211, 153)      // Emerald
        case .operatorSymbol: return (251, 191, 36) // Amber
        case .structure: return (192, 132, 252)    // Purple
        case .bracket: return (148, 163, 184)      // Slate
        }
    }

    public var color: Color {
        let (r, g, b) = rgb
        return Color(red: Double(r) / 255.0, green: Double(g) / 255.0, blue: Double(b) / 255.0)
    }

    public static func classify(_ token: String) -> TokenCategory {
        if token.count == 1, let ch = token.first {
            if ch.isNumber || ch == "." { return .number }
            if ch.isLetter { return .variable }
            if "+-=*/<>!:".contains(ch) { return .operatorSymbol }
            if "{}^_&\\".contains(ch) { return .structure }
            if "()[]|,".contains(ch) { return .bracket }
        }
        let greekOrVars: Set<String> = [
            "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\varepsilon",
            "\\zeta", "\\eta", "\\theta", "\\vartheta", "\\iota", "\\kappa",
            "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma",
            "\\tau", "\\upsilon", "\\phi", "\\varphi", "\\chi", "\\psi", "\\omega",
            "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi", "\\Sigma",
            "\\Upsilon", "\\Phi", "\\Psi", "\\Omega", "\\infty", "\\partial", "\\nabla",
            "\\hbar", "\\ell", "\\prime",
        ]
        if greekOrVars.contains(token) { return .variable }

        let operators: Set<String> = [
            "\\cdot", "\\times", "\\div", "\\pm", "\\mp", "\\leq", "\\geq", "\\le", "\\ge",
            "\\neq", "\\ne", "\\approx", "\\equiv", "\\sim", "\\simeq", "\\propto",
            "\\in", "\\notin", "\\subset", "\\subseteq", "\\supset", "\\supseteq",
            "\\cup", "\\cap", "\\rightarrow", "\\to", "\\leftarrow", "\\Rightarrow",
            "\\Leftrightarrow", "\\leftrightarrow", "\\mapsto", "\\forall", "\\exists",
            "\\neg", "\\land", "\\lor", "\\gt", "\\lt",
        ]
        if operators.contains(token) { return .operatorSymbol }

        let brackets: Set<String> = [
            "\\left", "\\right", "\\{", "\\}", "\\langle", "\\rangle", "\\lfloor", "\\rfloor",
            "\\lceil", "\\rceil", "\\|", "\\mid",
        ]
        if brackets.contains(token) { return .bracket }

        return .structure
    }
}

public struct DecodedToken: Sendable, Identifiable {
    public let id: Int
    public let tokenId: Int
    public let text: String
    public let confidence: Float
    public let category: TokenCategory
    /// Head-averaged cross-attention over the vision grid `[W_f * H_f]` (column-major), if vision was active.
    public let visionAttention: [Float]?

    public var isVisualToken: Bool {
        text != "{" && text != "}" && text != "^" && text != "_" && text != "&" && text != "\\\\"
    }
}

public struct ModelManifest: Codable, Sendable {
    public struct VisionConfig: Codable, Sendable {
        public let height: Int
        public let minWidth: Int
        public let maxWidth: Int
        public let widthBuckets: [Int]?
        public let padding: Int
        public let widthMultiple: Int
        public let channels: Int
        public let targetGlyphPx: Double?
        public let verticalStride: Int?
        public let horizontalStride: Int?

        public var resolvedWidthBuckets: [Int] {
            if let widthBuckets, !widthBuckets.isEmpty {
                return widthBuckets.sorted()
            }
            return [256, 512, 768, 1024].filter { $0 >= minWidth && $0 <= maxWidth }
        }

        public var resolvedTargetGlyphPx: Double {
            targetGlyphPx ?? 38.0
        }

        enum CodingKeys: String, CodingKey {
            case height
            case minWidth = "min_width"
            case maxWidth = "max_width"
            case widthBuckets = "width_buckets"
            case padding
            case widthMultiple = "width_multiple"
            case channels
            case targetGlyphPx = "target_glyph_px"
            case verticalStride = "vertical_stride"
            case horizontalStride = "horizontal_stride"
        }
    }

    public struct PreprocessingManifest: Codable, Sendable {
        public let maxPoints: Int
        public let minPointsPerStroke: Int
        public let resamplingEnabled: Bool
        public let resampleSpacing: Double
        public let minResampledPointsPerStroke: Int
        public let useTiming: Bool
        public let usePressure: Bool
        public let maxDeltaMs: Double
        public let normalization: String
        public let maxPointsPerStroke: Int

        // InkTeX v2 sub-stroke chunking parameters (optional for v1 backward compatibility)
        public let chunkSize: Int?
        public let chunkHop: Int?
        public let chunkBuckets: [Int]?
        public let maxChunks: Int?
        public let dedupeEps: Double?
        public let coordClip: Double?
        public let pointFeatureDim: Int?
        public let chunkMetaDim: Int?

        public var resolvedChunkSize: Int { chunkSize ?? 8 }
        public var resolvedChunkHop: Int { chunkHop ?? 7 }
        public var resolvedChunkBuckets: [Int] {
            if let chunkBuckets, !chunkBuckets.isEmpty {
                return chunkBuckets.sorted()
            }
            return [128, 256, 512, 768]
        }
        public var resolvedMaxChunks: Int { maxChunks ?? (resolvedChunkBuckets.last ?? 768) }
        public var resolvedDedupeEps: Double { dedupeEps ?? 1e-3 }
        public var resolvedCoordClip: Double { coordClip ?? 32.0 }

        enum CodingKeys: String, CodingKey {
            case maxPoints = "max_points"
            case minPointsPerStroke = "min_points_per_stroke"
            case resamplingEnabled = "resampling_enabled"
            case resampleSpacing = "resample_spacing"
            case minResampledPointsPerStroke = "min_resampled_points_per_stroke"
            case useTiming = "use_timing"
            case usePressure = "use_pressure"
            case maxDeltaMs = "max_delta_ms"
            case normalization
            case maxPointsPerStroke = "max_points_per_stroke"
            case chunkSize = "chunk_size"
            case chunkHop = "chunk_hop"
            case chunkBuckets = "chunk_buckets"
            case maxChunks = "max_chunks"
            case dedupeEps = "dedupe_eps"
            case coordClip = "coord_clip"
            case pointFeatureDim = "point_feature_dim"
            case chunkMetaDim = "chunk_meta_dim"
        }
    }

    public struct PersonalizationManifest: Codable, Sendable {
        public let styleDim: Int
        public let styleMaxStrokes: Int
        public let stylePoints: Int
        public let stylePointDim: Int
        public let filmLayers: [Int]
        public let dynamicLora: Bool
        public let loraLayers: [Int]
        public let loraRank: Int
        public let loraAlpha: Double
        public let strokeDim: Int

        enum CodingKeys: String, CodingKey {
            case styleDim = "style_dim"
            case styleMaxStrokes = "style_max_strokes"
            case stylePoints = "style_points"
            case stylePointDim = "style_point_dim"
            case filmLayers = "film_layers"
            case dynamicLora = "dynamic_lora"
            case loraLayers = "lora_layers"
            case loraRank = "lora_rank"
            case loraAlpha = "lora_alpha"
            case strokeDim = "stroke_dim"
        }
    }

    public struct MetricsSummary: Codable, Sendable {
        public let normalizedExact: Double?
        public let strictExact: Double?
        public let tokenCer: Double?

        enum CodingKeys: String, CodingKey {
            case normalizedExact = "normalized_exact"
            case strictExact = "strict_exact"
            case tokenCer = "token_cer"
        }
    }

    public struct FileMetadata: Codable, Sendable {
        public let file: String
        public let bytes: Int
        public let sha256: String
    }

    public let format: String
    public let modelName: String
    public let preset: String
    public let stage: Int?
    public let epoch: Int?
    public let quantization: String?
    public let hasR2L: Bool?
    public let sharedR2LDecoder: Bool?
    public let supportsBatchedDecoder: Bool?
    public let supportsPrefixVerify: Bool?
    public let staticKvLen: Int?
    public let staticShapes: Bool?
    public let r2lDecoderLayers: Int?
    public let dModel: Int
    public let decoderLayers: Int
    public let attentionHeads: Int
    public let headDim: Int
    public let strokeLayers: Int?
    public let strokeDim: Int?
    public let vocabSize: Int
    public let bosId: Int
    public let bosL2RId: Int?
    public let bosR2LId: Int?
    public let eosId: Int
    public let padId: Int
    public let unkId: Int
    public let maxPoints: Int
    public let maxTokens: Int
    public let inputDim: Int
    public let vision: VisionConfig
    public let preprocessing: PreprocessingManifest
    public let personalization: PersonalizationManifest?
    public let tokens: [String]
    public let metrics: MetricsSummary?
    public let files: [String: FileMetadata]?

    public var isV2Architecture: Bool {
        format == "inktex-onnx-v2"
    }

    public var resolvedBosL2RId: Int {
        bosL2RId ?? bosId
    }

    public var resolvedBosR2LId: Int? {
        if let bosR2LId { return bosR2LId }
        if isV2Architecture { return 2 }
        return nil
    }

    enum CodingKeys: String, CodingKey {
        case format
        case modelName = "model_name"
        case preset
        case stage
        case epoch
        case quantization
        case hasR2L = "has_r2l"
        case sharedR2LDecoder = "shared_r2l_decoder"
        case supportsBatchedDecoder = "supports_batched_decoder"
        case supportsPrefixVerify = "supports_prefix_verify"
        case staticKvLen = "static_kv_len"
        case staticShapes = "static_shapes"
        case r2lDecoderLayers = "r2l_decoder_layers"
        case dModel = "d_model"
        case decoderLayers = "decoder_layers"
        case attentionHeads = "attention_heads"
        case headDim = "head_dim"
        case strokeLayers = "stroke_layers"
        case strokeDim = "stroke_dim"
        case vocabSize = "vocab_size"
        case bosId = "bos_id"
        case bosL2RId = "bos_l2r_id"
        case bosR2LId = "bos_r2l_id"
        case eosId = "eos_id"
        case padId = "pad_id"
        case unkId = "unk_id"
        case maxPoints = "max_points"
        case maxTokens = "max_tokens"
        case inputDim = "input_dim"
        case vision
        case preprocessing
        case personalization
        case tokens
        case metrics
        case files
    }
}

public struct MemoryComponent: Sendable, Identifiable {
    public let id: String
    public let name: String
    public let ramMB: Double
    public let diskMB: Double
    public let coreMLEnabled: Bool
    public let colorRGB: (Double, Double, Double)

    public var color: Color {
        Color(red: colorRGB.0, green: colorRGB.1, blue: colorRGB.2)
    }
}

public struct MemorySnapshot: Sendable {
    public let baselineMB: Double
    public let components: [MemoryComponent]
    public let currentFootprintMB: Double
    public let peakFootprintMB: Double
    public let lastInferenceDeltaMB: Double
    public let availableHeadroomMB: Double

    public var componentsSumMB: Double {
        components.reduce(0) { $0 + $1.ramMB }
    }

    public var runtimeOverheadMB: Double {
        max(0, currentFootprintMB - baselineMB - componentsSumMB)
    }

    public var totalDiskMB: Double {
        components.reduce(0) { $0 + $1.diskMB }
    }

    public static let empty = MemorySnapshot(
        baselineMB: 0,
        components: [],
        currentFootprintMB: 0,
        peakFootprintMB: 0,
        lastInferenceDeltaMB: 0,
        availableHeadroomMB: 0
    )
}

public struct RecognitionResult: Sendable {
    public let latex: String
    public let decodedTokens: [DecodedToken]
    public let overallConfidence: Float
    public let totalMs: Double
    public let encodeMs: Double
    public let decodeMs: Double
    public let r2lMs: Double
    public let strokeCount: Int
    public let pointCount: Int
    public let stripHeight: Int
    public let stripWidth: Int
    public let visionGridW: Int
    public let visionGridH: Int
    public let stripLumBytes: [UInt8]
    public let stripOrderBytes: [UInt8]
    public let modality: ModalityMode
    public let decodingMode: DecodingMode
    public let r2lUsed: Bool
    public let r2lChangedWinner: Bool
    public let prefixVerifiedCount: Int
    public let encoderCacheHit: Bool
    public let beams: [BeamCandidate]
    public let truncated: Bool
    public let isV2Model: Bool
    public let chunkBucket: Int?
    public let activeProfileName: String?
    public let usedWriterStyle: Bool
    public let usedLoRAAdapter: Bool
    public let matchedCorrectionCache: Bool

    public var tokens: [String] { decodedTokens.map(\.text) }
}
