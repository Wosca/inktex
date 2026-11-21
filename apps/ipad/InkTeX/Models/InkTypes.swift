import Foundation
import UIKit

public struct RawStrokePoint: Sendable {
    public let x: Double
    public let y: Double
    public let timeMs: Double?

    public init(x: Double, y: Double, timeMs: Double?) {
        self.x = x
        self.y = y
        self.timeMs = timeMs
    }
}

public struct RawStroke: Sendable {
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

public struct ModelManifest: Codable, Sendable {
    public struct VisionConfig: Codable, Sendable {
        public let height: Int
        public let minWidth: Int
        public let maxWidth: Int
        public let padding: Int
        public let widthMultiple: Int
        public let channels: Int

        enum CodingKeys: String, CodingKey {
            case height
            case minWidth = "min_width"
            case maxWidth = "max_width"
            case padding
            case widthMultiple = "width_multiple"
            case channels
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

    public let format: String
    public let modelName: String
    public let preset: String
    public let stage: Int?
    public let epoch: Int?
    public let quantization: String?
    public let dModel: Int
    public let decoderLayers: Int
    public let attentionHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let bosId: Int
    public let eosId: Int
    public let padId: Int
    public let unkId: Int
    public let maxPoints: Int
    public let maxTokens: Int
    public let inputDim: Int
    public let vision: VisionConfig
    public let preprocessing: PreprocessingManifest
    public let tokens: [String]
    public let metrics: MetricsSummary?

    enum CodingKeys: String, CodingKey {
        case format
        case modelName = "model_name"
        case preset
        case stage
        case epoch
        case quantization
        case dModel = "d_model"
        case decoderLayers = "decoder_layers"
        case attentionHeads = "attention_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case bosId = "bos_id"
        case eosId = "eos_id"
        case padId = "pad_id"
        case unkId = "unk_id"
        case maxPoints = "max_points"
        case maxTokens = "max_tokens"
        case inputDim = "input_dim"
        case vision
        case preprocessing
        case tokens
        case metrics
    }
}

public struct RecognitionResult: Sendable {
    public let latex: String
    public let tokens: [String]
    public let totalMs: Double
    public let encodeMs: Double
    public let decodeMs: Double
    public let strokeCount: Int
    public let pointCount: Int
    public let stripWidth: Int
    public let modality: ModalityMode
    public let truncated: Bool
    public let debugStripPNGData: Data?
}
