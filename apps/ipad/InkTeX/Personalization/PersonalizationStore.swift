import CryptoKit
import Foundation

// MARK: - Personalization Data Models

public struct LoRAAdapterMetadata: Codable, Sendable, Equatable {
    public let name: String
    public let importedAt: Date
    public let dtype: String
    public let loraLayers: [Int]
    public let loraRank: Int
    public let loraAlpha: Double
    public let strokeDim: Int
    public let loraAShape: [Int]
    public let loraBShape: [Int]
    public let trainingSamples: Int
    public let notes: String
    public let sha256: String
    public let digest64: UInt64
    public let fileBytes: Int
}

public struct LoadedLoRAWeights: Sendable {
    public let metadata: LoRAAdapterMetadata
    public let loraA: [Float]
    public let loraB: [Float]
}

public struct WriterProfile: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public var name: String
    public var createdAt: Date
    public var updatedAt: Date
    public var calibrationStrokes: [RawStroke]
    /// Cached 64-d writer embedding computed by `writer_style_v2.onnx` (invalidated automatically if model hash changes).
    public var cachedStyleVector: [Float]?
    public var cachedStyleModelSha: String?
    public var styleDigest: UInt64
    public var loraAdapter: LoRAAdapterMetadata?

    public var calibrationStrokeCount: Int { calibrationStrokes.count }

    /// Matches `WriterStyleEncoder` confidence weighting `n / (n + 10)`.
    public var styleConfidenceWeight: Double {
        let n = Double(min(calibrationStrokes.count, InkPreprocessorV2.styleMaxStrokes))
        return n > 0 ? (n / (n + 10.0)) : 0.0
    }
}

public struct CorrectionSample: Codable, Sendable, Identifiable, Equatable {
    public let id: UUID
    public let profileId: UUID
    public let timestamp: Date
    public let strokes: [RawStroke]
    public let predictedLatex: String
    public let correctedLatex: String
    public let geometryHash: UInt64
}

/// Immutable snapshot handed to `InkTeXEngine` on every personalization state change.
/// Hashing `profileDigest` and `adapterDigest` into `fingerprintStrokes` ensures cached
/// encoder memory and prefix guesses are invalidated immediately when a profile or LoRA changes.
public struct PersonalizationState: Sendable {
    public let profileId: UUID?
    public let profileName: String?
    public let calibrationStrokes: [RawStroke]
    public let cachedStyleVector: [Float]?
    public let cachedStyleModelSha: String?
    public let styleEnabled: Bool
    public let loraEnabled: Bool
    public let loraWeights: LoadedLoRAWeights?
    public let profileDigest: UInt64
    public let adapterDigest: UInt64

    public static let none = PersonalizationState(
        profileId: nil,
        profileName: nil,
        calibrationStrokes: [],
        cachedStyleVector: nil,
        cachedStyleModelSha: nil,
        styleEnabled: false,
        loraEnabled: false,
        loraWeights: nil,
        profileDigest: 0,
        adapterDigest: 0
    )
}

/// Wire format of `.inktexlora` JSON files emitted by `ml/v2/export_onnx_v2.py::export_lora_adapter`.
private struct PortableLoRAFile: Codable {
    let format: String
    let adapterName: String
    let createdAt: String?
    let dtype: String
    let loraLayers: [Int]
    let loraRank: Int
    let loraAlpha: Double
    let strokeDim: Int
    let loraAShape: [Int]
    let loraBShape: [Int]
    let trainingSamples: Int?
    let notes: String?
    let sha256: String?
    let loraAB64: String
    let loraBB64: String

    enum CodingKeys: String, CodingKey {
        case format
        case adapterName = "adapter_name"
        case createdAt = "created_at"
        case dtype
        case loraLayers = "lora_layers"
        case loraRank = "lora_rank"
        case loraAlpha = "lora_alpha"
        case strokeDim = "stroke_dim"
        case loraAShape = "lora_a_shape"
        case loraBShape = "lora_b_shape"
        case trainingSamples = "training_samples"
        case notes
        case sha256
        case loraAB64 = "lora_a_b64"
        case loraBB64 = "lora_b_b64"
    }
}

private struct PersistedStoreState: Codable {
    var version: Int = 1
    var activeProfileId: UUID?
    var styleProfileEnabled: Bool = true
    var loraAdapterEnabled: Bool = true
    var correctionCacheEnabled: Bool = false
    var profiles: [WriterProfile] = []
    var corrections: [CorrectionSample] = []
}

// MARK: - Personalization Store

@MainActor
public final class PersonalizationStore: ObservableObject {
    @Published public private(set) var profiles: [WriterProfile] = []
    @Published public var activeProfileId: UUID? {
        didSet {
            if oldValue != activeProfileId {
                persistAndNotify()
            }
        }
    }
    @Published public var styleProfileEnabled: Bool = true {
        didSet {
            if oldValue != styleProfileEnabled {
                persistAndNotify()
            }
        }
    }
    @Published public var loraAdapterEnabled: Bool = true {
        didSet {
            if oldValue != loraAdapterEnabled {
                persistAndNotify()
            }
        }
    }
    @Published public var correctionCacheEnabled: Bool = false {
        didSet {
            if oldValue != correctionCacheEnabled {
                persistAndNotify()
            }
        }
    }
    @Published public private(set) var corrections: [CorrectionSample] = []

    public var onStateChanged: ((PersonalizationState) -> Void)?

    private var loadedLoRACache: [UUID: LoadedLoRAWeights] = [:]

    public var activeProfile: WriterProfile? {
        guard let activeProfileId else { return nil }
        return profiles.first(where: { $0.id == activeProfileId })
    }

    public var activeCorrections: [CorrectionSample] {
        guard let id = activeProfileId else { return [] }
        return corrections.filter { $0.profileId == id }
    }

    public var isAnyPersonalizationActive: Bool {
        guard let p = activeProfile else { return false }
        let styleActive = styleProfileEnabled && !p.calibrationStrokes.isEmpty
        let loraActive = loraAdapterEnabled && p.loraAdapter != nil
        return styleActive || loraActive
    }

    public init() {
        loadFromDisk()
        if profiles.isEmpty {
            let defaultProfile = WriterProfile(
                id: UUID(),
                name: "My Handwriting",
                createdAt: Date(),
                updatedAt: Date(),
                calibrationStrokes: [],
                cachedStyleVector: nil,
                cachedStyleModelSha: nil,
                styleDigest: 0,
                loraAdapter: nil
            )
            profiles = [defaultProfile]
            activeProfileId = defaultProfile.id
            saveToDisk()
        }
    }

    // MARK: - Profile Lifecycle

    @discardableResult
    public func createProfile(name: String) -> WriterProfile {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let profile = WriterProfile(
            id: UUID(),
            name: trimmed.isEmpty ? "Writer \(profiles.count + 1)" : trimmed,
            createdAt: Date(),
            updatedAt: Date(),
            calibrationStrokes: [],
            cachedStyleVector: nil,
            cachedStyleModelSha: nil,
            styleDigest: 0,
            loraAdapter: nil
        )
        profiles.append(profile)
        activeProfileId = profile.id
        persistAndNotify()
        return profile
    }

    public func renameProfile(id: UUID, to newName: String) {
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let idx = profiles.firstIndex(where: { $0.id == id }) else { return }
        profiles[idx].name = trimmed
        profiles[idx].updatedAt = Date()
        persistAndNotify()
    }

    public func deleteProfile(id: UUID) {
        profiles.removeAll { $0.id == id }
        corrections.removeAll { $0.profileId == id }
        loadedLoRACache.removeValue(forKey: id)
        try? FileManager.default.removeItem(at: adapterBinaryURL(for: id))
        if activeProfileId == id {
            activeProfileId = profiles.first?.id
        }
        persistAndNotify()
    }

    // MARK: - Tier 1: Zero-Shot Writer Style Calibration

    public func updateCalibration(
        for profileId: UUID,
        strokes: [RawStroke],
        styleVector: [Float]? = nil,
        modelSha: String? = nil
    ) {
        guard let idx = profiles.firstIndex(where: { $0.id == profileId }) else { return }
        let capped = Array(strokes.suffix(InkPreprocessorV2.styleMaxStrokes))
        profiles[idx].calibrationStrokes = capped
        profiles[idx].cachedStyleVector = styleVector
        profiles[idx].cachedStyleModelSha = modelSha
        profiles[idx].styleDigest = Self.computeStrokesDigest(capped)
        profiles[idx].updatedAt = Date()
        persistAndNotify()
    }

    public func storeComputedStyleVector(for profileId: UUID, vector: [Float], modelSha: String) {
        guard let idx = profiles.firstIndex(where: { $0.id == profileId }) else { return }
        profiles[idx].cachedStyleVector = vector
        profiles[idx].cachedStyleModelSha = modelSha
        profiles[idx].updatedAt = Date()
        saveToDisk()
    }

    public func clearCalibration(for profileId: UUID) {
        guard let idx = profiles.firstIndex(where: { $0.id == profileId }) else { return }
        profiles[idx].calibrationStrokes = []
        profiles[idx].cachedStyleVector = nil
        profiles[idx].cachedStyleModelSha = nil
        profiles[idx].styleDigest = 0
        profiles[idx].updatedAt = Date()
        persistAndNotify()
    }

    // MARK: - Tier 2: Dynamic LoRA Adapter Import / Management

    public func importLoRAAdapter(
        from securityScopedURL: URL,
        for profileId: UUID,
        expectedManifest: ModelManifest.PersonalizationManifest? = nil
    ) throws -> LoRAAdapterMetadata {
        let accessed = securityScopedURL.startAccessingSecurityScopedResource()
        defer {
            if accessed {
                securityScopedURL.stopAccessingSecurityScopedResource()
            }
        }

        let rawData = try Data(contentsOf: securityScopedURL)
        let parsed = try JSONDecoder().decode(PortableLoRAFile.self, from: rawData)
        guard parsed.format == "inktex-lora-v2" else {
            throw Self.makeError("Unsupported LoRA adapter format '\(parsed.format)'. Expected 'inktex-lora-v2'.")
        }
        guard let aBytes = Data(base64Encoded: parsed.loraAB64),
              let bBytes = Data(base64Encoded: parsed.loraBB64) else {
            throw Self.makeError("Corrupt base64 weight data inside LoRA adapter file.")
        }

        if let expectedSha = parsed.sha256, !expectedSha.isEmpty {
            var hasher = SHA256()
            hasher.update(data: aBytes)
            hasher.update(data: bBytes)
            let actualSha = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            if actualSha.lowercased() != expectedSha.lowercased() {
                throw Self.makeError("LoRA adapter SHA-256 verification failed.")
            }
        }

        if let expected = expectedManifest {
            if parsed.loraLayers != expected.loraLayers || parsed.loraRank != expected.loraRank || parsed.strokeDim != expected.strokeDim {
                throw Self.makeError(
                    "Adapter shape mismatch: adapter has layers \(parsed.loraLayers), r=\(parsed.loraRank), d=\(parsed.strokeDim), " +
                    "but active model expects layers \(expected.loraLayers), r=\(expected.loraRank), d=\(expected.strokeDim)."
                )
            }
        }

        let expectedACount = parsed.loraAShape.reduce(1, *)
        let expectedBCount = parsed.loraBShape.reduce(1, *)
        let loraA = try Self.decodeFloatArray(from: aBytes, dtype: parsed.dtype, expectedCount: expectedACount)
        let loraB = try Self.decodeFloatArray(from: bBytes, dtype: parsed.dtype, expectedCount: expectedBCount)

        var hasher = SHA256()
        hasher.update(data: aBytes)
        hasher.update(data: bBytes)
        let digestHex = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        let digest64 = Self.fnv1a64(Data(digestHex.utf8))

        let meta = LoRAAdapterMetadata(
            name: parsed.adapterName,
            importedAt: Date(),
            dtype: parsed.dtype,
            loraLayers: parsed.loraLayers,
            loraRank: parsed.loraRank,
            loraAlpha: parsed.loraAlpha,
            strokeDim: parsed.strokeDim,
            loraAShape: parsed.loraAShape,
            loraBShape: parsed.loraBShape,
            trainingSamples: parsed.trainingSamples ?? 0,
            notes: parsed.notes ?? "",
            sha256: digestHex,
            digest64: digest64,
            fileBytes: rawData.count
        )

        // Save compact Float32 binary pair on disk for instant load (<1 ms)
        try saveAdapterBinary(loraA: loraA, loraB: loraB, for: profileId)
        loadedLoRACache[profileId] = LoadedLoRAWeights(metadata: meta, loraA: loraA, loraB: loraB)

        guard let idx = profiles.firstIndex(where: { $0.id == profileId }) else {
            throw Self.makeError("Active writer profile no longer exists.")
        }
        profiles[idx].loraAdapter = meta
        profiles[idx].updatedAt = Date()
        persistAndNotify()
        return meta
    }

    public func removeLoRAAdapter(for profileId: UUID) {
        guard let idx = profiles.firstIndex(where: { $0.id == profileId }) else { return }
        profiles[idx].loraAdapter = nil
        profiles[idx].updatedAt = Date()
        loadedLoRACache.removeValue(forKey: profileId)
        try? FileManager.default.removeItem(at: adapterBinaryURL(for: profileId))
        persistAndNotify()
    }

    // MARK: - Tier 3: Correction Capture, Optional Lookup & Training JSONL Export

    public func saveCorrection(
        strokes: [RawStroke],
        predictedLatex: String,
        correctedLatex: String
    ) {
        guard let profileId = activeProfileId else { return }
        let cleaned = correctedLatex.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty, !strokes.isEmpty else { return }
        let geomHash = InkPreprocessorV2.inkGeometryHash(strokes)
        let sample = CorrectionSample(
            id: UUID(),
            profileId: profileId,
            timestamp: Date(),
            strokes: strokes,
            predictedLatex: predictedLatex,
            correctedLatex: cleaned,
            geometryHash: geomHash
        )
        // Replace existing exact-geometry correction for this profile if present so latest edit wins
        corrections.removeAll { $0.profileId == profileId && $0.geometryHash == geomHash }
        corrections.insert(sample, at: 0)
        saveToDisk()
    }

    public func deleteCorrection(id: UUID) {
        corrections.removeAll { $0.id == id }
        saveToDisk()
    }

    public func clearCorrections(for profileId: UUID) {
        corrections.removeAll { $0.profileId == profileId }
        saveToDisk()
    }

    /// Optional fast (<0.1 ms) exact geometric lookup when `correctionCacheEnabled` is turned on.
    public func lookupCachedCorrection(for strokes: [RawStroke]) -> CorrectionSample? {
        guard correctionCacheEnabled, let profileId = activeProfileId, !strokes.isEmpty else { return nil }
        let hash = InkPreprocessorV2.inkGeometryHash(strokes)
        return corrections.first {
            $0.profileId == profileId && $0.geometryHash == hash && $0.strokes.count == strokes.count
        }
    }

    /// Export saved corrections (+ calibration strokes) as JSONL ready for RTX 4070 Ti LoRA fine-tuning.
    public func exportCorrectionsJSONL(for profileId: UUID) throws -> URL {
        guard let profile = profiles.first(where: { $0.id == profileId }) else {
            throw Self.makeError("Profile not found.")
        }
        let samples = corrections.filter { $0.profileId == profileId }
        let iso = ISO8601DateFormatter()

        func serializeStrokes(_ strokes: [RawStroke]) -> [[[Double]]] {
            strokes.map { stroke in
                stroke.points.map { pt in
                    if let t = pt.timeMs {
                        return [pt.x, pt.y, t]
                    }
                    return [pt.x, pt.y]
                }
            }
        }

        var lines: [String] = []
        lines.reserveCapacity(samples.count + 1)

        // First line: profile metadata + calibration support strokes (if enrolled)
        let headerObj: [String: Any] = [
            "record_type": "writer_profile_header",
            "format": "inktex-personalization-v2",
            "writer_id": "ipad:\(profile.id.uuidString.lowercased())",
            "profile_name": profile.name,
            "exported_at": iso.string(from: Date()),
            "calibration_strokes": serializeStrokes(profile.calibrationStrokes),
            "sample_count": samples.count,
        ]
        let headerData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        if let line = String(data: headerData, encoding: .utf8) {
            lines.append(line)
        }

        for sample in samples {
            let row: [String: Any] = [
                "record_type": "correction_sample",
                "id": sample.id.uuidString.lowercased(),
                "writer_id": "ipad:\(profile.id.uuidString.lowercased())",
                "profile_name": profile.name,
                "timestamp": iso.string(from: sample.timestamp),
                "latex": sample.correctedLatex,
                "predicted_latex": sample.predictedLatex,
                "strokes": serializeStrokes(sample.strokes),
            ]
            let rowData = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            if let line = String(data: rowData, encoding: .utf8) {
                lines.append(line)
            }
        }

        let safeName = profile.name
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "/", with: "-")
        let exportDir = storageDirectory.appendingPathComponent("exports", isDirectory: true)
        try FileManager.default.createDirectory(at: exportDir, withIntermediateDirectories: true)
        let outURL = exportDir.appendingPathComponent("inktex_\(safeName)_corrections.jsonl")
        try (lines.joined(separator: "\n") + "\n").write(to: outURL, atomically: true, encoding: .utf8)
        return outURL
    }

    // MARK: - Current Runtime State Snapshot

    public func currentState() -> PersonalizationState {
        guard let profile = activeProfile else { return .none }
        let useStyle = styleProfileEnabled && !profile.calibrationStrokes.isEmpty
        var loadedWeights: LoadedLoRAWeights? = nil
        if loraAdapterEnabled, let meta = profile.loraAdapter {
            if let cached = loadedLoRACache[profile.id], cached.metadata.sha256 == meta.sha256 {
                loadedWeights = cached
            } else if let disk = try? loadAdapterBinary(meta: meta, for: profile.id) {
                loadedLoRACache[profile.id] = disk
                loadedWeights = disk
            }
        }
        let useLoRA = loraAdapterEnabled && loadedWeights != nil
        return PersonalizationState(
            profileId: profile.id,
            profileName: profile.name,
            calibrationStrokes: useStyle ? profile.calibrationStrokes : [],
            cachedStyleVector: useStyle ? profile.cachedStyleVector : nil,
            cachedStyleModelSha: useStyle ? profile.cachedStyleModelSha : nil,
            styleEnabled: useStyle,
            loraEnabled: useLoRA,
            loraWeights: useLoRA ? loadedWeights : nil,
            profileDigest: useStyle ? profile.styleDigest : 0,
            adapterDigest: useLoRA ? (loadedWeights?.metadata.digest64 ?? 0) : 0
        )
    }

    // MARK: - Disk Persistence

    private var storageDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("personalization", isDirectory: true)
    }

    private var stateFileURL: URL {
        storageDirectory.appendingPathComponent("profiles_v2.json")
    }

    private func adapterBinaryURL(for profileId: UUID) -> URL {
        storageDirectory
            .appendingPathComponent("adapters", isDirectory: true)
            .appendingPathComponent("\(profileId.uuidString).bin")
    }

    private func persistAndNotify() {
        saveToDisk()
        onStateChanged?(currentState())
    }

    private func saveToDisk() {
        do {
            try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
            let state = PersistedStoreState(
                version: 1,
                activeProfileId: activeProfileId,
                styleProfileEnabled: styleProfileEnabled,
                loraAdapterEnabled: loraAdapterEnabled,
                correctionCacheEnabled: correctionCacheEnabled,
                profiles: profiles,
                corrections: corrections
            )
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(state)
            try data.write(to: stateFileURL, options: .atomic)
        } catch {
            // Non-fatal persistence error; state stays in memory
        }
    }

    private func loadFromDisk() {
        guard let data = try? Data(contentsOf: stateFileURL) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let state = try? decoder.decode(PersistedStoreState.self, from: data) else { return }
        profiles = state.profiles
        activeProfileId = state.activeProfileId ?? state.profiles.first?.id
        styleProfileEnabled = state.styleProfileEnabled
        loraAdapterEnabled = state.loraAdapterEnabled
        correctionCacheEnabled = state.correctionCacheEnabled
        corrections = state.corrections
    }

    private func saveAdapterBinary(loraA: [Float], loraB: [Float], for profileId: UUID) throws {
        let url = adapterBinaryURL(for: profileId)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var data = Data(capacity: (loraA.count + loraB.count) * MemoryLayout<Float>.stride)
        loraA.withUnsafeBufferPointer { buf in
            if let base = buf.baseAddress {
                data.append(UnsafeBufferPointer(start: UnsafeRawPointer(base).assumingMemoryBound(to: UInt8.self), count: buf.count * MemoryLayout<Float>.stride))
            }
        }
        loraB.withUnsafeBufferPointer { buf in
            if let base = buf.baseAddress {
                data.append(UnsafeBufferPointer(start: UnsafeRawPointer(base).assumingMemoryBound(to: UInt8.self), count: buf.count * MemoryLayout<Float>.stride))
            }
        }
        try data.write(to: url, options: .atomic)
    }

    private func loadAdapterBinary(meta: LoRAAdapterMetadata, for profileId: UUID) throws -> LoadedLoRAWeights {
        let url = adapterBinaryURL(for: profileId)
        let data = try Data(contentsOf: url)
        let aCount = meta.loraAShape.reduce(1, *)
        let bCount = meta.loraBShape.reduce(1, *)
        let totalFloats = aCount + bCount
        guard data.count == totalFloats * MemoryLayout<Float>.stride else {
            throw Self.makeError("Cached LoRA binary size mismatch.")
        }
        var all = [Float](repeating: 0, count: totalFloats)
        _ = all.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst)
        }
        return LoadedLoRAWeights(
            metadata: meta,
            loraA: Array(all[0..<aCount]),
            loraB: Array(all[aCount..<totalFloats])
        )
    }

    // MARK: - Binary Decoding & Hashing Helpers

    private static func decodeFloatArray(from data: Data, dtype: String, expectedCount: Int) throws -> [Float] {
        if dtype == "fp32" {
            guard data.count == expectedCount * 4 else {
                throw makeError("Expected \(expectedCount * 4) bytes for fp32 tensor, got \(data.count).")
            }
            var out = [Float](repeating: 0, count: expectedCount)
            data.withUnsafeBytes { raw in
                for i in 0..<expectedCount {
                    let bits = raw.load(fromByteOffset: i * 4, as: UInt32.self).littleEndian
                    out[i] = Float(bitPattern: bits)
                }
            }
            return out
        } else if dtype == "fp16" {
            guard data.count == expectedCount * 2 else {
                throw makeError("Expected \(expectedCount * 2) bytes for fp16 tensor, got \(data.count).")
            }
            var out = [Float](repeating: 0, count: expectedCount)
            data.withUnsafeBytes { raw in
                for i in 0..<expectedCount {
                    let bits = raw.load(fromByteOffset: i * 2, as: UInt16.self).littleEndian
                    out[i] = float16BitsToFloat32(bits)
                }
            }
            return out
        } else {
            throw makeError("Unsupported LoRA tensor dtype '\(dtype)'.")
        }
    }

    /// Exact IEEE 754 binary16 -> binary32 conversion (zero external Accelerate dependency).
    private static func float16BitsToFloat32(_ h: UInt16) -> Float {
        let sign = UInt32((h >> 15) & 0x1) << 31
        let exp = UInt32((h >> 10) & 0x1F)
        let frac = UInt32(h & 0x03FF)
        if exp == 0 {
            if frac == 0 {
                return Float(bitPattern: sign)
            }
            // Subnormal
            var m = frac
            var e: Int32 = -14
            while (m & 0x0400) == 0 {
                m <<= 1
                e -= 1
            }
            m &= 0x03FF
            let fExp = UInt32(e + 127) << 23
            return Float(bitPattern: sign | fExp | (m << 13))
        } else if exp == 31 {
            return Float(bitPattern: sign | 0x7F80_0000 | (frac << 13))
        }
        let fExp = (exp + 112) << 23
        return Float(bitPattern: sign | fExp | (frac << 13))
    }

    private static func computeStrokesDigest(_ strokes: [RawStroke]) -> UInt64 {
        guard !strokes.isEmpty else { return 0 }
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
            }
        }
        return h == 0 ? 1 : h
    }

    private static func fnv1a64(_ data: Data) -> UInt64 {
        var h: UInt64 = 14695981039346656037
        let prime: UInt64 = 1099511628211
        for b in data {
            h ^= UInt64(b)
            h = h &* prime
        }
        return h == 0 ? 1 : h
    }

    private static func makeError(_ message: String) -> NSError {
        NSError(domain: "InkTeX.Personalization", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
