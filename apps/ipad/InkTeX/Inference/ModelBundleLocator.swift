import Foundation

public struct LocatedModelBundle: Sendable, Identifiable, Equatable {
    public var id: String { directory.standardizedFileURL.path }
    public let directory: URL
    public let sourceDescription: String
    public let manifest: ModelManifest

    public var isV2: Bool { manifest.isV2Architecture }

    public var writerStyleURL: URL? {
        guard isV2 else { return nil }
        let url = directory.appendingPathComponent(manifest.files?["writer_style"]?.file ?? "writer_style_v2.onnx")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    public var visionEncoderURL: URL {
        if isV2 {
            return directory.appendingPathComponent(manifest.files?["vision_encoder"]?.file ?? "vision_encoder_v2.onnx")
        }
        return directory.appendingPathComponent(manifest.files?["vision_encoder"]?.file ?? "vision_encoder.onnx")
    }

    public var strokeEncoderURL: URL {
        if isV2 {
            return directory.appendingPathComponent(manifest.files?["stroke_encoder"]?.file ?? "stroke_encoder_v2.onnx")
        }
        return directory.appendingPathComponent(manifest.files?["stroke_encoder"]?.file ?? "stroke_encoder.onnx")
    }

    public var memoryProjURL: URL {
        if isV2 {
            return directory.appendingPathComponent(manifest.files?["fusion_memproj"]?.file ?? "fusion_memproj_v2.onnx")
        }
        return directory.appendingPathComponent(manifest.files?["memory_proj"]?.file ?? "memory_proj.onnx")
    }

    public var decoderStepURL: URL {
        if isV2 {
            return directory.appendingPathComponent(manifest.files?["decoder_step"]?.file ?? "decoder_step_v2.onnx")
        }
        return directory.appendingPathComponent(manifest.files?["decoder_step"]?.file ?? "decoder_step.onnx")
    }

    public var r2lScorerURL: URL? {
        let filename = manifest.files?["r2l_scorer"]?.file ?? "r2l_scorer.onnx"
        let url = directory.appendingPathComponent(filename)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    public var displayTitle: String {
        var parts: [String] = [manifest.modelName]
        if isV2 && !manifest.modelName.lowercased().contains("v2") {
            parts.append("v2")
        }
        if let stage = manifest.stage, let epoch = manifest.epoch {
            if !manifest.modelName.contains("E\(epoch)") {
                parts.append("S\(stage)·E\(epoch)")
            }
        }
        if let exact = manifest.metrics?.normalizedExact {
            parts.append(String(format: "%.1f%%", exact * 100))
        }
        return parts.joined(separator: " · ")
    }

    public static func == (lhs: LocatedModelBundle, rhs: LocatedModelBundle) -> Bool {
        lhs.id == rhs.id
    }
}

public enum ModelBundleLocator {
    public static let requiredFilesV1 = [
        "manifest.json",
        "vision_encoder.onnx",
        "stroke_encoder.onnx",
        "memory_proj.onnx",
        "decoder_step.onnx",
    ]
    public static let optionalFilesV1 = [
        "r2l_scorer.onnx",
    ]
    public static let requiredFilesV2 = [
        "manifest.json",
        "writer_style_v2.onnx",
        "stroke_encoder_v2.onnx",
        "vision_encoder_v2.onnx",
        "fusion_memproj_v2.onnx",
        "decoder_step_v2.onnx",
    ]
    public static let optionalFilesV2 = [
        "r2l_scorer.onnx",
    ]
    /// Kept for legacy references.
    public static let requiredFiles = requiredFilesV1
    public static let optionalFiles = optionalFilesV1

    public static var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    public static var documentsBundleDirectory: URL {
        documentsDirectory.appendingPathComponent("model-bundle", isDirectory: true)
    }

    public static var importedModelsDirectory: URL {
        documentsDirectory.appendingPathComponent("models", isDirectory: true)
    }

    /// Discovers every valid ONNX bundle across Documents/ and the App Bundle,
    /// sorted with the highest-accuracy / latest-stage checkpoint first.
    public static func locateAll() -> [LocatedModelBundle] {
        let fm = FileManager.default
        let docs = documentsDirectory
        var candidates: [(URL, String)] = []

        // 1. Multi-model folder in Documents/models/*
        if let items = try? fm.contentsOfDirectory(at: importedModelsDirectory, includingPropertiesForKeys: [.isDirectoryKey]) {
            for url in items {
                candidates.append((url, "Imported (\(url.lastPathComponent))"))
            }
        }

        // 2. Any subfolder in Documents/ (including model-bundle, model-bundle-e17, etc.)
        if let items = try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: [.isDirectoryKey]) {
            for url in items where url.lastPathComponent != "models" {
                candidates.append((url, "Files (\(url.lastPathComponent))"))
            }
        }
        candidates.append((docs, "Documents"))

        // 3. App Bundle subfolders (any bundled model-bundle* folder reference) + bundle root
        if let resourceURL = Bundle.main.resourceURL {
            if let items = try? fm.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: [.isDirectoryKey]) {
                for url in items {
                    candidates.append((url, "App Bundle (\(url.lastPathComponent))"))
                }
            }
            candidates.append((resourceURL, "App Bundle"))
        }

        var seenPaths: Set<String> = []
        var found: [LocatedModelBundle] = []
        for (dir, label) in candidates {
            let stdPath = dir.standardizedFileURL.path
            if seenPaths.contains(stdPath) { continue }
            if let valid = validateBundle(at: dir, sourceDescription: label) {
                seenPaths.insert(stdPath)
                found.append(valid)
            }
        }

        // Sort: InkTeX v2 first, then highest stage, highest normalized_exact, highest epoch, static KV, then FP16 preference
        found.sort { a, b in
            let v2A = a.isV2 ? 1 : 0
            let v2B = b.isV2 ? 1 : 0
            if v2A != v2B { return v2A > v2B }
            let sA = a.manifest.stage ?? 0
            let sB = b.manifest.stage ?? 0
            if sA != sB { return sA > sB }
            let exA = a.manifest.metrics?.normalizedExact ?? -1
            let exB = b.manifest.metrics?.normalizedExact ?? -1
            if abs(exA - exB) > 1e-6 { return exA > exB }
            let epA = a.manifest.epoch ?? 0
            let epB = b.manifest.epoch ?? 0
            if epA != epB { return epA > epB }
            let v3A = (a.manifest.staticKvLen != nil) ? 1 : 0
            let v3B = (b.manifest.staticKvLen != nil) ? 1 : 0
            if v3A != v3B { return v3A > v3B }
            let fp16A = (a.manifest.quantization == "fp16") ? 1 : 0
            let fp16B = (b.manifest.quantization == "fp16") ? 1 : 0
            return fp16A > fp16B
        }
        return found
    }

    public static func locate(preferredPath: String? = nil) -> LocatedModelBundle? {
        let all = locateAll()
        if let preferredPath, let match = all.first(where: { $0.id == preferredPath }) {
            return match
        }
        return all.first
    }

    public static func validateBundle(at directory: URL, sourceDescription: String) -> LocatedModelBundle? {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard fm.fileExists(atPath: manifestURL.path),
              let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(ModelManifest.self, from: data) else {
            return nil
        }
        let bundle = LocatedModelBundle(directory: directory, sourceDescription: sourceDescription, manifest: manifest)
        if bundle.isV2 {
            guard let styleURL = bundle.writerStyleURL, fm.fileExists(atPath: styleURL.path) else {
                return nil
            }
            let requiredURLs = [
                bundle.strokeEncoderURL,
                bundle.visionEncoderURL,
                bundle.memoryProjURL,
                bundle.decoderStepURL,
            ]
            for url in requiredURLs where !fm.fileExists(atPath: url.path) {
                return nil
            }
        } else {
            for filename in requiredFilesV1 {
                if !fm.fileExists(atPath: directory.appendingPathComponent(filename).path) {
                    return nil
                }
            }
        }
        return bundle
    }

    public static func importBundle(from securityScopedFolder: URL) throws -> LocatedModelBundle {
        let accessed = securityScopedFolder.startAccessingSecurityScopedResource()
        defer {
            if accessed {
                securityScopedFolder.stopAccessingSecurityScopedResource()
            }
        }

        guard let validated = validateBundle(at: securityScopedFolder, sourceDescription: "Imported") else {
            throw NSError(
                domain: "InkTeX.ModelBundle",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Selected folder is missing manifest.json or required v1/v2 ONNX files."
                ]
            )
        }

        let fm = FileManager.default
        try fm.createDirectory(at: importedModelsDirectory, withIntermediateDirectories: true)
        let folderName = securityScopedFolder.lastPathComponent.isEmpty ? "model-bundle" : securityScopedFolder.lastPathComponent
        let dest = importedModelsDirectory.appendingPathComponent(folderName, isDirectory: true)
        if fm.fileExists(atPath: dest.path) {
            try fm.removeItem(at: dest)
        }
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)

        var filesToCopy: [URL] = [validated.directory.appendingPathComponent("manifest.json")]
        if validated.isV2 {
            if let ws = validated.writerStyleURL { filesToCopy.append(ws) }
            filesToCopy.append(contentsOf: [
                validated.strokeEncoderURL,
                validated.visionEncoderURL,
                validated.memoryProjURL,
                validated.decoderStepURL,
            ])
        } else {
            for filename in requiredFilesV1 where filename != "manifest.json" {
                filesToCopy.append(validated.directory.appendingPathComponent(filename))
            }
        }
        if let r2l = validated.r2lScorerURL {
            filesToCopy.append(r2l)
        }

        var copiedNames: Set<String> = []
        for srcFile in filesToCopy {
            let name = srcFile.lastPathComponent
            if copiedNames.insert(name).inserted && fm.fileExists(atPath: srcFile.path) {
                try fm.copyItem(at: srcFile, to: dest.appendingPathComponent(name))
            }
        }

        guard let installed = validateBundle(at: dest, sourceDescription: "Imported (\(folderName))") else {
            throw NSError(
                domain: "InkTeX.ModelBundle",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to verify imported model bundle."]
            )
        }
        return installed
    }

    public static func removeImportedBundle(_ bundle: LocatedModelBundle) throws {
        let fm = FileManager.default
        let docsPath = documentsDirectory.standardizedFileURL.path
        let targetPath = bundle.directory.standardizedFileURL.path
        guard targetPath.hasPrefix(docsPath), targetPath != docsPath else { return }
        if fm.fileExists(atPath: targetPath) {
            try fm.removeItem(atPath: targetPath)
        }
    }
}
