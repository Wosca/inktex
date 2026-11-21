import Foundation

public struct LocatedModelBundle: Sendable {
    public let directory: URL
    public let sourceDescription: String
    public let manifest: ModelManifest

    public var visionEncoderURL: URL { directory.appendingPathComponent("vision_encoder.onnx") }
    public var strokeEncoderURL: URL { directory.appendingPathComponent("stroke_encoder.onnx") }
    public var memoryProjURL: URL { directory.appendingPathComponent("memory_proj.onnx") }
    public var decoderStepURL: URL { directory.appendingPathComponent("decoder_step.onnx") }
    public var r2lScorerURL: URL? {
        let url = directory.appendingPathComponent("r2l_scorer.onnx")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

public enum ModelBundleLocator {
    public static let requiredFiles = [
        "manifest.json",
        "vision_encoder.onnx",
        "stroke_encoder.onnx",
        "memory_proj.onnx",
        "decoder_step.onnx",
    ]
    public static let optionalFiles = [
        "r2l_scorer.onnx",
    ]

    public static var documentsBundleDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("model-bundle", isDirectory: true)
    }

    public static func locate() -> LocatedModelBundle? {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]

        var candidates: [(URL, String)] = [
            (documentsBundleDirectory, "Imported (Files / AirDrop)"),
            (docs, "Documents"),
        ]

        if let bundledSubdir = Bundle.main.url(forResource: "model-bundle", withExtension: nil) {
            candidates.append((bundledSubdir, "App Bundle"))
        }
        if let resourceURL = Bundle.main.resourceURL {
            candidates.append((resourceURL, "App Bundle"))
        }

        for (dir, label) in candidates {
            if let loaded = validateBundle(at: dir, sourceDescription: label) {
                return loaded
            }
        }
        return nil
    }

    public static func validateBundle(at directory: URL, sourceDescription: String) -> LocatedModelBundle? {
        let fm = FileManager.default
        for filename in requiredFiles {
            if !fm.fileExists(atPath: directory.appendingPathComponent(filename).path) {
                return nil
            }
        }
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(ModelManifest.self, from: data) else {
            return nil
        }
        return LocatedModelBundle(directory: directory, sourceDescription: sourceDescription, manifest: manifest)
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
                        "Selected folder is missing one or more required files: \(requiredFiles.joined(separator: ", "))"
                ]
            )
        }

        let fm = FileManager.default
        let dest = documentsBundleDirectory
        if fm.fileExists(atPath: dest.path) {
            try fm.removeItem(at: dest)
        }
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)

        for filename in requiredFiles {
            let srcFile = validated.directory.appendingPathComponent(filename)
            let dstFile = dest.appendingPathComponent(filename)
            try fm.copyItem(at: srcFile, to: dstFile)
        }
        for filename in optionalFiles {
            let srcFile = validated.directory.appendingPathComponent(filename)
            if fm.fileExists(atPath: srcFile.path) {
                let dstFile = dest.appendingPathComponent(filename)
                try fm.copyItem(at: srcFile, to: dstFile)
            }
        }

        guard let installed = validateBundle(at: dest, sourceDescription: "Imported (Files / AirDrop)") else {
            throw NSError(
                domain: "InkTeX.ModelBundle",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to verify imported model bundle."]
            )
        }
        return installed
    }

    public static func removeImportedBundle() throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: documentsBundleDirectory.path) {
            try fm.removeItem(at: documentsBundleDirectory)
        }
    }
}
