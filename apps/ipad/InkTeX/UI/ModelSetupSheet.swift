import SwiftUI
import UniformTypeIdentifiers

struct ModelSetupSheet: View {
    @ObservedObject var engine: InkTeXEngine
    @Environment(\.dismiss) private var dismiss
    @State private var showingFolderPicker = false
    @State private var importError: String?

    var body: some View {
        NavigationStack {
            List {
                Section("Active Model") {
                    if let bundle = engine.loadedBundle {
                        LabeledContent("Model", value: bundle.manifest.modelName)
                        if let stage = bundle.manifest.stage, let epoch = bundle.manifest.epoch {
                            LabeledContent("Checkpoint", value: "Stage \(stage) · Epoch \(epoch)")
                        }
                        if let exact = bundle.manifest.metrics?.normalizedExact {
                            LabeledContent("Val Exact", value: String(format: "%.1f%%", exact * 100))
                        }
                        LabeledContent("Vocab", value: "\(bundle.manifest.vocabSize) tokens")
                        LabeledContent("Quantization", value: bundle.manifest.quantization ?? "fp32")
                        LabeledContent("R2L Reranker", value: engine.hasR2LScorer ? "Active (3-layer)" : "Not in bundle")
                        LabeledContent("Source", value: bundle.sourceDescription)
                    } else {
                        VStack(alignment: .leading, spacing: 6) {
                            Label("No model bundle loaded", systemImage: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .font(.headline)
                            Text("Import your exported `model-bundle` folder from Files / iCloud Drive / AirDrop, or bundle it directly in Xcode.")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }

                Section("Beam Search & R2L Reranking") {
                    Picker("Beam Width", selection: $engine.beamWidth) {
                        Text("2").tag(2)
                        Text("4 (Recommended)").tag(4)
                        Text("6").tag(6)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("R2L Weight (λ)")
                            Spacer()
                            Text(String(format: "%.2f", engine.r2lWeight))
                                .font(.callout.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $engine.r2lWeight, in: 0.0...1.5, step: 0.05)
                    }

                    Text("Combined candidate score = avg_logP(L2R) + λ · avg_logP(R2L). Higher λ gives the right-to-left decoder more say on nested structure and end-of-equation tokens.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Runtime Acceleration") {
                    Toggle("CoreML EP for Vision Encoder", isOn: Binding(
                        get: { engine.useCoreMLForVision },
                        set: { newValue in
                            engine.useCoreMLForVision = newValue
                            engine.reloadModel()
                        }
                    ))
                    Text("Uses Apple Neural Engine / GPU for ConvNeXt-Tiny while keeping dynamic-shape transformer steps on M1 performance cores.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Swap / Import Model Bundle") {
                    Button {
                        showingFolderPicker = true
                    } label: {
                        Label("Import model-bundle Folder from Files…", systemImage: "folder.badge.plus")
                    }

                    Button {
                        engine.reloadModel()
                    } label: {
                        Label("Reload Model", systemImage: "arrow.clockwise")
                    }

                    if engine.loadedBundle?.sourceDescription.contains("Imported") == true {
                        Button(role: .destructive) {
                            do {
                                try ModelBundleLocator.removeImportedBundle()
                                engine.reloadModel()
                            } catch {
                                importError = error.localizedDescription
                            }
                        } label: {
                            Label("Remove Imported Bundle", systemImage: "trash")
                        }
                    }

                    if let importError {
                        Text(importError)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                Section("Bundle Files") {
                    ForEach(ModelBundleLocator.requiredFiles, id: \.self) { file in
                        Text(file)
                            .font(.system(.caption, design: .monospaced))
                    }
                    ForEach(ModelBundleLocator.optionalFiles, id: \.self) { file in
                        HStack {
                            Text(file)
                                .font(.system(.caption, design: .monospaced))
                            Spacer()
                            Text("optional (R2L)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("InkTeX Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .fileImporter(
                isPresented: $showingFolderPicker,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false
            ) { result in
                importError = nil
                switch result {
                case .success(let urls):
                    guard let folder = urls.first else { return }
                    do {
                        _ = try ModelBundleLocator.importBundle(from: folder)
                        engine.reloadModel()
                    } catch {
                        importError = error.localizedDescription
                    }
                case .failure(let error):
                    importError = error.localizedDescription
                }
            }
        }
    }
}
