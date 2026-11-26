import SwiftUI
import UniformTypeIdentifiers

/// Dedicated sheet for InkTeX v2 Personalization:
///   - Tier 1: Zero-backprop Writer Style calibration pad (`writer_style_v2.onnx` -> FiLM in ink blocks 7/10/13)
///   - Tier 2: Hot-swappable Ink-Only LoRA adapter (`.inktexlora`, `r=8` in ink blocks 10-13)
///   - Tier 3: Personal correction capture, optional instant cache, and `.jsonl` export for PC/Mac LoRA fine-tuning
struct PersonalizationSheet: View {
    @ObservedObject var store: PersonalizationStore
    @ObservedObject var engine: InkTeXEngine
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme

    @StateObject private var calCanvas = PencilCanvasController()
    @State private var calStrokes: [RawStroke] = []
    @State private var isComputingStyle = false
    @State private var styleFeedback: String?
    @State private var styleError: String?

    @State private var showLoRAImporter = false
    @State private var loraError: String?
    @State private var loraSuccess: String?

    @State private var showNewProfileAlert = false
    @State private var newProfileName = ""
    @State private var exportedJSONLURL: URL?
    @State private var exportError: String?
    @State private var showOnboarding = false

    private static let samplePrompts = [
        "x^2 + \\alpha y = \\frac{1}{\\sqrt{2}}",
        "\\int_0^\\infty e^{-t}\\,dt = 1",
        "\\sum_{i=1}^{n} \\beta_i \\neq 0",
    ]

    var body: some View {
        NavigationStack {
            List {
                if !engine.isV2Model {
                    Section {
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: "info.circle.fill")
                                .foregroundStyle(.orange)
                            VStack(alignment: .leading, spacing: 4) {
                                Text("InkTeX v1 Bundle Loaded")
                                    .font(.subheadline.weight(.semibold))
                                Text("You can already enroll calibration strokes and collect personal corrections now. Zero-shot Writer Style (FiLM) and hot-swappable LoRA activate automatically once an InkTeX v2 bundle is loaded.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }

                // 0. Guided Adaptive Onboarding Hero
                Section {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 10) {
                            Image(systemName: "wand.and.stars")
                                .font(.title2.weight(.semibold))
                                .foregroundStyle(Color.accentColor)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Guided Handwriting Onboarding")
                                    .font(.headline)
                                Text("Evaluates your writing across 8 math categories, auto-calibrates Tier 1 FiLM, drills missed symbols, and builds your Tier 2 LoRA dataset.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }

                        Button {
                            showOnboarding = true
                        } label: {
                            Label(
                                (store.activeProfile?.calibrationStrokeCount ?? 0) > 0
                                    ? "Run Adaptive Drill / Re-Calibrate"
                                    : "Start Guided Onboarding (~2–5 min)",
                                systemImage: "pencil.tip.crop.circle.badge.plus"
                            )
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.regular)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Recommended Setup")
                }

                // 1. Writer Profile Switcher
                Section("Writer Profile") {
                    Picker("Active Writer", selection: $store.activeProfileId) {
                        ForEach(store.profiles) { profile in
                            Text(profile.name).tag(Optional(profile.id))
                        }
                    }

                    HStack {
                        Button {
                            newProfileName = ""
                            showNewProfileAlert = true
                        } label: {
                            Label("New Writer Profile", systemImage: "person.badge.plus")
                        }

                        Spacer()

                        if store.profiles.count > 1, let active = store.activeProfile {
                            Button(role: .destructive) {
                                store.deleteProfile(id: active.id)
                            } label: {
                                Text("Delete")
                                    .font(.subheadline)
                            }
                        }
                    }
                }

                // 2. Tier 1: Zero-Backprop On-Device Style Profile (FiLM)
                Section {
                    Toggle("Enable Zero-Shot Style (FiLM)", isOn: $store.styleProfileEnabled)

                    if let profile = store.activeProfile {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Enrolled Calibration Strokes")
                                    .font(.subheadline)
                                Text(
                                    profile.calibrationStrokeCount > 0
                                        ? "\(profile.calibrationStrokeCount) / \(InkPreprocessorV2.styleMaxStrokes) strokes · \(Int(round(profile.styleConfidenceWeight * 100)))% style weight"
                                        : "Not calibrated yet (exact base model behavior)"
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if profile.calibrationStrokeCount > 0 {
                                Button("Reset", role: .destructive) {
                                    store.clearCalibration(for: profile.id)
                                    styleFeedback = "Cleared style calibration."
                                }
                                .font(.caption.weight(.semibold))
                            }
                        }

                        // Compact interactive Apple Pencil Calibration Pad
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Draw 8–20 quick math strokes below (e.g., x² + α/y, ∫, ∑, √):")
                                .font(.caption)
                                .foregroundStyle(.secondary)

                            ZStack {
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .fill(colorScheme == .dark ? Color(white: 0.12) : Color.white)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                                            .strokeBorder(Color.secondary.opacity(0.25), lineWidth: 1)
                                    )

                                PencilCanvasView(
                                    controller: calCanvas,
                                    pencilOnly: false,
                                    isDark: colorScheme == .dark,
                                    onDrawingChanged: { strokes in
                                        calStrokes = strokes
                                    }
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                                if calCanvas.strokeCount == 0 {
                                    Text("Write a short equation here with Apple Pencil")
                                        .font(.caption)
                                        .foregroundStyle(.tertiary)
                                        .allowsHitTesting(false)
                                }
                            }
                            .frame(height: 150)

                            HStack(spacing: 10) {
                                Text("\(calStrokes.count) strokes drawn")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)

                                Spacer()

                                Button("Clear Pad") {
                                    calCanvas.clear()
                                    calStrokes = []
                                }
                                .font(.caption.weight(.medium))
                                .disabled(calStrokes.isEmpty)

                                Button {
                                    enrollCalibration(for: profile)
                                } label: {
                                    if isComputingStyle {
                                        ProgressView().controlSize(.small)
                                    } else {
                                        Label("Enroll Style", systemImage: "wand.and.stars")
                                            .font(.caption.weight(.semibold))
                                    }
                                }
                                .buttonStyle(.borderedProminent)
                                .controlSize(.small)
                                .disabled(calStrokes.isEmpty || isComputingStyle)
                            }

                            if let styleFeedback {
                                Text(styleFeedback)
                                    .font(.caption)
                                    .foregroundStyle(.green)
                            }
                            if let styleError {
                                Text(styleError)
                                    .font(.caption)
                                    .foregroundStyle(.red)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                } header: {
                    Text("Tier 1 · Instant Writer Style (FiLM)")
                } footer: {
                    Text("Runs writer_style_v2.onnx once on your strokes to compute a 64-d style embedding, cached in memory with 0 ms per-equation overhead.")
                }

                // 3. Tier 2: Hot-Swappable Ink-Only LoRA Adapter
                Section {
                    Toggle("Enable Ink-Only LoRA Adapter", isOn: $store.loraAdapterEnabled)

                    if let profile = store.activeProfile {
                        if let meta = profile.loraAdapter {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Label(meta.name, systemImage: "cpu.fill")
                                        .font(.subheadline.weight(.semibold))
                                    Spacer()
                                    Text("r=\(meta.loraRank) · \(meta.dtype.uppercased())")
                                        .font(.caption.monospacedDigit())
                                        .padding(.horizontal, 7)
                                        .padding(.vertical, 2)
                                        .background(Color.purple.opacity(0.16))
                                        .foregroundStyle(.purple)
                                        .clipShape(Capsule())
                                }
                                Text(
                                    String(
                                        format: "Blocks %@ · %.0f KB · SHA %@",
                                        meta.loraLayers.map { String($0 + 1) }.joined(separator: ","),
                                        Double(meta.fileBytes) / 1024.0,
                                        String(meta.sha256.prefix(10))
                                    )
                                )
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                                if meta.trainingSamples > 0 {
                                    Text("Trained on \(meta.trainingSamples) personal samples")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 2)

                            Button(role: .destructive) {
                                store.removeLoRAAdapter(for: profile.id)
                                loraSuccess = nil
                                loraError = nil
                            } label: {
                                Label("Remove LoRA Adapter", systemImage: "trash")
                            }
                        } else {
                            Text("No LoRA adapter installed for this profile. Base stroke weights are used.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Button {
                            showLoRAImporter = true
                        } label: {
                            Label("Import .inktexlora Adapter from Files…", systemImage: "square.and.arrow.down")
                        }

                        if let loraSuccess {
                            Text(loraSuccess)
                                .font(.caption)
                                .foregroundStyle(.green)
                        }
                        if let loraError {
                            Text(loraError)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                    }
                } header: {
                    Text("Tier 2 · Personal LoRA Adapter (Hot-Swap)")
                } footer: {
                    Text("Adapts Q/V/Out projections in stroke blocks 10–13 (~220 KB FP16) while keeping Vision & LaTeX Decoder frozen. Switches instantly with zero model reload.")
                }

                // 4. Tier 3: Personal Corrections & Export for Training
                Section {
                    Toggle("Instant Exact-Geometry Correction Cache", isOn: $store.correctionCacheEnabled)

                    if let profile = store.activeProfile {
                        let items = store.activeCorrections
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Saved Training Samples")
                                Text("\(store.activeOnboardingSampleCount) onboarding · \(store.activeManualCorrectionCount) manual corrections")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text("\(items.count)")
                                .font(.callout.monospacedDigit().weight(.semibold))
                                .foregroundStyle(.secondary)
                        }

                        if !items.isEmpty {
                            Button {
                                do {
                                    exportedJSONLURL = try store.exportCorrectionsJSONL(for: profile.id)
                                    exportError = nil
                                } catch {
                                    exportError = error.localizedDescription
                                }
                            } label: {
                                Label("Prepare JSONL for LoRA Training", systemImage: "doc.zipper")
                            }

                            if let url = exportedJSONLURL {
                                ShareLink(item: url) {
                                    Label("AirDrop / Share \(url.lastPathComponent)", systemImage: "square.and.arrow.up")
                                        .font(.subheadline.weight(.semibold))
                                }
                            }

                            if let exportError {
                                Text(exportError)
                                    .font(.caption)
                                    .foregroundStyle(.red)
                            }

                            ForEach(items.prefix(15)) { sample in
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(sample.correctedLatex)
                                        .font(.system(.callout, design: .monospaced).weight(.medium))
                                        .lineLimit(2)
                                    HStack(spacing: 8) {
                                        if sample.predictedLatex != sample.correctedLatex {
                                            Text("was: \(sample.predictedLatex)")
                                                .lineLimit(1)
                                        }
                                        Text("· \(sample.strokes.count) strokes")
                                    }
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                }
                                .swipeActions(edge: .trailing) {
                                    Button(role: .destructive) {
                                        store.deleteCorrection(id: sample.id)
                                    } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                            }
                        } else {
                            Text("Tap \"Correct\" on any recognized equation to log handwriting corrections. 30–100 samples are enough for a 30-second LoRA fine-tune.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Tier 3 · Corrections & Training Export")
                } footer: {
                    Text("Turn off the instant correction cache while benchmarking raw model accuracy.")
                }
            }
            .navigationTitle("Personalization")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .fullScreenCover(isPresented: $showOnboarding) {
                PersonalizationOnboardingView(store: store, engine: engine)
            }
            .alert("New Writer Profile", isPresented: $showNewProfileAlert) {
                TextField("Profile Name", text: $newProfileName)
                Button("Cancel", role: .cancel) {}
                Button("Create") {
                    _ = store.createProfile(name: newProfileName)
                }
            }
            .fileImporter(
                isPresented: $showLoRAImporter,
                allowedContentTypes: [.json, .data],
                allowsMultipleSelection: false
            ) { result in
                loraError = nil
                loraSuccess = nil
                guard let profileId = store.activeProfileId else { return }
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    do {
                        let meta = try store.importLoRAAdapter(
                            from: url,
                            for: profileId,
                            expectedManifest: engine.loadedBundle?.manifest.personalization
                        )
                        loraSuccess = "Loaded \(meta.name) (r=\(meta.loraRank), \(meta.loraLayers.count) blocks)."
                    } catch {
                        loraError = error.localizedDescription
                    }
                case .failure(let error):
                    loraError = error.localizedDescription
                }
            }
        }
    }

    private func enrollCalibration(for profile: WriterProfile) {
        let strokes = calStrokes
        guard !strokes.isEmpty else { return }
        styleError = nil
        styleFeedback = nil

        if !engine.isV2Model {
            store.updateCalibration(for: profile.id, strokes: strokes)
            styleFeedback = "Saved \(min(strokes.count, InkPreprocessorV2.styleMaxStrokes)) strokes. Embedding will compute automatically on v2 load."
            return
        }

        isComputingStyle = true
        Task {
            defer { isComputingStyle = false }
            do {
                let (vec, sha, ms) = try await engine.computeWriterStyle(strokes: strokes)
                store.updateCalibration(for: profile.id, strokes: strokes, styleVector: vec, modelSha: sha)
                let norm = sqrt(vec.reduce(0.0) { $0 + Double($1 * $1) })
                styleFeedback = String(
                    format: "Enrolled %d strokes in %.1f ms (style ‖z‖₂ = %.2f).",
                    min(strokes.count, InkPreprocessorV2.styleMaxStrokes),
                    ms,
                    norm
                )
            } catch {
                styleError = error.localizedDescription
            }
        }
    }
}

/// Quick inline sheet to fix a recognized LaTeX expression and save it to the active writer profile.
struct CorrectionEditorSheet: View {
    let strokes: [RawStroke]
    let predictedLatex: String
    let isDark: Bool
    var onSave: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var editedLatex: String

    init(strokes: [RawStroke], predictedLatex: String, isDark: Bool, onSave: @escaping (String) -> Void) {
        self.strokes = strokes
        self.predictedLatex = predictedLatex
        self.isDark = isDark
        self.onSave = onSave
        _editedLatex = State(initialValue: predictedLatex)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                KaTeXWebView(latex: editedLatex, isDark: isDark)
                    .frame(height: 140)
                    .padding(10)
                    .background(Color(uiColor: .secondarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

                VStack(alignment: .leading, spacing: 6) {
                    Text("Correct LaTeX (\(strokes.count) strokes)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    TextEditor(text: $editedLatex)
                        .font(.system(.body, design: .monospaced))
                        .autocorrectionDisabled(true)
                        .textInputAutocapitalization(.never)
                        .frame(minHeight: 100)
                        .padding(8)
                        .background(Color(uiColor: .secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }

                Spacer()
            }
            .padding(16)
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationTitle("Save Correction")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        onSave(editedLatex)
                        dismiss()
                    }
                    .disabled(editedLatex.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
    }
}
