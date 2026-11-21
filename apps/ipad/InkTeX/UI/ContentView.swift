import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var engine = InkTeXEngine()
    @StateObject private var canvasController = PencilCanvasController()
    @Environment(\.colorScheme) private var colorScheme

    @State private var modality: ModalityMode = .bimodal
    @State private var decodingMode: DecodingMode = .beamR2L
    @State private var pencilOnly: Bool = false
    @State private var autoRecognize: Bool = true
    @State private var expandPreview: Bool = false
    @State private var stripColorMode: StripColorMode = .confidence
    @State private var showInspectorSheet: Bool = false
    @State private var showModelSheet: Bool = false

    @State private var latestResult: RecognitionResult?
    @State private var isRefiningBeam: Bool = false
    @State private var errorMessage: String?
    @State private var copiedFeedback: Bool = false
    @State private var recognitionTask: Task<Void, Never>?
    @State private var requestGeneration: Int = 0

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if case .missingModel = engine.status {
                    missingModelBanner
                } else if case .failed(let msg) = engine.status {
                    errorBanner("Model load failed: \(msg)")
                } else if let errorMessage {
                    errorBanner(errorMessage)
                }

                // Scrollable Rendered LaTeX + Raw LaTeX Card
                previewPanel
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
                    .padding(.bottom, 10)

                // Main Apple Pencil Drawing Canvas
                ZStack {
                    DotGridBackground()
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .strokeBorder(Color.secondary.opacity(0.22), lineWidth: 1)
                        )

                    PencilCanvasView(
                        controller: canvasController,
                        pencilOnly: pencilOnly,
                        isDark: colorScheme == .dark,
                        onDrawingChanged: handleDrawingChanged
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

                    if canvasController.strokeCount == 0 {
                        VStack(spacing: 8) {
                            Image(systemName: "pencil.tip.crop.circle")
                                .font(.system(size: 38))
                                .foregroundStyle(.tertiary)
                            Text("Draw an equation with Apple Pencil")
                                .font(.headline)
                                .foregroundStyle(.secondary)
                            Text(pencilOnly ? "Apple Pencil only (palm rejection active)" : "Apple Pencil or finger")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .allowsHitTesting(false)
                    }
                }
                .padding(.horizontal, 16)

                // Dedicated Bottom Action Bar (instant Clear/Undo + Decoding Switcher)
                canvasActionBar
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                // Clean Settings icon only on top-left (no title/dot clutter)
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showModelSheet = true
                    } label: {
                        Image(systemName: settingsIconName)
                            .foregroundStyle(settingsIconColor)
                    }
                    .help("Model & Settings")
                }

                ToolbarItem(placement: .principal) {
                    Picker("Modality", selection: $modality) {
                        ForEach(ModalityMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 240)
                    .onChange(of: modality) { _, _ in
                        let strokes = canvasController.extractRawStrokes()
                        if !strokes.isEmpty {
                            triggerRecognition(strokes, immediate: true)
                        }
                    }
                }

                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        pencilOnly.toggle()
                    } label: {
                        Image(systemName: pencilOnly ? "applepencil.tip" : "hand.draw")
                    }
                    .help(pencilOnly ? "Pencil Only" : "Pencil + Touch")

                    Button {
                        autoRecognize.toggle()
                    } label: {
                        Image(systemName: autoRecognize ? "bolt.fill" : "bolt.slash")
                    }
                    .help(autoRecognize ? "Auto-recognize ON" : "Manual recognize")

                    Button {
                        showInspectorSheet = true
                    } label: {
                        Image(systemName: "eye")
                    }
                    .disabled(latestResult == nil)
                    .help("Inspect Vision Strip, Confidence & Beams")
                }
            }
            .sheet(isPresented: $showModelSheet) {
                ModelSetupSheet(engine: engine)
            }
            .sheet(isPresented: $showInspectorSheet) {
                if let result = latestResult {
                    InspectorSheet(result: result, stripColorMode: $stripColorMode)
                        .presentationDetents([.medium, .large])
                }
            }
        }
    }

    private var settingsIconName: String {
        switch engine.status {
        case .missingModel, .failed: return "exclamationmark.circle"
        default: return "gearshape"
        }
    }

    private var settingsIconColor: Color {
        switch engine.status {
        case .missingModel: return .orange
        case .failed: return .red
        default: return .primary
        }
    }

    private var canvasActionBar: some View {
        HStack(spacing: 10) {
            Picker("Decoding", selection: $decodingMode) {
                Text("Greedy").tag(DecodingMode.greedy)
                Text("Beam \(engine.beamWidth)").tag(DecodingMode.beam)
                Text(engine.hasR2LScorer ? "Beam+R2L" : "Beam+R2L*").tag(DecodingMode.beamR2L)
            }
            .pickerStyle(.segmented)
            .frame(width: 245)
            .controlSize(.small)
            .onChange(of: decodingMode) { _, _ in
                let strokes = canvasController.extractRawStrokes()
                if !strokes.isEmpty {
                    triggerRecognition(strokes, immediate: true)
                }
            }

            if let result = latestResult {
                HStack(spacing: 6) {
                    Text(String(format: "%.0f%% · %.0f ms", result.overallConfidence * 100, result.totalMs))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)

                    if isRefiningBeam {
                        ProgressView()
                            .controlSize(.mini)
                    } else if result.r2lChangedWinner {
                        Text("R2L Pick")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.purple.opacity(0.22))
                            .foregroundStyle(.purple)
                            .clipShape(Capsule())
                    } else if result.prefixVerifiedCount > 0 {
                        Text("\(result.prefixVerifiedCount) cached")
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.18))
                            .foregroundStyle(.green)
                            .clipShape(Capsule())
                    }
                }
            }

            Spacer()

            if !autoRecognize {
                Button {
                    triggerRecognition(canvasController.extractRawStrokes(), immediate: true)
                } label: {
                    Label("Recognize", systemImage: "sparkles")
                        .font(.subheadline.weight(.semibold))
                }
                .buttonStyle(.borderedProminent)
                .disabled(canvasController.strokeCount == 0 || engine.status != .ready)
            }

            Button {
                canvasController.undo()
            } label: {
                Image(systemName: "arrow.uturn.backward")
                    .font(.body.weight(.medium))
            }
            .buttonStyle(.bordered)
            .disabled(!canvasController.canUndo)

            Button {
                canvasController.redo()
            } label: {
                Image(systemName: "arrow.uturn.forward")
                    .font(.body.weight(.medium))
            }
            .buttonStyle(.bordered)
            .disabled(!canvasController.canRedo)

            Button(role: .destructive) {
                clearAll()
            } label: {
                Label("Clear", systemImage: "trash")
                    .font(.subheadline.weight(.medium))
            }
            .buttonStyle(.bordered)
            .disabled(canvasController.strokeCount == 0 && latestResult == nil)
        }
    }

    private func clearAll() {
        requestGeneration += 1
        recognitionTask?.cancel()
        recognitionTask = nil
        isRefiningBeam = false
        canvasController.clear()
        engine.clearCaches()
        latestResult = nil
        errorMessage = nil
    }

    private var previewPanel: some View {
        VStack(spacing: 8) {
            ZStack(alignment: .topTrailing) {
                KaTeXWebView(
                    latex: latestResult?.latex ?? "",
                    isDark: colorScheme == .dark
                )
                .frame(height: expandPreview ? 240 : 124)

                HStack(spacing: 6) {
                    if engine.isRecognizing && !isRefiningBeam {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            expandPreview.toggle()
                        }
                    } label: {
                        Image(systemName: expandPreview ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                            .font(.caption.weight(.semibold))
                            .padding(6)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .help(expandPreview ? "Compact Preview" : "Expand Preview for Multi-line Math")
                }
                .padding(6)
            }

            Divider()

            HStack(alignment: .center, spacing: 10) {
                if expandPreview {
                    ScrollView(.vertical, showsIndicators: true) {
                        Text(latestResult?.latex.isEmpty == false ? latestResult!.latex : "LaTeX output appears here")
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(latestResult?.latex.isEmpty == false ? .primary : .tertiary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 84)
                } else {
                    ScrollView(.horizontal, showsIndicators: true) {
                        Text(latestResult?.latex.isEmpty == false ? latestResult!.latex : "LaTeX output appears here")
                            .font(.system(.callout, design: .monospaced))
                            .foregroundStyle(latestResult?.latex.isEmpty == false ? .primary : .tertiary)
                            .textSelection(.enabled)
                            .lineLimit(1)
                    }
                }

                Button {
                    guard let tex = latestResult?.latex, !tex.isEmpty else { return }
                    UIPasteboard.general.string = tex
                    copiedFeedback = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) {
                        copiedFeedback = false
                    }
                } label: {
                    Label(copiedFeedback ? "Copied" : "Copy", systemImage: copiedFeedback ? "checkmark" : "doc.on.doc")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .disabled(latestResult?.latex.isEmpty != false)
            }
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.secondary.opacity(0.15), lineWidth: 1)
        )
    }

    private var missingModelBanner: some View {
        HStack {
            Image(systemName: "tray.and.arrow.down.fill")
                .foregroundStyle(.orange)
            Text("No model bundle found. Tap to import your exported `model-bundle` folder from Files / AirDrop.")
                .font(.subheadline)
            Spacer()
            Button("Import…") {
                showModelSheet = true
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.orange.opacity(0.14))
    }

    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.caption)
                .lineLimit(2)
            Spacer()
            Button("Dismiss") { errorMessage = nil }
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.red.opacity(0.14))
    }

    private func handleDrawingChanged(_ strokes: [RawStroke]) {
        if strokes.isEmpty {
            requestGeneration += 1
            recognitionTask?.cancel()
            isRefiningBeam = false
            engine.clearCaches()
            latestResult = nil
            errorMessage = nil
            return
        }
        guard autoRecognize else { return }
        triggerRecognition(strokes, immediate: false)
    }

    private func triggerRecognition(_ strokes: [RawStroke], immediate: Bool) {
        requestGeneration += 1
        let myGen = requestGeneration
        recognitionTask?.cancel()
        isRefiningBeam = false
        guard engine.status == .ready else { return }

        let selectedModality = modality
        let selectedDecoding = decodingMode
        let useTwoPhase = !immediate && engine.liveTwoPhase && selectedDecoding != .greedy

        recognitionTask = Task {
            if !immediate {
                try? await Task.sleep(nanoseconds: 240_000_000)
            }
            if Task.isCancelled || myGen != requestGeneration { return }

            if useTwoPhase {
                // Phase 1: Instant Prefix-Verified Greedy preview (~100-180ms!)
                do {
                    let fastResult = try await engine.recognize(
                        strokes: strokes,
                        modality: selectedModality,
                        decodingMode: .greedy
                    )
                    if !Task.isCancelled && myGen == requestGeneration {
                        latestResult = fastResult
                        errorMessage = nil
                    }
                } catch {
                    if !Task.isCancelled && myGen == requestGeneration {
                        errorMessage = error.localizedDescription
                    }
                    return
                }

                // Wait briefly for user to finish writing before running Beam + R2L polish
                try? await Task.sleep(nanoseconds: 550_000_000)
                if Task.isCancelled || myGen != requestGeneration { return }

                // Phase 2: Batched Beam + R2L polish (reuses cached encoder memory = 0ms encode!)
                isRefiningBeam = true
                defer {
                    if myGen == requestGeneration {
                        isRefiningBeam = false
                    }
                }
                do {
                    let refined = try await engine.recognize(
                        strokes: strokes,
                        modality: selectedModality,
                        decodingMode: selectedDecoding
                    )
                    if !Task.isCancelled && myGen == requestGeneration {
                        latestResult = refined
                        errorMessage = nil
                    }
                } catch {
                    if !Task.isCancelled && myGen == requestGeneration {
                        errorMessage = error.localizedDescription
                    }
                }
            } else {
                do {
                    let result = try await engine.recognize(
                        strokes: strokes,
                        modality: selectedModality,
                        decodingMode: selectedDecoding
                    )
                    if !Task.isCancelled && myGen == requestGeneration {
                        latestResult = result
                        errorMessage = nil
                    }
                } catch {
                    if !Task.isCancelled && myGen == requestGeneration {
                        errorMessage = error.localizedDescription
                    }
                }
            }
        }
    }
}

private struct DotGridBackground: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Canvas { context, size in
            let bg = colorScheme == .dark ? Color(white: 0.11) : Color.white
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(bg))

            let spacing: CGFloat = 24
            let dotColor = colorScheme == .dark ? Color.white.opacity(0.11) : Color.black.opacity(0.09)
            var y: CGFloat = spacing
            while y < size.height {
                var x: CGFloat = spacing
                while x < size.width {
                    let rect = CGRect(x: x - 1, y: y - 1, width: 2, height: 2)
                    context.fill(Path(ellipseIn: rect), with: .color(dotColor))
                    x += spacing
                }
                y += spacing
            }
        }
    }
}
