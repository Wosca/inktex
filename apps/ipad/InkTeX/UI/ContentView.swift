import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var engine = InkTeXEngine()
    @StateObject private var canvasController = PencilCanvasController()
    @Environment(\.colorScheme) private var colorScheme

    @State private var modality: ModalityMode = .bimodal
    @State private var pencilOnly: Bool = false
    @State private var autoRecognize: Bool = true
    @State private var showInspector: Bool = false
    @State private var expandPreview: Bool = false
    @State private var stripColorMode: StripColorMode = .confidence
    @State private var showModelSheet: Bool = false

    @State private var latestResult: RecognitionResult?
    @State private var errorMessage: String?
    @State private var copiedFeedback: Bool = false
    @State private var debounceTask: Task<Void, Never>?
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

                // Optional Vision Strip & Token Inspector Drawer
                if showInspector, let result = latestResult {
                    inspectorBar(result)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                // Main Apple Pencil Drawing Canvas (no overlaid buttons competing with PencilKit gestures)
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

                // Dedicated Bottom Action Bar (instant tap delivery, zero PencilKit gesture delay)
                canvasActionBar
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Text("InkTeX")
                        .font(.headline.weight(.bold))
                        .fixedSize()
                }

                ToolbarItem(placement: .topBarLeading) {
                    modelStatusChip
                }

                ToolbarItem(placement: .principal) {
                    Picker("Modality", selection: $modality) {
                        ForEach(ModalityMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 250)
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
                        withAnimation(.easeInOut(duration: 0.2)) {
                            showInspector.toggle()
                        }
                    } label: {
                        Image(systemName: showInspector ? "waveform.path.ecg.rectangle.fill" : "waveform.path.ecg.rectangle")
                    }
                    .help("Toggle Vision Strip & Confidence Inspector")

                    Button {
                        showModelSheet = true
                    } label: {
                        Image(systemName: "cpu")
                    }
                    .help("Model Settings & Import")
                }
            }
            .sheet(isPresented: $showModelSheet) {
                ModelSetupSheet(engine: engine)
            }
        }
    }

    private var canvasActionBar: some View {
        HStack(spacing: 10) {
            if let result = latestResult {
                Label(
                    String(format: "%.0f%% conf · %.0f ms · %d tok", result.overallConfidence * 100, result.totalMs, result.decodedTokens.count),
                    systemImage: "sparkles"
                )
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            } else {
                Text(pencilOnly ? "Pencil Only" : "Pencil + Touch")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
        debounceTask?.cancel()
        debounceTask = nil
        canvasController.clear()
        latestResult = nil
        errorMessage = nil
    }

    private var modelStatusChip: some View {
        Button {
            showModelSheet = true
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "circle.fill")
                    .font(.system(size: 7))
                    .foregroundStyle(statusColor)
                Text(statusLabel)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .fixedSize()
        }
    }

    private var statusColor: Color {
        switch engine.status {
        case .ready: return .green
        case .loading: return .orange
        case .missingModel: return .yellow
        case .failed: return .red
        }
    }

    private var statusLabel: String {
        switch engine.status {
        case .loading: return "Loading…"
        case .missingModel: return "Import Model"
        case .failed: return "Load Error"
        case .ready:
            if let m = engine.loadedBundle?.manifest, let stage = m.stage, let epoch = m.epoch {
                return "S\(stage) · E\(epoch)"
            }
            return "Ready"
        }
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
                    if engine.isRecognizing {
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

    private func inspectorBar(_ result: RecognitionResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("Strip \(result.stripWidth)×\(result.stripHeight) · \(result.strokeCount) strokes · \(result.pointCount) pts")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                Spacer()

                Picker("Strip Color", selection: $stripColorMode) {
                    ForEach(StripColorMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 230)
                .controlSize(.small)

                Text(String(format: "Enc %.0fms · Dec %.0fms", result.encodeMs, result.decodeMs))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            if let uiImg = StripRasterizer.renderVisualization(result: result, mode: stripColorMode) {
                Image(uiImage: uiImg)
                    .interpolation(.none)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 74)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            // Compact color legend matching selected visualization mode
            inspectorLegend(for: stripColorMode, modality: result.modality)

            // Token-by-token ribbon colored by confidence or symbol type
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 5) {
                    ForEach(result.decodedTokens) { tok in
                        let chipColor = tokenChipColor(tok)
                        HStack(spacing: 4) {
                            Text(tok.text)
                                .font(.system(.caption2, design: .monospaced).weight(.medium))
                            if stripColorMode == .confidence {
                                Text("\(Int(round(tok.confidence * 100)))%")
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(chipColor.opacity(0.20))
                        .overlay(
                            RoundedRectangle(cornerRadius: 5)
                                .strokeBorder(chipColor.opacity(0.55), lineWidth: 0.75)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 5))
                    }
                }
            }
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func tokenChipColor(_ tok: DecodedToken) -> Color {
        switch stripColorMode {
        case .strokeOrder:
            return .accentColor
        case .confidence:
            let (r, g, b) = StripRasterizer.confidenceRGB(tok.confidence)
            return Color(red: Double(r) / 255.0, green: Double(g) / 255.0, blue: Double(b) / 255.0)
        case .symbolType:
            return tok.category.color
        }
    }

    @ViewBuilder
    private func inspectorLegend(for mode: StripColorMode, modality: ModalityMode) -> some View {
        HStack(spacing: 12) {
            switch mode {
            case .strokeOrder:
                legendDot(color: Color(red: 40 / 255, green: 220 / 255, blue: 1), label: "First stroke")
                legendDot(color: Color(red: 1, green: 100 / 255, blue: 215 / 255), label: "Last stroke")
            case .confidence:
                legendDot(color: Color(red: 52 / 255, green: 211 / 255, blue: 153 / 255), label: "High (≥90%)")
                legendDot(color: Color(red: 250 / 255, green: 190 / 255, blue: 36 / 255), label: "Mid (~80%)")
                legendDot(color: Color(red: 248 / 255, green: 81 / 255, blue: 73 / 255), label: "Low (<65%)")
            case .symbolType:
                ForEach(TokenCategory.allCases) { cat in
                    legendDot(color: cat.color, label: cat.label)
                }
            }
            if modality == .stroke && mode != .strokeOrder {
                Spacer()
                Text("Switch to Bimodal/Vision for spatial strip coloring")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private func legendDot(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "circle.fill")
                .font(.system(size: 7))
                .foregroundStyle(color)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
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
            debounceTask?.cancel()
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
        debounceTask?.cancel()
        guard engine.status == .ready else { return }
        let selectedModality = modality

        debounceTask = Task {
            if !immediate {
                try? await Task.sleep(nanoseconds: 320_000_000)
            }
            if Task.isCancelled || myGen != requestGeneration { return }
            do {
                let result = try await engine.recognize(
                    strokes: strokes,
                    modality: selectedModality
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
