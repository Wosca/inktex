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
    @State private var showModelSheet: Bool = false

    @State private var latestResult: RecognitionResult?
    @State private var errorMessage: String?
    @State private var copiedFeedback: Bool = false
    @State private var debounceTask: Task<Void, Never>?

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

                // Rendered LaTeX Preview Card
                previewPanel
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
                    .padding(.bottom, 10)

                // Optional Debug Inspector Drawer
                if showInspector, let result = latestResult {
                    inspectorBar(result)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 10)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                // Main Apple Pencil Drawing Canvas
                ZStack(alignment: .bottomTrailing) {
                    DotGridBackground()
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .strokeBorder(Color.secondary.opacity(0.22), lineWidth: 1)
                        )

                    PencilCanvasView(
                        controller: canvasController,
                        pencilOnly: pencilOnly,
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

                    // Floating Quick Actions
                    HStack(spacing: 10) {
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
                                .frame(width: 36, height: 36)
                        }
                        .buttonStyle(.bordered)
                        .disabled(!canvasController.canUndo)

                        Button {
                            canvasController.redo()
                        } label: {
                            Image(systemName: "arrow.uturn.forward")
                                .font(.body.weight(.medium))
                                .frame(width: 36, height: 36)
                        }
                        .buttonStyle(.bordered)
                        .disabled(!canvasController.canRedo)

                        Button(role: .destructive) {
                            debounceTask?.cancel()
                            canvasController.clear()
                            latestResult = nil
                            errorMessage = nil
                        } label: {
                            Label("Clear", systemImage: "trash")
                                .font(.subheadline.weight(.medium))
                        }
                        .buttonStyle(.bordered)
                        .disabled(canvasController.strokeCount == 0)
                    }
                    .padding(14)
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 10) {
                        Text("InkTeX")
                            .font(.headline.weight(.bold))
                        modelStatusChip
                    }
                }

                ToolbarItem(placement: .principal) {
                    Picker("Modality", selection: $modality) {
                        ForEach(ModalityMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 260)
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
                    .help("Toggle Vision Strip & Token Inspector")

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

    private var modelStatusChip: some View {
        Button {
            showModelSheet = true
        } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusLabel)
                    .font(.caption.weight(.medium))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Color(uiColor: .secondarySystemGroupedBackground))
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
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
                .frame(height: 112)

                if engine.isRecognizing {
                    ProgressView()
                        .padding(10)
                }
            }

            Divider()

            HStack(spacing: 10) {
                ScrollView(.horizontal, showsIndicators: false) {
                    Text(latestResult?.latex.isEmpty == false ? latestResult!.latex : "LaTeX output appears here")
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(latestResult?.latex.isEmpty == false ? .primary : .tertiary)
                        .textSelection(.enabled)
                        .lineLimit(1)
                }

                Spacer(minLength: 8)

                if let result = latestResult {
                    Text(String(format: "%.0f ms · %d tok", result.totalMs, result.tokens.count))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(Capsule())
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
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.secondary.opacity(0.15), lineWidth: 1)
        )
    }

    private func inspectorBar(_ result: RecognitionResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Vision Strip (\(result.stripWidth)×128) · \(result.strokeCount) strokes · \(result.pointCount) pts")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "Encode %.0fms · Decode %.0fms", result.encodeMs, result.decodeMs))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            if let data = result.debugStripPNGData, let uiImg = UIImage(data: data) {
                Image(uiImage: uiImg)
                    .interpolation(.none)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 68)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(Array(result.tokens.enumerated()), id: \.offset) { _, tok in
                        Text(tok)
                            .font(.system(.caption2, design: .monospaced))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(Color.accentColor.opacity(0.14))
                            .clipShape(RoundedRectangle(cornerRadius: 5))
                    }
                }
            }
        }
        .padding(12)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
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
            debounceTask?.cancel()
            latestResult = nil
            errorMessage = nil
            return
        }
        guard autoRecognize else { return }
        triggerRecognition(strokes, immediate: false)
    }

    private func triggerRecognition(_ strokes: [RawStroke], immediate: Bool) {
        debounceTask?.cancel()
        guard engine.status == .ready else { return }
        let selectedModality = modality

        debounceTask = Task {
            if !immediate {
                try? await Task.sleep(nanoseconds: 320_000_000)
            }
            if Task.isCancelled { return }
            do {
                let result = try await engine.recognize(
                    strokes: strokes,
                    modality: selectedModality,
                    includeDebugStrip: true
                )
                if !Task.isCancelled {
                    latestResult = result
                    errorMessage = nil
                }
            } catch {
                if !Task.isCancelled {
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
