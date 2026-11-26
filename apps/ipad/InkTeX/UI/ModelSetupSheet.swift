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
                // 1. Live Hardware RAM Breakdown (exact Darwin phys_footprint per component!)
                Section {
                    MemoryBreakdownCard(
                        snapshot: engine.memorySnapshot,
                        onRefresh: { engine.refreshMemory() }
                    )
                    .listRowInsets(EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14))
                    Toggle("Show Live RAM in Bottom Bar", isOn: $engine.showLiveRamInBar)
                } header: {
                    Text("Memory Usage (Live)")
                }

                // 2. Active Model & Checkpoint Switcher
                Section("Active Model") {
                    if let bundle = engine.loadedBundle {
                        LabeledContent("Model", value: bundle.manifest.modelName)
                        LabeledContent(
                            "Architecture",
                            value: bundle.isV2
                                ? "InkTeX v2 (\(bundle.manifest.preset))"
                                : "InkTeX v1 (\(bundle.manifest.preset))"
                        )
                        if let stage = bundle.manifest.stage, let epoch = bundle.manifest.epoch {
                            LabeledContent("Checkpoint", value: "Stage \(stage) · Epoch \(epoch)")
                        }
                        if let exact = bundle.manifest.metrics?.normalizedExact {
                            LabeledContent("Val Exact", value: String(format: "%.2f%%", exact * 100))
                        }
                        LabeledContent("Quantization", value: (bundle.manifest.quantization ?? "fp32").uppercased())
                        LabeledContent(
                            "R2L Reranker",
                            value: engine.hasSharedR2L
                                ? "Shared Bidirectional (0 extra RAM)"
                                : (engine.hasR2LScorer ? "Active (3-layer)" : "Not in bundle")
                        )
                        if let pers = bundle.manifest.personalization {
                            LabeledContent(
                                "Personalization",
                                value: pers.dynamicLora
                                    ? "FiLM (\(pers.styleDim)-d) + Dynamic LoRA (r=\(pers.loraRank))"
                                    : "FiLM (\(pers.styleDim)-d)"
                            )
                        }
                        LabeledContent(
                            "Decoder Engine",
                            value: engine.staticKvLen != nil
                                ? "Static KV (\(engine.staticKvLen!)) + Prefix Verify"
                                : (engine.supportsV2FastDecode ? "Prefix Verify + Batched Beam" : "Legacy v1")
                        )
                        LabeledContent("Source", value: bundle.sourceDescription)
                    } else {
                        VStack(alignment: .leading, spacing: 6) {
                            Label("No model bundle loaded", systemImage: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .font(.headline)
                            Text("Import an exported `model-bundle` folder from Files / AirDrop, or add it in Xcode.")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }

                // 3. Multi-Model Switcher (automatically sorted best/latest first!)
                if engine.availableBundles.count > 1 {
                    Section("Available Models (Best First)") {
                        ForEach(engine.availableBundles) { bundle in
                            Button {
                                engine.selectBundle(bundle)
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(bundle.displayTitle)
                                            .font(.subheadline.weight(.medium))
                                            .foregroundStyle(.primary)
                                        Text("\(bundle.sourceDescription) · \((bundle.manifest.quantization ?? "fp32").uppercased())")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    if engine.loadedBundle == bundle {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(Color.accentColor)
                                    }
                                }
                            }
                        }
                    }
                }

                // 4. Live Decoding & Speed
                Section("Live Decoding & Speed") {
                    Toggle("Two-Phase Live Preview", isOn: $engine.liveTwoPhase)
                    Text("Shows instant prefix-verified Greedy while writing, then silently polishes with Batched Beam + R2L after you pause (reusing cached encoder memory).")
                        .font(.caption)
                        .foregroundStyle(.secondary)

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
                }

                // 5. Runtime Acceleration (CoreML MLProgram + On-Disk Cache)
                Section("CoreML Acceleration (MLProgram)") {
                    Picker("CoreML Profile", selection: Binding(
                        get: { engine.coreMLProfile },
                        set: { newValue in
                            engine.coreMLProfile = newValue
                            engine.reloadModel()
                        }
                    )) {
                        ForEach(CoreMLProfile.allCases) { profile in
                            Text(profile.title).tag(profile)
                        }
                    }

                    Text(engine.coreMLProfile.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Button {
                        engine.clearCoreMLDiskCache()
                        engine.reloadModel()
                    } label: {
                        Label("Clear CoreML Compiled Cache", systemImage: "arrow.triangle.2.circlepath")
                    }
                }

                // 6. Import / Manage Bundles
                Section("Import / Manage Models") {
                    Button {
                        showingFolderPicker = true
                    } label: {
                        Label("Import model-bundle Folder from Files…", systemImage: "folder.badge.plus")
                    }

                    Button {
                        engine.reloadModel()
                    } label: {
                        Label("Reload Active Model", systemImage: "arrow.clockwise")
                    }

                    if let active = engine.loadedBundle,
                       active.directory.standardizedFileURL.path.hasPrefix(ModelBundleLocator.documentsDirectory.standardizedFileURL.path) {
                        Button(role: .destructive) {
                            do {
                                try ModelBundleLocator.removeImportedBundle(active)
                                engine.reloadModel()
                            } catch {
                                importError = error.localizedDescription
                            }
                        } label: {
                            Label("Delete Active Imported Model", systemImage: "trash")
                        }
                    }

                    if let importError {
                        Text(importError)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                engine.refreshMemory()
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
                        let installed = try ModelBundleLocator.importBundle(from: folder)
                        engine.selectBundle(installed)
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

/// Visual per-component RAM monitor using real Darwin `phys_footprint` & `os_proc_available_memory()`.
struct MemoryBreakdownCard: View {
    let snapshot: MemorySnapshot
    var onRefresh: (() -> Void)? = nil

    private var segments: [(id: String, name: String, ramMB: Double, diskMB: Double?, coreML: Bool, color: Color)] {
        var rows: [(String, String, Double, Double?, Bool, Color)] = []
        rows.append(("baseline", "App Baseline (UI + PencilKit)", snapshot.baselineMB, nil, false, Color.gray))
        for comp in snapshot.components {
            rows.append((comp.id, comp.name, comp.ramMB, comp.diskMB, comp.coreMLEnabled, comp.color))
        }
        if snapshot.runtimeOverheadMB >= 0.5 {
            rows.append(("runtime", "KV-Cache & Runtime Buffers", snapshot.runtimeOverheadMB, nil, false, Color.orange))
        }
        return rows
    }

    private var barTotalMB: Double {
        max(1.0, segments.reduce(0) { $0 + max(0, $1.ramMB) })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Top summary row
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(format: "%.0f MB", snapshot.currentFootprintMB))
                        .font(.title2.weight(.bold).monospacedDigit())
                    Text(
                        String(
                            format: "Peak %.0f MB · %.1f GB free headroom",
                            snapshot.peakFootprintMB,
                            snapshot.availableHeadroomMB / 1024.0
                        )
                    )
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                }

                Spacer()

                if let onRefresh {
                    Button {
                        onRefresh()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.subheadline)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }

            // Stacked horizontal proportion bar
            GeometryReader { geo in
                HStack(spacing: 1.5) {
                    ForEach(segments, id: \.id) { seg in
                        let frac = max(0, seg.ramMB) / barTotalMB
                        if frac > 0.005 {
                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(seg.color)
                                .frame(width: max(4, (geo.size.width - CGFloat(segments.count) * 1.5) * frac))
                        }
                    }
                }
            }
            .frame(height: 12)
            .clipShape(Capsule())

            // Per-component table (Measured RAM + Disk size)
            VStack(spacing: 6) {
                ForEach(segments, id: \.id) { seg in
                    HStack(spacing: 8) {
                        Image(systemName: "circle.fill")
                            .font(.system(size: 8))
                            .foregroundStyle(seg.color)
                        Text(seg.name)
                            .font(.caption)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        if seg.coreML {
                            Text("CoreML")
                                .font(.system(size: 9, weight: .semibold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1.5)
                                .background(Color.blue.opacity(0.18))
                                .foregroundStyle(.blue)
                                .clipShape(Capsule())
                        }
                        Spacer()
                        if let disk = seg.diskMB {
                            Text(String(format: "%.0f MB disk", disk))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.tertiary)
                        }
                        Text(String(format: "%.1f MB", seg.ramMB))
                            .font(.caption.weight(.semibold).monospacedDigit())
                            .frame(width: 64, alignment: .trailing)
                    }
                }
            }

            if snapshot.lastInferenceDeltaMB > 0.05 {
                Text(String(format: "Last recognition transient RAM delta: +%.1f MB", snapshot.lastInferenceDeltaMB))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

/// Spacious, uncluttered Inspector sheet for inspecting the full-resolution Vision Strip,
/// per-character confidence heatmap, symbol types, token breakdown, L2R/R2L beams, and live RAM.
struct InspectorSheet: View {
    let result: RecognitionResult
    let memorySnapshot: MemorySnapshot
    @Binding var stripColorMode: StripColorMode
    var onRefreshMemory: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // 1. Large, crisp Vision Strip card
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Vision Strip")
                                .font(.headline)
                            Spacer()
                            Picker("Strip Mode", selection: $stripColorMode) {
                                ForEach(StripColorMode.allCases) { mode in
                                    Text(mode.title).tag(mode)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(width: 240)
                        }

                        if let uiImg = StripRasterizer.renderVisualization(result: result, mode: stripColorMode) {
                            ScrollView(.horizontal, showsIndicators: true) {
                                Image(uiImage: uiImg)
                                    .interpolation(.none)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(height: 168)
                                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                            }
                            .padding(8)
                            .background(Color(red: 18 / 255, green: 20 / 255, blue: 26 / 255))
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }

                        FlowLayout(spacing: 12) {
                            switch stripColorMode {
                            case .strokeOrder:
                                legendItem(color: Color(red: 40 / 255, green: 220 / 255, blue: 1), label: "First stroke")
                                legendItem(color: Color(red: 1, green: 100 / 255, blue: 215 / 255), label: "Last stroke")
                            case .confidence:
                                legendItem(color: Color(red: 52 / 255, green: 211 / 255, blue: 153 / 255), label: "High (≥90%)")
                                legendItem(color: Color(red: 250 / 255, green: 190 / 255, blue: 36 / 255), label: "Medium (~80%)")
                                legendItem(color: Color(red: 248 / 255, green: 81 / 255, blue: 73 / 255), label: "Low (<65%)")
                            case .symbolType:
                                ForEach(TokenCategory.allCases) { cat in
                                    legendItem(color: cat.color, label: cat.label)
                                }
                            }
                        }

                        if result.modality == .stroke && stripColorMode != .strokeOrder {
                            Text("Spatial strip coloring requires Bimodal or Vision mode.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(16)
                    .background(Color(uiColor: .secondarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                    // 2. Clean Timing & Telemetry Metrics
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)], spacing: 10) {
                        metricTile(title: "Total", value: String(format: "%.0f ms", result.totalMs))
                        metricTile(
                            title: "Encode",
                            value: result.encoderCacheHit ? "Cached (0 ms)" : String(format: "%.0f ms", result.encodeMs)
                        )
                        metricTile(title: "Decode", value: String(format: "%.0f ms", result.decodeMs))
                        if result.r2lUsed {
                            metricTile(title: "R2L Rerank", value: String(format: "%.0f ms", result.r2lMs))
                        } else if result.prefixVerifiedCount > 0 {
                            metricTile(title: "Prefix Cached", value: "\(result.prefixVerifiedCount) tok")
                        }
                        metricTile(title: "Confidence", value: String(format: "%.1f%%", result.overallConfidence * 100))
                        metricTile(title: "RAM Used", value: String(format: "%.0f MB", memorySnapshot.currentFootprintMB))
                        if let bucket = result.chunkBucket {
                            metricTile(title: "Ink Bucket", value: "\(bucket) chunks")
                        }
                        if result.usedWriterStyle || result.usedLoRAAdapter {
                            metricTile(
                                title: "Personalization",
                                value: result.usedWriterStyle && result.usedLoRAAdapter
                                    ? "FiLM + LoRA"
                                    : (result.usedLoRAAdapter ? "LoRA Only" : "FiLM Style")
                            )
                        }
                    }

                    // 3. Wrapping Token Breakdown
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Tokens (\(result.decodedTokens.count))")
                            .font(.headline)

                        FlowLayout(spacing: 6) {
                            ForEach(result.decodedTokens) { tok in
                                let chipColor = tokenColor(tok)
                                HStack(spacing: 4) {
                                    Text(tok.text)
                                        .font(.system(.callout, design: .monospaced).weight(.medium))
                                    Text("\(Int(round(tok.confidence * 100)))%")
                                        .font(.system(.caption2, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(chipColor.opacity(0.18))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 7)
                                        .strokeBorder(chipColor.opacity(0.55), lineWidth: 1)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 7))
                            }
                        }
                    }
                    .padding(16)
                    .background(Color(uiColor: .secondarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                    // 4. Top Beams (L2R vs R2L)
                    if result.beams.count > 1 {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text(result.r2lUsed ? "Beam Candidates (L2R + R2L)" : "Beam Candidates (L2R)")
                                    .font(.headline)
                                Spacer()
                                if result.r2lChangedWinner {
                                    Text("R2L Pick")
                                        .font(.caption.weight(.semibold))
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 3)
                                        .background(Color.purple.opacity(0.2))
                                        .foregroundStyle(.purple)
                                        .clipShape(Capsule())
                                }
                            }

                            ForEach(result.beams) { beam in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(spacing: 8) {
                                        Image(systemName: beam.isWinner ? "star.fill" : "circle")
                                            .font(.caption)
                                            .foregroundStyle(beam.wasPromotedByR2L ? Color.purple : (beam.isWinner ? Color.accentColor : Color.secondary))
                                        Text(beam.latex)
                                            .font(.system(.callout, design: .monospaced))
                                            .textSelection(.enabled)
                                    }
                                    HStack(spacing: 12) {
                                        Text(String(format: "L2R: %.3f", beam.l2rScore))
                                        if let r2l = beam.r2lScore {
                                            Text(String(format: "R2L: %.3f", r2l))
                                            Text(String(format: "Combined: %.3f", beam.combinedScore))
                                                .fontWeight(.semibold)
                                        }
                                    }
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .padding(.leading, 22)
                                }
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(beam.isWinner ? Color.accentColor.opacity(0.09) : Color.clear)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        }
                        .padding(16)
                        .background(Color(uiColor: .secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }

                    // 5. Per-Component RAM Breakdown Card
                    VStack(alignment: .leading, spacing: 10) {
                        Text("RAM Breakdown")
                            .font(.headline)
                        MemoryBreakdownCard(snapshot: memorySnapshot, onRefresh: onRefreshMemory)
                    }
                    .padding(16)
                    .background(Color(uiColor: .secondarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
                .padding(16)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationTitle("Inspector")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                onRefreshMemory()
            }
        }
    }

    private func tokenColor(_ tok: DecodedToken) -> Color {
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

    private func legendItem(color: Color, label: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "circle.fill")
                .font(.system(size: 8))
                .foregroundStyle(color)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func metricTile(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold).monospacedDigit())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: min(maxWidth, x), height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX && x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
