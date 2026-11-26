import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Guided, adaptive multi-tier personalization onboarding flow:
///   1. Sweeps across all 8 math categories for broad stroke & symbol coverage.
///   2. Auto-calibrates Tier 1 Writer Style (FiLM) on-device once >= 12-20 diverse strokes are drawn.
///   3. Evaluates every equation live with InkTeX and dynamically schedules extra drills targeting
///      the exact symbols & categories InkTeX misread or scored with low confidence.
///   4. Saves every labeled sample so Tier 2 LoRA training JSONL export & `.inktexlora` import
///      work right at the end of onboarding.
struct PersonalizationOnboardingView: View {
    @ObservedObject var store: PersonalizationStore
    @ObservedObject var engine: InkTeXEngine
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme

    @StateObject private var session = AdaptiveOnboardingSession()
    @StateObject private var canvasController = PencilCanvasController()

    @State private var selectedPreset: OnboardingPreset = .balanced
    @State private var pencilOnly: Bool = false
    @State private var lastSavedSampleId: UUID? = nil

    @State private var showNewProfileAlert: Bool = false
    @State private var newProfileName: String = ""
    @State private var exportedJSONLURL: URL? = nil
    @State private var exportError: String? = nil
    @State private var showLoRAImporter: Bool = false
    @State private var loraImportMessage: String? = nil
    @State private var loraImportError: String? = nil
    @State private var copiedCommandFeedback: Bool = false
    @State private var errorMessage: String? = nil

    var body: some View {
        NavigationStack {
            Group {
                switch session.phase {
                case .welcome:
                    welcomeView
                case .writing, .evaluating:
                    writingView
                case .feedback:
                    feedbackView
                case .summary:
                    summaryView
                }
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationTitle(navigationTitle)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(session.phase == .summary ? "Close" : "Cancel") {
                        finishAndDismiss()
                    }
                }
                if session.phase == .writing || session.phase == .feedback {
                    ToolbarItem(placement: .topBarTrailing) {
                        if session.completedCount > 0 {
                            Button("Finish") {
                                markCompletedIfNeeded()
                                session.finishEarly()
                            }
                            .fontWeight(.semibold)
                        }
                    }
                }
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
                loraImportError = nil
                loraImportMessage = nil
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
                        loraImportMessage = "Installed \(meta.name) (r=\(meta.loraRank)) — active now!"
                    } catch {
                        loraImportError = error.localizedDescription
                    }
                case .failure(let error):
                    loraImportError = error.localizedDescription
                }
            }
        }
    }

    private var navigationTitle: String {
        switch session.phase {
        case .welcome: return "Handwriting Onboarding"
        case .writing, .evaluating:
            return "Equation \(min(session.completedCount + 1, session.targetTotal)) of \(session.targetTotal)"
        case .feedback:
            return "Evaluation · \(session.completedCount)/\(session.targetTotal)"
        case .summary:
            return "Personalization Summary"
        }
    }

    // MARK: - 1. Welcome & Session Setup

    private var welcomeView: some View {
        ScrollView {
            VStack(spacing: 18) {
                // Hero card
                VStack(spacing: 10) {
                    ZStack {
                        Circle()
                            .fill(Color.accentColor.opacity(0.14))
                            .frame(width: 68, height: 68)
                        Image(systemName: "wand.and.stars")
                            .font(.system(size: 30, weight: .semibold))
                            .foregroundStyle(Color.accentColor)
                    }
                    Text("Train InkTeX on Your Handwriting")
                        .font(.title2.weight(.bold))
                        .multilineTextAlignment(.center)
                    Text("Write a guided sequence of math equations. InkTeX evaluates every answer live, calibrates your on-device style vector automatically, and serves extra drills on whatever symbols it misses.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 560)
                }
                .padding(.top, 8)

                if engine.status != .ready {
                    HStack(spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text("Load an InkTeX model bundle in Settings before starting live handwriting evaluation.")
                            .font(.subheadline)
                        Spacer()
                    }
                    .padding(14)
                    .background(Color.orange.opacity(0.14))
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                } else if !engine.isV2Model {
                    HStack(spacing: 10) {
                        Image(systemName: "info.circle.fill")
                            .foregroundStyle(.blue)
                        Text("InkTeX v1 bundle active: live evaluation, adaptive drilling, and training dataset export work now. Tier 1 FiLM & Tier 2 LoRA activate automatically on v2.")
                            .font(.caption)
                        Spacer()
                    }
                    .padding(12)
                    .background(Color.blue.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                }

                // Active Profile card
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Label("Writer Profile", systemImage: "person.crop.circle.fill")
                            .font(.headline)
                        Spacer()
                        Button {
                            newProfileName = ""
                            showNewProfileAlert = true
                        } label: {
                            Label("New Profile", systemImage: "plus")
                                .font(.caption.weight(.semibold))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    Picker("Active Writer", selection: $store.activeProfileId) {
                        ForEach(store.profiles) { profile in
                            Text(profile.name).tag(Optional(profile.id))
                        }
                    }
                    .pickerStyle(.segmented)

                    if let profile = store.activeProfile {
                        HStack(spacing: 10) {
                            statusPill(
                                label: profile.calibrationStrokeCount > 0
                                    ? "Tier 1 FiLM: \(profile.calibrationStrokeCount) strokes"
                                    : "Tier 1 FiLM: Not set",
                                active: profile.calibrationStrokeCount > 0,
                                color: .teal
                            )
                            statusPill(
                                label: profile.loraAdapter != nil
                                    ? "Tier 2 LoRA: Installed"
                                    : "Tier 2 Samples: \(store.activeCorrections.count)",
                                active: profile.loraAdapter != nil || store.activeCorrections.count >= 25,
                                color: .purple
                            )
                        }
                    }
                }
                .padding(16)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                // 3-Tier Walkthrough Card
                VStack(alignment: .leading, spacing: 12) {
                    Text("What This Session Collects")
                        .font(.headline)

                    tierRow(
                        badge: "Tier 1",
                        color: .teal,
                        title: "Zero-Shot Writer Style (FiLM)",
                        detail: "Automatically enrolled on-device from 20 diverse strokes in <1 ms — active immediately mid-session."
                    )
                    tierRow(
                        badge: "Tier 2",
                        color: .purple,
                        title: "Personal LoRA Training Set",
                        detail: "Covers all 8 math categories (digits, scripts, fractions, Greek, calculus, trig, logic, vectors) plus extra drills on misses, ready for 1-tap JSONL export."
                    )
                    tierRow(
                        badge: "Tier 3",
                        color: .blue,
                        title: "Live Benchmark & Weak-Symbol Report",
                        detail: "Pinpoints exact lookalike confusions (e.g. 2 vs z, α vs a, 5 vs s) and saves every labeled sample to your profile."
                    )
                }
                .padding(16)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                // Session Length Selector
                VStack(alignment: .leading, spacing: 10) {
                    Text("Choose Session Depth")
                        .font(.headline)

                    HStack(spacing: 10) {
                        ForEach(OnboardingPreset.allCases) { option in
                            let selected = (selectedPreset == option)
                            Button {
                                selectedPreset = option
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(option.title)
                                            .font(.subheadline.weight(.bold))
                                        if option == .balanced {
                                            Text("Best")
                                                .font(.system(size: 10, weight: .bold))
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 2)
                                                .background(Color.accentColor.opacity(0.2))
                                                .foregroundStyle(Color.accentColor)
                                                .clipShape(Capsule())
                                        }
                                    }
                                    Text("\(option.totalPrompts) equations")
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.primary)
                                    Text(option.subtitle)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .multilineTextAlignment(.leading)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(12)
                                .background(selected ? Color.accentColor.opacity(0.12) : Color(uiColor: .tertiarySystemGroupedBackground))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                                        .strokeBorder(selected ? Color.accentColor : Color.secondary.opacity(0.2), lineWidth: selected ? 2 : 1)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(16)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                // Start CTA
                Button {
                    let existing = store.activeProfile?.calibrationStrokes ?? []
                    session.start(preset: selectedPreset, existingCalibrationStrokes: existing)
                    canvasController.clear()
                    errorMessage = nil
                } label: {
                    Label("Start Guided Personalization", systemImage: "pencil.tip.crop.circle.badge.plus")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(engine.status != .ready)
            }
            .padding(20)
            .frame(maxWidth: 760)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - 2. Interactive Writing Screen

    private var writingView: some View {
        VStack(spacing: 10) {
            // Top Progress & Live Telemetry Strip
            VStack(spacing: 6) {
                ProgressView(
                    value: Double(session.completedCount),
                    total: Double(max(1, session.targetTotal))
                )
                .tint(session.isAdaptiveDrillPhase ? .purple : .accentColor)

                HStack(spacing: 8) {
                    Text(session.isAdaptiveDrillPhase ? "Adaptive Drill" : "Breadth Sweep")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background((session.isAdaptiveDrillPhase ? Color.purple : Color.accentColor).opacity(0.16))
                        .foregroundStyle(session.isAdaptiveDrillPhase ? .purple : .accentColor)
                        .clipShape(Capsule())

                    if !session.currentSelectionReason.isEmpty {
                        Text(session.currentSelectionReason)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }

                    Spacer()

                    if session.completedCount > 0 {
                        Text(String(format: "%.0f%% exact (%d/%d)", session.exactMatchRate * 100, session.exactMatchCount, session.completedCount))
                            .font(.caption.monospacedDigit().weight(.medium))
                            .foregroundStyle(.secondary)
                    }

                    // Tier 1 FiLM live status badge
                    let calCount = store.activeProfile?.calibrationStrokeCount ?? 0
                    HStack(spacing: 4) {
                        Image(systemName: (calCount >= 12 && engine.isV2Model) ? "checkmark.circle.fill" : "waveform")
                            .font(.caption2)
                        Text(
                            engine.isV2Model
                                ? (calCount >= 12 ? "FiLM Active" : "FiLM \(min(session.totalStrokesCollected + calCount, 20))/20")
                                : "v1 Capture"
                        )
                        .font(.caption2.weight(.semibold))
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background((calCount >= 12 && engine.isV2Model) ? Color.teal.opacity(0.18) : Color.secondary.opacity(0.14))
                    .foregroundStyle((calCount >= 12 && engine.isV2Model) ? .teal : .secondary)
                    .clipShape(Capsule())
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)

            // Target Prompt Card
            if let prompt = session.currentPrompt {
                VStack(spacing: 6) {
                    HStack {
                        Label(prompt.category.title, systemImage: prompt.category.iconName)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("Write this expression:")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }

                    KaTeXWebView(latex: prompt.latex, isDark: colorScheme == .dark)
                        .frame(height: 96)

                    HStack(spacing: 8) {
                        Text(prompt.latex)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                        Spacer()
                        Text(prompt.tip)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                    }
                }
                .padding(12)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(Color.secondary.opacity(0.16), lineWidth: 1)
                )
                .padding(.horizontal, 16)
            }

            if let errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                    Text(errorMessage).font(.caption)
                    Spacer()
                    Button("Dismiss") { self.errorMessage = nil }.font(.caption.weight(.semibold))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.red.opacity(0.14))
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .padding(.horizontal, 16)
            }

            // Spacious Apple Pencil Canvas
            ZStack {
                OnboardingDotGrid()
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(Color.secondary.opacity(0.22), lineWidth: 1)
                    )

                PencilCanvasView(
                    controller: canvasController,
                    pencilOnly: pencilOnly,
                    isDark: colorScheme == .dark,
                    onDrawingChanged: { _ in }
                )
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))

                if canvasController.strokeCount == 0 {
                    VStack(spacing: 6) {
                        Image(systemName: "applepencil.and.scribble")
                            .font(.system(size: 32))
                            .foregroundStyle(.tertiary)
                        Text("Write the equation above naturally with Apple Pencil")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .allowsHitTesting(false)
                }

                if session.phase == .evaluating {
                    ZStack {
                        Color.black.opacity(0.18)
                        HStack(spacing: 10) {
                            ProgressView()
                            Text("Evaluating with InkTeX…")
                                .font(.subheadline.weight(.semibold))
                        }
                        .padding(.horizontal, 18)
                        .padding(.vertical, 12)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                }
            }
            .padding(.horizontal, 16)

            // Bottom Action Bar
            HStack(spacing: 10) {
                Button {
                    pencilOnly.toggle()
                } label: {
                    Label(pencilOnly ? "Pencil Only" : "Pencil + Touch", systemImage: pencilOnly ? "applepencil.tip" : "hand.draw")
                        .font(.caption.weight(.medium))
                }
                .buttonStyle(.bordered)

                Button {
                    canvasController.undo()
                } label: {
                    Image(systemName: "arrow.uturn.backward")
                }
                .buttonStyle(.bordered)
                .disabled(!canvasController.canUndo || session.phase == .evaluating)

                Button {
                    canvasController.clear()
                } label: {
                    Label("Clear", systemImage: "trash")
                        .font(.caption.weight(.medium))
                }
                .buttonStyle(.bordered)
                .disabled(canvasController.strokeCount == 0 || session.phase == .evaluating)

                Spacer()

                Button("Skip") {
                    canvasController.clear()
                    session.skipCurrentPrompt()
                }
                .buttonStyle(.bordered)
                .disabled(session.phase == .evaluating)

                Button {
                    evaluateCurrentDrawing()
                } label: {
                    Label("Check My Writing", systemImage: "checkmark.seal.fill")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 6)
                }
                .buttonStyle(.borderedProminent)
                .disabled(canvasController.strokeCount == 0 || session.phase == .evaluating)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
        }
    }

    // MARK: - 3. Live Evaluation Feedback View

    private var feedbackView: some View {
        ScrollView {
            if let ev = session.latestEvaluation {
                VStack(spacing: 16) {
                    // Result Banner
                    HStack(spacing: 12) {
                        Image(systemName: ev.isExactMatch ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                            .font(.system(size: 32))
                            .foregroundStyle(ev.isExactMatch ? .green : .orange)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(ev.isExactMatch ? "Exact Match!" : "Great Training Signal — InkTeX Missed This")
                                .font(.headline)
                            Text(
                                ev.isExactMatch
                                    ? String(format: "100%% match · %.0f%% confidence · %.0f ms · saved to dataset", ev.confidence * 100, ev.latencyMs)
                                    : String(format: "%.0f%% token accuracy · saved with ground truth for LoRA & future drills", ev.tokenAccuracy * 100)
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    .padding(16)
                    .background((ev.isExactMatch ? Color.green : Color.orange).opacity(0.13))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                    if let banner = session.styleEnrollmentBanner {
                        HStack(spacing: 8) {
                            Image(systemName: "sparkles")
                                .foregroundStyle(.teal)
                            Text(banner)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.teal)
                            Spacer()
                        }
                        .padding(12)
                        .background(Color.teal.opacity(0.14))
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    }

                    // Side-by-side Target vs Prediction
                    HStack(alignment: .top, spacing: 12) {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Target Equation")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            KaTeXWebView(latex: ev.prompt.latex, isDark: colorScheme == .dark)
                                .frame(height: 112)
                            Text(ev.prompt.latex)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(uiColor: .secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("InkTeX Read")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Spacer()
                                if ev.usedWriterStyle || ev.usedLoRA {
                                    Text(ev.usedWriterStyle && ev.usedLoRA ? "FiLM+LoRA" : (ev.usedLoRA ? "LoRA" : "FiLM"))
                                        .font(.caption2.weight(.semibold))
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background(Color.teal.opacity(0.18))
                                        .foregroundStyle(.teal)
                                        .clipShape(Capsule())
                                }
                            }
                            KaTeXWebView(latex: ev.predictedLatex, isDark: colorScheme == .dark)
                                .frame(height: 112)
                            Text(ev.predictedLatex.isEmpty ? "(empty)" : ev.predictedLatex)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(ev.isExactMatch ? .green : .orange)
                                .textSelection(.enabled)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(uiColor: .secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }

                    // Specific Confusions & Adaptive Scheduling Feedback
                    if !ev.isExactMatch && (!ev.confusions.isEmpty || !ev.missedSymbols.isEmpty) {
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Adaptive Drill Updated", systemImage: "scope")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.purple)

                            if !ev.confusions.isEmpty {
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 8) {
                                        ForEach(ev.confusions) { c in
                                            HStack(spacing: 4) {
                                                Text(c.expected)
                                                    .font(.system(.caption, design: .monospaced).weight(.bold))
                                                    .foregroundStyle(.green)
                                                Image(systemName: "arrow.right")
                                                    .font(.caption2)
                                                    .foregroundStyle(.secondary)
                                                Text(c.predicted)
                                                    .font(.system(.caption, design: .monospaced).weight(.bold))
                                                    .foregroundStyle(.orange)
                                            }
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 5)
                                            .background(Color.purple.opacity(0.12))
                                            .clipShape(Capsule())
                                        }
                                    }
                                }
                            }

                            if !ev.missedSymbols.isEmpty {
                                Text("InkTeX queued more equations featuring \(ev.missedSymbols.prefix(4).joined(separator: ", ")) to strengthen your profile.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(uiColor: .secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }

                    // Next / Retry buttons
                    HStack(spacing: 12) {
                        Button {
                            if let id = lastSavedSampleId {
                                store.deleteCorrection(id: id)
                                lastSavedSampleId = nil
                            }
                            canvasController.clear()
                            session.retryCurrentPrompt()
                        } label: {
                            Label("Retry This One", systemImage: "arrow.counterclockwise")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)

                        Button {
                            lastSavedSampleId = nil
                            canvasController.clear()
                            if session.completedCount >= session.targetTotal {
                                markCompletedIfNeeded()
                            }
                            session.proceedAfterFeedback()
                        } label: {
                            Label(
                                session.completedCount >= session.targetTotal ? "View Summary" : "Next Equation",
                                systemImage: "arrow.right.circle.fill"
                            )
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                    }
                }
                .padding(20)
                .frame(maxWidth: 780)
                .frame(maxWidth: .infinity)
            }
        }
    }

    // MARK: - 4. Summary & All-Tiers Completion Hub

    private var summaryView: some View {
        ScrollView {
            VStack(spacing: 18) {
                // Scoreboard
                VStack(spacing: 12) {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 40))
                        .foregroundStyle(.green)
                    Text("Handwriting Profile Updated")
                        .font(.title2.weight(.bold))
                    Text("Collected \(session.completedCount) evaluated equations (\(session.totalStrokesCollected) strokes) across \(session.categorySummaries.filter { $0.attempted > 0 }.count) math categories.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)], spacing: 10) {
                        summaryTile(title: "Exact Match", value: String(format: "%.0f%%", session.exactMatchRate * 100))
                        summaryTile(title: "Token Accuracy", value: String(format: "%.0f%%", session.averageTokenAccuracy * 100))
                        summaryTile(title: "Session Eqs", value: "\(session.completedCount)")
                        summaryTile(title: "Total Profile Eqs", value: "\(store.activeCorrections.count)")
                    }
                }
                .padding(16)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                // All 3 Tiers Readiness & Actions
                VStack(alignment: .leading, spacing: 14) {
                    Text("All 3 Personalization Tiers")
                        .font(.headline)

                    // Tier 1
                    let calCount = store.activeProfile?.calibrationStrokeCount ?? 0
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: calCount > 0 ? "checkmark.circle.fill" : "circle.dashed")
                            .foregroundStyle(calCount > 0 ? .teal : .secondary)
                            .font(.title3)
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Tier 1 · Zero-Shot Writer Style (FiLM)")
                                .font(.subheadline.weight(.semibold))
                            Text(
                                calCount > 0
                                    ? "Active on-device (\(calCount)/\(InkPreprocessorV2.styleMaxStrokes) diverse strokes · \(Int(round((store.activeProfile?.styleConfidenceWeight ?? 0) * 100)))% style weight). Zero extra latency."
                                    : "Not enrolled yet — complete at least 2 equations to calibrate."
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }

                    Divider()

                    // Tier 2
                    let totalSamples = store.activeCorrections.count
                    let readyForLoRA = totalSamples >= 25
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: store.activeProfile?.loraAdapter != nil ? "checkmark.circle.fill" : (readyForLoRA ? "sparkles" : "clock"))
                                .foregroundStyle(.purple)
                                .font(.title3)
                            VStack(alignment: .leading, spacing: 3) {
                                Text("Tier 2 · Personal LoRA Adapter")
                                    .font(.subheadline.weight(.semibold))
                                Text(
                                    readyForLoRA
                                        ? "\(totalSamples) labeled samples ready! Export JSONL to your PC/Mac, run a ~30s LoRA fine-tune, and import `.inktexlora` right below."
                                        : "\(totalSamples) / 25 recommended samples collected. You can export now or tap \"Drill Weak Spots\" below for even stronger LoRA adaptation."
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                        }

                        if let profile = store.activeProfile, totalSamples > 0 {
                            HStack(spacing: 10) {
                                Button {
                                    do {
                                        exportedJSONLURL = try store.exportCorrectionsJSONL(for: profile.id)
                                        exportError = nil
                                    } catch {
                                        exportError = error.localizedDescription
                                    }
                                } label: {
                                    Label("Export Training JSONL", systemImage: "doc.zipper")
                                        .font(.caption.weight(.semibold))
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.purple)

                                if let url = exportedJSONLURL {
                                    ShareLink(item: url) {
                                        Label("AirDrop / Share", systemImage: "square.and.arrow.up")
                                            .font(.caption.weight(.semibold))
                                    }
                                    .buttonStyle(.bordered)
                                }

                                Button {
                                    showLoRAImporter = true
                                } label: {
                                    Label("Import .inktexlora", systemImage: "square.and.arrow.down")
                                        .font(.caption.weight(.semibold))
                                }
                                .buttonStyle(.bordered)
                            }

                            let safeName = profile.name
                                .replacingOccurrences(of: " ", with: "_")
                                .replacingOccurrences(of: "/", with: "-")
                            let trainCmd = "python ml/v2/train_personal_lora.py --checkpoint checkpoints/inktex_v2_release.pt --data inktex_\(safeName)_corrections.jsonl --output \(safeName.lowercased()).inktexlora --use-calibration-style"
                            HStack(spacing: 8) {
                                Text(trainCmd)
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                                Spacer()
                                Button {
                                    UIPasteboard.general.string = trainCmd
                                    copiedCommandFeedback = true
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                        copiedCommandFeedback = false
                                    }
                                } label: {
                                    Label(copiedCommandFeedback ? "Copied" : "Copy CLI", systemImage: copiedCommandFeedback ? "checkmark" : "doc.on.doc")
                                        .font(.caption2.weight(.semibold))
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                            }
                            .padding(10)
                            .background(Color(uiColor: .tertiarySystemGroupedBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }

                        if let exportError {
                            Text(exportError).font(.caption).foregroundStyle(.red)
                        }
                        if let loraImportMessage {
                            Text(loraImportMessage).font(.caption).foregroundStyle(.green)
                        }
                        if let loraImportError {
                            Text(loraImportError).font(.caption).foregroundStyle(.red)
                        }
                    }

                    Divider()

                    // Tier 3
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "tray.full.fill")
                            .foregroundStyle(.blue)
                            .font(.title3)
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Tier 3 · Personal Sample Library")
                                .font(.subheadline.weight(.semibold))
                            Text("\(store.activeOnboardingSampleCount) onboarding samples + \(store.activeManualCorrectionCount) manual corrections saved locally on iPad.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                }
                .padding(16)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                // 8-Category Coverage & Mastery Grid
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Category Coverage & Mastery")
                            .font(.headline)
                        Spacer()
                        if !session.topWeakSymbols.isEmpty {
                            Text("Weak spots found")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.orange)
                        }
                    }

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 165), spacing: 10)], spacing: 10) {
                        ForEach(session.categorySummaries) { item in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Image(systemName: item.category.iconName)
                                        .font(.caption)
                                        .foregroundStyle(Color.accentColor)
                                    Text(item.category.shortTitle)
                                        .font(.subheadline.weight(.semibold))
                                    Spacer()
                                    if item.attempted > 0 {
                                        Text("\(item.exactMatches)/\(item.attempted)")
                                            .font(.caption.monospacedDigit().weight(.semibold))
                                            .foregroundStyle(item.accuracy >= 0.8 ? .green : (item.accuracy >= 0.5 ? .orange : .red))
                                    } else {
                                        Text("—")
                                            .font(.caption)
                                            .foregroundStyle(.tertiary)
                                    }
                                }
                                ProgressView(value: item.attempted > 0 ? item.avgTokenAccuracy : 0)
                                    .tint(item.accuracy >= 0.8 ? .green : (item.accuracy >= 0.5 ? .orange : .red))

                                if !item.weakSymbols.isEmpty {
                                    Text("Missed: \(item.weakSymbols.joined(separator: " "))")
                                        .font(.system(.caption2, design: .monospaced))
                                        .foregroundStyle(.orange)
                                        .lineLimit(1)
                                } else if item.attempted > 0 {
                                    Text("Clean recognition")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                } else {
                                    Text("Not tested yet")
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                            .padding(12)
                            .background(Color(uiColor: .tertiarySystemGroupedBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }
                    }
                }
                .padding(16)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                // Bottom actions: Drill More vs Done
                HStack(spacing: 12) {
                    Button {
                        canvasController.clear()
                        session.extendWithTargetedDrills(extraCount: 8)
                    } label: {
                        Label("Drill Weakest Symbols (+8)", systemImage: "scope")
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .tint(.purple)
                    .controlSize(.large)

                    Button {
                        finishAndDismiss()
                    } label: {
                        Label("Done", systemImage: "checkmark")
                            .fontWeight(.semibold)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
            }
            .padding(20)
            .frame(maxWidth: 780)
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Actions & Evaluation Logic

    private func evaluateCurrentDrawing() {
        guard let prompt = session.currentPrompt else { return }
        let strokes = canvasController.extractRawStrokes()
        guard !strokes.isEmpty else { return }

        errorMessage = nil
        session.phase = .evaluating

        let reason = session.currentSelectionReason
        let profileId = store.activeProfileId

        Task {
            do {
                let result = try await engine.recognize(
                    strokes: strokes,
                    modality: .bimodal,
                    decodingMode: .beamR2L
                )
                let eval = OnboardingEvaluator.evaluate(
                    prompt: prompt,
                    result: result,
                    strokes: strokes,
                    selectionReason: reason
                )

                // Save labeled sample immediately to profile (feeds Tier 2 LoRA export)
                lastSavedSampleId = store.recordOnboardingSample(
                    strokes: strokes,
                    targetLatex: prompt.latex,
                    predictedLatex: result.latex,
                    category: prompt.category,
                    wasExactMatch: eval.isExactMatch
                )

                // Update adaptive scheduler & trigger Tier 1 FiLM enrollment when stroke pool hits threshold
                let strokesForStyle = session.recordEvaluation(eval)
                if let styleStrokes = strokesForStyle, let profileId {
                    if engine.isV2Model {
                        do {
                            let (vec, sha, ms) = try await engine.computeWriterStyle(strokes: styleStrokes)
                            store.updateCalibration(
                                for: profileId,
                                strokes: styleStrokes,
                                styleVector: vec,
                                modelSha: sha
                            )
                            session.markStyleEnrolled(strokeCount: styleStrokes.count, elapsedMs: ms)
                        } catch {
                            // Keep strokes saved even if style compute fails transiently
                            store.updateCalibration(for: profileId, strokes: styleStrokes)
                        }
                    } else {
                        store.updateCalibration(for: profileId, strokes: styleStrokes)
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
                session.phase = .writing
            }
        }
    }

    private func markCompletedIfNeeded() {
        if let profileId = store.activeProfileId, session.completedCount > 0 {
            store.markOnboardingCompleted(for: profileId)
        }
    }

    private func finishAndDismiss() {
        markCompletedIfNeeded()
        dismiss()
    }

    // MARK: - Small UI Builders

    private func tierRow(badge: String, color: Color, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(badge)
                .font(.caption2.weight(.bold))
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(color.opacity(0.16))
                .foregroundStyle(color)
                .clipShape(Capsule())
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func statusPill(label: String, active: Bool, color: Color) -> some View {
        Text(label)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background((active ? color : Color.secondary).opacity(0.15))
            .foregroundStyle(active ? color : .secondary)
            .clipShape(Capsule())
    }

    private func summaryTile(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.weight(.bold).monospacedDigit())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(uiColor: .tertiarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct OnboardingDotGrid: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Canvas { context, size in
            let bg = colorScheme == .dark ? Color(white: 0.11) : Color.white
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(bg))

            let spacing: CGFloat = 24
            let dotColor = colorScheme == .dark ? Color.white.opacity(0.10) : Color.black.opacity(0.08)
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
