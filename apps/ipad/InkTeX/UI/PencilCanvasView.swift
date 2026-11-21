import PencilKit
import SwiftUI
import UIKit

public final class PencilCanvasController: ObservableObject {
    weak var canvasView: PKCanvasView?
    @Published public private(set) var canUndo: Bool = false
    @Published public private(set) var canRedo: Bool = false
    @Published public private(set) var strokeCount: Int = 0

    public init() {}

    public func clear() {
        guard let canvasView else { return }
        canvasView.drawing = PKDrawing()
        canvasView.undoManager?.removeAllActions()
        syncState()
    }

    public func undo() {
        canvasView?.undoManager?.undo()
        syncState()
    }

    public func redo() {
        canvasView?.undoManager?.redo()
        syncState()
    }

    public func extractRawStrokes() -> [RawStroke] {
        guard let drawing = canvasView?.drawing else { return [] }
        var result: [RawStroke] = []
        result.reserveCapacity(drawing.strokes.count)

        for stroke in drawing.strokes {
            let path = stroke.path
            guard path.count > 0 else { continue }
            let transform = stroke.transform
            let baseMs = path.creationDate.timeIntervalSince1970 * 1000.0
            var pts: [RawStrokePoint] = []
            pts.reserveCapacity(path.count)
            for idx in 0..<path.count {
                let sample = path[idx]
                let loc = sample.location.applying(transform)
                pts.append(
                    RawStrokePoint(
                        x: Double(loc.x),
                        y: Double(loc.y),
                        timeMs: baseMs + sample.timeOffset * 1000.0
                    )
                )
            }
            if !pts.isEmpty {
                result.append(RawStroke(points: pts))
            }
        }
        return result
    }

    func syncState() {
        DispatchQueue.main.async { [weak self] in
            guard let self, let canvasView = self.canvasView else { return }
            self.canUndo = canvasView.undoManager?.canUndo ?? false
            self.canRedo = canvasView.undoManager?.canRedo ?? false
            self.strokeCount = canvasView.drawing.strokes.count
        }
    }
}

public struct PencilCanvasView: UIViewRepresentable {
    @ObservedObject var controller: PencilCanvasController
    var pencilOnly: Bool
    var onDrawingChanged: ([RawStroke]) -> Void

    public func makeCoordinator() -> Coordinator {
        Coordinator(controller: controller, onDrawingChanged: onDrawingChanged)
    }

    public func makeUIView(context: Context) -> PKCanvasView {
        let canvas = PKCanvasView()
        canvas.backgroundColor = .clear
        canvas.isOpaque = false
        canvas.drawingPolicy = pencilOnly ? .pencilOnly : .anyInput
        canvas.tool = PKInkingTool(.pen, color: .label, width: 3.5)
        canvas.delegate = context.coordinator
        canvas.alwaysBounceVertical = false
        canvas.alwaysBounceHorizontal = false
        controller.canvasView = canvas
        return canvas
    }

    public func updateUIView(_ uiView: PKCanvasView, context: Context) {
        uiView.drawingPolicy = pencilOnly ? .pencilOnly : .anyInput
        context.coordinator.onDrawingChanged = onDrawingChanged
    }

    public final class Coordinator: NSObject, PKCanvasViewDelegate {
        let controller: PencilCanvasController
        var onDrawingChanged: ([RawStroke]) -> Void

        init(controller: PencilCanvasController, onDrawingChanged: @escaping ([RawStroke]) -> Void) {
            self.controller = controller
            self.onDrawingChanged = onDrawingChanged
        }

        public func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
            controller.syncState()
            onDrawingChanged(controller.extractRawStrokes())
        }
    }
}
