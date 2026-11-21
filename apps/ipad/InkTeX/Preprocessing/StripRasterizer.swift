import Foundation
import UIKit

/// Exact Swift port of `export/inktex_model/raster.py` (`render_strokes_to_strip`),
/// plus real-time colorized strip visualization (stroke order, per-character confidence, or symbol type).
public enum StripRasterizer {
    public struct Output: Sendable {
        public let tensor: [Float]          // NCHW planar [1, 2, height, width], values in [0, 1]
        public let lumBytes: [UInt8]        // [height * width]
        public let orderBytes: [UInt8]      // [height * width]
        public let height: Int
        public let width: Int
        public let usedWidth: Int
    }

    public static func render(
        strokes: [RawStroke],
        config: ModelManifest.VisionConfig
    ) -> Output {
        let height = config.height
        let minWidth = config.minWidth
        let maxWidth = config.maxWidth
        let padding = config.padding
        let widthMultiple = config.widthMultiple
        let ss = 3

        let clean = strokes.map(\.points).filter { !$0.isEmpty }
        guard !clean.isEmpty else {
            let planeSize = height * minWidth
            return Output(
                tensor: [Float](repeating: 0, count: 2 * planeSize),
                lumBytes: [UInt8](repeating: 0, count: planeSize),
                orderBytes: [UInt8](repeating: 0, count: planeSize),
                height: height,
                width: minWidth,
                usedWidth: max(widthMultiple, min(minWidth, 2 * padding))
            )
        }

        var minX = Double.greatestFiniteMagnitude
        var maxX = -Double.greatestFiniteMagnitude
        var minY = Double.greatestFiniteMagnitude
        var maxY = -Double.greatestFiniteMagnitude
        var totalPoints = 0

        for stroke in clean {
            totalPoints += stroke.count
            for pt in stroke {
                minX = min(minX, pt.x)
                maxX = max(maxX, pt.x)
                minY = min(minY, pt.y)
                maxY = max(maxY, pt.y)
            }
        }

        let extentX = maxX - minX
        let extentY = maxY - minY
        let innerH = Double(height - 2 * padding)
        let innerW = Double(maxWidth - 2 * padding)

        var scales: [Double] = []
        if extentX > 1e-9 { scales.append(innerW / extentX) }
        if extentY > 1e-9 { scales.append(innerH / extentY) }
        let scale = scales.min() ?? 1.0

        let contentW = extentX * scale
        let contentH = extentY * scale
        let roundedW = Int(ceil(max(contentW + Double(2 * padding), 1.0) / Double(widthMultiple))) * widthMultiple
        let width = min(maxWidth, max(minWidth, roundedW))
        let offsetX = Double(padding) - minX * scale
        let offsetY = (Double(height) - contentH) / 2.0 - minY * scale

        let hiW = width * ss
        let hiH = height * ss
        let linePx = max(1, Int(Foundation.round(max(1.0, Double(height) / 48.0) * Double(ss))))
        let radius = Double(linePx) / 2.0
        let radiusSq = radius * radius

        var hiInk = [UInt8](repeating: 0, count: hiW * hiH)
        var hiOrder = [UInt8](repeating: 0, count: hiW * hiH)

        let denominator = Double(max(totalPoints - 1, 1))
        var globalStart = 0

        for stroke in clean {
            let scaledPts: [(Double, Double)] = stroke.map { pt in
                ((pt.x * scale + offsetX) * Double(ss), (pt.y * scale + offsetY) * Double(ss))
            }
            var keep: [Int] = [0]
            for i in 1..<scaledPts.count {
                let prev = scaledPts[keep[keep.count - 1]]
                let curr = scaledPts[i]
                if i == scaledPts.count - 1 || max(abs(curr.0 - prev.0), abs(curr.1 - prev.1)) >= 1.0 {
                    keep.append(i)
                }
            }

            var prevPt: (Double, Double)? = nil
            for idx in keep {
                let pt = scaledPts[idx]
                let prog = UInt8(min(255, max(1, 1 + Int(Foundation.rint(254.0 * Double(globalStart + idx) / denominator)))))
                if let p0 = prevPt {
                    drawCapsule(
                        x0: p0.0, y0: p0.1, x1: pt.0, y1: pt.1,
                        radius: radius, radiusSq: radiusSq,
                        width: hiW, height: hiH,
                        orderValue: prog,
                        ink: &hiInk, order: &hiOrder
                    )
                } else {
                    drawDisc(
                        cx: pt.0, cy: pt.1,
                        radius: radius, radiusSq: radiusSq,
                        width: hiW, height: hiH,
                        orderValue: prog,
                        ink: &hiInk, order: &hiOrder
                    )
                }
                prevPt = pt
            }
            globalStart += stroke.count
        }

        // Downsample 3x3: BOX average for Channel L, MAX for Channel A
        let planeSize = height * width
        var tensor = [Float](repeating: 0, count: 2 * planeSize)
        var lumBytes = [UInt8](repeating: 0, count: planeSize)
        var orderBytes = [UInt8](repeating: 0, count: planeSize)

        for y in 0..<height {
            let hy0 = y * ss
            for x in 0..<width {
                let hx0 = x * ss
                var inkSum = 0
                var orderMax: UInt8 = 0
                for dy in 0..<ss {
                    let rowOffset = (hy0 + dy) * hiW + hx0
                    for dx in 0..<ss {
                        let idx = rowOffset + dx
                        inkSum += Int(hiInk[idx])
                        let ord = hiOrder[idx]
                        if ord > orderMax { orderMax = ord }
                    }
                }
                let lumByte = UInt8(min(255, (inkSum + 4) / 9))
                let ordByte: UInt8 = lumByte > 0 ? max(orderMax, 1) : 0
                let dst = y * width + x
                lumBytes[dst] = lumByte
                orderBytes[dst] = ordByte
                tensor[dst] = Float(lumByte) / 255.0
                tensor[planeSize + dst] = Float(ordByte) / 255.0
            }
        }

        let usedWidth = min(width, max(widthMultiple, Int(ceil(contentW + Double(2 * padding)))))
        return Output(
            tensor: tensor,
            lumBytes: lumBytes,
            orderBytes: orderBytes,
            height: height,
            width: width,
            usedWidth: usedWidth
        )
    }

    /// Render a colorized visualization of the vision strip in < 1ms using decoder cross-attention.
    public static func renderVisualization(
        result: RecognitionResult,
        mode: StripColorMode
    ) -> UIImage? {
        let width = result.stripWidth
        let height = result.stripHeight
        let planeSize = width * height
        guard result.stripLumBytes.count == planeSize,
              result.stripOrderBytes.count == planeSize else {
            return nil
        }

        let gridW = result.visionGridW
        let gridH = result.visionGridH
        let gridCells = gridW * gridH
        let hasSpatialAttention = gridW > 0 && gridH > 0 && result.decodedTokens.contains { $0.visionAttention?.count == gridCells }

        // Precompute per-cell RGB on the [gridW * gridH] feature grid (column-major: col * gridH + row)
        var cellR = [Float](repeating: 180, count: max(1, gridCells))
        var cellG = [Float](repeating: 190, count: max(1, gridCells))
        var cellB = [Float](repeating: 210, count: max(1, gridCells))

        if hasSpatialAttention && (mode == .confidence || mode == .symbolType) {
            let visualTokens = result.decodedTokens.filter { $0.isVisualToken && $0.visionAttention?.count == gridCells }
            let sourceTokens = visualTokens.isEmpty ? result.decodedTokens.filter { $0.visionAttention?.count == gridCells } : visualTokens

            for cell in 0..<gridCells {
                if mode == .confidence {
                    var weightSum: Float = 0
                    var confSum: Float = 0
                    for tok in sourceTokens {
                        guard let attn = tok.visionAttention else { continue }
                        let w = attn[cell]
                        weightSum += w
                        confSum += w * tok.confidence
                    }
                    let conf = weightSum > 1e-6 ? (confSum / weightSum) : result.overallConfidence
                    let (r, g, b) = confidenceRGB(conf)
                    cellR[cell] = r
                    cellG[cell] = g
                    cellB[cell] = b
                } else {
                    // Dominant symbol category at this vision patch
                    var bestWeight: Float = -1
                    var bestCategory: TokenCategory = .structure
                    for tok in sourceTokens {
                        guard let attn = tok.visionAttention else { continue }
                        let w = attn[cell]
                        if w > bestWeight {
                            bestWeight = w
                            bestCategory = tok.category
                        }
                    }
                    let (r, g, b) = bestCategory.rgb
                    cellR[cell] = r
                    cellG[cell] = g
                    cellB[cell] = b
                }
            }
        }

        var rgba = [UInt8](repeating: 0, count: planeSize * 4)
        let fallbackConfRGB = confidenceRGB(result.overallConfidence)

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let lumByte = result.stripLumBytes[idx]
                let p = idx * 4

                if lumByte == 0 {
                    rgba[p + 0] = 18
                    rgba[p + 1] = 20
                    rgba[p + 2] = 26
                    rgba[p + 3] = 255
                    continue
                }

                let alpha = Float(lumByte) / 255.0
                let r: Float
                let g: Float
                let b: Float

                switch mode {
                case .strokeOrder:
                    let t = Float(result.stripOrderBytes[idx]) / 255.0
                    r = 40 + 215 * t
                    g = 220 - 120 * t
                    b = 255 - 40 * t

                case .confidence:
                    if hasSpatialAttention {
                        (r, g, b) = sampleBilinear(
                            x: x, y: y, width: width, height: height,
                            gridW: gridW, gridH: gridH,
                            cellR: cellR, cellG: cellG, cellB: cellB
                        )
                    } else {
                        (r, g, b) = fallbackConfRGB
                    }

                case .symbolType:
                    if hasSpatialAttention {
                        (r, g, b) = sampleBilinear(
                            x: x, y: y, width: width, height: height,
                            gridW: gridW, gridH: gridH,
                            cellR: cellR, cellG: cellG, cellB: cellB
                        )
                    } else {
                        (r, g, b) = (180, 196, 220)
                    }
                }

                // Blend over dark background for crisp readability
                rgba[p + 0] = UInt8(min(255, max(0, Int(18 * (1 - alpha) + r * alpha))))
                rgba[p + 1] = UInt8(min(255, max(0, Int(20 * (1 - alpha) + g * alpha))))
                rgba[p + 2] = UInt8(min(255, max(0, Int(26 * (1 - alpha) + b * alpha))))
                rgba[p + 3] = 255
            }
        }

        return makeUIImage(rgba: rgba, width: width, height: height)
    }

    /// Confidence gradient: Red (<0.6) -> Amber (0.8) -> Emerald (1.0)
    public static func confidenceRGB(_ conf: Float) -> (Float, Float, Float) {
        let c = min(1.0, max(0.0, conf))
        if c >= 0.80 {
            let t = (c - 0.80) / 0.20
            return (
                250 * (1 - t) + 52 * t,
                190 * (1 - t) + 211 * t,
                36 * (1 - t) + 153 * t
            )
        } else {
            let t = max(0.0, (c - 0.45) / 0.35)
            return (
                248 * (1 - t) + 250 * t,
                81 * (1 - t) + 190 * t,
                73 * (1 - t) + 36 * t
            )
        }
    }

    private static func sampleBilinear(
        x: Int, y: Int, width: Int, height: Int,
        gridW: Int, gridH: Int,
        cellR: [Float], cellG: [Float], cellB: [Float]
    ) -> (Float, Float, Float) {
        let fx = (Float(x) + 0.5) * Float(gridW) / Float(width) - 0.5
        let fy = (Float(y) + 0.5) * Float(gridH) / Float(height) - 0.5
        let x0 = max(0, min(gridW - 1, Int(floor(fx))))
        let x1 = max(0, min(gridW - 1, x0 + 1))
        let y0 = max(0, min(gridH - 1, Int(floor(fy))))
        let y1 = max(0, min(gridH - 1, y0 + 1))
        let tx = max(0, min(1, fx - Float(x0)))
        let ty = max(0, min(1, fy - Float(y0)))

        // Column-major index: col * gridH + row
        let i00 = x0 * gridH + y0
        let i10 = x1 * gridH + y0
        let i01 = x0 * gridH + y1
        let i11 = x1 * gridH + y1

        let w00 = (1 - tx) * (1 - ty)
        let w10 = tx * (1 - ty)
        let w01 = (1 - tx) * ty
        let w11 = tx * ty

        let r = w00 * cellR[i00] + w10 * cellR[i10] + w01 * cellR[i01] + w11 * cellR[i11]
        let g = w00 * cellG[i00] + w10 * cellG[i10] + w01 * cellG[i01] + w11 * cellG[i11]
        let b = w00 * cellB[i00] + w10 * cellB[i10] + w01 * cellB[i01] + w11 * cellB[i11]
        return (r, g, b)
    }

    private static func drawDisc(
        cx: Double, cy: Double,
        radius: Double, radiusSq: Double,
        width: Int, height: Int,
        orderValue: UInt8,
        ink: inout [UInt8], order: inout [UInt8]
    ) {
        let minX = max(0, Int(floor(cx - radius)))
        let maxX = min(width - 1, Int(ceil(cx + radius)))
        let minY = max(0, Int(floor(cy - radius)))
        let maxY = min(height - 1, Int(ceil(cy + radius)))
        guard minX <= maxX && minY <= maxY else { return }

        for y in minY...maxY {
            let dy = (Double(y) + 0.5) - cy
            let dySq = dy * dy
            let row = y * width
            for x in minX...maxX {
                let dx = (Double(x) + 0.5) - cx
                if dx * dx + dySq <= radiusSq {
                    ink[row + x] = 255
                    order[row + x] = orderValue
                }
            }
        }
    }

    private static func drawCapsule(
        x0: Double, y0: Double, x1: Double, y1: Double,
        radius: Double, radiusSq: Double,
        width: Int, height: Int,
        orderValue: UInt8,
        ink: inout [UInt8], order: inout [UInt8]
    ) {
        let minX = max(0, Int(floor(min(x0, x1) - radius)))
        let maxX = min(width - 1, Int(ceil(max(x0, x1) + radius)))
        let minY = max(0, Int(floor(min(y0, y1) - radius)))
        let maxY = min(height - 1, Int(ceil(max(y0, y1) + radius)))
        guard minX <= maxX && minY <= maxY else { return }

        let vx = x1 - x0
        let vy = y1 - y0
        let lenSq = vx * vx + vy * vy

        for y in minY...maxY {
            let py = Double(y) + 0.5
            let row = y * width
            for x in minX...maxX {
                let px = Double(x) + 0.5
                let distSq: Double
                if lenSq <= 1e-12 {
                    let dx = px - x0
                    let dy = py - y0
                    distSq = dx * dx + dy * dy
                } else {
                    let t = min(1.0, max(0.0, ((px - x0) * vx + (py - y0) * vy) / lenSq))
                    let cx = x0 + t * vx
                    let cy = y0 + t * vy
                    let dx = px - cx
                    let dy = py - cy
                    distSq = dx * dx + dy * dy
                }
                if distSq <= radiusSq {
                    ink[row + x] = 255
                    order[row + x] = orderValue
                }
            }
        }
    }

    private static func makeUIImage(rgba: [UInt8], width: Int, height: Int) -> UIImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cgImage = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: width * 4,
                  space: colorSpace,
                  bitmapInfo: bitmapInfo,
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              ) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }
}
