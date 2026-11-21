import Foundation
import UIKit

/// Exact Swift port of `export/inktex_model/raster.py` (`render_strokes_to_strip`).
/// Produces a 2-channel [1, 2, 128, W] float32 strip (Channel 0 = anti-aliased ink, Channel 1 = stroke-order progress).
public enum StripRasterizer {
    public struct Output: Sendable {
        public let tensor: [Float]          // NCHW planar [1, 2, height, width], values in [0, 1]
        public let height: Int
        public let width: Int
        public let usedWidth: Int
        public let debugPNGData: Data?
    }

    public static func render(
        strokes: [RawStroke],
        config: ModelManifest.VisionConfig,
        includeDebugImage: Bool = true
    ) -> Output {
        let height = config.height
        let minWidth = config.minWidth
        let maxWidth = config.maxWidth
        let padding = config.padding
        let widthMultiple = config.widthMultiple
        let ss = 3

        let clean = strokes.map(\.points).filter { !$0.isEmpty }
        guard !clean.isEmpty else {
            let blank = [Float](repeating: 0, count: 2 * height * minWidth)
            return Output(tensor: blank, height: height, width: minWidth, usedWidth: max(widthMultiple, min(minWidth, 2 * padding)), debugPNGData: nil)
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
        var debugRGBA = includeDebugImage ? [UInt8](repeating: 0, count: planeSize * 4) : []

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
                tensor[dst] = Float(lumByte) / 255.0
                tensor[planeSize + dst] = Float(ordByte) / 255.0

                if includeDebugImage {
                    let p = dst * 4
                    if lumByte > 0 {
                        // Cyan -> Magenta gradient by stroke order, brightness by ink luminance
                        let t = Float(ordByte) / 255.0
                        let alpha = Float(lumByte) / 255.0
                        debugRGBA[p + 0] = UInt8(min(255, Int((40 + 215 * t) * alpha)))
                        debugRGBA[p + 1] = UInt8(min(255, Int((220 - 120 * t) * alpha)))
                        debugRGBA[p + 2] = UInt8(min(255, Int((255 - 40 * t) * alpha)))
                        debugRGBA[p + 3] = 255
                    } else {
                        debugRGBA[p + 0] = 18
                        debugRGBA[p + 1] = 20
                        debugRGBA[p + 2] = 26
                        debugRGBA[p + 3] = 255
                    }
                }
            }
        }

        let usedWidth = min(width, max(widthMultiple, Int(ceil(contentW + Double(2 * padding)))))
        let pngData = includeDebugImage ? makePNG(rgba: debugRGBA, width: width, height: height) : nil
        return Output(tensor: tensor, height: height, width: width, usedWidth: usedWidth, debugPNGData: pngData)
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

    private static func makePNG(rgba: [UInt8], width: Int, height: Int) -> Data? {
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
        return UIImage(cgImage: cgImage).pngData()
    }
}
