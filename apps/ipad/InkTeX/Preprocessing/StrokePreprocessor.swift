import Foundation

/// Exact Swift port of `export/inktex_model/preprocessing.py` (Stage 2 glyph normalization).
public enum StrokePreprocessor {
    public struct Output: Sendable {
        public let features: [Float]        // [numPoints, 10] row-major
        public let paddingMask: [UInt8]     // [numPoints], all 0 (valid)
        public let pointCount: Int
        public let strokeCount: Int
    }

    public static func preprocess(
        strokes: [RawStroke],
        config: ModelManifest.PreprocessingManifest
    ) -> Output? {
        let nonEmpty = strokes.map(\.points).filter { !$0.isEmpty }
        guard !nonEmpty.isEmpty else { return nil }

        // 1. Normalization ("glyph" default for Stage 2 bimodal)
        let centerX: Double
        let centerY: Double
        let scale: Double

        if config.normalization == "glyph" {
            let heights = nonEmpty.map { stroke -> Double in
                let ys = stroke.map(\.y)
                return (ys.max() ?? 0) - (ys.min() ?? 0)
            }
            let usefulHeights = heights.filter { $0 > 1e-9 }
            let widths = nonEmpty.map { stroke -> Double in
                let xs = stroke.map(\.x)
                return (xs.max() ?? 0) - (xs.min() ?? 0)
            }
            let fallbackWidths = widths.filter { $0 > 1e-9 }
            let chosen = usefulHeights.isEmpty ? fallbackWidths : usefulHeights
            scale = max(chosen.isEmpty ? 1.0 : median(chosen), 1e-9)
            centerX = nonEmpty[0][0].x
            centerY = nonEmpty[0][0].y
        } else {
            var minX = Double.greatestFiniteMagnitude
            var maxX = -Double.greatestFiniteMagnitude
            var minY = Double.greatestFiniteMagnitude
            var maxY = -Double.greatestFiniteMagnitude
            for stroke in nonEmpty {
                for pt in stroke {
                    minX = min(minX, pt.x)
                    maxX = max(maxX, pt.x)
                    minY = min(minY, pt.y)
                    maxY = max(maxY, pt.y)
                }
            }
            centerX = (minX + maxX) / 2.0
            centerY = (minY + maxY) / 2.0
            scale = max(max(maxX - minX, maxY - minY), 1e-9)
        }

        var normalized = nonEmpty.map { stroke in
            stroke.map { pt in
                RawStrokePoint(
                    x: (pt.x - centerX) / scale,
                    y: (pt.y - centerY) / scale,
                    timeMs: pt.timeMs
                )
            }
        }

        // 2. Spatial arc-length resampling
        if config.resamplingEnabled && config.resampleSpacing > 0 {
            normalized = normalized.map { stroke in
                resampleByDistance(
                    stroke,
                    spacing: config.resampleSpacing,
                    minPoints: config.minResampledPointsPerStroke
                )
            }
        }

        // 3. Per-stroke point cap
        if config.normalization == "glyph" {
            normalized = normalized.map { resampleToCount($0, count: config.maxPointsPerStroke) }
        }

        // 4. Global working budget cap
        let totalPoints = normalized.reduce(0) { $0 + $1.count }
        if totalPoints > config.maxPoints {
            let counts = allocateCounts(
                lengths: normalized.map(\.count),
                budget: config.maxPoints,
                minimum: config.minPointsPerStroke
            )
            normalized = zip(normalized, counts).map { resampleToCount($0.0, count: $0.1) }
        }

        // 5. Build 10-channel feature matrix
        let finalPointCount = normalized.reduce(0) { $0 + $1.count }
        guard finalPointCount >= 1 else { return nil }

        // Ensure at least 2 points for dynamic-shape ONNX encoder guard (a 1-point tap is duplicated safely)
        if finalPointCount == 1, let only = normalized.first?.first {
            normalized = [[only, only]]
        }

        let actualPointCount = normalized.reduce(0) { $0 + $1.count }
        var features: [Float] = []
        features.reserveCapacity(actualPointCount * 10)

        var prevX: Float? = nil
        var prevY: Float? = nil
        var prevTime: Double? = nil
        let denom = Double(max(normalized.count - 1, 1))
        let logMaxDt = log1p(config.maxDeltaMs)

        for (strokeIdx, stroke) in normalized.enumerated() {
            let strokePos: Float = config.normalization == "glyph"
                ? Float(log1p(Double(strokeIdx)) / 8.0)
                : Float(Double(strokeIdx) / denom)

            for (ptIdx, pt) in stroke.enumerated() {
                let x = Float(pt.x)
                let y = Float(pt.y)
                let isStart = ptIdx == 0
                let isEnd = ptIdx == stroke.count - 1
                let dx: Float = (isStart || prevX == nil) ? 0.0 : (x - prevX!)
                let dy: Float = (isStart || prevY == nil) ? 0.0 : (y - prevY!)

                let dt: Float
                if config.useTiming, let t = pt.timeMs, let ptPrev = prevTime {
                    let deltaMs = max(0.0, t - ptPrev)
                    dt = Float(log1p(min(deltaMs, config.maxDeltaMs)) / logMaxDt)
                } else {
                    dt = 0.0
                }

                features.append(x)
                features.append(y)
                features.append(dx)
                features.append(dy)
                features.append(dt)
                features.append(0.0) // pressure (disabled in Stage 2)
                features.append(0.0) // pressure_available
                features.append(isStart ? 1.0 : 0.0)
                features.append(isEnd ? 1.0 : 0.0)
                features.append(strokePos)

                prevX = x
                prevY = y
                if let t = pt.timeMs {
                    prevTime = t
                }
            }
        }

        return Output(
            features: features,
            paddingMask: [UInt8](repeating: 0, count: actualPointCount),
            pointCount: actualPointCount,
            strokeCount: normalized.count
        )
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 {
            return sorted[mid]
        }
        return 0.5 * (sorted[mid - 1] + sorted[mid])
    }

    private static func interpolate(_ a: RawStrokePoint, _ b: RawStrokePoint, alpha: Double) -> RawStrokePoint {
        let t: Double?
        if let ta = a.timeMs, let tb = b.timeMs {
            t = ta + alpha * (tb - ta)
        } else {
            t = nil
        }
        return RawStrokePoint(
            x: a.x + alpha * (b.x - a.x),
            y: a.y + alpha * (b.y - a.y),
            timeMs: t
        )
    }

    private static func resampleByDistance(
        _ points: [RawStrokePoint],
        spacing: Double,
        minPoints: Int
    ) -> [RawStrokePoint] {
        guard points.count > 1, spacing > 0 else { return points }

        var cumulative = [Double](repeating: 0.0, count: points.count)
        for i in 1..<points.count {
            let dx = points[i].x - points[i - 1].x
            let dy = points[i].y - points[i - 1].y
            cumulative[i] = cumulative[i - 1] + hypot(dx, dy)
        }
        let total = cumulative[points.count - 1]
        if total <= 1e-12 {
            return [points[0]]
        }

        var targets: [Double] = []
        var cur = 0.0
        while cur < total {
            targets.append(cur)
            cur += spacing
        }
        if targets.isEmpty || abs(targets[targets.count - 1] - total) > 1e-9 * max(1.0, total) {
            targets.append(total)
        }

        let minimum = min(max(minPoints, 2), points.count)
        if targets.count < minimum {
            targets = (0..<minimum).map { i in
                minimum == 1 ? 0.0 : total * Double(i) / Double(minimum - 1)
            }
        }

        var result: [RawStrokePoint] = []
        result.reserveCapacity(targets.count)
        for target in targets {
            let right = min(upperBound(cumulative, value: target), points.count - 1)
            let left = max(0, right - 1)
            let span = cumulative[right] - cumulative[left]
            let alpha = span <= 1e-12 ? 0.0 : (target - cumulative[left]) / span
            result.append(interpolate(points[left], points[right], alpha: alpha))
        }
        return result
    }

    private static func upperBound(_ sorted: [Double], value: Double) -> Int {
        var lo = 0
        var hi = sorted.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if sorted[mid] <= value {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        return lo
    }

    private static func resampleToCount(_ points: [RawStrokePoint], count: Int) -> [RawStrokePoint] {
        guard count < points.count else { return points }
        if count <= 1 { return [points[0]] }
        var out: [RawStrokePoint] = []
        out.reserveCapacity(count)
        let last = points.count - 1
        for i in 0..<count {
            let idx: Int
            if i == 0 {
                idx = 0
            } else if i == count - 1 {
                idx = last
            } else {
                idx = Int(Foundation.rint(Double(last) * Double(i) / Double(count - 1)))
            }
            out.append(points[idx])
        }
        return out
    }

    private static func allocateCounts(lengths: [Int], budget: Int, minimum: Int) -> [Int] {
        var floors = lengths.map { min($0, max(1, minimum)) }
        if floors.reduce(0, +) > budget {
            floors = [Int](repeating: 1, count: lengths.count)
        }
        let remaining = budget - floors.reduce(0, +)
        let capacities = zip(lengths, floors).map { $0 - $1 }
        let totalCap = capacities.reduce(0, +)
        if remaining <= 0 || totalCap == 0 {
            return floors
        }
        let shares = capacities.map { Double(remaining) * Double($0) / Double(totalCap) }
        var counts = zip(zip(floors, capacities), shares).map { pair, share in
            pair.0 + min(pair.1, Int(share))
        }
        var spare = budget - counts.reduce(0, +)
        let order = (0..<lengths.count).sorted { a, b in
            (shares[a] - floor(shares[a])) > (shares[b] - floor(shares[b]))
        }
        for idx in order {
            if spare <= 0 { break }
            if counts[idx] < lengths[idx] {
                counts[idx] += 1
                spare -= 1
            }
        }
        return counts
    }
}
