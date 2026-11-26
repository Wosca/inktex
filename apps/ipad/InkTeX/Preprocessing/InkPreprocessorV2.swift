import Foundation

/// Exact FP64 -> FP32 Swift port of `ml/v2/ink_preprocess.py`:
///   - `0.05 * h` deterministic arc-length resampling with coarsening ladder
///   - Overlapping 8-pt sub-stroke chunks (`hop = 7`) + explicit per-point masks
///   - 20-dim point features & 10-dim chunk metadata padded to static CoreML buckets
///   - 16-pt raw-stroke WriterStyleEncoder featurization (`[1, 20, 16, 8]`, `[1, 20, 4]`, `[1, 20]`)
public enum InkPreprocessorV2 {
    public static let chunkSize = 8
    public static let chunkHop = 7
    public static let defaultBuckets = [128, 256, 512, 768]
    public static let pointFeatureDim = 20
    public static let chunkMetaDim = 10

    public static let styleMaxStrokes = 20
    public static let stylePoints = 16
    public static let stylePointDim = 8

    public struct ChunkedOutput: Sendable {
        public let chunkPoints: [Float]      // [bucket * chunkSize * 20]
        public let chunkPointMask: [UInt8]   // [bucket * chunkSize] (0 = valid, 1 = padded)
        public let chunkMeta: [Float]        // [bucket * 10]
        public let chunkMask: [UInt8]        // [bucket] (0 = valid, 1 = padded)
        public let validChunks: Int
        public let bucket: Int
        public let chunkSize: Int
        public let strokeCount: Int
        public let pointCount: Int
        public let glyphScale: Double
        public let coarsened: Bool
        public let overflowed: Bool
    }

    public struct StyleInput: Sendable {
        public let stylePoints: [Float]      // [maxStrokes * pointsPerStroke * 8]
        public let styleAspect: [Float]      // [maxStrokes * 4]
        public let styleMask: [UInt8]        // [maxStrokes] (0 = valid, 1 = padded)
        public let validStrokeCount: Int
        public let maxStrokes: Int
        public let pointsPerStroke: Int
    }

    private struct CleanStroke {
        let xs: [Double]
        let ys: [Double]
        let ts: [Double]?
    }

    private struct ResampledStroke {
        let xs: [Double]
        let ys: [Double]
        let ts: [Double]?
        let arc: Double
    }

    // MARK: - Public Entry Points

    public static func chunkInk(
        strokes: [RawStroke],
        config: ModelManifest.PreprocessingManifest,
        bucketOverride: Int? = nil
    ) -> ChunkedOutput {
        let kSize = config.resolvedChunkSize
        let hop = config.resolvedChunkHop
        let buckets = config.resolvedChunkBuckets
        let maxChunks = bucketOverride ?? config.resolvedMaxChunks
        let dedupeEps = config.resolvedDedupeEps
        let coordClip = config.resolvedCoordClip
        let baseSpacing = config.resampleSpacing > 0 ? config.resampleSpacing : 0.05
        let maxDeltaMs = config.maxDeltaMs > 0 ? config.maxDeltaMs : 2000.0
        let useTiming = config.useTiming

        let clean = cleanRawStrokes(strokes)
        guard !clean.isEmpty else {
            let b = bucketOverride ?? (buckets.first ?? 128)
            return ChunkedOutput(
                chunkPoints: [Float](repeating: 0, count: b * kSize * pointFeatureDim),
                chunkPointMask: [UInt8](repeating: 1, count: b * kSize),
                chunkMeta: [Float](repeating: 0, count: b * chunkMetaDim),
                chunkMask: [UInt8](repeating: 1, count: b),
                validChunks: 0,
                bucket: b,
                chunkSize: kSize,
                strokeCount: 0,
                pointCount: 0,
                glyphScale: 1.0,
                coarsened: false,
                overflowed: false
            )
        }

        let h = estimateGlyphScale(clean)
        let origin = estimateExpressionOrigin(clean)

        var deduped: [CleanStroke] = []
        deduped.reserveCapacity(clean.count)
        for s in clean {
            let nx = s.xs.map { ($0 - origin.0) / h }
            let ny = s.ys.map { ($0 - origin.1) / h }
            deduped.append(dedupeStroke(xs: nx, ys: ny, ts: s.ts, eps: dedupeEps))
        }

        func fits(_ list: [ResampledStroke]) -> Bool {
            countChunks(lengths: list.map { $0.xs.count }, chunkSize: kSize, hop: hop) <= maxChunks
        }

        var effectiveSpacing = baseSpacing
        var resampled = deduped.map { resampleStroke($0, spacing: effectiveSpacing) }
        var coarsened = false
        var overflowed = false

        if !fits(resampled) {
            coarsened = true
            let ladder: [Double] = [1.35, 1.8, 2.5, 3.5, 5.0, 7.5]
            for factor in ladder {
                let candSpacing = baseSpacing * factor
                let cand = deduped.map { resampleStroke($0, spacing: candSpacing) }
                effectiveSpacing = candSpacing
                resampled = cand
                if fits(resampled) {
                    break
                }
            }
            if !fits(resampled) {
                overflowed = true
            }
        }

        let totalPoints = max(resampled.reduce(0) { $0 + $1.xs.count }, 1)
        let logMaxDt = log1p(maxDeltaMs)
        let hasAnyTiming = useTiming && resampled.contains { $0.ts != nil }

        // Per-stroke 20-dim featurization (exact parity with Python resample_and_featurize_strokes)
        var perStrokeFeatures: [[Float]] = []
        perStrokeFeatures.reserveCapacity(resampled.count)

        var globalPtCursor = 0
        var prevStrokeEndX: Double? = nil
        var prevStrokeEndY: Double? = nil
        var prevTime: Double? = nil

        for (sIdx, st) in resampled.enumerated() {
            let nPts = st.xs.count
            var sMinX = st.xs[0], sMaxX = st.xs[0]
            var sMinY = st.ys[0], sMaxY = st.ys[0]
            for i in 1..<nPts {
                let x = st.xs[i], y = st.ys[i]
                if x < sMinX { sMinX = x }
                if x > sMaxX { sMaxX = x }
                if y < sMinY { sMinY = y }
                if y > sMaxY { sMaxY = y }
            }
            let sMidX = 0.5 * (sMinX + sMaxX)
            let sMidY = 0.5 * (sMinY + sMaxY)
            let sScale = max(max(sMaxX - sMinX, sMaxY - sMinY), 1e-4)

            var dx = [Double](repeating: 0, count: nPts)
            var dy = [Double](repeating: 0, count: nPts)
            var step = [Double](repeating: 0, count: nPts)
            if nPts > 1 {
                for i in 1..<nPts {
                    dx[i] = st.xs[i] - st.xs[i - 1]
                    dy[i] = st.ys[i] - st.ys[i - 1]
                    step[i] = hypot(dx[i], dy[i])
                }
            }

            var sinTh = [Double](repeating: 0, count: nPts)
            var cosTh = [Double](repeating: 1, count: nPts)
            var lastSin = 0.0
            var lastCos = 1.0
            if nPts > 1 {
                for i in 1..<nPts {
                    if step[i] > 1e-9 {
                        let th = atan2(dy[i], dx[i])
                        lastSin = sin(th)
                        lastCos = cos(th)
                    }
                    sinTh[i] = lastSin
                    cosTh[i] = lastCos
                }
                sinTh[0] = sinTh[1]
                cosTh[0] = cosTh[1]
            }

            var sinDth = [Double](repeating: 0, count: nPts)
            var cosDth = [Double](repeating: 1, count: nPts)
            if nPts > 2 {
                for i in 2..<nPts {
                    cosDth[i] = cosTh[i] * cosTh[i - 1] + sinTh[i] * sinTh[i - 1]
                    sinDth[i] = sinTh[i] * cosTh[i - 1] - cosTh[i] * sinTh[i - 1]
                }
            }

            var dts = [Double](repeating: 0, count: nPts)
            let strokeHasTiming = hasAnyTiming && st.ts != nil
            if strokeHasTiming, let ts = st.ts {
                if let pt = prevTime {
                    dts[0] = max(0.0, ts[0] - pt)
                }
                if nPts > 1 {
                    for i in 1..<nPts {
                        dts[i] = max(0.0, ts[i] - ts[i - 1])
                    }
                }
                prevTime = ts[nPts - 1]
            }

            let strokeIdxNorm = log1p(Double(sIdx)) / 5.0
            let effSpacingDenom = max(effectiveSpacing, 1e-4)
            let orderDenom = Double(max(totalPoints - 1, 1))

            var feat = [Float](repeating: 0, count: nPts * pointFeatureDim)
            for i in 0..<nPts {
                let base = i * pointFeatureDim
                let x = st.xs[i]
                let y = st.ys[i]
                let prog = nPts > 1 ? Double(i) / Double(nPts - 1) : 0.0
                let orderNorm = Double(globalPtCursor + i) / orderDenom

                feat[base + 0] = Float(clamp(x, -coordClip, coordClip))
                feat[base + 1] = Float(clamp(y, -coordClip, coordClip))
                feat[base + 2] = Float((x - sMidX) / sScale)
                feat[base + 3] = Float((y - sMidY) / sScale)
                feat[base + 4] = Float(clamp(dx[i], -coordClip, coordClip))
                feat[base + 5] = Float(clamp(dy[i], -coordClip, coordClip))
                feat[base + 6] = Float(sinTh[i])
                feat[base + 7] = Float(cosTh[i])
                feat[base + 8] = Float(sinDth[i])
                feat[base + 9] = Float(cosDth[i])
                feat[base + 10] = Float(log1p(step[i] / effSpacingDenom))
                feat[base + 11] = Float(prog)
                feat[base + 12] = Float(strokeIdxNorm)
                feat[base + 13] = Float(orderNorm)
                feat[base + 14] = (i == 0) ? 1.0 : 0.0
                feat[base + 15] = (i == nPts - 1) ? 1.0 : 0.0
                if strokeHasTiming {
                    feat[base + 16] = Float(log1p(min(dts[i], maxDeltaMs)) / logMaxDt)
                    feat[base + 17] = 1.0
                }
            }

            if let px = prevStrokeEndX, let py = prevStrokeEndY {
                feat[18] = Float(clamp(st.xs[0] - px, -coordClip, coordClip))
                feat[19] = Float(clamp(st.ys[0] - py, -coordClip, coordClip))
            }
            prevStrokeEndX = st.xs[nPts - 1]
            prevStrokeEndY = st.ys[nPts - 1]
            globalPtCursor += nPts
            perStrokeFeatures.append(feat)
        }

        // First count chunks so we allocate directly into the final static bucket buffer (zero intermediate stack copies!)
        let rawChunkCount = countChunks(lengths: resampled.map { $0.xs.count }, chunkSize: kSize, hop: hop)
        let cappedChunks = min(rawChunkCount, maxChunks)
        if rawChunkCount > maxChunks {
            overflowed = true
        }
        let bucket = bucketOverride ?? selectBucket(count: cappedChunks, buckets: buckets)
        if cappedChunks > bucket {
            overflowed = true
        }
        let targetMax = min(maxChunks, bucket)

        var chunkPoints = [Float](repeating: 0, count: bucket * kSize * pointFeatureDim)
        var chunkPointMask = [UInt8](repeating: 1, count: bucket * kSize)
        var chunkMeta = [Float](repeating: 0, count: bucket * chunkMetaDim)
        var chunkMask = [UInt8](repeating: 1, count: bucket)

        var chunkCursor = 0
        for (sIdx, st) in resampled.enumerated() {
            if chunkCursor >= targetMax {
                overflowed = true
                break
            }
            let nPts = st.xs.count
            let ptsFeat = perStrokeFeatures[sIdx]

            var sMinX = st.xs[0], sMaxX = st.xs[0]
            var sMinY = st.ys[0], sMaxY = st.ys[0]
            for i in 1..<nPts {
                if st.xs[i] < sMinX { sMinX = st.xs[i] }
                if st.xs[i] > sMaxX { sMaxX = st.xs[i] }
                if st.ys[i] < sMinY { sMinY = st.ys[i] }
                if st.ys[i] > sMaxY { sMaxY = st.ys[i] }
            }
            let sExtMax = max(sMaxX - sMinX, sMaxY - sMinY)
            let isDotStroke: Float = (nPts == 1 || (sExtMax < 0.22 && st.arc < 0.30)) ? 1.0 : 0.0
            let strokeIdxNorm = Float(log1p(Double(sIdx)) / 5.0)

            var starts: [Int] = []
            if nPts <= kSize {
                starts = [0]
            } else {
                var a = 0
                while a < nPts - kSize {
                    starts.append(a)
                    a += hop
                }
                let nextStart = (starts.last ?? -hop) + hop
                if nextStart < nPts {
                    starts.append(nextStart)
                }
            }

            for (localCIdx, a) in starts.enumerated() {
                if chunkCursor >= targetMax {
                    overflowed = true
                    break
                }
                let b = min(a + kSize, nPts)
                let validLen = b - a

                let cpDstBase = chunkCursor * kSize * pointFeatureDim
                let pmDstBase = chunkCursor * kSize
                for k in 0..<validLen {
                    let srcOff = (a + k) * pointFeatureDim
                    let dstOff = cpDstBase + k * pointFeatureDim
                    for f in 0..<pointFeatureDim {
                        chunkPoints[dstOff + f] = ptsFeat[srcOff + f]
                    }
                    chunkPointMask[pmDstBase + k] = 0
                }

                var sumX = 0.0, sumY = 0.0
                var cMinX = st.xs[a], cMaxX = st.xs[a]
                var cMinY = st.ys[a], cMaxY = st.ys[a]
                var cArc = 0.0
                for idx in a..<b {
                    let x = st.xs[idx], y = st.ys[idx]
                    sumX += x
                    sumY += y
                    if x < cMinX { cMinX = x }
                    if x > cMaxX { cMaxX = x }
                    if y < cMinY { cMinY = y }
                    if y > cMaxY { cMaxY = y }
                    if idx > a {
                        cArc += hypot(x - st.xs[idx - 1], y - st.ys[idx - 1])
                    }
                }
                let cx = sumX / Double(validLen)
                let cy = sumY / Double(validLen)
                let mBase = chunkCursor * chunkMetaDim
                chunkMeta[mBase + 0] = Float(clamp(cx, -coordClip, coordClip))
                chunkMeta[mBase + 1] = Float(clamp(cy, -coordClip, coordClip))
                chunkMeta[mBase + 2] = Float(log1p(cMaxX - cMinX))
                chunkMeta[mBase + 3] = Float(log1p(cMaxY - cMinY))
                chunkMeta[mBase + 4] = Float(log1p(cArc))
                chunkMeta[mBase + 5] = ptsFeat[a * pointFeatureDim + 11]
                chunkMeta[mBase + 6] = ptsFeat[(b - 1) * pointFeatureDim + 11]
                chunkMeta[mBase + 7] = strokeIdxNorm
                chunkMeta[mBase + 8] = Float(log1p(Double(localCIdx)) / 4.0)
                chunkMeta[mBase + 9] = isDotStroke
                chunkMask[chunkCursor] = 0
                chunkCursor += 1
            }
        }

        return ChunkedOutput(
            chunkPoints: chunkPoints,
            chunkPointMask: chunkPointMask,
            chunkMeta: chunkMeta,
            chunkMask: chunkMask,
            validChunks: chunkCursor,
            bucket: bucket,
            chunkSize: kSize,
            strokeCount: resampled.count,
            pointCount: totalPoints,
            glyphScale: h,
            coarsened: coarsened,
            overflowed: overflowed
        )
    }

    /// Exact Swift port of `extract_style_strokes` in `ml/v2/ink_preprocess.py`.
    public static func extractStyleStrokes(
        strokes: [RawStroke],
        maxStrokes: Int = styleMaxStrokes,
        pointsPerStroke: Int = stylePoints
    ) -> StyleInput {
        let clean = cleanRawStrokes(strokes)
        var outPts = [Float](repeating: 0, count: maxStrokes * pointsPerStroke * stylePointDim)
        var outAsp = [Float](repeating: 0, count: maxStrokes * 4)
        var outMask = [UInt8](repeating: 1, count: maxStrokes)

        guard !clean.isEmpty else {
            return StyleInput(
                stylePoints: outPts,
                styleAspect: outAsp,
                styleMask: outMask,
                validStrokeCount: 0,
                maxStrokes: maxStrokes,
                pointsPerStroke: pointsPerStroke
            )
        }

        let h = estimateGlyphScale(clean)
        let count = min(clean.count, maxStrokes)

        for idx in 0..<count {
            let s = clean[idx]
            let n = s.xs.count
            let snX = s.xs.map { $0 / h }
            let snY = s.ys.map { $0 / h }

            var cum = [Double](repeating: 0, count: n)
            if n > 1 {
                for i in 1..<n {
                    cum[i] = cum[i - 1] + hypot(snX[i] - snX[i - 1], snY[i] - snY[i - 1])
                }
            }
            let total = cum[n - 1]

            var rx = [Double](repeating: snX[0], count: pointsPerStroke)
            var ry = [Double](repeating: snY[0], count: pointsPerStroke)
            if total > 1e-9 && n > 1 {
                let denom = Double(max(pointsPerStroke - 1, 1))
                var targets = [Double](repeating: 0, count: pointsPerStroke)
                for p in 0..<pointsPerStroke {
                    targets[p] = total * Double(p) / denom
                }
                rx = linearInterp(targets: targets, xp: cum, fp: snX)
                ry = linearInterp(targets: targets, xp: cum, fp: snY)
            }

            var minX = rx[0], maxX = rx[0], minY = ry[0], maxY = ry[0]
            for p in 1..<pointsPerStroke {
                if rx[p] < minX { minX = rx[p] }
                if rx[p] > maxX { maxX = rx[p] }
                if ry[p] < minY { minY = ry[p] }
                if ry[p] > maxY { maxY = ry[p] }
            }
            let extX = maxX - minX
            let extY = maxY - minY
            let midX = 0.5 * (minX + maxX)
            let midY = 0.5 * (minY + maxY)
            let scale = max(max(extX, extY), 1e-4)

            var locX = [Double](repeating: 0, count: pointsPerStroke)
            var locY = [Double](repeating: 0, count: pointsPerStroke)
            for p in 0..<pointsPerStroke {
                locX[p] = (rx[p] - midX) / scale
                locY[p] = (ry[p] - midY) / scale
            }

            var dx = [Double](repeating: 0, count: pointsPerStroke)
            var dy = [Double](repeating: 0, count: pointsPerStroke)
            var step = [Double](repeating: 0, count: pointsPerStroke)
            for p in 1..<pointsPerStroke {
                dx[p] = locX[p] - locX[p - 1]
                dy[p] = locY[p] - locY[p - 1]
                step[p] = hypot(dx[p], dy[p])
            }

            var sinTh = [Double](repeating: 0, count: pointsPerStroke)
            var cosTh = [Double](repeating: 1, count: pointsPerStroke)
            var lastSin = 0.0, lastCos = 1.0
            for p in 1..<pointsPerStroke {
                if step[p] > 1e-9 {
                    let th = atan2(dy[p], dx[p])
                    lastSin = sin(th)
                    lastCos = cos(th)
                }
                sinTh[p] = lastSin
                cosTh[p] = lastCos
            }
            if pointsPerStroke > 1 {
                sinTh[0] = sinTh[1]
                cosTh[0] = cosTh[1]
            }

            var sinDth = [Double](repeating: 0, count: pointsPerStroke)
            var cosDth = [Double](repeating: 1, count: pointsPerStroke)
            if pointsPerStroke > 2 {
                for p in 2..<pointsPerStroke {
                    cosDth[p] = cosTh[p] * cosTh[p - 1] + sinTh[p] * sinTh[p - 1]
                    sinDth[p] = sinTh[p] * cosTh[p - 1] - cosTh[p] * sinTh[p - 1]
                }
            }

            let strokeBase = idx * pointsPerStroke * stylePointDim
            for p in 0..<pointsPerStroke {
                let base = strokeBase + p * stylePointDim
                outPts[base + 0] = Float(locX[p])
                outPts[base + 1] = Float(locY[p])
                outPts[base + 2] = Float(dx[p])
                outPts[base + 3] = Float(dy[p])
                outPts[base + 4] = Float(sinTh[p])
                outPts[base + 5] = Float(cosTh[p])
                outPts[base + 6] = Float(sinDth[p])
                outPts[base + 7] = Float(cosDth[p])
            }

            let endDx = locX[pointsPerStroke - 1] - locX[0]
            let endDy = locY[pointsPerStroke - 1] - locY[0]
            let dirAng = atan2(endDy, endDx)
            let aspBase = idx * 4
            outAsp[aspBase + 0] = Float(log1p(extX))
            outAsp[aspBase + 1] = Float(log1p(extY))
            outAsp[aspBase + 2] = Float(sin(dirAng))
            outAsp[aspBase + 3] = Float(cos(dirAng))
            outMask[idx] = 0
        }

        return StyleInput(
            stylePoints: outPts,
            styleAspect: outAsp,
            styleMask: outMask,
            validStrokeCount: count,
            maxStrokes: maxStrokes,
            pointsPerStroke: pointsPerStroke
        )
    }

    /// Public helper shared with `StripRasterizer.renderV2` for identical glyph scale `h`.
    public static func estimateGlyphScale(strokes: [RawStroke]) -> Double {
        let clean = cleanRawStrokes(strokes)
        return estimateGlyphScale(clean)
    }

    /// Scale- and translation-invariant geometric ink hash for optional on-device correction lookup.
    public static func inkGeometryHash(_ strokes: [RawStroke]) -> UInt64 {
        let clean = cleanRawStrokes(strokes)
        guard !clean.isEmpty else { return 0 }
        let h = estimateGlyphScale(clean)
        let origin = estimateExpressionOrigin(clean)

        var hash: UInt64 = 14695981039346656037
        let prime: UInt64 = 1099511628211
        func mix(_ v: UInt64) {
            hash ^= v
            hash = hash &* prime
        }
        mix(UInt64(clean.count))
        for s in clean {
            let n = s.xs.count
            // Quantize start, mid, end points + bbox to ~0.12*h bins so redrawing the same equation matches cleanly
            let idxs = n == 1 ? [0] : [0, n / 2, n - 1]
            for i in idxs {
                let qx = Int64(Foundation.round(((s.xs[i] - origin.0) / h) * 8.0))
                let qy = Int64(Foundation.round(((s.ys[i] - origin.1) / h) * 8.0))
                mix(UInt64(bitPattern: qx))
                mix(UInt64(bitPattern: qy))
            }
        }
        return hash
    }

    public static func selectBucket(count: Int, buckets: [Int] = defaultBuckets) -> Int {
        for b in buckets {
            if count <= b { return b }
        }
        return buckets.last ?? 768
    }

    // MARK: - Internal Geometry Helpers

    private static func cleanRawStrokes(_ strokes: [RawStroke]) -> [CleanStroke] {
        var out: [CleanStroke] = []
        out.reserveCapacity(strokes.count)
        for stroke in strokes {
            var xs: [Double] = []
            var ys: [Double] = []
            var ts: [Double] = []
            var allHaveTime = true
            xs.reserveCapacity(stroke.points.count)
            ys.reserveCapacity(stroke.points.count)
            ts.reserveCapacity(stroke.points.count)
            for pt in stroke.points {
                guard pt.x.isFinite && pt.y.isFinite else { continue }
                xs.append(pt.x)
                ys.append(pt.y)
                if let t = pt.timeMs, t.isFinite {
                    ts.append(t)
                } else {
                    allHaveTime = false
                }
            }
            if !xs.isEmpty {
                out.append(CleanStroke(xs: xs, ys: ys, ts: allHaveTime ? ts : nil))
            }
        }
        return out
    }

    private static func estimateGlyphScale(_ strokes: [CleanStroke]) -> Double {
        guard !strokes.isEmpty else { return 1.0 }
        var minX = Double.greatestFiniteMagnitude
        var maxX = -Double.greatestFiniteMagnitude
        var minY = Double.greatestFiniteMagnitude
        var maxY = -Double.greatestFiniteMagnitude

        var sizes: [Double] = []
        sizes.reserveCapacity(strokes.count)

        for s in strokes {
            var sMinX = s.xs[0], sMaxX = s.xs[0]
            var sMinY = s.ys[0], sMaxY = s.ys[0]
            for i in 0..<s.xs.count {
                let x = s.xs[i], y = s.ys[i]
                if x < sMinX { sMinX = x }
                if x > sMaxX { sMaxX = x }
                if y < sMinY { sMinY = y }
                if y > sMaxY { sMaxY = y }
            }
            if sMinX < minX { minX = sMinX }
            if sMaxX > maxX { maxX = sMaxX }
            if sMinY < minY { minY = sMinY }
            if sMaxY > maxY { maxY = sMaxY }
        }

        let exprH = maxY - minY
        let exprW = maxX - minX
        let exprMax = max(max(exprH, exprW), 1e-6)

        for s in strokes {
            var sMinX = s.xs[0], sMaxX = s.xs[0]
            var sMinY = s.ys[0], sMaxY = s.ys[0]
            for i in 1..<s.xs.count {
                let x = s.xs[i], y = s.ys[i]
                if x < sMinX { sMinX = x }
                if x > sMaxX { sMaxX = x }
                if y < sMinY { sMinY = y }
                if y > sMaxY { sMaxY = y }
            }
            let val = max(sMaxY - sMinY, 0.5 * (sMaxX - sMinX))
            if val > 1e-6 * exprMax {
                sizes.append(val)
            }
        }
        guard !sizes.isEmpty else { return exprMax }
        let h = median(sizes)
        let lo = exprH > 1e-6 ? 0.08 * exprH : 1e-6
        let hi = max(max(exprH, exprMax * 0.5), 1e-6)
        return clamp(h, max(lo, 1e-6), max(max(hi, lo), 1e-6))
    }

    private static func estimateExpressionOrigin(_ strokes: [CleanStroke]) -> (Double, Double) {
        var minX = Double.greatestFiniteMagnitude
        var maxX = -Double.greatestFiniteMagnitude
        var strokeYs: [Double] = []
        strokeYs.reserveCapacity(strokes.count)
        for s in strokes {
            var sMinY = s.ys[0], sMaxY = s.ys[0]
            for i in 0..<s.xs.count {
                let x = s.xs[i], y = s.ys[i]
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < sMinY { sMinY = y }
                if y > sMaxY { sMaxY = y }
            }
            strokeYs.append(0.5 * (sMinY + sMaxY))
        }
        return (0.5 * (minX + maxX), median(strokeYs))
    }

    private static func dedupeStroke(xs: [Double], ys: [Double], ts: [Double]?, eps: Double) -> CleanStroke {
        let n = xs.count
        if n <= 2 {
            return CleanStroke(xs: xs, ys: ys, ts: ts)
        }
        var keep: [Int] = [0]
        keep.reserveCapacity(n)
        for i in 1..<(n - 1) {
            let last = keep[keep.count - 1]
            if hypot(xs[i] - xs[last], ys[i] - ys[last]) >= eps {
                keep.append(i)
            }
        }
        keep.append(n - 1)
        let outX = keep.map { xs[$0] }
        let outY = keep.map { ys[$0] }
        let outT = ts.map { tArr in keep.map { tArr[$0] } }
        return CleanStroke(xs: outX, ys: outY, ts: outT)
    }

    private static func resampleStroke(_ stroke: CleanStroke, spacing: Double) -> ResampledStroke {
        let n = stroke.xs.count
        if n <= 1 || spacing <= 0 {
            return ResampledStroke(xs: stroke.xs, ys: stroke.ys, ts: stroke.ts, arc: 0.0)
        }
        var cum = [Double](repeating: 0, count: n)
        for i in 1..<n {
            cum[i] = cum[i - 1] + hypot(stroke.xs[i] - stroke.xs[i - 1], stroke.ys[i] - stroke.ys[i - 1])
        }
        let total = cum[n - 1]
        if total <= 1e-9 {
            return ResampledStroke(
                xs: [stroke.xs[0]],
                ys: [stroke.ys[0]],
                ts: stroke.ts.map { [$0[0]] },
                arc: 0.0
            )
        }
        let steps = Int(floor(total / spacing))
        var targets: [Double]
        if steps <= 0 {
            targets = [0.0, total]
        } else {
            targets = (0...steps).map { Double($0) * spacing }
            if total - targets[targets.count - 1] > 1e-6 * max(1.0, total) {
                targets.append(total)
            } else {
                targets[targets.count - 1] = total
            }
        }
        let rx = linearInterp(targets: targets, xp: cum, fp: stroke.xs)
        let ry = linearInterp(targets: targets, xp: cum, fp: stroke.ys)
        let rt = stroke.ts.map { linearInterp(targets: targets, xp: cum, fp: $0) }
        return ResampledStroke(xs: rx, ys: ry, ts: rt, arc: total)
    }

    private static func countChunks(lengths: [Int], chunkSize: Int, hop: Int) -> Int {
        var total = 0
        for n in lengths {
            if n <= chunkSize {
                total += 1
            } else {
                total += 1 + Int(ceil(Double(n - chunkSize) / Double(hop)))
            }
        }
        return total
    }

    private static func linearInterp(targets: [Double], xp: [Double], fp: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: targets.count)
        let lastIdx = xp.count - 1
        var seg = 0
        for (i, t) in targets.enumerated() {
            if t <= xp[0] {
                out[i] = fp[0]
                continue
            }
            if t >= xp[lastIdx] {
                out[i] = fp[lastIdx]
                continue
            }
            while seg < lastIdx - 1 && xp[seg + 1] < t {
                seg += 1
            }
            let x0 = xp[seg], x1 = xp[seg + 1]
            let denom = x1 - x0
            if denom <= 1e-12 {
                out[i] = fp[seg]
            } else {
                let alpha = (t - x0) / denom
                out[i] = fp[seg] + alpha * (fp[seg + 1] - fp[seg])
            }
        }
        return out
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0.0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 {
            return sorted[mid]
        }
        return 0.5 * (sorted[mid - 1] + sorted[mid])
    }

    @inline(__always)
    private static func clamp(_ v: Double, _ lo: Double, _ hi: Double) -> Double {
        min(hi, max(lo, v))
    }
}
