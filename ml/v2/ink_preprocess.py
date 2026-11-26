"""InkTeX v2 ink preprocessing: 0.05*h resampling, overlapping 8-pt sub-stroke chunks,
per-point chunk padding masks, shared point featurization for ablations, static bucket padding
(zero data-dependent shapes in ONNX/CoreML), and 16-pt raw-stroke style extraction.

All reference geometry is computed in FP64 (with median averaging the two middle values on even
counts, matching Swift's median()) before casting to FP32 so 1-point dots never underflow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np

CHUNK_SIZE = 8
CHUNK_HOP = 7
CHUNK_BUCKETS = (128, 256, 512, 768)
STYLE_POINTS = 16
STYLE_MAX_STROKES = 20

POINT_FEATURE_DIM = 20
POINT_FEATURE_NAMES = (
    "x_expr", "y_expr",              # 0, 1: expression-frame coords (centered, scaled by glyph h)
    "x_local", "y_local",            # 2, 3: stroke-local coords (centered at stroke mid, scaled by stroke extent)
    "dx", "dy",                      # 4, 5: within-stroke delta
    "sin_theta", "cos_theta",        # 6, 7: tangent direction
    "sin_dtheta", "cos_dtheta",      # 8, 9: turn angle (never feed raw angle to Swift/model!)
    "log_step",                      # 10: log1p(step / effective_spacing)
    "stroke_progress",               # 11: [0, 1] along stroke
    "stroke_index_norm",             # 12: log1p(stroke_idx) / 5.0
    "point_order_norm",              # 13: [0, 1] across whole expression
    "is_start", "is_end",            # 14, 15: stroke start / end
    "dt_norm", "time_avail",         # 16, 17: timing + availability
    "penup_dx", "penup_dy",          # 18, 19: pen-up jump from previous stroke end (start pts only)
)

CHUNK_META_DIM = 10
CHUNK_META_NAMES = (
    "cx", "cy",                      # 0, 1: chunk centroid in expression frame
    "log_w", "log_h",                # 2, 3: log1p(chunk bbox width / height)
    "log_arc",                       # 4: log1p(chunk arc length)
    "start_progress", "end_progress",  # 5, 6: stroke progress at chunk start/end
    "stroke_idx_norm",               # 7: log1p(stroke_idx) / 5.0
    "chunk_idx_in_stroke_norm",      # 8: log1p(local_chunk_idx) / 4.0
    "is_dot",                        # 9: 1.0 if small dot stroke
)

STYLE_POINT_DIM = 8  # x_loc, y_loc, dx, dy, sin_theta, cos_theta, sin_dtheta, cos_dtheta


@dataclass(slots=True)
class InkPreprocessV2Config:
    resample_spacing: float = 0.05
    chunk_size: int = CHUNK_SIZE
    chunk_hop: int = CHUNK_HOP
    buckets: tuple[int, ...] = CHUNK_BUCKETS
    max_chunks: int = CHUNK_BUCKETS[-1]
    dedupe_eps: float = 1e-3
    coord_clip: float = 32.0
    max_delta_ms: float = 2000.0
    use_timing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InkPreprocessV2Config":
        fields = cls.__dataclass_fields__
        cleaned = {}
        for k, v in value.items():
            if k in fields:
                cleaned[k] = tuple(v) if k == "buckets" else v
        return cls(**cleaned)


@dataclass(slots=True)
class ResampledStrokeSet:
    per_stroke_features: list[np.ndarray]  # list of [n_pts, 20] float32
    per_stroke_xy: list[np.ndarray]        # list of [n_pts, 2] float64 normalized
    per_stroke_arc: list[float]
    glyph_scale: float
    origin: np.ndarray
    effective_spacing: float
    coarsened: bool
    overflowed: bool


@dataclass(slots=True)
class ChunkedInk:
    chunk_points: np.ndarray       # [N_bucket, K, 20] float32
    chunk_point_mask: np.ndarray   # [N_bucket, K] bool (True = padded tail repeat point!)
    chunk_meta: np.ndarray         # [N_bucket, 10] float32
    chunk_mask: np.ndarray         # [N_bucket] bool (True = padded chunk)
    chunk_stroke_ids: np.ndarray   # [N_bucket] int64 (-1 for padded)
    chunk_centers_raw: np.ndarray  # [N_bucket, 2] float32 in original ink space
    valid_chunks: int
    stroke_count: int
    glyph_scale: float
    origin: np.ndarray
    bucket: int
    coarsened: bool
    overflowed: bool


def clean_raw_strokes(
    strokes: Sequence[np.ndarray],
    times: Sequence[np.ndarray | None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    out_xy: list[np.ndarray] = []
    out_t: list[np.ndarray | None] = []
    for idx, stroke in enumerate(strokes):
        arr = np.asarray(stroke, dtype=np.float64)
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            continue
        xy = arr[:, :2]
        finite = np.isfinite(xy).all(axis=1)
        if not finite.any():
            continue
        xy = xy[finite]
        t = None
        if times is not None and idx < len(times) and times[idx] is not None:
            t_arr = np.asarray(times[idx], dtype=np.float64)
            if len(t_arr) == len(arr) and np.isfinite(t_arr[finite]).all():
                t = t_arr[finite]
        out_xy.append(xy)
        out_t.append(t)
    return out_xy, out_t


def estimate_glyph_scale(strokes: Sequence[np.ndarray]) -> float:
    """Robust glyph height h = median of non-degenerate stroke max(height, 0.5*width),
    clamped safely against expression bounding-box height (FP64, matches Swift median)."""
    if not strokes:
        return 1.0
    xy = np.concatenate(strokes, axis=0)
    expr_h = float(np.ptp(xy[:, 1]))
    expr_w = float(np.ptp(xy[:, 0]))
    expr_max = max(expr_h, expr_w, 1e-6)

    sizes = []
    for s in strokes:
        sh = float(np.ptp(s[:, 1]))
        sw = float(np.ptp(s[:, 0]))
        val = max(sh, 0.5 * sw)
        if val > 1e-6 * expr_max:
            sizes.append(val)
    if not sizes:
        return expr_max
    h = float(np.median(sizes))
    lo = 0.08 * expr_h if expr_h > 1e-6 else 1e-6
    hi = max(expr_h, expr_max * 0.5, 1e-6)
    return float(np.clip(h, max(lo, 1e-6), max(hi, lo, 1e-6)))


def estimate_expression_origin(strokes: Sequence[np.ndarray]) -> np.ndarray:
    xy = np.concatenate(strokes, axis=0)
    x_mid = 0.5 * (float(xy[:, 0].min()) + float(xy[:, 0].max()))
    stroke_ys = [0.5 * (float(s[:, 1].min()) + float(s[:, 1].max())) for s in strokes]
    y_mid = float(np.median(stroke_ys))
    return np.array([x_mid, y_mid], dtype=np.float64)


def _dedupe_stroke(xy: np.ndarray, t: np.ndarray | None, eps: float) -> tuple[np.ndarray, np.ndarray | None]:
    if len(xy) <= 2:
        return xy, t
    keep = [0]
    for i in range(1, len(xy) - 1):
        if float(np.linalg.norm(xy[i] - xy[keep[-1]])) >= eps:
            keep.append(i)
    keep.append(len(xy) - 1)
    idx = np.asarray(keep, dtype=np.int64)
    return xy[idx], (t[idx] if t is not None else None)


def _resample_stroke(
    xy: np.ndarray,
    t: np.ndarray | None,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Deterministic arc-length resample using explicit integer count `floor(total / spacing)`
    so Python and Swift produce identical point counts at every boundary."""
    if len(xy) <= 1 or spacing <= 0:
        return xy.copy(), (t.copy() if t is not None else None), 0.0
    diffs = np.diff(xy, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total <= 1e-9:
        return xy[:1].copy(), (t[:1].copy() if t is not None else None), 0.0
    steps = int(math.floor(total / spacing))
    if steps <= 0:
        targets = np.array([0.0, total], dtype=np.float64)
    else:
        interior = np.arange(steps + 1, dtype=np.float64) * spacing
        if total - float(interior[-1]) > 1e-6 * max(1.0, total):
            targets = np.append(interior, total)
        else:
            targets = interior
            targets[-1] = total
    rx = np.interp(targets, cum, xy[:, 0])
    ry = np.interp(targets, cum, xy[:, 1])
    rxy = np.stack([rx, ry], axis=1)
    rt = np.interp(targets, cum, t) if t is not None else None
    return rxy, rt, total


def select_bucket(count: int, buckets: Sequence[int] = CHUNK_BUCKETS) -> int:
    for b in buckets:
        if count <= b:
            return int(b)
    return int(buckets[-1])


def _count_chunks_for_lengths(lengths: Sequence[int], chunk_size: int, hop: int) -> int:
    total = 0
    for n in lengths:
        if n <= chunk_size:
            total += 1
        else:
            total += 1 + math.ceil((n - chunk_size) / hop)
    return total


def resample_and_featurize_strokes(
    strokes: Sequence[np.ndarray],
    times: Sequence[np.ndarray | None] | None = None,
    config: InkPreprocessV2Config | None = None,
    *,
    max_total_points: int | None = None,
    max_total_chunks: int | None = None,
    drop_timing: bool = False,
    post_resample_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ResampledStrokeSet:
    """Shared endpoint-preserving resampler & 20-dim point featurizer used by BOTH
    `chunk_ink_v2` and the Phase 0B Point-Global ablation arm for 100% parity."""
    cfg = config or InkPreprocessV2Config()
    clean_xy, clean_t = clean_raw_strokes(strokes, times)
    if not clean_xy:
        return ResampledStrokeSet([], [], [], 1.0, np.zeros(2, dtype=np.float64), cfg.resample_spacing, False, False)

    h = estimate_glyph_scale(clean_xy)
    origin = estimate_expression_origin(clean_xy)
    norm_xy = [(s - origin) / h for s in clean_xy]
    deduped = [_dedupe_stroke(s, t, cfg.dedupe_eps) for s, t in zip(norm_xy, clean_t)]

    chunk_cap = cfg.max_chunks if max_total_chunks is None else max_total_chunks
    point_cap = max_total_points

    def fits(res: list[tuple[np.ndarray, np.ndarray | None, float]]) -> bool:
        if point_cap is not None and sum(len(s) for s, _, _ in res) > point_cap:
            return False
        if chunk_cap is not None:
            if _count_chunks_for_lengths([len(s) for s, _, _ in res], cfg.chunk_size, cfg.chunk_hop) > chunk_cap:
                return False
        return True

    effective_spacing = cfg.resample_spacing
    resampled = [_resample_stroke(s, t, effective_spacing) for s, t in deduped]
    coarsened = False
    overflowed = False
    if not fits(resampled):
        coarsened = True
        for factor in (1.35, 1.8, 2.5, 3.5, 5.0, 7.5):
            cand_spacing = cfg.resample_spacing * factor
            cand_res = [_resample_stroke(s, t, cand_spacing) for s, t in deduped]
            effective_spacing = cand_spacing
            resampled = cand_res
            if fits(resampled):
                break
        if not fits(resampled):
            overflowed = True

    # Apply subtle post-resample jitter here (never before resampling, so arc length isn't inflated!)
    if post_resample_jitter > 0.0 and rng is not None:
        jittered = []
        for s_xy, s_t, s_arc in resampled:
            j_xy = s_xy + rng.normal(0.0, post_resample_jitter, size=s_xy.shape)
            jittered.append((j_xy, s_t, s_arc))
        resampled = jittered

    total_points = max(sum(len(s) for s, _, _ in resampled), 1)
    log_max_dt = math.log1p(cfg.max_delta_ms)
    has_any_timing = cfg.use_timing and (not drop_timing) and any(t is not None for _, t, _ in resampled)

    per_stroke_features: list[np.ndarray] = []
    per_stroke_xy: list[np.ndarray] = []
    per_stroke_arc: list[float] = []

    global_pt_cursor = 0
    prev_stroke_end_xy: np.ndarray | None = None
    prev_time: float | None = None

    for s_idx, (s_xy, s_t, s_arc) in enumerate(resampled):
        n_pts = len(s_xy)
        s_min = s_xy.min(axis=0)
        s_max = s_xy.max(axis=0)
        s_mid = 0.5 * (s_min + s_max)
        s_scale = max(float((s_max - s_min).max()), 1e-4)

        pts_feat = np.zeros((n_pts, POINT_FEATURE_DIM), dtype=np.float32)
        dx = np.zeros(n_pts, dtype=np.float64)
        dy = np.zeros(n_pts, dtype=np.float64)
        if n_pts > 1:
            dx[1:] = s_xy[1:, 0] - s_xy[:-1, 0]
            dy[1:] = s_xy[1:, 1] - s_xy[:-1, 1]
        step = np.hypot(dx, dy)

        # Carry previous valid heading on degenerate zero-length segments
        sin_th = np.zeros(n_pts, dtype=np.float64)
        cos_th = np.ones(n_pts, dtype=np.float64)
        valid_seg = step > 1e-9
        theta = np.arctan2(dy, dx)
        last_sin, last_cos = 0.0, 1.0
        for i in range(1, n_pts):
            if valid_seg[i]:
                last_sin = math.sin(float(theta[i]))
                last_cos = math.cos(float(theta[i]))
            sin_th[i] = last_sin
            cos_th[i] = last_cos
        if n_pts > 1:
            sin_th[0], cos_th[0] = sin_th[1], cos_th[1]

        # Continuous turning angle via complex dot/cross (exact across +-pi, zero wrap discontinuity)
        sin_dth = np.zeros(n_pts, dtype=np.float64)
        cos_dth = np.ones(n_pts, dtype=np.float64)
        if n_pts > 2:
            # dtheta = theta[i] - theta[i-1]
            cos_dth[2:] = cos_th[2:] * cos_th[1:-1] + sin_th[2:] * sin_th[1:-1]
            sin_dth[2:] = sin_th[2:] * cos_th[1:-1] - cos_th[2:] * sin_th[1:-1]

        progress = np.linspace(0.0, 1.0, n_pts, dtype=np.float64) if n_pts > 1 else np.zeros(1, dtype=np.float64)
        stroke_idx_norm = math.log1p(s_idx) / 5.0
        order_norm = (global_pt_cursor + np.arange(n_pts, dtype=np.float64)) / max(total_points - 1, 1)
        global_pt_cursor += n_pts

        pts_feat[:, 0:2] = np.clip(s_xy, -cfg.coord_clip, cfg.coord_clip)
        pts_feat[:, 2:4] = (s_xy - s_mid) / s_scale
        pts_feat[:, 4] = np.clip(dx, -cfg.coord_clip, cfg.coord_clip)
        pts_feat[:, 5] = np.clip(dy, -cfg.coord_clip, cfg.coord_clip)
        pts_feat[:, 6] = sin_th
        pts_feat[:, 7] = cos_th
        pts_feat[:, 8] = sin_dth
        pts_feat[:, 9] = cos_dth
        pts_feat[:, 10] = np.log1p(step / max(effective_spacing, 1e-4))
        pts_feat[:, 11] = progress
        pts_feat[:, 12] = stroke_idx_norm
        pts_feat[:, 13] = order_norm
        pts_feat[0, 14] = 1.0
        pts_feat[-1, 15] = 1.0

        if has_any_timing and s_t is not None:
            pts_feat[:, 17] = 1.0
            dts = np.zeros(n_pts, dtype=np.float64)
            if prev_time is not None:
                dts[0] = max(0.0, float(s_t[0]) - prev_time)
            if n_pts > 1:
                dts[1:] = np.maximum(0.0, np.diff(s_t))
            pts_feat[:, 16] = np.log1p(np.minimum(dts, cfg.max_delta_ms)) / log_max_dt
            prev_time = float(s_t[-1])

        if prev_stroke_end_xy is not None:
            jump = np.clip(s_xy[0] - prev_stroke_end_xy, -cfg.coord_clip, cfg.coord_clip)
            pts_feat[0, 18] = float(jump[0])
            pts_feat[0, 19] = float(jump[1])
        prev_stroke_end_xy = s_xy[-1]

        per_stroke_features.append(pts_feat)
        per_stroke_xy.append(s_xy)
        per_stroke_arc.append(s_arc)

    return ResampledStrokeSet(
        per_stroke_features=per_stroke_features,
        per_stroke_xy=per_stroke_xy,
        per_stroke_arc=per_stroke_arc,
        glyph_scale=h,
        origin=origin,
        effective_spacing=effective_spacing,
        coarsened=coarsened,
        overflowed=overflowed,
    )


def chunk_ink_v2(
    strokes: Sequence[np.ndarray],
    times: Sequence[np.ndarray | None] | None = None,
    config: InkPreprocessV2Config | None = None,
    *,
    bucket_override: int | None = None,
    drop_timing: bool = False,
    max_total_points: int | None = None,
    post_resample_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ChunkedInk:
    """Convert raw strokes into static-bucketed overlapping 8-pt sub-stroke chunks with
    explicit per-point padding masks (zero endpoint over-weighting on tail chunks)."""
    cfg = config or InkPreprocessV2Config()
    res = resample_and_featurize_strokes(
        strokes,
        times,
        cfg,
        max_total_points=max_total_points,
        max_total_chunks=cfg.max_chunks,
        drop_timing=drop_timing,
        post_resample_jitter=post_resample_jitter,
        rng=rng,
    )
    if not res.per_stroke_features:
        b = int(bucket_override or cfg.buckets[0])
        return ChunkedInk(
            chunk_points=np.zeros((b, cfg.chunk_size, POINT_FEATURE_DIM), dtype=np.float32),
            chunk_point_mask=np.ones((b, cfg.chunk_size), dtype=bool),
            chunk_meta=np.zeros((b, CHUNK_META_DIM), dtype=np.float32),
            chunk_mask=np.ones((b,), dtype=bool),
            chunk_stroke_ids=np.full((b,), -1, dtype=np.int64),
            chunk_centers_raw=np.zeros((b, 2), dtype=np.float32),
            valid_chunks=0,
            stroke_count=0,
            glyph_scale=1.0,
            origin=np.zeros(2, dtype=np.float64),
            bucket=b,
            coarsened=False,
            overflowed=False,
        )

    chunk_pts_list: list[np.ndarray] = []
    chunk_pmask_list: list[np.ndarray] = []
    chunk_meta_list: list[np.ndarray] = []
    chunk_sid_list: list[int] = []
    chunk_center_raw_list: list[np.ndarray] = []
    overflowed = res.overflowed

    for s_idx, (pts_feat, s_xy, s_arc) in enumerate(
        zip(res.per_stroke_features, res.per_stroke_xy, res.per_stroke_arc)
    ):
        n_pts = len(pts_feat)
        s_ext = np.ptp(s_xy, axis=0)
        # Real handwritten dots have extent ~0.02-0.20*h and short arc (<0.30*h)
        is_dot_stroke = float(n_pts == 1 or (float(s_ext.max()) < 0.22 and s_arc < 0.30))
        stroke_idx_norm = math.log1p(s_idx) / 5.0
        progress = pts_feat[:, 11]

        if n_pts <= cfg.chunk_size:
            starts = [0]
        else:
            starts = list(range(0, n_pts - cfg.chunk_size, cfg.chunk_hop))
            next_start = starts[-1] + cfg.chunk_hop if starts else 0
            if next_start < n_pts:
                starts.append(next_start)

        for local_c_idx, a in enumerate(starts):
            if len(chunk_pts_list) >= cfg.max_chunks:
                overflowed = True
                break
            b = min(a + cfg.chunk_size, n_pts)
            valid_len = b - a
            sub_feat = np.zeros((cfg.chunk_size, POINT_FEATURE_DIM), dtype=np.float32)
            sub_pmask = np.ones((cfg.chunk_size,), dtype=bool)
            sub_feat[:valid_len] = pts_feat[a:b]
            sub_pmask[:valid_len] = False
            sub_xy = s_xy[a:b]

            c_center = sub_xy.mean(axis=0)
            c_ext = np.ptp(sub_xy, axis=0)
            c_arc = float(np.linalg.norm(np.diff(sub_xy, axis=0), axis=1).sum()) if len(sub_xy) > 1 else 0.0
            meta = np.array([
                float(np.clip(c_center[0], -cfg.coord_clip, cfg.coord_clip)),
                float(np.clip(c_center[1], -cfg.coord_clip, cfg.coord_clip)),
                math.log1p(float(c_ext[0])),
                math.log1p(float(c_ext[1])),
                math.log1p(c_arc),
                float(progress[a]),
                float(progress[b - 1]),
                stroke_idx_norm,
                math.log1p(local_c_idx) / 4.0,
                is_dot_stroke,
            ], dtype=np.float32)

            chunk_pts_list.append(sub_feat)
            chunk_pmask_list.append(sub_pmask)
            chunk_meta_list.append(meta)
            chunk_sid_list.append(s_idx)
            chunk_center_raw_list.append((c_center * res.glyph_scale + res.origin).astype(np.float32))

        if len(chunk_pts_list) >= cfg.max_chunks:
            break

    valid = len(chunk_pts_list)
    bucket = int(bucket_override) if bucket_override is not None else select_bucket(valid, cfg.buckets)
    if valid > bucket:
        overflowed = True
    valid_clamped = min(valid, bucket)

    chunk_points = np.zeros((bucket, cfg.chunk_size, POINT_FEATURE_DIM), dtype=np.float32)
    chunk_point_mask = np.ones((bucket, cfg.chunk_size), dtype=bool)
    chunk_meta = np.zeros((bucket, CHUNK_META_DIM), dtype=np.float32)
    chunk_mask = np.ones((bucket,), dtype=bool)
    chunk_stroke_ids = np.full((bucket,), -1, dtype=np.int64)
    chunk_centers_raw = np.zeros((bucket, 2), dtype=np.float32)

    if valid_clamped > 0:
        chunk_points[:valid_clamped] = np.stack(chunk_pts_list[:valid_clamped], axis=0)
        chunk_point_mask[:valid_clamped] = np.stack(chunk_pmask_list[:valid_clamped], axis=0)
        chunk_meta[:valid_clamped] = np.stack(chunk_meta_list[:valid_clamped], axis=0)
        chunk_mask[:valid_clamped] = False
        chunk_stroke_ids[:valid_clamped] = np.asarray(chunk_sid_list[:valid_clamped], dtype=np.int64)
        chunk_centers_raw[:valid_clamped] = np.stack(chunk_center_raw_list[:valid_clamped], axis=0)

    return ChunkedInk(
        chunk_points=chunk_points,
        chunk_point_mask=chunk_point_mask,
        chunk_meta=chunk_meta,
        chunk_mask=chunk_mask,
        chunk_stroke_ids=chunk_stroke_ids,
        chunk_centers_raw=chunk_centers_raw,
        valid_chunks=valid_clamped,
        stroke_count=len(res.per_stroke_features),
        glyph_scale=res.glyph_scale,
        origin=res.origin,
        bucket=bucket,
        coarsened=res.coarsened,
        overflowed=overflowed,
    )


def extract_style_strokes(
    strokes: Sequence[np.ndarray],
    max_strokes: int = STYLE_MAX_STROKES,
    points_per_stroke: int = STYLE_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_xy, _ = clean_raw_strokes(strokes)
    out_pts = np.zeros((max_strokes, points_per_stroke, STYLE_POINT_DIM), dtype=np.float32)
    out_asp = np.zeros((max_strokes, 4), dtype=np.float32)
    out_mask = np.ones((max_strokes,), dtype=bool)
    if not clean_xy:
        return out_pts, out_asp, out_mask

    h = estimate_glyph_scale(clean_xy)
    chosen = clean_xy[:max_strokes]
    for idx, s in enumerate(chosen):
        sn = s / h
        diffs = np.diff(sn, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(cum[-1])
        if total <= 1e-9 or len(sn) == 1:
            rxy = np.repeat(sn[:1], points_per_stroke, axis=0)
        else:
            targets = np.linspace(0.0, total, points_per_stroke, dtype=np.float64)
            rxy = np.stack([np.interp(targets, cum, sn[:, 0]), np.interp(targets, cum, sn[:, 1])], axis=1)

        mid = 0.5 * (rxy.min(axis=0) + rxy.max(axis=0))
        ext = np.ptp(rxy, axis=0)
        scale = max(float(ext.max()), 1e-4)
        loc = (rxy - mid) / scale

        dx = np.zeros(points_per_stroke, dtype=np.float64)
        dy = np.zeros(points_per_stroke, dtype=np.float64)
        dx[1:] = loc[1:, 0] - loc[:-1, 0]
        dy[1:] = loc[1:, 1] - loc[:-1, 1]
        step = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        sin_th = np.zeros(points_per_stroke, dtype=np.float64)
        cos_th = np.ones(points_per_stroke, dtype=np.float64)
        last_sin, last_cos = 0.0, 1.0
        for i in range(1, points_per_stroke):
            if step[i] > 1e-9:
                last_sin, last_cos = math.sin(float(theta[i])), math.cos(float(theta[i]))
            sin_th[i], cos_th[i] = last_sin, last_cos
        sin_th[0], cos_th[0] = sin_th[1], cos_th[1]
        cos_dth = np.ones(points_per_stroke, dtype=np.float64)
        sin_dth = np.zeros(points_per_stroke, dtype=np.float64)
        cos_dth[2:] = cos_th[2:] * cos_th[1:-1] + sin_th[2:] * sin_th[1:-1]
        sin_dth[2:] = sin_th[2:] * cos_th[1:-1] - cos_th[2:] * sin_th[1:-1]

        out_pts[idx, :, 0:2] = loc
        out_pts[idx, :, 2] = dx
        out_pts[idx, :, 3] = dy
        out_pts[idx, :, 4] = sin_th
        out_pts[idx, :, 5] = cos_th
        out_pts[idx, :, 6] = sin_dth
        out_pts[idx, :, 7] = cos_dth

        end_vec = loc[-1] - loc[0]
        dir_ang = math.atan2(float(end_vec[1]), float(end_vec[0]))
        out_asp[idx] = np.array([
            math.log1p(float(ext[0])),
            math.log1p(float(ext[1])),
            math.sin(dir_ang),
            math.cos(dir_ang),
        ], dtype=np.float32)
        out_mask[idx] = False

    return out_pts, out_asp, out_mask


def warp_raw_strokes(
    strokes: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    rotation_degrees: float = 4.5,
    shear: float = 0.08,
    anisotropic_scale: tuple[float, float] = (0.88, 1.12),
    stroke_shift: float = 0.02,
) -> list[np.ndarray]:
    """Single-transform affine + per-stroke shift applied ONCE to raw ink before BOTH
    v2 chunking AND v2 strip rasterization. Point jitter is intentionally applied AFTER
    resampling inside `resample_and_featurize_strokes` so arc length is never inflated."""
    clean_xy, _ = clean_raw_strokes(strokes)
    if not clean_xy:
        return []
    h = estimate_glyph_scale(clean_xy)
    origin = estimate_expression_origin(clean_xy)

    angle = math.radians(float(rng.uniform(-rotation_degrees, rotation_degrees)))
    sh = float(rng.uniform(-shear, shear))
    sx, sy = rng.uniform(*anisotropic_scale, size=2)
    c, s = math.cos(angle), math.sin(angle)
    mat = np.array([[c, -s], [s, c]]) @ np.array([[1.0, sh], [0.0, 1.0]]) @ np.diag([sx, sy])

    warped: list[np.ndarray] = []
    for st in clean_xy:
        pts = (st - origin) @ mat.T + origin
        if stroke_shift > 0:
            pts = pts + rng.normal(0.0, stroke_shift * h, size=(1, 2))
        warped.append(pts)
    return warped
