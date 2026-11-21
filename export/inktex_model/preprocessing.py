"""Standalone stroke preprocessing matching Stage 2 bimodal shard packing.

Produces 10-channel glyph-normalized features:
  [x, y, dx, dy, delta_time, pressure(=0), pressure_available(=0),
   stroke_start, stroke_end, log1p(stroke_index) / 8.0]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(slots=True)
class StrokePoint:
    x: float
    y: float
    time: float | None = None  # milliseconds


@dataclass(slots=True)
class PreprocessingConfig:
    max_points: int = 2048
    min_points_per_stroke: int = 2
    resampling_enabled: bool = True
    resample_spacing: float = 0.02
    min_resampled_points_per_stroke: int = 0
    use_timing: bool = True
    use_pressure: bool = False
    max_delta_ms: float = 2000.0
    normalization: str = "glyph"
    max_points_per_stroke: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interpolate(a: StrokePoint, b: StrokePoint, alpha: float) -> StrokePoint:
    t = None if (a.time is None or b.time is None) else float(a.time + alpha * (b.time - a.time))
    return StrokePoint(
        x=float(a.x + alpha * (b.x - a.x)),
        y=float(a.y + alpha * (b.y - a.y)),
        time=t,
    )


def resample_stroke_by_distance(
    points: Sequence[StrokePoint], spacing: float = 0.02, min_points: int = 0
) -> list[StrokePoint]:
    if len(points) <= 1 or spacing <= 0:
        return list(points)
    xy = np.asarray([(p.x, p.y) for p in points], dtype=np.float64)
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return [points[0]]
    targets = np.arange(0.0, total, spacing, dtype=np.float64)
    if not len(targets) or not math.isclose(float(targets[-1]), total):
        targets = np.append(targets, total)
    minimum = min(max(min_points, 2), len(points))
    if len(targets) < minimum:
        targets = np.linspace(0.0, total, minimum, dtype=np.float64)
    result: list[StrokePoint] = []
    for target in targets:
        right = min(int(np.searchsorted(cumulative, target, side="right")), len(points) - 1)
        left = max(0, right - 1)
        span = cumulative[right] - cumulative[left]
        alpha = 0.0 if span <= 1e-12 else float((target - cumulative[left]) / span)
        result.append(_interpolate(points[left], points[right], alpha))
    return result


def _resample_to_count(points: list[StrokePoint], count: int) -> list[StrokePoint]:
    if count >= len(points):
        return points
    if count == 1:
        return [points[0]]
    chosen = np.rint(np.linspace(0, len(points) - 1, count)).astype(int)
    chosen[0], chosen[-1] = 0, len(points) - 1
    return [points[int(i)] for i in chosen]


def _allocate_counts(lengths: list[int], budget: int, minimum: int) -> list[int]:
    if budget < len(lengths):
        raise ValueError(f"max_points={budget} cannot preserve {len(lengths)} non-empty strokes")
    floors = [min(length, max(1, minimum)) for length in lengths]
    if sum(floors) > budget:
        floors = [1] * len(lengths)
    remaining = budget - sum(floors)
    capacities = [length - floor for length, floor in zip(lengths, floors)]
    if remaining <= 0 or not any(capacities):
        return floors
    total_capacity = sum(capacities)
    shares = [remaining * c / total_capacity for c in capacities]
    counts = [f + min(c, int(s)) for f, c, s in zip(floors, capacities, shares)]
    spare = budget - sum(counts)
    order = sorted(range(len(lengths)), key=lambda i: shares[i] - int(shares[i]), reverse=True)
    for idx in order:
        if spare <= 0:
            break
        if counts[idx] < lengths[idx]:
            counts[idx] += 1
            spare -= 1
    return counts


def preprocess_strokes(
    strokes: Sequence[Sequence[StrokePoint]], config: PreprocessingConfig | None = None
) -> np.ndarray:
    """Convert raw strokes into float32 [N, 10] feature tensor."""
    cfg = config or PreprocessingConfig()
    non_empty = [list(s) for s in strokes if s]
    if not non_empty:
        raise ValueError("Cannot preprocess empty strokes")

    all_xy = np.asarray([(p.x, p.y) for s in non_empty for p in s], dtype=np.float64)
    if cfg.normalization == "glyph":
        heights = [max(p.y for p in s) - min(p.y for p in s) for s in non_empty]
        useful = [h for h in heights if h > 1e-9]
        widths = [max(p.x for p in s) - min(p.x for p in s) for s in non_empty]
        fallback = [w for w in widths if w > 1e-9]
        scale = max(float(np.median(useful or fallback)) if (useful or fallback) else 1.0, 1e-9)
        center = all_xy[0]
    else:
        low, high = all_xy.min(axis=0), all_xy.max(axis=0)
        center = (low + high) / 2.0
        scale = max(float((high - low).max()), 1e-9)

    normed = [
        [StrokePoint((p.x - center[0]) / scale, (p.y - center[1]) / scale, p.time) for p in s]
        for s in non_empty
    ]
    if cfg.resampling_enabled and cfg.resample_spacing > 0:
        normed = [
            resample_stroke_by_distance(s, cfg.resample_spacing, cfg.min_resampled_points_per_stroke)
            for s in normed
        ]
    if cfg.normalization == "glyph":
        normed = [_resample_to_count(s, cfg.max_points_per_stroke) for s in normed]
    total_pts = sum(len(s) for s in normed)
    if total_pts > cfg.max_points:
        counts = _allocate_counts([len(s) for s in normed], cfg.max_points, cfg.min_points_per_stroke)
        normed = [_resample_to_count(s, c) for s, c in zip(normed, counts)]

    rows: list[list[float]] = []
    previous_xy: np.ndarray | None = None
    previous_time: float | None = None
    denom = max(len(normed) - 1, 1)
    log_max_dt = math.log1p(cfg.max_delta_ms)

    for stroke_idx, stroke in enumerate(normed):
        for pt_idx, pt in enumerate(stroke):
            xy = np.asarray((pt.x, pt.y), dtype=np.float32)
            is_start = pt_idx == 0
            is_end = pt_idx == len(stroke) - 1
            delta = np.zeros(2, dtype=np.float32) if is_start or previous_xy is None else xy - previous_xy
            if cfg.use_timing and pt.time is not None and previous_time is not None:
                dt_ms = max(0.0, float(pt.time) - previous_time)
                dt = math.log1p(min(dt_ms, cfg.max_delta_ms)) / log_max_dt
            else:
                dt = 0.0
            stroke_pos = math.log1p(stroke_idx) / 8.0 if cfg.normalization == "glyph" else stroke_idx / denom
            rows.append([
                float(xy[0]), float(xy[1]), float(delta[0]), float(delta[1]),
                float(dt), 0.0, 0.0, float(is_start), float(is_end), float(stroke_pos),
            ])
            previous_xy = xy
            if pt.time is not None:
                previous_time = float(pt.time)

    return np.asarray(rows, dtype=np.float32)
