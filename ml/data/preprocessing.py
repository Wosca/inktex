from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterable, Sequence

import numpy as np

from data.importers.native import NativeImporter
from data.schema import StrokeExpression, StrokePoint


FEATURE_NAMES = (
    "x", "y", "dx", "dy", "delta_time", "pressure",
    "pressure_available", "stroke_start", "stroke_end", "stroke_index_normalized",
)


@dataclass(slots=True)
class PreprocessingConfig:
    # Retained for old checkpoints/CLI. Common-representation points are source ink.
    point_source: str = "raw"
    max_points: int = 1024
    min_points_per_stroke: int = 2
    resampling_enabled: bool = True
    resample_spacing: float = 0.02
    min_resampled_points_per_stroke: int = 0
    use_timing: bool = True
    use_pressure: bool = True
    max_delta_ms: float = 2000.0
    normalization: str = "expression"
    max_points_per_stroke: int = 64
    point_overflow: str = "reduce"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreprocessingConfig":
        aliases = dict(value)
        if "target_spacing" in aliases:
            aliases.setdefault("resample_spacing", aliases.pop("target_spacing"))
        resampling = aliases.pop("resampling", None)
        if isinstance(resampling, dict):
            aliases.setdefault("resampling_enabled", resampling.get("enabled", True))
            aliases.setdefault("resample_spacing", resampling.get("target_spacing", 0.02))
            aliases.setdefault("max_points", resampling.get("max_points", 1024))
        fields = cls.__dataclass_fields__
        return cls(**{key: val for key, val in aliases.items() if key in fields})


@dataclass(slots=True)
class ProcessedExample:
    features: np.ndarray
    latex: str
    original_latex: str
    example_id: str
    stroke_count: int
    original_point_count: int
    processed_point_count: int
    downsampled: bool
    points_by_stroke: list[np.ndarray]
    source: str
    source_dataset: str
    metadata: dict[str, Any]


def ensure_stroke_expression(example: StrokeExpression | dict[str, Any] | Any) -> StrokeExpression:
    if isinstance(example, StrokeExpression):
        return example
    if isinstance(example, dict):
        return NativeImporter().from_dict(example)
    loader = getattr(example, "load", None)
    if callable(loader):
        loaded = loader()
        if isinstance(loaded, StrokeExpression):
            return loaded
    raise TypeError(f"Unsupported stroke example type: {type(example).__name__}")


def _interpolate(a: StrokePoint, b: StrokePoint, alpha: float) -> StrokePoint:
    def optional(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return float(left + alpha * (right - left))

    pressure_available = a.pressure_available and b.pressure_available
    return StrokePoint(
        x=float(a.x + alpha * (b.x - a.x)),
        y=float(a.y + alpha * (b.y - a.y)),
        time=optional(a.time, b.time),
        relative_time=optional(a.relative_time, b.relative_time),
        delta_time=None,
        pressure=optional(a.pressure, b.pressure) if pressure_available else None,
        pressure_available=pressure_available,
        metadata=dict(a.metadata),
    )


def resample_stroke_by_distance(
    points: Sequence[StrokePoint], spacing: float = 0.02, min_points: int = 0
) -> list[StrokePoint]:
    """Resample one normalized stroke by arc length, preserving endpoints and sensors."""
    if len(points) <= 1 or spacing <= 0:
        return [replace(point, metadata=dict(point.metadata)) for point in points]
    xy = np.asarray([(point.x, point.y) for point in points], dtype=np.float64)
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return [replace(points[0], metadata=dict(points[0].metadata))]
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


spatially_resample = resample_stroke_by_distance


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
    shares = [remaining * capacity / total_capacity for capacity in capacities]
    counts = [floor + min(capacity, int(share)) for floor, capacity, share in zip(floors, capacities, shares)]
    spare = budget - sum(counts)
    order = sorted(range(len(lengths)), key=lambda index: shares[index] - int(shares[index]), reverse=True)
    for index in order:
        if spare <= 0:
            break
        if counts[index] < lengths[index]:
            counts[index] += 1
            spare -= 1
    return counts


def _resample_to_count(points: list[StrokePoint], count: int) -> list[StrokePoint]:
    if count >= len(points):
        return points
    if count == 1:
        return [points[0]]
    chosen = np.rint(np.linspace(0, len(points) - 1, count)).astype(int)
    chosen[0], chosen[-1] = 0, len(points) - 1
    return [points[int(index)] for index in chosen]


def limit_strokes(
    strokes: list[list[StrokePoint]], max_points: int, min_points_per_stroke: int
) -> tuple[list[list[StrokePoint]], bool]:
    total = sum(map(len, strokes))
    if total <= max_points:
        return strokes, False
    counts = _allocate_counts([len(stroke) for stroke in strokes], max_points, min_points_per_stroke)
    return [_resample_to_count(stroke, count) for stroke, count in zip(strokes, counts)], True


def _normalize(expression: StrokeExpression, mode: str = "expression") -> list[list[StrokePoint]]:
    points = [point for stroke in expression.strokes for point in stroke.points]
    if not points:
        raise ValueError(f"Expression {expression.id} contains no stroke points")
    xy = np.asarray([(point.x, point.y) for point in points], dtype=np.float64)
    low, high = xy.min(axis=0), xy.max(axis=0)
    center = (low + high) / 2.0
    scale = max(float((high - low).max()), 1e-9)
    if mode == "glyph":
        # Stroke height estimates writing size without depending on line width.
        heights = [max(p.y for p in s.points) - min(p.y for p in s.points)
                   for s in expression.strokes if s.points]
        useful = [height for height in heights if height > 1e-9]
        widths = [max(p.x for p in s.points) - min(p.x for p in s.points)
                  for s in expression.strokes if s.points]
        fallback = [width for width in widths if width > 1e-9]
        scale = max(float(np.median(useful or fallback)) if (useful or fallback) else 1.0, 1e-9)
        center = xy[0]  # appending terms cannot move the existing ink's origin
    elif mode != "expression":
        raise ValueError(f"Unknown ink normalization: {mode}")
    return [[replace(
        point,
        x=float((point.x - center[0]) / scale),
        y=float((point.y - center[1]) / scale),
        metadata=dict(point.metadata),
    ) for point in stroke.points] for stroke in expression.strokes if stroke.points]


def preprocess_example(
    example: StrokeExpression | dict[str, Any], config: PreprocessingConfig | None = None
) -> ProcessedExample:
    config = config or PreprocessingConfig()
    expression = ensure_stroke_expression(example)
    original_point_count = expression.point_count
    strokes = _normalize(expression, config.normalization)
    if config.resampling_enabled and config.resample_spacing > 0:
        strokes = [
            resample_stroke_by_distance(stroke, config.resample_spacing, config.min_resampled_points_per_stroke)
            for stroke in strokes
        ]
    if config.normalization == "glyph":
        strokes = [_resample_to_count(stroke, config.max_points_per_stroke) for stroke in strokes]
    if config.point_overflow not in ("reduce", "error"):
        raise ValueError("point_overflow must be reduce or error")
    if config.point_overflow == "error" and sum(map(len, strokes)) > config.max_points:
        raise ValueError(f"Expression {expression.id} exceeds the {config.max_points}-point working budget; "
                         "split at layout boundaries instead of discarding character detail")
    strokes, downsampled = limit_strokes(strokes, config.max_points, config.min_points_per_stroke)

    rows: list[list[float]] = []
    normalized_strokes: list[np.ndarray] = []
    previous_xy: np.ndarray | None = None
    previous_time: float | None = None
    denominator = max(len(strokes) - 1, 1)
    for stroke_index, stroke in enumerate(strokes):
        stroke_xy: list[list[float]] = []
        for point_index, point in enumerate(stroke):
            xy = np.asarray((point.x, point.y), dtype=np.float32)
            is_start = point_index == 0
            is_end = point_index == len(stroke) - 1
            delta = np.zeros(2, dtype=np.float32) if is_start or previous_xy is None else xy - previous_xy
            if config.use_timing and point.time is not None and previous_time is not None:
                dt_ms = max(0.0, float(point.time) - previous_time)
                dt = math.log1p(min(dt_ms, config.max_delta_ms)) / math.log1p(config.max_delta_ms)
            else:
                dt = 0.0
            pressure_available = bool(config.use_pressure and point.pressure_available and point.pressure is not None)
            pressure = float(point.pressure) if pressure_available else 0.0
            rows.append([
                float(xy[0]), float(xy[1]), float(delta[0]), float(delta[1]), float(dt), pressure,
                float(pressure_available), float(is_start), float(is_end),
                math.log1p(stroke_index) / 8.0 if config.normalization == "glyph" else stroke_index / denominator,
            ])
            stroke_xy.append([float(xy[0]), float(xy[1])])
            previous_xy = xy
            if point.time is not None:
                previous_time = float(point.time)
        normalized_strokes.append(np.asarray(stroke_xy, dtype=np.float32))

    return ProcessedExample(
        features=np.asarray(rows, dtype=np.float32),
        latex=expression.latex,
        original_latex=expression.latex_original,
        example_id=expression.id,
        stroke_count=len(strokes),
        original_point_count=original_point_count,
        processed_point_count=len(rows),
        downsampled=downsampled,
        points_by_stroke=normalized_strokes,
        source=expression.source,
        source_dataset=expression.source_dataset,
        metadata=dict(expression.metadata),
    )


def iter_original_xy(example: StrokeExpression | dict[str, Any], source: str = "raw") -> Iterable[np.ndarray]:
    expression = ensure_stroke_expression(example)
    for stroke in expression.strokes:
        if stroke.points:
            yield np.asarray([(point.x, point.y) for point in stroke.points], dtype=np.float32)
