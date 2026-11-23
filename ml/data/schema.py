from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable


@dataclass(slots=True)
class StrokePoint:
    """One recorded online-ink point. Missing sensor values stay missing."""

    x: float
    y: float
    time: float | None = None
    relative_time: float | None = None
    delta_time: float | None = None
    pressure: float | None = None
    pressure_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Stroke:
    points: list[StrokePoint]
    id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrokeExpression:
    id: str
    latex_original: str
    latex_normalized: str
    strokes: list[Stroke]
    source: str
    source_dataset: str
    original_id: str | None = None
    synthetic: bool = False
    split: str | None = None
    writer_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latex(self) -> str:
        return self.latex_normalized or self.latex_original

    @property
    def point_count(self) -> int:
        return sum(len(stroke.points) for stroke in self.strokes)

    @property
    def stroke_count(self) -> int:
        return len(self.strokes)


def normalize_expression_timing(expression: StrokeExpression) -> StrokeExpression:
    """Re-zero available times without manufacturing values for untimed ink."""
    available = [point.time for stroke in expression.strokes for point in stroke.points if point.time is not None]
    if not available:
        for stroke in expression.strokes:
            for point in stroke.points:
                point.relative_time = None
                point.delta_time = None
        expression.metadata["time_available"] = False
        return expression

    origin = float(available[0])
    previous: float | None = None
    for stroke in expression.strokes:
        for point in stroke.points:
            if point.time is None:
                point.relative_time = None
                point.delta_time = None
                continue
            current = float(point.time)
            point.relative_time = current - origin
            point.delta_time = 0.0 if previous is None else max(0.0, current - previous)
            previous = current
    expression.metadata["time_available"] = True
    return expression


def clone_expression(expression: StrokeExpression, strokes: Iterable[Stroke] | None = None) -> StrokeExpression:
    """Make an augmentation-safe copy while keeping source metadata."""
    copied_strokes = list(strokes) if strokes is not None else [
        Stroke(
            id=stroke.id,
            metadata=dict(stroke.metadata),
            points=[replace(point, metadata=dict(point.metadata)) for point in stroke.points],
        )
        for stroke in expression.strokes
    ]
    return replace(expression, strokes=copied_strokes, metadata=dict(expression.metadata))
