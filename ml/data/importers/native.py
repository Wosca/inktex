from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing
from tokenizer.latex_tokenizer import normalize_latex


def _raw(point: dict[str, Any]) -> dict[str, Any]:
    return point.get("raw", point)


class NativeImporter:
    """Adapter for unmodified collector JSON/JSONL payloads."""

    source_dataset = "ipad_collector"

    @staticmethod
    def is_example(value: Any) -> bool:
        return isinstance(value, dict) and isinstance(value.get("strokes"), list) and "latex" in value

    def from_dict(self, value: dict[str, Any]) -> StrokeExpression:
        if not self.is_example(value):
            raise ValueError("Native value is not a collector stroke example")
        strokes: list[Stroke] = []
        for stroke_index, raw_stroke in enumerate(value["strokes"]):
            points: list[StrokePoint] = []
            # Raw points are the source of truth. Stored resampled points are device-specific derived data.
            for wrapped in raw_stroke.get("rawPoints") or raw_stroke.get("resampledPoints") or []:
                raw = _raw(wrapped)
                pointer_type = str(raw.get("pointerType", "")).lower()
                pressure_available = pointer_type in {"pen", "stylus"} and raw.get("pressure") is not None
                timestamp = raw.get("timestamp", raw.get("relativeTime"))
                points.append(StrokePoint(
                    x=float(raw.get("x", 0.0)),
                    y=float(raw.get("y", 0.0)),
                    time=float(timestamp) if timestamp is not None else None,
                    pressure=float(raw["pressure"]) if pressure_available else None,
                    pressure_available=pressure_available,
                    metadata={"pointer_type": pointer_type or None},
                ))
            if points:
                strokes.append(Stroke(points=points, id=raw_stroke.get("id", stroke_index), metadata=dict(raw_stroke.get("metadata") or {})))

        if not strokes:
            raise ValueError(f"Native example {value.get('id', '<unknown>')} contains no stroke points")
        metadata = dict(value.get("metadata") or {})
        latex_original = str(value.get("latexOriginal", value.get("latex", "")))
        latex_normalized = str(value.get("latexNormalized") or normalize_latex(str(value.get("latex", ""))))
        writer = value.get("writerID", metadata.get("writerID", metadata.get("writerId")))
        expression = StrokeExpression(
            id=str(value.get("id", "unknown")),
            original_id=str(value.get("id", "unknown")),
            latex_original=latex_original,
            latex_normalized=latex_normalized,
            strokes=strokes,
            source="native",
            source_dataset=self.source_dataset,
            synthetic=False,
            writer_id=str(writer) if writer not in (None, "") else None,
            metadata=metadata | {
                "source": "native",
                "sourceDataset": self.source_dataset,
                "schemaVersion": value.get("schemaVersion"),
            },
        )
        return normalize_expression_timing(expression)

    def load_file(self, path: str | Path) -> list[StrokeExpression]:
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            values = loaded if isinstance(loaded, list) else [loaded]
        return [self.from_dict(value) for value in values if self.is_example(value)]

    def load(self, path: str | Path) -> list[StrokeExpression]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(root)
        files: Iterable[Path] = [root] if root.is_file() else sorted({*root.rglob("*.json"), *root.rglob("*.jsonl")})
        examples: list[StrokeExpression] = []
        seen: set[str] = set()
        for file in files:
            for expression in self.load_file(file):
                if expression.id not in seen:
                    examples.append(expression)
                    seen.add(expression.id)
        if not examples:
            raise ValueError(f"No native stroke examples found under {root}")
        return examples
