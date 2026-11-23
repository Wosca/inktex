from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing


SCHEMA_VERSION = "stroke-latex-v1"


@dataclass(slots=True)
class CanonicalReference:
    """Lightweight random-access reference to one canonical JSONL record."""

    path: Path
    offset: int
    id: str
    latex_original: str
    latex_normalized: str
    source: str
    source_dataset: str
    original_id: str | None
    synthetic: bool
    split: str | None
    writer_id: str | None
    metadata: dict[str, Any]
    stroke_count: int
    point_count: int

    @property
    def latex(self) -> str:
        return self.latex_normalized or self.latex_original

    def load(self) -> StrokeExpression:
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            line = handle.readline()
        if not line:
            raise ValueError(f"Canonical JSONL record no longer exists at {self.path}:{self.offset}")
        return expression_from_record(json.loads(line.decode("utf-8")))

    @classmethod
    def from_record(cls, path: Path, offset: int, value: dict[str, Any]) -> "CanonicalReference":
        if value.get("schemaVersion") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported canonical stroke schema at {path}:{offset}: {value.get('schemaVersion')!r}"
            )
        strokes = value.get("strokes") or []
        return cls(
            path=path, offset=offset, id=str(value["id"]),
            latex_original=str(value.get("latexOriginal") or value.get("latex") or ""),
            latex_normalized=str(value.get("latexNormalized") or value.get("latex") or ""),
            source=str(value.get("source", "external")),
            source_dataset=str(value.get("sourceDataset", "external")),
            original_id=value.get("originalId"), synthetic=bool(value.get("synthetic", False)),
            split=value.get("split"),
            writer_id=str(value["writerId"]) if value.get("writerId") not in (None, "") else None,
            metadata={**(value.get('metadata') or {}), "canonicalPath": str(path), "canonicalOffset": offset},
            stroke_count=len(strokes),
            point_count=sum(len(stroke.get("points") or []) for stroke in strokes),
        )


def expression_to_record(expression: StrokeExpression) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": expression.id,
        "originalId": expression.original_id,
        "source": expression.source,
        "sourceDataset": expression.source_dataset,
        "split": expression.split,
        "synthetic": expression.synthetic,
        "writerId": expression.writer_id,
        "latex": expression.latex,
        "latexOriginal": expression.latex_original,
        "latexNormalized": expression.latex_normalized,
        "strokes": [
            {
                "id": stroke.id,
                "points": [
                    {
                        "x": point.x,
                        "y": point.y,
                        **({"time": point.time} if point.time is not None else {}),
                        **({"pressure": point.pressure} if point.pressure is not None else {}),
                    }
                    for point in stroke.points
                ],
            }
            for stroke in expression.strokes
        ],
        "metadata": expression.metadata,
    }


def expression_from_record(value: dict[str, Any]) -> StrokeExpression:
    if value.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported canonical stroke schema: {value.get('schemaVersion')!r}")
    strokes = [
        Stroke(
            id=raw_stroke.get("id", stroke_index),
            points=[
                StrokePoint(
                    x=float(point["x"]), y=float(point["y"]),
                    time=float(point["time"]) if point.get("time") is not None else None,
                    pressure=float(point["pressure"]) if point.get("pressure") is not None else None,
                    pressure_available=point.get("pressure") is not None,
                )
                for point in raw_stroke.get("points", [])
            ],
        )
        for stroke_index, raw_stroke in enumerate(value.get("strokes", []))
        if raw_stroke.get("points")
    ]
    if not strokes:
        raise ValueError(f"Canonical example {value.get('id', '<unknown>')} contains no stroke points")
    expression = StrokeExpression(
        id=str(value["id"]), original_id=value.get("originalId"),
        latex_original=str(value.get("latexOriginal") or value.get("latex") or ""),
        latex_normalized=str(value.get("latexNormalized") or value.get("latex") or ""),
        strokes=strokes, source=str(value.get("source", "external")),
        source_dataset=str(value.get("sourceDataset", "external")),
        synthetic=bool(value.get("synthetic", False)), split=value.get("split"),
        writer_id=str(value["writerId"]) if value.get("writerId") not in (None, "") else None,
        metadata=dict(value.get("metadata") or {}),
    )
    return normalize_expression_timing(expression)


class CanonicalImporter:
    def iter_references(self, path: str | Path) -> Iterator[CanonicalReference]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(root)
        files: Iterable[Path] = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
        found = False
        seen: set[str] = set()
        for file in files:
            with file.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    value = json.loads(line.decode("utf-8"))
                    reference = CanonicalReference.from_record(file, offset, value)
                    if reference.id not in seen:
                        found = True
                        yield reference
                        seen.add(reference.id)
        if not found:
            raise ValueError(f"No canonical stroke examples found under {root}")

    def iter_file(self, path: str | Path) -> Iterator[StrokeExpression]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield expression_from_record(json.loads(line))

    def load_file(self, path: str | Path) -> list[StrokeExpression]:
        return list(self.iter_file(path))

    def iter_load(self, path: str | Path) -> Iterator[StrokeExpression]:
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(root)
        files: Iterable[Path] = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
        found = False
        seen: set[str] = set()
        for file in files:
            for expression in self.iter_file(file):
                if expression.id not in seen:
                    found = True
                    yield expression
                    seen.add(expression.id)
        if not found:
            raise ValueError(f"No canonical stroke examples found under {root}")

    def load(self, path: str | Path) -> list[StrokeExpression]:
        return list(self.iter_load(path))
