from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing


INKML_NAMESPACE = "http://www.w3.org/2003/InkML"
NS = {"ink": INKML_NAMESPACE}
DEFAULT_SPLITS = ("train", "valid", "test")


@dataclass(slots=True)
class MathWritingReference:
    """Lightweight accepted-sample record; InkML points are loaded on demand."""

    path: Path
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
        return MathWritingImporter().parse(self.path)

    def to_record(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "id": self.id, "latex_original": self.latex_original,
            "latex_normalized": self.latex_normalized, "source": self.source,
            "source_dataset": self.source_dataset, "original_id": self.original_id,
            "synthetic": self.synthetic, "split": self.split, "writer_id": self.writer_id,
            "metadata": self.metadata, "stroke_count": self.stroke_count,
            "point_count": self.point_count,
        }

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> "MathWritingReference":
        return cls(
            path=Path(value["path"]), id=str(value["id"]),
            latex_original=str(value.get("latex_original", "")),
            latex_normalized=str(value.get("latex_normalized", "")),
            source=str(value.get("source", "external")),
            source_dataset=str(value.get("source_dataset", "mathwriting")),
            original_id=value.get("original_id"), synthetic=bool(value.get("synthetic", False)),
            split=value.get("split"), writer_id=value.get("writer_id"),
            metadata=dict(value.get("metadata") or {}), stroke_count=int(value["stroke_count"]),
            point_count=int(value["point_count"]),
        )

    @classmethod
    def from_expression(cls, path: str | Path, expression: StrokeExpression) -> "MathWritingReference":
        return cls(
            path=Path(path), id=expression.id, latex_original=expression.latex_original,
            latex_normalized=expression.latex_normalized, source=expression.source,
            source_dataset=expression.source_dataset, original_id=expression.original_id,
            synthetic=expression.synthetic, split=expression.split, writer_id=expression.writer_id,
            metadata=dict(expression.metadata), stroke_count=expression.stroke_count,
            point_count=expression.point_count,
        )


def discover_mathwriting_files(
    root: str | Path,
    splits: tuple[str, ...] | list[str] = DEFAULT_SPLITS,
    include_synthetic: bool = False,
    include_symbols: bool = False,
) -> dict[str, list[Path]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    selected = list(splits)
    if include_synthetic and "synthetic" not in selected:
        selected.append("synthetic")
    if include_symbols and "symbols" not in selected:
        selected.append("symbols")
    discovered: dict[str, list[Path]] = {}
    for split in selected:
        directory = root / split
        discovered[split] = sorted(directory.glob("*.inkml")) if directory.is_dir() else []
    return discovered


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_number(value: str) -> float:
    # InkML permits a leading !, ', or " modifier. MathWriting ordinarily uses plain decimals.
    cleaned = value.strip().lstrip("!'\"")
    return float(cleaned)


class MathWritingImporter:
    source_dataset = "mathwriting"

    def parse(self, path: str | Path) -> StrokeExpression:
        path = Path(path)
        root = ET.parse(path).getroot()
        annotations = {
            element.attrib.get("type", ""): (element.text or "").strip()
            for element in root.iter()
            if _local_name(element.tag) == "annotation"
        }

        channels = [
            element.attrib.get("name", "").upper()
            for element in root.iter()
            if _local_name(element.tag) == "channel"
        ]
        if not channels:
            channels = ["X", "Y", "T"]
        try:
            x_index, y_index = channels.index("X"), channels.index("Y")
        except ValueError as error:
            raise ValueError(f"InkML {path} does not declare X and Y channels") from error
        time_index = channels.index("T") if "T" in channels else None

        strokes: list[Stroke] = []
        for trace_index, trace in enumerate(element for element in root.iter() if _local_name(element.tag) == "trace"):
            points: list[StrokePoint] = []
            trace_text = trace.text or ""
            for point_text in re.split(r"\s*,\s*", trace_text.strip()):
                if not point_text:
                    continue
                values = [_parse_number(value) for value in point_text.split()]
                required = max(x_index, y_index, time_index if time_index is not None else 0)
                if len(values) <= required:
                    raise ValueError(f"Malformed trace point in {path}: {point_text!r}")
                points.append(StrokePoint(
                    x=values[x_index],
                    y=values[y_index],
                    time=values[time_index] if time_index is not None else None,
                    pressure=None,
                    pressure_available=False,
                ))
            if points:
                strokes.append(Stroke(points=points, id=trace.attrib.get("id", trace_index)))

        if not strokes:
            raise ValueError(f"InkML {path} contains no trace points")
        sample_id = annotations.get("sampleId") or path.stem
        split = annotations.get("splitTagOriginal") or path.parent.name
        creation = annotations.get("inkCreationMethod", "")
        synthetic = split == "synthetic" or creation != "human"
        original = annotations.get("label", "")
        normalized = annotations.get("normalizedLabel") or original
        expression = StrokeExpression(
            id=f"mathwriting:{sample_id}",
            original_id=sample_id,
            latex_original=original,
            latex_normalized=normalized,
            strokes=strokes,
            source="external",
            source_dataset=self.source_dataset,
            synthetic=synthetic,
            split=split,
            metadata={
                "source": "external",
                "sourceDataset": self.source_dataset,
                "originalId": sample_id,
                "synthetic": synthetic,
                "split": split,
                "inkCreationMethod": creation or None,
                "labelCreationMethod": annotations.get("labelCreationMethod"),
                "time_available": time_index is not None,
                "path": str(path),
            },
        )
        return normalize_expression_timing(expression)
