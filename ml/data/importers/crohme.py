from __future__ import annotations

from pathlib import Path
import html
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from data.importers.mathwriting import _local_name, _parse_number
from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing
from tokenizer.latex_tokenizer import normalize_latex


def _clean_latex(value: str) -> str:
    result = value.strip()
    if len(result) >= 2 and result.startswith("$") and result.endswith("$"):
        result = result[1:-1].strip()
    if result.startswith(r"\[") and result.endswith(r"\]"):
        result = result[2:-2].strip()
    return normalize_latex(result)


def _infer_split(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if any(any(name in part for name in ("artificial", "synthetic")) for part in parts):
        return "synthetic"
    # Prefer the closest directory/filename marker. This avoids an unrelated
    # ancestor such as a temporary pytest directory being mistaken for a split.
    aliases = {
        "valid": "validation", "validation": "validation", "val": "validation",
        "test": "test", "testing": "test", "train": "train", "training": "train",
    }
    for part in reversed(parts):
        if part in aliases:
            return aliases[part]
    for part in reversed(parts):
        for token in re.split(r"[^a-z0-9]+", part):
            if token in aliases:
                return aliases[token]
    return None


class CrohmeImporter:
    source_dataset = "crohme"

    def parse(self, path: str | Path) -> StrokeExpression:
        path = Path(path)
        xml_repaired = False
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as original_error:
            # Some publisher-supplied artificial files place raw LaTeX '<' or
            # '&' characters inside annotation text. Repair only plain
            # annotation bodies, and only after strict XML parsing has failed.
            source = path.read_text(encoding="utf-8-sig")

            def repair_annotation(match: re.Match[str]) -> str:
                value = escape(html.unescape(match.group(2)))
                return f"{match.group(1)}{value}{match.group(3)}"

            repaired = re.sub(
                r"(<annotation\b[^>]*>)(.*?)(</annotation>)",
                repair_annotation,
                source,
                flags=re.DOTALL | re.IGNORECASE,
            )
            try:
                root = ET.fromstring(repaired)
                xml_repaired = True
            except ET.ParseError as repaired_error:
                raise ValueError(f"Malformed CROHME InkML: {path}: {original_error}") from repaired_error
        # CROHME has nested per-symbol truth annotations. Only direct children
        # describe the complete expression.
        annotations = {
            child.attrib.get("type", ""): (child.text or "").strip()
            for child in list(root) if _local_name(child.tag) == "annotation"
        }
        label = _clean_latex(
            annotations.get("truth") or annotations.get("label") or annotations.get("normalizedLabel") or ""
        )
        if not label or label.lower() == "ground truth":
            raise ValueError(f"CROHME InkML has no expression-level LaTeX truth: {path}")

        channels = [
            element.attrib.get("name", "").upper()
            for element in root.iter() if _local_name(element.tag) == "channel"
        ] or ["X", "Y"]
        try:
            x_index, y_index = channels.index("X"), channels.index("Y")
        except ValueError as error:
            raise ValueError(f"CROHME InkML does not declare X and Y channels: {path}") from error
        time_index = channels.index("T") if "T" in channels else None

        strokes: list[Stroke] = []
        for trace_index, trace in enumerate(element for element in root.iter() if _local_name(element.tag) == "trace"):
            points: list[StrokePoint] = []
            for point_text in re.split(r"\s*,\s*", (trace.text or "").strip()):
                if not point_text:
                    continue
                values = [_parse_number(value) for value in point_text.split()]
                required = max(x_index, y_index, time_index if time_index is not None else 0)
                if len(values) <= required:
                    raise ValueError(f"Malformed CROHME trace point in {path}: {point_text!r}")
                points.append(StrokePoint(
                    x=values[x_index], y=values[y_index],
                    time=values[time_index] if time_index is not None else None,
                    pressure=None, pressure_available=False,
                ))
            if points:
                strokes.append(Stroke(points=points, id=trace.attrib.get("id", trace_index)))
        if not strokes:
            raise ValueError(f"CROHME InkML contains no trace points: {path}")

        original_id = annotations.get("UI") or annotations.get("sampleId") or path.stem
        # Publisher UI values identify the source expression, not necessarily
        # this ink instance: tens of thousands of artificial variants reuse
        # them. The archive-relative path is the stable, unique sample key.
        lower_parts = [part.lower() for part in path.parts]
        try:
            inkml_index = lower_parts.index("inkml")
            sample_id = Path(*path.parts[inkml_index + 1:]).with_suffix("").as_posix()
        except ValueError:
            sample_id = path.with_suffix("").name
        split = _infer_split(path)
        expression = StrokeExpression(
            id=f"{self.source_dataset}:{sample_id}", original_id=original_id,
            latex_original=label, latex_normalized=label, strokes=strokes,
            source="external", source_dataset=self.source_dataset,
            synthetic=split == "synthetic", split=split,
            writer_id=annotations.get("writer") or None,
            metadata={
                "source": "external", "sourceDataset": self.source_dataset,
                "originalId": original_id, "split": split,
                "year": annotations.get("year"), "copyright": annotations.get("copyright"),
                "time_available": time_index is not None, "path": str(path),
                "xmlRepaired": xml_repaired,
            },
        )
        return normalize_expression_timing(expression)
