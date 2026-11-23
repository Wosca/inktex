from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator

from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing
from tokenizer.latex_tokenizer import normalize_latex


def _clean_latex(value: str) -> str:
    result = value.strip()
    if len(result) >= 2 and result.startswith("$") and result.endswith("$"):
        result = result[1:-1].strip()
    if result.startswith(r"\[") and result.endswith(r"\]"):
        result = result[2:-2].strip()
    return normalize_latex(result)


class MathBrushImporter:
    """Read the MathBrush corpus's text-based SCG ink plus paired .tex."""

    source_dataset = "mathbrush"

    def parse(self, path: str | Path, root: str | Path | None = None) -> StrokeExpression:
        path = Path(path)
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
        if not lines or lines[0] != "SCG_INK":
            raise ValueError(f"Not an SCG_INK file: {path}")
        try:
            stroke_count = int(lines[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid SCG stroke count: {path}") from error

        cursor = 2
        strokes: list[Stroke] = []
        for stroke_index in range(stroke_count):
            try:
                point_count = int(lines[cursor])
            except (IndexError, ValueError) as error:
                raise ValueError(f"Invalid SCG point count in {path}") from error
            cursor += 1
            points: list[StrokePoint] = []
            for _ in range(point_count):
                try:
                    x_text, y_text, *_ = lines[cursor].split()
                    points.append(StrokePoint(x=float(x_text), y=float(y_text)))
                except (IndexError, ValueError) as error:
                    raise ValueError(f"Invalid SCG point in {path} at line {cursor + 1}") from error
                cursor += 1
            if points:
                strokes.append(Stroke(id=stroke_index, points=points))
        if not strokes:
            raise ValueError(f"MathBrush example contains no stroke points: {path}")

        latex_path = path.with_suffix(".tex")
        if not latex_path.is_file():
            raise FileNotFoundError(f"Paired MathBrush LaTeX file not found: {latex_path}")
        label = _clean_latex(latex_path.read_text(encoding="utf-8-sig"))
        if not label:
            raise ValueError(f"Paired MathBrush LaTeX is empty: {latex_path}")

        writer = next((part for part in reversed(path.parts) if re.fullmatch(r"user\d+", part, re.I)), None)
        base = Path(root) if root is not None else path.parent
        try:
            original_id = path.relative_to(base).with_suffix("").as_posix()
        except ValueError:
            original_id = path.stem
        expression = StrokeExpression(
            id=f"{self.source_dataset}:{original_id}", original_id=original_id,
            latex_original=label, latex_normalized=label, strokes=strokes,
            source="external", source_dataset=self.source_dataset,
            synthetic=False, split=None, writer_id=writer,
            metadata={
                "source": "external", "sourceDataset": self.source_dataset,
                "originalId": original_id, "writerId": writer,
                "time_available": False, "path": str(path),
            },
        )
        return normalize_expression_timing(expression)

    def iter_load(self, root: str | Path) -> Iterator[StrokeExpression]:
        root = Path(root)
        found = False
        for path in sorted(root.rglob("*.ink")):
            if path.with_suffix(".tex").is_file():
                found = True
                yield self.parse(path, root)
        if not found:
            raise ValueError(f"No paired MathBrush .ink/.tex examples found under {root}")

    def load(self, root: str | Path) -> list[StrokeExpression]:
        return list(self.iter_load(root))
