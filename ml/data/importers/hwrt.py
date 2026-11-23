from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing
from tokenizer.latex_tokenizer import normalize_latex


def _first(row: dict[str, Any], *names: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return str(value).strip()
    return ""


class HwrtImporter:
    source_dataset = "hwrt"

    def _read_csv(self, path: Path) -> Iterator[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=";", quotechar="'")

    def iter_load(self, root: str | Path) -> Iterator[StrokeExpression]:
        root = Path(root)
        symbol_file = next(iter(root.rglob("symbols.csv")), None)
        if symbol_file is None:
            raise FileNotFoundError(f"symbols.csv not found under {root}")
        labels = {
            _first(row, "symbol_id", "id"): normalize_latex(_first(row, "latex", "formula", "label").strip("$"))
            for row in self._read_csv(symbol_file)
        }
        found = False
        for split, filename in (("train", "train-data.csv"), ("test", "test-data.csv")):
            data_file = next(iter(root.rglob(filename)), None)
            if data_file is None:
                continue
            for row_index, row in enumerate(self._read_csv(data_file)):
                symbol_id = _first(row, "symbol_id", "symbol")
                label = labels.get(symbol_id, "")
                raw_data = _first(row, "data", "recording", "strokes")
                if not label or not raw_data:
                    continue
                try:
                    loaded = json.loads(raw_data)
                except json.JSONDecodeError:
                    loaded = yaml.safe_load(raw_data)
                if not isinstance(loaded, list):
                    continue
                strokes: list[Stroke] = []
                for stroke_index, raw_stroke in enumerate(loaded):
                    if not isinstance(raw_stroke, list):
                        continue
                    points = [
                        StrokePoint(
                            x=float(point["x"]), y=float(point["y"]),
                            time=float(point["time"]) if point.get("time") is not None else None,
                            pressure=None, pressure_available=False,
                        )
                        for point in raw_stroke if isinstance(point, dict) and "x" in point and "y" in point
                    ]
                    if points:
                        strokes.append(Stroke(id=stroke_index, points=points))
                if not strokes:
                    continue
                writer_id = _first(row, "user_id", "writer_id", "user") or None
                sample_id = _first(row, "id", "recording_id") or f"{split}-{row_index}"
                expression = StrokeExpression(
                    id=f"{self.source_dataset}:{split}:{sample_id}", original_id=sample_id,
                    latex_original=label, latex_normalized=label, strokes=strokes,
                    source="external", source_dataset=self.source_dataset,
                    synthetic=False, split=split, writer_id=writer_id,
                    metadata={
                        "source": "external", "sourceDataset": self.source_dataset,
                        "originalId": sample_id, "split": split, "symbolId": symbol_id,
                        "userAgent": _first(row, "user_agent"), "path": str(data_file),
                    },
                )
                found = True
                yield normalize_expression_timing(expression)
        if not found:
            raise ValueError(f"No HWRT stroke examples found under {root}")

    def load(self, root: str | Path) -> list[StrokeExpression]:
        return list(self.iter_load(root))
