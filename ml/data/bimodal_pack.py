"""Audited bimodal shard packer and external dataset downloaders.

    python -m data.bimodal_pack --smoke --output <dir>
    python -m data.bimodal_pack --stages stage1 stage2 val --workers 8 --output <dir>
    python -m data.bimodal_pack --stages stage2 --output <dir> --fixed-tokenizer <stage1 shards>/tokenizer.json
    python -m data.bimodal_pack --download-detexify --download-unimer --unimer-limit 300000

Every training candidate is checked against a test-set leakage index before it
is written. Shards are planned in fixed candidate chunks, so an interrupted run
resumes by skipping chunks whose shard and sidecar already match the plan.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Sequence
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from data.bimodal_raster import (
    compute_dhash64, encode_png, normalize_offline_image_to_strip, render_strokes_to_strip,
)
from data.bimodal_shards import STAGE_DIRS, SYNTHETIC_SOURCES, ShardWriter, encode_npz, sha256_file
from data.canonical import expression_from_record
from data.importers.mathwriting import MathWritingImporter
from data.preprocessing import FEATURE_NAMES, PreprocessingConfig, preprocess_example
from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing
from tokenizer.latex_tokenizer import LatexTokenizer, tokenize_latex
from tokenizer.normalization import LABEL_VERSION, normalize_training_latex

FORMAT_VERSION = "bimodal-shards-v1"
DATASETS_ROOT = Path(r"E:\math-recogniser-datasets")
DEFAULT_CROHME = DATASETS_ROOT / "canonical" / "crohme23"
DEFAULT_HME = DATASETS_ROOT / "offline" / "hme100k"
DEFAULT_MATHWRITING = Path(r"E:\mathwriting-2024")
DEFAULT_STRUCTURE = Path(r"E:\math-recogniser-cache\structure\crohme-all-train-v2.json")
DEFAULT_EXTERNAL = DATASETS_ROOT / "external"
DEFAULT_OUTPUT = DATASETS_ROOT / "bimodal-shards"

PREPROCESSING = PreprocessingConfig(
    use_pressure=False, use_timing=True, normalization="glyph", max_points=2048, max_points_per_stroke=64,
)
RASTER = {"height": 128, "min_width": 64, "max_width": 640, "online_padding": 8, "offline_padding": 6}
SMOKE_TARGETS = {"stage1": 64, "stage2": 64, "val": 32}
VAL_QUOTA = 256

LEAKAGE_REASONS = frozenset({
    "test_id", "reserved_writer", "ink_hash", "latex_ink_hash", "latex_dhash", "dhash_exact", "dhash_near",
    "leaked_donor",
})
CONFUSABLE_TOKENS = frozenset({"y", "z", "C", "c", r"\gamma", "r", ".", ",", "X", "x", "v", "u", "1", "2", "8", "f"})
RARE_CONFUSABLES = CONFUSABLE_TOKENS - {"x", "1", "2"}
HARD_CONSTRUCTS = frozenset({r"\frac", r"\sqrt", "^", "_", r"\sum", r"\int", r"\lim", r"\prod"})

DETEXIFY_FILES = {
    "detexify.sql.gz": ("0ByuYordD0JBRV01NM2pmNlpfNUE", "0-CZHt-PBM7v0hty25FF5wsg"),
    "symbols.json": ("0ByuYordD0JBRU1Y3Q3VSNk9kdE0", "0-V2m8tmPfD8eyNe4GGrhSxw"),
    "README.txt": ("0ByuYordD0JBRb3hFUUlSRkEyNmM", "0-HEZkhosOQw_GQWHkHILphA"),
}
DETEXIFY_FOLDER = "https://drive.google.com/drive/folders/0ByuYordD0JBRWXUxRlhEMGtfVVE"
UNIMER_REPO = "wanderkid/UniMER_Dataset"
UNIMER_ARCHIVE = "UniMER-1M.zip"


class DownloadError(RuntimeError):
    pass


# --------------------------------------------------------------------------- labels and ink


def normalize_label(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    try:
        value = normalize_training_latex(raw)
    except (ValueError, RecursionError, IndexError):
        return None
    return value or None


def expression_strokes(expression: StrokeExpression) -> list[np.ndarray]:
    return [np.asarray([(point.x, point.y) for point in stroke.points], dtype=np.float64)
            for stroke in expression.strokes if stroke.points]


def feature_strokes(features: np.ndarray) -> list[np.ndarray]:
    starts = np.flatnonzero(features[:, 7] > 0.5)
    if not len(starts) or starts[0] != 0:
        starts = np.concatenate(([0], starts))
    bounds = [*starts.tolist(), len(features)]
    return [features[a:b, :2].astype(np.float64) for a, b in zip(bounds, bounds[1:]) if b > a]


def ink_geometry_hash(strokes: Sequence[np.ndarray], grid: int = 32, samples: int = 8) -> str:
    """Translation/scale invariant fingerprint of the stroke layout."""
    clean = [np.asarray(stroke, dtype=np.float64)[:, :2] for stroke in strokes if len(stroke)]
    if not clean:
        return ""
    xy = np.concatenate(clean)
    low = xy.min(axis=0)
    scale = max(float((xy.max(axis=0) - low).max()), 1e-9)
    digest = hashlib.sha1(str(len(clean)).encode())
    for stroke in clean:
        chosen = np.unique(np.rint(np.linspace(0, len(stroke) - 1, min(len(stroke), samples))).astype(int))
        digest.update(np.rint((stroke[chosen] - low) / scale * grid).astype(np.int16).tobytes())
        digest.update(b"|")
    return digest.hexdigest()[:20]


def _render_online(strokes: Sequence[np.ndarray], stroke_width: float | None = None):
    return render_strokes_to_strip(strokes, RASTER["height"], RASTER["min_width"], RASTER["max_width"],
                                   RASTER["online_padding"], stroke_width)


def audit_dhash(strokes: Sequence[np.ndarray]) -> str:
    """dHash of the canonical (default pen width) render; training renders jitter the pen."""
    return compute_dhash64(_render_online(strokes)[0])


def _has_timing(expression: StrokeExpression) -> bool:
    return bool(expression.metadata.get("time_available", True)) and any(
        point.time is not None for stroke in expression.strokes for point in stroke.points)


def _stroke_width(sample_id: str, jitter: bool) -> float | None:
    if not jitter:
        return None
    seed = int.from_bytes(hashlib.sha1(sample_id.encode()).digest()[:4], "little")
    return 2.0 + 2.0 * (seed / 0xFFFFFFFF)


# --------------------------------------------------------------------------- leakage index


def _low_information(value: int) -> bool:
    return not 8 <= value.bit_count() <= 56


class LeakageIndex:
    """Held-out test fingerprints: ids, ink geometry, normalized LaTeX and dHash."""

    def __init__(self, reserved_writers: Iterable[str] = (), near_distance: int = 2):
        self.test_ids: set[str] = set()
        self.ink: dict[str, tuple[str, str]] = {}
        self.latex_dhash: dict[str, list[tuple[int, str]]] = {}
        self.dhash_owner: dict[int, tuple[str, str]] = {}
        self.buckets: dict[tuple[int, int], list[int]] = {}
        self.reserved_writers = {str(writer) for writer in reserved_writers}
        self.near_distance = near_distance
        self.counts: Counter = Counter()
        self.complete = True

    def __len__(self) -> int:
        return len(self.test_ids)

    def add(self, *, test_id: str, source: str, normalized_latex: str | None, dhash: str,
            ink_hash: str | None = None) -> None:
        self.test_ids.add(test_id)
        self.counts[source] += 1
        latex = normalized_latex or ""
        value = int(dhash, 16)
        if ink_hash:
            self.ink.setdefault(ink_hash, (test_id, latex))
        if latex:
            self.latex_dhash.setdefault(latex, []).append((value, test_id))
        if value not in self.dhash_owner:
            self.dhash_owner[value] = (test_id, latex)
            for chunk in range(4):
                self.buckets.setdefault((chunk, (value >> (16 * chunk)) & 0xFFFF), []).append(value)

    def nearest(self, value: int) -> tuple[int, int] | None:
        # Hamming <= 2 over four 16-bit chunks implies at least two chunks match exactly.
        best = None
        for chunk in range(4):
            for other in self.buckets.get((chunk, (value >> (16 * chunk)) & 0xFFFF), ()):
                distance = (value ^ other).bit_count()
                if distance <= self.near_distance and (best is None or distance < best[0]):
                    best = (distance, other)
        return best

    def check(self, *, sample_id: str, normalized_latex: str, dhash: str | None, writer_id: str | None = None,
              ink_hash: str | None = None) -> tuple[str, str] | None:
        if sample_id in self.test_ids:
            return "test_id", sample_id
        if writer_id is not None and str(writer_id) in self.reserved_writers:
            return "reserved_writer", str(writer_id)
        value = int(dhash, 16) if dhash else None
        if ink_hash is not None:
            if ink_hash in self.ink:
                test_id, latex = self.ink[ink_hash]
                return ("latex_ink_hash" if latex == normalized_latex else "ink_hash"), test_id
            if value is not None:
                for other, test_id in self.latex_dhash.get(normalized_latex, ()):
                    if (value ^ other).bit_count() <= self.near_distance:
                        return "latex_dhash", test_id
            return None
        if value is None:
            return None
        match = self.nearest(value)
        if match is None:
            return None
        distance, other = match
        test_id, latex = self.dhash_owner[other]
        if _low_information(value) and latex != normalized_latex:
            return None
        return ("dhash_exact" if distance == 0 else "dhash_near"), test_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "near_distance": self.near_distance, "complete": self.complete, "counts": dict(self.counts),
            "reserved_writers": sorted(self.reserved_writers),
            "entries": [[test_id, latex, f"{value:016x}"] for value, (test_id, latex) in self.dhash_owner.items()],
            "latex_dhash": [[latex, test_id, f"{value:016x}"] for latex, values in self.latex_dhash.items()
                            for value, test_id in values],
            "ink": [[key, test_id, latex] for key, (test_id, latex) in self.ink.items()],
            "test_ids": sorted(self.test_ids),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LeakageIndex":
        index = cls(value.get("reserved_writers", ()), int(value.get("near_distance", 2)))
        index.complete = bool(value.get("complete", True))
        index.counts = Counter(value.get("counts", {}))
        index.test_ids = set(value.get("test_ids", ()))
        index.ink = {key: (test_id, latex) for key, test_id, latex in value.get("ink", ())}
        for latex, test_id, dhash in value.get("latex_dhash", ()):
            index.latex_dhash.setdefault(latex, []).append((int(dhash, 16), test_id))
        for test_id, latex, dhash in value.get("entries", ()):
            number = int(dhash, 16)
            if number not in index.dhash_owner:
                index.dhash_owner[number] = (test_id, latex)
                for chunk in range(4):
                    index.buckets.setdefault((chunk, (number >> (16 * chunk)) & 0xFFFF), []).append(number)
        return index


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True, slots=True)
class Candidate:
    source: str
    kind: str  # canonical | inkml | image | detexify | ast
    path: str = ""
    offset: int = -1
    id: str = ""
    latex: str = ""
    split: str = "train"
    writer_id: str | None = None


_CANONICAL_CACHE: dict[tuple[str, str, str], list[Candidate]] = {}


def canonical_candidates(path: Path, source: str, split: str) -> list[Candidate]:
    key = (str(path), source, split)
    if key not in _CANONICAL_CACHE:
        _CANONICAL_CACHE[key] = _scan_canonical(path, source, split)
    return _CANONICAL_CACHE[key]


def _scan_canonical(path: Path, source: str, split: str) -> list[Candidate]:
    result = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            writer = record.get("writerId")
            result.append(Candidate(
                source, "canonical", str(path), offset, str(record["id"]),
                str(record.get("latexNormalized") or record.get("latex") or ""), split,
                str(writer) if writer not in (None, "") else None,
            ))
    return result


def inkml_candidates(directory: Path, source: str, split: str) -> list[Candidate]:
    if not directory.is_dir():
        return []
    with os.scandir(directory) as entries:
        names = sorted(entry.name for entry in entries if entry.name.endswith(".inkml"))
    return [Candidate(source, "inkml", str(directory / name), -1, f"mathwriting:{name[:-6]}", "", split)
            for name in names]


def jsonl_image_candidates(root: Path, manifest: str, source: str, splits: set[str] | None) -> list[Candidate]:
    path = root / manifest
    if not path.is_file():
        return []
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split = str(record.get("split", "train"))
            if splits is not None and split not in splits:
                continue
            writer = record.get("writer")
            result.append(Candidate(source, "image", str(root / record["image"]), -1, str(record["id"]),
                                    str(record.get("latex", "")), split,
                                    str(writer) if writer not in (None, "") else None))
    return result


def detexify_candidates(path: Path) -> list[Candidate]:
    result = []
    if not path.is_file():
        return result
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                record = json.loads(line)
                result.append(Candidate("detexify", "detexify", str(path), offset, record["id"],
                                        record["latex"], "train"))
    return result


def _read_line(path: str, offset: int) -> dict[str, Any]:
    with open(path, "rb") as handle:
        handle.seek(offset)
        return json.loads(handle.readline())


def detexify_expression(record: dict[str, Any]) -> StrokeExpression:
    strokes = []
    for index, raw in enumerate(record["strokes"]):
        points = [StrokePoint(x=float(p[0]), y=float(p[1]), time=float(p[2]) if len(p) > 2 else None)
                  for p in raw if len(p) >= 2]
        if points:
            strokes.append(Stroke(points=points, id=index))
    if not strokes:
        raise ValueError(f"{record['id']} has no ink")
    expression = StrokeExpression(
        id=record["id"], latex_original=record["latex"], latex_normalized=record["latex"], strokes=strokes,
        source="external", source_dataset="detexify", split="train",
        metadata={"time_available": all(point.time is not None for s in strokes for point in s.points)},
    )
    return normalize_expression_timing(expression)


def load_expression(candidate: Candidate) -> StrokeExpression:
    if candidate.kind == "canonical":
        return expression_from_record(_read_line(candidate.path, candidate.offset))
    if candidate.kind == "inkml":
        return MathWritingImporter().parse(candidate.path)
    if candidate.kind == "detexify":
        return detexify_expression(_read_line(candidate.path, candidate.offset))
    raise ValueError(f"{candidate.kind} candidates carry no ink")


# --------------------------------------------------------------------------- sample building


@dataclass(slots=True)
class Drop:
    reason: str
    detail: str = ""


@dataclass(slots=True)
class Packed:
    meta: dict[str, Any]
    png: bytes
    npz: bytes | None


@dataclass
class PackContext:
    index: LeakageIndex
    max_tokens: int = 256
    unimer_max_tokens: int = 96
    unimer_min_ink_height: int = 44
    structure: dict[str, dict[str, Any]] = field(default_factory=dict)
    donors: list[Candidate] = field(default_factory=list)
    ast_seed: int = 1337
    ast_count: int = 0


_CONTEXT: PackContext | None = None
_COMPOSER: Any = None


def _init_worker(context: PackContext) -> None:
    global _CONTEXT, _COMPOSER
    _CONTEXT, _COMPOSER = context, None
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _context() -> PackContext:
    if _CONTEXT is None:
        raise RuntimeError("Packing context is not initialised")
    return _CONTEXT


def _label_or_drop(raw: str, max_tokens: int) -> str | Drop:
    label = normalize_label(raw)
    if label is None:
        return Drop("label_invalid", raw[:120])
    if len(tokenize_latex(label)) + 2 > max_tokens:
        return Drop("too_long")
    return label


def _structure_arrays(record: dict[str, Any] | None, stroke_count: int) -> dict[str, np.ndarray] | None:
    if not record or len(record["stroke_groups"]) != stroke_count:
        return None
    groups = np.asarray(record["stroke_groups"], dtype=np.int16)
    classes = np.asarray(record["symbol_classes"], dtype=np.int16)
    arrays = {"stroke_group_ids": groups, "stroke_symbol_ids": classes[groups]}
    if record.get("relations_available", True):
        arrays["relation_edges"] = np.asarray(record.get("relations") or [], dtype=np.int16).reshape(-1, 3)
    return arrays


def _meta(candidate: Candidate, latex: str, label: str, *, has_strokes: bool, has_timing: bool, image_width: int,
          stroke_count: int, writer_id: str | None, dhash: str | None, rendered: bool,
          sample_id: str | None = None, synthetic: bool | None = None) -> dict[str, Any]:
    meta = {
        "id": sample_id or candidate.id, "source": candidate.source, "split": candidate.split,
        "latex": latex, "normalized_latex": label, "has_strokes": has_strokes,
        "has_timing": bool(has_strokes and has_timing), "image_width": int(image_width),
        "stroke_count": int(stroke_count), "writer_id": writer_id,
        "synthetic": candidate.source in SYNTHETIC_SOURCES if synthetic is None else synthetic,
        "rendered": rendered,
    }
    if dhash:
        meta["dhash"] = dhash
    return meta


def build_online(candidate: Candidate, expression: StrokeExpression, *, with_strokes: bool,
                 jitter_width: bool) -> Packed | Drop:
    context = _context()
    if candidate.source == "mathwriting_train" and expression.synthetic:
        return Drop("not_human")
    raw = expression.latex or candidate.latex
    label = _label_or_drop(raw, context.max_tokens)
    if isinstance(label, Drop):
        return label
    strokes = expression_strokes(expression)
    if not strokes:
        return Drop("empty_ink")
    image, width = _render_online(strokes, _stroke_width(candidate.id, jitter_width))
    dhash = audit_dhash(strokes) if label in context.index.latex_dhash else None
    writer = expression.writer_id or candidate.writer_id
    leak = context.index.check(sample_id=candidate.id, normalized_latex=label, dhash=dhash, writer_id=writer,
                               ink_hash=ink_geometry_hash(strokes))
    if leak:
        return Drop(*leak)
    timed = _has_timing(expression)
    npz = None
    has_structure = False
    stroke_count = len(strokes)
    if with_strokes:
        processed = preprocess_example(expression, PREPROCESSING)
        arrays: dict[str, np.ndarray] = {"features": processed.features.astype(np.float16)}
        if not np.isfinite(arrays["features"]).all():
            return Drop("feature_overflow")
        if not timed:
            arrays["features"][:, 4] = 0
        structure = _structure_arrays(context.structure.get(candidate.id), processed.stroke_count)
        if structure:
            arrays.update(structure)
            has_structure = True
        npz = encode_npz(**arrays)
        stroke_count = processed.stroke_count
    meta = _meta(candidate, raw, label, has_strokes=with_strokes, has_timing=timed, image_width=width,
                 stroke_count=stroke_count, writer_id=writer, dhash=dhash, rendered=True)
    if with_strokes:
        meta["has_structure"] = has_structure
    return Packed(meta, encode_png(image), npz)


def build_offline(candidate: Candidate) -> Packed | Drop:
    context = _context()
    max_tokens = min(context.max_tokens, context.unimer_max_tokens) if candidate.source == "unimer" else context.max_tokens
    label = _label_or_drop(candidate.latex, max_tokens)
    if isinstance(label, Drop):
        return label
    image, width = normalize_offline_image_to_strip(Path(candidate.path), RASTER["height"], RASTER["min_width"],
                                                    RASTER["max_width"], RASTER["offline_padding"])
    bbox = image.getchannel("L").getbbox()
    if bbox is None:
        return Drop("blank_image")
    if candidate.source == "unimer" and (bbox[3] - bbox[1]) < context.unimer_min_ink_height:
        return Drop("squished_unimer")
    dhash = compute_dhash64(image)
    leak = context.index.check(sample_id=candidate.id, normalized_latex=label, dhash=dhash,
                               writer_id=candidate.writer_id)
    if leak:
        return Drop(*leak)
    meta = _meta(candidate, candidate.latex, label, has_strokes=False, has_timing=False, image_width=width,
                 stroke_count=0, writer_id=candidate.writer_id, dhash=dhash, rendered=False)
    return Packed(meta, encode_png(image), None)


class DonorBase:
    """Minimal ``ComposedDataset`` base over audited training-only donors."""

    def __init__(self, donors: list[Candidate], max_tokens: int):
        self.donors = donors
        self.examples = [SimpleNamespace(id=d.id, latex=d.latex, metadata={}) for d in donors]
        self.preprocessing = PREPROCESSING
        self.tokenizer = LatexTokenizer()
        self.max_tokens = max_tokens
        self.cache: dict[int, dict[str, Any]] = {}
        self.timed: dict[str, bool] = {}
        self.leaked: dict[str, tuple[str, str]] = {}

    def __len__(self) -> int:
        return len(self.donors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        if index in self.cache:
            return self.cache[index]
        if len(self.cache) > 20000:
            self.cache.clear()
        donor = self.donors[index]
        expression = load_expression(donor)
        strokes = expression_strokes(expression)
        leakage_index = _context().index
        leak = leakage_index.check(sample_id=donor.id, normalized_latex=donor.latex,
                                   dhash=audit_dhash(strokes) if donor.latex in leakage_index.latex_dhash else None,
                                   writer_id=expression.writer_id, ink_hash=ink_geometry_hash(strokes))
        if leak:
            self.leaked[donor.id] = leak
        self.timed[donor.id] = _has_timing(expression)
        processed = preprocess_example(expression, PREPROCESSING)
        item = {"features": torch.from_numpy(processed.features), "latex": donor.latex, "id": donor.id,
                "tokens": torch.tensor(self.tokenizer.encode(donor.latex)), "ctc_targets": None}
        self.cache[index] = item
        return item


def _composer() -> Any:
    global _COMPOSER
    if _COMPOSER is None:
        from data.composition import ComposedDataset
        context = _context()
        base = DonorBase(context.donors, context.max_tokens)
        composer = ComposedDataset(base, context.ast_count, context.ast_seed, styles=("sum", "product"))
        weights = []
        for index in composer.terms:
            tokens = set(tokenize_latex(base.donors[int(index)].latex))
            weights.append(1 + 2 * bool(tokens & RARE_CONFUSABLES) + bool(tokens & HARD_CONSTRUCTS))
        composer.terms = np.repeat(composer.terms, weights)
        _COMPOSER = composer
    return _COMPOSER


def build_ast(candidate: Candidate, *, with_strokes: bool) -> Packed | Drop:
    composer = _composer()
    base: DonorBase = composer.base
    try:
        item = composer[candidate.offset]
    except ValueError as error:
        return Drop("compose_error", str(error)[:160])
    leaked = [donor for donor in item["donor_ids"] if donor in base.leaked]
    if leaked:
        return Drop("leaked_donor", f"{leaked[0]}->{base.leaked[leaked[0]][1]}")
    label = _label_or_drop(item["latex"], _context().max_tokens)
    if isinstance(label, Drop):
        return label
    features = item["features"].numpy().astype(np.float32)
    timed = all(base.timed.get(donor, False) for donor in item["donor_ids"])
    strokes = feature_strokes(features)
    image, width = _render_online(strokes, _stroke_width(candidate.id, True))
    dhash = audit_dhash(strokes) if label in _context().index.latex_dhash else None
    leak = _context().index.check(sample_id=candidate.id, normalized_latex=label, dhash=dhash,
                                  ink_hash=ink_geometry_hash(strokes))
    if leak:
        return Drop(*leak)
    npz = None
    if with_strokes:
        stored = features.astype(np.float16)
        if not np.isfinite(stored).all():
            return Drop("feature_overflow")
        if not timed:
            stored[:, 4] = 0
        npz = encode_npz(features=stored)
    meta = _meta(candidate, item["latex"], label, has_strokes=with_strokes, has_timing=timed, image_width=width,
                 stroke_count=len(strokes), writer_id=None, dhash=dhash, rendered=True, synthetic=True)
    meta["donor_ids"] = list(item["donor_ids"])
    return Packed(meta, encode_png(image), npz)


def build_sample(candidate: Candidate, mode: str, upsample_hint: float | None = None) -> Packed | Drop:
    with_strokes = mode in ("bimodal", "eval")
    jitter = mode != "eval"
    if candidate.kind == "image":
        result = build_offline(candidate)
    elif candidate.kind == "ast":
        result = build_ast(candidate, with_strokes=with_strokes)
    else:
        result = build_online(candidate, load_expression(candidate), with_strokes=with_strokes, jitter_width=jitter)
    if isinstance(result, Packed) and upsample_hint and upsample_hint != 1:
        result.meta["upsample_hint"] = float(upsample_hint)
    return result


def index_entry(candidate: Candidate) -> dict[str, Any] | None:
    try:
        expression = load_expression(candidate)
    except Exception:
        return None
    strokes = expression_strokes(expression)
    if not strokes:
        return None
    return {"test_id": candidate.id, "source": candidate.source,
            "normalized_latex": normalize_label(expression.latex) or expression.latex,
            "dhash": audit_dhash(strokes), "ink_hash": ink_geometry_hash(strokes),
            "writer_id": expression.writer_id}


# --------------------------------------------------------------------------- chunk packing


@dataclass(slots=True)
class ChunkTask:
    stage: str
    directory: str
    index: int
    mode: str
    candidates: list[Candidate]
    plan_hash: str
    max_samples: int | None = None
    caps: dict[str, int] | None = None
    upsample: dict[str, float] = field(default_factory=dict)


def _shard_name(index: int) -> str:
    return f"shard-{index:05d}"


def pack_chunk(task: ChunkTask) -> dict[str, Any]:
    directory = Path(task.directory)
    shard_path = directory / f"{_shard_name(task.index)}.tar"
    sidecar = directory / f"{_shard_name(task.index)}.json"
    writer = ShardWriter(shard_path)
    candidates, accepted = Counter(), Counter()
    dropped: dict[str, Counter] = defaultdict(Counter)
    leaks, labels, failures = [], [], []
    structured = 0
    try:
        for candidate in task.candidates:
            if task.max_samples is not None and writer.samples >= task.max_samples:
                break
            if task.caps is not None and accepted[candidate.source] >= task.caps.get(candidate.source, 0):
                continue
            candidates[candidate.source] += 1
            try:
                result = build_sample(candidate, task.mode, task.upsample.get(candidate.source))
            except Exception as error:
                result = Drop("load_error", f"{type(error).__name__}: {error}"[:200])
            if isinstance(result, Drop):
                dropped[candidate.source][result.reason] += 1
                if result.reason in LEAKAGE_REASONS:
                    leaks.append({"id": candidate.id, "source": candidate.source, "reason": result.reason,
                                  "match": result.detail})
                elif result.reason == "load_error" and len(failures) < 20:
                    failures.append({"id": candidate.id, "error": result.detail})
                continue
            writer.write(f"s{task.index:05d}_{writer.samples:06d}", result.meta, result.png, result.npz)
            accepted[candidate.source] += 1
            labels.append(result.meta["normalized_latex"])
            structured += bool(result.meta.get("has_structure"))
    except BaseException:
        writer.abort()
        raise
    if writer.samples:
        shard = writer.close()
    else:
        writer.abort()
        shard_path.unlink(missing_ok=True)
        shard = {"file": None, "samples": 0, "bytes": 0, "sha256": None}
    summary = {
        **shard, "stage": task.stage, "index": task.index, "plan_hash": task.plan_hash,
        "sources": dict(accepted), "candidates": dict(candidates),
        "dropped": {source: dict(reasons) for source, reasons in dropped.items()},
        "leaks": leaks, "failures": failures, "with_structure": structured, "labels": labels,
    }
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, sidecar)
    return summary


def _sidecar_valid(directory: Path, index: int, plan_hash: str) -> dict[str, Any] | None:
    sidecar = directory / f"{_shard_name(index)}.json"
    if not sidecar.is_file():
        return None
    try:
        summary = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if summary.get("plan_hash") != plan_hash:
        return None
    if summary.get("file"):
        shard = directory / summary["file"]
        if not shard.is_file() or shard.stat().st_size != summary.get("bytes"):
            return None
    return summary


# --------------------------------------------------------------------------- downloads


def _drive_url(file_id: str, key: str) -> str:
    return (f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
            f"&resourcekey={key}&confirm=t")


def download_resumable(url: str, destination: Path, *, timeout: float = 60.0) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    start = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": "math-recogniser-bimodal-pack/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code == 416 and start:
            os.replace(part, destination)
            return destination
        raise DownloadError(f"HTTP {error.code} downloading {destination.name} from {url}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise DownloadError(
            f"Could not reach {url.split('/')[2]} to download {destination.name} ({error}). "
            f"Check the network connection and rerun; partial data in {part} resumes automatically.") from error
    with response:
        if "text/html" in (response.headers.get("Content-Type") or ""):
            raise DownloadError(
                f"Server returned an HTML page instead of {destination.name} (quota or sharing change). "
                f"Download it manually into {destination.parent}.")
        append = start > 0 and response.status == 206
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        total_bytes = int(total) if total.isdigit() else int(response.headers.get("Content-Length") or 0) + (
            start if append else 0)
        written = start if append else 0
        next_report = time.monotonic()
        try:
            with part.open("ab" if append else "wb") as handle:
                while block := response.read(1 << 20):
                    handle.write(block)
                    written += len(block)
                    if time.monotonic() >= next_report:
                        share = f" ({100 * written / total_bytes:.1f}%)" if total_bytes else ""
                        print(f"  {destination.name}: {written / 1e6:.1f} MB{share}", flush=True)
                        next_report = time.monotonic() + 10
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise DownloadError(f"Download of {destination.name} interrupted ({error}); rerun to resume.") from error
    if total_bytes and written < total_bytes:
        raise DownloadError(f"Download of {destination.name} incomplete ({written}/{total_bytes}); rerun to resume.")
    os.replace(part, destination)
    return destination


def _copy_unescape(value: str) -> str:
    if "\\" not in value:
        return value
    escapes = {"\\\\": "\\", "\\t": "\t", "\\n": "\n", "\\r": "\r"}
    output, index = [], 0
    while index < len(value):
        pair = value[index:index + 2]
        if pair in escapes:
            output.append(escapes[pair])
            index += 2
        else:
            output.append(value[index])
            index += 1
    return "".join(output)


def parse_detexify(raw: Path, output: Path, limit: int | None = None) -> dict[str, Any]:
    symbols = {entry["id"]: entry for entry in json.loads((raw / "symbols.json").read_text(encoding="utf-8"))}
    skipped: Counter = Counter()
    per_symbol: Counter = Counter()
    accepted = 0
    temporary = output.with_name(output.name + ".tmp")
    with gzip.open(raw / "detexify.sql.gz", "rt", encoding="utf-8") as handle, \
            temporary.open("w", encoding="utf-8") as sink:
        copying = False
        for line in handle:
            if not copying:
                copying = line.startswith("COPY samples")
                if copying and "(id, key, strokes)" not in line:
                    raise ValueError(f"Unexpected Detexify table layout: {line.strip()}")
                continue
            if line.startswith("\\."):
                break
            if limit is not None and accepted >= limit:
                break
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3 or parts[2] == "\\N":
                skipped["malformed"] += 1
                continue
            row_id, key, strokes_text = parts
            symbol = symbols.get(key)
            if symbol is None or not symbol.get("mathmode"):
                skipped["not_math_symbol"] += 1
                continue
            latex = normalize_label(symbol["command"])
            if latex is None:
                skipped["label_invalid"] += 1
                continue
            try:
                strokes = json.loads(_copy_unescape(strokes_text))
            except json.JSONDecodeError:
                skipped["malformed"] += 1
                continue
            strokes = [[point[:3] for point in stroke if isinstance(point, list) and len(point) >= 2]
                       for stroke in strokes if isinstance(stroke, list)]
            strokes = [stroke for stroke in strokes if stroke]
            if not strokes:
                skipped["empty_ink"] += 1
                continue
            sink.write(json.dumps({"id": f"detexify:{row_id}", "key": key, "command": symbol["command"],
                                   "latex": latex, "strokes": strokes}, separators=(",", ":")) + "\n")
            accepted += 1
            per_symbol[latex] += 1
    os.replace(temporary, output)
    return {"accepted": accepted, "symbols": len(per_symbol), "skipped": dict(skipped),
            "complete": limit is None or accepted < limit, "top_symbols": per_symbol.most_common(30)}


def download_detexify(root: Path, *, limit: int | None = None) -> dict[str, Any]:
    raw = root / "raw"
    for name, (file_id, key) in DETEXIFY_FILES.items():
        print(f"Detexify: {name}", flush=True)
        download_resumable(_drive_url(file_id, key), raw / name)
    manifest_path = root / "manifest.json"
    samples = root / "samples.jsonl"
    if samples.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") and manifest.get("limit") == limit:
            print(f"Detexify already parsed: {manifest['accepted']} samples")
            return manifest
    print("Detexify: parsing PostgreSQL dump", flush=True)
    summary = parse_detexify(raw, samples, limit)
    manifest = {
        **summary, "limit": limit, "source": DETEXIFY_FOLDER, "repository": "https://github.com/kirel/detexify-data",
        "license": "ODbL-1.0", "format": "samples.jsonl: {id, key, command, latex, strokes:[[[x,y,t],...],...]}",
        "sql_sha256": sha256_file(raw / "detexify.sql.gz"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detexify: {summary['accepted']} samples over {summary['symbols']} math symbols")
    return manifest


def unimer_label_reason(label: str, max_tokens: int) -> tuple[str | None, str | None]:
    if not label.strip():
        return None, "empty_label"
    if "\\\\" in label or "\\newline" in label:
        return None, "multi_line"
    if re.search(r"\\operatorname\s*\*", label):
        return None, "unsupported_command"
    normalized = normalize_label(label)
    if normalized is None:
        return None, "label_invalid"
    if len(tokenize_latex(normalized)) + 2 > max_tokens:
        return None, "too_long"
    return normalized, None


def unimer_image_reason(width: int, height: int) -> str | None:
    if width < 8 or height < 8:
        return "too_small"
    if height > 2.5 * width or height > 400:
        return "too_tall"
    if width > 12 * height:
        return "too_wide"
    return None


def download_unimer(root: Path, *, limit: int = 300000, max_tokens: int = 256, seed: int = 1337,
                    stream: bool = False, token: str | bool = False) -> dict[str, Any]:
    try:
        from huggingface_hub import HfFileSystem, hf_hub_download
    except ImportError as error:
        raise DownloadError("UniMER download needs huggingface_hub (pip install huggingface_hub)") from error
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    examples_path = root / "examples.jsonl"
    existing = set()
    if examples_path.is_file():
        with examples_path.open("r", encoding="utf-8") as handle:
            existing = {json.loads(line)["id"] for line in handle if line.strip()}
    try:
        if stream:
            if limit > 5000:
                print("UniMER: --unimer-stream issues one HTTP range request per image; "
                      "use the full download for large limits", flush=True)
            handle = HfFileSystem(token=token).open(f"datasets/{UNIMER_REPO}/{UNIMER_ARCHIVE}", "rb",
                                                    block_size=1 << 20)
        else:
            local_zip = root / "raw" / UNIMER_ARCHIVE
            if local_zip.is_file() and zipfile.is_zipfile(local_zip):
                print(f"UniMER: reusing local archive {local_zip} ({local_zip.stat().st_size / 1e9:.2f} GB)", flush=True)
                handle = open(local_zip, "rb")
            else:
                print(f"UniMER: downloading {UNIMER_ARCHIVE} (~2 GB, resumable)", flush=True)
                handle = open(hf_hub_download(UNIMER_REPO, UNIMER_ARCHIVE, repo_type="dataset",
                                              local_dir=root / "raw", token=token), "rb")
    except Exception as error:
        raise DownloadError(f"Could not fetch {UNIMER_REPO}/{UNIMER_ARCHIVE} ({type(error).__name__}: {error}). "
                            "Check the network (and --hf-token for gated mirrors); rerun to resume.") from error
    rejected: Counter = Counter()
    accepted = len(existing)
    with handle, zipfile.ZipFile(handle) as archive:
        labels = archive.read("UniMER-1M/train.txt").decode("utf-8").splitlines()
        present = {name for name in archive.namelist() if name.startswith("UniMER-1M/images/")}
        order = [index for index in range(len(labels)) if f"UniMER-1M/images/{index:07d}.png" in present]
        rejected["missing_image"] = len(labels) - len(order)
        random.Random(seed).shuffle(order)
        with examples_path.open("a", encoding="utf-8") as sink:
            for position, index in enumerate(order):
                if accepted >= limit:
                    break
                sample_id = f"unimer:{index:07d}"
                if sample_id in existing:
                    continue
                normalized, reason = unimer_label_reason(labels[index], max_tokens)
                if reason:
                    rejected[reason] += 1
                    continue
                data = archive.read(f"UniMER-1M/images/{index:07d}.png")
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        width, height = image.size
                except Exception:
                    rejected["bad_image"] += 1
                    continue
                reason = unimer_image_reason(width, height)
                if reason:
                    rejected[reason] += 1
                    continue
                relative = f"images/{index:07d}.png"
                (root / relative).write_bytes(data)
                sink.write(json.dumps({"id": sample_id, "image": relative, "latex": labels[index],
                                       "normalized_latex": normalized, "width": width, "height": height,
                                       "split": "train", "source": "unimer"}, ensure_ascii=False) + "\n")
                accepted += 1
                if accepted % 5000 == 0:
                    sink.flush()
                    print(f"  UniMER: {accepted}/{limit} accepted ({position + 1} scanned)", flush=True)
    manifest = {
        "repository": UNIMER_REPO, "archive": UNIMER_ARCHIVE, "license": "apache-2.0", "limit": limit,
        "accepted": accepted, "rejected_this_run": dict(rejected), "seed": seed, "max_tokens": max_tokens,
        "filters": {"multi_line": "labels with \\\\ or \\newline", "too_tall": "h > 2.5 w or h > 400 px",
                    "too_wide": "w > 24 h", "too_long": f"> {max_tokens} tokens"},
        "note": "UniMER-1M omits the HME100K portion (copyright); every image is re-audited against test sets "
                "by dHash at pack time.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"UniMER: {accepted} curated examples in {root}")
    return manifest


# --------------------------------------------------------------------------- planning


def _deterministic_subset(candidates: list[Candidate], fraction: float, seed: int) -> list[Candidate]:
    if fraction >= 1:
        return candidates
    threshold = int(max(0.0, fraction) * 0xFFFFFFFF)
    return [c for c in candidates
            if int.from_bytes(hashlib.sha1(f"{seed}:{c.id}".encode()).digest()[:4], "little") < threshold]


def _limit(candidates: list[Candidate], limit: int | None, seed: int) -> list[Candidate]:
    if limit is None or len(candidates) <= limit:
        return candidates
    chosen = list(candidates)
    random.Random(seed).shuffle(chosen)
    return chosen[:limit]


def _reserved_writers(value: str | None) -> set[str]:
    if not value:
        return set()
    path = Path(value)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return {str(item) for item in json.loads(text)}
        return {line.strip() for line in text.splitlines() if line.strip()}
    return {item.strip() for item in value.split(",") if item.strip()}


def test_candidates(args: argparse.Namespace) -> list[Candidate]:
    result = []
    crohme_test = Path(args.crohme_root) / "test.jsonl"
    if crohme_test.is_file():
        result += _limit(canonical_candidates(crohme_test, "crohme_test", "test"), args.leakage_limit, args.seed)
    result += _limit(inkml_candidates(Path(args.mathwriting_root) / "test", "mathwriting_test", "test"),
                     args.leakage_limit, args.seed)
    return result


def _index_key(args: argparse.Namespace, tests: list[Candidate], writers: set[str]) -> str:
    digest = hashlib.sha1(FORMAT_VERSION.encode())
    for candidate in tests:
        digest.update(f"{candidate.id}|{candidate.path}|{candidate.offset}\n".encode())
    digest.update(json.dumps(sorted(writers)).encode())
    digest.update(str(args.leakage_limit).encode())
    return digest.hexdigest()


def build_leakage_index(args: argparse.Namespace, output: Path) -> tuple[LeakageIndex, str]:
    writers = _reserved_writers(args.reserved_writers)
    tests = test_candidates(args)
    if not tests:
        raise SystemExit("No held-out test sets found; refusing to pack without a leakage audit")
    key = _index_key(args, tests, writers)
    cache = output / "leakage_index.json"
    if cache.is_file() and not args.rebuild_leakage:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("key") == key:
            index = LeakageIndex.from_dict(payload["index"])
            if args.block_test_writers:
                index.reserved_writers |= set(payload.get("test_writers", ()))
            print(f"Leakage index: reused {len(index)} test samples from {cache}")
            return index, key
    started = time.monotonic()
    print(f"Leakage index: fingerprinting {len(tests)} held-out test samples", flush=True)
    entries = _map(index_entry, tests, args.workers, chunksize=64)
    index = LeakageIndex(writers)
    index.complete = args.leakage_limit is None
    test_writers = set()
    for candidate, entry in zip(tests, entries):
        if entry is None:
            index.test_ids.add(candidate.id)
            index.counts[f"{candidate.source}_unreadable"] += 1
            continue
        index.add(test_id=entry["test_id"], source=entry["source"], normalized_latex=entry["normalized_latex"],
                  dhash=entry["dhash"], ink_hash=entry["ink_hash"])
        if entry["writer_id"]:
            test_writers.add(entry["writer_id"])
    for candidate in tests:
        if candidate.source == "crohme_test":
            subset = candidate.id.split("/")[1] if "/" in candidate.id else "unknown"
            index.counts[f"crohme_test:{subset}"] += 1
    output.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "index": index.to_dict(), "test_writers": sorted(test_writers)}),
                     encoding="utf-8")
    if args.block_test_writers:
        index.reserved_writers |= test_writers
    print(f"Leakage index: {len(index)} test samples in {time.monotonic() - started:.1f}s "
          f"({dict(index.counts)})", flush=True)
    return index, key


def _map(function, items: list[Any], workers: int, chunksize: int = 16) -> list[Any]:
    if workers <= 1 or len(items) < 2 * chunksize:
        return [function(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items, chunksize=chunksize))


def stage_sources(stage: str, args: argparse.Namespace) -> tuple[dict[str, list[Candidate]], str, dict[str, float]]:
    crohme, mathwriting, hme = Path(args.crohme_root), Path(args.mathwriting_root), Path(args.hme_root)
    external = Path(args.external_root)
    sources: dict[str, list[Candidate]] = {}
    if stage == "val":
        sources["crohme_val"] = canonical_candidates(crohme / "validation.jsonl", "crohme_val", "validation") \
            if (crohme / "validation.jsonl").is_file() else []
        sources["mathwriting_val"] = inkml_candidates(mathwriting / "valid", "mathwriting_val", "validation")
        return sources, "eval", {}
    crohme_train = canonical_candidates(crohme / "train.jsonl", "crohme_train", "train") \
        if (crohme / "train.jsonl").is_file() else []
    sources["crohme_train"] = crohme_train
    sources["mathwriting_train"] = inkml_candidates(mathwriting / "train", "mathwriting_train", "train")
    hme_train = jsonl_image_candidates(hme, "examples.jsonl", "hme100k", {"train"})
    if stage == "stage1":
        sources["hme100k"] = hme_train
        if args.with_unimer:
            sources["unimer"] = _limit(jsonl_image_candidates(external / "unimer", "examples.jsonl", "unimer", None),
                                       args.unimer_limit, args.seed)
        return sources, "image", {"crohme_train": float(args.crohme_upsample)}
    sources["mathwriting_synthetic"] = _limit(
        inkml_candidates(mathwriting / "synthetic", "mathwriting_synthetic", "synthetic"),
        args.mw_synthetic_limit, args.seed)
    if args.crohme_artificial and (crohme / "synthetic.jsonl").is_file():
        sources["crohme_artificial"] = _limit(
            canonical_candidates(crohme / "synthetic.jsonl", "crohme_artificial", "synthetic"),
            args.crohme_artificial_limit, args.seed)
    sources["hme100k"] = _deterministic_subset(hme_train, args.stage2_hme_fraction, args.seed)
    if args.ast_count:
        sources["ast_synthetic"] = [Candidate("ast_synthetic", "ast", "", index, f"ast:{args.seed}:{index}", "",
                                              "synthetic") for index in range(args.ast_count)]
    return sources, "bimodal", {}


def ast_donors(args: argparse.Namespace) -> list[Candidate]:
    from data.composition import composable_math_tokens
    crohme_train = Path(args.crohme_root) / "train.jsonl"
    pool: list[Candidate] = []
    if crohme_train.is_file():
        pool += canonical_candidates(crohme_train, "crohme_train", "train")
    symbols = inkml_candidates(Path(args.mathwriting_root) / "symbols", "mathwriting_symbols", "train")
    for candidate, label in zip(symbols, _map(_symbol_label, symbols, args.workers, chunksize=128)):
        if label is not None:
            pool.append(Candidate(candidate.source, "inkml", candidate.path, -1, candidate.id, label, "train"))
    vocabulary = {token for candidate in pool for token in tokenize_latex(normalize_label(candidate.latex) or "")}
    if args.with_detexify:
        per_label: Counter = Counter()
        for candidate in detexify_candidates(Path(args.external_root) / "detexify" / "samples.jsonl"):
            tokens = tokenize_latex(candidate.latex)
            if set(tokens) <= vocabulary and per_label[candidate.latex] < args.detexify_per_symbol:
                per_label[candidate.latex] += 1
                pool.append(candidate)
    donors = []
    for candidate in pool:
        label = normalize_label(candidate.latex)
        if label is None or composable_math_tokens(label) is None:
            continue
        donors.append(Candidate(candidate.source, candidate.kind, candidate.path, candidate.offset, candidate.id,
                                label, "train", candidate.writer_id))
    plus = [donor for donor in donors if donor.latex == "+"]
    if not plus:
        raise SystemExit("AST synthetic needs isolated handwritten '+' donors (MathWriting symbols/ or Detexify)")
    terms = _limit([donor for donor in donors if donor.latex != "+"], args.ast_donor_limit, args.seed)
    return sorted(terms + plus, key=lambda donor: donor.id)


def _symbol_label(candidate: Candidate) -> str | None:
    try:
        expression = MathWritingImporter().parse(candidate.path)
    except Exception:
        return None
    return expression.latex if expression.split in (None, "symbols", "train") else None


def _plan_hash(stage: str, mode: str, index_key: str, chunk: list[Candidate], extra: dict[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps({"format": FORMAT_VERSION, "stage": stage, "mode": mode, "index": index_key,
                                      "extra": extra, "raster": RASTER,
                                      "preprocessing": PREPROCESSING.to_dict()}, sort_keys=True).encode())
    for candidate in chunk:
        digest.update(f"{candidate.source}|{candidate.id}|{candidate.path}|{candidate.offset}\n".encode())
    return digest.hexdigest()


def plan_stage(stage: str, args: argparse.Namespace, output: Path, index_key: str) -> list[ChunkTask]:
    sources, mode, upsample = stage_sources(stage, args)
    for source, candidates in sources.items():
        print(f"  {stage}: {source} candidates {len(candidates)}", flush=True)
    directory = output / STAGE_DIRS[stage]
    rng = random.Random(f"{args.seed}:{stage}")
    if args.smoke or stage == "val":
        target = SMOKE_TARGETS[stage] if args.smoke else 2 * VAL_QUOTA
        active = [source for source, candidates in sources.items() if candidates]
        if stage == "val":
            caps = {source: target // 2 for source in sources}
        else:
            caps = {source: math.ceil(target / max(1, len(active))) for source in active}
        pool = []
        for source in active:
            chosen = list(sources[source])
            rng.shuffle(chosen)
            pool += chosen if stage == "val" else chosen[:3 * caps[source]]
        rng.shuffle(pool)
        extra = {"caps": caps, "target": target, "upsample": upsample, "max_tokens": args.max_tokens,
                 "unimer_max_tokens": args.unimer_max_tokens, "unimer_min_ink_height": args.unimer_min_ink_height}
        return [ChunkTask(stage, str(directory), 0, mode, pool, _plan_hash(stage, mode, index_key, pool, extra),
                          target, caps, upsample)]
    pool = [candidate for source in sources for candidate in _limit(sources[source], args.limit_per_source,
                                                                     args.seed)]
    pool.sort(key=lambda c: (c.source, c.id))
    rng.shuffle(pool)
    extra = {"upsample": upsample, "max_tokens": args.max_tokens, "ast_seed": args.seed,
             "unimer_max_tokens": args.unimer_max_tokens, "unimer_min_ink_height": args.unimer_min_ink_height}
    tasks = []
    for index, start in enumerate(range(0, len(pool), args.shard_size)):
        chunk = pool[start:start + args.shard_size]
        tasks.append(ChunkTask(stage, str(directory), index, mode, chunk,
                               _plan_hash(stage, mode, index_key, chunk, extra), None, None, upsample))
    return tasks


def run_stage(tasks: list[ChunkTask], pool: ProcessPoolExecutor | None, force: bool = False) -> list[dict[str, Any]]:
    if not tasks:
        return []
    directory = Path(tasks[0].directory)
    directory.mkdir(parents=True, exist_ok=True)
    planned = {_shard_name(task.index) for task in tasks}
    for stale in [*directory.glob("shard-*.tar"), *directory.glob("shard-*.json"), *directory.glob("*.tmp")]:
        if stale.name.split(".")[0] not in planned or stale.suffix == ".tmp":
            stale.unlink()
    summaries: dict[int, dict[str, Any]] = {}
    pending = []
    for task in tasks:
        existing = None if force else _sidecar_valid(directory, task.index, task.plan_hash)
        if existing is not None:
            summaries[task.index] = existing
        else:
            pending.append(task)
    if summaries:
        print(f"  {tasks[0].stage}: resuming, {len(summaries)}/{len(tasks)} shards already packed", flush=True)
    started = time.monotonic()
    results = pool.map(pack_chunk, pending) if pool is not None else map(pack_chunk, pending)
    for done, summary in enumerate(results, start=1):
        summaries[summary["index"]] = summary
        dropped = sum(sum(reasons.values()) for reasons in summary["dropped"].values())
        print(f"  {summary['stage']} {summary['file'] or '(empty)'}: {summary['samples']} samples, "
              f"{dropped} dropped [{done}/{len(pending)}, {time.monotonic() - started:.0f}s]", flush=True)
    return [summaries[index] for index in sorted(summaries)]


# --------------------------------------------------------------------------- manifest


def stage_summary(stage: str, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    sources, candidates = Counter(), Counter()
    dropped: dict[str, Counter] = defaultdict(Counter)
    for summary in summaries:
        sources.update(summary["sources"])
        candidates.update(summary["candidates"])
        for source, reasons in summary["dropped"].items():
            dropped[source].update(reasons)
    leakage = {source: sum(n for reason, n in reasons.items() if reason in LEAKAGE_REASONS)
               for source, reasons in dropped.items()}
    return {
        "directory": STAGE_DIRS[stage], "samples": sum(sources.values()), "sources": dict(sources),
        "candidates": dict(candidates), "dropped": {source: dict(reasons) for source, reasons in dropped.items()},
        "dropped_leakage": {source: n for source, n in leakage.items() if n},
        "with_structure": sum(summary.get("with_structure", 0) for summary in summaries),
        "shards": [{key: summary[key] for key in ("file", "samples", "bytes", "sha256", "sources")}
                   for summary in summaries if summary.get("file")],
    }


def _load_stage_summaries(directory: Path) -> list[dict[str, Any]]:
    summaries = []
    for sidecar in sorted(directory.glob("shard-*.json")):
        summary = json.loads(sidecar.read_text(encoding="utf-8"))
        if not summary.get("file") or (directory / summary["file"]).is_file():
            summaries.append(summary)
    return summaries


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_outputs(output: Path, args: argparse.Namespace, index: LeakageIndex,
                  structure_vocab: dict[str, Any] | None) -> dict[str, Any]:
    """Stages not repacked this run and without local sidecars keep their entry from an existing manifest."""
    previous = _read_json(output / "manifest.json").get("stages") or {}
    stages, labels, leaks, preserved = {}, [], [], []
    for stage, directory in STAGE_DIRS.items():
        summaries = _load_stage_summaries(output / directory) if (output / directory).is_dir() else []
        if not summaries and stage not in (args.stages or ()) and stage in previous:
            stages[stage] = previous[stage]
            preserved.append(stage)
            continue
        if not (output / directory).is_dir():
            continue
        stages[stage] = stage_summary(stage, summaries)
        for summary in summaries:
            labels += summary["labels"]
            leaks += [{"stage": stage, **leak} for leak in summary["leaks"]]
    if preserved:
        print(f"Manifest: kept existing entries for {', '.join(preserved)}", flush=True)
        previous_report = _read_json(output / "audit_report.json")
        leaks += [leak for leak in previous_report.get("leaks", []) if leak.get("stage") in preserved]
    tokenizer_info: dict[str, Any] = {"file": "tokenizer.json", "label_normalization": LABEL_VERSION}
    if args.fixed_tokenizer:
        source = Path(args.fixed_tokenizer)
        tokenizer = LatexTokenizer.load(source)
        if tokenizer.label_normalization != LABEL_VERSION:
            raise SystemExit(f"--fixed-tokenizer {source} uses label normalization "
                             f"{tokenizer.label_normalization!r}, packer needs {LABEL_VERSION!r}")
        if not (output / "tokenizer.json").is_file() or not os.path.samefile(source, output / "tokenizer.json"):
            shutil.copyfile(source, output / "tokenizer.json")
        known = set(tokenizer.tokens)
        oov = Counter(token for label in labels for token in tokenize_latex(label) if token not in known)
        oov_labels = sum(1 for label in labels if any(token not in known for token in tokenize_latex(label)))
        tokenizer_info.update(fixed_tokenizer=True, source=str(source), oov_labels=oov_labels,
                              oov_tokens=dict(oov.most_common(50)))
        if oov_labels:
            print(f"warning: {oov_labels} labels contain tokens outside the fixed tokenizer (encoded as <UNK>): "
                  f"{dict(oov.most_common(10))}", flush=True)
    else:
        if preserved:
            print(f"warning: tokenizer.json is rebuilt without the labels of {', '.join(preserved)}; "
                  "pass --fixed-tokenizer to keep the original vocabulary", flush=True)
        tokenizer = LatexTokenizer.build(labels)
        tokenizer.label_normalization = LABEL_VERSION
        tokenizer.save(output / "tokenizer.json")
        tokenizer_info["fixed_tokenizer"] = False
    tokenizer_info.update(size=len(tokenizer), tokenizer_sha256=sha256_file(output / "tokenizer.json"))
    leakage_totals = {stage: summary.get("dropped_leakage", {}) for stage, summary in stages.items()}
    manifest = {
        "format_version": FORMAT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "contract": {
            "json": ["id", "source", "split", "latex", "normalized_latex", "has_strokes", "has_timing",
                     "image_width", "stroke_count", "writer_id"],
            "png": "LA uint8, height 128, width multiple of 16 in [64, 640]; L 0=background 255=ink; "
                   "A 0=background/unknown order, 1..255 stroke-order progress",
            "npz": "features float16 [T, 10] (+ optional stroke_symbol_ids, stroke_group_ids int16 [S], "
                   "relation_edges int16 [E, 3]); present iff has_strokes",
        },
        "preprocessing": PREPROCESSING.to_dict(), "feature_names": list(FEATURE_NAMES), "raster": RASTER,
        "tokenizer": tokenizer_info,
        "structure": structure_vocab,
        "synthetic_sources": sorted(SYNTHETIC_SOURCES),
        "leakage": {"complete": index.complete, "test_samples": len(index), "counts": dict(index.counts),
                    "reserved_writers": len(index.reserved_writers), "near_distance": index.near_distance,
                    "dropped": leakage_totals},
        "stages": stages,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    by_reason: dict[str, Counter] = defaultdict(Counter)
    for leak in leaks:
        by_reason[leak["stage"]][f"{leak['source']}:{leak['reason']}"] += 1
    report = {
        "created_at": manifest["created_at"], "leakage_index": manifest["leakage"],
        "stages": {stage: {source: {"candidates": summary.get("candidates", {}).get(source, 0),
                                    "accepted": summary.get("sources", {}).get(source, 0),
                                    "dropped": summary.get("dropped", {}).get(source, {})}
                           for source in sorted(set(summary.get("candidates", {})) | set(summary.get("sources", {})))}
                   for stage, summary in stages.items()},
        "leak_counts": {stage: dict(counts) for stage, counts in by_reason.items()},
        "leaks": leaks[:5000],
    }
    (output / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stages", nargs="+", choices=("stage1", "stage2", "val"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true", help="repack shards even if a matching sidecar exists")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--crohme-root", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--mathwriting-root", type=Path, default=DEFAULT_MATHWRITING)
    parser.add_argument("--hme-root", type=Path, default=DEFAULT_HME)
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--crohme-upsample", type=float, default=3.0)
    parser.add_argument("--crohme-artificial", action="store_true")
    parser.add_argument("--crohme-artificial-limit", type=int)
    parser.add_argument("--mw-synthetic-limit", type=int)
    parser.add_argument("--stage2-hme-fraction", type=float, default=0.3)
    parser.add_argument("--with-unimer", action="store_true")
    parser.add_argument("--with-detexify", action="store_true")
    parser.add_argument("--detexify-per-symbol", type=int, default=30)
    parser.add_argument("--ast-count", type=int, default=0)
    parser.add_argument("--ast-donor-limit", type=int)
    parser.add_argument("--reserved-writers", help="comma list, .txt (one per line) or .json list")
    parser.add_argument("--block-test-writers", action="store_true")
    parser.add_argument("--leakage-limit", type=int, help="cap test samples per test source (smoke only)")
    parser.add_argument("--rebuild-leakage", action="store_true")
    parser.add_argument("--download-detexify", action="store_true")
    parser.add_argument("--detexify-limit", type=int)
    parser.add_argument("--download-unimer", action="store_true")
    parser.add_argument("--unimer-limit", type=int, default=300000)
    parser.add_argument("--unimer-max-tokens", type=int, default=96)
    parser.add_argument("--unimer-min-ink-height", type=int, default=44)
    parser.add_argument("--unimer-stream", action="store_true")
    parser.add_argument("--fixed-tokenizer", type=Path,
                        help="copy this tokenizer.json verbatim instead of rebuilding the vocabulary from packed labels")
    parser.add_argument("--hf-token")
    args = parser.parse_args(argv)
    if args.fixed_tokenizer is not None and not args.fixed_tokenizer.is_file():
        parser.error(f"--fixed-tokenizer {args.fixed_tokenizer} does not exist")
    if args.smoke:
        args.output = args.output or DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + "-smoke")
        if args.leakage_limit is None:
            args.leakage_limit = 256
        if args.ast_count:
            args.ast_count = min(args.ast_count, 16)
            args.ast_donor_limit = args.ast_donor_limit or 400
    args.output = args.output or DEFAULT_OUTPUT
    if args.stages is None:
        args.stages = [] if (args.download_detexify or args.download_unimer) else ["stage1", "stage2", "val"]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    external = Path(args.external_root)
    try:
        if args.download_detexify:
            download_detexify(external / "detexify", limit=args.detexify_limit)
        if args.download_unimer:
            download_unimer(external / "unimer", limit=args.unimer_limit, max_tokens=args.max_tokens,
                            seed=args.seed, stream=args.unimer_stream, token=args.hf_token or False)
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not args.stages:
        return 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    index, index_key = build_leakage_index(args, output)

    structure_vocab, structure = None, {}
    if "stage2" in args.stages and Path(args.structure).is_file():
        payload = json.loads(Path(args.structure).read_text(encoding="utf-8"))
        if payload.get("split") != "train":
            raise SystemExit("Structure annotations must come from the training split")
        structure = {key: value for key, value in payload["records"].items() if key.startswith("crohme:train/")}
        structure_vocab = {"symbols": payload["symbols"], "relation_classes": payload["relation_classes"],
                           "source": str(args.structure), "records": len(structure)}
    elif (output / "manifest.json").is_file():
        structure_vocab = json.loads((output / "manifest.json").read_text(encoding="utf-8")).get("structure")
    donors = ast_donors(args) if "stage2" in args.stages and args.ast_count else []
    if donors:
        print(f"AST synthetic: {len(donors)} training-only donors", flush=True)
    context = PackContext(index=index, max_tokens=args.max_tokens, unimer_max_tokens=args.unimer_max_tokens,
                          unimer_min_ink_height=args.unimer_min_ink_height, structure=structure, donors=donors,
                          ast_seed=args.seed, ast_count=args.ast_count)

    plans = {}
    for stage in args.stages:
        plans[stage] = plan_stage(stage, args, output, index_key)
    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(context,))
    else:
        _init_worker(context)
    try:
        for stage, tasks in plans.items():
            print(f"Packing {stage}: {len(tasks)} shard(s) -> {output / STAGE_DIRS[stage]}", flush=True)
            run_stage(tasks, pool, args.force)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    manifest = write_outputs(output, args, index, structure_vocab)
    for stage, summary in manifest["stages"].items():
        print(f"{stage}: {summary['samples']} samples in {len(summary['shards'])} shard(s) {summary['sources']}; "
              f"leakage drops {summary['dropped_leakage']}")
    print(f"Wrote {output / 'manifest.json'}, {output / 'audit_report.json'}, {output / 'tokenizer.json'}")
    return 0


if __name__ == "__main__":
    from data.bimodal_pack import main as _main  # workers must pickle data.bimodal_pack, not __main__

    raise SystemExit(_main())
