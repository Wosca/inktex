"""InkForge fast packed dataset builder & PyTorch training dataset (zero holdout leakage).

Packs real human online handwriting from:
  - MathWriting 2024 `train` (~230k human expressions)
  - CROHME 23 `train` (~12k human expressions)
  - IAMonDo v1 `train` (~32.5k human text lines + sliced 1-4 word `\\text{...}` phrases)
  - HWRT `train` (~151k human symbols)
  - Detexify `samples` (capped per symbol for balanced rare-symbol coverage)

Strictly excludes ALL test/validation holdouts (CROHME test/val, MathWriting test/val, IAMonDo test/val).
Supports cross-sample same-writer style conditioning: when a writer has >=2 samples in the training pool,
the Style Encoder sees strokes from a *different* equation by the same writer so `z` is forced to capture
pure handwriting style rather than equation content!
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from v2.ink_preprocess import (  # noqa: E402
    _dedupe_stroke,
    _resample_stroke,
    estimate_glyph_scale,
    extract_style_strokes,
    warp_raw_strokes,
)
from inkforge.model import InkForgeConfig  # noqa: E402
from inkforge.tokenizer import InkForgeTokenizer, tokenize_inkforge  # noqa: E402


DATASETS_ROOT = Path(r"E:\math-recogniser-datasets")
DEFAULT_CROHME_TRAIN = DATASETS_ROOT / "canonical" / "crohme23" / "train.jsonl"
DEFAULT_HWRT_TRAIN = DATASETS_ROOT / "canonical" / "hwrt" / "train.jsonl"
DEFAULT_IAMONDO_TRAIN = DATASETS_ROOT / "canonical" / "iamondo-v1" / "train.jsonl"
DEFAULT_DETEXIFY = DATASETS_ROOT / "external" / "detexify" / "samples.jsonl"
DEFAULT_MATHWRITING_TRAIN = Path(r"E:\mathwriting-2024\train")
DEFAULT_MATHWRITING_SYMBOLS = Path(r"E:\mathwriting-2024\symbols")
DEFAULT_CACHE_DIR = Path(r"E:\math-recogniser-training\inkforge_data")

SOURCE_NAMES = ("mathwriting", "crohme", "iamondo", "hwrt", "detexify", "mw_symbols")
SOURCE_TO_ID = {name: i for i, name in enumerate(SOURCE_NAMES)}


# ---------------------------------------------------------------------------
# Stroke Normalization & Sequence Conversion
# ---------------------------------------------------------------------------

def normalize_and_resample_for_inkforge(
    raw_strokes: Sequence[np.ndarray],
    *,
    spacing: float = 0.12,
    max_points: int = 512,
    coord_clip: float = 18.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Normalize strokes by glyph scale `h`, anchor first point at `(0, 0)`, resample to `spacing * h`.

    Returns `(xy [N, 2] float32, pen_state [N] uint8)` where:
      - `xy[0] == (0, 0)`
      - `pen_state[i] == 1` iff point `i` is the last point of its stroke (pen lifts after `i`), else `0`.
      - Stroke 0 is guaranteed to have >= 2 points (1-point dots get a tiny 0.01*h stub).
    """
    clean: list[np.ndarray] = []
    for st in raw_strokes:
        arr = np.asarray(st, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            continue
        xy = arr[:, :2]
        finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        xy = xy[finite]
        if len(xy) > 0:
            clean.append(xy)
    if not clean:
        return None

    h = estimate_glyph_scale(clean)
    if not math.isfinite(h) or h <= 1e-6:
        return None

    origin = clean[0][0].copy()
    deduped = [_dedupe_stroke((s - origin) / h, None, 1e-3)[0] for s in clean]

    # Coarsen slightly if raw expression exceeds max_points
    eff_spacing = spacing
    resampled = [_resample_stroke(s, None, eff_spacing)[0] for s in deduped]
    total_pts = sum(len(s) for s in resampled)
    if total_pts > max_points:
        for factor in (1.35, 1.75, 2.25, 3.0):
            resampled = [_resample_stroke(s, None, spacing * factor)[0] for s in deduped]
            total_pts = sum(len(s) for s in resampled)
            if total_pts <= max_points:
                break
    if total_pts > max_points or total_pts < 2:
        return None

    # Ensure Stroke 0 has >= 2 points so BOS -> step 0 is always a drawn segment on Stroke 0
    if len(resampled[0]) == 1:
        p0 = resampled[0][0]
        resampled[0] = np.stack([p0, p0 + np.array([0.01, 0.01], dtype=np.float64)], axis=0)
        if sum(len(s) for s in resampled) > max_points:
            return None

    pts_list: list[np.ndarray] = []
    pen_list: list[np.ndarray] = []
    for st in resampled:
        n = len(st)
        pts_list.append(st)
        pen = np.zeros((n,), dtype=np.uint8)
        pen[-1] = 1
        pen_list.append(pen)

    xy_all = np.concatenate(pts_list, axis=0).astype(np.float32)
    pen_all = np.concatenate(pen_list, axis=0)

    # Anchor exact origin
    xy_all -= xy_all[0:1]

    # Sanity check deltas & bounds
    diffs = np.diff(xy_all, axis=0)
    if np.abs(diffs).max() > coord_clip * 1.5:
        return None
    if np.abs(xy_all[:, 0]).max() > 36.0 or np.abs(xy_all[:, 1]).max() > 18.0:
        return None

    return xy_all, pen_all


def unpack_points_to_strokes(xy: np.ndarray, pen: np.ndarray) -> list[np.ndarray]:
    """Convert packed `(xy [N, 2], pen [N])` back to list of `[Ni, 2]` strokes."""
    strokes: list[np.ndarray] = []
    start = 0
    n = len(xy)
    for i in range(n):
        if pen[i] == 1 or i == n - 1:
            strokes.append(xy[start : i + 1].astype(np.float64))
            start = i + 1
    return strokes


# ---------------------------------------------------------------------------
# Fast Raw Dataset Parsers
# ---------------------------------------------------------------------------

def _parse_inkml_file(path_str: str) -> tuple[str, str, list[np.ndarray]] | None:
    """Parse one MathWriting `.inkml` file into `(label, writer_id, strokes)`."""
    p = Path(path_str)
    try:
        root = ET.parse(p).getroot()
    except Exception:
        return None

    label = ""
    raw_label = ""
    writer_id = ""
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "annotation":
            atype = elem.attrib.get("type", "")
            text = (elem.text or "").strip()
            if atype == "normalizedLabel" and text:
                label = text
            elif atype == "label" and text:
                raw_label = text
            elif atype in ("writer", "writerId", "author") and text:
                writer_id = text
    latex = label or raw_label
    if not latex:
        return None

    strokes: list[np.ndarray] = []
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] != "trace":
            continue
        pts: list[tuple[float, float]] = []
        for tok in (elem.text or "").strip().split(","):
            parts = tok.strip().split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if pts:
            strokes.append(np.asarray(pts, dtype=np.float64))
    if not strokes:
        return None
    return latex, writer_id, strokes


def _parse_jsonl_strokes(raw_strokes: Any) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    if not isinstance(raw_strokes, list):
        return out
    for s in raw_strokes:
        pts: list[tuple[float, float]] = []
        if isinstance(s, dict):
            for p in s.get("points", []):
                if isinstance(p, dict) and "x" in p and "y" in p:
                    pts.append((float(p["x"]), float(p["y"])))
        elif isinstance(s, list):
            for p in s:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
        if pts:
            out.append(np.asarray(pts, dtype=np.float64))
    return out


def _process_mw_batch(args: tuple[list[str], float, int]) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    paths, spacing, max_points = args
    results: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for p in paths:
        parsed = _parse_inkml_file(p)
        if parsed is None:
            continue
        latex, wid, strokes = parsed
        norm = normalize_and_resample_for_inkforge(strokes, spacing=spacing, max_points=max_points)
        if norm is None:
            continue
        xy, pen = norm
        results.append((latex, f"mw:{wid}" if wid else "", xy.astype(np.float16), pen))
    return results


def _slice_iamondo_words(
    latex: str,
    strokes: list[np.ndarray],
    max_words: int = 4,
) -> list[tuple[str, list[np.ndarray]]]:
    """Keep short IAMonDo textlines (<= 28 chars) directly as `\\text{...}` phrases."""
    s = latex.strip()
    if s.startswith(r"\text{") and s.endswith("}"):
        inner = s[6:-1].strip()
    else:
        inner = s
    if not inner or len(inner) > 36:
        return []
    words = inner.split()
    if not words or len(words) > max_words:
        return []
    # Only clean ASCII printable words/phrases
    if not all(32 <= ord(c) < 127 for c in inner):
        return []
    return [(r"\text{" + inner + "}", strokes)]


def _packed_record_digest(ids: Sequence[int], xy16: np.ndarray, pen: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(ids, dtype=np.uint16).tobytes())
    h.update(b"|")
    h.update(np.ascontiguousarray(xy16, dtype=np.float16).tobytes())
    h.update(b"|")
    h.update(np.ascontiguousarray(pen, dtype=np.uint8).tobytes())
    return h.digest()


def _split_records_leak_free(
    records: Sequence[tuple[list[int], str, int, np.ndarray, np.ndarray]],
    *,
    val_fraction: float = 0.015,
    seed: int = 42,
) -> tuple[list[tuple[list[int], str, int, np.ndarray, np.ndarray]], list[tuple[list[int], str, int, np.ndarray, np.ndarray]]]:
    """Group exact packed duplicates AND shared named writers before assigning groups to train/val.

    Guarantees:
      - Zero exact packed `(token_ids, xy16, pen)` overlap between train and val.
      - Zero named writer overlap between train and val.
    """
    n = len(records)
    if n == 0:
        return [], []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_digest: dict[bytes, int] = {}
    seen_writer: dict[str, int] = {}
    for i, (ids, wid, _sid, xy16, pen) in enumerate(records):
        dig = _packed_record_digest(ids, xy16, pen)
        prev_d = seen_digest.get(dig)
        if prev_d is not None:
            union(i, prev_d)
        else:
            seen_digest[dig] = i

        if wid:
            prev_w = seen_writer.get(wid)
            if prev_w is not None:
                union(i, prev_w)
            else:
                seen_writer[wid] = i

    groups_map: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(i)

    groups = list(groups_map.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    target_val = max(1, int(round(n * val_fraction)))
    max_single_group = max(64, int(target_val * 0.25))
    val_idxs: list[int] = []
    train_idxs: list[int] = []

    for g in groups:
        if len(val_idxs) < target_val and len(g) <= max_single_group:
            val_idxs.extend(g)
        else:
            train_idxs.extend(g)

    rng.shuffle(val_idxs)
    rng.shuffle(train_idxs)
    train_records = [records[i] for i in train_idxs]
    val_records = [records[i] for i in val_idxs]
    return train_records, val_records


def resplit_existing_cache_leak_free(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    spacing: float = 0.12,
    max_points: int = 512,
    max_text_len: int = 160,
    hwrt_cap_per_symbol: int = 400,
    val_fraction: float = 0.015,
    seed: int = 42,
) -> Path:
    """Fast (~20s) zero-leakage split rebuild: reuses packed writerless sources (MathWriting/Detexify/MW-symbols)
    and reloads true string writer IDs for CROHME/IAMonDo/HWRT, then groups exact duplicates + named writers.
    """
    t0 = time.perf_counter()
    tokenizer = InkForgeTokenizer.load(cache_dir / "tokenizer.json")
    rng = random.Random(seed)
    all_records: list[tuple[list[int], str, int, np.ndarray, np.ndarray]] = []

    # 1. Reuse already-packed writerless sources (0=mathwriting, 4=detexify, 5=mw_symbols)
    writerless_sids = {SOURCE_TO_ID["mathwriting"], SOURCE_TO_ID["detexify"], SOURCE_TO_ID["mw_symbols"]}
    for sp in ("train", "val"):
        sp_dir = cache_dir / sp
        if not (sp_dir / "source_ids.npy").is_file():
            continue
        pts_xy = np.load(sp_dir / "points_xy.npy")
        pts_pen = np.load(sp_dir / "points_pen.npy")
        pt_off = np.load(sp_dir / "pt_offsets.npy")
        toks = np.load(sp_dir / "tokens.npy")
        tok_off = np.load(sp_dir / "tok_offsets.npy")
        sids = np.load(sp_dir / "source_ids.npy")
        for i, sid in enumerate(sids.tolist()):
            if sid not in writerless_sids:
                continue
            p0, p1 = int(pt_off[i]), int(pt_off[i + 1])
            t0_i, t1_i = int(tok_off[i]), int(tok_off[i + 1])
            all_records.append(
                (toks[t0_i:t1_i].tolist(), "", int(sid), pts_xy[p0:p1].copy(), pts_pen[p0:p1].copy())
            )
    print(f"[resplit] loaded {len(all_records):,} packed writerless samples ({time.perf_counter()-t0:.1f}s)")

    def _add_if_valid(latex: str, wkey: str, sid: int, strokes: list[np.ndarray]) -> bool:
        toks_str = tokenize_inkforge(latex)
        if not toks_str or len(toks_str) + 2 > max_text_len:
            return False
        ids = tokenizer.encode(latex, max_len=max_text_len)
        if tokenizer.unk_id in ids:
            return False
        norm = normalize_and_resample_for_inkforge(strokes, spacing=spacing, max_points=max_points)
        if norm is None:
            return False
        xy, pen = norm
        all_records.append((ids, wkey, sid, xy.astype(np.float16), pen))
        return True

    # 2. Reload named-writer sources with true string writer keys (CROHME, IAMonDo, HWRT)
    if DEFAULT_CROHME_TRAIN.is_file():
        n0 = len(all_records)
        with DEFAULT_CROHME_TRAIN.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("synthetic"):
                    continue
                latex = (r.get("latexNormalized") or r.get("latex") or "").strip()
                if not latex:
                    continue
                wid = str(r.get("writerId") or "")
                _add_if_valid(latex, f"crohme:{wid}" if wid else "", SOURCE_TO_ID["crohme"], _parse_jsonl_strokes(r.get("strokes")))
        print(f"[resplit] crohme: {len(all_records)-n0:,} ({time.perf_counter()-t0:.1f}s)")

    if DEFAULT_IAMONDO_TRAIN.is_file():
        n0 = len(all_records)
        with DEFAULT_IAMONDO_TRAIN.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                latex = (r.get("latexNormalized") or r.get("latex") or "").strip()
                strokes = _parse_jsonl_strokes(r.get("strokes"))
                wid = str(r.get("writerId") or "")
                for sub_tex, sub_st in _slice_iamondo_words(latex, strokes):
                    _add_if_valid(sub_tex, f"iam:{wid}" if wid else "", SOURCE_TO_ID["iamondo"], sub_st)
        print(f"[resplit] iamondo: {len(all_records)-n0:,} ({time.perf_counter()-t0:.1f}s)")

    if DEFAULT_HWRT_TRAIN.is_file():
        n0 = len(all_records)
        sym_counts: Counter[str] = Counter()
        with DEFAULT_HWRT_TRAIN.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        rng.shuffle(lines)
        for line in lines:
            r = json.loads(line)
            latex = (r.get("latexNormalized") or r.get("latex") or "").strip()
            if not latex or sym_counts[latex] >= hwrt_cap_per_symbol:
                continue
            wid = str(r.get("writerId") or "")
            if _add_if_valid(latex, f"hwrt:{wid}" if wid else "", SOURCE_TO_ID["hwrt"], _parse_jsonl_strokes(r.get("strokes"))):
                sym_counts[latex] += 1
        print(f"[resplit] hwrt: {len(all_records)-n0:,} ({time.perf_counter()-t0:.1f}s)")

    train_records, val_records = _split_records_leak_free(all_records, val_fraction=val_fraction, seed=seed)

    # Strict zero-leakage verification BEFORE overwriting!
    train_digs = {_packed_record_digest(r[0], r[3], r[4]) for r in train_records}
    val_digs = {_packed_record_digest(r[0], r[3], r[4]) for r in val_records}
    dup_overlap = len(train_digs & val_digs)
    train_w = {r[1] for r in train_records if r[1]}
    val_w = {r[1] for r in val_records if r[1]}
    writer_overlap = len(train_w & val_w)
    assert dup_overlap == 0, f"Exact duplicate leakage detected: {dup_overlap}"
    assert writer_overlap == 0, f"Writer leakage detected: {writer_overlap}"
    print(f"[resplit] VERIFIED zero leakage: exact_dup_overlap=0, writer_overlap=0 (train_w={len(train_w)}, val_w={len(val_w)})")

    for split_name, split_recs in (("train", train_records), ("val", val_records)):
        _write_packed_split(cache_dir / split_name, split_recs)

    meta = {
        "format": "inkforge-cache-v2",
        "vocab_size": len(tokenizer),
        "spacing": spacing,
        "max_points": max_points,
        "max_text_len": max_text_len,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "sources": list(SOURCE_NAMES),
        "leak_free_grouped": True,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[resplit] DONE in {meta['elapsed_seconds']}s -> {cache_dir}")
    return cache_dir


def prepare_inkforge_cache(
    output_dir: Path = DEFAULT_CACHE_DIR,
    *,
    spacing: float = 0.12,
    max_points: int = 512,
    max_text_len: int = 160,
    detexify_cap_per_symbol: int = 250,
    hwrt_cap_per_symbol: int = 400,
    val_fraction: float = 0.015,
    workers: int = 8,
    seed: int = 42,
) -> Path:
    """Build compact memory-mappable numpy cache + tokenizer for fast RTX 4070 Ti InkForge training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    # Store intermediate tuples: (latex, writer_key, source_id, xy_fp16, pen_uint8)
    records: list[tuple[str, str, int, np.ndarray, np.ndarray]] = []

    # 1. CROHME 23 train (human only)
    if DEFAULT_CROHME_TRAIN.is_file():
        n_before = len(records)
        with DEFAULT_CROHME_TRAIN.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("synthetic"):
                    continue
                latex = (r.get("latexNormalized") or r.get("latex") or "").strip()
                if not latex:
                    continue
                strokes = _parse_jsonl_strokes(r.get("strokes"))
                norm = normalize_and_resample_for_inkforge(strokes, spacing=spacing, max_points=max_points)
                if norm is None:
                    continue
                xy, pen = norm
                wid = str(r.get("writerId") or "")
                records.append((latex, f"crohme:{wid}" if wid else "", SOURCE_TO_ID["crohme"], xy.astype(np.float16), pen))
        print(f"[pack] crohme train: {len(records) - n_before:,} samples ({time.perf_counter()-t0:.1f}s)")

    # 2. MathWriting 2024 train (real human online expressions!)
    if DEFAULT_MATHWRITING_TRAIN.is_dir():
        n_before = len(records)
        inkml_paths = [str(p) for p in sorted(DEFAULT_MATHWRITING_TRAIN.glob("*.inkml"))]
        chunk_size = 2048
        batches = [(inkml_paths[i : i + chunk_size], spacing, max_points) for i in range(0, len(inkml_paths), chunk_size)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res_list in ex.map(_process_mw_batch, batches):
                for latex, wid, xy16, pen in res_list:
                    records.append((latex, wid, SOURCE_TO_ID["mathwriting"], xy16, pen))
        print(f"[pack] mathwriting train: {len(records) - n_before:,} samples ({time.perf_counter()-t0:.1f}s)")

    # 3. MathWriting isolated symbols
    if DEFAULT_MATHWRITING_SYMBOLS.is_dir():
        n_before = len(records)
        sym_paths = [str(p) for p in sorted(DEFAULT_MATHWRITING_SYMBOLS.glob("*.inkml"))]
        batches = [(sym_paths[i : i + 1024], spacing, max_points) for i in range(0, len(sym_paths), 1024)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res_list in ex.map(_process_mw_batch, batches):
                for latex, wid, xy16, pen in res_list:
                    records.append((latex, wid, SOURCE_TO_ID["mw_symbols"], xy16, pen))
        print(f"[pack] mathwriting symbols: {len(records) - n_before:,} samples ({time.perf_counter()-t0:.1f}s)")

    # 4. IAMonDo v1 train (short human words/phrases for \text{...})
    if DEFAULT_IAMONDO_TRAIN.is_file():
        n_before = len(records)
        with DEFAULT_IAMONDO_TRAIN.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                latex = (r.get("latexNormalized") or r.get("latex") or "").strip()
                strokes = _parse_jsonl_strokes(r.get("strokes"))
                wid = str(r.get("writerId") or "")
                for sub_tex, sub_st in _slice_iamondo_words(latex, strokes):
                    norm = normalize_and_resample_for_inkforge(sub_st, spacing=spacing, max_points=max_points)
                    if norm is None:
                        continue
                    xy, pen = norm
                    records.append((sub_tex, f"iam:{wid}" if wid else "", SOURCE_TO_ID["iamondo"], xy.astype(np.float16), pen))
        print(f"[pack] iamondo train phrases: {len(records) - n_before:,} samples ({time.perf_counter()-t0:.1f}s)")

    # 5. HWRT train (balanced per-symbol cap)
    if DEFAULT_HWRT_TRAIN.is_file():
        n_before = len(records)
        sym_counts: Counter[str] = Counter()
        with DEFAULT_HWRT_TRAIN.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        rng.shuffle(lines)
        for line in lines:
            r = json.loads(line)
            latex = (r.get("latexNormalized") or r.get("latex") or "").strip()
            if not latex or sym_counts[latex] >= hwrt_cap_per_symbol:
                continue
            strokes = _parse_jsonl_strokes(r.get("strokes"))
            norm = normalize_and_resample_for_inkforge(strokes, spacing=spacing, max_points=max_points)
            if norm is None:
                continue
            xy, pen = norm
            wid = str(r.get("writerId") or "")
            sym_counts[latex] += 1
            records.append((latex, f"hwrt:{wid}" if wid else "", SOURCE_TO_ID["hwrt"], xy.astype(np.float16), pen))
        print(f"[pack] hwrt train (capped {hwrt_cap_per_symbol}/sym): {len(records) - n_before:,} samples ({time.perf_counter()-t0:.1f}s)")

    # 6. Detexify (balanced rare symbol coverage)
    if DEFAULT_DETEXIFY.is_file():
        n_before = len(records)
        sym_counts = Counter()
        with DEFAULT_DETEXIFY.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        rng.shuffle(lines)
        for line in lines:
            r = json.loads(line)
            latex = (r.get("latex") or r.get("command") or "").strip()
            if not latex or len(latex) > 32 or sym_counts[latex] >= detexify_cap_per_symbol:
                continue
            strokes = _parse_jsonl_strokes(r.get("strokes"))
            norm = normalize_and_resample_for_inkforge(strokes, spacing=spacing, max_points=max_points)
            if norm is None:
                continue
            xy, pen = norm
            sym_counts[latex] += 1
            records.append((latex, "", SOURCE_TO_ID["detexify"], xy.astype(np.float16), pen))
        print(f"[pack] detexify (capped {detexify_cap_per_symbol}/sym): {len(records) - n_before:,} samples ({time.perf_counter()-t0:.1f}s)")

    # Build expanded tokenizer on all kept labels
    tokenizer = InkForgeTokenizer.build((rec[0] for rec in records), min_frequency=8, max_vocab=600)
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"[pack] built tokenizer with {len(tokenizer)} tokens")

    # Filter out any record with <UNK> or > max_text_len
    filtered: list[tuple[list[int], str, int, np.ndarray, np.ndarray]] = []
    for latex, wid, sid, xy16, pen in records:
        toks = tokenize_inkforge(latex)
        if not toks or len(toks) + 2 > max_text_len:
            continue
        ids = tokenizer.encode(latex, max_len=max_text_len)
        if tokenizer.unk_id in ids:
            continue
        filtered.append((ids, wid, sid, xy16, pen))
    print(f"[pack] kept {len(filtered):,} / {len(records):,} samples after token filter")

    # Deterministic exact-duplicate + writer-grouped train/val split (zero leakage!)
    train_records, val_records = _split_records_leak_free(
        filtered,
        val_fraction=val_fraction,
        seed=seed,
    )

    for split_name, split_recs in (("train", train_records), ("val", val_records)):
        _write_packed_split(output_dir / split_name, split_recs)

    meta = {
        "format": "inkforge-cache-v2",
        "vocab_size": len(tokenizer),
        "spacing": spacing,
        "max_points": max_points,
        "max_text_len": max_text_len,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "sources": list(SOURCE_NAMES),
        "leak_free_grouped": True,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[pack] DONE in {meta['elapsed_seconds']}s -> {output_dir}")
    return output_dir


def _write_packed_split(split_dir: Path, recs: Sequence[tuple[list[int], str, int, np.ndarray, np.ndarray]]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    n = len(recs)
    pt_offsets = np.zeros((n + 1,), dtype=np.int64)
    tok_offsets = np.zeros((n + 1,), dtype=np.int64)
    source_ids = np.zeros((n,), dtype=np.uint8)
    writer_map: dict[str, int] = {"": 0}
    writer_ids = np.zeros((n,), dtype=np.int32)

    total_pts = 0
    total_toks = 0
    for i, (ids, wid, sid, xy16, _pen) in enumerate(recs):
        total_pts += len(xy16)
        total_toks += len(ids)
        pt_offsets[i + 1] = total_pts
        tok_offsets[i + 1] = total_toks
        source_ids[i] = sid
        if wid:
            if wid not in writer_map:
                writer_map[wid] = len(writer_map)
            writer_ids[i] = writer_map[wid]

    all_xy = np.empty((total_pts, 2), dtype=np.float16)
    all_pen = np.empty((total_pts,), dtype=np.uint8)
    all_toks = np.empty((total_toks,), dtype=np.uint16)

    for i, (ids, _wid, _sid, xy16, pen) in enumerate(recs):
        p0, p1 = int(pt_offsets[i]), int(pt_offsets[i + 1])
        t0, t1 = int(tok_offsets[i]), int(tok_offsets[i + 1])
        all_xy[p0:p1] = xy16
        all_pen[p0:p1] = pen
        all_toks[t0:t1] = np.asarray(ids, dtype=np.uint16)

    np.save(split_dir / "points_xy.npy", all_xy)
    np.save(split_dir / "points_pen.npy", all_pen)
    np.save(split_dir / "pt_offsets.npy", pt_offsets)
    np.save(split_dir / "tokens.npy", all_toks)
    np.save(split_dir / "tok_offsets.npy", tok_offsets)
    np.save(split_dir / "source_ids.npy", source_ids)
    np.save(split_dir / "writer_ids.npy", writer_ids)
    writer_names = [""] * len(writer_map)
    for wkey, wid_int in writer_map.items():
        writer_names[wid_int] = wkey
    (split_dir / "writer_names.json").write_text(json.dumps(writer_names), encoding="utf-8")
    print(
        f"  [{split_dir.name}] {n:,} samples | {total_pts:,} pts | "
        f"{len(writer_map)-1:,} named writers"
    )


# ---------------------------------------------------------------------------
# Fast Memory-Mapped PyTorch Dataset + Cross-Sample Same-Writer Conditioning
# ---------------------------------------------------------------------------

class InkForgeDataset(Dataset):
    """High-throughput memory-mapped dataset for InkForge training."""

    def __init__(
        self,
        cache_dir: Path,
        split: str = "train",
        *,
        cfg: InkForgeConfig | None = None,
        augment: bool = True,
        input_noise_std: float = 0.010,
        source_weights: dict[str, float] | None = None,
    ):
        self.split_dir = Path(cache_dir) / split
        self.cfg = cfg or InkForgeConfig()
        self.augment = augment
        self.input_noise_std = input_noise_std

        self.points_xy = np.load(self.split_dir / "points_xy.npy", mmap_mode="r")
        self.points_pen = np.load(self.split_dir / "points_pen.npy", mmap_mode="r")
        self.pt_offsets = np.load(self.split_dir / "pt_offsets.npy")
        self.tokens = np.load(self.split_dir / "tokens.npy", mmap_mode="r")
        self.tok_offsets = np.load(self.split_dir / "tok_offsets.npy")
        self.source_ids = np.load(self.split_dir / "source_ids.npy")
        self.writer_ids = np.load(self.split_dir / "writer_ids.npy")

        self.n_samples = len(self.source_ids)

        # Index samples per writer_id for cross-sample same-writer style conditioning!
        writer_buckets: dict[int, list[int]] = defaultdict(list)
        for idx, wid in enumerate(self.writer_ids.tolist()):
            if wid > 0:
                writer_buckets[wid].append(idx)
        self.writer_peers: dict[int, np.ndarray] = {
            wid: np.asarray(idxs, dtype=np.int64)
            for wid, idxs in writer_buckets.items()
            if len(idxs) >= 2
        }

        # Optional per-sample weights for balanced expression vs symbol sampling
        if source_weights is None:
            source_weights = {
                "mathwriting": 1.0,
                "crohme": 2.5,
                "iamondo": 1.5,
                "hwrt": 0.20,
                "detexify": 0.25,
                "mw_symbols": 0.30,
            }
        w_table = np.array([source_weights.get(name, 1.0) for name in SOURCE_NAMES], dtype=np.float64)
        self.sample_weights = w_table[self.source_ids]

    def __len__(self) -> int:
        return self.n_samples

    def _get_raw_strokes(self, idx: int) -> list[np.ndarray]:
        p0, p1 = int(self.pt_offsets[idx]), int(self.pt_offsets[idx + 1])
        xy = np.array(self.points_xy[p0:p1], dtype=np.float64)
        pen = np.array(self.points_pen[p0:p1], dtype=np.uint8)
        return unpack_points_to_strokes(xy, pen)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng() if self.augment else np.random.default_rng(int(idx) + 1000003)
        t0, t1 = int(self.tok_offsets[idx]), int(self.tok_offsets[idx + 1])
        tok_ids = np.array(self.tokens[t0:t1], dtype=np.int64)

        p0, p1 = int(self.pt_offsets[idx]), int(self.pt_offsets[idx + 1])
        xy = np.array(self.points_xy[p0:p1], dtype=np.float32)
        pen = np.array(self.points_pen[p0:p1], dtype=np.int64)

        # Mild realistic geometric augmentation (rotation, shear, anisotropic scale)
        if self.augment:
            angle = math.radians(float(rng.uniform(-3.0, 3.0)))
            ca, sa = math.cos(angle), math.sin(angle)
            sx = float(rng.uniform(0.92, 1.08))
            sy = float(rng.uniform(0.92, 1.08))
            sh = float(rng.uniform(-0.05, 0.05))
            x0, y0 = xy[:, 0] * sx, xy[:, 1] * sy
            x1 = x0 + sh * y0
            xy = np.stack([x1 * ca - y0 * sa, x1 * sa + y0 * ca], axis=-1).astype(np.float32)
            xy -= xy[0:1]

        # Build autoregressive training inputs & targets (length T = N, where step N-1 predicts EOS!)
        # Points: P[0], P[1], ..., P[N-1] with P[0]=(0,0).
        # Step 0..N-2: predicts delta P[t+1] - P[t] and pen[t+1] (0=down, 1=lift)
        # Step N-1:    predicts EOS (pen=2)
        n_pts = len(xy)
        n_steps = n_pts  # T = N (steps 0..N-2 are points 1..N-1; step N-1 is EOS)

        deltas = np.diff(xy, axis=0)  # [N-1, 2]
        target_dx = np.zeros((n_steps,), dtype=np.float32)
        target_dy = np.zeros((n_steps,), dtype=np.float32)
        target_pen = np.full((n_steps,), 2, dtype=np.int64)  # last step is EOS=2

        target_dx[:-1] = np.clip(deltas[:, 0], -self.cfg.coord_clip, self.cfg.coord_clip)
        target_dy[:-1] = np.clip(deltas[:, 1], -self.cfg.coord_clip, self.cfg.coord_clip)
        target_pen[:-1] = pen[1:]  # 0=pen_down, 1=pen_lift

        # Decoder input at step 0 is BOS; at step t>=1 is previous step's (dx, dy, abs_x, abs_y, pen_onehot)
        # Built from ONE self-consistent perturbed trajectory `xy_in` so `cumsum(in_dx) == in_ax`!
        stroke_in = np.zeros((n_steps, 7), dtype=np.float32)
        stroke_in[0, 6] = 1.0  # is_bos
        if n_steps > 1:
            xy_in = xy.copy()
            if self.augment and self.input_noise_std > 0:
                # Point i (i>=1) is a stroke start (after an air jump) iff pen[i-1] == 1
                is_stroke_start = (pen[:-1] == 1).astype(np.float32)
                is_drawn = 1.0 - is_stroke_start
                drift_x = np.cumsum(
                    is_stroke_start * rng.normal(0.0, self.input_noise_std * 1.5, size=(n_steps - 1,)).astype(np.float32)
                )
                drift_y = np.cumsum(
                    is_stroke_start * rng.normal(0.0, self.input_noise_std * 1.5, size=(n_steps - 1,)).astype(np.float32)
                )
                jitter_x = is_drawn * rng.normal(0.0, self.input_noise_std, size=(n_steps - 1,)).astype(np.float32)
                jitter_y = is_drawn * rng.normal(0.0, self.input_noise_std, size=(n_steps - 1,)).astype(np.float32)
                xy_in[1:, 0] += drift_x + jitter_x
                xy_in[1:, 1] += drift_y + jitter_y

            in_deltas = np.diff(xy_in, axis=0)
            in_dx = np.clip(in_deltas[:, 0], -self.cfg.coord_clip, self.cfg.coord_clip)
            in_dy = np.clip(in_deltas[:, 1], -self.cfg.coord_clip, self.cfg.coord_clip)
            in_ax = np.clip(xy_in[1:, 0], -self.cfg.abs_x_clip, self.cfg.abs_x_clip)
            in_ay = np.clip(xy_in[1:, 1], -self.cfg.abs_y_clip, self.cfg.abs_y_clip)

            stroke_in[1:, 0] = in_dx
            stroke_in[1:, 1] = in_dy
            stroke_in[1:, 2] = in_ax
            stroke_in[1:, 3] = in_ay
            stroke_in[1:, 4] = (target_pen[:-1] == 0).astype(np.float32)
            stroke_in[1:, 5] = (target_pen[:-1] == 1).astype(np.float32)

        # Style reference strokes: STRICTLY from a DIFFERENT equation by the SAME named writer!
        # Writerless / single-sample rows get empty style_strokes (style_mask = all True -> exact z = 0)
        # so the Style Encoder can NEVER leak the target equation's own strokes!
        wid = int(self.writer_ids[idx])
        style_strokes: list[np.ndarray]
        if wid in self.writer_peers:
            peers = self.writer_peers[wid]
            peer_idx = int(peers[int(rng.integers(0, len(peers)))])
            if peer_idx == idx and len(peers) > 1:
                peer_idx = int(peers[(int(np.searchsorted(peers, idx)) + 1) % len(peers)])
            style_strokes = self._get_raw_strokes(peer_idx)
        else:
            style_strokes = []

        if self.augment and style_strokes:
            style_strokes = warp_raw_strokes(
                style_strokes,
                rng,
                rotation_degrees=2.0,
                shear=0.03,
                anisotropic_scale=(0.95, 1.05),
                stroke_shift=0.01,
            )

        sp, sa, sm = extract_style_strokes(
            style_strokes,
            max_strokes=self.cfg.style_max_strokes,
            points_per_stroke=self.cfg.style_points,
        )

        return {
            "token_ids": tok_ids,
            "stroke_inputs": stroke_in,
            "target_dx": target_dx,
            "target_dy": target_dy,
            "target_pen": target_pen,
            "style_points": sp,
            "style_aspect": sa,
            "style_mask": sm,
            "source_id": int(self.source_ids[idx]),
        }


def collate_inkforge_batch(items: Sequence[dict[str, Any]], pad_id: int = 0) -> dict[str, Tensor]:
    b = len(items)
    max_l = max(len(it["token_ids"]) for it in items)
    max_t = max(len(it["target_pen"]) for it in items)

    token_ids = torch.full((b, max_l), pad_id, dtype=torch.long)
    stroke_inputs = torch.zeros((b, max_t, 7), dtype=torch.float32)
    target_dx = torch.zeros((b, max_t), dtype=torch.float32)
    target_dy = torch.zeros((b, max_t), dtype=torch.float32)
    target_pen = torch.full((b, max_t), 2, dtype=torch.long)
    stroke_mask = torch.zeros((b, max_t), dtype=torch.bool)

    style_points = torch.from_numpy(np.stack([it["style_points"] for it in items], axis=0))
    style_aspect = torch.from_numpy(np.stack([it["style_aspect"] for it in items], axis=0))
    style_mask = torch.from_numpy(np.stack([it["style_mask"] for it in items], axis=0)).bool()
    source_id = torch.tensor([int(it.get("source_id", 0)) for it in items], dtype=torch.long)

    for i, it in enumerate(items):
        l = len(it["token_ids"])
        t = len(it["target_pen"])
        token_ids[i, :l] = torch.from_numpy(it["token_ids"])
        stroke_inputs[i, :t] = torch.from_numpy(it["stroke_inputs"])
        target_dx[i, :t] = torch.from_numpy(it["target_dx"])
        target_dy[i, :t] = torch.from_numpy(it["target_dy"])
        target_pen[i, :t] = torch.from_numpy(it["target_pen"])
        stroke_mask[i, :t] = True

    return {
        "token_ids": token_ids,
        "stroke_inputs": stroke_inputs,
        "target_dx": target_dx,
        "target_dy": target_dy,
        "target_pen": target_pen,
        "stroke_mask": stroke_mask,
        "style_points": style_points,
        "style_aspect": style_aspect,
        "style_mask": style_mask,
        "source_id": source_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build packed binary InkForge training cache from local datasets")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--spacing", type=float, default=0.12)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resplit-only", action="store_true", help="Fast zero-leakage resplit reusing packed MathWriting/Detexify")
    args = parser.parse_args()
    if args.resplit_only:
        resplit_existing_cache_leak_free(
            cache_dir=args.output_dir,
            spacing=args.spacing,
            max_points=args.max_points,
        )
    else:
        prepare_inkforge_cache(
            output_dir=args.output_dir,
            spacing=args.spacing,
            max_points=args.max_points,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
