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
DEFAULT_MATHWRITING_SYNTHETIC = Path(r"E:\mathwriting-2024\synthetic")
DEFAULT_CACHE_DIR = Path(r"E:\math-recogniser-training\inkforge_data")
DEFAULT_V2_CACHE_DIR = Path(r"E:\math-recogniser-training\inkforge_data_v2")
DEFAULT_V3_CACHE_DIR = Path(r"E:\math-recogniser-training\inkforge_data_v3")

SOURCE_NAMES = (
    "mathwriting",
    "crohme",
    "iamondo",
    "hwrt",
    "detexify",
    "mw_symbols",
    "mw_synthetic",
    "composed",
)
SOURCE_TO_ID = {name: i for i, name in enumerate(SOURCE_NAMES)}
V3_ALIAS_SOURCE_ID = 9
V3_COMPOSED_SOURCE_ID = 10
V3_SOURCE_NAMES = {V3_ALIAS_SOURCE_ID: "real_alias", V3_COMPOSED_SOURCE_ID: "composed_v3"}

# LaTeX commands that are complete math atoms when used as a single-token script.
_SCRIPT_ATOM_COMMANDS = frozenset(r"""
\alpha \beta \gamma \delta \epsilon \varepsilon \zeta \eta \theta \vartheta
\iota \kappa \lambda \mu \nu \xi \pi \rho \sigma \tau \upsilon \phi \varphi
\chi \psi \omega \Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi
\Omega \infty \partial \nabla \ell \prime \ast \circ \times \pm \mp \hbar
""".split())


def _can_brace_script(token: str) -> bool:
    return token in _SCRIPT_ATOM_COMMANDS or (len(token) == 1 and token.isascii() and (token.isalnum() or token in "+-*=!.,"))


def expand_single_token_braces(tokens: Sequence[int], tokenizer: InkForgeTokenizer, max_len: int) -> list[int]:
    """Expand bare script atoms while retaining the complete sequence and EOS."""
    ids = tokenizer.token_to_id
    room = max_len - len(tokens)
    if room < 2 or not {"^", "_", "{", "}"}.issubset(ids):
        return list(tokens)
    markers = (ids["^"], ids["_"])
    expanded: list[int] = []
    i = 0
    while i < len(tokens):
        tok = int(tokens[i])
        expanded.append(tok)
        if room >= 2 and tok in markers and i + 1 < len(tokens):
            atom = int(tokens[i + 1])
            if _can_brace_script(tokenizer.tokens[atom]):
                expanded.extend((ids["{"], atom, ids["}"]))
                room -= 2
                i += 1
        i += 1
    return expanded


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


# ---------------------------------------------------------------------------
# Targeted Weak-Structure 2D Composed Math Synthesizer (Strictly from Train Donors)
# ---------------------------------------------------------------------------

def _normalize_donor_glyph(strokes: Sequence[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
    """Shift a symbol donor so its bbox starts at (0, 0) and scale height to ~1.0 (preserving width ratio)."""
    pts = [np.asarray(s, dtype=np.float64)[:, :2] for s in strokes if len(s) > 0]
    all_xy = np.concatenate(pts, axis=0)
    x0, y0 = float(all_xy[:, 0].min()), float(all_xy[:, 1].min())
    x1, y1 = float(all_xy[:, 0].max()), float(all_xy[:, 1].max())
    w = max(x1 - x0, 1e-3)
    h = max(y1 - y0, 1e-3)
    scale = 1.0 / max(h, 0.35 * w)
    out = [np.stack([(s[:, 0] - x0) * scale, (s[:, 1] - y0) * scale], axis=-1) for s in pts]
    return out, w * scale, h * scale


def _place_glyph(
    glyph_strokes: Sequence[np.ndarray],
    x_offset: float,
    y_offset: float,
    scale: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    sx = scale * float(rng.uniform(0.92, 1.08))
    sy = scale * float(rng.uniform(0.92, 1.08))
    jitter_y = float(rng.uniform(-0.04, 0.04)) * scale
    return [
        np.stack([s[:, 0] * sx + x_offset, s[:, 1] * sy + y_offset + jitter_y], axis=-1)
        for s in glyph_strokes
    ]


def _draw_radical_sign(
    x_rad_left: float,
    x_rad_right: float,
    y_top: float,
    y_bot: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate a realistic single-stroke hand-drawn square-root radical wrapping `[x_rad_left, x_rad_right]`."""
    h = max(y_bot - y_top, 0.8)
    y_roof = y_top - float(rng.uniform(0.14, 0.28)) * h
    y_v = y_bot + float(rng.uniform(0.02, 0.12)) * h
    y_mid = y_top + 0.55 * h
    x_hook = x_rad_left - float(rng.uniform(0.55, 0.80)) * h
    x_v = x_rad_left - float(rng.uniform(0.28, 0.42)) * h
    x_corner = x_rad_left - float(rng.uniform(0.05, 0.14)) * h
    x_end = x_rad_right + float(rng.uniform(0.12, 0.30)) * h

    p0 = np.array([x_hook, y_mid])
    p1 = np.array([x_v, y_v])
    p2 = np.array([x_corner, y_roof])
    p3 = np.array([x_end, y_roof + float(rng.uniform(-0.05, 0.05)) * h])

    seg1 = np.linspace(p0, p1, 6)
    seg2 = np.linspace(p1, p2, 10)[1:]
    n_roof = max(8, min(28, int(round((x_end - x_corner) * 5.0))))
    seg3 = np.linspace(p2, p3, n_roof)[1:]
    pts = np.concatenate([seg1, seg2, seg3], axis=0)
    # Mild organic hand tremor
    ts = np.linspace(0.0, 1.0, len(pts))
    pts[:, 1] += 0.018 * h * np.sin(ts * math.pi * float(rng.uniform(2.0, 4.0)) + float(rng.uniform(0, 6.28)))
    return [pts]


def _draw_frac_bar(x0: float, x1: float, y: float, rng: np.random.Generator) -> list[np.ndarray]:
    n = max(6, min(24, int(round((x1 - x0) * 5.0))))
    xs = np.linspace(x0, x1, n)
    ys = np.full(n, y) + np.linspace(0.0, float(rng.uniform(-0.04, 0.04)), n)
    ys += rng.normal(0.0, 0.008, size=n)
    return [np.stack([xs, ys], axis=-1)]


def _draw_vec_arrow(x0: float, x1: float, y: float, rng: np.random.Generator) -> list[np.ndarray]:
    w = max(x1 - x0, 0.5)
    shaft = np.linspace([x0, y], [x1, y + float(rng.uniform(-0.03, 0.03))], 7)
    tip = shaft[-1]
    head = np.array([
        [tip[0] - 0.22 * w, tip[1] - 0.14],
        [tip[0], tip[1]],
        [tip[0] - 0.22 * w, tip[1] + 0.14],
    ])
    return [shaft, head]


def _layout_linear_tokens(
    tokens: Sequence[str],
    bank: dict[str, list[tuple[list[np.ndarray], float, float]]],
    rng: np.random.Generator,
    *,
    scale: float = 1.0,
) -> tuple[list[np.ndarray], float, float, float] | None:
    """Lay out a flat sequence of tokens (supporting single-token `^` and `_` scripts)."""
    strokes: list[np.ndarray] = []
    cursor_x = 0.0
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "<SPACE>":
            cursor_x += 0.48 * scale
            i += 1
            continue
        if tok in ("^", "_") and i + 1 < n:
            script_tok = tokens[i + 1]
            cands = bank.get(script_tok)
            if not cands:
                return None
            g_st, gw, gh = cands[int(rng.integers(0, len(cands)))]
            sc = scale * 0.62
            y_off = (-0.42 * scale) if tok == "^" else (0.58 * scale)
            strokes.extend(_place_glyph(g_st, cursor_x, y_off, sc, rng))
            cursor_x += gw * sc + 0.14 * scale
            i += 2
            continue
        cands = bank.get(tok)
        if not cands:
            return None
        g_st, gw, gh = cands[int(rng.integers(0, len(cands)))]
        # Vertically center short operators (+, -, =, \cdot, \rightarrow, comma)
        if tok in ("-", "=", "+", r"\cdot", r"\rightarrow", r"\leq", r"\geq", r"\in", r"\cap", r"\cup", ","):
            y_off = (0.5 * (1.0 - gh) + (0.28 if tok == "," else 0.0)) * scale
        else:
            y_off = (1.0 - gh) * 0.5 * scale
        strokes.extend(_place_glyph(g_st, cursor_x, y_off, scale, rng))
        cursor_x += gw * scale + float(rng.uniform(0.14, 0.24)) * scale
        i += 1
    if not strokes:
        return None
    all_xy = np.concatenate(strokes, axis=0)
    return strokes, float(all_xy[:, 0].max()), float(all_xy[:, 1].min()), float(all_xy[:, 1].max())


def _generate_composed_weak_structures(
    symbol_bank: dict[str, list[tuple[list[np.ndarray], float, float]]],
    tokenizer: InkForgeTokenizer,
    *,
    count: int = 35000,
    spacing: float = 0.12,
    max_points: int = 512,
    max_text_len: int = 160,
    seed: int = 1337,
) -> list[tuple[list[int], str, int, np.ndarray, np.ndarray]]:
    """Synthesize clean 2D training examples targeting InkForge v1's weak structures (strictly train donors)."""
    rng = np.random.default_rng(seed)
    vars_pool = ["x", "y", "z", "t", "n", "k", "a", "b", "c", "r", "u", "v", "p", "q"]
    digits_pool = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ops_pool = ["+", "-"]
    greek_pool = [r"\alpha", r"\beta", r"\gamma", r"\theta", r"\lambda", r"\mu", r"\sigma", r"\omega", r"\phi"]
    words_pool = [
        "otherwise", "if", "where", "for", "all", "and", "or", "when", "then", "else", "true", "false",
    ]
    sid = SOURCE_TO_ID["composed"]
    out: list[tuple[list[int], str, int, np.ndarray, np.ndarray]] = []
    attempts = 0
    max_attempts = count * 5

    def _emit(latex: str, strokes: list[np.ndarray]) -> bool:
        toks = tokenize_inkforge(latex)
        if not toks or len(toks) + 2 > max_text_len:
            return False
        ids = tokenizer.encode(latex, max_len=max_text_len)
        if tokenizer.unk_id in ids:
            return False
        norm = normalize_and_resample_for_inkforge(strokes, spacing=spacing, max_points=max_points)
        if norm is None:
            return False
        xy, pen = norm
        out.append((ids, "", sid, xy.astype(np.float16), pen))
        return True

    while len(out) < count and attempts < max_attempts:
        attempts += 1
        u = float(rng.random())
        try:
            if u < 0.36:
                # Category 1: Radicals \sqrt{...} with varied algebraic radicands (fixes v1's #1 failure!)
                kind = int(rng.integers(0, 5))
                v1 = vars_pool[int(rng.integers(0, len(vars_pool)))]
                v2 = vars_pool[int(rng.integers(0, len(vars_pool)))]
                v3 = vars_pool[int(rng.integers(0, len(vars_pool)))]
                d1 = digits_pool[int(rng.integers(0, len(digits_pool)))]
                op = ops_pool[int(rng.integers(0, 2))]
                if kind == 0:
                    rad_tex = f"{v1}^2 {op} {d1}{v2}{v3}"
                    rad_toks = [v1, "^", "2", op, d1, v2, v3]
                elif kind == 1:
                    rad_tex = f"{v1}^2 {op} {v2}^2"
                    rad_toks = [v1, "^", "2", op, v2, "^", "2"]
                elif kind == 2:
                    rad_tex = f"{d1} {op} {v1}^2"
                    rad_toks = [d1, op, v1, "^", "2"]
                elif kind == 3:
                    rad_tex = f"{v1} {op} {v2}"
                    rad_toks = [v1, op, v2]
                else:
                    rad_tex = f"{d1}{v1} {op} {digits_pool[int(rng.integers(0, 9))]}"
                    rad_toks = [d1, v1, op, rad_tex.split()[-1]]

                res = _layout_linear_tokens(rad_toks, symbol_bank, rng, scale=1.0)
                if res is None:
                    continue
                rad_st, rx1, ry0, ry1 = res
                # Shift radicand right to make room for radical hook
                shift_x = 0.85
                rad_shifted = [np.stack([s[:, 0] + shift_x, s[:, 1]], axis=-1) for s in rad_st]
                rad_sign = _draw_radical_sign(shift_x, rx1 + shift_x, ry0, ry1, rng)
                # Optionally add prefix/suffix (e.g. y = \sqrt{...} or \sqrt{...} = c)
                if float(rng.random()) < 0.35:
                    lhs_v = vars_pool[int(rng.integers(0, len(vars_pool)))]
                    lhs = _layout_linear_tokens([lhs_v, "="], symbol_bank, rng, scale=1.0)
                    if lhs is None:
                        continue
                    lhs_st, lx1, _, _ = lhs
                    dx_r = lx1 + 0.35
                    all_st = lhs_st + [np.stack([s[:, 0] + dx_r, s[:, 1]], axis=-1) for s in (rad_sign + rad_shifted)]
                    _emit(f"{lhs_v} = \\sqrt{{{rad_tex}}}", all_st)
                else:
                    _emit(f"\\sqrt{{{rad_tex}}}", rad_sign + rad_shifted)

            elif u < 0.62:
                # Category 2: Sums / Products / Limits with stacked sub/sup bounds + fraction
                sub_kind = int(rng.integers(0, 3))
                if sub_kind == 0:
                    # \sum_{i=1}^n expr  or  \prod_{k=1}^n expr
                    big_op = r"\sum" if float(rng.random()) < 0.78 else r"\prod"
                    idx_v = "i" if float(rng.random()) < 0.6 else "k"
                    low_v = "1" if float(rng.random()) < 0.65 else "0"
                    hi_v = "n" if float(rng.random()) < 0.7 else r"\infty"
                    body_v = vars_pool[int(rng.integers(0, 8))]
                    op_cands = symbol_bank.get(big_op)
                    if not op_cands:
                        continue
                    g_op, op_w, op_h = op_cands[int(rng.integers(0, len(op_cands)))]
                    big_st = _place_glyph(g_op, 0.0, -0.25, 1.55, rng)
                    low_res = _layout_linear_tokens([idx_v, "=", low_v], symbol_bank, rng, scale=0.58)
                    hi_res = _layout_linear_tokens([hi_v], symbol_bank, rng, scale=0.62)
                    body_res = _layout_linear_tokens([body_v, "_", idx_v], symbol_bank, rng, scale=1.0)
                    if low_res is None or hi_res is None or body_res is None:
                        continue
                    low_st, lw, _, _ = low_res
                    hi_st, hw, _, _ = hi_res
                    body_st, _, _, _ = body_res
                    cx = 0.5 * op_w * 1.55
                    low_placed = [np.stack([s[:, 0] + cx - 0.5 * lw, s[:, 1] + 1.45], axis=-1) for s in low_st]
                    hi_placed = [np.stack([s[:, 0] + cx - 0.5 * hw, s[:, 1] - 0.95], axis=-1) for s in hi_st]
                    bx = max(op_w * 1.55, cx + 0.5 * lw) + 0.35
                    body_placed = [np.stack([s[:, 0] + bx, s[:, 1] + 0.05], axis=-1) for s in body_st]
                    _emit(f"{big_op}_{{{idx_v}={low_v}}}^{{{hi_v}}} {body_v}_{idx_v}", big_st + low_placed + hi_placed + body_placed)
                elif sub_kind == 1:
                    # \lim_{x \rightarrow 0} \frac{num}{den} = val
                    v = "x" if float(rng.random()) < 0.7 else "n"
                    tgt = "0" if v == "x" else r"\infty"
                    lim_res = _layout_linear_tokens(["l", "i", "m"], symbol_bank, rng, scale=0.95)
                    sub_res = _layout_linear_tokens([v, r"\rightarrow", tgt], symbol_bank, rng, scale=0.58)
                    num_toks = ["s", "i", "n", v] if float(rng.random()) < 0.5 else ["1"]
                    num_res = _layout_linear_tokens(num_toks, symbol_bank, rng, scale=0.88)
                    den_res = _layout_linear_tokens([v], symbol_bank, rng, scale=0.88)
                    val_tok = "1" if num_toks[0] == "s" else "0"
                    eq_res = _layout_linear_tokens(["=", val_tok], symbol_bank, rng, scale=1.0)
                    if None in (lim_res, sub_res, num_res, den_res, eq_res):
                        continue
                    lim_st, lim_w, _, _ = lim_res
                    sub_st, sub_w, _, _ = sub_res
                    num_st, nw, _, _ = num_res
                    den_st, dw, _, _ = den_res
                    eq_st, _, _, _ = eq_res
                    sub_placed = [np.stack([s[:, 0] + 0.5 * (lim_w - sub_w), s[:, 1] + 1.05], axis=-1) for s in sub_st]
                    fx = max(lim_w, 0.5 * (lim_w + sub_w)) + 0.42
                    fw = max(nw, dw) + 0.30
                    num_p = [np.stack([s[:, 0] + fx + 0.5 * (fw - nw), s[:, 1] - 0.62], axis=-1) for s in num_st]
                    bar_p = _draw_frac_bar(fx, fx + fw, 0.42, rng)
                    den_p = [np.stack([s[:, 0] + fx + 0.5 * (fw - dw), s[:, 1] + 0.62], axis=-1) for s in den_st]
                    eq_p = [np.stack([s[:, 0] + fx + fw + 0.30, s[:, 1]], axis=-1) for s in eq_st]
                    num_tex = f"\\sin {v}" if num_toks[0] == "s" else "1"
                    _emit(
                        f"\\lim_{{{v} \\rightarrow {tgt}}} \\frac{{{num_tex}}}{{{v}}} = {val_tok}",
                        lim_st + sub_placed + num_p + bar_p + den_p + eq_p,
                    )
                else:
                    # \frac{n(n+1)}{2} style fraction
                    v = vars_pool[int(rng.integers(0, 6))]
                    d1 = digits_pool[int(rng.integers(0, 5))]
                    d2 = digits_pool[int(rng.integers(1, 5))]
                    num_res = _layout_linear_tokens([v, "(", v, "+", d1, ")"], symbol_bank, rng, scale=0.88)
                    den_res = _layout_linear_tokens([d2], symbol_bank, rng, scale=0.88)
                    if num_res is None or den_res is None:
                        continue
                    num_st, nw, _, _ = num_res
                    den_st, dw, _, _ = den_res
                    fw = max(nw, dw) + 0.30
                    num_p = [np.stack([s[:, 0] + 0.5 * (fw - nw), s[:, 1] - 0.62], axis=-1) for s in num_st]
                    bar_p = _draw_frac_bar(0.0, fw, 0.42, rng)
                    den_p = [np.stack([s[:, 0] + 0.5 * (fw - dw), s[:, 1] + 0.62], axis=-1) for s in den_st]
                    _emit(f"\\frac{{{v}({v}+{d1})}}{{{d2}}}", num_p + bar_p + den_p)

            elif u < 0.85:
                # Category 3: \Pr(...), \forall x \in \mathbb{R}, \Delta x = \lambda \mu, \sin^2\theta+\cos^2\theta=1, \vec{u}\cdot\vec{v}=0
                k3 = int(rng.integers(0, 5))
                if k3 == 0:
                    a_c, b_c = ("A", "B") if float(rng.random()) < 0.7 else ("X", "Y")
                    op_set = r"\cap" if float(rng.random()) < 0.6 else r"\cup"
                    toks = ["P", "r", "(", a_c, op_set, b_c, ")", "=", "P", "r", "(", a_c, ")", "P", "r", "(", b_c, ")"]
                    res = _layout_linear_tokens(toks, symbol_bank, rng, scale=1.0)
                    if res is not None:
                        _emit(f"\\Pr({a_c} {op_set} {b_c}) = \\Pr({a_c})\\Pr({b_c})", res[0])
                elif k3 == 1:
                    g1 = greek_pool[int(rng.integers(0, len(greek_pool)))]
                    g2 = greek_pool[int(rng.integers(0, len(greek_pool)))]
                    v = vars_pool[int(rng.integers(0, 5))]
                    res = _layout_linear_tokens([r"\Delta", v, "=", g1, g2], symbol_bank, rng, scale=1.0)
                    if res is not None:
                        _emit(f"\\Delta {v} = {g1} {g2}", res[0])
                elif k3 == 2:
                    v = vars_pool[int(rng.integers(0, 4))]
                    rel = r"\geq" if float(rng.random()) < 0.6 else r"\leq"
                    res = _layout_linear_tokens([r"\forall", v, r"\in", "R", ",", v, "^", "2", rel, "0"], symbol_bank, rng, scale=1.0)
                    if res is not None:
                        _emit(f"\\forall {v} \\in \\mathbb{{R}}, {v}^2 {rel} 0", res[0])
                elif k3 == 3:
                    g = r"\theta" if float(rng.random()) < 0.65 else "x"
                    toks = ["s", "i", "n", "^", "2", g, "+", "c", "o", "s", "^", "2", g, "=", "1"]
                    res = _layout_linear_tokens(toks, symbol_bank, rng, scale=1.0)
                    if res is not None:
                        _emit(f"\\sin^2 {g} + \\cos^2 {g} = 1", res[0])
                else:
                    v1, v2 = ("u", "v") if float(rng.random()) < 0.6 else ("a", "b")
                    r1 = _layout_linear_tokens([v1], symbol_bank, rng, scale=1.0)
                    rd = _layout_linear_tokens([r"\cdot"], symbol_bank, rng, scale=1.0)
                    r2 = _layout_linear_tokens([v2], symbol_bank, rng, scale=1.0)
                    re_ = _layout_linear_tokens(["=", "0"], symbol_bank, rng, scale=1.0)
                    if None in (r1, rd, r2, re_):
                        continue
                    s1, w1, y1t, _ = r1
                    sd, wd, _, _ = rd
                    s2, w2, y2t, _ = r2
                    se, _, _, _ = re_
                    a1 = _draw_vec_arrow(0.0, max(w1, 0.55), y1t - 0.22, rng)
                    x_d = w1 + 0.28
                    sd_p = [np.stack([s[:, 0] + x_d, s[:, 1]], axis=-1) for s in sd]
                    x_2 = x_d + wd + 0.28
                    s2_p = [np.stack([s[:, 0] + x_2, s[:, 1]], axis=-1) for s in s2]
                    a2 = _draw_vec_arrow(x_2, x_2 + max(w2, 0.55), y2t - 0.22, rng)
                    x_e = x_2 + w2 + 0.30
                    se_p = [np.stack([s[:, 0] + x_e, s[:, 1]], axis=-1) for s in se]
                    _emit(f"\\vec{{{v1}}} \\cdot \\vec{{{v2}}} = 0", s1 + a1 + sd_p + s2_p + a2 + se_p)

            else:
                # Category 4: Clean \text{...} English words & piecewise conditions
                w = words_pool[int(rng.integers(0, len(words_pool)))]
                if w == "if" and float(rng.random()) < 0.7:
                    v = vars_pool[int(rng.integers(0, 4))]
                    rel = ">" if float(rng.random()) < 0.5 else r"\geq"
                    res = _layout_linear_tokens(["i", "f", "<SPACE>", v, rel, "0"], symbol_bank, rng, scale=1.0)
                    if res is not None:
                        _emit(f"\\text{{if }} {v} {rel} 0", res[0])
                else:
                    res = _layout_linear_tokens(list(w), symbol_bank, rng, scale=1.0)
                    if res is not None:
                        _emit(f"\\text{{{w}}}", res[0])
        except Exception:
            continue
    return out


def prepare_inkforge_v2_cache(
    base_cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_V2_CACHE_DIR,
    *,
    mw_synthetic_dir: Path = DEFAULT_MATHWRITING_SYNTHETIC,
    composed_count: int = 35000,
    spacing: float = 0.12,
    max_points: int = 512,
    max_text_len: int = 160,
    workers: int = 12,
    seed: int = 42,
) -> Path:
    """Build InkForge v2 cache: reuses leak-free real-human base cache + adds MathWriting synthetic (396k)
    + targeted 2D composed weak-structure math strictly to TRAIN (keeping VAL 100% real-human & identical!).
    """
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = InkForgeTokenizer.load(base_cache_dir / "tokenizer.json")
    tokenizer.save(output_dir / "tokenizer.json")

    def _load_split_records(sp_dir: Path) -> list[tuple[list[int], str, int, np.ndarray, np.ndarray]]:
        pts_xy = np.load(sp_dir / "points_xy.npy")
        pts_pen = np.load(sp_dir / "points_pen.npy")
        pt_off = np.load(sp_dir / "pt_offsets.npy")
        toks = np.load(sp_dir / "tokens.npy")
        tok_off = np.load(sp_dir / "tok_offsets.npy")
        sids = np.load(sp_dir / "source_ids.npy")
        wids = np.load(sp_dir / "writer_ids.npy")
        wnames = json.loads((sp_dir / "writer_names.json").read_text(encoding="utf-8"))
        recs: list[tuple[list[int], str, int, np.ndarray, np.ndarray]] = []
        for i in range(len(sids)):
            p0, p1 = int(pt_off[i]), int(pt_off[i + 1])
            t0_i, t1_i = int(tok_off[i]), int(tok_off[i + 1])
            wkey = wnames[int(wids[i])] if int(wids[i]) < len(wnames) else ""
            recs.append((toks[t0_i:t1_i].tolist(), wkey, int(sids[i]), pts_xy[p0:p1].copy(), pts_pen[p0:p1].copy()))
        return recs

    train_records = _load_split_records(base_cache_dir / "train")
    val_records = _load_split_records(base_cache_dir / "val")
    val_digests = {_packed_record_digest(r[0], r[3], r[4]) for r in val_records}
    print(
        f"[v2-pack] loaded base cache: train={len(train_records):,}, val={len(val_records):,} "
        f"({time.perf_counter()-t0:.1f}s)",
        flush=True,
    )

    # 1. Build train-only real-human symbol donor bank for targeted 2D composition (zero val leakage!)
    symbol_bank: dict[str, list[tuple[list[np.ndarray], float, float]]] = defaultdict(list)
    sym_sids = {SOURCE_TO_ID["hwrt"], SOURCE_TO_ID["detexify"], SOURCE_TO_ID["mw_symbols"]}
    for ids, _wkey, sid, xy16, pen in train_records:
        if sid not in sym_sids or len(ids) != 3:  # [BOS, tok, EOS]
            continue
        tok_str = tokenizer.tokens[ids[1]]
        if len(symbol_bank[tok_str]) >= 80:
            continue
        sts = unpack_points_to_strokes(xy16.astype(np.float64), pen)
        if sts:
            symbol_bank[tok_str].append(_normalize_donor_glyph(sts))
    print(f"[v2-pack] built train-only symbol donor bank: {len(symbol_bank)} unique symbols", flush=True)

    # 2. Parse Google MathWriting official synthetic/ split (~396k full expressions -> TRAIN ONLY!)
    mw_syn_added = 0
    if mw_synthetic_dir.is_dir():
        syn_paths = [str(p) for p in sorted(mw_synthetic_dir.glob("*.inkml"))]
        print(f"[v2-pack] parsing {len(syn_paths):,} MathWriting synthetic .inkml files ({workers} workers)...", flush=True)
        chunk_size = 2048
        batches = [
            (syn_paths[i : i + chunk_size], spacing, max_points)
            for i in range(0, len(syn_paths), chunk_size)
        ]
        sid_syn = SOURCE_TO_ID["mw_synthetic"]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for b_i, res_list in enumerate(ex.map(_process_mw_batch, batches)):
                for latex, _wid, xy16, pen in res_list:
                    toks = tokenize_inkforge(latex)
                    if not toks or len(toks) + 2 > max_text_len:
                        continue
                    ids = tokenizer.encode(latex, max_len=max_text_len)
                    if tokenizer.unk_id in ids:
                        continue
                    if _packed_record_digest(ids, xy16, pen) in val_digests:
                        continue
                    train_records.append((ids, "", sid_syn, xy16, pen))
                    mw_syn_added += 1
                if (b_i + 1) % 40 == 0:
                    print(
                        f"  [mw_synthetic] batch {b_i+1}/{len(batches)} | kept={mw_syn_added:,} "
                        f"({time.perf_counter()-t0:.1f}s)",
                        flush=True,
                    )
        print(f"[v2-pack] added mw_synthetic: {mw_syn_added:,} samples ({time.perf_counter()-t0:.1f}s)", flush=True)

    # 3. Synthesize targeted weak-structure 2D composed expressions strictly from train donors
    composed_recs = _generate_composed_weak_structures(
        symbol_bank,
        tokenizer,
        count=composed_count,
        spacing=spacing,
        max_points=max_points,
        max_text_len=max_text_len,
        seed=seed,
    )
    composed_added = 0
    for rec in composed_recs:
        if _packed_record_digest(rec[0], rec[3], rec[4]) not in val_digests:
            train_records.append(rec)
            composed_added += 1
    print(f"[v2-pack] added composed weak-structures: {composed_added:,} samples ({time.perf_counter()-t0:.1f}s)", flush=True)

    rng = random.Random(seed)
    rng.shuffle(train_records)

    # Count per source for transparency
    src_counts = Counter(SOURCE_NAMES[r[2]] for r in train_records)
    _write_packed_split(output_dir / "train", train_records)
    _write_packed_split(output_dir / "val", val_records)

    meta = {
        "format": "inkforge-cache-v2",
        "vocab_size": len(tokenizer),
        "spacing": spacing,
        "max_points": max_points,
        "max_text_len": max_text_len,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "sources": list(SOURCE_NAMES),
        "train_source_counts": dict(src_counts),
        "leak_free_grouped": True,
        "val_human_only": True,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[v2-pack] DONE in {meta['elapsed_seconds']}s -> {output_dir} | sources={dict(src_counts)}", flush=True)
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
        brace_aug_prob: float = 0.0,
        stroke_order_aug_prob: float = 0.0,
        stroke_jitter_std: float = 0.0,
    ):
        self.split_dir = Path(cache_dir) / split
        self.cfg = cfg or InkForgeConfig()
        self.augment = augment
        self.input_noise_std = input_noise_std
        if not (0.0 <= brace_aug_prob <= 1.0 and 0.0 <= stroke_order_aug_prob <= 1.0):
            raise ValueError("Augmentation probabilities must be in [0, 1]")
        if not math.isfinite(stroke_jitter_std) or stroke_jitter_std < 0.0:
            raise ValueError("stroke_jitter_std must be finite and nonnegative")
        self.brace_aug_prob = brace_aug_prob
        self.stroke_order_aug_prob = stroke_order_aug_prob
        self.stroke_jitter_std = stroke_jitter_std
        self.brace_tokenizer = InkForgeTokenizer.load(Path(cache_dir) / "tokenizer.json") if augment and brace_aug_prob else None

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
        for idx, (wid, sid) in enumerate(zip(self.writer_ids.tolist(), self.source_ids.tolist())):
            if wid > 0 and sid != V3_ALIAS_SOURCE_ID:
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
                "hwrt": 0.18,
                "detexify": 0.22,
                "mw_symbols": 0.25,
                "mw_synthetic": 0.45,
                "composed": 0.85,
                "real_alias": 1.0,
                "composed_v3": 0.70,
            }
        source_names = dict(enumerate(SOURCE_NAMES)) | V3_SOURCE_NAMES
        w_table = np.ones(max(source_names) + 1, dtype=np.float64)
        for sid, name in source_names.items():
            w_table[sid] = source_weights.get(name, 1.0)
        self.sample_weights = np.ones(self.n_samples, dtype=np.float64)
        mapped = (self.source_ids >= 0) & (self.source_ids < len(w_table))
        self.sample_weights[mapped] = w_table[self.source_ids[mapped]]

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
        if self.augment and self.brace_tokenizer is not None and rng.random() < self.brace_aug_prob:
            tok_ids = np.asarray(expand_single_token_braces(tok_ids, self.brace_tokenizer, self.cfg.max_text_len), dtype=np.int64)

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
            if self.stroke_order_aug_prob or self.stroke_jitter_std:
                bounds = np.flatnonzero(pen[:-1] == 1) + 1
                if len(bounds):
                    strokes = np.split(xy, bounds)
                    if self.stroke_order_aug_prob and rng.random() < self.stroke_order_aug_prob:
                        # Swap nearby consecutive strokes, retaining point order within each stroke.
                        pairs = [
                            j for j, (left, right) in enumerate(zip(strokes, strokes[1:]))
                            if (j > 0 or len(right) > 1)
                            and np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)) < 1.0
                        ]
                        if pairs:
                            j = int(rng.choice(pairs))
                            strokes[j], strokes[j + 1] = strokes[j + 1], strokes[j]
                            pen.fill(0)
                            pen[np.cumsum([len(stroke) for stroke in strokes]) - 1] = 1
                    if self.stroke_jitter_std:
                        shifts = rng.normal(0.0, self.stroke_jitter_std, size=(len(strokes), 2)).astype(np.float32)
                        strokes = [stroke + shift for stroke, shift in zip(strokes, shifts)]
                    xy = np.concatenate(strokes)
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
        style_strokes: list[np.ndarray] = []
        peers = self.writer_peers.get(wid)
        is_alias = self.source_ids[idx] == V3_ALIAS_SOURCE_ID
        if peers is not None:
            start = int(rng.integers(0, len(peers)))
            for offset in range(len(peers)):
                peer_idx = int(peers[(start + offset) % len(peers)])
                if peer_idx == idx:
                    continue
                if is_alias:
                    q0, q1 = self.pt_offsets[peer_idx:peer_idx + 2]
                    if (p1 - p0 == q1 - q0 and np.array_equal(self.points_xy[p0:p1], self.points_xy[q0:q1])
                            and np.array_equal(self.points_pen[p0:p1], self.points_pen[q0:q1])):
                        continue
                style_strokes = self._get_raw_strokes(peer_idx)
                break

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
    parser.add_argument("--build-v2", action="store_true", help="Build v2 cache: base real human + MathWriting synthetic (396k) + targeted composed math")
    parser.add_argument("--base-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--composed-count", type=int, default=35000)
    args = parser.parse_args()
    if args.build_v2:
        out_dir = DEFAULT_V2_CACHE_DIR if args.output_dir == DEFAULT_CACHE_DIR else args.output_dir
        prepare_inkforge_v2_cache(
            base_cache_dir=args.base_cache_dir,
            output_dir=out_dir,
            composed_count=args.composed_count,
            spacing=args.spacing,
            max_points=args.max_points,
            workers=args.workers,
        )
    elif args.resplit_only:
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
