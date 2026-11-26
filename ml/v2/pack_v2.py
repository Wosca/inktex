"""InkTeX v2 Audited Zero-Leakage Clean Shard Packer (`inktex-shards-v2`).

Fixes every P0/P1 audit finding from GPT-6.2-Sol & Claude Opus 5.5:
  1. Separate deny indices:
       - `train_deny`: CROHME test (3,499 + 192 writers), CROHME validation (1,702),
                       MathWriting test (7,644), MathWriting valid (15,674),
                       HME100K validation (7,149).
       - `val_crohme_deny`: CROHME test (IDs + 192 writers + ink/dHash) + MW test/valid + HME100K val.
       - `val_mw_deny`:     CROHME test/val + MW test + HME100K val.
  2. Fail-closed deny index:
       - Indexes test IDs and dataset-namespaced writer IDs (`crohme:<id>`) BEFORE parsing/rendering.
       - Stores ALL owners per sparse dHash entry and checks ALL matches within Hamming <= 2.
       - Asserts every required holdout source has > 0 indexed rows (and 192 CROHME test writers).
  3. Tokenizer v2 built ONLY from human training rows that already passed `train_deny`.
  4. UniMER requires explicit `split == "train"` + non-handwritten provenance + width-constrained
     squish filter (no false drops of short single-line 38px formulas).
  5. Validation splits skip stroke-ratio filtering and report exact OOV token coverage.
  6. Bounded worker submission + removes stale shard-*.tar before writing each split.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Sequence

import numpy as np

from data.bimodal_raster import compute_dhash64, encode_png
from data.bimodal_shards import ShardWriter, sha256_file
from data.canonical import expression_from_record
from data.importers.mathwriting import MathWritingImporter
from data.schema import StrokeExpression
from tokenizer.latex_tokenizer import tokenize_latex

from .shards_v2 import FORMAT_VERSION_V2, pack_raw_strokes_npz
from .tokenizer_v2 import (
    MAX_BODY_TOKENS_V2,
    LatexTokenizerV2,
    is_plausible_stroke_ratio,
    standardize_latex_v2,
)
from .vision_raster import (
    STRIP_HEIGHT_V2,
    normalize_offline_image_to_strip_v2,
    render_strokes_to_strip_v2,
)

DATASETS_ROOT = Path(r"E:\math-recogniser-datasets")
DEFAULT_CROHME = DATASETS_ROOT / "canonical" / "crohme23"
DEFAULT_HWRT = DATASETS_ROOT / "canonical" / "hwrt"
DEFAULT_HME = DATASETS_ROOT / "offline" / "hme100k"
DEFAULT_UNIMER = DATASETS_ROOT / "external" / "unimer"
DEFAULT_MATHWRITING = Path(r"E:\mathwriting-2024")
DEFAULT_OUTPUT = Path(r"E:\math-recogniser-training\v2-shards")

SHARD_SIZE_ONLINE = 4096
SHARD_SIZE_OFFLINE = 4096
VAL_MW_QUOTA = 1024
SUBMIT_BATCH = 4096


def namespace_writer(dataset_ns: str, writer_id: str | None) -> str | None:
    if writer_id in (None, ""):
        return None
    w = str(writer_id)
    return w if ":" in w else f"{dataset_ns}:{w}"


def expression_strokes_and_times(
    expression: StrokeExpression,
) -> tuple[list[np.ndarray], list[np.ndarray | None] | None]:
    strokes: list[np.ndarray] = []
    times: list[np.ndarray | None] = []
    has_any_t = bool(expression.metadata.get("time_available", True))
    for st in expression.strokes:
        if not st.points:
            continue
        xy = np.asarray([(p.x, p.y) for p in st.points], dtype=np.float64)
        strokes.append(xy)
        if has_any_t and all(p.time is not None for p in st.points):
            times.append(np.asarray([float(p.time) for p in st.points], dtype=np.float64))
        else:
            has_any_t = False
    return strokes, (times if has_any_t and len(times) == len(strokes) else None)


def ink_geometry_hash(strokes: Sequence[np.ndarray], grid: int = 32, samples: int = 8) -> str:
    clean = [np.asarray(s, dtype=np.float64)[:, :2] for s in strokes if len(s)]
    if not clean:
        return ""
    xy = np.concatenate(clean, axis=0)
    low = xy.min(axis=0)
    scale = max(float((xy.max(axis=0) - low).max()), 1e-9)
    digest = hashlib.sha1(str(len(clean)).encode())
    for st in clean:
        chosen = np.unique(np.rint(np.linspace(0, len(st) - 1, min(len(st), samples))).astype(int))
        digest.update(np.rint((st[chosen] - low) / scale * grid).astype(np.int16).tobytes())
        digest.update(b"|")
    return digest.hexdigest()[:20]


def _low_info_dhash(val: int) -> bool:
    return not (8 <= val.bit_count() <= 56)


class LeakageDenyIndexV2:
    """Fail-closed 4-way label-independent held-out deny index (ID, namespaced writer,
    ink geometry hash, multi-owner dHash-64)."""

    def __init__(self, reserved_writers: Iterable[str] = (), near_distance: int = 2):
        self.test_ids: set[str] = set()
        self.reserved_writers: set[str] = {str(w) for w in reserved_writers if w not in (None, "")}
        self.near_distance = int(near_distance)
        self.ink: dict[str, list[tuple[str, str]]] = {}
        # Multi-owner map: retains EVERY owner of a dHash so sparse-hash collisions never mask a match!
        self.dhash_owners: dict[int, list[tuple[str, str]]] = {}
        self.buckets: dict[tuple[int, int], list[int]] = {}
        self.counts: Counter[str] = Counter()

    def add(
        self,
        *,
        test_id: str,
        source: str,
        normalized_latex: str,
        dhash: str | None,
        ink_hash: str | None = None,
        writer_id: str | None = None,
    ) -> None:
        self.test_ids.add(test_id)
        self.counts[source] += 1
        if writer_id not in (None, ""):
            self.reserved_writers.add(str(writer_id))
        if ink_hash:
            self.ink.setdefault(ink_hash, []).append((test_id, normalized_latex))
        if dhash:
            val = int(dhash, 16)
            if val not in self.dhash_owners:
                self.dhash_owners[val] = [(test_id, normalized_latex)]
                for chunk in range(4):
                    self.buckets.setdefault((chunk, (val >> (16 * chunk)) & 0xFFFF), []).append(val)
            else:
                self.dhash_owners[val].append((test_id, normalized_latex))

    def check(
        self,
        *,
        sample_id: str,
        normalized_latex: str,
        dhash: str | None,
        writer_id: str | None = None,
        ink_hash: str | None = None,
    ) -> tuple[str, str] | None:
        if sample_id in self.test_ids:
            return "test_id", sample_id
        if writer_id not in (None, "") and str(writer_id) in self.reserved_writers:
            return "reserved_writer", str(writer_id)
        if ink_hash and ink_hash in self.ink:
            owners = self.ink[ink_hash]
            for tid, lab in owners:
                if lab == normalized_latex:
                    return "latex_ink_hash", tid
            return "ink_hash", owners[0][0]
        if dhash:
            val = int(dhash, 16)
            sparse = _low_info_dhash(val)
            seen_vals: set[int] = set()
            best_match: tuple[int, str] | None = None
            for chunk in range(4):
                for other in self.buckets.get((chunk, (val >> (16 * chunk)) & 0xFFFF), ()):
                    if other in seen_vals:
                        continue
                    seen_vals.add(other)
                    dist = (val ^ other).bit_count()
                    if dist > self.near_distance:
                        continue
                    owners = self.dhash_owners[other]
                    if sparse:
                        # Check EVERY owner of this sparse hash for label agreement
                        for tid, lab in owners:
                            if lab == normalized_latex:
                                if best_match is None or dist < best_match[0]:
                                    best_match = (dist, tid)
                    else:
                        if best_match is None or dist < best_match[0]:
                            best_match = (dist, owners[0][0])
            if best_match is not None:
                return ("dhash_exact" if best_match[0] == 0 else "dhash_near"), best_match[1]
        return None

    def to_summary(self) -> dict[str, Any]:
        return {
            "test_ids": len(self.test_ids),
            "reserved_writers": len(self.reserved_writers),
            "ink_hashes": len(self.ink),
            "dhashes": len(self.dhash_owners),
            "near_distance": self.near_distance,
            "counts": dict(self.counts),
        }


# --------------------------------------------------------------------------- Candidate Scanning


@dataclass(frozen=True, slots=True)
class CandidateV2:
    source: str
    dataset_ns: str  # crohme | mathwriting | hwrt | hme100k | unimer
    kind: str        # canonical | inkml | image
    path: str
    offset: int = -1
    id: str = ""
    latex: str = ""
    split: str = "train"
    writer_id: str | None = None


def scan_canonical(path: Path, source: str, dataset_ns: str, split: str) -> list[CandidateV2]:
    if not path.is_file():
        return []
    out: list[CandidateV2] = []
    with path.open("rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            w = namespace_writer(dataset_ns, rec.get("writerId"))
            out.append(CandidateV2(
                source=source,
                dataset_ns=dataset_ns,
                kind="canonical",
                path=str(path),
                offset=off,
                id=str(rec["id"]),
                latex=str(rec.get("latexNormalized") or rec.get("latex") or ""),
                split=split,
                writer_id=w,
            ))
    return out


def scan_mw_dir(directory: Path, source: str, split: str) -> list[CandidateV2]:
    if not directory.is_dir():
        return []
    with os.scandir(directory) as entries:
        names = sorted(e.name for e in entries if e.name.endswith(".inkml"))
    return [
        CandidateV2(
            source=source, dataset_ns="mathwriting", kind="inkml",
            path=str(directory / name), id=f"mathwriting:{name[:-6]}", split=split,
        )
        for name in names
    ]


def scan_image_jsonl(
    root: Path,
    filename: str,
    source: str,
    dataset_ns: str,
    allowed_splits: set[str],
) -> list[CandidateV2]:
    path = root / filename
    if not path.is_file():
        return []
    out: list[CandidateV2] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sp = str(rec.get("split", "")).lower()
            if sp not in allowed_splits:
                continue
            src_tag = str(rec.get("source", "")).lower()
            img_rel = str(rec.get("image", "")).lower()
            if source == "unimer_printed":
                if any(k in src_tag or k in img_rel for k in ("hme", "crohme", "hwe", "handwritten")):
                    continue
            w = namespace_writer(dataset_ns, rec.get("writer"))
            out.append(CandidateV2(
                source=source,
                dataset_ns=dataset_ns,
                kind="image",
                path=str(root / rec["image"]),
                id=str(rec["id"]),
                latex=str(rec.get("latex") or rec.get("original_latex") or ""),
                split=sp,
                writer_id=w,
            ))
    return out


def _read_jsonl_offset(path: str, offset: int) -> dict[str, Any]:
    with open(path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


def load_candidate_expression(cand: CandidateV2) -> StrokeExpression:
    if cand.kind == "canonical":
        return expression_from_record(_read_jsonl_offset(cand.path, cand.offset))
    if cand.kind == "inkml":
        return MathWritingImporter().parse(cand.path)
    raise ValueError(f"Unsupported ink kind: {cand.kind}")


# --------------------------------------------------------------------------- Worker Pipeline


@dataclass
class WorkerCtx:
    deny: LeakageDenyIndexV2
    tokenizer_tokens: frozenset[str] | None
    max_body_tokens: int
    unimer_max_body_tokens: int
    apply_stroke_ratio: bool


_WCTX: WorkerCtx | None = None


def _init_worker(ctx: WorkerCtx) -> None:
    global _WCTX
    _WCTX = ctx


def _index_one_holdout(cand: CandidateV2) -> dict[str, Any]:
    """Fail-closed holdout indexing: ALWAYS returns test_id & writer_id even if rendering fails."""
    base = {
        "test_id": cand.id,
        "source": cand.source,
        "normalized_latex": standardize_latex_v2(cand.latex) or cand.latex,
        "dhash": None,
        "ink_hash": None,
        "writer_id": cand.writer_id,
        "error": None,
    }
    try:
        if cand.kind == "image":
            strip = normalize_offline_image_to_strip_v2(Path(cand.path))
            base["dhash"] = compute_dhash64(strip.image)
            return base
        expr = load_candidate_expression(cand)
        raw = expr.latex or cand.latex
        base["normalized_latex"] = standardize_latex_v2(raw) or raw
        base["writer_id"] = namespace_writer(cand.dataset_ns, expr.writer_id) or cand.writer_id
        strokes, _ = expression_strokes_and_times(expr)
        if strokes:
            strip = render_strokes_to_strip_v2(strokes)
            base["dhash"] = compute_dhash64(strip.image)
            base["ink_hash"] = ink_geometry_hash(strokes)
        return base
    except Exception as exc:
        base["error"] = str(exc)[:120]
        return base


def _pack_one_candidate(cand: CandidateV2) -> tuple[str, Any]:
    assert _WCTX is not None
    ctx = _WCTX
    try:
        if cand.kind == "image":
            max_tok = min(ctx.max_body_tokens, ctx.unimer_max_body_tokens) if cand.source == "unimer_printed" else ctx.max_body_tokens
            std = standardize_latex_v2(cand.latex, max_body_tokens=max_tok)
            if std is None:
                return "drop", ("label_invalid", cand.source)
            if ctx.tokenizer_tokens is not None:
                if any(t not in ctx.tokenizer_tokens for t in tokenize_latex(std)):
                    return "drop", ("oov_token", cand.source)
            strip = normalize_offline_image_to_strip_v2(Path(cand.path))
            bbox = strip.image.getchannel("L").getbbox()
            if bbox is None:
                return "drop", ("blank_image", cand.source)
            # Filter squished UniMER by width-compression rather than fixed 48px threshold!
            if cand.source == "unimer_printed" and strip.width_constrained and (bbox[3] - bbox[1]) < 24:
                return "drop", ("squished_unimer", cand.source)
            dh = compute_dhash64(strip.image)
            leak = ctx.deny.check(
                sample_id=cand.id, normalized_latex=std, dhash=dh, writer_id=cand.writer_id,
            )
            if leak:
                return "drop", (f"leak:{leak[0]}", cand.source)
            meta = {
                "id": cand.id,
                "source": cand.source,
                "split": cand.split,
                "latex": cand.latex,
                "normalized_latex": std,
                "has_strokes": False,
                "has_timing": False,
                "image_width": int(strip.image_width),
                "bucket_width": int(strip.bucket_width),
                "stroke_count": 0,
                "writer_id": cand.writer_id,
                "synthetic": cand.source == "unimer_printed",
                "dhash": dh,
            }
            return "ok", (meta, encode_png(strip.image), None)

        expr = load_candidate_expression(cand)
        if cand.source == "mathwriting_train" and expr.synthetic:
            return "drop", ("not_human", cand.source)
        raw = expr.latex or cand.latex
        std = standardize_latex_v2(raw, max_body_tokens=ctx.max_body_tokens)
        if std is None:
            return "drop", ("label_invalid", cand.source)
        if ctx.tokenizer_tokens is not None:
            if any(t not in ctx.tokenizer_tokens for t in tokenize_latex(std)):
                return "drop", ("oov_token", cand.source)
        strokes, times = expression_strokes_and_times(expr)
        if not strokes:
            return "drop", ("empty_ink", cand.source)
        if ctx.apply_stroke_ratio and not is_plausible_stroke_ratio(std, len(strokes)):
            return "drop", ("implausible_stroke_ratio", cand.source)

        strip = render_strokes_to_strip_v2(strokes)
        dh = compute_dhash64(strip.image)
        ih = ink_geometry_hash(strokes)
        writer = namespace_writer(cand.dataset_ns, expr.writer_id) or cand.writer_id
        leak = ctx.deny.check(
            sample_id=cand.id, normalized_latex=std, dhash=dh, writer_id=writer, ink_hash=ih,
        )
        if leak:
            return "drop", (f"leak:{leak[0]}", cand.source)

        npz = pack_raw_strokes_npz(strokes, times)
        meta = {
            "id": cand.id,
            "source": cand.source,
            "split": cand.split,
            "latex": raw,
            "normalized_latex": std,
            "has_strokes": True,
            "has_timing": times is not None,
            "image_width": int(strip.image_width),
            "bucket_width": int(strip.bucket_width),
            "stroke_count": len(strokes),
            "writer_id": writer,
            "synthetic": cand.source in ("mathwriting_synthetic", "crohme_artificial"),
            "dhash": dh,
        }
        return "ok", (meta, encode_png(strip.image), npz)
    except Exception:
        return "drop", ("parse_error", cand.source)


def _admitted_vocab_label(cand: CandidateV2) -> str | None:
    """Return standardized label ONLY if human train candidate passes the FULL 4-way
    `train_deny` check (ID, namespaced writer, ink geometry hash, AND raster dHash)."""
    assert _WCTX is not None
    ctx = _WCTX
    try:
        if cand.kind == "image":
            std = standardize_latex_v2(cand.latex, max_body_tokens=ctx.max_body_tokens)
            if std is None:
                return None
            strip = normalize_offline_image_to_strip_v2(Path(cand.path))
            if strip.image.getchannel("L").getbbox() is None:
                return None
            dh = compute_dhash64(strip.image)
            if ctx.deny.check(sample_id=cand.id, normalized_latex=std, dhash=dh, writer_id=cand.writer_id):
                return None
            return std
        expr = load_candidate_expression(cand)
        if cand.source == "mathwriting_train" and expr.synthetic:
            return None
        raw = expr.latex or cand.latex
        std = standardize_latex_v2(raw, max_body_tokens=ctx.max_body_tokens)
        if std is None:
            return None
        strokes, _ = expression_strokes_and_times(expr)
        if not strokes or not is_plausible_stroke_ratio(std, len(strokes)):
            return None
        strip = render_strokes_to_strip_v2(strokes)
        dh = compute_dhash64(strip.image)
        ih = ink_geometry_hash(strokes)
        writer = namespace_writer(cand.dataset_ns, expr.writer_id) or cand.writer_id
        if ctx.deny.check(sample_id=cand.id, normalized_latex=std, dhash=dh, writer_id=writer, ink_hash=ih):
            return None
        return std
    except Exception:
        return None


# --------------------------------------------------------------------------- Main Orchestration


def index_holdout_entries(
    crohme_root: Path,
    mw_root: Path,
    hme_root: Path,
    *,
    workers: int,
    smoke: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Index every held-out split independently so train_deny, val_crohme_deny, and
    val_mw_deny each cover the exact right holdout union with zero blind spots."""
    groups: dict[str, list[CandidateV2]] = {
        "crohme_test": scan_canonical(crohme_root / "test.jsonl", "crohme_test", "crohme", "test"),
        "crohme_val": scan_canonical(crohme_root / "validation.jsonl", "crohme_val", "crohme", "val"),
        "mathwriting_test": scan_mw_dir(mw_root / "test", "mathwriting_test", "test"),
        "mathwriting_val": scan_mw_dir(mw_root / "valid", "mathwriting_val", "valid"),
        "hme100k_val": scan_image_jsonl(hme_root, "examples.jsonl", "hme100k_val", "hme100k", {"test", "val", "valid", "validation"}),
    }
    for name, items in groups.items():
        if not items:
            raise RuntimeError(f"Required holdout source {name!r} has 0 records! Refusing to pack fail-open.")
        if smoke:
            groups[name] = items[:64]

    flat = [c for items in groups.values() for c in items]
    print(f"[deny-index] Indexing {len(flat):,} held-out records across {len(groups)} splits ({workers} workers) ...", flush=True)
    t0 = time.perf_counter()
    results_by_source: dict[str, list[dict[str, Any]]] = {k: [] for k in groups}
    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(flat), SUBMIT_BATCH):
            chunk = flat[i : i + SUBMIT_BATCH]
            for entry in pool.map(_index_one_holdout, chunk, chunksize=64):
                results_by_source[entry["source"]].append(entry)
                if entry.get("error"):
                    errors += 1
    for name, items in groups.items():
        assert len(results_by_source[name]) == len(items), f"Indexed count mismatch for {name}"
    if not smoke:
        crohme_test_writers = {e["writer_id"] for e in results_by_source["crohme_test"] if e["writer_id"]}
        assert len(crohme_test_writers) == 192, f"Expected 192 CROHME test writers, got {len(crohme_test_writers)}"
    if errors > 0:
        sample_errs = [e["error"] for v in results_by_source.values() for e in v if e.get("error")][:5]
        raise RuntimeError(f"Holdout indexing hit {errors} errors (refusing fail-open): {sample_errs}")
    print(f"[deny-index] Done in {time.perf_counter() - t0:.1f}s | errors={errors} | counts={{k: len(v) for k, v in results_by_source.items()}}".replace("{k: len(v) for k, v in results_by_source.items()}", str({k: len(v) for k, v in results_by_source.items()})), flush=True)
    return results_by_source


def compose_deny_index(
    entries_by_source: dict[str, list[dict[str, Any]]],
    include_sources: Sequence[str],
    *,
    near_distance: int = 2,
) -> LeakageDenyIndexV2:
    deny = LeakageDenyIndexV2(near_distance=near_distance)
    for src in include_sources:
        for e in entries_by_source[src]:
            deny.add(
                test_id=e["test_id"],
                source=e["source"],
                normalized_latex=e["normalized_latex"],
                dhash=e["dhash"],
                ink_hash=e["ink_hash"],
                writer_id=e["writer_id"],
            )
    return deny


def build_v2_tokenizer(
    crohme_root: Path,
    mw_root: Path,
    hme_root: Path,
    train_deny: LeakageDenyIndexV2,
    *,
    workers: int,
    min_freq: int,
    smoke: bool,
) -> LatexTokenizerV2:
    cands: list[CandidateV2] = []
    cands.extend(scan_canonical(crohme_root / "train.jsonl", "crohme_train", "crohme", "train"))
    mw_tr = scan_mw_dir(mw_root / "train", "mathwriting_train", "train")
    hme_tr = scan_image_jsonl(hme_root, "examples.jsonl", "hme100k_train", "hme100k", {"train"})
    if smoke:
        mw_tr = mw_tr[:500]
        hme_tr = hme_tr[:500]
    cands.extend(mw_tr)
    cands.extend(hme_tr)

    ctx = WorkerCtx(
        deny=train_deny,
        tokenizer_tokens=None,
        max_body_tokens=MAX_BODY_TOKENS_V2,
        unimer_max_body_tokens=96,
        apply_stroke_ratio=False,
    )
    print(f"[tokenizer] Scanning {len(cands):,} deny-filtered human train labels ...", flush=True)
    t0 = time.perf_counter()
    labels: list[str] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        for i in range(0, len(cands), SUBMIT_BATCH * 4):
            batch = cands[i : i + SUBMIT_BATCH * 4]
            for lab in pool.map(_admitted_vocab_label, batch, chunksize=256):
                if lab is not None:
                    labels.append(lab)
    tok = LatexTokenizerV2.build(labels, min_frequency=1 if smoke else min_freq)
    print(f"[tokenizer] Built {len(tok)} tokens from {len(labels):,} deny-clean human labels in {time.perf_counter() - t0:.1f}s", flush=True)
    return tok


def write_split_shards(
    output_dir: Path,
    split_name: str,
    candidates: list[CandidateV2],
    ctx: WorkerCtx,
    *,
    shard_size: int,
    workers: int,
    seed: int,
    eval_tokenizer: LatexTokenizerV2 | None = None,
) -> dict[str, Any]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale shards so a repack never leaves orphan high-numbered files behind
    for old in split_dir.glob("shard-*.tar*"):
        old.unlink()

    rng = random.Random(seed)
    ordered = list(candidates)
    if split_name.startswith("train"):
        rng.shuffle(ordered)

    print(f"[{split_name}] Packing {len(ordered):,} candidates (shard_size={shard_size}) ...", flush=True)
    t0 = time.perf_counter()
    accepted_by_source: Counter[str] = Counter()
    dropped_by_reason: Counter[str] = Counter()
    shards_meta: list[dict[str, Any]] = []
    oov_samples = 0

    shard_idx = 0
    writer: ShardWriter | None = None
    in_shard = 0
    total_ok = 0

    def open_next() -> ShardWriter:
        nonlocal shard_idx, in_shard
        p = split_dir / f"shard-{shard_idx:05d}.tar"
        shard_idx += 1
        in_shard = 0
        return ShardWriter(p)

    # Bounded submission (SUBMIT_BATCH at a time) keeps RAM flat across 400k+ samples
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        for i in range(0, len(ordered), SUBMIT_BATCH):
            chunk = ordered[i : i + SUBMIT_BATCH]
            for status, payload in pool.map(_pack_one_candidate, chunk, chunksize=32):
                if status != "ok":
                    reason, src = payload
                    dropped_by_reason[f"{src}:{reason}"] += 1
                    continue
                meta, png, npz = payload
                if eval_tokenizer is not None and eval_tokenizer.has_unk(meta["normalized_latex"]):
                    oov_samples += 1
                if writer is None:
                    writer = open_next()
                key = f"{total_ok:08d}"
                writer.write(key, meta, png, npz)
                total_ok += 1
                in_shard += 1
                accepted_by_source[meta["source"]] += 1
                if in_shard >= shard_size:
                    shards_meta.append(writer.close())
                    writer = None
            if (i + len(chunk)) % (SUBMIT_BATCH * 4) == 0 or i + len(chunk) >= len(ordered):
                rate = total_ok / max(time.perf_counter() - t0, 1e-3)
                print(f"[{split_name}] {i + len(chunk):,}/{len(ordered):,} scanned | {total_ok:,} packed ({rate:.0f}/s)", flush=True)

    if writer is not None:
        shards_meta.append(writer.close())
    if total_ok == 0:
        raise RuntimeError(f"Split {split_name!r} packed 0 samples!")

    elapsed = time.perf_counter() - t0
    info: dict[str, Any] = {
        "directory": split_name,
        "samples": total_ok,
        "accepted_by_source": dict(accepted_by_source),
        "dropped_by_reason": dict(dropped_by_reason),
        "shards": shards_meta,
    }
    if eval_tokenizer is not None:
        info["oov_samples"] = oov_samples
        info["oov_rate"] = oov_samples / max(total_ok, 1)
    print(
        f"[{split_name}] Finished {total_ok:,} samples into {len(shards_meta)} shards in {elapsed:.1f}s "
        f"| dropped={sum(dropped_by_reason.values()):,}"
        + (f" | oov={oov_samples}/{total_ok}" if eval_tokenizer is not None else ""),
        flush=True,
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack audited zero-leakage InkTeX v2 clean shards.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crohme", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--mathwriting", type=Path, default=DEFAULT_MATHWRITING)
    parser.add_argument("--hwrt", type=Path, default=DEFAULT_HWRT)
    parser.add_argument("--hme100k", type=Path, default=DEFAULT_HME)
    parser.add_argument("--unimer", type=Path, default=DEFAULT_UNIMER)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--crohme-synth-cap", type=int, default=0,
                        help="Default 0: exclude unverified CROHME artificial donor writers")
    parser.add_argument("--mw-synth-cap", type=int, default=120000)
    parser.add_argument("--hwrt-cap", type=int, default=30000)
    parser.add_argument("--unimer-cap", type=int, default=60000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    # Remove stale manifest up front so a crashed partial pack can never be mistaken for complete
    (args.output / "manifest_v2.json").unlink(missing_ok=True)
    rng = random.Random(args.seed)

    # 1. Index ALL holdout splits fail-closed and build separate train / val deny indices
    holdout_entries = index_holdout_entries(
        args.crohme, args.mathwriting, args.hme100k, workers=args.workers, smoke=args.smoke,
    )
    train_deny = compose_deny_index(
        holdout_entries,
        ("crohme_test", "crohme_val", "mathwriting_test", "mathwriting_val", "hme100k_val"),
    )
    val_crohme_deny = compose_deny_index(
        holdout_entries,
        ("crohme_test", "mathwriting_test", "mathwriting_val", "hme100k_val"),
    )
    val_mw_deny = compose_deny_index(
        holdout_entries,
        ("crohme_test", "crohme_val", "mathwriting_test", "hme100k_val"),
    )

    # 2. Compact v2 tokenizer built strictly from train_deny-clean human training labels
    tokenizer = build_v2_tokenizer(
        args.crohme, args.mathwriting, args.hme100k, train_deny,
        workers=args.workers, min_freq=args.min_freq, smoke=args.smoke,
    )
    tok_path = args.output / "tokenizer_v2.json"
    tokenizer.save(tok_path)
    tok_set = frozenset(tokenizer.tokens)

    # 3. Assemble candidate pools (explicit split enforcement everywhere)
    crohme_tr = scan_canonical(args.crohme / "train.jsonl", "crohme_train", "crohme", "train")
    mw_tr = scan_mw_dir(args.mathwriting / "train", "mathwriting_train", "train")
    crohme_syn = scan_canonical(args.crohme / "synthetic.jsonl", "crohme_artificial", "crohme", "train")
    mw_syn = scan_mw_dir(args.mathwriting / "synthetic", "mathwriting_synthetic", "train")
    hwrt_tr = scan_canonical(args.hwrt / "train.jsonl", "hwrt_train", "hwrt", "train")

    hme_tr = scan_image_jsonl(args.hme100k, "examples.jsonl", "hme100k_train", "hme100k", {"train"})
    unimer_tr = scan_image_jsonl(args.unimer, "examples.jsonl", "unimer_printed", "unimer", {"train"})

    val_crohme = scan_canonical(args.crohme / "validation.jsonl", "crohme_val", "crohme", "val")
    val_mw_all = scan_mw_dir(args.mathwriting / "valid", "mathwriting_val", "val")
    rng.shuffle(val_mw_all)
    val_mw = val_mw_all[: (64 if args.smoke else VAL_MW_QUOTA)]

    if args.smoke:
        crohme_tr = crohme_tr[:64]
        mw_tr = mw_tr[:64]
        crohme_syn = crohme_syn[: min(32, args.crohme_synth_cap)]
        mw_syn = mw_syn[:32]
        hwrt_tr = hwrt_tr[:32]
        hme_tr = hme_tr[:64]
        unimer_tr = unimer_tr[:32]
        val_crohme = val_crohme[:64]
    else:
        rng.shuffle(crohme_syn)
        crohme_syn = crohme_syn[: args.crohme_synth_cap]
        rng.shuffle(mw_syn)
        mw_syn = mw_syn[: args.mw_synth_cap]
        rng.shuffle(hwrt_tr)
        hwrt_tr = hwrt_tr[: args.hwrt_cap]
        rng.shuffle(unimer_tr)
        unimer_tr = unimer_tr[: args.unimer_cap]

    train_ctx = WorkerCtx(
        deny=train_deny,
        tokenizer_tokens=tok_set,
        max_body_tokens=MAX_BODY_TOKENS_V2,
        unimer_max_body_tokens=94,
        apply_stroke_ratio=True,
    )
    val_crohme_ctx = WorkerCtx(
        deny=val_crohme_deny,
        tokenizer_tokens=None,
        max_body_tokens=MAX_BODY_TOKENS_V2,
        unimer_max_body_tokens=94,
        apply_stroke_ratio=False,
    )
    val_mw_ctx = WorkerCtx(
        deny=val_mw_deny,
        tokenizer_tokens=None,
        max_body_tokens=MAX_BODY_TOKENS_V2,
        unimer_max_body_tokens=94,
        apply_stroke_ratio=False,
    )

    stages: dict[str, Any] = {}
    stages["train_online"] = write_split_shards(
        args.output, "train_online",
        crohme_tr + mw_tr + crohme_syn + mw_syn + hwrt_tr,
        train_ctx,
        shard_size=64 if args.smoke else SHARD_SIZE_ONLINE,
        workers=args.workers,
        seed=args.seed,
    )
    stages["train_offline"] = write_split_shards(
        args.output, "train_offline",
        hme_tr + unimer_tr,
        train_ctx,
        shard_size=64 if args.smoke else SHARD_SIZE_OFFLINE,
        workers=args.workers,
        seed=args.seed + 1,
    )
    stages["val_crohme"] = write_split_shards(
        args.output, "val_crohme",
        val_crohme,
        val_crohme_ctx,
        shard_size=512,
        workers=args.workers,
        seed=args.seed + 2,
        eval_tokenizer=tokenizer,
    )
    stages["val_mw"] = write_split_shards(
        args.output, "val_mw",
        val_mw,
        val_mw_ctx,
        shard_size=512,
        workers=args.workers,
        seed=args.seed + 3,
        eval_tokenizer=tokenizer,
    )

    manifest = {
        "format": FORMAT_VERSION_V2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strip_height": STRIP_HEIGHT_V2,
        "tokenizer_file": tok_path.name,
        "tokenizer_sha256": sha256_file(tok_path),
        "vocab_size": len(tokenizer),
        "stages": stages,
    }
    (args.output / "manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit = {
        "format": FORMAT_VERSION_V2,
        "created_at": manifest["created_at"],
        "smoke": args.smoke,
        "deny_indices": {
            "train": train_deny.to_summary(),
            "val_crohme": val_crohme_deny.to_summary(),
            "val_mw": val_mw_deny.to_summary(),
        },
        "vocab_size": len(tokenizer),
        "stages": {
            name: {
                "samples": st["samples"],
                "accepted_by_source": st["accepted_by_source"],
                "dropped_by_reason": st["dropped_by_reason"],
                **({"oov_samples": st["oov_samples"], "oov_rate": st["oov_rate"]} if "oov_samples" in st else {}),
            }
            for name, st in stages.items()
        },
    }
    (args.output / "audit_report_v2.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[pack_v2] COMPLETE -> {args.output} (vocab={len(tokenizer)})", flush=True)


if __name__ == "__main__":
    main()
