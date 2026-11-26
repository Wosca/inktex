"""InkTeX v2 shard format (`inktex-shards-v2`), manifest-verified shard loading, single-transform
coupled augmentation, shared chunk/point featurization, and static-bucket collate.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import io
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence
import zlib

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from data.bimodal_raster import decode_png
from data.bimodal_shards import encode_npz, iter_shard

from .ink_preprocess import (
    CHUNK_BUCKETS,
    CHUNK_META_DIM,
    CHUNK_SIZE,
    POINT_FEATURE_DIM,
    InkPreprocessV2Config,
    chunk_ink_v2,
    resample_and_featurize_strokes,
    select_bucket,
    warp_raw_strokes,
)
from .tokenizer_v2 import MAX_SEQ_TOKENS_V2, LatexTokenizerV2
from .vision_raster import (
    STRIP_HEIGHT_V2,
    WIDTH_BUCKETS_V2,
    WIDTH_MULTIPLE_V2,
    render_strokes_to_strip_v2,
    select_width_bucket,
)

FORMAT_VERSION_V2 = "inktex-shards-v2"
SYNTHETIC_SOURCES_V2 = frozenset({
    "mathwriting_synthetic",
    "crohme_artificial",
    "composed_synthetic",
    "unimer_printed",
})


def pack_raw_strokes_npz(
    strokes: Sequence[np.ndarray],
    times: Sequence[np.ndarray | None] | None = None,
) -> bytes:
    clean_xy: list[np.ndarray] = []
    clean_t: list[np.ndarray] = []
    has_times = times is not None and any(t is not None for t in times)
    for i, s in enumerate(strokes):
        arr = np.asarray(s, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            continue
        arr = arr[:, :2]
        finite = np.isfinite(arr).all(axis=1)
        if not finite.any():
            continue
        arr = arr[finite]
        clean_xy.append(arr)
        if has_times:
            t_in = times[i] if (times is not None and i < len(times)) else None
            if t_in is not None and len(t_in) == len(s) and np.isfinite(np.asarray(t_in)[finite]).all():
                clean_t.append(np.asarray(t_in, dtype=np.float32)[finite])
            else:
                has_times = False
    if not clean_xy:
        raise ValueError("Cannot pack empty stroke list")
    starts = np.cumsum([0] + [len(s) for s in clean_xy[:-1]]).astype(np.int32)
    xy = np.concatenate(clean_xy, axis=0).astype(np.float32)
    arrays: dict[str, np.ndarray] = {"xy": xy, "stroke_starts": starts}
    if has_times and len(clean_t) == len(clean_xy):
        arrays["times"] = np.concatenate(clean_t, axis=0).astype(np.float32)
    return encode_npz(**arrays)


def unpack_raw_strokes_npz(data: bytes) -> tuple[list[np.ndarray], list[np.ndarray | None] | None]:
    with np.load(io.BytesIO(data)) as npz:
        xy = npz["xy"].astype(np.float64)
        starts = npz["stroke_starts"].astype(np.int64).tolist()
        edges = [*starts, len(xy)]
        strokes = [xy[a:b] for a, b in zip(edges, edges[1:]) if b > a]
        if "times" in npz:
            t = npz["times"].astype(np.float64)
            times: list[np.ndarray | None] | None = [t[a:b] for a, b in zip(edges, edges[1:]) if b > a]
        else:
            times = None
    return strokes, times


@dataclass(slots=True)
class AugmentV2Config:
    enabled: bool = True
    rotation_degrees: float = 4.5
    shear: float = 0.08
    anisotropic_scale: tuple[float, float] = (0.88, 1.12)
    post_resample_jitter: float = 0.004
    stroke_shift: float = 0.02
    timing_dropout: float = 0.50
    stroke_order_swap_prob: float = 0.06
    stroke_reverse_prob: float = 0.04
    scale_jitter: tuple[float, float] = (0.75, 1.25)
    stroke_width_range: tuple[float, float] = (1.6, 3.8)
    morphology_prob: float = 0.30
    intensity_jitter: float = 0.12
    vision_band_mask_prob: float = 0.40
    vision_band_mask_count: tuple[int, int] = (1, 2)
    vision_band_mask_width: tuple[int, int] = (16, 48)
    vision_band_mask_max_fraction: float = 0.25

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AugmentV2Config":
        known = {f.name: f.default for f in fields(cls)}
        unknown = set(value or {}) - set(known)
        if unknown:
            raise ValueError(f"Unknown AugmentV2Config keys: {sorted(unknown)}")
        return cls(**{
            k: tuple(v) if isinstance(known[k], tuple) else v
            for k, v in (value or {}).items()
        })


def _apply_stroke_order_aug(
    strokes: list[np.ndarray],
    times: list[np.ndarray | None] | None,
    rng: np.random.Generator,
    cfg: AugmentV2Config,
) -> tuple[list[np.ndarray], list[np.ndarray | None] | None]:
    """Perturb at most one adjacent stroke pair / stroke reversal without coupling
    timing availability to expression length (per-sample gate)."""
    strokes = list(strokes)
    times = list(times) if times is not None else None
    if len(strokes) >= 2 and cfg.stroke_order_swap_prob > 0 and rng.random() < cfg.stroke_order_swap_prob:
        i = int(rng.integers(0, len(strokes) - 1))
        strokes[i], strokes[i + 1] = strokes[i + 1], strokes[i]
        if times is not None:
            times[i], times[i + 1] = times[i + 1], times[i]
    if len(strokes) >= 1 and cfg.stroke_reverse_prob > 0 and rng.random() < cfg.stroke_reverse_prob:
        i = int(rng.integers(0, len(strokes)))
        if len(strokes[i]) >= 3:
            strokes[i] = strokes[i][::-1].copy()
            if times is not None and times[i] is not None:
                # Re-monotonize reversed stroke timestamps cleanly
                t_s = times[i]
                times[i] = t_s[0] + (t_s[-1] - t_s[::-1])
    return strokes, times


def _augment_raster_photometric(
    image: np.ndarray,
    image_width: int,
    rng: np.random.Generator,
    cfg: AugmentV2Config,
    *,
    allow_band_mask: bool,
) -> np.ndarray:
    ink, order = image[0].copy(), image[1].copy()
    had_order = bool(order.any())
    if rng.random() < cfg.morphology_prob:
        src = Image.fromarray(ink, "L")
        if rng.random() < 0.5:
            grown = np.asarray(src.filter(ImageFilter.MaxFilter(3)), dtype=np.float32)
            ink = np.rint(0.5 * ink + 0.5 * grown).astype(np.uint8)
            if had_order:
                spread = np.asarray(Image.fromarray(order, "L").filter(ImageFilter.MaxFilter(3)), dtype=np.uint8)
                order = np.where(order > 0, order, spread)
        else:
            shrunk = np.asarray(src.filter(ImageFilter.MinFilter(3)), dtype=np.float32)
            ink = np.rint(0.6 * ink + 0.4 * shrunk).astype(np.uint8)
    gain = 1.0 + float(rng.uniform(-cfg.intensity_jitter, cfg.intensity_jitter))
    ink = np.clip(np.rint(ink.astype(np.float32) * gain), 0, 255).astype(np.uint8)
    order = np.where(ink > 0, order, 0).astype(np.uint8)
    if had_order:
        order = np.where((ink > 0) & (order == 0), 1, order).astype(np.uint8)
    out = np.stack([ink, order], axis=0)
    if allow_band_mask and cfg.vision_band_mask_prob > 0 and rng.random() < cfg.vision_band_mask_prob:
        budget = int(cfg.vision_band_mask_max_fraction * image_width)
        lo_c, hi_c = cfg.vision_band_mask_count
        lo_w, hi_w = cfg.vision_band_mask_width
        for _ in range(int(rng.integers(lo_c, hi_c + 1))):
            w = min(int(rng.integers(lo_w, hi_w + 1)), budget)
            if w <= 0 or image_width <= w:
                break
            start = int(rng.integers(0, image_width - w + 1))
            out[:, :, start : start + w] = 0
            budget -= w
    return out


def resolve_v2_shards(root: str | Path, split: str) -> list[Path]:
    """Resolve exact manifest-listed shards when manifest_v2.json is present (preventing
    stale high-numbered shards from ever being read after a repack)."""
    p = Path(root)
    if p.is_file():
        return [p]
    manifest_path = p / "manifest_v2.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage = manifest.get("stages", {}).get(split)
        if stage is None:
            raise KeyError(f"Split {split!r} not found in {manifest_path}")
        files = [p / stage["directory"] / s["file"] for s in stage.get("shards", [])]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"Manifest-listed shards missing: {missing[:3]}")
        if not files:
            raise FileNotFoundError(f"Manifest split {split!r} has 0 shards")
        return files
    sub = p / split
    target = sub if sub.is_dir() else p
    files = sorted(target.glob("shard-*.tar"))
    if not files:
        raise FileNotFoundError(f"No shard-*.tar files found under {target}")
    return files


def _decode_v2_record(
    record: dict[str, Any],
    tokenizer: LatexTokenizerV2,
    ink_cfg: InkPreprocessV2Config,
    *,
    need_vision: bool,
    need_strokes: bool,
    featurize_mode: str = "chunk",  # "chunk" | "point"
    max_total_points: int | None = None,
    rng: np.random.Generator | None,
    augment: AugmentV2Config | None,
    max_seq_tokens: int | None,
    drop_unk: bool = True,
    drop_overflow: bool = True,
) -> dict[str, Any] | None:
    meta = json.loads(record["json"])
    label = meta["normalized_latex"]
    body = tokenizer.encode_body(label)
    if max_seq_tokens is not None and len(body) + 2 > max_seq_tokens:
        return None
    if drop_unk and tokenizer.unk_id in body:
        return None

    has_strokes = bool(meta.get("has_strokes")) and record.get("npz") is not None
    raw_strokes: list[np.ndarray] | None = None
    raw_times: list[np.ndarray | None] | None = None
    if has_strokes and (need_strokes or (need_vision and rng is not None and augment and augment.enabled)):
        raw_strokes, raw_times = unpack_raw_strokes_npz(record["npz"])

    do_aug = rng is not None and augment is not None and augment.enabled
    if raw_strokes is not None and do_aug:
        raw_strokes, raw_times = _apply_stroke_order_aug(raw_strokes, raw_times, rng, augment)
        raw_strokes = warp_raw_strokes(
            raw_strokes,
            rng,
            rotation_degrees=augment.rotation_degrees,
            shear=augment.shear,
            anisotropic_scale=augment.anisotropic_scale,
            stroke_shift=augment.stroke_shift,
        )

    chunk_points = chunk_point_mask = chunk_meta = chunk_mask = None
    point_features = None
    coarsened = False

    if need_strokes and raw_strokes is not None:
        drop_t = bool(do_aug and rng.random() < augment.timing_dropout)
        post_jit = augment.post_resample_jitter if do_aug else 0.0
        if featurize_mode == "chunk":
            chunked = chunk_ink_v2(
                raw_strokes,
                raw_times,
                ink_cfg,
                drop_timing=drop_t,
                max_total_points=max_total_points,
                post_resample_jitter=post_jit,
                rng=rng if do_aug else None,
            )
            if drop_overflow and chunked.overflowed:
                return None
            coarsened = chunked.coarsened
            v = max(chunked.valid_chunks, 1)
            chunk_points = torch.from_numpy(chunked.chunk_points[:v].copy())
            chunk_point_mask = torch.from_numpy(chunked.chunk_point_mask[:v].copy())
            chunk_meta = torch.from_numpy(chunked.chunk_meta[:v].copy())
            chunk_mask = torch.from_numpy(chunked.chunk_mask[:v].copy())
        elif featurize_mode == "point":
            res = resample_and_featurize_strokes(
                raw_strokes,
                raw_times,
                ink_cfg,
                max_total_points=max_total_points,
                max_total_chunks=ink_cfg.max_chunks,
                drop_timing=drop_t,
                post_resample_jitter=post_jit,
                rng=rng if do_aug else None,
            )
            if drop_overflow and res.overflowed:
                return None
            if not res.per_stroke_features:
                return None
            coarsened = res.coarsened
            pts = np.concatenate(res.per_stroke_features, axis=0)
            if max_total_points is not None and len(pts) > max_total_points:
                pts = pts[:max_total_points]
            point_features = torch.from_numpy(pts.copy())
        else:
            raise ValueError(f"Unknown featurize_mode: {featurize_mode}")

    image_tensor = None
    image_width = None
    if need_vision:
        if raw_strokes is not None and do_aug:
            sj = float(rng.uniform(*augment.scale_jitter))
            sw = float(rng.uniform(*augment.stroke_width_range))
            rendered = render_strokes_to_strip_v2(raw_strokes, scale_jitter=sj, stroke_width=sw)
            arr = np.ascontiguousarray(np.asarray(rendered.image, dtype=np.uint8).transpose(2, 0, 1))
            image_width = rendered.image_width
            arr = _augment_raster_photometric(arr, image_width, rng, augment, allow_band_mask=True)
            image_tensor = torch.from_numpy(arr)
        else:
            arr = decode_png(record["png"])
            image_width = int(min(arr.shape[-1], max(WIDTH_MULTIPLE_V2, int(meta["image_width"]))))
            if do_aug:
                arr = _augment_raster_photometric(arr, image_width, rng, augment, allow_band_mask=has_strokes)
            image_tensor = torch.from_numpy(np.ascontiguousarray(arr))

    l2r = torch.tensor([tokenizer.bos_l2r_id, *body, tokenizer.eos_id], dtype=torch.long)
    r2l = torch.tensor([tokenizer.bos_r2l_id, *reversed(body), tokenizer.eos_id], dtype=torch.long)
    return {
        "id": meta["id"],
        "source": meta["source"],
        "writer_id": meta.get("writer_id"),
        "latex": meta["latex"],
        "normalized_latex": label,
        "synthetic": bool(meta.get("synthetic", meta["source"] in SYNTHETIC_SOURCES_V2)),
        "coarsened": coarsened,
        "chunk_points": chunk_points,
        "chunk_point_mask": chunk_point_mask,
        "chunk_meta": chunk_meta,
        "chunk_mask": chunk_mask,
        "point_features": point_features,
        "image": image_tensor,
        "image_width": image_width,
        "l2r_tokens": l2r,
        "r2l_tokens": r2l,
    }


class InkTeXV2ShardDataset(IterableDataset):
    """Streaming v2 tar-shard dataset supporting stroke-only (chunk OR point), vision-only, and bimodal."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        tokenizer: LatexTokenizerV2 | str | Path,
        *,
        need_vision: bool = True,
        need_strokes: bool = True,
        featurize_mode: str = "chunk",
        max_total_points: int | None = None,
        train: bool = True,
        seed: int = 1337,
        shuffle_buffer: int = 2048,
        max_seq_tokens: int | None = MAX_SEQ_TOKENS_V2,
        clean_human_only: bool = False,
        allowed_sources: Sequence[str] | None = None,
        source_upsample: dict[str, float] | None = None,
        augment: AugmentV2Config | None = None,
        ink_cfg: InkPreprocessV2Config | None = None,
    ):
        self.shards = resolve_v2_shards(root, split)
        self.tokenizer = tokenizer if isinstance(tokenizer, LatexTokenizerV2) else LatexTokenizerV2.load(tokenizer)
        self.need_vision = need_vision
        self.need_strokes = need_strokes
        self.featurize_mode = featurize_mode
        self.max_total_points = max_total_points
        self.train = train
        self.seed = seed
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.max_seq_tokens = max_seq_tokens
        self.clean_human_only = clean_human_only
        self.allowed_sources = frozenset(allowed_sources) if allowed_sources else None
        self.source_upsample = {str(k): float(v) for k, v in source_upsample.items()} if source_upsample else None
        self.augment = (augment or AugmentV2Config()) if train else None
        self.ink_cfg = ink_cfg or InkPreprocessV2Config()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assignment(self) -> tuple[list[tuple[Path, int, int]], int]:
        info = get_worker_info()
        workers, worker = (info.num_workers, info.id) if info is not None else (1, 0)
        order = list(self.shards)
        if self.train:
            np.random.default_rng([self.seed, self.epoch]).shuffle(order)
        if workers <= len(order):
            return [(p, 0, 1) for p in order[worker::workers]], worker
        shard = worker % len(order)
        sharing = len(range(shard, workers, len(order)))
        return [(order[shard], worker // len(order), sharing)], worker

    def __iter__(self) -> Iterator[dict[str, Any]]:
        plan, reader = self._assignment()
        order_rng = np.random.default_rng([self.seed, self.epoch, reader])

        def raw_iter() -> Iterator[tuple[dict[str, Any], int]]:
            for path, offset, stride in plan:
                for pos, rec in enumerate(iter_shard(path, load_npz=self.need_strokes or self.train)):
                    if pos % stride != offset or rec["json"] is None:
                        continue
                    if self.need_vision and rec["png"] is None and rec.get("npz") is None:
                        continue
                    meta = json.loads(rec["json"])
                    if self.clean_human_only and meta.get("synthetic", meta.get("source") in SYNTHETIC_SOURCES_V2):
                        continue
                    if self.allowed_sources is not None and meta.get("source") not in self.allowed_sources:
                        continue
                    if self.need_strokes and not self.need_vision and not meta.get("has_strokes"):
                        continue
                    factor = self.source_upsample.get(meta.get("source"), 1.0) if (self.train and self.source_upsample) else 1.0
                    reps = int(math.floor(factor)) + int(order_rng.random() < factor - math.floor(factor))
                    for rep_idx in range(reps):
                        yield rec, rep_idx

        records: Iterator[tuple[dict[str, Any], int]] = raw_iter()
        if self.train and self.shuffle_buffer > 1:
            def shuffled(src: Iterator[tuple[dict[str, Any], int]]) -> Iterator[tuple[dict[str, Any], int]]:
                buf: list[tuple[dict[str, Any], int]] = []
                for item in src:
                    if len(buf) < self.shuffle_buffer:
                        buf.append(item)
                        continue
                    idx = int(order_rng.integers(len(buf)))
                    yield buf[idx]
                    buf[idx] = item
                order_rng.shuffle(buf)
                yield from buf
            records = shuffled(records)

        yielded = 0
        for rec, rep_idx in records:
            # Per-sample deterministic augmentation seed keyed by (seed, epoch, sample_id, rep_idx)
            # so chunk and point ablation arms receive IDENTICAL augmented strokes!
            aug_rng = None
            if self.train:
                sid = json.loads(rec["json"])["id"]
                h = zlib.crc32(sid.encode("utf-8")) & 0xFFFFFFFF
                aug_rng = np.random.default_rng([self.seed, self.epoch, h, rep_idx])
            sample = _decode_v2_record(
                rec,
                self.tokenizer,
                self.ink_cfg,
                need_vision=self.need_vision,
                need_strokes=self.need_strokes,
                featurize_mode=self.featurize_mode,
                max_total_points=self.max_total_points,
                rng=aug_rng,
                augment=self.augment,
                max_seq_tokens=self.max_seq_tokens,
                drop_unk=True,
                drop_overflow=True,
            )
            if sample is not None:
                yielded += 1
                yield sample
        if yielded == 0:
            raise RuntimeError("InkTeXV2ShardDataset yielded 0 samples — check split/filters/shards!")


class InkTeXV2EvalDataset(Dataset):
    """Deterministic in-memory map-style dataset for locked CROHME / MathWriting dev splits."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        tokenizer: LatexTokenizerV2 | str | Path,
        *,
        need_vision: bool = True,
        need_strokes: bool = True,
        featurize_mode: str = "chunk",
        max_total_points: int | None = None,
        ink_cfg: InkPreprocessV2Config | None = None,
    ):
        self.shards = resolve_v2_shards(root, split)
        self.tokenizer = tokenizer if isinstance(tokenizer, LatexTokenizerV2) else LatexTokenizerV2.load(tokenizer)
        self.need_vision = need_vision
        self.need_strokes = need_strokes
        self.featurize_mode = featurize_mode
        self.max_total_points = max_total_points
        self.ink_cfg = ink_cfg or InkPreprocessV2Config()
        self.records = [rec for p in self.shards for rec in iter_shard(p, load_npz=need_strokes)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = _decode_v2_record(
            self.records[idx],
            self.tokenizer,
            self.ink_cfg,
            need_vision=self.need_vision,
            need_strokes=self.need_strokes,
            featurize_mode=self.featurize_mode,
            max_total_points=self.max_total_points,
            rng=None,
            augment=None,
            max_seq_tokens=None,
            drop_unk=False,
            drop_overflow=False,
        )
        if sample is None:
            raise ValueError(f"Eval sample {idx} could not be decoded")
        return sample


def _pad_tokens(seqs: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.ones((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = False
    return out, mask


def collate_v2_batch(
    batch: Sequence[dict[str, Any]],
    *,
    pad_id: int = 0,
    static_buckets: bool = True,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate empty v2 batch")
    bsz = len(batch)

    # Chunked ink
    has_any_chunk = any(s.get("chunk_points") is not None for s in batch)
    chunk_points = chunk_point_mask = chunk_meta = chunk_mask = None
    if has_any_chunk:
        max_n = max(len(s["chunk_points"]) for s in batch if s.get("chunk_points") is not None)
        pad_n = select_bucket(max_n, CHUNK_BUCKETS) if static_buckets else max_n
        chunk_points = torch.zeros((bsz, pad_n, CHUNK_SIZE, POINT_FEATURE_DIM), dtype=torch.float32)
        chunk_point_mask = torch.ones((bsz, pad_n, CHUNK_SIZE), dtype=torch.bool)
        chunk_meta = torch.zeros((bsz, pad_n, CHUNK_META_DIM), dtype=torch.float32)
        chunk_mask = torch.ones((bsz, pad_n), dtype=torch.bool)
        for i, s in enumerate(batch):
            if s.get("chunk_points") is not None:
                n = min(len(s["chunk_points"]), pad_n)
                chunk_points[i, :n] = s["chunk_points"][:n]
                if s.get("chunk_point_mask") is not None:
                    chunk_point_mask[i, :n] = s["chunk_point_mask"][:n]
                else:
                    chunk_point_mask[i, :n] = False
                chunk_meta[i, :n] = s["chunk_meta"][:n]
                chunk_mask[i, :n] = s["chunk_mask"][:n]

    # Raw point-global ink (for Phase 0B ablation)
    has_any_point = any(s.get("point_features") is not None for s in batch)
    point_features = point_mask = None
    if has_any_point:
        max_p = max(len(s["point_features"]) for s in batch if s.get("point_features") is not None)
        point_features = torch.zeros((bsz, max_p, POINT_FEATURE_DIM), dtype=torch.float32)
        point_mask = torch.ones((bsz, max_p), dtype=torch.bool)
        for i, s in enumerate(batch):
            if s.get("point_features") is not None:
                n = len(s["point_features"])
                point_features[i, :n] = s["point_features"]
                point_mask[i, :n] = False

    # Vision strips (static_buckets=True by default so ConvNeXt-V2 GRN matches inference buckets!)
    has_any_vis = any(s.get("image") is not None for s in batch)
    images = image_widths = None
    if has_any_vis:
        max_w = max(int(s["image"].shape[2]) for s in batch if s.get("image") is not None)
        pad_w = select_width_bucket(max_w, WIDTH_BUCKETS_V2) if static_buckets else int(
            math.ceil(max_w / WIDTH_MULTIPLE_V2) * WIDTH_MULTIPLE_V2
        )
        images = torch.zeros((bsz, 2, STRIP_HEIGHT_V2, pad_w), dtype=torch.float32)
        # Missing images get width=0 -> VisionFPNEncoder masks ALL vision tokens!
        image_widths = torch.zeros((bsz,), dtype=torch.long)
        for i, s in enumerate(batch):
            if s.get("image") is not None:
                img = s["image"]
                w = min(int(img.shape[2]), pad_w)
                images[i, :, :, :w] = img[:, :, :w].float().div_(255.0)
                image_widths[i] = min(int(s["image_width"]), pad_w)

    l2r_tokens, l2r_mask = _pad_tokens([s["l2r_tokens"] for s in batch], pad_id)
    r2l_tokens, r2l_mask = _pad_tokens([s["r2l_tokens"] for s in batch], pad_id)

    return {
        "chunk_points": chunk_points,
        "chunk_point_mask": chunk_point_mask,
        "chunk_meta": chunk_meta,
        "chunk_mask": chunk_mask,
        "point_features": point_features,
        "point_mask": point_mask,
        "images": images,
        "image_widths": image_widths,
        "l2r_tokens": l2r_tokens,
        "l2r_mask": l2r_mask,
        "r2l_tokens": r2l_tokens,
        "r2l_mask": r2l_mask,
        "ids": [s["id"] for s in batch],
        "sources": [s["source"] for s in batch],
        "writer_ids": [s["writer_id"] for s in batch],
        "normalized_latex": [s["normalized_latex"] for s in batch],
        "synthetic": torch.tensor([s["synthetic"] for s in batch], dtype=torch.bool),
        "coarsened": torch.tensor([bool(s.get("coarsened", False)) for s in batch], dtype=torch.bool),
    }
