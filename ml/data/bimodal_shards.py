"""Sequential tar-shard loading for the bimodal (image + ink) recogniser.

Shard sample = ``<key>.json`` + ``<key>.png`` (LA strip) [+ ``<key>.npz`` ink].
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from data.bimodal_raster import WIDTH_MULTIPLE, decode_png, round_up
from data.preprocessing import PreprocessingConfig, _allocate_counts
from tokenizer.latex_tokenizer import LatexTokenizer, tokenize_latex
from tokenizer.standardize import standardize_latex

SYNTHETIC_SOURCES = frozenset({"mathwriting_synthetic", "crohme_artificial", "ast_synthetic", "unimer"})
STAGE_DIRS = {"stage1": "stage1_train", "stage2": "stage2_train", "val": "val_dev"}
DELTA_TIME = 4
STROKE_START, STROKE_END = 7, 8
MAX_WIDTH = 640
MIN_STRIP_WIDTH = 64
ONLINE_PADDING = 8
SHARD_MAX_DELTA_MS = 2000.0
STRAIGHT_COSINE = 0.95


class ShardWriter:
    """Write one tar shard atomically (``.tmp`` then rename) and checksum it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.name + ".tmp")
        self._tar = tarfile.open(self.temporary, "w", format=tarfile.USTAR_FORMAT)
        self.samples = 0

    def _add(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size, info.mtime, info.mode = len(data), 0, 0o644
        self._tar.addfile(info, io.BytesIO(data))

    def write(self, key: str, meta: dict[str, Any], png: bytes, npz: bytes | None = None) -> None:
        if "." in key or "/" in key:
            raise ValueError(f"Shard keys may not contain '.' or '/': {key!r}")
        if bool(meta.get("has_strokes")) != (npz is not None):
            raise ValueError(f"{meta.get('id')}: .npz must be present exactly when has_strokes is true")
        self._add(f"{key}.json", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self._add(f"{key}.png", png)
        if npz is not None:
            self._add(f"{key}.npz", npz)
        self.samples += 1

    def close(self) -> dict[str, Any]:
        self._tar.close()
        os.replace(self.temporary, self.path)
        return {"file": self.path.name, "samples": self.samples, "bytes": self.path.stat().st_size,
                "sha256": sha256_file(self.path)}

    def abort(self) -> None:
        self._tar.close()
        self.temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def encode_npz(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def iter_shard(path: str | Path, *, load_npz: bool = True) -> Iterator[dict[str, Any]]:
    """Yield raw ``{"key", "json", "png", "npz"}`` records in on-disk order."""
    current: dict[str, Any] | None = None
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, _, extension = member.name.rpartition("/")[2].partition(".")
            if current is not None and current["key"] != key:
                yield current
                current = None
            if current is None:
                current = {"key": key, "json": None, "png": None, "npz": None}
            if extension == "npz" and not load_npz:
                continue
            handle = tar.extractfile(member)
            current[extension] = handle.read() if handle is not None else None
    if current is not None:
        yield current


def resolve_shards(shards: str | Path | Sequence[str | Path], stage: str | None = None) -> list[Path]:
    if isinstance(shards, (str, Path)):
        root = Path(shards)
        if root.is_file():
            return [root]
        if stage and (root / STAGE_DIRS.get(stage, stage)).is_dir():
            root = root / STAGE_DIRS.get(stage, stage)
        files = sorted(root.glob("shard-*.tar"))
        if not files:
            raise FileNotFoundError(f"No shard-*.tar files under {root}")
        return files
    files = [Path(item) for item in shards]
    if not files:
        raise ValueError("No shards given")
    return files


def find_tokenizer(shards: Sequence[Path]) -> Path | None:
    for parent in (shards[0].parent, *shards[0].parents):
        if (parent / "tokenizer.json").is_file():
            return parent / "tokenizer.json"
    return None


def _load_tokenizer(value: LatexTokenizer | str | Path | None, shards: Sequence[Path]) -> LatexTokenizer:
    if isinstance(value, LatexTokenizer):
        return value
    path = Path(value) if value is not None else find_tokenizer(shards)
    if path is None:
        raise FileNotFoundError("tokenizer.json not found next to the shards; pass tokenizer=")
    return LatexTokenizer.load(path)


@dataclass(slots=True)
class BimodalAugmentConfig:
    timing_dropout: float = 0.5
    time_scale: tuple[float, float] = (0.7, 1.3)
    coordinate_scale: tuple[float, float] = (0.95, 1.05)
    image_probability: float = 0.8
    morphology_probability: float = 0.3
    intensity_jitter: float = 0.10
    width_jitter: float = 0.05
    coupled_geometry: bool = False
    rotation_degrees: float = 0.0
    shear: float = 0.0
    anisotropic_scale: tuple[float, float] = (0.95, 1.05)
    stroke_jitter: float = 0.0
    point_dropout: float = 0.0
    vision_band_mask_prob: float = 0.0
    vision_band_mask_count: tuple[int, int] = (1, 2)
    vision_band_mask_width: tuple[int, int] = (16, 40)
    vision_band_mask_max_fraction: float = 0.25

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "BimodalAugmentConfig":
        known = {field.name: field.default for field in fields(cls)}
        unknown = set(value or {}) - set(known)
        if unknown:
            raise ValueError(f"Unknown augment keys: {sorted(unknown)}")
        return cls(**{key: tuple(item) if isinstance(known[key], tuple) else item
                      for key, item in (value or {}).items()})


def stroke_bounds(features: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(features[:, STROKE_START] > 0.5)
    if not len(starts) or starts[0] != 0:
        starts = np.concatenate(([0], starts))
    edges = [*starts.tolist(), len(features)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def sample_affine(rng: np.random.Generator, config: BimodalAugmentConfig) -> np.ndarray:
    """One 2x2 map (scale, then x-shear, then rotation) shared by the ink and its strip.

    Ink features and strips are both x-right / y-down (the raster never flips y), so one
    matrix gives the same visual rotation and shear in both.
    """
    angle = math.radians(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
    shear = rng.uniform(-config.shear, config.shear)
    sx, sy = rng.uniform(*config.anisotropic_scale, size=2)
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos, -sin], [sin, cos]])
    return rotation @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([sx, sy])


def _glyph_scale(xy: np.ndarray, bounds: list[tuple[int, int]]) -> float:
    """Median stroke height (width fallback): the size statistic of glyph normalization."""
    for axis in (1, 0):
        sizes = [float(np.ptp(xy[a:b, axis])) for a, b in bounds]
        useful = [size for size in sizes if size > 1e-9]
        if useful:
            return float(np.median(useful))
    return 1.0


def _encode_ms(ms: np.ndarray, max_delta_ms: float) -> np.ndarray:
    return np.log1p(np.clip(ms, 0.0, max_delta_ms)) / math.log1p(max_delta_ms)


def _decode_ms(values: np.ndarray, max_delta_ms: float) -> np.ndarray:
    return np.expm1(np.clip(values, 0.0, 1.0) * math.log1p(max_delta_ms))


def _straight_interior_drops(xy: np.ndarray, bounds: list[tuple[int, int]], probability: float,
                             rng: np.random.Generator) -> np.ndarray:
    dropped = np.zeros(len(xy), dtype=bool)
    for a, b in bounds:
        if b - a < 3:
            continue
        before, after = xy[a + 1:b - 1] - xy[a:b - 2], xy[a + 2:b] - xy[a + 1:b - 1]
        norms = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
        cosine = np.where(norms > 1e-12, (before * after).sum(1) / np.maximum(norms, 1e-12), 1.0)
        dropped[a + 1:b - 1] = (cosine >= STRAIGHT_COSINE) & (rng.random(b - a - 2) < probability)
    dropped[1:] &= ~dropped[:-1]
    return dropped


def warp_stroke_features(features: np.ndarray, matrix: np.ndarray, rng: np.random.Generator,
                         config: BimodalAugmentConfig) -> np.ndarray:
    """Affine-warp the ink, then restore the glyph normalization the shards were packed with
    (first point and median stroke height unchanged), jitter, drop interior points and rebuild dx/dy.
    """
    bounds = stroke_bounds(features)
    xy = features[:, :2].astype(np.float64)
    centroid = xy.mean(0)
    warped = (xy - centroid) @ matrix.T + centroid
    before, after = _glyph_scale(xy, bounds), _glyph_scale(warped, bounds)
    factor = before / after if after > 1e-9 else 1.0
    warped = xy[0] + (warped - warped[0]) * factor
    if config.stroke_jitter > 0:
        warped = warped + rng.normal(0.0, config.stroke_jitter, size=warped.shape)
    result = features.astype(np.float32).copy()
    result[:, :2] = warped
    if config.point_dropout > 0:
        dropped = _straight_interior_drops(warped, bounds, config.point_dropout, rng)
        if dropped.any():
            if result[:, DELTA_TIME].any():
                ms = _decode_ms(result[:, DELTA_TIME].astype(np.float64), SHARD_MAX_DELTA_MS)
                ms[1:][dropped[:-1]] += ms[:-1][dropped[:-1]]
                result[:, DELTA_TIME] = _encode_ms(ms, SHARD_MAX_DELTA_MS)
            result = result[~dropped]
    delta = np.zeros_like(result[:, :2])
    delta[1:] = result[1:, :2] - result[:-1, :2]
    delta[result[:, STROKE_START] > 0.5] = 0.0
    result[:, 2:4] = delta
    return result


def augment_stroke_features(features: np.ndarray, has_timing: bool, rng: np.random.Generator,
                            config: BimodalAugmentConfig, matrix: np.ndarray | None = None) -> np.ndarray:
    if matrix is not None:
        result = warp_stroke_features(features, matrix, rng, config)
    else:
        result = features.copy()
        sx, sy = rng.uniform(*config.coordinate_scale, size=2)
        result[:, [0, 2]] *= sx
        result[:, [1, 3]] *= sy
    if has_timing:
        if rng.random() < config.timing_dropout:
            result[:, DELTA_TIME] = 0.0
        else:
            result[:, DELTA_TIME] *= rng.uniform(*config.time_scale)
    return result


def warp_image(image: np.ndarray, image_width: int, matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply ``matrix`` to the ink of a uint8 [2, H, W] strip, then refit it like the renderer
    (same fill of the drawable area, left margin kept, vertically centred)."""
    ink, order = image[0], image[1]
    height = ink.shape[0]
    ys, xs = np.nonzero(ink[:, :image_width])
    if not len(xs):
        return image, image_width
    pixels = np.stack([xs, ys], 1).astype(np.float64) + 0.5
    left, right = float(xs.min()), float(xs.max() + 1)
    center = np.array([(left + right) / 2, (ys.min() + ys.max() + 1) / 2])
    pixels -= center
    inner_h, inner_w = height - 2 * ONLINE_PADDING, MAX_WIDTH - 2 * ONLINE_PADDING

    def fill(extent: np.ndarray) -> float:
        return min(inner_w / max(extent[0], 1.0), inner_h / max(extent[1], 1.0))

    moved = pixels @ matrix.T
    refit = float(np.clip(fill(np.ptp(moved, 0) + 1) / fill(np.ptp(pixels, 0) + 1), 0.5, 2.0))
    forward = matrix * refit
    moved = pixels @ forward.T
    low, extent = moved.min(0) - 0.5, np.ptp(moved, 0) + 1
    shift = np.array([left, (height - extent[1]) / 2]) - low
    used = left + extent[0] + max(0.0, image_width - right)
    new_width = int(min(MAX_WIDTH, max(min(image.shape[-1], MIN_STRIP_WIDTH), round_up(used))))
    inverse = np.linalg.inv(forward)
    offset = center - inverse @ shift
    data = (inverse[0, 0], inverse[0, 1], offset[0], inverse[1, 0], inverse[1, 1], offset[1])
    size = (new_width, height)
    ink = np.asarray(Image.fromarray(ink, "L").transform(size, Image.Transform.AFFINE, data,
                                                         Image.Resampling.BILINEAR, fillcolor=0))
    order = np.asarray(Image.fromarray(order, "L").transform(size, Image.Transform.AFFINE, data,
                                                             Image.Resampling.NEAREST, fillcolor=0))
    width = int(min(new_width, max(WIDTH_MULTIPLE, math.ceil(used))))
    return np.stack([ink, order]), width


def mask_vision_bands(image: np.ndarray, image_width: int, rng: np.random.Generator,
                      config: BimodalAugmentConfig) -> np.ndarray:
    """Zero vertical bands of both channels so the decoder must read those columns from the ink."""
    if config.vision_band_mask_prob <= 0 or rng.random() >= config.vision_band_mask_prob:
        return image
    budget = int(config.vision_band_mask_max_fraction * image_width)
    low_count, high_count = config.vision_band_mask_count
    low_width, high_width = config.vision_band_mask_width
    result = image.copy()
    for _ in range(int(rng.integers(low_count, high_count + 1))):
        width = min(int(rng.integers(low_width, high_width + 1)), budget)
        if width <= 0:
            break
        start = int(rng.integers(0, image_width - width + 1))
        result[:, :, start:start + width] = 0
        budget -= width
    return result


def augment_image(image: np.ndarray, image_width: int, rng: np.random.Generator,
                  config: BimodalAugmentConfig, matrix: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Ink-only augmentation on uint8 [2, H, W]; background stays exactly 0.

    ``matrix`` (coupled geometry) replaces the independent width jitter.
    """
    had_order = bool(image[1].any())
    if matrix is not None:
        image, image_width = warp_image(image, image_width, matrix)
    elif rng.random() >= config.image_probability:
        return image, image_width
    ink, order = image[0], image[1]
    if matrix is not None and rng.random() >= config.image_probability:
        return _restore_order(ink, order, had_order), image_width
    if rng.random() < config.morphology_probability:
        source = Image.fromarray(ink, "L")
        if rng.random() < 0.5:
            grown = np.asarray(source.filter(ImageFilter.MaxFilter(3)), dtype=np.float32)
            ink = np.rint(0.5 * ink + 0.5 * grown).astype(np.uint8)
            spread = np.asarray(Image.fromarray(order, "L").filter(ImageFilter.MaxFilter(3)), dtype=np.uint8)
            order = np.where(order > 0, order, spread)
        else:
            shrunk = np.asarray(source.filter(ImageFilter.MinFilter(3)), dtype=np.float32)
            ink = np.rint(0.6 * ink + 0.4 * shrunk).astype(np.uint8)
    gain = 1.0 + rng.uniform(-config.intensity_jitter, config.intensity_jitter)
    ink = np.clip(np.rint(ink.astype(np.float32) * gain), 0, 255).astype(np.uint8)
    if matrix is None:
        height, width = ink.shape
        scale = 1.0 + rng.uniform(-config.width_jitter, config.width_jitter)
        target = int(min(MAX_WIDTH, max(WIDTH_MULTIPLE, round(width * scale))))
        if target != width:
            ink = np.asarray(Image.fromarray(ink, "L").resize((target, height), Image.Resampling.BILINEAR))
            order = np.asarray(Image.fromarray(order, "L").resize((target, height), Image.Resampling.NEAREST))
            image_width = int(min(target, max(WIDTH_MULTIPLE, round(image_width * target / width))))
    return _restore_order(ink, order, had_order, spread=matrix is not None), image_width


def _restore_order(ink: np.ndarray, order: np.ndarray, had_order: bool, spread: bool = True) -> np.ndarray:
    order = np.where(ink > 0, order, 0).astype(np.uint8)
    if had_order:
        missing = (ink > 0) & (order == 0)
        if not spread:
            order = np.where(missing, 1, order).astype(np.uint8)
        elif missing.any():
            spread = np.asarray(Image.fromarray(order, "L").filter(ImageFilter.MaxFilter(3)), dtype=np.uint8)
            order = np.where(missing, np.maximum(spread, 1), order).astype(np.uint8)
    return np.stack([ink, order])


def teacher_stroke_features(features: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Re-express shard (glyph-normalized) features in a Stroke2Latex checkpoint's own convention:
    normalization, arc-length spacing, point budget and stroke-index channel."""
    if config.normalization not in ("expression", "glyph"):
        raise ValueError(f"Unknown ink normalization: {config.normalization}")
    bounds = stroke_bounds(features)
    xy = features[:, :2].astype(np.float64)
    timed = bool(config.use_timing and features[:, DELTA_TIME].any())
    times = np.cumsum(_decode_ms(features[:, DELTA_TIME].astype(np.float64), SHARD_MAX_DELTA_MS)) if timed \
        else np.zeros(len(features))
    if config.normalization == "expression":
        low, high = xy.min(0), xy.max(0)
        center, scale = (low + high) / 2, max(float((high - low).max()), 1e-9)
    else:
        center, scale = xy[0], max(_glyph_scale(xy, bounds), 1e-9)
    xy = (xy - center) / scale
    strokes = []
    for a, b in bounds:
        points, stamps = xy[a:b], times[a:b]
        if config.resampling_enabled and config.resample_spacing > 0 and len(points) > 1:
            cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
            total = float(cumulative[-1])
            if total <= 1e-12:
                points, stamps = points[:1], stamps[:1]
            else:
                targets = np.arange(0.0, total, config.resample_spacing)
                if not len(targets) or not math.isclose(float(targets[-1]), total):
                    targets = np.append(targets, total)
                minimum = min(max(config.min_resampled_points_per_stroke, 2), len(points))
                if len(targets) < minimum:
                    targets = np.linspace(0.0, total, minimum)
                points = np.stack([np.interp(targets, cumulative, points[:, axis]) for axis in (0, 1)], 1)
                stamps = np.interp(targets, cumulative, stamps)
        if config.normalization == "glyph" and len(points) > config.max_points_per_stroke:
            chosen = _even_indices(len(points), config.max_points_per_stroke)
            points, stamps = points[chosen], stamps[chosen]
        strokes.append((points, stamps))
    if sum(len(points) for points, _ in strokes) > config.max_points:
        counts = _allocate_counts([len(points) for points, _ in strokes], config.max_points,
                                  config.min_points_per_stroke)
        strokes = [(points[chosen], stamps[chosen]) for (points, stamps), count in zip(strokes, counts)
                   for chosen in [_even_indices(len(points), count)]]
    denominator = max(len(strokes) - 1, 1)
    parts = []
    for index, (points, stamps) in enumerate(strokes):
        rows = np.zeros((len(points), features.shape[1]), dtype=np.float32)
        rows[:, :2] = points
        rows[1:, 2:4] = np.diff(points, axis=0)
        rows[0, STROKE_START] = 1.0
        rows[-1, STROKE_END] = 1.0
        rows[:, 9] = math.log1p(index) / 8.0 if config.normalization == "glyph" else index / denominator
        parts.append((rows, stamps))
    result = np.concatenate([rows for rows, _ in parts])
    if timed:
        stamps = np.concatenate([stamps for _, stamps in parts])
        delta = np.zeros(len(stamps))
        delta[1:] = np.diff(stamps)
        result[:, DELTA_TIME] = _encode_ms(delta, config.max_delta_ms)
    return result


def _even_indices(length: int, count: int) -> np.ndarray:
    if count >= length:
        return np.arange(length)
    if count == 1:
        return np.zeros(1, dtype=np.int64)
    chosen = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
    chosen[0], chosen[-1] = 0, length - 1
    return chosen


def _structure_targets(arrays: Any) -> dict[str, torch.Tensor] | None:
    if arrays is None or "stroke_symbol_ids" not in arrays:
        return None
    targets = {name: torch.from_numpy(arrays[name].astype(np.int64))
               for name in ("stroke_symbol_ids", "stroke_group_ids") if name in arrays}
    if "relation_edges" in arrays:
        targets["relation_edges"] = torch.from_numpy(arrays["relation_edges"].astype(np.int64).reshape(-1, 3))
    return targets


class _SampleDecoder:
    def __init__(self, tokenizer: LatexTokenizer, stage: str, max_tokens: int | None,
                 standardize_labels: bool = False, keep_unstandardized: bool = False,
                 teacher_preprocessing: PreprocessingConfig | None = None):
        self.tokenizer, self.stage, self.max_tokens = tokenizer, stage, max_tokens
        self.standardize_labels, self.keep_unstandardized = standardize_labels, keep_unstandardized
        self.teacher_preprocessing = teacher_preprocessing

    def encode(self, latex: str) -> list[int]:
        unk = self.tokenizer.unk_id
        return [self.tokenizer.token_to_id.get(token, unk) for token in tokenize_latex(latex)]

    def label(self, meta: dict[str, Any]) -> str | None:
        latex = meta["normalized_latex"]
        if not self.standardize_labels:
            return latex
        standard = standardize_latex(latex)
        return latex if standard is None and self.keep_unstandardized else standard

    def __call__(self, record: dict[str, Any], rng: np.random.Generator | None,
                 augment: BimodalAugmentConfig | None) -> dict[str, Any] | None:
        meta = json.loads(record["json"])
        label = self.label(meta)
        if label is None:
            return None
        body = self.encode(label)
        if self.max_tokens is not None and len(body) + 2 > self.max_tokens:
            return None
        image = decode_png(record["png"])
        image_width = int(min(image.shape[-1], max(WIDTH_MULTIPLE, int(meta["image_width"]))))
        features, structure, has_timing = None, None, False
        if self.stage != "stage1" and meta.get("has_strokes") and record.get("npz") is not None:
            with np.load(io.BytesIO(record["npz"])) as arrays:
                features = arrays["features"].astype(np.float32)
                structure = _structure_targets(arrays)
            has_timing = bool(meta.get("has_timing"))
        if rng is not None and augment is not None:
            if augment.coupled_geometry and features is not None:
                matrix = sample_affine(rng, augment)
                features = augment_stroke_features(features, has_timing, rng, augment, matrix)
                image, image_width = augment_image(image, image_width, rng, augment, matrix)
            else:
                image, image_width = augment_image(image, image_width, rng, augment)
                if features is not None:
                    features = augment_stroke_features(features, has_timing, rng, augment)
            if features is not None:
                image = mask_vision_bands(image, image_width, rng, augment)
        teacher = None
        if features is not None and self.teacher_preprocessing is not None:
            teacher = torch.from_numpy(teacher_stroke_features(features, self.teacher_preprocessing))
        bos, eos = self.tokenizer.bos_id, self.tokenizer.eos_id
        return {
            "id": meta["id"], "source": meta["source"], "split": meta.get("split"),
            "latex": meta["latex"], "normalized_latex": label,
            "synthetic": bool(meta.get("synthetic", meta["source"] in SYNTHETIC_SOURCES)),
            "is_anchor": meta.get("stage3_role") == "anchor", "has_stage3_role": "stage3_role" in meta,
            "image": torch.from_numpy(np.ascontiguousarray(image)), "image_width": image_width,
            "stroke_features": torch.from_numpy(features) if features is not None else None,
            "teacher_features": teacher, "has_timing": has_timing,
            "tokens": torch.tensor([bos, *body, eos], dtype=torch.long),
            "r2l_tokens": torch.tensor([bos, *reversed(body), eos], dtype=torch.long),
            "structure_targets": structure,
            "upsample_hint": float(meta.get("upsample_hint", 1.0)),
        }


def _is_synthetic(meta: dict[str, Any]) -> bool:
    return bool(meta.get("synthetic", meta.get("source") in SYNTHETIC_SOURCES))


class BimodalShardDataset(IterableDataset):
    """Streams shards sequentially with epoch-seeded shard order and a shuffle buffer.

    Call ``set_epoch(epoch)`` before each epoch (DataLoader workers receive a copy).
    """

    def __init__(
        self,
        shards: str | Path | Sequence[str | Path],
        *,
        stage: str = "stage2",
        tokenizer: LatexTokenizer | str | Path | None = None,
        train: bool = True,
        seed: int = 1337,
        shuffle_buffer: int = 1024,
        clean_human_only: bool = False,
        max_tokens: int | None = 256,
        augment: BimodalAugmentConfig | None = None,
        use_upsample_hint: bool = False,
        allowed_sources: Sequence[str] | None = None,
        source_upsample: dict[str, float] | None = None,
        standardize_labels: bool = False,
        teacher_preprocessing: PreprocessingConfig | None = None,
    ):
        if stage not in ("stage1", "stage2", "val"):
            raise ValueError("stage must be stage1, stage2 or val")
        self.shards = resolve_shards(shards, stage)
        self.stage, self.train, self.seed = stage, train, seed
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.clean_human_only, self.use_upsample_hint = clean_human_only, use_upsample_hint
        self.allowed_sources = frozenset(allowed_sources) if allowed_sources else None
        self.source_upsample = {str(key): float(value) for key, value in source_upsample.items()} \
            if source_upsample else None
        self.tokenizer = _load_tokenizer(tokenizer, self.shards)
        self.augment = (augment or BimodalAugmentConfig()) if train else None
        self.decoder = _SampleDecoder(self.tokenizer, stage, max_tokens, standardize_labels,
                                      teacher_preprocessing=teacher_preprocessing)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _assignment(self) -> tuple[list[tuple[Path, int, int]], int]:
        info = get_worker_info()
        workers, worker = (info.num_workers, info.id) if info is not None else (1, 0)
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        total, index = workers * world, rank * workers + worker
        order = list(self.shards)
        if self.train:
            np.random.default_rng([self.seed, self.epoch]).shuffle(order)
        if total <= len(order):
            return [(path, 0, 1) for path in order[index::total]], index
        # More readers than shards: readers sharing a shard interleave its samples.
        shard = index % len(order)
        sharing = len(range(shard, total, len(order)))
        return [(order[shard], index // len(order), sharing)], index

    def _records(self, plan: list[tuple[Path, int, int]]) -> Iterator[dict[str, Any]]:
        load_npz = self.stage != "stage1"
        for path, offset, stride in plan:
            for position, record in enumerate(iter_shard(path, load_npz=load_npz)):
                if position % stride != offset or record["json"] is None or record["png"] is None:
                    continue
                if self.clean_human_only or self.allowed_sources is not None:
                    meta = json.loads(record["json"])
                    if self.clean_human_only and _is_synthetic(meta):
                        continue
                    if self.allowed_sources is not None and meta.get("source") not in self.allowed_sources:
                        continue
                yield record

    def _pipeline(self) -> tuple[Iterator[dict[str, Any]], np.random.Generator, int]:
        plan, reader = self._assignment()
        rng = np.random.default_rng([self.seed, self.epoch, reader])
        records = self._records(plan)
        if self.train and (self.use_upsample_hint or self.source_upsample):
            records = self._upsampled(records, rng)
        if self.train and self.shuffle_buffer > 1:
            records = self._shuffled(records, rng)
        return records, rng, reader

    def __iter__(self) -> Iterator[dict[str, Any]]:
        records, rng, _ = self._pipeline()
        for record in records:
            sample = self.decoder(record, rng if self.train else None, self.augment)
            if sample is not None:
                yield sample

    def _upsampled(self, records: Iterator[dict[str, Any]], rng: np.random.Generator) -> Iterator[dict[str, Any]]:
        """``source_upsample`` overrides shard hints and may also downsample (factor < 1)."""
        for record in records:
            meta = json.loads(record["json"])
            if self.source_upsample is not None:
                factor, least = self.source_upsample.get(meta.get("source"), 1.0), 0
            else:
                factor, least = float(meta.get("upsample_hint", 1.0)), 1
            repeats = int(math.floor(factor)) + int(rng.random() < factor - math.floor(factor))
            for _ in range(max(least, repeats)):
                yield record

    def _shuffled(self, records: Iterator[dict[str, Any]], rng: np.random.Generator) -> Iterator[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []
        for record in records:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(record)
                continue
            index = int(rng.integers(len(buffer)))
            yield buffer[index]
            buffer[index] = record
        rng.shuffle(buffer)
        yield from buffer


class BimodalEvalDataset(Dataset):
    """Deterministic in-memory map-style dataset (val_dev is 512 samples)."""

    def __init__(self, shards: str | Path | Sequence[str | Path], *, stage: str = "val",
                 tokenizer: LatexTokenizer | str | Path | None = None, max_tokens: int | None = None,
                 allowed_sources: Sequence[str] | None = None, standardize_labels: bool = False):
        """Standardized references fall back to the stored label so no validation sample is dropped."""
        self.shards = resolve_shards(shards, stage)
        self.tokenizer = _load_tokenizer(tokenizer, self.shards)
        self.decoder = _SampleDecoder(self.tokenizer, "stage1" if stage == "stage1" else "stage2", max_tokens,
                                      standardize_labels, keep_unstandardized=True)
        allowed = frozenset(allowed_sources) if allowed_sources else None
        self.records = [record for path in self.shards for record in iter_shard(path, load_npz=stage != "stage1")
                        if allowed is None or json.loads(record["json"]).get("source") in allowed]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.decoder(self.records[index], None, None)
        if sample is None:
            raise ValueError(f"Sample {index} exceeds max_tokens")
        return sample


def _pad_tokens(sequences: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(sequence) for sequence in sequences)
    tokens = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    mask = torch.ones((len(sequences), length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = sequence
        mask[row, :len(sequence)] = False
    return tokens, mask


def _pad_points(sequences: list[torch.Tensor | None]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    present = [sequence for sequence in sequences if sequence is not None]
    if not present:
        return None, None
    length, channels = max(len(sequence) for sequence in present), present[0].shape[1]
    features = torch.zeros((len(sequences), length, channels), dtype=torch.float32)
    mask = torch.ones((len(sequences), length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        if sequence is not None:
            features[row, :len(sequence)] = sequence
            mask[row, :len(sequence)] = False
    return features, mask


def collate_bimodal_batch(batch: Sequence[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    height = int(batch[0]["image"].shape[1])
    width = int(math.ceil(max(int(sample["image"].shape[2]) for sample in batch) / WIDTH_MULTIPLE) * WIDTH_MULTIPLE)
    images = torch.zeros((len(batch), 2, height, width), dtype=torch.float32)
    for row, sample in enumerate(batch):
        image = sample["image"]
        if image.shape[1] != height:
            raise ValueError(f"Mixed strip heights in batch: {image.shape[1]} != {height}")
        images[row, :, :, :image.shape[2]] = image.float().div_(255.0)
    image_widths = torch.tensor([int(sample["image_width"]) for sample in batch], dtype=torch.long)
    has_strokes = torch.tensor([sample["stroke_features"] is not None for sample in batch], dtype=torch.bool)
    stroke_features, stroke_padding_mask = _pad_points([sample["stroke_features"] for sample in batch])
    teacher_features, teacher_padding_mask = _pad_points([sample.get("teacher_features") for sample in batch])
    tokens, token_padding_mask = _pad_tokens([sample["tokens"] for sample in batch], pad_id)
    r2l_tokens, r2l_token_padding_mask = _pad_tokens([sample["r2l_tokens"] for sample in batch], pad_id)
    return {
        "images": images, "image_widths": image_widths,
        "stroke_features": stroke_features, "stroke_padding_mask": stroke_padding_mask,
        "has_strokes": has_strokes,
        "teacher_features": teacher_features, "teacher_padding_mask": teacher_padding_mask,
        "has_timing": torch.tensor([bool(sample["has_timing"]) for sample in batch], dtype=torch.bool),
        "tokens": tokens, "token_padding_mask": token_padding_mask,
        "r2l_tokens": r2l_tokens, "r2l_token_padding_mask": r2l_token_padding_mask,
        "structure_targets": [sample["structure_targets"] for sample in batch],
        "ids": [sample["id"] for sample in batch], "sources": [sample["source"] for sample in batch],
        "latex": [sample["latex"] for sample in batch],
        "normalized_latex": [sample["normalized_latex"] for sample in batch],
        "synthetic": torch.tensor([bool(sample["synthetic"]) for sample in batch], dtype=torch.bool),
        "is_anchor": torch.tensor([bool(sample.get("is_anchor", False)) for sample in batch], dtype=torch.bool),
        "has_stage3_role": torch.tensor([bool(sample.get("has_stage3_role", False)) for sample in batch],
                                        dtype=torch.bool),
    }


def verify_manifest_checksums(root: str | Path) -> dict[str, bool]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for stage in manifest["stages"].values():
        for shard in stage["shards"]:
            path = root / stage["directory"] / shard["file"]
            results[f"{stage['directory']}/{shard['file']}"] = path.is_file() and sha256_file(path) == shard["sha256"]
    return results
