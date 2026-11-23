"""Sequential tar-shard loading for the bimodal (image + ink) recogniser.

Shard sample = ``<key>.json`` + ``<key>.png`` (LA strip) [+ ``<key>.npz`` ink].
"""
from __future__ import annotations

from dataclasses import dataclass
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

from data.bimodal_raster import WIDTH_MULTIPLE, decode_png
from tokenizer.latex_tokenizer import LatexTokenizer, tokenize_latex
from tokenizer.standardize import standardize_latex

SYNTHETIC_SOURCES = frozenset({"mathwriting_synthetic", "crohme_artificial", "ast_synthetic", "unimer"})
STAGE_DIRS = {"stage1": "stage1_train", "stage2": "stage2_train", "val": "val_dev"}
DELTA_TIME = 4
MAX_WIDTH = 640


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


def augment_stroke_features(features: np.ndarray, has_timing: bool, rng: np.random.Generator,
                            config: BimodalAugmentConfig) -> np.ndarray:
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


def augment_image(image: np.ndarray, image_width: int, rng: np.random.Generator,
                  config: BimodalAugmentConfig) -> tuple[np.ndarray, int]:
    """Mild ink-only augmentation on uint8 [2, H, W]; background stays exactly 0."""
    if rng.random() >= config.image_probability:
        return image, image_width
    ink, order = image[0], image[1]
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
    height, width = ink.shape
    scale = 1.0 + rng.uniform(-config.width_jitter, config.width_jitter)
    target = int(min(MAX_WIDTH, max(WIDTH_MULTIPLE, round(width * scale))))
    if target != width:
        ink = np.asarray(Image.fromarray(ink, "L").resize((target, height), Image.Resampling.BILINEAR))
        order = np.asarray(Image.fromarray(order, "L").resize((target, height), Image.Resampling.NEAREST))
        image_width = int(min(target, max(WIDTH_MULTIPLE, round(image_width * target / width))))
    order = np.where(ink > 0, order, 0).astype(np.uint8)
    if image[1].any():
        order = np.where((ink > 0) & (order == 0), 1, order).astype(np.uint8)
    return np.stack([ink, order]), image_width


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
                 standardize_labels: bool = False, keep_unstandardized: bool = False):
        self.tokenizer, self.stage, self.max_tokens = tokenizer, stage, max_tokens
        self.standardize_labels, self.keep_unstandardized = standardize_labels, keep_unstandardized

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
            image, image_width = augment_image(image, image_width, rng, augment)
            if features is not None:
                features = augment_stroke_features(features, has_timing, rng, augment)
        bos, eos = self.tokenizer.bos_id, self.tokenizer.eos_id
        return {
            "id": meta["id"], "source": meta["source"], "split": meta.get("split"),
            "latex": meta["latex"], "normalized_latex": label,
            "synthetic": bool(meta.get("synthetic", meta["source"] in SYNTHETIC_SOURCES)),
            "is_anchor": meta.get("stage3_role") == "anchor", "has_stage3_role": "stage3_role" in meta,
            "image": torch.from_numpy(np.ascontiguousarray(image)), "image_width": image_width,
            "stroke_features": torch.from_numpy(features) if features is not None else None,
            "has_timing": has_timing,
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
        self.decoder = _SampleDecoder(self.tokenizer, stage, max_tokens, standardize_labels)
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
    stroke_features = stroke_padding_mask = None
    if bool(has_strokes.any()):
        length = max(len(sample["stroke_features"]) for sample in batch if sample["stroke_features"] is not None)
        channels = next(sample["stroke_features"].shape[1] for sample in batch if sample["stroke_features"] is not None)
        stroke_features = torch.zeros((len(batch), length, channels), dtype=torch.float32)
        stroke_padding_mask = torch.ones((len(batch), length), dtype=torch.bool)
        for row, sample in enumerate(batch):
            features = sample["stroke_features"]
            if features is not None:
                stroke_features[row, :len(features)] = features
                stroke_padding_mask[row, :len(features)] = False
    tokens, token_padding_mask = _pad_tokens([sample["tokens"] for sample in batch], pad_id)
    r2l_tokens, r2l_token_padding_mask = _pad_tokens([sample["r2l_tokens"] for sample in batch], pad_id)
    return {
        "images": images, "image_widths": image_widths,
        "stroke_features": stroke_features, "stroke_padding_mask": stroke_padding_mask,
        "has_strokes": has_strokes,
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
