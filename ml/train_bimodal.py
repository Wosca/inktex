"""Resumable BimodalMath training, validation and Stage 3 hard-example mining.

    python train_bimodal.py --stage 1 --shards <shards> --output <run1>
    python train_bimodal.py --stage 2 --shards <shards> --output <run2> --init-from <run1>/best.pt
    python train_bimodal.py --stage 2 --config configs/bimodal_stage25_3.yaml --shards <shards> --output <run25> \
        --init-from <run2>/epoch-017.pt
    python train_bimodal.py --stage 2 --config configs/bimodal_stage26.yaml --shards <shards> --output <run26> \
        --init-from <run25>/best.pt [--teacher-checkpoint <stroke2latex>/best.pt]
    python train_bimodal.py --mine-hard --shards <shards> --init-from <run2>/best.pt
    python train_bimodal.py --stage 3 --shards <shards> --output <run3> --init-from <run2>/best.pt
    python train_bimodal.py --eval-only --shards <shards> --init-from <run>/best.pt

A session stops cleanly at --max-wall-minutes (recovery.pt, Drive/HF sync, exit 0) and the
next session continues with --resume auto. With --hf-sync-repo <user>/<repo> (HF_TOKEN in the
environment) run artifacts sync to a private Hugging Face dataset repo instead of a mounted Drive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data.bimodal_shards import (
    SYNTHETIC_SOURCES, BimodalAugmentConfig, BimodalEvalDataset, BimodalShardDataset, ShardWriter,
    collate_bimodal_batch, iter_shard, resolve_shards, sha256_file,
)
from data.preprocessing import PreprocessingConfig
from evaluation.recognition import canonical
from model.bimodal import BimodalMathModel, BimodalModelConfig, get_bimodal_config
from model.stroke_structure import structure_loss
from tokenizer.latex_tokenizer import LatexTokenizer, canonicalize_latex_tokens, tokenize_latex
from tokenizer.standardize import standardize_latex
from training.benchmark_metrics import reference_tokens
from training.checkpoint import load_checkpoint
from training.losses import sequence_cross_entropy
from training.metrics import edit_distance
from training.recovery import atomic_json, restore_rng, rng_state

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "bimodal_g4.yaml"
RECOVERY_FORMAT = "bimodal-recovery-v1"
CHECKPOINT_FORMAT = "bimodal-checkpoint-v1"
HARD_FORMAT = "bimodal-hard-v1"
HARD_DIRECTORY = "stage3_hard"
SYNC_FILES = ("recovery.pt", "best.pt", "metrics.jsonl", "status.json", "config_resolved.json")
SYNTAX_TOKENS = frozenset({"{", "}", "^", "_", "&", r"\frac", r"\sqrt", r"\text", r"\begin", r"\end", r"\mathrm",
                           r"\mathbf", r"\operatorname"})
STRUCTURE_TOKENS = frozenset({r"\frac", r"\sqrt", "^", "_", r"\sum", r"\int", r"\lim", r"\prod", r"\begin", r"\end",
                              "{", "}"})
DEFAULT_REWARD_WEIGHTS = {"exact_norm": 1.0, "strict_case_bonus": 0.25, "structure_skeleton": 0.5,
                          "edit_penalty": 0.5}
PLACEHOLDER_MARKS = ("xxx", "your", "placeholder", "changeme", "token_here", "<", "...")
BATCH_TENSORS = ("images", "image_widths", "stroke_features", "stroke_padding_mask", "has_strokes", "has_timing",
                 "tokens", "token_padding_mask", "r2l_tokens", "r2l_token_padding_mask", "synthetic", "is_anchor",
                 "has_stage3_role", "teacher_features", "teacher_padding_mask")
SYNC_FAILED_EXIT = 75


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- environment


def real_hf_token(token: str | None) -> bool:
    value = (token or "").strip()
    lowered = value.lower()
    return value.startswith("hf_") and len(value) >= 20 and not any(mark in lowered for mark in PLACEHOLDER_MARKS)


def redacted_argv(argv: list[str]) -> list[str]:
    redacted, hide = [], False
    for value in map(str, argv):
        if value.startswith("--hf-token="):
            value = "--hf-token=***"
        redacted.append("***" if hide else value)
        hide = value == "--hf-token"
    return redacted


def configure_environment(hf_token: str | None = None) -> str | None:
    """Returns the HF token for explicit use; public timm downloads never send one implicitly."""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    token = os.environ.get("HF_TOKEN")
    if not real_hf_token(token):
        os.environ.pop("HF_TOKEN", None)
        token = None
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    # timm imports huggingface_hub before main() runs, so its constants already read the environment.
    if (constants := sys.modules.get("huggingface_hub.constants")) is not None:
        constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return token.strip() if token else None


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    return device


def autocast(device: torch.device, dtype_name: str | None):
    if device.type != "cuda" or dtype_name in (None, "float32", "fp32"):
        return nullcontext()
    if dtype_name not in ("bfloat16", "bf16"):
        raise ValueError(f"Unsupported amp_dtype {dtype_name!r}; use bfloat16 or float32")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def maybe_compile(model: BimodalMathModel, enabled: bool, device: torch.device) -> list[str]:
    if not enabled:
        return []
    if device.type != "cuda":
        log("compile: skipped on CPU")
        return []
    if importlib.util.find_spec("triton") is None:
        log("compile: skipped, triton is unavailable on this platform")
        return []
    import torch._dynamo
    dynamo = torch._dynamo.config
    dynamo.suppress_errors = True
    limit = "recompile_limit" if hasattr(dynamo, "recompile_limit") else "cache_size_limit"
    setattr(dynamo, limit, max(64, getattr(dynamo, limit)))
    compiled = []
    for name in ("vision_encoder", "decoder", "r2l_decoder"):
        module = getattr(model, name, None)
        if module is None:
            continue
        try:
            module.compile(dynamic=True)
            compiled.append(name)
        except Exception as error:  # noqa: BLE001
            log(f"compile: {name} stays eager ({error})")
    log(f"compile: {', '.join(compiled) or 'nothing'} (in place, state_dict keys unchanged)")
    return compiled


# --------------------------------------------------------------------------- files


def file_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(existing + json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class DriveSync:
    """Copies run artifacts to Drive via ``.tmp`` + ``os.replace``, skipping unchanged files."""

    def __init__(self, output: Path, target: Path | None, every_minutes: float):
        self.output, self.target = output, target
        self.every = max(0.0, float(every_minutes)) * 60.0
        self.last = time.monotonic()
        self.copied: dict[str, tuple[int, int]] = {}

    def due(self) -> bool:
        return self.target is not None and self.every > 0 and time.monotonic() - self.last >= self.every

    def now(self) -> None:
        self.last = time.monotonic()
        if self.target is None:
            return
        for name in SYNC_FILES:
            source = self.output / name
            if not source.is_file():
                continue
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if self.copied.get(name) == signature and (self.target / name).is_file():
                continue
            try:
                atomic_copy(source, self.target / name)
                self.copied[name] = signature
            except OSError as error:
                log(f"sync: failed to copy {name} ({error})")


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


class HFSync:
    """Mirrors run artifacts to ``<prefix>/`` of a private HF dataset repo, one commit per sync."""

    RETRY_DELAYS = (2.0, 5.0, 10.0)

    def __init__(self, output: Path, repo: str, prefix: str, every_minutes: float, token: str | None):
        if not real_hf_token(token):
            raise SystemExit(f"--hf-sync-repo {repo} needs a write-scoped HF_TOKEN (hf_...) in the environment "
                             "or Colab Secrets")
        from huggingface_hub import HfApi
        self.output, self.repo, self.prefix, self.token = Path(output), repo, prefix.strip("/"), token
        self.api = HfApi(token=token)
        self.every = max(0.0, float(every_minutes)) * 60.0
        self.last = time.monotonic()
        self.uploaded: dict[str, tuple[int, int]] = {}
        self.healthy = True

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/tree/main/{self.prefix}"

    def remote(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def attempt(self, action, label: str, retries: int) -> Any:
        delays = self.RETRY_DELAYS[:retries]
        for index in range(len(delays) + 1):
            try:
                return action()
            except Exception as error:  # noqa: BLE001
                if index == len(delays):
                    raise
                log(f"hf-sync: {label} failed ({type(error).__name__}: {error}); retrying in {delays[index]:g}s")
                time.sleep(delays[index])

    def ensure_repo(self) -> None:
        try:
            self.attempt(lambda: self.api.create_repo(repo_id=self.repo, repo_type="dataset", private=True,
                                                      exist_ok=True), "create_repo", len(self.RETRY_DELAYS))
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"--hf-sync-repo {self.repo}: cannot create or access the private dataset repo "
                             f"({type(error).__name__}: {error}). Check HF_TOKEN has write scope.") from error
        log(f"hf-sync: {self.url}")

    def due(self) -> bool:
        return self.every > 0 and time.monotonic() - self.last >= self.every

    def changed(self, source: Path) -> list[tuple[str, tuple[int, int]]]:
        found = []
        for name in SYNC_FILES:
            path = source / name
            if path.is_file() and self.uploaded.get(name) != (signature := file_signature(path)):
                found.append((name, signature))
        return found

    def sync(self, source_dir: Path | None = None, force: bool = False) -> bool:
        """``force`` retries with backoff (epoch end, pause, exit); periodic syncs warn and move on."""
        if not force and not self.due():
            return True
        self.last = time.monotonic()
        source = Path(source_dir or self.output)
        pending = self.changed(source)
        if not pending:
            return True
        from huggingface_hub import CommitOperationAdd
        status = {}
        if (source / "status.json").is_file():
            try:
                status = json.loads((source / "status.json").read_text(encoding="utf-8"))
            except ValueError:
                pass
        message = (f"{self.prefix}: {status.get('state', 'sync')} epoch {status.get('epoch', '?')} "
                   f"step {status.get('global_step', '?')}")
        names = ", ".join(name for name, _ in pending)

        def commit():
            operations = [CommitOperationAdd(path_in_repo=self.remote(name), path_or_fileobj=str(source / name))
                          for name, _ in pending]
            return self.api.create_commit(repo_id=self.repo, repo_type="dataset", operations=operations,
                                          commit_message=message)

        started = time.perf_counter()
        try:
            self.attempt(commit, f"commit [{names}]", len(self.RETRY_DELAYS) if force else 0)
        except Exception as error:  # noqa: BLE001
            self.healthy = False
            level = "ERROR" if force else "warning"
            log(f"hf-sync {level}: upload of [{names}] failed ({type(error).__name__}: {error}); "
                f"{'artifacts remain only in ' + str(source) if force else 'training continues, retrying later'}")
            return False
        self.uploaded.update(pending)
        self.healthy = True
        log(f"hf-sync: committed [{names}] in {time.perf_counter() - started:.1f}s")
        return True

    def list_files(self) -> set[str]:
        return set(self.attempt(lambda: self.api.list_repo_files(self.repo, repo_type="dataset"), "list_repo_files",
                                len(self.RETRY_DELAYS)))

    def fetch(self, path_in_repo: str, destination: Path) -> Path:
        from huggingface_hub import hf_hub_download
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".hf-download-{destination.name}"
        try:
            downloaded = self.attempt(lambda: hf_hub_download(repo_id=self.repo, filename=path_in_repo,
                                                              repo_type="dataset", token=self.token,
                                                              local_dir=staging),
                                      f"download {path_in_repo}", len(self.RETRY_DELAYS))
            os.replace(downloaded, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return destination

    def download_file(self, path_in_repo: str, destination: Path) -> Path:
        if path_in_repo not in self.list_files():
            raise FileNotFoundError(f"hf://datasets/{self.repo}/{path_in_repo} does not exist")
        return self.fetch(path_in_repo, destination)

    def restore_run(self, output_dir: Path) -> bool:
        try:
            remote = self.list_files()
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"resume: cannot list {self.repo} ({error}); refusing to start fresh over a "
                             "possibly existing run") from error
        if self.remote("recovery.pt") not in remote:
            return False
        output_dir = Path(output_dir)
        for name in SYNC_FILES:
            if self.remote(name) in remote:
                self.uploaded[name] = file_signature(self.fetch(self.remote(name), output_dir / name))
        return True

    def squash(self) -> None:
        try:
            self.api.super_squash_history(repo_id=self.repo, repo_type="dataset",
                                          commit_message=f"squash: latest {self.prefix}")
            log("hf-sync: squashed repo history (superseded checkpoint versions are released)")
        except Exception as error:  # noqa: BLE001
            log(f"hf-sync warning: history squash failed ({type(error).__name__}: {error})")


# --------------------------------------------------------------------------- model and data wiring


def visible_token_ids(tokenizer: LatexTokenizer) -> list[int]:
    return [index for index, token in enumerate(tokenizer.tokens)
            if index >= 4 and token not in SYNTAX_TOKENS and not token.startswith("<")]


def manifest_structure(manifest: dict[str, Any]) -> tuple[list[str], list[str]] | None:
    structure = manifest.get("structure") or {}
    symbols, relations = structure.get("symbols"), structure.get("relation_classes")
    if not symbols or not relations:
        return None
    if relations[0] != "None":
        log("structure: manifest relation vocabulary does not start with None; structural heads disabled")
        return None
    return list(symbols), list(relations)


def build_model_config(config: dict[str, Any], tokenizer: LatexTokenizer, manifest: dict[str, Any], stage: int,
                       preset: str | None = None, pretrained: bool = True) -> BimodalModelConfig:
    values = dict(config.get("model") or {})
    configured = values.pop("preset", "bimodal_75m")
    chosen = preset or configured
    max_points, max_tokens = int(values.pop("max_points", 2048)), int(values.pop("max_tokens", 512))
    if chosen != configured:
        # Sizing keys belong to the YAML preset; a different preset keeps only semantics.
        values = {key: value for key, value in values.items() if key in ("position_encoding", "coverage")}
    fields = BimodalModelConfig.__dataclass_fields__
    overrides = {key: value for key, value in values.items() if key in fields}
    overrides.update(visible_token_ids=visible_token_ids(tokenizer), pad_id=tokenizer.pad_id,
                     bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    if stage >= 2 and (structure := manifest_structure(manifest)) is not None:
        overrides.update(structure_symbols=structure[0], structure_relations=structure[1])
    if not pretrained:
        overrides["vision_pretrained"] = False
    return get_bimodal_config(chosen, len(tokenizer), max_points, max_tokens, **overrides)


def checkpoint_weights(payload: dict[str, Any]) -> tuple[dict[str, torch.Tensor], str]:
    for key in ("ema_state_dict", "model_state_dict"):
        if payload.get(key) is not None:
            return payload[key], key
    raise ValueError("Checkpoint has neither ema_state_dict nor model_state_dict")


def model_from_payload(payload: dict[str, Any], device: torch.device) -> BimodalMathModel:
    config = BimodalModelConfig.from_dict({**payload["model_config"], "vision_pretrained": False})
    model = BimodalMathModel(config)
    state, _ = checkpoint_weights(payload)
    model.load_state_dict(state)
    return model.to(device)


def _summarize_names(names: list[str], depth: int = 3, limit: int = 12) -> str:
    groups = Counter(".".join(name.split(".")[:depth]) for name in names)
    parts = [f"{prefix} ({count})" for prefix, count in groups.most_common(limit)]
    if len(groups) > limit:
        parts.append(f"... {len(groups) - limit} more groups")
    return ", ".join(parts)


def load_partial(model: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> dict[str, Any]:
    own = model.state_dict()
    accepted, mismatched, unexpected = {}, [], []
    for name, value in state.items():
        name = name.replace("_orig_mod.", "")
        target = own.get(name)
        if target is None:
            unexpected.append(name)
        elif tuple(target.shape) != tuple(value.shape):
            mismatched.append(name)
        else:
            accepted[name] = value
    fresh = [name for name in own if name not in accepted]
    model.load_state_dict(accepted, strict=False)
    log(f"{label}: loaded {len(accepted)}/{len(own)} tensors")
    if mismatched:
        log(f"  shape mismatch, freshly initialized: {_summarize_names(mismatched)}")
    missing = [name for name in fresh if name not in set(mismatched)]
    if missing:
        log(f"  absent from checkpoint, freshly initialized: {_summarize_names(missing)}")
    if unexpected:
        log(f"  ignored checkpoint-only tensors: {_summarize_names(unexpected)}")
    return {"loaded": len(accepted), "total": len(own), "mismatched": mismatched, "missing": missing,
            "unexpected": unexpected}


def parameter_groups(model: torch.nn.Module, lr: float, vision_lr: float, weight_decay: float,
                     stroke_lr: float | None = None) -> list[dict]:
    """``stroke_lr=None`` keeps the stroke encoder in the main groups (the original optimizer layout)."""
    order = ("main", "stroke", "vision")
    rates = {"main": lr, "stroke": lr if stroke_lr is None else stroke_lr, "vision": vision_lr}
    buckets: dict[tuple[int, bool], list[torch.nn.Parameter]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("vision_encoder.backbone."):
            part = "vision"
        elif stroke_lr is not None and name.startswith("stroke_encoder."):
            part = "stroke"
        else:
            part = "main"
        decay = parameter.ndim > 1 and not name.endswith(".bias") and "modality_emb" not in name
        buckets[(order.index(part), decay)].append(parameter)
    groups = []
    for (part, decay), parameters in sorted(buckets.items()):
        base = rates[order[part]]
        groups.append({"params": parameters, "lr": base, "base_lr": base,
                       "weight_decay": weight_decay if decay else 0.0,
                       "name": f"{order[part]}_{'decay' if decay else 'no_decay'}"})
    return groups


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = float(decay), 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        ema = list(self.module.parameters())
        live = [parameter.detach() for parameter in model.parameters()]
        torch._foreach_lerp_(ema, live, 1.0 - decay)
        for ema_buffer, buffer in zip(self.module.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


@dataclass
class Schedule:
    warmup_steps: int
    main_steps: int
    floor: float
    cooldown_factor: float | None = None

    def factor(self, step: int, cooldown: bool) -> float:
        if cooldown and self.cooldown_factor is not None:
            return self.cooldown_factor
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        span = max(1, self.main_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / span))
        return self.floor + (1.0 - self.floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResumableShardDataset(BimodalShardDataset):
    """Skips the first ``skip_batches`` loader batches without decoding images.

    Batches come round-robin from loader workers, so each reader skips its own share.
    """

    def __init__(self, *args: Any, skip_batches: int = 0, batch_size: int = 1, loader_workers: int = 0,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.skip_batches, self.batch_size, self.loader_workers = int(skip_batches), batch_size, max(1, loader_workers)

    def _fits(self, record: dict[str, Any]) -> bool:
        latex = self.decoder.label(json.loads(record["json"]))
        if latex is None:
            return False
        return self.decoder.max_tokens is None or len(self.decoder.encode(latex)) + 2 <= self.decoder.max_tokens

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.skip_batches:
            yield from super().__iter__()
            return
        records, rng, reader = self._pipeline()
        skip = len(range(reader % self.loader_workers, self.skip_batches, self.loader_workers)) * self.batch_size
        for record in records:
            if skip:
                skip -= self._fits(record)
                continue
            sample = self.decoder(record, rng if self.train else None, self.augment)
            if sample is not None:
                yield sample


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in BATCH_TENSORS:
        if moved.get(key) is not None:
            moved[key] = moved[key].to(device, non_blocking=True)
    return moved


def align_batch(batch: dict[str, Any], token_multiple: int, point_multiple: int, pad_id: int) -> dict[str, Any]:
    """Round sequence lengths up so compiled kernels see fewer distinct shapes."""
    def pad_to(tensor, mask, multiple, value):
        extra = -tensor.shape[1] % multiple
        if not extra:
            return tensor, mask
        shape = (tensor.shape[0], extra, *tensor.shape[2:])
        return (torch.cat((tensor, tensor.new_full(shape, value)), 1),
                torch.cat((mask, mask.new_ones((mask.shape[0], extra))), 1))
    if token_multiple > 1:
        batch["tokens"], batch["token_padding_mask"] = pad_to(batch["tokens"], batch["token_padding_mask"],
                                                              token_multiple, pad_id)
        batch["r2l_tokens"], batch["r2l_token_padding_mask"] = pad_to(
            batch["r2l_tokens"], batch["r2l_token_padding_mask"], token_multiple, pad_id)
    if point_multiple > 1 and batch.get("stroke_features") is not None:
        batch["stroke_features"], batch["stroke_padding_mask"] = pad_to(
            batch["stroke_features"], batch["stroke_padding_mask"], point_multiple, 0.0)
    return batch


def encoder_inputs(batch: dict[str, Any], modality: str) -> dict[str, Any]:
    vision = modality != "stroke"
    ink = modality != "vision" and batch.get("stroke_features") is not None
    return {"images": batch["images"] if vision else None,
            "image_widths": batch["image_widths"] if vision else None,
            "stroke_features": batch["stroke_features"] if ink else None,
            "stroke_padding_mask": batch["stroke_padding_mask"] if ink else None,
            "has_strokes": batch["has_strokes"] if ink else None}


def structure_record(targets: dict[str, torch.Tensor] | None, symbols: int, relations: int) -> dict | None:
    """Shard targets -> the record format of ``model.stroke_structure.structure_loss``."""
    if not targets or "stroke_group_ids" not in targets or "stroke_symbol_ids" not in targets:
        return None
    groups, labels = targets["stroke_group_ids"].tolist(), targets["stroke_symbol_ids"].tolist()
    if not groups or len(groups) != len(labels) or min(groups) < 0:
        return None
    classes = [-1] * (max(groups) + 1)
    for group, label in zip(groups, labels):
        if not 0 <= label < symbols or classes[group] not in (-1, label):
            return None
        classes[group] = label
    if -1 in classes:
        return None
    edges = targets.get("relation_edges")
    relation_list = []
    if edges is not None:
        for left, right, relation in edges.tolist():
            if not (0 <= left < len(classes) and 0 <= right < len(classes) and 0 <= relation < relations):
                return None
            relation_list.append((left, right, relation))
    return {"stroke_groups": groups, "symbol_classes": classes, "relations": relation_list,
            "relations_available": edges is not None}


class TeacherKD:
    """Frozen stroke-only ``Stroke2Latex`` distilled into the student on the vocabulary both share.

    The teacher reads ``batch["teacher_features"]``: the augmented ink re-expressed in its own
    preprocessing (``data.bimodal_shards.teacher_stroke_features``).
    """

    def __init__(self, model: torch.nn.Module, teacher: LatexTokenizer, student: LatexTokenizer,
                 preprocessing: PreprocessingConfig, device: torch.device, amp_dtype: str | None):
        self.device, self.amp, self.preprocessing = device, amp_dtype, preprocessing
        self.model = model.to(device).eval().requires_grad_(False)
        if device.type == "cuda" and amp_dtype in ("bfloat16", "bf16"):
            self.model.to(dtype=torch.bfloat16)
        self.max_points, self.max_tokens = int(model.config.max_points), int(model.config.max_tokens)
        self.pad_id = student.pad_id
        mapping = torch.tensor([teacher.token_to_id.get(token, -1) for token in student.tokens], dtype=torch.long)
        excluded = {student.pad_id, student.bos_id, student.unk_id}
        shared = [index for index, target in enumerate(mapping.tolist()) if target >= 0 and index not in excluded]
        self.student_to_teacher = mapping.to(device)
        self.teacher_input = torch.where(mapping >= 0, mapping, torch.tensor(teacher.unk_id)).to(device)
        self.shared_student_ids = torch.tensor(shared, dtype=torch.long, device=device)
        self.shared_teacher_ids = mapping[shared].to(device)
        to_shared = torch.full((len(student),), -1, dtype=torch.long)
        to_shared[shared] = torch.arange(len(shared))
        self.student_to_shared = to_shared.to(device)

    @classmethod
    def load(cls, path: str | Path, student: LatexTokenizer, device: torch.device,
             amp_dtype: str | None) -> "TeacherKD":
        model, teacher, preprocessing, _ = load_checkpoint(path)
        return cls(model, teacher, student, preprocessing, device, amp_dtype)

    def summary(self) -> dict[str, Any]:
        return {"parameters": sum(parameter.numel() for parameter in self.model.parameters()),
                "shared_tokens": len(self.shared_student_ids), "max_points": self.max_points,
                "max_tokens": self.max_tokens, "normalization": self.preprocessing.normalization}

    def __call__(self, batch: dict[str, Any], student_logits: torch.Tensor, temperature: float,
                 min_target_prob: float) -> tuple[torch.Tensor, dict[str, float]]:
        zero = student_logits.new_zeros((), dtype=torch.float32)
        features = batch.get("teacher_features")
        if features is None:
            return zero, {"kd_rows": 0.0}
        padding = batch["teacher_padding_mask"]
        points = (~padding).sum(1)
        token_lengths = (~batch["token_padding_mask"]).sum(1)
        eligible = (batch["has_strokes"] & points.gt(0) & points.le(self.max_points)
                    & token_lengths.le(self.max_tokens))
        rows = eligible.nonzero().squeeze(1)
        if not len(rows):
            return zero, {"kd_rows": 0.0}
        width = int(points[rows].max())
        length = min(int(token_lengths[rows].max()) - 1, student_logits.shape[1])
        tokens = batch["tokens"][rows]
        with torch.no_grad(), autocast(self.device, self.amp):
            teacher_logits = self.model(features[rows, :width], padding[rows, :width],
                                        self.teacher_input[tokens[:, :length]],
                                        batch["token_padding_mask"][rows, :length])
        teacher_logits = teacher_logits.float()[..., self.shared_teacher_ids]
        student_shared = student_logits[rows, :length].float()[..., self.shared_student_ids]
        targets = tokens[:, 1:length + 1]
        valid = targets.ne(self.pad_id)
        shared_targets = self.student_to_shared[targets]
        target_prob = teacher_logits.softmax(-1).gather(-1, shared_targets.clamp_min(0)[..., None]).squeeze(-1)
        mask = valid & shared_targets.ge(0) & target_prob.ge(min_target_prob)
        teacher_log = (teacher_logits / temperature).log_softmax(-1)
        student_log = (student_shared / temperature).log_softmax(-1)
        divergence = (teacher_log.exp() * (teacher_log - student_log)).sum(-1) * temperature ** 2
        count = mask.sum()
        loss = torch.where(mask, divergence, torch.zeros_like(divergence)).sum() / count.clamp_min(1)
        return loss, {"kd_loss": float(loss.detach()), "kd_coverage": float(count) / max(1, int(valid.sum())),
                      "kd_rows": float(len(rows))}


# --------------------------------------------------------------------------- scoring


def normalized_tokens(latex: str) -> list[str]:
    return canonicalize_latex_tokens(canonical(latex))


def structural_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token in STRUCTURE_TOKENS]


def normalized_distance(left: list[str], right: list[str]) -> float:
    return edit_distance(left, right) / max(1, len(left), len(right))


def standard_form(latex: str) -> str:
    """Single-standard label; malformed or non-ASCII text falls back to the legacy canonical form."""
    return standardize_latex(latex) or canonical(latex)


def reward_terms(prediction: str, reference: str, finished: bool, weights: dict[str, float],
                 standardize: bool = False) -> dict[str, float]:
    if standardize:
        prediction, reference = standard_form(prediction), standard_form(reference)
    predicted, target = tokenize_latex(prediction), tokenize_latex(reference)
    exact = float(finished and normalized_tokens(prediction) == normalized_tokens(reference))
    strict = float(finished and predicted == target)
    skeleton = 1.0 - normalized_distance(structural_tokens(predicted), structural_tokens(target))
    edit = min(1.0, edit_distance(predicted, target) / max(1, len(target)))
    total = (weights.get("exact_norm", 0.0) * exact + weights.get("strict_case_bonus", 0.0) * strict
             + weights.get("structure_skeleton", 0.0) * skeleton - weights.get("edit_penalty", 0.0) * edit)
    return {"reward": total, "exact": exact, "strict": strict, "skeleton": skeleton, "edit": edit}


def source_group(source: str) -> str:
    return source.split("_")[0]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"samples": 0}
    errors = sum(row["cer_errors"] for row in rows)
    total = sum(row["cer_tokens"] for row in rows)
    return {"samples": len(rows),
            "normalized_exact": sum(row["normalized_exact"] for row in rows) / len(rows),
            "strict_exact": sum(row["strict_exact"] for row in rows) / len(rows),
            "token_cer": errors / max(1, total),
            "avg_tokens": sum(row["tokens"] for row in rows) / len(rows),
            "truncated": sum(row["truncated"] for row in rows) / len(rows)}


def score_prediction(prediction: str, reference: str, finished: bool, standardize: bool = False) -> dict[str, Any]:
    if standardize:
        prediction, reference = standard_form(prediction), standard_form(reference)
    target = reference_tokens(reference)
    return {"normalized_exact": float(normalized_tokens(prediction) == normalized_tokens(reference)),
            "strict_exact": float(tokenize_latex(prediction) == tokenize_latex(reference)),
            "cer_errors": edit_distance(reference_tokens(prediction), target), "cer_tokens": len(target),
            "tokens": len(tokenize_latex(prediction)), "truncated": float(not finished)}


def build_val_loader(shards: Path, tokenizer: LatexTokenizer, batch_size: int,
                     standardize: bool = False) -> DataLoader:
    dataset = BimodalEvalDataset(shards, stage="val", tokenizer=tokenizer, standardize_labels=standardize)
    widths = [int(json.loads(record["json"]).get("image_width", 0)) for record in dataset.records]
    order = sorted(range(len(dataset)), key=lambda index: widths[index])
    return DataLoader(dataset, batch_size=batch_size, sampler=order,
                      collate_fn=partial(collate_bimodal_batch, pad_id=tokenizer.pad_id))


@torch.no_grad()
def evaluate_val(model: BimodalMathModel, loader: DataLoader, tokenizer: LatexTokenizer, device: torch.device,
                 amp_dtype: str | None, modality: str, max_gen_tokens: int,
                 standardize: bool = False) -> tuple[dict[str, Any], list[dict]]:
    started = time.perf_counter()
    model.eval()
    rows = []
    for batch in loader:
        moved = to_device(batch, device)
        with autocast(device, amp_dtype):
            ids = model.decode_greedy(**encoder_inputs(moved, modality), max_tokens=max_gen_tokens + 1)
        ids = ids.cpu()
        finished = ids.eq(tokenizer.eos_id).any(1).tolist()
        for row, sequence in enumerate(ids.tolist()):
            prediction = tokenizer.decode(sequence)
            reference = batch["normalized_latex"][row]
            rows.append({"id": batch["ids"][row], "source": batch["sources"][row], "prediction": prediction,
                         "reference": reference,
                         **score_prediction(prediction, reference, finished[row], standardize)})
    by_source = defaultdict(list)
    for row in rows:
        by_source[source_group(row["source"])].append(row)
    metrics = {**summarize_rows(rows), "by_source": {name: summarize_rows(group) for name, group in
                                                     sorted(by_source.items())},
               "modality": modality, "seconds": time.perf_counter() - started}
    return metrics, rows


def is_better(metrics: dict[str, Any], best: dict[str, Any] | None, primary: str = "normalized_exact") -> bool:
    if best is None:
        return True
    if metrics[primary] != best[primary]:
        return metrics[primary] > best[primary]
    return metrics["token_cer"] < best["token_cer"]


# --------------------------------------------------------------------------- decoding


def _top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    ordered, index = probs.sort(-1, descending=True)
    ordered = ordered.masked_fill(ordered.cumsum(-1) - ordered > top_p, 0.0)
    return torch.zeros_like(probs).scatter(-1, index, ordered)


@torch.no_grad()
def sample_sequences(decoder: torch.nn.Module, memory: torch.Tensor, memory_mask: torch.Tensor,
                     tokenizer: LatexTokenizer, max_new_tokens: int, *, group: int = 1, greedy: bool = False,
                     temperature: float = 1.0, top_p: float = 1.0, generator: torch.Generator | None = None,
                     block_special: bool = True) -> dict[str, torch.Tensor]:
    """Cached L2R decoding. ``tokens [N, T]`` include EOS; ``valid`` marks generated positions."""
    from research.incremental import DecoderState
    state = DecoderState(decoder, memory, memory_mask)
    if group > 1:
        state.repeat_beams(group)
    count = memory.shape[0] * group
    device = memory.device
    token = torch.full((count, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(count, dtype=torch.bool, device=device)
    min_prob = torch.ones(count, device=device)
    blocked = torch.tensor([tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id], device=device)
    tokens, logps, alive = [], [], []
    for _ in range(max_new_tokens):
        logits = state.step(token)[0][:, -1].float()
        if block_special:
            logits = logits.index_fill(1, blocked, -1e9)
        if greedy:
            log_probs = logits.log_softmax(-1)
            choice = log_probs.argmax(-1)
        else:
            log_probs = (logits / temperature).log_softmax(-1)
            choice = torch.multinomial(_top_p(log_probs.exp(), top_p), 1, generator=generator).squeeze(1)
        chosen = log_probs.gather(1, choice[:, None]).squeeze(1)
        alive.append(~finished)
        choice = choice.masked_fill(finished, tokenizer.pad_id)
        chosen = chosen.masked_fill(finished, 0.0)
        min_prob = torch.where(finished, min_prob, torch.minimum(min_prob, chosen.exp()))
        tokens.append(choice)
        logps.append(chosen)
        finished = finished | choice.eq(tokenizer.eos_id)
        token = choice[:, None]
        if bool(finished.all()):
            break
    valid = torch.stack(alive, 1)
    return {"tokens": torch.stack(tokens, 1), "logp": torch.stack(logps, 1), "valid": valid,
            "lengths": valid.sum(1), "finished": finished, "min_prob": min_prob}


def policy_distribution(logits: torch.Tensor, tokenizer: LatexTokenizer, temperature: float) -> torch.Tensor:
    blocked = torch.tensor([tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id], device=logits.device)
    # A finite sentinel keeps the KL difference defined on blocked tokens.
    return (logits.float() / temperature).index_fill(-1, blocked, -1e9).log_softmax(-1)


# --------------------------------------------------------------------------- training


class Accumulator:
    def __init__(self, state: dict[str, Any] | None = None):
        state = state or {}
        self.sums: dict[str, float] = dict(state.get("sums", {}))
        self.counts: dict[str, int] = dict(state.get("counts", {}))

    def add(self, values: dict[str, float]) -> None:
        for key, value in values.items():
            if value is None or not math.isfinite(float(value)):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def means(self) -> dict[str, float]:
        return {key: self.sums[key] / self.counts[key] for key in sorted(self.sums) if self.counts.get(key)}

    def state(self) -> dict[str, Any]:
        return {"sums": self.sums, "counts": self.counts}


class Trainer:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]):
        self.args, self.config = args, config
        self.stage = int(args.stage)
        self.cfg = dict(config.get(f"stage{self.stage}") or {})
        if self.stage == 3 and int(self.cfg.get("group_size", 8)) < 2:
            raise SystemExit("GRPO needs group_size >= 2")
        for key in ("batch_size", "grad_accum", "epochs", "num_workers"):
            if getattr(args, key, None) is not None:
                self.cfg[key] = getattr(args, key)
        if int(self.cfg.get("grad_accum", 1)) < 1:
            raise SystemExit(f"grad_accum must be >= 1, got {self.cfg['grad_accum']}")
        self.validation = dict(config.get("validation") or {})
        if getattr(args, "eval_every_steps", None) is not None:
            self.validation["eval_every_steps"] = args.eval_every_steps
        self.standardize = bool(getattr(args, "standardize_labels", False)
                                or self.cfg.get("standardize_labels", config.get("standardize_labels", False)))
        self.step_val: tuple[int, dict[str, Any]] | None = None
        runtime = dict(config.get("runtime") or {})
        self.seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
        self.device = resolve_device(args.device)
        self.amp = self.cfg.get("amp_dtype", "bfloat16")
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.shards = Path(args.shards)
        self.manifest = json.loads((self.shards / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = LatexTokenizer.load(self.shards / "tokenizer.json")
        self.tokenizer_sha = file_sha256(self.shards / "tokenizer.json")
        self.manifest_sha = file_sha256(self.shards / "manifest.json")
        self.max_wall = float(args.max_wall_minutes if args.max_wall_minutes is not None
                              else runtime.get("default_max_wall_minutes", 0))
        sync_every = args.sync_every_minutes if args.sync_every_minutes is not None \
            else runtime.get("sync_every_minutes", 20)
        self.sync = DriveSync(self.output, Path(args.drive_sync_dir) if args.drive_sync_dir else None, sync_every)
        self.hf: HFSync | None = None
        if args.hf_sync_repo:
            self.hf = HFSync(self.output, args.hf_sync_repo, args.hf_sync_prefix, sync_every,
                             getattr(args, "hf_sync_token", None))
            self.hf.ensure_repo()
        self.keep_last = int(runtime.get("keep_last_checkpoints", 2))
        self.session_start = time.monotonic()
        self.epoch = self.batch_index = self.global_step = self.skipped_steps = 0
        self.elapsed_base = 0.0
        self.best_metric: dict[str, Any] | None = None
        self.last_val: dict[str, Any] | None = None
        self.epoch_stats = Accumulator()
        self.reference_checkpoint: str | None = None
        self.ref_model: BimodalMathModel | None = None
        if getattr(args, "teacher_checkpoint", None):
            self.cfg["teacher_checkpoint"] = str(args.teacher_checkpoint)
        self.teacher: TeacherKD | None = None
        self.state = "starting"

    # ----------------------------------------------------------------- state

    def elapsed(self) -> float:
        return self.elapsed_base + time.monotonic() - self.session_start

    def wall_reached(self) -> bool:
        return self.max_wall > 0 and time.monotonic() - self.session_start >= self.max_wall * 60.0

    def write_status(self, state: str, **extra: Any) -> None:
        self.state = state
        atomic_json(self.output / "status.json", {
            "state": state, "stage": self.stage, "epoch": self.epoch, "batch_index": self.batch_index,
            "global_step": self.global_step, "epochs": int(self.cfg.get("epochs", 0)),
            "best_metric": self.best_metric, "last_val": self.last_val,
            "elapsed_seconds": round(self.elapsed(), 1), "updated_at": utc_now(), **extra})

    def find_resume(self) -> Path | None:
        value = self.args.resume
        if not value:
            return None
        if value != "auto":
            path = Path(value)
            if not path.is_file():
                raise SystemExit(f"--resume {path} does not exist")
            return path
        local = self.output / "recovery.pt"
        remote = self.sync.target / "recovery.pt" if self.sync.target else None
        candidates = [path for path in (local, remote) if path is not None and path.is_file()]
        if not candidates and self.hf is not None and self.hf.restore_run(self.output):
            log(f"resume: restored run artifacts from {self.hf.url}")
            return local
        if not candidates:
            log("resume: no recovery.pt found, starting fresh")
            return None
        chosen = max(candidates, key=lambda path: path.stat().st_mtime)
        if chosen == remote:
            for name in SYNC_FILES:
                source, destination = remote.parent / name, self.output / name
                if source.is_file() and (not destination.is_file()
                                         or destination.stat().st_mtime < source.stat().st_mtime):
                    atomic_copy(source, destination)
            chosen = local
            log(f"resume: restored run artifacts from {remote.parent}")
        return chosen

    # ----------------------------------------------------------------- setup

    def fetch_init(self, resuming: bool) -> None:
        path = self.args.hf_init_path
        if not path or (self.args.init_from and Path(self.args.init_from).is_file()):
            return
        if resuming and self.stage != 3:
            return
        destination = self.output / "_init_from.pt"
        try:
            self.args.init_from = self.hf.download_file(path, destination)
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"--hf-init-path: cannot download {path} from {self.hf.repo} ({error})") from error
        log(f"init-from: downloaded hf://datasets/{self.hf.repo}/{path} -> {destination}")

    def setup(self) -> None:
        resume = self.find_resume()
        payload = load_payload(resume) if resume else None
        self.fetch_init(payload is not None)
        init = None
        if payload is not None:
            if payload.get("format") != RECOVERY_FORMAT or int(payload["stage"]) != self.stage:
                raise SystemExit(f"{resume} is not a stage {self.stage} recovery file")
            if payload.get("tokenizer_sha256") != self.tokenizer_sha:
                raise SystemExit("tokenizer.json differs from the one this run was started with")
            model_config = BimodalModelConfig.from_dict({**payload["model_config"], "vision_pretrained": False})
        else:
            if self.args.init_from:
                init = load_payload(self.args.init_from)
                if init.get("tokenizer_sha256") not in (None, self.tokenizer_sha):
                    log("init-from: checkpoint tokenizer differs from shards tokenizer.json")
            elif self.stage == 3:
                raise SystemExit("Stage 3 needs --init-from <stage2 best.pt>")
            if self.stage == 3:
                model_config = BimodalModelConfig.from_dict({**init["model_config"], "vision_pretrained": False})
            else:
                pretrained = not self.args.no_pretrained and init is None
                model_config = build_model_config(self.config, self.tokenizer, self.manifest, self.stage,
                                                  self.args.preset, pretrained)
                if model_config.vision_pretrained:
                    log(f"vision: downloading pretrained {model_config.vision_backbone}")
        if model_config.vocab_size != len(self.tokenizer):
            raise SystemExit(f"Model vocabulary {model_config.vocab_size} != tokenizer {len(self.tokenizer)}")
        seed_everything(self.seed)
        model = BimodalMathModel(model_config)
        self.init_report = None
        if payload is not None:
            model.load_state_dict(payload["model_state_dict"])
        elif init is not None:
            state, key = checkpoint_weights(init)
            self.init_report = load_partial(model, state, f"init-from {self.args.init_from} [{key}]")
        self.model = model.to(self.device)
        if self.freeze_vision:
            self.model.vision_encoder.requires_grad_(False)
            log("vision: encoder frozen (no gradients, eval mode)")
        self.model_config = model_config
        self.ema = EMA(self.model, float(self.cfg.get("ema_decay", 0.999)))
        if payload is not None:
            self.ema.module.load_state_dict(payload["ema_state_dict"])
            self.ema.updates = int(payload.get("ema_updates", 0))

        if self.stage == 3:
            self.reference_checkpoint = str(self.args.init_from) if self.args.init_from else \
                payload.get("reference_checkpoint")
            if not self.reference_checkpoint or not Path(self.reference_checkpoint).is_file():
                raise SystemExit(f"Stage 3 reference checkpoint missing: {self.reference_checkpoint}")
            reference = init if init is not None else load_payload(self.reference_checkpoint)
            self.ref_model = model_from_payload(reference, self.device).eval().requires_grad_(False)
            if self.device.type == "cuda":
                self.ref_model.to(dtype=torch.bfloat16)

        teacher_path = self.cfg.get("teacher_checkpoint")
        if teacher_path and self.stage == 2 and max(self.kd_weight(False), self.kd_weight(True)) > 0:
            self.teacher = TeacherKD.load(teacher_path, self.tokenizer, self.device, self.amp)
            log(f"teacher: {teacher_path} {json.dumps(self.teacher.summary())}")
        elif teacher_path:
            log(f"teacher: {teacher_path} unused (stage {self.stage}, teacher_kd_weight 0)")

        lr = float(self.cfg.get("lr", 1e-4))
        vision_lr = float(self.cfg.get("vision_lr", lr))
        stroke_lr = float(self.cfg["stroke_lr"]) if self.cfg.get("stroke_lr") is not None else None
        groups = parameter_groups(self.model, lr, vision_lr, float(self.cfg.get("weight_decay", 0.0)), stroke_lr)
        log("optimizer: " + ", ".join(f"{group['name']} lr={group['base_lr']:.1e} ({len(group['params'])})"
                                      for group in groups))
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8, fused=self.device.type == "cuda")
        if payload is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.schedule = Schedule(**payload["scheduler_state_dict"])
            self.epoch, self.batch_index = int(payload["epoch"]), int(payload["batch_index"])
            self.global_step, self.elapsed_base = int(payload["global_step"]), float(payload["elapsed_seconds"])
            self.skipped_steps = int(payload.get("skipped_steps", 0))
            self.best_metric, self.last_val = payload.get("best_metric"), payload.get("last_val")
            self.epoch_stats = Accumulator(payload.get("epoch_stats"))
            if int(payload.get("num_workers", self.workers)) != self.workers and self.batch_index:
                log("resume: worker count changed; the mid-epoch skip is approximate")
            if (int(payload.get("batch_size", 0)), int(payload.get("grad_accum", 1))) != (
                    int(self.cfg["batch_size"]), int(self.cfg.get("grad_accum", 1))):
                log("resume: batch_size/grad_accum changed; the LR schedule keeps its original step count")
            restore_rng(payload["rng"], self.device)
            log(f"resume: stage {self.stage} epoch {self.epoch} batch {self.batch_index} "
                f"step {self.global_step} from {resume}")
        else:
            self.schedule = self.build_schedule()
            metrics = self.output / "metrics.jsonl"
            if metrics.is_file():
                metrics.rename(self.output / f"metrics.{int(time.time())}.jsonl")
        self.compiled = maybe_compile(self.model, self.args.compile, self.device)
        self.val_loader = build_val_loader(self.shards, self.tokenizer, int(self.validation.get("batch_size", 64)),
                                           self.standardize)
        summary = self.model.parameter_summary()
        log(f"model {model_config.preset}: {json.dumps(summary)}")
        atomic_json(self.output / "config_resolved.json", {
            "created_at": utc_now(), "stage": self.stage, "argv": redacted_argv(sys.argv),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()
                     if not key.endswith("token")},
            "seed": self.seed, "stage_config": self.cfg, "validation": self.validation,
            "runtime": self.config.get("runtime"), "model_config": model_config.to_dict(),
            "parameters": summary, "schedule": self.schedule.to_dict(), "device": str(self.device),
            "torch": torch.__version__, "compiled": self.compiled, "init_from": str(self.args.init_from or ""),
            "init_report": {key: value if isinstance(value, int) else len(value)
                            for key, value in (self.init_report or {}).items()},
            "reference_checkpoint": self.reference_checkpoint, "standardize_labels": self.standardize,
            "teacher_checkpoint": str(self.cfg["teacher_checkpoint"]) if self.cfg.get("teacher_checkpoint") else None,
            "teacher": self.teacher.summary() if self.teacher is not None else None,
            "tokenizer_sha256": self.tokenizer_sha, "manifest_sha256": self.manifest_sha})

    @property
    def workers(self) -> int:
        return int(self.cfg.get("num_workers", 0))

    @property
    def freeze_vision(self) -> bool:
        return bool(self.cfg.get("freeze_vision", False))

    @property
    def log_modalities(self) -> bool:
        return self.stage >= 2 and bool(self.validation.get("log_modalities", True))

    def set_train_mode(self) -> None:
        self.model.train(self.stage != 3)
        if self.freeze_vision:
            self.model.vision_encoder.eval()

    @property
    def epochs(self) -> int:
        return int(self.cfg.get("epochs", 1))

    @property
    def cooldown_epochs(self) -> int:
        return int(self.cfg.get("cooldown_epochs", 0)) if self.stage == 2 else 0

    def is_cooldown(self, epoch: int) -> bool:
        return self.cooldown_epochs > 0 and epoch >= self.epochs - self.cooldown_epochs

    def kd_weight(self, cooldown: bool) -> float:
        override = self.cfg.get("cooldown_teacher_kd_weight") if cooldown else None
        return float(self.cfg.get("teacher_kd_weight", 0.0) if override is None else override)

    def augment_config(self, cooldown: bool) -> BimodalAugmentConfig | None:
        values = self.cfg.get("augment")
        if cooldown and self.cfg.get("cooldown_augment") is not None:
            values = {**(values or {}), **self.cfg["cooldown_augment"]}
        if values is None:
            return None
        config = BimodalAugmentConfig.from_dict(values)
        if cooldown and self.cfg.get("cooldown_disable_band_mask"):
            config.vision_band_mask_prob = 0.0
        return config

    def estimate_samples(self, clean: bool) -> int:
        if self.stage == 3:
            hard = json.loads((self.shards / "hard_manifest.json").read_text(encoding="utf-8")) \
                if (self.shards / "hard_manifest.json").is_file() else {}
            sources = hard.get("sources") or {}
            repeats = {}
        else:
            entry = self.manifest.get("stages", {}).get(f"stage{self.stage}", {})
            sources = entry.get("sources") or {}
            if not sources:
                return max(1, sum(int(shard.get("samples", 0)) for shard in entry.get("shards", [])))
            hint = float((self.manifest.get("args") or {}).get("crohme_upsample", 1.0))
            repeats = {"crohme_train": hint} if self.stage == 1 and self.cfg.get("use_upsample_hint") else {}
        if self.cfg.get("source_upsample"):
            repeats = {str(source): float(factor) for source, factor in self.cfg["source_upsample"].items()}
        allowed = set(self.cfg.get("allowed_sources") or sources)
        total = sum(count * repeats.get(source, 1.0) for source, count in sources.items()
                    if not (clean and source in SYNTHETIC_SOURCES) and source in allowed)
        return max(1, int(total))

    def build_schedule(self) -> Schedule:
        per_update = int(self.cfg["batch_size"]) * int(self.cfg.get("grad_accum", 1))
        main_epochs = max(1, self.epochs - self.cooldown_epochs)
        main_steps = main_epochs * math.ceil(self.estimate_samples(False) / per_update)
        if self.args.max_steps:
            main_steps = min(main_steps, int(self.args.max_steps))
        lr = float(self.cfg.get("lr", 1e-4))
        cooldown = float(self.cfg["cooldown_lr"]) / lr if self.cooldown_epochs and "cooldown_lr" in self.cfg else None
        floor = cooldown if cooldown is not None else float(self.cfg.get("min_lr_ratio", 1.0 if self.stage == 3
                                                                           else 0.05))
        warmup = int(float(self.cfg.get("warmup_ratio", 0.0)) * main_steps)
        return Schedule(warmup, max(1, main_steps), floor, cooldown)

    # ----------------------------------------------------------------- checkpoints

    def recovery_payload(self) -> dict[str, Any]:
        return {
            "format": RECOVERY_FORMAT, "stage": self.stage, "epoch": self.epoch, "batch_index": self.batch_index,
            "global_step": self.global_step, "elapsed_seconds": self.elapsed(),
            "model_config": self.model_config.to_dict(), "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.module.state_dict(), "ema_updates": self.ema.updates,
            "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.schedule.to_dict(),
            "best_metric": self.best_metric, "last_val": self.last_val, "tokenizer_sha256": self.tokenizer_sha,
            "manifest_sha256": self.manifest_sha, "rng": rng_state(self.device), "num_workers": self.workers,
            "batch_size": int(self.cfg["batch_size"]), "grad_accum": int(self.cfg.get("grad_accum", 1)),
            "epoch_stats": self.epoch_stats.state(), "skipped_steps": self.skipped_steps,
            "reference_checkpoint": self.reference_checkpoint, "train_config": self.cfg, "saved_at": utc_now(),
        }

    def weights_payload(self, metrics: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT, "stage": self.stage, "epoch": self.epoch, "global_step": self.global_step,
            "model_config": self.model_config.to_dict(), "model_state_dict": self.ema.module.state_dict(),
            "weights": "ema", "metrics": metrics, "tokenizer": self.tokenizer.to_dict(),
            "tokenizer_sha256": self.tokenizer_sha, "manifest_sha256": self.manifest_sha, "saved_at": utc_now(),
        }

    def save_recovery(self) -> None:
        atomic_torch_save(self.output / "recovery.pt", self.recovery_payload())

    def save_snapshot(self, epoch: int, metrics: dict[str, Any] | None) -> None:
        if self.keep_last <= 0:
            return
        atomic_torch_save(self.output / f"epoch-{epoch:03d}.pt", self.weights_payload(metrics))
        for stale in sorted(self.output.glob("epoch-*.pt"))[:-self.keep_last]:
            stale.unlink(missing_ok=True)

    def sync_due(self) -> bool:
        return self.sync.due() or (self.hf is not None and self.hf.due())

    def sync_now(self, force: bool = True) -> None:
        if force or self.sync.due():
            self.sync.now()
        if self.hf is not None:
            self.hf.sync(self.output, force=force)

    def checkpoint_and_sync(self, state: str = "running", force: bool = True, **extra: Any) -> None:
        self.save_recovery()
        self.write_status(state, **extra)
        self.sync_now(force)

    # ----------------------------------------------------------------- data

    def make_loader(self, epoch: int, cooldown: bool, skip: int) -> DataLoader:
        batch_size = int(self.cfg["batch_size"])
        if self.stage == 3:
            source, stage = self.shards / HARD_DIRECTORY, "stage2"
            if not source.is_dir():
                raise SystemExit(f"{source} missing; run --mine-hard first")
            max_tokens = int(self.cfg.get("max_gen_tokens", 160)) + 1
        else:
            source, stage = self.shards, f"stage{self.stage}"
            max_tokens = self.cfg.get("max_tokens", 256)
        dataset = ResumableShardDataset(
            source, stage=stage, tokenizer=self.tokenizer, train=True, seed=self.seed,
            shuffle_buffer=int(self.cfg.get("shuffle_buffer", 1024)),
            clean_human_only=cooldown or bool(self.cfg.get("clean_human_only", False)),
            max_tokens=max_tokens, use_upsample_hint=bool(self.cfg.get("use_upsample_hint", False)),
            allowed_sources=self.cfg.get("allowed_sources"), source_upsample=self.cfg.get("source_upsample"),
            standardize_labels=self.standardize, skip_batches=skip, batch_size=batch_size,
            loader_workers=self.workers, augment=self.augment_config(cooldown),
            teacher_preprocessing=self.teacher.preprocessing
            if self.teacher is not None and self.kd_weight(cooldown) > 0 else None)
        dataset.set_epoch(epoch)
        options: dict[str, Any] = {}
        if self.workers:
            options.update(prefetch_factor=4, persistent_workers=False)
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.workers,
                          collate_fn=partial(collate_bimodal_batch, pad_id=self.tokenizer.pad_id),
                          pin_memory=self.device.type == "cuda", **options)

    def prepare(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.compiled:
            batch = align_batch(batch, 8, 64, self.tokenizer.pad_id)
        return to_device(batch, self.device)

    # ----------------------------------------------------------------- objectives

    def supervised_step(self, batch: dict[str, Any], cooldown: bool) -> tuple[torch.Tensor, dict[str, float]]:
        cfg, pad = self.cfg, self.tokenizer.pad_id
        modality = cfg.get("modality_mode", "auto" if self.stage > 1 else "vision")
        tokens = batch["tokens"]
        r2l_weight = float(cfg.get("r2l_weight", 0.0))
        use_r2l = self.model.r2l_decoder is not None and r2l_weight > 0
        weights = {key: float(value) for key, value in (cfg.get("structure_loss_weights") or {}).items()}
        symbols = self.model_config.structure_symbols or []
        relations = self.model_config.structure_relations or []
        records = [structure_record(targets, len(symbols), len(relations)) if symbols else None
                   for targets in batch["structure_targets"]]
        need_structure = (bool(symbols) and any(weights.values()) and modality != "vision"
                          and batch.get("stroke_features") is not None and any(records))
        inputs = encoder_inputs(batch, modality)
        prefix = "cooldown_" if cooldown else ""
        vision_drop = float(cfg.get(f"{prefix}vision_drop_prob", cfg.get("vision_drop_prob", 0.0)))
        stroke_drop = float(cfg.get(f"{prefix}stroke_drop_prob", cfg.get("stroke_drop_prob", 0.0)))
        with autocast(self.device, self.amp):
            out = self.model(
                tokens[:, :-1], batch["token_padding_mask"][:, :-1],
                batch["r2l_tokens"][:, :-1] if use_r2l else None,
                batch["r2l_token_padding_mask"][:, :-1] if use_r2l else None,
                **inputs, modality_mode=modality,
                vision_drop_prob=vision_drop if modality == "auto" else 0.0,
                stroke_drop_prob=stroke_drop if modality == "auto" else 0.0,
                return_structure=need_structure)
        smoothing = float(cfg.get("label_smoothing", 0.0))
        ce = sequence_cross_entropy(out["logits"], tokens[:, 1:], pad, smoothing)
        loss = ce
        stats = {"ce": float(ce.detach())}
        kd_weight = self.kd_weight(cooldown)
        if self.teacher is not None and kd_weight > 0:
            kd, kd_stats = self.teacher(batch, out["logits"], float(cfg.get("teacher_kd_temp", 2.0)),
                                        float(cfg.get("teacher_kd_min_target_prob", 0.05)))
            loss = loss + kd_weight * kd
            stats.update(kd_stats)
        if use_r2l:
            r2l = sequence_cross_entropy(out["r2l_logits"], batch["r2l_tokens"][:, 1:], pad, smoothing)
            loss = loss + r2l_weight * r2l
            stats["r2l_ce"] = float(r2l.detach())
        structure = out.get("structure")
        if need_structure and structure is not None:
            mask = structure["stroke_mask"]
            aligned, misaligned = [], 0
            for position, row in enumerate(structure["batch_index"].tolist()):
                record = records[row]
                if record is not None and len(record["stroke_groups"]) != int((~mask[position]).sum()):
                    record, misaligned = None, misaligned + 1
                aligned.append(record)
            if any(record is not None for record in aligned):
                auxiliary, parts = structure_loss(structure, aligned, len(symbols), weights)
                loss = loss + auxiliary
                stats["structure"] = float(auxiliary.detach())
                stats.update({f"structure_{key}": float(parts[key].detach()) for key in weights if key in parts})
                stats["structure_rows"] = sum(record is not None for record in aligned)
            if misaligned:
                stats["structure_misaligned"] = misaligned
        if bool(batch["has_strokes"].any()):
            stats["ink_rows"] = float(batch["has_strokes"].float().mean())
        stats["loss"] = float(loss.detach())
        return loss, stats

    def grpo_step(self, batch: dict[str, Any], micro: int) -> tuple[torch.Tensor, dict[str, float]]:
        cfg, tokenizer = self.cfg, self.tokenizer
        group = int(cfg.get("group_size", 8))
        temperature = float(cfg.get("temperature", 1.0))
        clip = float(cfg.get("clip_eps", 0.2))
        weights = {**DEFAULT_REWARD_WEIGHTS, **(cfg.get("reward_weights") or {})}
        inputs = encoder_inputs(batch, "auto")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed * 1_000_003 + self.global_step * 131 + micro)
        with autocast(self.device, self.amp), torch.no_grad():
            memory, memory_mask, _ = self.model.encode_bimodal(**inputs)
            rollout = sample_sequences(self.model.decoder, memory, memory_mask, tokenizer,
                                       int(cfg.get("max_gen_tokens", 160)), group=group, temperature=temperature,
                                       top_p=float(cfg.get("top_p", 1.0)), generator=generator)
        sequences, lengths = rollout["tokens"], rollout["lengths"]
        rows = sequences.shape[0]
        ids, finished = sequences.cpu().tolist(), rollout["finished"].cpu().tolist()
        counts = lengths.cpu().tolist()
        terms, cache = [], {}
        for index in range(rows):
            prompt = index // group
            key = (prompt, tuple(ids[index][:counts[index]]))
            if key not in cache:
                cache[key] = reward_terms(tokenizer.decode(key[1]), batch["normalized_latex"][prompt],
                                          finished[index], weights, self.standardize)
            terms.append(cache[key])
        rewards = torch.tensor([term["reward"] for term in terms], device=self.device).view(-1, group)
        spread = rewards.std(1, unbiased=False, keepdim=True)
        varied = spread.gt(1e-6)
        advantages = torch.where(varied, (rewards - rewards.mean(1, keepdim=True)) / (spread + 1e-6),
                                 torch.zeros_like(rewards)).flatten()

        valid = rollout["valid"]
        bos = torch.full((rows, 1), tokenizer.bos_id, dtype=torch.long, device=self.device)
        decoder_input = torch.cat((bos, sequences[:, :-1]), 1)
        with autocast(self.device, self.amp):
            memory, memory_mask, _ = self.model.encode_bimodal(**inputs)
            logits = self.model.decoder(decoder_input, memory.repeat_interleave(group, 0),
                                        memory_mask.repeat_interleave(group, 0), ~valid)
            with torch.no_grad():
                ref_memory, ref_mask, _ = self.ref_model.encode_bimodal(**inputs)
                ref_logits = self.ref_model.decoder(decoder_input, ref_memory.repeat_interleave(group, 0),
                                                    ref_mask.repeat_interleave(group, 0), ~valid)
        log_probs = policy_distribution(logits, tokenizer, temperature)
        ref_log_probs = policy_distribution(ref_logits, tokenizer, temperature)
        chosen = log_probs.gather(-1, sequences[..., None]).squeeze(-1)
        ratio = torch.exp(torch.where(valid, chosen - rollout["logp"], torch.zeros_like(chosen)))
        advantage = advantages[:, None]
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
        denominator = lengths.clamp_min(1).to(surrogate.dtype)
        policy = -(torch.where(valid, surrogate, torch.zeros_like(surrogate)).sum(1) / denominator).mean()
        token_kl = (log_probs.exp() * (log_probs - ref_log_probs)).sum(-1)
        kl = (torch.where(valid, token_kl, torch.zeros_like(token_kl)).sum(1) / denominator).mean()
        loss = policy + float(cfg.get("kl_beta", 0.0)) * kl
        stats = {"policy_loss": float(policy.detach()), "kl": float(kl.detach()),
                 "reward": float(rewards.mean()), "reward_std": float(spread.mean()),
                 "varied_groups": float(varied.float().mean()),
                 "rollout_exact": sum(term["exact"] for term in terms) / rows,
                 "rollout_strict": sum(term["strict"] for term in terms) / rows,
                 "rollout_tokens": float(lengths.float().mean()),
                 "clip_fraction": float(((ratio - 1).abs() > clip).float()[valid].mean()) if bool(valid.any()) else 0}
        # Mined shards mark every row with stage3_role; teacher forcing then touches clean anchors only,
        # never the mined mistakes. Unmarked shards fall back to the human rows.
        roles = batch.get("has_stage3_role")
        if roles is not None and bool(roles.any()):
            anchors = batch["is_anchor"].nonzero().squeeze(1)
        else:
            anchors = (~batch["synthetic"]).nonzero().squeeze(1)
        anchor_weight = float(cfg.get("sft_anchor_weight", 0.0))
        r2l_weight = float(cfg.get("r2l_anchor_weight", 0.0)) if self.model.r2l_decoder is not None else 0.0
        if len(anchors) and (anchor_weight > 0 or r2l_weight > 0):
            anchor_memory, anchor_mask = memory[anchors], memory_mask[anchors]
            stats["anchor_rows"] = float(len(anchors))
            if anchor_weight > 0:
                tokens = batch["tokens"][anchors]
                with autocast(self.device, self.amp):
                    anchor_logits = self.model.decoder(tokens[:, :-1], anchor_memory, anchor_mask,
                                                       batch["token_padding_mask"][anchors, :-1])
                anchor = sequence_cross_entropy(anchor_logits, tokens[:, 1:], tokenizer.pad_id)
                loss = loss + anchor_weight * anchor
                stats["anchor_ce"] = float(anchor.detach())
            if r2l_weight > 0:
                r2l_tokens = batch["r2l_tokens"][anchors]
                with autocast(self.device, self.amp):
                    r2l_logits = self.model.r2l_decoder(r2l_tokens[:, :-1], anchor_memory, anchor_mask,
                                                        batch["r2l_token_padding_mask"][anchors, :-1])
                r2l = sequence_cross_entropy(r2l_logits, r2l_tokens[:, 1:], tokenizer.pad_id)
                loss = loss + r2l_weight * r2l
                stats["r2l_anchor_ce"] = float(r2l.detach())
        stats["loss"] = float(loss.detach())
        return loss, stats

    # ----------------------------------------------------------------- loop

    def optimizer_step(self, cooldown: bool) -> None:
        factor = self.schedule.factor(self.global_step, cooldown)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        parameters = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(parameters, float(self.cfg.get("grad_clip", 1.0)))
        if torch.isfinite(norm):
            self.optimizer.step()
            self.ema.update(self.model)
        else:
            self.skipped_steps += 1
            log(f"step {self.global_step}: non-finite gradient norm, update skipped")
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        self.last_grad_norm = float(norm)

    def validate(self) -> dict[str, Any]:
        modality = "vision" if self.stage == 1 else "auto"
        max_gen = int(self.validation.get("max_gen_tokens", 256))
        metrics, _ = evaluate_val(self.ema.module, self.val_loader, self.tokenizer, self.device, self.amp,
                                  modality, max_gen, self.standardize)
        if self.log_modalities:
            metrics["modalities"] = {}
            for name in ("vision", "stroke"):
                try:
                    single, _ = evaluate_val(self.ema.module, self.val_loader, self.tokenizer, self.device,
                                             self.amp, name, max_gen, self.standardize)
                except ValueError as error:
                    log(f"val {name}-only skipped ({error})")
                    continue
                metrics["modalities"][name] = {key: single[key] for key in (
                    "normalized_exact", "strict_exact", "token_cer", "by_source", "seconds")}
        return metrics

    def accept_val(self, val: dict[str, Any], epoch: int, label: str) -> bool:
        self.last_val = {key: val[key] for key in ("normalized_exact", "strict_exact", "token_cer", "avg_tokens")}
        improved = is_better(val, self.best_metric, self.validation.get("primary_metric", "normalized_exact"))
        if improved:
            self.best_metric = {**self.last_val, "epoch": epoch, "global_step": self.global_step}
            atomic_torch_save(self.output / "best.pt", self.weights_payload(val))
        by_source = " ".join(f"{name}={value['normalized_exact']:.3f}" for name, value in val["by_source"].items())
        modalities = " ".join(f"{name}-only={value['normalized_exact']:.3f}"
                              for name, value in (val.get("modalities") or {}).items())
        log(f"val {label}: normalized_exact={val['normalized_exact']:.4f} strict={val['strict_exact']:.4f} "
            f"token_cer={val['token_cer']:.4f} [{by_source}]{f' [{modalities}]' if modalities else ''} "
            f"{val['seconds']:.1f}s{' *best*' if improved else ''}")
        return improved

    def step_validate(self, epoch: int) -> None:
        val = self.validate()
        self.step_val = (self.global_step, val)
        improved = self.accept_val(val, epoch, f"step {self.global_step}")
        append_jsonl(self.output / "metrics.jsonl", {
            "type": "step_val", "stage": self.stage, "epoch": epoch, "global_step": self.global_step,
            "improved": improved, "val": val, "train": self.epoch_stats.means(),
            "lr": [group["lr"] for group in self.optimizer.param_groups],
            "elapsed_seconds": round(self.elapsed(), 1), "time": utc_now()})

    def end_epoch(self, epoch: int, cooldown: bool, partial_epoch: bool) -> dict[str, Any]:
        every = max(1, int(self.validation.get("eval_every_epochs", 1)))
        val = None
        if (epoch + 1) % every == 0 or epoch + 1 >= self.epochs or partial_epoch:
            if self.step_val is not None and self.step_val[0] == self.global_step:
                val = self.step_val[1]
            else:
                val = self.validate()
                self.accept_val(val, epoch, f"epoch {epoch}")
        record = {"type": "epoch", "stage": self.stage, "epoch": epoch, "global_step": self.global_step,
                  "cooldown": cooldown, "partial": partial_epoch, "train": self.epoch_stats.means(),
                  "train_samples": int(self.epoch_stats.sums.get("samples", 0)), "val": val,
                  "lr": [group["lr"] for group in self.optimizer.param_groups],
                  "skipped_steps": self.skipped_steps, "elapsed_seconds": round(self.elapsed(), 1), "time": utc_now()}
        if self.device.type == "cuda":
            record["peak_memory_gib"] = round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2)
        append_jsonl(self.output / "metrics.jsonl", record)
        self.save_snapshot(epoch, val)
        return record

    def train_loop(self) -> int:
        grad_accum = max(1, int(self.cfg.get("grad_accum", 1)))
        max_steps = int(self.args.max_steps or 0)
        log_every = max(1, int(self.args.log_every))
        eval_every = max(0, int(self.validation.get("eval_every_steps", 0) or 0))
        if self.epoch >= self.epochs:
            log("training already completed")
            self.write_status("completed")
            self.sync_now()
            return 0
        self.write_status("running")
        self.sync_now()
        self.optimizer.zero_grad(set_to_none=True)
        stop = None
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            cooldown = self.is_cooldown(epoch)
            if self.batch_index == 0:
                self.epoch_stats = Accumulator()
            log(f"stage {self.stage} epoch {epoch + 1}/{self.epochs}{' (cooldown: clean human only)' if cooldown else ''}"
                f"{f', resuming after batch {self.batch_index}' if self.batch_index else ''}")
            loader = self.make_loader(epoch, cooldown, self.batch_index)
            self.set_train_mode()
            pending, window_start, window_samples = 0, time.perf_counter(), 0
            for batch in loader:
                batch = self.prepare(batch)
                if self.stage == 3:
                    loss, stats = self.grpo_step(batch, self.batch_index)
                else:
                    loss, stats = self.supervised_step(batch, cooldown)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {self.global_step}: {stats}")
                (loss / grad_accum).backward()
                stats["samples"] = len(batch["ids"])
                self.epoch_stats.add(stats)
                self.batch_index += 1
                pending += 1
                window_samples += len(batch["ids"])
                if pending < grad_accum:
                    continue
                self.optimizer_step(cooldown)
                pending = 0
                if self.global_step % log_every == 0:
                    rate = window_samples / max(1e-6, time.perf_counter() - window_start)
                    shown = " ".join(f"{key}={value:.4f}" for key, value in stats.items()
                                     if key in ("loss", "ce", "r2l_ce", "structure", "kd_loss", "kd_coverage",
                                                "reward", "reward_std", "kl", "anchor_ce", "r2l_anchor_ce"))
                    rates = "/".join(f"{group['lr']:.2e}" for group in self.optimizer.param_groups
                                     if not group["name"].endswith("no_decay")) \
                        or f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    log(f"step {self.global_step} {shown} lr={rates} "
                        f"gnorm={self.last_grad_norm:.3f} {rate:.1f} samples/s")
                    window_start, window_samples = time.perf_counter(), 0
                if eval_every and self.global_step % eval_every == 0:
                    self.step_validate(epoch)
                    self.set_train_mode()
                    window_start, window_samples = time.perf_counter(), 0
                if max_steps and self.global_step >= max_steps:
                    stop = "max_steps"
                    break
                if self.wall_reached():
                    stop = "wall"
                    break
                if self.sync_due():
                    self.checkpoint_and_sync(force=False)
            if stop == "wall":
                log(f"wall limit {self.max_wall:g} min reached at step {self.global_step}; pausing")
                self.checkpoint_and_sync("paused_wall_limit", reason="max_wall_minutes")
                return 0
            if pending:
                self.optimizer_step(cooldown)
            del loader
            self.end_epoch(epoch, cooldown, partial_epoch=stop == "max_steps")
            if stop == "max_steps":
                self.checkpoint_and_sync("completed", reason="max_steps")
                log(f"max steps {max_steps} reached")
                return 0
            self.epoch, self.batch_index = epoch + 1, 0
            if self.epoch >= self.epochs:
                break
            if self.wall_reached():
                log(f"wall limit {self.max_wall:g} min reached after epoch {epoch}; pausing")
                self.checkpoint_and_sync("paused_wall_limit", reason="max_wall_minutes")
                return 0
            self.checkpoint_and_sync()
        self.checkpoint_and_sync("completed")
        log(f"stage {self.stage} completed: best {self.best_metric}")
        return 0

    def run(self) -> int:
        try:
            self.setup()
            code = self.train_loop()
        except BaseException as error:
            try:
                self.write_status("failed", error=f"{type(error).__name__}: {error}")
                self.sync_now()
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            if self.hf is not None and self.args.hf_squash_history and self.hf.healthy:
                self.hf.squash()
        if self.hf is not None and not self.hf.healthy:
            log(f"hf-sync ERROR: final upload failed; run artifacts are only in {self.output} (exit {SYNC_FAILED_EXIT})")
            return SYNC_FAILED_EXIT
        return code


# --------------------------------------------------------------------------- eval-only and mining


class Reservoir:
    def __init__(self, size: int, rng: random.Random):
        self.size, self.rng, self.seen, self.items = max(0, size), rng, 0, []

    def add(self, item: Any) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
        elif self.size and (slot := self.rng.randrange(self.seen)) < self.size:
            self.items[slot] = item


def checkpoint_modality(payload: dict[str, Any], override: str | None) -> str:
    return override or ("vision" if int(payload.get("stage", 2)) == 1 else "auto")


def eval_only(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if not args.init_from:
        raise SystemExit("--eval-only needs --init-from <checkpoint>")
    device = resolve_device(args.device)
    payload = load_payload(args.init_from)
    model = model_from_payload(payload, device).eval()
    shards = Path(args.shards)
    tokenizer = LatexTokenizer.load(shards / "tokenizer.json")
    validation = dict(config.get("validation") or {})
    standardize = bool(args.standardize_labels or config.get("standardize_labels", False))
    loader = build_val_loader(shards, tokenizer, int(args.batch_size or validation.get("batch_size", 64)), standardize)
    modality = checkpoint_modality(payload, args.eval_modality)
    amp = (config.get(f"stage{payload.get('stage', 2)}") or {}).get("amp_dtype", "bfloat16")
    metrics, rows = evaluate_val(model, loader, tokenizer, device, amp, modality,
                                 int(validation.get("max_gen_tokens", 256)), standardize)
    metrics.update(checkpoint=str(args.init_from), stage=payload.get("stage"), epoch=payload.get("epoch"),
                   global_step=payload.get("global_step"), standardize_labels=standardize)
    output = Path(args.output) if args.output else Path(args.init_from).parent
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "eval_metrics.json", metrics)
    with open(output / "eval_predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


def mine_hard(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if not args.init_from:
        raise SystemExit("--mine-hard needs --init-from <stage2 best.pt>")
    started = time.perf_counter()
    device = resolve_device(args.device)
    payload = load_payload(args.init_from)
    model = model_from_payload(payload, device).eval()
    shards = Path(args.shards)
    tokenizer = LatexTokenizer.load(shards / "tokenizer.json")
    stage3 = dict(config.get("stage3") or {})
    max_gen = int(stage3.get("max_gen_tokens", 160))
    batch_size = int(args.mine_batch_size or args.batch_size or (config.get("validation") or {}).get("batch_size", 64))
    workers = int(args.num_workers if args.num_workers is not None else stage3.get("num_workers", 0))
    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    standardize = bool(args.standardize_labels or stage3.get("standardize_labels",
                                                             config.get("standardize_labels", False)))
    paths = resolve_shards(shards, "stage2")
    dataset = BimodalShardDataset(paths, stage="stage2", tokenizer=tokenizer, train=False, max_tokens=max_gen + 1,
                                  allowed_sources=args.mine_sources, standardize_labels=standardize)
    log(f"mine: sources {args.mine_sources or 'all'}, standardized labels {'on' if standardize else 'off'}")
    options = {"prefetch_factor": 4} if workers else {}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        collate_fn=partial(collate_bimodal_batch, pad_id=tokenizer.pad_id),
                        pin_memory=device.type == "cuda", **options)
    amp = stage3.get("amp_dtype", "bfloat16")
    hard: list[tuple[float, int, str]] = []
    hard_info: dict[str, dict[str, Any]] = {}
    rng = random.Random(seed)
    confident = Reservoir(args.anchor_limit, rng)
    humans = Reservoir(args.anchor_limit + args.mine_limit, rng)
    scanned = mistakes = low_confidence = batches = counter = 0
    for batch in loader:
        moved = to_device(batch, device)
        with autocast(device, amp), torch.inference_mode():
            memory, memory_mask, _ = model.encode_bimodal(**encoder_inputs(moved, "auto"))
            result = sample_sequences(model.decoder, memory, memory_mask, tokenizer, max_gen, greedy=True,
                                      block_special=False)
        ids = result["tokens"].cpu().tolist()
        lengths = result["lengths"].cpu().tolist()
        finished = result["finished"].cpu().tolist()
        min_prob = result["min_prob"].float().cpu().tolist()
        mean_logp = (result["logp"].float().sum(1) / result["lengths"].clamp_min(1)).cpu().tolist()
        for row in range(len(ids)):
            scanned += 1
            identifier = batch["ids"][row]
            human = not bool(batch["synthetic"][row])
            if human:
                humans.add(identifier)
            prediction = tokenizer.decode(ids[row][:lengths[row]])
            reference = batch["normalized_latex"][row]
            if standardize:
                prediction_form, reference_form = standard_form(prediction), standard_form(reference)
            else:
                prediction_form, reference_form = prediction, reference
            wrong = not (finished[row] and normalized_tokens(prediction_form) == normalized_tokens(reference_form))
            unsure = min_prob[row] < args.mine_confidence
            mistakes += wrong
            low_confidence += unsure and not wrong
            if wrong or unsure:
                hardness = 2.0 * wrong + (1.0 - min_prob[row])
                counter += 1
                entry = (hardness, counter, identifier)
                if len(hard) < args.mine_limit:
                    heapq.heappush(hard, entry)
                elif entry > hard[0]:
                    _, _, evicted = heapq.heapreplace(hard, entry)
                    hard_info.pop(evicted, None)
                else:
                    continue
                hard_info[identifier] = {"prediction": prediction, "hardness": round(hardness, 4),
                                         "wrong": wrong, "min_prob": round(min_prob[row], 4),
                                         "mean_logp": round(mean_logp[row], 4)}
            elif human:
                confident.add(identifier)
        batches += 1
        if batches % 50 == 0:
            log(f"mine: scanned {scanned}, mistakes {mistakes}, low-confidence {low_confidence}")
        if args.mine_max_samples and scanned >= args.mine_max_samples:
            break
    selected = {identifier: "hard" for _, _, identifier in hard}
    # Prefer anchors the model already gets right; fill with other unselected human samples.
    anchor_count = 0
    for identifier in [*confident.items, *humans.items]:
        if anchor_count >= args.anchor_limit:
            break
        if identifier not in selected:
            selected[identifier] = "anchor"
            anchor_count += 1

    root = Path(args.output) if args.output else shards
    directory = root / HARD_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    for stale in [*directory.glob("shard-*.tar"), *directory.glob("*.tmp")]:
        stale.unlink()
    shard_size = 5000
    crohme_copies = max(1, int(round(args.crohme_upsample)))
    shards_written, writer, written, sources, roles, total = [], None, set(), Counter(), Counter(), 0
    try:
        for path in paths:
            for record in iter_shard(path):
                if record["json"] is None or record["png"] is None:
                    continue
                meta = json.loads(record["json"])
                role = selected.get(meta["id"])
                if role is None or meta["id"] in written:
                    continue
                meta["stage3_role"] = role
                if role == "hard":
                    meta["mined"] = hard_info.get(meta["id"])
                copies = crohme_copies if role == "hard" and meta["source"] == "crohme_train" else 1
                for _ in range(copies):
                    if writer is None:
                        writer = ShardWriter(directory / f"shard-{len(shards_written):05d}.tar")
                    writer.write(f"{total:08d}", meta, record["png"], record["npz"])
                    total += 1
                    sources[meta["source"]] += 1
                    roles[role] += 1
                    if writer.samples >= shard_size:
                        shards_written.append(writer.close())
                        writer = None
                written.add(meta["id"])
        if writer is not None:
            shards_written.append(writer.close())
            writer = None
    finally:
        if writer is not None:
            writer.abort()
    manifest = {
        "format_version": HARD_FORMAT, "created_at": utc_now(), "directory": HARD_DIRECTORY,
        "checkpoint": str(args.init_from), "checkpoint_stage": payload.get("stage"),
        "tokenizer_sha256": file_sha256(shards / "tokenizer.json"), "scanned": scanned, "mistakes": mistakes,
        "low_confidence": low_confidence, "confidence_threshold": args.mine_confidence,
        "max_gen_tokens": max_gen, "hard": roles["hard"], "anchors": roles["anchor"], "samples": total,
        "unique_samples": len(written), "mine_sources": args.mine_sources, "standardize_labels": standardize,
        "crohme_upsample": crohme_copies, "sources": dict(sources), "shards": shards_written,
        "seconds": round(time.perf_counter() - started, 1),
    }
    atomic_json(root / "hard_manifest.json", manifest)
    log(f"mine: {roles['hard']} hard + {roles['anchor']} anchors ({len(written)} unique) from {scanned} scanned "
        f"({mistakes} mistakes) -> {directory} {dict(sources)}")
    return 0


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3))
    parser.add_argument("--shards", type=Path, required=True,
                        help="directory with manifest.json, tokenizer.json, stage1_train/, stage2_train/, val_dev/")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--init-from", "--checkpoint", dest="init_from", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="frozen Stroke2Latex for shared-vocab logit distillation (overrides teacher_checkpoint)")
    parser.add_argument("--resume", help="'auto' or a recovery.pt path")
    parser.add_argument("--max-wall-minutes", type=float, help="0 = unlimited; default runtime.default_max_wall_minutes")
    parser.add_argument("--drive-sync-dir", type=Path)
    parser.add_argument("--sync-every-minutes", type=float)
    parser.add_argument("--hf-sync-repo", help="private HF dataset repo <user>/<name> for run artifacts (HF_TOKEN)")
    parser.add_argument("--hf-sync-prefix", help="folder inside --hf-sync-repo; default runs/stage<N>")
    parser.add_argument("--hf-init-path", help="file in --hf-sync-repo used as --init-from when that is not local")
    parser.add_argument("--hf-squash-history", action="store_true",
                        help="squash the repo history at session end so superseded checkpoints stop using storage")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--preset")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int, help="micro-batches per optimizer step (overrides stage config)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--hf-token")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-modality", choices=("auto", "vision", "stroke", "bimodal"))
    parser.add_argument("--mine-hard", action="store_true")
    parser.add_argument("--mine-limit", type=int, default=20000)
    parser.add_argument("--anchor-limit", type=int, default=5000)
    parser.add_argument("--mine-confidence", type=float, default=0.6,
                        help="correct predictions whose least likely token is below this are also hard")
    parser.add_argument("--mine-max-samples", type=int, default=0)
    parser.add_argument("--mine-batch-size", type=int)
    parser.add_argument("--mine-sources", nargs="+", help="mine only these sources, e.g. crohme_train mathwriting_train")
    parser.add_argument("--crohme-upsample", type=float, default=1.0,
                        help="copies of each mined crohme_train hard example written to stage3_hard/")
    parser.add_argument("--standardize-labels", action="store_true",
                        help="single-standard labels (tokenizer/standardize.py) for targets, validation, rewards "
                             "and mining; also enabled by standardize_labels in the YAML")
    parser.add_argument("--eval-every-steps", type=int,
                        help="validate every N optimizer steps (0 = epoch end only); overrides validation config")
    args = parser.parse_args(argv)
    if args.grad_accum is not None and args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if not (args.eval_only or args.mine_hard):
        if args.stage is None:
            parser.error("--stage is required unless --eval-only or --mine-hard")
        if args.output is None:
            parser.error("--output is required for training")
    if (args.hf_sync_prefix or args.hf_init_path or args.hf_squash_history) and not args.hf_sync_repo:
        parser.error("--hf-sync-prefix, --hf-init-path and --hf-squash-history need --hf-sync-repo")
    if args.hf_sync_repo:
        if args.eval_only or args.mine_hard:
            parser.error("--hf-sync-repo applies to training only")
        if args.hf_sync_repo.count("/") != 1 or not all(args.hf_sync_repo.split("/")):
            parser.error("--hf-sync-repo must look like <user>/<repo>")
        args.hf_sync_prefix = (args.hf_sync_prefix or f"runs/stage{args.stage}").strip("/")
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    args.hf_sync_token = configure_environment(args.hf_token)
    config = load_config(args.config)
    if args.mine_hard:
        return mine_hard(args, config)
    if args.eval_only:
        return eval_only(args, config)
    return Trainer(args, config).run()


if __name__ == "__main__":
    sys.exit(main())
