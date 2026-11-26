"""InkTeX v2 unified trainer supporting Stage A (Stroke Foundation), Stage B0 (Vision Alignment),
Stage B (High-Res Joint Bimodal Fusion), Stage C (Human Cooldown), and LoRA Personalization.

Hardened against every P0/P1 audit finding from Claude Opus 5.5 & GPT-6.2-Sol:
  - Hard fail on missing/invalid --resume path; saves recovery.pt every save_every_steps
  - Automatic Stage B0 init_skip_prefixes (never overwrites FCMAE ConvNeXt-V2 with Stage A random vision)
  - Multi-split weighted interleaving (MixedSplitV2Dataset for train_online + train_offline)
  - Zero-re-encode Stage B mutual KD gated strictly on undropped_both rows
  - Deduplicated EMA (no double-decay on tied embeddings!) with warmup min(d, (1+n)/(10+n))
  - Non-finite grad_norm guard + GradScaler skip detection
  - Activation checkpointing flag for StrokeChunkEncoder
  - Fast BF16 KV-cached greedy evaluation on full locked val_crohme & val_mw
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import yaml

from tokenizer.latex_tokenizer import tokenize_latex
from training.metrics import edit_distance

from .model import InkTeXV2Config, InkTeXV2Model, get_v2_config
from .shards_v2 import (
    AugmentV2Config,
    InkTeXV2EvalDataset,
    InkTeXV2ShardDataset,
    collate_v2_batch,
)
from .tokenizer_v2 import MAX_SEQ_TOKENS_V2, LatexTokenizerV2

CHECKPOINT_FORMAT_V2 = "inktex-v2-checkpoint-v1"


class ModelEMA:
    """Deduplicated EMA (never double-updates tied weights) with linear warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        seen: set[int] = set()
        for (_, sp), (_, mp) in zip(self.shadow.named_parameters(), model.named_parameters()):
            if id(sp) in seen or not mp.requires_grad:
                continue
            seen.add(id(sp))
            sp.mul_(d).add_(mp.detach().to(sp.dtype), alpha=1.0 - d)
        seen_b: set[int] = set()
        for (_, sb), (_, mb) in zip(self.shadow.named_buffers(), model.named_buffers()):
            if id(sb) in seen_b:
                continue
            seen_b.add(id(sb))
            sb.copy_(mb.detach())


class MixedSplitV2Dataset(IterableDataset):
    """Weighted stochastic interleaving of multiple InkTeXV2ShardDataset splits
    (e.g. train_online 0.88 + train_offline 0.12 for Stage B0 and Stage B)."""

    def __init__(self, datasets: list[InkTeXV2ShardDataset], weights: list[float], seed: int = 1337):
        if not datasets or len(datasets) != len(weights):
            raise ValueError("datasets and weights must be non-empty and equal length")
        self.datasets = datasets
        total_w = sum(weights)
        self.probs = [w / total_w for w in weights]
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for ds in self.datasets:
            ds.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if len(self.datasets) == 1:
            yield from self.datasets[0]
            return
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, self.epoch, wid, 991])
        # Give secondary splits a disjoint epoch range (self.epoch * 1000) so wrapping
        # across outer epochs NEVER replays identical secondary shuffle/augmentation seeds!
        sub_epochs = [self.epoch if i == 0 else self.epoch * 1000 for i in range(len(self.datasets))]
        for i in range(1, len(self.datasets)):
            self.datasets[i].set_epoch(sub_epochs[i])
        iters = [iter(ds) for ds in self.datasets]
        active = list(range(len(self.datasets)))
        weights = [float(p) for p in self.probs]

        def _active_probs() -> list[float]:
            tot = sum(weights[i] for i in active)
            return [weights[i] / tot for i in active]

        # Emit until the primary (index 0) dataset exhausts one full pass
        while True:
            choice = int(rng.choice(active, p=_active_probs())) if len(active) > 1 else active[0]
            try:
                yield next(iters[choice])
            except StopIteration:
                if choice == 0:
                    return
                sub_epochs[choice] += 1
                self.datasets[choice].set_epoch(sub_epochs[choice])
                iters[choice] = iter(self.datasets[choice])
                try:
                    yield next(iters[choice])
                except (StopIteration, RuntimeError):
                    active.remove(choice)
                    if not active:
                        return
            except RuntimeError as exc:
                if choice == 0:
                    raise
                print(f"[MixedSplitV2Dataset] Dropping secondary split idx={choice} on worker {wid}: {exc}", flush=True)
                active.remove(choice)
                if not active:
                    return


def corrupt_decoder_inputs(
    tokens: torch.Tensor,
    *,
    prob: float,
    vocab_size: int,
    special_max_id: int = 4,
) -> torch.Tensor:
    if prob <= 0.0 or tokens.shape[1] <= 1:
        return tokens
    out = tokens.clone()
    body = out[:, 1:]
    eligible = body.gt(special_max_id)
    drop = eligible & (torch.rand(body.shape, device=tokens.device) < prob)
    if bool(drop.any()):
        rand_toks = torch.randint(special_max_id + 1, vocab_size, body.shape, device=tokens.device)
        body[drop] = rand_toks[drop]
    return out


def compute_seq_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.02,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def build_optimizer_groups(
    model: InkTeXV2Model,
    *,
    lr: float,
    stroke_lr: float | None,
    vision_lr: float | None,
    decoder_lr: float | None,
    weight_decay: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[float, float], list[nn.Parameter]] = {}
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if name.startswith("stroke_encoder."):
            p_lr = stroke_lr if stroke_lr is not None else lr
        elif name.startswith("vision_encoder."):
            p_lr = vision_lr if vision_lr is not None else lr
        elif name.startswith("decoder."):
            p_lr = decoder_lr if decoder_lr is not None else lr
        else:
            p_lr = lr
        no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name
            or "embedding" in name
            or "gate" in name
            or "mod_emb" in name
        )
        p_wd = 0.0 if no_decay else weight_decay
        buckets.setdefault((p_lr, p_wd), []).append(param)
    return [
        {"params": params, "lr": p_lr, "base_lr": p_lr, "weight_decay": p_wd}
        for (p_lr, p_wd), params in buckets.items()
    ]


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


@torch.inference_mode()
def greedy_eval_v2(
    model: InkTeXV2Model,
    loader: DataLoader,
    tok: LatexTokenizerV2,
    device: torch.device,
    *,
    modality_mode: str = "auto",
    max_tokens: int = MAX_SEQ_TOKENS_V2,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Fast BF16 KV-cached greedy L2R evaluation on locked val_crohme / val_mw splits."""
    model.eval()
    exact = 0
    total = 0
    ed_sum = 0
    ref_len_sum = 0
    long_total = 0
    long_exact = 0
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for batch in loader:
        bsz = len(batch["normalized_latex"])
        images = batch["images"].to(device, non_blocking=True) if (batch["images"] is not None and modality_mode != "stroke") else None
        image_widths = batch["image_widths"].to(device, non_blocking=True) if images is not None else None
        cp = batch["chunk_points"].to(device, non_blocking=True) if (batch["chunk_points"] is not None and modality_mode != "vision") else None
        cpm = batch["chunk_point_mask"].to(device, non_blocking=True) if cp is not None else None
        cm = batch["chunk_meta"].to(device, non_blocking=True) if cp is not None else None
        ck = batch["chunk_mask"].to(device, non_blocking=True) if cp is not None else None

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            mem, mem_mask, _ = model.encode(
                images=images, image_widths=image_widths,
                chunk_points=cp, chunk_point_mask=cpm, chunk_meta=cm, chunk_mask=ck,
                modality_mode=modality_mode,
            )
            preds_t = model.decoder.decode_greedy_cached(
                mem,
                mem_mask,
                bos_id=model.config.bos_l2r_id,
                eos_id=model.config.eos_id,
                pad_id=model.config.pad_id,
                max_tokens=max_tokens,
            )

        preds = preds_t.cpu().tolist()
        refs = batch["normalized_latex"]
        take = bsz if max_samples is None else min(bsz, max_samples - total)
        for p_ids, ref in zip(preds[:take], refs[:take]):
            p_toks = tok.decode_tokens(p_ids)
            r_toks = tokenize_latex(ref)
            hit = int(p_toks == r_toks)
            exact += hit
            ed_sum += edit_distance(p_toks, r_toks)
            ref_len_sum += max(len(r_toks), 1)
            if len(r_toks) >= 40:
                long_total += 1
                long_exact += hit
            total += 1
        if max_samples is not None and total >= max_samples:
            break

    return {
        "samples": total,
        "exact": exact / max(total, 1),
        "token_cer": ed_sum / max(ref_len_sum, 1),
        "long40_samples": long_total,
        "long40_exact": long_exact / max(long_total, 1),
    }


def save_v2_checkpoint(
    path: Path,
    model: InkTeXV2Model,
    tokenizer: LatexTokenizerV2,
    *,
    stage: str,
    step: int,
    epoch: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEMA | None = None,
    best_score: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT_V2,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "global_step": step,
        "epoch": epoch,
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "model_state_dict": (ema.shadow if ema is not None else model).state_dict(),
        "lora_enabled": stage == "lora",
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["live_state_dict"] = model.state_dict()
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["best_score"] = best_score
        if ema is not None:
            payload["ema_updates"] = ema.updates
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_v2_checkpoint(path: Path, map_location: str = "cpu") -> tuple[InkTeXV2Model, LatexTokenizerV2, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT_V2:
        raise ValueError(f"Expected {CHECKPOINT_FORMAT_V2}, got {payload.get('format')!r}")
    cfg = InkTeXV2Config.from_dict({**payload["model_config"], "vision_pretrained": False})
    model = InkTeXV2Model(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if payload.get("lora_enabled"):
        model.set_lora_enabled(True)
    tok = LatexTokenizerV2.from_dict(payload["tokenizer"])
    return model, tok, payload


def _set_stage_train_modes(raw_model: InkTeXV2Model, stage: str) -> None:
    raw_model.train()
    if stage == "stage_a_stroke":
        raw_model.vision_encoder.eval()
        raw_model.fusion.blocks.eval()
    elif stage == "stage_b0_vision":
        raw_model.stroke_encoder.eval()
        raw_model.style_encoder.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InkTeX v2 (Stage A / B0 / B / C / LoRA).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-wall-minutes", type=float, default=0.0)
    parser.add_argument("--bench-steps", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=0, help="Optional override for config train.total_steps (budget-matched cosine schedule)")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional override for config train.batch_size")
    parser.add_argument("--grad-accum", type=int, default=0, help="Optional override for config train.grad_accum")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    cfg_yaml = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = str(cfg_yaml["stage"])
    tr_cfg = cfg_yaml["train"]
    if args.batch_size > 0:
        tr_cfg["batch_size"] = int(args.batch_size)
    if args.grad_accum > 0:
        tr_cfg["grad_accum"] = int(args.grad_accum)
    model_overrides = dict(cfg_yaml.get("model", {}))
    seed = int(cfg_yaml.get("seed", 1337))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = LatexTokenizerV2.load(args.shards / "tokenizer_v2.json")

    preset = model_overrides.pop("preset", "inktex_v2_100m")
    model_cfg = get_v2_config(preset, vocab_size=len(tokenizer), **model_overrides)
    model = InkTeXV2Model(model_cfg)
    model.stroke_encoder.grad_checkpointing = bool(tr_cfg.get("grad_checkpointing", False))

    if args.init_from is not None:
        if not args.init_from.is_file():
            raise FileNotFoundError(f"--init-from checkpoint not found: {args.init_from}")
        payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if "tokenizer" not in payload:
            raise ValueError(f"--init-from checkpoint {args.init_from} is missing embedded tokenizer metadata")
        if payload["tokenizer"] != tokenizer.to_dict():
            raise ValueError(f"--init-from tokenizer mismatch between {args.init_from} and {args.shards / 'tokenizer_v2.json'}")
        src_sd = payload["model_state_dict"]
        # CRITICAL: never overwrite FCMAE-pretrained ConvNeXt-V2 weights with Stage A's random vision weights!
        default_skips = ("vision_encoder.", "fusion.blocks.") if stage == "stage_b0_vision" else ()
        skip_prefixes = tuple(tr_cfg.get("init_skip_prefixes", default_skips))
        if skip_prefixes:
            src_sd = {k: v for k, v in src_sd.items() if not k.startswith(skip_prefixes)}
        missing, unexpected = model.load_state_dict(src_sd, strict=False)
        bad_missing = [k for k in missing if not (skip_prefixes and k.startswith(skip_prefixes))]
        if unexpected or bad_missing:
            raise RuntimeError(
                f"--init-from state_dict mismatch: unexpected={unexpected[:5]} ({len(unexpected)}), "
                f"unskipped_missing={bad_missing[:5]} ({len(bad_missing)})"
            )
        if stage == "stage_b0_vision" and not model_cfg.vision_pretrained:
            print("[init] WARNING: stage_b0_vision with vision_pretrained=False — ConvNeXt-V2 is randomly initialized!", flush=True)
        print(
            f"[init] Loaded {args.init_from} (skipped={skip_prefixes}) | "
            f"skipped_missing={len(missing)} unexpected=0",
            flush=True,
        )

    # Exact stage-specific freezing (preventing optimizer pollution and frozen-branch drift)
    if stage == "stage_a_stroke":
        for p in model.vision_encoder.parameters():
            p.requires_grad_(False)
        for p in model.fusion.blocks.parameters():
            p.requires_grad_(False)
        for p in model.style_encoder.parameters():
            p.requires_grad_(False)
        for p in model.lora_parameters():
            p.requires_grad_(False)
        for blk in model.stroke_encoder.blocks:
            if blk.has_film:
                blk.film.weight.requires_grad_(False)
                blk.film_scale.requires_grad_(False)
        model.set_lora_enabled(False)
    elif stage == "stage_b0_vision":
        for p in model.stroke_encoder.parameters():
            p.requires_grad_(False)
        for p in model.style_encoder.parameters():
            p.requires_grad_(False)
        for p in model.fusion.stroke_to_dec.parameters():
            p.requires_grad_(False)
        for p in model.fusion.stroke_norm.parameters():
            p.requires_grad_(False)
        model.fusion.stroke_mod_emb.requires_grad_(False)
        model.set_lora_enabled(False)
    elif stage == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lora_parameters():
            p.requires_grad_(True)
        model.set_lora_enabled(True)
    else:
        for p in model.lora_parameters():
            p.requires_grad_(False)
        model.set_lora_enabled(False)

    model.to(device)
    ema = ModelEMA(model, decay=float(tr_cfg.get("ema_decay", 0.999))) if stage != "lora" else None

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=True)

    modality_mode = str(tr_cfg.get("modality_mode", "stroke" if stage == "stage_a_stroke" else "auto"))
    need_vision = modality_mode != "stroke"
    need_strokes = modality_mode != "vision"
    static_buckets = bool(tr_cfg.get("static_buckets", need_vision))

    aug_cfg = AugmentV2Config.from_dict(tr_cfg.get("augment"))
    split_specs = tr_cfg.get("train_splits")
    if not split_specs:
        split_specs = [{"split": str(tr_cfg.get("train_split", "train_online")), "weight": 1.0}]
    sub_datasets = [
        InkTeXV2ShardDataset(
            args.shards,
            spec["split"],
            tokenizer,
            need_vision=need_vision,
            need_strokes=need_strokes,
            train=True,
            seed=seed + idx * 17,
            shuffle_buffer=int(tr_cfg.get("shuffle_buffer", 2048)),
            max_seq_tokens=int(tr_cfg.get("max_seq_tokens", MAX_SEQ_TOKENS_V2)),
            clean_human_only=bool(tr_cfg.get("clean_human_only", False)),
            allowed_sources=tr_cfg.get("allowed_sources"),
            source_upsample=tr_cfg.get("source_upsample"),
            augment=aug_cfg,
        )
        for idx, spec in enumerate(split_specs)
    ]
    tr_ds = MixedSplitV2Dataset(sub_datasets, [float(s.get("weight", 1.0)) for s in split_specs], seed=seed)

    collate_fn = partial(collate_v2_batch, pad_id=tokenizer.pad_id, static_buckets=static_buckets)
    tr_loader = DataLoader(
        tr_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 4)),
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    def _make_val_loader(split_name: str) -> DataLoader:
        ds = InkTeXV2EvalDataset(args.shards, split_name, tokenizer, need_vision=need_vision, need_strokes=need_strokes)
        val_bs = int(tr_cfg.get("val_batch_size", 32))
        if need_vision:
            from collections import defaultdict
            from .vision_raster import WIDTH_BUCKETS_V2, select_width_bucket

            groups: dict[int, list[int]] = defaultdict(list)
            for idx, rec in enumerate(ds.records):
                meta = json.loads(rec["json"])
                w = int(meta.get("image_width", WIDTH_BUCKETS_V2[0]))
                groups[select_width_bucket(w, WIDTH_BUCKETS_V2)].append(idx)
            batches = [
                idxs[s : s + val_bs]
                for b_key in sorted(groups.keys())
                for idxs in (groups[b_key],)
                for s in range(0, len(idxs), val_bs)
            ]
            return DataLoader(ds, batch_sampler=batches, num_workers=0, collate_fn=collate_fn)
        return DataLoader(ds, batch_size=val_bs, num_workers=0, collate_fn=collate_fn)

    val_crohme_loader = _make_val_loader("val_crohme")
    val_mw_loader = _make_val_loader("val_mw")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    groups = build_optimizer_groups(
        raw_model,
        lr=float(tr_cfg["lr"]),
        stroke_lr=tr_cfg.get("stroke_lr"),
        vision_lr=tr_cfg.get("vision_lr"),
        decoder_lr=tr_cfg.get("decoder_lr"),
        weight_decay=float(tr_cfg.get("weight_decay", 0.03)),
    )
    optimizer = torch.optim.AdamW(groups, fused=(device.type == "cuda"))

    use_amp = device.type == "cuda"
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    cfg_total_steps = int(args.total_steps) if args.total_steps > 0 else int(tr_cfg["total_steps"])
    total_steps = int(args.bench_steps) if args.bench_steps > 0 else cfg_total_steps
    grad_accum = int(tr_cfg.get("grad_accum", 1))
    warmup_steps = int(tr_cfg.get("warmup_steps", max(1, int(0.04 * total_steps))))
    min_lr_ratio = float(tr_cfg.get("min_lr_ratio", 0.05))
    r2l_weight = float(tr_cfg.get("r2l_weight", 0.30))
    label_smoothing = float(tr_cfg.get("label_smoothing", 0.02))
    token_corrupt_prob = float(tr_cfg.get("token_corrupt_prob", 0.0))
    vision_drop_prob = float(tr_cfg.get("vision_drop_prob", 0.0))
    stroke_drop_prob = float(tr_cfg.get("stroke_drop_prob", 0.0))
    mutual_kd_weight = float(tr_cfg.get("mutual_kd_weight", 0.0))
    eval_every = int(tr_cfg.get("eval_every_steps", 2000))
    save_every = int(tr_cfg.get("save_every_steps", 500))
    log_every = int(tr_cfg.get("log_every_steps", 100))
    val_max_raw = tr_cfg.get("val_max_samples", 0)
    val_max_samples = None if not val_max_raw or int(val_max_raw) <= 0 else int(val_max_raw)

    step = 0
    epoch = 0
    best_score = -1.0
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"--resume checkpoint does not exist: {args.resume}")
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" not in ck or "live_state_dict" not in ck:
            raise ValueError(f"--resume requires a recovery.pt with live_state_dict & optimizer_state_dict: {args.resume}")
        raw_model.load_state_dict(ck["live_state_dict"], strict=True)
        if ema is not None:
            ema.shadow.load_state_dict(ck["model_state_dict"], strict=True)
            ema.updates = int(ck.get("ema_updates", ck.get("global_step", 0)))
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scaler_state_dict" in ck:
            scaler.load_state_dict(ck["scaler_state_dict"])
        step = int(ck.get("global_step", 0))
        epoch = int(ck.get("epoch", 0))
        best_score = float(ck.get("best_score", -1.0))
        print(f"[resume] Resumed {args.resume} at step={step} epoch={epoch} best={best_score:.4f}", flush=True)

    trainable_m = sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1e6
    total_m = sum(p.numel() for p in raw_model.parameters()) / 1e6
    print(f"[train_v2:{stage}] total={total_m:.2f}M trainable={trainable_m:.2f}M device={device} steps={step}/{total_steps}", flush=True)

    metrics_file = args.output / "metrics.jsonl"
    t_wall = time.perf_counter()
    t_log = t_wall
    seen_log = 0
    loss_acc = torch.zeros((), device=device)
    skipped_steps = 0
    micro = 0
    micro_since_log = 0

    _set_stage_train_modes(raw_model, stage)
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        tr_ds.set_epoch(epoch)
        epoch += 1
        for batch in tr_loader:
            images = batch["images"].to(device, non_blocking=True) if (batch["images"] is not None and need_vision) else None
            image_widths = batch["image_widths"].to(device, non_blocking=True) if images is not None else None
            cp = batch["chunk_points"].to(device, non_blocking=True) if (batch["chunk_points"] is not None and need_strokes) else None
            cpm = batch["chunk_point_mask"].to(device, non_blocking=True) if cp is not None else None
            cm = batch["chunk_meta"].to(device, non_blocking=True) if cp is not None else None
            ck = batch["chunk_mask"].to(device, non_blocking=True) if cp is not None else None

            l2r = batch["l2r_tokens"].to(device, non_blocking=True)
            l2r_m = batch["l2r_mask"].to(device, non_blocking=True)
            r2l = batch["r2l_tokens"].to(device, non_blocking=True)
            r2l_m = batch["r2l_mask"].to(device, non_blocking=True)

            l2r_in = corrupt_decoder_inputs(l2r[:, :-1], prob=token_corrupt_prob, vocab_size=len(tokenizer))
            r2l_in = corrupt_decoder_inputs(r2l[:, :-1], prob=token_corrupt_prob, vocab_size=len(tokenizer))

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    l2r_in,
                    l2r_m[:, :-1],
                    r2l_decoder_input=r2l_in,
                    r2l_token_padding_mask=r2l_m[:, :-1],
                    images=images,
                    image_widths=image_widths,
                    chunk_points=cp,
                    chunk_point_mask=cpm,
                    chunk_meta=cm,
                    chunk_mask=ck,
                    modality_mode=modality_mode,
                    vision_drop_prob=vision_drop_prob,
                    stroke_drop_prob=stroke_drop_prob,
                )
                l2r_loss = compute_seq_ce(out["logits"], l2r[:, 1:], tokenizer.pad_id, label_smoothing)
                r2l_loss = compute_seq_ce(out["r2l_logits"], r2l[:, 1:], tokenizer.pad_id, label_smoothing)
                total_loss = l2r_loss + r2l_weight * r2l_loss

                # Zero-re-encode mutual KD: reuses stroke_mem from main pass, gated ONLY on undropped_both rows!
                aux = out.get("aux")
                if mutual_kd_weight > 0.0 and aux is not None and bool(aux["undropped_both"].any()):
                    rows = aux["undropped_both"]
                    st_logits = raw_model.decoder(
                        l2r_in[rows], aux["stroke_mem"][rows], aux["stroke_mask"][rows], l2r_m[rows, :-1],
                    )
                    valid_tok = l2r[rows, 1:].ne(tokenizer.pad_id)
                    p_bi = F.softmax(out["logits"][rows].detach().float(), dim=-1)
                    log_p_st = F.log_softmax(st_logits.float(), dim=-1)
                    kl = F.kl_div(log_p_st, p_bi, reduction="none").sum(dim=-1)
                    kd_loss = (kl * valid_tok).sum() / valid_tok.sum().clamp_min(1)
                    total_loss = total_loss + mutual_kd_weight * kd_loss

                loss = total_loss / grad_accum

            scaler.scale(loss).backward()
            loss_acc = loss_acc + total_loss.detach().float()
            seen_log += len(batch["normalized_latex"])
            micro += 1
            micro_since_log += 1

            if micro % grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(tr_cfg.get("grad_clip", 1.0)))
                if not bool(torch.isfinite(grad_norm)):
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                mult = lr_multiplier(step, total_steps, warmup_steps, min_lr_ratio)
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"] * mult

                prev_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Only advance step & EMA if FP16 GradScaler didn't skip the optimizer update
                if scaler.is_enabled() and scaler.get_scale() < prev_scale:
                    skipped_steps += 1
                    continue

                if ema is not None:
                    ema.update(raw_model)
                step += 1

                if step % log_every == 0 or step == total_steps:
                    dt = max(time.perf_counter() - t_log, 1e-3)
                    rate = seen_log / dt
                    avg_l = float(loss_acc.item()) / max(micro_since_log, 1)
                    vram_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if device.type == "cuda" else 0.0
                    print(
                        f"[{stage}] step {step}/{total_steps} ep={epoch} loss={avg_l:.4f} "
                        f"gnorm={float(grad_norm):.2f} lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"rate={rate:.1f} ex/s peak_vram={vram_gb:.2f}GB skip={skipped_steps}",
                        flush=True,
                    )
                    t_log = time.perf_counter()
                    seen_log = 0
                    loss_acc.zero_()
                    micro_since_log = 0

                if args.bench_steps == 0 and (step % save_every == 0 or step % eval_every == 0 or step == total_steps):
                    save_v2_checkpoint(
                        args.output / "recovery.pt", raw_model, tokenizer,
                        stage=stage, step=step, epoch=epoch, metrics={"step": step},
                        optimizer=optimizer, scaler=scaler, ema=ema, best_score=best_score,
                    )
                    print(f"[{stage}] saved recovery.pt at step {step}", flush=True)
                    t_log = time.perf_counter()

                if args.bench_steps == 0 and (step % eval_every == 0 or step == total_steps):
                    eval_model = ema.shadow if ema is not None else raw_model
                    eval_mode = "stroke" if modality_mode == "stroke" else ("vision" if modality_mode == "vision" else "auto")
                    vc = greedy_eval_v2(eval_model, val_crohme_loader, tokenizer, device, modality_mode=eval_mode, max_samples=val_max_samples)
                    vm = greedy_eval_v2(eval_model, val_mw_loader, tokenizer, device, modality_mode=eval_mode, max_samples=val_max_samples)
                    score = 0.5 * vc["exact"] + 0.5 * vm["exact"]
                    rec: dict[str, Any] = {
                        "step": step, "epoch": epoch, "score": score,
                        "val_crohme": vc, "val_mw": vm, "skipped_steps": skipped_steps,
                    }
                    if modality_mode == "auto" and stage in ("stage_b_joint", "stage_c_cooldown"):
                        vc_st = greedy_eval_v2(eval_model, val_crohme_loader, tokenizer, device, modality_mode="stroke", max_samples=val_max_samples)
                        rec["val_crohme_stroke_only"] = vc_st
                    with metrics_file.open("a", encoding="utf-8") as mf:
                        mf.write(json.dumps(rec) + "\n")
                    print(
                        f"[{stage}:eval@{step}] CROHME={vc['exact']:.2%} (cer={vc['token_cer']:.2%}, n={vc['samples']}) "
                        f"| MW={vm['exact']:.2%} (cer={vm['token_cer']:.2%}, n={vm['samples']}) | avg={score:.2%}",
                        flush=True,
                    )
                    save_v2_checkpoint(
                        args.output / "recovery.pt", raw_model, tokenizer,
                        stage=stage, step=step, epoch=epoch, metrics=rec,
                        optimizer=optimizer, scaler=scaler, ema=ema, best_score=max(best_score, score),
                    )
                    if score > best_score:
                        best_score = score
                        save_v2_checkpoint(
                            args.output / "best.pt", raw_model, tokenizer,
                            stage=stage, step=step, epoch=epoch, metrics=rec, ema=ema,
                        )
                        print(f"[{stage}] saved best.pt at step {step} (avg={score:.2%})", flush=True)
                    print(f"[{stage}] saved post-eval recovery.pt at step {step}", flush=True)
                    _set_stage_train_modes(raw_model, stage)
                    # Exclude eval/checkpoint time from training throughput timer!
                    t_log = time.perf_counter()

                if args.max_wall_minutes > 0 and (time.perf_counter() - t_wall) / 60.0 >= args.max_wall_minutes:
                    print(f"[{stage}] Wall-clock limit ({args.max_wall_minutes}m) reached at step {step}; saving recovery.pt", flush=True)
                    save_v2_checkpoint(
                        args.output / "recovery.pt", raw_model, tokenizer,
                        stage=stage, step=step, epoch=epoch, metrics={"paused": True},
                        optimizer=optimizer, scaler=scaler, ema=ema, best_score=best_score,
                    )
                    return

                if step >= total_steps:
                    break

    total_s = time.perf_counter() - t_wall
    print(f"[{stage}] DONE in {total_s/60:.1f}m | best_avg={best_score:.2%} | skipped={skipped_steps}", flush=True)


if __name__ == "__main__":
    main()
