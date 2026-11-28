"""Train InkForge v3 with bf16, EMA, and a real-data smoke benchmark.

From the repository root, run this script with --help or use
python -m inkforge.v3.train_v3 from ml/. The overnight defaults target one
RTX 4070 Ti. Use --smoke-steps 10 --cache-dir <v2 cache> to measure this
machine before starting a long run. --init-from-v2 accepts the v2 release.
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
from itertools import islice
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inkforge.dataset import InkForgeDataset, SOURCE_NAMES, collate_inkforge_batch
from inkforge.render import render_contact_sheet
from inkforge.tokenizer import InkForgeTokenizer
from inkforge.train_inkforge import (
    PREVIEW_PROMPTS,
    WeightedLengthBucketBatchSampler,
    _extract_named_writer_preview_styles,
    evaluate_val,
    export_writer_style_bank,
)
from inkforge.v3 import CHECKPOINT_FORMAT, InkForgeV3Config, InkForgeV3Model


DEFAULT_OUT_DIR = Path(r"E:\math-recogniser-training\inkforge_v3_model")


HARD_PREVIEW_PROMPTS = (
    r"\bar{A}+\overline{BC}",
    r"\ln(x)+\ln(y)=\ln(xy)",
    r"\sum_{n=0}^\infty \frac{x^n}{n!}",
    r"\xi+\zeta=\eta\kappa+\tau\rho",
    r"\hat{x}\cdot\bar{y}+\tilde{\phi}\equiv 0",
    r"\prod_{k=1}^{n}(1+x_k)",
    r"\lim_{x\to 0}\frac{e^x-1}{x}",
    r"\nabla\times\vec{E}",
    r"\frac{\sin x}{x}",
    r"\text{if } x > 0",
    r"\text{otherwise}",
    r"\sqrt{1+\sqrt{a^2+b^2}}",
)


@torch.no_grad()
def init_from_v2(model: InkForgeV3Model, tokenizer: InkForgeTokenizer, path: Path) -> int:
    """Transfer shape-compatible weights and remap embeddings by token string."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    old_cfg = payload["config"]
    state = payload.get("ema_state_dict", payload.get("model_state_dict"))
    if state is None:
        raise ValueError(f"No usable model weights in {path}")
    if payload.get("format") == CHECKPOINT_FORMAT or "canvas_in.weight" in state:
        raise ValueError("--init-from-v2 expects a v2 checkpoint. Use --resume for v3 training.")
    old_tokenizer = InkForgeTokenizer.from_dict(payload["tokenizer"])
    target = model.state_dict()
    transferred = 0
    same_heads = old_cfg["n_heads"] == model.cfg.n_heads
    for name, weight in target.items():
        if name == "text_encoder.embed.weight":
            old_embed = state[name]
            if old_embed.shape[-1] == weight.shape[-1]:
                pairs = [
                    (new_id, old_tokenizer.token_to_id[token])
                    for new_id, token in enumerate(tokenizer.tokens)
                    if token in old_tokenizer.token_to_id
                ]
                new_ids, old_ids = zip(*pairs)
                weight[list(new_ids)] = old_embed[list(old_ids)].to(weight)
                transferred += len(pairs) * weight.shape[-1]
        elif name in state and state[name].shape == weight.shape:
            # Different head counts change the attention partition despite equal projection shapes.
            if not same_heads and any(f".{part}." in name for part in ("qkv", "q_cross", "kv_cross")):
                continue
            # A changed component count changes the meaning of the output rows.
            if name.startswith("mdn_head.") and old_cfg["mdn_components"] != model.cfg.mdn_components:
                continue
            weight.copy_(state[name])
            transferred += weight.numel()

    # Newly added later blocks initially preserve the transferred residual stream.
    if old_cfg["d_model"] == model.cfg.d_model and same_heads:
        for block in model.dec_blocks[old_cfg["dec_layers"] :]:
            block.out1.weight.zero_()
            block.out2.weight.zero_()
            block.ff.w3.weight.zero_()
    if not transferred:
        raise ValueError(f"No compatible parameters in {path}")
    return transferred


@torch.inference_mode()
def render_previews(
    model: InkForgeV3Model,
    tokenizer: InkForgeTokenizer,
    data: InkForgeDataset,
    out_dir: Path,
    step: int,
    device: torch.device,
    cfg_scale: float,
) -> None:
    standard = tuple(PREVIEW_PROMPTS)
    prompts = standard + HARD_PREVIEW_PROMPTS
    encoded = [tokenizer.encode(p, max_len=model.cfg.max_text_len) for p in prompts]
    tokens = torch.full((len(encoded), max(map(len, encoded))), tokenizer.pad_id, dtype=torch.long)
    for i, ids in enumerate(encoded):
        tokens[i, :len(ids)] = torch.tensor(ids)
    tokens = tokens.to(device)
    writer_z = _extract_named_writer_preview_styles(model, data, len(prompts), device)
    variants = (
        ("neutral", torch.zeros_like(writer_z), 0.35),
        ("writer_neat", writer_z, 0.35),
        ("writer_natural", writer_z, 0.50),
    )
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for i, (name, z, temp) in enumerate(variants):
            torch.manual_seed(12345 + i)
            with torch.amp.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                strokes, meta = model.generate_batch(
                    tokens, z, temperature=temp, cfg_scale=cfg_scale,
                    max_steps=model.cfg.max_stroke_points, return_meta=True,
                )
            entries = list(zip(prompts, strokes))
            for tag, rows in (("standard", entries[:len(standard)]), ("hard", entries[len(standard):])):
                render_contact_sheet(
                    rows, out_dir / "previews" / f"step_{step:06d}_{tag}_{name}.png",
                    cols=3, tile_w=440, tile_h=150, color_by_order=False,
                )
            print(f"[PREVIEW {step}] {name} finished={sum(m['finished'] for m in meta)}/{len(prompts)}", flush=True)


def _save_checkpoint(
    path: Path,
    model: InkForgeV3Model,
    ema: InkForgeV3Model,
    tokenizer: InkForgeTokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_score: float,
) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "arch": "inkforge_v3",
        "step": step,
        "config": model.cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mdn": best_score,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
    }, tmp)
    tmp.replace(path)


def _lr(args: argparse.Namespace, step: int) -> float:
    if step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
    return args.lr * (0.05 + 0.95 * (1 + math.cos(math.pi * min(progress, 1.0))) / 2)


def _print_smoke(times: list[float], samples: int, max_points: int, max_tokens: int) -> None:
    elapsed = statistics.mean(times)
    stable = statistics.mean(times[min(2, len(times) - 1):])
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    print(
        f"[SMOKE] {len(times)} optimizer steps | {elapsed * 1000:.2f} ms/step all | "
        f"{stable * 1000:.2f} ms/step after first two | {samples / sum(times):.2f} samples/s | "
        f"peak allocated={peak:.3f} GiB | peak reserved={reserved:.3f} GiB | "
        f"longest batch T={max_points}, L={max_tokens} | all losses and gradient norms finite",
        flush=True,
    )
    print(
        f"[SMOKE] Training-only ETA from steady timing: "
        f"40k={stable * 40000 / 3600:.2f} h | 60k={stable * 60000 / 3600:.2f} h",
        flush=True,
    )
    if max(peak, reserved) >= 9.5:
        raise RuntimeError("Smoke benchmark exceeded the 9.5 GiB VRAM budget")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    if args.smoke_steps and not cuda:
        raise RuntimeError("The smoke benchmark requires CUDA")
    if cuda and not torch.cuda.is_bf16_supported():
        raise RuntimeError("InkForge v3 training requires CUDA bf16 support")

    tokenizer = InkForgeTokenizer.load(args.cache_dir / "tokenizer.json")
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume is not None:
        if resume.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("--resume requires a v3 training checkpoint")
        if resume["tokenizer"] != tokenizer.to_dict():
            raise ValueError("Resume checkpoint and cache tokenizers differ")
        cfg = InkForgeV3Config.from_dict(resume["config"])
    else:
        cfg = InkForgeV3Config(
            vocab_size=len(tokenizer),
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            token_roles=InkForgeV3Config.roles_for_tokens(tokenizer.tokens),
        )
    model = InkForgeV3Model(cfg, grad_checkpoint=args.grad_checkpoint)
    if args.init_from_v2:
        copied = init_from_v2(model, tokenizer, args.init_from_v2)
        print(f"[TRAIN] Warm-started {copied:,} parameters from {args.init_from_v2}", flush=True)
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    ema.grad_checkpoint = False
    if resume is not None:
        ema.load_state_dict(resume["ema_state_dict"])
    model = model.to(device).train()
    ema.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay, fused=cuda,
    )
    start, best = 0, float("inf")
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start, best = int(resume["step"]), float(resume["best_val_mdn"])
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"])
        if cuda and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume["cuda_rng_state"])
        del resume
    if start > args.total_steps:
        raise ValueError("Resume step exceeds --total-steps")

    data = InkForgeDataset(
        args.cache_dir,
        cfg=cfg,
        augment=True,
        input_noise_std=0,
        brace_aug_prob=args.brace_aug_prob,
        stroke_jitter_std=args.stroke_jitter_std,
    )
    collate = partial(collate_inkforge_batch, pad_id=cfg.pad_id)
    limit = args.smoke_steps or args.total_steps
    # Keep the same mega-batch shape in smoke and training so padding and timing are comparable.
    sampler = WeightedLengthBucketBatchSampler(
        data.sample_weights, np.diff(data.pt_offsets), args.batch_size,
        max(args.total_steps, limit) * args.grad_accum, seed=args.seed,
    )
    batches = DataLoader(
        data, batch_sampler=islice(sampler, start * args.grad_accum, limit * args.grad_accum),
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=cuda, persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_iter = iter(batches)

    val_loader = None
    out = args.out_dir or (args.resume.parent if args.resume else DEFAULT_OUT_DIR)
    if not args.smoke_steps:
        out.mkdir(parents=True, exist_ok=True)
        val = InkForgeDataset(args.cache_dir, split="val", cfg=cfg, augment=False, input_noise_std=0)
        human_ids = [SOURCE_NAMES.index(name) for name in ("mathwriting", "crohme", "iamondo")]
        indices = np.flatnonzero(np.isin(val.source_ids, human_ids)).tolist()
        if not indices:
            raise ValueError("Validation cache contains no human MathWriting, CROHME, or IAMONDO expressions")
        val_loader = DataLoader(
            Subset(val, indices), batch_size=args.batch_size, collate_fn=collate, pin_memory=cuda,
        )
    print(
        f"[TRAIN] {model.parameter_count():,} params on {device} | step={start}/{limit} | "
        f"batch={args.batch_size} x {args.grad_accum} | checkpoint={model.grad_checkpoint} | "
        f"named writer styles={len(data.writer_peers):,}",
        flush=True,
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    params = list(model.parameters())
    ema_params = list(ema.parameters())
    ema_buffers = list(ema.buffers())
    model_buffers = list(model.buffers())
    times: list[float] = []
    samples = max_points = max_tokens = 0
    metrics = ("loss", "mdn_loss", "pen_loss", "jump_mdn_loss", "jump_aux_loss", "pen_acc")
    running = torch.zeros(len(metrics) + 1, device=device)
    logged_steps = 0

    for step in range(start + 1, limit + 1):
        before = time.perf_counter()
        optimizer.param_groups[0]["lr"] = lr = _lr(args, step)
        kl = args.kl_weight * min(1.0, step / max(1, args.kl_warmup_steps))
        noise_progress = (step - args.warmup_steps) / max(1, args.noise_ramp_steps)
        noise = args.input_noise * min(1.0, max(0.0, noise_progress))
        if args.smoke_steps:
            noise = args.input_noise
        optimizer.zero_grad(set_to_none=True)
        loss_finite = torch.ones((), dtype=torch.bool, device=device)
        for _ in range(args.grad_accum):
            batch = next(train_iter)
            if args.smoke_steps:
                samples += batch["token_ids"].shape[0]
                max_points = max(max_points, batch["stroke_inputs"].shape[1])
                max_tokens = max(max_tokens, batch["token_ids"].shape[1])
            batch.pop("source_id")
            batch = {key: value.to(device, non_blocking=cuda) for key, value in batch.items()}
            with torch.amp.autocast(device.type, torch.bfloat16, enabled=cuda):
                losses = model.compute_loss(
                    **batch, kl_weight=kl, input_noise=noise, text_drop_prob=args.text_drop_prob,
                    guided_attn_weight=args.guided_attn_weight,
                )
                loss = losses["loss"] / args.grad_accum
            loss.backward()
            loss_finite &= loss.detach().isfinite()
            running[:-1].add_(torch.stack([losses[key].detach() for key in metrics]) / args.grad_accum)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
        if not (grad_norm.isfinite() & loss_finite):
            raise FloatingPointError(f"Non-finite loss or gradient at step {step}; latest checkpoint is intact")
        optimizer.step()
        with torch.no_grad():
            decay = min(args.ema_decay, (1 + step) / (10 + step))
            torch._foreach_lerp_(ema_params, params, 1 - decay)
            # These buffers describe token roles and rotary positions, which are constant.
            if step == start + 1:
                for dest, source in zip(ema_buffers, model_buffers):
                    dest.copy_(source)
            running[-1].add_(grad_norm)
        logged_steps += 1

        if args.smoke_steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            times.append(elapsed)
            print(
                f"[SMOKE {step}/{limit}] {elapsed * 1000:.2f} ms | "
                f"loss={losses['loss'].item():.4f} | grad_norm={grad_norm.item():.4f}",
                flush=True,
            )
            continue

        if step % args.log_every == 0 or step == limit:
            record = dict(zip((*metrics, "grad_norm"), (running / logged_steps).tolist()))
            record.update(step=step, split="train", lr=lr, input_noise=noise)
            if cuda:
                record["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
            _log(out, record)
            running.zero_()
            logged_steps = 0

        validate = step % args.eval_every == 0 or step == limit
        improved = False
        if validate:
            scores = evaluate_val(
                ema, val_loader, device, max_batches=args.val_max_batches or len(val_loader),
                guided_attn_weight=args.guided_attn_weight,
            )
            _log(out, {"step": step, "split": "VAL-EXPR", **scores})
            improved = scores["val_score"] < best
            if improved:
                best = scores["val_score"]
                _save_checkpoint(out / "best.pt", model, ema, tokenizer, optimizer, step, best)
        if validate or step % args.save_every == 0:
            _save_checkpoint(out / "latest.pt", model, ema, tokenizer, optimizer, step, best)
        if improved:
            _export_styles(ema, data, out, device)
        if step % args.preview_every == 0 or step == limit:
            render_previews(ema, tokenizer, data, out, step, device, args.preview_cfg_scale)

    if args.smoke_steps:
        _print_smoke(times, samples, max_points, max_tokens)
        return
    best_path = out / "best.pt"
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu", weights_only=True)
        ema.load_state_dict(payload["ema_state_dict"])
        del payload
    _export_styles(ema, data, out, device)
    print(f"[TRAIN] Finished at step {limit}. Best VAL-EXPR score={best:.4f}", flush=True)


def _export_styles(model: InkForgeV3Model, data: InkForgeDataset, out: Path, device: torch.device) -> None:
    temporary = out / "writer_styles.tmp.npz"
    export_writer_style_bank(model, data, temporary, device)
    temporary.replace(out / "writer_styles.npz")


def _log(out: Path, record: dict) -> None:
    line = json.dumps(record)
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(r"E:\math-recogniser-training\inkforge_data_v3"))
    parser.add_argument("--out-dir", type=Path, help="Defaults to the run directory or the resume checkpoint's folder")
    load = parser.add_mutually_exclusive_group()
    load.add_argument("--resume", type=Path)
    load.add_argument("--init-from-v2", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--kl-warmup-steps", type=int, default=4000)
    parser.add_argument("--noise-ramp-steps", type=int, default=7500)
    parser.add_argument("--input-noise", type=float, default=0.025)
    parser.add_argument("--brace-aug-prob", type=float, default=0.25)
    parser.add_argument("--stroke-jitter-std", type=float, default=0.004)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--kl-weight", type=float, default=0.025)
    parser.add_argument("--text-drop-prob", type=float, default=0.10)
    parser.add_argument("--guided-attn-weight", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--preview-every", type=int, default=2500)
    parser.add_argument("--val-max-batches", type=int, default=0, help="0 evaluates every human expression")
    parser.add_argument("--preview-cfg-scale", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("total_steps", "batch_size", "grad_accum", "log_every", "eval_every", "save_every", "preview_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.smoke_steps < 0 or args.num_workers < 0 or args.val_max_batches < 0:
        parser.error("Smoke steps, workers, and validation batch limit must be nonnegative")
    if args.resume and args.smoke_steps:
        parser.error("Use a fresh model or --init-from-v2 for a smoke benchmark")
    train(args)


if __name__ == "__main__":
    main()

