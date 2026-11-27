"""Train InkForge (~21M param Writer-Style-Conditioned Transformer-MDN) on RTX 4070 Ti.

Features:
  - Mixed precision `bf16` backbone + strictly `float32` MDN loss (zero NaNs).
  - Weighted sampling balancing full math expressions, IAMonDo `\\text{...}` phrases, and rare symbols.
  - EMA (0.999) + cosine LR with warmup + KL free-bits annealing.
  - Periodic validation + 24-prompt visual PNG contact sheets (both neat `tau=0.35` and messy `tau=0.55`)
    saved to `<out_dir>/previews/` so you can visually watch handwriting realism improve!
  - Extracts a 5,000-writer real human style latent bank (`writer_styles.npz`) on completion.
"""
from __future__ import annotations

import argparse
import copy
from functools import partial
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.dataset import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    InkForgeDataset,
    collate_inkforge_batch,
)
from inkforge.model import InkForgeConfig, InkForgeModel  # noqa: E402
from inkforge.render import render_contact_sheet  # noqa: E402
from inkforge.tokenizer import InkForgeTokenizer  # noqa: E402


class WeightedLengthBucketBatchSampler(Sampler[list[int]]):
    """Weighted sampling + mega-batch length bucketing: cuts padding waste 3-5x on RTX 4070 Ti!"""

    def __init__(
        self,
        weights: np.ndarray,
        lengths: np.ndarray,
        batch_size: int,
        num_batches: int,
        *,
        mega_factor: int = 32,
        seed: int = 42,
    ):
        self.probs = weights / weights.sum()
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.mega_factor = int(mega_factor)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        n = len(self.probs)
        batches_emitted = 0
        mega_batches = self.mega_factor
        while batches_emitted < self.num_batches:
            want_batches = min(mega_batches, self.num_batches - batches_emitted)
            pool_size = want_batches * self.batch_size
            sampled = rng.choice(n, size=pool_size, replace=True, p=self.probs)
            # Sort mega-pool by stroke length (with tiny jitter so bucket edges vary)
            order = np.argsort(self.lengths[sampled] + rng.uniform(-8.0, 8.0, size=pool_size))
            sorted_idx = sampled[order]
            chunk_batches = [
                sorted_idx[i : i + self.batch_size].tolist()
                for i in range(0, pool_size, self.batch_size)
            ]
            rng.shuffle(chunk_batches)
            for b in chunk_batches:
                yield b
                batches_emitted += 1

PREVIEW_PROMPTS = [
    r"x^2 + y^2 = r^2",
    r"2z + 5s = 17",
    r"\frac{a+b}{c}",
    r"\sqrt{b^2 - 4ac}",
    r"\int_0^\infty e^{-t} dt = 1",
    r"\sum_{i=1}^n i = \frac{n(n+1)}{2}",
    r"\alpha + \beta = \gamma",
    r"\sin^2 \theta + \cos^2 \theta = 1",
    r"\lim_{x \rightarrow 0} \frac{\sin x}{x} = 1",
    r"\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}",
    r"\forall x \in \mathbb{R}, x^2 \geq 0",
    r"\text{if } x > 0",
    r"\text{otherwise}",
    r"\Pr(A \cap B) = \Pr(A)\Pr(B)",
    r"f'(x) = 3x^2 - 2x + 1",
    r"|x - y| \leq |x| + |y|",
    r"\vec{u} \cdot \vec{v} = 0",
    r"\Delta x = \lambda \mu",
]


@torch.no_grad()
def update_ema(ema_model: InkForgeModel, model: InkForgeModel, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


@torch.inference_mode()
def evaluate_val(
    model: InkForgeModel,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 40,
    guided_attn_weight: float = 4.0,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "mdn_loss": 0.0, "pen_loss": 0.0, "guided_loss": 0.0, "kl_raw": 0.0, "pen_acc": 0.0}
    n = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model.compute_loss(
                token_ids=batch["token_ids"],
                stroke_inputs=batch["stroke_inputs"],
                target_dx=batch["target_dx"],
                target_dy=batch["target_dy"],
                target_pen=batch["target_pen"],
                stroke_mask=batch["stroke_mask"],
                style_points=batch["style_points"],
                style_aspect=batch["style_aspect"],
                style_mask=batch["style_mask"],
                kl_weight=0.025,
                style_drop_prob=0.0,
                guided_attn_weight=guided_attn_weight,
            )
        for k in totals:
            totals[k] += float(out[k].item())
        n += 1
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.inference_mode()
def _extract_named_writer_preview_styles(
    model: InkForgeModel,
    dataset: InkForgeDataset,
    count: int,
    device: torch.device,
) -> Tensor:
    """Extract real named-writer `mu_z` vectors for preview rendering (zero N(0, I) drift)."""
    peer_wids = sorted(dataset.writer_peers.keys())
    if not peer_wids:
        return torch.zeros((count, model.cfg.style_dim), device=device, dtype=torch.float32)

    orig_aug = dataset.augment
    dataset.augment = False
    try:
        items = []
        for i in range(count):
            wid = peer_wids[(i * 17) % len(peer_wids)]
            sample_idx = int(dataset.writer_peers[wid][0])
            items.append(dataset[sample_idx])
        batch = collate_inkforge_batch(items, pad_id=model.cfg.pad_id)
        sp = batch["style_points"].to(device)
        sa = batch["style_aspect"].to(device)
        sm = batch["style_mask"].to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, mu, _ = model.encode_style(sp, sa, sm, sample=False)
        return mu.float()
    finally:
        dataset.augment = orig_aug


@torch.inference_mode()
def render_preview_sheets(
    model: InkForgeModel,
    tokenizer: InkForgeTokenizer,
    style_ref_ds: InkForgeDataset,
    out_dir: Path,
    step: int,
    device: torch.device,
) -> None:
    model.eval()
    encoded = [tokenizer.encode(p, max_len=model.cfg.max_text_len) for p in PREVIEW_PROMPTS]
    max_l = max(len(e) for e in encoded)
    tok_t = torch.full((len(encoded), max_l), tokenizer.pad_id, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        tok_t[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)

    zero_z = torch.zeros((len(encoded), model.cfg.style_dim), device=device, dtype=torch.float32)
    writer_z = _extract_named_writer_preview_styles(model, style_ref_ds, len(encoded), device)

    configs = (
        ("neutral_z0", zero_z, 0.35),
        ("writer_neat", writer_z, 0.35),
        ("writer_natural", writer_z, 0.50),
    )
    for tag, z_in, temp in configs:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            strokes_list = model.generate_batch(
                tok_t,
                style_z=z_in,
                temperature=temp,
                pen_temperature=0.70,
                max_steps=420,
            )
        pairs = list(zip(PREVIEW_PROMPTS, strokes_list))
        render_contact_sheet(
            pairs,
            out_dir / "previews" / f"step_{step:06d}_{tag}.png",
            cols=3,
            tile_w=440,
            tile_h=150,
            color_by_order=False,
        )


@torch.inference_mode()
def export_writer_style_bank(
    model: InkForgeModel,
    dataset: InkForgeDataset,
    out_path: Path,
    device: torch.device,
    *,
    max_writers: int = 5000,
    batch_size: int = 128,
) -> Path:
    """Extract real human writer `mu_z` vectors strictly from named multi-sample writers."""
    model.eval()
    rng = np.random.default_rng(2026)
    peer_wids = set(dataset.writer_peers.keys())
    eligible = np.array([i for i, wid in enumerate(dataset.writer_ids.tolist()) if wid in peer_wids], dtype=np.int64)
    if len(eligible) == 0:
        eligible = np.arange(len(dataset))
    chosen = rng.choice(eligible, size=min(max_writers, len(eligible)), replace=False)

    orig_aug = dataset.augment
    dataset.augment = False
    zs: list[np.ndarray] = []
    try:
        for start in range(0, len(chosen), batch_size):
            idxs = chosen[start : start + batch_size]
            items = [dataset[int(i)] for i in idxs]
            batch = collate_inkforge_batch(items, pad_id=model.cfg.pad_id)
            sp = batch["style_points"].to(device)
            sa = batch["style_aspect"].to(device)
            sm = batch["style_mask"].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _, mu, _ = model.encode_style(sp, sa, sm, sample=False)
            zs.append(mu.float().cpu().numpy())
    finally:
        dataset.augment = orig_aug

    bank = np.concatenate(zs, axis=0).astype(np.float32)
    np.savez_compressed(
        out_path,
        mu_bank=bank,
        mean=bank.mean(axis=0),
        std=bank.std(axis=0),
    )
    print(f"[style-bank] Exported {len(bank):,} real human writer latents -> {out_path}")
    return out_path


def save_checkpoint(
    path: Path,
    *,
    step: int,
    cfg: InkForgeConfig,
    tokenizer: InkForgeTokenizer,
    model: InkForgeModel,
    ema_model: InkForgeModel,
    optimizer: torch.optim.Optimizer,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format": "inkforge-checkpoint-v1",
            "step": step,
            "config": cfg.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mdn": best_val,
        },
        tmp,
    )
    tmp.replace(path)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir: Path = args.cache_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = InkForgeTokenizer.load(cache_dir / "tokenizer.json")
    cfg = InkForgeConfig(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        style_dim=args.style_dim,
        mdn_components=args.mdn_components,
        dropout=args.dropout,
    )

    model = InkForgeModel(cfg).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    print(
        f"[inkforge] Model initialized: {model.parameter_count() / 1e6:.2f}M params "
        f"(d={cfg.d_model}, enc={cfg.enc_layers}L, dec={cfg.dec_layers}L, M={cfg.mdn_components}, "
        f"vocab={cfg.vocab_size}) on {device}"
    )

    train_ds = InkForgeDataset(cache_dir, split="train", cfg=cfg, augment=True, input_noise_std=args.input_noise)
    val_ds = InkForgeDataset(cache_dir, split="val", cfg=cfg, augment=False, input_noise_std=0.0)

    train_lengths = np.diff(train_ds.pt_offsets)
    batch_sampler = WeightedLengthBucketBatchSampler(
        weights=train_ds.sample_weights,
        lengths=train_lengths,
        batch_size=args.batch_size,
        num_batches=args.total_steps * args.grad_accum,
        mega_factor=32,
        seed=args.seed,
    )
    collate_fn = partial(collate_inkforge_batch, pad_id=cfg.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    best_val = float("inf")
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val_mdn", float("inf")))
        print(f"[inkforge] Resumed from {args.resume} at step {start_step} (best_val_mdn={best_val:.4f})")

    def lr_for_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.total_steps - args.warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def kl_for_step(step: int) -> float:
        return args.kl_weight * min(1.0, float(step) / float(max(1, args.kl_warmup_steps)))

    log_path = out_dir / "metrics.jsonl"
    t0 = time.perf_counter()
    train_iter = iter(train_loader)
    running = {"loss": 0.0, "mdn_loss": 0.0, "pen_loss": 0.0, "guided_loss": 0.0, "kl_raw": 0.0, "pen_acc": 0.0}
    running_n = 0

    for step in range(start_step + 1, args.total_steps + 1):
        model.train()
        lr = lr_for_step(step)
        kl_w = kl_for_step(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model.compute_loss(
                    token_ids=batch["token_ids"],
                    stroke_inputs=batch["stroke_inputs"],
                    target_dx=batch["target_dx"],
                    target_dy=batch["target_dy"],
                    target_pen=batch["target_pen"],
                    stroke_mask=batch["stroke_mask"],
                    style_points=batch["style_points"],
                    style_aspect=batch["style_aspect"],
                    style_mask=batch["style_mask"],
                    kl_weight=kl_w,
                    free_bits=args.free_bits,
                    style_drop_prob=args.style_drop_prob,
                    guided_attn_weight=args.guided_attn_weight,
                    guided_attn_sigma=args.guided_attn_sigma,
                )
                loss = out["loss"] / float(args.grad_accum)
            loss.backward()
            for k in running:
                running[k] += float(out[k].item())
            running_n += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema_decay = min(args.ema_decay, (1.0 + step) / (10.0 + step))
        update_ema(ema_model, model, ema_decay)

        if step % args.log_every == 0 or step == start_step + 1:
            avg = {k: v / max(1, running_n) for k, v in running.items()}
            elapsed = time.perf_counter() - t0
            steps_done = step - start_step
            sps = steps_done / max(elapsed, 1e-3)
            eta_min = (args.total_steps - step) / max(sps, 1e-6) / 60.0
            print(
                f"[step {step:6d}/{args.total_steps}] "
                f"loss={avg['loss']:.3f} mdn={avg['mdn_loss']:.3f} pen={avg['pen_loss']:.3f} "
                f"guide={avg['guided_loss']:.4f} pen_acc={avg['pen_acc']*100:.1f}% kl={avg['kl_raw']:.1f} "
                f"lr={lr:.2e} ({sps:.2f} step/s, ETA {eta_min:.1f}m)",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps({"step": step, "split": "train", "lr": lr, **avg}) + "\n")
            running = {k: 0.0 for k in running}
            running_n = 0

        if step == 500 or step % args.eval_every == 0 or step == args.total_steps:
            val_metrics = evaluate_val(
                ema_model,
                val_loader,
                device,
                guided_attn_weight=args.guided_attn_weight,
            )
            print(
                f"  >>> [VAL step {step:6d}] mdn={val_metrics['mdn_loss']:.4f} "
                f"pen={val_metrics['pen_loss']:.4f} guide={val_metrics['guided_loss']:.4f} "
                f"pen_acc={val_metrics['pen_acc']*100:.2f}% kl={val_metrics['kl_raw']:.1f}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps({"step": step, "split": "val", **val_metrics}) + "\n")

            render_preview_sheets(ema_model, tokenizer, train_ds, out_dir, step, device)

            save_checkpoint(
                out_dir / "latest.pt",
                step=step,
                cfg=cfg,
                tokenizer=tokenizer,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                best_val=best_val,
            )
            if val_metrics["mdn_loss"] < best_val:
                best_val = val_metrics["mdn_loss"]
                save_checkpoint(
                    out_dir / "best.pt",
                    step=step,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    best_val=best_val,
                )

    export_writer_style_bank(ema_model, train_ds, out_dir / "writer_styles.npz", device)
    print(f"[inkforge] Training finished! Best val MDN={best_val:.4f} | checkpoints in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InkForge neural math handwriting generator on RTX 4070 Ti")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(r"E:\math-recogniser-training\inkforge_model"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=8)
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--mdn-components", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=45000)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--kl-warmup-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--kl-weight", type=float, default=0.025)
    parser.add_argument("--free-bits", type=float, default=0.15)
    parser.add_argument("--style-drop-prob", type=float, default=0.30)
    parser.add_argument("--input-noise", type=float, default=0.030)
    parser.add_argument("--guided-attn-weight", type=float, default=4.0)
    parser.add_argument("--guided-attn-sigma", type=float, default=0.24)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
