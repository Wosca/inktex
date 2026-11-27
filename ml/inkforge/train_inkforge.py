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
    max_batches: int = 220,
    guided_attn_weight: float = 4.0,
    transition_weight: float = 4.0,
) -> dict[str, float]:
    """Population-weighted validation metrics (exact valid-step denominators)."""
    model.eval()
    sums = {
        "valid_count": 0.0,
        "coord_count": 0.0,
        "weighted_coord_sum": 0.0,
        "draw_count": 0.0,
        "jump_count": 0.0,
        "mdn_num": 0.0,
        "mdn_w_num": 0.0,
        "draw_mdn_num": 0.0,
        "jump_mdn_num": 0.0,
        "pen_num": 0.0,
        "pen_acc_num": 0.0,
        "guided_num": 0.0,
        "attn_max_num": 0.0,
        "cursor_overshoot_num": 0.0,
        "sigma_floor_num": 0.0,
        "cursor_end_sum": 0.0,
        "cursor_prog_sum": 0.0,
        "kl_sum": 0.0,
        "loss_sum": 0.0,
        "n_rows": 0.0,
        "n_batches": 0.0,
    }
    cursor_gate = 0.0
    for n_b, batch in enumerate(val_loader):
        if n_b >= max_batches:
            break
        bsz = int(batch["token_ids"].shape[0])
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
                transition_weight=transition_weight,
            )
        vc = float(out["valid_count"].item())
        cc = float(out["coord_count"].item())
        wc = float(out["weighted_coord_sum"].item())
        dc = float(out["draw_count"].item())
        jc = float(out["jump_count"].item())

        sums["valid_count"] += vc
        sums["coord_count"] += cc
        sums["weighted_coord_sum"] += wc
        sums["draw_count"] += dc
        sums["jump_count"] += jc
        sums["mdn_num"] += float(out["mdn_loss"].item()) * cc
        sums["mdn_w_num"] += float(out["mdn_w_loss"].item()) * wc
        sums["draw_mdn_num"] += float(out["draw_mdn_loss"].item()) * dc
        sums["jump_mdn_num"] += float(out["jump_mdn_loss"].item()) * jc
        sums["pen_num"] += float(out["pen_loss"].item()) * vc
        sums["pen_acc_num"] += float(out["pen_acc"].item()) * vc
        sums["guided_num"] += float(out["guided_loss"].item()) * vc
        sums["attn_max_num"] += float(out["attn_max_txt"].item()) * vc
        sums["cursor_overshoot_num"] += float(out["cursor_overshoot"].item()) * vc
        sums["sigma_floor_num"] += float(out["sigma_floor_frac"].item()) * cc
        sums["cursor_end_sum"] += float(out["cursor_end_loss"].item()) * bsz
        sums["cursor_prog_sum"] += float(out["cursor_progress"].item()) * bsz
        sums["kl_sum"] += float(out["kl_raw"].item())
        sums["loss_sum"] += float(out["loss"].item())
        sums["n_rows"] += bsz
        sums["n_batches"] += 1.0
        cursor_gate = float(out["cursor_gate"].item())

    vc = max(1.0, sums["valid_count"])
    cc = max(1.0, sums["coord_count"])
    wc = max(1.0, sums["weighted_coord_sum"])
    dc = max(1.0, sums["draw_count"])
    jc = max(1.0, sums["jump_count"])
    nr = max(1.0, sums["n_rows"])
    nb = max(1.0, sums["n_batches"])

    mdn_w = sums["mdn_w_num"] / wc
    pen = sums["pen_num"] / vc
    return {
        "loss": sums["loss_sum"] / nb,
        "val_score": mdn_w + 2.0 * pen,
        "mdn_loss": sums["mdn_num"] / cc,
        "mdn_w_loss": mdn_w,
        "draw_mdn_loss": sums["draw_mdn_num"] / dc,
        "jump_mdn_loss": sums["jump_mdn_num"] / jc,
        "pen_loss": pen,
        "guided_loss": sums["guided_num"] / vc,
        "attn_max_txt": sums["attn_max_num"] / vc,
        "cursor_end_loss": sums["cursor_end_sum"] / nr,
        "cursor_progress": sums["cursor_prog_sum"] / nr,
        "cursor_overshoot": sums["cursor_overshoot_num"] / vc,
        "cursor_gate": cursor_gate,
        "sigma_floor_frac": sums["sigma_floor_num"] / cc,
        "kl_raw": sums["kl_sum"] / nb,
        "pen_acc": sums["pen_acc_num"] / vc,
        "n_rows": sums["n_rows"],
    }


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
    *,
    cfg_scale: float = 1.0,
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
    devices = [device] if device.type == "cuda" else []
    summary_parts: list[str] = []
    with torch.random.fork_rng(devices=devices):
        for cfg_i, (tag, z_in, temp) in enumerate(configs):
            torch.manual_seed(12345 + cfg_i)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(12345 + cfg_i)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                strokes_list, meta_list = model.generate_batch(
                    tok_t,
                    style_z=z_in,
                    temperature=temp,
                    pen_temperature=1.0,
                    cfg_scale=cfg_scale,
                    max_steps=512,
                    return_meta=True,
                )
            done = sum(1 for m in meta_list if m["finished"])
            avg_steps = sum(m["steps"] for m in meta_list) / max(1, len(meta_list))
            summary_parts.append(f"{tag}:{done}/{len(meta_list)}(avg={avg_steps:.0f}st)")
            pairs = list(zip(PREVIEW_PROMPTS, strokes_list))
            render_contact_sheet(
                pairs,
                out_dir / "previews" / f"step_{step:06d}_{tag}.png",
                cols=3,
                tile_w=440,
                tile_h=150,
                color_by_order=False,
            )
    print(f"  >>> [PREVIEW step {step:6d}] " + " | ".join(summary_parts), flush=True)


@torch.inference_mode()
def export_writer_style_bank(
    model: InkForgeModel,
    dataset: InkForgeDataset,
    out_path: Path,
    device: torch.device,
    *,
    refs_per_writer: int = 5,
    batch_size: int = 128,
) -> Path:
    """Export ONE aggregated `mu_z` per unique named writer (equal representation across all writers!)."""
    model.eval()
    peer_wids = sorted(dataset.writer_peers.keys())
    if not peer_wids:
        bank = np.zeros((1, model.cfg.style_dim), dtype=np.float32)
        np.savez_compressed(out_path, mu_bank=bank, mean=bank.mean(axis=0), std=bank.std(axis=0))
        return out_path

    # Deterministically pick up to `refs_per_writer` evenly spaced samples for each writer
    sample_indices: list[int] = []
    writer_slices: list[tuple[int, int]] = []
    for wid in peer_wids:
        idxs = dataset.writer_peers[wid]
        k = min(refs_per_writer, len(idxs))
        pick = idxs[np.linspace(0, len(idxs) - 1, num=k, dtype=np.int64)]
        start_pos = len(sample_indices)
        sample_indices.extend(int(i) for i in pick.tolist())
        writer_slices.append((start_pos, len(sample_indices)))

    orig_aug = dataset.augment
    dataset.augment = False
    zs: list[np.ndarray] = []
    try:
        for start in range(0, len(sample_indices), batch_size):
            idxs = sample_indices[start : start + batch_size]
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

    all_z = np.concatenate(zs, axis=0).astype(np.float32)
    writer_bank = np.stack([all_z[a:b].mean(axis=0) for a, b in writer_slices], axis=0).astype(np.float32)
    np.savez_compressed(
        out_path,
        mu_bank=writer_bank,
        writer_ids=np.asarray(peer_wids, dtype=np.int32),
        mean=writer_bank.mean(axis=0),
        std=writer_bank.std(axis=0),
    )
    print(
        f"[style-bank] Exported {len(writer_bank):,} balanced unique writers "
        f"(<= {refs_per_writer} refs/writer) -> {out_path}",
        flush=True,
    )
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
        sigma_min=args.sigma_min,
    )

    model = InkForgeModel(cfg, grad_checkpoint=args.grad_checkpoint).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.grad_checkpoint = False
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
    val_expr_idxs = np.where(val_ds.source_ids <= 2)[0].tolist()
    val_sym_idxs = np.where(val_ds.source_ids >= 3)[0].tolist()
    val_expr_loader = DataLoader(
        torch.utils.data.Subset(val_ds, val_expr_idxs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    ) if val_expr_idxs else None
    val_sym_loader = DataLoader(
        torch.utils.data.Subset(val_ds, val_sym_idxs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    ) if val_sym_idxs else None

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
    running = {
        "loss": 0.0,
        "mdn_loss": 0.0,
        "draw_mdn_loss": 0.0,
        "jump_mdn_loss": 0.0,
        "pen_loss": 0.0,
        "guided_loss": 0.0,
        "attn_max_txt": 0.0,
        "cursor_end_loss": 0.0,
        "cursor_progress": 0.0,
        "cursor_overshoot": 0.0,
        "cursor_gate": 0.0,
        "kl_raw": 0.0,
        "pen_acc": 0.0,
    }
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
                    text_drop_prob=args.text_drop_prob,
                    guided_attn_weight=args.guided_attn_weight,
                    guided_attn_sigma=args.guided_attn_sigma,
                    transition_weight=args.transition_weight,
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
                f"loss={avg['loss']:.3f} draw={avg['draw_mdn_loss']:.3f} jump={avg['jump_mdn_loss']:.3f} "
                f"pen={avg['pen_loss']:.3f} guide={avg['guided_loss']:.4f} max_txt={avg['attn_max_txt']:.3f} "
                f"cur={avg['cursor_end_loss']:.3f}(p={avg['cursor_progress']:.2f},g={avg['cursor_gate']:.2f}) "
                f"pen_acc={avg['pen_acc']*100:.1f}% kl={avg['kl_raw']:.1f} "
                f"lr={lr:.2e} ({sps:.2f} step/s, ETA {eta_min:.1f}m)",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps({"step": step, "split": "train", "lr": lr, **avg}) + "\n")
            running = {k: 0.0 for k in running}
            running_n = 0

        if step == 500 or step % args.eval_every == 0 or step == args.total_steps:
            val_expr = (
                evaluate_val(
                    ema_model,
                    val_expr_loader,
                    device,
                    guided_attn_weight=args.guided_attn_weight,
                    transition_weight=args.transition_weight,
                )
                if val_expr_loader is not None
                else None
            )
            val_sym = (
                evaluate_val(
                    ema_model,
                    val_sym_loader,
                    device,
                    guided_attn_weight=args.guided_attn_weight,
                    transition_weight=args.transition_weight,
                )
                if val_sym_loader is not None
                else None
            )
            primary_val = val_expr if val_expr is not None else val_sym
            assert primary_val is not None
            if val_expr is not None:
                print(
                    f"  >>> [VAL-EXPR step {step:6d}] score={val_expr['val_score']:.4f} "
                    f"draw={val_expr['draw_mdn_loss']:.4f} jump={val_expr['jump_mdn_loss']:.4f} "
                    f"pen={val_expr['pen_loss']:.4f} guide={val_expr['guided_loss']:.4f} "
                    f"max_txt={val_expr['attn_max_txt']:.3f} cur={val_expr['cursor_end_loss']:.3f}"
                    f"(p={val_expr['cursor_progress']:.2f},ov={val_expr['cursor_overshoot']*100:.1f}%) "
                    f"pen_acc={val_expr['pen_acc']*100:.2f}% sig_flr={val_expr['sigma_floor_frac']*100:.2f}% "
                    f"n={int(val_expr['n_rows'])}",
                    flush=True,
                )
            if val_sym is not None:
                print(
                    f"  >>> [VAL-SYM  step {step:6d}] score={val_sym['val_score']:.4f} "
                    f"draw={val_sym['draw_mdn_loss']:.4f} jump={val_sym['jump_mdn_loss']:.4f} "
                    f"pen={val_sym['pen_loss']:.4f} guide={val_sym['guided_loss']:.4f} "
                    f"max_txt={val_sym['attn_max_txt']:.3f} pen_acc={val_sym['pen_acc']*100:.2f}% "
                    f"n={int(val_sym['n_rows'])}",
                    flush=True,
                )
            with log_path.open("a", encoding="utf-8") as lf:
                if val_expr is not None:
                    lf.write(json.dumps({"step": step, "split": "val_expr", **val_expr}) + "\n")
                if val_sym is not None:
                    lf.write(json.dumps({"step": step, "split": "val_sym", **val_sym}) + "\n")

            # Update best score and write checkpoints BEFORE rendering preview sheets!
            improved = primary_val["val_score"] < best_val
            if improved:
                best_val = primary_val["val_score"]
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
            if improved and step >= args.eval_every:
                export_writer_style_bank(ema_model, train_ds, out_dir / "writer_styles.npz", device)

            render_preview_sheets(
                ema_model, tokenizer, train_ds, out_dir, step, device, cfg_scale=args.preview_cfg_scale
            )

    # Export style bank strictly from best.pt's EMA weights if available!
    best_path = out_dir / "best.pt"
    if best_path.is_file():
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        ema_model.load_state_dict(best_ckpt["ema_state_dict"])
        print(f"[inkforge] Loaded best.pt (step={best_ckpt.get('step')}) for final writer style bank export", flush=True)
    export_writer_style_bank(ema_model, train_ds, out_dir / "writer_styles.npz", device)
    print(f"[inkforge] Training finished! Best val score={best_val:.4f} | checkpoints in {out_dir}")


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--kl-warmup-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--kl-weight", type=float, default=0.025)
    parser.add_argument("--free-bits", type=float, default=0.15)
    parser.add_argument("--style-drop-prob", type=float, default=0.30)
    parser.add_argument("--text-drop-prob", type=float, default=0.0)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--preview-cfg-scale", type=float, default=1.0)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--input-noise", type=float, default=0.010)
    parser.add_argument("--guided-attn-weight", type=float, default=4.0)
    parser.add_argument("--guided-attn-sigma", type=float, default=0.24)
    parser.add_argument("--transition-weight", type=float, default=4.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
