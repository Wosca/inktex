"""Train InkTeX v2 Tier 1 Zero-Shot Writer Style Encoder + FiLM (Frozen Base Backbone).

Trains ONLY `style_encoder.*` and `stroke_encoder.blocks.{film_layers}.film*` (~294K params)
on multi-writer human handwriting (`crohme_train` + `hwrt_train` from `v2-shards/train_online`)
with writer-coupled style transforms (identical writer slant/aspect/tilt applied to BOTH the
support strokes and the query equation) and `label_smoothing=0.0`, validating strictly on
completely UNSEEN real human writers in `val_crohme` (57 unseen writers, 1,230 equations).

Because `StrokeEncoderBlock.film` has `bias=False` and `WriterStyleEncoder` outputs exact zeros
when `style_mask` is all-True, uncalibrated base recognition remains 100.00% bit-for-bit identical
to the input checkpoint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from data.bimodal_shards import iter_shard  # noqa: E402
from v2.ink_preprocess import (  # noqa: E402
    STYLE_MAX_STROKES,
    STYLE_POINTS,
    InkPreprocessV2Config,
    chunk_ink_v2,
    clean_raw_strokes,
    estimate_expression_origin,
    estimate_glyph_scale,
    extract_style_strokes,
)
from v2.model import InkTeXV2Config, InkTeXV2Model  # noqa: E402
from v2.shards_v2 import (  # noqa: E402
    AugmentV2Config,
    _augment_raster_photometric,
    collate_v2_batch,
    resolve_v2_shards,
    unpack_raw_strokes_npz,
)
from v2.tokenizer_v2 import LatexTokenizerV2  # noqa: E402
from v2.vision_raster import (  # noqa: E402
    WIDTH_BUCKETS_V2,
    render_strokes_to_strip_v2,
)


def sample_writer_affine(rng: np.random.Generator) -> np.ndarray:
    """Draw a realistic writer-level linear style transform (slant/shear, aspect ratio, tilt)."""
    angle = math.radians(float(rng.uniform(-7.0, 7.0)))
    shear = float(rng.uniform(-0.22, 0.22))
    sx = float(rng.uniform(0.78, 1.28))
    sy = float(rng.uniform(0.82, 1.22))
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]]) @ np.array([[1.0, shear], [0.0, 1.0]]) @ np.diag([sx, sy])


def apply_linear_warp(
    strokes: list[np.ndarray],
    mat: np.ndarray,
    rng: np.random.Generator | None = None,
    stroke_shift: float = 0.0,
) -> list[np.ndarray]:
    clean_xy, _ = clean_raw_strokes(strokes)
    if not clean_xy:
        return []
    h = estimate_glyph_scale(clean_xy)
    origin = estimate_expression_origin(clean_xy)
    out: list[np.ndarray] = []
    for st in clean_xy:
        pts = (st - origin) @ mat.T + origin
        if rng is not None and stroke_shift > 0:
            pts = pts + rng.normal(0.0, stroke_shift * h, size=(1, 2))
        out.append(pts)
    return out


def load_writers_from_split(
    shards_root: Path,
    split: str,
    tokenizer: LatexTokenizerV2,
    *,
    min_samples_per_writer: int = 3,
    max_seq_tokens: int = 256,
) -> dict[str, list[dict[str, Any]]]:
    shard_paths = resolve_v2_shards(shards_root, split)
    by_writer: dict[str, list[dict[str, Any]]] = defaultdict(list)

    t0 = time.perf_counter()
    for sp in shard_paths:
        for rec in iter_shard(sp):
            if rec.get("npz") is None:
                continue
            meta = json.loads(rec["json"])
            if meta.get("synthetic"):
                continue
            wid = meta.get("writer_id")
            if not wid or str(wid).lower() in ("", "unknown", "none", "synthetic", "null"):
                continue
            src = meta.get("source", split)
            label = meta["normalized_latex"]
            body = tokenizer.encode_body(label)
            if not body or len(body) + 2 > max_seq_tokens or tokenizer.unk_id in body:
                continue
            key = f"{src}:{wid}"
            by_writer[key].append(
                {
                    "id": meta.get("id", ""),
                    "source": src,
                    "writer_key": key,
                    "latex": label,
                    "body": body,
                    "npz": rec["npz"],
                }
            )

    filtered = {wk: rows for wk, rows in by_writer.items() if len(rows) >= min_samples_per_writer}
    print(
        f"[data:{split}] Scanned {len(shard_paths)} shards in {time.perf_counter()-t0:.1f}s | "
        f"writers(>={min_samples_per_writer})={len(filtered)} ({sum(len(v) for v in filtered.values())} rows)"
    )
    return filtered


def build_support_style(
    writer_rows: list[dict[str, Any]],
    exclude_idx: int,
    rng: np.random.Generator,
    *,
    writer_mat: np.ndarray | None = None,
    min_strokes: int = 6,
    max_strokes: int = STYLE_MAX_STROKES,
    drop_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if drop_empty or len(writer_rows) <= 1:
        return extract_style_strokes([], max_strokes=STYLE_MAX_STROKES, points_per_stroke=STYLE_POINTS)

    other_indices = [i for i in range(len(writer_rows)) if i != exclude_idx]
    rng.shuffle(other_indices)
    target_strokes = int(rng.integers(min_strokes, max_strokes + 1))

    collected: list[np.ndarray] = []
    for idx in other_indices:
        strokes, _ = unpack_raw_strokes_npz(writer_rows[idx]["npz"])
        if not strokes:
            continue
        if writer_mat is not None:
            strokes = apply_linear_warp(strokes, writer_mat)
        collected.extend(strokes)
        if len(collected) >= target_strokes:
            break

    if not collected:
        return extract_style_strokes([], max_strokes=STYLE_MAX_STROKES, points_per_stroke=STYLE_POINTS)

    if len(collected) > target_strokes:
        start = int(rng.integers(0, len(collected) - target_strokes + 1))
        collected = collected[start : start + target_strokes]

    return extract_style_strokes(collected, max_strokes=STYLE_MAX_STROKES, points_per_stroke=STYLE_POINTS)


def build_sample_item(
    row: dict[str, Any],
    tokenizer: LatexTokenizerV2,
    ink_cfg: InkPreprocessV2Config,
    rng: np.random.Generator,
    augment: AugmentV2Config | None,
    *,
    writer_mat: np.ndarray | None = None,
) -> dict[str, Any] | None:
    strokes, times = unpack_raw_strokes_npz(row["npz"])
    if not strokes:
        return None

    if writer_mat is not None:
        strokes = apply_linear_warp(
            strokes,
            writer_mat,
            rng=rng if (augment and augment.enabled) else None,
            stroke_shift=augment.stroke_shift if (augment and augment.enabled) else 0.0,
        )
    if augment is not None and augment.enabled:
        if times is not None and rng.random() < augment.timing_dropout:
            times = None
        jitter = augment.post_resample_jitter
    else:
        jitter = 0.0

    chunked = chunk_ink_v2(strokes, times, config=ink_cfg, post_resample_jitter=jitter, rng=rng)
    v = int(chunked.valid_chunks)
    if v <= 0:
        return None

    raster = render_strokes_to_strip_v2(
        strokes,
        width_buckets=WIDTH_BUCKETS_V2,
        stroke_width=float(rng.uniform(1.8, 3.2)) if (augment and augment.enabled) else None,
    )
    img_np = np.asarray(raster.image, dtype=np.uint8).transpose(2, 0, 1)
    used_w = int(raster.image_width)
    if augment is not None and augment.enabled:
        img_np = _augment_raster_photometric(img_np, used_w, rng, augment, allow_band_mask=False)

    body = row["body"]
    return {
        "id": row["id"],
        "source": row["source"],
        "writer_id": row["writer_key"],
        "normalized_latex": row["latex"],
        "synthetic": False,
        "coarsened": bool(chunked.coarsened),
        "chunk_points": torch.from_numpy(chunked.chunk_points[:v].copy()),
        "chunk_point_mask": torch.from_numpy(chunked.chunk_point_mask[:v].copy()),
        "chunk_meta": torch.from_numpy(chunked.chunk_meta[:v].copy()),
        "chunk_mask": torch.from_numpy(chunked.chunk_mask[:v].copy()),
        "image": torch.from_numpy(img_np.copy()),
        "image_width": used_w,
        "l2r_tokens": torch.tensor([tokenizer.bos_l2r_id, *body, tokenizer.eos_id], dtype=torch.long),
        "r2l_tokens": torch.tensor([tokenizer.bos_r2l_id, *reversed(body), tokenizer.eos_id], dtype=torch.long),
    }


def sample_writer_balanced_batch(
    writers_map: dict[str, list[dict[str, Any]]],
    crohme_keys: list[str],
    hwrt_keys: list[str],
    batch_size: int,
    tokenizer: LatexTokenizerV2,
    ink_cfg: InkPreprocessV2Config,
    rng: np.random.Generator,
    augment: AugmentV2Config,
    *,
    crohme_prob: float = 0.78,
    coupled_warp_prob: float = 0.70,
    empty_style_prob: float = 0.10,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    items: list[dict[str, Any]] = []
    s_pts_list: list[np.ndarray] = []
    s_asp_list: list[np.ndarray] = []
    s_msk_list: list[np.ndarray] = []

    attempts = 0
    while len(items) < batch_size and attempts < batch_size * 4:
        attempts += 1
        use_crohme = bool(crohme_keys) and (not hwrt_keys or rng.random() < crohme_prob)
        pool = crohme_keys if use_crohme else hwrt_keys
        wk = pool[int(rng.integers(0, len(pool)))]
        rows = writers_map[wk]
        q_idx = int(rng.integers(0, len(rows)))

        # Apply the SAME writer-style transform to both support & query strokes!
        w_mat = sample_writer_affine(rng) if rng.random() < coupled_warp_prob else None
        item = build_sample_item(rows[q_idx], tokenizer, ink_cfg, rng, augment, writer_mat=w_mat)
        if item is None:
            continue

        drop = rng.random() < empty_style_prob
        sp, sa, sm = build_support_style(rows, q_idx, rng, writer_mat=w_mat, drop_empty=drop)
        items.append(item)
        s_pts_list.append(sp)
        s_asp_list.append(sa)
        s_msk_list.append(sm)

    batch = collate_v2_batch(items, pad_id=tokenizer.pad_id, static_buckets=True)
    return (
        batch,
        torch.from_numpy(np.stack(s_pts_list, axis=0)),
        torch.from_numpy(np.stack(s_asp_list, axis=0)),
        torch.from_numpy(np.stack(s_msk_list, axis=0)),
    )


def prepare_unseen_val_examples(
    val_writers: dict[str, list[dict[str, Any]]],
    tokenizer: LatexTokenizerV2,
    ink_cfg: InkPreprocessV2Config,
    *,
    max_per_writer: int = 10,
    max_total: int = 480,
    seed: int = 1234,
) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]:
    """Build deterministic real unseen-writer validation pairs from `val_crohme` (zero warp)."""
    rng = np.random.default_rng(seed)
    examples: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    for wk in sorted(val_writers.keys()):
        rows = val_writers[wk]
        if len(rows) < 2:
            continue
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        taken = 0
        for q_idx in idxs:
            item = build_sample_item(rows[q_idx], tokenizer, ink_cfg, rng, augment=None, writer_mat=None)
            if item is None:
                continue
            sp, sa, sm = build_support_style(rows, q_idx, rng, min_strokes=10, max_strokes=STYLE_MAX_STROKES)
            if not (~sm).any():
                continue
            examples.append((item, sp, sa, sm))
            taken += 1
            if taken >= max_per_writer or len(examples) >= max_total:
                break
        if len(examples) >= max_total:
            break
    print(f"[val] Built {len(examples)} real unseen-writer validation pairs from val_crohme")
    return examples


@torch.inference_mode()
def evaluate_unseen_writers(
    model: InkTeXV2Model,
    val_examples: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]],
    pad_id: int,
    device: torch.device,
    *,
    batch_size: int = 32,
    stroke_only: bool = False,
    use_style: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    style_norms: list[float] = []

    for start in range(0, len(val_examples), batch_size):
        chunk = val_examples[start : start + batch_size]
        batch = collate_v2_batch([c[0] for c in chunk], pad_id=pad_id, static_buckets=True)

        cp = batch["chunk_points"].to(device, non_blocking=True)
        cpm = batch["chunk_point_mask"].to(device, non_blocking=True)
        cm = batch["chunk_meta"].to(device, non_blocking=True)
        ck = batch["chunk_mask"].to(device, non_blocking=True)
        img = batch["images"].to(device, non_blocking=True)
        iw = batch["image_widths"].to(device, non_blocking=True)
        l2r = batch["l2r_tokens"].to(device, non_blocking=True)

        if use_style:
            sp = torch.from_numpy(np.stack([c[1] for c in chunk], axis=0)).to(device)
            sa = torch.from_numpy(np.stack([c[2] for c in chunk], axis=0)).to(device)
            sm = torch.from_numpy(np.stack([c[3] for c in chunk], axis=0)).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                ws = model.style_encoder(sp, sa, sm)
            style_norms.extend(ws.float().norm(dim=-1).cpu().tolist())
        else:
            ws = None

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(
                decoder_input=l2r[:, :-1],
                token_padding_mask=l2r[:, :-1].eq(pad_id),
                images=None if stroke_only else img,
                image_widths=None if stroke_only else iw,
                chunk_points=cp,
                chunk_point_mask=cpm,
                chunk_meta=cm,
                chunk_mask=ck,
                writer_style=ws,
                modality_mode="stroke" if stroke_only else "bimodal",
            )
            logits = out["logits"].float()
            targets = l2r[:, 1:]
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_id, reduction="sum")
            total_loss += float(loss_sum.item())
            total_tokens += int(targets.ne(pad_id).sum().item())

    return {
        "ce_loss": total_loss / max(1, total_tokens),
        "style_norm": float(np.mean(style_norms)) if style_norms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tier 1 WriterStyleEncoder + FiLM with frozen base model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, default=Path(r"E:\math-recogniser-training\v2-shards"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--stroke-only-prob", type=float, default=0.50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = InkTeXV2Config.from_dict(payload.get("model_config") or payload["config"])
    cfg.vision_pretrained = False
    model = InkTeXV2Model(cfg)
    state = payload.get("ema_state_dict") or payload.get("model_state_dict") or payload["model"]
    model.load_state_dict(state, strict=False)
    model.set_lora_enabled(False)
    model.to(device)

    trainable_params: list[nn.Parameter] = []
    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if name.startswith("style_encoder.") or ".film" in name:
            param.requires_grad = True
            trainable_params.append(param)
            trainable_names.append(name)
        else:
            param.requires_grad = False

    n_train = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable Tier-1 FiLM+Style params: {n_train:,} / {n_total:,} ({100.0 * n_train / n_total:.2f}%)")

    tokenizer = LatexTokenizerV2(payload["tokenizer"]["tokens"])
    ink_cfg = InkPreprocessV2Config()
    aug_cfg = AugmentV2Config(
        enabled=True,
        post_resample_jitter=0.003,
        stroke_shift=0.015,
        timing_dropout=0.40,
        vision_band_mask_prob=0.0,
    )

    train_writers = load_writers_from_split(args.shards_root, "train_online", tokenizer)
    val_writers = load_writers_from_split(args.shards_root, "val_crohme", tokenizer)
    crohme_keys = sorted(k for k in train_writers if k.startswith("crohme"))
    hwrt_keys = sorted(k for k in train_writers if not k.startswith("crohme"))

    val_examples = prepare_unseen_val_examples(val_writers, tokenizer, ink_cfg)

    base_bi = evaluate_unseen_writers(model, val_examples, tokenizer.pad_id, device, stroke_only=False, use_style=False)
    base_st = evaluate_unseen_writers(model, val_examples, tokenizer.pad_id, device, stroke_only=True, use_style=False)
    print(f"[baseline val_crohme unseen writers] bimodal CE={base_bi['ce_loss']:.5f} | stroke-only CE={base_st['ce_loss']:.5f}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        warmup = min(80, max(15, args.steps // 10))
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, args.steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rng = np.random.default_rng(args.seed)

    best_score = 0.5 * (base_bi["ce_loss"] + base_st["ce_loss"])
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in set(trainable_names)}
    best_step = 0
    best_val_info: dict[str, Any] = {
        "bimodal_ce": base_bi["ce_loss"],
        "stroke_only_ce": base_st["ce_loss"],
        "bimodal_delta": 0.0,
        "stroke_only_delta": 0.0,
        "style_norm": 0.0,
    }

    t0 = time.perf_counter()
    running_ce = 0.0
    log_n = 0

    for step in range(1, args.steps + 1):
        model.eval()
        model.style_encoder.train()

        batch, sp, sa, sm = sample_writer_balanced_batch(
            train_writers,
            crohme_keys,
            hwrt_keys,
            args.batch_size,
            tokenizer,
            ink_cfg,
            rng,
            aug_cfg,
        )
        cp = batch["chunk_points"].to(device, non_blocking=True)
        cpm = batch["chunk_point_mask"].to(device, non_blocking=True)
        cm = batch["chunk_meta"].to(device, non_blocking=True)
        ck = batch["chunk_mask"].to(device, non_blocking=True)
        img = batch["images"].to(device, non_blocking=True)
        iw = batch["image_widths"].to(device, non_blocking=True)
        l2r = batch["l2r_tokens"].to(device, non_blocking=True)
        r2l = batch["r2l_tokens"].to(device, non_blocking=True)
        stroke_only = rng.random() < args.stroke_only_prob

        sp = sp.to(device, non_blocking=True)
        sa = sa.to(device, non_blocking=True)
        sm = sm.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            ws = model.style_encoder(sp, sa, sm)
            out = model(
                decoder_input=l2r[:, :-1],
                token_padding_mask=l2r[:, :-1].eq(tokenizer.pad_id),
                r2l_decoder_input=r2l[:, :-1],
                r2l_token_padding_mask=r2l[:, :-1].eq(tokenizer.pad_id),
                images=None if stroke_only else img,
                image_widths=None if stroke_only else iw,
                chunk_points=cp,
                chunk_point_mask=cpm,
                chunk_meta=cm,
                chunk_mask=ck,
                writer_style=ws,
                modality_mode="stroke" if stroke_only else "bimodal",
            )
            # Zero label smoothing: never incentivize FiLM to degrade sharp base predictions!
            l2r_loss = F.cross_entropy(
                out["logits"].reshape(-1, cfg.vocab_size),
                l2r[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id,
                label_smoothing=0.0,
            )
            r2l_loss = F.cross_entropy(
                out["r2l_logits"].reshape(-1, cfg.vocab_size),
                r2l[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id,
                label_smoothing=0.0,
            )
            loss = l2r_loss + 0.5 * r2l_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        running_ce += float(loss.item())
        log_n += 1

        if step % 100 == 0 or step == 1:
            q_gate = float(torch.sigmoid(model.style_encoder.quality_gate).item())
            f_scales = [round(float(model.stroke_encoder.blocks[i].film_scale.tanh().item()), 4) for i in cfg.film_layers]
            print(
                f"[step {step:4d}/{args.steps}] ce={running_ce/log_n:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} q_gate={q_gate:.3f} film_scale={f_scales} ({time.perf_counter()-t0:.0f}s)"
            )
            running_ce = 0.0
            log_n = 0

        if step % args.eval_every == 0 or step == args.steps:
            val_bi = evaluate_unseen_writers(model, val_examples, tokenizer.pad_id, device, stroke_only=False, use_style=True)
            val_st = evaluate_unseen_writers(model, val_examples, tokenizer.pad_id, device, stroke_only=True, use_style=True)
            score = 0.5 * (val_bi["ce_loss"] + val_st["ce_loss"])
            d_bi = val_bi["ce_loss"] - base_bi["ce_loss"]
            d_st = val_st["ce_loss"] - base_st["ce_loss"]
            improved = score < best_score
            if improved:
                best_score = score
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in set(trainable_names)}
                best_val_info = {
                    "bimodal_ce": val_bi["ce_loss"],
                    "stroke_only_ce": val_st["ce_loss"],
                    "bimodal_delta": d_bi,
                    "stroke_only_delta": d_st,
                    "style_norm": val_bi["style_norm"],
                }
            print(
                f"  -> [val_crohme step {step}] bimodal CE={val_bi['ce_loss']:.5f} ({d_bi:+.5f}) | "
                f"stroke-only CE={val_st['ce_loss']:.5f} ({d_st:+.5f}) | "
                f"style_norm={val_bi['style_norm']:.2f} {'*BEST*' if improved else ''}"
            )

    out_sd = {k: v.clone() for k, v in state.items()}
    for k, v in best_state.items():
        out_sd[k] = v

    model.load_state_dict(out_sd, strict=False)
    model.eval()
    check_bi = evaluate_unseen_writers(model, val_examples, tokenizer.pad_id, device, stroke_only=False, use_style=False)
    assert abs(check_bi["ce_loss"] - base_bi["ce_loss"]) < 1e-5, (
        f"Zero-style invariance broken: {check_bi['ce_loss']} vs {base_bi['ce_loss']}"
    )

    out_payload = dict(payload)
    out_payload["model_state_dict"] = out_sd
    out_payload["tier1_writer_style"] = {
        "best_step": best_step,
        "baseline_val_crohme_bimodal_ce": base_bi["ce_loss"],
        "baseline_val_crohme_stroke_only_ce": base_st["ce_loss"],
        "best_val_crohme": best_val_info,
        "train_writers": len(train_writers),
        "val_writers": len(val_writers),
        "zero_style_check_ce": check_bi["ce_loss"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, args.output)
    print(f"[done] Saved release checkpoint with Tier-1 FiLM (best_step={best_step}) -> {args.output}")
    print(f"[summary] {json.dumps(out_payload['tier1_writer_style'], indent=2)}")


if __name__ == "__main__":
    main()
