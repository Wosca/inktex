"""Train InkTeX v2 Tier 2 Personal Ink-Only LoRA Adapter (`*.inktexlora`).

Trains ONLY the ink-only LoRA matrices (`q_proj`, `v_proj`, `out_proj` on stroke encoder blocks
`9, 10, 11, 12`, rank `r=8`, `alpha=16.0` -> ~110K params, ~221 KB FP16) while keeping 100% of
the base Stroke Encoder, Vision Encoder, Bimodal Fusion, and Shared Bidirectional Decoder frozen.

Supports BOTH:
  1. Native handwriting JSONL (`examples.jsonl` with `rawPoints`/`resampledPoints` strokes)
  2. iPad-exported correction JSONL (`inktex_<name>_corrections.jsonl` from `PersonalizationSheet`)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from v2.export_onnx_v2 import export_lora_adapter  # noqa: E402
from v2.ink_preprocess import (  # noqa: E402
    STYLE_MAX_STROKES,
    STYLE_POINTS,
    InkPreprocessV2Config,
    chunk_ink_v2,
    extract_style_strokes,
    warp_raw_strokes,
)
from v2.model import InkTeXV2Config, InkTeXV2Model  # noqa: E402
from v2.shards_v2 import (  # noqa: E402
    AugmentV2Config,
    _augment_raster_photometric,
    collate_v2_batch,
)
from v2.tokenizer_v2 import LatexTokenizerV2, standardize_latex_v2  # noqa: E402
from v2.vision_raster import (  # noqa: E402
    WIDTH_BUCKETS_V2,
    render_strokes_to_strip_v2,
)


def parse_strokes_record(raw_strokes: Any) -> tuple[list[np.ndarray], list[np.ndarray | None] | None]:
    """Parse strokes from either Native snapshot format or iPad exported corrections format."""
    xy_list: list[np.ndarray] = []
    t_list: list[np.ndarray | None] = []
    has_any_t = True

    if not isinstance(raw_strokes, list):
        return [], None

    for st in raw_strokes:
        pts_xy: list[list[float]] = []
        pts_t: list[float] = []
        st_has_t = True

        if isinstance(st, dict):
            pts = st.get("rawPoints") or st.get("resampledPoints") or st.get("points") or []
            for p in pts:
                if isinstance(p, dict) and isinstance(p.get("raw"), dict):
                    p = p["raw"]
                if isinstance(p, dict):
                    x, y = p.get("x"), p.get("y")
                    t = p.get("t", p.get("timeMs", p.get("timestamp")))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = p[0], p[1]
                    t = p[2] if len(p) >= 3 else None
                else:
                    continue
                if x is None or y is None:
                    continue
                xf, yf = float(x), float(y)
                if not (math.isfinite(xf) and math.isfinite(yf)):
                    continue
                pts_xy.append([xf, yf])
                if t is not None and math.isfinite(float(t)):
                    pts_t.append(float(t))
                else:
                    st_has_t = False
        elif isinstance(st, list):
            for p in st:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    xf, yf = float(p[0]), float(p[1])
                    if not (math.isfinite(xf) and math.isfinite(yf)):
                        continue
                    pts_xy.append([xf, yf])
                    if len(p) >= 3 and p[2] is not None and math.isfinite(float(p[2])):
                        pts_t.append(float(p[2]))
                    else:
                        st_has_t = False

        if pts_xy:
            xy_list.append(np.asarray(pts_xy, dtype=np.float64))
            if st_has_t and len(pts_t) == len(pts_xy):
                t_list.append(np.asarray(pts_t, dtype=np.float64))
            else:
                t_list.append(None)
                has_any_t = False

    return xy_list, (t_list if (has_any_t and xy_list) else None)


def load_personal_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Load personal examples + optional iPad header calibration strokes from JSONL."""
    examples: list[dict[str, Any]] = []
    calibration_strokes: list[np.ndarray] = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rec_type = obj.get("record_type")
        if rec_type == "writer_profile_header":
            cal_xy, _ = parse_strokes_record(obj.get("calibration_strokes", []))
            calibration_strokes.extend(cal_xy)
            continue

        latex = obj.get("latex") or obj.get("corrected_latex") or obj.get("normalized_latex")
        if not latex:
            continue
        strokes, times = parse_strokes_record(obj.get("strokes", []))
        if not strokes:
            continue
        examples.append(
            {
                "id": str(obj.get("id", f"row-{len(examples)}")),
                "latex": str(latex),
                "canonical_latex": standardize_latex_v2(str(latex)),
                "strokes": strokes,
                "times": times,
            }
        )
    return examples, calibration_strokes


def train_personal_lora(
    model: InkTeXV2Model,
    tokenizer: LatexTokenizerV2,
    examples: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    steps: int | None = None,
    batch_size: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    lora_dropout: float = 0.10,
    stroke_only_prob: float = 0.45,
    label_smoothing: float = 0.0,
    r2l_weight: float = 0.5,
    writer_style: torch.Tensor | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train ink-only LoRA in-place on `model` (all base weights stay 100% frozen)."""
    if not examples:
        raise ValueError("Cannot train personal LoRA with 0 examples")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    ink_cfg = InkPreprocessV2Config()

    valid: list[dict[str, Any]] = []
    for ex in examples:
        canon = ex.get("canonical_latex") or standardize_latex_v2(ex["latex"])
        body = tokenizer.encode_body(canon)
        if not body or len(body) + 2 > model.config.max_tokens or tokenizer.unk_id in body:
            continue
        valid.append({**ex, "body": body})
    if not valid:
        raise ValueError("No valid tokenizable examples for LoRA training")

    n_samples = len(valid)
    eff_bs = batch_size or (8 if n_samples <= 60 else 16)
    eff_bs = min(eff_bs, n_samples)
    if steps is None:
        if n_samples <= 30:
            eff_steps = 35
            lr = min(lr, 1.5e-4)
        elif n_samples <= 75:
            eff_steps = 110
        elif n_samples <= 200:
            eff_steps = 180
        else:
            eff_steps = 260
    else:
        eff_steps = int(steps)

    model.to(device)
    model.set_lora_enabled(True)
    lora_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            p.requires_grad = True
            lora_params.append(p)
        else:
            p.requires_grad = False

    for layer_idx in model.config.lora_layers:
        blk = model.stroke_encoder.blocks[layer_idx]
        for proj in (blk.q_proj, blk.v_proj, blk.out_proj):
            if proj.lora_a is not None and proj.lora_b is not None:
                torch.nn.init.kaiming_uniform_(proj.lora_a, a=math.sqrt(5))
                torch.nn.init.zeros_(proj.lora_b)
                if isinstance(proj.lora_drop, torch.nn.Dropout):
                    proj.lora_drop.p = lora_dropout

    aug_cfg = AugmentV2Config(
        enabled=True,
        rotation_degrees=3.0,
        shear=0.05,
        anisotropic_scale=(0.92, 1.08),
        post_resample_jitter=0.0025,
        stroke_shift=0.012,
        timing_dropout=0.35,
        stroke_order_swap_prob=0.0,
        stroke_reverse_prob=0.0,
        morphology_prob=0.25,
        intensity_jitter=0.08,
        vision_band_mask_prob=0.0,
    )

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        warmup = max(8, eff_steps // 8)
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, eff_steps - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ws_fixed = writer_style.detach().to(device) if writer_style is not None else None
    t0 = time.perf_counter()
    losses: list[float] = []

    for step in range(1, eff_steps + 1):
        model.eval()
        for layer_idx in model.config.lora_layers:
            blk = model.stroke_encoder.blocks[layer_idx]
            for proj in (blk.q_proj, blk.v_proj, blk.out_proj):
                proj.lora_drop.train()

        chosen_idx = rng.choice(n_samples, size=eff_bs, replace=(eff_bs > n_samples))
        items: list[dict[str, Any]] = []
        for i in chosen_idx:
            ex = valid[int(i)]
            st = warp_raw_strokes(
                ex["strokes"],
                rng,
                rotation_degrees=aug_cfg.rotation_degrees,
                shear=aug_cfg.shear,
                anisotropic_scale=aug_cfg.anisotropic_scale,
                stroke_shift=aug_cfg.stroke_shift,
            )
            tm = None if (ex["times"] is None or rng.random() < aug_cfg.timing_dropout) else ex["times"]
            chunked = chunk_ink_v2(st, tm, config=ink_cfg, post_resample_jitter=aug_cfg.post_resample_jitter, rng=rng)
            v = int(chunked.valid_chunks)
            if v <= 0:
                continue
            raster = render_strokes_to_strip_v2(
                st,
                width_buckets=WIDTH_BUCKETS_V2,
                stroke_width=float(rng.uniform(1.9, 3.0)),
            )
            img_np = np.asarray(raster.image, dtype=np.uint8).transpose(2, 0, 1)
            used_w = int(raster.image_width)
            img_np = _augment_raster_photometric(img_np, used_w, rng, aug_cfg, allow_band_mask=False)

            body = ex["body"]
            items.append(
                {
                    "id": ex["id"],
                    "source": "personal",
                    "writer_id": "personal",
                    "normalized_latex": ex["latex"],
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
            )
        if not items:
            continue

        batch = collate_v2_batch(items, pad_id=tokenizer.pad_id, static_buckets=True)
        cp = batch["chunk_points"].to(device, non_blocking=True)
        cpm = batch["chunk_point_mask"].to(device, non_blocking=True)
        cm = batch["chunk_meta"].to(device, non_blocking=True)
        ck = batch["chunk_mask"].to(device, non_blocking=True)
        img = batch["images"].to(device, non_blocking=True)
        iw = batch["image_widths"].to(device, non_blocking=True)
        l2r = batch["l2r_tokens"].to(device, non_blocking=True)
        r2l = batch["r2l_tokens"].to(device, non_blocking=True)
        stroke_only = rng.random() < stroke_only_prob
        ws = ws_fixed.expand(cp.shape[0], -1) if ws_fixed is not None else None

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
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
            l2r_loss = F.cross_entropy(
                out["logits"].reshape(-1, model.config.vocab_size),
                l2r[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id,
                label_smoothing=label_smoothing,
            )
            r2l_loss = F.cross_entropy(
                out["r2l_logits"].reshape(-1, model.config.vocab_size),
                r2l[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id,
                label_smoothing=label_smoothing,
            )
            loss = l2r_loss + r2l_weight * r2l_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

        if verbose and (step == 1 or step % max(1, eff_steps // 4) == 0 or step == eff_steps):
            print(
                f"  [LoRA N={n_samples} step {step:3d}/{eff_steps}] "
                f"loss={np.mean(losses[-10:]):.4f} lr={scheduler.get_last_lr()[0]:.2e} ({time.perf_counter()-t0:.1f}s)"
            )

    model.eval()
    return {
        "n_samples": n_samples,
        "steps": eff_steps,
        "batch_size": eff_bs,
        "final_loss": float(np.mean(losses[-10:])) if losses else 0.0,
        "train_seconds": round(time.perf_counter() - t0, 2),
        "trainable_params": sum(p.numel() for p in lora_params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a personal `.inktexlora` adapter from native or iPad corrections JSONL")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to InkTeX v2 release .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to native examples.jsonl or iPad corrections.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output `.inktexlora` path")
    parser.add_argument("--adapter-name", type=str, default="Personal Handwriting LoRA")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--use-calibration-style", action="store_true", help="Condition LoRA training on Tier-1 writer style from calibration strokes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = InkTeXV2Config.from_dict(payload.get("model_config") or payload["config"])
    cfg.vision_pretrained = False
    model = InkTeXV2Model(cfg)
    state = payload.get("ema_state_dict") or payload.get("model_state_dict") or payload["model"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    tokenizer = LatexTokenizerV2(payload["tokenizer"]["tokens"])

    examples, cal_strokes = load_personal_jsonl(args.data)
    print(f"[data] Loaded {len(examples)} personal examples (+ {len(cal_strokes)} calibration strokes) from {args.data}")

    ws = None
    if args.use_calibration_style:
        src_strokes = cal_strokes if cal_strokes else [s for ex in examples[:5] for s in ex["strokes"]][:STYLE_MAX_STROKES]
        sp, sa, sm = extract_style_strokes(src_strokes, max_strokes=STYLE_MAX_STROKES, points_per_stroke=STYLE_POINTS)
        with torch.inference_mode():
            ws = model.style_encoder(
                torch.from_numpy(sp)[None].to(device),
                torch.from_numpy(sa)[None].to(device),
                torch.from_numpy(sm)[None].to(device),
            )

    stats = train_personal_lora(
        model,
        tokenizer,
        examples,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        writer_style=ws,
        seed=args.seed,
    )
    export_lora_adapter(
        model,
        args.output,
        adapter_name=args.adapter_name,
        dtype="fp16",
        training_samples=stats["n_samples"],
        notes=args.notes or f"Trained in {stats['train_seconds']}s ({stats['steps']} steps, r={cfg.lora_rank})",
    )
    print(f"[done] Exported {args.output} ({args.output.stat().st_size / 1024:.1f} KB) | stats={stats}")


if __name__ == "__main__":
    main()
