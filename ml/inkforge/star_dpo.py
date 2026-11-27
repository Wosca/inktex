"""Stage 2 InkTeX-Guided STaR + DPO Post-Training for InkForge.

Eliminates autoregressive exposure bias and boosts hard-structure accuracy (`\\sqrt`, `\\sum`, `\\lim`,
`\\Pr`, long trig/calculus expressions) by using Oscar's frozen InkTeX v2 recognizer as an automated
round-trip reward judge:

  1. Rollout phase (InkForge on GPU): generate K candidates per prompt across UniMER + curated hard prompts.
  2. Judge phase (InkTeX v2 on GPU, InkForge parked on CPU -> zero VRAM squeeze on 12 GB RTX 4070 Ti):
     score every geometry-gated rollout with InkTeX v2 bimodal recognition (`exact match` under Standardizer v2).
  3. STaR + DPO phase (InkForge + frozen ref_model on GPU):
     - DPO loss on `(winner, loser)` rollout pairs from the same prompt.
     - STaR SFT anchor loss on verified winners.
     - Real human replay loss from `inkforge_data_v2` so handwriting naturalness & writer styles never drift.
"""
from __future__ import annotations

import argparse
import copy
from functools import partial
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.composer import (  # noqa: E402
    DEFAULT_UNIMER_JSONL,
    passes_geometry_gate,
    smooth_strokes_catmull_rom,
    synthesize_realistic_times,
)
from inkforge.dataset import (  # noqa: E402
    DEFAULT_V2_CACHE_DIR,
    InkForgeDataset,
    collate_inkforge_batch,
    normalize_and_resample_for_inkforge,
)
from inkforge.model import InkForgeConfig, InkForgeModel  # noqa: E402
from inkforge.tokenizer import InkForgeTokenizer, tokenize_inkforge  # noqa: E402
from inkforge.train_inkforge import (  # noqa: E402
    PREVIEW_PROMPTS,
    render_preview_sheets,
    save_checkpoint,
    update_ema,
)
from tokenizer.latex_tokenizer import tokenize_latex  # noqa: E402
from v2.eval_v2 import (  # noqa: E402
    _featurize_eval_batch,
    _prepare_row_vision,
    _safe_std_v2,
    load_v2_checkpoint,
)
from v2.vision_raster import WIDTH_BUCKETS_V2, select_width_bucket  # noqa: E402

DEFAULT_INKTEX_V2_CKPT = Path(r"E:\math-recogniser-training\inktex_v2\release_v2\inktex_v2_release.pt")


def build_hard_prompt_pool(
    tokenizer: InkForgeTokenizer,
    *,
    unimer_path: Path = DEFAULT_UNIMER_JSONL,
    max_unimer: int = 25000,
    seed: int = 42,
) -> list[str]:
    """Build a diverse LaTeX prompt pool heavy on radicals, sums, limits, fractions, trig, and STEM formulas."""
    rng = random.Random(seed)
    prompts: list[str] = []

    # 1. Curated programmatic hard-structure prompts (heavy on v1 failure modes)
    vars_a = ["x", "y", "z", "t", "n", "k", "a", "b", "c", "r", "u", "v"]
    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    greeks = [r"\alpha", r"\beta", r"\gamma", r"\theta", r"\lambda", r"\mu", r"\sigma", r"\omega"]
    for _ in range(4000):
        v1, v2, v3 = rng.sample(vars_a, 3)
        d1, d2 = rng.choice(digits), rng.choice(digits)
        op = rng.choice(["+", "-"])
        g1, g2 = rng.sample(greeks, 2)
        c = rng.randrange(10)
        if c == 0:
            p = f"\\sqrt{{{v1}^2 {op} {d1}{v2}{v3}}}"
        elif c == 1:
            p = f"\\sqrt{{{v1}^2 {op} {v2}^2}}"
        elif c == 2:
            p = f"\\sum_{{i=1}}^{v1} {v2}_i = \\frac{{{v1}({v1}+1)}}{{{d2}}}"
        elif c == 3:
            p = f"\\lim_{{{v1} \\rightarrow 0}} \\frac{{\\sin {v1}}}{{{v1}}} = 1"
        elif c == 4:
            p = f"\\sin^2 {g1} + \\cos^2 {g1} = 1"
        elif c == 5:
            p = f"\\Pr(A \\cap B) = \\Pr(A)\\Pr(B)"
        elif c == 6:
            p = f"\\forall {v1} \\in \\mathbb{{R}}, {v1}^2 \\geq 0"
        elif c == 7:
            p = f"\\int_0^\\infty e^{{-{v1}}} d{v1} = 1"
        elif c == 8:
            p = f"\\frac{{\\partial f}}{{\\partial {v1}}} + \\frac{{\\partial f}}{{\\partial {v2}}}"
        else:
            p = f"\\Delta {v1} = {g1} {g2}"
        if not tokenizer.has_unk(p) and len(tokenize_inkforge(p)) <= 48:
            prompts.append(p)

    # 2. Real-world STEM single-line formulas from UniMER
    if unimer_path.is_file():
        unimer_cands: list[str] = []
        with unimer_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(unimer_cands) >= max_unimer * 3:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                tex = (rec.get("latex") or rec.get("text") or "").strip()
                if not tex or len(tex) > 95 or r"\begin{" in tex or r"\\" in tex:
                    continue
                if tokenizer.has_unk(tex):
                    continue
                toks = tokenize_inkforge(tex)
                if 3 <= len(toks) <= 44:
                    unimer_cands.append(tex)
        rng.shuffle(unimer_cands)
        prompts.extend(unimer_cands[:max_unimer])

    # Deduplicate while preserving variety
    seen: set[str] = set()
    uniq: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    rng.shuffle(uniq)
    return uniq


def strokes_to_inkforge_item(
    strokes: Sequence[np.ndarray],
    token_ids: Sequence[int],
    style_z: np.ndarray,
    cfg: InkForgeConfig,
) -> dict[str, Any] | None:
    """Convert generated strokes into an autoregressive InkForge training item (inputs + targets)."""
    norm = normalize_and_resample_for_inkforge(
        strokes, spacing=cfg.resample_spacing, max_points=cfg.max_stroke_points, coord_clip=cfg.coord_clip
    )
    if norm is None:
        return None
    xy, pen = norm
    n_steps = len(xy)
    if n_steps < 2:
        return None
    deltas = np.diff(xy, axis=0)
    target_dx = np.zeros((n_steps,), dtype=np.float32)
    target_dy = np.zeros((n_steps,), dtype=np.float32)
    target_pen = np.full((n_steps,), 2, dtype=np.int64)
    target_dx[:-1] = np.clip(deltas[:, 0], -cfg.coord_clip, cfg.coord_clip)
    target_dy[:-1] = np.clip(deltas[:, 1], -cfg.coord_clip, cfg.coord_clip)
    target_pen[:-1] = pen[1:]

    stroke_in = np.zeros((n_steps, 7), dtype=np.float32)
    stroke_in[0, 6] = 1.0
    if n_steps > 1:
        stroke_in[1:, 0] = target_dx[:-1]
        stroke_in[1:, 1] = target_dy[:-1]
        stroke_in[1:, 2] = np.clip(xy[1:, 0], -cfg.abs_x_clip, cfg.abs_x_clip)
        stroke_in[1:, 3] = np.clip(xy[1:, 1], -cfg.abs_y_clip, cfg.abs_y_clip)
        stroke_in[1:, 4] = (target_pen[:-1] == 0).astype(np.float32)
        stroke_in[1:, 5] = (target_pen[:-1] == 1).astype(np.float32)

    return {
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "stroke_inputs": stroke_in,
        "target_dx": target_dx,
        "target_dy": target_dy,
        "target_pen": target_pen,
        "style_z": np.asarray(style_z, dtype=np.float32),
    }


def collate_rollout_items(items: Sequence[dict[str, Any]], pad_id: int = 0) -> dict[str, Tensor]:
    b = len(items)
    max_l = max(len(it["token_ids"]) for it in items)
    max_t = max(len(it["target_pen"]) for it in items)
    token_ids = torch.full((b, max_l), pad_id, dtype=torch.long)
    stroke_inputs = torch.zeros((b, max_t, 7), dtype=torch.float32)
    target_dx = torch.zeros((b, max_t), dtype=torch.float32)
    target_dy = torch.zeros((b, max_t), dtype=torch.float32)
    target_pen = torch.full((b, max_t), 2, dtype=torch.long)
    stroke_mask = torch.zeros((b, max_t), dtype=torch.bool)
    style_z = torch.from_numpy(np.stack([it["style_z"] for it in items], axis=0))
    for i, it in enumerate(items):
        l = len(it["token_ids"])
        t = len(it["target_pen"])
        token_ids[i, :l] = torch.from_numpy(it["token_ids"])
        stroke_inputs[i, :t] = torch.from_numpy(it["stroke_inputs"])
        target_dx[i, :t] = torch.from_numpy(it["target_dx"])
        target_dy[i, :t] = torch.from_numpy(it["target_dy"])
        target_pen[i, :t] = torch.from_numpy(it["target_pen"])
        stroke_mask[i, :t] = True
    return {
        "token_ids": token_ids,
        "stroke_inputs": stroke_inputs,
        "target_dx": target_dx,
        "target_dy": target_dy,
        "target_pen": target_pen,
        "stroke_mask": stroke_mask,
        "style_z": style_z,
    }


@torch.inference_mode()
def judge_rollouts_with_inktex(
    judge_model: torch.nn.Module,
    judge_tok: Any,
    rollout_rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[tuple[bool, str]]:
    """Run InkTeX v2 bimodal greedy recognition on generated stroke rollouts and return `(exact_ok, pred_latex)`."""
    judge_model.eval()
    n = len(rollout_rows)
    out: list[tuple[bool, str]] = [(False, "")] * n
    if n == 0:
        return out

    rng = np.random.default_rng(12345)
    bucket_groups: dict[int, list[int]] = {}
    prepared_rows: list[dict[str, Any] | None] = [None] * n
    for i, r in enumerate(rollout_rows):
        strokes = r["strokes"]
        if not r["finished"] or not passes_geometry_gate(r["latex"], strokes):
            continue
        smoothed = smooth_strokes_catmull_rom(strokes, subdivisions=2)
        times = synthesize_realistic_times(smoothed, rng)
        row_obj = {"id": f"r{i}", "strokes": smoothed, "times": times, "target": r["latex"]}
        try:
            img_t, w = _prepare_row_vision(row_obj)
            row_obj["_cached_image"] = img_t
            row_obj["_cached_width"] = w
            prepared_rows[i] = row_obj
            bk = select_width_bucket(w, WIDTH_BUCKETS_V2)
            bucket_groups.setdefault(bk, []).append(i)
        except Exception:
            continue

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for bk in sorted(bucket_groups.keys()):
        idxs = bucket_groups[bk]
        for start in range(0, len(idxs), batch_size):
            sub_idxs = idxs[start : start + batch_size]
            sub_rows = [prepared_rows[i] for i in sub_idxs]
            batch = _featurize_eval_batch(sub_rows, use_vision=True, use_strokes=True, pad_id=judge_tok.pad_id)
            cp = batch["chunk_points"].to(device, non_blocking=True)
            cpm = batch["chunk_point_mask"].to(device, non_blocking=True)
            cm = batch["chunk_meta"].to(device, non_blocking=True)
            ck = batch["chunk_mask"].to(device, non_blocking=True)
            imgs = batch["images"].to(device, non_blocking=True)
            img_w = batch["image_widths"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                mem, mem_mask, _ = judge_model.encode(
                    images=imgs,
                    image_widths=img_w,
                    chunk_points=cp,
                    chunk_point_mask=cpm,
                    chunk_meta=cm,
                    chunk_mask=ck,
                    modality_mode="auto",
                )
                pred_ids_list = judge_model.decoder.decode_greedy_cached(
                    mem,
                    mem_mask,
                    bos_id=judge_tok.bos_l2r_id,
                    eos_id=judge_tok.eos_id,
                    pad_id=judge_tok.pad_id,
                    max_tokens=96,
                ).cpu().tolist()
            for idx, pred_ids in zip(sub_idxs, pred_ids_list):
                pred_tex = judge_tok.decode(pred_ids)
                t_toks = tokenize_latex(_safe_std_v2(rollout_rows[idx]["latex"]))
                p_toks = tokenize_latex(_safe_std_v2(pred_tex))
                out[idx] = (bool(t_toks) and p_toks == t_toks, pred_tex)
    return out


def run_star_dpo(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load InkForge checkpoint
    ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    cfg = InkForgeConfig.from_dict(ckpt["config"])
    tokenizer = InkForgeTokenizer.from_dict(ckpt["tokenizer"])
    model = InkForgeModel(cfg, grad_checkpoint=args.grad_checkpoint)
    state_key = "ema_state_dict" if "ema_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])

    ema_model = copy.deepcopy(model)
    ema_model.grad_checkpoint = False
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    ref_model = copy.deepcopy(model)
    ref_model.grad_checkpoint = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(
        f"[star-dpo] Loaded InkForge init={args.init_ckpt} ({model.parameter_count()/1e6:.2f}M params, "
        f"sigma_min={cfg.sigma_min})",
        flush=True,
    )

    # Load writer style bank if available
    style_bank_path = args.init_ckpt.parent / "writer_styles.npz"
    if style_bank_path.is_file():
        z_bank = np.load(style_bank_path)["mu_bank"].astype(np.float32)
        print(f"[star-dpo] Loaded {len(z_bank)} writer styles from {style_bank_path}", flush=True)
    else:
        z_bank = np.zeros((1, cfg.style_dim), dtype=np.float32)

    # 2. Load frozen InkTeX v2 judge on CPU (swapped to GPU strictly during judging phase!)
    judge_model, judge_tok, _ = load_v2_checkpoint(
        args.judge_ckpt, None, torch.device("cpu"), use_ema=True
    )
    judge_model.eval()
    for p in judge_model.parameters():
        p.requires_grad_(False)
    print(f"[star-dpo] Loaded frozen InkTeX v2 judge from {args.judge_ckpt}", flush=True)

    # 3. Replay dataset (strictly real human samples so handwriting naturalness stays anchored!)
    replay_ds = InkForgeDataset(args.cache_dir, split="train", cfg=cfg, augment=True, input_noise_std=0.010)
    replay_loader = DataLoader(
        replay_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_inkforge_batch, pad_id=cfg.pad_id),
    )
    replay_iter = iter(replay_loader)

    # 4. Build hard prompt pool
    prompt_pool = build_hard_prompt_pool(tokenizer, seed=args.seed)
    print(f"[star-dpo] Built prompt pool: {len(prompt_pool):,} prompts", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )
    log_path = out_dir / "star_dpo_metrics.jsonl"
    rng = random.Random(args.seed)
    global_step = 0

    for rd in range(1, args.rounds + 1):
        t_rd = time.perf_counter()
        rd_prompts = rng.sample(prompt_pool, min(args.prompts_per_round, len(prompt_pool)))

        # ---- Phase 1: Batched Rollout Generation (ema_model on GPU) ----
        ema_model.to(device)
        ema_model.eval()
        rollout_rows: list[dict[str, Any]] = []
        encoded_prompts = [tokenizer.encode(p, max_len=cfg.max_text_len) for p in rd_prompts]
        k_roll = args.rollouts_per_prompt

        for start in range(0, len(rd_prompts), args.gen_batch_size):
            sub_p = rd_prompts[start : start + args.gen_batch_size]
            sub_enc = encoded_prompts[start : start + args.gen_batch_size]
            # Expand each prompt K times
            exp_p = [p for p in sub_p for _ in range(k_roll)]
            exp_enc = [e for e in sub_enc for _ in range(k_roll)]
            max_l = max(len(e) for e in exp_enc)
            tok_t = torch.full((len(exp_enc), max_l), tokenizer.pad_id, dtype=torch.long, device=device)
            for i, e in enumerate(exp_enc):
                tok_t[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)

            # Mix 50% neutral z=0 and 50% real human writer styles
            z_np = np.zeros((len(exp_enc), cfg.style_dim), dtype=np.float32)
            for i in range(len(exp_enc)):
                if i % 2 == 1 and len(z_bank) > 0:
                    z_np[i] = z_bank[rng.randrange(len(z_bank))]
            z_t = torch.from_numpy(z_np).to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                strokes_list, meta_list = ema_model.generate_batch(
                    tok_t,
                    style_z=z_t,
                    temperature=args.rollout_temp,
                    pen_temperature=1.0,
                    cfg_scale=args.cfg_scale,
                    max_steps=512,
                    return_meta=True,
                )
            for i in range(len(exp_enc)):
                rollout_rows.append({
                    "prompt_idx": start + (i // k_roll),
                    "latex": exp_p[i],
                    "token_ids": exp_enc[i],
                    "style_z": z_np[i],
                    "strokes": strokes_list[i],
                    "finished": meta_list[i]["finished"],
                })

        # Park ema_model on CPU before moving InkTeX v2 judge onto GPU
        ema_model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ---- Phase 2: InkTeX v2 Automated Reward Judging ----
        judge_model.to(device)
        verdicts = judge_rollouts_with_inktex(
            judge_model, judge_tok, rollout_rows, device, batch_size=args.judge_batch_size
        )
        judge_model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Group winners & losers by prompt
        by_prompt: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
        total_pass = 0
        for row, (ok, _pred) in zip(rollout_rows, verdicts):
            w_list, l_list = by_prompt.setdefault(row["prompt_idx"], ([], []))
            if ok:
                total_pass += 1
                w_list.append(row)
            else:
                l_list.append(row)

        star_winners: list[dict[str, Any]] = []
        dpo_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        prompts_with_win = 0
        for _pidx, (ws, ls) in by_prompt.items():
            if not ws:
                continue
            prompts_with_win += 1
            for w in ws[:2]:
                item_w = strokes_to_inkforge_item(w["strokes"], w["token_ids"], w["style_z"], cfg)
                if item_w is not None:
                    star_winners.append(item_w)
                    if ls:
                        l_cand = rng.choice(ls)
                        item_l = strokes_to_inkforge_item(
                            l_cand["strokes"], l_cand["token_ids"], l_cand["style_z"], cfg
                        )
                        if item_l is not None:
                            dpo_pairs.append((item_w, item_l))

        pass_rate = total_pass / max(1, len(rollout_rows))
        prompt_solve_rate = prompts_with_win / max(1, len(rd_prompts))
        print(
            f"[star-dpo round {rd}/{args.rounds}] rollout_pass={pass_rate*100:.1f}% "
            f"({total_pass}/{len(rollout_rows)}) | prompt_pass@{k_roll}={prompt_solve_rate*100:.1f}% "
            f"| winners={len(star_winners)} | dpo_pairs={len(dpo_pairs)} "
            f"({time.perf_counter()-t_rd:.1f}s)",
            flush=True,
        )
        if not dpo_pairs and not star_winners:
            print(f"[star-dpo round {rd}] no verified rollouts found, skipping update.", flush=True)
            continue

        # ---- Phase 3: STaR + DPO + Human Replay Updates ----
        model.to(device)
        ref_model.to(device)
        ema_model.to(device)
        model.train()
        ref_model.eval()

        rng.shuffle(dpo_pairs)
        rng.shuffle(star_winners)
        dpo_loss_sum = 0.0
        sft_loss_sum = 0.0
        replay_loss_sum = 0.0
        n_updates = 0

        bs = args.train_batch_size
        n_batches = max(len(dpo_pairs) // max(1, bs), len(star_winners) // max(1, bs), 1)
        for bi in range(min(n_batches, args.max_updates_per_round)):
            optimizer.zero_grad(set_to_none=True)
            total_step_loss = torch.zeros((), device=device, dtype=torch.float32)

            # 3a. DPO preference loss
            if dpo_pairs:
                sub_pairs = [dpo_pairs[(bi * bs + j) % len(dpo_pairs)] for j in range(min(bs, len(dpo_pairs)))]
                bw = {k: v.to(device, non_blocking=True) for k, v in collate_rollout_items([p[0] for p in sub_pairs], cfg.pad_id).items()}
                bl = {k: v.to(device, non_blocking=True) for k, v in collate_rollout_items([p[1] for p in sub_pairs], cfg.pad_id).items()}
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    lp_w = model.sequence_log_prob(**bw)
                    lp_l = model.sequence_log_prob(**bl)
                    with torch.no_grad():
                        ref_w = ref_model.sequence_log_prob(**bw)
                        ref_l = ref_model.sequence_log_prob(**bl)
                logits_dpo = args.dpo_beta * ((lp_w - ref_w) - (lp_l - ref_l))
                dpo_loss = -F.logsigmoid(logits_dpo).mean()
                sft_loss = -lp_w.mean()
                total_step_loss = total_step_loss + dpo_loss + args.sft_weight * sft_loss
                dpo_loss_sum += float(dpo_loss.item())
                sft_loss_sum += float(sft_loss.item())

            # 3b. Real human replay anchor (prevents style/naturalness drift)
            try:
                rep_batch = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                rep_batch = next(replay_iter)
            rep_batch = {k: v.to(device, non_blocking=True) for k, v in rep_batch.items()}
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                rep_out = model.compute_loss(
                    token_ids=rep_batch["token_ids"],
                    stroke_inputs=rep_batch["stroke_inputs"],
                    target_dx=rep_batch["target_dx"],
                    target_dy=rep_batch["target_dy"],
                    target_pen=rep_batch["target_pen"],
                    stroke_mask=rep_batch["stroke_mask"],
                    style_points=rep_batch["style_points"],
                    style_aspect=rep_batch["style_aspect"],
                    style_mask=rep_batch["style_mask"],
                    text_drop_prob=0.10,
                )
                rep_loss = rep_out["loss"]
            total_step_loss = total_step_loss + args.replay_weight * rep_loss
            replay_loss_sum += float(rep_loss.item())

            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            update_ema(ema_model, model, 0.995)
            n_updates += 1

        # Save checkpoint + render preview sheet after every round
        save_checkpoint(
            out_dir / "best.pt",
            step=global_step,
            cfg=cfg,
            tokenizer=tokenizer,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            best_val=-pass_rate,
        )
        render_preview_sheets(
            ema_model, tokenizer, replay_ds, out_dir, global_step, device, cfg_scale=args.cfg_scale
        )
        rec = {
            "round": rd,
            "global_step": global_step,
            "rollout_pass_rate": round(pass_rate, 4),
            "prompt_pass_at_k": round(prompt_solve_rate, 4),
            "winners": len(star_winners),
            "dpo_pairs": len(dpo_pairs),
            "dpo_loss": round(dpo_loss_sum / max(1, n_updates), 4),
            "sft_loss": round(sft_loss_sum / max(1, n_updates), 4),
            "replay_loss": round(replay_loss_sum / max(1, n_updates), 4),
            "elapsed_s": round(time.perf_counter() - t_rd, 1),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[star-dpo round {rd} done] {json.dumps(rec)}", flush=True)

        # Free GPU before next round
        model.to("cpu")
        ref_model.to("cpu")
        ema_model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[star-dpo] Finished {args.rounds} rounds -> {out_dir / 'best.pt'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="InkTeX-guided STaR + DPO post-training for InkForge")
    parser.add_argument("--init-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_V2_CACHE_DIR)
    parser.add_argument("--judge-ckpt", type=Path, default=DEFAULT_INKTEX_V2_CKPT)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--prompts-per-round", type=int, default=1200)
    parser.add_argument("--rollouts-per-prompt", type=int, default=6)
    parser.add_argument("--gen-batch-size", type=int, default=12)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=12)
    parser.add_argument("--max-updates-per-round", type=int, default=200)
    parser.add_argument("--rollout-temp", type=float, default=0.45)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--dpo-beta", type=float, default=0.15)
    parser.add_argument("--sft-weight", type=float, default=0.25)
    parser.add_argument("--replay-weight", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_star_dpo(args)


if __name__ == "__main__":
    main()
