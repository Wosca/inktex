"""Resumable overnight synthetic human-like math handwriting generator for RTX 4070 Ti.

Generates a balanced real-world STEM mix:
  - ~55% complex single-line STEM formulas mined from UniMER (300k real formulas, long expressions, rare symbols)
  - ~20% multi-line derivations (`\\` and `\\begin{aligned}`) with consistent single-writer style across lines
  - ~15% 2x2..3x3 matrices (`bmatrix`, `pmatrix`, `vmatrix`, `matrix`) with hand-drawn brackets
  - ~10% piecewise functions (`\\begin{cases}`) with real handwritten `\\text{if ...}` / `\\text{otherwise}`

Writes resumable `stroke-latex-v1` JSONL shards to `E:\\math-recogniser-training\\inkforge_synth\\`
plus crisp PNG contact sheets every 1,000 samples so you can visually inspect quality!
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import numpy as np
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.composer import (  # noqa: E402
    DEFAULT_UNIMER_JSONL,
    StructuredPrompt,
    assemble_structured_strokes,
    load_unimer_prompt_pool,
    passes_geometry_gate,
    sample_programmatic_structured_prompt,
    smooth_strokes_catmull_rom,
    synthesize_realistic_times,
)
from inkforge.model import InkForgeConfig, InkForgeModel  # noqa: E402
from inkforge.render import render_contact_sheet  # noqa: E402
from inkforge.star_dpo import (  # noqa: E402
    DEFAULT_INKTEX_V2_CKPT,
    _canonicalize_judge_latex,
    judge_rollouts_with_inktex,
    normalize_cell_for_inkforge,
)
from inkforge.tokenizer import InkForgeTokenizer, tokenize_inkforge  # noqa: E402
from tokenizer.latex_tokenizer import tokenize_latex  # noqa: E402
from v2.eval_v2 import _safe_std_v2, load_v2_checkpoint  # noqa: E402


_SINGLE_CHAR_AMBIG_MAP = {
    "C": "c", "K": "k", "O": "o", "P": "p", "S": "s",
    "V": "v", "W": "w", "X": "x", "Y": "y", "Z": "z",
    r"\Pi": r"\pi",
}


def _is_dangling_prefix_cell(latex: str) -> bool:
    """Short operator-bounded chunk cells like `f(x) =`, `1 +`, `-b \\pm`, `= f(x)`, or `+ 2xy`."""
    s = latex.strip()
    return len(tokenize_inkforge(s)) <= 12 and (
        s.endswith(("=", "+", "-", r"\pm", r"\mp")) or s.startswith(("=", "+", "-", r"\pm", r"\mp"))
    )


def _prefix_tokens_match(target_latex: str, pred_latex: str) -> bool:
    """Verify operator-bounded split chunks (catches `0 =` vs `r =`, allows leading/trailing op quirks)."""
    t_toks = tokenize_latex(_safe_std_v2(_canonicalize_judge_latex(target_latex)))
    p_toks = tokenize_latex(_safe_std_v2(_canonicalize_judge_latex(pred_latex)))
    if not t_toks or not p_toks:
        return False
    if p_toks == t_toks:
        return True
    ops = ("=", "+", "-", r"\pm", r"\mp")
    t_core = list(t_toks)
    p_core = list(p_toks)
    if t_core and t_core[-1] in ops:
        t_core = t_core[:-1]
    if p_core and p_core[-1] in ops:
        p_core = p_core[:-1]
    if t_core and t_core[0] in ops:
        t_core = t_core[1:]
    if p_core and p_core[0] in ops:
        p_core = p_core[1:]
    return bool(t_core) and p_core == t_core


def _single_token_case_match(target_latex: str, pred_latex: str) -> bool:
    """Isolated 1-glyph cells (`c`, `x`, `\\pi`) lack x-height context vs `C`, `X`, `\\Pi`."""
    t_toks = tokenize_latex(_safe_std_v2(_canonicalize_judge_latex(target_latex)))
    p_toks = tokenize_latex(_safe_std_v2(_canonicalize_judge_latex(pred_latex)))
    if len(t_toks) != 1 or len(p_toks) != 1:
        return False
    return _SINGLE_CHAR_AMBIG_MAP.get(t_toks[0], t_toks[0]) == _SINGLE_CHAR_AMBIG_MAP.get(p_toks[0], p_toks[0])


@torch.inference_mode()
def generate_verified_cells(
    model: InkForgeModel,
    tokenizer: InkForgeTokenizer,
    flat_cells: list[str],
    cell_zs_np: np.ndarray,
    cell_temps_np: np.ndarray,
    device: torch.device,
    *,
    cfg_scale: float = 1.5,
    max_chunk_seqs: int = 64,
    judge_model: torch.nn.Module | None = None,
    judge_tok: Any | None = None,
    judge_batch_size: int = 32,
    max_retries: int = 2,
) -> tuple[list[list[np.ndarray]], list[bool]]:
    """Generate flattened cells in length-bucketed OOM-safe chunks with InkTeX v2 verification + selective retry!

    Length-sorting pending cells before chunking cuts KV-cache padding waste 3-5x on RTX 4070 Ti.
    Failed cells are selectively regenerated using the EXACT SAME writer style `z`.
    """
    n = len(flat_cells)
    out_strokes: list[list[np.ndarray]] = [[] for _ in range(n)]
    cell_ok: list[bool] = [False] * n
    pending = list(range(n))
    all_enc = [tokenizer.encode(normalize_cell_for_inkforge(c), max_len=model.cfg.max_text_len) for c in flat_cells]

    for attempt in range(max_retries + 1):
        if not pending:
            break
        # Sort pending cells by token length so short matrix numbers finish in ~25 steps instead of 450!
        pending.sort(key=lambda idx: len(all_enc[idx]))
        temp_scale = 1.0 if attempt == 0 else max(0.75, 1.0 - 0.10 * attempt)
        cand_strokes: list[list[np.ndarray]] = []
        cand_meta: list[dict[str, Any]] = []

        chunk_sz = max_chunk_seqs
        pos_s = 0
        while pos_s < len(pending):
            sub_ids = pending[pos_s : pos_s + chunk_sz]
            enc = [all_enc[i] for i in sub_ids]
            max_l = max(len(e) for e in enc)
            tok_t = torch.full((len(sub_ids), max_l), tokenizer.pad_id, dtype=torch.long, device=device)
            for j, e in enumerate(enc):
                tok_t[j, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)
            z_t = torch.from_numpy(cell_zs_np[sub_ids]).to(device)
            t_t = torch.from_numpy((cell_temps_np[sub_ids] * temp_scale).astype(np.float32)).to(device)
            try:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    st_c, meta_c = model.generate_batch(
                        tok_t,
                        style_z=z_t,
                        temperature=t_t,
                        pen_temperature=1.0,
                        cfg_scale=cfg_scale,
                        max_steps=512,
                        return_meta=True,
                    )
                cand_strokes.extend(st_c)
                cand_meta.extend(meta_c)
                pos_s += len(sub_ids)
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if chunk_sz <= 8:
                    raise
                chunk_sz = max(8, chunk_sz // 2)

        # Geometry gate on every atomic cell first
        geom_pass_pos: list[int] = []
        for pos, idx in enumerate(pending):
            ok_geom = bool(cand_meta[pos]["finished"]) and passes_geometry_gate(
                flat_cells[idx], cand_strokes[pos], is_atomic_cell=True
            )
            if ok_geom:
                if judge_model is None:
                    out_strokes[idx] = cand_strokes[pos]
                    cell_ok[idx] = True
                else:
                    geom_pass_pos.append(pos)

        # Run fast InkTeX v2 bimodal recognition on ALL geometry-valid cells (including dangling prefixes!)
        if judge_model is not None and judge_tok is not None and geom_pass_pos:
            judge_rows = [
                {
                    "latex": flat_cells[pending[pos]],
                    "strokes": cand_strokes[pos],
                    "finished": True,
                }
                for pos in geom_pass_pos
            ]
            verdicts = judge_rollouts_with_inktex(
                judge_model, judge_tok, judge_rows, device, batch_size=judge_batch_size
            )
            for pos, (v_ok, pred_tex, _reason) in zip(geom_pass_pos, verdicts):
                idx = pending[pos]
                if (
                    v_ok
                    or (_is_dangling_prefix_cell(flat_cells[idx]) and _prefix_tokens_match(flat_cells[idx], pred_tex))
                    or _single_token_case_match(flat_cells[idx], pred_tex)
                ):
                    out_strokes[idx] = cand_strokes[pos]
                    cell_ok[idx] = True

        pending = [idx for idx in pending if not cell_ok[idx]]

    return out_strokes, cell_ok


def sample_writer_latents(
    n: int,
    style_dim: int,
    rng: np.random.Generator,
    bank: np.ndarray | None = None,
) -> np.ndarray:
    """Sample realistic human writer style vectors by interpolating real bank writers + prior noise."""
    if bank is None or len(bank) < 2:
        return rng.normal(0.0, 0.9, size=(n, style_dim)).astype(np.float32)

    idx_a = rng.integers(0, len(bank), size=n)
    idx_b = rng.integers(0, len(bank), size=n)
    alpha = rng.uniform(0.55, 1.0, size=(n, 1)).astype(np.float32)
    z = alpha * bank[idx_a] + (1.0 - alpha) * bank[idx_b]
    z += rng.normal(0.0, 0.12, size=(n, style_dim)).astype(np.float32)
    return z.astype(np.float32)


def strokes_to_v1_schema(strokes: list[np.ndarray], times: list[np.ndarray]) -> list[dict[str, Any]]:
    out = []
    for idx, (st, tm) in enumerate(zip(strokes, times)):
        pts = [
            {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "time": round(float(t), 2)}
            for p, t in zip(st, tm)
        ]
        if pts:
            out.append({"id": str(idx), "points": pts})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight InkForge synthetic human math handwriting generator")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained InkForge best.pt / latest.pt")
    parser.add_argument("--style-bank", type=Path, default=None, help="Optional writer_styles.npz from training")
    parser.add_argument("--unimer-jsonl", type=Path, default=DEFAULT_UNIMER_JSONL)
    parser.add_argument("--out-dir", type=Path, default=Path(r"E:\math-recogniser-training\inkforge_synth"))
    parser.add_argument("--target-count", type=int, default=200000, help="Total valid samples to generate")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=96, help="GPU cell batch size (96-128 fits RTX 4070 Ti easily)")
    parser.add_argument("--structured-prob", type=float, default=0.45, help="Fraction of multi-line/matrix/cases prompts")
    parser.add_argument("--temp-min", type=float, default=0.28)
    parser.add_argument("--temp-max", type=float, default=0.52)
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="Text CFG scale (1.5 for InkForge v2)")
    parser.add_argument("--judge-ckpt", type=Path, default=DEFAULT_INKTEX_V2_CKPT, help="InkTeX v2 judge for cell verification")
    parser.add_argument("--no-verify-cells", action="store_true", help="Disable InkTeX v2 atomic-cell verification")
    parser.add_argument("--max-cell-retries", type=int, default=2, help="Selective retries per failed atomic cell (same writer z)")
    parser.add_argument("--preview-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = InkForgeConfig.from_dict(ckpt["config"])
    tokenizer = InkForgeTokenizer.from_dict(ckpt["tokenizer"])
    model = InkForgeModel(cfg).to(device)
    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[gen] Loaded InkForge ({model.parameter_count()/1e6:.2f}M params, step={ckpt.get('step')}) on {device}")

    judge_model = None
    judge_tok = None
    if not args.no_verify_cells and args.judge_ckpt.is_file():
        judge_model, judge_tok, _ = load_v2_checkpoint(args.judge_ckpt, None, device, use_ema=True)
        judge_model.eval()
        for p in judge_model.parameters():
            p.requires_grad_(False)
        print(f"[gen] Loaded InkTeX v2 atomic-cell verifier (retries={args.max_cell_retries}) from {args.judge_ckpt}")

    style_bank = None
    bank_path = args.style_bank or (args.checkpoint.parent / "writer_styles.npz")
    if not bank_path.is_file():
        fallback_bank = Path(r"E:\math-recogniser-training\inkforge_v2_model\writer_styles.npz")
        if fallback_bank.is_file():
            bank_path = fallback_bank
    if bank_path.is_file():
        style_bank = np.load(bank_path)["mu_bank"]
        print(f"[gen] Loaded {len(style_bank):,} real human writer styles from {bank_path}")

    unimer_prompts = load_unimer_prompt_pool(tokenizer, args.unimer_jsonl, max_prompts=120000, seed=args.seed)
    print(f"[gen] Loaded {len(unimer_prompts):,} UniMER STEM prompts")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Resume cleanly if shards already exist
    existing_shards = sorted(out_dir.glob("inkforge_synth_*.jsonl"))
    generated = 0
    for sh in existing_shards:
        with sh.open("rb") as f:
            generated += sum(1 for _ in f)
    shard_idx = len(existing_shards)
    if generated > 0:
        print(f"[gen] Resuming from {generated:,} existing samples across {shard_idx} shard(s)")

    py_rng = random.Random(args.seed + generated)
    np_rng = np.random.default_rng(args.seed + generated)
    unimer_cursor = generated % max(1, len(unimer_prompts))

    t0 = time.perf_counter()
    start_generated = generated
    rejected = 0
    in_shard = 0
    preview_buf: list[tuple[str, list[np.ndarray]]] = []

    def open_shard(idx: int):
        return (out_dir / f"inkforge_synth_{idx:05d}.jsonl").open("a", encoding="utf-8")

    shard_f = open_shard(shard_idx)
    try:
        while generated < args.target_count:
            # Build a batch of prompts whose flattened cell count is ~args.batch_size
            prompts_batch: list[StructuredPrompt] = []
            flat_cells: list[str] = []
            cell_owner: list[int] = []
            while len(flat_cells) < args.batch_size:
                if unimer_prompts and py_rng.random() >= args.structured_prob:
                    pr = unimer_prompts[unimer_cursor % len(unimer_prompts)]
                    unimer_cursor += 1
                else:
                    pr = sample_programmatic_structured_prompt(py_rng)
                p_idx = len(prompts_batch)
                prompts_batch.append(pr)
                for c in pr.cells:
                    flat_cells.append(c)
                    cell_owner.append(p_idx)
                if len(prompts_batch) >= args.batch_size:
                    break

            # Assign ONE consistent writer latent `z` and temperature `tau` per equation (shared by all its cells!)
            n_prompts = len(prompts_batch)
            prompt_zs = sample_writer_latents(n_prompts, cfg.style_dim, np_rng, bank=style_bank)
            prompt_temps = np_rng.uniform(args.temp_min, args.temp_max, size=(n_prompts,)).astype(np.float32)

            cell_zs_np = prompt_zs[np.asarray(cell_owner)]
            cell_temps_np = prompt_temps[np.asarray(cell_owner)]
            cell_strokes_flat, cell_ok_flat = generate_verified_cells(
                model,
                tokenizer,
                flat_cells,
                cell_zs_np,
                cell_temps_np,
                device,
                cfg_scale=args.cfg_scale,
                max_chunk_seqs=64,
                judge_model=judge_model,
                judge_tok=judge_tok,
                max_retries=args.max_cell_retries,
            )

            # Group generated cells back per prompt (every atomic cell must pass!)
            grouped: list[list[list[np.ndarray]]] = [[] for _ in range(n_prompts)]
            grouped_ok: list[bool] = [True] * n_prompts
            for c_st, c_ok, owner in zip(cell_strokes_flat, cell_ok_flat, cell_owner):
                grouped[owner].append(c_st)
                if not c_ok:
                    grouped_ok[owner] = False

            for p_idx, pr in enumerate(prompts_batch):
                if generated >= args.target_count:
                    break
                if not grouped_ok[p_idx]:
                    rejected += 1
                    continue
                assembled = assemble_structured_strokes(pr, grouped[p_idx], np_rng)
                if not passes_geometry_gate(pr.latex, assembled):
                    rejected += 1
                    continue

                smoothed = smooth_strokes_catmull_rom(assembled, subdivisions=2)
                times = synthesize_realistic_times(smoothed, np_rng)
                rec = {
                    "schemaVersion": "stroke-latex-v1",
                    "id": f"inkforge:{generated:07d}",
                    "source": "synthetic",
                    "sourceDataset": f"inkforge_{pr.kind}",
                    "split": "train",
                    "synthetic": True,
                    "writerId": f"inkforge_w:{generated:07d}",
                    "latex": pr.latex,
                    "latexOriginal": pr.latex,
                    "latexNormalized": pr.latex,
                    "strokes": strokes_to_v1_schema(smoothed, times),
                    "metadata": {
                        "kind": pr.kind,
                        "temperature": round(float(prompt_temps[p_idx]), 3),
                    },
                }
                shard_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                generated += 1
                in_shard += 1

                if len(preview_buf) < 18:
                    preview_buf.append((f"[{pr.kind}] {pr.latex}", smoothed))

                if generated % args.preview_every == 0 and preview_buf:
                    render_contact_sheet(
                        preview_buf,
                        preview_dir / f"samples_{generated:07d}.png",
                        cols=3,
                        tile_w=460,
                        tile_h=175,
                    )
                    preview_buf = []

                if in_shard >= args.shard_size:
                    shard_f.close()
                    shard_idx += 1
                    in_shard = 0
                    shard_f = open_shard(shard_idx)

            done_now = generated - start_generated
            if done_now > 0 and done_now % 500 < n_prompts:
                elapsed = time.perf_counter() - t0
                rate = done_now / max(elapsed, 1e-3)
                eta_min = (args.target_count - generated) / max(rate, 1e-6) / 60.0
                accept_pct = 100.0 * done_now / max(1, done_now + rejected)
                print(
                    f"[gen] {generated:,}/{args.target_count:,} "
                    f"({rate*60:.0f} eq/min | accept={accept_pct:.1f}% | ETA {eta_min:.1f}m)",
                    flush=True,
                )
    finally:
        shard_f.close()

    print(f"[gen] DONE! Generated {generated:,} samples -> {out_dir}")


if __name__ == "__main__":
    main()
