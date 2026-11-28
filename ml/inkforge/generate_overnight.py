"""Resumable overnight synthetic human-like math handwriting generator for RTX 4070 Ti.

Combines stratified real UniMER labels with procedurally composed STEM formulas. Each
equation shares a writer style across its cells and keeps geometry- and recognition-verified
cells while retrying failures. Writes resumable stroke-latex-v1 JSONL shards and contact sheets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.composer import (  # noqa: E402
    DEFAULT_UNIMER_JSONL,
    StructuredPrompt,
    assemble_structured_strokes,
    load_stratified_unimer_pools,
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
    normalize_cell_for_inkforge,
)
from inkforge.tokenizer import InkForgeTokenizer, tokenize_inkforge  # noqa: E402
from inkforge.verify_openrouter import _normalize_otherwise  # noqa: E402
from tokenizer.latex_tokenizer import tokenize_latex  # noqa: E402
from v2.eval_v2 import (  # noqa: E402
    _featurize_eval_batch,
    _prepare_row_vision,
    _safe_std_v2,
    load_v2_checkpoint,
)
from v2.vision_raster import WIDTH_BUCKETS_V2, select_width_bucket  # noqa: E402


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


def _otherwise_match(target_latex: str, pred_latex: str) -> bool:
    word = r"\text{otherwise}"
    return _normalize_otherwise(target_latex.strip()) == word == _normalize_otherwise(pred_latex.strip())


def _decode_with_token_confidence(
    decoder: torch.nn.Module, memory: torch.Tensor, mask: torch.Tensor | None, tokenizer: Any,
) -> tuple[list[list[int]], list[float]]:
    # Capture the cached decoder's actual greedy distributions without a second decode.
    chosen_probabilities: list[torch.Tensor] = []

    def capture_probability(_module: Any, _args: Any, output: torch.Tensor) -> None:
        logits = output[:, -1].float()
        max_logit = logits.amax(dim=-1)
        chosen_probabilities.append((max_logit - logits.logsumexp(dim=-1)).exp())

    hook = decoder.output.register_forward_hook(capture_probability)
    try:
        ids = decoder.decode_greedy_cached(
            memory, mask, bos_id=tokenizer.bos_l2r_id, eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id, max_tokens=96,
        )
    finally:
        hook.remove()
    if not chosen_probabilities:
        return ids.cpu().tolist(), [0.0] * ids.shape[0]
    probs = torch.stack(chosen_probabilities, dim=1)
    content = torch.ones_like(ids, dtype=torch.bool)
    for special in (tokenizer.pad_id, tokenizer.bos_l2r_id, tokenizer.bos_r2l_id,
                    tokenizer.eos_id, tokenizer.unk_id):
        content &= ids.ne(special)
    minimums = probs.masked_fill(~content, 1.0).amin(dim=1)
    minimums = minimums.masked_fill(~content.any(dim=1), 0.0)
    return ids.cpu().tolist(), minimums.cpu().tolist()


@torch.inference_mode()
def judge_rollouts_with_confidence(
    judge_model: torch.nn.Module,
    judge_tok: Any,
    rollout_rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[tuple[bool, str, str, float]]:
    """Run the local InkTeX v2 judge and return match, prediction, reason, and minimum token probability."""
    judge_model.eval()
    verdicts = [(False, "", "raster_fail", 0.0)] * len(rollout_rows)
    buckets: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    rng = np.random.default_rng(12345)
    for idx, row in enumerate(rollout_rows):
        if not row["finished"]:
            verdicts[idx] = (False, "", "hit_cap", 0.0)
            continue
        if not passes_geometry_gate(row["latex"], row["strokes"], is_atomic_cell=True):
            verdicts[idx] = (False, "", "geom_fail", 0.0)
            continue
        strokes = smooth_strokes_catmull_rom(row["strokes"], subdivisions=2)
        features = {"id": f"r{idx}", "strokes": strokes,
                    "times": synthesize_realistic_times(strokes, rng), "target": row["latex"]}
        try:
            img, width = _prepare_row_vision(features)
        except (ValueError, RuntimeError, OSError):
            continue
        features["_cached_image"], features["_cached_width"] = img, width
        bucket = select_width_bucket(width, WIDTH_BUCKETS_V2)
        buckets.setdefault(bucket, []).append((idx, features))

    for bucket in sorted(buckets):
        rows = buckets[bucket]
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = _featurize_eval_batch(
                [row for _, row in batch_rows], use_vision=True, use_strokes=True, pad_id=judge_tok.pad_id,
            )
            inputs = {key: batch[key].to(device, non_blocking=True) for key in (
                "images", "image_widths", "chunk_points", "chunk_point_mask", "chunk_meta", "chunk_mask",
            )}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                mem, mem_mask, _ = judge_model.encode(**inputs, modality_mode="auto")
                predictions, confidences = _decode_with_token_confidence(judge_model.decoder, mem, mem_mask, judge_tok)
            for (idx, row), ids, min_prob in zip(batch_rows, predictions, confidences):
                pred = judge_tok.decode(ids)
                if judge_tok.eos_id not in ids:
                    verdicts[idx] = (False, pred, "decode_cap", min_prob)
                    continue
                target_tokens = tokenize_latex(_safe_std_v2(_canonicalize_judge_latex(row["target"])))
                pred_tokens = tokenize_latex(_safe_std_v2(_canonicalize_judge_latex(pred)))
                matches = bool(target_tokens) and pred_tokens == target_tokens
                verdicts[idx] = (matches, pred, "ok" if matches else "wrong_latex", min_prob)
    return verdicts


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
    max_retries: int = 3,
    cell_owner: Sequence[int] | None = None,
    min_judge_prob: float = 0.60,
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

    if cell_owner is not None and len(cell_owner) != n:
        raise ValueError("cell_owner must contain one equation index per cell")
    owner_counts = Counter(cell_owner) if cell_owner is not None else Counter()
    sampler_options = {"pen_temperature": 1.0}
    if "jump_temp_scale" in inspect.signature(model.generate_batch).parameters:
        sampler_options = {"pen_temperature": 0.80, "jump_temp_scale": 0.55}

    for attempt in range(max_retries + 2):
        if attempt > max_retries:
            if cell_owner is None:
                break
            failed_by_owner = Counter(cell_owner[idx] for idx in pending)
            pending = [idx for idx in pending
                       if owner_counts[cell_owner[idx]] >= 3 and failed_by_owner[cell_owner[idx]] == 1]
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
                        cfg_scale=cfg_scale,
                        max_steps=min(512, max(96, 18 * max_l + 48)),
                        return_meta=True,
                        **sampler_options,
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
            verdicts = judge_rollouts_with_confidence(
                judge_model, judge_tok, judge_rows, device, batch_size=judge_batch_size
            )
            for pos, (v_ok, pred_tex, reason, min_prob) in zip(geom_pass_pos, verdicts):
                if reason not in ("ok", "wrong_latex") or not min_prob >= min_judge_prob:
                    continue
                idx = pending[pos]
                target = flat_cells[idx]
                if (
                    v_ok or _otherwise_match(target, pred_tex)
                    or (_is_dangling_prefix_cell(target) and _prefix_tokens_match(target, pred_tex))
                    or _single_token_case_match(target, pred_tex)
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


def _load_generator_checkpoint(
    path: Path, device: torch.device,
) -> tuple[InkForgeModel, InkForgeTokenizer, dict[str, Any]]:
    try:
        from inkforge.v3 import load_inkforge_checkpoint
    except ModuleNotFoundError as exc:
        if exc.name != "inkforge.v3":
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") == "inkforge-checkpoint-v3":
            raise RuntimeError("Loading an InkForge v3 checkpoint requires the inkforge.v3 package") from exc
        cfg = InkForgeConfig.from_dict(checkpoint["config"])
        tokenizer = InkForgeTokenizer.from_dict(checkpoint["tokenizer"])
        model = InkForgeModel(cfg)
        model.load_state_dict(checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"])
        return model.to(device).eval(), tokenizer, checkpoint
    return load_inkforge_checkpoint(path, device, grad_checkpoint=False, prefer_ema=True)


_REAL_STRATA_WEIGHTS = {"single": 35, "long_tree": 35, "multiline": 12, "matrix": 10, "cases": 8}


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight InkForge synthetic human math handwriting generator")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained InkForge best.pt / latest.pt")
    parser.add_argument("--style-bank", type=Path, default=None, help="Optional writer_styles.npz from training")
    parser.add_argument("--unimer-jsonl", type=Path, default=DEFAULT_UNIMER_JSONL)
    parser.add_argument("--out-dir", type=Path, default=Path(r"E:\math-recogniser-training\inkforge_synth"))
    parser.add_argument("--target-count", type=int, default=200000, help="Total valid samples to generate")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=96, help="GPU cell batch size (96-128 fits RTX 4070 Ti easily)")
    parser.add_argument("--structured-prob", type=float, default=0.45, help="Fraction of prompts composed procedurally instead of sampled from UniMER")
    parser.add_argument("--temp-min", type=float, default=0.28)
    parser.add_argument("--temp-max", type=float, default=0.52)
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="Text CFG scale (1.5 for InkForge v2)")
    parser.add_argument("--judge-ckpt", type=Path, default=DEFAULT_INKTEX_V2_CKPT, help="InkTeX v2 judge for cell verification")
    parser.add_argument("--no-verify-cells", action="store_true", help="Disable InkTeX v2 atomic-cell verification")
    parser.add_argument("--max-cell-retries", type=int, default=3, help="Selective retries per failed cell with the same writer style")
    parser.add_argument("--min-judge-prob", type=float, default=0.60, help="Minimum InkTeX v2 greedy token probability (0 disables the confidence gate)")
    parser.add_argument("--preview-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0.0 <= args.min_judge_prob <= 1.0:
        parser.error("--min-judge-prob must be between 0 and 1")
    if args.max_cell_retries < 0:
        parser.error("--max-cell-retries must be nonnegative")
    if not args.no_verify_cells and not args.judge_ckpt.is_file():
        parser.error(f"InkTeX v2 judge checkpoint is missing at {args.judge_ckpt}. Set --judge-ckpt or --no-verify-cells.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, checkpoint = _load_generator_checkpoint(args.checkpoint, device)
    cfg = model.cfg
    step = checkpoint.get("step")
    del checkpoint
    print(f"[gen] Loaded InkForge ({model.parameter_count()/1e6:.2f}M params, step={step}) on {device}")

    judge_model = None
    judge_tok = None
    if not args.no_verify_cells:
        judge_model, judge_tok, _ = load_v2_checkpoint(args.judge_ckpt, None, device, use_ema=True)
        judge_model.eval()
        for p in judge_model.parameters():
            p.requires_grad_(False)
        print(f"[gen] Loaded InkTeX v2 verifier (retries={args.max_cell_retries}, "
              f"min_prob={args.min_judge_prob:.2f}) from {args.judge_ckpt}")

    style_bank = None
    bank_path = args.style_bank or (args.checkpoint.parent / "writer_styles.npz")
    if not bank_path.is_file():
        fallback_bank = Path(r"E:\math-recogniser-training\inkforge_v2_model\writer_styles.npz")
        if fallback_bank.is_file():
            bank_path = fallback_bank
    if bank_path.is_file():
        style_bank = np.load(bank_path)["mu_bank"]
        print(f"[gen] Loaded {len(style_bank):,} real human writer styles from {bank_path}")

    unimer_pools = load_stratified_unimer_pools(tokenizer, args.unimer_jsonl, max_prompts=120000, seed=args.seed)
    real_kinds = [kind for kind, pool in unimer_pools.items() if pool]
    real_weights = [_REAL_STRATA_WEIGHTS[kind] for kind in real_kinds]
    counts = ", ".join(f"{kind}={len(pool):,}" for kind, pool in unimer_pools.items())
    print(f"[gen] Loaded UniMER STEM prompts ({counts})")

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
    unimer_cursors = {kind: generated % len(unimer_pools[kind]) for kind in real_kinds}

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
                if real_kinds and py_rng.random() >= args.structured_prob:
                    kind = py_rng.choices(real_kinds, weights=real_weights)[0]
                    pool = unimer_pools[kind]
                    cursor = unimer_cursors[kind]
                    pr = pool[cursor]
                    unimer_cursors[kind] = (cursor + 1) % len(pool)
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
                cell_owner=cell_owner,
                min_judge_prob=args.min_judge_prob,
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
