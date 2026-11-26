"""InkTeX v2 Official Evaluator: Greedy, KV-Cached Beam-5, Shared Bidirectional R2L Reranking,
Oracle@5, Anti-Memorization Diagnostics, and v1-under-v2 Rescoring.

Evaluates InkTeX v2 checkpoints (Stage A stroke-only or Stage B/C bimodal) on:
  - Official unmodified CROHME 2019 (`CROHME2019_test`, 1,199) and CROHME 2023 (`CROHME2023_test`, 2,300),
    emitting a `predictions.json` directly compatible with `ml/evaluation/prepare.py` and WSL
    organizer `ml/evaluation/crohme.py` (SymLG / LgEval INTER D_B=0).
  - Official unmodified MathWriting Test (`mathwriting`, 7,644), reporting Google's exact
    reference-token CER (`training.benchmark_metrics.reference_token_cer`) on raw `normalizedLabel`,
    plus Standardizer v2 exact match, Oracle@K, and long-expression (>=40 tokens) exact match.
  - Locked v2 validation splits (`val_crohme`, `val_mw`) from `v2-shards`.
  - Oscar's native personal handwriting snapshot (`native`, 753 samples).

Anti-memorization diagnostics (when `--train-shards` is supplied):
  - Seen vs Unseen canonical formula exact match (based on `standardize_latex_v2(target)` in train).
  - Long-expression slice (`>=40` reference tokens).
  - Null-memory prior probe (16 samples decoded with all-masked memory to verify no blind hallucination).

Also includes `--rescore-json` to re-score any v1 `public_full_test.json` under Standardizer v2
for 100% apples-to-apples comparisons.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from data.canonical import expression_from_record
from data.importers.mathwriting import MathWritingImporter
from data.importers.native import NativeImporter
from tokenizer.latex_tokenizer import tokenize_latex
from training.benchmark_metrics import reference_token_cer
from training.metrics import edit_distance

from .ink_preprocess import (
    InkPreprocessV2Config,
    chunk_ink_v2,
)
from .model import InkTeXV2Model, SharedBidirectionalDecoder, get_v2_config
from .pack_v2 import (
    DEFAULT_CROHME,
    DEFAULT_MATHWRITING,
    expression_strokes_and_times,
)
from .shards_v2 import InkTeXV2EvalDataset, collate_v2_batch, resolve_v2_shards
from .tokenizer_v2 import MAX_SEQ_TOKENS_V2, LatexTokenizerV2, standardize_latex_v2
from .vision_raster import WIDTH_BUCKETS_V2, render_strokes_to_strip_v2, select_width_bucket

DEFAULT_NATIVE = Path(
    r"E:\math-recogniser-training\datasets\native-snapshots\full_synthetic_20260925_223326\examples.jsonl"
)


# --------------------------------------------------------------------------- KV-Cached Beam + R2L Reranker


@torch.inference_mode()
def decode_beam_cached(
    decoder: SharedBidirectionalDecoder,
    memory: torch.Tensor,                              # [B, M, D]
    memory_padding_mask: torch.Tensor | None = None,   # [B, M] True=pad
    *,
    beam_size: int = 5,
    max_tokens: int = MAX_SEQ_TOKENS_V2,
    length_penalty: float = 0.6,
    r2l_alpha: float = 0.0,
    bos_l2r_id: int | None = None,
    bos_r2l_id: int | None = None,
    eos_id: int | None = None,
    pad_id: int | None = None,
) -> dict[str, Any]:
    """Batched KV-cached L2R beam search with optional Shared Bidirectional R2L reranking.

    Returns:
      - `best_tokens`: list[list[int]] (length B, best hypothesis token IDs after BOS)
      - `beam_tokens`: list[list[list[int]]] (B x K hypotheses, sorted by final score)
      - `beam_scores`: list[list[float]] (B x K final scores)
      - `l2r_top1_tokens`: list[list[int]] (B, L2R top-1 before R2L reranking)
    """
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    bos_l2r = decoder.cfg.bos_l2r_id if bos_l2r_id is None else bos_l2r_id
    bos_r2l = decoder.cfg.bos_r2l_id if bos_r2l_id is None else bos_r2l_id
    eos = decoder.cfg.eos_id if eos_id is None else eos_id
    pad = decoder.cfg.pad_id if pad_id is None else pad_id

    bsz, mem_len, d_model = memory.shape
    if max_tokens <= 1:
        empty = [[] for _ in range(bsz)]
        return {
            "best_tokens": empty,
            "beam_tokens": [[[] for _ in range(beam_size)] for _ in range(bsz)],
            "beam_scores": [[0.0 for _ in range(beam_size)] for _ in range(bsz)],
            "l2r_top1_tokens": empty,
            "best_terminated": [False for _ in range(bsz)],
        }
    device = memory.device
    dtype = memory.dtype
    num_heads = decoder.cfg.decoder_heads
    head_dim = d_model // num_heads
    vocab = decoder.cfg.vocab_size
    k_beam = beam_size
    banned_ids = tuple({pad, bos_l2r, bos_r2l})

    # Precompute per-layer cross-attention memory K/V once
    mem_kv_base = [layer.project_memory(memory) for layer in decoder.layers]
    if memory_padding_mask is None:
        mem_bias_base = torch.zeros((bsz, 1, 1, mem_len), dtype=dtype, device=device)
    else:
        safe_mem = memory_padding_mask.clone()
        all_pad = safe_mem.all(dim=1, keepdim=True)
        first = torch.arange(mem_len, device=device).eq(0).unsqueeze(0)
        safe_mem = safe_mem & ~(all_pad & first)
        mem_bias_base = torch.zeros((bsz, 1, 1, mem_len), dtype=dtype, device=device)
        mem_bias_base = mem_bias_base.masked_fill(safe_mem[:, None, None, :], float("-inf"))

    slopes = decoder.alibi_slopes.to(dtype=dtype, device=device)[None, :, None, None]

    def _step(
        cur_ids: torch.Tensor,                         # [N, 1]
        step_idx: int,
        past_k: list[torch.Tensor | None],
        past_v: list[torch.Tensor | None],
        mem_kv: list[tuple[torch.Tensor, torch.Tensor]],
        mem_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        n = cur_ids.shape[0]
        x = decoder.token_embedding(cur_ids)
        dist = torch.arange(step_idx, -1, -1, device=device, dtype=dtype)[None, None, None, :]
        self_bias = -slopes * dist
        new_k: list[torch.Tensor] = []
        new_v: list[torch.Tensor] = []
        for li, layer in enumerate(decoder.layers):
            h = layer.norm1(x)
            qkv = layer.self_qkv(h).reshape(n, 1, 3, num_heads, head_dim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            if past_k[li] is None:
                fk, fv = k, v
            else:
                fk = torch.cat([past_k[li], k], dim=2)
                fv = torch.cat([past_v[li], v], dim=2)
            new_k.append(fk)
            new_v.append(fv)
            s_attn = F.scaled_dot_product_attention(q, fk, fv, attn_mask=self_bias)
            x = x + layer.self_out(s_attn.transpose(1, 2).reshape(n, 1, d_model))

            h2 = layer.norm2(x)
            cq = layer.cross_q(h2).reshape(n, 1, num_heads, head_dim).transpose(1, 2)
            mk, mv = mem_kv[li]
            c_attn = F.scaled_dot_product_attention(cq, mk, mv, attn_mask=mem_bias)
            x = x + layer.cross_out(c_attn.transpose(1, 2).reshape(n, 1, d_model))

            h3 = layer.norm3(x)
            x = x + layer.fc2(F.gelu(layer.fc1(h3)))
        logits = decoder.output(decoder.norm(x))[:, 0]
        return logits, new_k, new_v

    # Step 0: run single-beam BOS step [B, 1] -> top-K initial tokens
    bos_cur = torch.full((bsz, 1), bos_l2r, dtype=torch.long, device=device)
    logits0, pk0, pv0 = _step(
        bos_cur, 0, [None] * len(decoder.layers), [None] * len(decoder.layers), mem_kv_base, mem_bias_base
    )
    logp0 = F.log_softmax(logits0.float(), dim=-1)  # [B, V]
    if k_beam > 1:
        for bid in banned_ids:
            if 0 <= bid < vocab:
                logp0[:, bid] = -1e9
    top_lp, top_tok = torch.topk(logp0, k=min(k_beam, vocab), dim=-1)  # [B, K]
    if k_beam > vocab:
        pad_extra = k_beam - vocab
        top_lp = torch.cat([top_lp, torch.full((bsz, pad_extra), -1e9, device=device)], dim=1)
        top_tok = torch.cat([top_tok, torch.full((bsz, pad_extra), pad, dtype=torch.long, device=device)], dim=1)

    beam_scores = top_lp                                    # [B, K] raw sum of log-probs
    beam_tokens = top_tok.unsqueeze(-1)                     # [B, K, 1]
    beam_lens = torch.ones((bsz, k_beam), dtype=torch.long, device=device)
    finished = top_tok.eq(eos)                              # [B, K]

    # Expand memory K/V, bias, and step-0 self-attn K/V to [B * K, ...]
    mem_kv_rep = [
        (mk.repeat_interleave(k_beam, dim=0), mv.repeat_interleave(k_beam, dim=0))
        for mk, mv in mem_kv_base
    ]
    mem_bias_rep = mem_bias_base.repeat_interleave(k_beam, dim=0)
    past_k = [t.repeat_interleave(k_beam, dim=0) for t in pk0]
    past_v = [t.repeat_interleave(k_beam, dim=0) for t in pv0]
    batch_offsets = (torch.arange(bsz, device=device) * k_beam)[:, None]

    for step in range(1, max_tokens - 1):
        if bool(finished.all()):
            break
        cur = beam_tokens[:, :, -1].reshape(bsz * k_beam, 1)
        cur = cur.masked_fill(finished.reshape(bsz * k_beam, 1), pad)
        logits, full_k, full_v = _step(cur, step, past_k, past_v, mem_kv_rep, mem_bias_rep)
        logp = F.log_softmax(logits.float(), dim=-1).reshape(bsz, k_beam, vocab)
        if k_beam > 1:
            for bid in banned_ids:
                if 0 <= bid < vocab:
                    logp[:, :, bid] = -1e9

        if bool(finished.any()):
            fin_mask = finished[:, :, None]
            fin_lp = torch.full_like(logp, -1e9)
            fin_lp[:, :, pad] = 0.0
            logp = torch.where(fin_mask, fin_lp, logp)

        cand_raw = beam_scores[:, :, None] + logp           # [B, K, V]
        cand_lens = beam_lens + (~finished).long()          # [B, K]
        denom = cand_lens.float().pow(length_penalty)[:, :, None]
        cand_rank = cand_raw / denom

        flat_rank = cand_rank.reshape(bsz, k_beam * vocab)
        _, top_idx = torch.topk(flat_rank, k=k_beam, dim=-1)
        parent = top_idx // vocab                           # [B, K]
        next_tok = top_idx % vocab                          # [B, K]

        beam_scores = torch.gather(cand_raw.reshape(bsz, k_beam * vocab), 1, top_idx)
        beam_lens = torch.gather(cand_lens, 1, parent)
        parent_fin = torch.gather(finished, 1, parent)
        next_tok = next_tok.masked_fill(parent_fin, pad)
        finished = parent_fin | next_tok.eq(eos)

        gather_tok = parent.unsqueeze(-1).expand(-1, -1, beam_tokens.shape[2])
        beam_tokens = torch.cat([torch.gather(beam_tokens, 1, gather_tok), next_tok.unsqueeze(-1)], dim=-1)

        flat_gather = (batch_offsets + parent).reshape(-1)
        past_k = [fk.index_select(0, flat_gather) for fk in full_k]
        past_v = [fv.index_select(0, flat_gather) for fv in full_v]

    # Compute final length-normalized L2R scores
    l2r_norm = beam_scores / beam_lens.float().pow(length_penalty)  # [B, K]
    l2r_order = torch.argsort(l2r_norm, dim=-1, descending=True)
    l2r_top1 = torch.gather(
        beam_tokens, 1, l2r_order[:, :1, None].expand(-1, 1, beam_tokens.shape[2])
    )[:, 0].cpu().tolist()

    final_scores = l2r_norm
    if r2l_alpha > 0.0 and k_beam > 1:
        # Score all B * K hypotheses in reverse with ONE batched teacher-forced pass
        hyps_flat: list[list[int]] = []
        seq_cpu = beam_tokens.cpu().tolist()
        for b in range(bsz):
            for k in range(k_beam):
                body: list[int] = []
                for tid in seq_cpu[b][k]:
                    if tid == eos:
                        break
                    if tid in (pad, bos_l2r, bos_r2l):
                        continue
                    body.append(tid)
                hyps_flat.append(body[::-1])

        max_r2l = max((len(h) for h in hyps_flat), default=0) + 2  # BOS_R2L + body + EOS
        n_total = bsz * k_beam
        r2l_in = torch.full((n_total, max_r2l - 1), pad, dtype=torch.long, device=device)
        r2l_tgt = torch.full((n_total, max_r2l - 1), pad, dtype=torch.long, device=device)
        r2l_pad_mask = torch.ones((n_total, max_r2l - 1), dtype=torch.bool, device=device)
        r2l_lens = torch.ones(n_total, dtype=torch.float32, device=device)
        for i, rev in enumerate(hyps_flat):
            seq_in = [bos_r2l] + rev
            seq_tg = rev + [eos]
            L = len(seq_in)
            r2l_in[i, :L] = torch.tensor(seq_in, dtype=torch.long, device=device)
            r2l_tgt[i, :L] = torch.tensor(seq_tg, dtype=torch.long, device=device)
            r2l_pad_mask[i, :L] = False
            r2l_lens[i] = float(L)

        mem_rep = memory.repeat_interleave(k_beam, dim=0)
        mm_rep = memory_padding_mask.repeat_interleave(k_beam, dim=0) if memory_padding_mask is not None else None
        r2l_logits = decoder(r2l_in, mem_rep, mm_rep, r2l_pad_mask)  # [B*K, T, V]
        r2l_logp = F.log_softmax(r2l_logits.float(), dim=-1)
        tok_lp = torch.gather(r2l_logp, 2, r2l_tgt.unsqueeze(-1)).squeeze(-1)
        valid_pos = (~r2l_pad_mask).float()
        r2l_sum = (tok_lp * valid_pos).sum(dim=1)
        r2l_norm = (r2l_sum / r2l_lens.pow(length_penalty)).reshape(bsz, k_beam)
        final_scores = (1.0 - r2l_alpha) * l2r_norm + r2l_alpha * r2l_norm

    order = torch.argsort(final_scores, dim=-1, descending=True)
    sorted_scores = torch.gather(final_scores, 1, order).cpu().tolist()
    sorted_tokens = torch.gather(
        beam_tokens, 1, order.unsqueeze(-1).expand(-1, -1, beam_tokens.shape[2])
    ).cpu().tolist()
    sorted_fin = torch.gather(finished, 1, order).cpu().tolist()
    best_tokens = [row[0] for row in sorted_tokens]
    return {
        "best_tokens": best_tokens,
        "beam_tokens": sorted_tokens,
        "beam_scores": sorted_scores,
        "l2r_top1_tokens": l2r_top1,
        "best_terminated": [bool(row[0]) for row in sorted_fin],
    }


# --------------------------------------------------------------------------- Dataset Loaders


def _load_crohme_test(crohme_dir: Path) -> dict[str, list[dict[str, Any]]]:
    test_path = crohme_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"CROHME test file not found: {test_path}")
    editions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            expr = expression_from_record(rec)
            strokes, times = expression_strokes_and_times(expr)
            rec_id = str(rec["id"])
            parts = rec_id.split("/")
            edition = parts[1] if len(parts) > 1 else "CROHME_test"
            target = str(rec.get("latexNormalized") or rec.get("latex") or "")
            editions[edition].append({
                "id": rec_id,
                "target": target,
                "strokes": strokes,
                "times": times,
            })
    return dict(editions)


def _load_mathwriting_test(mw_dir: Path) -> list[dict[str, Any]]:
    test_dir = mw_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"MathWriting test directory not found: {test_dir}")
    inkml_files = sorted(test_dir.glob("*.inkml"))
    if not inkml_files:
        raise FileNotFoundError(f"No .inkml files found in {test_dir}")
    importer = MathWritingImporter()
    rows: list[dict[str, Any]] = []
    for path in inkml_files:
        expr = importer.parse(path)
        strokes, times = expression_strokes_and_times(expr)
        target = str(expr.latex_normalized or expr.latex_original or "")
        rows.append({
            "id": str(expr.id),
            "target": target,
            "strokes": strokes,
            "times": times,
        })
    return rows


def _load_native_snapshot(native_path: Path) -> list[dict[str, Any]]:
    if not native_path.is_file():
        raise FileNotFoundError(f"Native snapshot not found: {native_path}")
    expressions = NativeImporter().load_file(native_path)
    if not expressions:
        raise ValueError(f"Native snapshot yielded 0 examples: {native_path}")
    rows: list[dict[str, Any]] = []
    for expr in expressions:
        strokes, times = expression_strokes_and_times(expr)
        rows.append({
            "id": f"native:{expr.id}",
            "target": str(expr.latex_normalized or expr.latex_original or ""),
            "strokes": strokes,
            "times": times,
        })
    return rows


def _load_locked_val(
    shards_dir: Path, split: str, tok: LatexTokenizerV2, *, need_vision: bool = False, need_strokes: bool = True
) -> list[dict[str, Any]]:
    ds = InkTeXV2EvalDataset(shards_dir, split, tok, need_vision=need_vision, need_strokes=need_strokes)
    rows: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        s = ds[idx]
        rows.append({
            "id": s["id"],
            # Preserve the original benchmark reference (`latex`) so Google reference-token CER
            # and its standardization floor are measured accurately on val_mw / val_crohme.
            "target": str(s.get("latex") or s["normalized_latex"]),
            "precomputed": s,
        })
    return rows


def _build_seen_train_labels(
    shards_dir: Path, splits: Sequence[str] = ("train_online", "train_offline")
) -> set[str]:
    """Scan shard metadata JSONs across training splits to build exact set of standardized labels."""
    import tarfile

    seen: set[str] = set()
    for split in splits:
        split_dir = shards_dir / split
        if not split_dir.is_dir():
            continue
        for tar_path in resolve_v2_shards(shards_dir, split):
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if not member.name.endswith(".json"):
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    meta = json.loads(f.read().decode("utf-8"))
                    lbl = meta.get("normalized_latex")
                    if lbl:
                        seen.add(lbl)
    if not seen:
        raise ValueError(f"--seen-check found 0 training labels under {shards_dir} (splits={list(splits)})")
    return seen


# --------------------------------------------------------------------------- Batch Featurization + Evaluation


def _prepare_row_vision(row: dict[str, Any]) -> tuple[torch.Tensor, int]:
    """Return `(image_tensor [2,H,W] uint8, image_width)` for a row, rendering once if needed."""
    if "precomputed" in row and row["precomputed"].get("image") is not None:
        return row["precomputed"]["image"], int(row["precomputed"]["image_width"])
    rendered = render_strokes_to_strip_v2(row["strokes"])
    arr = np.ascontiguousarray(np.asarray(rendered.image, dtype=np.uint8).transpose(2, 0, 1))
    return torch.from_numpy(arr), int(rendered.image_width)


def _featurize_eval_batch(
    rows: Sequence[dict[str, Any]],
    *,
    use_vision: bool,
    use_strokes: bool = True,
    pad_id: int = 0,
    static_buckets: bool = True,
) -> dict[str, Any]:
    """Featurize raw stroke rows using the exact same `chunk_ink_v2`, `render_strokes_to_strip_v2`,
    and `collate_v2_batch` path as training and ONNX export."""
    ink_cfg = InkPreprocessV2Config()
    dummy_tok = torch.zeros((2,), dtype=torch.long)
    samples: list[dict[str, Any]] = []
    for r in rows:
        if "precomputed" in r:
            s = dict(r["precomputed"])
            if not use_vision:
                s["image"] = None
                s["image_width"] = None
            if not use_strokes:
                s["chunk_points"] = None
                s["chunk_point_mask"] = None
                s["chunk_meta"] = None
                s["chunk_mask"] = None
            samples.append(s)
            continue
        chunk_points = chunk_point_mask = chunk_meta = chunk_mask = None
        coarsened = False
        if use_strokes:
            strokes = r["strokes"]
            times = r.get("times")
            chunked = chunk_ink_v2(strokes, times, ink_cfg)
            v = max(chunked.valid_chunks, 1)
            coarsened = chunked.coarsened
            chunk_points = torch.from_numpy(chunked.chunk_points[:v].copy())
            chunk_point_mask = torch.from_numpy(chunked.chunk_point_mask[:v].copy())
            chunk_meta = torch.from_numpy(chunked.chunk_meta[:v].copy())
            chunk_mask = torch.from_numpy(chunked.chunk_mask[:v].copy())
        img_tensor = None
        img_width = None
        if use_vision:
            if "_cached_image" in r:
                img_tensor, img_width = r["_cached_image"], int(r["_cached_width"])
            else:
                img_tensor, img_width = _prepare_row_vision(r)
        samples.append({
            "id": r["id"],
            "source": "eval",
            "writer_id": None,
            "latex": r["target"],
            "normalized_latex": r["target"],
            "synthetic": False,
            "coarsened": coarsened,
            "chunk_points": chunk_points,
            "chunk_point_mask": chunk_point_mask,
            "chunk_meta": chunk_meta,
            "chunk_mask": chunk_mask,
            "image": img_tensor,
            "image_width": img_width,
            "l2r_tokens": dummy_tok,
            "r2l_tokens": dummy_tok,
        })
    return collate_v2_batch(samples, pad_id=pad_id, static_buckets=static_buckets)


def _safe_std_v2(latex: str) -> str:
    try:
        res = standardize_latex_v2(latex)
        return res if res is not None else latex.strip()
    except Exception:
        return latex.strip()


def score_predictions(
    predictions: Sequence[dict[str, Any]],
    *,
    is_mathwriting: bool = False,
    seen_labels: set[str] | None = None,
) -> dict[str, Any]:
    n = len(predictions)
    if n == 0:
        return {"examples": 0}
    exact = 0
    l2r_exact = 0
    oracle_exact = 0
    ed_sum = 0
    ref_len_sum = 0
    long_n = 0
    long_exact = 0
    seen_n = 0
    seen_exact = 0
    unseen_n = 0
    unseen_exact = 0

    raw_preds: list[str] = []
    raw_targets: list[str] = []

    for row in predictions:
        pred = str(row["prediction"])
        target = str(row["target"])
        raw_preds.append(pred)
        raw_targets.append(target)

        t_std = _safe_std_v2(target)
        p_std = _safe_std_v2(pred)
        t_toks = tokenize_latex(t_std)
        p_toks = tokenize_latex(p_std)

        ok = p_toks == t_toks
        if ok:
            exact += 1
        ed_sum += edit_distance(p_toks, t_toks)
        ref_len_sum += max(len(t_toks), 1)

        if "l2r_prediction" in row:
            if tokenize_latex(_safe_std_v2(str(row["l2r_prediction"]))) == t_toks:
                l2r_exact += 1
        elif ok:
            l2r_exact += 1

        hyps = row.get("beam_predictions")
        if hyps:
            if any(tokenize_latex(_safe_std_v2(str(h))) == t_toks for h in hyps):
                oracle_exact += 1
        elif ok:
            oracle_exact += 1

        if len(t_toks) >= 40:
            long_n += 1
            if ok:
                long_exact += 1

        if seen_labels is not None:
            if t_std in seen_labels:
                seen_n += 1
                if ok:
                    seen_exact += 1
            else:
                unseen_n += 1
                if ok:
                    unseen_exact += 1

    wrong_in_train = 0
    wrong_total = 0
    if seen_labels is not None:
        for row in predictions:
            t_std = _safe_std_v2(str(row["target"]))
            p_std = _safe_std_v2(str(row["prediction"]))
            if tokenize_latex(p_std) != tokenize_latex(t_std):
                wrong_total += 1
                if p_std in seen_labels:
                    wrong_in_train += 1

    metrics: dict[str, Any] = {
        "examples": n,
        "std_v2_exact": exact / n,
        "l2r_top1_std_v2_exact": l2r_exact / n,
        "oracle_k_std_v2_exact": oracle_exact / n,
        "std_v2_token_cer": ed_sum / max(ref_len_sum, 1),
        "long_40_examples": long_n,
        "long_40_std_v2_exact": (long_exact / long_n) if long_n > 0 else None,
    }
    if is_mathwriting:
        metrics["reference_token_cer"] = reference_token_cer(raw_preds, raw_targets)
        metrics["reference_token_cer_std_floor"] = reference_token_cer(
            [_safe_std_v2(t) for t in raw_targets], raw_targets
        )
    if seen_labels is not None:
        metrics["seen_examples"] = seen_n
        metrics["seen_std_v2_exact"] = (seen_exact / seen_n) if seen_n > 0 else None
        metrics["unseen_examples"] = unseen_n
        metrics["unseen_std_v2_exact"] = (unseen_exact / unseen_n) if unseen_n > 0 else None
        metrics["wrong_pred_in_train_rate"] = (wrong_in_train / wrong_total) if wrong_total > 0 else 0.0
    return metrics


@torch.inference_mode()
def evaluate_dataset(
    model: InkTeXV2Model,
    tok: LatexTokenizerV2,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    mode: str,
    decode_method: str,
    beam_size: int,
    r2l_alpha: float,
    length_penalty: float,
    batch_size: int,
    max_tokens: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
    is_mathwriting: bool = False,
    seen_labels: set[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    if mode not in ("stroke", "vision", "bimodal"):
        raise ValueError(f"Unknown eval mode: {mode}")
    use_vision = mode in ("vision", "bimodal")
    use_strokes = mode in ("stroke", "bimodal")
    modality_mode = mode if mode in ("stroke", "vision") else "auto"
    n = len(rows)
    predictions: list[dict[str, Any] | None] = [None] * n
    gpu_ms_total = 0.0

    # Group batches by each strip's OWN width bucket when vision is active so ConvNeXt-V2 GRN
    # padding is 100% independent of batch size and neighbor composition (matches single-sample iPad).
    work_rows: list[dict[str, Any]] = []
    if use_vision:
        bucket_groups: dict[int, list[int]] = defaultdict(list)
        for idx, r in enumerate(rows):
            if "precomputed" in r and r["precomputed"].get("image") is not None:
                w = int(r["precomputed"]["image_width"])
                work_rows.append(r)
            else:
                img_t, w = _prepare_row_vision(r)
                rc = dict(r)
                rc["_cached_image"] = img_t
                rc["_cached_width"] = w
                work_rows.append(rc)
            bucket_groups[select_width_bucket(w, WIDTH_BUCKETS_V2)].append(idx)
        batch_index_lists: list[list[int]] = []
        for b_key in sorted(bucket_groups.keys()):
            idxs = bucket_groups[b_key]
            for s in range(0, len(idxs), batch_size):
                batch_index_lists.append(idxs[s : s + batch_size])
    else:
        work_rows = list(rows)
        batch_index_lists = [list(range(s, min(s + batch_size, n))) for s in range(0, n, batch_size)]

    for batch_idxs in batch_index_lists:
        chunk_rows = [work_rows[i] for i in batch_idxs]
        batch = _featurize_eval_batch(chunk_rows, use_vision=use_vision, use_strokes=use_strokes, pad_id=tok.pad_id)
        cp = batch["chunk_points"].to(device, non_blocking=True) if batch["chunk_points"] is not None else None
        cpm = batch["chunk_point_mask"].to(device, non_blocking=True) if batch["chunk_point_mask"] is not None else None
        cm = batch["chunk_meta"].to(device, non_blocking=True) if batch["chunk_meta"] is not None else None
        ck = batch["chunk_mask"].to(device, non_blocking=True) if batch["chunk_mask"] is not None else None
        imgs = batch["images"].to(device, non_blocking=True) if batch["images"] is not None else None
        img_w = batch["image_widths"].to(device, non_blocking=True) if batch["image_widths"] is not None else None

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            mem, mem_mask, _ = model.encode(
                images=imgs, image_widths=img_w,
                chunk_points=cp, chunk_point_mask=cpm, chunk_meta=cm, chunk_mask=ck,
                modality_mode=modality_mode,
            )
            if decode_method == "greedy":
                best_ids = model.decoder.decode_greedy_cached(
                    mem, mem_mask,
                    bos_id=tok.bos_l2r_id, eos_id=tok.eos_id, pad_id=tok.pad_id,
                    max_tokens=max_tokens,
                ).cpu().tolist()
                beam_out = {
                    "best_tokens": best_ids,
                    "beam_tokens": [[ids] for ids in best_ids],
                    "l2r_top1_tokens": best_ids,
                    "best_terminated": [tok.eos_id in ids for ids in best_ids],
                }
            else:
                alpha = r2l_alpha if decode_method == "beam_r2l" else 0.0
                beam_out = decode_beam_cached(
                    model.decoder, mem, mem_mask,
                    beam_size=beam_size, max_tokens=max_tokens,
                    length_penalty=length_penalty, r2l_alpha=alpha,
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_ms_total += (time.perf_counter() - t0) * 1000.0

        for orig_idx, r, best_t, l2r_t, hyps_t, term in zip(
            batch_idxs, chunk_rows, beam_out["best_tokens"], beam_out["l2r_top1_tokens"],
            beam_out["beam_tokens"], beam_out["best_terminated"],
        ):
            rec_out: dict[str, Any] = {
                "id": r["id"],
                "target": r["target"],
                "prediction": tok.decode(best_t),
                "terminated": bool(term),
            }
            if decode_method != "greedy":
                rec_out["l2r_prediction"] = tok.decode(l2r_t)
                rec_out["beam_predictions"] = [tok.decode(h) for h in hyps_t]
            predictions[orig_idx] = rec_out

    final_preds = [p for p in predictions if p is not None]
    metrics = score_predictions(final_preds, is_mathwriting=is_mathwriting, seen_labels=seen_labels)
    metrics["gpu_ms_per_sample"] = gpu_ms_total / max(n, 1)
    metrics["unterminated_rate"] = sum(not p["terminated"] for p in final_preds) / max(n, 1)
    return {
        "examples": n,
        "metrics": metrics,
        "predictions": final_preds,
    }


@torch.inference_mode()
def null_memory_probe(
    model: InkTeXV2Model,
    tok: LatexTokenizerV2,
    device: torch.device,
    *,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Probe decoder unconditional prior on zeroed fully-masked memory (1 deterministic draw)."""
    model.eval()
    mem = torch.zeros((1, 4, model.config.d_model), device=device)
    mem_mask = torch.ones((1, 4), dtype=torch.bool, device=device)
    ids = model.decoder.decode_greedy_cached(
        mem, mem_mask,
        bos_id=tok.bos_l2r_id, eos_id=tok.eos_id, pad_id=tok.pad_id,
        max_tokens=max_tokens,
    ).cpu().tolist()[0]
    decoded = tok.decode(ids)
    return {"zero_memory_decode": decoded, "tokens": len(tokenize_latex(decoded))}


def load_v2_checkpoint(
    ckpt_path: Path,
    tokenizer_path: Path | None,
    device: torch.device,
    *,
    use_ema: bool = True,
) -> tuple[InkTeXV2Model, LatexTokenizerV2, dict[str, Any]]:
    from .model import InkTeXV2Config
    from .train_v2 import CHECKPOINT_FORMAT_V2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("format") not in (None, CHECKPOINT_FORMAT_V2):
        raise ValueError(f"Expected {CHECKPOINT_FORMAT_V2}, got {ckpt.get('format')!r}")

    if tokenizer_path is not None:
        tok = LatexTokenizerV2.load(tokenizer_path)
    elif "tokenizer" in ckpt:
        tok = LatexTokenizerV2.from_dict(ckpt["tokenizer"])
    else:
        cand = ckpt_path.parent / "tokenizer_v2.json"
        if not cand.is_file():
            cand = Path(r"E:\math-recogniser-training\v2-shards\tokenizer_v2.json")
        if not cand.is_file():
            raise FileNotFoundError("Could not locate tokenizer_v2.json; pass --tokenizer explicitly")
        tok = LatexTokenizerV2.load(cand)

    model_cfg_dict = ckpt.get("model_config")
    if model_cfg_dict is not None:
        cfg = InkTeXV2Config.from_dict({**model_cfg_dict, "vision_pretrained": False})
    else:
        preset = ckpt.get("config", {}).get("model", {}).get("preset", "inktex_v2_100m")
        cfg = get_v2_config(preset, vocab_size=len(tok), vision_pretrained=False)

    model = InkTeXV2Model(cfg)
    if not use_ema:
        if "live_state_dict" not in ckpt:
            raise ValueError(f"--no-ema requested, but {ckpt_path} has no live_state_dict (use recovery.pt)")
        state_key = "live_state_dict"
    elif "model_state_dict" in ckpt:
        state_key = "model_state_dict"
    else:
        raise KeyError(f"Checkpoint missing model_state_dict: keys={sorted(ckpt.keys())}")
    model.load_state_dict(ckpt[state_key], strict=True)
    if ckpt.get("lora_enabled"):
        model.set_lora_enabled(True)
    model.to(device).eval()
    return model, tok, ckpt


def rescore_predictions_json(json_path: Path, seen_labels: set[str] | None = None) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "source_file": str(json_path.resolve()),
        "checkpoint": data.get("checkpoint"),
        "datasets": {},
    }
    for name, ds_data in data.get("datasets", {}).items():
        preds = ds_data.get("predictions", [])
        is_mw = "mathwriting" in name.lower()
        out["datasets"][name] = score_predictions(preds, is_mathwriting=is_mw, seen_labels=seen_labels)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="InkTeX v2 Official Evaluator")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path (predictions + metrics)")
    parser.add_argument("--mode", choices=("stroke", "vision", "bimodal"), default="stroke")
    parser.add_argument("--decode", choices=("greedy", "beam", "beam_r2l"), default="beam_r2l")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--r2l-alpha", type=float, default=0.4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=MAX_SEQ_TOKENS_V2)
    parser.add_argument("--no-ema", action="store_true", help="Evaluate live_state_dict weights instead of EMA")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["crohme2019", "crohme2023", "mathwriting"],
        choices=("crohme2019", "crohme2023", "mathwriting", "val_crohme", "val_mw", "native"),
    )
    parser.add_argument("--crohme-dir", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--mathwriting-dir", type=Path, default=DEFAULT_MATHWRITING)
    parser.add_argument("--shards-dir", type=Path, default=Path(r"E:\math-recogniser-training\v2-shards"))
    parser.add_argument("--native-path", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--seen-check", action="store_true", help="Compute seen vs unseen formula split from train shards")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-dataset cap for fast smoke tests (0=all)")
    parser.add_argument("--rescore-json", type=Path, default=None, help="Re-score an existing v1/v2 predictions JSON under Standardizer v2")
    args = parser.parse_args()

    seen_labels: set[str] | None = None
    if args.seen_check:
        print("[eval_v2] Indexing train_online + train_offline standardized labels for seen/unseen split...", flush=True)
        seen_labels = _build_seen_train_labels(args.shards_dir, ("train_online", "train_offline"))
        print(f"[eval_v2] Indexed {len(seen_labels):,} unique standardized training labels", flush=True)

    if args.rescore_json is not None:
        summary = rescore_predictions_json(args.rescore_json, seen_labels=seen_labels)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --rescore-json is passed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    model, tok, ckpt = load_v2_checkpoint(args.checkpoint, args.tokenizer, device, use_ema=not args.no_ema)
    print(
        f"[eval_v2] Loaded {args.checkpoint} (step={ckpt.get('global_step')}, mode={args.mode}, "
        f"decode={args.decode}, K={args.beam_size}, r2l_alpha={args.r2l_alpha}) on {device}",
        flush=True,
    )

    crohme_editions: dict[str, list[dict[str, Any]]] | None = None
    if "crohme2019" in args.datasets or "crohme2023" in args.datasets:
        crohme_editions = _load_crohme_test(args.crohme_dir)

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": ckpt.get("global_step"),
        "split": "test_limited" if args.limit > 0 else "test",
        "mode": args.mode,
        "decode": args.decode,
        "beam_size": args.beam_size if args.decode != "greedy" else 1,
        "r2l_alpha": args.r2l_alpha if args.decode == "beam_r2l" else 0.0,
        "length_penalty": args.length_penalty,
        "null_memory_probe": null_memory_probe(model, tok, device),
        "datasets": {},
    }

    for ds_name in args.datasets:
        if ds_name == "crohme2019":
            out_key = "CROHME2019_test"
            rows = crohme_editions["CROHME2019_test"]
            is_mw = False
        elif ds_name == "crohme2023":
            out_key = "CROHME2023_test"
            rows = crohme_editions["CROHME2023_test"]
            is_mw = False
        elif ds_name == "mathwriting":
            out_key = "mathwriting"
            rows = _load_mathwriting_test(args.mathwriting_dir)
            is_mw = True
        elif ds_name in ("val_crohme", "val_mw"):
            out_key = ds_name
            rows = _load_locked_val(
                args.shards_dir, ds_name, tok,
                need_vision=(args.mode in ("vision", "bimodal")),
                need_strokes=(args.mode in ("stroke", "bimodal")),
            )
            is_mw = ds_name == "val_mw"
        elif ds_name == "native":
            out_key = "native"
            rows = _load_native_snapshot(args.native_path)
            is_mw = False
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

        if args.limit > 0:
            rows = rows[: args.limit]
        print(f"[eval_v2] Evaluating {out_key} ({len(rows):,} samples)...", flush=True)
        t0 = time.perf_counter()
        ds_res = evaluate_dataset(
            model, tok, rows, device,
            mode=args.mode, decode_method=args.decode,
            beam_size=args.beam_size, r2l_alpha=args.r2l_alpha,
            length_penalty=args.length_penalty,
            batch_size=args.batch_size, max_tokens=args.max_tokens,
            amp_dtype=amp_dtype, use_amp=use_amp,
            is_mathwriting=is_mw, seen_labels=seen_labels,
        )
        elapsed = time.perf_counter() - t0
        m = ds_res["metrics"]
        cer_str = f" ref_cer={m['reference_token_cer']:.4f}" if "reference_token_cer" in m else ""
        print(
            f"[eval_v2] {out_key}: exact={m['std_v2_exact']:.2%} (l2r={m['l2r_top1_std_v2_exact']:.2%}, "
            f"oracle@{result['beam_size']}={m['oracle_k_std_v2_exact']:.2%}) "
            f"tok_cer={m['std_v2_token_cer']:.4f}{cer_str} gpu={m['gpu_ms_per_sample']:.2f}ms/ex ({elapsed:.1f}s)",
            flush=True,
        )
        result["datasets"][out_key] = ds_res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_only = {
        k: v for k, v in result.items() if k != "datasets"
    }
    summary_only["datasets"] = {k: v["metrics"] for k, v in result["datasets"].items()}
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary_only, indent=2), encoding="utf-8")
    print(f"[eval_v2] Saved full predictions -> {args.output} and summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
