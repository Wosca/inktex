"""Verify numerical & decode parity between PyTorch BimodalMathModel and exported ONNX bundle."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import load_bimodal_checkpoint
from inktex_model.preprocessing import PreprocessingConfig, StrokePoint, preprocess_strokes
from inktex_model.raster import render_strokes_to_strip
from inktex_model.tokenizer import LatexTokenizer


def synthetic_x_squared_plus_1() -> list[list[StrokePoint]]:
    """Draw a clean handwritten-like 'x^2 + 1' with realistic timestamps."""
    strokes: list[list[StrokePoint]] = []
    t = 0.0

    def add(pts: list[tuple[float, float]]) -> None:
        nonlocal t
        stroke = []
        for x, y in pts:
            stroke.append(StrokePoint(x=x, y=y, time=t))
            t += 12.0
        strokes.append(stroke)
        t += 120.0

    # 'x' (two diagonals around x=[0..40], y=[40..80])
    add([(0 + 40 * i / 10, 40 + 40 * i / 10) for i in range(11)])
    add([(40 - 40 * i / 10, 40 + 40 * i / 10) for i in range(11)])
    # '^2' (superscript around x=[48..72], y=[10..38])
    add([
        (48, 18), (54, 11), (64, 11), (70, 16), (68, 24),
        (58, 31), (48, 38), (60, 38), (72, 38),
    ])
    # '+' (around x=[90..130], y=[40..80])
    add([(110, 40 + 40 * i / 8) for i in range(9)])
    add([(90 + 40 * i / 8, 60) for i in range(9)])
    # '1' (around x=[152..152], y=[36..80])
    add([(152, 36 + 44 * i / 10) for i in range(11)])
    return strokes


def run_onnx_greedy(
    bundle_dir: Path,
    images: np.ndarray | None,
    image_widths: np.ndarray | None,
    stroke_features: np.ndarray | None,
    stroke_mask_u8: np.ndarray | None,
    modality: str = "bimodal",
    max_tokens: int = 64,
) -> tuple[list[int], str, float, np.ndarray]:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = LatexTokenizer(manifest["tokens"])
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    vis_sess = ort.InferenceSession(str(bundle_dir / "vision_encoder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    str_sess = ort.InferenceSession(str(bundle_dir / "stroke_encoder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    proj_sess = ort.InferenceSession(str(bundle_dir / "memory_proj.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(bundle_dir / "decoder_step.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

    started = time.perf_counter()
    mems = []
    masks = []
    if modality in ("bimodal", "stroke") and stroke_features is not None:
        sm, smask = str_sess.run(None, {"stroke_features": stroke_features, "stroke_padding_mask": stroke_mask_u8})
        mems.append(sm)
        masks.append(smask)
    if modality in ("bimodal", "vision") and images is not None:
        vm, vmask = vis_sess.run(None, {"images": images, "image_widths": image_widths})
        mems.append(vm)
        masks.append(vmask)

    memory = np.concatenate(mems, axis=1) if len(mems) > 1 else mems[0]
    memory_mask = np.concatenate(masks, axis=1) if len(masks) > 1 else masks[0]

    mk, mv = proj_sess.run(None, {"memory": memory})
    num_layers = manifest["decoder_layers"]
    num_heads = manifest["attention_heads"]
    head_dim = manifest["head_dim"]
    mem_len = memory.shape[1]

    past_k = np.zeros((num_layers, 1, num_heads, 1, head_dim), dtype=np.float32)
    past_v = np.zeros((num_layers, 1, num_heads, 1, head_dim), dtype=np.float32)
    coverage = np.zeros((1, num_heads, 1, mem_len), dtype=np.float32)
    prev_weights = np.zeros((1, num_heads, 1, mem_len), dtype=np.float32)

    token = np.array([[manifest["bos_id"]]], dtype=np.int64)
    generated: list[int] = []
    first_logits: np.ndarray | None = None

    for _ in range(max_tokens):
        logits, past_k, past_v, coverage, prev_weights = dec_sess.run(
            None,
            {
                "token": token,
                "past_k": past_k,
                "past_v": past_v,
                "mk": mk,
                "mv": mv,
                "memory_mask": memory_mask,
                "coverage": coverage,
                "prev_weights": prev_weights,
            },
        )
        if first_logits is None:
            first_logits = logits
        next_id = int(np.argmax(logits[0]))
        generated.append(next_id)
        if next_id == manifest["eos_id"]:
            break
        token = np.array([[next_id]], dtype=np.int64)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert first_logits is not None
    return generated, tokenizer.decode(generated), elapsed_ms, first_logits


def verify_parity(checkpoint_path: Path, bundle_dir: Path, tol: float = 1e-3) -> None:
    model, tokenizer, _ = load_bimodal_checkpoint(checkpoint_path)
    strokes = synthetic_x_squared_plus_1()
    raw_xy = [np.asarray([(p.x, p.y) for p in s], dtype=np.float64) for s in strokes]

    strip_img, used_width = render_strokes_to_strip(raw_xy, height=128, min_width=64, max_width=640, padding=8)
    la = np.asarray(strip_img, dtype=np.uint8).transpose(2, 0, 1)  # [2, 128, W]
    images = (la.astype(np.float32) / 255.0)[None, ...]
    image_widths = np.array([used_width], dtype=np.int64)

    features = preprocess_strokes(strokes, PreprocessingConfig())[None, ...]
    stroke_mask_u8 = np.zeros((1, features.shape[1]), dtype=np.uint8)

    for modality in ("bimodal", "vision", "stroke"):
        with torch.inference_mode():
            pt_img = torch.from_numpy(images) if modality in ("bimodal", "vision") else None
            pt_w = torch.from_numpy(image_widths) if modality in ("bimodal", "vision") else None
            pt_str = torch.from_numpy(features) if modality in ("bimodal", "stroke") else None
            pt_smask = torch.zeros((1, features.shape[1]), dtype=torch.bool) if modality in ("bimodal", "stroke") else None
            pt_has = torch.tensor([True]) if modality in ("bimodal", "stroke") else None

            mem, mem_mask, _ = model.encode_bimodal(
                images=pt_img, image_widths=pt_w,
                stroke_features=pt_str, stroke_padding_mask=pt_smask,
                has_strokes=pt_has, modality_mode=modality,
            )
            pt_first_logits = model.decoder(
                torch.tensor([[tokenizer.bos_id]], dtype=torch.long),
                mem,
                mem_mask,
            )[:, 0, :].numpy()
            pt_tokens = model.decode_greedy(
                images=pt_img, image_widths=pt_w,
                stroke_features=pt_str, stroke_padding_mask=pt_smask,
                has_strokes=pt_has, max_tokens=64, cached=True,
            )[0].tolist()
            # Trim after EOS
            if tokenizer.eos_id in pt_tokens:
                pt_tokens = pt_tokens[: pt_tokens.index(tokenizer.eos_id) + 1]
            pt_latex = tokenizer.decode(pt_tokens)

        ort_tokens, ort_latex, elapsed_ms, ort_first_logits = run_onnx_greedy(
            bundle_dir, images, image_widths, features, stroke_mask_u8, modality=modality, max_tokens=64,
        )
        max_diff = float(np.max(np.abs(pt_first_logits - ort_first_logits)))
        print(
            f"[{modality:7s}] max_logit_diff={max_diff:.6g} | latency={elapsed_ms:.1f}ms | "
            f"PyTorch='{pt_latex}' | ONNX='{ort_latex}'"
        )
        if max_diff > tol:
            raise AssertionError(f"{modality}: logit diff {max_diff:.6g} exceeds tolerance {tol}")
        if pt_tokens != ort_tokens:
            raise AssertionError(f"{modality}: token mismatch PyTorch={pt_tokens} vs ONNX={ort_tokens}")

    # Also test single-stroke edge case & wide strip edge case
    single_stroke = [strokes[0]]
    s_xy = [np.asarray([(p.x, p.y) for p in s], dtype=np.float64) for s in single_stroke]
    s_img, s_w = render_strokes_to_strip(s_xy)
    s_la = (np.asarray(s_img, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]
    s_feat = preprocess_strokes(single_stroke, PreprocessingConfig())[None, ...]
    s_mask = np.zeros((1, s_feat.shape[1]), dtype=np.uint8)
    ort_toks, ort_tex, ms, _ = run_onnx_greedy(
        bundle_dir, s_la, np.array([s_w], dtype=np.int64), s_feat, s_mask, modality="bimodal", max_tokens=32,
    )
    print(f"[1-stroke edge case] latency={ms:.1f}ms | ONNX='{ort_tex}' ({len(ort_toks)} tokens)")

    # Verify R2L scorer parity if present
    r2l_path = bundle_dir / "r2l_scorer.onnx"
    if r2l_path.is_file() and model.r2l_decoder is not None:
        r2l_sess = ort.InferenceSession(str(r2l_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            mem, mem_mask, _ = model.encode_bimodal(
                images=torch.from_numpy(images),
                image_widths=torch.from_numpy(image_widths),
                stroke_features=torch.from_numpy(features),
                stroke_padding_mask=torch.zeros((1, features.shape[1]), dtype=torch.bool),
                has_strokes=torch.tensor([True]),
                modality_mode="bimodal",
            )
            # Candidate 1: true decode ({x^{2}}+1), Candidate 2: distractor (x+2)
            cand1 = [t for t in pt_tokens if t != tokenizer.eos_id]
            cand2 = [tokenizer.token_to_id[t] for t in ("x", "+", "2")]
            r2l_seqs = [
                [tokenizer.bos_id, *reversed(cand1), tokenizer.eos_id],
                [tokenizer.bos_id, *reversed(cand2), tokenizer.eos_id],
            ]
            max_len = max(len(s) for s in r2l_seqs)
            tok_mat = np.full((2, max_len), tokenizer.pad_id, dtype=np.int64)
            pad_mat = np.ones((2, max_len), dtype=np.uint8)
            for row, seq in enumerate(r2l_seqs):
                tok_mat[row, : len(seq)] = seq
                pad_mat[row, : len(seq)] = 0

            # PyTorch reference
            pt_inp = torch.from_numpy(tok_mat[:, :-1])
            pt_tgt = torch.from_numpy(tok_mat[:, 1:])
            pt_inp_mask = torch.from_numpy(pad_mat[:, :-1] != 0)
            pt_tgt_valid = torch.from_numpy(pad_mat[:, 1:] == 0).float()
            pt_logits = model.r2l_decoder(
                pt_inp,
                mem.expand(2, -1, -1),
                mem_mask.expand(2, -1),
                pt_inp_mask,
            )
            pt_lp = torch.nn.functional.log_softmax(pt_logits, dim=-1).gather(-1, pt_tgt.unsqueeze(-1)).squeeze(-1)
            pt_seq_lp = (pt_lp * pt_tgt_valid).sum(-1).numpy()
            pt_avg_lp = (pt_seq_lp / pt_tgt_valid.sum(-1).clamp_min(1.0).numpy())

        t0 = time.perf_counter()
        ort_seq_lp, ort_avg_lp = r2l_sess.run(
            None,
            {
                "r2l_tokens": tok_mat,
                "r2l_padding_mask": pad_mat,
                "memory": mem.numpy(),
                "memory_mask": mem_mask.to(torch.uint8).numpy(),
            },
        )
        r2l_ms = (time.perf_counter() - t0) * 1000.0
        r2l_diff = float(np.max(np.abs(pt_avg_lp - ort_avg_lp)))
        print(
            f"[r2l    ] max_avg_lp_diff={r2l_diff:.6g} | latency={r2l_ms:.1f}ms | "
            f"true_avg_lp={ort_avg_lp[0]:.4f} vs distractor_avg_lp={ort_avg_lp[1]:.4f}"
        )
        if r2l_diff > tol:
            raise AssertionError(f"r2l_scorer avg_log_prob diff {r2l_diff:.6g} exceeds tolerance {tol}")
        if not (ort_avg_lp[0] > ort_avg_lp[1]):
            raise AssertionError("R2L scorer did not rank true candidate above distractor!")

    print("ALL PARITY CHECKS PASSED!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PyTorch vs ONNX parity for an exported InkTeX bundle.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()
    verify_parity(args.checkpoint, args.bundle, tol=args.tol)


if __name__ == "__main__":
    main()
