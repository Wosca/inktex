"""Export an InkTeX (bimodal-checkpoint-v1) .pt checkpoint to split KV-cached ONNX graphs.

Usage:
    python export_onnx.py --checkpoint path/to/best.pt --output path/to/model-bundle
    python export_onnx.py --checkpoint path/to/best.pt --output path/to/model-bundle --quantize int8

Outputs inside <output>/:
    vision_encoder.onnx   (images [1,2,128,W], image_widths [1]) -> (vision_memory, vision_mask)
    stroke_encoder.onnx   (stroke_features [1,N,10], stroke_padding_mask [1,N]) -> (stroke_memory, stroke_mask)
    memory_proj.onnx      (memory [1,M,512]) -> (mk, mv)
    decoder_step.onnx     (token, past_k, past_v, mk, mv, memory_mask, coverage, prev_weights) ->
                          (logits, new_k, new_v, new_coverage, new_prev_weights)
    manifest.json         Model config, 907-token vocabulary, preprocessing config & SHA-256 checksums

All masks use uint8 (0 = valid, 1 = padded) so Apple's Objective-C / Swift ONNX Runtime bindings
(which expose UInt8 but not Boolean in ORTTensorElementDataType) work natively with zero glue.
"""
from __future__ import annotations

import argparse
import hashlib
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

from inktex_model.bimodal import BimodalMathModel, BimodalModelConfig, sinusoidal_2d
from inktex_model.preprocessing import PreprocessingConfig
from inktex_model.relative_position import alibi_slopes
from inktex_model.tokenizer import LatexTokenizer

NEG_MASK_VALUE = -1e9


def _project_qkv(attention: nn.MultiheadAttention, values: torch.Tensor, part: int) -> torch.Tensor:
    width = attention.embed_dim
    weight = attention.in_proj_weight[part * width : (part + 1) * width]
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[part * width : (part + 1) * width]
    projected = F.linear(values, weight, bias)
    return projected.reshape(projected.shape[0], projected.shape[1], attention.num_heads, width // attention.num_heads).transpose(1, 2)


class VisionEncoderExport(nn.Module):
    def __init__(self, model: BimodalMathModel):
        super().__init__()
        self.backbone = model.vision_encoder.backbone
        self.projection = model.vision_encoder.projection
        self.norm = model.vision_encoder.norm
        self.vision_norm = model.vision_norm
        self.vision_modality_emb = model.vision_modality_emb
        self.d_model = model.vision_encoder.d_model
        self.stride = model.vision_encoder.stride

    def forward(self, images: torch.Tensor, image_widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)[-1]
        batch, _, height, width = features.shape
        values = self.norm(self.projection(features.permute(0, 2, 3, 1)))
        values = values + sinusoidal_2d(height, width, self.d_model, values.device).to(values.dtype)
        tokens = values.transpose(1, 2).flatten(1, 2)
        memory = self.vision_norm(tokens) + self.vision_modality_emb
        valid = torch.div(image_widths.to(torch.long) + self.stride - 1, self.stride, rounding_mode="floor").clamp(1, width)
        mask = (torch.arange(width, device=tokens.device)[None] >= valid[:, None]).repeat_interleave(height, 1)
        return memory, mask.to(torch.uint8)


class StrokeEncoderExport(nn.Module):
    def __init__(self, model: BimodalMathModel):
        super().__init__()
        self.stroke_encoder = model.stroke_encoder
        self.stroke_norm = model.stroke_norm
        self.stroke_modality_emb = model.stroke_modality_emb

    def forward(self, stroke_features: torch.Tensor, stroke_padding_mask_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = stroke_padding_mask_u8.ne(0)
        memory = self.stroke_encoder(stroke_features, padding_mask)
        mask = self.stroke_encoder.memory_padding_mask(stroke_features, padding_mask, memory)
        memory = self.stroke_norm(memory) + self.stroke_modality_emb
        return memory, mask.to(torch.uint8)


class MemoryProjExport(nn.Module):
    def __init__(self, model: BimodalMathModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mks = []
        mvs = []
        for layer in self.decoder.transformer.layers:
            mks.append(_project_qkv(layer.multihead_attn, memory, 1))
            mvs.append(_project_qkv(layer.multihead_attn, memory, 2))
        return torch.stack(mks, dim=0), torch.stack(mvs, dim=0)


class DecoderStepExport(nn.Module):
    """Single KV-cached autoregressive decoder step with ALiBi and visibility-gated coverage.

    Slot 0 of ``past_k`` / ``past_v`` is always a dummy sentinel masked with ``-1e9`` in self-attention.
    This eliminates any ``T = 0`` empty-tensor edge case and avoids duplicating decoder weights across
    separate init/step graphs:
      - Step 0: pass ``token = [[bos_id]]``, ``past_k/v = zeros([L, 1, H, 1, D])``,
        ``coverage = zeros([1, H, 1, M])``, ``prev_weights = zeros([1, H, 1, M])``.
      - Step t >= 1: pass previous predicted token and the returned ``new_k/v``, ``new_coverage``,
        ``new_prev_weights``.
    """

    def __init__(self, model: BimodalMathModel):
        super().__init__()
        self.decoder = model.decoder
        num_heads = self.decoder.transformer.layers[0].self_attn.num_heads
        self.register_buffer("alibi_slopes", alibi_slopes(num_heads), persistent=False)
        self.register_buffer(
            "eos_one_hot",
            F.one_hot(torch.tensor(self.decoder.eos_id, dtype=torch.long), model.config.vocab_size).to(torch.float32),
            persistent=False,
        )

    def _attend(
        self,
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = _project_qkv(attention, query, 0)
        score = (q @ keys.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if bias is not None:
            score = score + bias
        weights = score.softmax(-1)
        attended = (weights @ values).transpose(1, 2).reshape(query.shape[0], 1, -1)
        return attention.out_proj(attended), weights

    def forward(
        self,
        token: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        mk: torch.Tensor,
        mv: torch.Tensor,
        memory_mask_u8: torch.Tensor,
        coverage: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.decoder
        memory_mask = memory_mask_u8.ne(0)
        values = decoder.token_embedding(token)

        if decoder.coverage:
            visible = decoder.visible_tokens[token].to(coverage.dtype)[:, None, :, None]
            coverage = coverage + prev_weights * visible

        total_len = past_k.shape[3] + 1
        positions = torch.arange(total_len, device=values.device, dtype=values.dtype)
        distance = (total_len - 1) - positions
        self_bias = -self.alibi_slopes.to(values.dtype)[None, :, None, None] * distance[None, None, None, :]
        dummy_slot = torch.arange(total_len, device=values.device).eq(0)[None, None, None, :]
        self_bias = self_bias.masked_fill(dummy_slot, NEG_MASK_VALUE)

        mask_bias = torch.zeros_like(memory_mask, dtype=values.dtype).masked_fill(memory_mask, NEG_MASK_VALUE)[:, None, None, :]

        new_ks = []
        new_vs = []
        new_prev_weights = prev_weights
        num_layers = len(decoder.transformer.layers)

        for index, layer in enumerate(decoder.transformer.layers):
            q = layer.norm1(values)
            key = _project_qkv(layer.self_attn, q, 1)
            val = _project_qkv(layer.self_attn, q, 2)
            full_k = torch.cat((past_k[index], key), dim=2)
            full_v = torch.cat((past_v[index], val), dim=2)
            new_ks.append(full_k)
            new_vs.append(full_v)

            attended, _ = self._attend(layer.self_attn, q, full_k, full_v, self_bias)
            values = values + attended

            q = layer.norm2(values)
            last = index == num_layers - 1
            attended, weights = self._attend(layer.multihead_attn, q, mk[index], mv[index], mask_bias)
            if last:
                weights = weights.masked_fill(memory_mask[:, None, None, :], 0.0)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
                new_prev_weights = weights
                if decoder.coverage:
                    coverage_bias = -F.softplus(decoder.coverage_strength) * coverage
                    attended, _ = self._attend(layer.multihead_attn, q, mk[index], mv[index], coverage_bias + mask_bias)
            values = values + attended

            q = layer.norm3(values)
            values = values + layer.linear2(layer.activation(layer.linear1(q)))

        logits = decoder.output(decoder.transformer.norm(values))[:, 0, :]
        if decoder.coverage:
            valid = (~memory_mask).to(values.dtype)
            uncovered = (torch.exp(-4.0 * coverage.mean(1)) * valid[:, None, :]).sum(-1) / valid.sum(-1, keepdim=True).clamp_min(1.0)
            logits = logits - F.softplus(decoder.eos_coverage_strength) * uncovered * self.eos_one_hot.to(logits.dtype)[None, :]

        return logits, torch.stack(new_ks, dim=0), torch.stack(new_vs, dim=0), coverage, new_prev_weights


class R2LScorerExport(nn.Module):
    """Batched teacher-forced right-to-left (R2L) beam reranker.

    Inputs:
        r2l_tokens:       int64 [B, T]  -> [BOS, y_L, ..., y_1, EOS, PAD...]
        r2l_padding_mask: uint8 [B, T]  -> 0 = valid, 1 = padded
        memory:           float32 [1, M, D] (broadcast to [B, M, D] inside the graph)
        memory_mask:      uint8 [1, M]  -> 0 = valid, 1 = padded

    Outputs:
        seq_log_prob:     float32 [B]   -> sum of target token log-probabilities
        avg_log_prob:     float32 [B]   -> length-normalized mean log-probability
    """

    def __init__(self, model: BimodalMathModel):
        super().__init__()
        if model.r2l_decoder is None:
            raise ValueError("Model has no r2l_decoder")
        self.decoder = model.r2l_decoder
        num_heads = self.decoder.transformer.layers[0].self_attn.num_heads
        self.register_buffer("alibi_slopes", alibi_slopes(num_heads), persistent=False)

    def _mha(
        self,
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key_src: torch.Tensor,
        val_src: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        q = _project_qkv(attention, query, 0)
        k = _project_qkv(attention, key_src, 1)
        v = _project_qkv(attention, val_src, 2)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias
        weights = score.softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return attention.out_proj(attended)

    def forward(
        self,
        r2l_tokens: torch.Tensor,
        r2l_padding_mask_u8: torch.Tensor,
        memory: torch.Tensor,
        memory_mask_u8: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = r2l_tokens.shape[0]
        inp = r2l_tokens[:, :-1]
        tgt = r2l_tokens[:, 1:]
        inp_pad = r2l_padding_mask_u8[:, :-1].ne(0)
        tgt_valid = r2l_padding_mask_u8[:, 1:].eq(0)

        length = inp.shape[1]
        values = self.decoder.token_embedding(inp)

        # Causal ALiBi self-attention bias [B, H, L, L]
        positions = torch.arange(length, device=values.device)
        dist = (positions[:, None] - positions[None, :]).abs().to(values.dtype)
        causal = positions[None, :] > positions[:, None]
        self_bias = -self.alibi_slopes.to(values.dtype)[:, None, None] * dist[None, :, :]
        self_bias = self_bias.masked_fill(causal[None, :, :], NEG_MASK_VALUE)
        self_bias = self_bias.unsqueeze(0).expand(batch, -1, -1, -1)
        # Mask padded keys (never mask position 0 to keep softmax finite on dummy rows)
        key_pad_bias = torch.zeros_like(inp_pad, dtype=values.dtype).masked_fill(inp_pad, NEG_MASK_VALUE)
        self_bias = self_bias + key_pad_bias[:, None, None, :]

        mem = memory.expand(batch, -1, -1)
        mem_pad = memory_mask_u8.ne(0)
        mem_bias = torch.zeros_like(mem_pad, dtype=values.dtype).masked_fill(mem_pad, NEG_MASK_VALUE)[:, None, None, :]
        mem_bias = mem_bias.expand(batch, -1, -1, -1)

        for layer in self.decoder.transformer.layers:
            q = layer.norm1(values)
            values = values + self._mha(layer.self_attn, q, q, q, self_bias)
            q = layer.norm2(values)
            values = values + self._mha(layer.multihead_attn, q, mem, mem, mem_bias)
            q = layer.norm3(values)
            values = values + layer.linear2(layer.activation(layer.linear1(q)))

        logits = self.decoder.output(self.decoder.transformer.norm(values))
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        valid_f = tgt_valid.to(logits.dtype)
        seq_log_prob = (token_lp * valid_f).sum(dim=-1)
        avg_log_prob = seq_log_prob / valid_f.sum(dim=-1).clamp_min(1.0)
        return seq_log_prob, avg_log_prob


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_bimodal_checkpoint(checkpoint_path: Path) -> tuple[BimodalMathModel, LatexTokenizer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "bimodal-checkpoint-v1":
        raise ValueError(f"Expected bimodal-checkpoint-v1, got {payload.get('format')!r}")
    config_dict = {**payload["model_config"], "vision_pretrained": False}
    config = BimodalModelConfig.from_dict(config_dict)
    model = BimodalMathModel(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    tokenizer = LatexTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload


def _quantize_int8(src: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = src.with_suffix(".fp32.onnx")
    src.replace(tmp)
    try:
        quantize_dynamic(
            model_input=str(tmp),
            model_output=str(src),
            weight_type=QuantType.QUInt8,
        )
    finally:
        tmp.unlink(missing_ok=True)


def export_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    quantize: str = "none",
    model_name: str = "InkTeX-75M",
) -> Path:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print(f"Loading checkpoint: {checkpoint_path}")
    model, tokenizer, payload = load_bimodal_checkpoint(checkpoint_path)
    cfg = model.config
    num_layers = cfg.decoder_layers
    num_heads = cfg.attention_heads
    head_dim = cfg.d_model // num_heads

    # Dummy tensors for tracing
    dummy_images = torch.zeros((1, cfg.vision_in_chans, cfg.vision_height, 128), dtype=torch.float32)
    dummy_widths = torch.tensor([96], dtype=torch.long)

    dummy_strokes = torch.zeros((1, 24, cfg.input_dim), dtype=torch.float32)
    dummy_strokes[0, 0, 7] = 1.0
    dummy_strokes[0, 11, 8] = 1.0
    dummy_strokes[0, 12, 7] = 1.0
    dummy_strokes[0, 23, 8] = 1.0
    dummy_stroke_mask = torch.zeros((1, 24), dtype=torch.uint8)

    dummy_memory = torch.zeros((1, 48, cfg.d_model), dtype=torch.float32)
    dummy_memory_mask = torch.zeros((1, 48), dtype=torch.uint8)

    dummy_token = torch.tensor([[tokenizer.bos_id]], dtype=torch.long)
    dummy_past_k = torch.zeros((num_layers, 1, num_heads, 3, head_dim), dtype=torch.float32)
    dummy_past_v = torch.zeros((num_layers, 1, num_heads, 3, head_dim), dtype=torch.float32)
    dummy_mk = torch.zeros((num_layers, 1, num_heads, 48, head_dim), dtype=torch.float32)
    dummy_mv = torch.zeros((num_layers, 1, num_heads, 48, head_dim), dtype=torch.float32)
    dummy_coverage = torch.zeros((1, num_heads, 1, 48), dtype=torch.float32)
    dummy_prev_weights = torch.zeros((1, num_heads, 1, 48), dtype=torch.float32)

    width_dim = torch.export.Dim("strip_width", min=64, max=cfg.vision_max_width)
    points_dim = torch.export.Dim("num_points", min=2, max=cfg.max_points)
    mem_dim = torch.export.Dim("mem_len", min=2, max=cfg.max_points)
    past_dim = torch.export.Dim("past_len", min=1, max=cfg.max_tokens)

    vision_path = output_dir / "vision_encoder.onnx"
    stroke_path = output_dir / "stroke_encoder.onnx"
    memproj_path = output_dir / "memory_proj.onnx"
    decoder_path = output_dir / "decoder_step.onnx"
    r2l_path = output_dir / "r2l_scorer.onnx"

    print("Exporting vision_encoder.onnx ...")
    torch.onnx.export(
        VisionEncoderExport(model).eval(),
        (dummy_images, dummy_widths),
        vision_path,
        input_names=["images", "image_widths"],
        output_names=["vision_memory", "vision_mask"],
        dynamic_shapes=({3: width_dim}, {}),
        dynamo=True,
        external_data=False,
    )

    print("Exporting stroke_encoder.onnx ...")
    torch.onnx.export(
        StrokeEncoderExport(model).eval(),
        (dummy_strokes, dummy_stroke_mask),
        stroke_path,
        input_names=["stroke_features", "stroke_padding_mask"],
        output_names=["stroke_memory", "stroke_mask"],
        dynamic_shapes=({1: points_dim}, {1: points_dim}),
        dynamo=True,
        external_data=False,
    )

    print("Exporting memory_proj.onnx ...")
    torch.onnx.export(
        MemoryProjExport(model).eval(),
        (dummy_memory,),
        memproj_path,
        input_names=["memory"],
        output_names=["mk", "mv"],
        dynamic_shapes=({1: mem_dim},),
        dynamo=True,
        external_data=False,
    )

    print("Exporting decoder_step.onnx ...")
    torch.onnx.export(
        DecoderStepExport(model).eval(),
        (dummy_token, dummy_past_k, dummy_past_v, dummy_mk, dummy_mv, dummy_memory_mask, dummy_coverage, dummy_prev_weights),
        decoder_path,
        input_names=["token", "past_k", "past_v", "mk", "mv", "memory_mask", "coverage", "prev_weights"],
        output_names=["logits", "new_k", "new_v", "new_coverage", "new_prev_weights"],
        dynamic_shapes=(
            {},
            {3: past_dim},
            {3: past_dim},
            {3: mem_dim},
            {3: mem_dim},
            {1: mem_dim},
            {3: mem_dim},
            {3: mem_dim},
        ),
        dynamo=True,
        external_data=False,
    )

    has_r2l = model.r2l_decoder is not None
    if has_r2l:
        print("Exporting r2l_scorer.onnx ...")
        beam_dim = torch.export.Dim("beam_batch", min=1, max=16)
        r2l_len_dim = torch.export.Dim("r2l_len", min=2, max=cfg.max_tokens)
        dummy_r2l_tokens = torch.tensor([[tokenizer.bos_id, 6, 7, tokenizer.eos_id],
                                         [tokenizer.bos_id, 6, tokenizer.eos_id, tokenizer.pad_id]], dtype=torch.long)
        dummy_r2l_mask = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.uint8)
        torch.onnx.export(
            R2LScorerExport(model).eval(),
            (dummy_r2l_tokens, dummy_r2l_mask, dummy_memory, dummy_memory_mask),
            r2l_path,
            input_names=["r2l_tokens", "r2l_padding_mask", "memory", "memory_mask"],
            output_names=["seq_log_prob", "avg_log_prob"],
            dynamic_shapes=(
                {0: beam_dim, 1: r2l_len_dim},
                {0: beam_dim, 1: r2l_len_dim},
                {1: mem_dim},
                {1: mem_dim},
            ),
            dynamo=True,
            external_data=False,
        )

    if quantize == "int8":
        quant_targets = [stroke_path, memproj_path, decoder_path]
        if has_r2l:
            quant_targets.append(r2l_path)
        print("Applying dynamic INT8 weight quantization to transformer graphs ...")
        for p in quant_targets:
            _quantize_int8(p)
        print("Keeping vision_encoder.onnx in FP32 for CoreML EP / ConvNeXt compatibility.")
    elif quantize != "none":
        raise ValueError(f"Unsupported --quantize mode: {quantize}")

    file_entries = [
        ("vision_encoder", vision_path),
        ("stroke_encoder", stroke_path),
        ("memory_proj", memproj_path),
        ("decoder_step", decoder_path),
    ]
    if has_r2l:
        file_entries.append(("r2l_scorer", r2l_path))

    files_meta = {}
    total_bytes = 0
    for key, p in file_entries:
        size = p.stat().st_size
        total_bytes += size
        files_meta[key] = {
            "file": p.name,
            "bytes": size,
            "sha256": sha256_file(p),
        }
        print(f"  {p.name}: {size / (1024 * 1024):.2f} MB")

    manifest = {
        "format": "inktex-onnx-bundle-v1",
        "model_name": model_name,
        "preset": cfg.preset,
        "stage": payload.get("stage"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "saved_at": payload.get("saved_at"),
        "metrics": payload.get("metrics"),
        "quantization": quantize,
        "has_r2l": has_r2l,
        "r2l_decoder_layers": cfg.r2l_decoder_layers if has_r2l else 0,
        "d_model": cfg.d_model,
        "decoder_layers": num_layers,
        "attention_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": cfg.vocab_size,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id,
        "unk_id": tokenizer.unk_id,
        "max_points": cfg.max_points,
        "max_tokens": cfg.max_tokens,
        "input_dim": cfg.input_dim,
        "vision": {
            "height": cfg.vision_height,
            "min_width": 64,
            "max_width": cfg.vision_max_width,
            "padding": 8,
            "width_multiple": 16,
            "channels": cfg.vision_in_chans,
        },
        "preprocessing": PreprocessingConfig().to_dict(),
        "tokens": tokenizer.tokens,
        "files": files_meta,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"Export complete in {elapsed:.1f}s -> {output_dir} ({total_bytes / (1024 * 1024):.2f} MB total)")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export InkTeX bimodal .pt checkpoint to split ONNX bundle.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to bimodal-checkpoint-v1 best.pt")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for ONNX bundle + manifest.json")
    parser.add_argument("--quantize", choices=("none", "int8"), default="none", help="Optional dynamic INT8 quantization")
    parser.add_argument("--name", default="InkTeX-75M", help="Human-readable model name recorded in manifest.json")
    args = parser.parse_args()
    export_bundle(args.checkpoint, args.output, quantize=args.quantize, model_name=args.name)


if __name__ == "__main__":
    main()
