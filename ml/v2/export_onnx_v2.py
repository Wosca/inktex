"""InkTeX v2 split ONNX export, manifest & LoRA adapter packaging, and PyTorch <-> ONNX parity verifier.

Zero .item() calls, zero data-dependent shapes, static/bucketed shapes ready for Apple ANE/CoreML!
Verifies ALL outputs across all 5 graphs PLUS multi-step incremental KV decoding (cache_pos = 0,1,2,3),
prefix verification at cache_pos > 0, and single-pass batched bidirectional R2L scoring.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .ink_preprocess import (
    CHUNK_BUCKETS,
    CHUNK_META_DIM,
    CHUNK_SIZE,
    POINT_FEATURE_DIM,
    STYLE_MAX_STROKES,
    STYLE_POINTS,
    STYLE_POINT_DIM,
    InkPreprocessV2Config,
)
from .model import InkTeXV2Config, InkTeXV2Model, alibi_slopes, get_v2_config
from .vision_raster import (
    PADDING_V2,
    STRIP_HEIGHT_V2,
    TARGET_GLYPH_PX_V2,
    WIDTH_BUCKETS_V2,
    WIDTH_MULTIPLE_V2,
)

NEG_MASK_VALUE = -1e4
STATIC_KV_LEN = 256
MANIFEST_FORMAT_V2 = "inktex-onnx-v2"
LORA_ADAPTER_FORMAT_V2 = "inktex-lora-v2"


class WriterStyleExport(nn.Module):
    def __init__(self, model: InkTeXV2Model):
        super().__init__()
        self.style_encoder = model.style_encoder

    def forward(
        self,
        style_points: torch.Tensor,
        style_aspect: torch.Tensor,
        style_mask_u8: torch.Tensor,
    ) -> torch.Tensor:
        return self.style_encoder(style_points, style_aspect, style_mask_u8.ne(0))


class StrokeEncoderV2Export(nn.Module):
    def __init__(self, model: InkTeXV2Model, dynamic_lora: bool = False):
        super().__init__()
        self.stroke_encoder = model.stroke_encoder
        self.fusion = model.fusion
        self.dynamic_lora = dynamic_lora
        self.lora_layers = tuple(model.config.lora_layers)

    def forward(
        self,
        chunk_points: torch.Tensor,
        chunk_point_mask_u8: torch.Tensor,
        chunk_meta: torch.Tensor,
        chunk_mask_u8: torch.Tensor,
        writer_style: torch.Tensor,
        lora_a: torch.Tensor | None = None,
        lora_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ext_lora = None
        if self.dynamic_lora and lora_a is not None and lora_b is not None:
            ext_lora = {}
            for slot, layer_idx in enumerate(self.lora_layers):
                ext_lora[layer_idx] = {
                    "q": (lora_a[slot, 0], lora_b[slot, 0]),
                    "v": (lora_a[slot, 1], lora_b[slot, 1]),
                    "o": (lora_a[slot, 2], lora_b[slot, 2]),
                }
        raw = self.stroke_encoder(
            chunk_points,
            chunk_meta,
            chunk_mask_u8.ne(0),
            writer_style=writer_style,
            external_lora=ext_lora,
            chunk_point_mask=chunk_point_mask_u8.ne(0),
            use_explicit_attn=True,
        )
        return self.fusion.project_stroke(raw)


class VisionEncoderV2Export(nn.Module):
    def __init__(self, model: InkTeXV2Model):
        super().__init__()
        self.vision_encoder = model.vision_encoder
        self.fusion = model.fusion

    def forward(self, images: torch.Tensor, image_widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok, mask = self.vision_encoder(images, image_widths)
        return self.fusion.project_vision(tok), mask.to(torch.uint8)


class FusionMemProjV2Export(nn.Module):
    def __init__(self, model: InkTeXV2Model):
        super().__init__()
        self.fusion = model.fusion
        self.decoder = model.decoder

    def forward(
        self,
        stroke_mem: torch.Tensor,
        stroke_mask_u8: torch.Tensor,
        vision_mem: torch.Tensor,
        vision_mask_u8: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, mask = self.fusion.fuse(
            stroke_mem,
            stroke_mask_u8.ne(0),
            vision_mem,
            vision_mask_u8.ne(0),
            use_explicit_attn=True,
        )
        mks, mvs = [], []
        for layer in self.decoder.layers:
            mk, mv = layer.project_memory(memory)
            mks.append(mk)
            mvs.append(mv)
        return memory, mask.to(torch.uint8), torch.stack(mks, dim=0), torch.stack(mvs, dim=0)


class DecoderStepV2Export(nn.Module):
    """Unified static-KV-cache shared bidirectional decoder (L2R + R2L + prefix verify)."""

    def __init__(self, model: InkTeXV2Model, max_kv: int = STATIC_KV_LEN):
        super().__init__()
        self.decoder = model.decoder
        self.max_kv = max_kv
        self.register_buffer("alibi_slopes", alibi_slopes(model.config.decoder_heads), persistent=False)

    def forward(
        self,
        token: torch.Tensor,           # [B, S] int64
        cache_pos: torch.Tensor,       # [1] int64
        past_k: torch.Tensor,          # [L, B, H, max_kv, Dh]
        past_v: torch.Tensor,          # [L, B, H, max_kv, Dh]
        mk: torch.Tensor,              # [L, 1, H, M, Dh]
        mv: torch.Tensor,              # [L, 1, H, M, Dh]
        memory_mask_u8: torch.Tensor,  # [1, M] uint8
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, step_len = token.shape
        num_heads = past_k.shape[2]
        head_dim = past_k.shape[4]
        mem_len = memory_mask_u8.shape[1]

        x = self.decoder.token_embedding(token)

        q_pos = (cache_pos[0] + torch.arange(step_len, device=x.device, dtype=torch.long)).clamp(0, self.max_kv - 1)
        k_pos = torch.arange(self.max_kv, device=x.device, dtype=torch.long)
        scatter_idx = q_pos.view(1, 1, step_len, 1).expand(bsz, num_heads, step_len, head_dim)

        dist = (q_pos[:, None] - k_pos[None, :]).clamp_min(0).to(x.dtype)
        self_bias = -self.alibi_slopes.to(x.dtype)[None, :, None, None] * dist[None, None, :, :]
        invalid_key = k_pos[None, :] > q_pos[:, None]
        self_bias = self_bias.masked_fill(invalid_key[None, None, :, :], NEG_MASK_VALUE)

        mem_mask = memory_mask_u8.ne(0)
        all_pad = mem_mask.all(dim=1, keepdim=True)
        first = torch.arange(mem_len, device=x.device).eq(0).unsqueeze(0)
        safe_mem = mem_mask & ~(all_pad & first)
        mem_bias = torch.zeros((1, 1, 1, mem_len), dtype=x.dtype, device=x.device)
        mem_bias = mem_bias.masked_fill(safe_mem[:, None, None, :], NEG_MASK_VALUE)

        new_ks, new_vs = [], []
        for idx, layer in enumerate(self.decoder.layers):
            h = layer.norm1(x)
            qkv = layer.self_qkv(h).reshape(bsz, step_len, 3, num_heads, head_dim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)

            full_k = past_k[idx].scatter(2, scatter_idx, k)
            full_v = past_v[idx].scatter(2, scatter_idx, v)
            new_ks.append(full_k)
            new_vs.append(full_v)

            s_scores = (q @ full_k.transpose(-1, -2)) / math.sqrt(head_dim) + self_bias
            s_attn = (s_scores.softmax(dim=-1) @ full_v).transpose(1, 2).reshape(bsz, step_len, -1)
            x = x + layer.self_out(s_attn)

            h2 = layer.norm2(x)
            cq = layer.cross_q(h2).reshape(bsz, step_len, num_heads, head_dim).transpose(1, 2)
            mk_b = mk[idx].expand(bsz, -1, -1, -1)
            mv_b = mv[idx].expand(bsz, -1, -1, -1)
            c_scores = (cq @ mk_b.transpose(-1, -2)) / math.sqrt(head_dim) + mem_bias
            c_attn = (c_scores.softmax(dim=-1) @ mv_b).transpose(1, 2).reshape(bsz, step_len, -1)
            x = x + layer.cross_out(c_attn)

            h3 = layer.norm3(x)
            x = x + layer.fc2(torch.nn.functional.gelu(layer.fc1(h3)))

        logits = self.decoder.output(self.decoder.norm(x))
        return logits, torch.stack(new_ks, dim=0), torch.stack(new_vs, dim=0)


def export_and_verify_v2(
    model: InkTeXV2Model,
    output_dir: Path,
    *,
    dynamic_lora: bool = True,
    static_kv_len: int = STATIC_KV_LEN,
    chunk_bucket: int = 128,
    vision_width: int = 256,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Export all 5 InkTeX v2 ONNX graphs and strictly assert PyTorch <-> ORT parity
    across all outputs AND multi-step incremental KV decoding (cache_pos = 0, 1, 2, 3)."""
    import onnxruntime as ort

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    cfg = model.config

    # 1. Writer style encoder
    d_sp = torch.randn(1, 20, 16, cfg.style_point_dim)
    d_sa = torch.randn(1, 20, 4)
    d_sm = torch.zeros(1, 20, dtype=torch.uint8)
    d_sm[0, 8:] = 1
    style_path = output_dir / "writer_style_v2.onnx"
    torch.onnx.export(
        WriterStyleExport(model).eval(),
        (d_sp, d_sa, d_sm),
        style_path,
        input_names=["style_points", "style_aspect", "style_mask"],
        output_names=["writer_style"],
        dynamo=True,
        external_data=False,
    )

    # 2. Stroke encoder
    d_cp = torch.randn(1, chunk_bucket, cfg.chunk_size, cfg.point_dim)
    d_cpm = torch.zeros(1, chunk_bucket, cfg.chunk_size, dtype=torch.uint8)
    d_cpm[0, chunk_bucket // 2 - 1, 5:] = 1
    d_cpm[0, chunk_bucket // 2 :, :] = 1
    d_cm = torch.randn(1, chunk_bucket, cfg.chunk_meta_dim)
    d_ck = torch.zeros(1, chunk_bucket, dtype=torch.uint8)
    d_ck[0, chunk_bucket // 2 :] = 1
    d_ws = torch.zeros(1, cfg.style_dim)
    stroke_path = output_dir / "stroke_encoder_v2.onnx"
    if dynamic_lora:
        d_la = torch.randn(len(cfg.lora_layers), 3, cfg.lora_rank, cfg.stroke_dim) * 0.01
        d_lb = torch.randn(len(cfg.lora_layers), 3, cfg.stroke_dim, cfg.lora_rank) * 0.01
        torch.onnx.export(
            StrokeEncoderV2Export(model, dynamic_lora=True).eval(),
            (d_cp, d_cpm, d_cm, d_ck, d_ws, d_la, d_lb),
            stroke_path,
            input_names=["chunk_points", "chunk_point_mask", "chunk_meta", "chunk_mask", "writer_style", "lora_a", "lora_b"],
            output_names=["stroke_memory"],
            dynamo=True,
            external_data=False,
        )
    else:
        torch.onnx.export(
            StrokeEncoderV2Export(model, dynamic_lora=False).eval(),
            (d_cp, d_cpm, d_cm, d_ck, d_ws),
            stroke_path,
            input_names=["chunk_points", "chunk_point_mask", "chunk_meta", "chunk_mask", "writer_style"],
            output_names=["stroke_memory"],
            dynamo=True,
            external_data=False,
        )

    # 3. Vision encoder
    d_img = torch.randn(1, cfg.vision_in_chans, cfg.vision_height, vision_width)
    d_iw = torch.tensor([vision_width - 32], dtype=torch.long)
    vision_path = output_dir / "vision_encoder_v2.onnx"
    torch.onnx.export(
        VisionEncoderV2Export(model).eval(),
        (d_img, d_iw),
        vision_path,
        input_names=["images", "image_widths"],
        output_names=["vision_memory", "vision_mask"],
        dynamo=True,
        external_data=False,
    )

    # 4. Fusion + Memory projection
    vis_tokens = (cfg.vision_height // 8) * (vision_width // 16)
    d_smem = torch.randn(1, chunk_bucket, cfg.d_model)
    d_vmem = torch.randn(1, vis_tokens, cfg.d_model)
    d_vmask = torch.zeros(1, vis_tokens, dtype=torch.uint8)
    d_vmask[0, vis_tokens - 48 :] = 1
    fusion_path = output_dir / "fusion_memproj_v2.onnx"
    torch.onnx.export(
        FusionMemProjV2Export(model).eval(),
        (d_smem, d_ck, d_vmem, d_vmask),
        fusion_path,
        input_names=["stroke_memory", "stroke_mask", "vision_memory", "vision_mask"],
        output_names=["memory", "memory_mask", "mk", "mv"],
        dynamo=True,
        external_data=False,
    )

    # 5. Unified static-KV shared bidirectional decoder step
    mem_len = chunk_bucket + vis_tokens
    head_dim = cfg.d_model // cfg.decoder_heads
    d_tok = torch.tensor([[cfg.bos_l2r_id, 5, 6, 7, 8], [cfg.bos_r2l_id, 8, 7, 6, 5]], dtype=torch.long)
    d_pos = torch.tensor([0], dtype=torch.long)
    d_pk = torch.zeros(cfg.decoder_layers, 2, cfg.decoder_heads, static_kv_len, head_dim)
    d_pv = torch.zeros(cfg.decoder_layers, 2, cfg.decoder_heads, static_kv_len, head_dim)
    d_mk = torch.randn(cfg.decoder_layers, 1, cfg.decoder_heads, mem_len, head_dim)
    d_mv = torch.randn(cfg.decoder_layers, 1, cfg.decoder_heads, mem_len, head_dim)
    d_mm = torch.zeros(1, mem_len, dtype=torch.uint8)
    dec_batch_dim = torch.export.Dim("dec_batch", min=1, max=8)
    step_len_dim = torch.export.Dim("step_len", min=1, max=static_kv_len - 1)
    decoder_path = output_dir / "decoder_step_v2.onnx"
    torch.onnx.export(
        DecoderStepV2Export(model, max_kv=static_kv_len).eval(),
        (d_tok[:, :3], d_pos, d_pk, d_pv, d_mk, d_mv, d_mm),
        decoder_path,
        input_names=["token", "cache_pos", "past_k", "past_v", "mk", "mv", "memory_mask"],
        output_names=["logits", "new_k", "new_v"],
        dynamic_shapes=(
            {0: dec_batch_dim, 1: step_len_dim},
            {},
            {1: dec_batch_dim},
            {1: dec_batch_dim},
            {},
            {},
            {},
        ),
        dynamo=True,
        external_data=False,
    )

    # Strict parity verification across ALL outputs + multi-step incremental KV decoding!
    max_diffs: dict[str, float] = {}
    with torch.no_grad():
        s_pt = WriterStyleExport(model)(d_sp, d_sa, d_sm).numpy()
        sess = ort.InferenceSession(str(style_path), providers=["CPUExecutionProvider"])
        s_ort = sess.run(None, {"style_points": d_sp.numpy(), "style_aspect": d_sa.numpy(), "style_mask": d_sm.numpy()})[0]
        max_diffs["writer_style"] = float(np.max(np.abs(s_pt - s_ort)))

        if dynamic_lora:
            st_pt = StrokeEncoderV2Export(model, dynamic_lora=True)(d_cp, d_cpm, d_cm, d_ck, d_ws, d_la, d_lb).numpy()
            sess = ort.InferenceSession(str(stroke_path), providers=["CPUExecutionProvider"])
            st_ort = sess.run(None, {
                "chunk_points": d_cp.numpy(), "chunk_point_mask": d_cpm.numpy(),
                "chunk_meta": d_cm.numpy(), "chunk_mask": d_ck.numpy(),
                "writer_style": d_ws.numpy(), "lora_a": d_la.numpy(), "lora_b": d_lb.numpy(),
            })[0]
        else:
            st_pt = StrokeEncoderV2Export(model, dynamic_lora=False)(d_cp, d_cpm, d_cm, d_ck, d_ws).numpy()
            sess = ort.InferenceSession(str(stroke_path), providers=["CPUExecutionProvider"])
            st_ort = sess.run(None, {
                "chunk_points": d_cp.numpy(), "chunk_point_mask": d_cpm.numpy(),
                "chunk_meta": d_cm.numpy(), "chunk_mask": d_ck.numpy(),
                "writer_style": d_ws.numpy(),
            })[0]
        max_diffs["stroke_encoder"] = float(np.max(np.abs(st_pt - st_ort)))

        v_pt, vm_pt = VisionEncoderV2Export(model)(d_img, d_iw)
        sess = ort.InferenceSession(str(vision_path), providers=["CPUExecutionProvider"])
        v_ort, vm_ort = sess.run(None, {"images": d_img.numpy(), "image_widths": d_iw.numpy()})
        max_diffs["vision_encoder_mem"] = float(np.max(np.abs(v_pt.numpy() - v_ort)))
        max_diffs["vision_encoder_mask"] = float(np.max(np.abs(vm_pt.numpy().astype(np.int32) - vm_ort.astype(np.int32))))

        f_mem_pt, f_mask_pt, f_mk_pt, f_mv_pt = FusionMemProjV2Export(model)(d_smem, d_ck, d_vmem, d_vmask)
        sess = ort.InferenceSession(str(fusion_path), providers=["CPUExecutionProvider"])
        f_mem_ort, f_mask_ort, f_mk_ort, f_mv_ort = sess.run(None, {
            "stroke_memory": d_smem.numpy(), "stroke_mask": d_ck.numpy(),
            "vision_memory": d_vmem.numpy(), "vision_mask": d_vmask.numpy(),
        })
        max_diffs["fusion_mem"] = float(np.max(np.abs(f_mem_pt.numpy() - f_mem_ort)))
        max_diffs["fusion_mask"] = float(np.max(np.abs(f_mask_pt.numpy().astype(np.int32) - f_mask_ort.astype(np.int32))))
        max_diffs["fusion_mk"] = float(np.max(np.abs(f_mk_pt.numpy() - f_mk_ort)))
        max_diffs["fusion_mv"] = float(np.max(np.abs(f_mv_pt.numpy() - f_mv_ort)))

        # Full-sequence PyTorch ground truth over all 5 tokens (both L2R and R2L!)
        full_pt = model.decoder(d_tok, f_mem_pt.expand(2, -1, -1), f_mask_pt.ne(0).expand(2, -1)).numpy()
        dec_sess = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

        # (A) Token-by-token incremental KV decode: cache_pos = 0, 1, 2, 3, 4
        pk = np.zeros_like(d_pk.numpy())
        pv = np.zeros_like(d_pv.numpy())
        step_logits = []
        for pos in range(d_tok.shape[1]):
            lg, pk, pv = dec_sess.run(None, {
                "token": d_tok[:, pos : pos + 1].numpy(),
                "cache_pos": np.array([pos], dtype=np.int64),
                "past_k": pk, "past_v": pv,
                "mk": f_mk_ort, "mv": f_mv_ort, "memory_mask": f_mask_ort,
            })
            step_logits.append(lg)
        inc_ort = np.concatenate(step_logits, axis=1)
        max_diffs["decoder_incremental_0_to_4"] = float(np.max(np.abs(full_pt - inc_ort)))

        # (B) Prefix block (0..2) followed by multi-token continuation at cache_pos=3 (3..4)
        pk2 = np.zeros_like(d_pk.numpy())
        pv2 = np.zeros_like(d_pv.numpy())
        lg_pre, pk2, pv2 = dec_sess.run(None, {
            "token": d_tok[:, :3].numpy(),
            "cache_pos": np.array([0], dtype=np.int64),
            "past_k": pk2, "past_v": pv2,
            "mk": f_mk_ort, "mv": f_mv_ort, "memory_mask": f_mask_ort,
        })
        lg_cont, _, _ = dec_sess.run(None, {
            "token": d_tok[:, 3:].numpy(),
            "cache_pos": np.array([3], dtype=np.int64),
            "past_k": pk2, "past_v": pv2,
            "mk": f_mk_ort, "mv": f_mv_ort, "memory_mask": f_mask_ort,
        })
        block_ort = np.concatenate([lg_pre, lg_cont], axis=1)
        max_diffs["decoder_prefix_plus_pos3"] = float(np.max(np.abs(full_pt - block_ort)))

    for k, diff in max_diffs.items():
        if diff > atol:
            raise AssertionError(f"ONNX parity check failed on {k}: diff={diff:.3e} > atol={atol:.3e}")

    return {
        "files": {
            "writer_style": str(style_path),
            "stroke_encoder": str(stroke_path),
            "vision_encoder": str(vision_path),
            "fusion_memproj": str(fusion_path),
            "decoder_step": str(decoder_path),
        },
        "max_abs_diff": max_diffs,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _convert_fp16_v2(src: Path, extra_block_ops: Sequence[str] = ()) -> None:
    """Convert an exported ONNX graph to CoreML-friendly FP16 while keeping IO in FP32
    and preserving FP32 precision on numerically sensitive reduction/norm ops."""
    import onnx
    from onnxruntime.transformers import float16

    model = onnx.load(str(src))
    block_list = list(float16.DEFAULT_OP_BLOCK_LIST) + [
        "LayerNormalization",
        "SimplifiedLayerNormalization",
        "Softmax",
        "LogSoftmax",
        "CumSum",
        "Exp",
        "Log",
        "Pow",
        "Sqrt",
        "ReduceMean",
        "ReduceSum",
        "ReduceMin",
        "ReduceMax",
        *extra_block_ops,
    ]
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=sorted(set(block_list)),
    )
    dedup_nodes = []
    seen_outputs: set[str] = set()
    seen_names: set[str] = set()
    for idx, node in enumerate(fp16_model.graph.node):
        if node.op_type == "Cast" and len(node.output) == 1 and node.output[0] in seen_outputs:
            continue
        if node.name:
            if node.name in seen_names:
                node.name = f"{node.name}_{idx}"
            seen_names.add(node.name)
        for out_name in node.output:
            if out_name:
                seen_outputs.add(out_name)
        dedup_nodes.append(node)
    del fp16_model.graph.node[:]
    fp16_model.graph.node.extend(dedup_nodes)
    onnx.save(fp16_model, str(src))


def export_lora_adapter(
    model: InkTeXV2Model,
    output_path: Path,
    *,
    adapter_name: str = "Personal LoRA",
    dtype: str = "fp16",
    training_samples: int = 0,
    notes: str = "",
) -> dict[str, Any]:
    """Export the model's ink-only LoRA matrices to a compact `.inktexlora` JSON file
    directly importable by the iPad InkTeX app for zero-reload hot-swapping."""
    cfg = model.config
    if not cfg.lora_layers or cfg.lora_rank <= 0:
        raise ValueError("Model config has no LoRA layers enabled")

    a_rows = []
    b_rows = []
    for layer_idx in cfg.lora_layers:
        blk = model.stroke_encoder.blocks[layer_idx]
        a_slot = []
        b_slot = []
        for proj in (blk.q_proj, blk.v_proj, blk.out_proj):
            if proj.lora_a is None or proj.lora_b is None:
                raise RuntimeError(f"Missing LoRA parameters on stroke block {layer_idx}")
            a_slot.append(proj.lora_a.detach().cpu().float().numpy())
            b_slot.append(proj.lora_b.detach().cpu().float().numpy())
        a_rows.append(np.stack(a_slot, axis=0))
        b_rows.append(np.stack(b_slot, axis=0))

    lora_a = np.ascontiguousarray(np.stack(a_rows, axis=0))  # [L_lora, 3, r, stroke_dim]
    lora_b = np.ascontiguousarray(np.stack(b_rows, axis=0))  # [L_lora, 3, stroke_dim, r]

    if dtype == "fp16":
        a_bytes = lora_a.astype("<f2").tobytes()
        b_bytes = lora_b.astype("<f2").tobytes()
    elif dtype == "fp32":
        a_bytes = lora_a.astype("<f4").tobytes()
        b_bytes = lora_b.astype("<f4").tobytes()
    else:
        raise ValueError(f"Unsupported LoRA adapter dtype: {dtype}")

    digest = hashlib.sha256(a_bytes + b_bytes).hexdigest()
    payload: dict[str, Any] = {
        "format": LORA_ADAPTER_FORMAT_V2,
        "adapter_name": adapter_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dtype": dtype,
        "lora_layers": list(cfg.lora_layers),
        "lora_rank": int(cfg.lora_rank),
        "lora_alpha": float(cfg.lora_alpha),
        "stroke_dim": int(cfg.stroke_dim),
        "lora_a_shape": list(lora_a.shape),
        "lora_b_shape": list(lora_b.shape),
        "training_samples": int(training_samples),
        "notes": notes,
        "sha256": digest,
        "lora_a_b64": base64.b64encode(a_bytes).decode("ascii"),
        "lora_b_b64": base64.b64encode(b_bytes).decode("ascii"),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_lora_adapter(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load an `.inktexlora` JSON file back into FP32 `(lora_a, lora_b, metadata)` arrays."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != LORA_ADAPTER_FORMAT_V2:
        raise ValueError(f"Unsupported LoRA adapter format: {payload.get('format')!r}")
    dtype = payload.get("dtype", "fp16")
    np_dt = "<f2" if dtype == "fp16" else "<f4"
    a_bytes = base64.b64decode(payload["lora_a_b64"])
    b_bytes = base64.b64decode(payload["lora_b_b64"])
    expected_sha = payload.get("sha256")
    actual_sha = hashlib.sha256(a_bytes + b_bytes).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"LoRA adapter SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    lora_a = np.frombuffer(a_bytes, dtype=np_dt).reshape(payload["lora_a_shape"]).astype(np.float32)
    lora_b = np.frombuffer(b_bytes, dtype=np_dt).reshape(payload["lora_b_shape"]).astype(np.float32)
    return lora_a, lora_b, payload


def build_manifest_v2(
    cfg: InkTeXV2Config,
    tokens: Sequence[str],
    files: dict[str, Path],
    *,
    model_name: str = "InkTeX-v2-100M",
    stage: int | None = None,
    epoch: int | None = None,
    quantization: str = "fp16",
    dynamic_lora: bool = True,
    static_shapes: bool = False,
    static_kv_len: int = STATIC_KV_LEN,
    chunk_buckets: Sequence[int] = CHUNK_BUCKETS,
    width_buckets: Sequence[int] = WIDTH_BUCKETS_V2,
    preprocess_cfg: InkPreprocessV2Config | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    pre = preprocess_cfg or InkPreprocessV2Config()
    file_meta = {}
    for key, path in files.items():
        file_meta[key] = {
            "file": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }

    c_buckets = [int(b) for b in chunk_buckets]
    w_buckets = [int(w) for w in width_buckets]
    return {
        "format": MANIFEST_FORMAT_V2,
        "model_name": model_name,
        "preset": cfg.preset,
        "stage": stage,
        "epoch": epoch,
        "quantization": quantization,
        "has_r2l": True,
        "shared_r2l_decoder": True,
        "supports_batched_decoder": True,
        "supports_prefix_verify": True,
        "static_kv_len": int(static_kv_len),
        "static_shapes": bool(static_shapes),
        "d_model": int(cfg.d_model),
        "decoder_layers": int(cfg.decoder_layers),
        "attention_heads": int(cfg.decoder_heads),
        "head_dim": int(cfg.d_model // cfg.decoder_heads),
        "stroke_layers": int(cfg.stroke_layers),
        "stroke_dim": int(cfg.stroke_dim),
        "vocab_size": int(cfg.vocab_size),
        "bos_id": int(cfg.bos_l2r_id),
        "bos_l2r_id": int(cfg.bos_l2r_id),
        "bos_r2l_id": int(cfg.bos_r2l_id),
        "eos_id": int(cfg.eos_id),
        "pad_id": int(cfg.pad_id),
        "unk_id": int(cfg.unk_id),
        "max_points": int(max(c_buckets) * cfg.chunk_size),
        "max_tokens": int(cfg.max_tokens),
        "input_dim": int(cfg.point_dim),
        "vision": {
            "height": int(cfg.vision_height),
            "min_width": int(min(w_buckets)),
            "max_width": int(max(w_buckets)),
            "width_buckets": w_buckets,
            "padding": int(PADDING_V2),
            "width_multiple": int(WIDTH_MULTIPLE_V2),
            "channels": int(cfg.vision_in_chans),
            "target_glyph_px": float(TARGET_GLYPH_PX_V2),
            "vertical_stride": 8,
            "horizontal_stride": 16,
        },
        "preprocessing": {
            "max_points": int(max(c_buckets) * cfg.chunk_size),
            "min_points_per_stroke": 1,
            "resampling_enabled": True,
            "resample_spacing": float(pre.resample_spacing),
            "min_resampled_points_per_stroke": 1,
            "use_timing": bool(pre.use_timing),
            "use_pressure": False,
            "max_delta_ms": float(pre.max_delta_ms),
            "normalization": "glyph_v2",
            "max_points_per_stroke": int(max(c_buckets) * cfg.chunk_size),
            "chunk_size": int(cfg.chunk_size),
            "chunk_hop": int(pre.chunk_hop),
            "chunk_buckets": c_buckets,
            "max_chunks": int(max(c_buckets)),
            "dedupe_eps": float(pre.dedupe_eps),
            "coord_clip": float(pre.coord_clip),
            "point_feature_dim": int(POINT_FEATURE_DIM),
            "chunk_meta_dim": int(CHUNK_META_DIM),
        },
        "personalization": {
            "style_dim": int(cfg.style_dim),
            "style_max_strokes": int(STYLE_MAX_STROKES),
            "style_points": int(STYLE_POINTS),
            "style_point_dim": int(STYLE_POINT_DIM),
            "film_layers": [int(i) for i in cfg.film_layers],
            "dynamic_lora": bool(dynamic_lora and len(cfg.lora_layers) > 0 and cfg.lora_rank > 0),
            "lora_layers": [int(i) for i in cfg.lora_layers],
            "lora_rank": int(cfg.lora_rank),
            "lora_alpha": float(cfg.lora_alpha),
            "stroke_dim": int(cfg.stroke_dim),
        },
        "tokens": list(tokens),
        "metrics": metrics,
        "files": file_meta,
    }


def export_bundle_v2(
    model: InkTeXV2Model,
    tokens: Sequence[str],
    output_dir: Path,
    *,
    model_name: str = "InkTeX-v2-100M",
    stage: int | None = None,
    epoch: int | None = None,
    quantize: str = "fp16",
    dynamic_lora: bool = True,
    dynamic_buckets: bool = True,
    static_kv_len: int = STATIC_KV_LEN,
    chunk_buckets: Sequence[int] = CHUNK_BUCKETS,
    width_buckets: Sequence[int] = WIDTH_BUCKETS_V2,
    metrics: dict[str, float] | None = None,
    verify: bool = True,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Export a full iPad-ready InkTeX v2 ONNX bundle (5 split graphs + manifest.json).

    When ``dynamic_buckets=True`` (default for multi-bucket iPad deployment), exports a single
    compact set of 5 graphs supporting all chunk buckets ``{128, 256, 512, 768}`` and width
    buckets ``{256, 512, 768, 1024}``. When ``dynamic_buckets=False``, exports fixed-shape
    graphs for ``(chunk_buckets[0], width_buckets[0])`` so CoreML can compile with
    ``RequireStaticInputShapes=1``.
    """
    import onnxruntime as ort

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    cfg = model.config
    c_buckets = tuple(int(b) for b in chunk_buckets)
    w_buckets = tuple(int(w) for w in width_buckets)
    use_dynamic_lora = bool(dynamic_lora and len(cfg.lora_layers) > 0 and cfg.lora_rank > 0)

    trace_chunks = c_buckets[0] if not dynamic_buckets else min(max(128, c_buckets[0]), c_buckets[-1])
    trace_width = w_buckets[0] if not dynamic_buckets else min(max(256, w_buckets[0]), w_buckets[-1])
    vis_h_tokens = cfg.vision_height // 8
    trace_vis_tokens = vis_h_tokens * (trace_width // 16)
    max_vis_tokens = vis_h_tokens * (max(w_buckets) // 16)
    min_vis_tokens = vis_h_tokens * (min(w_buckets) // 16)

    # 1. Writer Style Encoder (always fixed [1, 20, 16, 8] -> [1, style_dim])
    d_sp = torch.randn(1, STYLE_MAX_STROKES, STYLE_POINTS, cfg.style_point_dim)
    d_sa = torch.randn(1, STYLE_MAX_STROKES, 4)
    d_sm = torch.zeros(1, STYLE_MAX_STROKES, dtype=torch.uint8)
    d_sm[0, 8:] = 1
    style_path = output_dir / "writer_style_v2.onnx"
    torch.onnx.export(
        WriterStyleExport(model).eval(),
        (d_sp, d_sa, d_sm),
        style_path,
        input_names=["style_points", "style_aspect", "style_mask"],
        output_names=["writer_style"],
        dynamo=True,
        external_data=False,
    )

    # 2. Stroke Chunk Encoder
    d_cp = torch.randn(1, trace_chunks, cfg.chunk_size, cfg.point_dim)
    d_cpm = torch.zeros(1, trace_chunks, cfg.chunk_size, dtype=torch.uint8)
    d_cpm[0, trace_chunks // 2 - 1, 5:] = 1
    d_cpm[0, trace_chunks // 2 :, :] = 1
    d_cm = torch.randn(1, trace_chunks, cfg.chunk_meta_dim)
    d_ck = torch.zeros(1, trace_chunks, dtype=torch.uint8)
    d_ck[0, trace_chunks // 2 :] = 1
    d_ws = torch.zeros(1, cfg.style_dim)
    stroke_path = output_dir / "stroke_encoder_v2.onnx"

    stroke_dyn = None
    if dynamic_buckets:
        chunk_dim = torch.export.Dim("chunk_len", min=min(16, min(c_buckets)), max=max(c_buckets))
        stroke_dyn = (
            {1: chunk_dim},
            {1: chunk_dim},
            {1: chunk_dim},
            {1: chunk_dim},
            {},
        )

    if use_dynamic_lora:
        d_la = torch.randn(len(cfg.lora_layers), 3, cfg.lora_rank, cfg.stroke_dim) * 0.01
        d_lb = torch.randn(len(cfg.lora_layers), 3, cfg.stroke_dim, cfg.lora_rank) * 0.01
        torch.onnx.export(
            StrokeEncoderV2Export(model, dynamic_lora=True).eval(),
            (d_cp, d_cpm, d_cm, d_ck, d_ws, d_la, d_lb),
            stroke_path,
            input_names=["chunk_points", "chunk_point_mask", "chunk_meta", "chunk_mask", "writer_style", "lora_a", "lora_b"],
            output_names=["stroke_memory"],
            dynamic_shapes=(*stroke_dyn, {}, {}) if stroke_dyn is not None else None,
            dynamo=True,
            external_data=False,
        )
    else:
        d_la = None
        d_lb = None
        torch.onnx.export(
            StrokeEncoderV2Export(model, dynamic_lora=False).eval(),
            (d_cp, d_cpm, d_cm, d_ck, d_ws),
            stroke_path,
            input_names=["chunk_points", "chunk_point_mask", "chunk_meta", "chunk_mask", "writer_style"],
            output_names=["stroke_memory"],
            dynamic_shapes=stroke_dyn,
            dynamo=True,
            external_data=False,
        )

    # 3. Vision FPN Encoder
    d_img = torch.randn(1, cfg.vision_in_chans, cfg.vision_height, trace_width)
    d_iw = torch.tensor([max(16, trace_width - 32)], dtype=torch.long)
    vision_path = output_dir / "vision_encoder_v2.onnx"
    vision_dyn = None
    if dynamic_buckets and min(w_buckets) != max(w_buckets):
        w32_dim = torch.export.Dim("vis_w32", min=max(2, min(w_buckets) // 32), max=max(w_buckets) // 32)
        vision_dyn = ({3: 32 * w32_dim}, {})
    torch.onnx.export(
        VisionEncoderV2Export(model).eval(),
        (d_img, d_iw),
        vision_path,
        input_names=["images", "image_widths"],
        output_names=["vision_memory", "vision_mask"],
        dynamic_shapes=vision_dyn,
        dynamo=True,
        external_data=False,
    )

    # 4. Bimodal Fusion + Decoder Cross-KV Projection
    d_smem = torch.randn(1, trace_chunks, cfg.d_model)
    d_vmem = torch.randn(1, trace_vis_tokens, cfg.d_model)
    d_vmask = torch.zeros(1, trace_vis_tokens, dtype=torch.uint8)
    d_vmask[0, max(1, trace_vis_tokens - 48) :] = 1
    fusion_path = output_dir / "fusion_memproj_v2.onnx"
    fusion_dyn = None
    if dynamic_buckets:
        f_s_dim = torch.export.Dim("stroke_mem_len", min=1, max=max(c_buckets))
        f_v_dim = torch.export.Dim("vision_mem_len", min=min(vis_h_tokens, min_vis_tokens), max=max_vis_tokens)
        fusion_dyn = ({1: f_s_dim}, {1: f_s_dim}, {1: f_v_dim}, {1: f_v_dim})
    torch.onnx.export(
        FusionMemProjV2Export(model).eval(),
        (d_smem, d_ck, d_vmem, d_vmask),
        fusion_path,
        input_names=["stroke_memory", "stroke_mask", "vision_memory", "vision_mask"],
        output_names=["memory", "memory_mask", "mk", "mv"],
        dynamic_shapes=fusion_dyn,
        dynamo=True,
        external_data=False,
    )

    # 5. Unified Static-KV Shared Bidirectional Decoder (L2R + R2L + Prefix Verify)
    mem_len = trace_chunks + trace_vis_tokens
    head_dim = cfg.d_model // cfg.decoder_heads
    d_tok = torch.tensor([[cfg.bos_l2r_id, 5, 6, 7, 8], [cfg.bos_r2l_id, 8, 7, 6, 5]], dtype=torch.long)
    d_pos = torch.tensor([0], dtype=torch.long)
    d_pk = torch.zeros(cfg.decoder_layers, 2, cfg.decoder_heads, static_kv_len, head_dim)
    d_pv = torch.zeros(cfg.decoder_layers, 2, cfg.decoder_heads, static_kv_len, head_dim)
    d_mk = torch.randn(cfg.decoder_layers, 1, cfg.decoder_heads, mem_len, head_dim)
    d_mv = torch.randn(cfg.decoder_layers, 1, cfg.decoder_heads, mem_len, head_dim)
    d_mm = torch.zeros(1, mem_len, dtype=torch.uint8)
    dec_batch_dim = torch.export.Dim("dec_batch", min=1, max=16)
    step_len_dim = torch.export.Dim("step_len", min=1, max=static_kv_len - 1)
    dec_dyn: list[dict[int, Any]] = [
        {0: dec_batch_dim, 1: step_len_dim},
        {},
        {1: dec_batch_dim},
        {1: dec_batch_dim},
        {},
        {},
        {},
    ]
    if dynamic_buckets:
        mem_dim = torch.export.Dim("mem_len", min=2, max=max(c_buckets) + max_vis_tokens)
        dec_dyn[4] = {3: mem_dim}
        dec_dyn[5] = {3: mem_dim}
        dec_dyn[6] = {1: mem_dim}
    decoder_path = output_dir / "decoder_step_v2.onnx"
    torch.onnx.export(
        DecoderStepV2Export(model, max_kv=static_kv_len).eval(),
        (d_tok[:, :3], d_pos, d_pk, d_pv, d_mk, d_mv, d_mm),
        decoder_path,
        input_names=["token", "cache_pos", "past_k", "past_v", "mk", "mv", "memory_mask"],
        output_names=["logits", "new_k", "new_v"],
        dynamic_shapes=tuple(dec_dyn),
        dynamo=True,
        external_data=False,
    )

    if quantize == "fp16":
        _convert_fp16_v2(style_path)
        _convert_fp16_v2(stroke_path, extra_block_ops=("Div", "Sub"))
        _convert_fp16_v2(vision_path)
        _convert_fp16_v2(fusion_path)
        _convert_fp16_v2(decoder_path)
    elif quantize not in ("fp32", "none"):
        raise ValueError(f"Unsupported quantize mode for v2 export: {quantize}")

    max_diffs: dict[str, float] = {}
    if verify:
        eff_atol = max(atol, 2e-2) if quantize == "fp16" else atol
        with torch.no_grad():
            s_pt = WriterStyleExport(model)(d_sp, d_sa, d_sm).numpy()
            sess = ort.InferenceSession(str(style_path), providers=["CPUExecutionProvider"])
            s_ort = sess.run(None, {"style_points": d_sp.numpy(), "style_aspect": d_sa.numpy(), "style_mask": d_sm.numpy()})[0]
            max_diffs["writer_style"] = float(np.max(np.abs(s_pt - s_ort)))

            st_sess = ort.InferenceSession(str(stroke_path), providers=["CPUExecutionProvider"])
            if use_dynamic_lora and d_la is not None and d_lb is not None:
                st_pt = StrokeEncoderV2Export(model, dynamic_lora=True)(d_cp, d_cpm, d_cm, d_ck, d_ws, d_la, d_lb).numpy()
                st_ort = st_sess.run(None, {
                    "chunk_points": d_cp.numpy(), "chunk_point_mask": d_cpm.numpy(),
                    "chunk_meta": d_cm.numpy(), "chunk_mask": d_ck.numpy(),
                    "writer_style": d_ws.numpy(), "lora_a": d_la.numpy(), "lora_b": d_lb.numpy(),
                })[0]
            else:
                st_pt = StrokeEncoderV2Export(model, dynamic_lora=False)(d_cp, d_cpm, d_cm, d_ck, d_ws).numpy()
                st_ort = st_sess.run(None, {
                    "chunk_points": d_cp.numpy(), "chunk_point_mask": d_cpm.numpy(),
                    "chunk_meta": d_cm.numpy(), "chunk_mask": d_ck.numpy(),
                    "writer_style": d_ws.numpy(),
                })[0]
            max_diffs["stroke_encoder"] = float(np.max(np.abs(st_pt - st_ort)))

            v_pt, vm_pt = VisionEncoderV2Export(model)(d_img, d_iw)
            v_sess = ort.InferenceSession(str(vision_path), providers=["CPUExecutionProvider"])
            v_ort, vm_ort = v_sess.run(None, {"images": d_img.numpy(), "image_widths": d_iw.numpy()})
            max_diffs["vision_encoder_mem"] = float(np.max(np.abs(v_pt.numpy() - v_ort)))
            max_diffs["vision_encoder_mask"] = float(np.max(np.abs(vm_pt.numpy().astype(np.int32) - vm_ort.astype(np.int32))))

            f_mem_pt, f_mask_pt, f_mk_pt, f_mv_pt = FusionMemProjV2Export(model)(d_smem, d_ck, d_vmem, d_vmask)
            f_sess = ort.InferenceSession(str(fusion_path), providers=["CPUExecutionProvider"])
            f_mem_ort, f_mask_ort, f_mk_ort, f_mv_ort = f_sess.run(None, {
                "stroke_memory": d_smem.numpy(), "stroke_mask": d_ck.numpy(),
                "vision_memory": d_vmem.numpy(), "vision_mask": d_vmask.numpy(),
            })
            max_diffs["fusion_mk"] = float(np.max(np.abs(f_mk_pt.numpy() - f_mk_ort)))
            max_diffs["fusion_mv"] = float(np.max(np.abs(f_mv_pt.numpy() - f_mv_ort)))

            # Verify parallel batched L2R + R2L scoring in ONE single decoder_step_v2 call!
            full_pt = model.decoder(d_tok, f_mem_pt.expand(2, -1, -1), f_mask_pt.ne(0).expand(2, -1)).numpy()
            dec_sess = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
            par_ort, _, _ = dec_sess.run(None, {
                "token": d_tok.numpy(),
                "cache_pos": np.array([0], dtype=np.int64),
                "past_k": np.zeros_like(d_pk.numpy()),
                "past_v": np.zeros_like(d_pv.numpy()),
                "mk": f_mk_ort,
                "mv": f_mv_ort,
                "memory_mask": f_mask_ort,
            })
            max_diffs["decoder_parallel_l2r_r2l"] = float(np.max(np.abs(full_pt - par_ort)))

        for k, diff in max_diffs.items():
            if diff > eff_atol:
                raise AssertionError(f"ONNX bundle parity check failed on {k}: diff={diff:.3e} > atol={eff_atol:.3e}")

    file_map = {
        "writer_style": style_path,
        "stroke_encoder": stroke_path,
        "vision_encoder": vision_path,
        "fusion_memproj": fusion_path,
        "decoder_step": decoder_path,
    }
    manifest = build_manifest_v2(
        cfg,
        tokens,
        file_map,
        model_name=model_name,
        stage=stage,
        epoch=epoch,
        quantization="fp32" if quantize == "none" else quantize,
        dynamic_lora=use_dynamic_lora,
        static_shapes=not dynamic_buckets,
        static_kv_len=static_kv_len,
        chunk_buckets=c_buckets if dynamic_buckets else (trace_chunks,),
        width_buckets=w_buckets if dynamic_buckets else (trace_width,),
        metrics=metrics,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "manifest": str(manifest_path),
        "files": {k: str(v) for k, v in file_map.items()},
        "max_abs_diff": max_diffs,
    }


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export an InkTeX v2 checkpoint to an iPad ONNX bundle")
    parser.add_argument("--checkpoint", type=Path, help="Path to v2 .pt checkpoint (omit to run tiny self-test)")
    parser.add_argument("--output", type=Path, default=Path("model-bundle-v2"), help="Output bundle directory")
    parser.add_argument("--name", type=str, default="InkTeX-v2-100M", help="Display name in manifest.json")
    parser.add_argument("--quantize", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--static-buckets", action="store_true", help="Export single fixed bucket shapes for strict CoreML static compilation")
    parser.add_argument("--chunk-bucket", type=int, default=256, help="Chunk bucket when --static-buckets is set")
    parser.add_argument("--vision-width", type=int, default=512, help="Vision width bucket when --static-buckets is set")
    parser.add_argument("--no-dynamic-lora", action="store_true", help="Disable dynamic LoRA inputs (merge into base weights instead)")
    parser.add_argument("--static-kv-len", type=int, default=STATIC_KV_LEN)
    parser.add_argument("--export-lora", type=Path, default=None, help="Optional path to also export `.inktexlora` adapter file")
    args = parser.parse_args()

    if args.checkpoint is None:
        import tempfile

        tiny = InkTeXV2Model(get_v2_config("inktex_v2_tiny", vocab_size=128))
        print("inktex_v2_tiny params:", tiny.parameter_summary())
        full = InkTeXV2Model(get_v2_config("inktex_v2_100m", vocab_size=384, vision_pretrained=False))
        print("inktex_v2_100m params:", {k: round(v / 1e6, 2) for k, v in full.parameter_summary().items()})
        with tempfile.TemporaryDirectory() as tmp:
            report = export_and_verify_v2(tiny, Path(tmp), dynamic_lora=True, static_kv_len=64)
            print("ONNX parity report:", json.dumps(report["max_abs_diff"], indent=2))
        return

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = InkTeXV2Config.from_dict(payload["config"])
    cfg.vision_pretrained = False
    model = InkTeXV2Model(cfg)
    state = payload.get("ema_state_dict") or payload.get("model_state_dict") or payload.get("model")
    if state is None:
        raise KeyError("Checkpoint missing model_state_dict / ema_state_dict")
    model.load_state_dict(state, strict=False)
    model.eval()
    tokens = payload.get("tokenizer", {}).get("tokens") or [f"<TOK_{i}>" for i in range(cfg.vocab_size)]
    metrics = payload.get("metrics")

    if args.no_dynamic_lora:
        model.merge_lora_()

    report = export_bundle_v2(
        model,
        tokens,
        args.output,
        model_name=args.name,
        stage=payload.get("stage"),
        epoch=payload.get("epoch"),
        quantize=args.quantize,
        dynamic_lora=not args.no_dynamic_lora,
        dynamic_buckets=not args.static_buckets,
        static_kv_len=args.static_kv_len,
        chunk_buckets=(args.chunk_bucket,) if args.static_buckets else CHUNK_BUCKETS,
        width_buckets=(args.vision_width,) if args.static_buckets else WIDTH_BUCKETS_V2,
        metrics=metrics,
    )
    print("Exported v2 bundle:", json.dumps(report, indent=2))
    if args.export_lora is not None:
        export_lora_adapter(model, args.export_lora, adapter_name=f"{args.name} LoRA")
        print(f"Exported LoRA adapter -> {args.export_lora}")


if __name__ == "__main__":
    _cli()
