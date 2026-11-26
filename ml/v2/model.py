"""InkTeX v2: ~102M asymmetric encoder-heavy bimodal math model.

Architecture:
  - StrokeChunkEncoder:  13L x d=576 (9 heads, FFN 2304), global attention over 8-pt sub-stroke chunks,
                         zero-init FiLM writer conditioning (blocks 7/10/13), built-in LoRA r=8 (blocks 10-13),
                         zero .item() / data-dependent shapes for Apple ANE/CoreML static buckets.
  - WriterStyleEncoder:  16-pt raw-stroke Conv1D -> 64-d writer vector (zero-backprop on-device).
  - VisionFPNEncoder:    ConvNeXt-V2-Tiny (FCMAE init, 2-ch stem) + s8/s16/s32 FPN -> (1,2) horizontal
                         reduction -> 24 vertical rows at H=192, static W buckets {256,512,768,1024}.
  - BimodalCrossFusion:  2 ink->vision cross-attention blocks at d=384 with zero-init tanh gates.
  - SharedBidiDecoder:   6L x d=384 (6 heads, FFN 1536), shared L2R + R2L via <BOS_L2R>/<BOS_R2L>,
                         ALiBi, tied embeddings, fused SDPA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .ink_preprocess import CHUNK_META_DIM, CHUNK_SIZE, POINT_FEATURE_DIM, STYLE_POINT_DIM

MODALITY_MODES = ("auto", "vision", "stroke", "bimodal")


@dataclass(slots=True)
class InkTeXV2Config:
    preset: str = "inktex_v2_100m"
    vocab_size: int = 384
    pad_id: int = 0
    bos_l2r_id: int = 1
    bos_r2l_id: int = 2
    eos_id: int = 3
    unk_id: int = 4

    # Stroke encoder (~52M)
    point_dim: int = POINT_FEATURE_DIM
    chunk_meta_dim: int = CHUNK_META_DIM
    chunk_size: int = CHUNK_SIZE
    stroke_stem_dim: int = 256
    stroke_dim: int = 576
    stroke_layers: int = 13
    stroke_heads: int = 9
    stroke_ffn_dim: int = 2304
    stroke_dropout: float = 0.10
    stroke_drop_path: float = 0.10

    # Personalization (built in from Day 1)
    style_point_dim: int = STYLE_POINT_DIM
    style_dim: int = 64
    film_layers: tuple[int, ...] = (6, 9, 12)      # 0-indexed (blocks 7, 10, 13)
    lora_layers: tuple[int, ...] = (9, 10, 11, 12) # 0-indexed (blocks 10, 11, 12, 13)
    lora_rank: int = 8
    lora_alpha: float = 16.0

    # Vision encoder (~31M)
    vision_backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
    vision_pretrained: bool = True
    vision_in_chans: int = 2
    vision_fpn_dim: int = 192
    vision_height: int = 192
    vision_max_width: int = 1024
    vision_global_layers: int = 0  # optional global 2D-RoPE layers (0 default)

    # Fusion & Lean Shared Bidirectional Decoder (~16.6M)
    d_model: int = 384
    fusion_layers: int = 2
    fusion_heads: int = 6
    fusion_ffn_dim: int = 768
    decoder_layers: int = 6
    decoder_heads: int = 6
    decoder_ffn_dim: int = 1536
    decoder_dropout: float = 0.15
    max_tokens: int = 256

    # Auxiliary training-only stroke structure heads (0 = disabled)
    structure_symbol_classes: int = 0
    structure_relation_classes: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["film_layers"] = list(self.film_layers)
        d["lora_layers"] = list(self.lora_layers)
        return d

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InkTeXV2Config":
        fields = cls.__dataclass_fields__
        cleaned = {}
        for k, v in value.items():
            if k in fields:
                if k in ("film_layers", "lora_layers"):
                    cleaned[k] = tuple(v)
                else:
                    cleaned[k] = v
        return cls(**cleaned)


def get_v2_config(preset: str = "inktex_v2_100m", vocab_size: int = 384, **overrides: Any) -> InkTeXV2Config:
    if preset == "inktex_v2_100m":
        cfg = InkTeXV2Config(preset=preset, vocab_size=vocab_size)
    elif preset == "inktex_v2_ablation":
        # Small 6L d=384 model for Week-1 point-vs-chunk ablation (no FiLM/LoRA so capacity matches exactly)
        cfg = InkTeXV2Config(
            preset=preset,
            vocab_size=vocab_size,
            stroke_stem_dim=192,
            stroke_dim=384,
            stroke_layers=6,
            stroke_heads=6,
            stroke_ffn_dim=1536,
            film_layers=(),
            lora_layers=(),
            lora_rank=0,
            vision_backbone="convnext_atto",
            vision_pretrained=False,
            vision_fpn_dim=128,
            d_model=384,
            fusion_layers=1,
            decoder_layers=4,
            decoder_heads=6,
            decoder_ffn_dim=1536,
        )
    elif preset == "inktex_v2_tiny":
        # Ultra-fast CPU / Week-1 ONNX export prototype preset
        cfg = InkTeXV2Config(
            preset=preset,
            vocab_size=vocab_size,
            stroke_stem_dim=64,
            stroke_dim=128,
            stroke_layers=4,
            stroke_heads=4,
            stroke_ffn_dim=256,
            style_dim=32,
            film_layers=(1, 2, 3),
            lora_layers=(2, 3),
            lora_rank=4,
            lora_alpha=8.0,
            vision_backbone="convnext_atto",
            vision_pretrained=False,
            vision_fpn_dim=64,
            d_model=128,
            fusion_layers=1,
            fusion_heads=4,
            fusion_ffn_dim=256,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=256,
        )
    else:
        raise ValueError(f"Unknown InkTeX v2 preset: {preset}")
    return replace(cfg, **overrides)


# --------------------------------------------------------------------------- Primitives


def alibi_slopes(num_heads: int) -> torch.Tensor:
    def power_of_two(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(power_of_two(num_heads), dtype=torch.float32)
    closest = 2 ** math.floor(math.log2(num_heads))
    slopes = power_of_two(closest)
    extra = power_of_two(2 * closest)[0::2][: num_heads - closest]
    return torch.tensor(slopes + extra, dtype=torch.float32)


def sinusoidal_1d(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None] * freq[None, :]
    emb = torch.cat([pos.sin(), pos.cos()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.to(dtype)


def sinusoidal_2d_grid(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("2D sinusoidal encoding requires dim divisible by 4")
    quarter = dim // 4
    freq = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / quarter))
    rows = torch.arange(height, device=device, dtype=torch.float32)[:, None] * freq
    cols = torch.arange(width, device=device, dtype=torch.float32)[:, None] * freq
    r_emb = torch.cat([rows.sin(), rows.cos()], dim=-1)[:, None, :].expand(height, width, -1)
    c_emb = torch.cat([cols.sin(), cols.cos()], dim=-1)[None, :, :].expand(height, width, -1)
    return torch.cat([r_emb, c_emb], dim=-1).to(dtype)


def fourier_2d_coords(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous 2D Fourier features on arbitrary ``xy [..., 2]`` coordinates (FP16/ANE-safe)."""
    if dim % 4 != 0:
        raise ValueError("fourier_2d_coords requires dim divisible by 4")
    quarter = dim // 4
    # Capped max freq (4.0 rad/unit) keeps phase accurate in FP16 across +-32 glyph units
    freqs = torch.exp(
        torch.linspace(math.log(0.2), math.log(4.0), quarter, device=xy.device, dtype=torch.float32)
    )
    x_ang = xy[..., 0:1].float() * freqs
    y_ang = xy[..., 1:2].float() * freqs
    return torch.cat([x_ang.sin(), x_ang.cos(), y_ang.sin(), y_ang.cos()], dim=-1).to(xy.dtype)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask.div_(keep)


class LoRALinear(nn.Module):
    """Linear layer with optional built-in low-rank adapter (r=8, alpha=16).

    Supports BOTH:
      1. Internal trainable LoRA parameters (``use_internal_lora=True``) + ``merge_lora()``
      2. External dynamic graph inputs ``external_ab=(A, B)`` (for hot-swappable iPad ONNX)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 0.0
        self.lora_enabled = False
        self.lora_drop = nn.Dropout(lora_dropout) if self.lora_rank > 0 else nn.Identity()
        if self.lora_rank > 0:
            self.lora_a = nn.Parameter(torch.empty(self.lora_rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, self.lora_rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(
        self,
        x: torch.Tensor,
        external_ab: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        out = self.linear(x)
        if external_ab is not None and self.lora_rank > 0:
            a, b = external_ab
            # a: [r, in] or [1, r, in]; b: [out, r] or [1, out, r]
            if a.ndim == 3:
                a = a.squeeze(0)
            if b.ndim == 3:
                b = b.squeeze(0)
            delta = F.linear(F.linear(x, a), b) * self.scaling
            return out + delta
        if self.lora_enabled and self.lora_a is not None and self.lora_b is not None:
            delta = F.linear(F.linear(self.lora_drop(x), self.lora_a), self.lora_b) * self.scaling
            return out + delta
        return out

    @torch.no_grad()
    def merge_lora_(self) -> None:
        if self.lora_rank > 0 and self.lora_a is not None and self.lora_b is not None:
            self.linear.weight.add_((self.lora_b @ self.lora_a) * self.scaling)
            self.lora_b.zero_()
            self.lora_enabled = False


# --------------------------------------------------------------------------- Writer Style Encoder


class WriterStyleEncoder(nn.Module):
    """Zero-backprop on-device writer style encoder: 5-20 raw strokes -> 64-d writer vector.

    When all support strokes are masked (``n=0``), outputs an exact zero vector so base
    recognition is 100% invariant with no profile enrolled.
    """

    def __init__(self, point_dim: int = STYLE_POINT_DIM, style_dim: int = 64):
        super().__init__()
        self.style_dim = style_dim
        self.conv = nn.Sequential(
            nn.Conv1d(point_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stroke_mlp = nn.Sequential(
            nn.Linear(64 * 2 + 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.profile_mlp = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.GELU(),
            nn.Linear(128, style_dim),
        )
        self.quality_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        style_points: torch.Tensor,   # [B, S, P, 8]
        style_aspect: torch.Tensor,   # [B, S, 4]
        style_mask: torch.Tensor,     # [B, S] bool (True = padded)
    ) -> torch.Tensor:
        bsz, strokes, pts, feat = style_points.shape
        x = style_points.reshape(bsz * strokes, pts, feat).transpose(1, 2)  # [B*S, 8, P]
        h = self.conv(x)  # [B*S, 64, P]
        mean_p = h.mean(dim=-1)
        std_p = (h.var(dim=-1, unbiased=False) + 1e-5).sqrt()
        s_vec = torch.cat([mean_p, std_p, style_aspect.reshape(bsz * strokes, 4)], dim=-1)
        s_vec = self.stroke_mlp(s_vec).reshape(bsz, strokes, 128)

        valid = (~style_mask).to(s_vec.dtype)  # [B, S]
        count = valid.sum(dim=1, keepdim=True)  # [B, 1]
        denom = count.clamp_min(1.0)
        mean_s = (s_vec * valid.unsqueeze(-1)).sum(dim=1) / denom
        diff = (s_vec - mean_s.unsqueeze(1)) * valid.unsqueeze(-1)
        std_s = ((diff * diff).sum(dim=1) / denom + 1e-5).sqrt()

        raw = self.profile_mlp(torch.cat([mean_s, std_s], dim=-1))  # [B, style_dim]
        confidence = (count / (count + 10.0)) * torch.sigmoid(self.quality_gate)
        # Exact zero when count == 0
        has_any = count.gt(0.5).to(raw.dtype)
        return raw * confidence * has_any


# --------------------------------------------------------------------------- Stroke Chunk Encoder


class StrokeEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        drop_path: float,
        *,
        style_dim: int = 0,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = LoRALinear(dim, dim, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = LoRALinear(dim, dim, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.out_proj = LoRALinear(dim, dim, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.attn_drop = dropout
        self.proj_drop = nn.Dropout(dropout)
        self.dp1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.ffn_drop = nn.Dropout(dropout)
        self.dp2 = DropPath(drop_path)

        self.has_film = style_dim > 0
        if self.has_film:
            # bias=False guarantees exact zero-change when writer_style == 0 even after training!
            self.film = nn.Linear(style_dim, dim * 2, bias=False)
            nn.init.zeros_(self.film.weight)
            self.film_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,   # [B, N] bool (True = padded)
        writer_style: torch.Tensor | None = None,
        external_lora: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        use_explicit_attn: bool = False,
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape
        h = self.norm1(x)
        if self.has_film and writer_style is not None:
            gamma, beta = self.film(writer_style).chunk(2, dim=-1)
            strength = self.film_scale.tanh()
            h = h * (1.0 + strength * gamma.tanh().unsqueeze(1)) + strength * beta.tanh().unsqueeze(1)

        q = self.q_proj(h, external_lora.get("q") if external_lora else None)
        k = self.k_proj(h)
        v = self.v_proj(h, external_lora.get("v") if external_lora else None)

        q = q.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Safe key mask: never leave an entire row -inf (guards empty-input export)
        safe_mask = padding_mask.clone()
        all_pad = safe_mask.all(dim=1, keepdim=True)
        first = torch.arange(seq, device=x.device).eq(0).unsqueeze(0)
        safe_mask = safe_mask & ~(all_pad & first)

        if use_explicit_attn:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(safe_mask[:, None, None, :], -1e4)
            attn = scores.softmax(dim=-1) @ v
        else:
            attn_bias = torch.zeros((bsz, 1, 1, seq), dtype=q.dtype, device=x.device)
            attn_bias = attn_bias.masked_fill(safe_mask[:, None, None, :], float("-inf"))
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop if self.training else 0.0,
            )

        attn = attn.transpose(1, 2).reshape(bsz, seq, self.dim)
        attn = self.proj_drop(self.out_proj(attn, external_lora.get("o") if external_lora else None))
        x = x + self.dp1(attn)

        h2 = self.norm2(x)
        ffn = self.fc2(self.ffn_drop(F.gelu(self.fc1(h2))))
        x = x + self.dp2(self.ffn_drop(ffn))
        return x


class StrokeChunkEncoder(nn.Module):
    """Global sub-stroke encoder: point Conv1D stem -> attention pool -> 13L global Transformer."""

    def __init__(self, cfg: InkTeXV2Config):
        super().__init__()
        self.cfg = cfg
        stem = cfg.stroke_stem_dim
        self.point_in = nn.Linear(cfg.point_dim, stem)
        self.dw_conv1 = nn.Conv1d(stem, stem, kernel_size=3, padding=1, groups=stem)
        self.pw_conv1 = nn.Conv1d(stem, stem, kernel_size=1)
        self.dw_conv2 = nn.Conv1d(stem, stem, kernel_size=3, padding=1, groups=stem)
        self.pw_conv2 = nn.Conv1d(stem, stem, kernel_size=1)
        self.point_norm = nn.LayerNorm(stem)
        self.pool_score = nn.Linear(stem, 1)
        self.chunk_proj = nn.Linear(stem, cfg.stroke_dim)

        self.meta_mlp = nn.Sequential(
            nn.Linear(cfg.chunk_meta_dim, cfg.stroke_dim),
            nn.GELU(),
            nn.Linear(cfg.stroke_dim, cfg.stroke_dim),
        )
        self.fourier_proj = nn.Linear(cfg.stroke_dim, cfg.stroke_dim)
        self.in_norm = nn.LayerNorm(cfg.stroke_dim)
        self.in_drop = nn.Dropout(cfg.stroke_dropout)

        dpr = torch.linspace(0.0, cfg.stroke_drop_path, cfg.stroke_layers).tolist()
        film_set = set(cfg.film_layers)
        lora_set = set(cfg.lora_layers)
        self.blocks = nn.ModuleList([
            StrokeEncoderBlock(
                cfg.stroke_dim,
                cfg.stroke_heads,
                cfg.stroke_ffn_dim,
                cfg.stroke_dropout,
                dpr[i],
                style_dim=cfg.style_dim if i in film_set else 0,
                lora_rank=cfg.lora_rank if i in lora_set else 0,
                lora_alpha=cfg.lora_alpha,
            )
            for i in range(cfg.stroke_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.stroke_dim)
        self.grad_checkpointing = False

        # Optional training-only auxiliary structure heads
        self.symbol_head = (
            nn.Linear(cfg.stroke_dim, cfg.structure_symbol_classes)
            if cfg.structure_symbol_classes > 0
            else None
        )

    def forward(
        self,
        chunk_points: torch.Tensor,                  # [B, N, K, 20]
        chunk_meta: torch.Tensor,                    # [B, N, 10]
        chunk_mask: torch.Tensor,                    # [B, N] bool (True = padded chunk)
        writer_style: torch.Tensor | None = None,
        external_lora: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] | None = None,
        *,
        chunk_point_mask: torch.Tensor | None = None,  # [B, N, K] bool (True = padded tail point)
        use_explicit_attn: bool = False,
    ) -> torch.Tensor:
        bsz, n_chunks, k_pts, _ = chunk_points.shape
        p = F.gelu(self.point_in(chunk_points))  # [B, N, K, stem]
        if chunk_point_mask is not None:
            valid_pt = (~chunk_point_mask).to(p.dtype).unsqueeze(-1)
            p = p * valid_pt
        p_c = p.reshape(bsz * n_chunks, k_pts, -1).transpose(1, 2)  # [B*N, stem, K]
        p_c = p_c + F.gelu(self.pw_conv1(self.dw_conv1(p_c)))
        if chunk_point_mask is not None:
            p_c = p_c * (~chunk_point_mask).reshape(bsz * n_chunks, 1, k_pts).to(p_c.dtype)
        p_c = p_c + F.gelu(self.pw_conv2(self.dw_conv2(p_c)))
        p_h = self.point_norm(p_c.transpose(1, 2))  # [B*N, K, stem]

        pool_logits = self.pool_score(p_h)  # [B*N, K, 1]
        if chunk_point_mask is not None:
            pm = chunk_point_mask.reshape(bsz * n_chunks, k_pts)
            # Keep point 0 unmasked on all-padded chunks so softmax is always finite
            first_pt = torch.arange(k_pts, device=p_h.device).eq(0).unsqueeze(0)
            safe_pm = pm & ~(pm.all(dim=1, keepdim=True) & first_pt)
            pool_logits = pool_logits.masked_fill(safe_pm.unsqueeze(-1), -1e4)
        attn_w = pool_logits.softmax(dim=1)
        pooled = (p_h * attn_w).sum(dim=1).reshape(bsz, n_chunks, -1)  # [B, N, stem]

        tokens = self.chunk_proj(pooled) + self.meta_mlp(chunk_meta)
        # Additive continuous 2D Fourier geometry + 1D temporal chunk position
        fourier = fourier_2d_coords(chunk_meta[:, :, 0:2], self.cfg.stroke_dim)
        temporal = sinusoidal_1d(n_chunks, self.cfg.stroke_dim, tokens.device, tokens.dtype).unsqueeze(0)
        tokens = self.in_drop(self.in_norm(tokens + self.fourier_proj(fourier) + temporal))
        tokens = tokens.masked_fill(chunk_mask.unsqueeze(-1), 0.0)

        for idx, block in enumerate(self.blocks):
            blk_lora = external_lora.get(idx) if external_lora else None
            if self.grad_checkpointing and self.training and blk_lora is None and not use_explicit_attn:
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, chunk_mask, writer_style, use_reentrant=False,
                )
            else:
                tokens = block(
                    tokens,
                    chunk_mask,
                    writer_style=writer_style,
                    external_lora=blk_lora,
                    use_explicit_attn=use_explicit_attn,
                )

        out = self.final_norm(tokens)
        return out.masked_fill(chunk_mask.unsqueeze(-1), 0.0)


# --------------------------------------------------------------------------- Vision FPN Encoder


class VisionFPNEncoder(nn.Module):
    """ConvNeXt-V2-Tiny + s8/s16/s32 FPN -> (1, 2) horizontal reduction -> 24 vertical rows."""

    def __init__(self, cfg: InkTeXV2Config):
        super().__init__()
        import timm

        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.vision_backbone,
            pretrained=cfg.vision_pretrained,
            features_only=True,
            out_indices=(1, 2, 3),  # strides 8, 16, 32
            in_chans=3 if (cfg.vision_pretrained and cfg.vision_in_chans == 2) else cfg.vision_in_chans,
        )
        # Replace 3-ch pretrained RGB stem with exact grayscale RGB-sum in ch0 and 0 in ch1 (stroke order)
        if cfg.vision_pretrained and cfg.vision_in_chans == 2:
            stem_conv = getattr(self.backbone, "stem_0", None)
            if not isinstance(stem_conv, nn.Conv2d) or stem_conv.weight.shape[1] != 3:
                raise RuntimeError(f"Expected 3-ch Conv2d stem_0 on {cfg.vision_backbone}, got {type(stem_conv)}")
            new_stem = nn.Conv2d(
                2,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                dilation=stem_conv.dilation,
                groups=stem_conv.groups,
                bias=stem_conv.bias is not None,
            )
            with torch.no_grad():
                new_stem.weight[:, 0] = stem_conv.weight.sum(dim=1)
                new_stem.weight[:, 1].zero_()
                if stem_conv.bias is not None and new_stem.bias is not None:
                    new_stem.bias.copy_(stem_conv.bias)
            self.backbone.stem_0 = new_stem

        c3, c4, c5 = self.backbone.feature_info.channels()
        fpn = cfg.vision_fpn_dim
        self.lat3 = nn.Conv2d(c3, fpn, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, fpn, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, fpn, kernel_size=1)

        # Anisotropic (1, 2) reduction on s8 grid: preserves ALL 24 vertical rows (H/8)
        # while halving horizontal width to W/16 (384 / 768 / 1152 / 1536 tokens).
        self.aniso_dw = nn.Conv2d(fpn, fpn, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), groups=fpn)
        self.aniso_pw = nn.Conv2d(fpn, cfg.d_model, kernel_size=1)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.col_stride = 16  # effective horizontal stride after s8 * (1, 2)

    def forward(self, images: torch.Tensor, image_widths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        p3, p4, p5 = self.backbone(images)
        f5 = self.lat5(p5)
        f4 = self.lat4(p4) + F.interpolate(f5, size=p4.shape[-2:], mode="nearest")
        f3 = self.lat3(p3) + F.interpolate(f4, size=p3.shape[-2:], mode="nearest")

        feat = self.aniso_pw(F.gelu(self.aniso_dw(f3)))  # [B, d_model, H_f=24, W_f=W/16]
        bsz, _, h_f, w_f = feat.shape
        values = self.norm(feat.permute(0, 2, 3, 1))  # [B, H_f, W_f, d_model]
        values = values + sinusoidal_2d_grid(h_f, w_f, self.cfg.d_model, values.device, values.dtype)
        # Column-major reading order (W_f outer, H_f inner) so padded right columns sit at the tail
        tokens = values.transpose(1, 2).flatten(1, 2)  # [B, W_f * H_f, d_model]

        if image_widths is None:
            mask = torch.zeros((bsz, w_f * h_f), dtype=torch.bool, device=tokens.device)
        else:
            iw = image_widths.to(device=tokens.device, dtype=torch.long)
            # image_width <= 0 means absent image in a mixed batch -> mask ALL vision tokens!
            valid_cols = torch.where(
                iw > 0,
                torch.div(iw + self.col_stride - 1, self.col_stride, rounding_mode="floor").clamp(1, w_f),
                torch.zeros_like(iw),
            )
            mask = (torch.arange(w_f, device=tokens.device)[None] >= valid_cols[:, None]).repeat_interleave(h_f, dim=1)
        return tokens, mask


# --------------------------------------------------------------------------- Bimodal Cross-Attention Fusion


class CrossFusionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout
        self.drop = nn.Dropout(dropout)
        self.gate_attn = nn.Parameter(torch.zeros(()))

        self.norm_ffn = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.gate_ffn = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        ink: torch.Tensor,            # [B, N_s, D]
        vision: torch.Tensor,         # [B, N_v, D]
        vision_mask: torch.Tensor,    # [B, N_v] bool (True = padded)
        vision_avail: torch.Tensor,   # [B, 1, 1] float (1.0 if vision active else 0.0)
        *,
        use_explicit_attn: bool = False,
    ) -> torch.Tensor:
        bsz, n_s, _ = ink.shape
        n_v = vision.shape[1]
        q = self.q_proj(self.norm_q(ink)).reshape(bsz, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.norm_kv(vision)
        k = self.k_proj(kv).reshape(bsz, n_v, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).reshape(bsz, n_v, self.num_heads, self.head_dim).transpose(1, 2)

        safe_mask = vision_mask.clone()
        all_pad = safe_mask.all(dim=1, keepdim=True)
        first = torch.arange(n_v, device=ink.device).eq(0).unsqueeze(0)
        safe_mask = safe_mask & ~(all_pad & first)

        if use_explicit_attn:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(safe_mask[:, None, None, :], -1e4)
            attn = scores.softmax(dim=-1) @ v
        else:
            bias = torch.zeros((bsz, 1, 1, n_v), dtype=q.dtype, device=ink.device)
            bias = bias.masked_fill(safe_mask[:, None, None, :], float("-inf"))
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=bias,
                dropout_p=self.attn_drop if self.training else 0.0,
            )

        attn = self.drop(self.out_proj(attn.transpose(1, 2).reshape(bsz, n_s, self.d_model)))
        ink = ink + (self.gate_attn.tanh() * vision_avail) * attn
        ffn = self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.norm_ffn(ink))))))
        ink = ink + (self.gate_ffn.tanh() * vision_avail) * ffn
        return ink


class BimodalCrossFusion(nn.Module):
    def __init__(self, cfg: InkTeXV2Config):
        super().__init__()
        self.stroke_to_dec = nn.Linear(cfg.stroke_dim, cfg.d_model)
        self.stroke_norm = nn.LayerNorm(cfg.d_model)
        self.vision_norm = nn.LayerNorm(cfg.d_model)
        self.stroke_mod_emb = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.vision_mod_emb = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.stroke_mod_emb, std=0.02)
        nn.init.normal_(self.vision_mod_emb, std=0.02)
        self.blocks = nn.ModuleList([
            CrossFusionBlock(cfg.d_model, cfg.fusion_heads, cfg.fusion_ffn_dim, cfg.stroke_dropout)
            for _ in range(cfg.fusion_layers)
        ])

    def project_stroke(self, stroke_mem: torch.Tensor) -> torch.Tensor:
        return self.stroke_norm(self.stroke_to_dec(stroke_mem)) + self.stroke_mod_emb

    def project_vision(self, vision_mem: torch.Tensor) -> torch.Tensor:
        return self.vision_norm(vision_mem) + self.vision_mod_emb

    def fuse(
        self,
        stroke_mem: torch.Tensor,     # [B, N_s, d_model]
        stroke_mask: torch.Tensor,    # [B, N_s] bool
        vision_mem: torch.Tensor,     # [B, N_v, d_model]
        vision_mask: torch.Tensor,    # [B, N_v] bool
        *,
        use_explicit_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision_avail = (~vision_mask).any(dim=1).to(stroke_mem.dtype)[:, None, None]
        fused_ink = stroke_mem
        for blk in self.blocks:
            fused_ink = blk(fused_ink, vision_mem, vision_mask, vision_avail, use_explicit_attn=use_explicit_attn)
        fused_ink = fused_ink.masked_fill(stroke_mask.unsqueeze(-1), 0.0)
        memory = torch.cat([fused_ink, vision_mem], dim=1)
        mask = torch.cat([stroke_mask, vision_mask], dim=1)
        return memory, mask


# --------------------------------------------------------------------------- Lean Shared Bidirectional Decoder


class DecoderBlockV2(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.self_qkv = nn.Linear(d_model, d_model * 3)
        self.self_out = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_kv = nn.Linear(d_model, d_model * 2)
        self.cross_out = nn.Linear(d_model, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.attn_drop = dropout
        self.drop = nn.Dropout(dropout)

    def project_memory(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, mem_len, _ = memory.shape
        kv = self.cross_kv(memory).reshape(bsz, mem_len, 2, self.num_heads, self.head_dim)
        mk = kv[:, :, 0].transpose(1, 2)  # [B, H, M, Dh]
        mv = kv[:, :, 1].transpose(1, 2)
        return mk, mv

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_bias: torch.Tensor,        # [B or 1, H, T, T]
        mem_bias: torch.Tensor,         # [B, 1, 1, M]
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape
        h = self.norm1(x)
        qkv = self.self_qkv(h).reshape(bsz, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x + self.drop(self.self_out(attn.transpose(1, 2).reshape(bsz, seq, self.d_model)))

        h2 = self.norm2(x)
        cq = self.cross_q(h2).reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        mk, mv = self.project_memory(memory)
        c_attn = F.scaled_dot_product_attention(
            cq, mk, mv,
            attn_mask=mem_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x + self.drop(self.cross_out(c_attn.transpose(1, 2).reshape(bsz, seq, self.d_model)))

        h3 = self.norm3(x)
        x = x + self.drop(self.fc2(self.drop(F.gelu(self.fc1(h3)))))
        return x


class SharedBidirectionalDecoder(nn.Module):
    """Lean 6L d=384 shared decoder supporting BOTH L2R (<BOS_L2R>) and R2L (<BOS_R2L>)."""

    def __init__(self, cfg: InkTeXV2Config):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.drop = nn.Dropout(cfg.decoder_dropout)
        self.layers = nn.ModuleList([
            DecoderBlockV2(cfg.d_model, cfg.decoder_heads, cfg.decoder_ffn_dim, cfg.decoder_dropout)
            for _ in range(cfg.decoder_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.register_buffer("alibi_slopes", alibi_slopes(cfg.decoder_heads), persistent=False)

    @torch.inference_mode()
    def decode_greedy_cached(
        self,
        memory: torch.Tensor,                        # [B, M, D]
        memory_padding_mask: torch.Tensor | None,    # [B, M] bool
        *,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_tokens: int = 200,
    ) -> torch.Tensor:
        """Fast KV-cached greedy decoder (caches both memory cross-KV and incremental self-KV)."""
        bsz, mem_len, _ = memory.shape
        device = memory.device
        dtype = memory.dtype
        num_heads = self.cfg.decoder_heads
        head_dim = self.cfg.d_model // num_heads

        # Precompute per-layer cross-attention memory K/V once
        mem_kv = [layer.project_memory(memory) for layer in self.layers]
        if memory_padding_mask is None:
            mem_bias = torch.zeros((bsz, 1, 1, mem_len), dtype=dtype, device=device)
        else:
            safe_mem = memory_padding_mask.clone()
            all_pad = safe_mem.all(dim=1, keepdim=True)
            first = torch.arange(mem_len, device=device).eq(0).unsqueeze(0)
            safe_mem = safe_mem & ~(all_pad & first)
            mem_bias = torch.zeros((bsz, 1, 1, mem_len), dtype=dtype, device=device)
            mem_bias = mem_bias.masked_fill(safe_mem[:, None, None, :], float("-inf"))

        past_k: list[torch.Tensor | None] = [None] * len(self.layers)
        past_v: list[torch.Tensor | None] = [None] * len(self.layers)
        slopes = self.alibi_slopes.to(dtype=dtype, device=device)[None, :, None, None]

        cur = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        out_tokens: list[torch.Tensor] = []

        for step in range(max_tokens - 1):
            x = self.token_embedding(cur)  # [B, 1, D]
            k_len = step + 1
            dist = torch.arange(step, -1, -1, device=device, dtype=dtype)[None, None, None, :]
            self_bias = -slopes * dist  # [1, H, 1, k_len]

            for li, layer in enumerate(self.layers):
                h = layer.norm1(x)
                qkv = layer.self_qkv(h).reshape(bsz, 1, 3, num_heads, head_dim)
                q = qkv[:, :, 0].transpose(1, 2)
                k = qkv[:, :, 1].transpose(1, 2)
                v = qkv[:, :, 2].transpose(1, 2)
                if past_k[li] is None:
                    full_k, full_v = k, v
                else:
                    full_k = torch.cat([past_k[li], k], dim=2)
                    full_v = torch.cat([past_v[li], v], dim=2)
                past_k[li], past_v[li] = full_k, full_v
                s_attn = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=self_bias)
                x = x + layer.self_out(s_attn.transpose(1, 2).reshape(bsz, 1, self.cfg.d_model))

                h2 = layer.norm2(x)
                cq = layer.cross_q(h2).reshape(bsz, 1, num_heads, head_dim).transpose(1, 2)
                mk, mv = mem_kv[li]
                c_attn = F.scaled_dot_product_attention(cq, mk, mv, attn_mask=mem_bias)
                x = x + layer.cross_out(c_attn.transpose(1, 2).reshape(bsz, 1, self.cfg.d_model))

                h3 = layer.norm3(x)
                x = x + layer.fc2(F.gelu(layer.fc1(h3)))

            logits = self.output(self.norm(x))[:, 0]
            nxt = logits.argmax(dim=-1).masked_fill(finished, pad_id)
            out_tokens.append(nxt)
            finished |= nxt.eq(eos_id)
            if bool(finished.all()):
                break
            cur = nxt[:, None]

        return torch.stack(out_tokens, dim=1) if out_tokens else torch.empty((bsz, 0), dtype=torch.long, device=device)

    def forward(
        self,
        tokens: torch.Tensor,                         # [B, T]
        memory: torch.Tensor,                         # [B, M, D]
        memory_padding_mask: torch.Tensor | None = None,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq = tokens.shape
        x = self.drop(self.token_embedding(tokens))

        # Causal ALiBi self-attention bias
        pos = torch.arange(seq, device=tokens.device)
        dist = (pos[:, None] - pos[None, :]).clamp_min(0).to(x.dtype)
        causal = pos[None, :] > pos[:, None]
        self_bias = -self.alibi_slopes.to(x.dtype)[None, :, None, None] * dist[None, None, :, :]
        self_bias = self_bias.masked_fill(causal[None, None, :, :], float("-inf"))
        if token_padding_mask is not None:
            safe_tok = token_padding_mask.clone()
            safe_tok[:, 0] = False
            key_pad = torch.zeros((bsz, 1, 1, seq), dtype=x.dtype, device=x.device)
            key_pad = key_pad.masked_fill(safe_tok[:, None, None, :], float("-inf"))
            self_bias = self_bias + key_pad

        mem_len = memory.shape[1]
        if memory_padding_mask is None:
            mem_bias = torch.zeros((bsz, 1, 1, mem_len), dtype=x.dtype, device=x.device)
        else:
            safe_mem = memory_padding_mask.clone()
            all_pad = safe_mem.all(dim=1, keepdim=True)
            first = torch.arange(mem_len, device=x.device).eq(0).unsqueeze(0)
            safe_mem = safe_mem & ~(all_pad & first)
            mem_bias = torch.zeros((bsz, 1, 1, mem_len), dtype=x.dtype, device=x.device)
            mem_bias = mem_bias.masked_fill(safe_mem[:, None, None, :], float("-inf"))

        for layer in self.layers:
            x = layer(x, memory, self_bias, mem_bias)
        return self.output(self.norm(x))


# --------------------------------------------------------------------------- Full InkTeX v2 Model


class InkTeXV2Model(nn.Module):
    def __init__(self, cfg: InkTeXV2Config):
        super().__init__()
        self.config = cfg
        self.style_encoder = WriterStyleEncoder(cfg.style_point_dim, cfg.style_dim)
        self.stroke_encoder = StrokeChunkEncoder(cfg)
        self.vision_encoder = VisionFPNEncoder(cfg)
        self.fusion = BimodalCrossFusion(cfg)
        self.decoder = SharedBidirectionalDecoder(cfg)

    def set_lora_enabled(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear) and module.lora_rank > 0:
                module.lora_enabled = enabled

    def lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.modules():
            if isinstance(module, LoRALinear) and module.lora_rank > 0:
                params.extend([module.lora_a, module.lora_b])
        return params

    @torch.no_grad()
    def merge_lora_(self) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear) and module.lora_rank > 0:
                module.merge_lora_()

    def encode(
        self,
        *,
        images: torch.Tensor | None = None,
        image_widths: torch.Tensor | None = None,
        chunk_points: torch.Tensor | None = None,
        chunk_point_mask: torch.Tensor | None = None,
        chunk_meta: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
        writer_style: torch.Tensor | None = None,
        modality_mode: str = "auto",
        vision_drop_prob: float = 0.0,
        stroke_drop_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if modality_mode not in MODALITY_MODES:
            raise ValueError(f"Unknown modality_mode: {modality_mode}")
        if modality_mode == "vision":
            chunk_points = None
        elif modality_mode == "stroke":
            images = None

        stroke_raw = None
        stroke_pair = None
        if chunk_points is not None:
            if chunk_meta is None or chunk_mask is None:
                raise ValueError("chunk_meta and chunk_mask are required with chunk_points")
            stroke_raw = self.stroke_encoder(
                chunk_points, chunk_meta, chunk_mask,
                writer_style=writer_style, chunk_point_mask=chunk_point_mask,
            )
            stroke_mem = self.fusion.project_stroke(stroke_raw)
            stroke_pair = (stroke_mem, chunk_mask)

        vision_pair = None
        if images is not None:
            v_tok, v_mask = self.vision_encoder(images, image_widths)
            vision_pair = (self.fusion.project_vision(v_tok), v_mask)

        if stroke_pair is None and vision_pair is None:
            raise ValueError("Must supply images, ink chunks, or both")
        if stroke_pair is None:
            return vision_pair[0], vision_pair[1], None
        if vision_pair is None:
            return stroke_pair[0], stroke_pair[1], stroke_raw

        s_mem, s_mask = stroke_pair
        v_mem, v_mask = vision_pair
        undropped_both = (~s_mask).any(dim=1) & (~v_mask).any(dim=1)
        if self.training and (vision_drop_prob > 0.0 or stroke_drop_prob > 0.0):
            draw = torch.rand(undropped_both.shape, device=s_mem.device)
            drop_v = undropped_both & draw.lt(vision_drop_prob)
            drop_s = undropped_both & draw.ge(vision_drop_prob) & draw.lt(vision_drop_prob + stroke_drop_prob)
            v_mask = v_mask | drop_v[:, None]
            s_mask = s_mask | drop_s[:, None]
            undropped_both = undropped_both & ~drop_v & ~drop_s

        mem, mask = self.fusion.fuse(s_mem, s_mask, v_mem, v_mask)
        return mem, mask, {
            "stroke_raw": stroke_raw,
            "stroke_mem": s_mem,
            "stroke_mask": chunk_mask,
            "undropped_both": undropped_both,
        }

    def forward(
        self,
        decoder_input: torch.Tensor,
        token_padding_mask: torch.Tensor | None = None,
        r2l_decoder_input: torch.Tensor | None = None,
        r2l_token_padding_mask: torch.Tensor | None = None,
        *,
        images: torch.Tensor | None = None,
        image_widths: torch.Tensor | None = None,
        chunk_points: torch.Tensor | None = None,
        chunk_point_mask: torch.Tensor | None = None,
        chunk_meta: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
        writer_style: torch.Tensor | None = None,
        modality_mode: str = "auto",
        vision_drop_prob: float = 0.0,
        stroke_drop_prob: float = 0.0,
    ) -> dict[str, Any]:
        memory, memory_mask, aux = self.encode(
            images=images,
            image_widths=image_widths,
            chunk_points=chunk_points,
            chunk_point_mask=chunk_point_mask,
            chunk_meta=chunk_meta,
            chunk_mask=chunk_mask,
            writer_style=writer_style,
            modality_mode=modality_mode,
            vision_drop_prob=vision_drop_prob,
            stroke_drop_prob=stroke_drop_prob,
        )
        logits = self.decoder(decoder_input, memory, memory_mask, token_padding_mask)
        r2l_logits = None
        if r2l_decoder_input is not None:
            r2l_logits = self.decoder(r2l_decoder_input, memory, memory_mask, r2l_token_padding_mask)
        stroke_raw = aux["stroke_raw"] if isinstance(aux, dict) else aux
        sym_logits = None
        if stroke_raw is not None and self.stroke_encoder.symbol_head is not None:
            sym_logits = self.stroke_encoder.symbol_head(stroke_raw)
        return {
            "logits": logits,
            "r2l_logits": r2l_logits,
            "symbol_logits": sym_logits,
            "aux": aux if isinstance(aux, dict) else None,
        }

    def parameter_summary(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            seen: set[int] = set()
            total = 0
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
            return total

        lora = sum(p.numel() for p in self.lora_parameters())
        summary = {
            "stroke": count(self.stroke_encoder),
            "vision": count(self.vision_encoder),
            "fusion": count(self.fusion),
            "decoder": count(self.decoder),
            "style": count(self.style_encoder),
            "lora_subset": lora,
            "total": count(self),
        }
        return summary
