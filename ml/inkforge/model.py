"""InkForgeModel: ~21M-param Writer-Style-Conditioned Transformer-MDN for Human Math Ink.

Architecture highlights:
  - `InkForgeStyleEncoder` (VAE, 64-d latent `z`): encodes 1..24 reference strokes from a writer
    into `(mu_z, logvar_z)` regularized with free-bits KL divergence so new writers can be sampled
    from `N(0, I)` or from a real human writer style bank, and kept identical across every line/cell
    of a multi-line proof or matrix.
  - `LatexTextEncoder` (4L RoPE Transformer, d=384): encodes expanded LaTeX + `\\text{...}` tokens.
  - `StrokeMDNDecoder` (8L Causal RoPE + Cross-Attention Transformer with per-block AdaLN/FiLM from `z`
    and full KV-cache for fast batched RTX 4070 Ti generation):
      * Inputs per step: `[dx, dy, abs_x, abs_y, prev_pen_down, prev_pen_lift, is_bos]`
        (explicit 2D canvas coords `(abs_x, abs_y)` let the model accurately place denominators under
        numerators, align closing brackets, and return to the baseline after scripts!).
      * Outputs: 20-component Bivariate Gaussian Mixture (MDN) for `(dx, dy)` + 3-way pen state
        (`0=pen_down`, `1=pen_lift`, `2=eos`).
      * MDN negative log-likelihood is strictly computed in `float32` with clamped `sigma` and `rho`
        for 100% NaN-free `bf16` training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class InkForgeConfig:
    vocab_size: int = 512
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    enc_layers: int = 4
    dec_layers: int = 8
    dropout: float = 0.10
    style_dim: int = 64
    style_max_strokes: int = 24
    style_points: int = 16
    style_point_dim: int = 8
    mdn_components: int = 20
    max_text_len: int = 160
    max_stroke_points: int = 512
    resample_spacing: float = 0.12
    coord_clip: float = 18.0
    abs_x_clip: float = 24.0
    abs_y_clip: float = 12.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InkForgeConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
# RoPE + RMSNorm + SwiGLU building blocks (fast on RTX 4070 Ti with Flash SDPA)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        return self.cos_cached[offset : offset + seq_len], self.sin_cached[offset : offset + seq_len]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to `[B, H, T, D]`."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c = cos[None, None, :, :].to(x.dtype)
    s = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(round(d_ff * 2 / 3 / 64)) * 64
        self.w12 = nn.Linear(d_model, hidden * 2, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(a) * b))


# ---------------------------------------------------------------------------
# Writer Style VAE Encoder (64-d latent `z`)
# ---------------------------------------------------------------------------

class InkForgeStyleEncoder(nn.Module):
    """Encodes a writer's reference strokes `[B, S, P, 8]` + aspect `[B, S, 4]` -> `(mu, logvar)`."""

    def __init__(self, cfg: InkForgeConfig):
        super().__init__()
        d = 128
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.style_point_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, d, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.aspect_proj = nn.Linear(4, d)
        self.stroke_norm = RMSNorm(d)
        self.stroke_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.attn_score = nn.Linear(d, 1)
        self.out_norm = RMSNorm(d * 2)
        self.to_mu = nn.Linear(d * 2, cfg.style_dim)
        self.to_logvar = nn.Linear(d * 2, cfg.style_dim)

    def forward(
        self,
        style_points: Tensor,  # [B, S, P, 8]
        style_aspect: Tensor,  # [B, S, 4]
        style_mask: Tensor,    # [B, S] bool (True = padded)
    ) -> tuple[Tensor, Tensor]:
        b, s, p, f = style_points.shape
        x = style_points.reshape(b * s, p, f).transpose(1, 2)  # [B*S, F, P]
        h = self.conv(x)  # [B*S, D, P]
        # Mean + max pool across points within each stroke
        h_stroke = (0.5 * h.mean(dim=-1) + 0.5 * h.amax(dim=-1)).reshape(b, s, -1)
        h_stroke = self.stroke_norm(h_stroke + self.aspect_proj(style_aspect))
        h_stroke = h_stroke + self.stroke_mlp(h_stroke)

        valid = (~style_mask).float()  # [B, S]
        valid_count = valid.sum(dim=1, keepdim=True)  # [B, 1]

        # Attention-weighted mean + unweighted mean over valid strokes
        scores = self.attn_score(h_stroke).squeeze(-1)
        scores = scores.masked_fill(style_mask, -1e4)
        attn = torch.softmax(scores, dim=-1) * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        pooled_attn = (h_stroke * attn.unsqueeze(-1)).sum(dim=1)
        pooled_mean = (h_stroke * valid.unsqueeze(-1)).sum(dim=1) / valid_count.clamp(min=1.0)
        pooled = self.out_norm(torch.cat([pooled_attn, pooled_mean], dim=-1))

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-6.0, 2.0)

        # If a sample has 0 valid strokes, fall back cleanly to prior N(0, I)
        has_any = (valid_count > 0.5).float()
        mu = mu * has_any
        logvar = logvar * has_any
        return mu, logvar


# ---------------------------------------------------------------------------
# LaTeX Token Encoder (4L RoPE Transformer)
# ---------------------------------------------------------------------------

class TextEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout=dropout)

    def forward(self, x: Tensor, pad_mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = apply_rope(q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = (~pad_mask)[:, None, None, :]  # [B, 1, 1, T] True = keep
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.out(y))
        x = x + self.ff(self.norm2(x))
        return x


class LatexTextEncoder(nn.Module):
    def __init__(self, cfg: InkForgeConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=cfg.max_text_len + 16)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [TextEncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.enc_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)

    def forward(self, token_ids: Tensor, pad_mask: Tensor) -> Tensor:
        t = token_ids.shape[1]
        x = self.drop(self.embed(token_ids) * math.sqrt(self.embed.embedding_dim))
        cos, sin = self.rope(t, offset=0)
        for blk in self.blocks:
            x = blk(x, pad_mask, cos, sin)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Style-Conditioned Causal Stroke Decoder with KV-Cache
# ---------------------------------------------------------------------------

class StrokeDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, style_dim: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        # Self-attention
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out1 = nn.Linear(d_model, d_model, bias=False)

        # Cross-attention to LaTeX encoder (+ prepended writer style token)
        self.norm2 = RMSNorm(d_model)
        self.q_cross = nn.Linear(d_model, d_model, bias=False)
        self.kv_cross = nn.Linear(d_model, d_model * 2, bias=False)
        self.out2 = nn.Linear(d_model, d_model, bias=False)

        # FFN
        self.norm3 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

        # AdaLN / FiLM conditioning from writer style latent `z` (zero-initialized for stable start!)
        self.style_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(style_dim, d_model * 4),
        )
        nn.init.zeros_(self.style_film[-1].weight)
        nn.init.zeros_(self.style_film[-1].bias)

    def project_cross_kv(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        b, m, _ = memory.shape
        k, v = self.kv_cross(memory).chunk(2, dim=-1)
        k = k.view(b, m, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.n_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        memory_pad_mask: Tensor,
        style_z: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        cross_kv: tuple[Tensor, Tensor] | None = None,
        need_cross_attn: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor | None]:
        b, t, d = x.shape
        gamma1, beta1, gamma3, beta3 = self.style_film(style_z).chunk(4, dim=-1)

        # 1. Causal Self-Attention (with KV cache support)
        h = self.norm1(x) * (1.0 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = apply_rope(q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        use_causal = past_kv is None and t > 1
        y1 = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=use_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        y1 = y1.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.out1(y1))

        # 2. Cross-Attention to LaTeX tokens + writer style token
        h2 = self.norm2(x)
        q2 = self.q_cross(h2).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if cross_kv is not None:
            k2, v2 = cross_kv
        else:
            k2, v2 = self.project_cross_kv(memory)

        cross_attn_weights: Tensor | None = None
        if need_cross_attn:
            # Disable autocast explicitly so float32 matmul/softmax are NOT downcast to BF16!
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                scale = 1.0 / math.sqrt(self.head_dim)
                scores = torch.matmul(q2.float(), k2.float().transpose(-2, -1)) * scale  # [B, H, T, L]
                scores = scores.masked_fill(memory_pad_mask[:, None, None, :], float("-inf"))
                attn_probs = torch.softmax(scores, dim=-1)
                cross_attn_weights = attn_probs
                attn_drop = F.dropout(attn_probs, p=self.dropout_p, training=self.training)
                y2 = torch.matmul(attn_drop, v2.float()).to(x.dtype)
        else:
            cross_mask = (~memory_pad_mask)[:, None, None, :]
            y2 = F.scaled_dot_product_attention(
                q2, k2, v2,
                attn_mask=cross_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        y2 = y2.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.out2(y2))

        # 3. Style-modulated SwiGLU FFN
        h3 = self.norm3(x) * (1.0 + gamma3.unsqueeze(1)) + beta3.unsqueeze(1)
        x = x + self.ff(h3)
        return x, new_kv, cross_attn_weights


# ---------------------------------------------------------------------------
# Graves-Style Monotonic Gaussian Attention Window (Parallel cumsum + O(1) KV-cache!)
# ---------------------------------------------------------------------------

class MonotonicGaussianWindow(nn.Module):
    """Causal monotonic Gaussian cursor over LaTeX text memory (Graves 2013 modernized for Transformers).

    Uses `cumsum(softplus(delta_kappa))` in masked log-space so:
      - Training is 100% parallel (zero RNN bottleneck) while step-by-step KV-cache is bit-for-bit identical.
      - All K windows have symmetry-broken initialization so alpha/beta/dk receive distinct gradients.
      - Text-length-aware speed scaling adapts cleanly from 3-token symbols to 80-token expressions.
      - Masked log-space normalization guarantees `phi_norm.sum(-1) == 1.0` even on extreme traces.
    """

    def __init__(self, d_model: int, num_windows: int = 4, step_scale: float = 0.25):
        super().__init__()
        self.num_windows = num_windows
        self.step_scale = step_scale
        self.norm = RMSNorm(d_model)
        self.to_params = nn.Linear(d_model, num_windows * 3)
        # Symmetry-breaking init so the K windows immediately specialize in width and speed
        nn.init.normal_(self.to_params.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.to_params.bias[:num_windows].copy_(torch.linspace(0.3, -0.3, num_windows))
            self.to_params.bias[num_windows : 2 * num_windows].copy_(torch.linspace(-0.6, 1.5, num_windows))
            self.to_params.bias[2 * num_windows :].copy_(torch.linspace(-2.2, -0.7, num_windows))
        self.text_speed = nn.Linear(d_model + 1, num_windows)
        nn.init.zeros_(self.text_speed.weight)
        nn.init.zeros_(self.text_speed.bias)
        self.ctx_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        x: Tensor,                   # [B, T, D]
        text_mem: Tensor,            # [B, L, D]
        text_pad_mask: Tensor,       # [B, L] bool (True = padded)
        prev_kappa: Tensor | None = None,  # [B, K] for step-by-step generation
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns `(x_out [B, T, D], kappa [B, T, K], phi_norm [B, T, L], active_kappa [B, T])`."""
        l = text_mem.shape[1]

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_norm = self.norm(x).float()
            params = self.to_params(x_norm)  # [B, T, 3K]
            raw_alpha, raw_beta, raw_dk = params.chunk(3, dim=-1)

            # Text-length-aware speed prior (strictly constant across T -> exact KV-cache parity)
            valid_mem = (~text_pad_mask).float()  # [B, L]
            text_lens = valid_mem.sum(dim=1).clamp(min=2.0)  # [B]
            text_summary = (text_mem.float() * valid_mem.unsqueeze(-1)).sum(dim=1) / text_lens.unsqueeze(-1)
            len_feat = torch.log(text_lens / 20.0).unsqueeze(-1)  # [B, 1]
            length_prior = (text_lens / 20.0).clamp(min=0.35, max=4.0)[:, None, None]  # [B, 1, 1]
            speed_mult = (
                0.5 + F.softplus(self.text_speed(torch.cat([text_summary, len_feat], dim=-1))) / 1.3862944
            )[:, None, :]  # [B, 1, K], starts at exactly 1.0

            log_alpha = F.log_softmax(raw_alpha, dim=-1)                           # [B, T, K]
            alpha = log_alpha.exp()                                                # [B, T, K]
            beta = F.softplus(raw_beta).clamp(min=0.05, max=12.0)                  # [B, T, K]
            dk = F.softplus(raw_dk) * self.step_scale * length_prior * speed_mult  # [B, T, K]

            if prev_kappa is None:
                kappa = torch.cumsum(dk, dim=1)                                    # [B, T, K]
            else:
                kappa = prev_kappa.unsqueeze(1) + torch.cumsum(dk, dim=1)          # [B, T, K]

            u = torch.arange(l, device=x.device, dtype=torch.float32)[None, None, None, :]  # [1, 1, 1, L]
            diff = kappa.unsqueeze(-1) - u                                         # [B, T, K, L]
            # Masked log-space Gaussian mixture: never underflows even when kappa >> L
            log_phi_k = log_alpha.unsqueeze(-1) - beta.unsqueeze(-1) * (diff * diff)
            log_phi = torch.logsumexp(log_phi_k, dim=2)                            # [B, T, L]
            log_phi = log_phi.masked_fill(text_pad_mask[:, None, :], float("-inf"))
            phi_norm = torch.softmax(log_phi, dim=-1)                              # [B, T, L]

            active_kappa = (alpha * kappa).sum(dim=-1)                             # [B, T]
            ctx = torch.matmul(phi_norm, text_mem.float()).to(x.dtype)             # [B, T, D]

        x_out = x + self.gate.to(x.dtype) * self.ctx_proj(ctx)
        return x_out, kappa, phi_norm, active_kappa


# ---------------------------------------------------------------------------
# Full InkForgeModel
# ---------------------------------------------------------------------------

class InkForgeModel(nn.Module):
    """Writer-style-conditioned neural handwriting generator (~27M params)."""

    INPUT_DIM = 7  # [dx, dy, abs_x, abs_y, pen_down, pen_lift, is_bos]

    def __init__(self, cfg: InkForgeConfig):
        super().__init__()
        self.cfg = cfg
        self.style_encoder = InkForgeStyleEncoder(cfg)
        self.text_encoder = LatexTextEncoder(cfg)

        self.stroke_in = nn.Sequential(
            nn.Linear(self.INPUT_DIM, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.dec_rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=cfg.max_stroke_points + 64)
        self.monotonic_window = MonotonicGaussianWindow(cfg.d_model, num_windows=4, step_scale=0.25)
        self.dec_blocks = nn.ModuleList(
            [
                StrokeDecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.style_dim, cfg.dropout)
                for _ in range(cfg.dec_layers)
            ]
        )
        self.dec_norm = RMSNorm(cfg.d_model)

        # Output head: M*(6) MDN params (pi, mu_x, mu_y, raw_sx, raw_sy, raw_rho) + 3 pen logits
        self.mdn_head = nn.Linear(cfg.d_model, cfg.mdn_components * 6 + 3)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_style(
        self,
        style_points: Tensor,
        style_aspect: Tensor,
        style_mask: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.style_encoder(style_points, style_aspect, style_mask)
        has_any = (~style_mask).any(dim=1, keepdim=True).to(mu.dtype)
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            z = (mu + std * torch.randn_like(std)) * has_any
        else:
            z = mu * has_any
        return z, mu, logvar

    def encode_memory(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode strictly the LaTeX tokens (NO style_tok in cross-attention memory!)."""
        pad_mask = token_ids.eq(self.cfg.pad_id)
        text_mem = self.text_encoder(token_ids, pad_mask)  # [B, L, D]
        return text_mem, pad_mask

    def decode_features(
        self,
        stroke_in_feats: Tensor,  # [B, T, 7]
        memory: Tensor,
        mem_mask: Tensor,
        style_z: Tensor,
        *,
        return_guided_attns: bool = False,
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        t = stroke_in_feats.shape[1]
        x = self.stroke_in(stroke_in_feats)
        cos, sin = self.dec_rope(t, offset=0)
        guided_attns: list[Tensor] = []
        guide_start = max(1, len(self.dec_blocks) // 4)
        active_kappa_seq: Tensor | None = None
        for idx, blk in enumerate(self.dec_blocks):
            want = return_guided_attns and (idx >= guide_start)
            x, _, attn_w = blk(x, memory, mem_mask, style_z, cos, sin, need_cross_attn=want)
            if idx == 0:
                x, _, _, active_kappa_seq = self.monotonic_window(x, memory, mem_mask, prev_kappa=None)
            if attn_w is not None:
                guided_attns.append(attn_w)
        assert active_kappa_seq is not None
        return self.mdn_head(self.dec_norm(x)), guided_attns, active_kappa_seq

    @staticmethod
    def _guided_diagonal_attn_loss(
        attn_list: list[Tensor],
        stroke_mask: Tensor,    # [B, T] bool (True = valid)
        text_pad_mask: Tensor,  # [B, L] bool (True = padded)
        *,
        sigma: float = 0.24,
    ) -> tuple[Tensor, Tensor]:
        """Soft diagonal alignment guidance directly over the L text tokens (impossible to hack!).

        Returns `(guided_loss, mean_max_text_attn)` for direct telemetry monitoring.
        """
        if not attn_list:
            z = torch.zeros((), device=stroke_mask.device, dtype=torch.float32)
            return z, z
        b, t = stroke_mask.shape
        l = text_pad_mask.shape[1]
        t_lens = stroke_mask.float().sum(dim=1).clamp(min=2.0) - 1.0  # [B]
        l_lens = (~text_pad_mask).float().sum(dim=1).clamp(min=2.0) - 1.0  # [B]
        t_pos = torch.arange(t, device=stroke_mask.device, dtype=torch.float32)[None, :, None] / t_lens[:, None, None]
        l_pos = torch.arange(l, device=stroke_mask.device, dtype=torch.float32)[None, None, :] / l_lens[:, None, None]
        w_guide = 1.0 - torch.exp(-((t_pos - l_pos) ** 2) / (2.0 * sigma * sigma))  # [B, T, L]
        valid = stroke_mask[:, :, None] & (~text_pad_mask)[:, None, :]  # [B, T, L]
        w_masked = (w_guide * valid.float()).unsqueeze(1)  # [B, 1, T, L]

        n_heads = attn_list[0].shape[1]
        n_guided = max(1, n_heads // 2)
        denom = (stroke_mask.float().sum() * float(n_guided)).clamp(min=1.0)
        valid_t = stroke_mask[:, None, :].float()

        total_loss = torch.zeros((), device=stroke_mask.device, dtype=torch.float32)
        total_max = torch.zeros((), device=stroke_mask.device, dtype=torch.float32)
        for attn in attn_list:
            # All L columns are real LaTeX tokens summing to 1.0 (no style_tok escape hatch!)
            a_guided = attn[:, :n_guided, :, :].float()
            total_loss = total_loss + (a_guided * w_masked).sum() / denom
            total_max = total_max + (a_guided.amax(dim=-1) * valid_t).sum() / denom
        n_layers = float(len(attn_list))
        return total_loss / n_layers, total_max / n_layers

    def split_mdn_params(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Split raw head output `[B, T, 6M+3]` strictly in float32 for numerical stability."""
        m = self.cfg.mdn_components
        raw_f = raw.float()
        pi_logits = raw_f[..., :m]
        mu_x = raw_f[..., m : 2 * m]
        mu_y = raw_f[..., 2 * m : 3 * m]
        sigma_x = F.softplus(raw_f[..., 3 * m : 4 * m]).clamp(min=1e-3, max=10.0)
        sigma_y = F.softplus(raw_f[..., 4 * m : 5 * m]).clamp(min=1e-3, max=10.0)
        rho = 0.95 * torch.tanh(raw_f[..., 5 * m : 6 * m])
        pen_logits = raw_f[..., 6 * m : 6 * m + 3]
        return pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho, pen_logits

    def compute_loss(
        self,
        *,
        token_ids: Tensor,        # [B, L]
        stroke_inputs: Tensor,    # [B, T, 7]
        target_dx: Tensor,        # [B, T]
        target_dy: Tensor,        # [B, T]
        target_pen: Tensor,       # [B, T] long in {0: down, 1: lift, 2: eos}
        stroke_mask: Tensor,      # [B, T] bool (True = valid target step)
        style_points: Tensor,     # [B, S, P, 8]
        style_aspect: Tensor,     # [B, S, 4]
        style_mask: Tensor,       # [B, S] bool (True = padded)
        kl_weight: float = 0.025,
        free_bits: float = 0.15,
        style_drop_prob: float = 0.30,
        guided_attn_weight: float = 4.0,
        guided_attn_sigma: float = 0.24,
        transition_weight: float = 4.0,
        cursor_end_weight: float = 0.5,
    ) -> dict[str, Tensor]:
        z, mu, logvar = self.encode_style(style_points, style_aspect, style_mask, sample=True)
        has_style = (~style_mask).any(dim=1)  # [B] bool

        # Classifier-free style dropout on styled rows so LaTeX text always dominates over z
        if self.training and style_drop_prob > 0:
            keep = (torch.rand((z.shape[0], 1), device=z.device) >= style_drop_prob).to(z.dtype)
            z = z * keep

        memory, mem_mask = self.encode_memory(token_ids)
        raw, guided_attns, active_kappa_seq = self.decode_features(
            stroke_inputs,
            memory,
            mem_mask,
            z,
            return_guided_attns=True,
        )
        pi_logits, mu_x, mu_y, sx, sy, rho, pen_logits = self.split_mdn_params(raw)

        # Bivariate Gaussian Mixture NLL in float32
        dx = target_dx.float().unsqueeze(-1)  # [B, T, 1]
        dy = target_dy.float().unsqueeze(-1)
        nx = (dx - mu_x) / sx
        ny = (dy - mu_y) / sy
        one_minus_rho2 = (1.0 - rho * rho).clamp(min=1e-4)
        z_quad = (nx * nx + ny * ny - 2.0 * rho * nx * ny) / one_minus_rho2
        log_norm = (
            math.log(2.0 * math.pi)
            + torch.log(sx)
            + torch.log(sy)
            + 0.5 * torch.log(one_minus_rho2)
        )
        log_comp = -0.5 * z_quad - log_norm
        log_pi = F.log_softmax(pi_logits, dim=-1)
        mdn_nll = -torch.logsumexp(log_pi + log_comp, dim=-1)  # [B, T]

        # Upweight stroke-start / pen-lift air-jump steps so the decoder focuses on symbol placement & shapes
        # instead of spending 94% of MDN gradient on smooth intra-stroke momentum continuations!
        valid_f = stroke_mask.float()
        coord_mask = (stroke_mask & target_pen.ne(2)).float()
        is_bos = stroke_inputs[:, :, 6] > 0.5
        is_lift_jump = stroke_inputs[:, :, 5] > 0.5
        is_jump = is_bos | is_lift_jump
        is_first_drawn = torch.zeros_like(is_lift_jump)
        if is_lift_jump.shape[1] > 1:
            is_first_drawn[:, 1:] = is_lift_jump[:, :-1]
        step_w = torch.ones_like(coord_mask)
        step_w = torch.where(is_first_drawn, torch.full_like(step_w, 2.0), step_w)
        step_w = torch.where(is_jump, torch.full_like(step_w, transition_weight), step_w)
        weighted_coord = coord_mask * step_w
        weighted_coord_sum = weighted_coord.sum()
        coord_count = coord_mask.sum()
        mdn_loss = (mdn_nll * weighted_coord).sum() / weighted_coord_sum.clamp(min=1.0)
        mdn_unweighted = (mdn_nll * coord_mask).sum() / coord_count.clamp(min=1.0)

        jump_mask = coord_mask * is_jump.float()
        draw_mask = coord_mask * (~is_jump).float()
        jump_count = jump_mask.sum()
        draw_count = draw_mask.sum()
        jump_mdn_loss = (mdn_nll * jump_mask).sum() / jump_count.clamp(min=1.0)
        draw_mdn_loss = (mdn_nll * draw_mask).sum() / draw_count.clamp(min=1.0)

        # UNWEIGHTED calibrated pen-state cross-entropy (class weights bias EOS/lift sampling probabilities!)
        pen_ce = F.cross_entropy(
            pen_logits.reshape(-1, 3),
            target_pen.reshape(-1),
            reduction="none",
        ).view_as(valid_f)
        valid_count = valid_f.sum()
        pen_loss = (pen_ce * valid_f).sum() / valid_count.clamp(min=1.0)

        # Guided diagonal cross-attention loss strictly over the L LaTeX tokens
        guided_loss, attn_max_txt = self._guided_diagonal_attn_loss(
            guided_attns, stroke_mask, mem_mask, sigma=guided_attn_sigma
        )

        # End-of-sequence active cursor supervision: encourage active weighted kappa to reach L-1 at EOS
        last_t = (stroke_mask.long().sum(dim=1) - 1).clamp(min=0)  # [B]
        ar = torch.arange(token_ids.shape[0], device=token_ids.device)
        end_kappa = active_kappa_seq[ar, last_t]  # [B]
        target_end = ((~mem_mask).float().sum(dim=1) - 1.0).clamp(min=1.0)
        cursor_ratio = end_kappa / target_end
        cursor_end_loss = F.smooth_l1_loss(cursor_ratio, torch.ones_like(target_end))

        # Free-bits KL divergence strictly over rows that actually have real peer writer strokes
        if has_style.any():
            kl_per_dim = -0.5 * (
                1.0 + logvar[has_style].float() - mu[has_style].float().pow(2) - logvar[has_style].float().exp()
            )
            kl_dim_mean = kl_per_dim.mean(dim=0)
            kl_loss = torch.clamp(kl_dim_mean, min=free_bits).sum()
            kl_raw = kl_dim_mean.sum()
        else:
            kl_loss = torch.zeros((), device=mdn_loss.device, dtype=torch.float32)
            kl_raw = torch.zeros((), device=mdn_loss.device, dtype=torch.float32)

        total_loss = (
            mdn_loss
            + 2.0 * pen_loss
            + guided_attn_weight * guided_loss
            + cursor_end_weight * cursor_end_loss
            + kl_weight * kl_loss
        )

        with torch.no_grad():
            pen_pred = pen_logits.argmax(dim=-1)
            pen_correct = ((pen_pred == target_pen) & stroke_mask).float().sum()
            pen_acc = pen_correct / valid_count.clamp(min=1.0)
            cursor_progress = cursor_ratio.mean()
            cursor_overshoot = (
                ((active_kappa_seq > (target_end[:, None] + 1.0)) & stroke_mask).float().sum()
                / valid_count.clamp(min=1.0)
            )
            cursor_gate = self.monotonic_window.gate.detach().abs().float()
            # Sigma-floor occupancy on dominant mixture component
            top_comp = pi_logits.argmax(dim=-1, keepdim=True)
            top_sx = sx.gather(-1, top_comp).squeeze(-1)
            top_sy = sy.gather(-1, top_comp).squeeze(-1)
            sigma_floor = (((top_sx <= 2e-3) | (top_sy <= 2e-3)) & (coord_mask > 0.5)).float().sum()
            sigma_floor_frac = sigma_floor / coord_count.clamp(min=1.0)
            val_score = mdn_loss + 2.0 * pen_loss

        return {
            "loss": total_loss,
            "val_score": val_score.detach(),
            "mdn_loss": mdn_unweighted.detach(),
            "mdn_w_loss": mdn_loss.detach(),
            "draw_mdn_loss": draw_mdn_loss.detach(),
            "jump_mdn_loss": jump_mdn_loss.detach(),
            "pen_loss": pen_loss.detach(),
            "guided_loss": guided_loss.detach(),
            "attn_max_txt": attn_max_txt.detach(),
            "cursor_end_loss": cursor_end_loss.detach(),
            "cursor_progress": cursor_progress.detach(),
            "cursor_overshoot": cursor_overshoot.detach(),
            "cursor_gate": cursor_gate,
            "sigma_floor_frac": sigma_floor_frac.detach(),
            "kl_raw": kl_raw.detach(),
            "pen_acc": pen_acc.detach(),
            # Population denominators for exact validation aggregation
            "valid_count": valid_count.detach(),
            "coord_count": coord_count.detach(),
            "weighted_coord_sum": weighted_coord_sum.detach(),
            "draw_count": draw_count.detach(),
            "jump_count": jump_count.detach(),
        }

    # -----------------------------------------------------------------------
    # Fast Batched KV-Cached Generation (RTX 4070 Ti: ~1,500+ eqs/min!)
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def generate_batch(
        self,
        token_ids: Tensor,             # [B, L]
        style_z: Tensor | None = None, # [B, style_dim] (zeros=neutral if None)
        *,
        temperature: float | Tensor = 0.40,
        pen_temperature: float = 1.0,
        max_steps: int = 512,
        min_steps: int = 1,
        return_meta: bool = False,
    ) -> list[list[np.ndarray]] | tuple[list[list[np.ndarray]], list[dict[str, Any]]]:
        """Generate batched handwritten 2D strokes in glyph-normalized units (h = 1.0)."""
        device = token_ids.device
        b = token_ids.shape[0]
        if style_z is None:
            style_z = torch.zeros((b, self.cfg.style_dim), device=device)
        else:
            style_z = style_z.to(device)

        if isinstance(temperature, (int, float)):
            temp = torch.full((b,), float(temperature), device=device, dtype=torch.float32)
        else:
            temp = temperature.to(device=device, dtype=torch.float32).view(b)

        memory, mem_mask = self.encode_memory(token_ids)
        # Precompute cross-attention K/V once per batch (massive speedup!)
        cross_kvs = [blk.project_cross_kv(memory) for blk in self.dec_blocks]
        past_kvs: list[tuple[Tensor, Tensor] | None] = [None] * len(self.dec_blocks)
        kappa_state = torch.zeros((b, self.monotonic_window.num_windows), device=device, dtype=torch.float32)

        # Initial BOS stroke token: (dx=0, dy=0, abs_x=0, abs_y=0, down=0, lift=0, is_bos=1)
        cur_feat = torch.zeros((b, 1, self.INPUT_DIM), device=device, dtype=memory.dtype)
        cur_feat[:, 0, 6] = 1.0

        abs_x = torch.zeros((b,), device=device, dtype=torch.float32)
        abs_y = torch.zeros((b,), device=device, dtype=torch.float32)
        finished = torch.zeros((b,), device=device, dtype=torch.bool)
        # Training grammar invariant: every trace ends with pen_lift (1) right before EOS (2).
        # Mask EOS whenever the current stroke is still open (`~can_eos`).
        can_eos = torch.zeros((b,), device=device, dtype=torch.bool)
        steps_taken = torch.zeros((b,), device=device, dtype=torch.long)

        # Record trajectories on GPU then unpack once on CPU
        xs_hist = torch.zeros((b, max_steps), device=device, dtype=torch.float32)
        ys_hist = torch.zeros((b, max_steps), device=device, dtype=torch.float32)
        pen_hist = torch.full((b, max_steps), 2, device=device, dtype=torch.long)

        for step in range(max_steps):
            x = self.stroke_in(cur_feat)
            cos, sin = self.dec_rope(1, offset=step)
            for idx, blk in enumerate(self.dec_blocks):
                x, past_kvs[idx], _ = blk(
                    x,
                    memory,
                    mem_mask,
                    style_z,
                    cos,
                    sin,
                    past_kv=past_kvs[idx],
                    cross_kv=cross_kvs[idx],
                    need_cross_attn=False,
                )
                if idx == 0:
                    x, kappa_step, _, _ = self.monotonic_window(x, memory, mem_mask, prev_kappa=kappa_state)
                    kappa_state = kappa_step[:, 0, :]
            raw = self.mdn_head(self.dec_norm(x))
            pi_logits, mu_x, mu_y, sx, sy, rho, pen_logits = self.split_mdn_params(raw)
            pi_logits = pi_logits[:, 0, :]  # [B, M]
            mu_x = mu_x[:, 0, :]
            mu_y = mu_y[:, 0, :]
            sx = sx[:, 0, :]
            sy = sy[:, 0, :]
            rho = rho[:, 0, :]
            pen_logits = pen_logits[:, 0, :]  # [B, 3]

            # Forbid EOS before min_steps OR while a stroke is still open (require pen_lift before EOS!)
            forbid_eos = (~finished) & ((step < min_steps) | (~can_eos))
            if forbid_eos.any():
                pen_logits = pen_logits.clone()
                pen_logits[forbid_eos, 2] = -1e4

            # Sample pen state
            pen_probs = torch.softmax(pen_logits / max(pen_temperature, 1e-3), dim=-1)
            pen = torch.multinomial(pen_probs, num_samples=1).squeeze(-1)  # [B]
            pen = torch.where(finished, torch.full_like(pen, 2), pen)

            # Sample mixture component (temperature-sharpened pi + scaled std — Graves 2013 bias trick)
            pi_probs = torch.softmax(pi_logits / temp.clamp(min=0.05).unsqueeze(-1), dim=-1)
            comp = torch.multinomial(pi_probs, num_samples=1).squeeze(-1)  # [B]
            ar = torch.arange(b, device=device)

            mx = mu_x[ar, comp]
            my = mu_y[ar, comp]
            sig_x = sx[ar, comp] * temp.sqrt()
            sig_y = sy[ar, comp] * temp.sqrt()
            r = rho[ar, comp]

            eps1 = torch.randn((b,), device=device)
            eps2 = torch.randn((b,), device=device)
            dx = (mx + sig_x * eps1).clamp(-self.cfg.coord_clip, self.cfg.coord_clip)
            dy = (my + sig_y * (r * eps1 + torch.sqrt((1.0 - r * r).clamp(min=1e-4)) * eps2)).clamp(
                -self.cfg.coord_clip, self.cfg.coord_clip
            )

            # Update canvas coordinates for active items
            active = ~finished & pen.ne(2)
            abs_x = torch.where(active, abs_x + dx, abs_x)
            abs_y = torch.where(active, abs_y + dy, abs_y)
            steps_taken = steps_taken + (~finished).long()

            xs_hist[:, step] = abs_x
            ys_hist[:, step] = abs_y
            pen_hist[:, step] = pen

            can_eos = pen.eq(1)
            finished = finished | pen.eq(2)
            if finished.all():
                break

            # Prepare next-step input features
            next_feat = torch.zeros((b, 1, self.INPUT_DIM), device=device, dtype=memory.dtype)
            next_feat[:, 0, 0] = dx.to(memory.dtype)
            next_feat[:, 0, 1] = dy.to(memory.dtype)
            next_feat[:, 0, 2] = abs_x.clamp(-self.cfg.abs_x_clip, self.cfg.abs_x_clip).to(memory.dtype)
            next_feat[:, 0, 3] = abs_y.clamp(-self.cfg.abs_y_clip, self.cfg.abs_y_clip).to(memory.dtype)
            next_feat[:, 0, 4] = pen.eq(0).to(memory.dtype)
            next_feat[:, 0, 5] = pen.eq(1).to(memory.dtype)
            cur_feat = next_feat

        xs_np = xs_hist.cpu().numpy()
        ys_np = ys_hist.cpu().numpy()
        pen_np = pen_hist.cpu().numpy()
        finished_np = finished.cpu().numpy()
        steps_np = steps_taken.cpu().numpy()

        results: list[list[np.ndarray]] = []
        meta_list: list[dict[str, Any]] = []
        for i in range(b):
            strokes: list[np.ndarray] = []
            cur_pts: list[tuple[float, float]] = [(0.0, 0.0)]
            for s in range(max_steps):
                p = int(pen_np[i, s])
                if p == 2:
                    break
                cur_pts.append((float(xs_np[i, s]), float(ys_np[i, s])))
                if p == 1:
                    # End of stroke: flush current stroke, next point starts a new stroke
                    if len(cur_pts) >= 1:
                        strokes.append(np.asarray(cur_pts, dtype=np.float64))
                    cur_pts = []
            if cur_pts:
                strokes.append(np.asarray(cur_pts, dtype=np.float64))
            # Filter tiny phantom leading point if first stroke was immediately a pen-lift jump
            if len(strokes) > 1 and len(strokes[0]) == 1 and float(np.linalg.norm(strokes[0][0])) < 1e-6:
                strokes = strokes[1:]
            results.append(strokes)
            is_fin = bool(finished_np[i])
            meta_list.append({
                "finished": is_fin,
                "hit_cap": not is_fin,
                "steps": int(steps_np[i]),
            })
        if return_meta:
            return results, meta_list
        return results
