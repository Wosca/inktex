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
    coord_clip: float = 12.0
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
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
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
        return x, new_kv


# ---------------------------------------------------------------------------
# Full InkForgeModel
# ---------------------------------------------------------------------------

class InkForgeModel(nn.Module):
    """Writer-style-conditioned neural handwriting generator (~21M params)."""

    INPUT_DIM = 7  # [dx, dy, abs_x, abs_y, pen_down, pen_lift, is_bos]

    def __init__(self, cfg: InkForgeConfig):
        super().__init__()
        self.cfg = cfg
        self.style_encoder = InkForgeStyleEncoder(cfg)
        self.text_encoder = LatexTextEncoder(cfg)
        self.style_to_mem = nn.Linear(cfg.style_dim, cfg.d_model)

        self.stroke_in = nn.Sequential(
            nn.Linear(self.INPUT_DIM, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.dec_rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=cfg.max_stroke_points + 64)
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
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar

    def encode_memory(self, token_ids: Tensor, style_z: Tensor) -> tuple[Tensor, Tensor]:
        pad_mask = token_ids.eq(self.cfg.pad_id)
        text_mem = self.text_encoder(token_ids, pad_mask)  # [B, L, D]
        style_tok = self.style_to_mem(style_z).unsqueeze(1)  # [B, 1, D]
        memory = torch.cat([style_tok, text_mem], dim=1)  # [B, 1+L, D]
        zero_mask = torch.zeros((token_ids.shape[0], 1), dtype=torch.bool, device=token_ids.device)
        mem_mask = torch.cat([zero_mask, pad_mask], dim=1)
        return memory, mem_mask

    def decode_features(
        self,
        stroke_in_feats: Tensor,  # [B, T, 7]
        memory: Tensor,
        mem_mask: Tensor,
        style_z: Tensor,
    ) -> Tensor:
        t = stroke_in_feats.shape[1]
        x = self.stroke_in(stroke_in_feats)
        cos, sin = self.dec_rope(t, offset=0)
        for blk in self.dec_blocks:
            x, _ = blk(x, memory, mem_mask, style_z, cos, sin)
        return self.mdn_head(self.dec_norm(x))

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
        kl_weight: float = 0.01,
        free_bits: float = 0.25,
        style_drop_prob: float = 0.10,
    ) -> dict[str, Tensor]:
        z, mu, logvar = self.encode_style(style_points, style_aspect, style_mask, sample=True)

        # Occasional classifier-free style dropout so z=0 works cleanly as neutral handwriting
        if self.training and style_drop_prob > 0:
            keep = (torch.rand((z.shape[0], 1), device=z.device) >= style_drop_prob).to(z.dtype)
            z = z * keep

        memory, mem_mask = self.encode_memory(token_ids, z)
        raw = self.decode_features(stroke_inputs, memory, mem_mask, z)
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

        # Mask MDN loss to valid non-EOS steps (at EOS step, only pen_state=2 matters)
        valid_f = stroke_mask.float()
        coord_mask = (stroke_mask & target_pen.ne(2)).float()
        mdn_loss = (mdn_nll * coord_mask).sum() / coord_mask.sum().clamp(min=1.0)

        # Pen-state cross-entropy (upweight pen_lift & eos slightly to prevent missed lifts/runaways)
        pen_weights = torch.tensor([1.0, 2.0, 4.0], device=pen_logits.device, dtype=torch.float32)
        pen_ce = F.cross_entropy(
            pen_logits.reshape(-1, 3),
            target_pen.reshape(-1),
            weight=pen_weights,
            reduction="none",
        ).view_as(valid_f)
        pen_loss = (pen_ce * valid_f).sum() / valid_f.sum().clamp(min=1.0)

        # Free-bits KL divergence per latent dimension (prevents posterior collapse!)
        kl_per_dim = -0.5 * (1.0 + logvar.float() - mu.float().pow(2) - logvar.float().exp())  # [B, D_z]
        kl_dim_mean = kl_per_dim.mean(dim=0)  # [D_z]
        kl_loss = torch.clamp(kl_dim_mean, min=free_bits).sum()
        kl_raw = kl_dim_mean.sum()

        total_loss = mdn_loss + 1.5 * pen_loss + kl_weight * kl_loss

        with torch.no_grad():
            pen_pred = pen_logits.argmax(dim=-1)
            pen_acc = ((pen_pred == target_pen) & stroke_mask).float().sum() / valid_f.sum().clamp(min=1.0)

        return {
            "loss": total_loss,
            "mdn_loss": mdn_loss.detach(),
            "pen_loss": pen_loss.detach(),
            "kl_raw": kl_raw.detach(),
            "pen_acc": pen_acc.detach(),
        }

    # -----------------------------------------------------------------------
    # Fast Batched KV-Cached Generation (RTX 4070 Ti: ~1,500+ eqs/min!)
    # -----------------------------------------------------------------------

    @torch.inference_mode()
    def generate_batch(
        self,
        token_ids: Tensor,             # [B, L]
        style_z: Tensor | None = None, # [B, style_dim] (sampled from N(0, I) if None)
        *,
        temperature: float | Tensor = 0.45,
        pen_temperature: float = 0.70,
        max_steps: int = 480,
        min_steps: int = 6,
    ) -> list[list[np.ndarray]]:
        """Generate batched handwritten 2D strokes in glyph-normalized units (h = 1.0)."""
        device = token_ids.device
        b = token_ids.shape[0]
        if style_z is None:
            style_z = torch.randn((b, self.cfg.style_dim), device=device)
        else:
            style_z = style_z.to(device)

        if isinstance(temperature, (int, float)):
            temp = torch.full((b,), float(temperature), device=device, dtype=torch.float32)
        else:
            temp = temperature.to(device=device, dtype=torch.float32).view(b)

        memory, mem_mask = self.encode_memory(token_ids, style_z)
        # Precompute cross-attention K/V once per batch (massive speedup!)
        cross_kvs = [blk.project_cross_kv(memory) for blk in self.dec_blocks]
        past_kvs: list[tuple[Tensor, Tensor] | None] = [None] * len(self.dec_blocks)

        # Initial BOS stroke token: (dx=0, dy=0, abs_x=0, abs_y=0, down=0, lift=0, is_bos=1)
        cur_feat = torch.zeros((b, 1, self.INPUT_DIM), device=device, dtype=memory.dtype)
        cur_feat[:, 0, 6] = 1.0

        abs_x = torch.zeros((b,), device=device, dtype=torch.float32)
        abs_y = torch.zeros((b,), device=device, dtype=torch.float32)
        finished = torch.zeros((b,), device=device, dtype=torch.bool)

        # Record trajectories on GPU then unpack once on CPU
        xs_hist = torch.zeros((b, max_steps), device=device, dtype=torch.float32)
        ys_hist = torch.zeros((b, max_steps), device=device, dtype=torch.float32)
        pen_hist = torch.full((b, max_steps), 2, device=device, dtype=torch.long)

        for step in range(max_steps):
            x = self.stroke_in(cur_feat)
            cos, sin = self.dec_rope(1, offset=step)
            for idx, blk in enumerate(self.dec_blocks):
                x, past_kvs[idx] = blk(
                    x,
                    memory,
                    mem_mask,
                    style_z,
                    cos,
                    sin,
                    past_kv=past_kvs[idx],
                    cross_kv=cross_kvs[idx],
                )
            raw = self.mdn_head(self.dec_norm(x))
            pi_logits, mu_x, mu_y, sx, sy, rho, pen_logits = self.split_mdn_params(raw)
            pi_logits = pi_logits[:, 0, :]  # [B, M]
            mu_x = mu_x[:, 0, :]
            mu_y = mu_y[:, 0, :]
            sx = sx[:, 0, :]
            sy = sy[:, 0, :]
            rho = rho[:, 0, :]
            pen_logits = pen_logits[:, 0, :]  # [B, 3]

            # Prevent premature EOS before min_steps
            if step < min_steps:
                pen_logits = pen_logits.clone()
                pen_logits[:, 2] = -1e4

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

            xs_hist[:, step] = abs_x
            ys_hist[:, step] = abs_y
            pen_hist[:, step] = pen

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

        results: list[list[np.ndarray]] = []
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
        return results
