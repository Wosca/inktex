"""Causal stroke MDN with split reading-order and 2D cross-attention heads."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..model import (
    InkForgeConfig,
    InkForgeModel,
    InkForgeStyleEncoder,
    LatexTextEncoder,
    RMSNorm,
    RotaryEmbedding,
    StrokeDecoderBlock,
    apply_rope,
)


_SPATIAL_TOKENS = (
    frozenset(("^", "_")),
    frozenset(r"\frac \dfrac \tfrac \binom \dbinom \tbinom \sqrt \root \stackrel \overset \underset".split()),
    frozenset(r"\sum \int \iint \iiint \oint \prod \coprod \lim \limsup \liminf \bigcup \bigcap".split()),
    frozenset(r"\bar \overline \underline \hat \widehat \tilde \widetilde \vec \dot \ddot \overbrace \underbrace".split()),
)
_TOKEN_ROLES = {token: role for role, tokens in enumerate(_SPATIAL_TOKENS, 1) for token in tokens}
_NUM_ROLES = len(_SPATIAL_TOKENS) + 1
_CANVAS_DIM = 8


@dataclass
class InkForgeV3Config(InkForgeConfig):
    vocab_size: int = 600
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    enc_layers: int = 6
    dec_layers: int = 13
    abs_x_clip: float = 36.0
    abs_y_clip: float = 16.0
    sigma_min: float = 0.008
    token_roles: tuple[int, ...] = ()

    @staticmethod
    def roles_for_tokens(tokens: Sequence[str]) -> tuple[int, ...]:
        return tuple(_TOKEN_ROLES.get(token, 0) for token in tokens)


KV = tuple[Tensor, Tensor]


class _SpatialDecoderBlock(StrokeDecoderBlock):
    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        style_z: Tensor,
        cos: Tensor,
        sin: Tensor,
        cross_bias: Tensor,
        cross_kv: KV | None = None,
        cache: KV | None = None,
        offset: int = 0,
        guide: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch, steps, width = x.shape
        heads = self.n_heads
        head_dim = self.head_dim
        g1, b1, g3, b3 = self.style_film(style_z).unsqueeze(1).chunk(4, dim=-1)
        dropout = self.dropout_p if self.training else 0.0

        q, k, v = self.qkv(self.norm1(x) * (1.0 + g1) + b1).chunk(3, dim=-1)
        q = apply_rope(q.view(batch, steps, heads, head_dim).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(batch, steps, heads, head_dim).transpose(1, 2), cos, sin)
        v = v.view(batch, steps, heads, head_dim).transpose(1, 2)
        if cache is not None:
            k_cache, v_cache = cache
            k_cache[:, :, offset : offset + 1].copy_(k)
            v_cache[:, :, offset : offset + 1].copy_(v)
            k = k_cache[:, :, : offset + 1]
            v = v_cache[:, :, : offset + 1]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None, dropout_p=dropout)
        x = x + self.drop(self.out1(y.transpose(1, 2).reshape(batch, steps, width)))

        q = self.q_cross(self.norm2(x)).view(batch, steps, heads, head_dim).transpose(1, 2)
        k, v = self.project_cross_kv(memory) if cross_kv is None else cross_kv
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=cross_bias, dropout_p=dropout)
        guide_weights = None
        if guide:
            n = heads // 2
            scores = (q[:, :n] @ k[:, :n].transpose(-2, -1)).float() * (head_dim ** -0.5)
            guide_weights = (scores + cross_bias[:, :n].float()).softmax(-1).mean(dim=1, keepdim=True)
        x = x + self.drop(self.out2(y.transpose(1, 2).reshape(batch, steps, width)))
        return x + self.ff(self.norm3(x) * (1.0 + g3) + b3), guide_weights


@dataclass
class _TextContext:
    end: Tensor
    positions: Tensor
    pad_mask: Tensor
    is_spatial: Tensor
    anchors: Tensor


@dataclass
class _DecodeCache:
    memory: Tensor
    text: _TextContext
    style_z: Tensor
    cross: list[KV]
    history: list[KV]
    low: Tensor
    high: Tensor
    start: Tensor
    lift: Tensor
    cursor: Tensor


class InkForgeV3Model(InkForgeModel):
    """Uses the same autoregressive actions, stroke features, and style as v2."""

    def __init__(self, cfg: InkForgeV3Config, *, grad_checkpoint: bool = False):
        nn.Module.__init__(self)
        if cfg.n_heads < 2 or cfg.n_heads % 2 or cfg.d_model % (2 * cfg.n_heads):
            raise ValueError("V3 requires an even head count and an even head dimension")
        if cfg.dec_layers < 1 or cfg.max_stroke_points < 1:
            raise ValueError("V3 requires a decoder and a positive point limit")
        if cfg.token_roles and len(cfg.token_roles) != cfg.vocab_size:
            raise ValueError("token_roles must match vocab_size")
        self.cfg = cfg
        self.grad_checkpoint = grad_checkpoint
        self.register_buffer(
            "token_roles", torch.tensor(cfg.token_roles or (0,) * cfg.vocab_size), persistent=False
        )
        self.style_encoder = InkForgeStyleEncoder(cfg)
        self.text_encoder = LatexTextEncoder(cfg)
        self.role_embed = nn.Embedding(_NUM_ROLES, cfg.d_model, padding_idx=0)
        self.stroke_in = nn.Sequential(
            nn.Linear(self.INPUT_DIM, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.canvas_in = nn.Linear(_CANVAS_DIM, cfg.d_model, bias=False)
        self.dec_rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.max_stroke_points + 64)
        self.dec_blocks = nn.ModuleList([
            _SpatialDecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.style_dim, cfg.dropout)
            for _ in range(cfg.dec_layers)
        ])
        self.dec_norm = RMSNorm(cfg.d_model)
        self.mdn_head = nn.Linear(cfg.d_model, cfg.mdn_components * 6 + 3)
        self.cursor_speed = nn.Linear(self.INPUT_DIM, 1)
        self.cursor_gate = nn.Parameter(torch.tensor(0.6))
        self.token_anchors = nn.Linear(cfg.d_model, 2)
        self.anchor_gate = nn.Parameter(torch.tensor(0.25))

        nn.init.normal_(self.role_embed.weight[1:], std=0.02)
        nn.init.zeros_(self.canvas_in.weight)
        nn.init.zeros_(self.cursor_speed.weight)
        nn.init.zeros_(self.cursor_speed.bias)
        nn.init.normal_(self.token_anchors.weight, std=0.01)
        nn.init.zeros_(self.token_anchors.bias)

    def encode_memory(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        memory, mask = super().encode_memory(token_ids)
        return memory + self.role_embed(self.token_roles[token_ids]), mask

    def _text_context(self, memory: Tensor, pad_mask: Tensor, token_ids: Tensor | None) -> _TextContext:
        spatial = torch.zeros_like(pad_mask[:, 0])
        if token_ids is not None:
            spatial = (self.token_roles[token_ids].ne(0) & ~pad_mask).any(dim=-1)
        return _TextContext(
            end=(~pad_mask).sum(dim=-1, keepdim=True).float().sub(1).clamp_min(1),
            positions=torch.arange(memory.shape[1], device=memory.device, dtype=torch.float32),
            pad_mask=pad_mask,
            is_spatial=spatial,
            anchors=self.token_anchors(memory).float().tanh(),
        )

    def _canvas(self, xy: Tensor, low: Tensor, high: Tensor, start: Tensor) -> Tensor:
        scale = xy.new_tensor((self.cfg.abs_x_clip, self.cfg.abs_y_clip))
        center = (high + low) * 0.5
        relative = (xy - center) / ((high - low) * 0.5).clamp_min(0.5)
        return torch.cat([low / scale, high / scale, (xy - start).asinh(), relative], dim=-1)

    def _parallel_canvas(self, inputs: Tensor) -> Tensor:
        xy = inputs[..., 2:4].float()
        low = xy.cummin(dim=1).values
        high = xy.cummax(dim=1).values
        starts = F.pad(inputs[:, :-1, 5].gt(0.5), (1, 0))
        indices = torch.arange(inputs.shape[1], device=inputs.device)
        start_idx = torch.where(starts, indices, 0).cummax(dim=1).values
        start = xy.gather(1, start_idx.unsqueeze(-1).expand_as(xy))
        return self._canvas(xy, low, high, start)

    def _cross_bias(
        self, inputs: Tensor, canvas: Tensor, text: _TextContext, dtype: torch.dtype, cursor: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        # The same positive increments produce parallel and cached cursor positions.
        speed = F.softplus(self.cursor_speed(inputs).squeeze(-1).float()) / math.log(2.0)
        advance = speed * (text.end / 160)
        if cursor is not None:
            advance = advance + cursor
        else:
            advance = advance.cumsum(dim=1)
        distance = advance.minimum(text.end)[:, :, None] - text.positions
        width = (text.end * 0.12).clamp_min(0.85).unsqueeze(-1)
        strength = self.cursor_gate.clamp(0, 2) * torch.where(text.is_spatial, 0.25, 1.0)
        monotonic = -(distance / width).square().log1p() * strength[:, None, None]

        # Free heads compare learned token anchors with the pen's position relative
        # to the evolving canvas. The bounded bias leaves every text position visible.
        relative = canvas[..., -2:]
        spatial = -(relative[:, :, None] - text.anchors[:, None, :]).square().sum(dim=-1)
        spatial = spatial * self.anchor_gate.tanh()
        n = self.cfg.n_heads // 2
        bias = torch.stack((monotonic, spatial), dim=1)
        bias = bias.masked_fill(text.pad_mask[:, None, None, :], float("-inf")).to(dtype)
        # CUDA efficient SDPA requires row strides aligned to eight elements.
        length = bias.shape[-1]
        bias = F.pad(bias, (0, (-length) % 8), value=float("-inf"))
        bias = bias.repeat_interleave(n, dim=1)[..., :length]
        return bias, advance

    def decode_features(
        self,
        stroke_in_feats: Tensor,
        memory: Tensor,
        mem_mask: Tensor,
        style_z: Tensor,
        *,
        return_guided_attns: bool = False,
        token_ids: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        canvas = self._parallel_canvas(stroke_in_feats)
        x = self.stroke_in(stroke_in_feats) + self.canvas_in(canvas)
        cos, sin = self.dec_rope(stroke_in_feats.shape[1])
        text = self._text_context(memory, mem_mask, token_ids)
        bias, cursor = self._cross_bias(stroke_in_feats, canvas, text, x.dtype, None)
        attns = []
        for i, block in enumerate(self.dec_blocks):
            # Supervise the final block's reading-order heads, leaving free heads unconstrained.
            guide = return_guided_attns and i == len(self.dec_blocks) - 1
            args = (x, memory, style_z, cos, sin, bias)
            if self.grad_checkpoint and torch.is_grad_enabled():
                x, weights = checkpoint(block, *args, guide=guide, use_reentrant=False)
            else:
                x, weights = block(*args, guide=guide)
            if weights is not None:
                attns.append(weights)
        return self.mdn_head(self.dec_norm(x)), attns, cursor

    def _make_cache(self, token_ids: Tensor, style_z: Tensor, max_steps: int) -> _DecodeCache:
        memory, mask = self.encode_memory(token_ids)
        cross = [block.project_cross_kv(memory) for block in self.dec_blocks]
        batch, heads, _, dim = cross[0][0].shape
        history = [
            (k.new_empty((batch, heads, max_steps, dim)), v.new_empty((batch, heads, max_steps, dim)))
            for k, v in cross
        ]
        zero_xy = torch.zeros((batch, 1, 2), device=memory.device)
        return _DecodeCache(
            memory, self._text_context(memory, mask, token_ids), style_z, cross, history,
            zero_xy, zero_xy, zero_xy,
            torch.zeros((batch, 1, 1), dtype=torch.bool, device=memory.device),
            zero_xy[..., 0],
        )

    def _decode_step(self, inputs: Tensor, cache: _DecodeCache, offset: int) -> Tensor:
        xy = inputs[..., 2:4].float()
        cache.low = cache.low.minimum(xy)
        cache.high = cache.high.maximum(xy)
        cache.start = torch.where(cache.lift, xy, cache.start)
        cache.lift = inputs[..., 5:6].gt(0.5)
        canvas = self._canvas(xy, cache.low, cache.high, cache.start)
        x = self.stroke_in(inputs) + self.canvas_in(canvas)
        cos, sin = self.dec_rope(1, offset=offset)
        bias, cache.cursor = self._cross_bias(inputs, canvas, cache.text, x.dtype, cache.cursor)
        for block, cross_kv, history in zip(self.dec_blocks, cache.cross, cache.history):
            x, _ = block(x, cache.memory, cache.style_z, cos, sin, bias, cross_kv, history, offset)
        return self.dec_norm(x)

    @staticmethod
    def _component_log_prob(
        params: tuple[Tensor, ...], target_dx: Tensor, target_dy: Tensor
    ) -> Tensor:
        pi, mx, my, sx, sy, rho, _ = params
        nx = (target_dx.float().unsqueeze(-1) - mx) / sx
        ny = (target_dy.float().unsqueeze(-1) - my) / sy
        residual = (ny - rho * nx) / (1 - rho.square()).sqrt()
        return (
            F.log_softmax(pi, dim=-1)
            - 0.5 * (nx.square() + residual.square() + torch.log1p(-rho.square()))
            - sx.log() - sy.log() - math.log(2 * math.pi)
        )

    def sequence_log_prob(
        self,
        *,
        token_ids: Tensor,
        stroke_inputs: Tensor,
        target_dx: Tensor,
        target_dy: Tensor,
        target_pen: Tensor,
        stroke_mask: Tensor,
        style_z: Tensor,
        pen_weight: float = 1.0,
        length_normalize: bool = False,
        mask_illegal_eos: bool = True,
    ) -> Tensor:
        memory, mask = self.encode_memory(token_ids)
        raw, _, _ = self.decode_features(stroke_inputs, memory, mask, style_z, token_ids=token_ids)
        params = self.split_mdn_params(raw)
        pen = params[-1]
        if mask_illegal_eos:
            # Permit legacy human labels where the final lift was not explicitly recorded.
            forbid = stroke_mask & stroke_inputs[..., 5].le(0.5) & target_pen.ne(2)
            pen = torch.cat((pen[..., :2], pen[..., 2:3].masked_fill(forbid[..., None], -1e4)), dim=-1)
        log_pen = F.log_softmax(pen, dim=-1).gather(-1, target_pen.unsqueeze(-1)).squeeze(-1)
        log_xy = self._component_log_prob(params, target_dx, target_dy).logsumexp(dim=-1)
        coord = stroke_mask & target_pen.ne(2)
        scores = log_xy.masked_fill(~coord, 0).sum(dim=1)
        scores = scores + pen_weight * log_pen.masked_fill(~stroke_mask, 0).sum(dim=1)
        if length_normalize:
            scores = scores / stroke_mask.sum(dim=1).clamp_min(1)
        return scores

    def _perturb_inputs(self, inputs: Tensor, mask: Tensor, noise: float) -> Tensor:
        # Clean input-only perturbation (matches v2's proven exposure-bias recipe):
        # perturb observed (dx, dy, abs_x, abs_y) slightly while keeping ground-truth
        # (target_dx, target_dy) strictly clean so MDN output scales (sx, sy) stay sharp.
        valid = (mask & inputs[..., 6].le(0.5)).unsqueeze(-1)
        xy_limit = inputs.new_tensor((self.cfg.abs_x_clip, self.cfg.abs_y_clip))
        delta_noise = torch.randn_like(inputs[..., :2].float()) * noise * valid
        xy_noise = torch.randn_like(inputs[..., 2:4].float()) * (0.5 * noise) * valid
        deltas = (inputs[..., :2] + delta_noise.to(inputs.dtype)).clamp(-self.cfg.coord_clip, self.cfg.coord_clip)
        xy = (inputs[..., 2:4] + xy_noise.to(inputs.dtype)).clamp(-xy_limit, xy_limit)
        return torch.cat((deltas, xy, inputs[..., 4:]), dim=-1)

    def compute_loss(
        self,
        *,
        token_ids: Tensor,
        stroke_inputs: Tensor,
        target_dx: Tensor,
        target_dy: Tensor,
        target_pen: Tensor,
        stroke_mask: Tensor,
        style_points: Tensor,
        style_aspect: Tensor,
        style_mask: Tensor,
        kl_weight: float = 0.025,
        free_bits: float = 0.15,
        style_drop_prob: float = 0.30,
        text_drop_prob: float = 0.0,
        guided_attn_weight: float = 2.0,
        guided_attn_sigma: float = 0.24,
        transition_weight: float = 4.0,
        cursor_end_weight: float = 0.5,
        jump_aux_weight: float = 0.25,
        input_noise: float = 0.0,
    ) -> dict[str, Tensor]:
        z, mu, logvar = self.encode_style(style_points, style_aspect, style_mask)
        if self.training and style_drop_prob > 0:
            z = z * (torch.rand_like(z[:, :1]) >= style_drop_prob)
        keep_text = torch.ones_like(token_ids[:, 0], dtype=torch.bool)
        if self.training and text_drop_prob > 0:
            keep_text = torch.rand(keep_text.shape, device=z.device) >= text_drop_prob
            token_ids = torch.where(keep_text[:, None], token_ids, self._make_uncond_tokens(token_ids))

        if self.training and input_noise > 0:
            stroke_inputs = self._perturb_inputs(stroke_inputs, stroke_mask, input_noise)

        memory, mask = self.encode_memory(token_ids)
        raw, attns, cursor = self.decode_features(
            stroke_inputs, memory, mask, z,
            token_ids=token_ids, return_guided_attns=guided_attn_weight > 0,
        )
        params = self.split_mdn_params(raw)
        pi, mx, my, sx, sy, _, pen_logits = params
        component_logp = self._component_log_prob(params, target_dx, target_dy)
        coord_nll = -component_logp.logsumexp(dim=-1)
        coord = stroke_mask & target_pen.ne(2)
        lift = stroke_inputs[..., 5].gt(0.5)
        jump = lift | stroke_inputs[..., 6].gt(0.5)
        after_jump = F.pad(lift[:, :-1], (1, 0))
        weights = torch.where(jump, transition_weight, torch.where(after_jump, 2.0, 1.0)) * coord
        valid_count = stroke_mask.sum()
        coord_count = coord.sum()
        jump_mask = coord & jump
        draw_mask = coord & ~jump
        weighted_count = weights.sum()
        mdn_loss = (coord_nll * weights).sum() / weighted_count.clamp_min(1)
        pen_nll = F.cross_entropy(pen_logits.transpose(1, 2), target_pen, reduction="none")
        pen_loss = (pen_nll * stroke_mask).sum() / valid_count.clamp_min(1)

        # Regress the posterior mode so ambiguous placements keep separate mixture components.
        mode = component_logp.detach().argmax(dim=-1, keepdim=True)
        jump_dx = mx.gather(-1, mode).squeeze(-1)
        jump_dy = my.gather(-1, mode).squeeze(-1)
        endpoint_error = (
            F.smooth_l1_loss(jump_dx, target_dx.float(), reduction="none")
            + F.smooth_l1_loss(jump_dy, target_dy.float(), reduction="none")
        )
        air = coord & lift
        jump_aux = (endpoint_error * air).sum() / air.sum().clamp_min(1)

        # Stack/accent scripts can change the writing order at any point in the
        # prompt. Without token-to-stroke labels, skip diagonal supervision on these rows.
        linear = keep_text & ~self.token_roles[token_ids].ne(0).any(dim=-1)
        guide_mask = stroke_mask & linear[:, None]
        guide_loss, attn_max = self._guided_diagonal_attn_loss(attns, guide_mask, mask, sigma=guided_attn_sigma)
        last = (stroke_mask.sum(dim=1, keepdim=True) - 1).clamp_min(0)
        text_end = ((~mask).sum(dim=1) - 1).clamp_min(1)
        ratio = cursor.gather(1, last).squeeze(1) / text_end
        cursor_end = (F.smooth_l1_loss(ratio, torch.ones_like(ratio), reduction="none") * linear).sum()
        cursor_end = cursor_end / linear.sum().clamp_min(1)

        styled = (~style_mask).any(dim=1, keepdim=True)
        kl_dims = -0.5 * (1 + logvar.float() - mu.float().square() - logvar.float().exp())
        kl_dims = (kl_dims * styled).sum(dim=0) / styled.sum().clamp_min(1)
        kl_loss = kl_dims.clamp_min(free_bits).sum() * styled.any()

        val_score = mdn_loss + 2 * pen_loss
        loss = (
            val_score + guided_attn_weight * guide_loss + cursor_end_weight * cursor_end
            + kl_weight * kl_loss + jump_aux_weight * jump_aux
        )
        with torch.no_grad():
            top = pi.argmax(dim=-1, keepdim=True)
            sigma = sx.minimum(sy).gather(-1, top).squeeze(-1)
            sigma_floor = sigma.le(max(2e-3, 1.25 * self.cfg.sigma_min)) & coord
            metrics = {
                "val_score": val_score,
                "mdn_loss": (coord_nll * coord).sum() / coord_count.clamp_min(1),
                "mdn_w_loss": mdn_loss,
                "draw_mdn_loss": (coord_nll * draw_mask).sum() / draw_mask.sum().clamp_min(1),
                "jump_mdn_loss": (coord_nll * jump_mask).sum() / jump_mask.sum().clamp_min(1),
                "jump_aux_loss": jump_aux,
                "pen_loss": pen_loss,
                "guided_loss": guide_loss,
                "attn_max_txt": attn_max,
                "cursor_end_loss": cursor_end,
                "cursor_progress": (ratio * linear).sum() / linear.sum().clamp_min(1),
                "cursor_overshoot": ((cursor > text_end[:, None] + 1) & guide_mask).sum()
                / guide_mask.sum().clamp_min(1),
                "cursor_gate": self.cursor_gate.detach(),
                "sigma_floor_frac": sigma_floor.sum() / coord_count.clamp_min(1),
                "kl_raw": kl_dims.sum(),
                "pen_acc": ((pen_logits.argmax(dim=-1) == target_pen) & stroke_mask).sum()
                / valid_count.clamp_min(1),
                "valid_count": valid_count,
                "coord_count": coord_count,
                "weighted_coord_sum": weighted_count,
                "draw_count": draw_mask.sum(),
                "jump_count": jump_mask.sum(),
            }
        return {"loss": loss, **{key: value.detach() for key, value in metrics.items()}}


    @staticmethod
    def _sample_categorical(probabilities: Tensor) -> Tensor:
        cdf = probabilities.cumsum(dim=-1)
        draw = torch.rand_like(probabilities[:, :1]) * cdf[:, -1:]
        return (cdf <= draw).sum(dim=-1).clamp_max(probabilities.shape[-1] - 1)

    @staticmethod
    def _mixture_probs(logits: Tensor, temperature: Tensor, min_p: float, top_p: float) -> Tensor:
        logits = logits / temperature[:, None].clamp_min(0.05)
        if min_p > 0:
            logits = logits.masked_fill(logits < logits.amax(dim=-1, keepdim=True) + math.log(min_p), -torch.inf)
        probs = logits.softmax(dim=-1)
        if top_p < 1:
            sorted_probs, indices = probs.sort(dim=-1, descending=True)
            outside = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
            sorted_probs = sorted_probs.masked_fill(outside, 0)
            probs = torch.zeros_like(probs).scatter(-1, indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    @torch.inference_mode()
    def generate_batch(
        self,
        token_ids: Tensor,
        style_z: Tensor | None = None,
        *,
        temperature: float | Tensor = 0.40,
        pen_temperature: float = 0.80,
        jump_temp_scale: float = 0.55,
        cfg_scale: float = 1.0,
        max_steps: int = 512,
        min_steps: int = 1,
        return_meta: bool = False,
        return_actions: bool = False,
        min_p: float = 0.03,
        top_p: float = 1.0,
        pen_cfg_scale: float | None = None,
        cfg_ramp_steps: int = 0,
    ) -> list[list[np.ndarray]] | tuple[list[list[np.ndarray]], list[dict[str, Any]]]:
        """Generate strokes, optionally emitting the actual autoregressive actions for DPO."""
        if not 1 <= max_steps <= self.dec_rope.cos_cached.shape[0] or min_steps < 0:
            raise ValueError("Generation step limits must fit the decoder's RoPE cache")
        if not 0 <= min_p <= 1 or not 0 < top_p <= 1 or cfg_ramp_steps < 0:
            raise ValueError("Invalid mixture truncation or CFG ramp")
        if not all(math.isfinite(t) and t > 0 for t in (pen_temperature, jump_temp_scale)):
            raise ValueError("Pen temperature and jump_temp_scale must be finite and positive")
        device = token_ids.device
        batch = token_ids.shape[0]
        if batch == 0:
            return ([], []) if return_meta else []
        style_z = (
            torch.zeros((batch, self.cfg.style_dim), device=device)
            if style_z is None else style_z.to(device)
        )
        temp = torch.as_tensor(temperature, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        if not (temp.isfinite() & temp.ge(0)).all():
            raise ValueError("Coordinate temperatures must be finite and nonnegative")
        if pen_cfg_scale is None:
            pen_cfg_scale = 1 + 0.5 * (cfg_scale - 1)
        use_cfg = cfg_scale != 1 or pen_cfg_scale != 1
        if use_cfg:
            token_ids = torch.cat((token_ids, self._make_uncond_tokens(token_ids)), dim=0)
            style_z = torch.cat((style_z, style_z), dim=0)
        cache = self._make_cache(token_ids, style_z, max_steps)

        features = torch.zeros((batch, 1, self.INPUT_DIM), device=device)
        features[..., 6] = 1
        xy = torch.zeros((batch, 2), device=device)
        xy_limit = xy.new_tensor((self.cfg.abs_x_clip, self.cfg.abs_y_clip))
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        lift = torch.zeros_like(done)
        steps = torch.zeros(batch, dtype=torch.long, device=device)
        xy_history = torch.empty((batch, max_steps, 2), device=device)
        pen_history = torch.empty((batch, max_steps), dtype=torch.long, device=device)
        deltas = torch.empty_like(xy_history) if return_actions else None

        for step in range(max_steps):
            feat_run = torch.cat((features, features), dim=0) if use_cfg else features
            hidden = self._decode_step(feat_run, cache, step)
            if use_cfg:
                conditioned, unconditioned = hidden.float().chunk(2)
                ramp = min(1., (step + 1) / cfg_ramp_steps) if cfg_ramp_steps else 1.
                difference = conditioned - unconditioned
                geom = self.mdn_head(conditioned + ramp * (cfg_scale - 1) * difference)
                if pen_cfg_scale != cfg_scale:
                    pen_features = conditioned + ramp * (pen_cfg_scale - 1) * difference
                    pen = F.linear(pen_features, self.mdn_head.weight[-3:], self.mdn_head.bias[-3:])
                    geom = torch.cat((geom[..., :-3], pen), dim=-1)
            else:
                geom = self.mdn_head(hidden)
            pi, mx, my, sx, sy, rho, logits = self.split_mdn_params(geom[:, 0])

            forbid = ~lift | (step < min_steps)
            logits[:, 2].masked_fill_(forbid, -torch.inf)
            pen = self._sample_categorical((logits / pen_temperature).softmax(dim=-1))
            pen = torch.where(done, 2, pen)

            coord_temp = torch.where(lift, temp * jump_temp_scale, temp)
            component = self._sample_categorical(self._mixture_probs(pi, coord_temp, min_p, top_p))
            select = component[:, None]
            means = torch.cat((mx.gather(-1, select), my.gather(-1, select)), dim=-1)
            stds = torch.cat((sx.gather(-1, select), sy.gather(-1, select)), dim=-1)
            corr = rho.gather(-1, select)
            eps1, eps2 = torch.randn_like(xy).chunk(2, dim=-1)
            eps = torch.cat((eps1, corr * eps1 + (1 - corr.square()).sqrt() * eps2), dim=-1)
            delta = (means + stds * coord_temp.sqrt()[:, None] * eps).clamp(
                -self.cfg.coord_clip, self.cfg.coord_clip,
            )
            delta = torch.where(pen.ne(2)[:, None], delta, 0)
            xy = xy + delta
            xy_history[:, step] = xy
            pen_history[:, step] = pen
            if deltas is not None:
                deltas[:, step] = delta
            steps = steps + ~done
            done = done | pen.eq(2)
            if done.all():
                break
            lift = pen.eq(1)
            flags = torch.stack((pen.eq(0), lift, torch.zeros_like(lift)), dim=-1)
            features = torch.cat((delta, xy.clamp(-xy_limit, xy_limit), flags), dim=-1).unsqueeze(1)

        points = xy_history[:, : step + 1].cpu().numpy()
        pens = pen_history[:, : step + 1].cpu().numpy()
        written = deltas[:, : step + 1].cpu().numpy() if deltas is not None else None
        counts = steps.cpu().tolist()
        complete = done.cpu().tolist()
        strokes, metadata = [], []
        for i, count in enumerate(counts):
            p = pens[i, :count]
            xy_i = points[i, :count]
            n_points = count - int(complete[i])
            path = np.concatenate((np.zeros((1, 2), dtype=np.float64), xy_i[:n_points]), axis=0)
            boundaries = np.flatnonzero(p[:n_points] == 1) + 2
            if len(boundaries) and boundaries[-1] == len(path):
                boundaries = boundaries[:-1]
            strokes.append(list(np.split(path, boundaries)))
            entry: dict[str, Any] = {"steps": count, "finished": complete[i], "hit_cap": not complete[i]}
            if written is not None:
                s_in = np.zeros((count, self.INPUT_DIM), dtype=np.float32)
                s_in[0, 6] = 1
                s_in[1:, :2] = written[i, : count - 1]
                s_in[1:, 2] = np.clip(xy_i[:-1, 0], -self.cfg.abs_x_clip, self.cfg.abs_x_clip)
                s_in[1:, 3] = np.clip(xy_i[:-1, 1], -self.cfg.abs_y_clip, self.cfg.abs_y_clip)
                s_in[1:, 4] = p[:-1] == 0
                s_in[1:, 5] = p[:-1] == 1
                entry["raw_trace"] = {
                    "stroke_inputs": s_in,
                    "target_dx": written[i, :count, 0].copy(),
                    "target_dy": written[i, :count, 1].copy(),
                    "target_pen": p.copy(),
                }
            metadata.append(entry)
        return (strokes, metadata) if return_meta else strokes

