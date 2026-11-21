"""Incremental decoder state for reference greedy decoding in PyTorch."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .relative_position import alibi_slopes


def project(attention, values, part: int):
    width = attention.embed_dim
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[part * width:(part + 1) * width]
    result = F.linear(values, attention.in_proj_weight[part * width:(part + 1) * width], bias)
    return result.reshape(result.shape[0], result.shape[1], attention.num_heads, width // attention.num_heads).transpose(1, 2)


class DecoderState:
    def __init__(self, decoder, memory, memory_mask):
        if decoder.training:
            raise ValueError("Cached decoding requires an eval-mode decoder")
        self.decoder, self.memory, self.mask = decoder, memory, memory_mask
        self.layers = [
            dict(
                k=None,
                v=None,
                mk=project(layer.multihead_attn, memory, 1),
                mv=project(layer.multihead_attn, memory, 2),
            )
            for layer in decoder.transformer.layers
        ]
        self.coverage = memory.new_zeros(
            memory.shape[0], decoder.transformer.layers[-1].self_attn.num_heads, 1, memory.shape[1]
        )
        self.previous = None
        self.length = 0

    def attend(self, attention, query, keys, values, bias=None):
        q = project(attention, query, 0)
        score = q @ keys.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if bias is not None:
            score = score + bias
        weights = score.softmax(-1)
        attended = (weights @ values).transpose(1, 2).reshape(query.shape[0], 1, -1)
        return attention.out_proj(attended), weights

    def step(self, token):
        decoder = self.decoder
        values = decoder.token_embedding(token)
        if decoder.position_embedding is not None:
            values = values + decoder.position_embedding(torch.full_like(token, self.length))
        if self.previous is not None and decoder.coverage:
            self.coverage = self.coverage + self.previous * decoder.visible_tokens[token][:, None, :, None]
        final = None
        for index, (layer, state) in enumerate(zip(decoder.transformer.layers, self.layers)):
            q = layer.norm1(values)
            key, value = project(layer.self_attn, q, 1), project(layer.self_attn, q, 2)
            state["k"] = key if state["k"] is None else torch.cat((state["k"], key), 2)
            state["v"] = value if state["v"] is None else torch.cat((state["v"], value), 2)
            bias = None
            if decoder.position_embedding is None:
                distance = self.length - torch.arange(self.length + 1, device=values.device)
                bias = -alibi_slopes(layer.self_attn.num_heads, values.device)[None, :, None, None] * distance[None, None, None]
            attended, _ = self.attend(layer.self_attn, q, state["k"], state["v"], bias)
            values = values + attended
            q = layer.norm2(values)
            mask_bias = None if self.mask is None else values.new_zeros(self.mask.shape).masked_fill(
                self.mask, float("-inf")
            )[:, None, None]
            last = index == len(self.layers) - 1
            attended, weights = self.attend(layer.multihead_attn, q, state["mk"], state["mv"], mask_bias)
            if last:
                if self.mask is not None:
                    weights = weights.masked_fill(self.mask[:, None, None], 0)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
                self.previous = weights
                if decoder.coverage:
                    coverage_bias = -F.softplus(decoder.coverage_strength) * self.coverage
                    attended, weights = self.attend(
                        layer.multihead_attn, q, state["mk"], state["mv"],
                        coverage_bias if mask_bias is None else coverage_bias + mask_bias,
                    )
                final = weights.mean(1)
            values = values + attended
            q = layer.norm3(values)
            values = values + layer.linear2(layer.activation(layer.linear1(q)))
        logits = decoder.output(decoder.transformer.norm(values))
        if decoder.coverage:
            valid = torch.ones_like(self.memory[..., 0], dtype=torch.bool) if self.mask is None else ~self.mask
            uncovered = (torch.exp(-4 * self.coverage.mean(1)) * valid[:, None]).sum(-1) / valid.sum(-1)[:, None].clamp_min(1)
            eos = F.one_hot(torch.tensor(decoder.eos_id, device=values.device), logits.shape[-1]).to(logits.dtype)
            logits = logits - F.softplus(decoder.eos_coverage_strength) * uncovered[..., None] * eos
        self.length += 1
        return logits, final
