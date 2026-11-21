"""Parameter-free relative attention biases; no learned position ceiling."""
import math

import torch


def alibi_slopes(heads, device=None):
    def power_of_two(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start ** (i + 1) for i in range(n)]
    power = 2 ** int(math.log2(heads))
    values = power_of_two(power)
    if heads != power:
        values += power_of_two(2 * power)[::2][:heads - power]
    return torch.tensor(values, device=device, dtype=torch.float32)


def sequence_bias(length, heads, batch, device, causal=False):
    positions = torch.arange(length, device=device)
    distance = (positions[:, None] - positions[None, :]).abs().float()
    bias = -alibi_slopes(heads, device)[:, None, None] * distance
    if causal:
        bias = bias.masked_fill(positions[None, :] > positions[:, None], float('-inf'))
    return bias.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * heads, length, length)


def spatial_bias(center, heads):
    delta = center[:, None, :] - center[:, :, None]
    distance = delta.abs().sum(-1).clamp_max(64)
    directions = torch.tensor([(math.cos(2 * math.pi * h / heads), math.sin(2 * math.pi * h / heads))
                               for h in range(heads)], device=center.device, dtype=torch.float32)
    # Distance alone cannot distinguish left from right or above from below.
    # Bounded signed offsets preserve that layout information at any line width.
    directional = (delta.tanh()[:, None] * directions[None, :, None, None]).sum(-1)
    return (-distance[:, None] * alibi_slopes(heads, center.device)[None, :, None, None]
            + .25 * directional).flatten(0, 1)


def encode_relative(transformer, values, bias, padding_bias):
    """Preserve floating relative biases in training, inference and ONNX.

    The fused TransformerEncoderLayer inference path can interpret an additive
    bias differently from SDPA. Explicit pre-norm layers avoid that path without
    changing the process-wide fastpath setting or historical model behavior.
    """
    for layer in transformer.layers:
        query = layer.norm1(values)
        attended = layer.self_attn(query, query, query, attn_mask=bias,
                                   key_padding_mask=padding_bias, need_weights=False)[0]
        values = values + layer.dropout1(attended)
        query = layer.norm2(values)
        values = values + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(query)))))
    return transformer.norm(values) if transformer.norm is not None else values
