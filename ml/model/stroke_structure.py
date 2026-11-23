"""Stroke hierarchy with auxiliary symbol/layout heads and graph inference."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .initialization import initialize_transformer


def stroke_membership(features, padding_mask):
    valid = ~padding_mask
    ids = (features[:, :, 7].gt(.5).long().cumsum(1) - 1).clamp_min(0)
    # Dynamic stroke count: no fixed-capacity truncation of long expressions.
    # A masked second slot for a one-stroke input avoids the ONNX exporter's
    # ambiguous one-versus-many broadcasting guard. All observed strokes remain.
    slot_count = (torch.where(valid, ids, 0).amax(dim=(0, 1)) + 1).clamp_min(2).item()
    torch._check_is_size(slot_count)
    torch._check(slot_count >= 2)
    slots = torch.arange(slot_count, device=features.device)
    membership = ids[:, None, :].eq(slots[None, :, None]) & valid[:, None, :]
    counts = membership.sum(-1)
    weights = membership.to(features.dtype) / counts.clamp_min(1)[..., None]
    return ids, membership, weights, counts.eq(0)


def stroke_geometry(features, membership, weights):
    xy = features[:, :, :2]
    # Rounding a dot's center in FP16/BF16, then dividing that error by its
    # 1e-4 local scale, invents a large displacement for a one-point stroke.
    # Keep this geometric reduction in FP32 under mixed precision.
    with torch.autocast(device_type=features.device.type, enabled=False):
        center = weights.float() @ xy.float()
    lower = xy[:, None].masked_fill(~membership[..., None], float('inf')).amin(2)
    upper = xy[:, None].masked_fill(~membership[..., None], -float('inf')).amax(2)
    present = membership.any(-1)[..., None]
    lower, upper = torch.where(present, lower, 0), torch.where(present, upper, 0)
    extent = (upper - lower).clamp_min(1e-4)
    # Uniform local scale retains each stroke's aspect ratio.
    scale = extent.amax(-1, keepdim=True)
    return center, extent, scale


class StrokeStructure(nn.Module):
    def __init__(self, width, heads, feedforward, dropout, symbol_classes, relation_classes):
        super().__init__()
        self.local_projection = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.geometry_projection = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        block = nn.TransformerEncoderLayer(width, heads, feedforward, dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(block, 2, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        initialize_transformer(self.context)
        self.input_gate = nn.Parameter(torch.zeros(()))
        self.context_gate = nn.Parameter(torch.zeros(()))
        self.symbol_head = nn.Linear(width, symbol_classes)
        rank = min(64, width)
        self.group_left = nn.Linear(width, rank)
        self.group_right = nn.Linear(width, rank)
        self.relation_left = nn.Linear(width, rank * relation_classes)
        self.relation_right = nn.Linear(width, rank * relation_classes)
        self.relation_geometry = nn.Sequential(nn.Linear(6, rank), nn.GELU(), nn.Linear(rank, relation_classes))
        self.count_head = nn.Linear(width, symbol_classes)
        self.rank, self.relation_classes = rank, relation_classes
        self.counter_kind = 'mean'
        self.density_counter = None
        self.count_projection = None
        self.count_gate = None

    def configure_counter(self, kind='mean', condition_counts=False):
        if kind not in ('mean', 'density'):
            raise ValueError('Unknown structural counter')
        if self.density_counter is not None and kind != 'density':
            raise ValueError('Cannot remove a checkpoint density counter')
        if self.count_projection is not None and not condition_counts:
            raise ValueError('Cannot remove checkpoint count conditioning')
        width = self.symbol_head.in_features
        # Modules initialize on CPU before moving to the model's device. Keep
        # that initialization from changing downstream loader/augmentation seeds.
        with torch.random.fork_rng(devices=[]):
            if kind == 'density' and self.density_counter is None:
                from .stroke_density import StrokeDensityCounter
                self.density_counter = StrokeDensityCounter(width).to(self.symbol_head.weight.device)
            if condition_counts and self.count_projection is None:
                self.count_projection = nn.Sequential(nn.Linear(self.symbol_head.out_features, width), nn.LayerNorm(width)).to(self.symbol_head.weight.device)
                self.count_gate = nn.Parameter(torch.zeros((), device=self.symbol_head.weight.device))
        self.counter_kind = kind

    def count_outputs(self, contextual, mask):
        if self.counter_kind == 'density':
            symbol_logits = self.symbol_head(contextual)
            density = self.density_counter(contextual, symbol_logits, mask)
            return density['count'].log1p(), density['mass']
        expression = (contextual * (~mask)[..., None]).sum(1) / (~mask).sum(1).clamp_min(1)[:, None]
        return self.count_head(expression), None

    def local_features(self, features, padding_mask):
        ids, membership, weights, stroke_mask = stroke_membership(features, padding_mask)
        center, extent, scale = stroke_geometry(features, membership, weights)
        point_center = center.gather(1, ids[..., None].expand(-1, -1, 2))
        point_scale = scale.gather(1, ids[..., None])
        local = torch.cat(((features[..., :2] - point_center) / point_scale,
                           features[..., 2:4] / point_scale, features[..., 7:9]), -1)
        return self.local_projection(local), (ids, membership, weights, stroke_mask, center, extent)

    def enrich(self, memory, layout):
        ids, _, weights, mask, center, extent = layout
        relative = getattr(self, 'relative_positions', False)
        geometry = torch.cat((torch.zeros_like(center) if relative else center, extent.log(), extent), -1)
        # Pool within observed strokes only, retaining every point for the decoder.
        pooled = weights.to(memory.dtype) @ memory
        bias = None
        context_mask = mask
        if relative:
            from .relative_position import encode_relative, spatial_bias
            heads = self.context.layers[0].self_attn.num_heads
            bias = spatial_bias(center, heads).to(pooled.dtype)
            context_mask = torch.zeros_like(mask, dtype=pooled.dtype).masked_fill(mask, float('-inf'))
        values = pooled + self.geometry_projection(geometry)
        contextual = (encode_relative(self.context, values, bias, context_mask) if relative else
                      self.context(values, src_key_padding_mask=context_mask, is_causal=False))
        contextual = contextual.masked_fill(mask[..., None], 0)
        enriched = memory + self.context_gate.tanh() * contextual.gather(1, ids[..., None].expand(-1, -1, memory.shape[-1]))
        if self.count_projection is not None:
            log_counts, _ = self.count_outputs(contextual, mask)
            count_bias = self.count_gate.tanh() * self.count_projection(log_counts.clamp_min(0)).unsqueeze(1)
            enriched = enriched + count_bias
            if relative:
                contextual = contextual + count_bias
        return enriched, contextual

    def auxiliary(self, contextual, layout):
        _, _, _, mask, center, extent = layout
        left, right = self.group_left(contextual), self.group_right(contextual)
        group = (left @ right.transpose(-1, -2)) / math.sqrt(self.rank)
        group = (group + group.transpose(-1, -2)) * .5
        batch, length, _ = contextual.shape
        left = self.relation_left(contextual).reshape(batch, length, self.relation_classes, self.rank)
        right = self.relation_right(contextual).reshape(batch, length, self.relation_classes, self.rank)
        relations = torch.einsum('bicd,bjcd->bijc', left, right) / math.sqrt(self.rank)
        delta = center[:, None, :, :] - center[:, :, None, :]
        ratio = extent[:, None].log() - extent[:, :, None].log()
        distance = (delta / extent[:, :, None].amax(-1, keepdim=True)).clamp(-32, 32)
        relations = relations + self.relation_geometry(torch.cat((delta, ratio, distance), -1))
        log_counts, mass = self.count_outputs(contextual, mask)
        return {'symbol': self.symbol_head(contextual), 'group': group, 'relation': relations,
                'count': log_counts, 'stroke_mask': mask, 'density_mass': mass,
                'center': center, 'extent': extent}


def count_regression_loss(prediction, log_counts, balance_classes=False):
    if not balance_classes:
        return F.smooth_l1_loss(prediction, log_counts)
    errors = F.smooth_l1_loss(prediction, log_counts, reduction='none')
    present = log_counts.gt(0).to(errors.dtype)
    absent = 1 - present
    present_n, absent_n = present.sum(), absent.sum()
    mass = present / present_n.clamp_min(1) + absent / absent_n.clamp_min(1)
    partitions = present_n.gt(0).to(errors.dtype) + absent_n.gt(0).to(errors.dtype)
    return (errors * mass).sum() / partitions.clamp_min(1)


def structure_loss(outputs, records, symbol_classes, weights, *, balance_count_classes=False):
    """Average per expression; balance rare positive grouping/relation pairs.

    Unannotated examples contribute no auxiliary loss. Count targets count symbols,
    never strokes or LaTeX syntax. No ground truth is fed into the encoder.
    """
    zero = outputs['symbol'].sum() * 0
    pieces = {key: [] for key in ('symbol', 'group', 'relation', 'count')}
    if outputs.get('density_mass') is not None:
        pieces['density_mass'] = []
    for row, record in enumerate(records):
        if record is None:
            continue
        device = outputs['symbol'].device
        groups = torch.tensor(record['stroke_groups'], device=device)
        n = len(groups)
        if n != int((~outputs['stroke_mask'][row]).sum()):
            raise ValueError('Structural labels no longer align with input stroke boundaries')
        classes = torch.tensor(record['symbol_classes'], device=device)
        labels = classes[groups]
        per_stroke = F.cross_entropy(outputs['symbol'][row, :n].float(), labels, reduction='none')
        # Each symbol has equal mass even when written with several strokes.
        group_sizes = torch.bincount(groups, minlength=len(classes)).clamp_min(1)
        pieces['symbol'].append((per_stroke / group_sizes[groups]).sum() / len(classes))
        offdiag = ~torch.eye(n, device=device, dtype=torch.bool)
        same = groups[:, None].eq(groups[None, :])
        group_losses = F.binary_cross_entropy_with_logits(outputs['group'][row, :n, :n].float(), same.float(), reduction='none')
        balanced = [group_losses[offdiag & same].mean()] if (offdiag & same).any() else []
        if (offdiag & ~same).any():
            balanced.append(group_losses[offdiag & ~same].mean())
        if balanced:
            pieces['group'].append(torch.stack(balanced).mean())
        graph = torch.zeros((len(classes), len(classes)), device=device, dtype=torch.long)
        for left, right, relation in record['relations']:
            graph[left, right] = relation
        targets = graph[groups[:, None], groups[None, :]]
        rel_losses = F.cross_entropy(outputs['relation'][row, :n, :n].float().reshape(n*n, -1), targets.reshape(-1), reduction='none').reshape(n,n)
        # Within-symbol edges are handled by grouping rather than layout loss.
        valid = offdiag & ~same
        positive, negative = valid & targets.ne(0), valid & targets.eq(0)
        balanced = [rel_losses[positive].mean()] if positive.any() else []
        if negative.any():
            balanced.append(rel_losses[negative].mean())
        if balanced and record.get('relations_available', True):
            pieces['relation'].append(torch.stack(balanced).mean())
        counts = torch.bincount(classes, minlength=symbol_classes).float().log1p()
        pieces['count'].append(count_regression_loss(outputs['count'][row].float(), counts,
                                                     balance_count_classes))
        if outputs.get('density_mass') is not None:
            from .stroke_density import stroke_mass_loss
            pieces['density_mass'].append(stroke_mass_loss(outputs['density_mass'][row, :n], groups, len(classes)))
    values = {key: torch.stack(value).mean() if value else zero for key, value in pieces.items()}
    return sum(float(weights.get(key, 0)) * value for key, value in values.items()), values
