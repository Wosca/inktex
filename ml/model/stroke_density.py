"""Experimental additive glyph counting from contextual stroke features.

Each nonempty stroke belongs to one annotated symbol. Its predicted mass is
between zero and one; a k-stroke symbol has per-stroke target mass 1/k. Summing
class probabilities times mass gives soft glyph counts without treating every
stroke as a separate symbol. Reference groups are training targets only.

It is available as an opt-in structural counter, with reference groups used
only by its training loss. It does not implement the image-based CAN architecture.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StrokeDensityCounter(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mass_head = nn.Linear(width, 1)

    def forward(self, contextual, symbol_logits, stroke_mask):
        contextual = contextual.masked_fill(stroke_mask[..., None], 0)
        symbol_logits = symbol_logits.masked_fill(stroke_mask[..., None], 0)
        mass = self.mass_head(contextual).float().squeeze(-1).sigmoid()
        mass = mass.masked_fill(stroke_mask, 0)
        probabilities = symbol_logits.float().softmax(-1)
        # Mask probabilities too: padded logits must never contribute, including
        # NaNs in unused slots. Soft counts retain gradients to the glyph head.
        probabilities = probabilities.masked_fill(stroke_mask[..., None], 0)
        counts = (probabilities * mass[..., None]).sum(1)
        return {'mass': mass, 'count': counts, 'stroke_mask': stroke_mask}


def stroke_mass_loss(mass, groups, symbol_count):
    if not symbol_count or not len(groups) or groups.min() < 0 or groups.max() >= symbol_count:
        raise ValueError('Invalid symbol groups for density targets')
    sizes = torch.bincount(groups, minlength=symbol_count)
    if bool(sizes.eq(0).any()):
        raise ValueError('Every reference symbol must have at least one stroke')
    shares = sizes[groups].float().reciprocal()
    errors = F.smooth_l1_loss(mass, shares, reduction='none')
    return (errors * shares).sum() / symbol_count


def stroke_density_loss(outputs, records, symbol_classes: int):
    """Group-balanced local mass regression plus log1p global count regression.

The two losses are separate so a future experiment can declare their weights.
Unannotated rows supply neither targets nor gradients. Symbols receive equal
local-loss mass regardless of how many strokes they have.
"""
    mass_losses, count_losses = [], []
    for index, record in enumerate(records):
        if record is None:
            continue
        groups = torch.as_tensor(record['stroke_groups'], device=outputs['mass'].device)
        classes = torch.as_tensor(record['symbol_classes'], device=groups.device)
        n = len(groups)
        if n != int((~outputs['stroke_mask'][index]).sum()):
            raise ValueError('Density targets do not match input stroke boundaries')
        if bool(outputs['stroke_mask'][index, :n].any()):
            raise ValueError('Density supervision requires contiguous nonempty stroke slots')
        if not len(classes) or classes.min() < 0 or classes.max() >= symbol_classes:
            raise ValueError('Unknown density glyph class')
        mass_losses.append(stroke_mass_loss(outputs['mass'][index, :n], groups, len(classes)))
        target = torch.bincount(classes, minlength=symbol_classes).float()
        count_losses.append(F.smooth_l1_loss(outputs['count'][index].log1p(), target.log1p()))
    zero = outputs['count'].sum() * 0
    return {'mass': torch.stack(mass_losses).mean() if mass_losses else zero,
            'count': torch.stack(count_losses).mean() if count_losses else zero}
