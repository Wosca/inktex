from __future__ import annotations

import torch
from torch import nn


def preservation_loss(adapted, baseline, targets):
    """Keep base-correct training distributions; labels never gate inference."""
    keep = baseline.argmax(-1).eq(targets)
    divergence = nn.functional.kl_div(adapted.float().log_softmax(-1),
                                    baseline.detach().float().softmax(-1), reduction='none').sum(-1)
    return (divergence * keep).sum() / keep.sum().clamp_min(1)


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, label_smoothing: float = 0.0,
                           *, reduction: str = 'token') -> torch.Tensor:
    if reduction not in {'token', 'equation'}:
        raise ValueError('Sequence reduction must be token or equation')
    loss = nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
        reduction='mean' if reduction == 'token' else 'none',
    )
    if reduction == 'token':
        return loss
    valid = targets.ne(pad_id)
    counts = valid.sum(1)
    if bool(counts.eq(0).any()):
        raise ValueError('Every equation must have a non-padding target')
    return (loss.reshape_as(targets).sum(1) / counts).mean()


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, targets: torch.Tensor,
                      pad_id: int, temperature: float = 2.0, *, reduction: str = 'token') -> torch.Tensor:
    """Token-mean KL(teacher || student), including EOS and excluding padding."""
    if student.shape != teacher.shape or student.shape[:2] != targets.shape:
        raise ValueError("Teacher and student must use identical token IDs and target positions")
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    if reduction not in {'token', 'equation'}:
        raise ValueError('Distillation reduction must be token or equation')
    valid = targets.ne(pad_id)
    if not bool(valid.any()):
        raise ValueError("Distillation requires at least one target token")
    # Slice first so arbitrary padded logits cannot introduce NaNs into the loss.
    student_log = nn.functional.log_softmax(student[valid].float() / temperature, dim=-1)
    teacher_log = nn.functional.log_softmax(teacher[valid].detach().float() / temperature, dim=-1)
    errors = nn.functional.kl_div(student_log, teacher_log, reduction='none', log_target=True).sum(-1) * temperature**2
    if reduction == 'token':
        return errors.mean()
    counts = valid.sum(1)
    if bool(counts.eq(0).any()):
        raise ValueError('Every equation must have a non-padding target')
    rows = torch.arange(targets.shape[0], device=targets.device)[:, None].expand_as(targets)[valid]
    return (torch.zeros(targets.shape[0], device=student.device, dtype=errors.dtype)
            .scatter_add(0, rows, errors) / counts).mean()
