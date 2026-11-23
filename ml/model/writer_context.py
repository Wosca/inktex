"""A shared, label-grounded context encoder; users never own model weights."""
from __future__ import annotations

import torch
from torch import nn


def summarize_ink(memory: torch.Tensor, mask: torch.Tensor, slots: int = 16) -> torch.Tensor:
    """Cache a bounded summary of SUPPORT ink, never pool the query expression."""
    result = []
    for sample, padding in zip(memory, mask):
        valid = sample[~padding]
        if not len(valid):
            raise ValueError("Support ink cannot be empty")
        result.append(torch.nn.functional.adaptive_avg_pool1d(valid.T.unsqueeze(0), slots)[0].T)
    return torch.stack(result)


class WriterContext(nn.Module):
    def __init__(self, width: int, heads: int, slots: int = 32, kind: str = "pooled"):
        super().__init__()
        self.slots = slots
        self.width = width
        if kind not in {"pooled", "symbols", "aligned", "conditioned", "calibrated", "grounded"}:
            raise ValueError("Unknown writer context kind")
        self.kind = kind
        if kind in {"aligned", "calibrated"}:
            self.retrieval_projection = nn.Linear(width, width, bias=False)
            nn.init.eye_(self.retrieval_projection.weight)
            self.token_gain = nn.Parameter(torch.zeros(slots))
            self.register_buffer('global_reference', torch.zeros(slots, width))
            self.register_buffer('reference_counts', torch.zeros(slots))
            return
        self.label_projection = nn.Linear(width, width)
        if kind == 'grounded':
            self.register_buffer('global_reference', torch.zeros(slots, width))
            self.register_buffer('reference_counts', torch.zeros(slots))
            self.register_buffer('label_reference', torch.zeros(slots, width))
        self.ink_alignment = nn.MultiheadAttention(width, heads, batch_first=True) if kind not in {'conditioned', 'grounded'} else None
        self.support_norm = nn.LayerNorm(width) if kind not in {'conditioned', 'grounded'} else None
        self.profile_queries = nn.Parameter(torch.randn(1, slots, width) * 0.02) if kind == "pooled" else None
        self.profile_pool = nn.MultiheadAttention(width, heads, batch_first=True) if kind == "pooled" else None
        self.profile_norm = nn.LayerNorm(width)
        self.query_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.residual = nn.Linear(width, width, bias=kind != 'grounded')
        # Adding this branch preserves the original recognizer exactly at initialization.
        nn.init.zeros_(self.residual.weight)
        if self.residual.bias is not None:
            nn.init.zeros_(self.residual.bias)

    def build(self, ink: torch.Tensor, labels: torch.Tensor, label_padding: torch.Tensor, label_ids: torch.Tensor | None = None) -> torch.Tensor:
        if not 1 <= ink.shape[0] <= 50:
            raise ValueError("A profile requires 1–50 support equations")
        if bool(label_padding.all(dim=1).any()):
            raise ValueError("Each support equation needs a nonempty known label")
        if self.kind in {"aligned", "conditioned", "calibrated", "grounded"}:
            if ink.shape != labels.shape:
                raise ValueError('Aligned profiles require one ink vector per known next token')
            if label_ids is None or label_ids.shape != label_padding.shape:
                raise ValueError('Aligned profiles require known support token IDs')
            ids = label_ids[~label_padding]
            if bool(((ids < 0) | (ids >= self.slots)).any()):
                raise ValueError('Support token ID outside profile vocabulary')
            grounded = ink if self.kind in {'aligned', 'calibrated'} else ink + self.label_projection(labels)
            totals = ink.new_zeros(self.slots, ink.shape[-1])
            totals.index_add_(0, ids, grounded[~label_padding])
            counts = torch.bincount(ids, minlength=self.slots).to(ink)
            profile = totals / counts.clamp_min(1)[:, None]
            if self.kind in {'conditioned', 'grounded'}:
                profile = self.profile_norm(profile).masked_fill(counts[:, None].eq(0), 0)
            return profile.unsqueeze(0)
        aligned, _ = self.ink_alignment(labels, ink, ink, need_weights=False)
        grounded = self.support_norm(aligned + self.label_projection(labels))
        if self.kind == "symbols":
            if label_ids is None or label_ids.shape != label_padding.shape:
                raise ValueError("Symbol profiles require known support token IDs")
            ids = label_ids[~label_padding]
            if bool(((ids < 0) | (ids >= self.slots)).any()):
                raise ValueError("Support token ID outside profile vocabulary")
            totals = grounded.new_zeros(self.slots, grounded.shape[-1])
            totals.index_add_(0, ids, grounded[~label_padding])
            counts = torch.bincount(ids, minlength=self.slots).to(grounded)
            profile = self.profile_norm(totals / counts.clamp_min(1)[:, None])
            # Empty symbols remain absent even after trainable normalization bias.
            return profile.masked_fill(counts[:, None].eq(0), 0).unsqueeze(0)
        valid = grounded[~label_padding].unsqueeze(0)
        profile, _ = self.profile_pool(self.profile_queries, valid, valid, need_weights=False)
        return self.profile_norm(profile)

    def condition(self, memory: torch.Tensor, profile: torch.Tensor | None) -> torch.Tensor:
        if profile is None:
            return memory
        self.validate_profile(profile, memory.shape[0], memory.shape[-1])
        if self.kind in {"aligned", "calibrated"}:
            # Retrieval is applied to NEXT-token logits, not pooled query ink.
            return memory
        if profile.ndim != 3 or profile.shape[1:] != (self.slots, memory.shape[-1]):
            raise ValueError("Writer profile shape does not match this checkpoint")
        if profile.shape[0] not in (1, memory.shape[0]):
            raise ValueError("Writer profile batch does not match query batch")
        profile = profile.expand(memory.shape[0], -1, -1).to(memory)
        padding = profile.abs().sum(dim=-1).eq(0) if self.kind in {"symbols", "conditioned", "grounded"} else None
        if padding is not None and bool(padding.all(dim=1).any()):
            raise ValueError("Symbol profile cannot be empty")
        if self.kind == 'grounded':
            # Compare against public ink with EXACTLY the same covered token set.
            # This cancels label-set/common-style shifts, including affine biases.
            padding = padding | self.reference_counts.eq(0)[None]
            active = ~padding.all(dim=1)
            if not bool(active.any()):
                return memory
            if not bool(active.all()):
                indices = active.nonzero().flatten()
                return memory.index_copy(0, indices, self.condition(memory[active], profile[active]))
            population = self.profile_norm(self.global_reference + self.label_projection(self.label_reference))
            population = population[None].expand_as(profile).to(memory)
            personal, _ = self.query_attention(memory, profile, profile, key_padding_mask=padding, need_weights=False)
            reference, _ = self.query_attention(memory, population, population, key_padding_mask=padding, need_weights=False)
            correction = self.residual(self.query_norm(personal) - self.query_norm(reference))
            # A hard per-position RMS limit; no division/sqrt singularity at init.
            rms = correction.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            limit = .05 * memory.float().square().mean(-1, keepdim=True).sqrt()
            correction = correction * (limit / rms).clamp(max=1).to(correction)
            return memory + correction
        style, _ = self.query_attention(memory, profile, profile, key_padding_mask=padding, need_weights=False)
        return memory + self.residual(self.query_norm(style))

    def validate_profile(self, profile, batch, width):
        if profile.ndim != 3 or profile.shape[1:] != (self.slots, width):
            raise ValueError('Writer profile shape does not match this checkpoint')
        if profile.shape[0] not in (1, batch):
            raise ValueError('Writer profile batch does not match query batch')
        if not bool(torch.isfinite(profile).all()):
            raise ValueError('Writer profile contains nonfinite values')
        if self.kind in {'aligned', 'calibrated'} and bool(profile.abs().sum(dim=-1).eq(0).all(dim=-1).any()):
            raise ValueError('Aligned profile cannot be empty')

    def retrieve_logits(self, logits, aligned_ink, profile):
        """Personalized similarity relative to public training prototypes.

        A covered token gets a correction only for the difference between its
        user's ink and its population ink. Missing tokens receive zero bias.
        Zero-initialized bounded gains preserve the original model initially.
        """
        self.validate_profile(profile, logits.shape[0], aligned_ink.shape[-1])
        profile = profile.expand(logits.shape[0], -1, -1).to(aligned_ink)
        normalize = torch.nn.functional.normalize
        query = normalize(self.retrieval_projection(aligned_ink), dim=-1)
        personal = normalize(self.retrieval_projection(profile), dim=-1)
        population = normalize(self.retrieval_projection(self.global_reference.to(aligned_ink)), dim=-1)
        delta = torch.bmm(query, personal.transpose(1, 2)) - query @ population.T
        covered = profile.abs().sum(dim=-1).ne(0) & self.reference_counts.gt(0)[None]
        gain = 4 * torch.tanh(self.token_gain)
        correction = delta * gain * covered[:, None, :]
        if self.kind == 'calibrated':
            # No common memory shift: only supported symbol logits can move,
            # relative to training-population ink. Confident base decisions bypass
            # the branch exactly, and uncertain decisions get at most one logit.
            confidence = logits.detach().float().softmax(-1).amax(-1, keepdim=True)
            trust = ((0.9 - confidence) / 0.4).clamp(0, 1).to(logits)
            correction = correction.clamp(-1, 1) * trust
        return logits + correction
