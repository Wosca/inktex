from __future__ import annotations

import torch
from torch import nn

from .initialization import initialize_transformer


class LatexDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        max_tokens: int,
        position_encoding: str = "learned",
        coverage: bool = False,
        visible_token_ids: list[int] | None = None,
        eos_id: int = 2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = position_encoding
        self.position_embedding = nn.Embedding(max_tokens, d_model) if position_encoding == "learned" else None
        self.coverage = coverage
        self.eos_id = eos_id
        if coverage:
            self.coverage_strength = nn.Parameter(torch.tensor(0.0))
            self.eos_coverage_strength = nn.Parameter(torch.tensor(2.0))
            visible = torch.zeros(vocab_size, dtype=torch.bool)
            visible[visible_token_ids or []] = True
            self.register_buffer('visible_tokens', visible)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
        initialize_transformer(self.transformer)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)
        self.max_tokens = max_tokens
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        length = tokens.shape[1]
        if self.position_encoding == "relative":
            return self._relative_forward(tokens, memory, memory_padding_mask, token_padding_mask)[0]
        if length > self.max_tokens:
            raise ValueError(f"Target has {length} tokens, decoder maximum is {self.max_tokens}")
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        embedded = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=tokens.device), diagonal=1)
        decoded = self.transformer(
            embedded,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=token_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
            tgt_is_causal=True,
        )
        return self.output(decoded)

    def _relative_forward(self, tokens, memory, memory_mask, token_mask):
        from torch.nn import functional as F
        from .relative_position import sequence_bias
        batch, length = tokens.shape
        heads = self.transformer.layers[0].self_attn.num_heads
        values = self.dropout(self.token_embedding(tokens))
        causal = sequence_bias(length, heads, batch, tokens.device, causal=True).to(values.dtype)
        def floating_mask(mask):
            return None if mask is None else torch.zeros_like(mask, dtype=values.dtype).masked_fill(mask, float('-inf'))
        token_bias, memory_bias = floating_mask(token_mask), floating_mask(memory_mask)
        final_attention = None
        coverage = None
        for index, layer in enumerate(self.transformer.layers):
            query = layer.norm1(values)
            attended = layer.self_attn(query, query, query, attn_mask=causal,
                                       key_padding_mask=token_bias, need_weights=False)[0]
            values = values + layer.dropout1(attended)
            query = layer.norm2(values)
            last = index == len(self.transformer.layers) - 1
            attended, weights = layer.multihead_attn(query, memory, memory, key_padding_mask=memory_bias,
                                                      need_weights=last, average_attn_weights=False)
            if last:
                if memory_mask is not None:
                    weights = weights.masked_fill(memory_mask[:, None, None], 0)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
                if self.coverage:
                    # Row j predicts token j+1. Only observed VISIBLE tokens can
                    # consume ink; braces and TeX commands must not exhaust it.
                    visible = torch.cat((self.visible_tokens[tokens[:, 1:]],
                                         torch.zeros_like(tokens[:, :1], dtype=torch.bool)), 1)
                    consumed = weights * visible[:, None, :, None]
                    coverage = (consumed.cumsum(2) - consumed).clamp_min(0)
                    bias = -F.softplus(self.coverage_strength) * coverage
                    attended, refined = layer.multihead_attn(query, memory, memory,
                        key_padding_mask=memory_bias, attn_mask=bias.flatten(0, 1).to(values.dtype),
                        need_weights=True, average_attn_weights=False)
                    weights = refined
                final_attention = weights.mean(1)
            values = values + layer.dropout2(attended)
            query = layer.norm3(values)
            values = values + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(query)))))
        logits = self.output(self.transformer.norm(values))
        if coverage is not None:
            valid = torch.ones_like(memory[..., 0], dtype=torch.bool) if memory_mask is None else ~memory_mask
            uncovered = (torch.exp(-4 * coverage.mean(1)) * valid[:, None]).sum(-1) / valid.sum(-1)[:, None].clamp_min(1)
            eos = F.one_hot(torch.tensor(self.eos_id, device=tokens.device), logits.shape[-1]).to(logits.dtype)
            logits = logits - F.softplus(self.eos_coverage_strength) * uncovered[..., None] * eos
        return logits, final_attention

    def forward_aligned(self, tokens, memory, memory_padding_mask=None, token_padding_mask=None, last_only=False):
        """Return logits and ink aligned to each NEXT token by trained attention.

        Row t predicts label t+1, so known support labels must be shifted against
        BOS-prefixed decoder inputs. Attention reads the complete ink sequence.
        The temporary hook observes the normal decoder without changing logits.
        """
        if self.position_encoding == "relative":
            logits, weights = self._relative_forward(tokens, memory, memory_padding_mask, token_padding_mask)
            if last_only:
                logits, weights = logits[:, -1:], weights[:, -1:]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            return logits, weights @ memory, weights
        captured = []
        attention = self.transformer.layers[-1].multihead_attn

        def remember(module, args, kwargs):
            captured.append((args, kwargs))

        hook = attention.register_forward_pre_hook(remember, with_kwargs=True)
        try:
            logits = self(tokens, memory, memory_padding_mask, token_padding_mask)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one final decoder cross-attention call')
        args, kwargs = captured[0]
        if last_only:
            args = (args[0][:, -1:], *args[1:])
            logits = logits[:, -1:]
            if token_padding_mask is not None:
                token_padding_mask = token_padding_mask[:, -1:]
        _, weights = attention(*args, **{**kwargs, 'need_weights': True, 'average_attn_weights': True})
        # Re-normalize: dropout can change row sums if used during training.
        if memory_padding_mask is not None:
            weights = weights.masked_fill(memory_padding_mask[:, None, :], 0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        aligned = torch.bmm(weights, memory)
        if token_padding_mask is not None:
            aligned = aligned.masked_fill(token_padding_mask[:, :, None], 0)
        return logits, aligned, weights
