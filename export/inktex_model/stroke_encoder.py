from __future__ import annotations

import torch
from torch import nn

from .initialization import initialize_transformer


class StrokeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        max_points: int,
        position_encoding: str = "learned",
    ):
        super().__init__()
        self.feature_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.spatial_projection = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.position_encoding = position_encoding
        self.position_embedding = nn.Embedding(max_points, d_model) if position_encoding == "learned" else None
        if position_encoding not in ("learned", "relative"):
            raise ValueError("Unknown positional encoding")
        self.boundary_embedding = nn.Embedding(4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Nested tensors currently do not support norm_first; disable the optimisation explicitly.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        initialize_transformer(self.transformer)
        self.dropout = nn.Dropout(dropout)
        self.max_points = max_points
        self.structure = None
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.boundary_embedding.weight, mean=0.0, std=0.02)

    def enable_structure(self, symbol_classes: int, relation_classes: int) -> None:
        from .stroke_structure import StrokeStructure
        block = self.transformer.layers[0]
        self.structure = StrokeStructure(block.self_attn.embed_dim, block.self_attn.num_heads,
                                         block.linear1.out_features, block.dropout.p,
                                         symbol_classes, relation_classes).to(self.feature_projection[0].weight.device)
        self.structure.relative_positions = self.position_encoding == "relative"

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._encode(features, padding_mask, None)[0]

    def forward_with_structure(self, features, padding_mask):
        return self._encode(features, padding_mask, 'structure')

    def forward_with_characters(self, features, padding_mask):
        return self._encode(features, padding_mask, 'characters')

    def _encode(self, features, padding_mask, output_kind):
        length = features.shape[1]
        if length > self.max_points and self.position_encoding == "learned":
            raise ValueError(f"Input has {length} points, encoder maximum is {self.max_points}")
        positions = torch.arange(length, device=features.device).unsqueeze(0)
        starts = (features[:, :, 7] > 0.5).long()
        ends = (features[:, :, 8] > 0.5).long()
        boundary_state = starts + 2 * ends
        layout = None
        local = None
        projected_features = features
        if self.position_encoding == "relative":
            if self.structure is None:
                raise ValueError("The hierarchical relative encoder requires stroke structure")
            if padding_mask is None:
                padding_mask = torch.zeros(features.shape[:2], device=features.device, dtype=torch.bool)
            local, layout = self.structure.local_features(features, padding_mask)
            ids, _, _, _, center, extent = layout
            point_center = center.gather(1, ids[..., None].expand(-1, -1, 2))
            scale = extent.amax(-1, keepdim=True).gather(1, ids[..., None])
            projected_features = torch.cat(((features[..., :2] - point_center) / scale,
                                            features[..., 2:4] / scale, features[..., 4:9],
                                            torch.zeros_like(features[..., 9:10])), -1)
        embedded = (
            self.feature_projection(projected_features)
            + self.spatial_projection(projected_features[:, :, :2])
            + self.boundary_embedding(boundary_state)
        )
        if self.position_embedding is not None:
            embedded = embedded + self.position_embedding(positions)
        if self.structure is not None:
            if padding_mask is None:
                padding_mask = torch.zeros(features.shape[:2], device=features.device, dtype=torch.bool)
            if layout is None:
                local, layout = self.structure.local_features(features, padding_mask)
            embedded = embedded + self.structure.input_gate.tanh() * local
        if self.position_encoding == "relative":
            memory = self._local_encode(self.dropout(embedded), padding_mask)
        else:
            memory = self.transformer(self.dropout(embedded), src_key_padding_mask=padding_mask)
        outputs = None
        if self.structure is not None:
            memory, contextual = self.structure.enrich(memory, layout)
            if output_kind == 'structure':
                outputs = self.structure.auxiliary(contextual, layout)
        if output_kind is not None and (self.position_encoding == 'relative' or output_kind == 'characters'):
            outputs = outputs or {}
            outputs['point_memory'] = memory
        if self.position_encoding == "relative":
            # The decoder attends to strokes, not thousands of individual points.
            memory = contextual
        return memory, outputs

    def memory_padding_mask(self, features, padding_mask, memory):
        if self.position_encoding == "learned":
            return padding_mask
        counts = ((features[..., 7] > .5) & ~padding_mask).sum(1)
        return torch.arange(memory.shape[1], device=memory.device)[None] >= counts[:, None]

    def _local_encode(self, embedded, padding_mask):
        from torch.nn import functional as F
        from .relative_position import encode_relative, sequence_bias
        batch, length, width = embedded.shape
        window = 64
        # Keep at least two windows, including an empty sentinel. This avoids
        # export specializing the one-window shape and preserves dynamic input
        # lengths on both sides of a window boundary.
        extra = 2 * window - length % window
        values = F.pad(embedded, (0, 0, 0, extra)).reshape(-1, window, width)
        mask = F.pad(padding_mask, (0, extra), value=True).reshape(-1, window)
        # An empty window needs a dummy key to avoid all-masked attention NaNs.
        empty = mask.all(1)
        first = torch.arange(window, device=mask.device).eq(0)[None]
        safe_mask = mask & ~(empty[:, None] & first)
        heads = self.transformer.layers[0].self_attn.num_heads
        bias = sequence_bias(window, heads, values.shape[0], values.device)
        mask_bias = torch.zeros_like(safe_mask, dtype=values.dtype).masked_fill(safe_mask, float('-inf'))
        encoded = encode_relative(self.transformer, values, bias.to(values.dtype), mask_bias)
        encoded = encoded.masked_fill(mask[..., None], 0)
        return encoded.reshape(batch, -1, width)[:, :length]
