"""BimodalMath: online ink and a rendered strip share one LaTeX decoder.

Decoder memory is ``[stroke tokens, vision tokens]``. Vision tokens are
column-major (``W_f`` outer, ``H_f`` inner): they follow left-to-right reading
order and the padded columns of narrow strips occupy the tail.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import torch
from torch import nn

from .latex_decoder import LatexDecoder
from .stroke_encoder import StrokeEncoder

MODALITY_MODES = ('auto', 'vision', 'stroke', 'bimodal')


@dataclass(slots=True)
class BimodalModelConfig:
    preset: str = "bimodal_75m"
    vocab_size: int = 256
    input_dim: int = 10
    d_model: int = 512
    stroke_encoder_layers: int = 8
    decoder_layers: int = 6
    r2l_decoder_layers: int = 3
    attention_heads: int = 8
    feedforward_dim: int = 2048
    dropout: float = 0.1
    max_points: int = 2048
    max_tokens: int = 512
    position_encoding: str = "relative"
    coverage: bool = True
    visible_token_ids: list[int] | None = None
    eos_id: int = 2
    bos_id: int = 1
    pad_id: int = 0
    vision_backbone: str = "convnext_tiny.fb_in22k_ft_in1k_384"
    vision_in_chans: int = 2
    vision_out_index: int = 2
    vision_pretrained: bool = True
    vision_height: int = 128
    vision_max_width: int = 640
    structure_symbols: list[str] | None = None
    structure_relations: list[str] | None = None
    structure_counter: str = "mean"
    structure_count_conditioning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BimodalModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: val for key, val in value.items() if key in fields})


def get_bimodal_config(preset: str, vocab_size: int, max_points: int = 2048, max_tokens: int = 512,
                       **overrides: Any) -> BimodalModelConfig:
    if preset == "bimodal_75m":
        config = BimodalModelConfig(vocab_size=vocab_size, max_points=max_points, max_tokens=max_tokens)
    elif preset == "bimodal_tiny":
        config = BimodalModelConfig(
            preset="bimodal_tiny", vocab_size=vocab_size, d_model=64, stroke_encoder_layers=2,
            decoder_layers=2, r2l_decoder_layers=1, attention_heads=4, feedforward_dim=128,
            max_points=max_points, max_tokens=max_tokens, vision_backbone="convnext_atto",
            vision_pretrained=False,
        )
    else:
        raise ValueError(f"Unknown bimodal preset: {preset}")
    return replace(config, **overrides)


def sinusoidal_2d(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    """``[height, width, dim]``: first half encodes the row, second half the column."""
    if dim % 4:
        raise ValueError("2D sinusoidal encoding requires d_model divisible by 4")
    quarter = dim // 4
    frequencies = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / quarter))
    rows = torch.arange(height, device=device, dtype=torch.float32)[:, None] * frequencies
    cols = torch.arange(width, device=device, dtype=torch.float32)[:, None] * frequencies
    rows = torch.cat((rows.sin(), rows.cos()), -1)[:, None].expand(height, width, -1)
    cols = torch.cat((cols.sin(), cols.cos()), -1)[None].expand(height, width, -1)
    return torch.cat((rows, cols), -1)


class VisionStripEncoder(nn.Module):
    def __init__(self, backbone: str, d_model: int, in_chans: int = 2, out_index: int = 2,
                 pretrained: bool = True, max_width: int = 640):
        super().__init__()
        import timm
        # timm adapts the pretrained RGB stem to ``in_chans`` when loading weights.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(out_index,), in_chans=in_chans)
        self.channels = self.backbone.feature_info.channels()[-1]
        self.stride = self.backbone.feature_info.reduction()[-1]
        self.projection = nn.Linear(self.channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.max_width = max_width

    def forward(self, images: torch.Tensor, image_widths: torch.Tensor | None = None):
        """``images [B, C, H, W]`` -> tokens ``[B, W_f * H_f, d_model]`` and padding mask ``[B, W_f * H_f]``."""
        if images.shape[-1] > self.max_width:
            raise ValueError(f"Strip width {images.shape[-1]} exceeds maximum {self.max_width}")
        features = self.backbone(images)[-1]
        batch, _, height, width = features.shape
        values = self.norm(self.projection(features.permute(0, 2, 3, 1)))
        values = values + sinusoidal_2d(height, width, self.d_model, values.device).to(values.dtype)
        tokens = values.transpose(1, 2).flatten(1, 2)
        return tokens, self.padding_mask(image_widths, batch, height, width, tokens.device)

    def padding_mask(self, image_widths, batch, height, width, device):
        if image_widths is None:
            return torch.zeros((batch, width * height), dtype=torch.bool, device=device)
        widths = image_widths.to(device=device, dtype=torch.long)
        valid = torch.div(widths + self.stride - 1, self.stride, rounding_mode='floor').clamp(1, width)
        return (torch.arange(width, device=device)[None] >= valid[:, None]).repeat_interleave(height, 1)


class BimodalMathModel(nn.Module):
    def __init__(self, config: BimodalModelConfig):
        super().__init__()
        self.config = config
        width = config.d_model
        self.vision_encoder = VisionStripEncoder(
            config.vision_backbone, width, config.vision_in_chans, config.vision_out_index,
            config.vision_pretrained, config.vision_max_width,
        )
        self.stroke_encoder = StrokeEncoder(
            config.input_dim, width, config.stroke_encoder_layers, config.attention_heads,
            config.feedforward_dim, config.dropout, config.max_points, config.position_encoding,
        )
        if config.structure_symbols:
            if not config.structure_relations or config.structure_relations[0] != 'None':
                raise ValueError('Structural relation vocabulary must begin with None')
            self.stroke_encoder.enable_structure(len(config.structure_symbols), len(config.structure_relations))
        elif config.position_encoding == "relative":
            # Relative encoding pools points into stroke memory through the
            # structure module; without a vocabulary its auxiliary heads are unused.
            self.stroke_encoder.enable_structure(1, 1)
        if config.structure_counter != 'mean' or config.structure_count_conditioning:
            if not config.structure_symbols:
                raise ValueError('Count variants require a structural vocabulary')
            self.stroke_encoder.structure.configure_counter(config.structure_counter,
                                                            config.structure_count_conditioning)
        self.stroke_norm = nn.LayerNorm(width)
        self.vision_norm = nn.LayerNorm(width)
        self.stroke_modality_emb = nn.Parameter(torch.zeros(1, 1, width))
        self.vision_modality_emb = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.stroke_modality_emb, std=0.02)
        nn.init.normal_(self.vision_modality_emb, std=0.02)
        self.decoder = LatexDecoder(
            config.vocab_size, width, config.decoder_layers, config.attention_heads,
            config.feedforward_dim, config.dropout, config.max_tokens, config.position_encoding,
            config.coverage, config.visible_token_ids, config.eos_id,
        )
        self.r2l_decoder = None
        if config.r2l_decoder_layers:
            self.r2l_decoder = LatexDecoder(
                config.vocab_size, width, config.r2l_decoder_layers, config.attention_heads,
                config.feedforward_dim, config.dropout, config.max_tokens, config.position_encoding,
                False, None, config.eos_id,
            )
            self.r2l_decoder.token_embedding = self.decoder.token_embedding
            self.r2l_decoder.output.weight = self.decoder.token_embedding.weight

    def encode_bimodal(self, images=None, image_widths=None, stroke_features=None, stroke_padding_mask=None,
                       has_strokes=None, modality_mode="auto", vision_drop_prob=0.0, stroke_drop_prob=0.0,
                       return_structure=False):
        """Return ``(memory, memory_padding_mask, structure)``.

        Images cover every sample. Strokes are encoded only for samples with ink
        and ``has_strokes``; other rows get fully masked stroke slots. Structure
        outputs cover encoded stroke rows, listed in ``structure['batch_index']``.
        """
        if modality_mode not in MODALITY_MODES:
            raise ValueError(f"Unknown modality mode: {modality_mode}")
        if modality_mode == "vision":
            stroke_features = None
        elif modality_mode == "stroke":
            images = None
        elif modality_mode == "bimodal" and (images is None or stroke_features is None):
            raise ValueError("Bimodal mode requires images and stroke features")
        if images is None and stroke_features is None:
            raise ValueError("Provide images, stroke features, or both")
        if images is not None and stroke_features is not None and images.shape[0] != stroke_features.shape[0]:
            raise ValueError("Image and stroke batches differ in size")
        if min(vision_drop_prob, stroke_drop_prob) < 0 or vision_drop_prob + stroke_drop_prob > 1:
            raise ValueError("Modality drop probabilities must be non-negative and sum to at most 1")
        vision = None
        if images is not None:
            memory, mask = self.vision_encoder(images, image_widths)
            vision = self.vision_norm(memory) + self.vision_modality_emb, mask
        stroke, structure = None, None
        if stroke_features is not None:
            stroke, structure = self._encode_strokes(stroke_features, stroke_padding_mask, has_strokes,
                                                     return_structure, required=vision is None)
        if stroke is None:
            return (*vision, None)
        if vision is None:
            return (*stroke, structure)
        (stroke_memory, stroke_mask), (vision_memory, vision_mask) = stroke, vision
        if self.training and (vision_drop_prob or stroke_drop_prob):
            # Every vision row keeps at least one column, so ``both`` rows can lose either stream.
            both = (~stroke_mask).any(1)
            draw = torch.rand(both.shape, device=both.device)
            drop_vision = both & draw.lt(vision_drop_prob)
            drop_stroke = both & draw.ge(vision_drop_prob) & draw.lt(vision_drop_prob + stroke_drop_prob)
            vision_mask = vision_mask | drop_vision[:, None]
            stroke_mask = stroke_mask | drop_stroke[:, None]
        return (torch.cat((stroke_memory, vision_memory), 1), torch.cat((stroke_mask, vision_mask), 1),
                structure)

    def _encode_strokes(self, features, padding_mask, has_strokes, return_structure, required):
        batch = features.shape[0]
        if padding_mask is None:
            padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
        present = ((features[..., 7] > .5) & ~padding_mask).any(1)
        if has_strokes is not None:
            present = present & has_strokes.to(device=present.device, dtype=torch.bool)
        rows = present.nonzero().squeeze(1)
        if required and len(rows) != batch:
            raise ValueError("Every sample without an image needs ink")
        if not len(rows):
            return None, None
        subset = len(rows) != batch
        if subset:
            features, padding_mask = features[rows], padding_mask[rows]
        structure = None
        if return_structure and self.config.structure_symbols:
            memory, structure = self.stroke_encoder.forward_with_structure(features, padding_mask)
            structure.pop('point_memory', None)
            structure['batch_index'] = rows
        else:
            memory = self.stroke_encoder(features, padding_mask)
        mask = self.stroke_encoder.memory_padding_mask(features, padding_mask, memory)
        memory = self.stroke_norm(memory) + self.stroke_modality_emb
        if subset:
            memory = memory.new_zeros((batch, *memory.shape[1:])).index_copy(0, rows, memory)
            mask = mask.new_ones((batch, mask.shape[1])).index_copy(0, rows, mask)
        return (memory, mask), structure

    def forward(self, decoder_input, token_padding_mask=None, r2l_decoder_input=None, r2l_token_padding_mask=None,
                *, images=None, image_widths=None, stroke_features=None, stroke_padding_mask=None,
                has_strokes=None, modality_mode="auto", vision_drop_prob=0.0, stroke_drop_prob=0.0,
                return_structure=False) -> dict[str, Any]:
        memory, memory_mask, structure = self.encode_bimodal(
            images, image_widths, stroke_features, stroke_padding_mask, has_strokes, modality_mode,
            vision_drop_prob, stroke_drop_prob, return_structure,
        )
        logits = self.decoder(decoder_input, memory, memory_mask, token_padding_mask)
        r2l_logits = None
        if r2l_decoder_input is not None:
            if self.r2l_decoder is None:
                raise ValueError("This model has no right-to-left decoder")
            r2l_logits = self.r2l_decoder(r2l_decoder_input, memory, memory_mask, r2l_token_padding_mask)
        return {"logits": logits, "r2l_logits": r2l_logits, "structure": structure}

    @torch.inference_mode()
    def decode_greedy(self, images=None, image_widths=None, stroke_features=None, stroke_padding_mask=None,
                      has_strokes=None, max_tokens=None, cached=True) -> torch.Tensor:
        """Greedy L2R tokens ``[B, L]`` without BOS; positions after EOS hold ``pad_id``."""
        self.eval()
        memory, memory_mask, _ = self.encode_bimodal(images, image_widths, stroke_features,
                                                     stroke_padding_mask, has_strokes)
        limit = max_tokens or self.config.max_tokens
        if self.config.position_encoding == "learned":
            limit = min(limit, self.config.max_tokens)
        batch = memory.shape[0]
        tokens = torch.full((batch, 1), self.config.bos_id, dtype=torch.long, device=memory.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        state = None
        if cached:
            from research.incremental import DecoderState
            state = DecoderState(self.decoder, memory, memory_mask)
        for _ in range(limit - 1):
            if state is not None:
                logits = state.step(tokens[:, -1:])[0][:, -1]
            else:
                logits = self.decoder(tokens, memory, memory_mask)[:, -1]
            next_token = logits.argmax(-1).masked_fill(finished, self.config.pad_id)
            tokens = torch.cat((tokens, next_token[:, None]), 1)
            finished |= next_token.eq(self.config.eos_id)
            if bool(finished.all()):
                break
        return tokens[:, 1:]

    def parameter_summary(self) -> dict[str, int]:
        def count(*parts, exclude=()):
            skip = {id(parameter) for parameter in exclude}
            parameters = {id(p): p for part in parts
                          for p in (part.parameters() if isinstance(part, nn.Module) else [part])}
            return sum(p.numel() for key, p in parameters.items() if key not in skip)

        summary = {
            "vision": count(self.vision_encoder),
            "stroke": count(self.stroke_encoder),
            "fusion": count(self.stroke_norm, self.vision_norm, self.stroke_modality_emb, self.vision_modality_emb),
            "decoder": count(self.decoder),
            "r2l_decoder": count(self.r2l_decoder, exclude=list(self.decoder.parameters())) if self.r2l_decoder else 0,
            "total": count(self),
        }
        summary["inference_total"] = summary["total"] - summary["r2l_decoder"]
        return summary
