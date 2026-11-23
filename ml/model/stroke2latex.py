from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from typing import Any

import torch
from torch import nn

from .latex_decoder import LatexDecoder
from .stroke_encoder import StrokeEncoder
from .writer_context import WriterContext


@dataclass(slots=True)
class ModelConfig:
    preset: str = "tiny"
    input_dim: int = 10
    vocab_size: int = 128
    d_model: int = 192
    encoder_layers: int = 4
    decoder_layers: int = 3
    attention_heads: int = 6
    feedforward_dim: int = 512
    dropout: float = 0.1
    max_points: int = 1024
    max_tokens: int = 256
    writer_context_slots: int = 0
    writer_context_kind: str = "pooled"
    structure_symbols: list[str] | None = None
    structure_relations: list[str] | None = None
    structure_counter: str = 'mean'
    structure_count_conditioning: bool = False
    position_encoding: str = "learned"
    coverage: bool = False
    visible_token_ids: list[int] | None = None
    character_ctc: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: val for key, val in value.items() if key in fields})


def get_model_config(preset: str, vocab_size: int, max_points: int = 1024, max_tokens: int = 256) -> ModelConfig:
    if preset == "tiny":
        return ModelConfig(preset="tiny", vocab_size=vocab_size, max_points=max_points, max_tokens=max_tokens)
    if preset == "small":
        return ModelConfig(
            preset="small", vocab_size=vocab_size, d_model=256, encoder_layers=6, decoder_layers=4,
            attention_heads=8, feedforward_dim=768, max_points=max_points, max_tokens=max_tokens,
        )
    if preset == "compact":
        return ModelConfig(
            preset="compact", vocab_size=vocab_size, d_model=384, encoder_layers=6, decoder_layers=4,
            attention_heads=8, feedforward_dim=1152, max_points=max_points, max_tokens=max_tokens,
        )
    if preset == "adaptive":
        return ModelConfig(
            preset="adaptive", vocab_size=vocab_size, d_model=320, encoder_layers=6, decoder_layers=4,
            attention_heads=8, feedforward_dim=960, max_points=max_points, max_tokens=max_tokens,
        )
    if preset == "teacher":
        return ModelConfig(
            preset="teacher", vocab_size=vocab_size, d_model=512, encoder_layers=8, decoder_layers=6,
            attention_heads=8, feedforward_dim=2048, max_points=max_points, max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown model preset: {preset}")


class Stroke2Latex(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = StrokeEncoder(
            config.input_dim, config.d_model, config.encoder_layers, config.attention_heads,
            config.feedforward_dim, config.dropout, config.max_points,
            config.position_encoding,
        )
        self.decoder = LatexDecoder(
            config.vocab_size, config.d_model, config.decoder_layers, config.attention_heads,
            config.feedforward_dim, config.dropout, config.max_tokens,
            config.position_encoding, config.coverage, config.visible_token_ids,
        )
        self.writer_context = WriterContext(config.d_model, config.attention_heads, config.writer_context_slots, config.writer_context_kind) if config.writer_context_slots else None
        self.character_head = nn.Linear(config.d_model, config.vocab_size) if config.character_ctc else None
        if config.structure_symbols:
            if not config.structure_relations or config.structure_relations[0] != 'None':
                raise ValueError('Structural relation vocabulary must begin with None')
            self.encoder.enable_structure(len(config.structure_symbols), len(config.structure_relations))
            self.encoder.structure.configure_counter(config.structure_counter, config.structure_count_conditioning)
        elif config.structure_counter != 'mean' or config.structure_count_conditioning:
            raise ValueError('Count variants require the structural encoder')

    def enable_structure(self, symbols: list[str], relations: list[str]) -> None:
        if self.encoder.structure is not None:
            if symbols != self.config.structure_symbols or relations != self.config.structure_relations:
                raise ValueError('Structural annotation vocabulary differs from checkpoint')
            return
        if not symbols or not relations or relations[0] != 'None':
            raise ValueError('Invalid structural annotation vocabulary')
        self.config.structure_symbols = list(symbols)
        self.config.structure_relations = list(relations)
        self.encoder.enable_structure(len(symbols), len(relations))

    def configure_structure_counter(self, counter='mean', condition_counts=False):
        if self.encoder.structure is None:
            raise ValueError('Enable structural encoding before its count variant')
        self.encoder.structure.configure_counter(counter, condition_counts)
        self.config.structure_counter = counter
        self.config.structure_count_conditioning = bool(condition_counts)

    def forward_with_structure(self, features, point_padding_mask, decoder_input, token_padding_mask=None):
        memory, auxiliary = self.encoder.forward_with_structure(features, point_padding_mask)
        memory_mask = self.encoder.memory_padding_mask(features, point_padding_mask, memory)
        if self.character_head is not None:
            auxiliary = auxiliary or {}
            auxiliary['ctc_logits'] = self.character_head(auxiliary.pop('point_memory', memory))
        elif auxiliary:
            auxiliary.pop('point_memory', None)
        return self.decode(decoder_input, memory, memory_mask, token_padding_mask), auxiliary

    def encode(self, features, point_padding_mask):
        memory = self.encoder(features, point_padding_mask)
        return memory, self.encoder.memory_padding_mask(features, point_padding_mask, memory)

    @torch.inference_mode()
    def recognize_graph(self, features, point_padding_mask):
        from .layout_graph import assemble_prediction
        if self.encoder.structure is None:
            raise ValueError('This checkpoint has no layout recognizer')
        _, outputs = self.encoder.forward_with_structure(features, point_padding_mask)
        return [assemble_prediction(outputs, self.config.structure_symbols, self.config.structure_relations, row)
                for row in range(features.shape[0])]

    @torch.inference_mode()
    def transcribe_text(self, features, point_padding_mask, tokenizer):
        """Parallel character recognition for words/lines; no language-model EOS."""
        if self.character_head is None:
            raise ValueError('This checkpoint has no character recognizer')
        _, outputs = self.encoder.forward_with_characters(features, point_padding_mask)
        logits = self.character_head(outputs['point_memory'])
        allowed = torch.tensor([i == tokenizer.pad_id or len(token) == 1 for i, token in enumerate(tokenizer.tokens)],
                               device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(~allowed, -float('inf'))
        ids = logits.argmax(-1).masked_fill(point_padding_mask, tokenizer.pad_id)
        results = []
        for row in ids:
            collapsed = row.unique_consecutive().tolist()
            results.append(''.join(tokenizer.tokens[i] for i in collapsed if i != tokenizer.pad_id))
        return results

    def enable_writer_context(self, slots: int = 32, kind: str = "pooled") -> None:
        if self.writer_context is not None:
            raise ValueError("Writer context is already enabled")
        if kind == 'grounded' and self.decoder.position_embedding is not None:
            raise ValueError('Grounded writer profiles require a relative-position checkpoint')
        if kind in {"symbols", "aligned", "conditioned", "calibrated", "grounded"}:
            slots = self.config.vocab_size
        if not 1 <= slots <= 512:
            raise ValueError("Writer context slots must be between 1 and 512")
        self.config.writer_context_slots = slots
        self.config.writer_context_kind = kind
        self.writer_context = WriterContext(self.config.d_model, self.config.attention_heads, slots, kind).to(next(self.parameters()).device)
        if kind == 'grounded':
            self.writer_context.label_reference.copy_(self.decoder.token_embedding.weight.detach())

    def condition_memory(self, memory: torch.Tensor, writer_profile: torch.Tensor | None = None) -> torch.Tensor:
        if writer_profile is not None and self.writer_context is None:
            raise ValueError("This checkpoint does not support writer profiles")
        return self.writer_context.condition(memory, writer_profile) if self.writer_context is not None else memory

    def forward(
        self,
        features: torch.Tensor,
        point_padding_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        token_padding_mask: torch.Tensor | None = None,
        writer_profile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, memory_mask = self.encode(features, point_padding_mask)
        return self.decode(decoder_input, memory, memory_mask, token_padding_mask, writer_profile)

    def decode(self, tokens, memory, memory_mask=None, token_mask=None, writer_profile=None, last_only=False):
        if writer_profile is not None and self.writer_context is not None and self.writer_context.kind in {'aligned', 'calibrated'}:
            self.writer_context.validate_profile(writer_profile, tokens.shape[0], memory.shape[-1])
            logits, ink, _ = self.decoder.forward_aligned(tokens, memory, memory_mask, token_mask, last_only=last_only)
            return self.writer_context.retrieve_logits(logits, ink, writer_profile)
        return self.decoder(tokens, self.condition_memory(memory, writer_profile), memory_mask, token_mask)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def expand_depth(self, encoder_layers: int, decoder_layers: int) -> None:
        """Add identity residual blocks, preserving pretrained predictions initially."""
        for transformer, requested in ((self.encoder.transformer, encoder_layers), (self.decoder.transformer, decoder_layers)):
            if requested < len(transformer.layers):
                raise ValueError("Cannot remove pretrained layers")
            for _ in range(requested - len(transformer.layers)):
                layer = copy.deepcopy(transformer.layers[-1])
                with torch.no_grad():
                    for projection in (layer.self_attn.out_proj, layer.linear2):
                        projection.weight.zero_()
                        projection.bias.zero_()
                    if hasattr(layer, "multihead_attn"):
                        layer.multihead_attn.out_proj.weight.zero_()
                        layer.multihead_attn.out_proj.bias.zero_()
                transformer.layers.append(layer)
            transformer.num_layers = len(transformer.layers)
        self.config.encoder_layers = encoder_layers
        self.config.decoder_layers = decoder_layers

    @torch.no_grad()
    def resize_vocabulary(self, vocab_size: int, initialize_from: dict[int, int] | None = None) -> None:
        """Grow tied decoder embeddings while preserving every existing row."""
        old_embedding = self.decoder.token_embedding
        old_size, width = old_embedding.weight.shape
        if vocab_size < old_size:
            raise ValueError("Vocabulary shrinking is not supported")
        if vocab_size == old_size:
            return
        new_embedding = nn.Embedding(
            vocab_size, width, device=old_embedding.weight.device, dtype=old_embedding.weight.dtype
        )
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
        new_embedding.weight[:old_size].copy_(old_embedding.weight)
        for destination, source in (initialize_from or {}).items():
            if not (old_size <= destination < vocab_size and 0 <= source < old_size):
                raise ValueError("Vocabulary initializer indices are out of range")
            new_embedding.weight[destination].copy_(old_embedding.weight[source])
        new_output = nn.Linear(
            width, vocab_size, bias=False,
            device=old_embedding.weight.device, dtype=old_embedding.weight.dtype,
        )
        new_output.weight = new_embedding.weight
        self.decoder.token_embedding = new_embedding
        self.decoder.output = new_output
        if self.character_head is not None:
            old_head = self.character_head
            self.character_head = nn.Linear(width, vocab_size, device=old_head.weight.device, dtype=old_head.weight.dtype)
            self.character_head.weight[:old_size].copy_(old_head.weight)
            self.character_head.bias[:old_size].copy_(old_head.bias)
        if self.decoder.coverage:
            visible = torch.zeros(vocab_size, device=new_embedding.weight.device, dtype=torch.bool)
            visible[:old_size] = self.decoder.visible_tokens
            self.decoder.visible_tokens = visible
        self.config.vocab_size = vocab_size

    @torch.inference_mode()
    def generate(
        self,
        features: torch.Tensor,
        point_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_tokens: int | None = None,
        beam_width: int = 1,
        writer_profile: torch.Tensor | None = None,
        return_nbest: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if beam_width not in (1, 3, 5):
            raise ValueError("beam_width must be 1, 3, or 5")
        if return_nbest and beam_width == 1:
            raise ValueError("N-best output requires beam search")
        memory, point_padding_mask = self.encode(features, point_padding_mask)
        memory = self.condition_memory(memory, writer_profile)
        limit = max_tokens or self.config.max_tokens
        if self.config.position_encoding == 'learned':
            limit = min(limit, self.config.max_tokens)
        if beam_width == 1:
            return self._greedy(memory, point_padding_mask, bos_id, eos_id, pad_id, limit,
                                writer_profile=writer_profile if self.config.writer_context_kind in {'aligned', 'calibrated'} else None)
        return self._beam(memory, point_padding_mask, bos_id, eos_id, pad_id, limit, beam_width,
                          writer_profile=writer_profile if self.config.writer_context_kind in {'aligned', 'calibrated'} else None,
                          return_nbest=return_nbest)

    def _greedy(self, memory, memory_mask, bos_id, eos_id, pad_id, limit, writer_profile=None):
        batch_size = memory.shape[0]
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=memory.device)
        confidences: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        for _ in range(limit - 1):
            logits = self.decode(generated, memory, memory_mask, writer_profile=writer_profile, last_only=True)[:, -1]
            probabilities = torch.softmax(logits, dim=-1)
            confidence, next_token = probabilities.max(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
            confidence = torch.where(finished, torch.ones_like(confidence), confidence)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            confidences.append(confidence)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
        confidence_tensor = torch.stack(confidences, dim=1) if confidences else torch.empty((batch_size, 0), device=memory.device)
        return {"token_ids": generated[:, 1:], "token_confidences": confidence_tensor, 'truncated': ~finished}

    def _beam(self, memory, memory_mask, bos_id, eos_id, pad_id, limit, width, writer_profile=None,
              *, return_nbest=False):
        """Decode beams together, preserving the existing length-normalized search.

        Completed beams retain their score and true length while their output is
        padded. Candidate selection is independent for every input expression.
        """
        batch, device = memory.shape[0], memory.device
        expanded_memory = memory.repeat_interleave(width, dim=0)
        expanded_mask = memory_mask.repeat_interleave(width, dim=0) if memory_mask is not None else None
        if writer_profile is not None and writer_profile.shape[0] != 1:
            writer_profile = writer_profile.repeat_interleave(width, dim=0)
        sequences = torch.full((batch, width, 1), bos_id, device=device, dtype=torch.long)
        confidences = torch.empty((batch, width, 0), device=device)
        scores = torch.full((batch, width), -float('inf'), device=device)
        scores[:, 0] = 0
        lengths = torch.zeros((batch, width), device=device, dtype=torch.long)
        finished = torch.zeros((batch, width), device=device, dtype=torch.bool)
        batch_ids = torch.arange(batch, device=device)[:, None]
        for _ in range(limit - 1):
            prefix = sequences.reshape(batch * width, -1)
            logits = self.decode(prefix, expanded_memory, expanded_mask,
                                 writer_profile=writer_profile, last_only=True)[:, -1]
            log_probs = torch.log_softmax(logits.float(), -1).reshape(batch, width, -1)
            # Finished hypotheses contribute exactly one unchanged candidate.
            log_probs = log_probs.masked_fill(finished[..., None], -float('inf'))
            log_probs[:, :, pad_id] = torch.where(finished, 0., log_probs[:, :, pad_id])
            values, tokens = log_probs.topk(width, dim=-1)
            candidate_scores = scores[..., None] + values
            candidate_lengths = lengths + (~finished).long()
            ranked = candidate_scores / candidate_lengths.clamp_min(1).float()[..., None].pow(.7)
            chosen = ranked.reshape(batch, -1).topk(width, dim=-1).indices
            parents = chosen // width
            emitted = tokens.reshape(batch, -1).gather(1, chosen)
            probabilities = values.reshape(batch, -1).gather(1, chosen).exp()
            scores = candidate_scores.reshape(batch, -1).gather(1, chosen)
            lengths = candidate_lengths.gather(1, parents)
            finished = finished.gather(1, parents) | emitted.eq(eos_id)
            sequences = torch.cat((sequences[batch_ids, parents], emitted[..., None]), -1)
            confidences = torch.cat((confidences[batch_ids, parents], probabilities[..., None]), -1)
            if bool(finished.all()):
                break
        result = {
            "token_ids": sequences[:, 0, 1:],
            "token_confidences": confidences[:, 0],
            'truncated': ~finished[:, 0],
        }
        if return_nbest:
            result["nbest"] = {
                "token_ids": sequences[:, :, 1:],
                "lengths": lengths,
                "log_probs": scores,
                "truncated": ~finished,
            }
        return result
