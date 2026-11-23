import json

import pytest
import torch

from model.bimodal import BimodalMathModel, BimodalModelConfig, get_bimodal_config
from research.incremental import DecoderState

VOCAB = 32
STROKE_COUNTS = (3, 2, 4, 1)


def tiny(**overrides):
    torch.manual_seed(0)
    config = get_bimodal_config("bimodal_tiny", VOCAB, visible_token_ids=list(range(4, VOCAB)), **overrides)
    return BimodalMathModel(config)


def strokes(counts=STROKE_COUNTS, points_per_stroke=4):
    generator = torch.Generator().manual_seed(1)
    length = max(counts) * points_per_stroke + 3
    features = torch.zeros(len(counts), length, 10)
    mask = torch.ones(len(counts), length, dtype=torch.bool)
    for row, count in enumerate(counts):
        n = count * points_per_stroke
        features[row, :n, :2] = torch.rand(n, 2, generator=generator)
        features[row, :n, 2:4] = torch.randn(n, 2, generator=generator) * .05
        features[row, :n:points_per_stroke, 7] = 1
        features[row, points_per_stroke - 1:n:points_per_stroke, 8] = 1
        mask[row, :n] = False
    return features, mask


def strips(widths=(128, 96, 160, 64)):
    generator = torch.Generator().manual_seed(2)
    images = torch.zeros(len(widths), 2, 128, max(widths))
    for row, width in enumerate(widths):
        images[row, :, :, :width] = torch.rand(2, 128, width, generator=generator)
    return images, torch.tensor(widths)


def tokens(batch=4, length=7):
    generator = torch.Generator().manual_seed(3)
    values = torch.randint(4, VOCAB, (batch, length), generator=generator)
    values[:, 0] = 1
    return values


def calls(module):
    seen = []
    module.register_forward_hook(lambda *_: seen.append(1))
    return seen


def grads(module):
    return [p.grad for p in module.parameters()]


def assert_finite_grads(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_vision_only_skips_stroke_encoder():
    model = tiny().train()
    stroke_calls = calls(model.stroke_encoder)
    images, widths = strips()
    out = model(tokens(), images=images, image_widths=widths)
    assert out["logits"].shape == (4, 7, VOCAB) and out["structure"] is None
    out["logits"].float().logsumexp(-1).mean().backward()
    assert not stroke_calls
    assert all(g is None for g in grads(model.stroke_encoder)) and model.stroke_modality_emb.grad is None
    assert any(g is not None and g.abs().sum() > 0 for g in grads(model.vision_encoder))
    assert_finite_grads(model)


def test_stroke_only_skips_vision_encoder():
    model = tiny().train()
    vision_calls = calls(model.vision_encoder)
    features, mask = strokes()
    memory, memory_mask, structure = model.encode_bimodal(stroke_features=features, stroke_padding_mask=mask,
                                                          return_structure=True)
    assert memory.shape == (4, max(STROKE_COUNTS), 64) and structure is None
    assert (~memory_mask).sum(1).tolist() == list(STROKE_COUNTS)
    out = model(tokens(), stroke_features=features, stroke_padding_mask=mask)
    out["logits"].float().logsumexp(-1).mean().backward()
    assert not vision_calls
    assert all(g is None for g in grads(model.vision_encoder)) and model.vision_modality_emb.grad is None
    assert any(g is not None for g in grads(model.stroke_encoder))
    assert_finite_grads(model)
    with pytest.raises(ValueError, match="needs ink"):
        model.encode_bimodal(stroke_features=features, stroke_padding_mask=mask,
                             has_strokes=torch.tensor([True, False, True, True]))


def test_mixed_batch_masks_offline_rows_and_stays_finite():
    model = tiny(structure_symbols=["x", "2", "+"], structure_relations=["None", "Right", "Sup"]).train()
    features, mask = strokes()
    has_strokes = torch.tensor([True, False, True, False])
    features[~has_strokes] = float("nan")
    mask[~has_strokes] = True
    images, widths = strips()
    memory, memory_mask, structure = model.encode_bimodal(images, widths, features, mask, has_strokes,
                                                          return_structure=True)
    stroke_slots = max(STROKE_COUNTS[0], STROKE_COUNTS[2])
    vision_tokens = 8 * (images.shape[-1] // 16)
    assert memory.shape == (4, stroke_slots + vision_tokens, 64)
    assert memory_mask.shape == (4, stroke_slots + vision_tokens) and memory_mask.dtype == torch.bool
    assert (~memory_mask[:, :stroke_slots]).sum(1).tolist() == [3, 0, 4, 0]
    assert structure["batch_index"].tolist() == [0, 2]
    assert structure["symbol"].shape == (2, stroke_slots, 3)
    assert torch.isfinite(memory).all()
    out = model(tokens(), images=images, image_widths=widths, stroke_features=features, stroke_padding_mask=mask,
                has_strokes=has_strokes, return_structure=True)
    assert torch.isfinite(out["logits"]).all() and out["r2l_logits"] is None
    (out["logits"].float().logsumexp(-1).mean() + out["structure"]["symbol"].float().pow(2).mean()).backward()
    assert_finite_grads(model)
    assert any(g is not None for g in grads(model.stroke_encoder))


def test_variable_widths_align_column_major_padding():
    model = tiny().eval()
    widths = (64, 256, 512, 72)
    images, image_widths = strips(widths)
    memory, mask = model.vision_encoder(images, image_widths)
    columns = 512 // 16
    assert memory.shape == (4, columns * 8, 64) and mask.shape == (4, columns * 8)
    for row, width in enumerate(widths):
        valid = -(-width // 16)
        expected = torch.arange(columns).repeat_interleave(8) >= valid
        assert torch.equal(mask[row], expected)
    assert not model.vision_encoder(images)[1].any()
    for width in (64, 256, 512):
        single = torch.rand(1, 2, 128, width)
        assert model.vision_encoder(single, torch.tensor([width]))[0].shape == (1, width // 16 * 8, 64)


def test_modality_dropout_never_masks_a_whole_sample():
    model = tiny().train()
    features, mask = strokes()
    images, widths = strips()
    has_strokes = torch.tensor([True, False, True, True])
    slots = max(STROKE_COUNTS)
    dropped = {"vision": 0, "stroke": 0}
    for _ in range(64):
        _, memory_mask, _ = model.encode_bimodal(images, widths, features, mask, has_strokes,
                                                 vision_drop_prob=.5, stroke_drop_prob=.5)
        assert (~memory_mask).any(1).all()
        stroke_lost = memory_mask[:, :slots].all(1)
        vision_lost = memory_mask[:, slots:].all(1)
        assert not (stroke_lost & vision_lost).any()
        assert torch.equal(stroke_lost[[0, 2, 3]] ^ vision_lost[[0, 2, 3]], torch.ones(3, dtype=torch.bool))
        assert stroke_lost[1] and not vision_lost[1]
        dropped["vision"] += int(vision_lost.sum())
        dropped["stroke"] += int(stroke_lost[[0, 2, 3]].sum())
    assert dropped["vision"] > 0 and dropped["stroke"] > 0
    model.eval()
    _, memory_mask, _ = model.encode_bimodal(images, widths, features, mask, has_strokes,
                                             vision_drop_prob=.5, stroke_drop_prob=.5)
    assert (~memory_mask[[0, 2, 3], :slots]).any(1).all()
    with pytest.raises(ValueError, match="sum to at most 1"):
        model.encode_bimodal(images, widths, features, mask, vision_drop_prob=.7, stroke_drop_prob=.6)


def test_r2l_decoder_shares_embedding_and_is_optional():
    model = tiny().train()
    assert model.r2l_decoder.token_embedding.weight is model.decoder.token_embedding.weight
    assert model.r2l_decoder.output.weight is model.decoder.token_embedding.weight
    images, widths = strips()
    features, mask = strokes()
    forward = tokens()
    reverse = torch.cat((forward[:, :1], forward[:, 1:].flip(1)), 1)
    out = model(forward, None, reverse, images=images, image_widths=widths,
                stroke_features=features, stroke_padding_mask=mask)
    assert out["r2l_logits"].shape == out["logits"].shape and torch.isfinite(out["r2l_logits"]).all()
    out["r2l_logits"].float().logsumexp(-1).mean().backward()
    assert model.decoder.token_embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.decoder.transformer.parameters())
    summary = model.parameter_summary()
    assert summary["r2l_decoder"] == sum(p.numel() for p in model.r2l_decoder.transformer.parameters())
    plain = tiny(r2l_decoder_layers=0)
    assert plain.r2l_decoder is None and plain.parameter_summary()["r2l_decoder"] == 0
    with pytest.raises(ValueError, match="right-to-left"):
        plain(forward, None, reverse, images=images, image_widths=widths)


def test_bf16_autocast_forward_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tiny().to(device).train()
    images, widths = strips()
    features, mask = strokes()
    forward = tokens().to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        out = model(forward, None, forward, images=images.to(device), image_widths=widths.to(device),
                    stroke_features=features.to(device), stroke_padding_mask=mask.to(device),
                    has_strokes=torch.tensor([True, False, True, True], device=device),
                    vision_drop_prob=.2, stroke_drop_prob=.2)
        loss = out["logits"].float().logsumexp(-1).mean() + out["r2l_logits"].float().logsumexp(-1).mean()
    assert torch.isfinite(out["logits"].float()).all()
    loss.backward()
    assert torch.isfinite(loss)
    assert_finite_grads(model)


@pytest.mark.parametrize("mode", ["vision", "stroke", "bimodal"])
def test_greedy_decoding_is_valid_and_matches_full_recompute(mode):
    model = tiny().train()
    images, widths = strips()
    features, mask = strokes()
    inputs = {"images": images, "image_widths": widths} if mode != "stroke" else {}
    if mode != "vision":
        inputs |= {"stroke_features": features, "stroke_padding_mask": mask}
    cached = model.decode_greedy(**inputs, max_tokens=12)
    assert not model.training
    assert cached.dtype == torch.long and cached.shape[0] == 4 and 1 <= cached.shape[1] <= 11
    assert ((cached >= 0) & (cached < VOCAB)).all()
    for row in cached.tolist():
        if 2 in row:
            assert all(value == 0 for value in row[row.index(2) + 1:])
    assert torch.equal(cached, model.decode_greedy(**inputs, max_tokens=12, cached=False))
    with torch.inference_mode():
        memory, memory_mask, _ = model.encode_bimodal(**inputs)
        state = DecoderState(model.decoder, memory, memory_mask)
        prefix = tokens(length=6)
        for step in range(prefix.shape[1]):
            stepped = state.step(prefix[:, step:step + 1])[0][:, -1]
            full = model.decoder(prefix[:, :step + 1], memory, memory_mask)[:, -1]
            torch.testing.assert_close(stepped, full, atol=1e-4, rtol=1e-4)


def test_bimodal_75m_parameter_budget():
    model = BimodalMathModel(get_bimodal_config("bimodal_75m", 256, vision_pretrained=False))
    summary = model.parameter_summary()
    print("bimodal_75m", json.dumps(summary))
    assert 65_000_000 <= summary["total"] <= 85_000_000
    parts = summary["vision"] + summary["stroke"] + summary["fusion"] + summary["decoder"] + summary["r2l_decoder"]
    assert parts == summary["total"]
    assert summary["inference_total"] == summary["total"] - summary["r2l_decoder"]
    assert model.vision_encoder.channels == 384 and model.vision_encoder.stride == 16


def test_config_round_trip():
    config = get_bimodal_config("bimodal_tiny", VOCAB, visible_token_ids=[4, 5], structure_symbols=["x"],
                                structure_relations=["None", "Right"], dropout=0.0)
    payload = json.loads(json.dumps(config.to_dict()))
    assert BimodalModelConfig.from_dict({**payload, "retired_field": 1}) == config
    assert config.preset == "bimodal_tiny" and config.dropout == 0.0 and config.vision_backbone == "convnext_atto"
    default = get_bimodal_config("bimodal_75m", 300)
    assert default == BimodalModelConfig(vocab_size=300)
    with pytest.raises(ValueError, match="Unknown bimodal preset"):
        get_bimodal_config("bimodal_huge", VOCAB)
    with pytest.raises(ValueError, match="Count variants"):
        BimodalMathModel(get_bimodal_config("bimodal_tiny", VOCAB, structure_counter="density"))
