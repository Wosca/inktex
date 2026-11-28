"""Regression tests for InkForge v3 training, inference, and v2 transfer."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.model import InkForgeConfig, InkForgeModel
from inkforge.tokenizer import InkForgeTokenizer
from inkforge.v3 import CHECKPOINT_FORMAT, InkForgeV3Config, InkForgeV3Model, load_inkforge_checkpoint
from inkforge.v3.train_v3 import _save_checkpoint, init_from_v2


@pytest.fixture
def model_inputs():
    torch.manual_seed(42)
    tokenizer = InkForgeTokenizer(["x", "y", "=", "+", r"\bar", "{", "}", r"\hat"])
    cfg = InkForgeV3Config(
        vocab_size=len(tokenizer),
        token_roles=InkForgeV3Config.roles_for_tokens(tokenizer.tokens),
        d_model=64,
        n_heads=4,
        d_ff=192,
        enc_layers=1,
        dec_layers=2,
        style_dim=16,
        style_max_strokes=4,
        style_points=8,
        mdn_components=4,
        max_stroke_points=32,
        dropout=0,
    )
    tokens = [tokenizer.encode(expr) for expr in ("x=y", r"\hat{x}+\bar{y}")]
    tokens = torch.tensor([row + [0] * (len(tokens[1]) - len(row)) for row in tokens])
    pen = torch.tensor([[0, 1, 0, 0, 1, 0, 1, 2], [1, 0, 1, 0, 1, 2, 2, 2]])
    b, t = pen.shape
    coords = torch.tensor([
        [[0, 0], [0.15, 0], [0.10, -0.25], [15, 0.2], [30, 0.2], [29, 0.3], [26, -0.4], [25.8, -0.2]],
        [[0, 0], [0.05, 0.4], [-0.05, -0.3], [0.15, -0.25], [-0.1, 0.35], [0, 0.4], [0., 0.], [0., 0.]],
    ])
    actions = torch.diff(coords, dim=1, append=coords[:, -1:])
    stroke_mask = torch.arange(t)[None] < torch.tensor([[t], [6]])
    actions = actions.masked_fill((pen == 2)[..., None], 0)
    inputs = torch.zeros((b, t, 7))
    inputs[:, 0, 6] = 1
    inputs[:, 1:, :2] = actions[:, :-1]
    inputs[:, :, 2:4] = coords
    inputs[:, 1:, 4] = pen[:, :-1].eq(0)
    inputs[:, 1:, 5] = pen[:, :-1].eq(1)
    inputs = inputs.masked_fill(~stroke_mask[..., None], 0)
    return tokenizer, cfg, {
        "token_ids": tokens,
        "stroke_inputs": inputs,
        "target_dx": actions[..., 0],
        "target_dy": actions[..., 1],
        "target_pen": pen,
        "stroke_mask": stroke_mask,
        "style_points": torch.randn(b, cfg.style_max_strokes, cfg.style_points, cfg.style_point_dim),
        "style_aspect": torch.randn(b, cfg.style_max_strokes, 4),
        "style_mask": torch.tensor([[False, False, True, True], [True, True, True, True]]),
    }


def test_v3_bfloat16_backward_checkpoint_and_selective_guidance(model_inputs):
    _, cfg, data = model_inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InkForgeV3Model(cfg, grad_checkpoint=True).to(device)
    data = {key: value.to(device) for key, value in data.items()}
    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        result = model.compute_loss(**data, input_noise=0.03, style_drop_prob=0)
    assert all(value.isfinite().all() for value in result.values())
    assert result["guided_loss"] > 0 and result["jump_aux_loss"] > 0
    result["loss"].backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    assert model.canvas_in.weight.grad.abs().sum() > 0
    assert model.token_anchors.weight.grad.abs().sum() > 0

    with torch.no_grad():
        spatial = {key: value[1:] for key, value in data.items()}
        spatial_result = model.compute_loss(**spatial)
        assert spatial_result["guided_loss"] == 0
        assert spatial_result["cursor_end_loss"] == 0
        dropped = model.compute_loss(**data, text_drop_prob=1)
        assert dropped["guided_loss"] == 0
        assert dropped["cursor_end_loss"] == 0
        z, mu, v = model.encode_style(spatial["style_points"], spatial["style_aspect"], spatial["style_mask"])
        assert all(torch.count_nonzero(x) == 0 for x in (z, mu, v))


def test_v3_parallel_cache_and_dpo_likelihood_parity(model_inputs):
    _, cfg, data = model_inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InkForgeV3Model(cfg).eval().to(device)
    data = {key: value.to(device) for key, value in data.items() if not key.startswith("style_")}
    tokens, inputs = data["token_ids"], data["stroke_inputs"]
    style = torch.randn(tokens.shape[0], cfg.style_dim, device=device)
    with torch.no_grad():
        model.canvas_in.weight.normal_(std=0.1)
        model.cursor_speed.weight.normal_(std=0.02)
        model.dec_blocks[0].style_film[-1].weight.normal_(std=0.02)
        memory, mask = model.encode_memory(tokens)
        parallel, guide, cursor = model.decode_features(
            inputs, memory, mask, style, token_ids=tokens, return_guided_attns=True,
        )
        cache = model._make_cache(tokens, style, inputs.shape[1])
        cached = torch.cat([
            model.mdn_head(model._decode_step(inputs[:, i:i + 1], cache, i))
            for i in range(inputs.shape[1])
        ], dim=1)
        assert (parallel - cached).abs().max().item() < 1e-3
        assert cursor.diff(dim=1).min() >= 0
        assert len(guide) == 1 and guide[0].shape[1] == 1
        torch.testing.assert_close(guide[0].sum(dim=-1), torch.ones_like(guide[0][..., 0]))

        pi, mx, my, sx, sy, rho, pen_logits = model.split_mdn_params(cached)
        lower = torch.stack((
            torch.stack((sx, torch.zeros_like(sx)), dim=-1),
            torch.stack((sy * rho, sy * (1 - rho.square()).sqrt()), dim=-1),
        ), dim=-2)
        normal = torch.distributions.MultivariateNormal(torch.stack((mx, my), dim=-1), scale_tril=lower)
        mixture = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=pi), normal)
        pen = data["target_pen"]
        valid = data["stroke_mask"]
        forbid = valid & inputs[..., 5].le(0.5) & pen.ne(2)
        pen_logits[..., 2].masked_fill_(forbid, -1e4)
        log_pen = torch.distributions.Categorical(logits=pen_logits).log_prob(pen)
        log_xy = mixture.log_prob(torch.stack((data["target_dx"], data["target_dy"]), dim=-1))
        expected = (log_xy.masked_fill(~(valid & pen.ne(2)), 0) + 1.4 * log_pen.masked_fill(~valid, 0)).sum(1)
        actual = model.sequence_log_prob(**data, style_z=style, pen_weight=1.4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-3)
        normalized = model.sequence_log_prob(**data, style_z=style, pen_weight=1.4, length_normalize=True)
        torch.testing.assert_close(normalized, actual / valid.sum(1))

    (-model.sequence_log_prob(**data, style_z=style, length_normalize=True).mean()).backward()
    assert model.mdn_head.weight.grad.isfinite().all()


def test_v3_generation_eos_cfg_traces_and_mixture_truncation(model_inputs):
    _, cfg, data = model_inputs
    model = InkForgeV3Model(cfg).eval()
    tokens = data["token_ids"]
    with torch.no_grad():
        model.mdn_head.weight.zero_()
        model.mdn_head.bias[-3:] = torch.tensor([-100., 3., 30.])
    strokes, meta = model.generate_batch(
        tokens, temperature=torch.tensor([0.3, 0.45]), pen_temperature=0.8, cfg_scale=1.4,
        pen_cfg_scale=1.15, cfg_ramp_steps=3, jump_temp_scale=0.55, top_p=0.90,
        max_steps=5, min_steps=2, return_meta=True, return_actions=True,
    )
    assert len(strokes) == len(meta) == 2 and all(m["finished"] for m in meta)
    for generated, m in zip(strokes, meta):
        raw = m["raw_trace"]
        np.testing.assert_array_equal(raw["target_pen"], [1, 1, 2])
        np.testing.assert_array_equal(raw["stroke_inputs"][:, 5], [0, 1, 1])
        np.testing.assert_array_equal(raw["stroke_inputs"][1:, 0], raw["target_dx"][:-1])
        assert raw["target_dx"][-1] == raw["target_dy"][-1] == 0
        assert np.isfinite(np.concatenate(generated)).all()

    _, capped = model.generate_batch(
        tokens, max_steps=2, min_steps=5, return_meta=True, return_actions=True,
    )
    assert all(m["steps"] == 2 and m["hit_cap"] and not m["finished"] for m in capped)
    assert all(2 not in m["raw_trace"]["target_pen"] for m in capped)

    logits = torch.tensor([[5.0, 1.0, -2.0], [5.0, 4.0, 3.0]])
    probs = model._mixture_probs(logits, torch.ones(2), min_p=0.03, top_p=0.7)
    assert probs[0, 0] == 1 and probs[0, 1:].sum() == 0
    assert probs[1, 1] > 0 and probs[1, 2] == 0
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2))


@pytest.mark.parametrize("version", [2, 3])
def test_jump_temperature_only_changes_post_lift_coordinate_sampling(model_inputs, version):
    _, cfg, data = model_inputs
    cfg = replace(cfg, mdn_components=1)
    model = InkForgeV3Model(cfg).eval() if version == 3 else InkForgeModel(InkForgeConfig.from_dict(cfg.to_dict())).eval()
    with torch.no_grad():
        model.mdn_head.weight.zero_()
        model.mdn_head.bias.zero_()
        model.mdn_head.bias[-3:] = torch.tensor([-100.0, 100.0, -100.0])
    tokens = data["token_ids"][:1]

    def draw(scale):
        torch.manual_seed(181)
        _, meta = model.generate_batch(
            tokens, temperature=0.4, jump_temp_scale=scale,
            max_steps=2, return_meta=True, return_actions=True,
        )
        return meta[0]["raw_trace"]

    base, cool = draw(1.0), draw(0.25)
    for axis in ("target_dx", "target_dy"):
        np.testing.assert_allclose(cool[axis], base[axis] * [1, 0.5], atol=1e-7)


def test_v3_checkpoint_round_trip_and_v2_vocabulary_transfer(model_inputs, tmp_path):
    tokenizer, cfg, _ = model_inputs
    v2 = InkForgeModel(InkForgeConfig.from_dict(cfg.to_dict()))
    source_tokenizer = InkForgeTokenizer(list(reversed(tokenizer.tokens[5:])))
    path_v2 = tmp_path / "v2.pt"
    torch.save({
        "config": v2.cfg.to_dict(), "tokenizer": source_tokenizer.to_dict(), "ema_state_dict": v2.state_dict(),
    }, path_v2)
    loaded_v2, loaded_tokenizer, _ = load_inkforge_checkpoint(path_v2, "cpu")
    assert type(loaded_v2) is InkForgeModel and loaded_tokenizer.tokens == source_tokenizer.tokens

    model = InkForgeV3Model(replace(cfg, dec_layers=3))
    copied = init_from_v2(model, tokenizer, path_v2)
    assert 0 < copied < model.parameter_count()
    src_id = source_tokenizer.token_to_id["x"]
    dst_id = tokenizer.token_to_id["x"]
    torch.testing.assert_close(model.text_encoder.embed.weight[dst_id], v2.text_encoder.embed.weight[src_id])
    assert model.dec_blocks[-1].out2.weight.count_nonzero() == 0

    ema = copy.deepcopy(model)
    with torch.no_grad():
        ema.mdn_head.bias.add_(0.25)
    path_v3 = tmp_path / "latest.pt"
    optimizer = torch.optim.AdamW(model.parameters())
    _save_checkpoint(path_v3, model, ema, tokenizer, optimizer, 4, -2.5)
    clone, clone_tokenizer, payload = load_inkforge_checkpoint(path_v3, "cpu", grad_checkpoint=True)
    assert payload["format"] == CHECKPOINT_FORMAT and payload["step"] == 4
    assert clone.cfg == model.cfg and clone.grad_checkpoint and not clone.training
    assert clone_tokenizer.tokens == tokenizer.tokens
    assert set(ema.state_dict()) == set(clone.state_dict())
    for name, weight in ema.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[name], weight)
    unaveraged, _, _ = load_inkforge_checkpoint(path_v3, "cpu", prefer_ema=False)
    torch.testing.assert_close(unaveraged.mdn_head.bias, model.mdn_head.bias)


def test_v3_input_noise_keeps_bos_pen_and_targets_uncorrupted(model_inputs):
    _, cfg, data = model_inputs
    model = InkForgeV3Model(cfg).train()
    orig_inputs = data["stroke_inputs"].clone()
    orig_dx = data["target_dx"].clone()
    orig_dy = data["target_dy"].clone()
    perturbed = model._perturb_inputs(orig_inputs, data["stroke_mask"], noise=0.02)
    # BOS step (col 6 == 1) and pen-state/padding flags (cols 4:7) must never be perturbed
    torch.testing.assert_close(perturbed[:, 0, :4], orig_inputs[:, 0, :4])
    torch.testing.assert_close(perturbed[..., 4:], orig_inputs[..., 4:])
    # Perturbation must stay local & bounded (no cumulative stroke drift!), and targets untouched
    assert (perturbed[..., :2] - orig_inputs[..., :2]).abs().max().item() < 0.20
    # Verify a model with sharp fixed MDN output on clean targets keeps nearly identical NLL under input_noise
    with torch.no_grad():
        res_clean = model.compute_loss(**data, input_noise=0.0, style_drop_prob=0.0)
        res_noisy = model.compute_loss(**data, input_noise=0.01, style_drop_prob=0.0)
        assert abs(res_noisy["mdn_loss"].item() - res_clean["mdn_loss"].item()) < 0.5
    torch.testing.assert_close(data["target_dx"], orig_dx)
    torch.testing.assert_close(data["target_dy"], orig_dy)

