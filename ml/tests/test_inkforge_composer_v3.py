"""CPU coverage for InkForge v3 prompting, assembly, and local verification."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkforge import generate_overnight as overnight  # noqa: E402
from inkforge.composer import (  # noqa: E402
    StructuredPrompt,
    assemble_structured_strokes,
    draw_human_delimiter,
    draw_human_hline,
    draw_human_sqrt,
    load_stratified_unimer_pools,
    load_unimer_prompt_pool,
    parse_unimer_prompt,
    passes_geometry_gate,
    sample_programmatic_structured_prompt,
    synthesize_realistic_times,
)
from inkforge.tokenizer import InkForgeTokenizer  # noqa: E402
from inkforge.verify_openrouter import _normalize_for_vlm_compare  # noqa: E402
from v2.tokenizer_v2 import LatexTokenizerV2  # noqa: E402


@pytest.fixture
def tokenizer():
    return InkForgeTokenizer.default()


@pytest.mark.parametrize("latex", [
    r"\mathrm{^{7}} + 2",
    r"F_{\mathrm{~C~Z~}} = 1",
    r"\dot{\sigma}_{\mathrm{~e~x~t~}} = y",
    r"_{\mathrm{x}} + 1",
    r"{^2 + y = 3}",
    r"\mathfrak{I} + 1",
    r"x_{\mathrm{GPa} = 2",
    r"x + 1} = 2",
    r"x + y = .",
])
def test_unimer_rejects_malformed_labels(tokenizer, latex):
    assert parse_unimer_prompt(latex, tokenizer) is None


def test_unimer_rewrites_clean_mathrm_and_terminal_punctuation(tokenizer):
    assert parse_unimer_prompt(r"E_{\mathrm{GPa}} = 1.25.", tokenizer).latex == r"E_{\text{GPa}} = 1.25"
    assert parse_unimer_prompt(r"y = \sin\mathrm{x},", tokenizer).latex == r"y = \sin x"
    assert parse_unimer_prompt(r"x = \{a,b\}", tokenizer).latex == r"x = \{a,b\}"
    assert parse_unimer_prompt(r"\begin{array}{r}{x=3.5.}\end{array}", tokenizer).latex == "x=3.5"


def test_stratified_loader_reserves_slots_for_late_real_layouts(tmp_path, tokenizer):
    path = tmp_path / "unimer.jsonl"
    labels = [
        *(f"x + y = {i}" for i in range(30)),
        r"f(x) = \frac{x^2 + 3x - 1 + \frac{a}{b}}{\sqrt{b^2 - 4ac} + 2xy}",
        r"\begin{array}{rl}x&=2y\\y&=1\end{array}",
        r"A = \left[\begin{array}{cc}1&2\\3&4\end{array}\right]",
        r"f(x)=\left\{\begin{array}{ll}x&x>0\\0&\text{otherwise}\end{array}\right.",
    ]
    records = [{"normalized_latex": label, "split": "train"} for label in labels]
    records.append({"latex": "x + y = 999", "split": "test"})
    path.write_text("\n".join([*(json.dumps(row) for row in records), "{bad JSON", "null"]), encoding="utf-8")

    pools = load_stratified_unimer_pools(tokenizer, path, max_prompts=10, seed=8)
    assert pools == load_stratified_unimer_pools(tokenizer, path, max_prompts=10, seed=8)
    assert set(pools) == {"single", "long_tree", "multiline", "matrix", "cases"}
    assert sum(map(len, pools.values())) == 10
    assert all(pools.values())
    assert pools["multiline"][0].cells == ["x =2y", "y =1"]
    assert pools["matrix"][0].cells == ["A =", "1", "2", "3", "4"]
    assert all("999" not in prompt.latex for pool in pools.values() for prompt in pool)
    assert all(p.kind == "single" for p in load_unimer_prompt_pool(tokenizer, path, max_prompts=10))


def test_seeded_compositional_prompts_cover_stem_and_remain_known(tokenizer):
    rng = random.Random(42)
    prompts = [sample_programmatic_structured_prompt(rng) for _ in range(256)]
    families = {p.meta["subkind"] for p in prompts}
    assert len(families) >= 32
    assert len({p.latex for p in prompts}) >= 250
    assert {"cases_2", "cases_3", "cases_4", "augmented", "block_matrix", "matrix_product",
            "driven_ode", "heat_equation", "joint_probability", "continued_frac"} <= families
    for prompt in prompts:
        assert not tokenizer.has_unk(prompt.latex)
        assert all(not tokenizer.has_unk(c) and not c.startswith(("^", "_")) for c in prompt.cells)
        if prompt.meta["subkind"] == "integral_eval":
            assert r"\mid" not in prompt.latex and "|_" in prompt.latex


def _upright(height: float, x: float = 0.0) -> np.ndarray:
    return np.array([[x, height], [x + 0.16 * height, 0.0]])


def _equals(width: float, x: float = 0.0) -> list[np.ndarray]:
    return [np.array([[x, y], [x + width, y]]) for y in (0.15, 0.42)]


def test_aligned_rows_restore_x_height_across_short_and_long_cells():
    cells = ["u =", "8", "2u + 6 =", "22"]
    tree = {"op": "aligned_eq", "rows": [
        [{"op": "cell", "i": 0}, {"op": "cell", "i": 1}],
        [{"op": "cell", "i": 2}, {"op": "cell", "i": 3}],
    ]}
    prompt = StructuredPrompt("tree", r"u=8\\2u+6=22", cells, {"tree": tree})
    generated = [
        [_upright(2.0), *_equals(1.6, 1.0)],
        [_upright(0.95)],
        [_upright(1.0, x) for x in (0.0, 1.0, 2.0)] + _equals(0.8, 3.0),
        [_upright(1.0), _upright(1.0, 0.7)],
    ]
    ink = assemble_structured_strokes(prompt, generated, np.random.default_rng(15))
    assert np.ptp(ink[0][:, 1]) == pytest.approx(np.ptp(ink[4][:, 1]))
    assert np.ptp(ink[1][:, 0]) == pytest.approx(np.ptp(ink[7][:, 0]))
    assert np.ptp(generated[0][0][:, 1]) == 2.0


def test_hcat_scales_isolated_equals_next_to_a_composed_radical():
    tree = {"op": "hcat", "items": [
        {"op": "cell", "i": 0}, {"op": "cell", "i": 1},
        {"op": "sqrt", "body": {"op": "cell", "i": 2}},
    ]}
    prompt = StructuredPrompt("tree", r"L=\sqrt{b}", ["L", "=", "b"], {"tree": tree})
    ink = assemble_structured_strokes(
        prompt, [[_upright(1.0)], _equals(1.70), [_upright(1.0)]], np.random.default_rng(1),
    )
    assert np.ptp(ink[1][:, 0]) == pytest.approx(0.85)
    assert np.ptp(ink[2][:, 0]) == pytest.approx(0.85)


def test_fraction_bar_and_closing_delimiter_change_stroke_order():
    frac = {"op": "frac", "num": {"op": "cell", "i": 0}, "den": {"op": "cell", "i": 1}}
    root = {"op": "delim", "left": "[", "right": "]", "body": frac}
    prompt = StructuredPrompt("tree", r"\left[\frac{a}{b}\right]", ["a", "b"], {"tree": root})
    bar_positions: Counter[int] = Counter()
    early_closings = 0
    for seed in range(80):
        ink = assemble_structured_strokes(prompt, [[_upright(1.0)], [_upright(1.0)]], np.random.default_rng(seed))
        x_right = max(np.mean(st[:, 0]) for st in ink)
        early_closings += np.mean(ink[1][:, 0]) == x_right
        content = [st for st in ink if np.ptp(st[:, 1]) < 1.5]
        bar_pos = next(i for i, st in enumerate(content) if np.ptp(st[:, 0]) > 0.7)
        bar_positions[bar_pos] += 1
    assert 12 <= bar_positions[2] <= 40
    assert bar_positions[1] + bar_positions[2] == 80
    assert 5 <= early_closings <= 35


@pytest.mark.parametrize("draw,args", [
    (draw_human_hline, (0.0, 4.0, 1.0)),
    (draw_human_sqrt, (0.0, 0.0, 4.0, 2.0)),
    (draw_human_delimiter, ("|", 0.0, 0.0, 4.0)),
])
def test_procedural_point_spacing_and_timestamps_vary(draw, args):
    rng = np.random.default_rng(90)
    stroke, = draw(*args, rng, slant=0.05)
    spacing = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    assert np.std(spacing) / np.mean(spacing) > 0.12
    first, empty, second = synthesize_realistic_times([stroke, np.empty((0, 2)), stroke], rng)
    assert len(empty) == 0 and second[0] > first[-1]
    assert (np.diff(first) > 0).all() and not np.allclose(np.diff(first), np.diff(second))


def test_short_prefix_geometry_accepts_uprights_and_rejects_runaways():
    for prefix in ("I =", "A =", "x =", r"E_{\text{SEI}} =", r"E_{\mathrm{SEI}}"):
        assert passes_geometry_gate(prefix, [_upright(5.0), *_equals(1.0, 1.4)], is_atomic_cell=True)
    assert passes_geometry_gate("A =", [
        np.array([[0.0, 5.0], [1.8, 0.0], [4.0, 5.0]]),
        *_equals(1.0, 4.5),
    ], is_atomic_cell=True)
    assert not passes_geometry_gate("I =", [
        _upright(1.0), *_equals(1.0, 1.4), np.array([[3.0, 0.0], [8.0, 5.0]]),
    ], is_atomic_cell=True)
    assert not passes_geometry_gate("I =", [_upright(11.0), *_equals(1.0)], is_atomic_cell=True)


class _Generator:
    cfg = SimpleNamespace(max_text_len=160)

    def __init__(self, finish_on=None):
        self.calls = []
        self.attempts: Counter[int] = Counter()
        self.finish_on = finish_on or {}

    def _generate(self, token_ids, style_z, max_steps, pen_temperature, jump_temp_scale):
        owners = style_z[:, 0].tolist()
        self.calls.append((token_ids.shape[1], max_steps, pen_temperature, jump_temp_scale))
        info = []
        for owner in owners:
            self.attempts[owner] += 1
            info.append({"finished": self.attempts[owner] >= self.finish_on.get(owner, 1)})
        return [[_upright(1.0)] for _ in owners], info


class _ModernGenerator(_Generator):
    def generate_batch(self, token_ids, *, style_z, temperature, pen_temperature, jump_temp_scale,
                       cfg_scale, max_steps, return_meta):
        return self._generate(token_ids, style_z, max_steps, pen_temperature, jump_temp_scale)


class _LegacyGenerator(_Generator):
    def generate_batch(self, token_ids, *, style_z, temperature, pen_temperature,
                       cfg_scale, max_steps, return_meta):
        return self._generate(token_ids, style_z, max_steps, pen_temperature, None)


@pytest.mark.parametrize("generator,options", [(_ModernGenerator, (0.80, 0.55)), (_LegacyGenerator, (1.0, None))])
def test_length_adaptive_step_cap_and_sampler_signature(tokenizer, generator, options):
    model = generator()
    labels = ["x" * 60, "x" * 8, "x"]
    overnight.generate_verified_cells(
        model, tokenizer, labels, np.arange(3).reshape(-1, 1), np.ones(3),
        torch.device("cpu"), max_chunk_seqs=1, max_retries=0,
    )
    assert model.calls == [(3, 102, *options), (10, 228, *options), (62, 512, *options)]


def test_selective_bonus_retry_only_for_one_failure_in_a_multi_cell_equation(tokenizer):
    model = _ModernGenerator({2: 5, 4: 5, 5: 5, 7: 5})
    _, ok = overnight.generate_verified_cells(
        model, tokenizer, ["x"] * 8, np.arange(8).reshape(-1, 1), np.ones(8),
        torch.device("cpu"), cell_owner=[0, 0, 0, 1, 1, 1, 2, 2],
    )
    assert ok == [True, True, True, True, False, False, True, False]
    assert model.attempts == {0: 1, 1: 1, 2: 5, 3: 1, 4: 4, 5: 4, 6: 1, 7: 4}


def test_greedy_confidence_excludes_eos_and_other_specials():
    tok = LatexTokenizerV2(["x", "y"])

    class Decoder(torch.nn.Module):
        output = torch.nn.Identity()

        def decode_greedy_cached(self, memory, mask, **params):
            top_ids = [(tok.token_to_id["x"], tok.token_to_id["x"]), (tok.eos_id, tok.unk_id),
                       (tok.pad_id, tok.token_to_id["y"]), (tok.pad_id, tok.eos_id)]
            strength = [(1.0, 5.0), (0.1, 0.1), (0.1, 3.0), (0.1, 0.1)]
            result = []
            for ids, scores in zip(top_ids, strength):
                logits = torch.zeros(2, 1, len(tok))
                logits[torch.arange(2), 0, torch.tensor(ids)] = torch.tensor(scores)
                result.append(self.output(logits)[:, -1].argmax(-1))
            return torch.stack(result, dim=1)

    decoder = Decoder()
    ids, confidence = overnight._decode_with_token_confidence(decoder, torch.empty(0), None, tok)
    assert tok.eos_id in ids[0]
    assert confidence == pytest.approx([np.exp(s) / (np.exp(s) + len(tok) - 1) for s in (1.0, 3.0)])
    assert not decoder.output._forward_hooks


def test_confidence_gate_applies_to_exact_matches_and_otherwise_alias(tokenizer, monkeypatch):
    verdicts = [(True, "x", "ok", 0.59), (False, "Otherwise", "wrong_latex", 0.80),
                (False, "Otherrwise", "wrong_latex", 0.99), (False, "Otherwise", "decode_cap", 0.99)]
    monkeypatch.setattr(overnight, "judge_rollouts_with_confidence", lambda *_args, **_kwargs: verdicts)
    labels = ["x", *[r"\text{otherwise}"] * 3]

    class ManyStrokeGenerator(_ModernGenerator):
        def _generate(self, *args, **kwargs):
            _, meta = super()._generate(*args, **kwargs)
            return [[_upright(1.0, 0.5 * i) for i in range(4)] for _ in meta], meta

    def generate(threshold):
        return overnight.generate_verified_cells(
            ManyStrokeGenerator(), tokenizer, labels, np.arange(4).reshape(-1, 1), np.ones(4),
            torch.device("cpu"), judge_model=object(), judge_tok=object(), max_retries=0,
            min_judge_prob=threshold,
        )[1]

    assert generate(0.60) == [False, True, False, False]
    assert generate(0.0) == [True, True, False, False]


def test_vlm_folds_mid_and_exact_otherwise_only():
    normalize = _normalize_for_vlm_compare
    assert normalize(r"\Pr(B \mid A)") == normalize(r"\Pr(B|A)")
    assert normalize(r"a\midpoint b") != normalize(r"a|point b")
    assert normalize(r"\text{otherwise}") == normalize("Otherwise")
    assert normalize(r"\text{otherwise}") != normalize("Otherrwise")
