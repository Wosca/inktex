"""Regression coverage for v3 cache assembly and the confidence-gated judge."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge import star_dpo  # noqa: E402
from inkforge.build_data_v3 import (  # noqa: E402
    RARE_GREEKS,
    ZERO_COUNT_TOKENS,
    PackedSplit,
    _ink_digest,
    build_data_v3,
    relabel_function_words,
)
from inkforge.dataset import (  # noqa: E402
    V3_ALIAS_SOURCE_ID,
    V3_COMPOSED_SOURCE_ID,
    InkForgeDataset,
    _write_packed_split,
    expand_single_token_braces,
)
from inkforge.model import InkForgeConfig, InkForgeModel  # noqa: E402
from inkforge.tokenizer import InkForgeTokenizer, tokenize_inkforge  # noqa: E402


def _record(tokenizer: InkForgeTokenizer, label: str, serial: int, writer: str = "", source: int = 0):
    size = 0.20 + serial * 0.007
    xy = np.asarray([[0, 0], [size, size / 2], [size * 2, size],
                     [size, size], [size / 2, size * 2]], dtype=np.float16)
    return tokenizer.encode(label), writer, source, xy, np.array([0, 0, 1, 0, 1], dtype=np.uint8)


@pytest.fixture(scope="module")
def mini_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("inkforge-v3")
    base = root / "base"
    tok = InkForgeTokenizer.default()
    donors = set(tok.tokens[5:]) - set(ZERO_COUNT_TOKENS) - {"{", "}", "^", "_", r"\text"}
    train = []
    # Write single-token donors directly to avoid standalone tokenizer aliases or dropped tokens.
    for i, sym in enumerate(sorted(donors), 1):
        _, writer, sid, xy, pen = _record(tok, sym, i, source=3)
        train.append(([tok.bos_id, tok.token_to_id[sym], tok.eos_id], writer, sid, xy, pen))
    expressions = (r"\overline{x}", r"\hat{x}", r"\tilde{x}", r"\frac{2ln2}{3}", "x^n_0", r"\text{ln}", "maxwell")
    train.extend(_record(tok, latex, serial, "train-writer") for serial, latex in enumerate(expressions, 500))
    train.append(_record(tok, "y=2", 700, source=200))
    # An existing v2 row has identical ink under another label. Its ink cannot seed an augmentation.
    train.append(_record(tok, r"\overline{y}", 900))
    val = [_record(tok, "v=0", 900, "val-writer")]
    _write_packed_split(base / "train", train)
    _write_packed_split(base / "val", val)
    (base / "val" / "original-note.txt").write_bytes(b"original split note\r\n")
    tok.save(base / "tokenizer.json")
    (base / "meta.json").write_text(json.dumps({"max_text_len": 160, "max_points": 512}), encoding="utf-8")
    out = build_data_v3(base, root / "v3", composed_count=160, seed=6)
    return tok, base, out


def test_function_aliases_and_single_script_bracing():
    assert relabel_function_words(tokenize_inkforge(r"\frac{2ln2}{3}+arcsin(x)")) == tokenize_inkforge(r"\frac{2\ln2}{3}+\arcsin(x)")
    unchanged = r"\text{exp}+maxwell+arccosine+x^ln"
    assert relabel_function_words(tokenize_inkforge(unchanged)) == tokenize_inkforge(unchanged)
    tok = InkForgeTokenizer.default()
    original = tok.encode(r"x^n_0+e^\infty")
    braced = expand_single_token_braces(original, tok, 160)
    assert tok.decode(braced) == r"x^{n}_{0}+e^{\infty}"
    assert expand_single_token_braces(braced, tok, 160) == braced
    assert expand_single_token_braces(original, tok, len(original) + 1) == original
    assert star_dpo._encode_rollout_prompt(r"\ln x^n_0+e^{\infty}", tok, 160, v3=True) == tok.encode(r"\ln x^n_0+e^\infty")[:-2] + [
        tok.token_to_id["{"], tok.token_to_id[r"\infty"], tok.token_to_id["}"], tok.eos_id,
    ]


def test_builder_preserves_base_and_covers_all_zero_count_tokens(mini_cache):
    tok, base, out = mini_cache
    old, new, val = (PackedSplit(path) for path in (base / "train", out / "train", out / "val"))
    for original in (base / "val").iterdir():
        assert original.read_bytes() == (out / "val" / original.name).read_bytes()
    assert (out / "tokenizer.json").read_bytes() == (base / "tokenizer.json").read_bytes()
    for name in ("xy", "pen", "sources", "writers", "point_offsets", "token_offsets", "tokens"):
        data = getattr(old, name)
        np.testing.assert_array_equal(data, getattr(new, name)[: len(data)])
    assert new.writer_names == old.writer_names
    assert set(new.sources[len(old):]) == {V3_ALIAS_SOURCE_ID, V3_COMPOSED_SOURCE_ID}
    assert len(new) == len(old) + 165
    assert all(_ink_digest(*new.ink(i)) != _ink_digest(*val.ink(0)) for i in range(len(old), len(new)))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["base_train_samples"] == len(old)
    assert meta["train_samples"] == len(new)
    assert meta["val_samples"] == len(val)
    assert meta["train_source_counts"]["composed_v3"] == 160
    assert meta["val_human_only"] and meta["leak_free_grouped"]
    for token in ZERO_COUNT_TOKENS:
        assert meta["token_frequencies"][token]["before"] == 0
        assert meta["token_frequencies"][token]["after"] == np.count_nonzero(new.tokens == tok.token_to_id[token]) > 0
    for token in RARE_GREEKS:
        counts = meta["token_frequencies"][token]
        assert counts["after"] > counts["before"]


def test_unchanged_alias_ink_and_valid_flowing_compositions(mini_cache):
    tok, base, out = mini_cache
    old, new = PackedSplit(base / "train"), PackedSplit(out / "train")
    expected = {
        r"\bar{x}": r"\overline{x}", r"\widehat{x}": r"\hat{x}", r"\widetilde{x}": r"\tilde{x}",
        r"\frac{2\ln2}{3}": r"\frac{2ln2}{3}", "x^{n}_{0}": "x^n_0",
    }
    originals = {tok.decode(old.label(i)): i for i in range(len(old))}
    for i in np.flatnonzero(new.sources == V3_ALIAS_SOURCE_ID):
        original = originals[expected.pop(tok.decode(new.label(i)))]
        assert new.writers[i] == old.writers[original] > 0
        for created, prior in zip(new.ink(i), old.ink(original)):
            np.testing.assert_array_equal(created, prior)
    assert not expected

    labels = set()
    for i in np.flatnonzero(new.sources == V3_COMPOSED_SOURCE_ID):
        ids = new.label(i)
        xy, pen = new.ink(i)
        assert new.writers[i] == 0
        assert tok.unk_id not in ids
        labels.add(tok.decode(ids))
        assert 2 <= len(xy) == len(pen) <= 512
        assert np.isfinite(xy).all() and np.all(xy[0] == 0)
        assert set(pen) == {0, 1} and pen[0] == 0 and pen[-1] == 1
    assert {r"\xi+\zeta=\eta\kappa+\tau\rho", r"\ln(x+1)", r"\hat{x}\cdot\bar{y}+\tilde{\phi}\equiv0"} <= labels
    assert any(label.startswith(r"\sum_{k=0}^{n}") for label in labels)


def test_dataset_source_weights_and_online_augmentation(mini_cache):
    tok, _, out = mini_cache
    packed = PackedSplit(out / "train")
    idx = next(i for i in range(len(packed)) if tok.decode(packed.label(i)) == "x^n_0")
    default = InkForgeDataset(out, augment=False)
    assert default.sample_weights[packed.sources == V3_ALIAS_SOURCE_ID][0] == 1.0
    assert default.sample_weights[packed.sources == V3_COMPOSED_SOURCE_ID][0] == 0.70
    assert default.sample_weights[packed.sources == 200][0] == 1.0
    assert tok.decode(default[idx]["token_ids"]) == "x^n_0"
    augmented = InkForgeDataset(out, brace_aug_prob=1.0, stroke_order_aug_prob=1.0, stroke_jitter_std=0.01)
    row = augmented[idx]
    assert tok.decode(row["token_ids"]) == "x^{n}_{0}"
    assert set(row["target_pen"]) == {0, 1, 2}
    assert row["target_pen"][-1] == 2
    assert np.isfinite(row["stroke_inputs"]).all()
    np.testing.assert_allclose(row["stroke_inputs"][1:, :2].cumsum(axis=0), row["stroke_inputs"][1:, 2:4], atol=1e-6)
    augmented.augment = False
    np.testing.assert_array_equal(augmented[idx]["token_ids"], default[idx]["token_ids"])
    assert all(packed.sources[peer] != V3_ALIAS_SOURCE_ID for peers in default.writer_peers.values() for peer in peers)


class _GreedyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Identity()
        self.calls = 0

    def decode_greedy_cached(self, memory, _mask, *, bos_id, eos_id, pad_id, max_tokens):
        self.calls += 1
        row = memory.flatten().long()
        # First emitted token is barely confident for row 1. Row 4 has an uncertain EOS.
        emitted = torch.tensor([[3, 2, 0], [3, 2, 0], [4, 2, 0], [3, 3, 2], [3, 2, 0]])[row]
        probability = torch.tensor([[.96, .96, .20], [.52, .96, .20], [.80, .96, .20],
                                    [.96, .96, .96], [.96, .55, .20]])[row]
        picks = []
        for step in range(emitted.shape[1]):
            token = emitted[:, step, None]
            prob = probability[:, step, None]
            distribution = ((1 - prob) / 4).expand(-1, 5).clone().scatter_(1, token, prob)
            logits = self.output(distribution.log()[:, None, :])
            picks.append(logits[:, 0].argmax(dim=-1))
        return torch.stack(picks, dim=1)


class _Judge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _GreedyDecoder()

    def encode(self, *, images, **kwargs):
        return images, None, None


def test_confidence_gate_reuses_cached_decode_and_ignores_post_eos(monkeypatch):
    monkeypatch.setattr(star_dpo, "passes_geometry_gate", lambda *args: True)
    monkeypatch.setattr(star_dpo, "smooth_strokes_catmull_rom", lambda strokes, **kwargs: strokes)
    monkeypatch.setattr(star_dpo, "synthesize_realistic_times", lambda *args: [])
    monkeypatch.setattr(star_dpo, "_prepare_row_vision", lambda row: (torch.tensor(int(row["id"][1:])), 64))

    def featurize(rows, **kwargs):
        batch = dict.fromkeys(("chunk_points", "chunk_point_mask", "chunk_meta", "chunk_mask", "image_widths"), torch.empty(0))
        batch["images"] = torch.stack([row["_cached_image"] for row in rows])
        return batch

    monkeypatch.setattr(star_dpo, "_featurize_eval_batch", featurize)
    tok = SimpleNamespace(bos_l2r_id=1, eos_id=2, pad_id=0,
                          decode=lambda ids: "".join({3: "x", 4: "y"}.get(i, "") for i in ids))
    rows = [{"strokes": [], "finished": True, "latex": label} for label in ("x", "x", "x", "xx", "x")]
    judge = _Judge()
    legacy = star_dpo.judge_rollouts_with_inktex(judge, tok, rows, torch.device("cpu"))
    assert [result[2] for result in legacy] == ["ok", "ok", "wrong_latex", "ok", "ok"]
    gated = star_dpo.judge_rollouts_with_inktex(judge, tok, rows, torch.device("cpu"), min_token_prob=0.60)
    assert gated == [(True, "x", "ok"), (False, "x", "low_conf"), (False, "y", "wrong_latex"),
                     (True, "xx", "ok"), (False, "x", "low_conf")]
    assert judge.decoder.calls == 2
    assert not judge.decoder.output._forward_hooks


@pytest.mark.parametrize("arch", ("v2", "v3", "v3-arch-tag"))
def test_policy_checkpoint_round_trip(tmp_path, arch):
    tok = InkForgeTokenizer.default()
    config = dict(vocab_size=len(tok), d_model=32, n_heads=4, d_ff=64, enc_layers=1, dec_layers=1)
    is_v3 = arch.startswith("v3")
    if is_v3:
        v3 = pytest.importorskip("inkforge.v3")
        cfg = v3.InkForgeV3Config(**config, token_roles=v3.InkForgeV3Config.roles_for_tokens(tok.tokens))
        model = v3.InkForgeV3Model(cfg)
    else:
        cfg = InkForgeConfig(**config)
        model = InkForgeModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "policy.pt"
    star_dpo._save_policy(path, v3=is_v3, step=0, cfg=cfg, tokenizer=tok, model=model, ema_model=model,
                          optimizer=optimizer, best_val=0.0)
    if arch == "v3-arch-tag":
        payload = torch.load(path, weights_only=True)
        assert payload.pop("format") == "inkforge-checkpoint-v3"
        torch.save(payload, path)
    loaded, loaded_tok, loaded_as_v3 = star_dpo._load_policy(path, grad_checkpoint=False)
    assert loaded_as_v3 == is_v3 and type(loaded) is type(model)
    assert loaded_tok.tokens == tok.tokens
    for key in model.state_dict():
        torch.testing.assert_close(loaded.state_dict()[key], model.state_dict()[key])
