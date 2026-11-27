"""End-to-end unit tests for InkForge tokenizer, model, MDN loss, KV-cached generation, and 2D composer."""
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.composer import (  # noqa: E402
    assemble_structured_strokes,
    parse_unimer_prompt,
    passes_geometry_gate,
    sample_programmatic_structured_prompt,
    smooth_strokes_catmull_rom,
    synthesize_realistic_times,
)
from inkforge.dataset import (  # noqa: E402
    _generate_composed_weak_structures,
    _normalize_donor_glyph,
    _packed_record_digest,
    _split_records_leak_free,
    normalize_and_resample_for_inkforge,
    unpack_points_to_strokes,
)
from inkforge.model import InkForgeConfig, InkForgeModel, MonotonicGaussianWindow  # noqa: E402
from inkforge.render import render_contact_sheet  # noqa: E402
from inkforge.tokenizer import InkForgeTokenizer, tokenize_inkforge  # noqa: E402


def test_tokenizer_preserves_text_spaces_and_math():
    tok = InkForgeTokenizer.default()
    expr = r"\text{for all} x \geq 0"
    toks = tokenize_inkforge(expr)
    assert "<SPACE>" in toks
    ids = tok.encode(expr)
    assert tok.unk_id not in ids
    decoded = tok.decode(ids)
    assert r"\text{for all}" in decoded
    assert r"\geq" in decoded


def test_normalize_and_unpack_roundtrip():
    s0 = np.array([[10.0, 20.0], [10.0, 30.0], [20.0, 30.0]], dtype=np.float64)
    s1 = np.array([[25.0, 20.0], [25.0, 30.0]], dtype=np.float64)
    norm = normalize_and_resample_for_inkforge([s0, s1], spacing=0.12)
    assert norm is not None
    xy, pen = norm
    assert np.allclose(xy[0], [0.0, 0.0], atol=1e-5)
    strokes = unpack_points_to_strokes(xy, pen)
    assert len(strokes) == 2
    assert len(strokes[0]) >= 2 and len(strokes[1]) >= 2


def test_model_forward_backward_and_kv_generate(tmp_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = InkForgeTokenizer.default()
    cfg = InkForgeConfig(
        vocab_size=len(tok),
        d_model=128,
        n_heads=4,
        d_ff=384,
        enc_layers=2,
        dec_layers=2,
        style_dim=32,
        mdn_components=8,
        max_stroke_points=64,
    )
    model = InkForgeModel(cfg).to(device)
    b, l, t = 4, 16, 24
    token_ids = torch.randint(5, len(tok), (b, l), device=device)
    stroke_inputs = torch.randn((b, t, 7), device=device)
    target_dx = torch.randn((b, t), device=device) * 0.2
    target_dy = torch.randn((b, t), device=device) * 0.2
    target_pen = torch.randint(0, 3, (b, t), device=device)
    stroke_mask = torch.ones((b, t), dtype=torch.bool, device=device)
    style_points = torch.randn((b, cfg.style_max_strokes, cfg.style_points, 8), device=device)
    style_aspect = torch.randn((b, cfg.style_max_strokes, 4), device=device)
    style_mask = torch.zeros((b, cfg.style_max_strokes), dtype=torch.bool, device=device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model.compute_loss(
            token_ids=token_ids,
            stroke_inputs=stroke_inputs,
            target_dx=target_dx,
            target_dy=target_dy,
            target_pen=target_pen,
            stroke_mask=stroke_mask,
            style_points=style_points,
            style_aspect=style_aspect,
            style_mask=style_mask,
            guided_attn_weight=4.0,
        )
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["guided_loss"]) and float(out["guided_loss"].item()) > 0.0
    out["loss"].backward()

    # Verify empty style_mask produces EXACT z=0 even in training mode (zero N(0,I) noise injection!)
    model.train()
    empty_mask = torch.ones((b, cfg.style_max_strokes), dtype=torch.bool, device=device)
    z_empty, mu_empty, _ = model.encode_style(style_points, style_aspect, empty_mask, sample=True)
    assert float(z_empty.abs().max().item()) == 0.0
    assert float(mu_empty.abs().max().item()) == 0.0

    # Verify parallel decode_features matches step-by-step KV-cached forward WITH mixed text padding!
    model.eval()
    with torch.no_grad():
        padded_toks = token_ids.clone()
        padded_toks[1, 10:] = tok.pad_id
        padded_toks[2, 6:] = tok.pad_id
        z0 = torch.zeros((b, cfg.style_dim), device=device)
        mem, mm = model.encode_memory(padded_toks)
        raw_par, _, _ = model.decode_features(stroke_inputs, mem, mm, z0, return_guided_attns=False)
        raw_guided, _, _ = model.decode_features(stroke_inputs, mem, mm, z0, return_guided_attns=True)
        assert float((raw_par - raw_guided).abs().max().item()) < 1e-4

        # Also check BF16 autocast guided vs SDPA parity (autocast disabled inside explicit score matmul!)
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_bf16_sdpa, _, _ = model.decode_features(stroke_inputs, mem, mm, z0, return_guided_attns=False)
                raw_bf16_guid, _, _ = model.decode_features(stroke_inputs, mem, mm, z0, return_guided_attns=True)
            bf16_diff = float((raw_bf16_sdpa - raw_bf16_guid).abs().max().item())
            assert bf16_diff < 5e-2, f"BF16 guided vs SDPA mismatch too large: {bf16_diff}"

        cross_kvs = [blk.project_cross_kv(mem) for blk in model.dec_blocks]
        past_kvs = [None] * len(model.dec_blocks)
        k_state = torch.zeros((b, model.monotonic_window.num_windows), device=device)
        step_outs = []
        for s in range(t):
            x = model.stroke_in(stroke_inputs[:, s : s + 1])
            cos, sin = model.dec_rope(1, offset=s)
            for idx, blk in enumerate(model.dec_blocks):
                x, past_kvs[idx], _ = blk(
                    x, mem, mm, z0, cos, sin, past_kv=past_kvs[idx], cross_kv=cross_kvs[idx]
                )
                if idx == 0:
                    x, ks, _, _ = model.monotonic_window(x, mem, mm, prev_kappa=k_state)
                    k_state = ks[:, 0, :]
            step_outs.append(model.mdn_head(model.dec_norm(x)))
        raw_seq = torch.cat(step_outs, dim=1)
        max_diff = float((raw_par - raw_seq).abs().max().item())
        assert max_diff < 1e-4, f"Parallel vs KV-cache mismatch: {max_diff}"

    model.eval()
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        gen, meta = model.generate_batch(token_ids, temperature=0.45, max_steps=32, return_meta=True)
    assert len(gen) == b and len(meta) == b
    assert all(len(st) >= 1 for st in gen)

    # Verify cap hit reporting with tiny max_steps
    _, meta_cap = model.generate_batch(token_ids, temperature=0.45, max_steps=2, min_steps=5, return_meta=True)
    assert all(m["hit_cap"] and not m["finished"] for m in meta_cap)

    sheet = render_contact_sheet([(r"x^2+y^2=1", gen[0]), (r"\alpha+\beta", gen[1])], tmp_path / "grid.png", cols=2)
    assert sheet.is_file() and sheet.stat().st_size > 500


def test_monotonic_window_symmetry_and_no_underflow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = MonotonicGaussianWindow(d_model=64, num_windows=4, step_scale=0.25).to(device)
    # Extreme 10-token / 450-point trace where naive exp(-beta*diff^2) underflows to 0!
    b, t, l, d = 2, 450, 10, 64
    x = torch.randn((b, t, d), device=device, requires_grad=True)
    mem = torch.randn((b, l, d), device=device)
    pad = torch.zeros((b, l), dtype=torch.bool, device=device)
    pad[1, 6:] = True

    x_out, kappa, phi_norm, active_kappa = win(x, mem, pad)
    assert torch.isfinite(phi_norm).all()
    assert torch.allclose(phi_norm.sum(dim=-1), torch.ones((b, t), device=device), atol=1e-5)
    assert float(phi_norm[1, :, 6:].abs().max().item()) == 0.0

    loss = x_out.pow(2).mean() + active_kappa[:, -1].mean()
    loss.backward()
    b_grad = win.to_params.bias.grad
    assert b_grad is not None
    alpha_grad = b_grad[:4]
    beta_grad = b_grad[4:8]
    dk_grad = b_grad[8:12]
    # All four windows must have non-zero alpha gradients and distinct beta/dk gradients!
    assert float(alpha_grad.abs().max().item()) > 1e-6
    assert float(beta_grad.std().item()) > 1e-6
    assert float(dk_grad.std().item()) > 1e-6


def test_leak_free_split_groups_duplicates_and_writers():
    recs = []
    # 50 identical copies of a symbol + 40 distinct writers with 3 samples each
    xy_dup = np.array([[0.0, 0.0], [0.2, 0.2], [0.4, 0.0]], dtype=np.float16)
    pen_dup = np.array([0, 0, 1], dtype=np.uint8)
    for _ in range(50):
        recs.append(([1, 10, 2], "", 3, xy_dup, pen_dup))
    for w in range(40):
        for j in range(3):
            xy = np.array([[0.0, 0.0], [0.1 * (w + 1), 0.1 * (j + 1)]], dtype=np.float16)
            pen = np.array([0, 1], dtype=np.uint8)
            recs.append(([1, 20 + j, 2], f"w:{w}", 1, xy, pen))
    train_r, val_r = _split_records_leak_free(recs, val_fraction=0.20, seed=7)
    train_digs = {_packed_record_digest(r[0], r[3], r[4]) for r in train_r}
    val_digs = {_packed_record_digest(r[0], r[3], r[4]) for r in val_r}
    assert len(train_digs & val_digs) == 0
    train_w = {r[1] for r in train_r if r[1]}
    val_w = {r[1] for r in val_r if r[1]}
    assert len(train_w & val_w) == 0


def test_structured_prompt_parsing_and_2d_assembly():
    tok = InkForgeTokenizer.default()
    mat = parse_unimer_prompt(r"\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}", tok)
    assert mat is not None and mat.kind == "matrix" and len(mat.cells) == 4

    cases = parse_unimer_prompt(r"f(x) = \begin{cases} x^2 & \text{if } x \geq 0 \\ -x & \text{otherwise} \end{cases}", tok)
    assert cases is not None and cases.kind == "cases" and len(cases.cells) == 5

    py_rng = random.Random(7)
    np_rng = np.random.default_rng(7)
    for _ in range(12):
        pr = sample_programmatic_structured_prompt(py_rng)
        dummy_cells = [
            [
                np.array([[k * 0.5, 0.0], [k * 0.5 + 0.25, 0.3], [k * 0.5 + 0.4, 0.8]])
                for k in range(max(2, len(tokenize_inkforge(c)) // 2))
            ]
            for c in pr.cells
        ]
        assembled = assemble_structured_strokes(pr, dummy_cells, np_rng)
        assert len(assembled) >= len(pr.cells)
        smoothed = smooth_strokes_catmull_rom(assembled, subdivisions=2)
        times = synthesize_realistic_times(smoothed, np_rng)
        assert len(times) == len(smoothed)
        assert passes_geometry_gate(pr.latex, smoothed)


def test_v2_sigma_min_text_cfg_grad_checkpoint_and_dpo_logprob():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = InkForgeTokenizer.default()
    cfg = InkForgeConfig(
        vocab_size=len(tok),
        d_model=128,
        n_heads=4,
        d_ff=384,
        enc_layers=2,
        dec_layers=2,
        style_dim=32,
        mdn_components=8,
        max_stroke_points=64,
        sigma_min=0.008,
    )
    model = InkForgeModel(cfg, grad_checkpoint=True).to(device)
    b, l, t = 4, 14, 20
    token_ids = torch.randint(5, len(tok), (b, l), device=device)
    token_ids[:, 0] = tok.bos_id
    token_ids[:, -1] = tok.eos_id
    stroke_inputs = torch.randn((b, t, 7), device=device)
    target_dx = torch.randn((b, t), device=device) * 0.2
    target_dy = torch.randn((b, t), device=device) * 0.2
    target_pen = torch.randint(0, 3, (b, t), device=device)
    stroke_mask = torch.ones((b, t), dtype=torch.bool, device=device)
    style_points = torch.randn((b, cfg.style_max_strokes, cfg.style_points, 8), device=device)
    style_aspect = torch.randn((b, cfg.style_max_strokes, 4), device=device)
    style_mask = torch.zeros((b, cfg.style_max_strokes), dtype=torch.bool, device=device)

    # 1. Verify sigma_min floor enforcement
    raw_neg = torch.full((b, t, cfg.mdn_components * 6 + 3), -30.0, device=device)
    _, _, _, sx, sy, _, _ = model.split_mdn_params(raw_neg)
    assert float(sx.min().item()) >= 0.008 - 1e-6
    assert float(sy.min().item()) >= 0.008 - 1e-6

    # 2. Verify text_drop_prob=1.0 (all rows dropped) stays finite & cleanly zeroes guided/cursor loss
    model.train()
    out_drop = model.compute_loss(
        token_ids=token_ids,
        stroke_inputs=stroke_inputs,
        target_dx=target_dx,
        target_dy=target_dy,
        target_pen=target_pen,
        stroke_mask=stroke_mask,
        style_points=style_points,
        style_aspect=style_aspect,
        style_mask=style_mask,
        text_drop_prob=1.0,
    )
    assert torch.isfinite(out_drop["loss"])
    assert float(out_drop["guided_loss"].item()) == 0.0
    assert float(out_drop["cursor_end_loss"].item()) == 0.0
    out_drop["loss"].backward()

    # 3. Verify sequence_log_prob supports STaR / DPO gradients cleanly
    model.zero_grad(set_to_none=True)
    z0 = torch.zeros((b, cfg.style_dim), device=device)
    logp = model.sequence_log_prob(
        token_ids=token_ids,
        stroke_inputs=stroke_inputs,
        target_dx=target_dx,
        target_dy=target_dy,
        target_pen=target_pen,
        stroke_mask=stroke_mask,
        style_z=z0,
    )
    assert logp.shape == (b,) and torch.isfinite(logp).all()
    (-logp.mean()).backward()

    # 4. Verify CFG generation (cfg_scale=1.6) runs cleanly with KV-cache
    model.eval()
    gen_cfg, meta_cfg = model.generate_batch(
        token_ids, style_z=z0, temperature=0.40, cfg_scale=1.6, max_steps=24, return_meta=True
    )
    assert len(gen_cfg) == b and len(meta_cfg) == b
    assert all(len(st) >= 1 for st in gen_cfg)


def test_composed_weak_structure_generator():
    tok = InkForgeTokenizer.default()
    # Build minimal dummy donor bank covering all ASCII + core tokens used by composer
    dummy_st = [np.array([[0.0, 0.0], [0.25, 0.45], [0.5, 0.9]], dtype=np.float64)]
    norm_glyph = _normalize_donor_glyph(dummy_st)
    bank = {t: [norm_glyph] for t in tok.tokens if t not in ("<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SPACE>")}
    recs = _generate_composed_weak_structures(bank, tok, count=40, seed=7)
    assert len(recs) == 40
    decoded = [tok.decode(r[0]) for r in recs]
    assert any(r"\sqrt" in d for d in decoded)
    assert any(r"\sum" in d or r"\lim" in d or r"\frac" in d for d in decoded)


def test_star_dpo_rollout_collation_and_prompt_pool():
    import copy
    from inkforge.star_dpo import (
        _canonicalize_judge_latex,
        build_hard_prompt_pool,
        collate_rollout_items,
        raw_trace_to_inkforge_item,
        strokes_to_inkforge_item,
    )

    tok = InkForgeTokenizer.default()
    cfg = InkForgeConfig(vocab_size=len(tok), d_model=128, n_heads=4, d_ff=384, enc_layers=2, dec_layers=2, style_dim=16)
    pool = build_hard_prompt_pool(tok, unimer_path=Path("__nonexistent__.jsonl"), seed=7)
    assert len(pool) > 50
    assert any(r"\sqrt" in p for p in pool)
    assert _canonicalize_judge_latex(r"\mathrm{Pr}(A)") == r"\Pr(A)"

    st = [np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.3]], dtype=np.float64)]
    it = strokes_to_inkforge_item(st, tok.encode(r"x^2+1"), np.zeros((16,), dtype=np.float32), cfg)
    assert it is not None
    batch = collate_rollout_items([it, it], pad_id=tok.pad_id)
    assert batch["token_ids"].shape[0] == 2
    assert batch["style_z"].shape == (2, 16)

    # Verify raw_trace preserves exact action length & NEVER fabricates EOS on capped rollouts,
    # and verify model.eval() with grad_checkpoint matches ref_model identically (0.0 DPO margin at init)!
    model = InkForgeModel(cfg, grad_checkpoint=True)
    ref = copy.deepcopy(model)
    ref.grad_checkpoint = False
    model.eval()
    ref.eval()

    tok_ids = torch.tensor([tok.encode(r"x^2+1")], dtype=torch.long)
    z0 = torch.zeros((1, 16), dtype=torch.float32)
    _gen, meta_cap = model.generate_batch(
        tok_ids, style_z=z0, temperature=0.4, max_steps=5, min_steps=10, return_meta=True, return_actions=True
    )
    assert meta_cap[0]["hit_cap"] and not meta_cap[0]["finished"]
    raw_cap = meta_cap[0]["raw_trace"]
    assert len(raw_cap["target_pen"]) == 5
    assert int((raw_cap["target_pen"] == 2).sum()) == 0  # NO fabricated EOS!

    raw_item = raw_trace_to_inkforge_item(raw_cap, tok.encode(r"x^2+1"), np.zeros((16,), dtype=np.float32))
    assert raw_item is not None
    rb = collate_rollout_items([raw_item], pad_id=tok.pad_id)
    lp_pol = model.sequence_log_prob(**rb, pen_weight=1.0, length_normalize=False, mask_illegal_eos=True)
    with torch.no_grad():
        lp_ref = ref.sequence_log_prob(**rb, pen_weight=1.0, length_normalize=False, mask_illegal_eos=True)
    assert float((lp_pol - lp_ref).abs().max().item()) < 1e-5
    (-lp_pol.mean()).backward()
