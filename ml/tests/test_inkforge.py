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
from inkforge.dataset import normalize_and_resample_for_inkforge, unpack_points_to_strokes  # noqa: E402
from inkforge.model import InkForgeConfig, InkForgeModel  # noqa: E402
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
        )
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    model.eval()
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        gen = model.generate_batch(token_ids, temperature=0.45, max_steps=32)
    assert len(gen) == b
    assert all(len(st) >= 1 for st in gen)

    sheet = render_contact_sheet([(r"x^2+y^2=1", gen[0]), (r"\alpha+\beta", gen[1])], tmp_path / "grid.png", cols=2)
    assert sheet.is_file() and sheet.stat().st_size > 500


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
