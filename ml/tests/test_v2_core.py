"""Comprehensive unit & parity tests for InkTeX v2 Phase 0 (post-audit hardened)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from v2.ablation_stroke import StrokeAblationModel
from v2.export_onnx_v2 import export_and_verify_v2
from v2.ink_preprocess import (
    CHUNK_BUCKETS,
    CHUNK_META_DIM,
    CHUNK_SIZE,
    POINT_FEATURE_DIM,
    InkPreprocessV2Config,
    chunk_ink_v2,
    extract_style_strokes,
    resample_and_featurize_strokes,
    warp_raw_strokes,
)
from v2.model import InkTeXV2Model, get_v2_config
from v2.pack_v2 import LeakageDenyIndexV2, ink_geometry_hash, namespace_writer
from v2.shards_v2 import collate_v2_batch, pack_raw_strokes_npz, unpack_raw_strokes_npz
from v2.tokenizer_v2 import (
    FORBIDDEN_V2_TOKENS,
    LatexTokenizerV2,
    count_visible_tokens,
    is_plausible_stroke_ratio,
    standardize_latex_v2,
)
from v2.vision_raster import STRIP_HEIGHT_V2, WIDTH_BUCKETS_V2, render_strokes_to_strip_v2


def _make_sample_strokes(num_strokes: int = 8) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    strokes = []
    for i in range(num_strokes):
        x0 = i * 40.0
        t = np.linspace(0.0, 1.0, 24)
        xs = x0 + 25.0 * t + rng.normal(0, 0.5, size=t.shape)
        ys = 30.0 * np.sin(np.pi * t) + rng.normal(0, 0.5, size=t.shape)
        strokes.append(np.stack([xs, ys], axis=1))
    # 1-point dot stroke
    strokes.append(np.array([[120.0, -25.0]], dtype=np.float64))
    return strokes


def test_chunk_ink_v2_buckets_masks_and_point_parity():
    strokes = _make_sample_strokes(10)
    cfg = InkPreprocessV2Config()
    out = chunk_ink_v2(strokes, config=cfg)
    assert out.bucket in CHUNK_BUCKETS
    assert out.chunk_points.shape == (out.bucket, CHUNK_SIZE, POINT_FEATURE_DIM)
    assert out.chunk_point_mask.shape == (out.bucket, CHUNK_SIZE)
    assert out.chunk_meta.shape == (out.bucket, CHUNK_META_DIM)
    assert out.valid_chunks > 10
    assert not out.chunk_mask[: out.valid_chunks].any()
    assert out.chunk_mask[out.valid_chunks :].all()
    assert out.chunk_point_mask[out.valid_chunks :].all()
    assert np.isfinite(out.chunk_points).all()
    # Dot stroke preserved as 1 valid point + 7 masked tail points with is_dot == 1.0
    dot_idx = out.valid_chunks - 1
    assert float(out.chunk_meta[dot_idx, 9]) == 1.0
    assert not out.chunk_point_mask[dot_idx, 0]
    assert out.chunk_point_mask[dot_idx, 1:].all()

    # Shared point featurizer matches chunk points identically at the same spacing
    res = resample_and_featurize_strokes(strokes, config=cfg)
    assert np.allclose(out.chunk_points[0, :8], res.per_stroke_features[0][:8], atol=1e-6)


def test_warp_and_raster_v2_alignment():
    strokes = _make_sample_strokes(12)
    rng = np.random.default_rng(7)
    warped = warp_raw_strokes(strokes, rng)
    assert len(warped) == len(strokes)
    strip = render_strokes_to_strip_v2(warped)
    assert strip.image.size == (strip.bucket_width, STRIP_HEIGHT_V2)
    assert strip.bucket_width in WIDTH_BUCKETS_V2
    assert strip.image_width <= strip.bucket_width
    arr = np.asarray(strip.image)
    assert arr.shape == (STRIP_HEIGHT_V2, strip.bucket_width, 2)
    assert arr[:, :, 0].max() > 0
    assert arr[:, :, 1].max() > 0


def test_bias_free_film_zero_support_invariance_after_training():
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).eval()
    # Simulate post-training non-zero FiLM weights!
    with torch.no_grad():
        for blk in model.stroke_encoder.blocks:
            if blk.has_film:
                blk.film.weight.normal_(0.0, 0.5)
                blk.film_scale.fill_(0.8)

    pts, asp, mask = extract_style_strokes([])
    cp = torch.randn(1, 32, cfg.chunk_size, cfg.point_dim)
    cpm = torch.zeros(1, 32, cfg.chunk_size, dtype=torch.bool)
    cm = torch.randn(1, 32, cfg.chunk_meta_dim)
    ck = torch.zeros(1, 32, dtype=torch.bool)
    with torch.no_grad():
        z = model.style_encoder(
            torch.from_numpy(pts).unsqueeze(0),
            torch.from_numpy(asp).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )
        assert torch.all(z == 0.0)
        out_none = model.stroke_encoder(cp, cm, ck, writer_style=None, chunk_point_mask=cpm)
        out_zero = model.stroke_encoder(cp, cm, ck, writer_style=z, chunk_point_mask=cpm)
    assert float((out_none - out_zero).abs().max()) == 0.0


def test_missing_image_in_mixed_batch_is_fully_masked():
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).eval()
    images = torch.randn(2, 2, cfg.vision_height, 256)
    widths = torch.tensor([192, 0], dtype=torch.long)  # row 1 has NO image
    with torch.no_grad():
        _, mask = model.vision_encoder(images, widths)
    assert not mask[0].all()
    assert mask[1].all()


def test_v2_lora_internal_external_and_merge_parity():
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).eval()
    with torch.no_grad():
        for p in model.lora_parameters():
            p.normal_(0.0, 0.05)

    cp = torch.randn(1, 32, cfg.chunk_size, cfg.point_dim)
    cpm = torch.zeros(1, 32, cfg.chunk_size, dtype=torch.bool)
    cm = torch.randn(1, 32, cfg.chunk_meta_dim)
    ck = torch.zeros(1, 32, dtype=torch.bool)
    ck[0, 20:] = True

    model.set_lora_enabled(False)
    with torch.no_grad():
        base_out = model.stroke_encoder(cp, cm, ck, chunk_point_mask=cpm)

    model.set_lora_enabled(True)
    with torch.no_grad():
        internal_out = model.stroke_encoder(cp, cm, ck, chunk_point_mask=cpm)
    assert float((internal_out - base_out).abs().max()) > 1e-3

    ext = {}
    for layer_idx in cfg.lora_layers:
        blk = model.stroke_encoder.blocks[layer_idx]
        ext[layer_idx] = {
            "q": (blk.q_proj.lora_a.detach(), blk.q_proj.lora_b.detach()),
            "v": (blk.v_proj.lora_a.detach(), blk.v_proj.lora_b.detach()),
            "o": (blk.out_proj.lora_a.detach(), blk.out_proj.lora_b.detach()),
        }
    model.set_lora_enabled(False)
    with torch.no_grad():
        external_out = model.stroke_encoder(cp, cm, ck, external_lora=ext, chunk_point_mask=cpm)
    assert float((internal_out - external_out).abs().max()) < 1e-5

    model.merge_lora_()
    with torch.no_grad():
        merged_out = model.stroke_encoder(cp, cm, ck, chunk_point_mask=cpm)
    assert float((internal_out - merged_out).abs().max()) < 1e-5


def test_v2_forward_backward_shared_bidi_decoder():
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).train()
    images = torch.randn(2, 2, cfg.vision_height, 256)
    widths = torch.tensor([200, 180], dtype=torch.long)
    cp = torch.randn(2, 32, cfg.chunk_size, cfg.point_dim)
    cpm = torch.zeros(2, 32, cfg.chunk_size, dtype=torch.bool)
    cm = torch.randn(2, 32, cfg.chunk_meta_dim)
    ck = torch.zeros(2, 32, dtype=torch.bool)
    ck[:, 24:] = True
    l2r = torch.randint(5, 60, (2, 12))
    l2r[:, 0] = cfg.bos_l2r_id
    r2l = torch.randint(5, 60, (2, 12))
    r2l[:, 0] = cfg.bos_r2l_id

    out = model(
        l2r,
        r2l_decoder_input=r2l,
        images=images,
        image_widths=widths,
        chunk_points=cp,
        chunk_point_mask=cpm,
        chunk_meta=cm,
        chunk_mask=ck,
        vision_drop_prob=0.25,
        stroke_drop_prob=0.15,
    )
    loss = out["logits"].square().mean() + out["r2l_logits"].square().mean()
    loss.backward()
    assert model.stroke_encoder.point_in.weight.grad is not None
    assert model.vision_encoder.lat3.weight.grad is not None
    assert model.decoder.token_embedding.weight.grad is not None


def test_v2_100m_parameter_budget():
    cfg = get_v2_config("inktex_v2_100m", vocab_size=384, vision_pretrained=False)
    model = InkTeXV2Model(cfg)
    s = model.parameter_summary()
    total_m = s["total"] / 1e6
    assert 90.0 <= total_m <= 112.0
    enc_frac = (s["stroke"] + s["vision"]) / s["total"]
    assert enc_frac >= 0.78


def test_v2_onnx_export_and_incremental_kv_parity(tmp_path: Path):
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).eval()
    report = export_and_verify_v2(
        model, tmp_path, dynamic_lora=True, static_kv_len=32, chunk_bucket=32, vision_width=256, atol=1e-4,
    )
    assert "decoder_incremental_0_to_4" in report["max_abs_diff"]
    assert "decoder_prefix_plus_pos3" in report["max_abs_diff"]


def test_v2_bundle_manifest_dynamic_buckets_and_lora_roundtrip(tmp_path: Path):
    import json
    import onnxruntime as ort
    from v2.export_onnx_v2 import (
        MANIFEST_FORMAT_V2,
        export_bundle_v2,
        export_lora_adapter,
        load_lora_adapter,
    )

    torch.manual_seed(3)
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).eval()
    with torch.no_grad():
        for p in model.lora_parameters():
            p.normal_(0.0, 0.05)

    tokens = ["<PAD>", "<BOS_L2R>", "<BOS_R2L>", "<EOS>", "<UNK>"] + [f"t{i}" for i in range(5, 64)]
    bundle_dir = tmp_path / "bundle_v2"
    report = export_bundle_v2(
        model,
        tokens,
        bundle_dir,
        model_name="InkTeX-v2-Tiny-Test",
        quantize="fp32",
        dynamic_lora=True,
        dynamic_buckets=True,
        static_kv_len=32,
        chunk_buckets=(32, 64),
        width_buckets=(256, 512),
        atol=1e-4,
    )

    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert manifest["format"] == MANIFEST_FORMAT_V2
    assert manifest["shared_r2l_decoder"] is True
    assert manifest["bos_l2r_id"] == cfg.bos_l2r_id
    assert manifest["bos_r2l_id"] == cfg.bos_r2l_id
    assert manifest["vision"]["width_buckets"] == [256, 512]
    assert manifest["preprocessing"]["chunk_buckets"] == [32, 64]
    assert manifest["personalization"]["dynamic_lora"] is True
    assert set(manifest["files"].keys()) == {
        "writer_style", "stroke_encoder", "vision_encoder", "fusion_memproj", "decoder_step"
    }

    # Export & reload portable .inktexlora (FP16) and verify exact ONNX hot-swap against PyTorch!
    lora_path = tmp_path / "oscar.inktexlora"
    export_lora_adapter(model, lora_path, adapter_name="Oscar Test", dtype="fp16", training_samples=42)
    la, lb, meta = load_lora_adapter(lora_path)
    assert meta["format"] == "inktex-lora-v2"
    assert la.shape == (len(cfg.lora_layers), 3, cfg.lora_rank, cfg.stroke_dim)
    assert lb.shape == (len(cfg.lora_layers), 3, cfg.stroke_dim, cfg.lora_rank)

    # Test BOTH bucket sizes (32 and 64 chunks, 256 and 512 vision width) on the single dynamic bundle!
    stroke_sess = ort.InferenceSession(report["files"]["stroke_encoder"], providers=["CPUExecutionProvider"])
    vis_sess = ort.InferenceSession(report["files"]["vision_encoder"], providers=["CPUExecutionProvider"])
    for cb in (32, 64):
        cp = torch.randn(1, cb, cfg.chunk_size, cfg.point_dim)
        cpm = torch.zeros(1, cb, cfg.chunk_size, dtype=torch.uint8)
        cm = torch.randn(1, cb, cfg.chunk_meta_dim)
        ck = torch.zeros(1, cb, dtype=torch.uint8)
        ws = torch.zeros(1, cfg.style_dim)
        model.set_lora_enabled(True)
        with torch.no_grad():
            pt_out = model.fusion.project_stroke(
                model.stroke_encoder(cp, cm, ck.ne(0), writer_style=ws, chunk_point_mask=cpm.ne(0))
            ).numpy()
        model.set_lora_enabled(False)
        ort_out = stroke_sess.run(None, {
            "chunk_points": cp.numpy(), "chunk_point_mask": cpm.numpy(),
            "chunk_meta": cm.numpy(), "chunk_mask": ck.numpy(),
            "writer_style": ws.numpy(), "lora_a": la, "lora_b": lb,
        })[0]
        assert float(np.max(np.abs(pt_out - ort_out))) < 5e-3

    for vw in (256, 512):
        img = torch.randn(1, cfg.vision_in_chans, cfg.vision_height, vw)
        iw = torch.tensor([vw - 16], dtype=torch.long)
        v_ort, vm_ort = vis_sess.run(None, {"images": img.numpy(), "image_widths": iw.numpy()})
        assert v_ort.shape == (1, (cfg.vision_height // 8) * (vw // 16), cfg.d_model)
        assert vm_ort.shape == (1, (cfg.vision_height // 8) * (vw // 16))


def test_v2_tokenizer_and_multi_owner_deny_index():
    s = standardize_latex_v2(r"\left( \dfrac{x}{2} \right) \to y")
    assert s == r"(\frac{x}{2})\rightarrow y"
    assert count_visible_tokens(r"\begin{array}{c}x\end{array}") == 1
    assert standardize_latex_v2(r"\textbf{\mathrm{x}}") is None or r"\mathrm" not in standardize_latex_v2(r"\textbf{\mathrm{x}}")

    tok = LatexTokenizerV2.build([s], min_frequency=1)
    assert not (set(tok.tokens) & FORBIDDEN_V2_TOKENS)
    l2r = tok.encode_l2r(s)
    r2l = tok.encode_r2l(s)
    assert tok.decode(l2r) == s
    assert tok.decode(r2l, is_r2l=True) == s
    assert is_plausible_stroke_ratio(r"\frac{x}{2}", 4)
    assert not is_plausible_stroke_ratio("a+b+c+d+e+f+g+h+i+j", 1)

    # Multi-owner sparse dHash + namespaced writers check
    deny = LeakageDenyIndexV2(near_distance=2)
    sparse_dh = "0000000000000003"  # only 2 bits set -> sparse!
    strokes = _make_sample_strokes(4)
    ih = ink_geometry_hash(strokes)
    deny.add(test_id="t1", source="crohme_test", normalized_latex="a", dhash=sparse_dh, ink_hash=ih, writer_id=namespace_writer("crohme", "45"))
    deny.add(test_id="t2", source="crohme_test", normalized_latex="b", dhash=sparse_dh, ink_hash=None, writer_id=None)
    # Second owner on sparse hash must still be caught!
    assert deny.check(sample_id="other", normalized_latex="b", dhash=sparse_dh) == ("dhash_exact", "t2")
    # Namespaced writer blocks crohme:45 but NOT hwrt:45!
    assert deny.check(sample_id="other", normalized_latex="z", dhash=None, writer_id=namespace_writer("crohme", "45")) == ("reserved_writer", "crohme:45")
    assert deny.check(sample_id="other", normalized_latex="z", dhash=None, writer_id=namespace_writer("hwrt", "45")) is None


def test_v2_ablation_arms_matched_capacity_and_forward_backward():
    strokes = _make_sample_strokes(6)
    times = [np.linspace(i * 100.0, i * 100.0 + 50.0, len(s)) for i, s in enumerate(strokes)]
    raw_npz = pack_raw_strokes_npz(strokes, times)
    u_strokes, u_times = unpack_raw_strokes_npz(raw_npz)

    tok = LatexTokenizerV2.build(["x+1", r"\frac{a}{b}"], min_frequency=1)
    cfg = get_v2_config("inktex_v2_ablation", vocab_size=len(tok))
    m_chunk = StrokeAblationModel(cfg, "chunk")
    m_point = StrokeAblationModel(cfg, "point")
    # Nearly matched parameter count (only meta_mlp + pool_score differ by ~0.3M)
    c_params = sum(p.numel() for p in m_chunk.parameters())
    p_params = sum(p.numel() for p in m_point.parameters())
    assert abs(c_params - p_params) / max(c_params, 1) < 0.03

    chunked = chunk_ink_v2(u_strokes, u_times)
    v = chunked.valid_chunks
    c_batch = collate_v2_batch([{
        "id": "s1", "source": "crohme_train", "writer_id": None, "latex": "x+1", "normalized_latex": "x+1",
        "synthetic": False, "coarsened": False,
        "chunk_points": torch.from_numpy(chunked.chunk_points[:v]),
        "chunk_point_mask": torch.from_numpy(chunked.chunk_point_mask[:v]),
        "chunk_meta": torch.from_numpy(chunked.chunk_meta[:v]),
        "chunk_mask": torch.from_numpy(chunked.chunk_mask[:v]),
        "l2r_tokens": torch.tensor(tok.encode_l2r("x+1")),
        "r2l_tokens": torch.tensor(tok.encode_r2l("x+1")),
    }], static_buckets=False)
    l2r_l, r2l_l = m_chunk(c_batch, torch.device("cpu"))
    (l2r_l.square().mean() + r2l_l.square().mean()).backward()

    res = resample_and_featurize_strokes(u_strokes, u_times)
    pts = np.concatenate(res.per_stroke_features, axis=0)
    p_batch = collate_v2_batch([{
        "id": "s1", "source": "crohme_train", "writer_id": None, "latex": "x+1", "normalized_latex": "x+1",
        "synthetic": False, "coarsened": False,
        "point_features": torch.from_numpy(pts),
        "l2r_tokens": torch.tensor(tok.encode_l2r("x+1")),
        "r2l_tokens": torch.tensor(tok.encode_r2l("x+1")),
    }], static_buckets=False)
    l2r_p, r2l_p = m_point(p_batch, torch.device("cpu"))
    (l2r_p.square().mean() + r2l_p.square().mean()).backward()


def test_v2_cached_greedy_matches_uncached_exactly():
    torch.manual_seed(0)
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).eval()
    mem = torch.randn(2, 24, cfg.d_model)
    mem_mask = torch.zeros(2, 24, dtype=torch.bool)
    mem_mask[0, 18:] = True
    mem_mask[1, 20:] = True

    with torch.no_grad():
        cached = model.decoder.decode_greedy_cached(
            mem, mem_mask, bos_id=cfg.bos_l2r_id, eos_id=cfg.eos_id, pad_id=cfg.pad_id, max_tokens=16,
        )
        # Uncached reference
        tokens = torch.full((2, 1), cfg.bos_l2r_id, dtype=torch.long)
        finished = torch.zeros(2, dtype=torch.bool)
        for _ in range(15):
            nxt = model.decoder(tokens, mem, mem_mask)[:, -1].argmax(dim=-1).masked_fill(finished, cfg.pad_id)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            finished |= nxt.eq(cfg.eos_id)
            if bool(finished.all()):
                break
        uncached = tokens[:, 1:]
    assert torch.equal(cached, uncached)


def test_v2_beam_and_r2l_rerank_parity():
    from v2.eval_v2 import decode_beam_cached

    torch.manual_seed(1)
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=64)
    model = InkTeXV2Model(cfg).eval()
    mem = torch.randn(2, 20, cfg.d_model)
    mem_mask = torch.zeros(2, 20, dtype=torch.bool)
    mem_mask[0, 15:] = True

    with torch.no_grad():
        greedy = model.decoder.decode_greedy_cached(
            mem, mem_mask, bos_id=cfg.bos_l2r_id, eos_id=cfg.eos_id, pad_id=cfg.pad_id, max_tokens=16
        ).tolist()
        beam1 = decode_beam_cached(model.decoder, mem, mem_mask, beam_size=1, max_tokens=16, r2l_alpha=0.0)
        assert beam1["best_tokens"] == greedy

        beam4 = decode_beam_cached(
            model.decoder, mem, mem_mask, beam_size=4, max_tokens=16, length_penalty=0.6, r2l_alpha=0.4
        )
        assert len(beam4["best_tokens"]) == 2
        assert len(beam4["beam_tokens"][0]) == 4
        assert len(beam4["beam_scores"][0]) == 4
        assert all(np.isfinite(s) for row in beam4["beam_scores"] for s in row)
        assert beam4["beam_scores"][0] == sorted(beam4["beam_scores"][0], reverse=True)


def test_v2_eval_bucket_invariance_and_mixed_split_resilience():
    from v2.eval_v2 import evaluate_dataset
    from v2.train_v2 import MixedSplitV2Dataset

    # 1. Width-bucketed bimodal & vision eval must be 100% invariant to wide neighbor strips
    torch.manual_seed(7)
    tok = LatexTokenizerV2.build(["x+1", "a+b+c+d+e+f"], min_frequency=1)
    cfg = get_v2_config("inktex_v2_tiny", vocab_size=len(tok))
    model = InkTeXV2Model(cfg).eval()
    narrow_strokes = _make_sample_strokes(2)
    wide_strokes = [np.stack([np.linspace(i * 60.0, i * 60.0 + 45.0, 20), np.linspace(0.0, 30.0, 20)], axis=1) for i in range(18)]
    row_narrow = {"id": "n1", "target": "x+1", "strokes": narrow_strokes, "times": None}
    row_wide = {"id": "w1", "target": "a+b+c+d+e+f", "strokes": wide_strokes, "times": None}

    for mode in ("vision", "bimodal"):
        solo = evaluate_dataset(
            model, tok, [row_narrow], torch.device("cpu"),
            mode=mode, decode_method="greedy", beam_size=1, r2l_alpha=0.0,
            length_penalty=0.6, batch_size=4, max_tokens=12,
            amp_dtype=torch.float32, use_amp=False,
        )
        paired = evaluate_dataset(
            model, tok, [row_narrow, row_wide], torch.device("cpu"),
            mode=mode, decode_method="greedy", beam_size=1, r2l_alpha=0.0,
            length_penalty=0.6, batch_size=4, max_tokens=12,
            amp_dtype=torch.float32, use_amp=False,
        )
        assert solo["predictions"][0]["prediction"] == paired["predictions"][0]["prediction"]

    # 2. MixedSplitV2Dataset: survives secondary RuntimeError (empty split) & uses disjoint secondary epochs
    class _DummyPrimary(torch.utils.data.IterableDataset):
        def __init__(self):
            self.epoch = 0
        def set_epoch(self, e):
            self.epoch = e
        def __iter__(self):
            for i in range(5):
                yield ("p", self.epoch, i)

    class _EmptySecondary(torch.utils.data.IterableDataset):
        def __init__(self):
            self.seen_epochs = []
        def set_epoch(self, e):
            self.seen_epochs.append(e)
        def __iter__(self):
            raise RuntimeError("Empty filtered split")
            yield  # pragma: no cover

    class _ShortSecondary(torch.utils.data.IterableDataset):
        def __init__(self):
            self.seen_epochs = []
            self.epoch = 0
        def set_epoch(self, e):
            self.epoch = e
            self.seen_epochs.append(e)
        def __iter__(self):
            yield ("s", self.epoch)

    prim = _DummyPrimary()
    empty_sec = _EmptySecondary()
    m1 = MixedSplitV2Dataset([prim, empty_sec], [0.5, 0.5], seed=42)
    m1.set_epoch(1)
    out1 = list(m1)
    assert len(out1) == 5 and all(x[0] == "p" for x in out1)

    short_sec = _ShortSecondary()
    m2 = MixedSplitV2Dataset([_DummyPrimary(), short_sec], [0.5, 0.5], seed=42)
    m2.set_epoch(1)
    list(m2)
    e1 = list(short_sec.seen_epochs)
    short_sec.seen_epochs.clear()
    m2.set_epoch(2)
    list(m2)
    e2 = list(short_sec.seen_epochs)
    assert set(e1).isdisjoint(set(e2)), f"Secondary epochs replayed: {e1} vs {e2}"



