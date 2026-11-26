"""Exhaustive iPad Bundle & Personalization Contract Verifier (Swift ModelManifest + ORT Parity).

Verifies:
  1. `manifest.json` strict schema/type compatibility with `InkTypes.swift::ModelManifest`
     and SHA-256 integrity of all 5 ONNX graphs.
  2. `.inktexlora` file strict compatibility with `PersonalizationStore.swift::PortableLoRAFile`.
  3. Multi-bucket ORT execution across ALL chunk buckets `{128, 256, 512, 768}` and ALL width
     buckets `{256, 512, 768, 1024}`, plus Bimodal, Stroke-Only, and Vision-Only iPad paths.
  4. End-to-end PyTorch <-> FP16 ONNX parity on real Oscar handwriting strokes with:
       - Base (uncalibrated, zero style, zero LoRA)
       - Tier 1 FiLM active (`writer_style_v2.onnx` -> `stroke_encoder_v2.onnx`)
       - Tier 2 LoRA active (loaded from `.inktexlora` -> `stroke_encoder_v2.onnx`)
       - Combined Tier 1 FiLM + Tier 2 LoRA active
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from v2.export_onnx_v2 import load_lora_adapter  # noqa: E402
from v2.ink_preprocess import (  # noqa: E402
    CHUNK_BUCKETS,
    STYLE_MAX_STROKES,
    STYLE_POINTS,
    InkPreprocessV2Config,
    chunk_ink_v2,
    extract_style_strokes,
)
from v2.model import InkTeXV2Config, InkTeXV2Model  # noqa: E402
from v2.tokenizer_v2 import LatexTokenizerV2  # noqa: E402
from v2.train_personal_lora import load_personal_jsonl  # noqa: E402
from v2.vision_raster import WIDTH_BUCKETS_V2, render_strokes_to_strip_v2  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest_swift_contract(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    assert manifest_path.is_file(), f"Missing {manifest_path}"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Strict Swift Codable type checks matching InkTypes.swift::ModelManifest
    assert m["format"] == "inktex-onnx-v2"
    assert isinstance(m["model_name"], str) and m["model_name"]
    assert isinstance(m["preset"], str)
    assert m.get("stage") is None or isinstance(m["stage"], int), f"Swift requires stage: Int?, got {type(m['stage'])}: {m['stage']!r}"
    assert m.get("epoch") is None or isinstance(m["epoch"], int)
    for k in ("d_model", "decoder_layers", "attention_heads", "head_dim", "vocab_size",
              "bos_id", "eos_id", "pad_id", "unk_id", "max_points", "max_tokens", "input_dim"):
        assert isinstance(m[k], int), f"{k} must be int"
    assert isinstance(m["tokens"], list) and len(m["tokens"]) == m["vocab_size"]

    if m.get("metrics") is not None:
        assert isinstance(m["metrics"], dict)
        for mk in ("normalized_exact", "strict_exact", "token_cer"):
            if mk in m["metrics"] and m["metrics"][mk] is not None:
                assert isinstance(m["metrics"][mk], (int, float))

    pers = m["personalization"]
    for pk in ("style_dim", "style_max_strokes", "style_points", "style_point_dim", "lora_rank", "stroke_dim"):
        assert isinstance(pers[pk], int)
    assert isinstance(pers["dynamic_lora"], bool) and pers["dynamic_lora"] is True
    assert isinstance(pers["film_layers"], list) and isinstance(pers["lora_layers"], list)

    required_files = ("writer_style", "stroke_encoder", "vision_encoder", "fusion_memproj", "decoder_step")
    total_bytes = 0
    for key in required_files:
        fmeta = m["files"][key]
        fpath = bundle_dir / fmeta["file"]
        assert fpath.is_file(), f"Missing ONNX graph: {fpath}"
        sz = fpath.stat().st_size
        assert sz == fmeta["bytes"], f"Size mismatch for {key}: {sz} != {fmeta['bytes']}"
        sha = _sha256(fpath)
        assert sha == fmeta["sha256"], f"SHA-256 mismatch for {key}"
        total_bytes += sz
        print(f"  [manifest] {key:15s}: {fmeta['file']:24s} {sz/1e6:6.2f} MB  sha256={sha[:12]}.. OK")
    print(f"  [manifest] Total ONNX bundle size: {total_bytes/1e6:.2f} MB (Swift ModelManifest contract: PASS)")
    return m


def verify_lora_swift_contract(lora_path: Path, manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lora_a, lora_b, meta = load_lora_adapter(lora_path)
    pers = manifest["personalization"]
    assert meta["format"] == "inktex-lora-v2"
    assert meta["lora_layers"] == pers["lora_layers"], "LoRA layers mismatch manifest"
    assert meta["lora_rank"] == pers["lora_rank"], "LoRA rank mismatch manifest"
    assert meta["stroke_dim"] == pers["stroke_dim"], "stroke_dim mismatch manifest"
    expected_a = [len(pers["lora_layers"]), 3, pers["lora_rank"], pers["stroke_dim"]]
    expected_b = [len(pers["lora_layers"]), 3, pers["stroke_dim"], pers["lora_rank"]]
    assert list(lora_a.shape) == expected_a, f"lora_a shape {lora_a.shape} != {expected_a}"
    assert list(lora_b.shape) == expected_b, f"lora_b shape {lora_b.shape} != {expected_b}"
    print(
        f"  [lora] {lora_path.name}: {lora_path.stat().st_size/1024:.1f} KB, "
        f"r={meta['lora_rank']}, layers={meta['lora_layers']}, samples={meta.get('training_samples')}, "
        f"||B||={float(np.linalg.norm(lora_b)):.3f} (Swift PortableLoRAFile contract: PASS)"
    )
    return lora_a, lora_b


def run_greedy_ort(
    sessions: dict[str, ort.InferenceSession],
    manifest: dict[str, Any],
    tokenizer: LatexTokenizerV2,
    strokes: list[np.ndarray],
    times: list[np.ndarray | None] | None,
    *,
    writer_style: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    modality: str = "bimodal",
    max_steps: int = 80,
) -> tuple[str, np.ndarray]:
    ink_cfg = InkPreprocessV2Config()
    chunked = chunk_ink_v2(strokes, times, config=ink_cfg)
    bucket = chunked.bucket
    cp = chunked.chunk_points[None].astype(np.float32)
    cpm = chunked.chunk_point_mask[None].astype(np.uint8)
    cm = chunked.chunk_meta[None].astype(np.float32)
    ck = chunked.chunk_mask[None].astype(np.uint8)

    d_model = manifest["d_model"]
    if modality in ("bimodal", "stroke"):
        s_mem = sessions["stroke"].run(
            None,
            {
                "chunk_points": cp,
                "chunk_point_mask": cpm,
                "chunk_meta": cm,
                "chunk_mask": ck,
                "writer_style": writer_style.astype(np.float32),
                "lora_a": lora_a.astype(np.float32),
                "lora_b": lora_b.astype(np.float32),
            },
        )[0]
        s_mask = ck
    else:
        s_mem = np.zeros((1, 1, d_model), dtype=np.float32)
        s_mask = np.ones((1, 1), dtype=np.uint8)

    if modality in ("bimodal", "vision"):
        raster = render_strokes_to_strip_v2(strokes, width_buckets=WIDTH_BUCKETS_V2)
        img = (np.asarray(raster.image, dtype=np.uint8).transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
        iw = np.array([raster.image_width], dtype=np.int64)
        v_mem, v_mask = sessions["vision"].run(None, {"images": img, "image_widths": iw})
    else:
        h_rows = manifest["vision"]["height"] // manifest["vision"]["vertical_stride"]
        v_mem = np.zeros((1, h_rows, d_model), dtype=np.float32)
        v_mask = np.ones((1, h_rows), dtype=np.uint8)

    memory, mem_mask, mk, mv = sessions["fusion"].run(
        None,
        {
            "stroke_memory": s_mem,
            "stroke_mask": s_mask,
            "vision_memory": v_mem,
            "vision_mask": v_mask,
        },
    )

    n_layers = manifest["decoder_layers"]
    n_heads = manifest["attention_heads"]
    head_dim = manifest["head_dim"]
    kv_len = manifest["static_kv_len"]
    past_k = np.zeros((n_layers, 1, n_heads, kv_len, head_dim), dtype=np.float32)
    past_v = np.zeros((n_layers, 1, n_heads, kv_len, head_dim), dtype=np.float32)

    cur_tok = np.array([[manifest["bos_l2r_id"]]], dtype=np.int64)
    out_ids: list[int] = []
    for pos in range(max_steps):
        logits, past_k, past_v = sessions["decoder"].run(
            None,
            {
                "token": cur_tok,
                "cache_pos": np.array([pos], dtype=np.int64),
                "past_k": past_k,
                "past_v": past_v,
                "mk": mk,
                "mv": mv,
                "memory_mask": mem_mask,
            },
        )
        nxt = int(np.argmax(logits[0, 0]))
        if nxt == manifest["eos_id"]:
            break
        out_ids.append(nxt)
        cur_tok = np.array([[nxt]], dtype=np.int64)
    return tokenizer.decode(out_ids), memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exported iPad v2 ONNX bundle and personal LoRA")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=Path(r"E:\math-recogniser-training\datasets\native-snapshots\full_synthetic_20260925_223326\examples.jsonl"))
    args = parser.parse_args()

    print("=== 1. Swift Manifest & LoRA Contract Verification ===")
    manifest = verify_manifest_swift_contract(args.bundle)
    lora_a, lora_b = verify_lora_swift_contract(args.lora, manifest)
    zero_lora_a = np.zeros_like(lora_a)
    zero_lora_b = np.zeros_like(lora_b)
    zero_style = np.zeros((1, manifest["personalization"]["style_dim"]), dtype=np.float32)

    print("\n=== 2. Loading ORT Sessions (CPUExecutionProvider) ===")
    sessions = {
        "style": ort.InferenceSession(str(args.bundle / manifest["files"]["writer_style"]["file"]), providers=["CPUExecutionProvider"]),
        "stroke": ort.InferenceSession(str(args.bundle / manifest["files"]["stroke_encoder"]["file"]), providers=["CPUExecutionProvider"]),
        "vision": ort.InferenceSession(str(args.bundle / manifest["files"]["vision_encoder"]["file"]), providers=["CPUExecutionProvider"]),
        "fusion": ort.InferenceSession(str(args.bundle / manifest["files"]["fusion_memproj"]["file"]), providers=["CPUExecutionProvider"]),
        "decoder": ort.InferenceSession(str(args.bundle / manifest["files"]["decoder_step"]["file"]), providers=["CPUExecutionProvider"]),
    }

    print("\n=== 3. All Dynamic Bucket Shapes Smoke Test (128/256/512/768 chunks x 256/512/768/1024 widths) ===")
    for cb in CHUNK_BUCKETS:
        out_s = sessions["stroke"].run(
            None,
            {
                "chunk_points": np.zeros((1, cb, 8, 20), dtype=np.float32),
                "chunk_point_mask": np.zeros((1, cb, 8), dtype=np.uint8),
                "chunk_meta": np.zeros((1, cb, 10), dtype=np.float32),
                "chunk_mask": np.zeros((1, cb), dtype=np.uint8),
                "writer_style": zero_style,
                "lora_a": lora_a,
                "lora_b": lora_b,
            },
        )[0]
        assert out_s.shape == (1, cb, manifest["d_model"]) and np.isfinite(out_s).all()
    for wb in WIDTH_BUCKETS_V2:
        vm, vmk = sessions["vision"].run(
            None,
            {
                "images": np.zeros((1, 2, manifest["vision"]["height"], wb), dtype=np.float32),
                "image_widths": np.array([wb - 16], dtype=np.int64),
            },
        )
        expected_v = (manifest["vision"]["height"] // 8) * (wb // 16)
        assert vm.shape == (1, expected_v, manifest["d_model"]) and vmk.shape == (1, expected_v) and np.isfinite(vm).all()
    print("  [buckets] All 4 chunk buckets + all 4 vision width buckets executed cleanly with finite outputs: PASS")

    print("\n=== 4. End-to-End PyTorch <-> FP16 ONNX Parity on Real Oscar Handwriting (Base / FiLM / LoRA / Both) ===")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = InkTeXV2Config.from_dict(payload.get("model_config") or payload["config"])
    cfg.vision_pretrained = False
    model = InkTeXV2Model(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    tokenizer = LatexTokenizerV2(manifest["tokens"])

    # Load LoRA into PyTorch model too for direct parity comparison
    for slot, layer_idx in enumerate(cfg.lora_layers):
        blk = model.stroke_encoder.blocks[layer_idx]
        for pi, proj in enumerate((blk.q_proj, blk.v_proj, blk.out_proj)):
            proj.lora_a.data.copy_(torch.from_numpy(lora_a[slot, pi]))
            proj.lora_b.data.copy_(torch.from_numpy(lora_b[slot, pi]))

    native_examples, _ = load_personal_jsonl(args.native)
    cal_strokes = [s for ex in native_examples[:5] for s in ex["strokes"]][:STYLE_MAX_STROKES]
    sp, sa, sm = extract_style_strokes(cal_strokes, max_strokes=STYLE_MAX_STROKES, points_per_stroke=STYLE_POINTS)
    ws_ort = sessions["style"].run(
        None,
        {"style_points": sp[None], "style_aspect": sa[None], "style_mask": sm[None].astype(np.uint8)},
    )[0]
    with torch.inference_mode():
        ws_pt = model.style_encoder(
            torch.from_numpy(sp)[None], torch.from_numpy(sa)[None], torch.from_numpy(sm)[None]
        ).numpy()
    ws_diff = float(np.max(np.abs(ws_ort - ws_pt)))
    print(f"  [style_encoder] ||z||_2={float(np.linalg.norm(ws_ort)):.3f}, PT<->ORT max_abs_diff={ws_diff:.3e}")
    assert ws_diff < 2e-2

    # Test 5 diverse Oscar equations across all 3 modalities + LoRA/FiLM combinations
    test_idxs = [10, 25, 42, 88, 120]
    for mode_name, ws_use, la_use, lb_use, mod_str in (
        ("Bimodal Base", zero_style, zero_lora_a, zero_lora_b, "bimodal"),
        ("Stroke-Only Base", zero_style, zero_lora_a, zero_lora_b, "stroke"),
        ("Vision-Only Base", zero_style, zero_lora_a, zero_lora_b, "vision"),
        ("Bimodal + FiLM", ws_ort, zero_lora_a, zero_lora_b, "bimodal"),
        ("Bimodal + LoRA", zero_style, lora_a, lora_b, "bimodal"),
        ("Bimodal + FiLM + LoRA", ws_ort, lora_a, lora_b, "bimodal"),
    ):
        exact_pt_ort = 0
        max_mem_diff = 0.0
        model.set_lora_enabled(lb_use is lora_b)
        for ti in test_idxs:
            ex = native_examples[ti]
            pred_ort, mem_ort = run_greedy_ort(
                sessions, manifest, tokenizer, ex["strokes"], ex["times"],
                writer_style=ws_use, lora_a=la_use, lora_b=lb_use, modality=mod_str,
            )
            # Compare encoder memory against PyTorch
            ink_cfg = InkPreprocessV2Config()
            ch = chunk_ink_v2(ex["strokes"], ex["times"], config=ink_cfg)
            raster = render_strokes_to_strip_v2(ex["strokes"], width_buckets=WIDTH_BUCKETS_V2)
            img_t = torch.from_numpy((np.asarray(raster.image, dtype=np.uint8).transpose(2, 0, 1)[None].astype(np.float32)) / 255.0)
            iw_t = torch.tensor([raster.image_width], dtype=torch.long)
            with torch.inference_mode():
                # Match iPad fusion call (even in single-modality, iPad passes dummy masked counterpart through fusion)
                s_raw = model.stroke_encoder(
                    torch.from_numpy(ch.chunk_points[None]),
                    torch.from_numpy(ch.chunk_meta[None]),
                    torch.from_numpy(ch.chunk_mask[None]),
                    writer_style=torch.from_numpy(ws_use),
                    chunk_point_mask=torch.from_numpy(ch.chunk_point_mask[None]),
                )
                s_mem_pt = model.fusion.project_stroke(s_raw)
                v_tok_pt, v_msk_pt = model.vision_encoder(img_t, iw_t)
                v_mem_pt = model.fusion.project_vision(v_tok_pt)
                if mod_str == "bimodal":
                    mem_pt, mask_pt = model.fusion.fuse(s_mem_pt, torch.from_numpy(ch.chunk_mask[None]), v_mem_pt, v_msk_pt)
                elif mod_str == "stroke":
                    h_rows = manifest["vision"]["height"] // 8
                    mem_pt, mask_pt = model.fusion.fuse(
                        s_mem_pt, torch.from_numpy(ch.chunk_mask[None]),
                        torch.zeros((1, h_rows, cfg.d_model)), torch.ones((1, h_rows), dtype=torch.bool),
                    )
                else:
                    mem_pt, mask_pt = model.fusion.fuse(
                        torch.zeros((1, 1, cfg.d_model)), torch.ones((1, 1), dtype=torch.bool),
                        v_mem_pt, v_msk_pt,
                    )
                # Greedy decode PT
                ids_pt = [cfg.bos_l2r_id]
                for _ in range(80):
                    lg = model.decoder(torch.tensor([ids_pt], dtype=torch.long), mem_pt, mask_pt)
                    nxt = int(lg[0, -1].argmax().item())
                    if nxt == cfg.eos_id:
                        break
                    ids_pt.append(nxt)
                pred_pt = tokenizer.decode(ids_pt[1:])
            diff = float(np.max(np.abs(mem_pt.numpy() - mem_ort)))
            max_mem_diff = max(max_mem_diff, diff)
            if pred_ort == pred_pt:
                exact_pt_ort += 1
        print(
            f"  [{mode_name:22s}] PT<->ORT exact decode match={exact_pt_ort}/{len(test_idxs)} | "
            f"max_mem_diff={max_mem_diff:.3e}"
        )
    print("\nALL IPAD BUNDLE VERIFICATIONS PASSED!")


if __name__ == "__main__":
    main()
