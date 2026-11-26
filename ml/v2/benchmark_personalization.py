"""Controlled Held-Out Personalization Benchmark on Oscar's Native Handwriting.

Zero-leakage evaluation protocol:
  - Groups all 753 Oscar native equations by canonical formula (`standardize_latex_v2(latex)`).
  - Deterministically partitions canonical groups into:
      * Adaptation Pool (253 equations, disjoint canonical formulas)
      * Strict Held-Out Native Test Set (500 equations, 0% canonical overlap)
  - Benchmarks realistic user budgets on the exact same 500 held-out equations (Beam-5 + R2L alpha=0.4)
    in BOTH Bimodal and Stroke-Only modes, PLUS CROHME 2019 Test (1,199 eqs) non-regression check:
      1. Base (0 eqs) — uncalibrated `inktex_v2_release.pt`
      2. Tier 1 · FiLM (Zero-Shot, 20 strokes / ~3-5 eqs, 0 gradient steps)
      3. Tier 2 · LoRA (N = 25 eqs) — ~2 min onboarding
      4. Tier 2 · LoRA (N = 50 eqs) — a few sessions of saved corrections
      5. Tier 2 · LoRA (N = 150 eqs) — ~1-2 weeks of daily use
      6. Tier 1 FiLM + Tier 2 LoRA (N = 150 eqs) — combined personalization
  - Also exports verified `.inktexlora` files for iPad deployment (including a production
    `oscar_personal_all753.inktexlora` clearly kept separate from the held-out benchmark!).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from v2.eval_v2 import (  # noqa: E402
    DEFAULT_CROHME,
    DEFAULT_NATIVE,
    _load_crohme_test,
    _prepare_row_vision,
    decode_beam_cached,
    score_predictions,
)
from v2.export_onnx_v2 import export_lora_adapter  # noqa: E402
from v2.ink_preprocess import (  # noqa: E402
    STYLE_MAX_STROKES,
    STYLE_POINTS,
    InkPreprocessV2Config,
    chunk_ink_v2,
    extract_style_strokes,
)
from v2.model import InkTeXV2Config, InkTeXV2Model  # noqa: E402
from v2.shards_v2 import collate_v2_batch  # noqa: E402
from v2.tokenizer_v2 import LatexTokenizerV2, standardize_latex_v2  # noqa: E402
from v2.train_personal_lora import load_personal_jsonl, train_personal_lora  # noqa: E402


def split_native_disjoint(
    examples: Sequence[dict[str, Any]],
    *,
    test_size: int = 500,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition examples into (adaptation_pool, held_out_test) with 0% canonical overlap."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        groups[ex["canonical_latex"]].append(ex)

    keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))

    held_out: list[dict[str, Any]] = []
    pool: list[dict[str, Any]] = []
    for idx in order:
        grp = groups[keys[int(idx)]]
        if len(held_out) + len(grp) <= test_size:
            held_out.extend(grp)
        else:
            pool.extend(grp)

    # Assert strict zero canonical overlap
    held_canons = {ex["canonical_latex"] for ex in held_out}
    pool_canons = {ex["canonical_latex"] for ex in pool}
    assert held_canons.isdisjoint(pool_canons), "Canonical formula leakage detected between pool and held-out test!"
    return pool, held_out


def precompute_eval_rows(
    examples: Sequence[dict[str, Any]],
    ink_cfg: InkPreprocessV2Config,
) -> list[dict[str, Any]]:
    """Pre-chunk and pre-rasterize eval rows once so repeated runs across tiers are blazing fast."""
    out: list[dict[str, Any]] = []
    dummy_tok = torch.zeros((2,), dtype=torch.long)
    for ex in examples:
        strokes = ex["strokes"]
        times = ex.get("times")
        target = ex.get("target") or ex["latex"]
        chunked = chunk_ink_v2(strokes, times, config=ink_cfg)
        v = max(int(chunked.valid_chunks), 1)
        row_for_vis = {"strokes": strokes, "id": ex["id"], "target": target}
        img_tensor, img_width = _prepare_row_vision(row_for_vis)
        out.append(
            {
                "id": ex["id"],
                "target": target,
                "canonical_latex": standardize_latex_v2(target),
                "sample": {
                    "id": ex["id"],
                    "source": "eval",
                    "writer_id": None,
                    "latex": target,
                    "normalized_latex": target,
                    "synthetic": False,
                    "coarsened": bool(chunked.coarsened),
                    "chunk_points": torch.from_numpy(chunked.chunk_points[:v].copy()),
                    "chunk_point_mask": torch.from_numpy(chunked.chunk_point_mask[:v].copy()),
                    "chunk_meta": torch.from_numpy(chunked.chunk_meta[:v].copy()),
                    "chunk_mask": torch.from_numpy(chunked.chunk_mask[:v].copy()),
                    "image": img_tensor,
                    "image_width": int(img_width),
                    "l2r_tokens": dummy_tok,
                    "r2l_tokens": dummy_tok,
                },
            }
        )
    return out


@torch.inference_mode()
def evaluate_cached_rows(
    model: InkTeXV2Model,
    tokenizer: LatexTokenizerV2,
    cached_rows: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    modality_mode: str = "bimodal",
    writer_style: torch.Tensor | None = None,
    beam_size: int = 5,
    r2l_alpha: float = 0.4,
    batch_size: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    preds: list[dict[str, Any]] = []
    ws_dev = writer_style.detach().to(device) if writer_style is not None else None

    for start in range(0, len(cached_rows), batch_size):
        chunk = cached_rows[start : start + batch_size]
        batch = collate_v2_batch([r["sample"] for r in chunk], pad_id=tokenizer.pad_id, static_buckets=True)

        cp = batch["chunk_points"].to(device, non_blocking=True)
        cpm = batch["chunk_point_mask"].to(device, non_blocking=True)
        cm = batch["chunk_meta"].to(device, non_blocking=True)
        ck = batch["chunk_mask"].to(device, non_blocking=True)
        img = batch["images"].to(device, non_blocking=True) if modality_mode != "stroke" else None
        iw = batch["image_widths"].to(device, non_blocking=True) if modality_mode != "stroke" else None
        ws = ws_dev.expand(cp.shape[0], -1) if ws_dev is not None else None

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            memory, mem_mask, _ = model.encode(
                images=img,
                image_widths=iw,
                chunk_points=cp,
                chunk_point_mask=cpm,
                chunk_meta=cm,
                chunk_mask=ck,
                writer_style=ws,
                modality_mode=modality_mode,
            )
            dec = decode_beam_cached(
                model.decoder,
                memory,
                mem_mask,
                beam_size=beam_size,
                r2l_alpha=r2l_alpha,
            )

        for i, r in enumerate(chunk):
            pred_str = tokenizer.decode(dec["best_tokens"][i])
            l2r_str = tokenizer.decode(dec["l2r_top1_tokens"][i])
            beam_strs = [tokenizer.decode(b) for b in dec["beam_tokens"][i]]
            preds.append(
                {
                    "id": r["id"],
                    "target": r["target"],
                    "prediction": pred_str,
                    "l2r_prediction": l2r_str,
                    "beam_predictions": beam_strs,
                }
            )

    summary = score_predictions(preds)
    return summary, preds


def extract_calibration_style(
    model: InkTeXV2Model,
    pool_examples: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    max_strokes: int = STYLE_MAX_STROKES,
) -> tuple[torch.Tensor, int, int]:
    """Collect up to 20 calibration strokes from the first few pool equations (simulates iPad Calibration Pad)."""
    strokes: list[np.ndarray] = []
    eqs_used = 0
    for ex in pool_examples:
        strokes.extend(ex["strokes"])
        eqs_used += 1
        if len(strokes) >= max_strokes:
            break
    chosen = strokes[:max_strokes]
    sp, sa, sm = extract_style_strokes(chosen, max_strokes=STYLE_MAX_STROKES, points_per_stroke=STYLE_POINTS)
    with torch.inference_mode():
        ws = model.style_encoder(
            torch.from_numpy(sp)[None].to(device),
            torch.from_numpy(sa)[None].to(device),
            torch.from_numpy(sm)[None].to(device),
        )
    return ws, len(chosen), eqs_used


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Oscar handwriting personalization benchmark")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--r2l-alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = InkTeXV2Config.from_dict(payload.get("model_config") or payload["config"])
    cfg.vision_pretrained = False
    base_state = payload.get("ema_state_dict") or payload.get("model_state_dict") or payload["model"]
    model = InkTeXV2Model(cfg)
    model.load_state_dict(base_state, strict=False)
    model.to(device).eval()
    tokenizer = LatexTokenizerV2(payload["tokenizer"]["tokens"])
    ink_cfg = InkPreprocessV2Config()

    native_examples, _ = load_personal_jsonl(args.native)
    pool, held_out = split_native_disjoint(native_examples, test_size=args.test_size, seed=args.seed)
    print(
        f"[split] Total Oscar native={len(native_examples)} | "
        f"Adaptation Pool={len(pool)} ({len({e['canonical_latex'] for e in pool})} unique formulas) | "
        f"Strict Held-Out Test={len(held_out)} ({len({e['canonical_latex'] for e in held_out})} unique formulas, 0% overlap)"
    )

    t0 = time.perf_counter()
    native_test_cached = precompute_eval_rows(held_out, ink_cfg)
    crohme19_raw = _load_crohme_test(DEFAULT_CROHME)["CROHME2019_test"]
    crohme19_cached = precompute_eval_rows(crohme19_raw, ink_cfg)
    print(
        f"[cache] Precomputed {len(native_test_cached)} Oscar held-out + {len(crohme19_cached)} CROHME19 test rows "
        f"in {time.perf_counter()-t0:.1f}s"
    )

    ws_oscar, cal_strokes_n, cal_eqs_n = extract_calibration_style(model, pool, device)
    print(f"[tier1] Oscar calibration style from {cal_strokes_n} strokes ({cal_eqs_n} eqs): ||z||_2 = {float(ws_oscar.norm().item()):.3f}")

    results: list[dict[str, Any]] = []
    pred_store: dict[str, list[dict[str, Any]]] = {}

    def run_eval_tier(
        key: str,
        label: str,
        *,
        n_adapt: int,
        writer_style: torch.Tensor | None,
        use_lora: bool,
        train_stats: dict[str, Any] | None = None,
        adapter_kb: float = 0.0,
    ) -> None:
        model.set_lora_enabled(use_lora)
        t_ev = time.perf_counter()
        nat_bi, nat_bi_preds = evaluate_cached_rows(
            model, tokenizer, native_test_cached, device,
            modality_mode="bimodal", writer_style=writer_style,
            beam_size=args.beam_size, r2l_alpha=args.r2l_alpha,
        )
        nat_st, nat_st_preds = evaluate_cached_rows(
            model, tokenizer, native_test_cached, device,
            modality_mode="stroke", writer_style=writer_style,
            beam_size=args.beam_size, r2l_alpha=args.r2l_alpha,
        )
        c19_bi, _ = evaluate_cached_rows(
            model, tokenizer, crohme19_cached, device,
            modality_mode="bimodal", writer_style=None,  # CROHME19 evaluated without Oscar's style, testing LoRA non-regression
            beam_size=args.beam_size, r2l_alpha=args.r2l_alpha,
        )
        pred_store[key] = nat_bi_preds
        rec = {
            "key": key,
            "label": label,
            "n_adapt": n_adapt,
            "train_seconds": train_stats["train_seconds"] if train_stats else 0.0,
            "adapter_kb": round(adapter_kb, 1),
            "native_bimodal_exact": nat_bi["std_v2_exact"],
            "native_bimodal_cer": nat_bi["std_v2_token_cer"],
            "native_bimodal_oracle5": nat_bi["oracle_k_std_v2_exact"],
            "native_stroke_exact": nat_st["std_v2_exact"],
            "native_stroke_cer": nat_st["std_v2_token_cer"],
            "native_stroke_oracle5": nat_st["oracle_k_std_v2_exact"],
            "crohme19_bimodal_exact": c19_bi["std_v2_exact"],
            "crohme19_bimodal_cer": c19_bi["std_v2_token_cer"],
            "eval_seconds": round(time.perf_counter() - t_ev, 1),
        }
        results.append(rec)
        print(
            f"[{label}] "
            f"Native Bi={100*rec['native_bimodal_exact']:.2f}% (CER {100*rec['native_bimodal_cer']:.2f}%, O@5 {100*rec['native_bimodal_oracle5']:.2f}%) | "
            f"Native Stroke={100*rec['native_stroke_exact']:.2f}% (CER {100*rec['native_stroke_cer']:.2f}%) | "
            f"CROHME19={100*rec['crohme19_bimodal_exact']:.2f}% | "
            f"train={rec['train_seconds']:.1f}s eval={rec['eval_seconds']:.1f}s"
        )

    # 1. Base (0 examples, zero style, no LoRA)
    run_eval_tier("base", "Base (0 eqs)", n_adapt=0, writer_style=None, use_lora=False)

    # 2. Tier 1 Zero-Shot FiLM (20 calibration strokes, 0 gradient steps)
    run_eval_tier(
        "tier1_film",
        f"Tier 1 · FiLM Zero-Shot ({cal_strokes_n} strokes / {cal_eqs_n} eqs)",
        n_adapt=cal_eqs_n,
        writer_style=ws_oscar,
        use_lora=False,
    )

    # 3-5. Tier 2 Ink-Only LoRA at realistic user budgets: N = 25, 50, 150
    for n_budget, steps_budget, tag in (
        (25, 70, "lora_25"),
        (50, 110, "lora_50"),
        (150, 180, "lora_150"),
    ):
        subset = pool[:n_budget]
        t_stats = train_personal_lora(
            model, tokenizer, subset, device=device, steps=steps_budget, lr=3e-4, seed=42
        )
        lora_path = args.out_dir / f"oscar_benchmark_{tag}.inktexlora"
        export_lora_adapter(
            model,
            lora_path,
            adapter_name=f"Oscar Benchmark LoRA ({n_budget} eqs)",
            dtype="fp16",
            training_samples=n_budget,
            notes=f"Held-out benchmark adapter trained on {n_budget} disjoint Oscar equations ({t_stats['train_seconds']}s)",
        )
        kb = lora_path.stat().st_size / 1024.0
        run_eval_tier(
            tag,
            f"Tier 2 · LoRA (N={n_budget} eqs)",
            n_adapt=n_budget,
            writer_style=None,
            use_lora=True,
            train_stats=t_stats,
            adapter_kb=kb,
        )

    # 6. Tier 1 FiLM + Tier 2 LoRA (N=150 eqs) combined (trained WITH Oscar's FiLM active!)
    t_stats_combo = train_personal_lora(
        model, tokenizer, pool[:150], device=device, steps=180, lr=3e-4, writer_style=ws_oscar, seed=42
    )
    combo_path = args.out_dir / "oscar_benchmark_film_lora_150.inktexlora"
    export_lora_adapter(
        model,
        combo_path,
        adapter_name="Oscar Benchmark FiLM+LoRA (150 eqs)",
        dtype="fp16",
        training_samples=150,
        notes=f"Trained with Tier-1 FiLM active on 150 disjoint Oscar equations ({t_stats_combo['train_seconds']}s)",
    )
    run_eval_tier(
        "film_lora_150",
        "Tier 1 FiLM + Tier 2 LoRA (N=150 eqs)",
        n_adapt=150,
        writer_style=ws_oscar,
        use_lora=True,
        train_stats=t_stats_combo,
        adapter_kb=combo_path.stat().st_size / 1024.0,
    )

    # Collect qualitative win/loss examples comparing Base vs Best LoRA (N=150)
    base_preds = pred_store["base"]
    best_preds = pred_store["lora_150"]
    wins: list[dict[str, str]] = []
    regressions: list[dict[str, str]] = []
    for bp, lp in zip(base_preds, best_preds):
        t_std = standardize_latex_v2(bp["target"])
        b_ok = standardize_latex_v2(bp["prediction"]) == t_std
        l_ok = standardize_latex_v2(lp["prediction"]) == t_std
        if not b_ok and l_ok:
            wins.append({"id": bp["id"], "target": bp["target"], "base": bp["prediction"], "personalized": lp["prediction"]})
        elif b_ok and not l_ok:
            regressions.append({"id": bp["id"], "target": bp["target"], "base": bp["prediction"], "personalized": lp["prediction"]})
    print(f"[diff Base -> LoRA-150] Fixed={len(wins)} equations | Regressed={len(regressions)} equations | Net=+{len(wins)-len(regressions)}")

    # 7. Production Personal Adapter trained on ALL 753 Oscar equations for real iPad deployment
    print("\n[production] Training Oscar full production LoRA adapter (all 753 native equations)...")
    prod_stats = train_personal_lora(
        model, tokenizer, native_examples, device=device, steps=260, lr=3e-4, seed=42
    )
    prod_path = args.out_dir / "oscar_personal_full753.inktexlora"
    export_lora_adapter(
        model,
        prod_path,
        adapter_name="Oscar Personal Handwriting (753 eqs)",
        dtype="fp16",
        training_samples=len(native_examples),
        notes=f"Production personal ink-only LoRA trained on all {len(native_examples)} Oscar handwriting samples ({prod_stats['train_seconds']}s)",
    )
    model.set_lora_enabled(True)
    prod_c19, _ = evaluate_cached_rows(
        model, tokenizer, crohme19_cached, device,
        modality_mode="bimodal", writer_style=None,
        beam_size=args.beam_size, r2l_alpha=args.r2l_alpha,
    )
    print(
        f"[production] Exported {prod_path} ({prod_path.stat().st_size/1024:.1f} KB) | "
        f"CROHME19 non-regression={100*prod_c19['std_v2_exact']:.2f}% (CER {100*prod_c19['std_v2_token_cer']:.2f}%)"
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "native_dataset": str(args.native),
        "split": {
            "total": len(native_examples),
            "adaptation_pool": len(pool),
            "held_out_test": len(held_out),
            "canonical_overlap": 0,
            "calibration_strokes": cal_strokes_n,
            "calibration_equations": cal_eqs_n,
        },
        "beam_size": args.beam_size,
        "r2l_alpha": args.r2l_alpha,
        "results": results,
        "diff_base_to_lora150": {
            "fixed_count": len(wins),
            "regressed_count": len(regressions),
            "sample_wins": wins[:15],
            "sample_regressions": regressions[:10],
        },
        "production_adapter": {
            "path": str(prod_path),
            "bytes": prod_path.stat().st_size,
            "training_samples": len(native_examples),
            "train_seconds": prod_stats["train_seconds"],
            "crohme19_bimodal_exact": prod_c19["std_v2_exact"],
            "crohme19_bimodal_cer": prod_c19["std_v2_token_cer"],
        },
    }
    report_path = args.out_dir / "personalization_benchmark_results.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] Full benchmark report -> {report_path}")


if __name__ == "__main__":
    main()
