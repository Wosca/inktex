"""Standalone inference CLI for InkTeX-75M (BimodalMathModel).

Supports both:
  - Online handwriting (.inkml files or directories) -> full bimodal (strokes + rendered strip)
  - Offline math images (.png/.jpg/.jpeg/.webp files or directories) -> vision-only

Recommended decoding (default): Beam-5 with R2L bidirectional reranking
(--beam-width 5 --r2l-weight 0.5 --length-penalty 0.7).

Examples:
    python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --inkml sample.inkml
    python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --inkml ./test_inkml/ --output preds.json
    python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --image formula.png --show-nbest
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ML = Path(__file__).resolve().parent
if str(REPO_ML) not in sys.path:
    sys.path.insert(0, str(REPO_ML))

import numpy as np
import torch

from data.bimodal_pack import PREPROCESSING, RASTER, _has_timing, expression_strokes
from data.bimodal_raster import WIDTH_MULTIPLE, normalize_offline_image_to_strip, render_strokes_to_strip
from data.importers.crohme import CrohmeImporter
from data.importers.mathwriting import MathWritingImporter
from data.preprocessing import preprocess_example
from eval_benchmarks import decode_beam_and_r2l
from tokenizer.latex_tokenizer import LatexTokenizer
from train_bimodal import autocast, model_from_payload

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(payload: dict[str, Any], tokenizer_path: Path | None) -> LatexTokenizer:
    if tokenizer_path is not None:
        return LatexTokenizer.load(tokenizer_path)
    if payload.get("tokenizer") is not None:
        return LatexTokenizer.from_dict(payload["tokenizer"])
    fallback = REPO_ML / "tokenizer" / "bimodal_tokenizer.json"
    if fallback.is_file():
        return LatexTokenizer.load(fallback)
    raise FileNotFoundError("No embedded tokenizer in checkpoint and --tokenizer not provided")


def parse_inkml(path: Path):
    """Parse CROHME or MathWriting InkML into a CanonicalExpression."""
    for importer in (MathWritingImporter(), CrohmeImporter()):
        try:
            expr = importer.parse(path)
            if expr.strokes:
                return expr
        except Exception:
            continue
    raise ValueError(f"Could not parse strokes from {path}")


def prepare_inkml_sample(path: Path) -> dict[str, Any]:
    expr = parse_inkml(path)
    strokes = expression_strokes(expr)
    pil_img, width = render_strokes_to_strip(
        strokes, RASTER["height"], RASTER["min_width"], RASTER["max_width"], RASTER["online_padding"], None
    )
    image = np.ascontiguousarray(np.transpose(np.asarray(pil_img, dtype=np.uint8), (2, 0, 1)))
    processed = preprocess_example(expr, PREPROCESSING)
    features = np.ascontiguousarray(processed.features.astype(np.float32))
    if not _has_timing(expr):
        features[:, 4] = 0.0
    return {
        "id": str(path),
        "mode": "bimodal",
        "target": getattr(expr, "latex", "") or "",
        "image": image,
        "image_width": width,
        "stroke_features": features,
    }


def prepare_image_sample(path: Path) -> dict[str, Any]:
    pil_img, width = normalize_offline_image_to_strip(
        path, RASTER["height"], RASTER["min_width"], RASTER["max_width"], RASTER["offline_padding"]
    )
    image = np.ascontiguousarray(np.transpose(np.asarray(pil_img, dtype=np.uint8), (2, 0, 1)))
    return {
        "id": str(path),
        "mode": "vision",
        "target": "",
        "image": image,
        "image_width": width,
        "stroke_features": None,
    }


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    height = int(batch[0]["image"].shape[1])
    width = int(math.ceil(max(int(s["image"].shape[2]) for s in batch) / WIDTH_MULTIPLE) * WIDTH_MULTIPLE)
    images = torch.zeros((len(batch), 2, height, width), dtype=torch.float32)
    for row, sample in enumerate(batch):
        img = torch.from_numpy(sample["image"])
        images[row, :, :, :img.shape[2]] = img.float().div_(255.0)
    image_widths = torch.tensor([int(s["image_width"]) for s in batch], dtype=torch.long)

    has_strokes = torch.tensor([s["stroke_features"] is not None for s in batch], dtype=torch.bool)
    if bool(has_strokes.any()):
        max_len = max(len(s["stroke_features"]) for s in batch if s["stroke_features"] is not None)
        channels = next(s["stroke_features"].shape[1] for s in batch if s["stroke_features"] is not None)
        stroke_features = torch.zeros((len(batch), max_len, channels), dtype=torch.float32)
        stroke_padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)
        for row, sample in enumerate(batch):
            if sample["stroke_features"] is not None:
                f = torch.from_numpy(sample["stroke_features"])
                stroke_features[row, :len(f)] = f
                stroke_padding_mask[row, :len(f)] = False
    else:
        stroke_features = None
        stroke_padding_mask = None

    return {
        "images": images,
        "image_widths": image_widths,
        "stroke_features": stroke_features,
        "stroke_padding_mask": stroke_padding_mask,
        "has_strokes": has_strokes,
    }


def collect_paths(target: Path, suffixes: set[str]) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    raise FileNotFoundError(f"Path not found: {target}")


@torch.inference_mode()
def run_inference(
    model,
    tokenizer: LatexTokenizer,
    samples: list[dict[str, Any]],
    device: torch.device,
    *,
    beam_width: int = 5,
    r2l_weight: float = 0.5,
    length_penalty: float = 0.7,
    greedy: bool = False,
    batch_size: int = 16,
    max_tokens: int = 256,
) -> list[dict[str, Any]]:
    use_r2l = (not greedy) and beam_width > 1 and r2l_weight > 0 and model.r2l_decoder is not None
    results: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        batch = collate_samples(chunk)
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if greedy or beam_width <= 1:
            amp = "bfloat16" if device.type == "cuda" else "float32"
            with autocast(device, amp):
                ids = model.decode_greedy(
                    images=moved["images"],
                    image_widths=moved["image_widths"],
                    stroke_features=moved["stroke_features"],
                    stroke_padding_mask=moved["stroke_padding_mask"],
                    has_strokes=moved["has_strokes"],
                    max_tokens=max_tokens,
                )
            ids_cpu = ids.cpu().tolist()
            finished = ids.eq(tokenizer.eos_id).any(1).cpu().tolist()
            for row, sample in enumerate(chunk):
                pred = tokenizer.decode(ids_cpu[row])
                results.append({
                    "id": sample["id"],
                    "mode": sample["mode"],
                    "decoding": "greedy",
                    "prediction": pred,
                    "truncated": not finished[row],
                    "target": sample["target"],
                })
        else:
            out = decode_beam_and_r2l(
                model,
                moved,
                tokenizer,
                device,
                width=beam_width,
                max_tokens=max_tokens,
                length_penalty=length_penalty,
            )
            for row, sample in enumerate(chunk):
                cands = []
                for k in range(beam_width):
                    pred = tokenizer.decode(out["token_ids"][row][k])
                    l2r_norm = out["l2r_scores"][row][k] / (max(1, out["l2r_lengths"][row][k]) ** length_penalty)
                    r2l_norm = out["r2l_scores"][row][k] / (max(1, out["r2l_lengths"][row][k]) ** length_penalty)
                    combined = (1.0 - r2l_weight) * l2r_norm + r2l_weight * r2l_norm if use_r2l else l2r_norm
                    cands.append({
                        "prediction": pred,
                        "truncated": not out["finished"][row][k],
                        "l2r_score": round(l2r_norm, 5),
                        "r2l_score": round(r2l_norm, 5),
                        "combined": round(combined, 5),
                    })
                ranked = sorted(cands, key=lambda c: c["combined"], reverse=True)
                best = ranked[0]
                results.append({
                    "id": sample["id"],
                    "mode": sample["mode"],
                    "decoding": f"beam{beam_width}_r2l" if use_r2l else f"beam{beam_width}",
                    "prediction": best["prediction"],
                    "truncated": best["truncated"],
                    "target": sample["target"],
                    "candidates": ranked,
                })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to InkTeX .pt checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=None, help="Optional tokenizer.json (defaults to embedded)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inkml", type=Path, help="Single .inkml file or directory of .inkml files (bimodal)")
    group.add_argument("--image", type=Path, help="Single image file or directory of math images (vision-only)")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width (recommended: 5)")
    parser.add_argument("--r2l-weight", type=float, default=0.5, help="R2L reranking weight in [0, 1] (recommended: 0.5)")
    parser.add_argument("--length-penalty", type=float, default=0.7, help="Beam length penalty (recommended: 0.7)")
    parser.add_argument("--greedy", action="store_true", help="Use fast single-pass greedy decoding instead of Beam+R2L")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--show-nbest", action="store_true", help="Print all top-K beam candidates")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output file")
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = load_tokenizer(payload, args.tokenizer)
    model = model_from_payload(payload, device).eval()
    max_tokens = int(payload["model_config"].get("max_tokens", 256))

    if args.inkml is not None:
        paths = collect_paths(args.inkml, {".inkml"})
        samples = [prepare_inkml_sample(p) for p in paths]
    else:
        paths = collect_paths(args.image, IMAGE_SUFFIXES)
        samples = [prepare_image_sample(p) for p in paths]

    if not samples:
        raise SystemExit("No matching input files found")

    results = run_inference(
        model,
        tokenizer,
        samples,
        device,
        beam_width=args.beam_width,
        r2l_weight=args.r2l_weight,
        length_penalty=args.length_penalty,
        greedy=args.greedy,
        batch_size=args.batch_size,
        max_tokens=max_tokens,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(results)} predictions to {args.output} ({time.perf_counter() - t0:.1f}s)")
    else:
        for row in results:
            name = Path(row["id"]).name
            print(f"{name}\t{row['prediction']}")
            if row.get("target"):
                print(f"  target: {row['target']}")
            if args.show_nbest and row.get("candidates"):
                for rank, cand in enumerate(row["candidates"], 1):
                    print(f"  #{rank} (comb={cand['combined']:.4f} l2r={cand['l2r_score']:.4f} r2l={cand['r2l_score']:.4f}): {cand['prediction']}")


if __name__ == "__main__":
    main()
