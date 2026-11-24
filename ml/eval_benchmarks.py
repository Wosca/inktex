"""Evaluate a BimodalMathModel checkpoint on official CROHME 2019/2023 (greedy, Beam-5, Beam-5+R2L)
and MathWriting test (7,644, Beam-5 & Beam-5+R2L) using exact unmodified ground-truth scorers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ML = Path(__file__).resolve().parent
if str(REPO_ML) not in sys.path:
    sys.path.insert(0, str(REPO_ML))

import numpy as np
import torch

from data.bimodal_pack import PREPROCESSING, RASTER, _has_timing, expression_strokes
from data.bimodal_raster import WIDTH_MULTIPLE, render_strokes_to_strip
from data.canonical import expression_from_record
from data.importers.mathwriting import MathWritingImporter
from data.preprocessing import preprocess_example
from evaluation.prepare import prepare_predictions
from evaluation.recognition import canonical
from research.incremental import DecoderState
from tokenizer.latex_tokenizer import LatexTokenizer
from train_bimodal import autocast, model_from_payload, score_prediction, summarize_rows
from training.benchmark_metrics import reference_token_cer
from training.structure_validation import wsl_path

BEAM_WIDTH = 5
LENGTH_PENALTY = 0.7
R2L_WEIGHT = 0.5


def _preprocess_expr(expr) -> tuple[np.ndarray, int, np.ndarray]:
    strokes = expression_strokes(expr)
    pil_img, width = render_strokes_to_strip(
        strokes, RASTER["height"], RASTER["min_width"], RASTER["max_width"], RASTER["online_padding"], None
    )
    image = np.ascontiguousarray(np.transpose(np.asarray(pil_img, dtype=np.uint8), (2, 0, 1)))
    processed = preprocess_example(expr, PREPROCESSING)
    features = np.ascontiguousarray(processed.features.astype(np.float32))
    if not _has_timing(expr):
        features[:, 4] = 0.0
    return image, width, features


def load_crohme_samples(test_jsonl: Path) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    samples = []
    with test_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            expr = expression_from_record(record)
            image, width, features = _preprocess_expr(expr)
            edition = record["id"].split("/")[1]
            samples.append({
                "id": record["id"],
                "edition": edition,
                "target": record.get("latex") or record.get("latexOriginal") or "",
                "normalized_target": record.get("latexNormalized") or record.get("latex") or "",
                "image": image,
                "image_width": width,
                "stroke_features": features,
            })
    print(f"Preprocessed {len(samples)} CROHME test equations in {time.perf_counter() - t0:.1f}s", flush=True)
    return samples


def load_mathwriting_samples(mw_test_dir: Path) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    importer = MathWritingImporter()
    paths = sorted(mw_test_dir.glob("*.inkml"))
    samples = []
    for path in paths:
        expr = importer.parse(path)
        image, width, features = _preprocess_expr(expr)
        samples.append({
            "id": expr.id,
            "edition": "mathwriting",
            "target": expr.latex,
            "normalized_target": expr.latex,
            "image": image,
            "image_width": width,
            "stroke_features": features,
        })
    print(f"Preprocessed {len(samples)} MathWriting test equations in {time.perf_counter() - t0:.1f}s", flush=True)
    return samples


def collate_eval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    height = int(batch[0]["image"].shape[1])
    width = int(math.ceil(max(int(s["image"].shape[2]) for s in batch) / WIDTH_MULTIPLE) * WIDTH_MULTIPLE)
    images = torch.zeros((len(batch), 2, height, width), dtype=torch.float32)
    for row, s in enumerate(batch):
        img = torch.from_numpy(s["image"])
        images[row, :, :, :img.shape[2]] = img.float().div_(255.0)
    image_widths = torch.tensor([int(s["image_width"]) for s in batch], dtype=torch.long)
    max_len = max(len(s["stroke_features"]) for s in batch)
    channels = batch[0]["stroke_features"].shape[1]
    stroke_features = torch.zeros((len(batch), max_len, channels), dtype=torch.float32)
    stroke_padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)
    for row, s in enumerate(batch):
        f = torch.from_numpy(s["stroke_features"])
        stroke_features[row, :len(f)] = f
        stroke_padding_mask[row, :len(f)] = False
    return {
        "images": images,
        "image_widths": image_widths,
        "stroke_features": stroke_features,
        "stroke_padding_mask": stroke_padding_mask,
        "has_strokes": torch.ones(len(batch), dtype=torch.bool),
    }


@torch.inference_mode()
def decode_beam_and_r2l(
    model,
    batch: dict[str, torch.Tensor],
    tokenizer: LatexTokenizer,
    device: torch.device,
    width: int = BEAM_WIDTH,
    max_tokens: int = 256,
    length_penalty: float = LENGTH_PENALTY,
) -> dict[str, Any]:
    bos_id, eos_id, pad_id = tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id
    with autocast(device, "bfloat16"):
        memory, memory_mask, _ = model.encode_bimodal(
            images=batch["images"],
            image_widths=batch["image_widths"],
            stroke_features=batch["stroke_features"],
            stroke_padding_mask=batch["stroke_padding_mask"],
            has_strokes=batch["has_strokes"],
        )

    bsize = memory.shape[0]
    state = DecoderState(model.decoder, memory, memory_mask)
    state.repeat_beams(width)

    sequences = torch.full((bsize, width, 1), bos_id, device=device, dtype=torch.long)
    scores = torch.full((bsize, width), -float("inf"), device=device, dtype=torch.float32)
    scores[:, 0] = 0.0
    lengths = torch.zeros((bsize, width), device=device, dtype=torch.long)
    finished = torch.zeros((bsize, width), device=device, dtype=torch.bool)
    batch_ids = torch.arange(bsize, device=device)[:, None]

    for _ in range(max_tokens - 1):
        step_token = sequences[:, :, -1].reshape(bsize * width, 1)
        with autocast(device, "bfloat16"):
            logits, _ = state.step(step_token, fast=True)
        log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1).reshape(bsize, width, -1)
        log_probs = log_probs.masked_fill(finished[..., None], -float("inf"))
        log_probs[:, :, pad_id] = torch.where(finished, 0.0, log_probs[:, :, pad_id])

        values, tokens = log_probs.topk(width, dim=-1)
        candidate_scores = scores[..., None] + values
        candidate_lengths = lengths + (~finished).long()
        ranked = candidate_scores / candidate_lengths.clamp_min(1).float()[..., None].pow(length_penalty)
        chosen = ranked.reshape(bsize, -1).topk(width, dim=-1).indices

        parents = chosen // width
        emitted = tokens.reshape(bsize, -1).gather(1, chosen)
        scores = candidate_scores.reshape(bsize, -1).gather(1, chosen)
        lengths = candidate_lengths.gather(1, parents)
        finished = finished.gather(1, parents) | emitted.eq(eos_id)
        sequences = torch.cat((sequences[batch_ids, parents], emitted[..., None]), dim=-1)

        flat_parents = (batch_ids * width + parents).reshape(-1)
        state.reorder_beams(flat_parents)

        if bool(finished.all()):
            break

    seq_cpu = sequences[:, :, 1:].cpu().tolist()
    r2l_seqs: list[list[int]] = []
    r2l_lens: list[int] = []
    for b in range(bsize):
        for k in range(width):
            body = []
            for tok in seq_cpu[b][k]:
                if tok == eos_id:
                    break
                if tok != pad_id:
                    body.append(tok)
            r2l = [bos_id, *reversed(body), eos_id]
            if len(r2l) > max_tokens:
                r2l = r2l[:max_tokens - 1] + [eos_id]
            r2l_seqs.append(r2l)
            r2l_lens.append(len(r2l) - 1)

    max_r2l = max(len(s) for s in r2l_seqs)
    r2l_tensor = torch.full((bsize * width, max_r2l), pad_id, dtype=torch.long, device=device)
    for row, s in enumerate(r2l_seqs):
        r2l_tensor[row, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    r2l_in = r2l_tensor[:, :-1]
    r2l_tgt = r2l_tensor[:, 1:]
    r2l_in_pad = r2l_in.eq(pad_id)
    r2l_tgt_valid = r2l_tgt.ne(pad_id)

    expanded_memory = memory.repeat_interleave(width, dim=0)
    expanded_mask = memory_mask.repeat_interleave(width, dim=0) if memory_mask is not None else None
    with autocast(device, "bfloat16"):
        r2l_logits = model.r2l_decoder(r2l_in, expanded_memory, expanded_mask, r2l_in_pad)
    r2l_logp = torch.log_softmax(r2l_logits.float(), dim=-1).gather(-1, r2l_tgt.unsqueeze(-1)).squeeze(-1)
    r2l_scores = (r2l_logp * r2l_tgt_valid.float()).sum(dim=-1).reshape(bsize, width)
    r2l_lengths = torch.tensor(r2l_lens, dtype=torch.long, device=device).reshape(bsize, width)

    return {
        "token_ids": seq_cpu,
        "l2r_scores": scores.cpu().tolist(),
        "l2r_lengths": lengths.cpu().tolist(),
        "r2l_scores": r2l_scores.cpu().tolist(),
        "r2l_lengths": r2l_lengths.cpu().tolist(),
        "finished": finished.cpu().tolist(),
    }


def run_crohme_organizer(
    name: str,
    ckpt_path: Path,
    out_dir: Path,
    samples: list[dict[str, Any]],
    preds_by_idx: dict[int, tuple[str, bool]],
    tools_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    datasets_out: dict[str, dict[str, Any]] = {}
    proxy_summary: dict[str, Any] = {}
    for edition in ("CROHME2019_test", "CROHME2023_test"):
        ed_preds = []
        ed_rows = []
        for idx, s in enumerate(samples):
            if s["edition"] != edition:
                continue
            pred, truncated = preds_by_idx[idx]
            ed_preds.append({"id": s["id"], "target": s["target"], "prediction": pred, "truncated": truncated})
            ed_rows.append(score_prediction(pred, s["normalized_target"], not truncated))
        datasets_out[edition] = {"predictions": ed_preds}
        proxy_summary[edition] = summarize_rows(ed_rows)
        print(
            f"  [{name}] {edition} proxy: norm={proxy_summary[edition]['normalized_exact']:.2%}, "
            f"cer={proxy_summary[edition]['token_cer']:.2%}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.json"
    pred_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path.resolve()),
        "split": "test",
        "datasets": datasets_out,
    }, ensure_ascii=False), encoding="utf-8")

    org_dir = out_dir / "organizer"
    if not (org_dir / "official_results.json").exists():
        t_org = time.perf_counter()
        prepare_predictions(pred_path, org_dir, {
            "tools": str(tools_path),
            "archive": str(archive_path),
            "symbol_normalization": "crohme_symbols_srt_v1",
        })
        scorer = REPO_ML / "evaluation" / "crohme.py"
        subprocess.run([
            "wsl", "-d", "Ubuntu", "--",
            "python3", "-u", wsl_path(scorer),
            "--tools", wsl_path(tools_path),
            "--work", wsl_path(org_dir),
            "--workers", "8",
        ], check=True)
        print(f"  [{name}] Organizer finished in {time.perf_counter() - t_org:.1f}s", flush=True)

    official = json.loads((org_dir / "official_results.json").read_text(encoding="utf-8"))
    summary = {
        "name": name,
        "checkpoint": str(ckpt_path),
        "latex_proxy": proxy_summary,
        "official": {
            ed: {k: v[k] for k in ("equations", "correct", "exprate", "full_test_equations", "full_test_exprate_bounds", "correct_with_0_to_3_errors") if k in v}
            for ed, v in official["datasets"].items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for ed, v in summary["official"].items():
        print(f"  [{name}] OFFICIAL {ed}: {v['exprate']:.2%} ({v['correct']}/{v['equations']})", flush=True)
    return summary


def score_mathwriting(name: str, samples: list[dict[str, Any]], preds_by_idx: dict[int, tuple[str, bool]]) -> dict[str, Any]:
    preds, targets, rows = [], [], []
    strict = canon = trunc = 0
    for idx, s in enumerate(samples):
        pred, truncated = preds_by_idx[idx]
        target = s["target"]
        preds.append(pred)
        targets.append(target)
        strict += int(pred == target)
        canon += int(canonical(pred) == canonical(target))
        trunc += int(truncated)
        rows.append(score_prediction(pred, s["normalized_target"], not truncated))
    n = len(samples)
    cer = reference_token_cer(preds, targets)
    proxy = summarize_rows(rows)
    res = {
        "name": name,
        "examples": n,
        "reference_token_cer": cer,
        "canonical_exact": canon / n,
        "strict_exact": strict / n,
        "normalized_exact": proxy["normalized_exact"],
        "token_cer": proxy["token_cer"],
        "truncated": trunc,
    }
    print(
        f"  [{name}] OFFICIAL MathWriting (n={n}): ref_token_cer={cer:.3%}, "
        f"canonical_exact={res['canonical_exact']:.2%}, norm_exact={res['normalized_exact']:.2%}, "
        f"strict={res['strict_exact']:.2%}",
        flush=True,
    )
    return res


@torch.inference_mode()
def run_greedy(model, samples: list[dict[str, Any]], tokenizer: LatexTokenizer, device: torch.device,
               max_gen: int, batch_size: int = 32, tag: str = "") -> dict[int, tuple[str, bool]]:
    t0 = time.perf_counter()
    order = sorted(range(len(samples)), key=lambda i: samples[i]["image_width"])
    preds_by_idx: dict[int, tuple[str, bool]] = {}
    for start in range(0, len(order), batch_size):
        chunk_indices = order[start:start + batch_size]
        batch = collate_eval([samples[i] for i in chunk_indices])
        moved = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast(device, "bfloat16"):
            ids = model.decode_greedy(
                images=moved["images"], image_widths=moved["image_widths"],
                stroke_features=moved["stroke_features"], stroke_padding_mask=moved["stroke_padding_mask"],
                has_strokes=moved["has_strokes"], max_tokens=max_gen,
            )
        ids = ids.cpu()
        finished = ids.eq(tokenizer.eos_id).any(1).tolist()
        for row, idx in enumerate(chunk_indices):
            preds_by_idx[idx] = (tokenizer.decode(ids[row].tolist()), not finished[row])
        done = min(start + batch_size, len(order))
        if done % 1024 == 0 or done == len(order):
            print(f"  [{tag}] greedy decoded {done}/{len(order)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return preds_by_idx


@torch.inference_mode()
def run_beam_r2l(model, samples: list[dict[str, Any]], tokenizer: LatexTokenizer, device: torch.device,
                 max_gen: int, batch_size: int = 16, tag: str = "",
                 beam_width: int = BEAM_WIDTH, length_penalty: float = LENGTH_PENALTY,
                 r2l_weight: float = R2L_WEIGHT) -> tuple[dict[int, tuple[str, bool]], dict[int, tuple[str, bool]], dict[int, list[dict[str, Any]]]]:
    t0 = time.perf_counter()
    order = sorted(range(len(samples)), key=lambda i: samples[i]["image_width"])
    beam_only_preds: dict[int, tuple[str, bool]] = {}
    beam_r2l_preds: dict[int, tuple[str, bool]] = {}
    nbest_dump: dict[int, list[dict[str, Any]]] = {}
    for start in range(0, len(order), batch_size):
        chunk_indices = order[start:start + batch_size]
        batch = collate_eval([samples[i] for i in chunk_indices])
        moved = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = decode_beam_and_r2l(model, moved, tokenizer, device, width=beam_width,
                                  max_tokens=max_gen, length_penalty=length_penalty)
        for row, idx in enumerate(chunk_indices):
            cands = []
            for k in range(beam_width):
                pred = tokenizer.decode(out["token_ids"][row][k])
                l2r_norm = out["l2r_scores"][row][k] / (max(1, out["l2r_lengths"][row][k]) ** length_penalty)
                r2l_norm = out["r2l_scores"][row][k] / (max(1, out["r2l_lengths"][row][k]) ** length_penalty)
                combined = (1.0 - r2l_weight) * l2r_norm + r2l_weight * r2l_norm
                cands.append({
                    "prediction": pred,
                    "truncated": not out["finished"][row][k],
                    "l2r_norm": l2r_norm,
                    "r2l_norm": r2l_norm,
                    "combined": combined,
                })
            nbest_dump[idx] = cands
            beam_only_preds[idx] = (cands[0]["prediction"], cands[0]["truncated"])
            best_r2l = max(cands, key=lambda c: c["combined"])
            beam_r2l_preds[idx] = (best_r2l["prediction"], best_r2l["truncated"])
        done = min(start + batch_size, len(order))
        if done % 1024 == 0 or done == len(order):
            print(f"  [{tag}] beam{beam_width}+r2l decoded {done}/{len(order)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return beam_only_preds, beam_r2l_preds, nbest_dump


def _resolve_tokenizer(payload: dict[str, Any], tokenizer_path: Path | None) -> LatexTokenizer:
    if tokenizer_path is not None:
        return LatexTokenizer.load(tokenizer_path)
    if payload.get("tokenizer") is not None:
        return LatexTokenizer.from_dict(payload["tokenizer"])
    fallback = REPO_ML / "tokenizer" / "bimodal_tokenizer.json"
    if fallback.is_file():
        return LatexTokenizer.load(fallback)
    raise FileNotFoundError("No tokenizer embedded in checkpoint and --tokenizer not provided")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None, help="Optional tokenizer.json (defaults to checkpoint embedded tokenizer)")
    parser.add_argument("--crohme-test", type=Path, default=Path(r"E:\math-recogniser-datasets\canonical\crohme23\test.jsonl"))
    parser.add_argument("--crohme-tools", type=Path, default=Path(r"E:\math-recogniser-training\tools\crohme-eval"))
    parser.add_argument("--crohme-archive", type=Path, default=Path(r"E:\math-recogniser-datasets\archives\CROHME23.zip"))
    parser.add_argument("--mathwriting-test", type=Path, default=Path(r"E:\mathwriting-2024\test"))
    parser.add_argument("--datasets", nargs="+", choices=("crohme", "mathwriting"), default=["crohme", "mathwriting"])
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH, help="Recommended: 5")
    parser.add_argument("--r2l-weight", type=float, default=R2L_WEIGHT, help="Recommended: 0.5")
    parser.add_argument("--length-penalty", type=float, default=LENGTH_PENALTY, help="Recommended: 0.7")
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== [{args.name}] Loading {args.checkpoint} ===", flush=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = _resolve_tokenizer(payload, args.tokenizer)
    model = model_from_payload(payload, device)
    model.eval()
    max_gen = int(payload["model_config"].get("max_tokens", 256))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "name": args.name,
        "checkpoint": str(args.checkpoint),
        "decoding": {"beam_width": args.beam_width, "r2l_weight": args.r2l_weight, "length_penalty": args.length_penalty},
    }

    if "crohme" in args.datasets:
        crohme_samples = load_crohme_samples(args.crohme_test)
        if not args.skip_greedy:
            greedy_preds = run_greedy(model, crohme_samples, tokenizer, device, max_gen, batch_size=32, tag=f"{args.name}_crohme")
            report["crohme_greedy"] = run_crohme_organizer(
                f"{args.name}_greedy", args.checkpoint, args.out_dir / "crohme_greedy",
                crohme_samples, greedy_preds, args.crohme_tools, args.crohme_archive,
            )
        beam_only, beam_r2l, nbest = run_beam_r2l(
            model, crohme_samples, tokenizer, device, max_gen, batch_size=args.batch_size,
            tag=f"{args.name}_crohme", beam_width=args.beam_width,
            length_penalty=args.length_penalty, r2l_weight=args.r2l_weight,
        )
        (args.out_dir / "crohme_nbest.json").write_text(
            json.dumps([{"id": crohme_samples[i]["id"], "candidates": nbest[i]} for i in range(len(crohme_samples))], ensure_ascii=False),
            encoding="utf-8",
        )
        report["crohme_beam5"] = run_crohme_organizer(
            f"{args.name}_beam{args.beam_width}", args.checkpoint, args.out_dir / f"crohme_beam{args.beam_width}_only",
            crohme_samples, beam_only, args.crohme_tools, args.crohme_archive,
        )
        report["crohme_beam5_r2l"] = run_crohme_organizer(
            f"{args.name}_beam{args.beam_width}_r2l", args.checkpoint, args.out_dir / f"crohme_beam{args.beam_width}_r2l",
            crohme_samples, beam_r2l, args.crohme_tools, args.crohme_archive,
        )
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if "mathwriting" in args.datasets:
        mw_samples = load_mathwriting_samples(args.mathwriting_test)
        beam_only, beam_r2l, _ = run_beam_r2l(
            model, mw_samples, tokenizer, device, max_gen, batch_size=args.batch_size,
            tag=f"{args.name}_mw", beam_width=args.beam_width,
            length_penalty=args.length_penalty, r2l_weight=args.r2l_weight,
        )
        report["mathwriting_beam5"] = score_mathwriting(f"{args.name}_mw_beam{args.beam_width}", mw_samples, beam_only)
        report["mathwriting_beam5_r2l"] = score_mathwriting(f"{args.name}_mw_beam{args.beam_width}_r2l", mw_samples, beam_r2l)
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== FULL BENCHMARK REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
