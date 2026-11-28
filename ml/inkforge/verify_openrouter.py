"""Independent Free-VLM Quality Auditor for InkForge Synthetic Shards via OpenRouter.

Uses OpenRouter's free vision models (default: `openrouter/free`, which automatically routes
image payloads to an active vision-capable free model such as `nvidia/nemotron-3.5-super-vl:free`
or `google/gemma-4-26b-a4b-it:free`) to audit ASSEMBLED 2D handwritten equations (multi-line proofs,
big nested fractions, matrices, piecewise cases) completely independently of InkTeX v2!

Throughput hack: packs `tiles_per_call` (default 9 = 3x3 grid) equations onto ONE clean numbered
contact-sheet PNG (NO ground-truth LaTeX drawn in the image so the VLM cannot cheat!) and asks for
a compact JSON verdict per tile, multiplying free-tier quota by 9x!
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.render import render_strokes_tile  # noqa: E402
from inkforge.star_dpo import _canonicalize_judge_latex  # noqa: E402
from tokenizer.latex_tokenizer import tokenize_latex  # noqa: E402
from v2.eval_v2 import _safe_std_v2  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_FREE_MODELS = [
    "nvidia/nemotron-3.5-super-vl:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free",
    "google/gemma-4-31b-it:free",
]


def _normalize_for_vlm_compare(latex: str) -> list[str]:
    """Canonicalize LaTeX for blind VLM-vs-ground-truth comparison (folds harmless formatting/env aliases)."""
    s = _canonicalize_judge_latex(latex or "")
    s = s.strip().strip("$").strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    # Fold delimiter-wrapped \begin{matrix} / \begin{array} into vmatrix/bmatrix/pmatrix/cases equivalently
    s = s.replace(r"\\end{", r"\end{")
    s = re.sub(r"\\left\\\{\s*\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}\s*\\right\.", r"\\begin{cases}\1\\end{cases}", s, flags=re.DOTALL)
    s = re.sub(r"\\left\|\s*\\begin\{matrix\}(.*?)\\end\{matrix\}\s*\\right\|", r"\\begin{vmatrix}\1\\end{vmatrix}", s, flags=re.DOTALL)
    s = re.sub(r"\\left\[\s*\\begin\{matrix\}(.*?)\\end\{matrix\}\s*\\right\]", r"\\begin{bmatrix}\1\\end{bmatrix}", s, flags=re.DOTALL)
    s = re.sub(r"\\left\(\s*\\begin\{matrix\}(.*?)\\end\{matrix\}\s*\\right\)", r"\\begin{pmatrix}\1\\end{pmatrix}", s, flags=re.DOTALL)
    # Strip harmless trailing \\ immediately before \end{...} (VLMs routinely write `otherwise \\ \end{cases}`)
    s = re.sub(r"\\\\\s*(?=\\end\{)", " ", s)
    # Fold \widetilde -> \tilde, \widehat -> \hat, \triangle -> \Delta, arrow aliases, sizing commands, and aligned wrappers
    s = s.replace(r"\widetilde", r"\tilde").replace(r"\widehat", r"\hat")
    s = s.replace(r"\triangle", r"\Delta").replace(r"\implies", r"\Rightarrow").replace(r"\iff", r"\Leftrightarrow")
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b", "", s)
    s = re.sub(r"\\begin\{(aligned|align\*?|gather\*?)\}", "", s)
    s = re.sub(r"\\end\{(aligned|align\*?|gather\*?)\}", "", s)
    s = s.replace("&", " ")
    s = re.sub(r"\\(?:qquad|quad|,|;|!|:)\b", " ", s)
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", " ")
    toks = tokenize_latex(_safe_std_v2(s))
    # Fold visually identical handwritten uppercase/lowercase single letters (c/C, p/P, s/S, v/V, w/W, x/X, y/Y, z/Z, k/K, o/O)
    ambig_upper = {"C": "c", "K": "k", "O": "o", "P": "p", "S": "s", "V": "v", "W": "w", "X": "x", "Y": "y", "Z": "z"}
    return [ambig_upper.get(t, t) for t in toks]


def record_strokes_to_arrays(rec: dict[str, Any]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for st in rec.get("strokes", []):
        pts = st.get("points", []) if isinstance(st, dict) else st
        xy = []
        for p in pts:
            if isinstance(p, dict) and "x" in p and "y" in p:
                xy.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                xy.append((float(p[0]), float(p[1])))
        if xy:
            out.append(np.asarray(xy, dtype=np.float64))
    return out


def render_blind_audit_grid(
    strokes_batch: Sequence[list[np.ndarray]],
    *,
    cols: int | None = None,
    tile_w: int | None = None,
    tile_h: int | None = None,
) -> bytes:
    """Render numbered tiles `#1..#N` onto a high-res PNG WITHOUT printing ground-truth LaTeX.

    For small batches (`<= 3` tiles), defaults to a 1-column vertical stack (`900x240` per equation)
    giving maximum pixel clarity for long equations, subscripts, accents, and rare Greek letters!
    """
    n = len(strokes_batch)
    if cols is None:
        cols = 1 if n <= 3 else (2 if n <= 6 else 3)
    if tile_w is None:
        tile_w = 900 if cols == 1 else (620 if cols == 2 else 480)
    if tile_h is None:
        tile_h = 240
    rows = max(1, (n + cols - 1) // cols)
    pad = 8
    canvas = Image.new("RGB", (cols * (tile_w + pad) + pad, rows * (tile_h + pad) + pad), (232, 235, 240))
    for idx, strokes in enumerate(strokes_batch):
        r, c = divmod(idx, cols)
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)
        tile = render_strokes_tile(strokes, f"#{idx + 1}", width=tile_w, height=tile_h, color_by_order=False)
        canvas.paste(tile, (x0, y0))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def parse_vlm_verdict_json(raw_text: str, count: int) -> dict[int, dict[str, Any]]:
    """Robustly extract `{1: {'ok': bool, 'reason': str, 'latex': str}, ...}` from a VLM JSON response."""
    text = (raw_text or "").strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start_obj = text.find("{")
        start_arr = text.find("[")
        starts = [x for x in (start_obj, start_arr) if x >= 0]
        if starts:
            s0 = min(starts)
            end_ch = "}" if text[s0] == "{" else "]"
            e0 = text.rfind(end_ch)
            if e0 > s0:
                text = text[s0 : e0 + 1]

    # Fix unescaped LaTeX backslashes inside JSON strings (very common when VLMs output \frac, \theta, \sqrt!)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        parsed = json.loads(repaired)

    out: dict[int, dict[str, Any]] = {}
    if isinstance(parsed, list):
        for i, item in enumerate(parsed):
            if isinstance(item, dict):
                tid = int(item.get("id", i + 1))
                out[tid] = {
                    "ok": bool(item.get("ok", item.get("legible", False))),
                    "reason": str(item.get("reason", "")),
                    "latex": str(item.get("latex", "")),
                }
    elif isinstance(parsed, dict):
        items_src = parsed.get("results", parsed) if "results" in parsed and isinstance(parsed["results"], (dict, list)) else parsed
        if isinstance(items_src, list):
            for i, item in enumerate(items_src):
                if isinstance(item, dict):
                    tid = int(item.get("id", i + 1))
                    out[tid] = {
                        "ok": bool(item.get("ok", item.get("legible", False))),
                        "reason": str(item.get("reason", "")),
                        "latex": str(item.get("latex", "")),
                    }
        elif isinstance(items_src, dict):
            for k, v in items_src.items():
                digits = re.sub(r"[^0-9]", "", str(k))
                if not digits:
                    continue
                tid = int(digits)
                if isinstance(v, bool):
                    out[tid] = {"ok": v, "reason": "", "latex": ""}
                elif isinstance(v, str):
                    out[tid] = {"ok": True, "reason": "", "latex": v}
                elif isinstance(v, dict):
                    out[tid] = {
                        "ok": bool(v.get("ok", v.get("legible", False))),
                        "reason": str(v.get("reason", "")),
                        "latex": str(v.get("latex", "")),
                    }
    # Ensure every 1..count key exists
    for idx in range(1, count + 1):
        out.setdefault(idx, {"ok": False, "reason": "missing_verdict", "latex": ""})
    return out


def call_openrouter_audit_grid(
    png_bytes: bytes,
    target_latex_list: Sequence[str],
    api_key: str,
    *,
    models: Sequence[str] = DEFAULT_FREE_MODELS,
    mode: str = "blind",
    timeout_s: int = 60,
    max_retries: int = 5,
) -> tuple[dict[int, dict[str, Any]], str]:
    """Send a numbered grid PNG to OpenRouter free vision models (`blind` transcription or `verify` mode)."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    n_tiles = len(target_latex_list)
    if mode == "blind":
        prompt = (
            f"The attached image contains {n_tiles} numbered handwritten math tiles (#1 to #{n_tiles}).\n"
            "For EACH tile, blindly transcribe LITERALLY every symbol actually drawn into standard LaTeX.\n"
            "CRITICAL ANTI-HALLUCINATION RULE: Do NOT auto-complete famous textbook formulas (like Bayes' theorem, "
            "Binomial theorem, or Quadratic formula) from memory! If a bar (\\bar{A}), parenthesis, exponent, "
            "or character is missing, miswritten, or repeated in the ink, transcribe the literal broken ink and "
            'set "legible": false.\n'
            "Use \\\\ for line breaks, \\begin{bmatrix}/\\begin{pmatrix}/\\begin{vmatrix} for matrices, "
            "and \\begin{cases} for piecewise functions.\n"
            'Return ONLY valid JSON (double-escape every LaTeX backslash like \\\\frac) mapping tile number '
            'to {"latex": "...", "legible": true/false, "reason": "short note"}, e.g.:\n'
            '{"1": {"latex": "x^2 + y^2 = 1", "legible": true, "reason": "clear"}}'
        )
    else:
        numbered_targets = "\n".join(f"#{i + 1}: {tex}" for i, tex in enumerate(target_latex_list))
        prompt = (
            "You are a strict math handwriting quality judge.\n"
            f"The attached image shows numbered handwritten math tiles (#1 to #{n_tiles}).\n"
            "First read each tile independently character-by-character, then compare against the intended LaTeX:\n"
            f"{numbered_targets}\n\n"
            "REJECT any tile with a wrong digit/letter (e.g. 0 instead of r), missing symbol/operator, "
            "garbled word (e.g. Siii instead of sin), giant stray slash, or overlapping lines.\n"
            'Return ONLY valid JSON (double-escape backslashes) mapping tile number to '
            '{"ok": true/false, "latex": "what you actually read", "reason": "short note"}.'
        )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        model_id = models[attempt % len(models)]
        payload: dict[str, Any] = {
            "model": model_id,
            "temperature": 0.0,
            "max_tokens": 2200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        }
        if model_id != "openrouter/free":
            payload["reasoning"] = {"effort": "none"}
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Wosca/inktex",
                "X-Title": "InkForge Synthetic Data Auditor",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip() or (msg.get("reasoning") or "").strip()
            used_model = str(data.get("model", model_id))
            verdicts = parse_vlm_verdict_json(content, n_tiles)
            if mode == "blind":
                for idx, target_tex in enumerate(target_latex_list, start=1):
                    v = verdicts[idx]
                    pred_tex = v.get("latex", "")
                    legible = bool(v.get("ok", False))
                    t_toks = _normalize_for_vlm_compare(target_tex)
                    p_toks = _normalize_for_vlm_compare(pred_tex)
                    exact = bool(t_toks) and (p_toks == t_toks)
                    v["ok"] = legible and exact
                    if not legible:
                        v["reason"] = f"illegible: {v.get('reason', '')}".strip()
                    elif not exact:
                        v["reason"] = f"blind_mismatch: read `{pred_tex}`"
                    else:
                        v["reason"] = "blind_exact_match"
            return verdicts, used_model
        except urllib.error.HTTPError as e:
            last_err = e
            wait_s = min(20.0, 2.0 * (attempt + 1))
            if e.code in (429, 502, 503, 504):
                time.sleep(wait_s)
                continue
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            raise RuntimeError(f"OpenRouter HTTP {e.code} on {model_id}: {body}") from e
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"OpenRouter audit failed after {max_retries} retries: {last_err}")


def call_openrouter_single_recheck(
    strokes: Sequence[np.ndarray],
    target_latex: str,
    pass1_latex: str,
    api_key: str,
    *,
    models: Sequence[str] = DEFAULT_FREE_MODELS,
    timeout_s: int = 60,
    max_retries: int = 4,
) -> tuple[dict[str, Any], str]:
    """Individual high-res (960x320) BLIND single-equation re-transcription on Pass-1 disagreements.

    Re-renders ONLY the failed equation at full 960x320 resolution with zero neighboring tiles and
    runs a blind transcription pass with the strongest vision model (`models[0]`) — eliminating both
    multi-tile interference and target-prompt sycophancy!
    """
    png_bytes = render_blind_audit_grid([list(strokes)], cols=1, tile_w=960, tile_h=320)
    verdicts, used_model = call_openrouter_audit_grid(
        png_bytes,
        [target_latex],
        api_key,
        models=models,
        mode="blind",
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    v = verdicts[1]
    return {
        "ok": bool(v.get("ok", False)),
        "latex": str(v.get("latex", "")),
        "reason": str(v.get("reason", "")),
    }, used_model


def audit_shard(
    input_jsonl: Path,
    out_verified: Path,
    out_rejected: Path,
    api_key: str,
    *,
    models: Sequence[str] = DEFAULT_FREE_MODELS,
    mode: str = "blind",
    tiles_per_call: int = 3,
    recheck_failed: bool = True,
    min_delay_s: float = 3.2,
    max_samples: int = 0,
) -> dict[str, int]:
    """Resumable batch VLM auditor over a `stroke-latex-v1` JSONL shard."""
    out_verified.parent.mkdir(parents=True, exist_ok=True)
    out_rejected.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    for p in (out_verified, out_rejected):
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        done_ids.add(str(json.loads(line).get("id", "")))
                    except Exception:
                        continue

    pending_recs: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if str(rec.get("id", "")) in done_ids:
                continue
            pending_recs.append(rec)
            if max_samples > 0 and len(pending_recs) >= max_samples:
                break

    stats = {"accepted": 0, "rejected": 0, "calls": 0}
    if not pending_recs:
        print(f"[openrouter-audit] Nothing left to audit in {input_jsonl}")
        return stats

    print(f"[openrouter-audit] Auditing {len(pending_recs):,} samples ({tiles_per_call}/call) via {models[0]}...")
    with out_verified.open("a", encoding="utf-8") as fv, out_rejected.open("a", encoding="utf-8") as fr:
        for start in range(0, len(pending_recs), tiles_per_call):
            batch = pending_recs[start : start + tiles_per_call]
            t0 = time.perf_counter()
            strokes_batch = [record_strokes_to_arrays(r) for r in batch]
            targets = [str(r.get("latexNormalized") or r.get("latex") or "") for r in batch]
            png_bytes = render_blind_audit_grid(strokes_batch)
            verdicts, used_model = call_openrouter_audit_grid(png_bytes, targets, api_key, models=models, mode=mode)
            stats["calls"] += 1

            for i, rec in enumerate(batch):
                v = verdicts.get(i + 1, {"ok": False, "reason": "missing", "latex": ""})
                recheck_info: dict[str, Any] | None = None
                if (not v["ok"]) and recheck_failed and strokes_batch[i]:
                    try:
                        time.sleep(max(1.2, min_delay_s * 0.5))
                        rv, r_model = call_openrouter_single_recheck(
                            strokes_batch[i],
                            targets[i],
                            v.get("latex", ""),
                            api_key,
                            models=models,
                        )
                        stats["calls"] += 1
                        recheck_info = {"model": r_model, **rv}
                        if rv["ok"]:
                            v = {
                                "ok": True,
                                "latex": rv.get("latex", v.get("latex", "")),
                                "reason": f"rescued_single_recheck: {rv.get('reason', '')}",
                            }
                    except Exception as e:
                        recheck_info = {"error": str(e)}

                rec.setdefault("metadata", {})
                rec["metadata"]["vlm_audit"] = {
                    "model": used_model,
                    "mode": mode,
                    "ok": v["ok"],
                    "reason": v["reason"],
                    "vlm_latex": v.get("latex", ""),
                    **({"recheck": recheck_info} if recheck_info is not None else {}),
                }
                if v["ok"]:
                    fv.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1
                else:
                    fr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["rejected"] += 1
            fv.flush()
            fr.flush()

            total_done = stats["accepted"] + stats["rejected"]
            acc_pct = 100.0 * stats["accepted"] / max(1, total_done)
            print(
                f"[openrouter-audit] {total_done}/{len(pending_recs)} | "
                f"accept={acc_pct:.1f}% ({stats['accepted']} ok / {stats['rejected']} rej) | "
                f"model={used_model}",
                flush=True,
            )
            elapsed = time.perf_counter() - t0
            if elapsed < min_delay_s and start + tiles_per_call < len(pending_recs):
                time.sleep(min_delay_s - elapsed)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit InkForge synthetic shards using free OpenRouter vision models")
    parser.add_argument("--input", type=Path, required=True, help="Input stroke-latex-v1 JSONL shard")
    parser.add_argument("--out-verified", type=Path, required=True, help="Output verified JSONL")
    parser.add_argument("--out-rejected", type=Path, default=None, help="Output rejected JSONL")
    parser.add_argument("--models", nargs="+", default=DEFAULT_FREE_MODELS, help="OpenRouter model IDs to rotate")
    parser.add_argument("--mode", choices=("blind", "verify"), default="blind", help="blind=zero-sycophancy transcribe+compare")
    parser.add_argument("--tiles-per-call", type=int, default=3, help="Equations packed per high-res grid API call (default 3 = 1x3 vertical strip)")
    parser.add_argument("--no-recheck-failed", action="store_true", help="Disable individual high-res single-image tiebreaker on Pass-1 fails")
    parser.add_argument("--min-delay", type=float, default=3.2, help="Min seconds between free-tier calls")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick sampling audit (0=all)")
    parser.add_argument("--api-key-file", type=Path, default=None, help="Optional path to file containing OPENROUTER_API_KEY")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key and args.api_key_file and args.api_key_file.is_file():
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY environment variable (or --api-key-file).")

    out_rej = args.out_rejected or args.out_verified.with_name(args.out_verified.stem + "_rejected.jsonl")
    audit_shard(
        args.input,
        args.out_verified,
        out_rej,
        api_key,
        models=args.models,
        mode=args.mode,
        tiles_per_call=args.tiles_per_call,
        recheck_failed=(not args.no_recheck_failed),
        min_delay_s=args.min_delay,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
