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

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
]


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
    cols: int = 3,
    tile_w: int = 480,
    tile_h: int = 240,
) -> bytes:
    """Render numbered tiles `#1..#N` onto a clean PNG WITHOUT printing ground-truth LaTeX."""
    n = len(strokes_batch)
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
    """Robustly extract `{1: {'ok': bool, 'reason': str}, ...}` from a VLM JSON response."""
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

    parsed = json.loads(text)
    out: dict[int, dict[str, Any]] = {}
    if isinstance(parsed, list):
        for i, item in enumerate(parsed):
            if isinstance(item, dict):
                tid = int(item.get("id", i + 1))
                out[tid] = {"ok": bool(item.get("ok", False)), "reason": str(item.get("reason", ""))}
    elif isinstance(parsed, dict):
        items_src = parsed.get("results", parsed) if "results" in parsed and isinstance(parsed["results"], (dict, list)) else parsed
        if isinstance(items_src, list):
            for i, item in enumerate(items_src):
                if isinstance(item, dict):
                    tid = int(item.get("id", i + 1))
                    out[tid] = {"ok": bool(item.get("ok", False)), "reason": str(item.get("reason", ""))}
        elif isinstance(items_src, dict):
            for k, v in items_src.items():
                digits = re.sub(r"[^0-9]", "", str(k))
                if not digits:
                    continue
                tid = int(digits)
                if isinstance(v, bool):
                    out[tid] = {"ok": v, "reason": ""}
                elif isinstance(v, dict):
                    out[tid] = {"ok": bool(v.get("ok", False)), "reason": str(v.get("reason", ""))}
    # Ensure every 1..count key exists
    for idx in range(1, count + 1):
        out.setdefault(idx, {"ok": False, "reason": "missing_verdict"})
    return out


def call_openrouter_audit_grid(
    png_bytes: bytes,
    target_latex_list: Sequence[str],
    api_key: str,
    *,
    models: Sequence[str] = DEFAULT_FREE_MODELS,
    timeout_s: int = 60,
    max_retries: int = 4,
) -> tuple[dict[int, dict[str, Any]], str]:
    """Send a numbered grid PNG + candidate LaTeX list to OpenRouter free vision models."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    numbered_targets = "\n".join(f"#{i + 1}: {tex}" for i, tex in enumerate(target_latex_list))
    prompt = (
        "You are a strict math handwriting quality judge.\n"
        "The attached image shows numbered handwritten math tiles (#1 to #"
        f"{len(target_latex_list)}).\n"
        "Below is the intended LaTeX for each tile:\n"
        f"{numbered_targets}\n\n"
        "For EACH tile, check whether the handwritten ink is clearly legible human handwriting "
        "that faithfully matches the intended LaTeX (accept normal handwriting variants like "
        "\\to vs \\rightarrow, \\le vs \\leq, slightly messy penmanship, or Pr vs \\Pr; REJECT "
        "missing symbols, wrong digits/letters, illegible scribbles, giant stray slashes, or overlapping lines).\n"
        'Return ONLY valid JSON mapping tile number string to {"ok": true/false, "reason": "short note"}, e.g.:\n'
        '{"1": {"ok": true, "reason": "clear match"}, "2": {"ok": false, "reason": "missing exponent"}}'
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        model_id = models[attempt % len(models)]
        payload = {
            "model": model_id,
            "temperature": 0.0,
            "max_tokens": 900,
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
            content = data["choices"][0]["message"]["content"]
            used_model = str(data.get("model", model_id))
            verdicts = parse_vlm_verdict_json(content, len(target_latex_list))
            return verdicts, used_model
        except urllib.error.HTTPError as e:
            last_err = e
            wait_s = min(30.0, 2.5 * (2 ** attempt))
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


def audit_shard(
    input_jsonl: Path,
    out_verified: Path,
    out_rejected: Path,
    api_key: str,
    *,
    models: Sequence[str] = DEFAULT_FREE_MODELS,
    tiles_per_call: int = 9,
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
            png_bytes = render_blind_audit_grid(strokes_batch, cols=3)
            verdicts, used_model = call_openrouter_audit_grid(png_bytes, targets, api_key, models=models)
            stats["calls"] += 1

            for i, rec in enumerate(batch):
                v = verdicts.get(i + 1, {"ok": False, "reason": "missing"})
                rec.setdefault("metadata", {})
                rec["metadata"]["vlm_audit"] = {"model": used_model, "ok": v["ok"], "reason": v["reason"]}
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
    parser.add_argument("--tiles-per-call", type=int, default=9, help="Equations packed per 3x3 grid API call")
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
        tiles_per_call=args.tiles_per_call,
        min_delay_s=args.min_delay,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
