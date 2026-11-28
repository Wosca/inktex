"""Hierarchical 2D Math Ink Composer & Real-World STEM Prompt Miner for InkForge.

Solves the two biggest weaknesses of single-line HMER generators:
  1. Generates SINGLE-LINE STEM equations (from UniMER 300k + curated STEM templates) directly via InkForge.
  2. Generates MULTI-LINE DERIVATIONS (`\\`, `\\begin{aligned}`), MATRICES (`bmatrix`, `pmatrix`, `vmatrix`, `matrix`),
     and PIECEWISE SYSTEMS (`cases`) by generating EVERY row/cell with the EXACT SAME writer style latent `z`,
     then laying them out in 2D with human alignment physics (`=`-alignment or margin wander, slight baseline tilt,
     and organically hand-drawn tall brackets `[ ]`, `( )`, `| |`, `{`).
  3. Synthesizes realistic Apple-Pencil-like timestamps + Catmull-Rom smoothing + geometric legibility validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from .tokenizer import InkForgeTokenizer, detokenize_inkforge, tokenize_inkforge

DEFAULT_UNIMER_JSONL = Path(r"E:\math-recogniser-datasets\external\unimer\examples.jsonl")


# ---------------------------------------------------------------------------
# Post-Processing: Smoothing, Synthetic Timing & Quality Gate
# ---------------------------------------------------------------------------

def smooth_strokes_catmull_rom(strokes: Sequence[np.ndarray], subdivisions: int = 2) -> list[np.ndarray]:
    """Gently upsample generated strokes with centripetal Catmull-Rom splines for natural pen fluidity."""
    out: list[np.ndarray] = []
    for st in strokes:
        arr = np.asarray(st, dtype=np.float64)[:, :2]
        n = len(arr)
        if n <= 2 or subdivisions <= 1:
            out.append(arr)
            continue
        pts: list[np.ndarray] = [arr[0]]
        for i in range(n - 1):
            p0 = arr[max(0, i - 1)]
            p1 = arr[i]
            p2 = arr[i + 1]
            p3 = arr[min(n - 1, i + 2)]
            for s in range(1, subdivisions + 1):
                t = s / subdivisions
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                pts.append(pt)
        out.append(np.asarray(pts, dtype=np.float64))
    return out


def synthesize_realistic_times(
    strokes: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate human-like timestamps (ms) proportional to arc length + pen-lift travel time."""
    times: list[np.ndarray] = []
    cursor_ms = float(rng.uniform(0.0, 50.0))
    # Human writing speed: ~180-340 ms per glyph-height unit of ink
    ms_per_unit = float(rng.uniform(170.0, 320.0))
    prev_end: np.ndarray | None = None

    for st in strokes:
        n = len(st)
        if n == 0:
            times.append(np.empty(0, dtype=np.float64))
            continue
        if prev_end is not None:
            air_dist = float(np.linalg.norm(st[0] - prev_end))
            cursor_ms += float(rng.uniform(75.0, 190.0)) + air_dist * float(rng.uniform(25.0, 65.0))
        t_arr = np.empty((n,), dtype=np.float64)
        t_arr[0] = cursor_ms
        if n > 1:
            deltas = np.diff(st[:, :2], axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            travel = np.cumsum(distances)
            progress = (travel - 0.5 * distances) / max(float(travel[-1]), 1e-6)
            # Ease in/out, slow on tight bends, and vary speed across the writer's strokes.
            pace = 1.0 + rng.uniform(0.15, 0.50) * (2.0 * progress - 1.0) ** 2
            pace *= 1.0 + 0.18 * np.sin(2.0 * math.pi * progress + rng.uniform(0, 2.0 * math.pi))
            unit = deltas / np.maximum(distances[:, None], 1e-6)
            pace[1:] += 0.25 * (1.0 - (unit[:-1] * unit[1:]).sum(axis=1))
            intervals = distances * ms_per_unit * rng.uniform(0.72, 1.35) * pace
            t_arr[1:] = cursor_ms + np.cumsum(np.maximum(3.0, intervals * rng.uniform(0.85, 1.15, n - 1)))
            cursor_ms = float(t_arr[-1])
        times.append(t_arr)
        prev_end = st[-1]
    return times


def strokes_bbox(strokes: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    pts = [s[:, :2] for s in strokes if len(s) > 0]
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    all_xy = np.concatenate(pts, axis=0)
    return (
        float(all_xy[:, 0].min()),
        float(all_xy[:, 1].min()),
        float(all_xy[:, 0].max()),
        float(all_xy[:, 1].max()),
    )


def passes_geometry_gate(
    latex: str,
    strokes: Sequence[np.ndarray],
    *,
    is_atomic_cell: bool = False,
) -> bool:
    """Reject degenerate/runaway generated strokes before saving to disk."""
    if not strokes:
        return False
    pts = [s[:, :2] for s in strokes if len(s) > 0]
    if not pts:
        return False
    all_xy = np.concatenate(pts, axis=0)
    if not np.isfinite(all_xy).all():
        return False
    x0, y0, x1, y1 = strokes_bbox(pts)
    w, h = x1 - x0, y1 - y0
    if max(w, h) < 0.20 or max(w, h) > 95.0:
        return False

    clean_tex = re.sub(r"\\(?:begin|end)\{[A-Za-z*]+\}", "", latex)
    toks = tokenize_inkforge(clean_tex)
    vis_est = max(1, sum(1 for t in toks if t not in ("{", "}", "^", "_", "<SPACE>", "&", r"\\")))

    if is_atomic_cell:
        has_tall_op = any(t in toks for t in (r"\frac", r"\int", r"\sum", r"\prod", r"\begin", r"\sqrt"))
        # A couple of horizontal glyphs can set the rollout's scale, making its one letter tall.
        single_scripted_glyph = toks[1:2] in (["_"], ["^"]) and not any(
            tok in ("=", "+", "-", "/", "<", ">") for tok in toks[2:]
        )
        short_cell = vis_est <= 3 or (
            vis_est <= 6 and (latex.rstrip().endswith("=") or single_scripted_glyph)
        )
        max_h = 8.5 if has_tall_op or short_cell else 4.8
        if h > max_h or (short_cell and w > 15.0):
            return False
        heights = [float(np.ptp(st[:, 1])) for st in pts]
        stroke_cap = 6.5 if has_tall_op else max_h if short_cell else 3.8
        if max(heights) > stroke_cap:
            return False
        if short_cell and not has_tall_op and "/" not in toks:
            for idx, (st, sh) in enumerate(zip(pts, heights)):
                if sh <= 3.8 or float(np.ptp(st[:, 0])) <= 0.32 * sh:
                    continue
                chord = float(np.linalg.norm(st[-1] - st[0]))
                length = float(np.linalg.norm(np.diff(st, axis=0), axis=1).sum())
                has_peer = any(i != idx and height >= 0.68 * sh for i, height in enumerate(heights))
                if chord >= 0.95 * length and not has_peer:
                    return False

    n_strokes = len(pts)
    if vis_est >= 4 and n_strokes < max(2, int(vis_est * 0.25)):
        return False
    if n_strokes > max(28, vis_est * 6):
        return False
    return True


# ---------------------------------------------------------------------------
# Hand-Drawn Organic Tall Delimiters (`[ ]`, `( )`, `| |`, `{`)
# ---------------------------------------------------------------------------

def _stroke_progress(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample increasing offsets with both local jitter and a smooth density change."""
    pace = rng.lognormal(0.0, 0.32, size=n - 1)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    pace *= 1.0 + 0.27 * np.sin(np.linspace(phase, phase + math.pi, n - 1))
    progress = np.concatenate(([0.0], np.cumsum(pace)))
    return progress / progress[-1]


def draw_human_delimiter(
    kind: str,
    x_center: float,
    y_top: float,
    y_bot: float,
    rng: np.random.Generator,
    *,
    slant: float = 0.0,
) -> list[np.ndarray]:
    """Generate realistic hand-drawn vertical delimiters with organic wobble, slant, and hook variance."""
    height = max(y_bot - y_top, 1.0)
    y_t = y_top - float(rng.uniform(0.08, 0.25))
    y_b = y_bot + float(rng.uniform(0.08, 0.25))
    h = y_b - y_t
    n = max(14, min(42, int(round(h * 6.5))))
    ts = _stroke_progress(n, rng)
    ys = y_t + ts * h

    # Smooth low-frequency hand tremor
    phase1 = float(rng.uniform(0, 2 * math.pi))
    phase2 = float(rng.uniform(0, 2 * math.pi))
    wobble = 0.028 * np.sin(ts * math.pi * float(rng.uniform(1.5, 3.0)) + phase1) + 0.014 * np.sin(
        ts * math.pi * 5.0 + phase2
    )
    slant_shift = (0.5 - ts) * h * slant

    if kind == "|":
        xs = x_center + wobble + slant_shift
        return [np.stack([xs, ys], axis=-1)]

    if kind in ("(", ")"):
        sign = -1.0 if kind == "(" else 1.0
        bulge = min(0.55, max(0.24, 0.14 * h)) * float(rng.uniform(0.85, 1.18))
        # Quadratic arc: curves outward at center, inward at top/bottom
        arc = sign * bulge * (1.0 - (2.0 * ts - 1.0) ** 2)
        xs = x_center + arc - sign * 0.5 * bulge + wobble + slant_shift
        return [np.stack([xs, ys], axis=-1)]

    if kind in ("[", "]"):
        sign = 1.0 if kind == "[" else -1.0
        hook = min(0.45, max(0.22, 0.12 * h)) * float(rng.uniform(0.85, 1.15))
        # Single continuous stroke: top hook -> vertical spine -> bottom hook (how humans draw [ and ])
        mid_x = x_center + wobble + slant_shift
        mid_y = ys
        top_t, bot_t = _stroke_progress(5, rng), _stroke_progress(5, rng)
        top_x = mid_x[0] + sign * hook * (1.0 - top_t)
        bot_x = mid_x[-1] + sign * hook * bot_t
        hook_tilt = rng.normal(0.18 * slant, 0.045)
        top_y = y_t + (top_x - mid_x[0]) * hook_tilt
        bot_y = y_b + (bot_x - mid_x[-1]) * hook_tilt
        xs = np.concatenate([top_x[:-1], mid_x, bot_x[1:]])
        ys_full = np.concatenate([top_y[:-1], mid_y, bot_y[1:]])
        return [np.stack([xs, ys_full], axis=-1)]

    if kind == "{":
        # Curly brace with independently shaped upper and lower hooks
        mid_y = 0.5 * (y_t + y_b) + float(rng.uniform(-0.08, 0.08))
        depth = min(0.50, max(0.25, 0.14 * h)) * float(rng.uniform(0.88, 1.15))
        half = n // 2
        t_top = _stroke_progress(half, rng)
        t_bot = _stroke_progress(n - half, rng)
        y_u = y_t + t_top * (mid_y - y_t)
        y_l = mid_y + t_bot * (y_b - mid_y)
        # Top half curves from right hook -> spine -> left tip
        x_u = x_center + depth * (0.5 * (1.0 - t_top) ** 2 - 0.65 * (t_top ** 3))
        x_l = x_center + depth * (0.5 * (t_bot ** 2) - 0.65 * ((1.0 - t_bot) ** 3))
        ys_full = np.concatenate([y_u, y_l[1:]])
        xs = np.concatenate([x_u, x_l[1:]]) + np.interp(ys_full, ys, wobble + slant_shift)
        return [np.stack([xs, ys_full], axis=-1)]

    return []


def draw_human_hline(
    x_left: float,
    x_right: float,
    y_bar: float,
    rng: np.random.Generator,
    *,
    slant: float = 0.0,
) -> list[np.ndarray]:
    """Generate a natural hand-drawn horizontal fraction bar (vinculum) with subtle tilt & hand wobble."""
    w = max(x_right - x_left, 0.6)
    n = max(10, min(64, int(round(w * 6.0))))
    ts = _stroke_progress(n, rng)
    xs = x_left + ts * w
    slope = float(rng.normal(0.22 * slant, 0.015))
    arch = float(rng.normal(0.0, 0.035)) * np.sin(ts * math.pi)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    wobble = 0.016 * np.sin(ts * math.pi * float(rng.uniform(2.0, 4.0)) + phase)
    ys = y_bar + (ts - 0.5) * w * slope + arch + wobble
    return [np.stack([xs, ys], axis=-1)]


def draw_human_sqrt(
    x_left: float,
    y_top: float,
    x_right: float,
    y_bot: float,
    rng: np.random.Generator,
    *,
    slant: float = 0.0,
) -> list[np.ndarray]:
    """Generate an organic hand-drawn radical check + overbar wrapping `(x_left..x_right, y_top..y_bot)`."""
    h = max(y_bot - y_top, 1.0)
    y_roof = y_top - float(rng.uniform(0.16, 0.30))
    y_valley = y_bot + float(rng.uniform(0.02, 0.14))
    y_hook = y_top + h * float(rng.uniform(0.45, 0.62))

    check_w = min(1.15, max(0.55, 0.32 * h))
    x_valley = x_left - float(rng.uniform(0.24, 0.42))
    x_hook = x_valley - check_w * float(rng.uniform(0.45, 0.65))
    x_peak = x_left - float(rng.uniform(0.02, 0.10))
    x_end = x_right + float(rng.uniform(0.05, 0.12))

    # 1. Small upward/downward hook -> 2. downstroke to valley -> 3. diagonal up to roof -> 4. overbar
    check = np.array([(x_hook - 0.14, y_hook + 0.10), (x_hook, y_hook),
                      (x_valley, y_valley), (x_peak, y_roof)])
    segments = []
    for start, end, n in zip(check, check[1:], (4, max(6, round(h * 3)), max(8, round(h * 4.5)))):
        t = _stroke_progress(n, rng)
        segment = start + t[:, None] * (end - start)
        segment[:, 0] += rng.normal(0.0, 0.04) * np.sin(t * math.pi)
        segments.append(segment[:-1])

    n_roof = max(10, min(64, round((x_end - x_peak) * 5.5)))
    ts = _stroke_progress(n_roof, rng)
    roof_x = x_peak + ts * (x_end - x_peak)
    roof_y = y_roof + (roof_x - x_peak) * rng.normal(0.22 * slant, 0.012)
    roof_y += rng.normal(0.0, 0.025) * np.sin(ts * math.pi)
    pts = np.concatenate([*segments, np.stack([roof_x, roof_y], axis=-1)])
    pts[:, 0] += (y_roof - pts[:, 1]) * slant
    return [pts]


# ---------------------------------------------------------------------------
# Structured LaTeX Prompt Bank (UniMER + Multi-Line + Matrices + Cases)
# ---------------------------------------------------------------------------

@dataclass
class StructuredPrompt:
    kind: str  # "single", "multiline", "matrix", "cases", "tree"
    latex: str
    cells: list[str]  # Atomic single-line sub-expressions to generate with InkForge
    meta: dict[str, Any]


_TEX_ATOM_RE = re.compile(r"\\(?:[A-Za-z]+|.)|[{}&=]", re.DOTALL)


def _extract_balanced_brace(s: str, open_idx: int) -> tuple[str, int] | None:
    """Given `s[open_idx] == '{'`, return `(inner, close_idx)` or `None` if unbalanced."""
    if open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    for tok in _TEX_ATOM_RE.finditer(s, open_idx):
        if tok[0] == "{":
            depth += 1
        elif tok[0] == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : tok.start()], tok.start()
    return None


def _find_top_level_split(s: str, ops: tuple[str, ...]) -> tuple[int, str] | None:
    """Find a top-level operator (`=` or `+`/`-`) outside `{...}` and `(...)` closest to string midpoint."""
    bdepth = 0
    pdepth = 0
    mid = len(s) * 0.5
    best: tuple[float, int, str] | None = None
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            # Skip escaped command or escaped brace
            i += 2
            while i < len(s) and s[i].isalpha():
                i += 1
            continue
        if ch == "{":
            bdepth += 1
        elif ch == "}":
            bdepth = max(0, bdepth - 1)
        elif ch in ("(", "["):
            pdepth += 1
        elif ch in (")", "]"):
            pdepth = max(0, pdepth - 1)
        elif bdepth == 0 and pdepth == 0 and ch in ops:
            # Skip unary leading sign at start or right after `=` / `(` / `,`
            prev_nonspace = s[:i].rstrip()
            if ch in ("+", "-") and (not prev_nonspace or prev_nonspace[-1] in ("=", "(", "[", ",", "+", "-")):
                i += 1
                continue
            left_s = s[:i].strip()
            right_s = s[i + 1 :].strip()
            if len(tokenize_inkforge(left_s)) >= 3 and len(tokenize_inkforge(right_s)) >= 3:
                dist = abs(i - mid)
                if best is None or dist < best[0]:
                    best = (dist, i, ch)
        i += 1
    if best is None:
        return None
    return best[1], best[2]


def _decompose_big_frac_tree(
    latex: str,
    tokenizer: InkForgeTokenizer,
    cells: list[str],
    *,
    depth: int = 0,
    max_depth: int = 3,
) -> dict[str, Any] | None:
    """Recursively decompose long equations, big `\\frac`, `\\sqrt`, and `\\binom` into an `InkNode` tree.

    Decomposes:
      1. Long equations (`>= 18` tokens) at top-level `=` so multi-part formulas never hit the 512-step cap.
      2. Big/nested top-level `\\frac{NUM}{DEN}`.
      3. Compound/long top-level `\\sqrt{BODY}`.
      4. Top-level `\\binom{N}{K}` into tall-parenthesis stacked binomial nodes.
      5. Very long expressions (`>= 22` tokens) at top-level `+`/`-`.
    """
    s = latex.strip()
    if not s:
        return None
    toks = tokenize_inkforge(s)
    has_frac = r"\frac" in s
    has_sqrt = r"\sqrt" in s
    has_binom = r"\binom" in s

    if toks and toks[0] in ("^", "_"):
        return None

    if depth >= max_depth:
        if _valid_unimer_cell(s, tokenizer, 44):
            idx = len(cells)
            cells.append(s)
            return {"op": "cell", "i": idx}
        return None

    # 1. Split long equations (`>= 18` tokens) at top-level `=` so neither side hits 512 steps!
    if len(toks) >= 18 and "=" in s:
        eq_split = _find_top_level_split(s, ("=",))
        if eq_split is not None:
            checkpoint = len(cells)
            pos, _ = eq_split
            left_s = s[:pos].strip()
            right_s = s[pos + 1 :].strip()
            # Keep short prefix like `x =` or `f(x) =` attached, otherwise emit explicit `=` cell
            if len(tokenize_inkforge(left_s)) <= 5 and r"\frac" not in left_s and r"\int" not in left_s:
                l_node = _decompose_big_frac_tree(left_s + " =", tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                r_node = _decompose_big_frac_tree(right_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if l_node is not None and r_node is not None:
                    return {"op": "hcat", "items": [l_node, r_node]}
            else:
                l_node = _decompose_big_frac_tree(left_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                r_node = _decompose_big_frac_tree("= " + right_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if l_node is not None and r_node is not None:
                    return {"op": "hcat", "items": [l_node, r_node]}

            del cells[checkpoint:]

    # 2. Scan for top-level `\frac`, `\sqrt`, or `\binom` (brace depth == 0)
    if has_frac or has_sqrt or has_binom:
        bdepth = 0
        i = 0
        found_frac: tuple[int, str, str, int] | None = None
        found_sqrt: tuple[int, str, int] | None = None
        found_binom: tuple[int, str, str, int] | None = None
        while i < len(s):
            ch = s[i]
            if ch == "{" and (i == 0 or s[i - 1] != "\\"):
                bdepth += 1
            elif ch == "}" and (i == 0 or s[i - 1] != "\\"):
                bdepth = max(0, bdepth - 1)
            elif bdepth == 0 and s.startswith(r"\frac", i) and (i + 5 < len(s) and s[i + 5] in ("{", " ")):
                j = i + 5
                while j < len(s) and s[j] == " ":
                    j += 1
                num_res = _extract_balanced_brace(s, j)
                if num_res is not None:
                    num_s, num_end = num_res
                    k = num_end + 1
                    while k < len(s) and s[k] == " ":
                        k += 1
                    den_res = _extract_balanced_brace(s, k)
                    if den_res is not None:
                        den_s, den_end = den_res
                        num_t = tokenize_inkforge(num_s)
                        den_t = tokenize_inkforge(den_s)
                        if (
                            max(len(num_t), len(den_t)) >= 6
                            or r"\frac" in num_s
                            or r"\frac" in den_s
                            or r"\sqrt" in num_s
                            or r"\sqrt" in den_s
                        ):
                            found_frac = (i, num_s.strip(), den_s.strip(), den_end + 1)
                            break
            elif bdepth == 0 and found_binom is None and s.startswith(r"\binom", i) and (i + 6 < len(s) and s[i + 6] in ("{", " ")):
                j = i + 6
                while j < len(s) and s[j] == " ":
                    j += 1
                top_res = _extract_balanced_brace(s, j)
                if top_res is not None:
                    top_s, top_end = top_res
                    k = top_end + 1
                    while k < len(s) and s[k] == " ":
                        k += 1
                    bot_res = _extract_balanced_brace(s, k)
                    if bot_res is not None:
                        bot_s, bot_end = bot_res
                        found_binom = (i, top_s.strip(), bot_s.strip(), bot_end + 1)
            elif bdepth == 0 and found_sqrt is None and s.startswith(r"\sqrt", i) and (i + 5 < len(s) and s[i + 5] in ("{", " ")):
                j = i + 5
                while j < len(s) and s[j] == " ":
                    j += 1
                rad_res = _extract_balanced_brace(s, j)
                if rad_res is not None:
                    rad_s, rad_end = rad_res
                    rad_t = tokenize_inkforge(rad_s)
                    if len(rad_t) >= 4 or (i > 0 or rad_end + 1 < len(s)):
                        found_sqrt = (i, rad_s.strip(), rad_end + 1)
            i += 1

        if found_frac is not None:
            f_start, num_s, den_s, f_end = found_frac
            prefix = s[:f_start].strip()
            suffix = s[f_end:].strip()
            items: list[dict[str, Any]] = []
            if prefix:
                p_node = _decompose_big_frac_tree(prefix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if p_node is None:
                    return None
                items.append(p_node)
            num_node = _decompose_big_frac_tree(num_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
            den_node = _decompose_big_frac_tree(den_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
            if num_node is None or den_node is None:
                return None
            items.append({"op": "frac", "num": num_node, "den": den_node, "scale": 0.94 if depth == 0 else 0.84})
            if suffix:
                s_node = _decompose_big_frac_tree(suffix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if s_node is None:
                    return None
                items.append(s_node)
            return items[0] if len(items) == 1 else {"op": "hcat", "items": items}

        if found_binom is not None:
            b_start, top_s, bot_s, b_end = found_binom
            prefix = s[:b_start].strip()
            suffix = s[b_end:].strip()
            items = []
            if prefix:
                p_node = _decompose_big_frac_tree(prefix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if p_node is None:
                    return None
                items.append(p_node)
            top_node = _decompose_big_frac_tree(top_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
            bot_node = _decompose_big_frac_tree(bot_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
            if top_node is None or bot_node is None:
                return None
            items.append({"op": "delim", "left": "(", "right": ")", "body": {"op": "binom", "top": top_node, "bot": bot_node}})
            if suffix:
                s_node = _decompose_big_frac_tree(suffix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if s_node is None:
                    return None
                items.append(s_node)
            return items[0] if len(items) == 1 else {"op": "hcat", "items": items}

        if found_sqrt is not None:
            s_start, rad_s, s_end = found_sqrt
            prefix = s[:s_start].strip()
            suffix = s[s_end:].strip()
            items = []
            if prefix:
                p_node = _decompose_big_frac_tree(prefix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if p_node is None:
                    return None
                items.append(p_node)
            rad_node = _decompose_big_frac_tree(rad_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
            if rad_node is None:
                return None
            items.append({"op": "sqrt", "body": rad_node})
            if suffix:
                s_node = _decompose_big_frac_tree(suffix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if s_node is None:
                    return None
                items.append(s_node)
            return items[0] if len(items) == 1 else {"op": "hcat", "items": items}

    # 2b. Tall/long parenthesized group containing `\frac` or >= 12 tokens -> `delim("(", ")", body)`!
    if len(toks) >= 16 and "(" in s and ")" in s and (has_frac or len(toks) >= 20):
        bdepth = 0
        pdepth = 0
        open_p = -1
        found_paren: tuple[int, str, int] | None = None
        for idx_p, ch in enumerate(s):
            if ch == "{" and (idx_p == 0 or s[idx_p - 1] != "\\"):
                bdepth += 1
            elif ch == "}" and (idx_p == 0 or s[idx_p - 1] != "\\"):
                bdepth = max(0, bdepth - 1)
            elif bdepth == 0 and ch == "(":
                if pdepth == 0:
                    open_p = idx_p
                pdepth += 1
            elif bdepth == 0 and ch == ")" and pdepth > 0:
                pdepth -= 1
                if pdepth == 0 and open_p >= 0:
                    inner_p = s[open_p + 1 : idx_p].strip()
                    if r"\frac" in inner_p or len(tokenize_inkforge(inner_p)) >= 12:
                        found_paren = (open_p, inner_p, idx_p + 1)
                        break
                    open_p = -1
        if found_paren is not None:
            p_start, inner_p, p_end = found_paren
            prefix = s[:p_start].strip()
            suffix = s[p_end:].strip()
            # Never detach a trailing exponent/subscript like `(1 + 1/x)^x` from its parenthesized base!
            if not suffix.startswith(("^", "_")) and (
                prefix or suffix or r"\frac" in inner_p or len(tokenize_inkforge(inner_p)) >= 16
            ):
                items = []
                if prefix:
                    p_node = _decompose_big_frac_tree(prefix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                    if p_node is None:
                        return None
                    items.append(p_node)
                in_node = _decompose_big_frac_tree(inner_p, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                if in_node is None:
                    return None
                items.append({"op": "delim", "left": "(", "right": ")", "body": in_node})
                if suffix:
                    s_node = _decompose_big_frac_tree(suffix, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
                    if s_node is None:
                        return None
                    items.append(s_node)
                return items[0] if len(items) == 1 else {"op": "hcat", "items": items}

    # 3. Very long expression (`>= 20` tokens) -> split cleanly at top-level `+` / `-`!
    if len(toks) >= 20:
        pm_split = _find_top_level_split(s, ("+", "-"))
        if pm_split is not None:
            checkpoint = len(cells)
            pos, op_ch = pm_split
            left_s = s[:pos].strip()
            right_s = s[pos + 1 :].strip()
            l_node = _decompose_big_frac_tree(left_s, tokenizer, cells, depth=depth + 1, max_depth=max_depth)
            r_node = _decompose_big_frac_tree(f"{op_ch} {right_s}", tokenizer, cells, depth=depth + 1, max_depth=max_depth)
            if l_node is not None and r_node is not None:
                return {"op": "hcat", "items": [l_node, r_node]}
            del cells[checkpoint:]

    if _valid_unimer_cell(s, tokenizer, 56):
        idx = len(cells)
        cells.append(s)
        return {"op": "cell", "i": idx}
    return None


_UNIMER_MATHRM_RE = re.compile(r"\\mathrm\s*(?=\{)")
_UNIMER_TRAILING_OPERATOR_RE = re.compile(
    r"(?:[=+_^\-]|\\(?:times|cdot|div|pm|mp|leq|geq|neq|in|to|rightarrow))$"
)
_ENV_ARRAY_RE = re.compile(r"\\begin\{array\}\{([clr]+)\}(.*?)\\end\{array\}", re.DOTALL)


def _split_tex_fields(s: str, separator: str) -> list[str]:
    """Split rows or columns at the outer brace level, respecting escaped separators."""
    depth, start = 0, 0
    fields = []
    for tok in _TEX_ATOM_RE.finditer(s):
        if tok[0] == "{":
            depth += 1
        elif tok[0] == "}":
            depth -= 1
        elif tok[0] == separator and depth == 0:
            fields.append(s[start : tok.start()].strip())
            start = tok.end()
    fields.append(s[start:].strip())
    return fields


def _unwrap_tex_group(s: str) -> str:
    group = _extract_balanced_brace(s, 0)
    while group is not None and group[1] == len(s) - 1:
        s = group[0].strip()
        group = _extract_balanced_brace(s, 0)
    return s


def _normalize_unimer_array(s: str) -> str:
    """Recover the layout of simple UniMER arrays, including one-row wrappers."""
    match = _ENV_ARRAY_RE.search(s)
    if match is None or r"\begin" in match[2]:
        return s
    prefix, suffix = s[:match.start()].rstrip(), s[match.end():].lstrip()
    wrappers = ((r"\{", ".", "cases"), ("[", "]", "bmatrix"),
                ("(", ")", "pmatrix"), ("|", "|", "vmatrix"))
    env = ""
    for left, right, kind in wrappers:
        left_match = re.search(r"\\left\s*" + re.escape(left) + r"$", prefix)
        right_match = re.match(r"\\right\s*" + re.escape(right), suffix)
        if left_match and right_match:
            env = kind
            prefix, suffix = prefix[:left_match.start()].rstrip(), suffix[right_match.end():].lstrip()
            break

    rows = [[_unwrap_tex_group(c) for c in _split_tex_fields(row, "&")]
            for row in _split_tex_fields(match[2], r"\\") if row]
    if not rows:
        return s
    if not env and len(rows) == 1:
        return " ".join((prefix, *rows[0], suffix)).strip()
    if not env:
        env = "matrix" if len(match[1]) > 1 and set(match[1]) == {"c"} else "aligned"
    body = r" \\ ".join(" & ".join(row) for row in rows)
    return f"{prefix} \\begin{{{env}}}{body}\\end{{{env}}} {suffix}".strip()


def _clean_unimer_latex(latex: str) -> str | None:
    if not isinstance(latex, str):
        return None
    s = latex.strip()
    if not s or len(s) > 260 or re.search(r"\\mathfrak(?![A-Za-z])", s):
        return None
    depth = 0
    for tok in _TEX_ATOM_RE.finditer(s):
        depth += int(tok[0] == "{") - int(tok[0] == "}")
        if depth < 0:
            return None
    if depth:
        return None

    rewritten = []
    end = 0
    for match in _UNIMER_MATHRM_RE.finditer(s):
        if match.start() < end:
            continue
        body, close = _extract_balanced_brace(s, match.end())
        if not body.strip() or "~" in body or body.lstrip().startswith(("^", "_")):
            return None
        if re.fullmatch(r"[A-Za-z]+", body.strip()):
            letters = body.strip()
            replacement = f" {letters} " if len(letters) == 1 else rf"\text{{{letters}}}"
            rewritten.extend((s[end:match.start()], replacement))
            end = close + 1
    s = "".join((*rewritten, s[end:]))
    s = _unwrap_tex_group(_normalize_unimer_array(s).strip())
    # A terminal sentence mark can be removed without affecting decimal points or \right.
    if s.endswith(("..", ",,")):
        return None
    if s.endswith((".", ",")) and not re.search(r"\\right\s*\.$", s):
        s = _unwrap_tex_group(s[:-1].rstrip())
    tokens = tokenize_inkforge(s)
    if not tokens or tokens[0] in ("^", "_"):
        return None
    if _UNIMER_TRAILING_OPERATOR_RE.search(s):
        return None
    return s


def _valid_unimer_cell(s: str, tokenizer: InkForgeTokenizer, max_tokens: int) -> bool:
    tokens = tokenize_inkforge(s)
    return (
        1 <= len(tokens) <= max_tokens
        and tokens[0] not in ("_", "^")
        and not any(t in tokens for t in (r"\begin", r"\end", "&", r"\\"))
        and not tokenizer.has_unk(s)
    )


_ENV_MATRIX_RE = re.compile(
    r"^\\begin\{(bmatrix|pmatrix|vmatrix|matrix)\}(.*?)\\end\{\1\}$",
    re.DOTALL,
)
_ENV_CASES_RE = re.compile(
    r"^\\begin\{cases\}(.*?)\\end\{cases\}$",
    re.DOTALL,
)
_ENV_ALIGN_RE = re.compile(
    r"^\\begin\{(aligned|align\*?|gather\*?)\}(.*?)\\end\{\1\}$",
    re.DOTALL,
)


_INLINE_MATRIX_RE = re.compile(
    r"\\begin\{(bmatrix|pmatrix|vmatrix|matrix)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)


def _parse_matrix_equation(s: str, tokenizer: InkForgeTokenizer) -> StructuredPrompt | None:
    """Parse multi-matrix equations like `A [a b \\ c d] [x \\ y] = [ax+by \\ cx+dy]` into a 2D tree."""
    matches = list(_INLINE_MATRIX_RE.finditer(s))
    if not matches:
        return None
    cells: list[str] = []
    items: list[dict[str, Any]] = []
    pos = 0
    for m in matches:
        gap = s[pos : m.start()].strip()
        if gap:
            if not _valid_unimer_cell(gap, tokenizer, 24):
                return None
            idx = len(cells)
            cells.append(gap)
            items.append({"op": "cell", "i": idx})
        env, body = m.group(1), m.group(2).strip()
        rows_raw = [r for r in _split_tex_fields(body, r"\\") if r]
        if not (2 <= len(rows_raw) <= 4):
            return None
        grid_nodes: list[list[dict[str, Any]]] = []
        for r in rows_raw:
            cols = _split_tex_fields(r, "&")
            if not (1 <= len(cols) <= 4):
                return None
            row_n: list[dict[str, Any]] = []
            for c in cols:
                if not _valid_unimer_cell(c, tokenizer, 24):
                    return None
                idx = len(cells)
                cells.append(c)
                row_n.append({"op": "cell", "i": idx})
            grid_nodes.append(row_n)
        items.append({"op": "matrix", "env": env, "grid": grid_nodes})
        pos = m.end()
    tail = s[pos:].strip()
    if tail:
        if not _valid_unimer_cell(tail, tokenizer, 24):
            return None
        idx = len(cells)
        cells.append(tail)
        items.append({"op": "cell", "i": idx})
    tree = items[0] if len(items) == 1 else {"op": "hcat", "items": items, "gap": 0.42}
    return StructuredPrompt(kind="tree", latex=s, cells=cells, meta={"tree": tree, "subkind": "matrix_eq"})


def parse_unimer_prompt(latex: str, tokenizer: InkForgeTokenizer) -> StructuredPrompt | None:
    """Parse a clean UniMER formula into either a single-line or structured matrix/cases/multiline prompt."""
    s = _clean_unimer_latex(latex)
    if s is None or len(s) > 220:
        return None

    # 1. Pure single Matrix environment
    m = _ENV_MATRIX_RE.match(s)
    if m and r"\end{" not in m.group(2):
        env, body = m.group(1), m.group(2).strip()
        rows_raw = [r for r in _split_tex_fields(body, r"\\") if r]
        if not (2 <= len(rows_raw) <= 4):
            return None
        grid: list[list[str]] = []
        cells: list[str] = []
        for r in rows_raw:
            cols = _split_tex_fields(r, "&")
            if not (1 <= len(cols) <= 4):
                return None
            for c in cols:
                if not _valid_unimer_cell(c, tokenizer, 24):
                    return None
                cells.append(c)
            grid.append(cols)
        n_cols = max(len(r) for r in grid)
        return StructuredPrompt(
            kind="matrix",
            latex=s,
            cells=cells,
            meta={"env": env, "rows": len(grid), "cols": n_cols, "row_lens": [len(r) for r in grid]},
        )

    # 2. Pure Cases environment (optionally prefixed with `f(x) = `)
    if r"\begin{cases}" in s and s.endswith(r"\end{cases}"):
        idx = s.index(r"\begin{cases}")
        prefix = s[:idx].strip()
        m_c = _ENV_CASES_RE.match(s[idx:])
        if m_c:
            body = m_c.group(1).strip()
            rows_raw = [r for r in _split_tex_fields(body, r"\\") if r]
            if 2 <= len(rows_raw) <= 4:
                cells = []
                if prefix:
                    if not _valid_unimer_cell(prefix, tokenizer, 24):
                        return None
                    cells.append(prefix)
                row_pairs: list[tuple[int, int | None]] = []
                ok = True
                for r in rows_raw:
                    parts = _split_tex_fields(r, "&")
                    if not (1 <= len(parts) <= 2) or not _valid_unimer_cell(parts[0], tokenizer, 44):
                        ok = False
                        break
                    v_idx = len(cells)
                    cells.append(parts[0])
                    c_idx = None
                    if len(parts) == 2 and parts[1]:
                        if not _valid_unimer_cell(parts[1], tokenizer, 44):
                            ok = False
                            break
                        c_idx = len(cells)
                        cells.append(parts[1])
                    row_pairs.append((v_idx, c_idx))
                if ok:
                    return StructuredPrompt(
                        kind="cases",
                        latex=s,
                        cells=cells,
                        meta={"has_prefix": bool(prefix), "row_pairs": row_pairs},
                    )

    # 3. Aligned / Multi-line environment
    m_a = _ENV_ALIGN_RE.match(s)
    env, body = (m_a[1], m_a[2]) if m_a else ("flat", s)
    rows = _split_tex_fields(body, r"\\")
    if 2 <= len(rows) <= 6:
        if rows[-1] == "":
            rows.pop()
        lines = [" ".join(_split_tex_fields(row, "&")).strip() for row in rows]
        if 2 <= len(lines) <= 5 and all(_valid_unimer_cell(line, tokenizer, 50) for line in lines):
            return StructuredPrompt(kind="multiline", latex=s, cells=lines, meta={"env": env})

    # 3b. Multi-matrix equation (e.g. `A [..] [..] = [..]`)
    if _INLINE_MATRIX_RE.search(s) and r"\begin{cases}" not in s and r"\begin{aligned}" not in s:
        meq = _parse_matrix_equation(s, tokenizer)
        if meq is not None:
            return meq

    # 4. Standard single-line formula (or auto-decomposed long equation / big frac / sqrt / binom!)
    if r"\begin" in s or r"\end" in s or "&" in s or r"\\" in s:
        return None
    toks = tokenize_inkforge(s)
    if not (4 <= len(toks) <= 120) or tokenizer.has_unk(s):
        return None
    if len(toks) >= 18 or r"\binom" in s or (r"\frac" in s and len(toks) >= 16) or (r"\sqrt" in s and len(toks) >= 12):
        cells: list[str] = []
        tree = _decompose_big_frac_tree(s, tokenizer, cells, depth=0, max_depth=3)
        if tree is not None and len(cells) >= 2:
            return StructuredPrompt(kind="tree", latex=s, cells=cells, meta={"tree": tree, "subkind": "long_tree"})
    if len(toks) > 80:
        return None
    return StructuredPrompt(kind="single", latex=s, cells=[s], meta={})


def _tree_cell(cells: list[str], expr: str) -> dict[str, Any]:
    idx = len(cells)
    cells.append(expr)
    return {"op": "cell", "i": idx}


@dataclass(frozen=True)
class _MathExpr:
    latex: str
    op: str = "cell"
    children: tuple[_MathExpr, ...] = ()
    delimiters: tuple[str, str] = ("(", ")")

    def emit(self, cells: list[str]) -> dict[str, Any]:
        if self.op == "cell":
            return _tree_cell(cells, self.latex)
        nodes = [child.emit(cells) for child in self.children]
        if self.op == "hcat":
            return {"op": "hcat", "items": nodes}
        if self.op == "frac":
            return {"op": "frac", "num": nodes[0], "den": nodes[1]}
        if self.op == "sqrt":
            return {"op": "sqrt", "body": nodes[0]}
        if self.op == "binom":
            return {"op": "delim", "left": "(", "right": ")",
                    "body": {"op": "binom", "top": nodes[0], "bot": nodes[1]}}
        left, right = self.delimiters
        node = {"op": "delim", "left": left, "right": right, "body": nodes[0]}
        if len(nodes) == 2:
            node["sup"] = nodes[1]
        return node


def _math_join(*parts: _MathExpr | str) -> _MathExpr:
    children = tuple(part if isinstance(part, _MathExpr) else _MathExpr(part) for part in parts)
    latex = " ".join(child.latex for child in children)
    # Keep linear expressions in one rollout so the recognizer can see their context.
    if all(child.op == "cell" for child in children):
        return _MathExpr(latex)
    return _MathExpr(latex, "hcat", children)


def _math_frac(num: _MathExpr, den: _MathExpr) -> _MathExpr:
    return _MathExpr(rf"\frac{{{num.latex}}}{{{den.latex}}}", "frac", (num, den))


def _math_sqrt(body: _MathExpr) -> _MathExpr:
    return _MathExpr(rf"\sqrt{{{body.latex}}}", "sqrt", (body,))


def _random_math_expr(rng: random.Random, variables: tuple[str, ...], depth: int) -> _MathExpr:
    v = rng.choice(variables)
    if depth <= 0:
        return _MathExpr(rng.choice((v, f"{v}_{{{rng.randint(1, 9)}}}", str(rng.randint(1, 19)))))

    op = rng.choice(("sum", "product", "power", "trig", "log", "absolute", "frac", "sqrt"))
    if op == "sum":
        a = _random_math_expr(rng, variables, depth - 1)
        b = _random_math_expr(rng, variables, depth - 1)
        return _math_join(a, rng.choice(("+", "-")), b)
    if op == "product":
        return _MathExpr(rf"{rng.randint(2, 19)} {v}^{{{rng.randint(2, 8)}}}")
    if op == "power":
        return _MathExpr(rf"({v} {rng.choice(('+', '-'))} {rng.randint(1, 12)})^{{{rng.randint(2, 8)}}}")
    if op == "trig":
        function = rng.choice((r"\sin", r"\cos", r"\tan", r"\sinh", r"\cosh"))
        return _MathExpr(rf"{function}({rng.randint(1, 9)} {v})")
    if op == "log":
        function = rng.choice((r"\ln", r"\log", r"\exp"))
        return _MathExpr(rf"{function}(1 + {v}^{{{2 * rng.randint(1, 3)}}})")
    if op == "absolute":
        return _MathExpr(rf"|{v} - {rng.randint(1, 13)}|")
    if op == "frac":
        num = _random_math_expr(rng, variables, depth - 1)
        den = _math_join(_MathExpr(rf"{v}^{{{2 * rng.randint(1, 3)}}}"), "+", str(rng.randint(1, 12)))
        return _math_frac(num, den)
    return _math_sqrt(_MathExpr(rf"{v}^{{{2 * rng.randint(1, 3)}}} + {rng.randint(1, 12)}"))


_DERIVATION_FAMILIES = (
    "linear_isolation", "quadratic_expand", "power_rule", "chain_rule", "integral_eval",
    "trig_identity", "log_power", "exponential_ode", "driven_ode", "heat_equation",
    "wave_equation", "joint_probability", "complement", "geometric_series", "telescoping_product",
    "exponential_limit", "absolute_inequality", "convexity",
)


def _sample_derivation(rng: random.Random, variables: tuple[str, ...]) -> StructuredPrompt:
    mode = rng.choice(_DERIVATION_FAMILIES)
    v, w, g = variables
    a, b, c = rng.sample(range(2, 16), 3)
    name = rng.choice(("f", "g", "h", "F", "P"))
    f = f"{name}({v})"
    t = rng.choice([time for time in ("t", "s", r"\tau") if time not in (v, w)])
    k = rng.choice(("k", "j", "i", "m"))
    n = rng.choice(("n", "N", "m"))
    if n == k:
        n = "N"

    if mode == "linear_isolation":
        op = rng.choice(("+", "-"))
        rhs = a*c + b if op == "+" else a*c - b
        lines = [rf"{a}{v} {op} {b} = {rhs}", rf"{a}{v} = {a*c}", rf"{v} = {c}"]
    elif mode == "quadratic_expand":
        op = rng.choice(("+", "-"))
        lines = [rf"({v} {op} {a})^2 = {b}", rf"{v}^2 {op} {2*a}{v} + {a*a} = {b}"]
    elif mode == "power_rule":
        lines = [rf"{f} = {a}{v}^{{{b}}} {rng.choice(('+', '-'))} {c}",
                 rf"{name}'({v}) = {a*b}{v}^{{{b-1}}}"]
    elif mode == "chain_rule":
        fn, derivative = rng.choice(((r"\sin", r"\cos"), (r"\cos", r"-\sin"), (r"\exp", r"\exp")))
        argument = rf"{a}{v}^{{{b}}} + {c}"
        lines = [rf"{f} = {fn}({argument})",
                 rf"{name}'({v}) = {derivative}({argument}) \cdot {a*b}{v}^{{{b-1}}}"]
    elif mode == "integral_eval":
        lower, upper = sorted(rng.sample(range(0, 10), 2))
        power = rng.randint(1, 4)
        symbol = rng.choice(("I", "A", "S", r"\Phi"))
        lines = [rf"{symbol} = \int_{{{lower}}}^{{{upper}}} {a}{v}^{{{power}}} d{v}",
                 rf"{symbol} = \frac{{{a}}}{{{power+1}}}{v}^{{{power+1}}}|_{{{lower}}}^{{{upper}}}",
                 rf"{symbol} = \frac{{{a*(upper**(power+1)-lower**(power+1))}}}{{{power+1}}}"]
    elif mode == "trig_identity":
        argument = rf"({a} {g} + {b})"
        fn, other = rng.choice(((r"\sin", r"\cos"), (r"\cos", r"\sin")))
        lines = [rf"{fn}^2{argument} + {other}^2{argument} = 1",
                 rf"{a}{fn}^2{argument} = {a}(1 - {other}^2{argument})"]
    elif mode == "log_power":
        use_ln = rng.random() < 0.5
        fn = r"\ln" if use_ln else rf"\log_{{{b}}}"
        den = rf"1+{v}^2" if use_ln else rf"(1+{v}^2)\ln {b}"
        lines = [rf"{f} = {fn}((1+{v}^2)^{{{a}}})",
                 rf"{f} = {a}{fn}(1+{v}^2)", rf"{name}'({v}) = \frac{{{2*a}{v}}}{{{den}}}"]
    elif mode == "exponential_ode":
        lines = [rf"{w}'({t}) = {g} {w}({t})", rf"{w}(0) = {a}",
                 rf"{w}({t}) = {a} e^{{{g} {t}}}"]
    elif mode == "driven_ode":
        trig, derivative = rng.choice(((r"\sin", r"\cos"), (r"\cos", r"-\sin")))
        slope = rf"{a}" if derivative[0] != "-" else f"-{a}"
        lines = [rf"{w}'({t}) = {slope} {derivative.lstrip('-')}({b}{t})",
                 rf"{w}({t}) = \frac{{{a}}}{{{b}}}{trig}({b}{t}) + {c}"]
    elif mode in ("heat_equation", "wave_equation"):
        fn = rng.choice((r"\sin", r"\cos"))
        space = rf"{fn}({b}{v})"
        coefficient = rf"{a}" if rng.random() < 0.5 else g
        if mode == "heat_equation":
            lines = [rf"{w}({v},{t}) = {space} e^{{-{coefficient} {b*b}{t}}}",
                     rf"\frac{{\partial {w}}}{{\partial {t}}} = {coefficient} \frac{{\partial^2 {w}}}{{\partial {v}^2}}"]
        else:
            lines = [rf"{w}({v},{t}) = {fn}({b}({v}-{coefficient} {t}))",
                     rf"\frac{{\partial^2 {w}}}{{\partial {t}^2}} = {{{coefficient}}}^2 \frac{{\partial^2 {w}}}{{\partial {v}^2}}"]
    elif mode in ("joint_probability", "complement"):
        u, z = rng.sample(("A", "B", "C", "D", "E", "F", "G", "H"), 2)
        probability = rng.choice((r"\Pr", "P"))
        if mode == "joint_probability":
            lines = [rf"{probability}({u}\cap {z}) = {probability}({u}|{z}){probability}({z})",
                     rf"{probability}({u}|{z}) = \frac{{{probability}({u}\cap {z})}}{{{probability}({z})}}"]
        else:
            lines = [rf"{probability}({u}) = \frac{{{a}}}{{{a+b}}}",
                     rf"{probability}({u}^c) = 1-{probability}({u}) = \frac{{{b}}}{{{a+b}}}"]
    elif mode == "geometric_series":
        ratio = rng.choice((w, g))
        lines = [rf"S_{{{n}}} = \sum_{{{k}=0}}^{{{n}-1}} {ratio}^{{{k}}}",
                 rf"(1-{ratio})S_{{{n}}} = 1-{ratio}^{{{n}}}",
                 rf"S_{{{n}}} = \frac{{1-{ratio}^{{{n}}}}}{{1-{ratio}}}"]
    elif mode == "telescoping_product":
        lines = [rf"P_{{{n}}} = \prod_{{{k}={a}}}^{{{n}}} \frac{{{k}+1}}{{{k}}}",
                 rf"P_{{{n}}} = \frac{{{a+1}}}{{{a}}}\frac{{{a+2}}}{{{a+1}}}\cdots\frac{{{n}+1}}{{{n}}}",
                 rf"P_{{{n}}} = \frac{{{n}+1}}{{{a}}}"]
    elif mode == "exponential_limit":
        lines = [rf"S_{{{n}}} = (1+\frac{{{a}}}{{{n}}})^{{{b}{n}}}",
                 rf"\lim_{{{n}\rightarrow\infty}} S_{{{n}}} = e^{{{a*b}}}"]
    elif mode == "absolute_inequality":
        lo, hi = rng.choice((("<", "<"), (r"\leq", r"\leq")))
        lines = [rf"|{a}{v} - {b}| {hi} {c}",
                 rf"{b-c} {lo} {a}{v} {hi} {b+c}",
                 rf"\frac{{{b-c}}}{{{a}}} {lo} {v} {hi} \frac{{{b+c}}}{{{a}}}"]
    else:
        p = 2 * rng.randint(1, 4)
        lines = [rf"{f} = {a}{v}^{{{p}}} + {b}{v} + {c}",
                 rf"{name}''({v}) = {a*p*(p-1)}{v}^{{{p-2}}} \geq 0"]

    layout = rng.choice(("aligned", "aligned_eq", "flat", "vstack"))
    parts = [_split_tex_fields(line, "=") for line in lines]
    if layout == "aligned_eq" and all(len(part) > 1 for part in parts):
        cells: list[str] = []
        rows = [[_tree_cell(cells, lhs + " ="), _tree_cell(cells, " = ".join(rhs))]
                for lhs, *rhs in parts]
        body = r" \\ ".join(lhs + " &= " + " = ".join(rhs) for lhs, *rhs in parts)
        return StructuredPrompt("tree", rf"\begin{{aligned}}{body}\end{{aligned}}", cells,
                                {"tree": {"op": "aligned_eq", "rows": rows}, "subkind": mode})
    body = r" \\ ".join(lines)
    if layout == "vstack":
        cells = []
        nodes = [_tree_cell(cells, line) for line in lines]
        return StructuredPrompt("tree", body, cells, {"tree": {"op": "vstack", "lines": nodes}, "subkind": mode})
    if layout == "aligned":
        body = rf"\begin{{aligned}}{body}\end{{aligned}}"
    return StructuredPrompt("multiline", body, lines, {"env": "aligned" if layout == "aligned" else "flat", "subkind": mode})


def _sample_expression_tree(rng: random.Random, variables: tuple[str, ...]) -> StructuredPrompt:
    mode = rng.choice(("big_frac", "nested_frac", "continued_frac", "big_sqrt", "tall_delim",
                       "binomial_expansion", "definite_integral", "bounded_sum", "bounded_product", "function_limit"))
    v, w, g = variables
    a, b, c = rng.sample(range(2, 15), 3)
    name = rng.choice(("F", "G", "L", "R", "S", r"\Phi", r"\Psi"))
    expression = _random_math_expr(rng, variables, rng.randint(1, 3))
    second = _random_math_expr(rng, variables, rng.randint(1, 2))
    if mode == "big_frac":
        expr = _math_frac(expression, _math_join(second, "+", f"{a}{v}^2"))
    elif mode == "nested_frac":
        expr = _math_frac(_math_join(f"{v}^2 +", _math_frac(expression, _MathExpr(str(a)))),
                          _math_frac(_MathExpr(rf"{w}^2 + {b}"), second))
    elif mode == "continued_frac":
        expr = _MathExpr(str(c))
        for _ in range(rng.randint(2, 3)):
            expr = _math_join(str(rng.randint(1, 9)), "+",
                              _math_frac(_MathExpr(rng.choice(("1", str(a), v))), expr))
    elif mode == "big_sqrt":
        expr = _math_sqrt(_math_join(f"{v}^2 +", _math_frac(expression, _MathExpr(rf"{w}^2 + {a}"))))
    elif mode == "tall_delim":
        left, right = rng.choice((("(", ")"), ("[", "]"), ("|", "|")))
        inner = _math_frac(expression, second)
        power = str(rng.randint(2, 6)) if left != "|" and rng.random() < 0.65 else ""
        exponent = rf"^{{{power}}}" if power else ""
        children = (inner, _MathExpr(power)) if power else (inner,)
        expr = _MathExpr(rf"\left{left}{inner.latex}\right{right}{exponent}", "delim", children, (left, right))
    elif mode == "binomial_expansion":
        index = rng.choice(("i", "j", "k"))
        upper = rng.choice(("n", "m", str(a)))
        binom = _MathExpr(rf"\binom{{{upper}}}{{{index}}}", "binom", (_MathExpr(upper), _MathExpr(index)))
        expr = _math_join(rf"({v}+{w})^{{{upper}}} = \sum_{{{index}=0}}^{{{upper}}}",
                          binom, rf"{v}^{{{index}}}{w}^{{{upper}-{index}}}")
    elif mode == "definite_integral":
        lower = rng.choice(("0", str(-a), g))
        upper = rng.choice((str(b), r"\pi", r"\infty"))
        expr = _math_join(rf"\int_{{{lower}}}^{{{upper}}}",
                          _math_frac(_MathExpr(rf"\cos({a}{v})"), _MathExpr(rf"{v}^2 + {c}")),
                          rf"d{v}")
    elif mode in ("bounded_sum", "bounded_product"):
        name_op = r"\sum" if mode == "bounded_sum" else r"\prod"
        index = rng.choice(("i", "j", "k"))
        upper = rng.choice(("n", "m", str(b)))
        expr = _math_join(rf"{name_op}_{{{index}={rng.randint(0, 3)}}}^{{{upper}}}",
                          _math_frac(_MathExpr(rf"{index}^2 + {a}{v}"), _MathExpr(rf"{index}^2 + {c}")))
    else:
        upper = rng.choice(("0", str(a), r"\infty"))
        expr = _math_join(rf"\lim_{{{v}\rightarrow {upper}}}",
                          _math_frac(_MathExpr(rf"{v}^2 + {b}{v} + {a}"), _MathExpr(rf"{v}^2 + {c}")))
    if mode != "binomial_expansion" and rng.random() < 0.78:
        expr = _math_join(rng.choice((name, f"{name}({v})", w)) + " =", expr)
    cells: list[str] = []
    tree = expr.emit(cells)
    return StructuredPrompt("tree", expr.latex, cells, {"tree": tree, "subkind": mode})


def _sample_matrix_prompt(rng: random.Random, variables: tuple[str, ...]) -> StructuredPrompt:
    mode = rng.choice(("matrix", "determinant", "matrix_vector", "matrix_product", "eigenvalues",
                       "augmented", "block_matrix"))
    env = rng.choice(("bmatrix", "pmatrix"))
    size = rng.choice((2, 3))
    a, b, c = rng.sample(("A", "B", "C", "D", "H", "M", "Q", "R", "S"), 3)
    cells: list[str] = []

    def entry() -> _MathExpr:
        if mode in ("matrix_vector", "matrix_product") or rng.random() < 0.68:
            return _MathExpr(rng.choice((str(rng.randint(-9, 14)), rng.choice(variables))))
        return _random_math_expr(rng, variables, 1)

    def matrix(grid: list[list[_MathExpr]], kind: str = env, divider: int | None = None):
        body = r" \\ ".join(" & ".join(e.latex for e in row) for row in grid)
        if divider is None:
            latex = rf"\begin{{{kind}}}{body}\end{{{kind}}}"
        else:
            spec = "c" * divider + "|" + "c" * (len(grid[0]) - divider)
            latex = rf"\left[\begin{{array}}{{{spec}}}{body}\end{{array}}\right]"
        node = {"op": "matrix", "env": kind, "grid": [[e.emit(cells) for e in row] for row in grid]}
        if divider is not None:
            node["column_divider"] = divider
        return latex, node

    cols = size if mode != "matrix" else rng.choice((2, 3))
    grid = [[entry() for _ in range(cols)] for _ in range(size)]
    items: list[tuple[str, dict[str, Any]]] = []

    def add(text: str) -> None:
        items.append((text, _tree_cell(cells, text)))

    if mode == "matrix":
        if rng.random() < 0.64:
            add(f"{a} =")
        items.append(matrix(grid))
    elif mode == "determinant":
        if rng.random() < 0.55:
            add(rf"\det {a} =")
        items.append(matrix(grid, "vmatrix"))
    elif mode == "matrix_vector":
        vector = [_MathExpr(x) for x in variables[:size]]
        items.extend((matrix(grid), matrix([[x] for x in vector])))
        add("=")
        result = [[_MathExpr(" + ".join(rf"({e.latex}) {v.latex}" for e, v in zip(row, vector)))] for row in grid]
        items.append(matrix(result))
    elif mode == "matrix_product":
        if rng.random() < 0.70:
            add(f"{a}{b} =")
        width = rng.choice((2, 3))
        items.extend((matrix(grid), matrix([[entry() for _ in range(width)] for _ in range(size)])))
    elif mode == "eigenvalues":
        lam = rng.choice((r"\lambda", r"\mu", r"\theta"))
        v = rng.choice(("v", "w", "u"))
        if rng.random() < 0.50:
            add(rf"({a} - {lam} I) {v} = 0")
            add(rf"\Rightarrow \det({a}-{lam} I) = 0")
        else:
            grid = [[_math_join(e, "-", lam) if i == j else e for j, e in enumerate(row)]
                    for i, row in enumerate(grid)]
            items.append(matrix(grid, "vmatrix"))
            add("= 0")
    elif mode == "augmented":
        add(f"{a} =")
        items.append(matrix([row + [entry()] for row in grid], "bmatrix", divider=size))
    else:
        m, n = rng.sample(("m", "n", "p", "q", str(size), str(size + 1)), 2)
        zero = _MathExpr(rng.choice(("0", rf"0_{{{m}\times {n}}}")))
        block_a = _MathExpr(rng.choice((a, rf"{a}^T", rf"{a}_{{{m}}}")))
        block_b = _MathExpr(rng.choice((b, rf"{b}^T", rf"{b}_{{{n}}}")))
        identity = _MathExpr(rng.choice((rf"I_{{{m}}}", rf"I_{{{n}}}", r"\mathbb{I}")))
        add(f"{c} =")
        placement = rng.choice((
            ((block_a, block_b), (zero, identity)),
            ((block_a, zero), (zero, block_b)),
            ((block_a, zero), (block_b, identity)),
        ))
        items.append(matrix([list(row) for row in placement]))

    label = " ".join(latex for latex, _ in items)
    nodes = [node for _, node in items]
    if len(nodes) == 1 and nodes[0]["op"] == "matrix":
        tree = nodes[0]
    else:
        tree = {"op": "hcat", "items": nodes, "gap": 0.44}
    return StructuredPrompt("tree", label, cells, {"tree": tree, "subkind": mode})


def _sample_cases_prompt(rng: random.Random, variables: tuple[str, ...]) -> StructuredPrompt:
    v, _, g = variables
    name = rng.choice(("f", "g", "h", "F", "p", "q"))
    count = rng.randint(2, 4)
    cut = rng.randint(-12, 12)
    upper = cut + rng.randint(1, 12)
    word = r"\text{otherwise}"
    membership = rng.choice((rf"{v} \in [{cut},{upper}]", rf"{v} \in ({cut},{upper})"))
    if count == 2:
        conditions = rng.choice((
            (rf"{v} < {cut}", rf"{v} \geq {cut}"),
            (rf"{v} \leq {cut}", rf"{v} > {cut}"),
            (rf"{v} = {cut}", rf"{v} \neq {cut}"),
            (membership, word),
            (rf"{v} > {g}", word),
        ))
        if rng.random() < 0.30:
            conditions = (conditions[0], word)
    elif count == 3:
        conditions = rng.choice((
            (rf"{v} < {cut}", rf"{v} = {cut}", rf"{v} > {cut}"),
            (rf"{v} \leq {cut}", rf"{v} \geq {upper}", word),
            (rf"{v} < {cut}", rf"{v} \in [{cut},{upper}]", rf"{v} > {upper}"),
        ))
    else:
        conditions = (rf"{v} < {cut}", rf"{v} = {cut}", rf"{cut} < {v} \leq {upper}", word)
    conditions = [condition if condition == word or rng.random() < 0.45 else rf"\text{{if }} {condition}"
                  for condition in conditions]
    expressions = [_random_math_expr(rng, variables, rng.choice((1, 1, 2))).latex for _ in conditions]
    rows = list(zip(expressions, conditions))
    prefix = rf"{name}({v}) ="
    body = r" \\ ".join(f"{value} & {condition}" for value, condition in rows)
    cells = [prefix, *(item for row in rows for item in row)]
    pairs = [(1 + 2*i, 2 + 2*i) for i in range(len(rows))]
    return StructuredPrompt("cases", rf"{prefix} \begin{{cases}}{body}\end{{cases}}", cells,
                            {"has_prefix": True, "row_pairs": pairs, "subkind": f"cases_{count}"})


def _sample_variables(rng: random.Random) -> tuple[str, str, str]:
    v, w = rng.sample(("x", "y", "t", "u", "z", "r", "s", "p", "q"), 2)
    g = rng.choice((r"\alpha", r"\beta", r"\gamma", r"\theta", r"\lambda", r"\mu", r"\phi", r"\omega"))
    return v, w, g


def sample_big_fraction_or_multistep_prompt(rng: random.Random) -> StructuredPrompt:
    """Sample a laid-out expression tree or multiline derivation."""
    variables = _sample_variables(rng)
    if rng.random() < 0.25:
        return _sample_derivation(rng, variables)
    return _sample_expression_tree(rng, variables)


def sample_programmatic_structured_prompt(rng: random.Random) -> StructuredPrompt:
    """Compose a fresh STEM expression from randomized symbols, operators, and layouts."""
    sampler = rng.choices(
        (_sample_derivation, _sample_expression_tree, _sample_matrix_prompt, _sample_cases_prompt),
        weights=(33, 23, 27, 17),
    )[0]
    return sampler(rng, _sample_variables(rng))


# ---------------------------------------------------------------------------
# 2D Spatial Assembly with Human Layout Physics
# ---------------------------------------------------------------------------

def _shift_scale_strokes(
    strokes: Sequence[np.ndarray],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    scale: float = 1.0,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for s in strokes:
        arr = s[:, :2].astype(np.float64, copy=True)
        if scale != 1.0:
            arr *= scale
        arr[:, 0] += dx
        arr[:, 1] += dy
        out.append(arr)
    return out


def _assemble_ink_node(
    node: dict[str, Any],
    cell_strokes: list[list[np.ndarray]],
    rng: np.random.Generator,
    writer_slant: float,
) -> tuple[list[np.ndarray], tuple[float, float, float, float], float]:
    """Recursively assemble an `InkNode` tree into `(strokes, bbox, y_math_axis)`."""
    op = node["op"]

    if op == "cell":
        st = cell_strokes[int(node["i"])] if int(node["i"]) < len(cell_strokes) else []
        if not st:
            return [], (0.0, 0.0, 0.5, 1.0), 0.5
        x0, y0, x1, y1 = strokes_bbox(st)
        # Anchor top-left at (0, 0); math axis at vertical midpoint of ink
        shifted = _shift_scale_strokes(st, dx=-x0, dy=-y0)
        w, h = max(x1 - x0, 0.25), max(y1 - y0, 0.4)
        return shifted, (0.0, 0.0, w, h), 0.5 * h

    if op == "hcat":
        parts = [_assemble_ink_node(ch, cell_strokes, rng, writer_slant) for ch in node["items"]]
        parts = [p for p in parts if p[0]]
        if not parts:
            return [], (0.0, 0.0, 0.5, 1.0), 0.5
        gap_base = float(node.get("gap", 0.52))
        out: list[np.ndarray] = []
        x_cur = 0.0
        for st, (x0, _y0, x1, _y1), y_ax in parts:
            dx = x_cur - x0
            dy = -y_ax + float(rng.normal(0.0, 0.03))
            out.extend(_shift_scale_strokes(st, dx=dx, dy=dy))
            x_cur += (x1 - x0) + gap_base * float(rng.uniform(0.80, 1.25))
        bx0, by0, bx1, by1 = strokes_bbox(out)
        # Re-anchor top-left at (0, 0), preserving exact y_math_axis = -by0
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), -by0

    if op == "frac":
        scale = float(node.get("scale", 0.94))
        num_st, (nx0, ny0, nx1, ny1), _ = _assemble_ink_node(node["num"], cell_strokes, rng, writer_slant)
        den_st, (dx0, dy0, dx1, dy1), _ = _assemble_ink_node(node["den"], cell_strokes, rng, writer_slant)
        nw, nh = (nx1 - nx0) * scale, (ny1 - ny0) * scale
        dw = (dx1 - dx0) * scale
        bar_pad = float(rng.uniform(0.24, 0.46))
        bar_w = max(nw, dw, 0.9) + 2.0 * bar_pad
        v_gap_top = float(rng.uniform(0.28, 0.46))
        den_has_sqrt = (node["den"].get("op") == "sqrt") or any(
            ch.get("op") == "sqrt" for ch in node["den"].get("items", []) if isinstance(ch, dict)
        )
        v_gap_bot = float(rng.uniform(0.44, 0.64) if den_has_sqrt else rng.uniform(0.28, 0.46))
        y_bar = nh + v_gap_top

        # Center the numerator and denominator before choosing the stroke order.
        num_dx = 0.5 * (bar_w - nw) - nx0 * scale + float(rng.normal(0.0, 0.08))
        num_dy = -ny0 * scale
        num_ink = _shift_scale_strokes(num_st, dx=num_dx, dy=num_dy, scale=scale)
        bar_ink = draw_human_hline(0.0, bar_w, y_bar, rng, slant=writer_slant)
        den_dx = 0.5 * (bar_w - dw) - dx0 * scale + float(rng.normal(0.0, 0.08))
        den_dy = (y_bar + v_gap_bot) - dy0 * scale
        den_ink = _shift_scale_strokes(den_st, dx=den_dx, dy=den_dy, scale=scale)
        out = num_ink + (den_ink + bar_ink if rng.random() < 0.30 else bar_ink + den_ink)

        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), y_bar - by0

    if op == "binom":
        # Stacked binomial numerator over denominator (no bar; wrapped by `delim("(", ")")`)
        scale = float(node.get("scale", 0.90))
        top_st, (tx0, ty0, tx1, ty1), _ = _assemble_ink_node(node["top"], cell_strokes, rng, writer_slant)
        bot_st, (bx0_i, by0_i, bx1_i, by1_i), _ = _assemble_ink_node(node["bot"], cell_strokes, rng, writer_slant)
        tw, th = (tx1 - tx0) * scale, (ty1 - ty0) * scale
        bw_i = (bx1_i - bx0_i) * scale
        col_w = max(tw, bw_i, 0.7)
        v_gap = float(rng.uniform(0.32, 0.52))
        out = []
        out.extend(_shift_scale_strokes(top_st, dx=0.5 * (col_w - tw) - tx0 * scale, dy=-ty0 * scale, scale=scale))
        out.extend(_shift_scale_strokes(bot_st, dx=0.5 * (col_w - bw_i) - bx0_i * scale, dy=(th + v_gap) - by0_i * scale, scale=scale))
        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), 0.5 * (by1 - by0)

    if op == "matrix":
        env = str(node.get("env", "bmatrix"))
        grid_nodes: list[list[dict[str, Any]]] = node["grid"]
        n_rows = len(grid_nodes)
        n_cols = max(len(r) for r in grid_nodes)
        assembled_grid = [
            [_assemble_ink_node(ch, cell_strokes, rng, writer_slant) for ch in r]
            for r in grid_nodes
        ]
        col_w = [0.7] * n_cols
        row_h = [0.9] * n_rows
        for r in range(n_rows):
            for c, (_st, box, _ax) in enumerate(assembled_grid[r]):
                col_w[c] = max(col_w[c], box[2] - box[0])
                row_h[r] = max(row_h[r], box[3] - box[1])
        col_gap = float(rng.uniform(0.55, 0.95))
        row_gap = float(rng.uniform(0.48, 0.82))
        col_xs, xc = [], 0.0
        for cw in col_w:
            col_xs.append(xc)
            xc += cw + col_gap
        mat_w = xc - col_gap
        row_ys, yc = [], 0.0
        for rh in row_h:
            row_ys.append(yc)
            yc += rh + row_gap
        mat_h = yc - row_gap

        left_d, right_d = {"bmatrix": ("[", "]"), "pmatrix": ("(", ")"), "vmatrix": ("|", "|")}.get(env, ("", ""))
        out = []
        x_start = 0.0
        if left_d:
            out.extend(draw_human_delimiter(left_d, 0.0, 0.0, mat_h, rng, slant=writer_slant))
            x_start = float(rng.uniform(0.42, 0.65))
        for r in range(n_rows):
            for c, (st, (x0, y0, x1, y1), _ax) in enumerate(assembled_grid[r]):
                if not st:
                    continue
                cw, ch = x1 - x0, y1 - y0
                dx = x_start + col_xs[c] + 0.5 * (col_w[c] - cw) - x0 + float(rng.normal(0.0, 0.06))
                dy = row_ys[r] + 0.5 * (row_h[r] - ch) - y0 + float(rng.normal(0.0, 0.05))
                out.extend(_shift_scale_strokes(st, dx=dx, dy=dy))
        if divider := node.get("column_divider"):
            x_divider = x_start + col_xs[divider] - 0.5 * col_gap
            out.extend(draw_human_delimiter("|", x_divider, 0.0, mat_h, rng, slant=writer_slant))
        if right_d:
            closing = draw_human_delimiter(
                right_d, x_start + mat_w + float(rng.uniform(0.42, 0.65)), 0.0, mat_h, rng, slant=writer_slant
            )
            if rng.random() < 0.22:
                out[1:1] = closing
            else:
                out.extend(closing)
        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), 0.5 * (by1 - by0)

    if op == "sqrt":
        body_st, (x0, y0, x1, y1), y_ax = _assemble_ink_node(node["body"], cell_strokes, rng, writer_slant)
        if not body_st:
            return [], (0.0, 0.0, 0.5, 1.0), 0.5
        rad_st = draw_human_sqrt(x0, y0, x1, y1, rng, slant=writer_slant)
        out = body_st + rad_st if rng.random() < 0.22 else rad_st + body_st
        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), y_ax - by0

    if op == "delim":
        body_st, (x0, y0, x1, y1), y_ax = _assemble_ink_node(node["body"], cell_strokes, rng, writer_slant)
        if not body_st:
            return [], (0.0, 0.0, 0.5, 1.0), 0.5
        bw = x1 - x0
        pad_l = float(rng.uniform(0.36, 0.56))
        pad_r = float(rng.uniform(0.36, 0.56))
        left_d = draw_human_delimiter(str(node["left"]), 0.0, y0, y1, rng, slant=writer_slant)
        body_shifted = _shift_scale_strokes(body_st, dx=pad_l - x0)
        right_x = pad_l + bw + pad_r
        right_d = draw_human_delimiter(str(node["right"]), right_x, y0, y1, rng, slant=writer_slant)
        out = left_d + (right_d + body_shifted if rng.random() < 0.22 else body_shifted + right_d)
        if node.get("sup") is not None:
            sup_st, (sx0, sy0, sx1, sy1), _ = _assemble_ink_node(node["sup"], cell_strokes, rng, writer_slant)
            if sup_st:
                sup_scale = 0.72
                sup_dx = right_x + float(rng.uniform(0.24, 0.42)) - sx0 * sup_scale
                sup_dy = y0 - (sy1 - sy0) * sup_scale * 0.45 - sy0 * sup_scale
                out.extend(_shift_scale_strokes(sup_st, dx=sup_dx, dy=sup_dy, scale=sup_scale))
        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), y_ax - by0

    if op == "vstack":
        lines = [_assemble_ink_node(ln, cell_strokes, rng, writer_slant) for ln in node["lines"]]
        lines = [ln for ln in lines if ln[0]]
        if not lines:
            return [], (0.0, 0.0, 0.5, 1.0), 0.5
        out = []
        y_cur = 0.0
        for st, (x0, y0, x1, y1), _ in lines:
            lh = max(y1 - y0, 1.0)
            x_shift = -x0 + float(rng.normal(0.0, 0.24))
            y_shift = y_cur - y0
            tilt = math.radians(float(rng.normal(0.0, 1.0)))
            ct, st_sin = math.cos(tilt), math.sin(tilt)
            for s in st:
                sh = s[:, :2].copy()
                sh[:, 0] += x_shift
                sh[:, 1] += y_shift
                rx = sh[:, 0] * ct - (sh[:, 1] - y_cur) * st_sin
                ry = sh[:, 0] * st_sin + (sh[:, 1] - y_cur) * ct + y_cur
                out.append(np.stack([rx, ry], axis=-1))
            y_cur += lh + float(rng.uniform(0.65, 1.25))
        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), 0.5 * (by1 - by0)

    if op == "aligned_eq":
        # Align `=` column vertically across all lines (LHS right-aligned, RHS left-aligned!)
        parsed_rows = []
        max_lhs_w = 0.5
        for lhs_n, rhs_n in node["rows"]:
            l_st, l_box, l_ax = _assemble_ink_node(lhs_n, cell_strokes, rng, writer_slant)
            r_st, r_box, r_ax = _assemble_ink_node(rhs_n, cell_strokes, rng, writer_slant)
            lw = l_box[2] - l_box[0]
            max_lhs_w = max(max_lhs_w, lw)
            parsed_rows.append((l_st, l_box, l_ax, r_st, r_box, r_ax))
        out = []
        y_cur = 0.0
        eq_gap = float(rng.uniform(0.45, 0.72))
        for l_st, (lx0, ly0, lx1, ly1), l_ax, r_st, (rx0, ry0, rx1, ry1), r_ax in parsed_rows:
            lw = lx1 - lx0
            row_ax = max(l_ax, r_ax)
            row_bot = max((ly1 - ly0) - l_ax, (ry1 - ry0) - r_ax)
            row_h = max(row_ax + row_bot, 1.0)
            y_axis_here = y_cur + row_ax
            if l_st:
                dx_l = (max_lhs_w - lw) - lx0 + float(rng.normal(0.0, 0.08))
                dy_l = y_axis_here - l_ax
                out.extend(_shift_scale_strokes(l_st, dx=dx_l, dy=dy_l))
            if r_st:
                dx_r = (max_lhs_w + eq_gap) - rx0 + float(rng.normal(0.0, 0.08))
                dy_r = y_axis_here - r_ax
                out.extend(_shift_scale_strokes(r_st, dx=dx_r, dy=dy_r))
            y_cur += row_h + float(rng.uniform(0.60, 1.10))
        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), 0.5 * (by1 - by0)

    return [], (0.0, 0.0, 0.5, 1.0), 0.5


def _estimate_cell_x_height(latex: str, strokes: Sequence[np.ndarray]) -> float | None:
    """Measure typical upright ink, filtering short marks and horizontal operators."""
    tokens = tokenize_inkforge(latex)
    spans = [(float(np.ptp(st[:, 0])), float(np.ptp(st[:, 1]))) for st in strokes if len(st) > 1]
    heights = [h for w, h in spans if h >= 0.10 and h > 0.55 * w]
    if heights:
        if len(heights) >= 3:
            low, high = np.percentile(heights, [20, 80])
            # Keep the main body when there is a stray tall delimiter or a small script.
            heights = [h for h in heights if 0.75 * low <= h <= 1.20 * high]
        return float(np.median(heights))
    if tokens in (["="], ["-"], [r"\leq"], [r"\geq"], [r"\approx"]):
        widths = [w for w, _ in spans if w >= 0.10]
        if widths:
            return float(np.median(widths)) / 0.85
    return None


def _scale_sibling_cells(
    labels: Sequence[str], cell_strokes: list[list[np.ndarray]],
) -> list[list[np.ndarray]]:
    """Restore a shared glyph scale before any subtree applies its own layout scaling."""
    heights = [_estimate_cell_x_height(label, st) for label, st in zip(labels, cell_strokes)]
    upright_heights = [h for label, h in zip(labels, heights)
                       if h is not None and re.search(r"[a-zA-Z0-9]", label)]
    target = float(np.clip(np.median(upright_heights), 0.75, 1.10)) if upright_heights else 0.90
    return [
        _shift_scale_strokes(st, scale=float(np.clip(target / heights[i], 0.13, 3.5)))
        if i < len(heights) and heights[i] is not None else st
        for i, st in enumerate(cell_strokes)
    ]


def assemble_structured_strokes(
    prompt: StructuredPrompt,
    cell_strokes: list[list[np.ndarray]],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Assemble individually generated InkForge cells into a cohesive 2D multi-line / fraction / matrix / cases canvas."""
    if prompt.kind == "single":
        return cell_strokes[0] if cell_strokes else []

    cell_strokes = _scale_sibling_cells(prompt.cells, cell_strokes)
    writer_slant = float(rng.uniform(-0.08, 0.08))

    if prompt.kind == "tree":
        strokes, _, _ = _assemble_ink_node(prompt.meta["tree"], cell_strokes, rng, writer_slant)
        return strokes

    if prompt.kind == "multiline":
        out: list[np.ndarray] = []
        y_cursor = 0.0
        for idx, st_list in enumerate(cell_strokes):
            if not st_list:
                continue
            x0, y0, x1, y1 = strokes_bbox(st_list)
            line_h = max(y1 - y0, 1.0)
            # Natural human left-margin wander + subtle line-to-line tilt
            x_shift = -x0 + float(rng.normal(0.0, 0.22))
            y_shift = y_cursor - y0
            tilt = math.radians(float(rng.normal(0.0, 1.2)))
            ct, st_sin = math.cos(tilt), math.sin(tilt)
            for s in st_list:
                shifted = s[:, :2].copy()
                shifted[:, 0] += x_shift
                shifted[:, 1] += y_shift
                # Subtle rotation around line start
                rx = shifted[:, 0] * ct - (shifted[:, 1] - y_cursor) * st_sin
                ry = shifted[:, 0] * st_sin + (shifted[:, 1] - y_cursor) * ct + y_cursor
                out.append(np.stack([rx, ry], axis=-1))
            y_cursor += line_h + float(rng.uniform(0.55, 1.15))
        return out

    if prompt.kind == "matrix":
        has_prefix = bool(prompt.meta.get("has_prefix", False))
        offset = 1 if has_prefix else 0
        n_rows = int(prompt.meta["rows"])
        n_cols = int(prompt.meta["cols"])
        row_lens = list(prompt.meta["row_lens"])
        env = str(prompt.meta.get("env", "bmatrix"))

        grid_strokes: list[list[list[np.ndarray]]] = []
        c_idx = offset
        for rl in row_lens:
            row_items = []
            for _ in range(rl):
                row_items.append(cell_strokes[c_idx] if c_idx < len(cell_strokes) else [])
                c_idx += 1
            grid_strokes.append(row_items)

        col_widths = [1.0] * n_cols
        row_heights = [1.0] * n_rows
        cell_boxes: list[list[tuple[float, float, float, float]]] = []
        for r in range(n_rows):
            row_b = []
            for c in range(len(grid_strokes[r])):
                b = strokes_bbox(grid_strokes[r][c])
                row_b.append(b)
                col_widths[c] = max(col_widths[c], max(b[2] - b[0], 0.5))
                row_heights[r] = max(row_heights[r], max(b[3] - b[1], 0.8))
            cell_boxes.append(row_b)

        col_gap = float(rng.uniform(0.65, 1.15))
        row_gap = float(rng.uniform(0.55, 0.95))
        col_xs = []
        xc = 0.0
        for cw in col_widths:
            col_xs.append(xc)
            xc += cw + col_gap
        mat_w = xc - col_gap

        row_ys = []
        yc = 0.0
        for rh in row_heights:
            row_ys.append(yc)
            yc += rh + row_gap
        mat_h = yc - row_gap

        out = []
        x_mat_start = 0.0
        if has_prefix and cell_strokes[0]:
            px0, py0, px1, py1 = strokes_bbox(cell_strokes[0])
            pw, ph = px1 - px0, py1 - py0
            py_shift = (0.5 * mat_h) - 0.5 * (py0 + py1)
            for s in cell_strokes[0]:
                ss = s[:, :2].copy()
                ss[:, 0] -= px0
                ss[:, 1] += py_shift
                out.append(ss)
            x_mat_start = pw + float(rng.uniform(0.65, 1.15))

        left_delim, right_delim = {
            "bmatrix": ("[", "]"),
            "pmatrix": ("(", ")"),
            "vmatrix": ("|", "|"),
            "matrix": ("", ""),
        }.get(env, ("[", "]"))

        if left_delim:
            out.extend(draw_human_delimiter(left_delim, x_mat_start, 0.0, mat_h, rng, slant=writer_slant))
            x_cells_start = x_mat_start + float(rng.uniform(0.45, 0.72))
        else:
            x_cells_start = x_mat_start

        for r in range(n_rows):
            for c in range(len(grid_strokes[r])):
                st_list = grid_strokes[r][c]
                if not st_list:
                    continue
                x0, y0, x1, y1 = cell_boxes[r][c]
                cw, ch = x1 - x0, y1 - y0
                cx = x_cells_start + col_xs[c] + 0.5 * (col_widths[c] - cw) - x0 + float(rng.normal(0.0, 0.08))
                cy = row_ys[r] + 0.5 * (row_heights[r] - ch) - y0 + float(rng.normal(0.0, 0.06))
                for s in st_list:
                    ss = s[:, :2].copy()
                    ss[:, 0] += cx
                    ss[:, 1] += cy
                    out.append(ss)

        if right_delim:
            x_right = x_cells_start + mat_w + float(rng.uniform(0.45, 0.72))
            closing = draw_human_delimiter(right_delim, x_right, 0.0, mat_h, rng, slant=writer_slant)
            if rng.random() < 0.22:
                first_cell_ink = len(cell_strokes[0]) + 1 if has_prefix else 1
                out[first_cell_ink:first_cell_ink] = closing
            else:
                out.extend(closing)

        return out

    if prompt.kind == "cases":
        has_prefix = bool(prompt.meta.get("has_prefix", False))
        row_pairs: list[tuple[int, int | None]] = list(prompt.meta["row_pairs"])
        row_heights: list[float] = []
        val_widths: list[float] = []
        for vi, ci in row_pairs:
            vb = strokes_bbox(cell_strokes[vi])
            vw, vh = max(vb[2] - vb[0], 0.5), max(vb[3] - vb[1], 0.8)
            ch = 0.0
            if ci is not None and cell_strokes[ci]:
                cb = strokes_bbox(cell_strokes[ci])
                ch = max(cb[3] - cb[1], 0.8)
            row_heights.append(max(vh, ch))
            val_widths.append(vw)

        row_gap = float(rng.uniform(0.55, 0.95))
        max_val_w = max(val_widths) if val_widths else 1.5
        row_ys = []
        yc = 0.0
        for rh in row_heights:
            row_ys.append(yc)
            yc += rh + row_gap
        total_h = yc - row_gap

        out = []
        x_brace = 0.0
        if has_prefix and cell_strokes[0]:
            px0, py0, px1, py1 = strokes_bbox(cell_strokes[0])
            pw = px1 - px0
            py_shift = (0.5 * total_h) - 0.5 * (py0 + py1)
            for s in cell_strokes[0]:
                ss = s[:, :2].copy()
                ss[:, 0] -= px0
                ss[:, 1] += py_shift
                out.append(ss)
            x_brace = pw + float(rng.uniform(0.55, 0.95))

        out.extend(draw_human_delimiter("{", x_brace, 0.0, total_h, rng, slant=writer_slant))
        x_val_start = x_brace + float(rng.uniform(0.55, 0.85))
        x_cond_start = x_val_start + max_val_w + float(rng.uniform(0.85, 1.35))

        for r, (vi, ci) in enumerate(row_pairs):
            if cell_strokes[vi]:
                x0, y0, x1, y1 = strokes_bbox(cell_strokes[vi])
                vh = y1 - y0
                dy = row_ys[r] + 0.5 * (row_heights[r] - vh) - y0 + float(rng.normal(0.0, 0.05))
                dx = x_val_start - x0 + float(rng.normal(0.0, 0.08))
                for s in cell_strokes[vi]:
                    ss = s[:, :2].copy()
                    ss[:, 0] += dx
                    ss[:, 1] += dy
                    out.append(ss)
            if ci is not None and cell_strokes[ci]:
                x0, y0, x1, y1 = strokes_bbox(cell_strokes[ci])
                ch = y1 - y0
                dy = row_ys[r] + 0.5 * (row_heights[r] - ch) - y0 + float(rng.normal(0.0, 0.05))
                dx = x_cond_start - x0 + float(rng.normal(0.0, 0.08))
                for s in cell_strokes[ci]:
                    ss = s[:, :2].copy()
                    ss[:, 0] += dx
                    ss[:, 1] += dy
                    out.append(ss)
        return out

    return []


# ---------------------------------------------------------------------------
# Prompt Stream Loader
# ---------------------------------------------------------------------------

UNIMER_STRATA = ("single", "long_tree", "multiline", "matrix", "cases")


def _iter_unimer_prompts(tokenizer: InkForgeTokenizer, path: Path) -> Iterator[StructuredPrompt]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict) or row.get("split", "train") != "train":
                continue
            prompt = parse_unimer_prompt(row.get("normalized_latex") or row.get("latex") or "", tokenizer)
            if prompt is not None:
                yield prompt


def load_unimer_prompt_pool(
    tokenizer: InkForgeTokenizer,
    unimer_path: Path = DEFAULT_UNIMER_JSONL,
    *,
    max_prompts: int = 120000,
    seed: int = 2026,
) -> list[StructuredPrompt]:
    """Load and shuffle the first valid training prompts, preserving the flat API."""
    prompts = list(islice(_iter_unimer_prompts(tokenizer, unimer_path), max(0, max_prompts)))
    random.Random(seed).shuffle(prompts)
    return prompts


def load_stratified_unimer_pools(
    tokenizer: InkForgeTokenizer,
    unimer_path: Path = DEFAULT_UNIMER_JSONL,
    *,
    max_prompts: int = 120000,
    seed: int = 2026,
) -> dict[str, list[StructuredPrompt]]:
    """Reserve slots for every structure across the full file, returning at most max_prompts."""
    pools: dict[str, list[StructuredPrompt]] = {kind: [] for kind in UNIMER_STRATA}
    if max_prompts <= 0:
        return pools
    # Reserve slots for rare layouts even when the first pages are entirely single-line.
    quota = max(1, max_prompts // len(pools))
    spare: list[tuple[str, StructuredPrompt]] = []
    seen: set[str] = set()
    count = 0
    for prompt in _iter_unimer_prompts(tokenizer, unimer_path):
        kind = prompt.kind
        if kind == "tree":
            kind = "matrix" if prompt.meta.get("subkind") == "matrix_eq" else "long_tree"
        key = detokenize_inkforge(tokenize_inkforge(prompt.latex))
        if key in seen:
            continue
        if len(pools[kind]) < quota:
            pools[kind].append(prompt)
            seen.add(key)
            count += 1
            if count >= max_prompts:
                break
        elif len(spare) < max_prompts:
            spare.append((kind, prompt))
            seen.add(key)

    rng = random.Random(seed)
    rng.shuffle(spare)
    for kind, prompt in spare[:max_prompts - count]:
        pools[kind].append(prompt)
    for prompts in pools.values():
        rng.shuffle(prompts)
    return pools
