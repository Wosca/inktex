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
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Sequence

import numpy as np
import torch

from .model import InkForgeModel
from .tokenizer import InkForgeTokenizer, tokenize_inkforge

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
        if prev_end is not None and n > 0:
            air_dist = float(np.linalg.norm(st[0] - prev_end))
            cursor_ms += float(rng.uniform(75.0, 190.0)) + air_dist * float(rng.uniform(25.0, 65.0))
        t_arr = np.empty((n,), dtype=np.float64)
        t_arr[0] = cursor_ms
        for i in range(1, n):
            dist = float(np.linalg.norm(st[i] - st[i - 1]))
            dt = max(4.0, dist * ms_per_unit * float(rng.uniform(0.85, 1.18)))
            cursor_ms += dt
            t_arr[i] = cursor_ms
        times.append(t_arr)
        if n > 0:
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

    # Reject stray giant slash strokes / blown-up height on plain single-line atomic cells
    if is_atomic_cell:
        has_tall_op = any(k in latex for k in (r"\frac", r"\int", r"\sum", r"\prod", r"\begin", r"\sqrt"))
        max_h_allowed = 8.5 if has_tall_op else 4.8
        if h > max_h_allowed:
            return False
        stroke_hs = [float(s[:, 1].max() - s[:, 1].min()) for s in pts]
        if max(stroke_hs) > (6.5 if has_tall_op else 3.8):
            return False

    # Strip environment names before estimating visible symbol count
    clean_tex = re.sub(r"\\(?:begin|end)\{[A-Za-z*]+\}", "", latex)
    toks = tokenize_inkforge(clean_tex)
    vis_est = max(1, sum(1 for t in toks if t not in ("{", "}", "^", "_", "<SPACE>", "&", r"\\")))
    n_strokes = len(pts)
    if vis_est >= 4 and n_strokes < max(2, int(vis_est * 0.25)):
        return False
    if n_strokes > max(28, vis_est * 6):
        return False
    return True


# ---------------------------------------------------------------------------
# Hand-Drawn Organic Tall Delimiters (`[ ]`, `( )`, `| |`, `{`)
# ---------------------------------------------------------------------------

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
    ts = np.linspace(0.0, 1.0, n)
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
        xs = x_center + sign * bulge * np.sin(ts * math.pi) * 0.0
        # Quadratic arc: curves outward at center, inward at top/bottom
        arc = sign * bulge * (1.0 - (2.0 * ts - 1.0) ** 2)
        xs = x_center + arc - sign * 0.5 * bulge + wobble + slant_shift
        return [np.stack([xs, ys], axis=-1)]

    if kind in ("[", "]"):
        sign = 1.0 if kind == "[" else -1.0
        hook = min(0.45, max(0.22, 0.12 * h)) * float(rng.uniform(0.85, 1.15))
        # Single continuous stroke: top hook -> vertical spine -> bottom hook (how humans draw [ and ])
        top_x = np.linspace(x_center + sign * hook, x_center, 5)
        top_y = np.full(5, y_t) + rng.normal(0, 0.015, size=5)
        mid_x = x_center + wobble + slant_shift
        mid_y = ys
        bot_x = np.linspace(x_center, x_center + sign * hook, 5)
        bot_y = np.full(5, y_b) + rng.normal(0, 0.015, size=5)
        xs = np.concatenate([top_x[:-1], mid_x, bot_x[1:]])
        ys_full = np.concatenate([top_y[:-1], mid_y, bot_y[1:]])
        return [np.stack([xs, ys_full], axis=-1)]

    if kind == "{":
        # Classic 1- or 2-stroke curly brace
        mid_y = 0.5 * (y_t + y_b) + float(rng.uniform(-0.08, 0.08))
        depth = min(0.50, max(0.25, 0.14 * h)) * float(rng.uniform(0.88, 1.15))
        half = n // 2
        t_top = np.linspace(0.0, 1.0, half)
        t_bot = np.linspace(0.0, 1.0, n - half)
        y_u = y_t + t_top * (mid_y - y_t)
        y_l = mid_y + t_bot * (y_b - mid_y)
        # Top half curves from right hook -> spine -> left tip
        x_u = x_center + depth * (0.5 * (1.0 - t_top) ** 2 - 0.65 * (t_top ** 3))
        x_l = x_center + depth * (0.5 * (t_bot ** 2) - 0.65 * ((1.0 - t_bot) ** 3))
        xs = np.concatenate([x_u, x_l[1:]]) + wobble[: len(x_u) + len(x_l) - 1] + slant_shift[: len(x_u) + len(x_l) - 1]
        ys_full = np.concatenate([y_u, y_l[1:]])
        return [np.stack([xs, ys_full], axis=-1)]

    return []


def draw_human_hline(
    x_left: float,
    x_right: float,
    y_bar: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate a natural hand-drawn horizontal fraction bar (vinculum) with subtle tilt & hand wobble."""
    w = max(x_right - x_left, 0.6)
    n = max(10, min(64, int(round(w * 6.0))))
    ts = np.linspace(0.0, 1.0, n)
    xs = x_left + ts * w
    slope = float(rng.normal(0.0, 0.015))
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
) -> list[np.ndarray]:
    """Generate an organic hand-drawn radical check + overbar wrapping `(x_left..x_right, y_top..y_bot)`."""
    h = max(y_bot - y_top, 1.0)
    w = max(x_right - x_left, 0.8)
    y_roof = y_top - float(rng.uniform(0.16, 0.30))
    y_valley = y_bot + float(rng.uniform(0.02, 0.14))
    y_hook = y_top + h * float(rng.uniform(0.45, 0.62))

    check_w = min(1.15, max(0.55, 0.32 * h))
    x_valley = x_left - float(rng.uniform(0.24, 0.42))
    x_hook = x_valley - check_w * float(rng.uniform(0.45, 0.65))
    x_peak = x_left - float(rng.uniform(0.02, 0.12))
    x_end = x_right + float(rng.uniform(0.14, 0.32))

    # 1. Small upward/downward hook -> 2. downstroke to valley -> 3. diagonal up to roof -> 4. overbar
    seg1 = np.stack([
        np.linspace(x_hook - 0.14, x_hook, 4),
        np.linspace(y_hook + 0.10, y_hook, 4),
    ], axis=-1)
    seg2 = np.stack([
        np.linspace(x_hook, x_valley, max(6, int(round(h * 3.0)))),
        np.linspace(y_hook, y_valley, max(6, int(round(h * 3.0)))),
    ], axis=-1)
    seg3 = np.stack([
        np.linspace(x_valley, x_peak, max(8, int(round(h * 4.5)))),
        np.linspace(y_valley, y_roof, max(8, int(round(h * 4.5)))),
    ], axis=-1)
    n_roof = max(10, min(64, int(round((x_end - x_peak) * 5.5))))
    ts = np.linspace(0.0, 1.0, n_roof)
    roof_x = x_peak + ts * (x_end - x_peak)
    roof_y = y_roof + (ts - 0.5) * (x_end - x_peak) * float(rng.normal(0.0, 0.012)) + 0.018 * np.sin(
        ts * math.pi * float(rng.uniform(1.5, 3.0)) + float(rng.uniform(0, 2 * math.pi))
    )
    seg4 = np.stack([roof_x, roof_y], axis=-1)
    pts = np.concatenate([seg1[:-1], seg2[:-1], seg3[:-1], seg4], axis=0)
    # Subtle micro-tremor along the check
    pts[:, 0] += rng.normal(0.0, 0.008, size=len(pts))
    pts[:, 1] += rng.normal(0.0, 0.008, size=len(pts))
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


def _extract_balanced_brace(s: str, open_idx: int) -> tuple[str, int] | None:
    """Given `s[open_idx] == '{'`, return `(inner, close_idx)` or `None` if unbalanced."""
    if open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif s[i] == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : i], i
    return None


def _decompose_big_frac_tree(
    latex: str,
    tokenizer: InkForgeTokenizer,
    cells: list[str],
    *,
    depth: int = 0,
    max_depth: int = 2,
) -> dict[str, Any] | None:
    """Recursively decompose large top-level `\\frac{NUM}{DEN}` and `\\sqrt{BODY}` into an `InkNode` tree.

    Keeps short simple fractions like `\\frac{1}{2}` or `\\sqrt{x}` as native single InkForge cells,
    while decomposing big fractions and compound radicals so InkForge never blows up on them.
    """
    s = latex.strip()
    if not s:
        return None
    toks = tokenize_inkforge(s)
    has_frac = r"\frac" in s
    has_sqrt = r"\sqrt" in s
    if depth >= max_depth or (not has_frac and not has_sqrt) or (not has_sqrt and len(toks) < 16):
        if 1 <= len(toks) <= 44 and not tokenizer.has_unk(s):
            idx = len(cells)
            cells.append(s)
            return {"op": "cell", "i": idx}
        return None

    # 1. Find first top-level `\frac{...}{...}` (brace depth == 0)
    bdepth = 0
    i = 0
    found_frac: tuple[int, str, str, int] | None = None
    found_sqrt: tuple[int, str, int] | None = None
    while i < len(s):
        ch = s[i]
        if ch == "{" and (i == 0 or s[i - 1] != "\\"):
            bdepth += 1
        elif ch == "}" and (i == 0 or s[i - 1] != "\\"):
            bdepth -= 1
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
                    if max(len(num_t), len(den_t)) >= 6 or r"\frac" in num_s or r"\frac" in den_s or r"\sqrt" in num_s or r"\sqrt" in den_s:
                        found_frac = (i, num_s.strip(), den_s.strip(), den_end + 1)
                        break
        elif bdepth == 0 and found_sqrt is None and s.startswith(r"\sqrt", i) and (i + 5 < len(s) and s[i + 5] in ("{", " ")):
            j = i + 5
            while j < len(s) and s[j] == " ":
                j += 1
            rad_res = _extract_balanced_brace(s, j)
            if rad_res is not None:
                rad_s, rad_end = rad_res
                rad_t = tokenize_inkforge(rad_s)
                # Decompose compound or long radicals (e.g. \sqrt{b^2-4ac} + 2xy or \sqrt{\frac{...}{...}})
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

    if 1 <= len(toks) <= 56 and not tokenizer.has_unk(s):
        idx = len(cells)
        cells.append(s)
        return {"op": "cell", "i": idx}
    return None


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


def parse_unimer_prompt(latex: str, tokenizer: InkForgeTokenizer) -> StructuredPrompt | None:
    """Parse a clean UniMER formula into either a single-line or structured matrix/cases/multiline prompt."""
    s = (latex or "").strip()
    if not s or len(s) > 220:
        return None

    # 1. Pure Matrix environment
    m = _ENV_MATRIX_RE.match(s)
    if m:
        env, body = m.group(1), m.group(2).strip()
        rows_raw = [r.strip() for r in body.split(r"\\") if r.strip()]
        if not (2 <= len(rows_raw) <= 4):
            return None
        grid: list[list[str]] = []
        cells: list[str] = []
        for r in rows_raw:
            cols = [c.strip() for c in r.split("&")]
            if not (1 <= len(cols) <= 4):
                return None
            for c in cols:
                if not c or tokenizer.has_unk(c) or len(tokenize_inkforge(c)) > 24:
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
            rows_raw = [r.strip() for r in body.split(r"\\") if r.strip()]
            if 2 <= len(rows_raw) <= 4:
                cells = []
                if prefix:
                    if tokenizer.has_unk(prefix) or len(tokenize_inkforge(prefix)) > 24:
                        return None
                    cells.append(prefix)
                row_pairs: list[tuple[int, int | None]] = []
                ok = True
                for r in rows_raw:
                    parts = [p.strip() for p in r.split("&")]
                    if not (1 <= len(parts) <= 2) or not parts[0] or tokenizer.has_unk(parts[0]):
                        ok = False
                        break
                    v_idx = len(cells)
                    cells.append(parts[0])
                    c_idx = None
                    if len(parts) == 2 and parts[1]:
                        if tokenizer.has_unk(parts[1]):
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
    if m_a:
        env, body = m_a.group(1), m_a.group(2).strip()
        lines = [ln.replace("&", "").strip() for ln in body.split(r"\\") if ln.strip()]
        if 2 <= len(lines) <= 5 and all(not tokenizer.has_unk(ln) and len(tokenize_inkforge(ln)) <= 50 for ln in lines):
            return StructuredPrompt(
                kind="multiline",
                latex=s,
                cells=lines,
                meta={"env": env},
            )

    # 4. Standard single-line formula (or auto-decomposed big/nested fraction!)
    if r"\begin" in s or r"\end" in s or "&" in s or r"\\" in s:
        return None
    toks = tokenize_inkforge(s)
    if not (2 <= len(toks) <= 110) or tokenizer.has_unk(s):
        return None
    if r"\frac" in s and len(toks) >= 20:
        cells: list[str] = []
        tree = _decompose_big_frac_tree(s, tokenizer, cells, depth=0, max_depth=2)
        if tree is not None and len(cells) >= 2:
            return StructuredPrompt(kind="tree", latex=s, cells=cells, meta={"tree": tree, "subkind": "big_frac"})
    if len(toks) > 80:
        return None
    return StructuredPrompt(kind="single", latex=s, cells=[s], meta={})


def _tree_cell(cells: list[str], expr: str) -> dict[str, Any]:
    idx = len(cells)
    cells.append(expr)
    return {"op": "cell", "i": idx}


def sample_big_fraction_or_multistep_prompt(rng: random.Random) -> StructuredPrompt:
    """Synthesize big nested fractions, big radicals, tall delimited expressions, and multi-line fraction proofs."""
    v = rng.choice(["x", "y", "t", "u", "z", "n"])
    v2 = "y" if v != "y" else "x"
    g = rng.choice([r"\alpha", r"\beta", r"\theta", r"\lambda", r"\omega"])
    a = rng.randint(2, 8)
    b = rng.randint(2, 7)
    c = rng.randint(2, 9)
    mode = rng.choice([
        "big_frac_eq",
        "nested_frac",
        "big_sqrt_frac",
        "tall_delim_frac",
        "multiline_frac_proof",
        "multiline_aligned_eq",
    ])
    cells: list[str] = []

    if mode == "big_frac_eq":
        lhs = rng.choice([f"f({v}) =", f"{v2} =", r"\Delta =", "I =", "S ="])
        num = rng.choice([
            f"{a}{v}^2 + {b}{v} - {c}",
            f"\\int_0^{b} {v}^2 d{v} + {a}{v}",
            f"\\sin^2 {g} + \\cos^2 {g} + {a}{v}",
            f"\\sum_{{i=1}}^{v} i + {b}{v}^2",
            f"({v} + {a})({v} - {b}) + {c}",
        ])
        den_kind = rng.randrange(4)
        if den_kind == 0:
            rad_inner = f"{v}^2 + {a}{v} + {c}"
            den_tex = f"\\sqrt{{{rad_inner}}}"
            den_node: dict[str, Any] = {"op": "sqrt", "body": _tree_cell(cells, rad_inner)}
        elif den_kind == 1:
            rad_inner = f"{a}^2 + {b}^2"
            den_tex = f"\\sqrt{{{rad_inner}}} + {v}"
            den_node = {
                "op": "hcat",
                "items": [{"op": "sqrt", "body": _tree_cell(cells, rad_inner)}, _tree_cell(cells, f"+ {v}")],
            }
        elif den_kind == 2:
            den_tex = f"{b}{v}^2 - {a}{v} + {c}"
            den_node = _tree_cell(cells, den_tex)
        else:
            den_tex = f"({v} + {a})^2 + {b}"
            den_node = _tree_cell(cells, den_tex)
        tail = rng.choice(["", f"+ {c}", "= 0", f"= {b}{v}"])
        items = [_tree_cell(cells, lhs), {"op": "frac", "num": _tree_cell(cells, num), "den": den_node, "scale": 0.96}]
        if tail:
            items.append(_tree_cell(cells, tail))
        latex = f"{lhs} \\frac{{{num}}}{{{den_tex}}}" + (f" {tail}" if tail else "")
        return StructuredPrompt(kind="tree", latex=latex, cells=cells, meta={"tree": {"op": "hcat", "items": items}, "subkind": "big_frac"})

    if mode == "nested_frac":
        lhs = rng.choice([f"f({v}) =", "R =", "H =", ""])
        n_left = rng.choice([f"{a}{v} +", "1 +", f"{v}^2 -"])
        n_num, n_den = rng.choice([(f"{b}", f"{v} + {c}"), (f"{v}", f"{a}"), (f"{a}{v}", f"{v} - {b}")])
        d_left = rng.choice([f"{v} -", "1 +", f"{b}{v} +"])
        d_num, d_den = rng.choice([(f"{c}", f"{v} + {a}"), (f"{b}", f"{v}^2 + 1"), (f"{v}", f"{c}")])
        num_node = {
            "op": "hcat",
            "items": [
                _tree_cell(cells, n_left),
                {"op": "frac", "num": _tree_cell(cells, n_num), "den": _tree_cell(cells, n_den), "scale": 0.82},
            ],
        }
        den_node = {
            "op": "hcat",
            "items": [
                _tree_cell(cells, d_left),
                {"op": "frac", "num": _tree_cell(cells, d_num), "den": _tree_cell(cells, d_den), "scale": 0.82},
            ],
        }
        frac_node = {"op": "frac", "num": num_node, "den": den_node, "scale": 0.94}
        if lhs:
            tree = {"op": "hcat", "items": [_tree_cell(cells, lhs), frac_node]}
            latex = f"{lhs} \\frac{{{n_left} \\frac{{{n_num}}}{{{n_den}}}}}{{{d_left} \\frac{{{d_num}}}{{{d_den}}}}}"
        else:
            tree = frac_node
            latex = f"\\frac{{{n_left} \\frac{{{n_num}}}{{{n_den}}}}}{{{d_left} \\frac{{{d_num}}}{{{d_den}}}}}"
        return StructuredPrompt(kind="tree", latex=latex, cells=cells, meta={"tree": tree, "subkind": "nested_frac"})

    if mode == "big_sqrt_frac":
        lhs = rng.choice([f"{v} =", "r =", "L =", ""])
        num = rng.choice([f"{a}{v}^2 + {b}{v} + {c}", f"({v} + {a})^2 - {b}", f"{v}^3 + {c}{v}"])
        den = rng.choice([f"{b}{v} + {c}", f"{a}{v}^2 + 1", f"{v}^2 + {b}^2"])
        inner = {"op": "frac", "num": _tree_cell(cells, num), "den": _tree_cell(cells, den), "scale": 0.92}
        sq_node = {"op": "sqrt", "body": inner}
        if lhs:
            tree = {"op": "hcat", "items": [_tree_cell(cells, lhs), sq_node]}
            latex = f"{lhs} \\sqrt{{\\frac{{{num}}}{{{den}}}}}"
        else:
            tree = sq_node
            latex = f"\\sqrt{{\\frac{{{num}}}{{{den}}}}}"
        return StructuredPrompt(kind="tree", latex=latex, cells=cells, meta={"tree": tree, "subkind": "big_sqrt"})

    if mode == "tall_delim_frac":
        l_d, r_d = rng.choice([("(", ")"), ("[", "]"), ("|", "|")])
        exp_s = rng.choice(["2", "n", "-1", ""]) if l_d != "|" else ""
        num = rng.choice([f"{a}{v} + {b}", f"{v}^2 - {c}", f"\\sin {g} + {a}"])
        den = rng.choice([f"{c}{v} - {a}", f"{v} + {b}", f"\\cos {g} + {b}"])
        frac_node = {"op": "frac", "num": _tree_cell(cells, num), "den": _tree_cell(cells, den), "scale": 0.94}
        sup_node = _tree_cell(cells, exp_s) if exp_s else None
        delim_node = {"op": "delim", "left": l_d, "right": r_d, "body": frac_node, "sup": sup_node}
        rhs = f"= {a + b}"
        tree = {"op": "hcat", "items": [delim_node, _tree_cell(cells, rhs)]}
        latex = f"\\left{l_d} \\frac{{{num}}}{{{den}}} \\right{r_d}" + (f"^{{{exp_s}}}" if exp_s else "") + f" {rhs}"
        return StructuredPrompt(kind="tree", latex=latex, cells=cells, meta={"tree": tree, "subkind": "tall_delim"})

    if mode == "multiline_frac_proof":
        # 3-line calculus/algebra derivation with big composed fractions on EVERY line (same writer!)
        line1 = {
            "op": "hcat",
            "items": [
                _tree_cell(cells, f"f({v}) ="),
                {"op": "frac", "num": _tree_cell(cells, f"{a}{v}^2 + {b}"), "den": _tree_cell(cells, f"{v} + {c}"), "scale": 0.94},
            ],
        }
        line2 = {
            "op": "hcat",
            "items": [
                _tree_cell(cells, f"f'({v}) ="),
                {
                    "op": "frac",
                    "num": _tree_cell(cells, f"{2*a}{v}({v} + {c}) - ({a}{v}^2 + {b})"),
                    "den": _tree_cell(cells, f"({v} + {c})^2"),
                    "scale": 0.94,
                },
            ],
        }
        line3 = {
            "op": "hcat",
            "items": [
                _tree_cell(cells, f"f'({v}) ="),
                {
                    "op": "frac",
                    "num": _tree_cell(cells, f"{a}{v}^2 + {2*a*c}{v} - {b}"),
                    "den": _tree_cell(cells, f"({v} + {c})^2"),
                    "scale": 0.94,
                },
            ],
        }
        l1 = f"f({v}) = \\frac{{{a}{v}^2 + {b}}}{{{v} + {c}}}"
        l2 = f"f'({v}) = \\frac{{{2*a}{v}({v} + {c}) - ({a}{v}^2 + {b})}}{{({v} + {c})^2}}"
        l3 = f"f'({v}) = \\frac{{{a}{v}^2 + {2*a*c}{v} - {b}}}{{({v} + {c})^2}}"
        latex = r"\begin{aligned}" + r" \\ ".join([l1, l2, l3]) + r"\end{aligned}"
        return StructuredPrompt(
            kind="tree",
            latex=latex,
            cells=cells,
            meta={"tree": {"op": "vstack", "lines": [line1, line2, line3]}, "subkind": "multiline_frac"},
        )

    # multiline_aligned_eq: true `=`-column aligned multi-step equation solving!
    rhs_val = a * c + b
    d = rng.randint(2, 8)
    rows_tex = [
        (f"{a}{v} + {b} + {d}", f"= {rhs_val} + {d}"),
        (f"{a}{v} + {b}", f"= {rhs_val}"),
        (f"{a}{v}", f"= {rhs_val - b}"),
        (f"{v}", f"= {c}"),
    ]
    n_lines = rng.choice([3, 4])
    chosen = rows_tex[1:] if n_lines == 3 else rows_tex
    eq_rows = [(_tree_cell(cells, lhs_s), _tree_cell(cells, rhs_s)) for lhs_s, rhs_s in chosen]
    latex = r"\begin{aligned}" + r" \\ ".join(f"{l_s} & {r_s}" for l_s, r_s in chosen) + r"\end{aligned}"
    return StructuredPrompt(
        kind="tree",
        latex=latex,
        cells=cells,
        meta={"tree": {"op": "aligned_eq", "rows": eq_rows}, "subkind": "aligned_eq"},
    )


def sample_programmatic_structured_prompt(rng: random.Random) -> StructuredPrompt:
    """Synthesize diverse multi-line derivations, big/nested fractions, radicals, matrices, and piecewise functions."""
    choice = rng.random()

    # 0. Big / nested fractions, big radicals, tall delimiters & multi-line fraction derivations (30%)
    if choice < 0.30:
        return sample_big_fraction_or_multistep_prompt(rng)

    # A. Multi-line algebra / calculus derivation (28%)
    if choice < 0.58:
        v = rng.choice(["x", "y", "t", "u", "z", "n"])
        a = rng.randint(2, 9)
        b = rng.randint(1, 9)
        c = rng.randint(1, 12)
        template = rng.choice(["linear_solve", "quadratic_expand", "deriv_chain", "integral_eval", "system2"])
        if template == "linear_solve":
            rhs = a * c + b
            lines = [
                f"{a}{v} + {b} = {rhs}",
                f"{a}{v} = {rhs - b}",
                f"{v} = {c}",
            ]
        elif template == "quadratic_expand":
            lines = [
                f"({v} + {b})^2 = {c}",
                f"{v}^2 + {2*b}{v} + {b*b} = {c}",
                f"{v}^2 + {2*b}{v} + {b*b - c} = 0" if b * b >= c else f"{v}^2 + {2*b}{v} = {c - b*b}",
            ]
        elif template == "deriv_chain":
            g = rng.choice([r"\sin", r"\cos", r"\ln", r"\exp"])
            lines = [
                f"f({v}) = {g}({a}{v}^{b})",
                f"f'({v}) = {a*b}{v}^{b-1} \\cdot {g}({a}{v}^{b})" if b > 1 else f"f'({v}) = {a} {g}({a}{v})",
            ]
        elif template == "integral_eval":
            lines = [
                f"I = \\int_0^{b} {a}{v} d{v}",
                f"I = \\frac{{{a}}}{{2}} {v}^2 \\mid_0^{b}",
                f"I = \\frac{{{a * b * b}}}{{2}}",
            ]
        else:
            v2 = "y" if v != "y" else "z"
            lines = [
                f"{a}{v} + {b}{v2} = {c}",
                f"{b}{v} - {v2} = {a}",
            ]
        use_aligned = rng.random() < 0.55
        if use_aligned:
            latex = r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"
        else:
            latex = r" \\ ".join(lines)
        return StructuredPrompt(kind="multiline", latex=latex, cells=lines, meta={"env": "aligned" if use_aligned else "flat"})

    # B. Matrix (2x2, 2x3, 3x3) with bmatrix / pmatrix / vmatrix (33%)
    if choice < 0.75:
        env = rng.choice(["bmatrix", "pmatrix", "vmatrix", "matrix"])
        rows = rng.choice([2, 2, 3])
        cols = rng.choice([2, 2, 3])
        if env == "vmatrix":
            cols = rows
        pool = [
            "0", "1", "-1", "2", "3", "-2", "4", "5", "a", "b", "c", "d",
            "x", "y", "z", r"\alpha", r"\beta", r"\lambda", r"\cos\theta", r"\sin\theta",
            r"-\sin\theta", "1-x", r"\frac{1}{2}", "2x", "x^2",
        ]
        grid = [[rng.choice(pool) for _ in range(cols)] for _ in range(rows)]
        body = r" \\ ".join(" & ".join(r) for r in grid)
        prefix = ""
        if rng.random() < 0.45:
            mat_name = rng.choice(["A", "B", "M", "R", "T", r"\Sigma"])
            prefix = f"{mat_name} = "
        latex = prefix + r"\begin{" + env + "}" + body + r"\end{" + env + "}"
        cells = ([prefix.strip()] if prefix else []) + [c for r in grid for c in r]
        return StructuredPrompt(
            kind="matrix",
            latex=latex,
            cells=cells,
            meta={
                "env": env,
                "rows": rows,
                "cols": cols,
                "row_lens": [cols] * rows,
                "has_prefix": bool(prefix),
            },
        )

    # C. Piecewise function `\begin{cases}` with `\text{if ...}` / `\text{otherwise}` (25%)
    v = rng.choice(["x", "t", "n"])
    fname = rng.choice(["f", "g", "h", "p"])
    a = rng.randint(1, 6)
    b = rng.randint(1, 5)
    case_rows = [
        (f"{v}^2 + {a}", rf"\text{{if }} {v} \geq 0"),
        (f"-{b}{v}", rf"\text{{if }} {v} < 0"),
    ]
    if rng.random() < 0.45:
        case_rows = [
            (f"{a}{v}", rf"\text{{if }} {v} > {b}"),
            ("0", rf"\text{{if }} {v} = {b}"),
            (f"-{v}", r"\text{otherwise}"),
        ]
    prefix = f"{fname}({v}) ="
    body = r" \\ ".join(f"{val} & {cond}" for val, cond in case_rows)
    latex = f"{prefix} \\begin{{cases}}{body}\\end{{cases}}"
    cells = [prefix]
    row_pairs: list[tuple[int, int | None]] = []
    for val, cond in case_rows:
        vi = len(cells)
        cells.append(val)
        ci = len(cells)
        cells.append(cond)
        row_pairs.append((vi, ci))
    return StructuredPrompt(
        kind="cases",
        latex=latex,
        cells=cells,
        meta={"has_prefix": True, "row_pairs": row_pairs},
    )


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
        dw, dh = (dx1 - dx0) * scale, (dy1 - dy0) * scale
        bar_pad = float(rng.uniform(0.24, 0.46))
        bar_w = max(nw, dw, 0.9) + 2.0 * bar_pad
        v_gap_top = float(rng.uniform(0.28, 0.46))
        v_gap_bot = float(rng.uniform(0.28, 0.46))
        y_bar = nh + v_gap_top

        out: list[np.ndarray] = []
        # Numerator centered above bar (natural order: numerator -> bar -> denominator)
        num_dx = 0.5 * (bar_w - nw) - nx0 * scale + float(rng.normal(0.0, 0.08))
        num_dy = -ny0 * scale
        out.extend(_shift_scale_strokes(num_st, dx=num_dx, dy=num_dy, scale=scale))
        out.extend(draw_human_hline(0.0, bar_w, y_bar, rng))
        den_dx = 0.5 * (bar_w - dw) - dx0 * scale + float(rng.normal(0.0, 0.08))
        den_dy = (y_bar + v_gap_bot) - dy0 * scale
        out.extend(_shift_scale_strokes(den_st, dx=den_dx, dy=den_dy, scale=scale))

        bx0, by0, bx1, by1 = strokes_bbox(out)
        out = _shift_scale_strokes(out, dx=-bx0, dy=-by0)
        return out, (0.0, 0.0, bx1 - bx0, by1 - by0), y_bar - by0

    if op == "sqrt":
        body_st, (x0, y0, x1, y1), y_ax = _assemble_ink_node(node["body"], cell_strokes, rng, writer_slant)
        if not body_st:
            return [], (0.0, 0.0, 0.5, 1.0), 0.5
        rad_st = draw_human_sqrt(x0, y0, x1, y1, rng)
        out = rad_st + body_st
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
        out = left_d + body_shifted + right_d
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


def assemble_structured_strokes(
    prompt: StructuredPrompt,
    cell_strokes: list[list[np.ndarray]],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Assemble individually generated InkForge cells into a cohesive 2D multi-line / fraction / matrix / cases canvas."""
    if prompt.kind == "single":
        return cell_strokes[0] if cell_strokes else []

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
            out.extend(draw_human_delimiter(right_delim, x_right, 0.0, mat_h, rng, slant=writer_slant))

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

def load_unimer_prompt_pool(
    tokenizer: InkForgeTokenizer,
    unimer_path: Path = DEFAULT_UNIMER_JSONL,
    *,
    max_prompts: int = 120000,
    seed: int = 2026,
) -> list[StructuredPrompt]:
    """Load real-world STEM formulas (single-line, matrices, cases, aligned) from UniMER."""
    prompts: list[StructuredPrompt] = []
    if not unimer_path.is_file():
        return prompts
    with unimer_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("split", "train") != "train":
                continue
            tex = r.get("normalized_latex") or r.get("latex") or ""
            sp = parse_unimer_prompt(tex, tokenizer)
            if sp is not None:
                prompts.append(sp)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts
