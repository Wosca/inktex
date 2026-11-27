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


def passes_geometry_gate(latex: str, strokes: Sequence[np.ndarray]) -> bool:
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


# ---------------------------------------------------------------------------
# Structured LaTeX Prompt Bank (UniMER + Multi-Line + Matrices + Cases)
# ---------------------------------------------------------------------------

@dataclass
class StructuredPrompt:
    kind: str  # "single", "multiline", "matrix", "cases"
    latex: str
    cells: list[str]  # Atomic single-line sub-expressions to generate with InkForge
    meta: dict[str, Any]


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

    # 4. Standard single-line formula (reject unparsed \begin/\end environments)
    if r"\begin" in s or r"\end" in s or "&" in s or r"\\" in s:
        return None
    toks = tokenize_inkforge(s)
    if not (2 <= len(toks) <= 80) or tokenizer.has_unk(s):
        return None
    return StructuredPrompt(kind="single", latex=s, cells=[s], meta={})


def sample_programmatic_structured_prompt(rng: random.Random) -> StructuredPrompt:
    """Synthesize diverse multi-line derivations, matrices, systems, and piecewise functions."""
    choice = rng.random()

    # A. Multi-line algebra / calculus derivation (40%)
    if choice < 0.42:
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

def assemble_structured_strokes(
    prompt: StructuredPrompt,
    cell_strokes: list[list[np.ndarray]],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Assemble individually generated InkForge cells into a cohesive 2D multi-line / matrix / cases canvas."""
    if prompt.kind == "single":
        return cell_strokes[0] if cell_strokes else []

    writer_slant = float(rng.uniform(-0.08, 0.08))

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
