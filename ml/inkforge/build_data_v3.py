"""Append targeted v3 training rows to a packed, leak-free InkForge v2 cache."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import filecmp
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Iterator, Sequence

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from inkforge.dataset import (  # noqa: E402
    DEFAULT_V2_CACHE_DIR,
    DEFAULT_V3_CACHE_DIR,
    SOURCE_NAMES,
    V3_ALIAS_SOURCE_ID,
    V3_COMPOSED_SOURCE_ID,
    V3_SOURCE_NAMES,
    _normalize_donor_glyph,
    expand_single_token_braces,
    unpack_points_to_strokes,
)
from inkforge.tokenizer import InkForgeTokenizer, tokenize_inkforge  # noqa: E402


ACCENT_ALIASES = {r"\overline": (r"\bar", 8000), r"\hat": (r"\widehat", 3000), r"\tilde": (r"\widetilde", 3000)}
FUNCTION_WORDS = tuple("ln exp max min det cot sec csc sinh cosh tanh arcsin arccos arctan gcd dim ker deg arg".split())
ZERO_COUNT_TOKENS = tuple(r"""# % @ \arccos \arcsin \arctan \arg \bar \binom \boxed \cosh \cot \csc \ddot
\deg \det \dim \exp \gcd \ker \ln \mathbf \max \min \sec \sinh \tanh \widehat \widetilde ~""".split())
RARE_GREEKS = tuple(r"\xi \zeta \eta \kappa \tau \rho \chi \varphi \Psi \Phi \Omega \Gamma \Lambda".split())
REPORTED_TOKENS = (*ZERO_COUNT_TOKENS, *RARE_GREEKS, r"\log", r"\tan")

COMPOSITION_TEMPLATES = (
    r"\ln(x+1)", r"\ln(ax^2+b)", r"\log_2(x)", r"\exp(-x^2)", r"\max(a,b)", r"\min(x,y)", r"\det(A)",
    r"\tan\theta", r"\cot\theta", r"\sec x", r"\csc x", r"\sinh x", r"\cosh x", r"\tanh x",
    r"\arctan x", r"\arcsin x", r"\arccos x", r"\gcd(a,b)", r"\dim V", r"\ker T", r"\deg p", r"\arg z",
    r"\xi + \zeta = \eta \kappa + \tau \rho", r"\Psi(x,t) = \Phi(x) e^{-i\omega t}",
    r"\Omega = \Lambda \Delta \pm \chi \varphi", r"\Gamma(n+1) = n!",
    r"\hat{x} \cdot \bar{y} + \tilde{\phi} \equiv 0", r"P(B|\bar{A})P(\bar{A})", r"\ddot{x} + \omega^2 x = 0",
    r"\widehat{AB} + \widetilde{CD}", r"\mathbf{v} = (x,y)", r"\boxed{x=1}",
    r"\sum_{k=0}^{n} a_k x^k", r"\sum_{n=0}^{\infty} \frac{x^n}{n!}",
    r"\prod_{k=1}^{n} (1 + \frac{1}{k})", r"\lim_{x \rightarrow \infty} \frac{1}{x} = 0", r"\lim_{x \rightarrow \infty} \ldots",
    r"\int_{0}^{\infty} e^{-t} dt = 1", r"\binom{n}{k}",
    r"\text{if } x > 0", r"\text{if } x \neq 0", r"\text{if } x = 0", r"\text{otherwise}",
    r"\forall x \in \mathbb{R}, \exists y \in \mathbb{Z}", r"x \notin A \Rightarrow A \subset B", r"p \Leftrightarrow q \neq r",
    r"\#1 + #2", r"p = 50%", r"x@y", r"a~b",
    r"\xi_0 = \zeta + \eta", r"\rho(\tau) = \kappa \chi - \varphi", r"\Phi(x) \neq \Psi(x)",
    r"\Gamma(z) \Gamma(1-z)", r"\Omega \subset \Lambda", r"\ln x + \ln y = \ln(xy)",
    r"\log_{10}(x) = \frac{\ln x}{\ln 10}", r"\sinh^2 x + 1 = \cosh^2 x", r"\tan x = \frac{\sin x}{\cos x}",
)

_WORD_RUN = re.compile(r"[a-zA-Z]+")
_FUNCTION_SET = frozenset(FUNCTION_WORDS)
_WORD_OPERATORS = _FUNCTION_SET | {"log", "tan", "sin", "cos", "lim"}
_ACCENTS = {r"\hat", r"\widehat", r"\tilde", r"\widetilde", r"\bar", r"\ddot"}
_LIMIT_OPERATORS = {r"\sum", r"\prod", r"\lim"}
_INLINE_TEXT = {r"\text", r"\mathrm", r"\mathbf", r"\mathbb", r"\mathcal"}
_DESCENDERS = frozenset("gjpqy")
_SHORT_LETTERS = frozenset("aceimnorsuvwxz")
_SMALL_OPERATORS = {"+", "-", "=", "<", ">", r"\cdot", r"\equiv", r"\in", r"\notin", r"\subset", r"\neq", r"\pm",
                    r"\rightarrow", r"\Rightarrow", r"\Leftrightarrow", r"\sim"}
_GLYPH_ALIASES = {"#": r"\#", "%": r"\%", r"\widehat": r"\hat", r"\widetilde": r"\tilde"}
_ARRAY_FIELDS = ("points_xy", "points_pen", "pt_offsets", "tokens", "tok_offsets", "source_ids", "writer_ids")


class PackedSplit:
    def __init__(self, path: Path):
        (self.xy, self.pen, self.point_offsets, self.tokens, self.token_offsets, self.sources, self.writers) = (
            np.load(path / f"{name}.npy", mmap_mode="r") for name in _ARRAY_FIELDS
        )
        self.writer_names = json.loads((path / "writer_names.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.sources)

    def ink(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.point_offsets[row : row + 2]
        return self.xy[lo:hi], self.pen[lo:hi]

    def label(self, row: int) -> np.ndarray:
        lo, hi = self.token_offsets[row : row + 2]
        return self.tokens[lo:hi]

    def verify(self, vocab_size: int) -> None:
        if len(self.writers) != len(self) or len(self.pen) != len(self.xy) or self.xy.shape != (len(self.pen), 2):
            raise ValueError("Packed split has inconsistent array lengths")
        for offsets, size in ((self.point_offsets, len(self.xy)), (self.token_offsets, len(self.tokens))):
            if len(offsets) != len(self) + 1 or offsets[0] != 0 or offsets[-1] != size or np.any(np.diff(offsets) <= 0):
                raise ValueError("Packed split has invalid offsets")
        if (self.pen.max(initial=0) > 1 or np.any(self.pen[self.point_offsets[1:] - 1] != 1)
                or self.writers.min(initial=0) < 0 or self.writers.max(initial=0) >= len(self.writer_names)
                or self.tokens.max(initial=0) >= vocab_size):
            raise ValueError("Packed split has invalid pen, writer, or token IDs")
        for start in range(0, len(self.xy), 1_000_000):
            if not np.isfinite(self.xy[start : start + 1_000_000]).all():
                raise ValueError("Packed split contains NaN or infinite points")


@dataclass
class AddedRow:
    tokens: np.ndarray
    xy: np.ndarray
    pen: np.ndarray
    source: int
    writer: int = 0


def relabel_function_words(tokens: Sequence[str]) -> list[str]:
    """Replace entire plain-letter math runs, preserving words in text/style groups."""
    result: list[str] = []
    depth = 0
    protected_depth: int | None = None
    letter_runs = "".join(t if len(t) == 1 and t.isascii() and t.isalpha() else "\0" for t in tokens)
    matches = {match.start(): match for match in _WORD_RUN.finditer(letter_runs) if match.group() in _FUNCTION_SET}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "{":
            depth += 1
            if i and tokens[i - 1] in _INLINE_TEXT and protected_depth is None:
                protected_depth = depth
        elif tok == "}":
            if protected_depth == depth:
                protected_depth = None
            depth -= 1
        match = matches.get(i) if protected_depth is None else None
        if match is not None and (not i or tokens[i - 1] not in ("^", "_")):
            result.append("\\" + match.group())
            i = match.end()
        else:
            result.append(tok)
            i += 1
    return result


def _ink_digest(xy: np.ndarray, pen: np.ndarray) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(memoryview(xy).cast("B"))
    digest.update(memoryview(pen).cast("B"))
    return digest.digest()


def _verify_base_leakage(train: PackedSplit, val: PackedSplit) -> set[bytes]:
    train_names = {train.writer_names[i] for i in np.unique(train.writers)} - {""}
    val_names = {val.writer_names[i] for i in np.unique(val.writers)} - {""}
    if train_names & val_names or np.any(val.sources > 5):
        raise ValueError("Base validation overlaps named training writers or contains synthetic sources")
    val_labels: dict[bytes, set[bytes]] = defaultdict(set)
    for row in range(len(val)):
        val_labels[_ink_digest(*val.ink(row))].add(val.label(row).tobytes())
    candidate_lengths = np.isin(np.diff(train.point_offsets), np.unique(np.diff(val.point_offsets)))
    for row in np.flatnonzero(candidate_lengths):
        labels = val_labels.get(_ink_digest(*train.ink(row)))
        if labels is not None and train.label(row).tobytes() in labels:
            raise ValueError(f"Base train row {row} duplicates a validation record")
    return set(val_labels)


def _rows_at_token_positions(split: PackedSplit, positions: np.ndarray) -> np.ndarray:
    rows = np.unique(np.searchsorted(split.token_offsets, positions, side="right") - 1)
    return rows[split.sources[rows] <= 5]


def _alias_rows(
    split: PackedSplit, tokenizer: InkForgeTokenizer, val_ink: set[bytes], max_len: int, rng: np.random.Generator,
) -> Iterator[AddedRow]:
    vocab = tokenizer.token_to_id
    for old, (new, cap) in ACCENT_ALIASES.items():
        if old not in vocab or new not in vocab or "{" not in vocab:
            continue
        positions = np.flatnonzero((split.tokens[:-1] == vocab[old]) & (split.tokens[1:] == vocab["{"]))
        rows = _rows_at_token_positions(split, positions)
        count = 0
        for row in rng.permutation(rows):
            xy, pen = split.ink(row)
            if _ink_digest(xy, pen) in val_ink:
                continue
            ids = split.label(row).copy()
            relabeled = np.flatnonzero((ids[:-1] == vocab[old]) & (ids[1:] == vocab["{"]))
            ids[relabeled] = vocab[new]
            yield AddedRow(ids, xy, pen, V3_ALIAS_SOURCE_ID, int(split.writers[row]))
            count += 1
            if count == cap:
                break

    # Find plausible function starts in the flat buffer, then check full math-token boundaries row by row.
    first = split.tokens[:-1]
    second = split.tokens[1:]
    candidates = np.zeros(len(first), dtype=np.bool_)
    for a, b in sorted({word[:2] for word in FUNCTION_WORDS}):
        if a in vocab and b in vocab:
            candidates |= (first == vocab[a]) & (second == vocab[b])
    for row in _rows_at_token_positions(split, np.flatnonzero(candidates)):
        original = [tokenizer.tokens[t] for t in split.label(row)]
        updated = relabel_function_words(original)
        if updated != original and all(tok in vocab for tok in updated):
            xy, pen = split.ink(row)
            if _ink_digest(xy, pen) in val_ink:
                continue
            ids = np.fromiter((vocab[tok] for tok in updated), dtype=np.uint16)
            yield AddedRow(ids, xy, pen, V3_ALIAS_SOURCE_ID, int(split.writers[row]))

    markers = [vocab[mark] for mark in "^_" if mark in vocab]
    rows = _rows_at_token_positions(split, np.flatnonzero(np.isin(split.tokens, markers)))
    count = 0
    for row in rng.permutation(rows):
        original = split.label(row)
        ids = expand_single_token_braces(original, tokenizer, max_len)
        if len(ids) != len(original):
            xy, pen = split.ink(row)
            if _ink_digest(xy, pen) in val_ink:
                continue
            yield AddedRow(np.asarray(ids, dtype=np.uint16), xy, pen, V3_ALIAS_SOURCE_ID, int(split.writers[row]))
            count += 1
            if count == 15_000:
                break


@dataclass(frozen=True)
class Glyph:
    symbol: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Box:
    glyphs: tuple[Glyph, ...]
    width: float

    @property
    def top(self) -> float:
        return min(glyph.y for glyph in self.glyphs)

    @property
    def bottom(self) -> float:
        return max(glyph.y + glyph.height for glyph in self.glyphs)

    def place(self, x: float, y: float = 0.0, scale: float = 1.0) -> tuple[Glyph, ...]:
        return tuple(Glyph(g.symbol, x + g.x * scale, y + g.y * scale, g.width * scale, g.height * scale) for g in self.glyphs)


@dataclass(frozen=True)
class Donors:
    xy: np.ndarray
    pen: np.ndarray
    aspect: float


def _pack_donors(strokes: Sequence[tuple[list[np.ndarray], float, float]]) -> Donors:
    length = max(sum(map(len, glyph)) for glyph, _, _ in strokes)
    xy = np.zeros((len(strokes), length, 2), dtype=np.float32)
    pen = np.full((len(strokes), length), 255, dtype=np.uint8)
    for i, (glyph, width, height) in enumerate(strokes):
        endpoints = np.cumsum([len(stroke) for stroke in glyph])
        xy[i, : endpoints[-1]] = np.concatenate(glyph) / [width, height]
        pen[i, : endpoints[-1]] = 0
        pen[i, endpoints - 1] = 1
    return Donors(xy, pen, float(np.median([w / h for _, w, h in strokes])))


def _load_donors(
    split: PackedSplit, tokenizer: InkForgeTokenizer, rng: np.random.Generator, *, val_ink: set[bytes] | None = None,
) -> dict[str, Donors]:
    rows = np.flatnonzero((np.diff(split.token_offsets) == 3) & (split.sources <= 5))
    by_token: dict[int, list[int]] = defaultdict(list)
    for row in rng.permutation(rows):
        tok = int(split.tokens[split.token_offsets[row] + 1])
        if tok > tokenizer.space_id and len(by_token[tok]) < 48:
            if val_ink and _ink_digest(*split.ink(row)) in val_ink:
                continue
            by_token[tok].append(int(row))
    bank = {}
    for tok, glyph_rows in by_token.items():
        normal = []
        for row in glyph_rows:
            glyph, width, height = _normalize_donor_glyph(unpack_points_to_strokes(*split.ink(row)))
            if sum(map(len, glyph)) <= 96:
                normal.append((glyph, width, height))
        if normal:
            bank[tokenizer.tokens[tok]] = _pack_donors(normal)

    ring = np.linspace(0.15 * np.pi, 2.0 * np.pi, 25)
    paths = {
        "line": [np.array([[0, 0.4], [0.25, 0.55], [0.5, 0.5], [0.75, 0.54], [1, 0.45]])],
        "box": [np.array([[0, 0.02], [0.51, 0], [1, 0.02], [0.99, 0.54], [1, 1], [0.52, 0.99], [0, 1], [0.01, 0.48], [0, 0.02]])],
        "ring": [np.column_stack((0.5 + 0.49 * np.cos(ring), 0.5 + 0.49 * np.sin(ring)))],
        "vertical": [np.array([[0.5, 0], [0.52, 0.4], [0.48, 0.7], [0.5, 1]])],
        "diagonal": [np.array([[1, 0], [0.67, 0.34], [0.33, 0.67], [0, 1]])],
    }
    bank.update({name: _pack_donors([_normalize_donor_glyph(strokes)]) for name, strokes in paths.items()})
    return bank


class Layout:
    """Compile the fixed LaTeX template grammar into positioned donor glyphs."""

    def __init__(self, tokens: Sequence[str], bank: dict[str, Donors]):
        self.tokens = iter(tokens)
        self.next = next(self.tokens, None)
        self.bank = bank

    def take(self) -> str:
        tok = self.next
        if tok is None:
            raise ValueError("Unexpected end of composition template")
        self.next = next(self.tokens, None)
        return tok

    def glyph(self, tok: str) -> Box:
        if tok in ("<SPACE>", "~"):
            return Box((), 0.48)
        if tok == "@":
            inner = self.glyph("a")
            return Box((*inner.place(0.28, -0.10, 0.68), Glyph("ring", 0, -0.93, 1.15, 1.14)), 1.25)
        sym = _GLYPH_ALIASES.get(tok, tok)
        aspect = self.bank[sym].aspect
        height = 1.0
        bottom = 0.0
        if tok in _SHORT_LETTERS:
            height = 0.73
        elif tok in _DESCENDERS:
            height, bottom = 1.0, 0.22
        elif tok in _SMALL_OPERATORS or tok in (".", ","):
            height = 0.14 if tok in (".", r"\cdot") else float(np.clip(0.76 / aspect, 0.16, 0.79))
            bottom = 0.12 if tok == "," else 0.0 if tok == "." else -0.44 + height / 2
        elif tok in (r"\sum", r"\prod", r"\int", "(", ")", "|"):
            height, bottom = 1.2, 0.12
        width = float(np.clip(aspect * height, 0.09, 1.65))
        return Box((Glyph(sym, 0, bottom - height, width, height),), width + 0.13)

    def atom(self) -> Box:
        tok = self.take()
        if tok == "{":
            group = self.sequence()
            if self.take() != "}":
                raise ValueError("Unclosed composition template group")
            return group
        if tok.startswith("\\") and tok[1:] in _WORD_OPERATORS:
            return self.join(self.glyph(letter) for letter in tok[1:])
        if tok in (r"\frac", r"\binom"):
            num, den = self.atom(), self.atom()
            scale = 0.72
            width = max(num.width, den.width) * scale + 0.24
            glyphs = (*num.place((width - scale * num.width) / 2, -0.64 - num.bottom * scale, scale),
                      *den.place((width - scale * den.width) / 2, -0.29 - den.top * scale, scale))
            if tok == r"\frac":
                return Box((*glyphs, Glyph("line", 0, -0.50, width, 0.035)), width + 0.12)
            return Box((Glyph("(", 0, -1.44, 0.34, 1.95), *Box(glyphs, width).place(0.38),
                        Glyph(")", width + 0.35, -1.44, 0.34, 1.95)), width + 0.77)
        if tok in _ACCENTS:
            child = self.atom()
            width = max(0.26, child.width - 0.11)
            if tok == r"\ddot":
                marks = (Glyph(".", width * 0.27, child.top - 0.27, 0.10, 0.10),
                         Glyph(".", width * 0.70, child.top - 0.27, 0.10, 0.10))
            else:
                height = 0.045 if tok == r"\bar" else 0.25
                symbol = (r"\overline" if r"\overline" in self.bank else "line") if tok == r"\bar" else _GLYPH_ALIASES.get(tok, tok)
                marks = (Glyph(symbol, 0.02, child.top - height - 0.09, width, height),)
            return Box((*child.glyphs, *marks), child.width)
        if tok in _INLINE_TEXT or tok == r"\boxed":
            child = self.atom()
            if tok == r"\boxed":
                return Box((*child.place(0.17), Glyph("box", 0, child.top - 0.18, child.width + 0.24,
                                                       child.bottom - child.top + 0.36)), child.width + 0.40)
            if tok == r"\mathbb":
                g = child.glyphs[0]
                if g.symbol == "Z":
                    mark = Glyph("diagonal", g.x + g.width * 0.18, g.y + 0.05, g.width * 0.68, g.height - 0.10)
                else:
                    mark = Glyph("vertical", g.x + g.width * 0.20, g.y, 0.045, g.height)
                return Box((*child.glyphs, mark), child.width)
            if tok == r"\mathbf":
                return Box(child.place(0, scale=1.09), child.width * 1.09)
            return child
        return self.glyph(tok)

    @staticmethod
    def join(boxes: Iterator[Box]) -> Box:
        glyphs: list[Glyph] = []
        width = 0.0
        for box in boxes:
            glyphs.extend(box.place(width))
            width += box.width
        return Box(tuple(glyphs), width)

    def sequence(self) -> Box:
        def atoms() -> Iterator[Box]:
            while self.next not in (None, "}"):
                is_limit = self.next in _LIMIT_OPERATORS
                base = self.atom()
                scripts: dict[str, Box] = {}
                while self.next in ("^", "_"):
                    mark = self.take()
                    scripts[mark] = self.atom()
                if not scripts:
                    yield base
                    continue
                scale = 0.53 if is_limit else 0.63
                sw = max(script.width for script in scripts.values()) * scale
                width = max(base.width, sw) if is_limit else base.width + sw
                glyphs = list(base.place((width - base.width) / 2 if is_limit else 0))
                for mark, script in scripts.items():
                    x = (width - scale * script.width) / 2 if is_limit else base.width - 0.06
                    if mark == "^":
                        y = (base.top - 0.08 if is_limit else -0.58) - scale * script.bottom
                    else:
                        y = (base.bottom + 0.08 if is_limit else -0.18) - scale * script.top
                    glyphs.extend(script.place(x, y, scale))
                yield Box(tuple(glyphs), width)
        return self.join(atoms())


class DonorLayout:
    """Sample independent real glyphs in bulk and perturb their placement per row."""

    def __init__(self, box: Box, bank: dict[str, Donors]):
        glyphs = box.glyphs
        self.origins = np.asarray([(g.x, g.y) for g in glyphs], dtype=np.float32)
        self.size = np.asarray([(g.width, g.height) for g in glyphs], dtype=np.float32)
        donors = [bank[g.symbol] for g in glyphs]
        self.choices = np.asarray([len(donor.xy) for donor in donors])
        self.glyphs = np.arange(len(donors))
        shape = (len(donors), int(max(self.choices)), max(donor.xy.shape[1] for donor in donors))
        self.xy = np.zeros((*shape, 2), dtype=np.float32)
        self.pen = np.full(shape, 255, dtype=np.uint8)
        for i, donor in enumerate(donors):
            count, length = donor.pen.shape
            self.xy[i, :count, :length] = donor.xy
            self.pen[i, :count, :length] = donor.pen

    def sample(self, count: int, max_points: int, rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        picks = rng.integers(self.choices, size=(count, len(self.glyphs)))
        xy = self.xy[self.glyphs, picks]
        pen = self.pen[self.glyphs, picks]
        variation = rng.uniform(0.94, 1.06, size=(count, len(self.glyphs), 2)).astype(np.float32)
        xy *= (self.size * variation)[:, :, None, :]
        origins = np.broadcast_to(self.origins, variation.shape).copy()
        x_pos = origins[..., 0]
        x_pos *= rng.uniform(0.91, 1.10, size=(count, 1)).astype(np.float32)
        phase = rng.uniform(0, 2 * np.pi, size=(count, 1)).astype(np.float32)
        origins[..., 1] += 0.07 * np.sin(1.3 * x_pos + phase)
        x_pos += 0.05 * np.sin(1.8 * x_pos + phase)
        origins += rng.normal(0, 0.022, size=origins.shape).astype(np.float32)
        xy += origins[:, :, None, :]
        xy[..., 0] += xy[..., 1] * rng.uniform(-0.13, 0.16, size=(count, 1, 1)).astype(np.float32)
        mask = pen != 255
        offsets = np.concatenate(([0], np.cumsum(np.count_nonzero(mask, axis=(1, 2)))))
        xy, pen = xy[mask], pen[mask]
        for start, end in zip(offsets[:-1], offsets[1:]):
            coords, states = xy[start:end], pen[start:end]
            if len(coords) > max_points:
                idx = np.arange(len(coords))
                endpoints = (states == 1) | np.r_[True, states[:-1] == 1]
                keep = endpoints | (idx % (1 + len(coords) // max_points) == 0)
                coords, states = coords[keep], states[keep]
            if len(coords) < 2 or len(coords) > max_points or states[0] != 0:
                continue
            coords -= coords[:1]
            scale = max(1.0, float(np.abs(coords[:, 0]).max()) / 23, float(np.abs(coords[:, 1]).max()) / 11)
            yield (coords / scale).astype(np.float16), states.copy()


def _compose_rows(
    bank: dict[str, Donors], tokenizer: InkForgeTokenizer, val_ink: set[bytes], *, count: int,
    max_points: int, max_text_len: int, rng: np.random.Generator,
    templates: Sequence[str] = COMPOSITION_TEMPLATES,
) -> Iterator[AddedRow]:
    compiled = []
    missing: set[str] = set()
    for latex in templates:
        tokens = tokenize_inkforge(latex)
        if latex.startswith(r"\text{if }"):
            tokens.insert(tokens.index("}"), "<SPACE>")
        # Include closely related labels while keeping the original expressions in the training set.
        alternate = {"x": "u", "y": "v"}
        variants = (tokens, [alternate.get(token, token) for token in tokens]) if "x" in tokens else (tokens,)
        for variant in variants:
            try:
                ids = [tokenizer.bos_id, *(tokenizer.token_to_id[tok] for tok in variant), tokenizer.eos_id]
                box = Layout(variant, bank).sequence()
                layout = DonorLayout(box, bank)
            except KeyError as exc:
                missing.add(str(exc.args[0]))
                continue
            if len(ids) > max_text_len:
                continue
            braced = expand_single_token_braces(ids, tokenizer, max_text_len)
            compiled.append((np.asarray(ids, dtype=np.uint16), np.asarray(braced, dtype=np.uint16), layout))
    if count and not compiled:
        raise ValueError(f"Cannot compose any templates. Missing tokenizer tokens or donors: {sorted(missing)}")
    if missing:
        print(f"[v3-pack] Skipped layouts with missing tokens or donors: {sorted(missing)}", flush=True)
    for i, (plain, braced, layout) in enumerate(compiled):
        target = count // len(compiled) + (i < count % len(compiled))
        added = 0
        attempts = 0
        while added < target and attempts < max(256, target * 4):
            requested = min(128, target - added)
            attempts += requested
            for xy, pen in layout.sample(requested, max_points, rng):
                if _ink_digest(xy, pen) in val_ink:
                    continue
                yield AddedRow(braced if added % 2 else plain, xy, pen, V3_COMPOSED_SOURCE_ID)
                added += 1
        if added < target:
            raise ValueError(f"Generated {added}/{target} valid rows for {tokenizer.decode(plain)!r}")


def _write_train(path: Path, train: PackedSplit, additions: Sequence[AddedRow]) -> None:
    path.mkdir()
    point_offsets = np.append(train.point_offsets, len(train.xy) + np.cumsum([len(row.xy) for row in additions], dtype=np.int64))
    token_offsets = np.append(train.token_offsets, len(train.tokens) + np.cumsum([len(row.tokens) for row in additions], dtype=np.int64))
    point_total, token_total = int(point_offsets[-1]), int(token_offsets[-1])
    xy = np.lib.format.open_memmap(path / "points_xy.npy", mode="w+", dtype=train.xy.dtype, shape=(point_total, 2))
    pen = np.lib.format.open_memmap(path / "points_pen.npy", mode="w+", dtype=train.pen.dtype, shape=(point_total,))
    tokens = np.lib.format.open_memmap(path / "tokens.npy", mode="w+", dtype=train.tokens.dtype, shape=(token_total,))
    xy[: len(train.xy)], pen[: len(train.pen)], tokens[: len(train.tokens)] = train.xy, train.pen, train.tokens
    for i, row in enumerate(additions, len(train)):
        p0, p1 = point_offsets[i : i + 2]
        t0, t1 = token_offsets[i : i + 2]
        xy[p0:p1], pen[p0:p1], tokens[t0:t1] = row.xy, row.pen, row.tokens
    for array in (xy, pen, tokens):
        array.flush()
    np.save(path / "pt_offsets.npy", point_offsets)
    np.save(path / "tok_offsets.npy", token_offsets)
    np.save(path / "source_ids.npy", np.concatenate((train.sources, np.asarray([row.source for row in additions], dtype=train.sources.dtype))))
    np.save(path / "writer_ids.npy", np.concatenate((train.writers, np.asarray([row.writer for row in additions], dtype=train.writers.dtype))))
    (path / "writer_names.json").write_text(json.dumps(train.writer_names, ensure_ascii=False), encoding="utf-8")


def build_data_v3(
    base_cache: Path, out_dir: Path, *, composed_count: int = 95_000, seed: int = 42,
) -> Path:
    """Write the v3 cache atomically, retaining the exact v2 train prefix and val files."""
    start = time.perf_counter()
    base_cache, out_dir = Path(base_cache).resolve(), Path(out_dir).resolve()
    if out_dir.is_relative_to(base_cache) or base_cache.is_relative_to(out_dir):
        raise ValueError("Base and output cache paths must be separate")
    if out_dir.exists():
        raise FileExistsError(f"Output cache already exists: {out_dir}")
    if composed_count < 0:
        raise ValueError("composed_count must be nonnegative")
    meta = json.loads((base_cache / "meta.json").read_text(encoding="utf-8"))
    tokenizer = InkForgeTokenizer.load(base_cache / "tokenizer.json")
    train, val = PackedSplit(base_cache / "train"), PackedSplit(base_cache / "val")
    train.verify(len(tokenizer))
    val.verify(len(tokenizer))
    if np.isin(train.sources, list(V3_SOURCE_NAMES)).any():
        raise ValueError("Base cache already contains v3 source IDs")
    val_ink = _verify_base_leakage(train, val)
    print(f"[v3-pack] Verified base splits: {len(train):,} train, {len(val):,} validation", flush=True)

    rng = np.random.default_rng(seed)
    max_text_len = int(meta.get("max_text_len", 160))
    additions = list(_alias_rows(train, tokenizer, val_ink, max_text_len, rng))
    print(f"[v3-pack] Selected {len(additions):,} alias and bracing rows", flush=True)
    bank = _load_donors(train, tokenizer, rng, val_ink=val_ink)
    additions.extend(_compose_rows(bank, tokenizer, val_ink, count=composed_count,
                                   max_points=int(meta.get("max_points", 512)), max_text_len=max_text_len, rng=rng))
    print(f"[v3-pack] Composed {composed_count:,} train-donor rows", flush=True)

    before = np.bincount(train.tokens, minlength=len(tokenizer))
    appended_tokens = np.concatenate([row.tokens for row in additions]) if additions else np.empty(0, dtype=np.uint16)
    after = before + np.bincount(appended_tokens, minlength=len(tokenizer))
    report = {
        tok: {"before": int(before[tid]), "after": int(after[tid])} if tid is not None else {"before": 0, "after": 0}
        for tok in REPORTED_TOKENS for tid in (tokenizer.token_to_id.get(tok),)
    }
    uncovered = [tok for tok, counts in report.items() if not counts["after"]]
    if uncovered:
        raise ValueError(f"Training token frequencies are still zero: {uncovered}")
    for tok, counts in report.items():
        print(f"  {tok:>12} {counts['before']:>7,} -> {counts['after']:>7,}")

    sources = list(meta.get("sources", SOURCE_NAMES))
    while len(sources) <= V3_COMPOSED_SOURCE_ID:
        sid = len(sources)
        sources.append(V3_SOURCE_NAMES.get(sid, f"source_{sid}"))
    for sid, name in V3_SOURCE_NAMES.items():
        sources[sid] = name
    totals = Counter(map(int, train.sources))
    totals.update(row.source for row in additions)
    meta.update({
        "format": "inkforge-cache-v3",
        "vocab_size": len(tokenizer),
        "train_samples": len(train) + len(additions),
        "val_samples": len(val),
        "base_train_samples": len(train),
        "sources": sources,
        "train_source_counts": {sources[sid] if sid < len(sources) else f"source_{sid}": n for sid, n in sorted(totals.items())},
        "token_frequencies": report,
        "leak_free_grouped": True,
        "val_human_only": True,
    })

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}-", dir=out_dir.parent) as stage:
        packed = Path(stage) / "cache"
        packed.mkdir()
        _write_train(packed / "train", train, additions)
        PackedSplit(packed / "train").verify(len(tokenizer))
        shutil.copytree(base_cache / "val", packed / "val")
        for path in (base_cache / "val").rglob("*"):
            if path.is_file() and not filecmp.cmp(path, packed / "val" / path.relative_to(base_cache / "val"), shallow=False):
                raise OSError(f"Copied validation file differs: {path.name}")
        shutil.copyfile(base_cache / "tokenizer.json", packed / "tokenizer.json")
        meta["elapsed_seconds"] = round(time.perf_counter() - start, 1)
        (packed / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        packed.rename(out_dir)
    print(f"[v3-pack] Verified finite points and unchanged, leak-free validation. "
          f"Wrote {meta['train_samples']:,} train rows in {meta['elapsed_seconds']}s to {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_V2_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V3_CACHE_DIR)
    parser.add_argument("--composed-count", type=int, default=95_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_data_v3(args.base_cache, args.out_dir, composed_count=args.composed_count, seed=args.seed)


if __name__ == "__main__":
    main()
