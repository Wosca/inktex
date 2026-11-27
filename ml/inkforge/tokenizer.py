"""Expanded LaTeX + inline text tokenizer for InkForge handwriting generation.

Preserves explicit spaces inside `\\text{...}` (via `<SPACE>`) so IAMonDo words & math text phrases
can be generated naturally alongside full LaTeX math commands, Greek letters, and structures.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SPACE>")

# Always keep full printable ASCII + core LaTeX math commands in the base vocabulary
_BASE_ASCII = list(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "{}[]()^_+-=*/,.|!<>:&'?;:@#%\"~"
)
_CORE_LATEX = [
    r"\frac", r"\sqrt", r"\sum", r"\int", r"\iint", r"\oint", r"\prod", r"\lim",
    r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc", r"\arcsin", r"\arccos", r"\arctan",
    r"\sinh", r"\cosh", r"\tanh", r"\log", r"\ln", r"\exp", r"\det", r"\max", r"\min", r"\gcd",
    r"\Pr", r"\arg", r"\dim", r"\ker", r"\deg",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", r"\zeta", r"\eta",
    r"\theta", r"\vartheta", r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi",
    r"\rho", r"\sigma", r"\tau", r"\upsilon", r"\phi", r"\varphi", r"\chi", r"\psi", r"\omega",
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi", r"\Pi", r"\Sigma", r"\Upsilon",
    r"\Phi", r"\Psi", r"\Omega",
    r"\infty", r"\partial", r"\nabla", r"\hbar", r"\ell", r"\emptyset", r"\aleph", r"\angle",
    r"\pm", r"\mp", r"\times", r"\cdot", r"\div", r"\ast", r"\circ", r"\bullet", r"\oplus", r"\otimes",
    r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\sim", r"\simeq", r"\cong", r"\propto",
    r"\ll", r"\gg", r"\in", r"\notin", r"\ni", r"\subset", r"\subseteq", r"\supset", r"\supseteq",
    r"\cup", r"\cap", r"\setminus", r"\forall", r"\exists", r"\neg", r"\wedge", r"\vee",
    r"\rightarrow", r"\leftarrow", r"\leftrightarrow", r"\Rightarrow", r"\Leftarrow", r"\Leftrightarrow",
    r"\mapsto", r"\uparrow", r"\downarrow", r"\perp", r"\parallel", r"\mid", r"\vdash", r"\therefore",
    r"\because", r"\ldots", r"\cdots", r"\dots", r"\vdots", r"\ddots",
    r"\vec", r"\hat", r"\widehat", r"\bar", r"\overline", r"\underline", r"\tilde", r"\widetilde",
    r"\dot", r"\ddot", r"\prime", r"\mathbb", r"\mathcal", r"\mathbf", r"\mathrm", r"\text",
    r"\boxed", r"\binom", r"\{", r"\}", r"\|", r"\#", r"\%", r"\_", r"\\",
    r"\begin", r"\end",
]

_ALIAS_MAP = {
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\ne": r"\neq",
    r"\to": r"\rightarrow",
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\cdotp": r"\cdot",
    r"\vert": "|",
    r"\lbrack": "[",
    r"\rbrack": "]",
    r"\mbox": r"\text",
    r"\operatorname": r"\text",
    r"\textrm": r"\text",
}

_DROP_TOKENS = frozenset({
    r"\left", r"\right", r"\limits", r"\nolimits", r"\displaystyle", r"\textstyle",
    r"\,", r"\;", r"\:", r"\!", r"\quad", r"\qquad",
})


def tokenize_inkforge(latex: str) -> list[str]:
    """Tokenize LaTeX + `\\text{...}` into command/character tokens, preserving spaces inside `\\text{...}`."""
    s = (latex or "").strip()
    if not s:
        return []

    tokens: list[str] = []
    n = len(s)
    i = 0
    in_text_brace_depth = 0
    pending_text = False

    while i < n:
        ch = s[i]
        if ch.isspace():
            if in_text_brace_depth > 0:
                if tokens and tokens[-1] != "<SPACE>":
                    tokens.append("<SPACE>")
            i += 1
            continue

        if ch == "\\":
            j = i + 1
            if j < n:
                if s[j].isalpha():
                    while j < n and s[j].isalpha():
                        j += 1
                    cmd = s[i:j]
                else:
                    cmd = s[i : j + 1]
                    j += 1
            else:
                cmd = "\\"
            i = j
            if cmd in _DROP_TOKENS:
                continue
            cmd = _ALIAS_MAP.get(cmd, cmd)
            if cmd == r"\text":
                pending_text = True
            tokens.append(cmd)
            continue

        if ch == "{":
            if pending_text:
                in_text_brace_depth = 1
                pending_text = False
            elif in_text_brace_depth > 0:
                in_text_brace_depth += 1
            tokens.append(ch)
            i += 1
            continue

        if ch == "}":
            if in_text_brace_depth > 0:
                in_text_brace_depth -= 1
                if tokens and tokens[-1] == "<SPACE>":
                    tokens.pop()
            tokens.append(ch)
            i += 1
            continue

        pending_text = False
        tokens.append(ch)
        i += 1

    # Collapse single-token brace groups after ^ or _ (e.g., x^{2} -> x^2) for cleaner generation
    collapsed: list[str] = []
    k = 0
    m = len(tokens)
    while k < m:
        if (
            tokens[k] in ("^", "_")
            and k + 3 < m
            and tokens[k + 1] == "{"
            and tokens[k + 3] == "}"
            and tokens[k + 2] not in ("{", "}", "<SPACE>")
        ):
            collapsed.append(tokens[k])
            collapsed.append(tokens[k + 2])
            k += 4
            continue
        collapsed.append(tokens[k])
        k += 1

    return collapsed


def detokenize_inkforge(tokens: Sequence[str]) -> str:
    """Reconstruct readable LaTeX from InkForge tokens."""
    out: list[str] = []
    prev: str | None = None
    for tok in tokens:
        if tok in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
            continue
        if tok == "<SPACE>":
            out.append(" ")
            prev = " "
            continue
        if prev and prev.startswith("\\") and len(prev) >= 2 and prev[1:].isalpha() and tok and tok[0].isalpha():
            out.append(" ")
        out.append(tok)
        prev = tok
    return "".join(out)


class InkForgeTokenizer:
    """Compact yet expressive token vocabulary for InkForge (~400-550 tokens)."""

    def __init__(self, tokens: Iterable[str] | None = None):
        ordered = list(SPECIAL_TOKENS)
        seen = set(ordered)
        for tok in (tokens or []):
            if tok not in seen:
                ordered.append(tok)
                seen.add(tok)
        self.tokens: list[str] = ordered
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(ordered)}

    @classmethod
    def build(cls, labels: Iterable[str], *, min_frequency: int = 10, max_vocab: int = 600) -> "InkForgeTokenizer":
        counts: Counter[str] = Counter()
        for lab in labels:
            counts.update(tokenize_inkforge(lab))
        kept = set(_BASE_ASCII) | set(_CORE_LATEX)
        for tok, c in counts.most_common():
            if len(kept) + len(SPECIAL_TOKENS) >= max_vocab:
                break
            if c >= min_frequency and tok not in SPECIAL_TOKENS:
                kept.add(tok)
        return cls(sorted(kept))

    @classmethod
    def default(cls) -> "InkForgeTokenizer":
        return cls(sorted(set(_BASE_ASCII) | set(_CORE_LATEX)))

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def space_id(self) -> int:
        return 4

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, latex: str, *, max_len: int = 160) -> list[int]:
        toks = tokenize_inkforge(latex)[: max(1, max_len - 2)]
        unk = self.unk_id
        return [self.bos_id, *(self.token_to_id.get(t, unk) for t in toks), self.eos_id]

    def has_unk(self, latex: str) -> bool:
        return any(t not in self.token_to_id for t in tokenize_inkforge(latex))

    def decode(self, ids: Sequence[int]) -> str:
        toks: list[str] = []
        for idx in ids:
            i = int(idx)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            if 0 <= i < len(self.tokens):
                toks.append(self.tokens[i])
        return detokenize_inkforge(toks)

    def to_dict(self) -> dict[str, object]:
        return {"format": "inkforge-tokenizer-v1", "tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "InkForgeTokenizer":
        toks = list(payload["tokens"])
        if tuple(toks[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"Unexpected InkForge special tokens: {toks[:5]!r}")
        return cls(toks[len(SPECIAL_TOKENS) :])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InkForgeTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
