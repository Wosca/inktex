"""InkTeX v2 Standardizer & Compact Bidirectional Tokenizer (~280-310 tokens).

Special tokens match InkTeXV2Config exactly:
    0: <PAD>
    1: <BOS_L2R>
    2: <BOS_R2L>
    3: <EOS>
    4: <UNK>

Built strictly AFTER standardize_latex_v2 so dead presentation/alias tokens
(\\left, \\right, \\to, \\mathrm, \\dfrac, etc.) never enter the vocabulary.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable, Sequence

from tokenizer.latex_tokenizer import detokenize_latex, tokenize_latex
from tokenizer.standardize import standardize_latex

STANDARD_V2_VERSION = "ink-standard-v2"
SPECIAL_TOKENS_V2 = ("<PAD>", "<BOS_L2R>", "<BOS_R2L>", "<EOS>", "<UNK>")
# Max body tokens = 198 so [BOS, *body, EOS] fits strictly inside 200 tokens everywhere!
MAX_BODY_TOKENS_V2 = 198
MAX_SEQ_TOKENS_V2 = MAX_BODY_TOKENS_V2 + 2  # 200

SYNTAX_ONLY_TOKENS = frozenset({
    "{", "}", "^", "_", r"\frac", r"\sqrt", r"\binom", r"\mbox", r"\mathbf",
    r"\mathbb", r"\mathcal", r"\mathsf", r"\overline", r"\underline",
    r"\vec", r"\hat", r"\widehat", r"\bar", r"\tilde", r"\widetilde",
    r"\dot", r"\ddot", r"\begin", r"\end", "&", r"\\", " ",
})

FORBIDDEN_V2_TOKENS = frozenset({
    r"\left", r"\right", r"\to", r"\dfrac", r"\tfrac", r"\le", r"\ge", r"\ne",
    r"\cdotp", r"\limits", r"\nolimits", r"\displaystyle", r"\textstyle",
    r"\mathrm", r"\mathit", r"\text", r"\operatorname",
})


@lru_cache(maxsize=1 << 16)
def standardize_latex_v2(latex: str, max_body_tokens: int = MAX_BODY_TOKENS_V2) -> str | None:
    """Canonicalize LaTeX for v2 training; returns None on malformed, non-ASCII,
    forbidden-token, or over-length labels."""
    if not latex or not latex.strip():
        return None
    std = standardize_latex(latex)
    if not std:
        return None
    tokens = tokenize_latex(std)
    if not tokens or len(tokens) > max_body_tokens:
        return None
    # Reject any surviving forbidden wrappers (e.g. nested \textbf{\mathrm{x}})
    if any(t in FORBIDDEN_V2_TOKENS for t in tokens):
        return None
    roundtrip = detokenize_latex(tokens)
    if tokenize_latex(roundtrip) != tokens:
        return None
    return std


def count_visible_tokens(latex: str) -> int:
    """Count visible math symbols, skipping environment names and array column specs."""
    tokens = tokenize_latex(latex)
    visible = 0
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in (r"\begin", r"\end"):
            i += 1
            # Skip `{env_name}` (and optional `{cols}` after `\begin{array}`)
            groups_to_skip = 1
            env_name = ""
            if i < n and tokens[i] == "{":
                j = i + 1
                while j < n and tokens[j] != "}":
                    env_name += tokens[j]
                    j += 1
                i = min(j + 1, n)
                if tok == r"\begin" and env_name == "array" and i < n and tokens[i] == "{":
                    j = i + 1
                    while j < n and tokens[j] != "}":
                        j += 1
                    i = min(j + 1, n)
            continue
        if tok not in SYNTAX_ONLY_TOKENS:
            visible += 1
        i += 1
    return visible


def is_plausible_stroke_ratio(latex: str, stroke_count: int) -> bool:
    """Reject severe training label-vs-ink mismatches (never applied to validation splits)."""
    if stroke_count <= 0:
        return False
    vis = max(count_visible_tokens(latex), 1)
    if vis >= 8 and stroke_count < max(2, int(vis * 0.18)):
        return False
    if stroke_count >= 15 and vis == 1:
        return False
    if stroke_count > max(24, vis * 10):
        return False
    return True


class LatexTokenizerV2:
    """Compact bidirectional LaTeX tokenizer for InkTeX v2."""

    def __init__(self, tokens: Iterable[str] | None = None):
        ordered = list(SPECIAL_TOKENS_V2)
        seen = set(ordered)
        for token in (tokens or []):
            if token not in seen and token not in FORBIDDEN_V2_TOKENS:
                ordered.append(token)
                seen.add(token)
        self.tokens: list[str] = ordered
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(ordered)}
        self.standard_version = STANDARD_V2_VERSION

    @classmethod
    def build(
        cls,
        standardized_labels: Iterable[str],
        *,
        min_frequency: int = 5,
        allowlist: Iterable[str] = (),
    ) -> "LatexTokenizerV2":
        counts: Counter[str] = Counter()
        for label in standardized_labels:
            counts.update(tokenize_latex(label))
        kept = {tok for tok, c in counts.items() if c >= min_frequency}
        kept.update(t for t in allowlist if t and t not in FORBIDDEN_V2_TOKENS)
        kept.update(list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]()^_+-=*/,.|!<>:&'"))
        return cls(sorted(kept))

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_l2r_id(self) -> int:
        return 1

    @property
    def bos_r2l_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def unk_id(self) -> int:
        return 4

    def __len__(self) -> int:
        return len(self.tokens)

    def encode_body(self, standardized_latex: str) -> list[int]:
        unk = self.unk_id
        return [self.token_to_id.get(tok, unk) for tok in tokenize_latex(standardized_latex)]

    def has_unk(self, standardized_latex: str) -> bool:
        return any(tok not in self.token_to_id for tok in tokenize_latex(standardized_latex))

    def encode_l2r(self, standardized_latex: str) -> list[int]:
        return [self.bos_l2r_id, *self.encode_body(standardized_latex), self.eos_id]

    def encode_r2l(self, standardized_latex: str) -> list[int]:
        return [self.bos_r2l_id, *reversed(self.encode_body(standardized_latex)), self.eos_id]

    def decode_tokens(self, token_ids: Sequence[int], *, is_r2l: bool = False) -> list[str]:
        """Return raw token list (preserving `<UNK>` as one single token for exact CER scoring)."""
        specials = {self.pad_id, self.bos_l2r_id, self.bos_r2l_id, self.eos_id}
        body: list[str] = []
        for idx in token_ids:
            i = int(idx)
            if i == self.eos_id:
                break
            if i in specials:
                continue
            if 0 <= i < len(self.tokens):
                body.append(self.tokens[i])
        if is_r2l:
            body.reverse()
        return body

    def decode(self, token_ids: Sequence[int], *, is_r2l: bool = False) -> str:
        return detokenize_latex(self.decode_tokens(token_ids, is_r2l=is_r2l))

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "inktex-tokenizer-v2",
            "standard_version": self.standard_version,
            "tokens": list(self.tokens),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LatexTokenizerV2":
        tokens = list(payload["tokens"])
        if tuple(tokens[: len(SPECIAL_TOKENS_V2)]) != SPECIAL_TOKENS_V2:
            raise ValueError(f"Tokenizer v2 special tokens mismatch: {tokens[:5]!r}")
        tok = cls(tokens[len(SPECIAL_TOKENS_V2) :])
        tok.standard_version = str(payload.get("standard_version", STANDARD_V2_VERSION))
        return tok

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LatexTokenizerV2":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
