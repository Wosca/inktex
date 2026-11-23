from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable


SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
CORE_COMMANDS = (
    r"\frac", r"\sqrt", r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\pi",
    r"\theta", r"\alpha", r"\beta", r"\gamma", r"\delta", r"\lambda", r"\mu",
    r"\sigma", r"\phi", r"\int", r"\sum", r"\prod", r"\lim", r"\infty",
    r"\left", r"\right", r"\cdot", r"\times", r"\div", r"\pm", r"\leq", r"\geq",
    r"\neq", r"\approx", r"\partial", r"\mathrm", r"\mathbf", r"\overline", r"\vec",
    r"\prime", r"\rightarrow", r"\to",
)
CORE_SYMBOLS = tuple("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]()^_+-=*/,.|!<>:&%")
TOKEN_PATTERN = re.compile(r"\\[A-Za-z]+|\\.|\d|[A-Za-z]|[^\s]")
CONTROL_WORD_PATTERN = re.compile(r"\\[A-Za-z]+\Z")
SEMANTIC_ALIASES = {
    r"\to": r"\rightarrow",
    r"\dfrac": r"\frac",
    r"\tfrac": r"\frac",
    r"\le": r"\leq",
    r"\ge": r"\geq",
    r"\ne": r"\neq",
    r"\cdotp": r"\cdot",
}
PRESENTATION_ONLY_TOKENS = {
    r"\,", r"\!", r"\:", r"\;", r"\ ", r"\quad", r"\qquad", r"\left", r"\right",
}


def tokenize_latex(latex: str) -> list[str]:
    # Spaces are insignificant in math, but carry word boundaries inside text.
    result = []
    depth = 0
    text_depth = None
    pending_text = False
    end = 0
    for match in TOKEN_PATTERN.finditer(latex.strip()):
        token = match[0]
        if text_depth is not None and match.start() > end:
            result.append(" ")
        if token == "{":
            depth += 1
            if pending_text:
                text_depth = depth
                pending_text = False
        elif token == "}":
            if text_depth == depth:
                text_depth = None
            depth -= 1
        if token in (r"\text", r"\mbox", r"\operatorname"):
            pending_text = True
        result.append(token)
        end = match.end()
    return result


def detokenize_latex(tokens: Iterable[str]) -> str:
    """Join tokens without accidentally merging a TeX control word with a letter."""
    result: list[str] = []
    previous: str | None = None
    for token in tokens:
        if previous is not None and CONTROL_WORD_PATTERN.fullmatch(previous) and token[:1].isalpha():
            result.append(" ")
        result.append(token)
        previous = token
    return "".join(result)


def normalize_latex(latex: str) -> str:
    """Canonicalize whitespace while retaining separators required by TeX."""
    return detokenize_latex(tokenize_latex(latex))


def canonicalize_latex_tokens(latex: str) -> list[str]:
    """Return a conservative semantic token form for evaluation.

    It removes presentation-only spacing, maps common command aliases, converts
    apostrophe primes, and removes braces only when they wrap one atom. It does
    not use algebraic identities or reorder expressions.
    """
    tokens = tokenize_latex(latex)

    # Model predictions can contain hundreds of nested groups. Use an explicit
    # stack so scoring never depends on Python's call-stack limit.
    groups: list[list[str]] = [[]]

    def close_group():
        inner = groups.pop()
        groups[-1].extend(inner if len(inner) == 1 else ["{", *inner, "}"])

    for token in tokens:
        if token == '{':
            groups.append([])
        elif token == '}' and len(groups) > 1:
            close_group()
        elif token in PRESENTATION_ONLY_TOKENS:
            continue
        elif token == "'":
            groups[-1].extend(('^', r'\prime'))
        else:
            groups[-1].append(SEMANTIC_ALIASES.get(token, token))
    while len(groups) > 1:
        close_group()
    return groups[0]


class LatexTokenizer:
    def __init__(self, tokens: Iterable[str] | None = None):
        self.label_normalization = None
        ordered = list(SPECIAL_TOKENS)
        for token in (*CORE_COMMANDS, *CORE_SYMBOLS, *(tokens or [])):
            if token not in ordered:
                ordered.append(token)
        self.tokens = ordered
        self.token_to_id = {token: index for index, token in enumerate(ordered)}

    @classmethod
    def build(cls, latex_values: Iterable[str], min_frequency: int = 1) -> "LatexTokenizer":
        counts = Counter(token for latex in latex_values for token in tokenize_latex(latex))
        learned = sorted(token for token, count in counts.items() if count >= min_frequency)
        return cls(learned)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def __len__(self) -> int:
        return len(self.tokens)

    def add_tokens(self, tokens: Iterable[str]) -> list[str]:
        added: list[str] = []
        for token in sorted(set(tokens)):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.tokens)
                self.tokens.append(token)
                added.append(token)
        return added

    def encode(self, latex: str, add_special_tokens: bool = True) -> list[int]:
        if self.label_normalization:
            from .normalization import normalize_training_latex
            latex = normalize_training_latex(latex)
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize_latex(latex)]
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        result: list[str] = []
        for index in ids:
            token = self.tokens[int(index)] if 0 <= int(index) < len(self.tokens) else "<UNK>"
            if token == "<EOS>" and skip_special_tokens:
                break
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            result.append(token)
        return detokenize_latex(result)

    def ids_to_tokens(self, ids: Iterable[int], skip_special_tokens: bool = True) -> list[str]:
        values: list[str] = []
        for index in ids:
            token = self.tokens[int(index)] if 0 <= int(index) < len(self.tokens) else "<UNK>"
            if token == "<EOS>" and skip_special_tokens:
                break
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            values.append(token)
        return values

    def to_dict(self) -> dict[str, object]:
        return {"version": 2, "tokens": self.tokens, "label_normalization": self.label_normalization}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LatexTokenizer":
        instance = cls([])
        instance.tokens = [str(token) for token in value["tokens"]]  # type: ignore[index]
        instance.token_to_id = {token: index for index, token in enumerate(instance.tokens)}
        instance.label_normalization = value.get("label_normalization")
        if instance.label_normalization:
            from .normalization import LABEL_VERSION
            if instance.label_normalization != LABEL_VERSION:
                raise ValueError('Unsupported checkpoint label normalization')
        return instance

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LatexTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
