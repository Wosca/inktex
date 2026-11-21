"""Minimal LaTeX tokenizer for InkTeX checkpoint loading and decoding."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable

SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
CONTROL_WORD_PATTERN = re.compile(r"\\[A-Za-z]+\Z")


def detokenize_latex(tokens: Iterable[str]) -> str:
    """Join tokens without merging a TeX control word with a following letter."""
    result: list[str] = []
    previous: str | None = None
    for token in tokens:
        if previous is not None and CONTROL_WORD_PATTERN.fullmatch(previous) and token[:1].isalpha():
            result.append(" ")
        result.append(token)
        previous = token
    return "".join(result)


class LatexTokenizer:
    def __init__(self, tokens: list[str], label_normalization: str | None = None):
        self.tokens = [str(t) for t in tokens]
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.label_normalization = label_normalization

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

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        result: list[str] = []
        for index in ids:
            idx = int(index)
            token = self.tokens[idx] if 0 <= idx < len(self.tokens) else "<UNK>"
            if token == "<EOS>" and skip_special_tokens:
                break
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            result.append(token)
        return detokenize_latex(result)

    def ids_to_tokens(self, ids: Iterable[int], skip_special_tokens: bool = True) -> list[str]:
        result: list[str] = []
        for index in ids:
            idx = int(index)
            token = self.tokens[idx] if 0 <= idx < len(self.tokens) else "<UNK>"
            if token == "<EOS>" and skip_special_tokens:
                break
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            result.append(token)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "tokens": self.tokens,
            "label_normalization": self.label_normalization,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LatexTokenizer":
        return cls(
            tokens=[str(t) for t in value["tokens"]],  # type: ignore[index]
            label_normalization=value.get("label_normalization"),  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: str | Path) -> "LatexTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
