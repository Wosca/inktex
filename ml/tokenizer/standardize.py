"""One spelling per symbol for CROHME and MathWriting training labels.

Runs on top of ``normalize_training_latex`` and never leaves the shipped 907-token vocabulary.
``\\text`` is not in that vocabulary, so ``\\mbox`` is the single text wrapper.
"""
from __future__ import annotations

from functools import lru_cache
import re

from .latex_tokenizer import detokenize_latex, tokenize_latex
from .normalization import ARGUMENTS, FUNCTIONS, normalize_training_latex

STANDARD_VERSION = "ink-standard-v1"
TEXT_COMMAND = r"\mbox"
_GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon "
          "phi chi psi omega").split()
UNICODE = {
    "·": r"\cdot", "⋅": r"\cdot", "∙": r"\cdot", "±": r"\pm", "∓": r"\mp", "′": "'", "″": "''", "‴": "'''",
    "−": "-", "–": "-", "×": r"\times", "÷": r"\div", "∞": r"\infty", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "≈": r"\approx", "≡": r"\equiv", "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "…": r"\ldots", "⋯": r"\ldots",
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "√": r"\sqrt", "∂": r"\partial", "∇": r"\nabla", "∈": r"\in",
    "∉": r"\notin", "∀": r"\forall", "∃": r"\exists", "∅": r"\emptyset", "∪": r"\cup", "∩": r"\cap",
    "⊂": r"\subset", "⊆": r"\subseteq", "∧": r"\wedge", "∨": r"\vee", "¬": r"\neg", "°": r"^{\circ}",
    "∠": r"\angle", "⊥": r"\perp", "∥": r"\parallel", "∝": r"\propto", "∼": r"\sim",
    **{letter: "\\" + name for letter, name in zip("αβγδεζηθικλμνξπρστυφχψω", _GREEK)},
    "ς": r"\varsigma", "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi", "Π": r"\Pi",
    "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}
ALIASES = {
    r"\lt": "<", r"\gt": ">", r"\to": r"\rightarrow", r"\gets": r"\leftarrow",
    **{command: r"\ldots" for command in (r"\cdots", r"\dots", r"\dotsc", r"\dotsb", r"\dotsm", r"\dotsi",
                                          r"\dotso", r"\hdots")},
    r"\longrightarrow": r"\rightarrow", r"\longleftarrow": r"\leftarrow", r"\longleftrightarrow": r"\leftrightarrow",
    r"\Longrightarrow": r"\Rightarrow", r"\Longleftarrow": r"\Leftarrow", r"\Longleftrightarrow": r"\Leftrightarrow",
    r"\iff": r"\Leftrightarrow", r"\implies": r"\Rightarrow",
    r"\bigwedge": r"\wedge", r"\bigvee": r"\vee", r"\land": r"\wedge", r"\lor": r"\vee",
}
PRESENTATION = frozenset({r"\limits", r"\nolimits", r"\displaylimits", r"\displaystyle", r"\textstyle",
                          r"\scriptstyle", r"\scriptscriptstyle"})
TEXT_WRAPPERS = frozenset({r"\mbox", r"\text", r"\operatorname"})
FONT_WRAPPERS = frozenset({r"\mathrm", r"\mathit"})
OPERATORS = frozenset(FUNCTIONS | {"arg", "deg", "dim", "ker", "sup", "coth", "lg"})
SIGN_PROMOTED = frozenset({"exp", "sin", "cos", "tan", "log", "ln", "lim"})
# Arguments of these are text or environment specs and are copied verbatim.
RAW_COMMANDS = frozenset({r"\begin", r"\end", r"\textbf", r"\textit", r"\textrm", r"\texttt", r"\textsf",
                          r"\textnormal", r"\textup", r"\textmd", r"\textsl", r"\textsc", r"\emph", r"\hbox",
                          r"\fbox", r"\makebox", r"\framebox", r"\textsuperscript", r"\textsubscript"})
ARG_COUNTS = {
    **ARGUMENTS, r"\cfrac": 2, r"\overset": 2, r"\underset": 2, r"\stackrel": 2, r"\sideset": 2,
    **{command: 1 for command in (
        r"\xrightarrow", r"\xleftarrow", r"\boxed", r"\underbrace", r"\overbrace", r"\overrightarrow",
        r"\overleftarrow", r"\overleftrightarrow", r"\underrightarrow", r"\underleftarrow", r"\check", r"\breve",
        r"\acute", r"\grave", r"\mathring", r"\boldsymbol", r"\pmb", r"\mathfrak", r"\mathscr", r"\mathtt",
        r"\phantom", r"\hphantom", r"\vphantom", r"\pmod", r"\substack", r"\textcircled", r"\underbar",
        r"\overbar", r"\widecheck", r"\Vec", r"\Hat", r"\Bar", r"\Tilde", r"\Dot", r"\Ddot", r"\Check", r"\Breve",
        r"\mathop", r"\mathbin", r"\mathrel", r"\mathord", r"\mathopen", r"\mathclose", r"\mathinner",
        r"\mathbfcal", r"\mathbfit", r"\mathsfit", r"\mathnormal", r"\textbf", r"\textit", r"\textrm")},
}
ZERO_ARG = frozenset(
    {"\\" + name for name in _GREEK} | {"\\" + name.capitalize() for name in _GREEK}
    | {"\\" + name for name in OPERATORS}
    | {"\\" + name for name in (
        "varepsilon vartheta varphi varrho varsigma varpi varkappa sum prod coprod int iint iiint oint lim "
        "bigcup bigcap bigoplus bigotimes bigodot bigsqcup cdot times div pm mp ast star circ bullet ldots "
        "vdots ddots infty partial nabla prime emptyset varnothing forall exists nexists neg in notin ni "
        "leq geq neq approx equiv sim simeq cong propto ll gg subset supset subseteq supseteq cup cap wedge vee "
        "setminus perp parallel mid nmid angle triangle square rightarrow leftarrow leftrightarrow Rightarrow "
        "Leftarrow Leftrightarrow uparrow downarrow mapsto hbar ell Re Im aleph wp top bot not therefore because "
        "langle rangle lfloor rfloor lceil rceil vert Vert lbrace rbrace backslash colon oplus otimes odot "
        "leqslant geqslant prec succ preceq succeq doteq models vdash dashv pmod bmod mod quad qquad "
        "Delta Gamma Theta Lambda Xi Pi Sigma Phi Psi Omega Upsilon").split()})
OPENERS = {"(": ")", "[": "]", r"\{": r"\}", "|": "|", r"\|": r"\|", r"\langle": r"\rangle",
           r"\lfloor": r"\rfloor", r"\lceil": r"\rceil", r"\lvert": r"\rvert", r"\lVert": r"\rVert"}
INFIX = frozenset({r"\over", r"\choose", r"\atop", r"\above", r"\brace", r"\brack"})
_CONTROL_WORD = re.compile(r"\\[A-Za-z]+\Z")
_ALNUM = re.compile(r"[A-Za-z0-9]+\Z")
MAX_PASSES = 8


def _is_command(item: object) -> bool:
    return isinstance(item, str) and _CONTROL_WORD.match(item) is not None


def _takes_arguments(item: object) -> bool:
    return _is_command(item) and item not in ZERO_ARG


def _parse(tokens: list[str]) -> list:
    stack: list[list] = [[]]
    for token in tokens:
        if token == "{":
            stack.append([])
        elif token == "}":
            if len(stack) == 1:
                raise ValueError("Closing TeX group before opening group")
            group = stack.pop()
            stack[-1].append(group)
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise ValueError("Unbalanced TeX groups")
    return stack[0]


def _flatten(sequence: list) -> list[str]:
    tokens: list[str] = []
    for item in sequence:
        if isinstance(item, list):
            tokens.extend(("{", *_flatten(item), "}"))
        else:
            tokens.append(item)
    return tokens


def _word(group: list) -> str | None:
    if not all(isinstance(item, str) for item in group):
        return None
    start, end = 0, len(group)
    while start < end and group[start] == " ":
        start += 1
    while end > start and group[end - 1] == " ":
        end -= 1
    letters = group[start:end]
    if not all(len(item) == 1 for item in letters):
        return None
    word = "".join(letters)
    return word if not word or _ALNUM.match(word) else None


def _unwrap(command: str, group: list) -> list:
    if command in FONT_WRAPPERS:
        group = _process(group)
    word = _word(group)
    if word == "":
        return []
    if word in OPERATORS:
        return ["\\" + word]
    if word is not None:
        return [list(word)]
    return [group] if command in FONT_WRAPPERS else [TEXT_COMMAND, group]


def _promote(sequence: list) -> list:
    """``exp-\\frac{v}{k}`` -> ``\\exp-\\frac{v}{k}``; the whole letter run must be the name."""
    result, index = [], 0
    while index < len(sequence):
        end = index
        while end < len(sequence) and isinstance(sequence[end], str) and len(sequence[end]) == 1 \
                and sequence[end].isalpha():
            end += 1
        word = "".join(sequence[index:end])
        if word in SIGN_PROMOTED and end < len(sequence) and sequence[end] in ("+", "-"):
            result.append("\\" + word)
        else:
            result.extend(sequence[index:end] if end > index else [sequence[index]])
        index = max(end, index + 1)
    return result


def _swap_scripts(sequence: list) -> list:
    result, index = [], 0
    while index < len(sequence):
        window = sequence[index:index + 4]
        if len(window) == 4 and window[0] == "^" and window[2] == "_" and isinstance(window[1], list) \
                and isinstance(window[3], list):
            result.extend(("_", window[3], "^", window[1]))
            index += 4
        else:
            result.append(sequence[index])
            index += 1
    return result


def _skip_optional(sequence: list, index: int) -> int:
    if index < len(sequence) and sequence[index] == "[":
        while index < len(sequence) and sequence[index] != "]":
            index += 1
        index += 1
    return index


def _argument_positions(sequence: list) -> set[int]:
    protected: set[int] = set()
    for index, item in enumerate(sequence):
        if item in ("^", "_"):
            if index + 1 < len(sequence) and isinstance(sequence[index + 1], list):
                protected.add(index + 1)
        elif _takes_arguments(item):
            position = _skip_optional(sequence, index + 1) if item == r"\sqrt" else index + 1
            count = ARG_COUNTS.get(item)
            if count is None:
                while position < len(sequence) and isinstance(sequence[position], list):
                    protected.add(position)
                    position += 1
            else:
                for _ in range(count):
                    if position < len(sequence) and isinstance(sequence[position], list):
                        protected.add(position)
                    position += 1
    return protected


def _delimited(inner: list) -> bool:
    opener = inner[0] if len(inner) >= 2 and isinstance(inner[0], str) else None
    closer = OPENERS.get(opener) if opener is not None else None
    if closer is None or inner[-1] != closer:
        return False
    if opener == closer:
        return opener not in inner[1:-1]
    depth = 0
    for position, item in enumerate(inner):
        if item == opener:
            depth += 1
        elif item == closer:
            depth -= 1
            if depth == 0:
                return position == len(inner) - 1
    return False


def _single_atom(inner: list) -> bool:
    if len(inner) == 1:
        return True
    if _delimited(inner):
        return True
    head = inner[0]
    count = ARG_COUNTS.get(head) if isinstance(head, str) else None
    if count is None:
        return False
    rest = inner[_skip_optional(inner, 1) if head == r"\sqrt" else 1:]
    return len(rest) == count and all(isinstance(item, list) for item in rest)


def _strippable(inner: list, following: object) -> bool:
    script = following in ("^", "_")
    if not inner:
        return not script
    if inner[0] in ("^", "_") or inner[-1] in ("^", "_"):
        return False
    if any(isinstance(item, str) and (item in INFIX or item in ("&", r"\\")) for item in inner):
        return False
    if following is not None and _takes_arguments(inner[-1]):
        return False
    return _single_atom(inner) if script else True


def _strip_braces(sequence: list) -> list:
    protected = _argument_positions(sequence)
    result = []
    for index, item in enumerate(sequence):
        following = sequence[index + 1] if index + 1 < len(sequence) else None
        if isinstance(item, list) and index not in protected and _strippable(item, following):
            result.extend(item)
        else:
            result.append(item)
    return result


def _process(sequence: list) -> list:
    sequence = [ALIASES.get(item, item) if isinstance(item, str) else item
                for item in sequence if not (isinstance(item, str) and item in PRESENTATION)]
    result, index = [], 0
    while index < len(sequence):
        item = sequence[index]
        index += 1
        if isinstance(item, list):
            result.append(_process(item))
        elif item in TEXT_WRAPPERS or item in FONT_WRAPPERS:
            if index < len(sequence) and isinstance(sequence[index], list):
                result.extend(_unwrap(item, sequence[index]))
                index += 1
            else:
                result.append(item)
        elif item in RAW_COMMANDS:
            result.append(item)
            while index < len(sequence) and isinstance(sequence[index], list):
                result.append(sequence[index])
                index += 1
                if item not in (r"\begin", r"\end"):
                    break
        else:
            result.append(item)
    return _strip_braces(_swap_scripts(_promote(result)))


def _replace_unicode(latex: str) -> str:
    if latex.isascii():
        return latex
    return "".join(f" {UNICODE[c]} " if UNICODE.get(c, "").startswith("\\") else UNICODE.get(c, c) for c in latex)


def _standardize_once(latex: str) -> str | None:
    normalized = normalize_training_latex(_replace_unicode(latex))
    if not normalized.isascii():
        return None
    tokens = _flatten(_process(_parse(tokenize_latex(normalized))))
    depth = 0
    for token in tokens:
        depth += (token == "{") - (token == "}")
        if depth < 0:
            return None
    return detokenize_latex(tokens) if depth == 0 else None


@lru_cache(maxsize=1 << 16)
def standardize_latex(latex: str) -> str | None:
    """The single-standard label, or ``None`` when ``latex`` is malformed or not plain ASCII math."""
    try:
        current = _standardize_once(latex)
        for _ in range(MAX_PASSES):
            if not current:
                return None
            following = _standardize_once(current)
            if following == current:
                return current
            current = following
    except (ValueError, RecursionError, IndexError):
        return None
    return None
