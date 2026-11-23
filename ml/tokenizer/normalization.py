"""One syntax-preserving training label for all ink sources.

Original references remain untouched. This is TeX normalization, not algebraic
simplification: x^12 must never become x^{12}, and words are never math operators.
"""
from __future__ import annotations

import re
import unicodedata

from .latex_tokenizer import detokenize_latex, tokenize_latex

LABEL_VERSION = "ink-label-v2"
ALIASES = {
    r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\le": r"\leq",
    r"\ge": r"\geq", r"\ne": r"\neq", r"\to": r"\rightarrow",
    r"\cdotp": r"\cdot", r"\dots": r"\ldots", r"\dotsc": r"\ldots",
    "…": r"\ldots", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "−": "-", "×": r"\times", "÷": r"\div", "∞": r"\infty",
}
SPACING = {r"\,", r"\!", r"\:", r"\;", r"\ ", r"\quad", r"\qquad", r"\enspace", r"\thinspace"}
SIZING = {r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr"}
FUNCTIONS = {"sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "log", "ln", "exp", "lim", "max", "min", "det", "gcd"}
ARGUMENTS = {r"\frac": 2, r"\binom": 2, r"\sqrt": 1,
             **{f"\\{name}": 1 for name in ("overline", "underline", "vec", "hat", "widehat", "bar", "tilde", "widetilde", "dot", "ddot", "mathbf", "mathbb", "mathcal", "mathsf", "mathit", "mathrm", "text", "operatorname", "mbox")}}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def text_to_latex(text: str) -> str:
    escaped = normalize_text(text)
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                    "%": r"\%", "&": r"\&", "$": r"\$", "#": r"\#", "_": r"\_",
                    '^': r'\textasciicircum{}', '~': r'\textasciitilde{}'}
    return r"\text{" + "".join(replacements.get(c, c) for c in escaped) + "}"


def latex_to_text(latex: str) -> str:
    value = latex.strip()
    if value.startswith((r'\text{', r'\mbox{')) and value.endswith('}'):
        value = value[value.index('{')+1:-1]
    for command, character in ((r'\textbackslash{}', '\\'), (r'\textasciicircum{}', '^'),
                               (r'\textasciitilde{}', '~')):
        value = value.replace(command, character)
    value = re.sub(r'\\([{}%&$#_])', r'\1', value)
    return normalize_text(value)


def normalize_training_latex(latex: str) -> str:
    tokens = tokenize_latex(unicodedata.normalize("NFC", latex).strip())
    # Isolated-symbol corpora label the visible radical/accent itself. It has
    # no operand in this task; retain the glyph instead of inventing an operand
    # or discarding valid symbol supervision.
    if len(tokens) == 1 and ALIASES.get(tokens[0], tokens[0]) in ARGUMENTS:
        return ALIASES.get(tokens[0], tokens[0])
    position = 0

    def group(text=False):
        nonlocal position
        output = []
        while position < len(tokens) and tokens[position] != "}":
            token = tokens[position]
            position += 1
            if token == "{":
                inner = group(text)
                output.extend(("{", *inner, "}"))
            elif text:
                output.append(token)
            elif token in SPACING:
                continue
            elif token in SIZING:
                if position < len(tokens) and tokens[position] == ".":
                    position += 1
            elif token in ("^", "_"):
                output.extend((token, "{", *atom(), "}"))
            elif token == "'":
                primes = [r"\prime"]
                while position < len(tokens) and tokens[position] == "'":
                    primes.append(r"\prime")
                    position += 1
                output.extend(("^", "{", *primes, "}"))
            elif token == "." and tokens[position:position + 2] == [".", "."]:
                position += 2
                output.append(r"\ldots")
            else:
                token = ALIASES.get(token, token)
                if token in ARGUMENTS:
                    arguments = []
                    optional = []
                    if token == r"\sqrt" and position < len(tokens) and tokens[position] == "[":
                        position += 1
                        while position < len(tokens) and tokens[position] != "]":
                            optional.append(tokens[position]); position += 1
                        if position == len(tokens):
                            raise ValueError("Unclosed root index")
                        position += 1
                        optional = ["[", *optional, "]"]
                    for _ in range(ARGUMENTS[token]):
                        arguments.append(atom(token in (r"\text", r"\mbox", r"\operatorname")))
                    name = "".join(arguments[0]) if token in (r"\operatorname", r"\mathrm") else ""
                    if name in FUNCTIONS:
                        output.append("\\" + name)
                    else:
                        output.extend((token, *optional))
                        for argument in arguments:
                            output.extend(("{", *argument, "}"))
                else:
                    output.append(token)
        if position < len(tokens) and tokens[position] == "}":
            position += 1
        for infix, command in ((r"\over", r"\frac"), (r"\choose", r"\binom")):
            depth = 0
            for i, token in enumerate(output):
                depth += (token == '{') - (token == '}')
                if token == infix and depth == 0:
                    if not output[:i] or not output[i+1:]:
                        raise ValueError('Missing infix TeX argument')
                    return [command, '{', *output[:i], '}', '{', *output[i+1:], '}']
        return output

    def atom(text=False):
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Missing TeX argument")
        token = tokens[position]; position += 1
        if token == '}':
            raise ValueError('Missing TeX argument before closing group')
        if token == "{":
            return group(text)
        # A control sequence is one TeX atom, including its own arguments.
        token = ALIASES.get(token, token)
        result = [token]
        if token == r"\sqrt" and position < len(tokens) and tokens[position] == '[':
            result.append('['); position += 1
            while position < len(tokens) and tokens[position] != ']':
                result.append(tokens[position]); position += 1
            if position == len(tokens):
                raise ValueError('Unclosed root index')
            result.append(']'); position += 1
        if token in ARGUMENTS and not text:
            for _ in range(ARGUMENTS[token]):
                result.extend(("{", *atom(), "}"))
        return result

    if tokens.count("{") != tokens.count("}"):
        raise ValueError("Unbalanced TeX groups")
    balance = 0
    for token in tokens:
        balance += (token == '{') - (token == '}')
        if balance < 0:
            raise ValueError('Closing TeX group before opening group')
    normalized = detokenize_latex(group())
    # Only promote bare function names in clear argument contexts, outside text.
    # In particular, do not turn the word 'cosine' or a standalone 'cos' into cos().
    def promote(value):
        pattern = r"(?<![A-Za-z\\])(" + "|".join(sorted(FUNCTIONS, key=len, reverse=True)) + r")(?=\^|_|\(|\\|\s+[A-Za-z0-9])"
        return re.sub(pattern, lambda m: "\\" + m[1], value)
    result, math = [], []
    canonical_tokens = tokenize_latex(normalized)
    cursor = 0
    while cursor < len(canonical_tokens):
        token = canonical_tokens[cursor]
        if token not in (r'\text', r'\mbox', r'\operatorname'):
            math.append(token); cursor += 1
            continue
        result.append(promote(detokenize_latex(math))); math = []
        protected = [token]; cursor += 1
        depth = 0
        while cursor < len(canonical_tokens):
            token = canonical_tokens[cursor]; cursor += 1
            protected.append(token)
            depth += (token == '{') - (token == '}')
            if depth == 0:
                break
        result.append(detokenize_latex(protected))
    result.append(promote(detokenize_latex(math)))
    # Promotion can make a separator between two control sequences redundant.
    # Finish with the same token serializer used for every other target so that
    # normalizing an already normalized label never changes its token stream.
    return detokenize_latex(tokenize_latex("".join(result)))
