"""MathWriting reference-token CER (distinct from character similarity).

Token boundaries follow Google's Apache-2.0 reference notebook:
https://github.com/google-research/google-research/blob/master/mathwriting/mathwriting_code_examples.ipynb
No semantic normalization is applied here; use the dataset's normalizedLabel.
"""
import re

from training.metrics import edit_distance

COMMAND = re.compile(r"\\(?:mathbb\{[a-zA-Z]\}|begin\{[a-z]+\}|end\{[a-z]+\}|operatorname\*|[a-zA-Z]+|.)")


def reference_tokens(value: str) -> list[str]:
    tokens = []
    while value:
        match = COMMAND.match(value) if value.startswith("\\") else None
        token = match.group() if match else value[0]
        tokens.append(token)
        value = value[len(token):]
    return tokens


def reference_token_cer(predictions: list[str], targets: list[str]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("Prediction and reference counts must match")
    errors = total = 0
    for prediction, target in zip(predictions, targets):
        reference = reference_tokens(target)
        errors += edit_distance(reference_tokens(prediction), reference)
        total += len(reference)
    if not total:
        raise ValueError("CER requires nonempty references")
    return errors / total
