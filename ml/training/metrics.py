from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from tokenizer.latex_tokenizer import canonicalize_latex_tokens, detokenize_latex


BALANCED_METRIC_NAMES = (
    "token_accuracy",
    "exact_match",
    "normalized_exact_match",
    "average_token_edit_distance",
    "normalized_edit_similarity",
)


def edit_distance(left: Sequence[object], right: Sequence[object]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_item != right_item)))
        previous = current
    return previous[-1]


@dataclass(slots=True)
class SequenceMetrics:
    token_accuracy: float
    exact_match: float
    normalized_exact_match: float
    average_token_edit_distance: float
    normalized_edit_similarity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_metrics(predictions: Sequence[str], targets: Sequence[str], tokenize) -> SequenceMetrics:
    if not targets:
        return SequenceMetrics(0, 0, 0, 0, 0)
    correct_tokens = total_tokens = edit_total = 0
    exact = normalized_exact = similarity_total = 0.0
    for prediction, target in zip(predictions, targets):
        predicted_tokens, target_tokens = tokenize(prediction), tokenize(target)
        correct_tokens += sum(a == b for a, b in zip(predicted_tokens, target_tokens))
        total_tokens += max(len(predicted_tokens), len(target_tokens), 1)
        distance = edit_distance(predicted_tokens, target_tokens)
        edit_total += distance
        exact += prediction == target
        canonical_prediction = canonicalize_latex_tokens(prediction)
        canonical_target = canonicalize_latex_tokens(target)
        normalized_prediction = detokenize_latex(canonical_prediction)
        normalized_target = detokenize_latex(canonical_target)
        normalized_exact += canonical_prediction == canonical_target
        character_distance = edit_distance(normalized_prediction, normalized_target)
        similarity_total += 1.0 - character_distance / max(len(normalized_prediction), len(normalized_target), 1)
    count = len(targets)
    return SequenceMetrics(
        token_accuracy=correct_tokens / total_tokens,
        exact_match=exact / count,
        normalized_exact_match=normalized_exact / count,
        average_token_edit_distance=edit_total / count,
        normalized_edit_similarity=similarity_total / count,
    )


def calculate_balanced_metrics(by_source: dict[str, dict[str, float]]) -> dict[str, float]:
    """Give each source equal influence regardless of its validation-set size."""
    if not by_source:
        return {f"balanced_{name}": 0.0 for name in BALANCED_METRIC_NAMES}
    source_count = len(by_source)
    return {
        f"balanced_{name}": sum(source[name] for source in by_source.values()) / source_count
        for name in BALANCED_METRIC_NAMES
    }
