"""Recognition summaries shared by private adapters and writer experiments."""
from collections import defaultdict

from tokenizer.latex_tokenizer import tokenize_latex
from tokenizer.normalization import latex_to_text, normalize_training_latex
from training.metrics import edit_distance


def canonical(value):
    try:
        return normalize_training_latex(value)
    except (ValueError, RecursionError):
        return value


def summarize_predictions(rows):
    if not rows:
        return {'queries': 0, 'writers': 0}
    errors = sum(edit_distance(tokenize_latex(canonical(row['prediction'])),
                              tokenize_latex(row['target'])) for row in rows)
    tokens = sum(len(tokenize_latex(row['target'])) for row in rows)
    by_writer = defaultdict(list)
    for row in rows:
        by_writer[tuple(row['writer'])].append(row)
    result = {'queries': len(rows), 'writers': len(by_writer),
              'canonical_exact_match': sum(canonical(row['prediction']) == row['target'] for row in rows) / len(rows),
              'token_error_rate': errors / max(tokens, 1),
              'mean_latency_ms': sum(row['latency_ms'] for row in rows) / len(rows),
              'macro_writer_exact': sum(sum(canonical(row['prediction']) == row['target'] for row in writer_rows)
                                        / len(writer_rows) for writer_rows in by_writer.values()) / len(by_writer)}
    if all(row['writer'][0] == 'iamondo' for row in rows):
        pairs = [(latex_to_text(row['prediction']), latex_to_text(row['target'])) for row in rows]
        result['text_character_error_rate'] = sum(edit_distance(p, t) for p, t in pairs) / max(sum(len(t) for _, t in pairs), 1)
    return result
