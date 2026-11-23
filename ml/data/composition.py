"""Recombine training ink into longer expressions; never draw from holdouts."""
from dataclasses import dataclass

import numpy as np
import torch

from tokenizer.latex_tokenizer import detokenize_latex, tokenize_latex
from tokenizer.normalization import ARGUMENTS, normalize_text, text_to_latex


@dataclass(slots=True)
class CompositionReference:
    id: str
    source_dataset: str = 'composed'
    synthetic: bool = True
    point_count: int = 2048


def composable_math_tokens(label):
    """Only complete math terms can be joined to unrelated handwritten ink.

    A dangling backslash consumes the first token of the next donor, including
    a structural brace. A leading script would instead attach to that donor's
    predecessor. Keep both out of virtual expressions without rewriting ink or
    changing the base dataset/checkpoint contract.
    """
    tokens = tokenize_latex(label)
    if not tokens or '\\' in tokens or tokens[0] in {'^', '_'} or label in ARGUMENTS:
        return None
    return tokens


def composition_pools(examples):
    """Validate training donors before expensive preprocessing/cache building."""
    words, terms, plus = [], [], []
    for i, example in enumerate(examples):
        label = example.latex
        text = (getattr(example, 'metadata', {}) or {}).get('transcription')
        if text and len(text) <= 40:
            words.append(i)
        elif label == '+':
            plus.append(i)
        else:
            tokens = composable_math_tokens(label)
            if (tokens is not None and len(tokens) <= 64 and '\\begin' not in label
                    and not any(c in label for c in ('=', '<', '>', '&'))
                    and '\\text' not in label):
                terms.append(i)
    if not terms:
        raise ValueError('Length composition needs short mathematical terms in the training split')
    if not plus:
        raise ValueError('Length composition needs handwritten + donors in the training split; '
                         'enable data.mathwriting.include_symbols in the recipe')
    # Keep arrays: Generator.choice otherwise converts large Python pools per draw.
    return tuple(np.asarray(pool, dtype=np.int64) for pool in (words, terms, plus))


class ComposedDataset:
    def __init__(self, base, count, seed=1337, *, styles=('sum',), token_budgets=None):
        if base.preprocessing.normalization != 'glyph':
            raise ValueError('Length composition requires glyph-normalized ink')
        self.base, self.seed = base, seed
        self.feature_cache = None
        self.examples = [CompositionReference(f'composed:train:{i}') for i in range(count)]
        self.words, self.terms, self.plus = composition_pools(base.examples)
        if not styles or set(styles) - {'sum', 'product'}:
            raise ValueError('Composition styles must be sum and/or product')
        self.styles = tuple(styles)
        if token_budgets is not None and (not token_budgets or any(
                not isinstance(budget, int) or not 4 <= budget <= base.max_tokens - 2 for budget in token_budgets)):
            raise ValueError('Composition token budgets must fit the decoder working budget')
        self.token_budgets = tuple(token_budgets) if token_budgets else None

    def set_epoch(self, seed):
        self.seed = seed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        text_mode = len(self.words) > 0 and rng.random() < .25
        style = self.styles[0] if len(self.styles) == 1 else str(rng.choice(self.styles))
        pool = self.words if text_mode else self.terms
        target_tokens = int(rng.choice(self.token_budgets)) if self.token_budgets and not text_mode else None
        requested = 512 if target_tokens else int(rng.integers(4, 17))
        pieces, labels, text_parts, donor_ids = [], [], [], []
        cursor = 0.0
        point_count = token_count = 0
        for component in range(requested):
            candidate = self.base[int(rng.choice(pool))]
            if component and not text_mode and style == 'sum':
                candidates = [self.base[int(rng.choice(self.plus))], candidate]
            else:
                candidates = [candidate]
            next_labels = labels + [item['latex'] for item in candidates]
            if text_mode:
                raw_text = ''.join(self.base.tokenizer.tokens[t] for t in candidate['ctc_targets'])
                next_text = [*text_parts, raw_text]
                label = text_to_latex(' '.join(next_text))
                next_tokens = len(self.base.tokenizer.encode(label)) - 2
            else:
                # All donors already have canonical tokens. Count incrementally
                # instead of repeatedly parsing an ever longer expression.
                next_tokens = token_count + sum(len(item['tokens']) - 2 for item in candidates)
            points = point_count + sum(len(item['features']) for item in candidates)
            if points > self.base.preprocessing.max_points or next_tokens + 2 > self.base.max_tokens:
                break
            point_count, token_count = points, next_tokens
            labels = next_labels
            if text_mode:
                text_parts = next_text
            for item in candidates:
                features = item['features'].clone()
                xy = features[:, :2]
                features[:, 0] += cursor - float(xy[:, 0].min())
                features[:, 1] -= float(torch.quantile(xy[:, 1], .75))
                cursor = float(features[:, 0].max()) + float(rng.uniform(.35, .7))
                pieces.append(features)
                donor_ids.append(item['id'])
            if target_tokens and token_count >= target_tokens:
                break
        if not pieces:
            raise ValueError('A composition donor exceeds the working budget')
        features = torch.cat(pieces)
        # Rebuild stroke-order features across the combined ink, without making
        # existing strokes depend on the final number of strokes.
        stroke_ids = features[:, 7].gt(.5).long().cumsum(0) - 1
        features[:, 9] = stroke_ids.float().log1p() / 8
        label = text_to_latex(' '.join(text_parts)) if text_mode else detokenize_latex(
            t for value in labels for t in tokenize_latex(value))
        text = normalize_text(' '.join(text_parts)) if text_mode else None
        try:
            tokens = self.base.tokenizer.encode(label)
        except ValueError as error:
            raise ValueError(f'Invalid composed target {self.examples[index].id}, seed {self.seed}: {error}') from error
        if len(tokens) > self.base.max_tokens:
            raise ValueError('Canonical composed label exceeds the decoder working budget')
        return {'features': features, 'tokens': torch.tensor(tokens),
                'latex': label, 'normalized_latex': label, 'id': self.examples[index].id,
                'source_dataset': 'composed', 'source': 'synthetic', 'downsampled': False,
                'structure': None, 'donor_ids': donor_ids, 'requested_token_budget': target_tokens,
                'text': text, 'granularity': 'line' if text is not None else None,
                'ctc_targets': [self.base.tokenizer.token_to_id[c] for c in text] if text else None}
