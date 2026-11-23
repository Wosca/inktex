"""IAMonDo words/lines, including compressed InkML and canvas transforms.

Subsets 0+1 train, 2 validates, 3 and independent subset 4 remain held out.
Formulas have no transcriptions in v1.0 and are not invented as training labels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from data.schema import Stroke, StrokeExpression, StrokePoint, normalize_expression_timing
from tokenizer.normalization import normalize_text, text_to_latex

XML_ID = '{http://www.w3.org/XML/1998/namespace}id'
NUMBER = re.compile(r'''([!'\"]?)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|\*|\?)''')


def decode_trace(text: str, channels: int = 4) -> np.ndarray:
    """InkML explicit, first and second differences, inherited per channel."""
    previous = np.zeros(channels)
    velocity = np.zeros(channels)
    acceleration = np.zeros(channels)
    modes = ['!'] * channels
    result = []
    for index, point in enumerate(text.strip().split(',')):
        if not point.strip():
            continue
        values = NUMBER.findall(point)
        if len(values) != channels or NUMBER.sub('', point).strip():
            raise ValueError(f'Malformed IAMonDo trace point: {point!r}')
        decoded = []
        for channel, (prefix, value) in enumerate(values):
            if prefix:
                modes[channel] = prefix
            mode = modes[channel]
            if index == 0 and (mode != '!' or value in ('*', '?')):
                raise ValueError('InkML trace must start with explicit channel values')
            if value == '?':
                raise ValueError('Missing regular InkML channel')
            if value == '*':
                number = previous[channel] if mode == '!' else (velocity[channel] if mode == "'" else acceleration[channel])
            else:
                number = float(value)
            if mode == '"':
                acceleration[channel] = number
                velocity[channel] += number
                number = previous[channel] + velocity[channel]
            elif mode == "'":
                velocity[channel] = number
                number += previous[channel]
            else:
                velocity[channel] = number - previous[channel]
            previous[channel] = number
            decoded.append(number)
        result.append(decoded)
    array = np.asarray(result, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError('Invalid IAMonDo trace coordinates')
    return array


class IAMonDoImporter:
    def __init__(self):
        self.report = Counter()
        self.missing_documents = []
        self.unassigned_documents = []

    def parse(self, path: Path, subset: int):
        root = ET.parse(path).getroot()
        # v1.0 has unqualified tags; accept namespaced InkML as well.
        for element in root.iter():
            element.tag = element.tag.rsplit('}', 1)[-1]
        annotations = {a.get('type'): a.text for a in root.findall('annotation')}
        writer = annotations.get('authorId')
        if not writer:
            raise ValueError(f'IAMonDo document {path.name} has no writer identity')
        matrix_element = root.find('.//canvasTransform/mapping/matrix')
        matrix = np.eye(4, 5)
        if matrix_element is not None:
            rows = [row for row in (matrix_element.text or '').split(',') if row.strip()]
            matrix = np.asarray([[float(v) for v in row.split()] for row in rows])
            if matrix.shape != (4, 5):
                raise ValueError('Unsupported IAMonDo canvas transform')
        traces = {}
        for index, trace in enumerate(root.findall('trace')):
            trace_id = trace.get(XML_ID) or trace.get('id')
            raw = decode_trace(trace.text or '')
            transformed = np.c_[raw, np.ones(len(raw))] @ matrix.T
            points = [StrokePoint(x=float(v[0]), y=float(v[1]), time=float(v[2] * 1000),
                                  pressure=float(np.clip(v[3] / 255, 0, 1)), pressure_available=True)
                      for v in transformed]
            traces[trace_id] = (index, Stroke(points, id=trace_id))
        split = 'train' if subset < 2 else ('validation' if subset == 2 else 'test')
        word_rotations = {}
        for line in root.iter('traceView'):
            if not any(a.get('type') == 'type' and a.text == 'Textline' for a in line.findall('annotation')):
                continue
            line_words = [word for word in line.iter('traceView')
                          if any(a.get('type') == 'type' and a.text == 'Word' for a in word.findall('annotation'))]
            centers = []
            for word in line_words:
                refs = [v.get('traceDataRef', '').lstrip('#') for v in word.iter('traceView') if v.get('traceDataRef')]
                points = [p for ref in refs if ref in traces for p in traces[ref][1].points]
                if points:
                    centers.append(np.mean([(p.x, p.y) for p in points], axis=0))
            if len(centers) >= 2:
                delta = centers[-1] - centers[0]
                angle = -float(np.arctan2(delta[1], delta[0]))
                for word in line_words:
                    word_rotations[id(word)] = angle
        for group_index, group in enumerate(root.iter('traceView')):
            labels = {a.get('type'): a.text or '' for a in group.findall('annotation')}
            kind = labels.get('type')
            if kind == 'Formula':
                self.report['formulas_without_transcription'] += 1
            if kind not in ('Word', 'Textline'):
                continue
            text = normalize_text(labels.get('transcription', ''))
            if not text:
                self.report['empty_transcription'] += 1
                continue
            words = [child for child in group.iter('traceView')
                     if any(a.get('type') == 'type' and a.text == 'Word' for a in child.findall('annotation'))]
            if kind == 'Textline' and len(words) < 2:
                self.report['single_word_line_duplicates'] += 1
                continue
            refs = [v.get('traceDataRef', '').lstrip('#') for v in group.iter('traceView') if v.get('traceDataRef')]
            if not refs or len(set(refs)) != len(refs) or any(ref not in traces for ref in refs):
                raise ValueError(f'Invalid IAMonDo ink grouping in {path.name}:{group_index}')
            strokes = [traces[ref][1] for ref in sorted(refs, key=lambda ref: traces[ref][0])]
            # Normalize the direction of oriented text using annotated word order.
            centers = []
            for word in words:
                word_refs = [v.get('traceDataRef', '').lstrip('#') for v in word.iter('traceView') if v.get('traceDataRef')]
                points = [p for ref in word_refs for p in traces[ref][1].points]
                if points:
                    centers.append(np.mean([(p.x, p.y) for p in points], axis=0))
            rotation = word_rotations.get(id(group), 0.0)
            if kind == 'Textline' and len(centers) > 1:
                direction = centers[-1] - centers[0]
                rotation = -float(np.arctan2(direction[1], direction[0]))
            if abs(rotation) > .25:
                cosine, sine = np.cos(rotation), np.sin(rotation)
                strokes = [Stroke([replace(p, x=float(cosine*p.x - sine*p.y),
                                           y=float(sine*p.x + cosine*p.y)) for p in stroke.points], id=stroke.id)
                           for stroke in strokes]
            label = text_to_latex(text)
            expression = StrokeExpression(id=f'iamondo:{path.stem}:{group_index}',
                original_id=f'{path.stem}:{group_index}', latex_original=label, latex_normalized=label,
                strokes=strokes, source='external', source_dataset='iamondo', split=split,
                writer_id=f'iamondo:{writer}', metadata={'transcription': text, 'granularity': kind.lower(),
                'document_id': path.stem, 'subset': subset, 'rotation_radians': rotation,
                'inkml_path': str(path), 'time_unit': 'milliseconds', 'pressure_scale': 255})
            self.report[f'{split}_{kind.lower()}'] += 1
            yield normalize_expression_timing(expression)

    def load(self, root: Path):
        roots = [root] if (root/'0.set').exists() else list(root.rglob('0.set'))
        roots = [p if p.is_dir() else p.parent for p in roots]
        if len(roots) != 1:
            raise ValueError('Expected one IAMonDo directory containing 0.set through 4.set')
        root = roots[0]
        seen, writers = set(), {name: set() for name in ('train', 'validation', 'test')}
        for subset in range(5):
            for name in (root/f'{subset}.set').read_text().splitlines():
                name = name.strip()
                if not name:
                    continue
                if name in seen:
                    raise ValueError('IAMonDo document occurs in multiple predefined subsets')
                if not (root/name).exists():
                    self.missing_documents.append(name)
                    self.report['missing_predefined_documents'] += 1
                    continue
                seen.add(name)
                for expression in self.parse(root/name, subset):
                    writers[expression.split].add(expression.writer_id)
                    yield expression
        if any(writers[a] & writers[b] for a, b in (('train','validation'), ('train','test'), ('validation','test'))):
            raise ValueError('IAMonDo predefined subsets overlap writers')
        self.unassigned_documents = sorted({p.name for p in root.glob('*.inkml')} - seen)
        # Some v1.0 archive documents have no official assignment. Keep them
        # outside training; assigning them randomly could leak held-out writers.
        self.report['unassigned_documents_excluded'] = len(self.unassigned_documents)
        self.report['documents'] = len(seen)
        for split, ids in writers.items():
            self.report[f'{split}_writers'] = len(ids)
