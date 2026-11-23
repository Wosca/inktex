"""Assemble predicted symbol/layout graphs without a sequence-length ceiling.

Only confidently connected trees are accepted. Unsupported/ambiguous layouts
remain candidates for the sequence decoder, never silently concatenated pieces.
"""
from __future__ import annotations

from tokenizer.latex_tokenizer import detokenize_latex


def graph_to_latex(symbols, edges):
    children = {i: {} for i in range(len(symbols))}
    parents = {}
    for parent, child, relation in edges:
        if not (0 <= parent < len(symbols) and 0 <= child < len(symbols)) or parent == child:
            raise ValueError('Invalid layout edge')
        if relation not in ('Right', 'Sup', 'Sub', 'Above', 'Below', 'Inside'):
            raise ValueError('Unsupported layout relation')
        if child in parents:
            raise ValueError('Ambiguous symbol parent')
        parents[child] = parent
        children[parent].setdefault(relation, []).append(child)
    roots = set(range(len(symbols))) - parents.keys()
    if len(roots) != 1:
        raise ValueError('Disconnected or cyclic layout graph')
    visited = set()

    def region(nodes):
        if len(nodes) != 1:
            raise ValueError('Ambiguous branch ordering')
        return nodes[0]

    tokens, pending = [], [next(iter(roots))]
    while pending:
        node = pending.pop()
        if isinstance(node, str):
            tokens.append(node)
            continue
        if node in visited:
            raise ValueError('Cyclic layout graph')
        visited.add(node)
        label, branches = symbols[node], children[node]
        if label in ('-', r'\frac') and 'Above' in branches and 'Below' in branches:
            value = [r'\frac', '{', region(branches['Above']), '}', '{', region(branches['Below']), '}']
        elif label == r'\sqrt' and 'Inside' in branches:
            value = [r'\sqrt']
            if 'Above' in branches:
                value.extend(('[', region(branches['Above']), ']'))
            value.extend(('{', region(branches['Inside']), '}'))
        else:
            if 'Inside' in branches:
                raise ValueError('Unsupported inside layout')
            value = [label]
            for relations, marker in ((('Below', 'Sub'), '_'), (('Above', 'Sup'), '^')):
                nodes = [child for relation in relations for child in branches.get(relation, [])]
                if nodes:
                    value.extend((marker, '{', region(nodes), '}'))
        if 'Right' in branches:
            value.append(region(branches['Right']))
        # Iterative emission also handles long baselines beyond Python's usual
        # recursion limit, including nested branches inside those baselines.
        pending.extend(reversed(value))
    if len(visited) != len(symbols):
        raise ValueError('Layout graph omitted symbols')
    return detokenize_latex(tokens)


def assemble_prediction(outputs, symbol_names, relation_names, row=0, threshold=.8):
    """Pool stroke groups, label symbols, and choose one confident parent each."""
    import torch
    present = (~outputs['stroke_mask'][row]).nonzero().flatten().tolist()
    roots = list(range(len(present)))
    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]; i = roots[i]
        return i
    groups = outputs['group'][row].sigmoid()
    centers = outputs['center'][row]
    near = (centers[:, None] - centers[None]).abs().sum(-1) <= 4
    allowed = (groups >= threshold) & near
    valid = ~outputs['stroke_mask'][row]
    pairs = (allowed & valid[:, None] & valid[None]).triu(1).nonzero().cpu().tolist()
    for i, j in pairs:
        roots[find(j)] = find(i)
    members = {}
    for i in present:
        members.setdefault(find(i), []).append(i)
    members = list(members.values())
    symbols, scores = [], []
    probabilities = outputs['symbol'][row].softmax(-1)
    pooling = probabilities.new_zeros((len(members), len(probabilities)))
    for index, group in enumerate(members):
        pooling[index, group] = 1 / len(group)
    pooled = (pooling @ probabilities).cpu()
    for distribution in pooled:
        confidence, label = distribution.max(0)
        if float(confidence) < threshold:
            raise ValueError('Uncertain character in layout graph')
        symbols.append(symbol_names[int(label)]); scores.append(float(confidence))
    relation = outputs['relation'][row].softmax(-1)
    relation = torch.einsum('is,stc,jt->ijc', pooling, relation, pooling).cpu()
    edges = []
    for child in range(len(members)):
        best = None
        for parent in range(len(members)):
            if child == parent:
                continue
            distribution = relation[parent, child]
            confidence, label = distribution[1:].max(0)
            candidate = (float(confidence), parent, child, relation_names[int(label)+1])
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best and best[0] >= threshold:
            scores.append(best[0]); edges.append(best[1:])
    latex = graph_to_latex(symbols, edges)
    return {'latex': latex, 'confidence': min(scores), 'symbols': symbols, 'edges': edges}
