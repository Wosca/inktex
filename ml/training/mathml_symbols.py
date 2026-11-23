"""Data-independent adapter from typesetting numbers to CROHME glyph objects."""
import re
import csv
from io import StringIO


def split_numeric_symbols(math, soup):
    """Split numeric MathML atoms, preserving their enclosing structural position.

    Pandoc emits <mn>24</mn> or <mn>8.8</mn> as one typesetting number.
    CROHME recognizes individual glyphs. Never consult reference graphs here.
    Functions/identifiers are untouched. Apply before organizer ID assignment.
    """
    for node in list(math.find_all('mn')):
        value = node.get_text().strip()
        if len(value) <= 1 or not re.fullmatch(r'[0-9.,]+', value):
            continue
        row = soup.new_tag('mrow')
        for character in value:
            leaf = soup.new_tag('mn')
            leaf.string = character
            row.append(leaf)
        node.replace_with(row)
    return math


def normalize_mathtext(math, soup):
    """Preserve visible alphanumeric glyphs which the old filter deletes."""
    for node in list(math.find_all('mtext')):
        value = node.get_text().strip()
        if not re.fullmatch(r'[A-Za-z0-9]+', value):
            continue
        row = soup.new_tag('mrow')
        for character in value:
            leaf = soup.new_tag('mn' if character.isdigit() else 'mi')
            leaf.string = character
            row.append(leaf)
        node.replace_with(row)
    return math


def remove_redundant_spatial_edges(text):
    """Convert redundant containment fan-out to a symbol relation tree.

    The legacy converter can emit sqrt->a Inside, a->b Right AND sqrt->b
    Inside. In the CROHME SRT representation, the latter is redundant. Remove
    it only when an alternate same-relation anchor reaches the child through
    an explicit Right chain. No reference graph or token label is consulted.
    """
    lines = text.splitlines(keepends=True)
    edges = {}
    for index, line in enumerate(lines):
        fields = next(csv.reader(StringIO(line), skipinitialspace=True), [])
        if fields and fields[0] == 'R' and len(fields) >= 5:
            edges[index] = tuple(fields[1:4])
    right = {}
    for parent, child, relation in edges.values():
        if relation == 'Right':
            right.setdefault(parent, set()).add(child)
    remove = set()
    for index, (parent, child, relation) in edges.items():
        if relation == 'Right':
            continue
        anchors = [c for p, c, r in edges.values() if p == parent and r == relation and c != child]
        seen, pending = set(), list(anchors)
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(right.get(current, set()) - seen)
        if child in seen:
            remove.add(index)
    return ''.join(line for index, line in enumerate(lines) if index not in remove)
