"""Give a lecture notebook stable section anchors, and a line of slide links under each section heading.

Usage: python scripts/notebook_sections.py 2 3 4

Reads lectureN/data/sections.json (the deck's numbered sections and the notebook cell each one runs in),
rewrites lectureN/lectureN.ipynb in place without touching outputs, and writes lectureN/data/notebook-anchors.json,
which the deck checks its links against. The notebook generators call apply() after writing, so a rebuild keeps
the anchors. Headings map to anchors as: the title cell -> top, "## k. ..." -> nb-k, any other "## ..." -> nb-exit,
plus any heading listed under "headings" in sections.json.
"""
import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SITE = 'https://nipunbatra.github.io/generative/lectures'


def anchor_for(index, heading, extra):
    if heading in extra:
        return extra[heading]
    if index == 0 and heading.startswith('# '):
        return 'top'
    numbered = re.match(r'## (\d+)\. ', heading)
    if numbered:
        return f'nb-{numbered[1]}'
    if heading.startswith('## '):
        return 'nb-exit'
    return None


def apply(number):
    folder = ROOT / f'lecture{number}'
    spec = json.loads((folder / 'data/sections.json').read_text())
    deck = f'{SITE}/lecture{number}/'
    links = {'top': [f'[Lecture {number} at a glance]({deck}#contents/1)']}
    for num, title, first, anchor in spec['sections']:
        links.setdefault(anchor, []).append(f'[{num} {title}]({deck}#{first}/1)')
    path = folder / f'lecture{number}.ipynb'
    notebook = nbformat.read(path, as_version=4)
    anchors = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'markdown':
            continue
        lines = cell.source.split('\n')
        anchor = anchor_for(index, lines[0], spec.get('headings', {}))
        if not anchor:
            continue
        if anchor in anchors:
            raise SystemExit(f'lecture{number}: two headings map to {anchor}')
        if anchor in links:
            body = [line for line in lines[1:] if not line.startswith('Slides: ')]
            while body and not body[0].strip():
                body.pop(0)
            cell.source = '\n'.join([lines[0], '', 'Slides: ' + ' · '.join(links[anchor])] + (['', *body] if body else []))
        cell.id = anchor
        cell.metadata['id'] = anchor
        anchors.append(anchor)
    missing = sorted({anchor for *_, anchor in spec['sections']} - set(anchors))
    if missing:
        raise SystemExit(f'lecture{number}: no notebook heading for {missing}')
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    (folder / 'data/notebook-anchors.json').write_text(json.dumps(anchors) + '\n')
    return anchors


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        print(f'lecture{arg}:', ' '.join(apply(int(arg))))
