"""Prepare official CROHME scoring tools and prediction/ground-truth pairs.

Downloads a pinned copy of CoMER's bundled organizer tools, with original
licenses and git-blob hash checks. No tools are executed by this preparation step.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

REVISION = 'ee15cdbe02467fd29f386fd554b4bc2185110fcc'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tools', type=Path, default=Path('E:/math-recogniser-training/tools/crohme-eval'))
    parser.add_argument('--archive', type=Path, default=Path('E:/math-recogniser-datasets/archives/CROHME23.zip'))
    parser.add_argument('--symbol-normalization', choices=('none', 'crohme_numeric_v1', 'crohme_symbols_srt_v1'), default='crohme_symbols_srt_v1')
    args = parser.parse_args()
    prepare_predictions(args.predictions, args.output, {'tools': str(args.tools), 'archive': str(args.archive),
                                                       'symbol_normalization': args.symbol_normalization})


def prepare_predictions(predictions, output, config):
    from types import SimpleNamespace
    args = SimpleNamespace(predictions=Path(predictions), output=Path(output),
        tools=Path(config.get('tools', 'E:/math-recogniser-training/tools/crohme-eval')),
        archive=Path(config.get('archive', 'E:/math-recogniser-datasets/archives/CROHME23.zip')),
        symbol_normalization=config.get('symbol_normalization', 'crohme_symbols_srt_v1'))
    provenance_path = args.tools / 'provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance['revision'] != REVISION:
            raise ValueError('Unexpected organizer tool revision')
        files = [{'path': row['path'], 'sha': row['git_blob_sha1']} for row in provenance['files']]
    else:
        tree = json.load(urllib.request.urlopen(f'https://api.github.com/repos/Green-Wood/CoMER/git/trees/{REVISION}?recursive=1'))
        files = [item for item in tree['tree'] if item['type'] == 'blob' and item['path'].startswith(('convert2symLG/', 'lgeval/'))
                 and not item['path'].endswith(('.swp', '.pyc'))]
    root = args.tools.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def download(item):
        target = (root / item['path']).resolve()
        if root not in target.parents:
            raise ValueError('Invalid tool path')
        if target.exists():
            data = target.read_bytes()
        else:
            data = urllib.request.urlopen(f'https://raw.githubusercontent.com/Green-Wood/CoMER/{REVISION}/{item["path"]}').read()
        digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if digest != item['sha']:
            raise ValueError(f'Tool content differs from pinned revision: {item["path"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return {'path': item['path'], 'git_blob_sha1': digest}

    with ThreadPoolExecutor(max_workers=12) as pool:
        manifest = list(pool.map(download, files))
    (root / 'provenance.json').write_text(json.dumps({'repository': 'https://github.com/Green-Wood/CoMER', 'revision': REVISION, 'files': manifest}, indent=2), encoding='utf-8')
    result = json.loads(args.predictions.read_text(encoding='utf-8'))
    if result.get('split') != 'test':
        raise ValueError('Test graph preparation requires explicitly labeled test predictions')
    if args.output.exists():
        raise FileExistsError('Choose a new scoring output directory')
    args.output.mkdir(parents=True)
    counts, missing_graphs, full_counts = {}, {}, {}
    with zipfile.ZipFile(args.archive) as archive:
        archive_names = set(archive.namelist())
        for edition, data in result['datasets'].items():
            if not edition.startswith('CROHME'):
                continue
            directory = args.output / edition
            missing_graphs[edition] = []
            full_counts[edition] = len(data['predictions'])
            for child in ('tex', 'ground_truth', 'mml', 'inkml', 'lg', 'pred_symlg'):
                (directory / child).mkdir(parents=True)
            for row in data['predictions']:
                stem = row['id'].rsplit('/', 1)[-1]
                if Path(stem).name != stem or stem in {'.', '..'}:
                    raise ValueError('Unsafe equation filename')
                gt = f'TC11_CROHME23/SymLG/test/{edition}/{stem}.lg'
                if gt not in archive_names:
                    missing_graphs[edition].append(stem)
                    continue
                (directory / 'ground_truth' / f'{stem}.lg').write_bytes(archive.read(gt))
                (directory / 'tex' / f'{stem}.txt').write_text('$$\n' + row['prediction'] + '\n$$\n', encoding='utf-8')
            counts[edition] = len(data['predictions']) - len(missing_graphs[edition])
    (args.output / 'manifest.json').write_text(json.dumps({'checkpoint': result['checkpoint'], 'predictions': str(args.predictions.resolve()),
        'tool_revision': REVISION, 'predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        'ground_truth_archive': str(args.archive.resolve()), 'counts': counts,
        'symbol_normalization': args.symbol_normalization,
        'normalization_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'training/mathml_symbols.py').read_bytes()).hexdigest(),
        'full_counts': full_counts, 'missing_official_graphs': missing_graphs}, indent=2), encoding='utf-8')
    print(f'Prepared {counts} with pinned tools at {root}', flush=True)


if __name__ == '__main__':
    main()
