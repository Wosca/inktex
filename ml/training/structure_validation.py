"""Select on validation organizer graph ExpRate, using the pinned test scorer.

This adapter accepts validation references only. The scorer's legacy JSON names
include `full_test_*`; here those fields describe validation, never test results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def wsl_path(path: Path) -> str:
    path = path.resolve()
    if not path.drive or len(path.drive) != 2:
        raise ValueError('WSL scoring requires an absolute Windows drive path')
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]


class StructureValidation:
    def __init__(self, config, references, output):
        self.references = references
        self.output = Path(output)
        self.tools = Path(config.get('tools', 'E:/math-recogniser-training/tools/crohme-eval'))
        self.archive = Path(config.get('archive', 'E:/math-recogniser-datasets/archives/CROHME23.zip'))
        self.distro = str(config.get('wsl_distribution', 'Ubuntu'))
        self.workers = int(config.get('workers', 8))
        self.symbol_normalization = config.get('symbol_normalization', 'none')
        self.adapter_hash = hashlib.sha256((Path(__file__).parent / 'mathml_symbols.py').read_bytes()).hexdigest()
        self.provenance = json.loads((self.tools / 'provenance.json').read_text())
        from evaluation.prepare import REVISION
        if self.provenance['revision'] != REVISION:
            raise ValueError('Unexpected organizer tool revision')
        for item in self.provenance['files']:
            data = (self.tools / item['path']).read_bytes()
            if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != item['git_blob_sha1']:
                raise ValueError('Organizer tool modified: ' + item['path'])
        for ref in references:
            if ref.source_dataset == 'crohme' and (ref.split != 'validation' or not ref.id.startswith('crohme:val/')):
                raise ValueError('Structure selection must use CROHME validation only')
        if config.get('reference_audit'):
            audit_path = Path(config['reference_audit'])
            audit = json.loads(audit_path.read_text())
            if audit.get('symbol_normalization') != self.symbol_normalization:
                raise ValueError('Reference round-trip audit used a different scoring protocol')
            if audit.get('normalization_sha256') not in (None, self.adapter_hash):
                raise ValueError('Reference round-trip audit used different normalization code')
            expected = {ref.id: ref.latex for ref in references if ref.source_dataset == 'crohme'}
            predictions = json.loads((audit_path.parent / 'predictions.json').read_text())
            if {row['id']: row['prediction'] for row in predictions} != expected:
                raise ValueError('Reference audit does not match the current validation equations and labels')
            correct = sum(value['correct'] for value in audit['datasets'].values())
            total = sum(value['equations'] for value in audit['datasets'].values())
            if total != len(expected) or correct / total < .98:
                raise ValueError('Reference round-trip audit must cover validation and score at least 98%')
        self.preflight()

    def preflight(self):
        """Verify the actual WSL runtime and validation archive before training."""
        references = [ref for ref in self.references if ref.source_dataset == 'crohme']
        if not references:
            raise ValueError('No CROHME validation references')
        with zipfile.ZipFile(self.archive) as archive:
            names = set(archive.namelist())
            for ref in references:
                _, edition, stem = ref.id.split(':', 1)[1].split('/')
                key = f'TC11_CROHME23/SymLG/val/{edition}/{stem}.lg'
                if key in names:
                    archive.read(key)  # Verify ZIP CRC while the epoch has no work to lose.
        script = '''import sys, subprocess, importlib.util
from pathlib import Path
from bs4 import BeautifulSoup
import lxml, numpy
tools = Path(sys.argv[1])
sys.path.insert(0, str(tools / 'lgeval/src'))
import compareTools
from lg import Lg
from lgio import writeMetrics
spec = importlib.util.spec_from_file_location('organizer', tools / 'convert2symLG/process_mml.py')
organizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(organizer)
print(subprocess.check_output(['pandoc', '--version'], text=True, timeout=10).splitlines()[0])
subprocess.run(['perl', '-MXML::LibXML', '-e', 'exit 0'], check=True, capture_output=True, timeout=10)
print('WSL organizer runtime ready')
'''
        try:
            process = subprocess.run(['wsl', '-d', self.distro, '--', 'python3', '-c', script,
                                      wsl_path(self.tools)], capture_output=True, text=True,
                                     errors='replace', timeout=60, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, 'stderr', '') or str(error)
            raise RuntimeError(f'Graph-scoring preflight failed before training: {detail}') from error
        print(process.stdout.strip(), flush=True)

    def score(self, predictions, epoch):
        if len(predictions) != len(self.references):
            raise ValueError('Validation predictions are not aligned with references')
        work = self.output / f'validation_graph_epoch_{epoch:03d}'
        attempt = 0
        # A stopped validation can leave artifacts before latest.pt is saved.
        # Keep each attempt intact and let epoch-boundary resume score again.
        while work.exists():
            attempt += 1
            work = self.output / f'validation_graph_epoch_{epoch:03d}_attempt_{attempt:02d}'
        work.mkdir(parents=True)
        grouped = {}
        for ref, prediction in zip(self.references, predictions):
            if ref.source_dataset == 'crohme':
                _, edition, stem = ref.id.split(':', 1)[1].split('/')
                grouped.setdefault(edition, []).append((stem, prediction))
        if not grouped:
            raise ValueError('No CROHME validation references')
        counts, full_counts, missing = {}, {}, {}
        with zipfile.ZipFile(self.archive) as archive:
            names = set(archive.namelist())
            for edition, rows in grouped.items():
                directory = work / edition
                for child in ('tex', 'ground_truth', 'mml', 'inkml', 'lg', 'pred_symlg'):
                    (directory / child).mkdir(parents=True)
                full_counts[edition], missing[edition] = len(rows), []
                for stem, prediction in rows:
                    if Path(stem).name != stem or stem in ('.', '..'):
                        raise ValueError('Unsafe validation ID')
                    key = f'TC11_CROHME23/SymLG/val/{edition}/{stem}.lg'
                    if key not in names:
                        missing[edition].append(stem)
                        continue
                    (directory / 'ground_truth' / f'{stem}.lg').write_bytes(archive.read(key))
                    (directory / 'tex' / f'{stem}.txt').write_text('$$\n' + prediction + '\n$$\n', encoding='utf-8')
                counts[edition] = len(rows) - len(missing[edition])
        (work / 'predictions.json').write_text(json.dumps([
            {'id': ref.id, 'prediction': prediction} for ref, prediction in zip(self.references, predictions)
            if ref.source_dataset == 'crohme'], ensure_ascii=False), encoding='utf-8')
        (work / 'manifest.json').write_text(json.dumps({'checkpoint': f'validation epoch {epoch}',
            'scope': 'validation only', 'tool_revision': self.provenance['revision'],
            'symbol_normalization': self.symbol_normalization,
            'normalization_sha256': self.adapter_hash,
            'counts': counts, 'full_counts': full_counts, 'missing_official_graphs': missing,
            'predictions_sha256': hashlib.sha256((work / 'predictions.json').read_bytes()).hexdigest()}), encoding='utf-8')
        script = Path(__file__).resolve().parents[1] / 'evaluation' / 'crohme.py'
        subprocess.run(['wsl', '-d', self.distro, '--', 'python3', '-u', wsl_path(script),
                        '--tools', wsl_path(self.tools), '--work', wsl_path(work),
                        '--workers', str(self.workers)], check=True, timeout=1800)
        result = json.loads((work / 'official_results.json').read_text())
        if set(result['datasets']) != set(counts) or any(
                value['equations'] != counts[edition] for edition, value in result['datasets'].items()):
            raise ValueError('Graph scorer changed the validation denominator; checkpoint selection refused')
        correct = sum(value['correct'] for value in result['datasets'].values())
        equations = sum(value['equations'] for value in result['datasets'].values())
        return {'structure_exprate': correct / equations, 'structure_correct': correct,
                'structure_equations': equations, 'structure_full_equations': sum(full_counts.values()),
                'structure_report': str(work / 'official_results.json')}
