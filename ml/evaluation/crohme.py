"""Run pinned organizer LaTeX/SymLG conversion and LgEval without shell wrappers.

Run in Linux/WSL with pandoc, Perl XML::LibXML, bs4, lxml and numpy installed.
All ground-truth equations remain in the denominator, including conversion errors.
"""
import argparse
import hashlib
from contextlib import redirect_stderr
from concurrent.futures import ThreadPoolExecutor
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def empty_prediction(directory: Path, identity: str, reason: str) -> None:
    for folder in ('lg', 'pred_symlg'):
        (directory / folder / (identity + '.lg')).write_text(
            f'# Failed conversion: {reason}; counted as incorrect\n', encoding='utf-8')


def render_prediction(file: Path, directory: Path) -> dict | None:
    """A malformed model output is an incorrect prediction, not a fatal epoch."""
    from tokenizer.latex_tokenizer import tokenize_latex
    output = directory / 'mml' / (file.stem + '.mml')
    # Scoring can be retried in place; never reuse a partial/stale conversion.
    for folder, suffix in (('mml', '.mml'), ('inkml', '.inkml'), ('lg', '.lg'), ('pred_symlg', '.lg')):
        (directory / folder / (file.stem + suffix)).unlink(missing_ok=True)
    depth = 0
    for token in tokenize_latex(file.read_text(encoding='utf-8')):
        depth += (token == '{') - (token == '}')
        if depth < 0:
            break
    if depth:
        empty_prediction(directory, file.stem, 'unbalanced TeX groups')
        return {'id': file.stem, 'stage': 'pandoc', 'failed': True, 'message': 'Unbalanced TeX groups'}
    command = ['pandoc', '--mathml', '-s', '-f', 'markdown', '-t', 'html',
               '--metadata=pagetitle:CROHME', str(file), '-o', str(output)]
    try:
        process = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=30)
    except subprocess.TimeoutExpired:
        output.unlink(missing_ok=True)
        empty_prediction(directory, file.stem, 'pandoc timed out after 30 seconds')
        return {'id': file.stem, 'stage': 'pandoc', 'failed': True, 'timeout_seconds': 30}
    if process.returncode:
        output.unlink(missing_ok=True)
        empty_prediction(directory, file.stem, 'pandoc returned an error')
    return {'id': file.stem, 'stage': 'pandoc', 'failed': bool(process.returncode),
            'returncode': process.returncode, 'message': process.stderr[:1000]} if process.returncode or process.stderr else None


def convert_prediction(file: Path, directory: Path, tools: Path, normalization: str) -> dict | None:
    try:
        process = subprocess.run(['perl', str(tools / 'convert2symLG/mml2lg.pl'), '-s', str(file)],
                                 capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        empty_prediction(directory, file.stem, 'Perl converter timed out after 30 seconds')
        return {'id': file.stem, 'stage': 'perl', 'failed': True, 'timeout_seconds': 30}
    if process.returncode:
        empty_prediction(directory, file.stem, 'Perl converter returned an error')
        return {'id': file.stem, 'stage': 'perl', 'failed': True, 'returncode': process.returncode,
                'message': process.stderr.decode(errors='replace')[:1000]}
    output_data = process.stdout
    if normalization == 'crohme_symbols_srt_v1':
        from training.mathml_symbols import remove_redundant_spatial_edges
        try:
            output_data = remove_redundant_spatial_edges(output_data.decode('utf-8')).encode('utf-8')
        except UnicodeDecodeError:
            pass
    encoding_error = store_prediction_graph(directory / 'lg' / (file.stem + '.lg'), output_data)
    if encoding_error:
        empty_prediction(directory, file.stem, 'invalid UTF-8')
        return {'id': file.stem, 'stage': 'lg_encoding', 'failed': True, 'message': encoding_error}
    return {'id': file.stem, 'stage': 'perl', 'message': process.stderr.decode(errors='replace')[:1000]} if process.stderr else None


def tag_prediction(file: Path, directory: Path, tools: Path) -> dict | None:
    # Organizer node tagging follows graph parents; a cyclic bad prediction
    # can hang it. Isolate each equation rather than losing the whole corpus.
    try:
        process = subprocess.run([sys.executable, str(tools / 'convert2symLG/update_nodeTags.py'),
                                  str(file), str(directory / 'pred_symlg')],
                                 capture_output=True, text=True, errors='replace', timeout=30)
    except subprocess.TimeoutExpired:
        empty_prediction(directory, file.stem, 'node tagging timed out after 30 seconds')
        return {'id': file.stem, 'stage': 'node_tags', 'failed': True, 'timeout_seconds': 30}
    if process.returncode:
        empty_prediction(directory, file.stem, 'node tagging returned an error')
        return {'id': file.stem, 'stage': 'node_tags', 'failed': True, 'returncode': process.returncode,
                'message': process.stderr[:1000]}
    return None


def store_prediction_graph(path: Path, data: bytes) -> str | None:
    """Keep invalid converter bytes for audit; score that prediction as empty."""
    try:
        data.decode('utf-8')
    except UnicodeDecodeError as error:
        path.with_suffix('.invalid-utf8').write_bytes(data)
        path.write_text('# Failed UTF-8 conversion: counted as incorrect\n', encoding='utf-8')
        return str(error)
    path.write_bytes(data)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    tools, work = args.tools.resolve(), args.work.resolve()
    manifest = json.loads((work / 'manifest.json').read_text())
    symbol_normalization = manifest.get('symbol_normalization', 'none')
    if symbol_normalization not in ('none', 'crohme_numeric_v1', 'crohme_symbols_srt_v1'):
        raise ValueError('Unknown MathML symbol normalization')
    adapter_hash = hashlib.sha256((Path(__file__).resolve().parents[1] / 'training/mathml_symbols.py').read_bytes()).hexdigest()
    if manifest.get('normalization_sha256') not in (None, adapter_hash):
        raise ValueError('Symbol normalization changed since scoring preparation')
    provenance = json.loads((tools / 'provenance.json').read_text())
    if provenance['revision'] != manifest['tool_revision']:
        raise ValueError('Scoring tool revision differs from preparation')
    sys.path.insert(0, str(tools / 'lgeval/src'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import compareTools
    from lg import Lg
    from lgio import writeMetrics
    from bs4 import BeautifulSoup
    spec = importlib.util.spec_from_file_location('organizer_process_mml', tools / 'convert2symLG/process_mml.py')
    organizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(organizer)
    compareTools.cmpNodes = compareTools.intersectMetric
    compareTools.cmpEdges = compareTools.intersectMetric
    result = {'checkpoint': manifest['checkpoint'], 'tool_revision': provenance['revision'],
              'symbol_normalization': symbol_normalization,
              'normalization_sha256': adapter_hash,
              'pandoc': subprocess.check_output(['pandoc', '--version'], text=True, timeout=30).splitlines()[0],
              'metric': 'Organizer SymLG/LgEval INTER D_B=0; provided official graphs only, missing ground truth reported explicitly',
              'datasets': {}}
    for edition, expected in manifest['counts'].items():
        directory = work / edition
        gt_files = sorted((directory / 'ground_truth').glob('*.lg'))
        if len(gt_files) != expected:
            raise ValueError('Ground-truth denominator differs from predictions')
        tex_files = sorted((directory / 'tex').glob('*.txt'))
        if {file.stem for file in tex_files} != {file.stem for file in gt_files}:
            raise ValueError('LaTeX predictions do not match the ground-truth equation IDs')
        # The comparison library must reproduce a perfect ground-truth self score.
        first = Lg(str(gt_files[0]))
        second = Lg(str(gt_files[0]))
        first.labelMissingEdges()
        second.labelMissingEdges()
        if dict(first.compare(second)[0])['D_B'] != 0:
            raise ValueError('Organizer scorer failed ground-truth self comparison')

        print(f'{edition}: rendering {expected} LaTeX predictions', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            warnings = [value for value in pool.map(lambda file: render_prediction(file, directory), tex_files) if value]
        failed_mathml = set()
        # Call the unchanged organizer functions per file so one malformed
        # prediction is an error in the score, rather than aborting the corpus.
        for file in sorted((directory / 'mml').glob('*.mml')):
            try:
                soup = BeautifulSoup(file.read_text(), 'lxml')
                if not soup('math'):
                    raise ValueError('No MathML math tag')
                if symbol_normalization == 'crohme_symbols_srt_v1':
                    from training.mathml_symbols import normalize_mathtext
                    normalize_mathtext(soup('math')[0], soup)
                clean = organizer.remove_unknown_tags(soup('math')[0])
                if symbol_normalization in ('crohme_numeric_v1', 'crohme_symbols_srt_v1'):
                    from training.mathml_symbols import split_numeric_symbols
                    clean = split_numeric_symbols(clean, soup)
                organizer.write_mml(organizer.add_ID(clean), file.name, str(directory / 'inkml') + os.sep)
            except Exception as error:
                failed_mathml.add(file.stem)
                warnings.append({'id': file.stem, 'stage': 'mathml', 'message': str(error)})
                (directory / 'lg' / (file.stem + '.lg')).write_text('# Failed MathML conversion\n')

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            warnings.extend(value for value in pool.map(lambda file: convert_prediction(file, directory, tools, symbol_normalization),
                sorted(file for file in (directory / 'inkml').glob('*.inkml') if file.stem not in failed_mathml)) if value)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            warnings.extend(value for value in pool.map(lambda file: tag_prediction(file, directory, tools),
                                                        sorted((directory / 'lg').glob('*.lg'))) if value)
        failures = failed_mathml | {item['id'] for item in warnings if item.get('failed')}
        missing, rows, invalid_gt = sorted(failures), [], []
        metric_file = directory / 'metrics.csv'
        with metric_file.open('w') as stream, (directory / 'graph_diagnostics.log').open('w') as diagnostics, redirect_stderr(diagnostics):
            for index, target in enumerate(gt_files):
                predicted = directory / 'pred_symlg' / target.name
                if target.stem in failures:
                    empty_prediction(directory, target.stem, 'conversion failed')
                if not predicted.exists() or not predicted.read_text().strip():
                    predicted.write_text('# Failed conversion: counted as an incorrect prediction\n')
                    if target.stem not in missing:
                        missing.append(target.stem)
                output_graph, target_graph = Lg(str(predicted)), Lg(str(target))
                if target_graph.error:
                    invalid_gt.append(target.stem)
                    continue
                output_graph.labelMissingEdges()
                target_graph.labelMissingEdges()
                values = output_graph.compare(target_graph)
                stream.write(f'*M,{predicted},{target}\n')
                writeMetrics(values, stream)
                metrics = dict(values[0])
                rows.append({'id': target.stem, 'label_graph_errors': metrics['D_B'],
                             'correct_structure': metrics['hasCorrectStructure']})
                if (index + 1) % 500 == 0:
                    print(f'{edition}: scored {index+1}/{expected}', flush=True)
        summary = subprocess.run([sys.executable, str(tools / 'lgeval/src/sumMetric.py'), edition, str(metric_file)],
                                 capture_output=True, text=True, errors='replace', timeout=120)
        (directory / 'Summary.txt').write_text(summary.stdout + summary.stderr)
        if summary.returncode:
            raise RuntimeError('Organizer metric summary failed')
        counts = [sum(row['label_graph_errors'] <= threshold for row in rows) for threshold in range(4)]
        full_count = manifest.get('full_counts', manifest['counts'])[edition]
        missing_gt = manifest.get('missing_official_graphs', {}).get(edition, [])
        scorable = len(rows)
        if not scorable:
            raise ValueError('No valid official ground-truth graphs')
        result['datasets'][edition] = {'equations': scorable, 'correct': counts[0], 'exprate': counts[0] / scorable,
                                      'full_test_equations': full_count, 'missing_official_graphs': missing_gt,
                                      'invalid_official_graphs': invalid_gt,
                                      'full_test_exprate_bounds': [counts[0] / full_count, (counts[0] + len(missing_gt) + len(invalid_gt)) / full_count],
                                      'correct_with_0_to_3_errors': counts, 'missing_conversions': missing,
                                      'conversion_warnings': warnings, 'per_equation': rows}
        print(f'{edition}: scorable ExpRate {counts[0]}/{scorable} = {counts[0]/scorable:.3%}; invalid ground truth {len(invalid_gt)}; missing conversions {len(missing)}', flush=True)
        (work / 'official_results.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
