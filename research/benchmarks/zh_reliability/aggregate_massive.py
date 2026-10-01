"""Audit baseline/calibrated runs, or recompute the compact historical evidence on CPU."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.benchmarks.zh_reliability.calibration import fit_temperature, validate_calibration
from research.benchmarks.zh_reliability.data import atomic_json, digest, file_hash, partition, read_data
from research.benchmarks.zh_reliability.metrics import paired_bootstrap, probability_record, summarize

CHILDREN = ('base-calibration', 'E0', 'E1')
CONDITIONS = ('E0', 'E1')
SCHEMA = 'massive-alarm-calibration-evidence-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def numerically_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(numerically_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(numerically_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def evidence_prediction(record, condition):
    """Rebuild probabilities without source text, a tokenizer, model weights or CUDA."""
    row = {'id': record['id'], 'group_id': record['group_id'], 'task_id': record['task_id'],
           'question': {'type': record['type'], 'criteria': dict.fromkeys(record['option_order'], '')},
           'label': record['gold'], 'tags': ['public_localized', 'alarm']}
    logits = record['raw_logits'] if condition == 'E0' else record['calibrated_raw_logits']
    temperature = record['temperature'] if condition == 'E0' else record['calibrated_temperature']
    result = probability_record(row, logits, temperature)
    result.update(sample_hash=record['sample_hash'], split=record['split'],
                  tokens={'input_tokens': record['input_tokens'], 'truncated': record['truncated']})
    return result


def compact_summary(rows):
    result = summarize(rows)
    return {key: result[key] for key in ('overall', 'by_type', 'definitions')}


def recompute(artifact, bootstrap=True):
    """Recompute evidence and validate pairing/fit; do not trust stored summary values."""
    require(artifact['schema'] == SCHEMA, 'unsupported evidence schema')
    records = artifact['records']
    model = artifact['provenance']['model']
    require(model['content_hash'] == digest(model['files']), 'model identity hash mismatch')
    require(model['model_id'] == 'convaiinnovations/laya-multilingual'
            and model['revision'] == 'e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'
            and model['trained'] is False, 'expected pinned original multilingual checkpoint')
    require(artifact['calibration']['minimum_per_type'] == 30 and artifact['calibration']['bounds'] == [.5, 5.],
            'calibration protocol mismatch')
    from laya.common import QTYPES, temp_bucket
    original = artifact['calibration']['original']
    temperature_config = original['lang_temperatures'].get('zh', original)
    require(artifact['records_sha256'] == digest(records), 'decision record hash mismatch')
    require(len({r['id'] for r in records}) == len(records), 'duplicate decision IDs')
    require(all(r['split'] in ('calibration', 'test') for r in records), 'unexpected evidence split')
    groups = {}
    source_types = {}
    source_bindings = {}
    for rec in records:
        require(rec['type'] in ('choice', 'noul'), 'unsupported question type')
        require(rec['gold'] in rec['option_order'] and len(set(rec['option_order'])) == len(rec['option_order']),
                'invalid gold/options')
        require(len(rec['raw_logits']) == len(rec['option_order']), 'logit/options mismatch')
        require(len(rec['sample_hash']) == 64 and len(rec['source_row_hash']) == 64, 'invalid source hash')
        require(rec['group_id'] not in groups or groups[rec['group_id']] == rec['split'], 'group leakage')
        groups[rec['group_id']] = rec['split']
        source = (rec['split'], rec['source_id'])
        binding = (rec['split'], rec['group_id'], rec['source_row_hash'])
        require(rec['source_id'] not in source_bindings or source_bindings[rec['source_id']] == binding,
                'source utterance split/group/hash mismatch')
        source_bindings[rec['source_id']] = binding
        qt = QTYPES[rec['type']]
        expected_temperature = temperature_config['temperature_by_options'].get(
            temp_bucket(qt, len(rec['option_order'])), temperature_config['temperature'][qt])
        require(rec['temperature'] == expected_temperature, 'baseline temperature mismatch')
        source_types.setdefault(source, []).append(rec['type'])
        if rec['split'] == 'test':
            require(rec['calibrated_raw_logits'] == rec['raw_logits'], 'calibration changed raw decision logits')
    require(all(sorted(types) == ['choice', 'noul'] for types in source_types.values()),
            'source utterances must keep both question types together')
    counts = Counter(r['split'] for r in records)
    group_counts = Counter(groups.values())
    require(counts == {'calibration': 128, 'test': 186}, 'fixed subset decision counts mismatch')
    require(group_counts == {'calibration': 62, 'test': 89}, 'fixed subset group counts mismatch')
    calibration = [evidence_prediction(r, 'E0') for r in records if r['split'] == 'calibration']
    fitted = {}
    for kind in ('choice', 'noul'):
        selected = [r for r in calibration if r['question']['type'] == kind]
        require(len(selected) == 64, 'calibration type count mismatch')
        fitted[kind] = fit_temperature(selected, minimum=30)
        require(numerically_equal(fitted[kind], artifact['calibration']['fitted'][kind]),
                'fitted temperature/objective mismatch: ' + kind)
    test = [r for r in records if r['split'] == 'test']
    for rec in test:
        require(rec['calibrated_temperature'] == artifact['calibration']['fitted'][rec['type']]['temperature'],
                'test calibration temperature mismatch')
    predictions = {c: [evidence_prediction(r, c) for r in test] for c in CONDITIONS}
    require(all(a['predicted'] == b['predicted'] for a, b in zip(predictions['E0'], predictions['E1'])),
            'positive scalar temperature changed argmax')
    result = {c: compact_summary(predictions[c]) for c in CONDITIONS}
    result['fitted'] = fitted
    result['counts'] = {'calibration_decisions': 128, 'calibration_groups': 62,
                        'test_decisions': 186, 'test_groups': 89, 'test_source_utterances': 93}
    if bootstrap:
        config = artifact['analysis']['paired_group_bootstrap']
        require(config == {'repeats': 1000, 'seed': 20260924, 'min_groups': 20}, 'bootstrap protocol mismatch')
        result['paired'] = {'E1-E0': paired_bootstrap(predictions['E0'], predictions['E1'], **config)}
    return result


def verify_artifact(artifact, bootstrap=True):
    """Return recomputed results; reject changed records, fit, pairing or stored metrics."""
    result = recompute(artifact, bootstrap)
    for key, value in result.items():
        require(numerically_equal(value, artifact['summary'][key]), 'stored summary mismatch: ' + key)
    return result


def render_report(artifact, result):
    provenance = artifact['provenance']
    lines = ['# MASSIVE Chinese alarm calibration evaluation', '',
             f"Measurement date: **{provenance['measurement_date']}**. "
             f"Execution base: `{provenance['execution_base_commit']}`; "
             f"contribution: `{provenance['contribution_commit']}`.", '',
             provenance['version_note'], '',
             'E0 and E1 use the same multilingual checkpoint and raw decision logits. E1 fits one '
             'temperature per question type on calibration data. No model weights are updated.', '',
             '| Condition | Slice | n | Accuracy | Macro F1 | NLL | Brier | ECE |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for kind, scores in [('overall', result[condition]['overall']), *result[condition]['by_type'].items()]:
            values = ' | '.join(f'{scores[k]:.6f}' for k in ('accuracy', 'macro_f1', 'nll', 'brier', 'ece'))
            lines.append(f"| {condition} | {kind} | {scores['n']} | {values} |")
    lines += ['', 'Positive scalar temperature preserves argmax: accuracy and F1 are unchanged.', '',
              '| Type | Calibration n | Temperature | Bound reached |', '|---|---:|---:|---|']
    for kind, fit in result['fitted'].items():
        lines.append(f"| {kind} | {fit['n']} | {fit['temperature']:.6f} | {fit['boundary_hit']} |")
    if 'paired' in result:
        lines += ['', 'Paired differences are E1 minus E0; negative NLL, Brier and ECE differences are better.', '',
                  '| Metric | Difference | 95% group bootstrap interval |', '|---|---:|---|']
        for name, value in result['paired']['E1-E0']['after_minus_before'].items():
            lo, hi = value['ci95']
            lines.append(f"| {name} | {value['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', '## Scope and limits', '',
              '- Test: 93 source utterances, 89 heuristic groups, 186 correlated decisions. Calibration: '
              '64 source utterances, 62 groups, 128 decisions. The paired question types are not independent samples.',
              '- This is a derived MASSIVE 1.1 alarm subset with upstream human intent-review evidence, '
              'not the full benchmark, naturally occurring customer-service logs, or independent local human review.',
              '- One fixed split/seed; 1,000 group bootstrap draws. Intervals condition on the fixed checkpoint '
              'and fitted temperatures, omit calibration uncertainty, and are descriptive without multiplicity correction.',
              '- Unknown checkpoint exposure to MASSIVE/SLURP; grouping may miss semantic dependencies. '
              'These results do not establish general Chinese improvement or unchanged behavior on other tasks.',
              '- The baseline noul fit reaches T=5, the permitted upper bound. Calibration settings were fixed '
              'without choosing on test. Risk/coverage does not improve at every coverage.',
              '- Configured max_len=512, head_max_len=192, FP16. Historical test inputs were at most 60 tokens; '
              'this is not a 512-token stress measurement.',
              '- The compact evidence omits source text and retains source IDs/hashes. It can recompute numerical '
              'results and check pairing; full source-schema/annotation checks require the prepared public dataset.', '']
    return '\n'.join(lines)


def verify_run_predictions(predictions, rows, split, model, selection, condition):
    expected = {r['id']: r for r in rows if split['assignments'][r['id']] == selection}
    require(len(predictions) == len(expected) == len({r['id'] for r in predictions}), 'prediction count/duplicate mismatch')
    require({r['id'] for r in predictions} == set(expected), 'prediction membership mismatch')
    for rec in predictions:
        row = expected[rec['id']]
        require(rec['split'] == selection and rec['condition'] == condition, 'prediction partition/condition mismatch')
        require(rec['model'] == model and rec['sample_hash'] == digest(row) == split['row_hashes'][rec['id']],
                'prediction model/source mismatch')
        require(all(rec.get(k) == v for k, v in row.items()), 'prediction source row changed')
        rebuilt = probability_record(row, rec['raw_logits'], rec['temperature'])
        require(all(numerically_equal(rec[k], rebuilt[k]) for k in
                    ('option_order', 'gold', 'predicted', 'probabilities', 'log_probabilities', 'confidence', 'p_true')),
                'probability reconstruction mismatch')


def collect_run(root):
    """Read only the three evaluation children; never modify the supplied run."""
    root = Path(root)
    protocol = read_json(root / 'protocol.json')
    manifest = read_json(root / 'manifest.json')
    require(file_hash(root / 'protocol.json') == manifest['protocol_sha256'], 'protocol hash mismatch')
    rows = read_data(protocol['data'], formal=True)
    require(file_hash(protocol['data']) == protocol['data_file_sha256'], 'source file hash mismatch')
    require(file_hash(protocol['split_manifest']) == protocol['split_file_sha256'], 'split file hash mismatch')
    split = partition(rows, protocol['split_manifest'], 20260924, True)
    children, predictions, hashes = {}, {}, {}
    for name in CHILDREN:
        directory = root / name
        child = read_json(directory / 'manifest.json')
        require(child['status'] == 'SUCCESS' and child['gpu_experiment'] == 'SUCCESS' and not child['smoke_only'],
                name + ': unsuccessful/formally ineligible run')
        expected = {'formal': True, 'seed': 20260924, 'max_len': 512, 'head_max_len': 192, 'dtype': 'fp16'}
        require(all(child['config'].get(k) == v for k, v in expected.items()), name + ': config mismatch')
        require(child['command'] == ('calibrate' if name == 'base-calibration' else 'eval'), 'wrong command')
        require(read_json(directory / 'split_manifest.json') == split, name + ': split mismatch')
        require(read_json(directory / 'model_identity.json') == child['model'], name + ': model metadata mismatch')
        children[name] = child
        predictions[name] = read_jsonl(directory / 'predictions.jsonl')
        hashes.update({name + '/' + file: file_hash(directory / file) for file in
                       ('manifest.json', 'predictions.jsonl', 'metrics.json', 'split_manifest.json',
                        'model_identity.json', 'environment.json')})
    model = children['E0']['model']
    require(all(child['model'] == model for child in children.values()), 'model differs between conditions')
    require(model['content_hash'] == digest(model['files']), 'model identity content hash mismatch')
    require(model['trained'] is False, 'expected original checkpoint')
    for name in CHILDREN:
        verify_run_predictions(predictions[name], rows, split, model,
                               'calibration' if name == 'base-calibration' else 'test',
                               'E0' if name == 'base-calibration' else name)
    calibration = read_json(root / 'base-calibration/calibration.json')
    validate_calibration(calibration, model, split)
    require(calibration['minimum_per_type'] == 30 and calibration['smoke_only'] is False, 'invalid calibration minimum/mode')
    require(calibration['calibration_ids'] == [r['id'] for r in predictions['base-calibration']], 'calibration IDs differ')
    hashes['base-calibration/calibration.json'] = file_hash(root / 'base-calibration/calibration.json')
    after = {r['id']: r for r in predictions['E1']}
    records = []
    for rec in predictions['base-calibration'] + predictions['E0']:
        upstream = rec['provenance']['upstream_annotation']
        compact = {key: rec[key] for key in ('id', 'group_id', 'task_id', 'split', 'sample_hash', 'option_order',
                                            'gold', 'raw_logits', 'temperature')}
        compact.update(type=rec['question']['type'], source_id=upstream['source_id'],
                       source_row_hash=upstream['source_row_hash'], input_tokens=rec['tokens']['input_tokens'],
                       truncated=rec['tokens']['truncated'])
        if rec['split'] == 'test':
            paired = after[rec['id']]
            require(paired['gold'] == rec['gold'] and paired['group_id'] == rec['group_id'], 'pairing changed')
            compact.update(calibrated_raw_logits=paired['raw_logits'], calibrated_temperature=paired['temperature'])
        records.append(compact)
    environment = read_json(root / 'E0/environment.json')
    safe_environment = {k: environment[k] for k in ('python', 'os', 'dependencies', 'cuda', 'cudnn')}
    source_hashes = {Path(k).name: v for k, v in environment.get('source_hashes', {}).items()
                     if Path(k).name in ('__main__.py', 'data.py', 'metrics.py', 'calibration.py', 'runtime.py', 'prepare_massive.py')}
    artifact = {'schema': SCHEMA, 'provenance': {
        'measurement_date': protocol['created_utc'][:10], 'execution_base_commit': environment['commit'],
        'contribution_commit': 'unrecorded', 'version_note': 'Source version recorded by the execution environment; '
        'recomputing this artifact does not rerun model inference.', 'execution_worktree_dirty': environment['dirty'],
        'execution_source_hashes': source_hashes, 'original_file_sha256': hashes,
        'dataset': 'MASSIVE 1.1 zh-CN alarm subset', 'dataset_license': 'CC-BY-4.0',
        'publisher_notice': 'https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md',
        'data_hash': split['data_hash'], 'split_hash': split['split_hash'],
        'data_file_sha256': protocol['data_file_sha256'], 'split_file_sha256': protocol['split_file_sha256'],
        'model': model, 'environment': safe_environment,
        'evaluation_config': {k: children['E0']['config'][k] for k in
                              ('seed', 'max_len', 'head_max_len', 'dtype', 'device')}},
        'analysis': {'paired_group_bootstrap': {'repeats': 1000, 'seed': 20260924, 'min_groups': 20}},
        'calibration': {'minimum_per_type': 30, 'bounds': [.5, 5.], 'fitted': calibration['fitted'],
                        'original': calibration['original']}, 'records_sha256': digest(records), 'records': records}
    artifact['summary'] = recompute(artifact)
    for name in CONDITIONS:
        source = read_json(root / name / 'metrics.json')
        require(numerically_equal(artifact['summary'][name], {k: source[k] for k in artifact['summary'][name]}),
                'recomputed metrics differ from measured child: ' + name)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, nargs='?')
    parser.add_argument('--artifact', type=Path, help='verify compact evidence without model/data downloads or CUDA')
    parser.add_argument('--report', type=Path, help='write Markdown rebuilt from verified values')
    args = parser.parse_args(argv)
    if (args.run_dir is None) == (args.artifact is None):
        parser.error('provide either RUN_DIR or --artifact')
    try:
        artifact = read_json(args.artifact) if args.artifact else collect_run(args.run_dir)
        result = verify_artifact(artifact) if args.artifact else artifact['summary']
        if args.artifact:
            if args.report:
                args.report.write_text(render_report(artifact, result), encoding='utf-8')
        else:
            root = args.run_dir
            atomic_json(root / 'calibration_evidence.json', artifact)
            atomic_json(root / 'metrics.json', result)
            (args.report or root / 'report.md').write_text(render_report(artifact, result), encoding='utf-8')
            manifest = read_json(root / 'manifest.json')
            manifest.update(status='SUCCESS', gpu_experiment='SUCCESS', active_stage='complete', completed=list(CHILDREN),
                            completed_utc=datetime.now(timezone.utc).isoformat(),
                            evidence_sha256=file_hash(root / 'calibration_evidence.json'))
            atomic_json(root / 'manifest.json', manifest)
        print('SUCCESS: E0/E1 pairing, fitted temperatures, metrics and 1,000 group bootstrap draws verified')
        return 0
    except Exception as error:
        if args.run_dir:
            manifest_path = args.run_dir / 'manifest.json'
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            manifest.update(status='FAILED', gpu_experiment='FAILED', active_stage='aggregation',
                            error={'stage': 'aggregation', 'type': type(error).__name__, 'reason': str(error)})
            atomic_json(manifest_path, manifest)
            (args.run_dir / 'aggregation-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('verification failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
