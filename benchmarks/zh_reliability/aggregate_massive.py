"""Verify and aggregate the fixed MASSIVE experiment without loading a model."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.zh_reliability.__main__ import report, write_predictions
from benchmarks.zh_reliability.calibration import validate_calibration
from benchmarks.zh_reliability.data import atomic_json, digest, file_hash, partition, read_data
from benchmarks.zh_reliability.metrics import paired_bootstrap, probability_record, summarize

CHILDREN = ('base-calibration', 'head', 'head-calibration', 'E0', 'E1', 'E2', 'E3')
CONDITIONS = ('E0', 'E1', 'E2', 'E3')


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
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def verify_identity(identity, source):
    """Check the original source; Agent may patch only its private tokenizer copy."""
    require(identity['content_hash'] == digest(identity['files']), 'model identity content hash mismatch')
    for name, expected in identity['files'].items():
        path = Path(source) / name
        require(path.is_relative_to(Path(source)) and '..' not in Path(name).parts,
                'unsafe identity file path')
        require(file_hash(path) == expected, 'model source file hash mismatch: ' + str(path))
    return {'source': str(source), 'identity': identity, 'files_verified': len(identity['files'])}


def verify_predictions(predictions, rows, split, model, selection, condition):
    expected = {r['id']: r for r in rows if split['assignments'][r['id']] == selection}
    selected = [p for p in predictions if p.get('split') == selection]
    require(len(selected) == len(predictions), condition + ': unexpected prediction partition')
    require(len(selected) == len(expected), condition + ': incomplete predictions')
    require(len({r['id'] for r in selected}) == len(selected), condition + ': duplicate prediction IDs')
    require({r['id'] for r in selected} == set(expected), condition + ': prediction membership mismatch')
    for rec in selected:
        source = expected[rec['id']]
        require(rec['sample_hash'] == digest(source) == split['row_hashes'][rec['id']],
                condition + ': prediction sample hash mismatch')
        require(all(rec.get(k) == v for k, v in source.items()), condition + ': source row changed')
        require(rec['model'] == model, condition + ': prediction model identity mismatch')
        require(rec['condition'] == condition, condition + ': condition label mismatch')
        require(rec['gold'] == source['label'] and rec['group_id'] == source['group_id'],
                condition + ': label/group mismatch')
        require(split['assignments'][rec['id']] == selection, condition + ': split assignment mismatch')
        rebuilt = probability_record(source, rec['raw_logits'], rec['temperature'])
        for key in ('option_order', 'gold', 'predicted', 'probabilities', 'log_probabilities', 'confidence', 'p_true'):
            require(rec[key] == rebuilt[key], condition + ': invalid ' + key)
        require(isinstance(rec.get('tokens'), dict), condition + ': missing token diagnostics')
    return sorted(selected, key=lambda r: r['id'])


def verify_calibration(artifact, predictions, model, split, rows):
    validate_calibration(artifact, model, split)
    require(artifact['smoke_only'] is False, 'smoke-only calibration prohibited')
    require(artifact['minimum_per_type'] >= 30, 'calibration minimum below protocol')
    expected = {r['id'] for r in rows if split['assignments'][r['id']] == 'calibration'}
    require(set(artifact['calibration_ids']) == expected, 'calibration sample membership mismatch')
    require(artifact['calibration_ids'] == [r['id'] for r in predictions],
            'calibration artifact/prediction ordering mismatch')
    counts = Counter(r['question']['type'] for r in predictions)
    for kind in ('choice', 'noul'):
        fit = artifact['fitted'][kind]
        require(fit['status'] == 'SUCCESS' and fit['n'] == counts[kind] == 64,
                'calibration fitting status/count mismatch: ' + kind)
        require(.5 <= fit['temperature'] <= 5., 'calibration temperature outside fixed bounds')
        require(fit['boundary_hit'] == (fit['temperature'] in (.5, 5.)), 'temperature boundary flag mismatch')
        require(fit['nll_fitted'] <= fit['nll_t1'] + 1e-12, 'calibration increased fit objective')


def verify_training(root, manifest, split):
    metrics = read_json(root / 'head/metrics.json')
    saved = read_json(root / 'head/checkpoints/training_result.json')
    require(all(metrics.get(k) == v for k, v in saved.items()), 'training summary/result mismatch')
    require(metrics['status'] == 'SUCCESS' and metrics['steps'] == 198,
            'training did not complete 198 successful optimizer steps')
    require(metrics['checkpoint_every'] == 11, 'training checkpoint interval mismatch')
    require(metrics['frozen_unchanged'] is True and metrics['decision_updated'] is True,
            'frozen/updated parameter verification failed')
    require(metrics['reload_raw_logits_max_difference'] == 0, 'export/reload changed raw logits')
    cfg = read_json(root / 'head/checkpoints/training_config.json')
    contract = cfg['contract']
    require(contract['model'] == manifest['model'], 'training source identity mismatch')
    require(contract['data_hash'] == split['data_hash'] and contract['split_hash'] == split['split_hash'],
            'training data/split hash mismatch')
    names = contract['trainable_names']
    require(names and all(n.startswith(('head.', 'type_emb.', 'scorer.')) for n in names),
            'unexpected trainable parameters')
    expected = {'max_steps': 198, 'checkpoint_every': 11, 'micro_batch': 1, 'accumulation': 8,
                'lr': .0001, 'weight_decay': .01, 'max_grad_norm': 1., 'full': True, 'seed': 20260924}
    require(all(cfg['requested'].get(k) == v for k, v in expected.items()), 'training config mismatch')
    logs = read_jsonl(root / 'head/checkpoints/training.jsonl')
    success = [r for r in logs if r['status'] == 'SUCCESS']
    require(sum(r['status'] == 'SKIPPED' for r in logs) == metrics['skipped_updates'],
            'skipped optimizer update count mismatch')
    require([r['step'] for r in success] == list(range(1, 199)), 'successful update log is incomplete')
    checked = [r for r in success if r['dev_nll'] is not None]
    require([r['step'] for r in checked] == list(range(11, 199, 11)), 'dev-selection schedule mismatch')
    best = min(checked, key=lambda r: (r['dev_nll'], r['step']))
    require(metrics['best_dev_nll'] == best['dev_nll'], 'best dev NLL mismatch')
    require(Path(metrics['best_path']).name == f"step-{best['step']:05d}", 'wrong selected checkpoint')
    progress = read_json(Path(metrics['best_path']) / 'progress.json')
    require(progress['contract'] == contract, 'selected checkpoint contract mismatch')
    for file, key in [('head.safetensors', 'head_hash'), ('local_state.pt', 'state_hash')]:
        require(file_hash(Path(metrics['best_path']) / file) == progress[key], 'selected checkpoint hash mismatch')
    return metrics, cfg, logs, best['step']


def append_report(root, metrics, selected_step):
    lines = ['\n## Fixed public-data experiment\n',
             'The test set contains **93 source utterances, 89 heuristic source groups, and 186 correlated decisions** '
             '(93 choice and 93 noul). Each source utterance generates both question types.\n',
             'E0 is the original checkpoint with runtime-clamped original temperatures; E1 adds base calibration. '
             'E2 is the dev-selected trained checkpoint with neutral decision temperature T=1; '
             'E3 adds calibration fitted after checkpoint selection.\n',
             f'Training completed 198 successful optimizer updates. Dev NLL was checked every 11 successful updates; '
             f'step {selected_step} was selected. The encoder/action parameters were verified unchanged, decision '
             'parameters changed, and same-device export/reload raw logits matched exactly. '
             'Action output quality was not supervised or evaluated.\n',
             '| Condition | Type | n | Accuracy | Macro F1 | NLL | Brier | ECE |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for kind, result in metrics[condition]['by_type'].items():
            values = ' | '.join(f'{result[k]:.6f}' for k in ('accuracy', 'macro_f1', 'nll', 'brier', 'ece'))
            lines.append(f"| {condition} | {kind} | {result['n']} | {values} |")
    lines.extend(['\n### Paired differences and 95% cluster bootstrap intervals\n',
                  'All values are **after minus before**. Negative NLL, Brier, and ECE changes are better; '
                  'positive accuracy and F1 changes are better. Groups are resampled together, preserving '
                  'the paired conditions and the two question types.\n',
                  '| Comparison | Metric | Difference | 95% percentile interval |',
                  '|---|---|---:|---|'])
    for comparison, result in metrics['paired'].items():
        for metric, value in result['after_minus_before'].items():
            lo, hi = value['ci95']
            lines.append(f"| {comparison} | {metric} | {value['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(['\n### Calibration and boundaries\n',
                  '| Model | Type | n | Temperature | Boundary hit | Fit NLL at T=1 | Fit NLL after |',
                  '|---|---|---:|---:|---|---:|---:|'])
    for model, artifact in metrics['calibration'].items():
        for kind, fit in artifact['fitted'].items():
            lines.append(f"| {model} | {kind} | {fit['n']} | {fit['temperature']:.6f} | "
                         f"{fit['boundary_hit']} | {fit['nll_t1']:.6f} | {fit['nll_fitted']:.6f} |")
    lines.extend(['\n### Interpretation limits\n',
                  '- This is a single-seed, narrow MASSIVE 1.1 alarm-intent experiment using public human-localized '
                  'data with retained upstream intent-review evidence. There was no independent local human review. '
                  'It is not naturally occurring customer-service data or the full MASSIVE benchmark.',
                  '- Data, splits, hyperparameters, dev-selection schedule and calibration settings were fixed before '
                  'test predictions. All 198 successful updates were completed regardless of intermediate dev scores.',
                  '- The bootstrap uses 1,000 draws with seed 20260924. Its intervals are conditional on the fixed '
                  'trained and calibrated models; they exclude training and calibration uncertainty. Multiple comparisons '
                  'are descriptive and are not corrected significance tests.',
                  '- The grouping heuristic may miss semantic dependencies. Base-model exposure to MASSIVE/SLURP is '
                  'unknown. These results cannot establish general Chinese-language improvement.',
                  '- max_len=512 is the configured limit. These are short utterances, not a 512-token stress test. '
                  'Timing uses 10 warmup batches and 186 measured single-decision batches per stage; GPUs were not '
                  'exclusively reserved. No new FP32 experiment is included.',
                  '- Verification records and artifact hashes are in `verification.json`; per-condition environments, '
                  'loading memory and official API parity are retained.\n'])
    with (root / 'report.md').open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def aggregate(root):
    root = Path(root).resolve()
    manifest = read_json(root / 'manifest.json')
    protocol = read_json(root / 'protocol.json')
    require(file_hash(root / 'protocol.json') == manifest['protocol_sha256'], 'protocol changed')
    require(file_hash(protocol['data']) == protocol['data_file_sha256'], 'source data changed')
    require(file_hash(protocol['split_manifest']) == protocol['split_file_sha256'], 'source split changed')
    rows = read_data(protocol['data'], formal=True)
    split = partition(rows, protocol['split_manifest'], 20260924, True)
    require(split['counts']['test'] == 186 and split['group_counts']['test'] == 89,
            'unexpected fixed test counts')
    test_rows = [r for r in rows if split['assignments'][r['id']] == 'test']
    source_ids = {r['provenance']['upstream_annotation']['source_id'] for r in test_rows}
    require(len(source_ids) == 93 and Counter(r['question']['type'] for r in test_rows) == {'choice': 93, 'noul': 93},
            'unexpected source utterance/type count')
    verification = {'status': 'RUNNING', 'protocol_sha256': manifest['protocol_sha256'],
                    'source_data_sha256': file_hash(protocol['data']), 'children': {}}
    children, child_metrics, raw_predictions = {}, {}, {}
    for name in CHILDREN:
        directory = root / name
        child = read_json(directory / 'manifest.json')
        require(child['status'] == 'SUCCESS' and child['gpu_experiment'] == 'SUCCESS', name + ': unsuccessful run')
        require(child['smoke_only'] is False, name + ': smoke run prohibited')
        expected = {'formal': True, 'seed': 20260924, 'max_len': 512, 'head_max_len': 192, 'dtype': 'fp16'}
        command = 'train-head' if name == 'head' else ('eval' if name in CONDITIONS else 'calibrate')
        require(child['command'] == command, name + ': command mismatch')
        require(child['config']['device'] == ('TITAN RTX' if name == 'head' else 'RTX 2080 Ti'),
                name + ': device selector mismatch')
        require(all(child['config'].get(k) == v for k, v in expected.items()), name + ': config mismatch')
        require(read_json(directory / 'split_manifest.json') == split, name + ': split changed')
        require(read_json(directory / 'model_identity.json') == child['model'], name + ': model metadata mismatch')
        children[name] = child
        child_metrics[name] = read_json(directory / 'metrics.json')
        raw_predictions[name] = read_jsonl(directory / 'predictions.jsonl')
        files = ('manifest.json', 'metrics.json', 'predictions.jsonl', 'split_manifest.json',
                 'environment.json', 'model_identity.json', 'loading_memory.json')
        verification['children'][name] = {'files': {f: file_hash(directory / f) for f in files}}
    base, tuned = children['E0']['model'], children['head']['exported_model']
    require(base['trained'] is False and tuned['trained'] is True, 'base/tuned identity type mismatch')
    require(base != tuned, 'trained checkpoint identical to base')
    for name in ('base-calibration', 'head', 'E1'):
        require(children[name]['model'] == base, name + ': wrong base identity')
    for name in ('head-calibration', 'E2', 'E3'):
        require(children[name]['model'] == tuned, name + ': wrong tuned identity')
    require(base['revision'] == protocol['model_revision'], 'base model revision differs from fixed protocol')
    if 'base_model_weights_sha256' in protocol:
        require(file_hash(Path(protocol['base_model']) / 'model.safetensors') == protocol['base_model_weights_sha256'],
                'base model weights changed since protocol initialization')
    if 'base_model_lineage_sha256' in protocol:
        require(file_hash(Path(protocol['base_model']) / 'zh_lineage.json') == protocol['base_model_lineage_sha256'],
                'base model lineage changed since protocol initialization')
    verification['base_model'] = verify_identity(base, protocol['base_model'])
    verification['tuned_model'] = verify_identity(tuned, root / 'head/exported')
    lineage = read_json(root / 'head/exported/zh_lineage.json')
    require(lineage['base_hash'] == base['content_hash'] and lineage['trained'] is True,
            'exported model lineage mismatch')
    verification['lineage_sha256'] = file_hash(root / 'head/exported/zh_lineage.json')
    training, train_config, train_logs, selected_step = verify_training(root, children['head'], split)
    verify_predictions(raw_predictions['head'], rows, split, tuned, 'dev', 'E2')
    artifacts = {}
    from laya.common import QTYPES, temp_bucket
    for name, model, condition in [('base-calibration', base, 'E0'), ('head-calibration', tuned, 'E2')]:
        verify_predictions(raw_predictions[name], rows, split, model, 'calibration', condition)
        artifact = read_json(root / name / 'calibration.json')
        verify_calibration(artifact, raw_predictions[name], model, split, rows)
        artifacts[name] = artifact
        verification['children'][name]['files']['calibration.json'] = file_hash(root / name / 'calibration.json')
    predictions = {}
    metrics = {}
    for name in CONDITIONS:
        model = base if name in ('E0', 'E1') else tuned
        predictions[name] = verify_predictions(raw_predictions[name], rows, split, model, 'test', name)
        metrics[name] = summarize(predictions[name])
        parity = child_metrics[name]['official_parity']
        require(parity['n'] == 4 and parity['max_absolute_error'] <= parity['rounding_tolerance'],
                name + ': official API parity failed')
        if name in ('E1', 'E3'):
            artifact_name = 'base-calibration' if name == 'E1' else 'head-calibration'
            applied_path = Path(children[name]['config']['calibration'])
            require(file_hash(applied_path) == file_hash(root / artifact_name / 'calibration.json'),
                    name + ': supplied calibration artifact differs')
        require(all(numerically_equal(child_metrics[name][key], value) for key, value in metrics[name].items()),
                name + ': reported metrics do not reproduce from predictions')
        for rec in predictions[name]:
            if name == 'E0':
                original = artifacts['base-calibration']['original']
                temperature_config = original['lang_temperatures'].get('zh', original)
                qt = QTYPES[rec['question']['type']]
                bucket = temp_bucket(qt, len(rec['option_order']))
                expected_temperature = temperature_config['temperature_by_options'].get(
                    bucket, temperature_config['temperature'][qt])
                require(rec['temperature'] == expected_temperature, 'E0 changed runtime original temperature')
            if name == 'E2':
                require(rec['temperature'] == 1., 'E2 must use neutral temperature')
            if name in ('E1', 'E3'):
                artifact = artifacts['base-calibration' if name == 'E1' else 'head-calibration']
                require(rec['temperature'] == artifact['fitted'][rec['question']['type']]['temperature'],
                        name + ': fitted temperature not applied')
    for before, after in [('E0', 'E1'), ('E2', 'E3')]:
        require(all(a['raw_logits'] == b['raw_logits'] for a, b in zip(predictions[before], predictions[after])),
                after + ': calibration changed raw model logits')
    metrics['paired'] = {}
    for after, before in [('E1', 'E0'), ('E3', 'E2'), ('E3', 'E1'), ('E3', 'E0')]:
        result = paired_bootstrap(predictions[before], predictions[after], repeats=1000, seed=20260924, min_groups=20)
        require(result['status'] == 'SUCCESS' and result['n_groups'] == 89, 'paired bootstrap unsuccessful')
        metrics['paired'][after + '-' + before] = result
    metrics['training'] = training
    metrics['training_config'] = train_config
    metrics['calibration'] = {name: {'fitted': value['fitted'], 'minimum_per_type': value['minimum_per_type'],
                                   'smoke_only': value['smoke_only'], 'original': value['original']}
                              for name, value in artifacts.items()}
    metrics['official_parity'] = {name: child_metrics[name]['official_parity'] for name in CONDITIONS}
    metrics['loading_memory'] = {name: read_json(root / name / 'loading_memory.json') for name in CHILDREN}
    for name in ('E0', 'E2'):
        metrics[name + '_measurement'] = child_metrics[name]['measurement']
        measurement = metrics[name + '_measurement']
        require(measurement['warmup_batches'] == 10 and measurement['measured_batches'] == 186
                and measurement['batch_decisions'] == 1, name + ': timing protocol mismatch')
    metrics['test_sample_counts'] = {'source_utterances': 93, 'groups': 89, 'correlated_decisions': 186}
    metrics['token_diagnostics'] = {
        name: {'max_input_tokens': max(r['tokens']['input_tokens'] for r in predictions[name]),
               'truncated_decisions': sum(r['tokens']['truncated'] for r in predictions[name])}
        for name in CONDITIONS}
    verification['training_artifacts'] = {
        str(p.relative_to(root)): file_hash(p) for p in (root / 'head/checkpoints').rglob('*') if p.is_file()}
    verification['successful_updates'] = sum(r['status'] == 'SUCCESS' for r in train_logs)
    verification['selected_step'] = selected_step
    verification['aggregation_code_sha256'] = file_hash(Path(__file__))
    verification['prediction_checks'] = ['exact split membership', 'unique IDs', 'row/sample hashes',
        'unchanged gold labels and groups', 'model identities', 'probabilities rebuilt from raw logits',
        'calibration split and model binding', 'paired raw-logit equality under temperature changes']
    verification['status'] = 'SUCCESS'
    verification['verified_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(root / 'metrics.json', metrics)
    write_predictions(root / 'predictions.jsonl', [rec for name in CONDITIONS for rec in predictions[name]])
    shutil.copyfile(root / 'E0/split_manifest.json', root / 'split_manifest.json')
    shutil.copyfile(root / 'E0/environment.json', root / 'environment.json')
    atomic_json(root / 'environments.json', {name: read_json(root / name / 'environment.json') for name in CHILDREN})
    verification['aggregate_files'] = {name: file_hash(root / name) for name in
                                      ('metrics.json', 'predictions.jsonl', 'split_manifest.json', 'environment.json',
                                       'environments.json')}
    atomic_json(root / 'verification.json', verification)
    manifest.update(status='SUCCESS', gpu_experiment='SUCCESS', active_stage='complete', completed=list(CHILDREN) + ['aggregation'],
                    data_provenance=children['E0']['data_provenance'], protocol=protocol,
                    verification_sha256=file_hash(root / 'verification.json'),
                    completed_utc=datetime.now(timezone.utc).isoformat())
    manifest.pop('error', None)
    atomic_json(root / 'manifest.json', manifest)
    report(root)
    append_report(root, metrics, selected_step)
    print(str(root / 'report.md'), flush=True)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print('Usage: python -m benchmarks.zh_reliability.aggregate_massive EXPERIMENT_DIR', file=sys.stderr)
        return 2
    root = Path(args[0])
    try:
        aggregate(root)
        return 0
    except Exception as error:
        manifest = read_json(root / 'manifest.json') if (root / 'manifest.json').exists() else {}
        manifest.update(status='FAILED', gpu_experiment='FAILED', active_stage='aggregation',
                        error={'stage': 'aggregation', 'type': type(error).__name__, 'reason': str(error)})
        atomic_json(root / 'manifest.json', manifest)
        (root / 'aggregation-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('aggregation failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
