"""Reproducible opt-in experiments: python -m research.benchmarks.zh_reliability --help."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import uuid

from .data import FIXTURES, SPLITS, atomic_json, partition, read_data
from .runtime import environment, select_device


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('check-env', 'import-data', 'smoke', 'eval', 'calibrate', 'report'):
        s = sub.add_parser(name)
        s.add_argument('--run-dir', type=Path, help='fresh output directory; existing runs are never overwritten')
        s.add_argument('--data', type=Path, default=FIXTURES)
        s.add_argument('--formal', action='store_true', help='require licensed data, documented human annotation and fixed splits')
        s.add_argument('--split-manifest', type=Path)
        s.add_argument('--seed', type=int, default=20260924)
        if name in ('smoke', 'eval', 'calibrate'):
            s.add_argument('--model', default='convaiinnovations/laya-multilingual')
            s.add_argument('--revision', default='main', help='resolved to immutable HF SHA before download')
            s.add_argument('--download', action='store_true', help='allow target-model-only download; default offline')
            s.add_argument('--device', default='cuda:0', help='unique visible GPU name, UUID or process cuda:index')
            s.add_argument('--dtype', choices=('fp16', 'fp32'), default='fp16')
            s.add_argument('--max-len', type=int, default=256 if name == 'smoke' else 512)
            s.add_argument('--head-max-len', type=int, default=192)
        if name in ('eval', 'smoke'):
            s.add_argument('--warmup', type=int, default=10)
            s.add_argument('--batches', type=int, default=30)
        if name == 'eval':
            s.add_argument('--split', choices=SPLITS, default='test')
            s.add_argument('--calibration', type=Path)
            s.add_argument('--allow-smoke-calibration', action='store_true')
            s.add_argument('--measure', action='store_true')
        if name in ('calibrate', 'smoke'):
            s.add_argument('--minimum', type=int, default=1 if name == 'smoke' else 30)
        if name == 'calibrate':
            s.add_argument('--smoke-only', action='store_true')
        if name == 'smoke':
            s.add_argument('--gpu', action='store_true', help='opt in to actual Laya inference, calibration and precision measurements')
        if name == 'report':
            s.add_argument('--source', type=Path, required=True, help='existing result directory (read only)')
    return p


def write_predictions(path, rows):
    # Runs are private until manifest finalization; each row retains raw unrounded decision logits.
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def report(run, destination=None):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    metrics = json.loads((run / 'metrics.json').read_text())
    text = '# Chinese decision reliability run\n\n'
    text += f"Status: **{manifest['status']}**. Run: `{run.name}`.\n\n"
    data_info = manifest.get('data_provenance', manifest.get('source_manifest', {}).get('data_provenance', {}))
    if 'public_human_annotated' in data_info.get('kinds', []):
        text += 'Public human-localized data with retained upstream annotation evidence; no independent local human review. '
        text += 'This derived task/subset is not the full source benchmark or evidence of general Chinese improvement. '
    elif data_info.get('formal'):
        text += 'Source/annotation declarations and fixed splits were validated; this does not establish representative sampling or independent label correctness. '
    else:
        text += 'Agent-authored fixtures are unverified diagnostics, not a standard evaluation or evidence of general Chinese improvement. '
    text += 'Action outputs are outside this evaluation. No test-set temperature selection.\n\n'
    text += 'Full slices, per-sample evidence and configuration: `metrics.json`, `predictions.jsonl`, '
    text += '`split_manifest.json`, `environment.json`, `manifest.json`.\n\n'
    conditions = [(k, v['overall']) for k, v in metrics.items() if isinstance(v, dict) and 'overall' in v]
    if 'overall' in metrics:
        conditions = [('evaluated split', metrics['overall'])]
    if conditions:
        text += '| Condition | n | Accuracy | Task macro F1 | NLL | Brier | ECE |\n'
        text += '|---|---:|---:|---:|---:|---:|---:|\n'
        for k, m in conditions:
            values = [str(m['n'])] + [f"{m[x]:.6f}" if m[x] is not None else 'null'
                                       for x in ('accuracy', 'macro_f1', 'nll', 'brier', 'ece')]
            text += '| ' + k + ' | ' + ' | '.join(values) + ' |\n'
        text += '\n'
    timing = [(k, v) for k, v in metrics.items() if isinstance(v, dict) and 'end_to_end' in v]
    if timing:
        text += '| Device / dtype | E2E batch p50 ms | E2E batch p95 ms | state/s | Model p50 ms | Peak allocated MiB |\n'
        text += '|---|---:|---:|---:|---:|---:|\n'
        for k, v in timing:
            e, m = v['end_to_end'], v['model_stage']
            text += (f"| {k} | {e['p50_batch_ms']:.3f} | {e['p95_batch_ms']:.3f} | "
                     f"{e['state_per_s']:.2f} | {m['p50_batch_ms']:.3f} | {e['max_memory_allocated']/2**20:.1f} |\n")
        text += '\nLatency is per batch (one state/decision); model stage excludes H2D/D2H.\n\n'
    details = {k: v for k, v in metrics.items() if k not in {x[0] for x in conditions + timing}}
    text += '## Verification and numerical diagnostics\n\n```json\n'
    text += json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + '\n```\n'
    text += '\n## Execution\n\n```json\n' + json.dumps(manifest, ensure_ascii=False, indent=2) + '\n```\n'
    target = Path(destination) if destination else run / 'report.md'
    target.write_text(text, encoding='utf-8')


def load_adapter(args, env, run):
    import torch
    from laya.agent import Agent
    from .runtime import Adapter, ensure_device, resolve_model
    device = select_device(args.device, env)
    model_input, identity = resolve_model(args.model, args.revision, not args.download, run)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    agent = Agent(str(model_input), device=device, fast=False, compile=False)
    ensure_device(agent, device)
    atomic_json(Path(run) / 'loading_memory.json', {'device': device,
        'max_memory_allocated': torch.cuda.max_memory_allocated(device),
        'max_memory_reserved': torch.cuda.max_memory_reserved(device), 'scope': 'model loading only'})
    return Adapter(agent, identity, args.max_len, args.head_max_len, args.dtype), model_input


def evaluate(adapter, rows, split, calibration=None, smoke=False, selection='test'):
    import copy
    from .calibration import validate_calibration
    from .metrics import summarize
    adapter.agent.lang_temperatures = (validate_calibration(calibration, adapter.identity, split, smoke)
        if calibration else copy.deepcopy(adapter.original['lang_temperatures']))
    condition = 'E1' if calibration else 'E0'
    selected = [r for r in rows if split['assignments'][r['id']] == selection]
    predictions = adapter.predict(selected, condition, split)
    return predictions, summarize(predictions)


def gpu_smoke(args, env, run, rows, split, manifest):
    import copy
    import numpy as np
    import torch
    from .calibration import calibrate
    from .metrics import paired_bootstrap
    from .runtime import measure
    device = select_device(args.device, env)
    results = {'devices': {args.device: {'status': 'AVAILABLE', 'device': device}}}
    manifest['gpu_experiment'] = 'PARTIAL'
    base_dir = run / 'base'
    base_dir.mkdir()
    adapter, _ = load_adapter(args, env, base_dir)
    manifest['model'] = adapter.identity
    subset = [
        r for kind in ('choice', 'noul') for r in [x for x in rows if x['question']['type'] == kind][:4]]
    results['official_parity'] = adapter.parity(subset)
    cal_rows = [r for r in rows if split['assignments'][r['id']] == 'calibration']
    base_preds, results['E0'] = evaluate(adapter, rows, split)
    cal_preds = adapter.predict(cal_rows, 'E0', split)
    artifact = calibrate(cal_preds, adapter.identity, split, adapter.original,
                         run / 'calibration.json', args.minimum, True)
    results['calibration_E1'] = artifact['fitted']
    if any(x['status'] == 'SKIPPED' for x in artifact['fitted'].values()):
        manifest['status'] = 'PARTIAL'
    fitted_preds, results['E1'] = evaluate(adapter, rows, split, artifact, True)
    all_predictions = base_preds + fitted_preds + cal_preds
    results['E1-E0'] = (paired_bootstrap(base_preds, fitted_preds) if args.formal
        else {'status': 'SKIPPED', 'reason': 'unverified fixtures cannot support population inference'})
    manifest['completed'].append('E0/E1')
    write_predictions(run / 'predictions.jsonl', all_predictions)
    atomic_json(run / 'metrics.json', results)
    atomic_json(run / 'manifest.json', manifest)
    # Precision comparisons use the original checkpoint temperatures and fixed samples.
    adapter.agent.lang_temperatures = copy.deepcopy(adapter.original['lang_temperatures'])
    numeric = {}
    for dtype in ('fp16', 'fp32'):
        key = args.device + '/' + dtype
        adapter.dtype = dtype
        adapter.agent.amp_enabled = dtype == 'fp16'
        adapter.agent.dtype = torch.float16 if dtype == 'fp16' else torch.float32
        numeric[dtype] = adapter.predict(subset, 'numeric-' + dtype, split)
        results[key] = measure(adapter, subset, args.warmup, args.batches)
        all_predictions += numeric[dtype]
        write_predictions(run / 'predictions.jsonl', all_predictions)
        atomic_json(run / 'metrics.json', results)
    delta = np.concatenate([np.abs(np.array(x['probabilities']) - y['probabilities'])
                            for x, y in zip(numeric['fp16'], numeric['fp32'])])
    results['numerical_comparisons'] = {args.device + '/fp16 vs ' + args.device + '/fp32': {
        'n': len(subset), 'argmax_agreement': float(np.mean([x['predicted'] == y['predicted']
                                                          for x, y in zip(numeric['fp16'], numeric['fp32'])])),
        'probability_abs_mean': float(delta.mean()), 'probability_abs_max': float(delta.max()),
        'probability_abs_p50_p95_p99': np.quantile(delta, [.5, .95, .99]).tolist()}}
    manifest['completed'].append('FP16/FP32 measurements on requested device')
    return results


def main(argv=None):
    args = parser().parse_args(argv)
    run = args.run_dir or Path('runs') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    if run.exists():
        print('refusing existing run directory: ' + str(run), file=sys.stderr)
        return 2
    run.mkdir(parents=True)
    manifest = {'run_id': run.name, 'status': 'RUNNING', 'command': args.command, 'completed': [],
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'smoke_only': args.command == 'smoke' or getattr(args, 'smoke_only', False),
                'gpu_experiment': 'NOT_RUN'}
    atomic_json(run / 'manifest.json', manifest)
    atomic_json(run / 'metrics.json', {'status': 'NOT_RUN'})
    write_predictions(run / 'predictions.jsonl', [])
    stage = 'environment'
    try:
        env = environment()
        atomic_json(run / 'environment.json', env)
        if args.command == 'report':
            import shutil
            stage = 'report'
            manifest['source_manifest'] = json.loads((args.source / 'manifest.json').read_text())
            for name in ('metrics.json', 'predictions.jsonl', 'split_manifest.json'):
                shutil.copyfile(args.source / name, run / name)
            shutil.copyfile(args.source / 'environment.json', run / 'source_environment.json')
            manifest['completed'].append('report generated from ' + str(args.source))
        else:
            stage = 'data'
            rows = read_data(args.data, args.formal)
            manifest['data_provenance'] = {
                'formal': args.formal,
                'kinds': sorted({r['provenance']['kind'] for r in rows}),
                'licenses': sorted({r['provenance'].get('license', 'unspecified') for r in rows}),
                'human_verified_rows': sum(r['provenance']['human_verified'] for r in rows),
                'upstream_review_rows': sum('upstream_annotation' in r['provenance'] for r in rows)}
            split = partition(rows, args.split_manifest, args.seed, args.formal)
            atomic_json(run / 'split_manifest.json', split)
            manifest['completed'].append('data/schema/group partitions')
            atomic_json(run / 'metrics.json', {'data_counts': split['counts'], 'groups': split['group_counts']})
            if args.command == 'check-env':
                manifest['completed'].append('CUDA execution probes')
                if env['gpu_status'] == 'FAILED':
                    raise RuntimeError('CUDA execution probe failed; see environment.json')
            elif args.command == 'smoke' and not args.gpu:
                import subprocess
                stage = 'offline tests'
                completed = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                            '-p', 'test_zh_*.py', '-v'], capture_output=True, text=True)
                (run / 'offline-tests.log').write_text(completed.stdout + completed.stderr)
                if completed.returncode:
                    raise RuntimeError('offline tests failed; see offline-tests.log')
                atomic_json(run / 'metrics.json', {'offline_tests': 'SUCCESS', 'gpu': 'NOT_RUN',
                    'reason': 'default smoke is offline; opt in using --gpu', 'data_counts': split['counts']})
                manifest['completed'].append('offline tests (tiny/mock inference only)')
                manifest['status'] = 'PARTIAL'
            elif args.command == 'smoke':
                stage = 'GPU smoke'
                if not env['gpus']:
                    manifest['status'] = 'SKIPPED'
                    manifest['reason'] = 'CUDA unavailable; run default offline smoke'
                else:
                    results = gpu_smoke(args, env, run, rows, split, manifest)
                    atomic_json(run / 'metrics.json', results)
                    manifest['gpu_experiment'] = ('SUCCESS' if manifest['status'] == 'RUNNING'
                                                  else manifest['status'])
            elif args.command not in ('check-env', 'import-data'):
                stage = 'model loading'
                adapter, _ = load_adapter(args, env, run)
                manifest['model'] = adapter.identity
                stage = args.command
                if args.command == 'eval':
                    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
                    preds, metrics = evaluate(adapter, rows, split, calibration, args.allow_smoke_calibration, args.split)
                    metrics['official_parity'] = adapter.parity([r for r in rows if split['assignments'][r['id']] == args.split][:4])
                    if args.measure:
                        from .runtime import measure
                        metrics['measurement'] = measure(adapter, [r for r in rows if split['assignments'][r['id']] == args.split],
                                                         args.warmup, args.batches)
                    write_predictions(run / 'predictions.jsonl', preds)
                elif args.command == 'calibrate':
                    from .calibration import calibrate
                    preds, metrics = evaluate(adapter, rows, split, selection='calibration')
                    artifact = calibrate(preds, adapter.identity, split, adapter.original, run / 'calibration.json',
                                         args.minimum, args.smoke_only)
                    metrics['fitted'] = artifact['fitted']
                    if any(x['status'] == 'SKIPPED' for x in artifact['fitted'].values()):
                        manifest['status'] = 'PARTIAL'
                    write_predictions(run / 'predictions.jsonl', preds)
                atomic_json(run / 'metrics.json', metrics)
                manifest['completed'].append(args.command)
                manifest['gpu_experiment'] = 'SUCCESS'
        if manifest['status'] == 'RUNNING':
            manifest['status'] = 'SUCCESS'
        atomic_json(run / 'manifest.json', manifest)
        report(run)
        print(str(run.resolve()))
        return 0
    except Exception as e:
        if manifest['gpu_experiment'] != 'NOT_RUN':
            manifest['gpu_experiment'] = 'FAILED'
        manifest.update(status='FAILED', error={'stage': stage, 'type': type(e).__name__, 'reason': str(e)})
        atomic_json(run / 'manifest.json', manifest)
        (run / 'error.log').write_text(traceback.format_exc())
        report(run)
        print(f'{stage}: {e}; diagnostics: {run}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
