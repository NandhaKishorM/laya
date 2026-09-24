"""Reproducible opt-in experiments: python -m benchmarks.zh_reliability --help."""
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
    for name in ('check-env', 'import-data', 'smoke', 'eval', 'calibrate', 'train-head', 'resume', 'report'):
        s = sub.add_parser(name)
        s.add_argument('--run-dir', type=Path, help='fresh output directory; existing runs are never overwritten')
        s.add_argument('--data', type=Path, default=FIXTURES)
        s.add_argument('--formal', action='store_true', help='require licensed data, documented human annotation and fixed splits')
        s.add_argument('--split-manifest', type=Path)
        s.add_argument('--seed', type=int, default=20260924)
        if name in ('smoke', 'eval', 'calibrate', 'train-head', 'resume'):
            s.add_argument('--model', default='convaiinnovations/laya-multilingual')
            s.add_argument('--revision', default='main', help='resolved to immutable HF SHA before download')
            s.add_argument('--download', action='store_true', help='allow target-model-only download; default offline')
            s.add_argument('--device', default='RTX 2080 Ti', help='unique visible GPU name, UUID or process cuda:index')
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
        if name in ('train-head', 'resume', 'smoke'):
            s.add_argument('--max-steps', type=int, default=2 if name == 'smoke' else 20)
            s.add_argument('--micro-batch', type=int, default=1)
            s.add_argument('--accumulation', type=int, default=8)
            s.add_argument('--checkpoint-every', type=int, default=1,
                           help='evaluate dev and save every N successful updates, plus the final update')
            s.add_argument('--lr', type=float, default=1e-4)
            s.add_argument('--weight-decay', type=float, default=.01)
            s.add_argument('--max-grad-norm', type=float, default=1.)
            s.add_argument('--full', action='store_true', help='explicitly allow more than 20 optimizer steps')
        if name == 'resume':
            s.add_argument('--checkpoint', type=Path, required=True, help='trusted local step directory')
        if name == 'smoke':
            s.add_argument('--gpu', action='store_true', help='opt in to actual Laya inference/training and dual-card measurements')
            s.add_argument('--train-device', default='TITAN RTX')
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
    text += 'Action outputs are not supervised or validated. No test-set model/temperature selection.\n\n'
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


def load_adapter(args, env, run, selector=None, source=None):
    import torch
    from laya.agent import Agent
    from .runtime import Adapter, ensure_device, resolve_model
    device = select_device(selector or args.device, env)
    model_input, identity = resolve_model(source or args.model, args.revision, not args.download, run)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    agent = Agent(str(model_input), device=device, fast=False, compile=False)
    ensure_device(agent, device)
    atomic_json(Path(run) / 'loading_memory.json', {'device': device,
        'max_memory_allocated': torch.cuda.max_memory_allocated(device),
        'max_memory_reserved': torch.cuda.max_memory_reserved(device), 'scope': 'model loading only'})
    return Adapter(agent, identity, args.max_len, args.head_max_len, args.dtype), model_input


def training_config(args):
    return {k: getattr(args, k) for k in ('seed', 'max_steps', 'micro_batch', 'accumulation', 'lr',
                                        'weight_decay', 'max_grad_norm', 'full', 'checkpoint_every')}


def evaluate(adapter, rows, split, calibration=None, smoke=False, selection='test'):
    from .calibration import validate_calibration
    from .metrics import summarize
    if calibration:
        adapter.agent.lang_temperatures = validate_calibration(calibration, adapter.identity, split, smoke)
    condition = ('E3' if calibration else 'E2') if adapter.identity['trained'] else ('E1' if calibration else 'E0')
    selected = [r for r in rows if split['assignments'][r['id']] == selection]
    predictions = adapter.predict(selected, condition, split, neutral=condition == 'E2')
    return predictions, summarize(predictions)


def gpu_smoke(args, env, run, rows, split, manifest):
    import gc
    import numpy as np
    import torch
    from .calibration import calibrate
    from .metrics import paired_bootstrap
    from .runtime import measure
    from .training import export_agent, train
    if args.max_steps > 20:
        raise ValueError('smoke never exceeds 20 steps; use train-head --full for longer training')
    all_predictions, results = [], {'devices': {}}
    available = {}
    for selector in (args.train_device, args.device):
        try:
            available[selector] = select_device(selector, env)
            results['devices'][selector] = {'status': 'AVAILABLE', 'device': available[selector]}
        except RuntimeError as e:
            results['devices'][selector] = {'status': 'NOT_RUN', 'reason': str(e)}
            manifest['status'] = 'PARTIAL'
    if not available:
        manifest['status'] = 'SKIPPED'
        return results
    eval_selector = args.device if args.device in available else args.train_device
    base_dir = run / 'base'
    base_dir.mkdir()
    adapter, _ = load_adapter(args, env, base_dir, eval_selector)
    manifest['gpu_experiment'] = 'PARTIAL'
    manifest['model'] = adapter.identity
    subset = [
        r for kind in ('choice', 'noul') for r in [x for x in rows if x['question']['type'] == kind][:4]]
    results['official_parity'] = adapter.parity(subset)
    cal_rows = [r for r in rows if split['assignments'][r['id']] == 'calibration']
    for trained in (False, True):
        if trained and args.train_device not in available:
            for key in ('training', 'E2', 'E3'):
                results[key] = {'status': 'NOT_RUN', 'reason': 'requested training GPU unavailable'}
            continue
        if trained:
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
            training_dir = run / 'train'
            training_dir.mkdir()
            adapter, model_input = load_adapter(args, env, training_dir, args.train_device)
            results['training'] = train(adapter, rows, split, training_dir / 'checkpoints', training_config(args))
            # Lock dev-selected weights and save expected logits before independent-card reload.
            reference = adapter.predict(cal_rows, 'reload-reference', split, neutral=True)
            exported = training_dir / 'exported'
            export_agent(adapter, model_input, exported)
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
            reload_dir = run / 'reloaded'
            reload_dir.mkdir()
            adapter, _ = load_adapter(args, env, reload_dir, eval_selector, source=str(exported))
            reloaded = adapter.predict(cal_rows, 'reload-check', split, neutral=True)
            delta = np.concatenate([np.abs(np.array(a['probabilities']) - b['probabilities'])
                                     for a, b in zip(reference, reloaded)])
            results['reload_comparison'] = {'max_probability_abs_difference': float(delta.max()),
                'mean_probability_abs_difference': float(delta.mean()),
                'argmax_agreement': float(np.mean([a['predicted'] == b['predicted'] for a, b in zip(reference, reloaded)])),
                'training_device': available[args.train_device], 'evaluation_device': available[eval_selector],
                'independent_evaluation': available[args.train_device] != available[eval_selector],
                'note': 'reload comparison; no claim of cross-device bitwise equality'}
            results['reloaded_official_parity'] = adapter.parity(subset)
        e = 'E2' if trained else 'E0'
        base_preds, results[e] = evaluate(adapter, rows, split)
        cal_preds = adapter.predict(cal_rows, e, split, neutral=trained)
        artifact = calibrate(cal_preds, adapter.identity, split, adapter.original,
                             run / ('calibration-head.json' if trained else 'calibration.json'), args.minimum, True)
        results['calibration_E3' if trained else 'calibration_E1'] = artifact['fitted']
        if any(x['status'] == 'SKIPPED' for x in artifact['fitted'].values()):
            manifest['status'] = 'PARTIAL'
        fitted_preds, results['E3' if trained else 'E1'] = evaluate(adapter, rows, split, artifact, True)
        all_predictions += base_preds + fitted_preds + cal_preds
        results[('E3-E2' if trained else 'E1-E0')] = (paired_bootstrap(base_preds, fitted_preds) if args.formal
            else {'status': 'SKIPPED', 'reason': 'unverified fixtures cannot support population inference'})
        manifest['completed'].append('E2/E3' if trained else 'E0/E1')
        write_predictions(run / 'predictions.jsonl', all_predictions)
        atomic_json(run / 'metrics.json', results)
        atomic_json(run / 'manifest.json', manifest)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    # Numerical comparisons use the same BASE checkpoint and fixed samples on each card, sequentially.
    numeric = {}
    for selector in dict.fromkeys((args.train_device, args.device)):
        if selector not in available:
            continue
        for dtype in ('fp16', 'fp32'):
            key = selector + '/' + dtype
            directory = run / ('numeric-' + str(len(numeric)))
            directory.mkdir()
            adapter, _ = load_adapter(args, env, directory, selector)
            adapter.dtype = dtype
            adapter.agent.amp_enabled = dtype == 'fp16'
            adapter.agent.dtype = torch.float16 if dtype == 'fp16' else torch.float32
            preds = adapter.predict(subset, 'numeric-' + dtype, split)
            results[key] = measure(adapter, subset, args.warmup, args.batches)
            numeric[key] = preds
            all_predictions += preds
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
            write_predictions(run / 'predictions.jsonl', all_predictions)
            atomic_json(run / 'metrics.json', results)
    pairs = [(args.train_device + '/fp16', args.train_device + '/fp32'),
             (args.device + '/fp16', args.device + '/fp32'),
             (args.train_device + '/fp16', args.device + '/fp16'),
             (args.train_device + '/fp32', args.device + '/fp32')]
    results['numerical_comparisons'] = {}
    for a, b in pairs:
        if a not in numeric or b not in numeric:
            results['numerical_comparisons'][a + ' vs ' + b] = {'status': 'NOT_RUN', 'reason': 'device unavailable'}
            continue
        delta = np.concatenate([np.abs(np.array(x['probabilities']) - y['probabilities'])
                                for x, y in zip(numeric[a], numeric[b])])
        results['numerical_comparisons'][a + ' vs ' + b] = {
            'n': len(subset), 'argmax_agreement': float(np.mean([x['predicted'] == y['predicted']
                                                              for x, y in zip(numeric[a], numeric[b])])),
            'probability_abs_mean': float(delta.mean()), 'probability_abs_max': float(delta.max()),
            'probability_abs_p50_p95_p99': np.quantile(delta, [.5, .95, .99]).tolist()}
    manifest['completed'].append('FP16/FP32 measurements on available requested devices')
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
                manifest['completed'].append('offline tests (tiny/mock models only)')
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
                adapter, model_input = load_adapter(args, env, run)
                manifest['model'] = adapter.identity
                from .metrics import summarize
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
                else:
                    from .training import train, export_agent
                    metrics = train(adapter, rows, split, run / 'checkpoints', training_config(args),
                                    args.checkpoint if args.command == 'resume' else None)
                    reference = adapter.predict([r for r in rows if split['assignments'][r['id']] == 'dev'],
                                                'E2', split, neutral=True)
                    export_agent(adapter, model_input, run / 'exported')
                    # Validate serialization on the same device before reporting successful export.
                    from laya.agent import Agent
                    from .runtime import Adapter, ensure_device, resolve_model
                    import gc
                    import torch
                    device = adapter.device
                    del adapter
                    gc.collect()
                    torch.cuda.empty_cache()
                    validation_dir = run / 'reload-validation'
                    validation_dir.mkdir()
                    source, identity = resolve_model(str(run / 'exported'), args.revision, True, validation_dir)
                    agent = Agent(str(source), device=device, fast=False, compile=False)
                    ensure_device(agent, device)
                    manifest['exported_model'] = identity
                    adapter = Adapter(agent, identity, args.max_len, args.head_max_len, args.dtype)
                    reloaded = adapter.predict([r for r in rows if split['assignments'][r['id']] == 'dev'], 'E2', split, True)
                    # Compare raw logits so neutral temperature cannot hide a serialization error.
                    import numpy as np
                    delta = max(float(np.max(np.abs(np.array(a['raw_logits']) - b['raw_logits'])))
                                for a, b in zip(reference, reloaded))
                    if delta != 0:
                        raise AssertionError('same-device reload changed raw logits: ' + str(delta))
                    metrics['reload_raw_logits_max_difference'] = delta
                    metrics['dev'] = summarize(reloaded)
                    write_predictions(run / 'predictions.jsonl', reloaded)
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
