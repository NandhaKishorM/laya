"""Validate and rescore saved runs offline. No model or API dependencies."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from metrics import summarize
from prompts import CRITERIA, digest, interpret, requests_for

ROOT = Path(__file__).resolve().parent


def load_cases():
    cases = [json.loads(line) for line in (ROOT / 'data/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    if digest(cases) != manifest['cases_sha256']:
        raise ValueError('Dataset hash mismatch')
    if digest({c['id']: requests_for(c) for c in cases}) != manifest['requests_sha256']:
        raise ValueError('Prompt hash mismatch')
    if len(cases) != 64 or len({c['id'] for c in cases}) != 64:
        raise ValueError('Expected 64 unique frozen cases')
    return cases, manifest


def audit_run(folder):
    cases, manifest = load_cases()
    by_id = {c['id']: c for c in cases}
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    for key in ['cases_sha256', 'requests_sha256']:
        if metadata[key] != manifest[key]:
            raise ValueError('Run uses a different frozen protocol')
    ids = metadata.get('evaluated_ids', list(by_id))
    modes = metadata.get('modes', ['choice', 'four_noul'])
    repeats = metadata['repeats']
    if (not isinstance(repeats, int) or repeats < 1 or not ids or len(set(ids)) != len(ids)
            or not set(ids) <= set(by_id) or not modes or len(set(modes)) != len(modes)
            or not set(modes) <= {'choice', 'four_noul'}):
        raise ValueError('Invalid run metadata')
    rows = [json.loads(line) for line in (folder / 'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(case_id, mode, repeat) for case_id in ids for mode in modes for repeat in range(repeats)}
    seen = set()
    for row in rows:
        key = (row['id'], row['mode'], row['repeat'])
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unexpected prediction')
        seen.add(key)
        case = by_id[row['id']]
        if row['expected'] != case['expected'] or row['family'] != case['family']:
            raise ValueError('Reference label/family was altered')
        if row['request_sha256'] != digest(requests_for(case)[row['mode']]):
            raise ValueError('Request hash mismatch')
        if row['status'] == 'error':
            if 'predicted' in row:
                raise ValueError('Failed requests must not have a scored prediction')
            continue
        if row['status'] != 'ok' or row.get('predicted') not in CRITERIA:
            raise ValueError('Invalid status or label')
        if not math.isfinite(row['elapsed_ms']) or row['elapsed_ms'] < 0:
            raise ValueError('Invalid latency')
        decoded = interpret(row['mode'], row['response']['answers'])
        for name, value in decoded.items():
            if row.get(name) != value:
                raise ValueError('Saved prediction differs from raw response')
    if seen != expected:
        raise ValueError('Incomplete run: missing predictions')
    return {mode: summarize([r for r in rows if r['mode'] == mode], expected_repeats=repeats) for mode in modes}


def verify_archive():
    source = json.loads((ROOT / 'SOURCE.json').read_text(encoding='utf-8'))
    for item in source['files']:
        name, expected = item['path'], item['sha256']
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid archive path')
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Archived source bytes changed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='One run containing metadata.json and raw.jsonl')
    parser.add_argument('--write-summary', type=Path, help='Optional destination for recomputed JSON')
    args = parser.parse_args()
    if args.run_dir is None:
        verify_archive()
    folders = [args.run_dir] if args.run_dir else [ROOT / 'results/v1' / name for name in ['jev', 'laya']]
    results = {p.name: audit_run(p) for p in folders}
    if args.write_summary:
        if args.write_summary.exists():
            raise SystemExit('Refusing to overwrite an existing summary')
        args.write_summary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n',
                                      encoding='utf-8', newline='\n')
    for name, modes in results.items():
        for mode, result in modes.items():
            print(f"{name} / {mode}: {result['correct']}/{result['n']}; "
                  f"{result['failed_requests']} failed requests; p50={result['timing']['p50_ms']} ms")


if __name__ == '__main__':
    main()
