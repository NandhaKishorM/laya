"""Offline replay of typed evidence contracts and archived CPU measurements.

This research module never imports Laya, loads weights or executes an action.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import constraint_kernel_oracles as oracle
import constraint_kernel_prepared as prepared
import constraint_kernel_v6 as reference

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stable(result):
    return {k: v for k, v in result.items() if k != 'wall_ms'}


@dataclass(frozen=True)
class LegacyDeclaration:
    """Negative authority control; deliberately not the v6 declaration class."""
    task_id: str
    revision: int
    source: str
    request: str


def declaration(row):
    data = {k: row['input'][k] for k in ('task_id', 'revision', 'source', 'request')}
    mode = row.get('declaration', 'current')
    if mode == 'none': return None
    if mode == 'forged': return data
    if mode == 'legacy': return LegacyDeclaration(**data)
    if mode == 'stale_id': data['task_id'] += 'x'
    if mode == 'stale_revision': data['revision'] += 1
    if mode == 'stale_source': data['source'] += ' '
    if mode == 'stale_request': data['request'] += ' '
    cls = reference.TypedDeclaration
    if mode == 'subclass':
        class Subclass(reference.TypedDeclaration): pass
        cls = Subclass
    try:
        result = cls(**data)
    except (ValueError, UnicodeError):
        return None
    if mode == 'bad_version': object.__setattr__(result, 'api_version', 'other')
    return result


def verify():
    manifest = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if actual != expected: raise ValueError('frozen export changed: ' + name)
    archive = json.loads((HERE / 'archive.json').read_text(encoding='utf-8'))
    rows = oracle.cases() + archive['extras']
    if len(rows) != 1129 or len({r['id'] for r in rows}) != 1129:
        raise ValueError('protocol fixture identity/count')
    if digest(rows) != manifest['protocol_sha256']:
        raise ValueError('protocol generator differs from archived inputs')
    controls = archive['dense']
    if len(controls) != 14 or len({r['id'] for r in controls}) != 14:
        raise ValueError('dense fixture identity/count')
    if digest(controls) != manifest['controls_sha256']:
        raise ValueError('dense inputs differ from archived workload')
    return dict(rows=rows, projections=archive['projections'], controls=controls, archive=archive)


def stats(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(ordered), median=statistics.median(ordered),
                p95=ordered[math.ceil(.95 * len(ordered)) - 1])


def audit_samples(plan):
    archived = plan['archive']['benchmark']
    dense = {r['id']: r for r in plan['archive']['dense']}
    protocol = {r['id']: r for r in plan['rows']}
    if set(archived['summary']) != set(dense): raise ValueError('control summary identity')
    if set(archived['cohorts']) != {r['cohort'] for r in protocol.values()}:
        raise ValueError('cohort summary identity')
    expected = {(pool, key, rep, engine)
                for pool, rows, repeats in [('controls', dense, 10), ('protocol', protocol, 5)]
                for key in rows for rep in range(repeats) for engine in ('v6', 'prepared')}
    seen = set()
    for sample in archived['timings']:
        key = tuple(sample[k] for k in ('pool', 'id', 'rep', 'engine'))
        if key not in expected or key in seen: raise ValueError('sample identity/multiplicity')
        ms = sample['ms']
        if type(ms) not in (int, float) or not math.isfinite(ms) or ms < 0:
            raise ValueError('sample duration')
        seen.add(key)
    if seen != expected or len(seen) != archived['observations']:
        raise ValueError('missing archived samples')
    for key, summary in archived['summary'].items():
        for engine in ('v6', 'prepared'):
            measured = stats([s['ms'] for s in archived['timings']
                              if s['pool'] == 'controls' and s['id'] == key and s['engine'] == engine])
            if measured != summary[engine]: raise ValueError('control statistics: ' + key)
        if summary['speedup_mean'] != summary['v6']['mean'] / summary['prepared']['mean']:
            raise ValueError('speedup ratio: ' + key)
    for cohort, summary in archived['cohorts'].items():
        for engine in ('v6', 'prepared'):
            measured = stats([s['ms'] for s in archived['timings'] if s['pool'] == 'protocol'
                              and s['engine'] == engine and protocol[s['id']]['cohort'] == cohort])
            if measured != summary[engine]: raise ValueError('cohort statistics: ' + cohort)
    return len(seen)


def audit():
    plan = verify()
    observed, claims, byte_checks = 0, 0, 0
    for pool in ('rows', 'projections', 'controls'):
        digests = plan['archive']['full_result_digests']['protocol' if pool == 'rows' else pool]
        if len(digests) != len(plan[pool]) or set(digests) != {r['id'] for r in plan[pool]}:
            raise ValueError('full-result archive identity/count')
        for row in plan[pool]:
            kwargs = dict(row['input'], declaration=declaration(row), proposal=row.get('proposal'))
            before, after = reference.check(**kwargs), prepared.check(**kwargs)
            if stable(before) != stable(after) or digest(stable(after)) != digests[row['id']]:
                raise ValueError('full-result mismatch: ' + row['id'])
            if not row.get('self_extension') and any(after[f] != v for f, v in row['expected'].items()):
                raise ValueError('independent stage oracle: ' + row['id'])
            if pool == 'rows' and row['expected_claims'] is not None:
                if [c['grounding'] for c in after['claims']] != row['expected_claims']:
                    raise ValueError('per-claim oracle: ' + row['id'])
                claims += len(row['expected_claims'])
            for field, records in [('source', after['source_records']), ('candidate', after['claims']),
                                   ('request', [after['task']] if after['task'] else [])]:
                raw = row['input'][field].encode('utf-8')
                for record in records:
                    lo, hi = record['span']
                    if not 0 <= lo < hi <= len(raw) or raw[lo:hi].decode() != record['raw']:
                        raise ValueError('original-byte span: ' + row['id'])
                    if pool == 'rows': byte_checks += 1
            for claim in after['claims']:
                witness = claim['witness']
                if witness:
                    if len(witness['source_indices']) != len(witness['source_spans']):
                        raise ValueError('witness multiplicity')
                    for index, span in zip(witness['source_indices'], witness['source_spans']):
                        if after['source_records'][index]['span'] != span:
                            raise ValueError('witness source span')
                        if pool == 'rows': byte_checks += 1
            if after['whole_answer_certified'] or after['proposal_used']:
                raise ValueError('authority boundary')
            observed += 1
    return dict(protocol_cases=len(plan['rows']), projections=len(plan['projections']),
                dense_controls=len(plan['controls']),
                current_replay_observations=observed, per_claim_checks=claims,
                original_byte_witness_checks=byte_checks, archived_timing_samples=audit_samples(plan),
                full_result_parity=True, model_inference_performed=False,
                historical_model_retention='aggregate provenance only; not rerun by this audit')


def benchmark(repeats):
    plan = verify(); rows = {r['id']: r for r in plan['archive']['dense']}
    approvals = {key: declaration(row) for key, row in rows.items()}
    jobs = [(key, engine) for key in rows for _ in range(repeats) for engine in ('v6', 'prepared')]
    random.Random(101820682).shuffle(jobs)
    timings, outputs = [], {}
    for key, engine in jobs:
        module = reference if engine == 'v6' else prepared
        started = time.perf_counter()
        result = module.check(**rows[key]['input'], declaration=approvals[key])
        elapsed = (time.perf_counter() - started) * 1000
        if key in outputs and outputs[key] != stable(result): raise ValueError('benchmark parity')
        outputs[key] = stable(result)
        if digest(stable(result)) != plan['archive']['full_result_digests']['controls'][key]:
            raise ValueError('benchmark full-result mismatch: ' + key)
        if any(result[f] != v for f, v in rows[key]['expected'].items()):
            raise ValueError('benchmark oracle')
        timings.append(dict(id=key, engine=engine, ms=elapsed))
    return {key: {engine: stats([t['ms'] for t in timings if t['id'] == key and t['engine'] == engine])
                  for engine in ('v6', 'prepared')} for key in rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true', help='also time fresh identical-input CPU checks')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100: parser.error('--repeats must be between 1 and 100')
    result = audit()
    if args.benchmark: result['fresh_benchmark_ms'] = benchmark(args.repeats)
    print(json.dumps(result, indent=2, allow_nan=False))
