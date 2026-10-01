"""Metrics use valid decision probabilities, never action/entropy confidence."""
import math
from collections import defaultdict
import numpy as np


def log_probs(logits, temperature=1.0, mask=None):
    from laya.common import TEMP_MIN, TEMP_MAX
    if not math.isfinite(temperature) or not TEMP_MIN <= temperature <= TEMP_MAX:
        raise ValueError('temperature outside runtime range')
    z = np.asarray(logits, dtype=np.float64)
    if mask is not None:
        z = z[np.asarray(mask, dtype=bool)]
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError('invalid decision logits')
    z = z / temperature
    z -= z.max()
    return z - np.log(np.exp(z).sum())


def probability_record(row, logits, temperature):
    from .data import digest, options
    order = options(row)
    if len(logits) != len(order):
        raise ValueError('logits/options mismatch')
    lp = log_probs(logits, temperature)
    p = np.exp(lp)
    return {**row, 'sample_hash': digest(row), 'option_order': order, 'gold': row['label'], 'raw_logits': list(logits),
            'temperature': temperature, 'probabilities': p.tolist(), 'log_probabilities': lp.tolist(),
            'predicted': order[int(p.argmax())], 'confidence': float(p.max()),
            'p_true': float(p[1]) if row['question']['type'] == 'noul' else None}


def quality(rows):
    if not rows:
        return {'n': 0, 'accuracy': None, 'macro_f1': None, 'nll': None, 'brier': None, 'ece': None,
                'reason': 'empty group'}
    correct = np.array([r['predicted'] == r['gold'] for r in rows], dtype=float)
    conf = np.array([max(r['probabilities']) for r in rows])
    nll, brier = [], []
    task_rows = defaultdict(list)
    for r in rows:
        task_rows[r['task_id']].append(r)
        g = r['option_order'].index(r['gold'])
        nll.append(-r['log_probabilities'][g])
        brier.append(sum((p - (j == g)) ** 2 for j, p in enumerate(r['probabilities'])))
    f1 = []
    for task in task_rows.values():
        labels = sorted({x for r in task for x in r['option_order']})
        scores = []
        for label in labels:
            tp = sum(r['predicted'] == r['gold'] == label for r in task)
            fp = sum(r['predicted'] == label and r['gold'] != label for r in task)
            fn = sum(r['predicted'] != label and r['gold'] == label for r in task)
            scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        f1.append(float(np.mean(scores)))
    bucket = np.minimum(14, (conf * 15).astype(int))
    ece = sum(np.mean(bucket == b) * abs(correct[bucket == b].mean() - conf[bucket == b].mean())
              for b in range(15) if (bucket == b).any())
    # Ties broken lexically by stable ID, independently of labels/correctness.
    ordered = sorted(rows, key=lambda r: (-max(r['probabilities']), r['id']))
    risk = []
    for c in (.25, .5, .75, .9, 1.):
        k = max(1, math.ceil(c * len(rows)))
        risk.append({'target_coverage': c, 'coverage': k / len(rows), 'n': k,
                     'risk': sum(r['predicted'] != r['gold'] for r in ordered[:k]) / k})
    return {'n': len(rows), 'accuracy': float(correct.mean()), 'macro_f1': float(np.mean(f1)),
            'nll': float(np.mean(nll)), 'brier': float(np.mean(brier)), 'ece': float(ece),
            'risk_coverage': risk}


def summarize(rows):
    result = {'overall': quality(rows), 'definitions': {
        'macro_f1': 'equal mean of task-local macro F1; all task options, zero for undefined F1',
        'brier': 'mean sum of squared errors over all valid classes (including both noul classes)',
        'ece': '15 equal bins [i/15,(i+1)/15), last bin includes 1; confidence=max(p)',
        'risk_ties': 'descending max(p), then lexical sample ID; ceil target count; diagnostic only'}}
    for name, getter in [('task', lambda r: [r['task_id']]),
                         ('type', lambda r: [r['question']['type']]),
                         ('option_count', lambda r: [str(len(r['option_order']))]),
                         ('phenomenon', lambda r: r['tags']),
                         ('truncation', lambda r: [str(r.get('tokens', {}).get('truncated', False))])]:
        groups = defaultdict(list)
        for r in rows:
            for key in getter(r):
                groups[key].append(r)
        if name == 'type':
            for key in ('choice', 'noul'):
                groups.setdefault(key, [])
        result['by_' + name] = {k: quality(v) for k, v in sorted(groups.items())}
    pairs = defaultdict(list)
    for r in rows:
        if r.get('relation'):
            pairs[r['relation']['pair_id']].append(r)
    result['relations'] = {}
    for kind in ('preserve', 'flip'):
        selected = [v for v in pairs.values() if len(v) == 2 and v[0]['relation']['kind'] == kind]
        result['relations'][kind] = {
            'n_pairs': len(selected),
            'consistent_or_changed': (sum((v[0]['predicted'] == v[1]['predicted']) == (kind == 'preserve')
                                          for v in selected) / len(selected)) if selected else None,
            'both_correct': (sum(all(r['predicted'] == r['gold'] for r in v) for v in selected)
                             / len(selected)) if selected else None}
    return result


def paired_bootstrap(before, after, repeats=1000, seed=20260924, min_groups=20):
    a, b = {r['id']: r for r in before}, {r['id']: r for r in after}
    if a.keys() != b.keys() or any(a[k]['group_id'] != b[k]['group_id'] or a[k]['gold'] != b[k]['gold'] for k in a):
        raise ValueError('paired bootstrap requires identical samples/groups/labels')
    groups = sorted({r['group_id'] for r in before})
    if len(groups) < min_groups:
        return {'status': 'SKIPPED', 'reason': 'too few independent groups', 'n_groups': len(groups)}
    ids = {g: [k for k in a if a[k]['group_id'] == g] for g in groups}
    rng = np.random.default_rng(seed)
    keys = ('accuracy', 'macro_f1', 'nll', 'brier', 'ece')
    deltas = {k: [] for k in keys}
    for _ in range(repeats):
        selected = [i for g in rng.choice(groups, len(groups), replace=True) for i in ids[g]]
        qa, qb = quality([a[i] for i in selected]), quality([b[i] for i in selected])
        for k in keys:
            deltas[k].append(qb[k] - qa[k])
    qa, qb = quality(before), quality(after)
    return {'status': 'SUCCESS', 'n_groups': len(groups), 'repeats': repeats, 'seed': seed,
            'after_minus_before': {k: {'delta': qb[k] - qa[k], 'ci95': np.quantile(v, [.025, .975]).tolist()}
                                   for k, v in deltas.items()}}
