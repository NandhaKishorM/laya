"""Fit bounded scalar temperatures from unscaled calibration logits only."""
import copy
import math
from .data import atomic_json, digest
from .metrics import log_probs


def fit_temperature(rows, minimum=30):
    from laya.common import TEMP_MIN, TEMP_MAX
    if len(rows) < minimum:
        return {'status': 'SKIPPED', 'n': len(rows), 'reason': 'insufficient calibration samples',
                'temperature': None, 'boundary_hit': False}
    def loss(beta):
        return sum(-log_probs(r['raw_logits'], 1 / beta)[r['option_order'].index(r['gold'])]
                   for r in rows) / len(rows)
    # NLL is convex in inverse temperature. Bounded golden-section, no new dependency.
    lo, hi = 1 / TEMP_MAX, 1 / TEMP_MIN
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        x, y = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if loss(x) < loss(y):
            hi = y
        else:
            lo = x
    candidates = [TEMP_MIN, TEMP_MAX, 1., 1 / ((lo + hi) / 2)]
    t = min(candidates, key=lambda v: (loss(1 / v), abs(v - 1)))
    return {'status': 'SUCCESS', 'n': len(rows), 'temperature': t,
            'boundary_hit': t in (TEMP_MIN, TEMP_MAX), 'nll_t1': loss(1.), 'nll_fitted': loss(1 / t)}


def calibrate(rows, model, split, original, path=None, minimum=30, smoke=False):
    if minimum < 30 and not smoke:
        raise ValueError('minimum < 30 is smoke_only')
    if not rows or any(r['split'] != 'calibration' for r in rows):
        raise ValueError('fit only on calibration partition')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate calibration predictions')
    for r in rows:
        if (split['assignments'].get(r['id']) != 'calibration' or r.get('model') != model
                or r.get('sample_hash') != split['row_hashes'].get(r['id'])):
            raise ValueError('calibration model/partition mismatch')
    cfg = copy.deepcopy(original)
    base = cfg.get('lang_temperatures', {}).get('zh', {
        'temperature': cfg['temperature'], 'temperature_by_options': cfg['temperature_by_options']})
    override = copy.deepcopy(base)
    fitted = {}
    for kind, index in [('choice', 0), ('noul', 2)]:
        fitted[kind] = fit_temperature([r for r in rows if r['question']['type'] == kind], minimum)
        if fitted[kind]['status'] == 'SUCCESS':
            override['temperature'][index] = fitted[kind]['temperature']
            override['temperature_by_options'] = {
                k: v for k, v in override['temperature_by_options'].items() if not k.startswith(kind + ':')}
    out = {'model': model, 'data_hash': split['data_hash'], 'split_hash': split['split_hash'],
           'calibration_hash': digest([split['row_hashes'][r['id']] for r in rows]),
           'calibration_ids': [r['id'] for r in rows], 'smoke_only': smoke,
           'minimum_per_type': minimum, 'fitted': fitted, 'original': original,
           'lang_temperatures': {**cfg.get('lang_temperatures', {}), 'zh': override}}
    if path:
        atomic_json(path, out)
    return out


def validate_calibration(artifact, model, split, allow_smoke=False):
    from laya.common import clamp_temperature
    if artifact['model'] != model or artifact['data_hash'] != split['data_hash']:
        raise ValueError('calibration model or data hash mismatch')
    ids = artifact['calibration_ids']
    if (artifact['split_hash'] != split['split_hash'] or len(set(ids)) != len(ids)
            or any(split['assignments'].get(i) != 'calibration' for i in ids)
            or artifact['calibration_hash'] != digest([split['row_hashes'][i] for i in ids])):
        raise ValueError('calibration partition/hash mismatch')
    if artifact['smoke_only'] and not allow_smoke:
        raise ValueError('smoke calibration requires explicit --allow-smoke-calibration')
    for cfg in artifact['lang_temperatures'].values():
        if len(cfg['temperature']) != 3:
            raise ValueError('invalid temperature vector')
        for t in list(cfg['temperature']) + list(cfg['temperature_by_options'].values()):
            if not isinstance(t, (int, float)) or not math.isfinite(t) or clamp_temperature(t) != t:
                raise ValueError('illegal exported temperature')
    for kind, index in [('choice', 0), ('noul', 2)]:
        fit = artifact['fitted'][kind]
        cfg = artifact['lang_temperatures']['zh']
        if fit['status'] == 'SUCCESS' and (cfg['temperature'][index] != fit['temperature'] or
                any(k.startswith(kind + ':') for k in cfg['temperature_by_options'])):
            raise ValueError('fitted temperature shadowed by stale option bucket')
    return artifact['lang_temperatures']
