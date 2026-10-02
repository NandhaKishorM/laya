"""Fit bounded scalar temperatures from unscaled calibration logits only."""
import copy
import decimal
import math
from .data import atomic_json, digest

FIT_METHOD = 'decimal50-derivative96-temperature10-v1'


def fit_temperature(rows, minimum=30):
    from laya.common import TEMP_MIN, TEMP_MAX
    if minimum < 1:
        raise ValueError('minimum calibration sample count must be positive')
    if len(rows) < minimum:
        return {'status': 'SKIPPED', 'n': len(rows), 'reason': 'insufficient calibration samples',
                'temperature': None, 'boundary_hit': False}
    # NumPy/libm loss comparisons become indistinguishable near the optimum. Use
    # correctly rounded Decimal exp/ln and a fixed context, independent of ambient
    # precision/traps. The derivative of NLL is monotone in inverse temperature.
    context = decimal.Context(prec=50, rounding=decimal.ROUND_HALF_EVEN,
                              Emin=-999999, Emax=999999, capitals=1, clamp=0,
                              traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])
    with decimal.localcontext(context):
        D = decimal.Decimal
        zero, one = D(0), D(1)
        prepared = []
        for row in rows:
            try:
                logits = [float(x) for x in row['raw_logits']]
                order = row['option_order']
                gold = order.index(row['gold'])
            except (KeyError, TypeError, ValueError, OverflowError) as e:
                raise ValueError('invalid calibration logits, option order or gold label') from e
            if not logits or len(logits) != len(order) or not all(math.isfinite(x) for x in logits):
                raise ValueError('calibration logits must be finite and match option order')
            values = [D.from_float(x) for x in logits]
            maximum = max(values)
            margins = [x - maximum for x in values]
            prepared.append((tuple(sorted(margins)), margins[gold]))
        # Row/option permutations do not change the mathematical objective or its
        # finite-precision accumulation order. Keep the exact input float values.
        prepared.sort()

        def derivative(beta):
            total = zero
            for margins, gold in prepared:
                weights = [(x * beta).exp() if x else one for x in margins]
                total += sum((w * x for w, x in zip(weights, margins)), zero) / sum(weights, zero) - gold
            return total

        def loss(temperature):
            beta = one / temperature
            return sum((sum(((x * beta).exp() if x else one for x in margins), zero).ln() - beta * gold
                        for margins, gold in prepared), zero) / len(prepared)

        lower, upper = D.from_float(float(TEMP_MIN)), D.from_float(float(TEMP_MAX))
        lo, hi = one / upper, one / lower
        if all(all(x == zero for x in margins) for margins, _ in prepared):
            estimate = one
        elif derivative(lo) >= zero:
            estimate = upper
        elif derivative(hi) <= zero:
            estimate = lower
        else:
            for _ in range(96):
                mid = (lo + hi) / 2
                value = derivative(mid)
                if value == zero:
                    lo = hi = mid
                    break
                if value < zero:
                    lo = mid
                else:
                    hi = mid
            estimate = one / ((lo + hi) / 2)
        estimate = min(upper, max(lower, estimate.quantize(D('1e-10'))))
        candidates = sorted({lower, upper, one, estimate}, key=lambda t: (abs(t - one), t))
        losses = {t: loss(t) for t in candidates}
        best = min(losses.values())
        # Flat or numerically indistinguishable fits prefer T=1, then the nearest
        # candidate. This absolute NLL tolerance is far below reported precision.
        t = next(t for t in candidates if losses[t] - best <= D('1e-30'))
        nll_t1, nll_fitted = float(losses[one]), float(losses[t])
        if not math.isfinite(nll_t1) or not math.isfinite(nll_fitted):
            raise ValueError('calibration NLL exceeds finite float range')
        return {'status': 'SUCCESS', 'n': len(rows), 'temperature': float(t),
                'boundary_hit': t in (lower, upper), 'nll_t1': nll_t1, 'nll_fitted': nll_fitted}


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
    out = {'method': FIT_METHOD, 'model': model, 'data_hash': split['data_hash'], 'split_hash': split['split_hash'],
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
