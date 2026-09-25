"""Fit and persist per-bucket temperatures for Laya logits.

Ports the notebook LBFGS fitter (NLL on softmax(z/T)) and groups records by
`temp_bucket(qtype, k)` so the runtime's `temperature_by_options` lookup has a writer.
Fitting is CPU-side: records are `(qtype, logits, target, k)` and do not load Hub weights.

Two sample floors, on purpose:

- `MIN_BUCKET_N` is the per-bucket floor. Buckets with fewer examples are omitted from
  `temperature_by_options`; the type-level scalar covers them at lookup time.
- `MIN_TYPE_N` is the lower floor used only for that type-level scalar. It stays at the
  previous floor of 10 so a realistic dataset that fills no single bucket still gets a
  temperature instead of remaining at 1.0. Do not raise it to `MIN_BUCKET_N`.

Fitted temperatures go through `clamp_temperature` and therefore `[TEMP_MIN, TEMP_MAX]`
from `laya.common`, the same guard checkpoint load uses. There is no second pair of
bounds in this module.
"""
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    clamp_temperature,
    collate_items,
    ece_score,
    temp_bucket,
)

# Per-bucket floor. Below this, `temperature_by_options` omits the bucket.
MIN_BUCKET_N = 2000
# Type-level floor only. Lower than MIN_BUCKET_N; see the module docstring.
MIN_TYPE_N = 10
# Fraction of each bucket held out when `compute_ece=True`. Unused otherwise.
ECE_HOLDOUT_FRAC = 0.2
# Payload schema. Files written before identity was recorded have no version key;
# readers treat that absence as version 1.
CALIBRATION_VERSION = 2
N_QTYPES = len(QTYPES)

# (qtype:int, logits:1d, target:1d, k:int)
Record = Tuple[int, Any, Any, int]

_CONFIG_IDENTITY_SKIP = ("temperature", "temperature_by_options")


def _vec(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _pairs_to_tensors(pairs: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
    kmax = max(len(_vec(z)) for z, _ in pairs)
    z_mat = torch.full((len(pairs), kmax), -1e4)
    t_mat = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        z = _vec(z)
        t = _vec(t)
        n = min(len(z), len(t))
        z_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(z[:n]))
        t_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(t[:n]))
    return z_mat, t_mat


def fit_one_temperature(pairs: Sequence, min_n: Optional[int] = None) -> float:
    """Fit one scalar T by NLL + LBFGS on log T.

    The result is `clamp_temperature` of the optimised scale, so it lies in
    `[TEMP_MIN, TEMP_MAX]` (or is the neutral 1.0 when the value is not a number).
    Returns 1.0 when fewer than `min_n` pairs are given. `min_n` defaults to
    `MIN_BUCKET_N` (the per-bucket floor). Type-level fits pass `MIN_TYPE_N`, which
    is lower, so a dataset that fills no bucket still gets a scalar instead of
    staying at 1.0.
    """
    if min_n is None:
        min_n = MIN_BUCKET_N
    sel = list(pairs)
    if len(sel) < min_n:
        return 1.0
    z_mat, t_mat = _pairs_to_tensors(sel)
    with torch.enable_grad():
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(t_mat * torch.log_softmax(z_mat / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
    fitted = float(log_t.detach().exp().item())
    return clamp_temperature(fitted, TEMP_MIN, TEMP_MAX)


def _iter_records(records: Iterable) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
    out = []
    for rec in records:
        if len(rec) == 4:
            qtype, logits, target, k = rec
        elif len(rec) == 3:
            qtype, logits, target = rec
            k = len(_vec(logits))
        else:
            raise ValueError("record must be (qtype, logits, target[, k])")
        logits = _vec(logits)
        target = _vec(target)
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        out.append((int(qtype), logits[:k], target[:k], k))
    return out


def _softmax(z, t_scale: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t_scale))
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _ece_report(recs, temperature, temperature_by_options) -> Dict[str, Any]:
    conf_b, cor_b, conf_a, cor_a = [], [], [], []
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        p0 = _softmax(z[:k], 1.0)
        conf_b.append(float(p0.max()))
        cor_b.append(float(int(p0.argmax()) == y))
        t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        p1 = _softmax(z[:k], t_scale)
        conf_a.append(float(p1.max()))
        cor_a.append(float(int(p1.argmax()) == y))
    return {
        "ece_before": ece_score(np.asarray(conf_b), np.asarray(cor_b)),
        "ece_after": ece_score(np.asarray(conf_a), np.asarray(cor_a)),
        "n": float(len(recs)),
    }


def _group_records(recs):
    by_type = {qt: [] for qt in range(N_QTYPES)}
    by_bucket = {}
    for qt, z, t, k in recs:
        if qt in by_type:
            by_type[qt].append((z, t))
        key = temp_bucket(qt, k)
        by_bucket.setdefault(key, []).append((z, t))
    return by_type, by_bucket


def _ece_split(recs, seed):
    """Split records into fit and eval, stratified by `temp_bucket`.

    Returns `(fit_recs, eval_recs, excluded_buckets)`. The split is a deterministic
    partition (`numpy.random.RandomState(seed)`): every record is in exactly one side.
    A bucket whose fit side would fall below `MIN_BUCKET_N` is not split; all of its
    records stay in `fit_recs` and its key is listed in `excluded_buckets`.
    """
    by_bucket = {}
    for i, rec in enumerate(recs):
        qt, _z, _t, k = rec
        by_bucket.setdefault(temp_bucket(qt, k), []).append(i)

    rng = np.random.RandomState(seed)
    fit_idx = []
    eval_idx = []
    excluded = []
    for key in sorted(by_bucket):
        idxs = by_bucket[key]
        n = len(idxs)
        n_eval = int(round(n * ECE_HOLDOUT_FRAC))
        if n - n_eval < MIN_BUCKET_N:
            fit_idx.extend(idxs)
            excluded.append(key)
            continue
        perm = rng.permutation(n)
        for j in perm[:n_eval]:
            eval_idx.append(idxs[int(j)])
        for j in perm[n_eval:]:
            fit_idx.append(idxs[int(j)])

    fit_idx.sort()
    eval_idx.sort()
    fit_recs = [recs[i] for i in fit_idx]
    eval_recs = [recs[i] for i in eval_idx]
    return fit_recs, eval_recs, excluded


def _fit_groups(by_type, by_bucket):
    temperature = [1.0] * N_QTYPES
    for qt in range(N_QTYPES):
        pairs = by_type[qt]
        # MIN_TYPE_N, not MIN_BUCKET_N: the scalar has to exist when every bucket is small.
        if len(pairs) >= MIN_TYPE_N:
            temperature[qt] = fit_one_temperature(pairs, min_n=MIN_TYPE_N)

    temperature_by_options = {}
    for key, pairs in by_bucket.items():
        if len(pairs) < MIN_BUCKET_N:
            continue
        temperature_by_options[key] = fit_one_temperature(pairs)
    return temperature, temperature_by_options


def fit_temperature_map(records: Iterable, compute_ece: bool = False, seed: int = 0) -> Dict[str, Any]:
    """Fit type-level scalars and per-bucket temperatures.

    `MIN_BUCKET_N` is the per-bucket floor: smaller buckets are omitted from
    `temperature_by_options` and the type-level scalar covers them. `MIN_TYPE_N` is
    the separate, lower floor for that scalar only.

    `compute_ece=False` (the default, and the path `Agent.fit_temperatures` stores)
    fits on every record and returns no `report` key. `seed` is ignored on this path.

    `compute_ece=True` holds out `ECE_HOLDOUT_FRAC` of each bucket, stratified by
    `temp_bucket`, using `seed` so the same records always split the same way. Temperatures
    are fit on the remainder only and ECE is scored only on the held-out records.
    `report["n"]` is the number of records passed in; `report["n_eval"]` is the held-out
    count the ECE rests on. A bucket that would drop below `MIN_BUCKET_N` after the
    holdout is fit on all of its records, left out of the eval set, and named in
    `report["buckets_excluded_from_eval"]` instead of being dropped. `n_by_bucket`
    always counts the full input, including when the fit itself used a subset.
    """
    recs = _iter_records(records)
    all_by_type, all_buckets = _group_records(recs)
    n_by_bucket = {key: len(pairs) for key, pairs in all_buckets.items()}

    if compute_ece:
        _fit_recs, eval_recs, excluded = _ece_split(recs, seed)
        by_type, by_bucket = _group_records(_fit_recs)
    else:
        eval_recs = None
        excluded = []
        by_type, by_bucket = all_by_type, all_buckets

    temperature, temperature_by_options = _fit_groups(by_type, by_bucket)
    out: Dict[str, Any] = {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
    }
    if compute_ece:
        report = _ece_report(eval_recs, temperature, temperature_by_options)
        # `n` stays the size of the input. `n_eval` is the held-out count ECE used.
        report["n"] = float(len(recs))
        report["n_eval"] = float(len(eval_recs))
        report["buckets_excluded_from_eval"] = excluded
        out["report"] = report
    return out


fit_temperatures = fit_temperature_map


def records_from_labeled(agent, pairs: Sequence) -> List[Record]:
    """Collect CPU records from `(state, questions, targets)`.

    Logits are the raw rows `_decode_answers` scales, not the probabilities it returns.
    `_encode_state` builds one state's items, `_forward` runs them, and `_option_logits`
    slices each question to the same marker width `_decode_answers` divides by temperature.
    No Hub download and no `model.safetensors` write. Tests stay weight-free by passing
    synthetic records to `fit_temperature_map` instead of calling this.
    """
    # Imported lazily: `agent` imports this module at load, and `_option_logits` is defined
    # on the way through that import.
    from .agent import _option_logits

    records: List[Record] = []
    for state, questions, targets in pairs:
        ids = list(questions.keys())
        if not ids:
            continue
        for qid in ids:
            agent._check_question(qid, questions[qid])
        internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
        items = agent._encode_state(state, ids, internal)
        batch = collate_items([items], agent.tok.pad_token_id)
        if batch is None:
            continue
        logits, _act = agent._forward(batch)
        rows = _option_logits(logits, items, 0)
        for j, qid in enumerate(ids):
            k = len(items[j]["markers"])
            qt = int(items[j]["qtype"])
            records.append((qt, _vec(rows[j]), _vec(targets[qid])[:k], k))
    return records


def _config_identity(cfg):
    """Checkpoint config without the temperature fields this file itself stores."""
    if cfg is None:
        return None
    return {str(k): cfg[k] for k in cfg if k not in _CONFIG_IDENTITY_SKIP}


def calibration_payload(
    temperature,
    temperature_by_options,
    model_id_or_path=None,
    subfolder=None,
    config=None,
) -> Dict[str, Any]:
    """JSON body written by `Agent.save_calibration` (no weights).

    `version` is `CALIBRATION_VERSION`. Payloads written before identity was recorded have
    no version key; `apply_calibration_payload` treats that as version 1 and still loads.

    `model_id_or_path`, `subfolder`, and `config` say which checkpoint the map was fitted
    against. `config` is `rl_agent_config.json` without `temperature` /
    `temperature_by_options` (those live at the top of this payload).
    """
    return {
        "version": CALIBRATION_VERSION,
        "temperature": [float(x) for x in temperature],
        "temperature_by_options": {
            str(k): float(v) for k, v in dict(temperature_by_options).items()
        },
        "model_id_or_path": model_id_or_path,
        "subfolder": subfolder,
        "config": _config_identity(config),
    }


def _warn_if_identity_mismatch(obj, payload) -> None:
    recorded_model = payload.get("model_id_or_path")
    recorded_sub = payload.get("subfolder")
    recorded_cfg = payload.get("config")
    current_model = getattr(obj, "model_id_or_path", None)
    current_sub = getattr(obj, "subfolder", None)
    current_cfg = _config_identity(getattr(obj, "cfg", None))
    if (recorded_model, recorded_sub, recorded_cfg) == (current_model, current_sub, current_cfg):
        return
    cfg_note = ""
    if recorded_cfg != current_cfg:
        cfg_note = " Checkpoint config identity differs."
    warnings.warn(
        "calibration was fitted for model_id_or_path=%r subfolder=%r but this agent is "
        "model_id_or_path=%r subfolder=%r.%s Loading the temperatures anyway."
        % (recorded_model, recorded_sub, current_model, current_sub, cfg_note),
        UserWarning,
        stacklevel=3,
    )


def _rejected_temperatures(entries):
    """Names whose raw value is not the number `clamp_temperature` kept, matching checkpoint load."""
    rejected = []
    for name, raw, applied in entries:
        try:
            if float(raw) == applied:
                continue
        except (TypeError, ValueError):
            pass
        rejected.append("%s=%r -> %g" % (name, raw, applied))
    return rejected


def _install_temperatures(obj, temperature, temperature_by_options, warn: bool = True) -> None:
    """Store a map the way checkpoint load does: raw values, and the clamped ones inference reads.

    `warn=False` is the fitter path: those values were already passed through `clamp_temperature`,
    so a second pass must not report them as a bad file.
    """
    raw_t = list(temperature)
    applied_t = [clamp_temperature(t) for t in raw_t]
    raw_b = {str(k): v for k, v in dict(temperature_by_options).items()}
    applied_b = {k: clamp_temperature(v) for k, v in raw_b.items()}
    obj.temperature_raw = raw_t
    obj.temperature = applied_t
    obj.temperature_by_options_raw = raw_b
    obj.temperature_by_options = applied_b
    if not warn:
        return
    entries = [("temperature[%d]" % i, raw, applied_t[i]) for i, raw in enumerate(raw_t)]
    entries += [(key, raw_b[key], applied_b[key]) for key in raw_b]
    rejected = _rejected_temperatures(entries)
    if rejected:
        warnings.warn(
            "laya: calibration temperatures are invalid or outside [%g, %g]; "
            "using %s. Treat confidence from the affected entries as uncalibrated."
            % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
            RuntimeWarning,
            stacklevel=3,
        )


def apply_calibration_payload(obj, payload: Dict[str, Any]) -> None:
    """Copy `temperature` and `temperature_by_options` from a calibration payload onto `obj`.

    A missing `version` is version 1 (temperatures only, no checkpoint identity). Version
    `CALIBRATION_VERSION` records the checkpoint the map was fitted for; a mismatch warns
    and still loads, so an older file never becomes a hard failure. Each value is passed
    through `clamp_temperature`, so a non-numeric or out-of-range entry cannot crash a later
    forward the way an unclamped zero used to.
    """
    version = payload.get("version", 1)
    if version is None:
        version = 1
    temps = payload.get("temperature")
    if temps is None or len(temps) != N_QTYPES:
        raise ValueError("calibration JSON must contain temperature: [3 floats]")
    if int(version) >= CALIBRATION_VERSION:
        _warn_if_identity_mismatch(obj, payload)
    _install_temperatures(obj, temps, payload.get("temperature_by_options") or {})
