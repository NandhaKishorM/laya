"""Shadow-eval gate: a candidate payload ships only when it does not regress.

Slice 4 of outcome-logged calibration. Fitting (slice 3) produces a payload; this
module decides whether the deployment should install it, by scoring the same pinned
eval records twice -- once under the live temperatures, once under the candidate's --
and comparing. Temperature scaling never changes an argmax, so accuracy is the wrong
thing to gate on: the comparison is calibration quality (ECE) plus what the abstention
gate would have done (coverage and gated error at the candidate thresholds).

Everything here is pure: eval records are the `(qtype, logits, target, k)` tuples
`records_from_labeled` builds, so the one model-dependent step (running the pinned
eval set) stays outside. `fit_payload_from_records(..., compute_ece=True)` already
scores its own holdout the same way; this module scores any records under any two
temperature maps, which is what makes it a gate rather than a report.
"""

import math

from .calibrate import N_QTYPES
from .common import temp_bucket
from .confidence import check_min_confidence

#: Bins in the expected-calibration-error histogram. Coarse on purpose: the gate reads
#: the number, it does not debug the curve.
ECE_BINS = 10


def _softmax_row(logits):
    """Softmax, shift-invariant: subtract the max so extreme logits cannot overflow."""
    shifted = [float(z) for z in logits]
    if not shifted:
        return []
    m = max(shifted)
    exps = [math.exp(z - m) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def _ece(pairs, n_bins=ECE_BINS):
    """Mean |accuracy - confidence| over confidence bins, weighted by bin mass."""
    bins = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append(correct)
    n = len(pairs)
    if n == 0:
        return 0.0
    return sum(abs(sum(b) / len(b) - (i + 0.5) / n_bins) * len(b) / n
               for i, b in enumerate(bins) if b)


def _cut_for(thresholds, min_confidence, bucket):
    if thresholds:
        return thresholds.get(bucket, thresholds.get("default", 0.0))
    return min_confidence if min_confidence is not None else 0.0


def evaluate_records(records, temperature, temperature_by_options, thresholds=None,
                     min_confidence=None):
    """Score eval records under one temperature map, returning (rows, summary).

    `temperature` is the per-type list and `temperature_by_options` the per-bucket dict
    from a fit (or the live agent's current values). Confidence is the calibrated max
    probability rounded the way the runtime reports it, so the gate below sees what a
    deployment would gate on. `thresholds` is a per-bucket map (validated like core's
    `min_confidence` maps) or `min_confidence` one cut for every answer; with neither,
    nothing abstains. An empty map is that same "no gate", not an error: it is what a
    fit on too little data returns.
    """
    from .confidence import check_min_confidence_map

    if thresholds:
        thresholds = check_min_confidence_map(thresholds)
    if min_confidence is not None:
        min_confidence = check_min_confidence(min_confidence)
    if len(temperature) != N_QTYPES:
        raise ValueError("temperature must have one scale per question type (%d), got %r"
                         % (N_QTYPES, temperature))
    recs = list(records)
    if not recs:
        raise ValueError("evaluate_records needs at least one record")
    rows = []
    for qt, logits, target, k in recs:
        if isinstance(qt, bool) or qt not in range(N_QTYPES):
            raise ValueError("record question type must be in range(%d), got %r"
                             % (N_QTYPES, qt))
        z = [float(v) for v in logits][:k]
        y = list(target)[:k]
        bucket = temp_bucket(qt, k)
        scale = temperature_by_options.get(bucket, temperature[qt])
        probs = _softmax_row([v / scale for v in z])
        conf = round(max(probs), 4)
        correct = int(probs.index(max(probs)) == y.index(max(y)))
        cut = _cut_for(thresholds, min_confidence, bucket)
        rows.append({"qtype": qt, "bucket": bucket,
                     "confidence": conf, "correct": bool(correct),
                     "abstained": bool(conf < cut)})
    n = len(rows)
    accepted = [r for r in rows if not r["abstained"]]
    summary = {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "ece": _ece([(r["confidence"], r["correct"]) for r in rows]),
        "coverage": len(accepted) / n,
        "gated_error": (sum(not r["correct"] for r in accepted) / len(accepted)
                        if accepted else None),
    }
    return rows, summary


def compare_reports(before, after):
    """After-minus-before deltas for the gate's three numbers (plus accuracy, informational)."""
    return {key: after[key] - before[key]
            for key in ("accuracy", "ece", "coverage", "gated_error")
            if after[key] is not None and before[key] is not None}


def should_ship(before, after, *, tol_ece=0.002, tol_error=0.0, tol_coverage=0.0):
    """Whether the candidate behind `after` may replace what `before` scored.

    Ships only when nothing regresses: ECE within `tol_ece`, gated error within
    `tol_error`, coverage within `tol_coverage`. A candidate whose gate accepts
    nothing has no gated error to compare, which fails closed -- a gate that answers
    no question is not an improvement. Returns `(ship, reasons)` where reasons names
    every check with its numbers, so a blocked rollout says exactly what blocked it.
    """
    reasons = []
    ok = True

    def check(name, good, detail):
        nonlocal ok
        reasons.append("%s %s: %s" % (name, "ok" if good else "BLOCKED", detail))
        ok = ok and good

    check("ece", after["ece"] <= before["ece"] + tol_ece,
          "%.4f -> %.4f (tol %.4f)" % (before["ece"], after["ece"], tol_ece))
    for key, tol in (("gated_error", tol_error), ("coverage", tol_coverage)):
        b, a = before[key], after[key]
        if b is None or a is None:
            check(key, False, "undefined (empty accepted set)")
        elif key == "coverage":
            check(key, a >= b - tol, "%.3f -> %.3f (tol %.3f)" % (b, a, tol))
        else:
            check(key, a <= b + tol, "%.4f -> %.4f (tol %.4f)" % (b, a, tol))
    return ok, reasons
