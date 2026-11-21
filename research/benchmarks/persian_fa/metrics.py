"""Model-free scoring. Pure Python, no third-party imports.

A *row* is one recorded response (see ``run.py``)::

    {"case_id", "mode", "repeat", "prediction", "probabilities", "answer_confidence",
     "latency_ms", "error"}

Failed requests (``error`` set or ``prediction`` missing) stay in the denominator and count
as wrong. Quality is scored on one preselected repeat (default: the first), never the best.
"""
from collections import Counter, defaultdict
from statistics import median

from prompts import LABELS

ECE_BINS = 10


def _percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def expected_calibration_error(confidences, corrects, bins=ECE_BINS):
    """Standard ECE over the probability of the reported answer."""
    if not confidences:
        return None
    buckets = defaultdict(list)
    for c, ok in zip(confidences, corrects):
        idx = min(int(c * bins), bins - 1)
        buckets[idx].append((c, ok))
    n = len(confidences)
    ece = 0.0
    for items in buckets.values():
        avg_c = sum(c for c, _ in items) / len(items)
        acc = sum(1 for _, ok in items if ok) / len(items)
        ece += len(items) / n * abs(avg_c - acc)
    return ece


def score(cases, rows, mode, repeat=0):
    """Metrics for one prompt mode on one repeat. ``cases`` is the list from cases.jsonl."""
    gold = {c["id"]: c for c in cases}
    picked = {r["case_id"]: r for r in rows if r["mode"] == mode and r["repeat"] == repeat}

    n = len(gold)
    correct_ids, failures = [], 0
    confusion = {g: Counter() for g in LABELS}
    fam_tot, fam_ok = Counter(), Counter()
    lab_tot, lab_ok = Counter(), Counter()
    trap_tot, trap_ok = Counter(), Counter()
    confs, oks = [], []
    false_cancel = missed_cancel = 0

    for cid, case in gold.items():
        r = picked.get(cid)
        pred = None if (r is None or r.get("error")) else r.get("prediction")
        if pred is None:
            failures += 1
        ok = pred == case["label"]
        fam_tot[case["family"]] += 1
        lab_tot[case["label"]] += 1
        confusion[case["label"]][pred if pred is not None else "FAILED"] += 1
        if ok:
            correct_ids.append(cid)
            fam_ok[case["family"]] += 1
            lab_ok[case["label"]] += 1
        if case.get("trap"):
            trap_tot[case["trap"]] += 1
            trap_ok[case["trap"]] += ok
        if pred == "cancel" and case["label"] != "cancel":
            false_cancel += 1
        if case["label"] == "cancel" and pred != "cancel":
            missed_cancel += 1
        if pred is not None and r.get("answer_confidence") is not None:
            confs.append(float(r["answer_confidence"]))
            oks.append(ok)

    # macro F1 over the four labels
    f1s = []
    for lab in LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in LABELS if g != lab)
        fn = lab_tot[lab] - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    lat = [r["latency_ms"] for r in rows
           if r["mode"] == mode and not r.get("error") and r.get("latency_ms") is not None]

    return {
        "mode": mode,
        "repeat": repeat,
        "n": n,
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s),
        "failures": failures,
        "false_cancel": false_cancel,
        "missed_cancel": missed_cancel,
        "by_family": {f: {"correct": fam_ok[f], "n": fam_tot[f]} for f in fam_tot},
        "by_label": {lab: {"correct": lab_ok[lab], "n": lab_tot[lab]} for lab in LABELS},
        "by_trap": {t: {"correct": trap_ok[t], "n": trap_tot[t]} for t in sorted(trap_tot)},
        "confusion": {g: dict(confusion[g]) for g in LABELS},
        "mean_answer_confidence": sum(confs) / len(confs) if confs else None,
        "ece": expected_calibration_error(confs, oks),
        "latency_ms_p50": median(lat) if lat else None,
        "latency_ms_p90": _percentile(lat, 0.9),
    }


def repeat_agreement(rows, mode):
    """Fraction of cases whose prediction is identical across every recorded repeat."""
    by_case = defaultdict(set)
    for r in rows:
        if r["mode"] == mode:
            by_case[r["case_id"]].add(r.get("prediction") if not r.get("error") else "FAILED")
    if not by_case:
        return None
    return sum(1 for s in by_case.values() if len(s) == 1) / len(by_case)
