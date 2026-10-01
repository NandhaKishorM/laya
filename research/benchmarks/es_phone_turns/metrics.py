"""Metrics recomputed from per-case records alone. Standard library only."""
from collections import Counter, defaultdict

# Everything the agent can DO. "ninguno" is the only answer that does nothing.
KEEP_TALKING = "ninguno"


def accuracy(records):
    return sum(r["predicted"] == r["gold"] for r in records) / len(records) if records else None


def macro_f1(records, labels):
    scores = []
    for label in labels:
        tp = sum(r["predicted"] == label and r["gold"] == label for r in records)
        fp = sum(r["predicted"] == label and r["gold"] != label for r in records)
        fn = sum(r["predicted"] != label and r["gold"] == label for r in records)
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else None


def ece(records, bins=10):
    """Expected calibration error over the confidence of the reported answer."""
    if not records:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [r for r in records
                  if (lo < r["confidence"] <= hi) or (b == 0 and r["confidence"] == 0.0)]
        if not inside:
            continue
        conf = sum(r["confidence"] for r in inside) / len(inside)
        acc = sum(r["predicted"] == r["gold"] for r in inside) / len(inside)
        total += len(inside) / len(records) * abs(conf - acc)
    return total


def by_tag(records):
    groups = defaultdict(list)
    for r in records:
        for tag in r["tags"]:
            groups[tag].append(r)
    return {tag: {"n": len(rs), "accuracy": accuracy(rs)} for tag, rs in sorted(groups.items())}


def confusion(records):
    return dict(Counter((r["gold"], r["predicted"]) for r in records if r["gold"] != r["predicted"]))


def wrong_actions(records):
    """The mistakes a phone agent cannot afford: it ACTED, and should not have, or acted
    on the wrong destination. Answering 'ninguno' by mistake only costs a slower turn."""
    return [r for r in records
            if r["predicted"] != KEEP_TALKING and r["predicted"] != r["gold"]]


def fast_path(records, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """What a confidence gate buys. A turn takes the fast path when the model names an
    ACTION with at least this confidence; every other turn goes to the language model.

    coverage        share of all turns that skip the language model
    action_recall   share of the turns that truly were actions and were taken fast
    wrong           fast turns that acted wrongly (the expensive mistake)
    precision       share of fast turns that acted correctly
    """
    actions = [r for r in records if r["gold"] != KEEP_TALKING]
    table = []
    for t in thresholds:
        fast = [r for r in records
                if r["predicted"] != KEEP_TALKING and r["confidence"] >= t]
        right = [r for r in fast if r["predicted"] == r["gold"]]
        table.append({
            "threshold": t,
            "fast": len(fast),
            "coverage": len(fast) / len(records) if records else None,
            "action_recall": len(right) / len(actions) if actions else None,
            "wrong": len(fast) - len(right),
            "precision": len(right) / len(fast) if fast else None,
        })
    return table


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


def summarize(records, labels):
    headline = [r for r in records if "borderline" not in r["tags"]]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    majority = Counter(r["gold"] for r in headline).most_common(1)[0][1] / len(headline)
    return {
        "n": len(headline),
        "accuracy": accuracy(headline),
        "macro_f1": macro_f1(headline, labels),
        "ece": ece(headline),
        "mean_confidence": sum(r["confidence"] for r in headline) / len(headline),
        "majority_baseline": majority,
        "wrong_actions": len(wrong_actions(headline)),
        "by_tag": by_tag(headline),
        "confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion(headline).items())},
        "fast_path": fast_path(headline),
        "latency_ms": {
            "p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
            "min": min(latencies) if latencies else None,
        },
        "borderline": [
            {"id": r["id"], "text": r["text"], "gold": r["gold"],
             "predicted": r["predicted"], "confidence": round(r["confidence"], 4)}
            for r in records if "borderline" in r["tags"]],
    }
