"""Recompute ECE under both definitions of `confidence`, from results already in the repo.

  USE_TF=0 python3 research/scripts/confidence_definitions.py
  USE_TF=0 python3 research/scripts/confidence_definitions.py --write

`confidence` carries two definitions. `choice` and `score` answers report normalized entropy,
`1 - H(p) / log(k)`; `noul` answers report `max(p_true, 1 - p_true)`, which over two options is
`max(p)`. Both benchmark harnesses in this directory take `conf = max(probs)` before calling
`ece_score`, so every published ECE figure describes `max(p)` -- not the number a `choice` or
`score` answer actually returns.

This script measures the gap on shipped output rather than on a synthetic distribution. It
reads the per-question probabilities recorded by the Chinese workflow benchmark and reports,
for each definition, the accuracy-weighted calibration error and the reliability bins behind
it. Nothing is re-run: the probabilities are read from the checked-in JSONL.

ECE alone does not settle it, and the output says so. ECE is sensitive to the scale of the
number, not only to how well it orders answers: a confidence that is systematically compressed
can land near a low accuracy and score a low ECE without tracking correctness at all. So the
report also carries two scale-free readings -- AUROC of confidence against correctness, and the
accuracy you get by keeping only the most confident half -- plus what actually happens at the
0.85 gate the README's "Automated Confidence Gating" section uses.

It also checks the two invariants the comparison rests on, and fails loudly if either breaks:
the shipped `choice` confidence really is `confidence_from_probs`, and the shipped `noul`
confidence really is `max(p)` -- so `answer_confidence` is a no-op on every `noul` answer here.

Caveat worth stating up front: the only per-question probabilities checked into this repository
are this benchmark's, and Laya scores near chance on it (4 options, accuracy around 0.31). Every
calibration number below inherits that. The comparison between the two definitions is still
meaningful -- both are computed on the same predictions -- but neither is a calibration figure
for the checkpoint in general.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from laya.common import answer_confidence, confidence_from_probs, ece_score  # noqa: E402

RESULTS = os.path.join(REPO, "research", "benchmarks", "feishu_zh", "results", "v1")
OUT = os.path.join(REPO, "research", "results", "confidence_definitions.json")
BINS = 15
# The stored probabilities are rounded to four decimals, so entropy recomputed from them lands
# a little away from the value the run reported off full-precision probabilities.
ROUNDING_SLACK = 2e-3


def load(model):
    path = os.path.join(RESULTS, model, "raw.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_laya(rows):
    """The invariants below describe Laya's own answer shape, so they are only checked there."""
    return any(str((r.get("response") or {}).get("model", "")).startswith("laya") for r in rows)


def choice_rows(rows):
    """(probability vector, gold index, shipped confidence) for every answered choice row."""
    out = []
    for r in rows:
        if r.get("mode") != "choice" or r.get("status") != "ok" or not r.get("probabilities"):
            continue
        labels = list(r["probabilities"])
        if r["expected"] not in labels:
            continue
        p = np.array([float(r["probabilities"][k]) for k in labels], dtype=float)
        out.append((p, labels.index(r["expected"]), r.get("confidence")))
    return out


def noul_answers(rows):
    """Every individual noul answer across the four-question rows."""
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        for ans in (r.get("response") or {}).get("answers", {}).values():
            if ans.get("type") == "noul" and "noul" in ans and "confidence" in ans:
                out.append((float(ans["noul"]), float(ans["confidence"])))
    return out


def auroc(conf, correct):
    """Probability that a correct answer outranks an incorrect one. Scale-free, so it isolates
    ordering from the units the confidence happens to be reported in."""
    pos, neg = conf[correct > 0.5], conf[correct <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    values = np.concatenate([pos, neg])
    for v in np.unique(values):                      # average ranks over ties
        tie = values == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def acc_at_coverage(conf, correct, coverage):
    """Accuracy over the most confident `coverage` share -- what selective gating actually buys."""
    k = max(1, int(round(len(conf) * coverage)))
    return float(correct[np.argsort(-conf, kind="mergesort")[:k]].mean())


def gate(conf, correct, threshold=0.85):
    """What the README's gating example does at this threshold: how much is automated, and how
    much of that is right."""
    sel = conf >= threshold
    return {"threshold": threshold, "automated": int(sel.sum()),
            "automated_share": round(float(sel.mean()), 4),
            "accuracy_of_automated": round(float(correct[sel].mean()), 4) if sel.any() else None}


def reliability(conf, correct, bins=BINS):
    """The per-bin table ECE sums over, so the single number can be read rather than trusted."""
    edges = np.linspace(0, 1, bins + 1)
    table = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            table.append({"bin": "%.2f-%.2f" % (lo, hi), "n": int(sel.sum()),
                          "mean_confidence": round(float(conf[sel].mean()), 4),
                          "accuracy": round(float(correct[sel].mean()), 4),
                          "gap": round(float(conf[sel].mean() - correct[sel].mean()), 4)})
    return table


def measure(rows):
    probs = [p for p, _, _ in rows]
    gold = np.array([g for _, g, _ in rows])
    pred = np.array([int(np.argmax(p)) for p in probs])
    correct = (pred == gold).astype(float)

    top = np.array([answer_confidence(p, len(p)) for p in probs])
    ent = np.array([confidence_from_probs(p, len(p)) for p in probs])

    def block(conf):
        return {"mean_confidence": round(float(conf.mean()), 4),
                "ece": round(ece_score(conf, correct), 4),
                "auroc": round(auroc(conf, correct), 4),
                "acc_at_50_coverage": round(acc_at_coverage(conf, correct, 0.5), 4),
                "gate_at_0.85": gate(conf, correct),
                "reliability": reliability(conf, correct)}

    return {"n": len(rows), "accuracy": round(float(correct.mean()), 4),
            "answer_confidence": block(top), "entropy_confidence": block(ent)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="also write %s" % os.path.relpath(OUT, REPO))
    ap.add_argument("--models", nargs="+", default=["laya", "jev"])
    args = ap.parse_args()

    report = {"source": os.path.relpath(RESULTS, REPO), "bins": BINS, "models": {}}
    problems = []

    for model in args.models:
        rows = load(model)
        ch = choice_rows(rows)
        if not ch:
            problems.append("%s: no answered choice rows" % model)
            continue
        report["models"][model] = m = measure(ch)

        if not is_laya(rows):
            # Another engine's `confidence` is its own business; only the comparison table above
            # is meaningful for it.
            m["invariants_checked"] = False
            continue
        m["invariants_checked"] = True

        # Invariant 1: the shipped choice confidence is the entropy definition.
        drift = max((abs(c - confidence_from_probs(p, len(p))) for p, _, c in ch if c is not None),
                    default=None)
        m["shipped_choice_confidence_is_entropy"] = drift is not None and drift <= ROUNDING_SLACK
        m["choice_confidence_max_drift"] = None if drift is None else round(drift, 6)
        if drift is None or drift > ROUNDING_SLACK:
            problems.append("%s: shipped choice confidence is not confidence_from_probs "
                            "(max drift %s)" % (model, drift))

        # Invariant 2: the shipped noul confidence is max(p), so `answer_confidence` adds nothing
        # there. That is the whole reason this change can be additive.
        nouls = noul_answers(rows)
        noul_drift = max((abs(c - answer_confidence(np.array([1.0 - t, t]), 2)) for t, c in nouls),
                         default=None)
        m["noul_answers"] = len(nouls)
        m["shipped_noul_confidence_is_max_p"] = noul_drift is not None and noul_drift <= ROUNDING_SLACK
        m["noul_confidence_max_drift"] = None if noul_drift is None else round(noul_drift, 6)
        if noul_drift is None or noul_drift > ROUNDING_SLACK:
            problems.append("%s: shipped noul confidence is not max(p) (max drift %s)"
                            % (model, noul_drift))

    print("Source : %s" % report["source"])
    print("Bins   : %d\n" % BINS)
    for model, m in report["models"].items():
        print("%s   n=%d   accuracy=%.4f" % (model, m["n"], m["accuracy"]))
        print("   %-28s %8s %8s %8s %8s" % ("confidence definition", "mean", "ECE", "AUROC", "acc@50%"))
        for key, label in (("answer_confidence", "max(p)"),
                           ("entropy_confidence", "1 - H/log(k)")):
            b = m[key]
            print("   %-28s %8.4f %8.4f %8.4f %8.4f"
                  % (label, b["mean_confidence"], b["ece"], b["auroc"], b["acc_at_50_coverage"]))
        print("   gate at 0.85 (the README's example):")
        for key, label in (("answer_confidence", "max(p)"),
                           ("entropy_confidence", "1 - H/log(k)")):
            g = m[key]["gate_at_0.85"]
            acc = "n/a" if g["accuracy_of_automated"] is None else "%.4f" % g["accuracy_of_automated"]
            print("      %-28s automated %3d/%d (%.1f%%), of which correct: %s"
                  % (label, g["automated"], m["n"], 100 * g["automated_share"], acc))
        print()

    for model, m in report["models"].items():
        print("\n%s: reliability bins, confidence = max(p)" % model)
        for b in m["answer_confidence"]["reliability"]:
            print("   %-12s n=%-4d confidence=%.4f accuracy=%.4f gap=%+.4f"
                  % (b["bin"], b["n"], b["mean_confidence"], b["accuracy"], b["gap"]))
        print("%s: reliability bins, confidence = 1 - H/log(k)" % model)
        for b in m["entropy_confidence"]["reliability"]:
            print("   %-12s n=%-4d confidence=%.4f accuracy=%.4f gap=%+.4f"
                  % (b["bin"], b["n"], b["mean_confidence"], b["accuracy"], b["gap"]))

    print("Invariants (Laya answers only; another engine's `confidence` is its own definition):")
    for model, m in report["models"].items():
        if not m.get("invariants_checked"):
            print("   %-6s skipped, not a Laya response shape" % model)
            continue
        print("   %-6s shipped choice confidence is 1 - H/log(k)   : %-5s (max drift %.2g over %d)"
              % (model, m["shipped_choice_confidence_is_entropy"],
                 m["choice_confidence_max_drift"], m["n"]))
        print("   %-6s shipped noul confidence is max(p)           : %-5s (max drift %.2g over %d)"
              % (model, m["shipped_noul_confidence_is_max_p"],
                 m["noul_confidence_max_drift"], m["noul_answers"]))
        print("   %-6s -> `answer_confidence` changes nothing on a noul answer." % model)

    if args.write:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        print("\nwrote %s" % os.path.relpath(OUT, REPO))

    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  " + p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
