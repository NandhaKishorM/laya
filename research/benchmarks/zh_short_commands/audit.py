"""Re-derive every published number from an archive, offline.

    python research/benchmarks/zh_short_commands/audit.py
    python research/benchmarks/zh_short_commands/audit.py --run-dir DIR

No model, no network and no third-party package: the frozen cases, the frozen prompts
and the per-case records are enough to recompute each ``report`` block. What is checked:

* the three hashes (cases, prompts, and the ones the archive recorded) still agree, so
  a record cannot be swapped for one scored under a different prompt;
* every record names a real case, and its text/state/instructions/gold are re-built
  from ``prompts.py`` rather than trusted;
* every config is present with the right multiplicity and no duplicates;
* probabilities are finite and in [0, 1], and the prediction follows from them;
* ``report`` is recomputed from ``cases`` alone and must match what was archived.

The metrics below are re-implemented without numpy on purpose. ``tests/test_audit.py``
pins them against ``research/eval/laya_eval.py``, so this file cannot drift from the
harness that produced the numbers.
"""
import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import prompts  # noqa: E402

from research.eval.laya_eval import ECE_BINS  # noqa: E402

DEFAULT_RUN = os.path.join(HERE, "results", "v1", "laya-multilingual")
PLACES = 4          # the archive's own rounding for accuracy / macro_f1 / ece


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_frozen():
    """The two frozen inputs plus the manifest that pins them."""
    cases = []
    with open(os.path.join(HERE, "data", "cases.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    manifest = load_json(os.path.join(HERE, "data", "manifest.json"))
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case id in data/cases.jsonl")
    if any(c["gold"] not in prompts.LABELS for c in cases):
        raise ValueError("a gold label is outside LABELS")
    for key, path in (("cases_sha256", "data/cases.jsonl"), ("prompts_sha256", "prompts.py")):
        if sha256_file(os.path.join(HERE, path)) != manifest[key]:
            raise ValueError(
                "%s no longer matches the manifest: %s changed after the archive was "
                "produced, so the archived numbers cannot be re-derived from it" % (key, path))
    if manifest["cases"] != len(cases):
        raise ValueError("manifest count %r != %d cases" % (manifest["cases"], len(cases)))
    return cases, manifest


# ------------------------------------------------------------------ metrics (no numpy)
def macro_f1(gold, pred):
    """Unweighted mean F1 over the classes gold and pred use, as laya_eval computes it."""
    scores = []
    for cls in sorted(set(gold) | set(pred)):
        tp = sum(1 for g, p in zip(gold, pred) if p == cls and g == cls)
        fp = sum(1 for g, p in zip(gold, pred) if p == cls and g != cls)
        fn = sum(1 for g, p in zip(gold, pred) if p != cls and g == cls)
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else float("nan")


def ece(confidence, correct, bins=ECE_BINS):
    """Equal-width bins; the first is closed at the bottom, matching laya_eval."""
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [j for j, c in enumerate(confidence)
               if (c >= lo if i == 0 else c > lo) and c <= hi]
        if sel:
            total += (len(sel) / len(confidence)) * abs(
                sum(confidence[j] for j in sel) / len(sel)
                - sum(correct[j] for j in sel) / len(sel))
    return total


def mean(values):
    return sum(values) / len(values)


def std(values):
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def close(got, want, places):
    """Equal once both are rounded to the archive's precision."""
    return abs(got - want) <= 0.5 * 10 ** -places


def check(name, got, want, places=PLACES):
    if not close(got, want, places):
        raise ValueError("%s: re-derived %.8f, archived %s" % (name, got, want))


# ---------------------------------------------------------------------------- records
def audit_records(doc, cases, manifest):
    """Every record must describe a real request, and no request may be missing."""
    by_id = {c["id"]: c for c in cases}
    configs = doc["config"]["configs"]
    if configs != list(manifest["ladder"]["choice"]) + list(manifest["ladder"]["noul"]):
        raise ValueError("the archive scored a different ladder than the manifest freezes")
    expected = {}
    for config in configs:
        if config in prompts.CHOICE_CONFIGS:
            expected[config] = {c["id"]: {None} for c in cases}
        else:
            expected[config] = {c["id"]: set(prompts.NOUL_DIMENSIONS) for c in cases}
    seen = {}
    temperatures = {}
    for record in doc["cases"]:
        config = record["config"]
        if config not in expected:
            raise ValueError("record for a config the archive did not score: %r" % config)
        case_id, dim = record["case_id"], record.get("dimension")
        if case_id not in by_id:
            raise ValueError("record for an unknown case: %r" % case_id)
        if dim in seen.setdefault(config, {}).get(case_id, set()):
            raise ValueError("duplicate record: %s / %s / %s" % (config, case_id, dim))
        seen.setdefault(config, {}).setdefault(case_id, set()).add(dim)
        case = by_id[case_id]
        where = "%s / %s / %s" % (config, case_id, dim)
        if record["text"] != case["text"] or record["gold_label"] != case["gold"]:
            raise ValueError("%s: the frozen text or gold label was altered" % where)
        if record["state"] != prompts.state_for(config, case["text"]):
            raise ValueError("%s: state is not what this rung sends" % where)
        if record["family"] != case["family"]:
            raise ValueError("%s: family was altered" % where)
        temperatures.setdefault(config, set()).add(record["temperature"])
        if dim is None:
            expected_instructions = prompts.choice_question(config)["intent"]["instructions"]
            if record["instructions"] != expected_instructions:
                raise ValueError("%s: instructions are not what this rung sends" % where)
            if record["options"] != list(prompts.CHOICE_CRITERIA):
                raise ValueError("%s: options are not the label policy" % where)
            for key in ("probability", "p_gold", "confidence"):
                if not 0.0 <= record[key] <= 1.0 or not math.isfinite(record[key]):
                    raise ValueError("%s: %s is not a probability" % (where, key))
            if record["pred_label"] != prompts.LABELS[record["pred_index"]]:
                raise ValueError("%s: pred_label/pred_index disagree" % where)
            if record["correct"] != int(record["pred_index"] == record["gold_index"]):
                raise ValueError("%s: correct contradicts the prediction" % where)
            if record["probability"] != record["confidence"]:
                raise ValueError("%s: the recorded probability is not the argmax one" % where)
            if record["pred_label"] != case["gold"] and record["correct"]:
                raise ValueError("%s: scored correct but the label differs" % where)
        else:
            if dim not in prompts.NOUL_DIMENSIONS:
                raise ValueError("%s: not a frozen dimension" % where)
            expected_instructions = prompts.noul_questions(config)[dim]["instructions"]
            if record["instructions"] != expected_instructions:
                raise ValueError("%s: instructions are not what this rung sends" % where)
            if record["options"] != ["false", "true"]:
                raise ValueError("%s: noul options must be false/true" % where)
            if not 0.0 <= record["p_true"] <= 1.0 or not math.isfinite(record["p_true"]):
                raise ValueError("%s: p_true is not a probability" % where)
            if record["pred_bool"] != (record["p_true"] >= 0.5):
                raise ValueError("%s: pred_bool does not follow from p_true" % where)
            if record["gold_bool"] != bool(prompts.NOUL_GOLD[dim](case["gold"])):
                raise ValueError("%s: gold_bool does not follow from the gold label" % where)
            if record["correct"] != int(record["pred_bool"] == record["gold_bool"]):
                raise ValueError("%s: correct contradicts the prediction" % where)
            if record["confidence"] != round(max(record["p_true"], 1.0 - record["p_true"]), 6):
                raise ValueError("%s: confidence is not the two-way max" % where)
    missing = [(config, case_id, dim)
               for config, per_case in expected.items()
               for case_id, dims in per_case.items()
               for dim in dims
               if dim not in seen.get(config, {}).get(case_id, set())]
    if missing:
        raise ValueError("incomplete archive: %d missing records, first %r" % (len(missing), missing[0]))
    for config, values in temperatures.items():
        if len(values) != 1 or doc["report"][config]["temperature"] not in values:
            raise ValueError("%s: the report's temperature is not the one the records used" % config)
    return configs


def audit_report(doc):
    """Recompute every report block from the per-case records."""
    for config, report in doc["report"].items():
        records = [r for r in doc["cases"] if r["config"] == config]
        confidences = [r["confidence"] for r in records]
        corrects = [r["correct"] for r in records]
        check("%s/accuracy" % config, mean(corrects), report["accuracy"])
        check("%s/mean_confidence" % config, mean(confidences), report["mean_confidence"], 6)
        if config in prompts.NOUL_CONFIGS:
            check("%s/n"% config, len(records), report["n_cases"] * report["n_dimensions"])
            check("%s/n_cases" % config, len(records) / len(prompts.NOUL_DIMENSIONS), report["n_cases"])
            check("%s/n_dimensions" % config, len(prompts.NOUL_DIMENSIONS), report["n_dimensions"])
            check("%s/std_confidence" % config, std(confidences), report["std_confidence"], 6)
            for dim, value in report["accuracy_by_dimension"].items():
                subset = [r["correct"] for r in records if r["dimension"] == dim]
                if not subset:
                    raise ValueError("%s: %s has no records" % (config, dim))
                check("%s/%s" % (config, dim), mean(subset), value, 6)
            if sorted(report["accuracy_by_dimension"]) != sorted(prompts.NOUL_DIMENSIONS):
                raise ValueError("%s: report dims are not the frozen ones" % config)
        else:
            check("%s/n" % config, len(records), report["n"])
            golds = [r["gold_index"] for r in records]
            preds = [r["pred_index"] for r in records]
            check("%s/macro_f1" % config, macro_f1(golds, preds), report["macro_f1"])
            check("%s/ece" % config, ece(confidences, corrects), report["ece"])
            order = sorted(range(len(records)), key=lambda i: (-confidences[i], i))
            top_half = [corrects[i] for i in order[:max(1, len(records) // 2)]]
            check("%s/acc_at_50_coverage" % config, mean(top_half),
                  report["acc_at_50_coverage"])
        if set(report) - {"n", "n_cases", "n_dimensions", "accuracy", "macro_f1", "ece",
                          "mean_confidence", "std_confidence", "acc_at_50_coverage",
                          "temperature", "accuracy_by_dimension"}:
            raise ValueError("%s: the report carries an unexplained field" % config)


def audit_provenance(doc):
    """Say out loud what was scored. A real run must pin a weight, not just a repo id."""
    checkpoint = doc["config"]["checkpoint"]
    for key in ("checkpoint", "subfolder", "device"):
        if not checkpoint.get(key):
            raise ValueError("provenance is missing %r" % key)
    if checkpoint.get("stub"):
        return "stub run: no checkpoint was loaded"
    for key in ("weight_sha256", "weight_bytes"):
        if not checkpoint.get(key):
            raise ValueError(
                "no %s: the archive does not say which weights produced these numbers, "
                "and the reported hash is the only way to tell two checkpoints apart" % key)
    return "%s/%s %s bytes sha256 %s" % (
        checkpoint["checkpoint"], checkpoint["subfolder"], checkpoint["weight_bytes"],
        checkpoint["weight_sha256"][:16] + "...")


def audit_run(folder=DEFAULT_RUN):
    cases, manifest = load_frozen()
    path = os.path.join(folder, "report.json")
    if not os.path.exists(path):
        raise ValueError("no report.json under %s" % folder)
    doc = load_json(path)
    if sorted(doc) != ["cases", "config", "report", "summary"]:
        raise ValueError("unexpected document parts: %r" % sorted(doc))
    for key in ("cases_sha256", "prompts_sha256"):
        if doc["config"][key] != manifest[key]:
            raise ValueError("the archive was scored under a different frozen protocol")
    if not isinstance(doc["config"]["unclamped"], bool):
        raise ValueError("`unclamped` must be recorded as a boolean")
    configs = audit_records(doc, cases, manifest)
    audit_report(doc)
    for config in configs:
        if doc["summary"][config]["accuracy"] != doc["report"][config]["accuracy"]:
            raise ValueError("summary/%s disagrees with its report block" % config)
    return {"folder": os.path.basename(os.path.normpath(folder)),
            "cases": len(cases), "configs": configs, "records": len(doc["cases"]),
            "provenance": audit_provenance(doc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=DEFAULT_RUN,
                        help="directory holding report.json (default: the committed archive)")
    args = parser.parse_args(argv)
    try:
        result = audit_run(args.run_dir)
    except ValueError as exc:
        print("FAIL %s" % exc)
        return 1
    print("ok   %s: %d cases, %d configs, %d records" % (
        result["folder"], result["cases"], len(result["configs"]), result["records"]))
    print("ok   re-derived from cases alone: accuracy, macro_f1, ece, mean/std confidence")
    print("ok   provenance: %s" % result["provenance"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
