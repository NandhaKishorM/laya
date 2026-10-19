"""Paired choice-context evaluation through the public prediction API.

Run python -m research.eval.context_robustness --help. Saved-run analysis needs
no weights or torch. The dataset author supplies variants and expected labels;
this module cannot establish semantic equivalence.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

from laya.confidence import answer_confidence_value, apply_confidence_gate, check_min_confidence
from laya.evals import ChoiceAccuracy

CASE_SCHEMA = "laya-context-robustness-case/1"
RUN_SCHEMA = "laya-context-robustness-run/1"
REPORT_SCHEMA = "laya-context-robustness-report/1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key %r" % key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-finite JSON value %s" % value)


def parse(text):
    value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    encoded(value)  # Also rejects exponents that overflow, such as 1e999.
    return value


def _text(value, where):
    require(isinstance(value, str) and bool(value.strip()), where + " must be a nonempty string")


def _fields(value, required, optional, where):
    require(isinstance(value, dict), where + " must be an object")
    require(set(required) <= set(value), where + " is missing required fields")
    require(set(value) <= set(required) | set(optional), where + " has unknown fields")


def validate_cases(cases):
    require(isinstance(cases, list) and bool(cases), "cases must be a nonempty list")
    encoded(cases)
    seen = set()
    for index, case in enumerate(cases):
        where = "case %d" % (index + 1)
        _fields(case, ("schema", "id", "state", "questions", "expected", "gold_rationale", "variants"), (), where)
        _text(case["id"], where + ".id")
        require(case["id"] not in seen, where + ": duplicate case id")
        seen.add(case["id"])
        where += " (%s)" % case["id"]
        require(case["schema"] == CASE_SCHEMA, where + ": unknown schema")
        _text(case["gold_rationale"], where + ".gold_rationale")
        require(type(case["state"]) in (str, dict, list), where + ": unsupported state root type")
        questions = case["questions"]
        require(isinstance(questions, dict) and len(questions) == 1, where + ": exactly one question required")
        qid, q = next(iter(questions.items()))
        _text(qid, where + ".question id")
        _fields(q, ("type", "instructions", "criteria"), (), where + ".question")
        require(q["type"] == "choice", where + ": only choice is supported")
        _text(q["instructions"], where + ".instructions")
        criteria = q["criteria"]
        require(isinstance(criteria, dict) and len(criteria) >= 2, where + ": at least two ordered criteria required")
        for label, description in criteria.items():
            _text(label, where + ".label")
            require(description is None or isinstance(description, str), where + ": invalid option description")
        require(isinstance(case["expected"], str) and case["expected"] in criteria, where + ": unknown expected label")
        variants = case["variants"]
        require(isinstance(variants, list) and bool(variants), where + ": variants must be nonempty")
        by_id = {}
        for variant in variants:
            _fields(variant, ("id", "kind", "state", "rationale"), ("family", "control_id"), where + ".variant")
            vid = variant["id"]
            _text(vid, where + ".variant.id")
            require(vid != "clean" and vid not in by_id, where + ": reserved or duplicate variant id")
            by_id[vid] = variant
            require(variant["kind"] in ("benign", "adversarial"), where + ": unknown variant kind")
            require(type(variant["state"]) is type(case["state"]), where + ": state root type changed")
            _text(variant["rationale"], where + ".rationale")
            if "family" in variant:
                _text(variant["family"], where + ".family")
            if variant["kind"] == "benign":
                require("control_id" not in variant, where + ": benign variant cannot name a control")
        for variant in variants:
            if variant["kind"] == "adversarial":
                control = variant.get("control_id")
                require(isinstance(control, str) and control in by_id
                        and by_id[control]["kind"] == "benign", where + ": invalid benign control_id")
    return cases


def load_cases(path):
    cases = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = parse(line)
            validate_cases([case])
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError("%s:%d: %s" % (path, lineno, exc)) from exc
        cases.append(case)
    return validate_cases(cases)


def semantic_identity():
    import laya
    import laya.confidence
    import laya.evals
    paths = {"collector": Path(__file__), "runner": Path(__file__).with_name("_context_runner.py"),
             "confidence": Path(laya.confidence.__file__),
             "evals": Path(laya.evals.__file__)}
    return {"laya_version": laya.__version__, "source_sha256": {
        name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name, path in paths.items()}}


def validate_settings(settings):
    _fields(settings, ("max_len", "head_max_len", "min_confidence"), (), "settings")
    for key in ("max_len", "head_max_len"):
        require(type(settings[key]) is int and settings[key] > 0, key + " must be a positive integer")
    require(settings["head_max_len"] < settings["max_len"], "head_max_len must be smaller than max_len")
    validate_threshold(settings["min_confidence"])


def validate_threshold(threshold):
    require(threshold is None or (type(threshold) in (int, float) and math.isfinite(threshold)
                                 and 0 <= threshold <= 1), "invalid min_confidence")
    if threshold is not None:
        check_min_confidence(threshold)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def observation(case, result, threshold):
    """Validate public answers without reconstructing choices from rounded probabilities."""
    require(isinstance(result, dict), "predict must return an object")
    encoded(result)
    qid, q = next(iter(case["questions"].items()))
    answers = result.get("answers")
    require(isinstance(answers, dict) and set(answers) == {qid}, "unexpected answer question ids")
    answer = answers[qid]
    require(isinstance(answer, dict) and answer.get("type") == "choice", "expected a choice answer")
    require(isinstance(answer.get("choice"), str) and answer["choice"] in q["criteria"], "unknown returned choice")
    probs = answer.get("probabilities")
    require(isinstance(probs, dict) and set(probs) == set(q["criteria"]), "probability labels differ")
    require(all(_number(p) for p in probs.values()), "invalid probability")
    require(abs(sum(probs.values()) - 1) <= len(probs) * 0.00005 + 1e-12, "probabilities do not sum to one")
    conf = answer_confidence_value(answer)
    require(_number(conf), "missing or invalid answer_confidence")
    gate_keys = ("abstention", "abstention_threshold", "low_confidence")
    if threshold is None:
        require(not any(key in answer for key in gate_keys), "unexpected gate on an ungated run")
        accepted = None
    else:
        require(_number(answer.get("abstention_threshold")), "missing or invalid gate threshold")
        if "low_confidence" in answer:
            require(type(answer["low_confidence"]) is bool, "invalid low_confidence flag")
        replay = deepcopy(result)
        for key in gate_keys:
            replay["answers"][qid].pop(key, None)
        apply_confidence_gate([replay], threshold)
        require(all(answer.get(key) == replay["answers"][qid].get(key) for key in gate_keys),
                "recorded gate differs from native gate")
        accepted = answer["abstention"] == "passed"
    usage = result.get("usage")
    truncated = None
    if usage is not None:
        require(isinstance(usage, dict), "usage must be an object")
        if "truncated" in usage:
            require(type(usage["truncated"]) is bool, "invalid truncation flag")
            truncated = usage["truncated"]
        for key in ("state_tokens", "state_tokens_dropped"):
            if key in usage:
                require(type(usage[key]) is int and usage[key] >= 0, "invalid " + key)
        if "state_tokens_dropped" in usage and truncated is not None:
            require(truncated == (usage["state_tokens_dropped"] > 0), "contradictory truncation usage")
        if "state_tokens" in usage and "state_tokens_dropped" in usage:
            require(usage["state_tokens_dropped"] <= usage["state_tokens"], "dropped more state tokens than present")
    return {"choice": answer["choice"], "correct": bool(ChoiceAccuracy().score(answer, case["expected"])),
            "accepted": accepted, "truncated": truncated, "usage": usage}


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _stratum(observations):
    values = [o["truncated"] for o in observations]
    return "truncated" if True in values else "unknown" if None in values else "untruncated"


def _policy(items):
    accepted = sum(x["accepted"] for x in items)
    wrong = sum(x["accepted"] and not x["correct"] for x in items)
    return {"accepted_coverage": ratio(accepted, len(items)), "wrong_and_accepted": ratio(wrong, len(items)),
            "error_given_accepted": ratio(wrong, accepted)}


def _pairs(pairs, gated):
    table = {k: 0 for k in ("CC", "CW", "WC", "WW")}
    for _, clean, variant in pairs:
        table[("C" if clean["correct"] else "W") + ("C" if variant["correct"] else "W")] += 1
    n, nc = len(pairs), table["CC"] + table["CW"]
    by_case = defaultdict(list)
    for cid, clean, variant in pairs:
        by_case[cid].append((clean, variant))
    correct_cases = [items for items in by_case.values() if items[0][0]["correct"]]
    result = {"n_pairs": n, "n_unique_cases": len(by_case), "transitions": table,
              "variants_per_case": {cid: len(items) for cid, items in by_case.items()},
              "answer_flip_rate": ratio(sum(c["choice"] != v["choice"] for _, c, v in pairs), n),
              "clean_accuracy_pair_weighted": ratio(nc, n),
              "variant_accuracy": ratio(table["CC"] + table["WC"], n),
              "accuracy_delta": ratio(table["WC"] - table["CW"], n),
              "new_error_given_clean_correct": ratio(table["CW"], nc),
              "case_any_new_error": ratio(sum(any(not v["correct"] for _, v in items)
                                               for items in correct_cases), len(correct_cases))}
    if gated:
        newly_accepted = sum(c["correct"] and not v["correct"] and v["accepted"] for _, c, v in pairs)
        retained = sum(c["correct"] and c["accepted"] and not v["correct"] and v["accepted"] for _, c, v in pairs)
        result["policy"] = {"clean": _policy([c for _, c, _ in pairs]),
                            "variant": _policy([v for _, _, v in pairs]),
                            "new_wrong_accepted_given_clean_correct": ratio(newly_accepted, nc),
                            "gate_miss_on_new_errors": ratio(newly_accepted, table["CW"]),
                            "retained_acceptance_new_error": ratio(retained, sum(c["correct"] and c["accepted"]
                                                                                for _, c, _ in pairs))}
    return result


def _matched(triples, gated):
    eligible = [row for row in triples if row[2]["correct"]]
    table = {k: 0 for k in ("CC", "CW", "WC", "WW")}
    for _, _, _, benign, attack in eligible:
        table[("C" if benign["correct"] else "W") + ("C" if attack["correct"] else "W")] += 1
    controls = Counter((cid, control) for cid, control, _, _, _ in triples)
    result = {"n_triplets": len(triples), "n_unique_cases": len({row[0] for row in triples}),
              "n_unique_controls": len(controls), "n_clean_correct_triplets": len(eligible),
              "control_reuse": [{"case_id": cid, "control_id": control, "n_adversarial": count}
                                for (cid, control), count in controls.items()],
              "benign_adversarial_transitions_given_clean_correct": table,
              "matched_error_delta": ratio(table["CW"] - table["WC"], len(eligible))}
    if gated:
        bwrong = sum(not b["correct"] and b["accepted"] for _, _, _, b, _ in eligible)
        awrong = sum(not a["correct"] and a["accepted"] for _, _, _, _, a in eligible)
        result["policy"] = {"benign_wrong_accepted": ratio(bwrong, len(eligible)),
                            "adversarial_wrong_accepted": ratio(awrong, len(eligible)),
                            "matched_wrong_accepted_delta": ratio(awrong - bwrong, len(eligible))}
    return result


def summarize(cases, results, threshold=None):
    """Results keyed by (case_id, variant_id); clean is a reserved variant id."""
    validate_cases(cases)
    validate_threshold(threshold)
    expected = {(c["id"], v["id"]) for c in cases for v in [{"id": "clean"}] + c["variants"]}
    require(set(results) == expected, "missing or extra observation ids")
    groups = {}
    for case in cases:
        qhash = digest(case["questions"])
        group = groups.setdefault(qhash, {"clean": [], "pairs": defaultdict(list), "triples": []})
        observed = {v["id"]: observation(case, results[(case["id"], v["id"])], threshold)
                    for v in [{"id": "clean"}] + case["variants"]}
        clean = observed["clean"]
        group["clean"].append(clean)
        for variant in case["variants"]:
            value = observed[variant["id"]]
            group["pairs"][variant["kind"]].append((case["id"], clean, value))
            if variant["kind"] == "adversarial":
                group["triples"].append((case["id"], variant["control_id"], clean,
                                         observed[variant["control_id"]], value))
    report = {"schema": REPORT_SCHEMA, "policy": {"configured": threshold is not None, "threshold": threshold},
              "groups": {}}
    for qhash, group in groups.items():
        out = {"n_clean_cases": len(group["clean"]),
               "unique_clean_accuracy": ratio(sum(o["correct"] for o in group["clean"]), len(group["clean"])),
               "by_kind": {}, "matched_controls": _matched(group["triples"], threshold is not None)}
        for kind, pairs in group["pairs"].items():
            out["by_kind"][kind] = _pairs(pairs, threshold is not None)
            out["by_kind"][kind]["truncation"] = {
                name: _pairs([p for p in pairs if _stratum(p[1:]) == name], threshold is not None)
                for name in ("untruncated", "truncated", "unknown")}
        out["matched_controls"]["truncation"] = {
            name: _matched([t for t in group["triples"] if _stratum(t[2:]) == name], threshold is not None)
            for name in ("untruncated", "truncated", "unknown")}
        report["groups"][qhash] = out
    return report


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value) + b"\n")
    temporary.replace(path)


def collect(cases, runner_factory, out, settings, dataset_sha256):
    """Own a new output directory; leave failed/incomplete evidence on interruption."""
    validate_cases(cases)
    validate_settings(settings)
    cases, settings = deepcopy(cases), deepcopy(settings)
    identity = {"cases": cases, "settings": settings, "dataset_bytes_sha256": dataset_sha256,
                "input_sha256": digest(cases), "semantics": semantic_identity()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": RUN_SCHEMA, "status": "running", "identity": identity, "completed": 0}
    _write(out / "manifest.json", manifest)
    try:
        with runner_factory() as runner:
            require(isinstance(runner.provenance, dict) and bool(runner.provenance), "missing runner provenance")
            identity["provenance"] = deepcopy(runner.provenance)
            run_id = digest(identity)
            manifest["run_id"] = run_id
            _write(out / "manifest.json", manifest)
            with (out / "observations.jsonl").open("xb") as handle:
                for case in cases:
                    for variant in [{"id": "clean", "state": case["state"]}] + case["variants"]:
                        manifest["current"] = [case["id"], variant["id"]]
                        result = runner.predict(deepcopy(variant["state"]), deepcopy(case["questions"]), **settings)
                        # Preserve a malformed JSON-compatible response before refusing it.
                        row = {"run_id": run_id, "case_id": case["id"], "variant_id": variant["id"],
                               "index": manifest["completed"], "response": result}
                        handle.write(encoded(row) + b"\n")
                        handle.flush()
                        observation(case, result, settings["min_confidence"])
                        manifest["completed"] += 1
        require(semantic_identity() == identity["semantics"], "semantic source changed during collection")
        manifest.pop("current", None)
        manifest.update(status="complete", observations_sha256=file_digest(out / "observations.jsonl"))
        _write(out / "manifest.json", manifest)
        report = summarize_saved(out)
        _write(out / "report.json", report)
        return report
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        _write(out / "manifest.json", manifest)
        raise


def summarize_saved(out):
    out = Path(out)
    manifest = parse((out / "manifest.json").read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and manifest.get("schema") == RUN_SCHEMA, "unknown run schema")
    require(manifest.get("status") == "complete", "run is not complete")
    identity = manifest["identity"]
    require(isinstance(identity, dict) and identity.get("semantics") == semantic_identity(),
            "incompatible analysis environment; use the recorded semantic sources")
    require(digest(identity) == manifest.get("run_id"), "run identity changed")
    cases, settings = validate_cases(identity["cases"]), identity["settings"]
    validate_settings(settings)
    require(digest(cases) == identity.get("input_sha256"), "input identity changed")
    path = out / "observations.jsonl"
    require(file_digest(path) == manifest.get("observations_sha256"), "observations file changed")
    planned = [(c["id"], v["id"]) for c in cases for v in [{"id": "clean"}] + c["variants"]]
    results = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = parse(line)
            _fields(row, ("run_id", "case_id", "variant_id", "index", "response"), (), "observation")
            key = (row["case_id"], row["variant_id"])
            require(index < len(planned) and key == planned[index] and type(row["index"]) is int
                    and row["index"] == index, "observation order/ids differ from plan")
            require(row["run_id"] == manifest["run_id"], "observation belongs to another run")
            results[key] = row["response"]
    require(len(results) == len(planned) and type(manifest["completed"]) is int
            and manifest["completed"] == len(planned), "missing observations")
    report = summarize(cases, results, settings["min_confidence"])
    report["run_id"] = manifest["run_id"]
    report["observations_sha256"] = manifest["observations_sha256"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("dataset")
    run = commands.add_parser("run")
    run.add_argument("dataset")
    run.add_argument("--out", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--revision", help="required 40-character SHA for a Hub model; omit for a local directory")
    run.add_argument("--subfolder")
    run.add_argument("--max-len", type=int, required=True)
    run.add_argument("--head-max-len", type=int, required=True)
    run.add_argument("--min-confidence", type=float)
    summary = commands.add_parser("summarize")
    summary.add_argument("run_dir")
    summary.add_argument("--out", help="new output file; omit to print JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            cases = load_cases(args.dataset)
            print(json.dumps({"cases": len(cases), "input_sha256": digest(cases)}))
        elif args.command == "summarize":
            report = summarize_saved(args.run_dir)
            if args.out:
                with Path(args.out).open("xb") as handle:
                    handle.write(encoded(report) + b"\n")
            else:
                print(json.dumps(report, indent=2, allow_nan=False))
        else:
            cases = load_cases(args.dataset)
            settings = {"max_len": args.max_len, "head_max_len": args.head_max_len,
                        "min_confidence": args.min_confidence}
            validate_settings(settings)
            from ._context_runner import cpu_runner
            collect(cases, lambda: cpu_runner(args.model, args.revision, args.subfolder),
                    args.out, settings, file_digest(args.dataset))
            print("Saved complete run and report to %s" % args.out)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print("context-robustness: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
