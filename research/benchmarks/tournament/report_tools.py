"""Compact research reports and recompute descriptive or paired results offline."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.benchmarks.tournament.diagnostics import paired_comparison  # noqa: E402

SCHEMA = "laya-study-evidence/1"
LABEL_FIELDS = ("expected", "choice")
ORDER_FIELDS = ("shortlist", "finalists", "decision_order")
CASE_FIELDS = ("id", "state_sha256", "expected", "choice", "correct", "confidence",
               "outcome", "shortlist", "finalists", "decision_order", "gold_rank",
               "decision_calls", "rank", "query_text_sha256", "prefixed_query_text_sha256")
REPORT_FIELDS = ("schema", "status", "mode", "parameters", "identity", "environment",
                 "source_sha256", "embedder", "runtime", "checkpoint", "ordering", "memory",
                 "arm_definitions", "comparison_family", "planned_rows", "warmup_case_ids",
                 "reference_checkpoint", "query_format", "decision_model_loaded", "interpretation",
                 "embedding_truncation", "rows")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") == SCHEMA:
        return expand(value)
    value["evidence_provenance"] = {"source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "omitted": None}
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def compact(report, source_sha256):
    require(report.get("status") == "complete", "cannot publish an incomplete report")
    labels, orders = [], []
    label_index, order_index = {}, {}

    def label(text):
        require(isinstance(text, str), "label must be text")
        if text not in label_index:
            label_index[text] = len(labels)
            labels.append(text)
        return label_index[text]

    def order(texts):
        require(isinstance(texts, list), "candidate order must be a list")
        indices = tuple(label(text) for text in texts)
        if indices not in order_index:
            order_index[indices] = len(orders)
            orders.append(list(indices))
        return order_index[indices]

    def arm(value):
        result = {k: copy.deepcopy(v) for k, v in value.items() if k != "cases"}
        if "cases" not in value:
            return result
        fields = [k for k in CASE_FIELDS if any(k in row for row in value["cases"])]
        rows = []
        diagnostics = {"questions": 0, "truncated_questions": 0, "state_tokens_dropped": 0}
        for row in value["cases"]:
            require(not (set(row) - set(CASE_FIELDS) - {"calls", "groups"}), "unrecognized case fields")
            present = [key for key in fields if key in row]
            values = []
            for key in present:
                item = row[key]
                if key in LABEL_FIELDS:
                    item = label(item)
                elif key in ORDER_FIELDS:
                    item = order(item)
                values.append(item)
            rows.append([sum(1 << fields.index(key) for key in present), values])
            for call in row.get("calls", []):
                for question in call["questions"].values():
                    diagnostics["questions"] += 1
                    diagnostics["truncated_questions"] += int(question["truncated"])
                    diagnostics["state_tokens_dropped"] += question["state_tokens_dropped"]
        result["case_columns"] = fields
        result["encoded_cases"] = rows
        result["trace_counts"] = diagnostics
        return result

    result = {k: copy.deepcopy(report[k]) for k in REPORT_FIELDS if k in report}
    if result.get("parameters", {}).get("arms_file"):
        result["parameters"]["arms_file"] = Path(result["parameters"]["arms_file"]).name
    if "strategies" in report:
        result["strategies"] = {name: arm(value) for name, value in report["strategies"].items()}
    if "retrieval" in report:
        result["retrieval"] = arm(report["retrieval"])
    if "timing" in report:
        result["timing"] = copy.deepcopy(report["timing"])
        for value in result["timing"]["arms"].values():
            value.pop("passes_ms", None)  # Identical observations remain in samples.
    return {"schema": SCHEMA, "source_report_sha256": source_sha256,
            "omitted": "token-level call traces and group descriptions; retained per-case candidates, outcomes, call counts and timing samples",
            "labels": labels, "orders": orders, "report": result}


def expand(evidence):
    require(evidence.get("schema") == SCHEMA, "unknown evidence schema")
    labels, orders = evidence["labels"], evidence["orders"]

    def entry(sequence, index):
        require(type(index) is int and 0 <= index < len(sequence), "invalid evidence reference")
        return sequence[index]

    def arm(value):
        result = copy.deepcopy(value)
        if "encoded_cases" not in result:
            return result
        fields = result.pop("case_columns")
        require(len(set(fields)) == len(fields) and set(fields) <= set(CASE_FIELDS), "invalid case columns")
        rows = []
        for mask, values in result.pop("encoded_cases"):
            require(type(mask) is int and 0 <= mask < 1 << len(fields), "invalid presence mask")
            present = [key for i, key in enumerate(fields) if mask & (1 << i)]
            require(len(values) == len(present), "case width differs")
            row = dict(zip(present, values))
            for key in LABEL_FIELDS:
                if key in row:
                    row[key] = entry(labels, row[key])
            for key in ORDER_FIELDS:
                if key in row:
                    row[key] = [entry(labels, i) for i in entry(orders, row[key])]
            rows.append(row)
        result["cases"] = rows
        return result

    result = copy.deepcopy(evidence["report"])
    if "strategies" in result:
        result["strategies"] = {name: arm(value) for name, value in result["strategies"].items()}
    if "retrieval" in result:
        result["retrieval"] = arm(result["retrieval"])
    result["evidence_provenance"] = {"source_report_sha256": evidence["source_report_sha256"],
                                    "omitted": evidence["omitted"]}
    return result


def case_outcome(row):
    """Reconstruct the first irreversible loss from retained candidate sets."""
    kept, final = row.get("shortlist"), row.get("finalists")
    require(isinstance(kept, list) and isinstance(final, list) and kept and final,
            "stage evidence requires candidate and finalist lists")
    require(len(set(kept)) == len(kept) and len(set(final)) == len(final)
            and set(final) <= set(kept) and row["choice"] in final,
            "invalid stage candidate sets")
    if "decision_order" in row:
        order = row["decision_order"]
        require(len(order) == len(kept) and set(order) == set(kept),
                "decision order changes candidate membership")
    if row["expected"] not in kept:
        return "retrieval_miss"
    if row["expected"] not in final:
        return "elimination_miss"
    return "correct" if row["choice"] == row["expected"] else "final_miss"


def get_arm(report, name):
    require(report.get("status") == "complete", "report is incomplete")
    if "retrieval" in report:
        require(name == "top1", "retrieval-only report has only top1")
        value = report["retrieval"]
    else:
        value = report["strategies"][name]
    require(value.get("status") == "complete", "arm is incomplete")
    require(isinstance(value.get("identity"), dict)
            and all(value["identity"].get(key) == item for key, item in report["identity"].items()),
            "arm identity differs from report identity")
    rows = value["cases"]
    require(rows and len({row["id"] for row in rows}) == len(rows), "missing or duplicate rows")
    if "rows" in report:
        require([(row["id"], row["state_sha256"]) for row in rows]
                == [(row["id"], row["state_sha256"]) for row in report["rows"]],
                "cases differ from the report row manifest")
    has_stages = any("outcome" in row for row in rows)
    for row in rows:
        require(type(row["correct"]) is bool and row["correct"] == (row["choice"] == row["expected"]),
                "correctness differs from predicted label")
        if has_stages:
            require(row.get("outcome") == case_outcome(row), "outcome differs from candidate stages")
    return value


def query_contract(report):
    return report.get("query_format", "instructions plus serialized state; model query prefix retained")


def pair(candidate, baseline, candidate_arm, baseline_arm, confidence=0.95, mixed_query=False):
    different = query_contract(candidate) != query_contract(baseline)
    require(not different or mixed_query, "different query contracts require --mixed-query")
    left, right = get_arm(candidate, candidate_arm), get_arm(baseline, baseline_arm)
    result = paired_comparison(left, right, confidence=confidence, samples=10000, seed=0)
    result.update(candidate=candidate_arm, baseline=baseline_arm,
                  candidate_query=query_contract(candidate), baseline_query=query_contract(baseline),
                  interpretation="mixed-query pipeline comparison" if different else
                  "paired pipeline comparison; differing k, model or prefixes do not isolate a decision-stage effect",
                  candidate_correct=sum(row["correct"] for row in left["cases"]),
                  baseline_correct=sum(row["correct"] for row in right["cases"]),
                  candidate_provenance=candidate.get("evidence_provenance"),
                  baseline_provenance=baseline.get("evidence_provenance"))
    return result


def summary(report):
    require(report.get("status") == "complete", "report is incomplete")
    result = {"identity": report["identity"], "parameters": report.get("parameters"),
              "query_contract": query_contract(report), "environment": report["environment"],
              "evidence_provenance": report.get("evidence_provenance")}
    if "timing" in report:
        timing = report["timing"]
        require(timing["status"] == "complete", "timing is incomplete")
        ids, passes, arms = timing["case_ids"], timing["passes"], timing["arms"]
        require(type(passes) is int and passes > 0 and arms, "invalid timing passes or arms")
        require(ids and len(set(ids)) == len(ids), "duplicate timing identities")
        for field in ("rows_per_pass", "independent_requests"):
            if field in timing:
                require(type(timing[field]) is int and timing[field] == len(ids),
                        "declared timing request count differs from case identities")
        if "rows" in report:
            require([row["id"] for row in report["rows"]] == ids
                    and all(isinstance(row.get("state_sha256"), str) and row["state_sha256"]
                            for row in report["rows"]), "timing row manifest differs from case identities")
        samples = timing["samples"]
        expected = {(p, row, arm) for p in range(passes) for row in ids for arm in arms}
        observed = {(row["pass"], row["id"], row["arm"]) for row in samples}
        require(len(samples) == len(expected) and observed == expected, "incomplete or duplicate timing observations")
        require(all(isinstance(row["ms"], (int, float)) and not isinstance(row["ms"], bool)
                    and math.isfinite(row["ms"]) and row["ms"] >= 0 for row in samples), "invalid timing")
        result["timing"] = {}
        for name in arms:
            values = [row["ms"] for row in samples if row["arm"] == name]
            result["timing"][name] = {"p50_ms": float(np.quantile(values, .5)),
                                      "p95_ms": float(np.quantile(values, .95)),
                                      "distinct_cases": len(ids), "passes": passes, "observations": len(values)}
        result["timing_scope"] = timing["scope"]
        result["interpretation"] = "Descriptive within-run quantiles; repeated passes are not independent cases."
    else:
        names = ["top1"] if "retrieval" in report else list(report["strategies"])
        result["strategies"] = {}
        for name in names:
            if report.get("strategies", {}).get(name, {}).get("status") == "infeasible":
                result["strategies"][name] = {"status": "infeasible"}
                continue
            arm = get_arm(report, name)
            rows = arm["cases"]
            counts = {}
            for row in rows:
                if "outcome" in row:
                    outcome = case_outcome(row)
                    counts[outcome] = counts.get(outcome, 0) + 1
            result["strategies"][name] = {"rows": len(rows), "correct": sum(row["correct"] for row in rows),
                                          "accuracy": sum(row["correct"] for row in rows) / len(rows),
                                          "stages": counts, "equivalent_to": arm.get("equivalent_to")}
    return result


def verify_grid(reports, selection, selector):
    """Check the retained grid contract, then reuse the frozen selection rule.

    This deliberately does not call the raw report validator: compact evidence
    omits the token traces that validator requires.
    """
    expected = {f"grid-dev200-{v}-k{k}.json" for v in selector.VARIANTS for k in selector.KS}
    require({name for name in reports if name.startswith("grid-dev200-")} == expected,
            "incomplete or unexpected compact development grid")
    grid, hashes, anchor = {}, {}, None
    for variant in selector.VARIANTS:
        for k in selector.KS:
            name = f"grid-dev200-{variant}-k{k}.json"
            report = reports[name]
            require(report["schema"] == "laya-dedicated-shortlist/amendment-v2"
                    and report["parameters"] == selector.parameters(variant=variant, k=k),
                    f"{name}: grid parameters differ")
            require(all(type(report["parameters"][field]) is int for field in
                        ("limit", "offset", "seed", "k", "group_size", "threads", "passes")),
                    f"{name}: grid numeric parameters must be integers")
            require(report["embedder"] == selector.SPECS[variant], f"{name}: model contract differs")
            require(report["ordering"] == {"seed": None, "global_labels": None, "policy": "natural"},
                    f"{name}: grid order differs")
            identity, rows = report["identity"], report["rows"]
            require(len(rows) == 200 and identity["dataset"] == "banking77"
                    and identity["split"] == "train" and identity["dataset_revision"] == selector.DATASET_REVISION
                    and identity["checkpoint_revision"] == selector.CHECKPOINT
                    and identity["case_ids_sha256"] == selector.fingerprint([r["id"] for r in rows]),
                    f"{name}: dev200 identities differ")
            require(set(report["strategies"]) == set(selector.METHODS), f"{name}: grid arms differ")
            gold = None
            for method in selector.METHODS:
                arm = get_arm(report, method)
                require(arm["k"] == k and arm["group_size"] == 16
                        and arm["identity"] == dict(identity, embedder_id=selector.fingerprint(report["embedder"])),
                        f"{name}: arm configuration differs")
                cases = arm["cases"]
                current = [(r["id"], r["state_sha256"], r["expected"]) for r in cases]
                require(gold is None or gold == current, f"{name}: gold labels differ between arms")
                gold = current
                for row in cases:
                    require(len(row["shortlist"]) == k and row["decision_order"] == row["shortlist"],
                            f"{name}: retained shortlist or order differs")
                    calls = 2 if method == "combined" and k > 16 else 1
                    require(type(row["decision_calls"]) is int and row["decision_calls"] == calls,
                            f"{name}: decision call count differs")
                    if calls == 1:
                        require(row["finalists"] == row["shortlist"], f"{name}: single-pass finalists differ")
            left, right = (report["strategies"][method] for method in selector.METHODS)
            require([r["shortlist"] for r in left["cases"]] == [r["shortlist"] for r in right["cases"]],
                    f"{name}: candidates differ between arms")
            if k <= 16:
                require(right.get("equivalent_to") == "shortlist" and right.get("degenerate_single_pass") is True
                        and left["cases"] == right["cases"], f"{name}: single-pass alias differs")
            else:
                require("equivalent_to" not in right and "degenerate_single_pass" not in right,
                        f"{name}: unexpected single-pass alias")
            common = {key: report[key] for key in ("identity", "rows", "checkpoint", "source_sha256", "environment")}
            common["gold"] = gold
            require(anchor is None or common == anchor, f"{name}: shared grid identities differ")
            anchor = common
            grid[variant, k] = report
            hashes[f"{variant}/k{k}"] = report["evidence_provenance"]["source_report_sha256"]
    require(selection["grid_sha256"] == hashes, "selection grid source hashes differ")
    selected = selector.choose(grid)
    require(selection["selected"] == selected, "saved selection differs from development outcomes")
    return {"cells": len(grid), "rows_per_cell": 200, "selected": selected,
            "source_hashes_match": True, "rule": "most correct, smaller k, then Nomic QD before QQ"}


def verify_pilot(reports, selection):
    """Recount the original development-only lightweight model comparison."""
    accuracies = {model: [] for model in ("bge", "minilm", "nomic")}
    contracts = {}
    for dataset in ("banking77", "clinc150", "massive"):
        anchor = None
        for model in accuracies:
            name = f"dev200-{dataset}-{model}.json"
            require(name in reports, f"missing pilot report: {name}")
            report = reports[name]
            params = report["parameters"]
            expected = dict(mode="quality", dataset=dataset, split="dev", limit=200,
                            seed=0, embedder=model, k=20, group_size=16)
            require(all(params.get(key) == value for key, value in expected.items()),
                    f"{name}: pilot parameters differ")
            require(contracts.setdefault(model, report["embedder"]) == report["embedder"],
                    f"{name}: pilot model contract differs")
            arm = get_arm(report, "shortlist")
            cases = arm["cases"]
            expected_split = "train" if dataset == "banking77" else "validation"
            require(len(cases) == 200 and report["identity"]["split"] == expected_split,
                    f"{name}: pilot is not dev200")
            common = (report["identity"], [(r["id"], r["state_sha256"], r["expected"]) for r in cases])
            require(anchor is None or common == anchor, f"{name}: shared pilot identities differ")
            anchor = common
            accuracies[model].append(sum(r["correct"] for r in cases) / len(cases))
    macro = {model: sum(values) / len(values) for model, values in accuracies.items()}
    stored = selection["macro_shortlist_accuracy"]
    require(set(stored) == {"bge", "minilm"} and all(
        math.isclose(stored[model], macro[model], rel_tol=0, abs_tol=1e-12) for model in stored),
        "pilot macro accuracy differs from saved model selection")
    require(selection["based_only_on"] == "development", "model selection must use development rows")
    require(abs(macro["bge"] - macro["minilm"]) >= .01 and selection["tie_break"] is None,
            "pilot tie-break needs separate latency/RSS evidence; this verifier covers the accuracy decision")
    selected = max(stored, key=lambda model: macro[model])
    require(selection["selected"] == selected, "saved pilot model choice differs")
    return {"reports": 9, "rows_per_report": 200, "macro_shortlist_accuracy": macro,
            "selected": selected, "nomic": "reference only; excluded from lightweight selection"}


def verify(directory):
    """Verify the published bundle without loading checkpoints or raw traces."""
    if __package__:
        from . import select_configurations as selector
    else:
        import select_configurations as selector

    directory = Path(directory).resolve()
    manifest = selector.read(directory / "manifest.json")
    require(manifest.get("schema") == "laya-study-manifest/1", "unknown manifest schema")
    files = manifest["files"]
    require(isinstance(files, list) and files, "manifest has no files")
    documents, reports, summaries = {}, {}, {}
    for item in files:
        name = item["file"]
        require(isinstance(name, str) and name and "/" not in name and "\\" not in name
                and ":" not in name and name not in {".", "..", "manifest.json"}, "invalid manifest filename")
        require(name not in documents, "duplicate manifest filename")
        path = directory / name
        require(path.resolve().parent == directory, "manifest entry escapes the evidence directory")
        data = path.read_bytes()
        require(selector.digest(item["sha256"]) and hashlib.sha256(data).hexdigest() == item["sha256"],
                f"{name}: file SHA256 differs")
        if "bytes" in item:
            require(type(item["bytes"]) is int and item["bytes"] == len(data), f"{name}: byte count differs")
        value = selector.loads(data.decode("utf-8"))
        documents[name] = value
        if value.get("schema") == SCHEMA:
            require(selector.digest(value["source_report_sha256"])
                    and item.get("source_report_sha256") == value["source_report_sha256"],
                    f"{name}: source report hash differs")
            reports[name] = expand(value)
            summaries[name] = summary(reports[name])
        else:
            require("source_report_sha256" not in item, f"{name}: expected compact report")
    for name in ("measurement.json", "selected-configurations.json", "model-selection.json",
                 "planned-samples.json", "frozen-protocol.json"):
        require(name in documents, f"missing required metadata: {name}")
    measurement = documents["measurement.json"]
    require(measurement.get("schema") == "laya-study-measurement/1", "unknown measurement schema")
    result = {"manifest_files": len(files), "compact_reports": len(reports),
              "quality_reports": sum("strategies" in value for value in summaries.values()),
              "timing_reports": sum("timing" in value for value in summaries.values()),
              "quality_case_records": sum(arm["rows"] for value in summaries.values()
                  for arm in value.get("strategies", {}).values() if "rows" in arm),
              "timing_observations": sum(arm["observations"] for value in summaries.values()
                  for arm in value.get("timing", {}).values()),
              "grid_selection": verify_grid(reports, documents["selected-configurations.json"], selector),
              "counts_note": "Case records repeat across arms/reports; observations include repeated passes. Neither total is an independent sample count.",
              "limits": ["Hashes verify the listed compact files and recorded raw-source identities, not unavailable raw bytes.",
                         "Omitted token-level calls and group traces cannot be re-audited from compact evidence.",
                         "This offline check does not rerun models or establish when predictions were first inspected."]}
    for declared, actual in (("complete_report_files", result["compact_reports"]),
                             ("timing_observations", result["timing_observations"]),
                             ("completed_matrix_jobs", sum(item.get("phase") == "amendment"
                                 and item["file"] in reports for item in files))):
        require(type(measurement.get(declared)) is int and measurement[declared] == actual,
                f"measurement inventory differs: {declared}")
    unique = measurement.get("timing_unique_requests_per_cell")
    require(type(unique) is int and unique > 0 and all(arm["distinct_cases"] == unique
            for value in summaries.values() for arm in value.get("timing", {}).values()),
            "measurement timing request count differs")
    result["measurement_inventory_matches"] = True
    result["pilot_selection"] = verify_pilot(reports, documents["model-selection.json"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("compact")
    pack.add_argument("report")
    pack.add_argument("--out", required=True)
    describe = sub.add_parser("summarize")
    describe.add_argument("report")
    describe.add_argument("--out")
    compare = sub.add_parser("pair")
    compare.add_argument("candidate")
    compare.add_argument("baseline")
    compare.add_argument("--candidate-arm", required=True)
    compare.add_argument("--baseline-arm", required=True)
    compare.add_argument("--confidence", type=float, default=.95)
    compare.add_argument("--mixed-query", action="store_true")
    compare.add_argument("--out")
    check = sub.add_parser("verify")
    check.add_argument("directory")
    check.add_argument("--out")
    args = parser.parse_args()
    if args.command == "compact":
        path = Path(args.report)
        source = json.loads(path.read_text(encoding="utf-8"))
        require(source.get("schema") != SCHEMA, "input is already compact")
        result = compact(source, hashlib.sha256(path.read_bytes()).hexdigest())
    elif args.command == "summarize":
        result = summary(read(args.report))
    elif args.command == "verify":
        result = verify(args.directory)
    else:
        result = pair(read(args.candidate), read(args.baseline), args.candidate_arm, args.baseline_arm,
                      args.confidence, args.mixed_query)
    if args.out:
        write_new(args.out, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
