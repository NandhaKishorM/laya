"""Re-derives every published number from the archive. Standard library only, no network.

    python research/benchmarks/es_phone_turns/audit.py

Checks that the frozen cases and prompts are the ones the results were produced with, that
every record answers a case that exists with the gold it has, and that the summary stored
beside the records is the one the records produce. Exits non-zero on the first thing that
does not hold.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402
import prompts  # noqa: E402

FROZEN = {
    "data/cases.jsonl": "6ea5c7a9ff26b8c2",
    "data/clinic.jsonl": "545990173a6667b7",
    "data/independent_cases.jsonl": "7d412fbf8683963f",
    "data/independent_clinic.jsonl": "ad6b345d9a2edfc7",
    "data/frustration.jsonl": "0274d9f4143c1819",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def close(a, b):
    if a is None or b is None:
        return a is b
    return abs(a - b) < 1e-9


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def audit(results_dir=None):
    failures = []
    for relative, expected in FROZEN.items():
        check(digest(HERE / relative) == expected,
              f"{relative} is not the frozen file (expected {expected})", failures)

    results_dir = Path(results_dir) if results_dir else HERE / "results"
    reports = sorted(results_dir.glob("*.json"))
    check(bool(reports), "no result files to audit", failures)

    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        dataset = report["meta"].get("dataset", "cases")
        cases = {}
        for line in (HERE / "data" / f"{dataset}.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                cases[row["id"]] = row

        records = report["records"]
        check(len(records) == len(cases),
              f"{path.name}: {len(records)} records for {len(cases)} cases", failures)
        check(len({r["id"] for r in records}) == len(records),
              f"{path.name}: a case was answered twice", failures)

        labels = set(prompts.labels(dataset))
        for record in records:
            case = cases.get(record["id"])
            if case is None:
                failures.append(f"{path.name}: {record['id']} is not a case")
                continue
            check(record["text"] == case["text"] and record["gold"] == case["gold"]
                  and record["tags"] == case["tags"],
                  f"{path.name}: {record['id']} does not match its case", failures)
            check(record["predicted"] in labels | {"invalid"},
                  f"{path.name}: {record['id']} predicted {record['predicted']!r}", failures)
            check(0.0 <= record["confidence"] <= 1.0,
                  f"{path.name}: {record['id']} has confidence {record['confidence']}", failures)
            if "probabilities" in record:
                check(abs(sum(record["probabilities"].values()) - 1.0) < 1e-3,
                      f"{path.name}: {record['id']} probabilities do not add up to 1", failures)
                best = max(record["probabilities"], key=record["probabilities"].get)
                check(best == record["predicted"],
                      f"{path.name}: {record['id']} reports {record['predicted']!r} "
                      f"but its most likely option is {best!r}", failures)

        rebuilt = metrics.summarize(records, prompts.labels(dataset))
        stored = report["summary"]
        for key in ("accuracy", "macro_f1", "ece", "mean_confidence", "majority_baseline"):
            check(close(rebuilt[key], stored[key]),
                  f"{path.name}: {key} is {stored[key]} in the file, {rebuilt[key]} recomputed",
                  failures)
        for key in ("n", "wrong_actions", "confusion", "by_tag", "fast_path"):
            check(rebuilt[key] == stored[key],
                  f"{path.name}: {key} does not match the records", failures)

    return reports, failures


def main():
    reports, failures = audit(sys.argv[1] if len(sys.argv) > 1 else None)
    for failure in failures:
        print("FAIL", failure)
    if failures:
        sys.exit(1)
    print(f"ok: {len(FROZEN)} frozen files and {len(reports)} result files re-derived")


if __name__ == "__main__":
    main()
