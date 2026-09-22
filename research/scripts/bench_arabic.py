"""Reproducible Arabic regression benchmark using the public Laya SDK.

Run from any directory: python /path/to/research/scripts/bench_arabic.py --help
No model is loaded with --routing-only. Otherwise only multilingual is loaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GROUPS = ("msa", "emirati", "mixed")
CATEGORIES = ("billing", "technical", "delivery", "general")
DEFAULT_DATA = ROOT / "research/data/arabic_triage.json"


def questions(language):
    from laya import arabic_triage_questions
    if language == "ar":
        return arabic_triage_questions()
    if language != "en":
        raise ValueError("question language must be ar or en")
    return {
        "category": {"type": "choice", "instructions": "Which team should handle the customer's request in message?",
                     "criteria": dict(zip(CATEGORIES, [
                         "invoices, payments, refunds", "technical errors, login or app problems",
                         "order delivery, shipping, shipment tracking", "general inquiries, opening hours, service information"]))},
        "refund_requested": {"type": "noul", "instructions": "Does the customer explicitly request money back in message?",
                             "criteria": {"false": "does not request money back", "true": "requests money back"}},
        "urgency": {"type": "score", "instructions": "How urgent is the request in message?",
                    "criteria": ["not urgent, can wait", "needs attention soon or today without a fixed deadline",
                                 "blocking work or an explicit hard deadline that cannot move"]},
    }


def load_cases(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or not data.get("provenance") or not data.get("cases"):
        raise ValueError("dataset needs schema_version=1, provenance and nonempty cases")
    seen = set()
    for case in data["cases"]:
        if not case.get("id") or case["id"] in seen:
            raise ValueError("case IDs must be unique and nonempty")
        seen.add(case["id"])
        gold = case.get("expected", {})
        if (case.get("group") not in GROUPS or case.get("expected_route") != "multilingual"
                or not isinstance(case.get("state", {}).get("message"), str)
                or not case["state"]["message"].strip()
                or gold.get("category") not in CATEGORIES
                or type(gold.get("refund_requested")) is not bool
                or type(gold.get("urgency")) is not int or gold["urgency"] not in (0, 1, 2)):
            raise ValueError("invalid Arabic case: " + case["id"])
    return data


def mean(values):
    return sum(values) / len(values)


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def ece(confidences, correct, bins=10):
    # Include zero in the first bin and one in the last bin.
    buckets = [[] for _ in range(bins)]
    for confidence, hit in zip(confidences, correct):
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, hit))
    return sum(len(b) / len(correct) * abs(mean([x[0] for x in b]) - mean([x[1] for x in b]))
               for b in buckets if b)


def summarize(rows):
    result = {"cases": len(rows), "routing_accuracy": mean([r["route"]["model"] == r["expected_route"] for r in rows])}
    if not all("answers" in r for r in rows):
        return result
    cat_hits, refund_hits, cat_conf, refund_conf, brier, mae, ordinal_hits = [], [], [], [], [], [], []
    for row in rows:
        answer, gold = row["answers"], row["expected"]
        cat = answer["category"]
        p = answer["refund_requested"]["noul"]
        score = answer["urgency"]["score"]
        if (not math.isfinite(p) or not 0 <= p <= 1 or not math.isfinite(score) or not 0 <= score <= 2
                or set(cat["probabilities"]) != set(CATEGORIES)
                or cat["choice"] not in CATEGORIES
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in cat["probabilities"].values())
                or abs(sum(cat["probabilities"].values()) - 1) > 0.001):
            raise ValueError("invalid prediction for " + row["id"])
        cat_hits.append(cat["choice"] == gold["category"])
        cat_conf.append(cat["probabilities"][cat["choice"]])
        refund_hits.append((p >= 0.5) == gold["refund_requested"])
        refund_conf.append(max(p, 1 - p))
        brier.append((p - int(gold["refund_requested"])) ** 2)
        mae.append(abs(score - gold["urgency"]))
        ordinal_hits.append(min(2, int(score + 0.5)) == gold["urgency"])
    latency = [r["latency_ms"] for r in rows]
    result.update({
        "category_accuracy": mean(cat_hits),
        "category_ece": ece(cat_conf, cat_hits),
        "refund_accuracy": mean(refund_hits),
        "refund_brier": mean(brier),
        "refund_ece": ece(refund_conf, refund_hits),
        "urgency_mae": mean(mae),
        "urgency_nearest_level_accuracy": mean(ordinal_hits),
        "latency_p50_ms": percentile(latency, 0.5),
        "latency_p95_ms": percentile(latency, 0.95),
        "baselines": {
            "category_majority_accuracy": max(sum(r["expected"]["category"] == c for r in rows) for c in CATEGORIES) / len(rows),
            "refund_majority_accuracy": max(sum(r["expected"]["refund_requested"] == v for r in rows) for v in (False, True)) / len(rows),
            "urgency_constant_1_mae": mean([abs(1 - r["expected"]["urgency"]) for r in rows]),
        },
    })
    return result


def run(data, router, languages, routing_only=False):
    rows = []
    for language in languages:
        qs = questions(language)
        for case in data["cases"]:
            decision = dict(router.route(case["state"], qs))
            row = dict(case, question_language=language, route=decision)
            if not routing_only:
                start = time.perf_counter()
                response = router.predict(case["state"], qs)
                row.update(answers=response["answers"], route=response["routing"],
                           latency_ms=1000 * (time.perf_counter() - start), usage=response["usage"])
            rows.append(row)
    summaries = {}
    for language in languages:
        for group in ("all", *GROUPS):
            selected = [r for r in rows if r["question_language"] == language and (group == "all" or r["group"] == group)]
            if selected:
                summaries[f"{language}/{group}"] = summarize(selected)
    return {"summaries": summaries, "cases": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routing-only", action="store_true")
    parser.add_argument("--question-language", choices=("ar", "en", "both"), default="both")
    parser.add_argument("--model-path", help="Local multilingual checkpoint directory; otherwise download the bundled checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    data = load_cases(args.data)
    from laya import Router
    router = Router(device=args.device, models={"multilingual": args.model_path} if args.model_path else None)
    languages = ["ar", "en"] if args.question_language == "both" else [args.question_language]
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "routing_only" if args.routing_only else "model_inference",
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "dataset_provenance": data["provenance"], "limitations": data.get("limitations", []),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted((ROOT / "laya").glob("*.py")), Path(__file__).resolve()]},
        "platform": platform.platform(), "python": platform.python_version(),
        "question_languages": languages, "requested_device": args.device,
        "packages": {p: importlib.metadata.version(p) for p in ("laya", "torch", "transformers", "numpy")},
        "metric_notes": "ECE uses predicted-label probabilities, not Laya entropy confidence. Latency is warm SDK time for 3 questions per case. Rounding uses half-up for ordinal levels. Groups are parallel synthetic scenarios.",
    }
    if not args.routing_only:
        import torch
        # Fail before model loading if the corpus would need an unexpected checkpoint.
        for case in data["cases"]:
            if router.route(case["state"]).model != "multilingual":
                raise ValueError("case does not route to multilingual: " + case["id"])
        agent = router.load("multilingual")
        metadata.update(actual_device=str(agent.device), checkpoint=router.models["multilingual"],
                        model_config=agent.cfg, temperatures=agent.temperature,
                        temperature_by_options=agent.temperature_by_options,
                        cpu_threads=torch.get_num_threads(), machine=platform.machine())
        if args.model_path:
            metadata["checkpoint_path"] = str(Path(args.model_path).resolve())
        else:
            from huggingface_hub import try_to_load_from_cache
            metadata["checkpoint_path"] = try_to_load_from_cache(
                "convaiinnovations/laya", "multilingual/rl_agent_config.json")
        for language in languages:
            router.predict(data["cases"][0]["state"], questions(language))
    report = dict(metadata=metadata, **run(data, router, languages, args.routing_only))
    # A failed run never writes a misleading partial success report.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summaries"], indent=2))
    return 0 if all(s["routing_accuracy"] == 1 for s in report["summaries"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
