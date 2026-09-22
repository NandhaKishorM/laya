"""Compare original state-only routing with Arabic-aware routing across workflows.

Synthetic evaluation only; no fitting or prompt selection is performed here.
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
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.scripts.bench_arabic import ece, mean
from laya import Router
from laya.lang import analyse, latin_profile, state_text

GROUPS = ("msa", "emirati", "mixed", "english_control")
DATA = ROOT / "research/data/arabic_workflows.json"


def original_route(state):
    """Original default policy for these fixtures: dominant script, then Latin LID.

    This reconstructs routing only, not an old model runtime. The fixtures have
    no explicit hints or typed-decisions signatures. Arabic question text was
    ignored and minority Arabic text did not affect the Latin-language guess.
    """
    det = analyse(state)
    if det["script"] == "unknown":
        return "english"
    if det["script"] != "latin":
        return "multilingual"
    latin = latin_profile(state_text(state))
    english = latin["language"] == "en" or (latin["language"] is None and not latin["looks_non_english"])
    return "english" if english else "multilingual"


def load_data(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or not data.get("provenance") or not data.get("cases"):
        raise ValueError("invalid workflow dataset")
    seen = set()
    for case in data["cases"]:
        if not case.get("id") or case["id"] in seen or case.get("group") not in GROUPS:
            raise ValueError("invalid or duplicate case")
        seen.add(case["id"])
        if not case.get("state", {}).get("message"):
            raise ValueError("missing state")
        variants = data["workflows"][case["workflow"]]
        for language in ("ar", "en"):
            q = variants[language]["decision"]
            gold = case["expected"]
            valid = ((q["type"] == "choice" and isinstance(gold, str) and gold in q["criteria"])
                     or (q["type"] == "noul" and type(gold) is bool)
                     or (q["type"] == "score" and type(gold) is int and 0 <= gold < len(q["criteria"])))
            if not valid:
                raise ValueError("invalid label: " + case["id"])
        ar, en = variants["ar"]["decision"], variants["en"]["decision"]
        if ar["type"] != en["type"] or (ar["type"] == "choice" and list(ar["criteria"]) != list(en["criteria"])):
            raise ValueError("question language contracts differ")
        if ar["type"] == "score" and len(ar["criteria"]) != len(en["criteria"]):
            raise ValueError("score scale differs")
    return data


def summarize(rows, policy):
    metrics = {"cases": len(rows), "routing_accuracy": mean([r[policy]["model"] == r["expected_route"] for r in rows])}
    if "answer" not in rows[0][policy]:
        return metrics
    kinds = {r["type"] for r in rows}
    if len(kinds) != 1:
        raise ValueError("summarize one decision type at a time")
    kind = rows[0]["type"]
    hits, confidence, errors, brier = [], [], [], []
    for row in rows:
        answer, gold = row[policy]["answer"], row["expected"]
        if kind == "score":
            score = answer["score"]
            if not math.isfinite(score) or not 0 <= score <= row["max_score"]:
                raise ValueError("invalid score")
            errors.append(abs(score - gold))
            hits.append(int(score + 0.5) == gold)
        elif kind == "noul":
            p = answer["noul"]
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid probability")
            hits.append((p >= 0.5) == gold)
            confidence.append(max(p, 1 - p))
            brier.append((p - gold) ** 2)
        else:
            probs = answer["probabilities"]
            if (set(probs) != set(row["labels"]) or answer["choice"] not in probs
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values())
                    or abs(sum(probs.values()) - 1) > 0.001):
                raise ValueError("invalid choice probabilities")
            hits.append(answer["choice"] == gold)
            confidence.append(probs[answer["choice"]])
    metrics["accuracy" if kind != "score" else "nearest_level_accuracy"] = mean(hits)
    if errors:
        metrics["mae"] = mean(errors)
        metrics["best_constant_mae"] = min(mean([abs(c - r["expected"]) for r in rows])
                                           for c in range(rows[0]["max_score"] + 1))
    else:
        metrics["ece"] = ece(confidence, hits)
        metrics["majority_accuracy"] = max(sum(r["expected"] == label for r in rows)
                                           for label in {r["expected"] for r in rows}) / len(rows)
    if brier:
        metrics["brier"] = mean(brier)
    return metrics


def evaluate(data, router, routing_only=False):
    rows = []
    for language in ("ar", "en"):
        for case in data["cases"]:
            questions = data["workflows"][case["workflow"]][language]
            q = questions["decision"]
            new = router.route(case["state"], questions)
            old = original_route(case["state"])
            # English controls need multilingual only when the questions are Arabic.
            expected_route = "english" if language == "en" and case["group"] == "english_control" else "multilingual"
            row = dict(case, question_language=language, type=q["type"], expected_route=expected_route,
                       original={"model": old}, current={"model": new.model, "reason": new.reason})
            if q["type"] == "score":
                row["max_score"] = len(q["criteria"]) - 1
            elif q["type"] == "choice":
                row["labels"] = list(q["criteria"])
            if not routing_only:
                for policy in ("current", "original"):
                    if policy == "original" and old == new.model:
                        row[policy].update(answer=row["current"]["answer"], reused_prediction=True)
                        continue
                    # Route selection is the independent variable. Both paths use
                    # the same Agent.predict implementation, weights and questions.
                    agent = router.load(row[policy]["model"])
                    result = agent.predict(case["state"], questions)
                    row[policy]["answer"] = result["answers"]["decision"]
            rows.append(row)
    summaries = {}
    for language in ("ar", "en"):
        for workflow in data["workflows"]:
            for group in ("all", *GROUPS):
                selected = [r for r in rows if r["workflow"] == workflow and r["question_language"] == language
                            and (group == "all" or r["group"] == group)]
                if selected:
                    summaries[f"{language}/{workflow}/{group}"] = {
                        p: summarize(selected, p) for p in ("original", "current")}
    return {"summaries": summaries, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--routing-only", action="store_true")
    parser.add_argument("--english-path")
    parser.add_argument("--multilingual-path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data = load_data(args.data)
    models = {k: v for k, v in [("english", args.english_path), ("multilingual", args.multilingual_path)] if v}
    router = Router(models=models, device=args.device, max_loaded=2)
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "provenance": data["provenance"],
            "mode": "routing_only" if args.routing_only else "model_inference",
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [*sorted((ROOT / "laya").glob("*.py")),
                                        ROOT / "research/scripts/bench_arabic.py", Path(__file__).resolve()]},
            "platform": platform.platform(), "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ("laya", "torch", "transformers", "numpy")},
            "policy_comparison": "Original default dominant-script/state-only routing vs current Arabic-aware routing; same runtime and checkpoints. Predictions reused where routes agree. No statistical independence across parallel scenarios.",
            "models": {}}
    if not args.routing_only:
        for name in ("english", "multilingual"):
            agent = router.load(name)
            spec = router.models[name]
            if isinstance(spec, tuple):
                from huggingface_hub import try_to_load_from_cache
                repo, sub = spec
                path = try_to_load_from_cache(repo, (sub + "/" if sub else "") + "rl_agent_config.json")
            else:
                path = str(Path(spec).resolve())
            meta["models"][name] = {"checkpoint": spec, "resolved_path": path, "actual_device": str(agent.device),
                                    "config": agent.cfg, "temperatures": agent.temperature,
                                    "temperature_by_options": agent.temperature_by_options}
    start = time.perf_counter()
    report = {"metadata": meta, **evaluate(data, router, args.routing_only)}
    meta["evaluation_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for key, result in report["summaries"].items():
        if key.endswith("/all"):
            print(key, json.dumps(result), flush=True)
    return int(any(s["current"]["routing_accuracy"] != 1 for s in report["summaries"].values()))


if __name__ == "__main__":
    sys.exit(main())
