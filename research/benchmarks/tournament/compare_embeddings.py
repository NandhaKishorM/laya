"""Compare dedicated retrieval and tournament on the same pinned intent rows.

Run --help for quality, retrieval-only and independent end-to-end latency passes.
The existing benchmark and its historical results are left intact.
"""
import argparse
import copy
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import laya  # noqa: E402
from laya.common import render_options  # noqa: E402
from laya.evals import Dataset, Example, evaluate, file_fingerprint, questions_fingerprint  # noqa: E402
from laya.evals_shortlist import evaluate_shortlist  # noqa: E402
from laya.shortlist import predict_shortlist, predict_tournament, shortlist_choice  # noqa: E402
from research.benchmarks.tournament.benchmark import DATASETS, REVISIONS  # noqa: E402
from research.benchmarks.tournament.diagnostics import (  # noqa: E402
    fingerprint, outcome, paired_comparison, stage_metrics, token_diagnostics,
)
from research.benchmarks.tournament.embedders import RoleEmbedder, SPECS, TextEncoder, truncation_summary  # noqa: E402

CHECKPOINT_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
STRATEGIES = ("single", "wide", "builtin", "tournament", "top1", "shortlist", "combined", "shortlist-wide")
SHORTLIST_STRATEGIES = ("builtin", "shortlist", "combined", "shortlist-wide")
DATASET_REPOS = {"banking77": "mteb/banking77", "clinc150": "clinc/clinc_oos",
                 "massive": "SetFit/amazon_massive_intent_en-US"}


def load_cases(name, split, limit=None, seed=0):
    loader, instructions = DATASETS[name]
    actual_split = ("train" if name == "banking77" else "validation") if split == "dev" else split
    names, rows = loader(actual_split)
    labels = [name.replace("_", " ") for name in names]
    if len(set(labels)) != len(labels):
        raise ValueError("label humanization is not unique")
    questions = {"q": {"type": "choice", "instructions": instructions, "criteria": dict.fromkeys(labels)}}
    indices = list(range(len(rows)))
    if limit is not None:
        if isinstance(limit, bool) or limit < 1 or limit > len(rows):
            raise ValueError("limit must be between 1 and the split size")
        # A pilot must be a prefix of the larger development sample. Sampling only
        # `limit` items can switch Random.sample's algorithm as the limit grows.
        indices = random.Random(seed).sample(indices, len(indices))[:limit]
    cases, examples = [], []
    for i in indices:
        text, gold = rows[i]
        gold = gold.replace("_", " ")
        if gold not in labels:
            raise ValueError("unknown gold label")
        state = {"message": text}
        ident = "%s:%s:%d" % (name, actual_split, i)
        cases.append({"id": ident, "source_index": i, "state": state,
                      "state_sha256": fingerprint(state), "expected": gold})
        examples.append(Example(state, questions, {"q": gold}, tags=(ident,)))
    dataset = Dataset(examples)
    identity = {"dataset": name, "split": actual_split, "dataset_revision": REVISIONS[DATASET_REPOS[name]],
                "questions_sha256": questions_fingerprint(dataset),
                "case_ids_sha256": fingerprint([c["id"] for c in cases]), "seed": seed,
                "sampling": "seeded permutation prefix" if limit is not None else "official order"}
    return cases, dataset, identity


class CallRecorder:
    """Observe each real decision call; never supply labels from the gold data."""

    def __init__(self, agent):
        self.agent, self.calls = agent, []

    def predict(self, state, questions, **kwargs):
        original = copy.deepcopy(questions)
        result = self.agent.predict(state, questions, **kwargs)
        answers = result.get("answers", {})
        for qid, q in original.items():
            if q.get("type") == "choice":
                answer = answers.get(qid, {})
                if answer.get("type") != "choice" or answer.get("choice") not in q["criteria"]:
                    raise ValueError("decision returned a missing or out-of-set choice for %r" % qid)
        dtype = str(self.agent.dtype_for(len(questions))) if hasattr(self.agent, "dtype_for") else None
        self.calls.append({"questions": original, "kwargs": dict(kwargs),
                           "usage": copy.deepcopy(result.get("usage", {})), "dtype": dtype})
        return result


class Predictor:
    """The decision stage, shared by evaluate_shortlist and the live timing path."""

    def __init__(self, agent, strategy, group_size=16, planned=None, record=False):
        if strategy not in STRATEGIES:
            raise ValueError("unknown strategy %r" % strategy)
        self.agent, self.strategy, self.group_size = agent, strategy, group_size
        self.planned, self.record, self.history = planned, record, []

    def predict(self, state, questions, model=None, **kwargs):
        if model is not None:
            raise ValueError("this comparison uses one pinned Agent, not a Router")
        if self.strategy in ("wide", "shortlist-wide"):
            kwargs = dict(kwargs, head_max_len=512, max_len=1024)
        case = None
        if self.planned is not None:
            index = len(self.history)
            if index >= len(self.planned) or fingerprint(state) != self.planned[index]["state_sha256"]:
                raise ValueError("evaluation call does not match the next planned case")
            case = self.planned[index]
        target = CallRecorder(self.agent) if self.record else self.agent
        if self.strategy in ("tournament", "combined"):
            result = predict_tournament(target, state, questions, self.group_size, **kwargs)
        else:
            result = target.predict(state, questions, **kwargs)
        if self.record:
            self.history.append({"id": case["id"] if case else None, "result": result, "calls": target.calls})
        return result


def predict_strategy(predictor, state, questions, embed_fn=None, k=20):
    """A live request path: embedding and all decision rounds are inside this call."""
    if predictor.strategy == "top1":
        q = questions["q"]
        return shortlist_choice(state, q["criteria"], embed_fn, k=1, instructions=q["instructions"])[0]
    if predictor.strategy in SHORTLIST_STRATEGIES:
        return predict_shortlist(predictor, state, questions, embed_fn, k=k)
    return predictor.predict(state, questions)


def reset_cache(embed_fn, question):
    if embed_fn is None:
        return
    embed_fn.cache_clear()
    texts = render_options({"t": "choice", "crit": question["criteria"]})
    if hasattr(embed_fn, "warm_options"):
        embed_fn.warm_options(texts)
    else:
        embed_fn(texts)


def warmup_cases(name, measured, seed):
    candidates, _, _ = load_cases(name, "dev")
    excluded = {case["state_sha256"] for case in measured}
    candidates = [case for case in candidates if case["state_sha256"] not in excluded]
    if len(candidates) < 3:
        raise ValueError("not enough disjoint development warmups")
    return random.Random(seed + 104729).sample(candidates, 3)


def infeasible_budget(agent, state, questions, strategy):
    # These heads do not depend on a retrieved or predicted candidate set.
    if strategy not in ("single", "wide"):
        return None
    budget = ({"max_len": 1024, "head_max_len": 512} if strategy == "wide" else
              {key: agent.cfg.get(key, default) for key, default in (("max_len", 512), ("head_max_len", 192))})
    info = token_diagnostics(agent.tok, state, questions["q"], **budget)
    if info["markers_kept"] != len(questions["q"]["criteria"]):
        return {"status": "infeasible", "reason": "option markers exceed the sequence budget",
                "budget": budget, "diagnostics": info}
    return None


def diagnose_calls(agent, state, calls):
    out = []
    for call in calls:
        budget = {key: call["kwargs"].get(key, agent.cfg.get(key, default))
                  for key, default in (("max_len", 512), ("head_max_len", 192))}
        questions = {}
        for qid, question in call["questions"].items():
            if question.get("type") == "choice":
                questions[qid] = dict(labels=list(question["criteria"]),
                                     **token_diagnostics(agent.tok, state, question, **budget))
        out.append({"budget": budget, "questions": questions, "usage": call["usage"], "dtype": call["dtype"]})
    return out


def quality(agent, dataset, cases, identity, strategy, embed_fn=None, k=20, group_size=16, diagnose=True):
    """Use the existing evaluator; add tournament observations by checked case id."""
    if len(dataset.examples) != len(cases) or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("missing or duplicate planned cases")
    planned = [{key: case[key] for key in ("id", "state_sha256")} for case in cases]
    runner = Predictor(agent, strategy, group_size, planned=planned, record=True)
    if strategy in SHORTLIST_STRATEGIES:
        report = evaluate_shortlist(runner, dataset, embed_fn, k=k,
                                    checkpoint_id=identity["checkpoint_revision"],
                                    embedder_id=identity["embedder_id"])
    else:
        report = evaluate(runner, dataset)
    records = {record["id"]: record for record in runner.history}
    if len(records) != len(cases) or len(report.cases) != len(cases):
        raise ValueError("evaluation did not return every planned case")
    source = {case["id"]: case for case in cases}
    original = list(dataset.examples[0].questions["q"]["criteria"])
    observations = []
    for case in report.cases:
        ident, = case["tags"]
        raw = records.pop(ident)
        sample = source[ident]
        if case["qid"] != "q" or case["expected"] != sample["expected"]:
            raise ValueError("mismatched evaluation case")
        kept = case.get("shortlist", {}).get("labels", original)
        finalists = raw["result"].get("tournament", {}).get("q", {}).get("labels", kept)
        label = case["answer"]["choice"]
        phase = outcome(sample["expected"], label, original, kept, finalists)
        item = {"id": ident, "state_sha256": sample["state_sha256"], "expected": sample["expected"],
                "choice": label, "correct": case["correct"], "confidence": case["confidence"],
                "outcome": phase, "shortlist": kept, "finalists": finalists,
                "gold_rank": kept.index(sample["expected"]) + 1 if sample["expected"] in kept else None,
                "decision_calls": len(raw["calls"])}
        if diagnose:
            item["calls"] = diagnose_calls(agent, sample["state"], raw["calls"])
        observations.append(item)
    result = {"status": "complete", "identity": identity, "strategy": strategy, "k": k,
              "group_size": group_size, "overall": report.overall,
              "stages": stage_metrics(observations), "cases": observations}
    # The quality evaluator includes trace collection. Use the separate mode for latency claims.
    result["overall"] = {key: value for key, value in result["overall"].items()
                         if not key.startswith(("latency_", "cost_per_decision_"))}
    return result


def retrieval(cases, questions, embed_fn, ks=(16, 20, 32, 48)):
    q = questions["q"]
    labels = list(q["criteria"])
    # Ranking every option requires k < n; these fixed intent sets all have at least 60 labels.
    top = min(max(ks), len(labels) - 1)
    rows = []
    for case in cases:
        ranked = shortlist_choice(case["state"], q["criteria"], embed_fn, top, instructions=q["instructions"])
        rows.append({"id": case["id"], "state_sha256": case["state_sha256"], "expected": case["expected"],
                     "choice": ranked[0], "correct": ranked[0] == case["expected"],
                     "rank": ranked.index(case["expected"]) + 1 if case["expected"] in ranked else None})
    return {"accuracy": sum(r["correct"] for r in rows) / len(rows),
            "recall": {str(k): sum(r["rank"] is not None and r["rank"] <= k for r in rows) / len(rows)
                       for k in ks}, "cases": rows}


def synchronize(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def latency(agent, cases, questions, strategy, embed_fn=None, k=20, group_size=16,
            passes=3, clock=time.perf_counter):
    if passes < 1:
        raise ValueError("passes must be positive")
    runner = Predictor(agent, strategy, group_size)
    all_ms, cache = [], []
    for iteration in range(passes):
        reset_cache(embed_fn, questions["q"])
        values = []
        for case in cases:
            synchronize(agent.device)
            start = clock()
            predict_strategy(runner, case["state"], questions, embed_fn, k)
            synchronize(agent.device)
            values.append((clock() - start) * 1000)
        all_ms.append(values)
        cache.append(embed_fn.cache_info() if embed_fn is not None else None)
    return {"passes_ms": all_ms, "p50_ms": float(np.quantile(all_ms, .5)),
            "p95_ms": float(np.quantile(all_ms, .95)), "cache": cache,
            "strategy": strategy, "k": k, "group_size": group_size, "status": "complete",
            "rows_per_pass": len(cases), "scope": "query embedding + ranking + all decision rounds"}


def timing_schedule(strategies, passes):
    return [(iteration, strategy) for iteration in range(passes)
            for strategy in strategies[iteration % len(strategies):] + strategies[:iteration % len(strategies)]]


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment():
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"platform": platform.platform(), "python": platform.python_version(),
            "versions": versions, "laya_version": laya.__version__, "code_commit": commit,
            "runner_sha256": {name: file_fingerprint(str(Path(__file__).with_name(name)))
                              for name in ("compare_embeddings.py", "diagnostics.py", "embedders.py", "benchmark.py")}}


def memory_snapshot(device):
    """Process lifetime peak RSS, and current/peak device allocation where available."""
    import torch
    result = {"peak_rss_bytes": None, "scope": "whole process lifetime, including all resident models"}
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "paged_peak", "paged", "nonpaged_peak",
                                                   "nonpaged", "pagefile", "pagefile_peak")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if get_memory(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            result["peak_rss_bytes"] = counters.peak
    else:
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        except ImportError:
            pass
    if str(device).startswith("cuda"):
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("quality", "recall", "latency"), default="quality")
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", choices=sorted(SPECS), default="nomic")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--checkpoint-revision", default=CHECKPOINT_REVISION)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--local-files-only", action="store_true", help="load model weights only from cache; set HF_HUB_OFFLINE=1 for cached datasets too")
    ap.add_argument("--primary", action="store_true",
                    help="use 97.5%% intervals for the two preselected BANKING77 test comparisons")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.threads < 1 or args.k < 1 or args.group_size < 2 or args.passes < 1:
        ap.error("threads/k/passes must be positive and group-size at least two")
    if args.primary and (args.dataset != "banking77" or args.split != "test" or args.embedder == "nomic"):
        ap.error("primary comparisons require BANKING77 test and the development-selected lightweight model")
    if len(args.checkpoint_revision) != 40 or any(c not in "0123456789abcdef" for c in args.checkpoint_revision):
        ap.error("checkpoint-revision must be an immutable 40-character commit")
    strategies = args.strategies.split(",")
    if len(set(strategies)) != len(strategies) or any(s not in STRATEGIES for s in strategies):
        ap.error("unknown or duplicate strategy")
    if args.split == "dev" and args.limit is None:
        args.limit = 600
    import torch
    torch.set_num_threads(args.threads)
    output = {"schema": "laya-dedicated-shortlist/1", "status": "running", "mode": args.mode,
              "parameters": {key: value for key, value in vars(args).items() if key != "out"},
              "strategies": {}}
    write_report(args.out, output)
    agent, dedicated = None, None
    try:
        cases, dataset, identity = load_cases(args.dataset, args.split, args.limit, args.seed)
        questions = dataset.examples[0].questions
        output.update(identity=dict(identity, checkpoint_revision=args.checkpoint_revision),
                      environment=dict(environment(), threads=args.threads, device=args.device), planned_rows=len(cases))
        needed = args.mode == "recall" or any(s in ("top1", "shortlist", "combined", "shortlist-wide") for s in strategies)
        if needed:
            started = time.perf_counter()
            encoder = TextEncoder(SPECS[args.embedder], args.device, local_files_only=args.local_files_only)
            dedicated = RoleEmbedder(encoder, SPECS[args.embedder])
            output["embedder"] = dedicated.identity()
            output["embedding_acquire_and_load_ms"] = (time.perf_counter() - started) * 1000
            output["embedding_truncation"] = truncation_summary(encoder, cases, questions["q"])
            output["memory_after_embedding"] = memory_snapshot(args.device)
        if args.mode == "recall":
            reset_cache(dedicated, questions["q"])
            output["retrieval"] = retrieval(cases, questions, dedicated)
        else:
            started = time.perf_counter()
            from huggingface_hub import snapshot_download
            checkpoint = Path(snapshot_download("convaiinnovations/laya", revision=args.checkpoint_revision,
                              local_files_only=args.local_files_only,
                              allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]))
            agent = laya.load(str(checkpoint), device=args.device)
            output["decision_acquire_and_load_ms"] = (time.perf_counter() - started) * 1000
            output["checkpoint"] = {"repo": "convaiinnovations/laya", "revision": args.checkpoint_revision,
                                    "config": agent.cfg, "parallel_options": agent.parallel_options,
                                    "effective_temperature": agent.temperature,
                                    "effective_temperature_by_options": agent.temperature_by_options,
                                    "temperature_clamped": (agent.temperature != agent.temperature_raw or
                                       agent.temperature_by_options != agent.temperature_by_options_raw),
                                    "sha256": {name: file_fingerprint(str(checkpoint / name)) for name in
                                               ("model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json")}}
            builtin = laya.cached_embed_fn(laya.embed_fn_from_agent(agent)) if "builtin" in strategies else None
            warm = warmup_cases(args.dataset, cases, args.seed)
            output["warmup_case_ids"] = [row["id"] for row in warm]
            schedule = timing_schedule(strategies, args.passes) if args.mode == "latency" else [(0, s) for s in strategies]
            if args.mode == "latency":
                output["timing_schedule"] = schedule
            for iteration, strategy in schedule:
                infeasible = infeasible_budget(agent, cases[0]["state"], questions, strategy)
                if infeasible is not None:
                    output["strategies"][strategy] = infeasible
                    write_report(args.out, output)
                    print(strategy, "infeasible", flush=True)
                    continue
                embed = builtin if strategy == "builtin" else dedicated if strategy in SHORTLIST_STRATEGIES + ("top1",) else None
                reset_cache(embed, questions["q"])
                runner = Predictor(agent, strategy, args.group_size)
                for row in warm:
                    predict_strategy(runner, row["state"], questions, embed, args.k)
                reset_cache(embed, questions["q"])
                embedder_id = (fingerprint(output["embedder"]) if embed is not None and embed is dedicated else
                               "laya-encoder@" + args.checkpoint_revision if embed is not None else "none")
                ident = dict(output["identity"], embedder_id=embedder_id)
                if args.mode == "latency":
                    result = latency(agent, cases, questions, strategy, embed, args.k, args.group_size, passes=1)
                    result["identity"] = ident
                    previous = output["strategies"].get(strategy)
                    if previous is not None:
                        result["passes_ms"] = previous["passes_ms"] + result["passes_ms"]
                        result["cache"] = previous["cache"] + result["cache"]
                        result["p50_ms"] = float(np.quantile(result["passes_ms"], .5))
                        result["p95_ms"] = float(np.quantile(result["passes_ms"], .95))
                    result["status"] = "complete" if iteration + 1 == args.passes else "running"
                elif strategy == "top1":
                    result = dict(retrieval(cases, questions, embed), status="complete", identity=ident)
                else:
                    result = quality(agent, dataset, cases, ident, strategy, embed, args.k, args.group_size)
                output["strategies"][strategy] = result
                output["memory"] = memory_snapshot(args.device)
                write_report(args.out, output)
                print(strategy, result.get("stages", {}).get("accuracy", result.get("accuracy", result.get("p50_ms"))), flush=True)
            if args.mode == "quality" and "combined" in output["strategies"]:
                output["comparison_family"] = "primary" if args.primary else "exploratory"
                output["paired"] = {other: paired_comparison(output["strategies"]["combined"], result,
                                    confidence=.975 if args.primary else .95)
                                    for other, result in output["strategies"].items()
                                    if other in ("shortlist", "tournament") and result.get("status") == "complete"}
        output["memory"] = memory_snapshot(args.device)
        output["status"] = "complete"
    except Exception as exc:
        output.update(status="failed", error="%s: %s" % (type(exc).__name__, exc))
        raise
    finally:
        write_report(args.out, output)


if __name__ == "__main__":
    main()
