"""Versioned development, ordering and timing extensions to the fixed-k study."""
import argparse
import copy
from dataclasses import replace
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import laya  # noqa: E402
from laya.evals import Dataset, Example, file_fingerprint, questions_fingerprint  # noqa: E402
from research.benchmarks.tournament import compare_embeddings as fixed  # noqa: E402
from research.benchmarks.tournament.diagnostics import fingerprint, outcome, stage_metrics  # noqa: E402
from research.benchmarks.tournament.embedders import RoleEmbedder, SPECS, TextEncoder  # noqa: E402

SCHEMA = "laya-dedicated-shortlist/amendment-v2"
VARIANTS = {"minilm": SPECS["minilm"], "nomic-qd": SPECS["nomic"],
            "nomic-qq": replace(SPECS["nomic"], option_prefix="search_query: ")}
QUALITY_STRATEGIES = ("top1", "shortlist", "combined", "tournament")


def selected_cases(name, split, limit, offset, seed):
    if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("offset must be nonnegative and limit positive")
    cases, dataset, identity = fixed.load_cases(name, split, limit + offset, seed)
    cases = cases[offset:]
    dataset = Dataset(dataset.examples[offset:])
    identity = dict(identity, case_ids_sha256=fingerprint([c["id"] for c in cases]),
                    questions_sha256=questions_fingerprint(dataset), offset=offset)
    return cases, dataset, identity


def label_order(questions, seed):
    labels = list(questions["q"]["criteria"])
    if seed is not None:
        random.Random(seed).shuffle(labels)
    return labels


class OrderedPredictor(fixed.Predictor):
    """Reorder the already selected candidates before grouping, preserving membership."""

    def __init__(self, agent, strategy, group_size, order=None):
        super().__init__(agent, strategy, group_size)
        if order is not None and len(order) != len(set(order)):
            raise ValueError("global label order contains duplicates")
        self.order = list(order) if order is not None else None
        self.presented = None

    def predict(self, state, questions, **kwargs):
        if self.order is not None:
            questions = copy.deepcopy(questions)
            for question in questions.values():
                if question.get("type") != "choice":
                    continue
                original = question["criteria"]
                if question.get("option_order") is not None:
                    raise ValueError("permutation requires canonical criteria without option_order")
                if not isinstance(original, dict) or not set(original) <= set(self.order):
                    raise ValueError("candidate set differs from the declared label universe")
                question["criteria"] = {label: original[label] for label in self.order if label in original}
                if set(question["criteria"]) != set(original):
                    raise ValueError("ordering changed retrieval membership")
        self.presented = list(questions["q"]["criteria"])
        return super().predict(state, questions, **kwargs)


def runtime_identity(agent, encoder, requested_device):
    from research.benchmarks.tournament.interleaved import _device, _placement
    requested = _device(requested_device)
    result = {"requested_device": requested, "laya": _placement(agent, requested),
              "cpu_fallback_count": getattr(agent, "cpu_fallback_count", 0),
              "amp_failures": getattr(agent, "_amp_failures", 0),
              "autocast_dtype": str(getattr(agent, "dtype", None)),
              "amp_enabled": getattr(agent, "amp_enabled", None)}
    if encoder is not None:
        result["encoder"] = _placement(encoder, requested)
    return result


class CheckedAgent:
    """Catch a scoped fallback before another tournament round can reset it."""

    def __init__(self, agent, encoder, requested, runtime):
        self.agent, self.encoder, self.requested, self.runtime = agent, encoder, requested, runtime

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def predict(self, *args, **kwargs):
        if runtime_identity(self.agent, self.encoder, self.requested) != self.runtime:
            raise RuntimeError("device, dtype or fallback counters changed before decision")
        result = self.agent.predict(*args, **kwargs)
        if runtime_identity(self.agent, self.encoder, self.requested) != self.runtime:
            raise RuntimeError("device, dtype or fallback counters changed during decision")
        return result


def quality(agent, cases, questions, identity, strategy, embedder, k=20, group_size=16,
            order=None, diagnose=False, requested_device=None):
    """Record actual live predictions; labels enter only the subsequent analysis."""
    rows = []
    if strategy not in ("shortlist", "combined", "tournament"):
        raise ValueError("quality supports dedicated shortlist/combined and tournament")
    if set(questions) != {"q"} or questions["q"].get("type") != "choice":
        raise ValueError("amendment requires one choice question named q")
    original = list(questions["q"]["criteria"])
    if order is not None and (len(order) != len(original) or set(order) != set(original)):
        raise ValueError("global order must be a permutation of the canonical labels")
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("missing or duplicate planned cases")
    if identity.get("case_ids_sha256") != fingerprint([case["id"] for case in cases]):
        raise ValueError("planned case identity differs")
    schema = Dataset([Example(cases[0]["state"], questions, {})])
    if identity.get("questions_sha256") != questions_fingerprint(schema):
        raise ValueError("question identity differs")
    if any(case["expected"] not in original for case in cases):
        raise ValueError("unknown gold label")
    encoder = getattr(embedder, "encoder", None)
    requested_device = requested_device or agent.device
    runtime = runtime_identity(agent, encoder, requested_device)
    guarded = CheckedAgent(agent, encoder, requested_device, runtime)
    fixed.reset_cache(embedder, questions["q"])
    for case in cases:
        if fingerprint(case["state"]) != case["state_sha256"]:
            raise ValueError("case content changed")
        recorder = fixed.CallRecorder(guarded)
        predictor = OrderedPredictor(recorder, strategy, group_size, order)
        result = fixed.predict_strategy(predictor, case["state"], questions, embedder, k)
        if runtime_identity(agent, encoder, requested_device) != runtime:
            raise RuntimeError("device, dtype or fallback counters changed during quality evaluation")
        answer = result["answers"]["q"]
        shortlist = result.get("shortlist", {}).get("q", {})
        kept = shortlist.get("labels", original)
        if strategy in fixed.SHORTLIST_STRATEGIES and ("labels" not in shortlist or len(kept) != min(k, len(original))):
            raise ValueError("shortlist metadata missing or inconsistent")
        finalists = result.get("tournament", {}).get("q", {}).get("labels", predictor.presented)
        gold = case["expected"]
        row = {"id": case["id"], "state_sha256": case["state_sha256"], "expected": gold,
               "choice": answer["choice"], "correct": answer["choice"] == gold,
               "confidence": answer.get("answer_confidence"), "shortlist": kept,
               "decision_order": predictor.presented, "finalists": finalists,
               "gold_rank": kept.index(gold) + 1 if strategy != "tournament" and gold in kept else None,
               "decision_calls": len(recorder.calls),
               "outcome": outcome(gold, answer["choice"], original, kept, finalists),
               "groups": [[list(q["criteria"]) for q in call["questions"].values()]
                          for call in recorder.calls]}
        if diagnose:
            row["calls"] = fixed.diagnose_calls(agent, case["state"], recorder.calls)
        rows.append(row)
    return {"status": "complete", "identity": copy.deepcopy(identity), "strategy": strategy, "k": k,
            "runtime": runtime,
            "group_size": group_size, "stages": stage_metrics(rows), "cases": rows}


def alias_single_pass(result):
    if (result.get("status") != "complete" or result["strategy"] != "shortlist" or
            result["k"] > result["group_size"] or not result["cases"] or
            any(row["decision_calls"] != 1 for row in result["cases"])):
        raise ValueError("only a shortlist fitting one group is an equivalent combined configuration")
    result = copy.deepcopy(result)
    result.update(strategy="combined", equivalent_to="shortlist", degenerate_single_pass=True)
    return result


def source_identity():
    directory = Path(__file__).parent
    return {name: file_fingerprint(str(directory / name)) for name in
            ("amended_comparison.py", "interleaved.py", "compare_embeddings.py",
             "embedders.py", "diagnostics.py", "benchmark.py")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("quality", "latency"), default="quality")
    ap.add_argument("--dataset", choices=sorted(fixed.DATASETS), default="banking77")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="minilm")
    ap.add_argument("--strategies", default="shortlist,combined")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--permutation-seed", type=int)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--arms-file", help="timing arm records with id/strategy/k/embedder/group_size")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    strategies = args.strategies.split(",")
    if (args.threads < 1 or args.k < 1 or args.group_size < 2 or args.limit < 1 or args.offset < 0
            or args.passes < 1 or len(set(strategies)) != len(strategies)
            or any(s not in fixed.STRATEGIES for s in strategies)):
        ap.error("invalid strategy or execution parameters")
    if args.mode == "latency" and args.permutation_seed is not None:
        ap.error("order diagnostics are quality-only")
    if args.mode == "quality" and any(s not in QUALITY_STRATEGIES for s in strategies):
        ap.error("quality supports only top1, shortlist, combined and tournament")
    if args.mode == "quality" and args.arms_file is not None:
        ap.error("arms-file is latency-only")
    if Path(args.out).exists():
        ap.error("refusing to overwrite an existing report")
    import torch
    torch.set_num_threads(args.threads)
    report = {"schema": SCHEMA, "status": "running", "strategies": {},
              "parameters": {key: value for key, value in vars(args).items() if key != "out"},
              "source_sha256": source_identity(), "environment": fixed.environment()}
    fixed.write_report(args.out, report)
    try:
        cases, dataset, identity = selected_cases(args.dataset, args.split, args.limit, args.offset, args.seed)
        identity["checkpoint_revision"] = fixed.CHECKPOINT_REVISION
        questions = dataset.examples[0].questions
        report.update(identity=identity, rows=[{key: c[key] for key in ("id", "state_sha256")} for c in cases])
        spec = VARIANTS[args.variant]
        encoder = TextEncoder(spec, args.device, local_files_only=args.local_files_only)
        embedder = RoleEmbedder(encoder, spec)
        report["embedder"] = embedder.identity()
        identity = dict(identity, embedder_id=fingerprint(report["embedder"]))
        from huggingface_hub import snapshot_download
        checkpoint = Path(snapshot_download("convaiinnovations/laya", revision=fixed.CHECKPOINT_REVISION,
                          local_files_only=args.local_files_only,
                          allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]))
        agent = laya.load(str(checkpoint), device=args.device)
        report["runtime"] = {"requested_device": args.device, "actual_laya_device": str(agent.device),
                             "actual_embedder_device": str(encoder.device),
                             "torch_threads": torch.get_num_threads(),
                             "torch_interop_threads": torch.get_num_interop_threads()}
        runtime_before = runtime_identity(agent, encoder, args.device)
        report["runtime"]["placement_before"] = runtime_before
        report["checkpoint"] = {"revision": fixed.CHECKPOINT_REVISION,
                                "sha256": {name: file_fingerprint(str(checkpoint / name)) for name in
                                           ("model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json")}}
        if args.mode == "latency":
            import json
            from research.benchmarks.tournament.interleaved import Arm, LatencyFailure, run_interleaved
            definitions = (json.loads(Path(args.arms_file).read_text(encoding="utf-8")) if args.arms_file else
                           [{"id": s, "strategy": s, "k": args.k, "group_size": args.group_size,
                             "embedder": args.variant if s in fixed.SHORTLIST_STRATEGIES + ("top1",)
                             and s != "builtin" else None} for s in strategies])
            arms = [Arm(**row) for row in definitions]
            keys = {arm.embedder for arm in arms if arm.embedder is not None}
            if any(VARIANTS[key].repo != spec.repo for key in keys):
                raise ValueError("a latency process may share one dedicated model's weights only")
            report["arm_definitions"] = definitions
            encoders = {key: encoder for key in keys}
            specs = {key: VARIANTS[key] for key in keys}
            def progress(result):
                report["timing"] = result
                fixed.write_report(args.out, report)
            try:
                report["timing"] = run_interleaved(agent, cases, questions, arms, encoders, specs,
                    fixed.warmup_cases(args.dataset, cases, args.seed), requested_device=args.device,
                    passes=args.passes, progress=progress)
            except LatencyFailure as exc:
                report["timing"] = exc.report
                raise
        else:
            order = label_order(questions, args.permutation_seed) if args.permutation_seed is not None else None
            report["ordering"] = {"seed": args.permutation_seed, "global_labels": order,
                                   "policy": "global permutation restricted to retrieved set" if order else "natural"}
            for strategy in strategies:
                fixed.reset_cache(embedder, questions["q"])
                if strategy == "combined" and args.k <= args.group_size and "shortlist" in report["strategies"]:
                    result = alias_single_pass(report["strategies"]["shortlist"])
                elif strategy == "top1":
                    result = dict(fixed.retrieval(cases, questions, embedder), identity=identity,
                                  status="complete", strategy="top1", k=1, group_size=args.group_size)
                else:
                    result = quality(agent, cases, questions, identity, strategy,
                        embedder if strategy in fixed.SHORTLIST_STRATEGIES else None,
                        args.k, args.group_size, order, args.diagnose, requested_device=args.device)
                report["strategies"][strategy] = result
                fixed.write_report(args.out, report)
                print(strategy, result.get("stages", {}).get("accuracy", result.get("accuracy")), flush=True)
        report["memory"] = fixed.memory_snapshot(args.device)
        runtime_after = runtime_identity(agent, encoder, args.device)
        report["runtime"]["placement_after"] = runtime_after
        if runtime_after != runtime_before:
            raise RuntimeError("device, dtype or fallback counters changed during execution")
        if source_identity() != report["source_sha256"]:
            raise RuntimeError("source changed during execution")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error="%s: %s" % (type(exc).__name__, exc))
        raise
    finally:
        fixed.write_report(args.out, report)


if __name__ == "__main__":
    main()
