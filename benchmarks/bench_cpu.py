"""Paired CPU inference latency, unrounded parity and peak-RSS evidence.

Fetch the pinned public checkpoints with `python verify/checkpoints.py --fetch`, then run:
    python benchmarks/bench_cpu.py --checkpoint multilingual --json result.json
    python benchmarks/bench_cpu.py --checkpoint multilingual --memory baseline --json rss-baseline.json
    python benchmarks/bench_cpu.py --checkpoint multilingual --memory selected --json rss-selected.json

Uses the public fast-path parity corpus plus long, wide-option and padded-batch fixtures.
No private data, remote inference, tuning, compilation or checkpoint changes are involved.
Run memory variants in separate processes and performance runs without competing benchmarks.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import transformers

from laya import Agent
from laya.common import temp_bucket
from benchmarks.parity_fast import states
from verify.checkpoints import sha256


def fixtures(suite="standard"):
    if suite == "long-options":
        topics = ("account access", "password reset", "two factor authentication", "account closure",
                  "email changes", "phone changes", "address changes", "identity verification",
                  "invoice corrections", "duplicate charges", "failed payments", "refund requests",
                  "subscription cancellation", "subscription upgrades", "discount eligibility",
                  "tax documents", "bank transfers", "card replacement", "card activation",
                  "card limits", "cash withdrawals", "foreign exchange", "delivery delays",
                  "order tracking", "damaged orders", "product returns", "product availability",
                  "software crashes", "installation errors", "network failures", "data exports",
                  "privacy requests", "general feedback")
        question = {"topic": {"type": "choice", "instructions": "Choose the support topic matching the request.",
                    "criteria": {topic: "Select this category when the customer needs assistance with " + topic
                                 for topic in topics}}}
        return [{"name": "long-options/%d" % length,
                 "state": {"body": "My invoice contains the wrong billing address. Please correct the invoice. " * length},
                 "questions": question, "kwargs": {"max_len": length, "head_max_len": 768}}
                for length in (795, 910, 1200, 1499)]
    cases = [{"name": name, "state": state, "questions": qs, "kwargs": {}}
             for name, state, qs in states()]
    question = {"team": {"type": "choice", "instructions": "Which team should handle this request?",
                         "criteria": {"billing": "payments and invoices", "support": "software failures"}}}
    for length in (512, 1024, 2048, 4096):
        cases.append({"name": "long/%d" % length,
                      "state": {"body": "The invoice needs a correction. " * length},
                      "questions": question, "kwargs": {"max_len": length}})
    for count in (1, 32):
        qs = {"route": {"type": "choice", "instructions": "Select the best topic.",
                        "criteria": {"topic%d" % i: "Topic number %d" % i for i in range(count)}}}
        cases.append({"name": "options/%d" % count, "state": {"body": "Discuss topic number zero."},
                      "questions": qs, "kwargs": {"max_len": 1024, "head_max_len": 768}})
    cases.append({"name": "batch/padded", "state": [{"body": "Please correct the invoice."},
                   {"body": "The software fails. " * 60}, {"body": "I need a refund. " * 150}],
                  "questions": question, "kwargs": {"batch_size": 2, "sort_by_length": True}})
    schema = {"type": "object", "properties": {
        "team": {"type": "string", "enum": ["billing", "support", "sales"]},
        "urgent": {"type": "boolean", "description": "Is this urgent?"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["team", "urgent", "priority"]}
    cases.append({"name": "structured/mixed", "state": {"body": "Please fix the invoice today."},
                  "schema": schema, "kwargs": {}})
    cases.append({"name": "structured/batch", "state": [{"body": "Fix the invoice today."},
                   {"body": "The software fails. " * 60}], "schema": schema,
                  "kwargs": {"batch_size": 4, "sort_by_length": True}})
    return cases


def public_call(agent, case):
    if "schema" in case:
        method = agent.decide_batch if isinstance(case["state"], list) else agent.decide
        return method(case["state"], case["schema"], return_details=True,
                      min_confidence=0.8, **case["kwargs"])
    method = agent.predict_batch if isinstance(case["state"], list) else agent.predict
    # Fixed before inference: exercise the public confidence gate, too.
    return method(case["state"], case["questions"], min_confidence=0.8, **case["kwargs"])


@contextmanager
def mode(variant, optimization="combined"):
    with ExitStack() as stack:
        if variant == "baseline" or optimization == "windows":
            stack.enter_context(patch("laya.common._can_use_cpu_head", return_value=False))
        if variant == "baseline" or optimization == "head":
            stack.enter_context(patch("laya.common._can_use_cpu_windows", return_value=False))
        yield


def answers(result):
    if isinstance(result, list):
        return [answers(x) for x in result]
    if hasattr(result, "values") and not isinstance(result, dict):
        return {"values": result.values, "answers": result.answers}
    return result["answers"]


def compare_public(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        return max((compare_public(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list):
        assert len(a) == len(b)
        return max((compare_public(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, float):
        delta = abs(a - b)
        assert delta <= 0.000100001, (a, b)
        return delta
    assert a == b, (a, b)  # labels, booleans, gate flags and integer answers
    return 0.0


def capture(agent, case, variant, optimization):
    rows = []

    def save(module, args, output):
        logits, act = (t.detach().float().numpy().copy() for t in output)
        rows.append({"logits": logits, "act": act, "counts": args[3].sum(-1).tolist(),
                     "types": args[4].tolist(), "shape": list(args[0].shape)})

    handle = agent.model.register_forward_hook(save)
    try:
        with mode(variant, optimization):
            result = public_call(agent, case)
    finally:
        handle.remove()
    return rows, answers(result)


def parity(agent, before, after):
    assert len(before) == len(after)
    result = {"questions": 0, "max_logit_delta": 0.0, "max_action_logit_delta": 0.0,
              "max_probability_delta": 0.0, "argmax_flips": 0}
    for a, b in zip(before, after):
        assert (a["shape"], a["counts"], a["types"]) == (b["shape"], b["counts"], b["types"])
        for key, summary in (("logits", "max_logit_delta"), ("act", "max_action_logit_delta")):
            np.testing.assert_allclose(a[key], b[key], atol=1e-4, rtol=1e-5)
            result[summary] = max(result[summary], float(np.max(np.abs(a[key] - b[key]))))
        for i, (count, qt) in enumerate(zip(a["counts"], a["types"])):
            temperature = agent.temperature_by_options.get(temp_bucket(qt, count), agent.temperature[qt])
            probs = []
            for output in (a, b):
                z = output["logits"][i, :count] / temperature
                p = np.exp(z - z.max())
                probs.append(p / p.sum())
            np.testing.assert_allclose(*probs, atol=1e-5, rtol=1e-5)
            result["max_probability_delta"] = max(result["max_probability_delta"], float(abs(probs[0] - probs[1]).max()))
            result["argmax_flips"] += int(probs[0].argmax() != probs[1].argmax())
            result["questions"] += 1
    assert result["argmax_flips"] == 0, result
    return result


def rss_mib():
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=("english", "multilingual", "typed-decisions"), required=True)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--repeats", type=positive, default=5)
    parser.add_argument("--limit", type=positive, help="smoke run only; omit for the full corpus")
    parser.add_argument("--case", action="append", help="repeat to select named fixtures for diagnostic reruns")
    parser.add_argument("--optimization", choices=("combined", "head", "windows"), default="combined")
    parser.add_argument("--suite", choices=("standard", "long-options"), default="standard")
    parser.add_argument("--memory", choices=("baseline", "selected"))
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error("output already exists; choose a fresh path")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    manifest = json.loads((ROOT / "verify/checkpoints.json").read_text())
    entry = next(c for c in manifest["checkpoints"] if c["name"] == args.checkpoint)
    weights = ROOT / "models" / entry["path"]
    digest = sha256(weights)
    assert digest == entry["sha256"], "checkpoint differs from verify/checkpoints.json"
    agent = Agent(str(weights.parent), device="cpu", fast=False, compile=False)
    agent.amp_enabled = False
    assert next(agent.model.parameters()).dtype == torch.float32
    versions = [(id(p), p._version) for p in agent.model.parameters()]
    cases = fixtures(args.suite)
    if args.case:
        unknown = set(args.case) - {case["name"] for case in cases}
        if unknown:
            parser.error("unknown cases: %s" % sorted(unknown))
        cases = [case for case in cases if case["name"] in args.case]
    if args.limit:
        cases = cases[:args.limit]
    report = {"checkpoint": args.checkpoint, "checkpoint_sha256": digest, "hub_revision": manifest["revision"],
              "platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
              "threads": args.threads, "interop_threads": 1, "repeats": args.repeats,
              "optimization": args.optimization, "suite": args.suite,
              "common_sha256": sha256(ROOT / "laya/common.py"), "harness_sha256": sha256(Path(__file__)),
              "fixtures_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
              "memory_variant": args.memory, "peak_rss_after_load_mib": rss_mib(), "cases": []}
    with torch.inference_mode():
        for index, case in enumerate(cases):
            if args.memory:
                with mode(args.memory, args.optimization):
                    public_call(agent, case)
                row = {"name": case["name"], "peak_rss_mib": rss_mib()}
            else:
                # These untimed parity calls also warm both variants for this input shape.
                before, public_before = capture(agent, case, "baseline", args.optimization)
                after, public_after = capture(agent, case, "selected", args.optimization)
                row = {"name": case["name"], **parity(agent, before, after),
                       "max_public_delta": compare_public(public_before, public_after),
                       "forward_shapes": [r["shape"] for r in before],
                       "option_counts": [r["counts"] for r in before], "ms": {"baseline": [], "selected": []},
                       "cpu_ms": {"baseline": [], "selected": []}}
                for repeat in range(args.repeats):
                    order = ("baseline", "selected") if (repeat + index) % 2 == 0 else ("selected", "baseline")
                    for variant in order:
                        with mode(variant, args.optimization):
                            cpu_start = time.process_time_ns()
                            start = time.perf_counter_ns()
                            public_call(agent, case)
                            elapsed = (time.perf_counter_ns() - start) / 1e6
                            cpu_elapsed = (time.process_time_ns() - cpu_start) / 1e6
                        row["ms"][variant].append(elapsed)
                        row["cpu_ms"][variant].append(cpu_elapsed)
                row["median_ms"] = {k: statistics.median(v) for k, v in row["ms"].items()}
                row["median_cpu_ms"] = {k: statistics.median(v) for k, v in row["cpu_ms"].items()}
                row["latency_reduction_pct"] = 100 * (1 - row["median_ms"]["selected"] / row["median_ms"]["baseline"])
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
    assert versions == [(id(p), p._version) for p in agent.model.parameters()], "parameters changed"
    assert sha256(weights) == digest, "checkpoint changed"
    report["peak_rss_mib"] = rss_mib()
    report["parameters_unchanged"] = True
    if not args.memory:
        rows = report["cases"]
        report["summary"] = {"questions": sum(r["questions"] for r in rows),
            "argmax_flips": sum(r["argmax_flips"] for r in rows),
            **{key: max(r[key] for r in rows) for key in
               ("max_logit_delta", "max_action_logit_delta", "max_probability_delta", "max_public_delta")},
            "sum_median_ms": {v: sum(r["median_ms"][v] for r in rows) for v in ("baseline", "selected")}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report.get("summary", {"peak_rss_mib": report["peak_rss_mib"]})), flush=True)


if __name__ == "__main__":
    main()
