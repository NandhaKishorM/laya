"""Compare contiguous and length-grouped real-model batches on synthetic tickets.

This is a throughput experiment, not an accuracy benchmark. All timing includes
tokenization, collation, inference and decoding, but excludes model loading.
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import transformers

from laya import Agent
from laya import agent as agent_module

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this support ticket?",
             "criteria": {"billing": "payments, charges and refunds", "technical": "bugs and outages",
                          "sales": "pricing and new purchases"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need urgent help?"},
    "sentiment": {"type": "score", "instructions": "How satisfied is the customer?",
                  "criteria": ["very dissatisfied", "neutral", "very satisfied"]},
}
TICKETS = [
    "I was charged twice. Please refund the duplicate payment today.",
    "The app crashes at login and I cannot access my work. Please help immediately.",
    "Could you share pricing for twenty seats? We are planning next quarter's budget.",
    "Thank you for resolving my invoice problem. Everything works well now.",
    "I am unable to upload a document. There is no rush; I can try tomorrow.",
    "I need a quote for the enterprise plan. I am happy with the trial so far.",
]
DETAIL = "For context, I checked the account details and reviewed the earlier support conversation. "


def states_for(count, workload):
    rng = random.Random(912)
    states = []
    for i in range(count):
        repeats = rng.choice([0, 0, 0, 1, 2, 4, 8, 20]) if workload == "mixed" else 2
        states.append({"ticket": TICKETS[i % len(TICKETS)], "reference": i,
                       "context": DETAIL * repeats})
    rng.shuffle(states)
    return states


def compare(before, after):
    assert len(before) == len(after)
    max_delta = 0.0
    flips = 0
    usage_mismatches = 0
    def numeric(a, b):
        nonlocal max_delta
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                numeric(a[key], b[key])
        elif isinstance(a, (int, float)):
            max_delta = max(max_delta, abs(a - b))
        elif a != b:
            assert isinstance(a, str) and isinstance(b, str)
    for a, b in zip(before, after):
        usage_mismatches += a["usage"] != b["usage"]
        for qid, ans in a["answers"].items():
            new = b["answers"][qid]
            flips += (ans["action"]["act_probability"] >= 0.5) != (new["action"]["act_probability"] >= 0.5)
            if ans["type"] == "choice":
                flips += ans["choice"] != new["choice"]
            elif ans["type"] == "noul":
                flips += (ans["noul"] >= 0.5) != (new["noul"] >= 0.5)
            else:
                flips += max(ans["probabilities"], key=ans["probabilities"].get) != max(
                    new["probabilities"], key=new["probabilities"].get)
            numeric(ans, new)
    return {"decision_flips": flips, "max_numeric_delta": max_delta,
            "usage_mismatches": usage_mismatches, "results_equal": before == after}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--questions", type=int, default=1, choices=[1, 3])
    p.add_argument("--workload", choices=["mixed", "uniform"], default="mixed")
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", default="1e28ac20c0896b1c37a744cd11f740eb98f8b178")
    p.add_argument("--memory-mode", choices=["original", "grouped"],
                   help="Profile one mode in a fresh process (requires psutil), then exit.")
    args = p.parse_args()
    if min(args.count, args.batch_size, args.rounds, args.threads) < 1:
        p.error("count, batch size, rounds and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    ag = Agent(args.model, device=args.device)
    # Execute the actual upstream method, so added default-path overhead is included.
    source = subprocess.check_output(["git", "show", args.baseline + ":laya/agent.py"], encoding="utf-8")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Agent")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "predict_batch")
    scope = dict(vars(agent_module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<upstream predict_batch>", "exec"), scope)  # noqa: S102
    baseline = scope["predict_batch"].__get__(ag)
    questions = dict(list(QUESTIONS.items())[:args.questions])
    states = states_for(args.count, args.workload)
    ag.predict_batch(states[:args.batch_size], questions, batch_size=args.batch_size)
    def require_device():
        if ag.device.type != args.device:
            raise RuntimeError("Device fallback occurred; this benchmark run is invalid")
    require_device()
    if args.memory_mode:
        # Run each mode in a fresh process: allocator retention from a previous mode
        # otherwise makes RSS peaks incomparable. This excludes model-load transients.
        import psutil
        process = psutil.Process()
        samples = [process.memory_info().rss]
        stop = threading.Event()
        def sample_rss():
            while not stop.wait(0.01):
                samples.append(process.memory_info().rss)
        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        try:
            if args.memory_mode == "grouped":
                results = ag.predict_batch(states, questions, batch_size=args.batch_size,
                                           sort_by_length=True)
            else:
                results = baseline(states, questions, batch_size=args.batch_size)
            require_device()
            samples.append(process.memory_info().rss)
        finally:
            stop.set()
            sampler.join()
        report = {"args": vars(args), "start_rss_bytes": samples[0],
                  "sampled_peak_rss_bytes": max(samples), "sample_interval_ms": 10,
                  "samples": len(samples), "results": len(results),
                  "results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
        return 0
    original_forward = ag._forward
    stats = {}
    def measured_forward(batch):
        stats["padded_slots"] += batch["input_ids"].numel()
        stats["max_batch_slots"] = max(stats["max_batch_slots"], batch["input_ids"].numel())
        start = time.perf_counter()
        out = original_forward(batch)
        stats["forward_seconds"].append(time.perf_counter() - start)
        completed = len(stats["forward_seconds"])
        if completed % 25 == 0:
            print(json.dumps({"forward_batches": completed, "sorted": sorted_,
                              "elapsed_seconds": time.perf_counter() - pass_started}), flush=True)
        return out
    ag._forward = measured_forward
    times = {False: [], True: []}
    runs = []
    gate_failed = False
    for r in range(args.rounds):
        outputs = {}
        for sorted_ in ([False, True] if r % 2 == 0 else [True, False]):
            stats = {"padded_slots": 0, "max_batch_slots": 0, "forward_seconds": []}
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            pass_started = start = time.perf_counter()
            if sorted_:
                outputs[sorted_] = ag.predict_batch(states, questions, batch_size=args.batch_size,
                                                   sort_by_length=True)
            else:
                outputs[sorted_] = baseline(states, questions, batch_size=args.batch_size)
            seconds = time.perf_counter() - start
            require_device()
            if args.device == "cuda":
                stats["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            times[sorted_].append(seconds)
            run = {"round": r, "sorted": sorted_, "seconds": seconds, **stats}
            runs.append(run)
            print(json.dumps({k: v for k, v in run.items() if k != "forward_seconds"}), flush=True)
        parity = compare(outputs[False], outputs[True])
        print(json.dumps(parity), flush=True)
        if parity["decision_flips"] or parity["usage_mismatches"] or parity["max_numeric_delta"] > 0.001:
            print("CONSISTENCY GATE FAILED", flush=True)
            gate_failed = True
            break
    report = {"args": vars(args), "torch": torch.__version__, "device": str(ag.device),
              "transformers": transformers.__version__, "python": platform.python_version(),
              "platform": platform.platform(), "processor": platform.processor(),
              "states_sha256": hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest(),
              "agent_source_sha256": hashlib.sha256(Path(agent_module.__file__).read_bytes()).hexdigest(),
              "runs": runs, "parity": parity, "consistency_gate_passed": not gate_failed,
              "speedup": statistics.median(times[False]) / statistics.median(times[True])}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}), flush=True)
    return int(gate_failed)


if __name__ == "__main__":
    sys.exit(main())
