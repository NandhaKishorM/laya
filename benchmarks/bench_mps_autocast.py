#!/usr/bin/env python
"""MPS fp16 autocast against fp32, request by request, on Apple silicon.

    python benchmarks/bench_mps_autocast.py --checkpoint english
    python benchmarks/bench_mps_autocast.py --checkpoint multilingual --out results.json

`Agent` autocasts an MPS forward to fp16 at or above `mps_amp_min_rows` question rows
(default 5, `LAYA_MPS_AMP_MIN_ROWS`). This script measures what that gate decides, for one
checkpoint, offline after the checkpoint is cached:

* latency: every request runs once in fp32 and once in fp16 on the same loaded agent, back
  to back, with the order alternating. The two modes then see the same machine load, so the
  paired difference holds on a machine that is not idle. Reported per (state length, rows):
  median of each mode, and the median of the per-request difference.
* answers: for the same requests, the largest and the 95th-percentile probability change
  between the two modes, and how many decisions changed (choice: the option; score: the
  most likely level; noul: the side of 0.5).

The mode is switched through `agent.mps_amp_min_rows` (1 = fp16 for every request, above the
row count = fp32), so nothing but the gate differs between the two runs of a request.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

BUNDLE = "convaiinnovations/laya"
QUESTIONS = [
    {"type": "choice", "instructions": "Which department should handle this request?",
     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                  "logistics": "delivery tracking and delivery problems", "other": "everything else"}},
    {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
    {"type": "score", "instructions": "How urgent is this request?",
     "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    {"type": "noul", "instructions": "Is the user angry?"},
    {"type": "choice", "instructions": "What should happen next?",
     "criteria": {"reply": "answer the user directly", "escalate": "hand over to a human",
                  "wait": "nothing can be done yet"}},
    {"type": "noul", "instructions": "Does the user ask to speak to a human?"},
    {"type": "noul", "instructions": "Is a delivery late?"},
]
SUBJECTS = ["my order #4411", "the March invoice", "the export feature", "my account login"]
PROBLEMS = [
    "was charged twice and I want the duplicate back",
    "has not arrived although tracking says it was delivered last week",
    "stopped working this morning and the whole team is blocked",
    "may or may not be fine, I honestly cannot tell from the page",
    "is wrong, but before anything else I would like to know my options",
    "was fine, thank you, though the refund you promised has not shown up yet",
]
SHORT = [{"subject": "About " + s, "body": "Hello, %s %s." % (s, p)} for s in SUBJECTS for p in PROBLEMS]
# A long state is a thread: the request plus the five that follow it in the list.
LONG = [{"thread": [SHORT[(i + k) % len(SHORT)] for k in range(6)]} for i in range(len(SHORT))]


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return platform.processor()


def call(agent, state, questions, min_rows):
    agent.mps_amp_min_rows = min_rows
    t0 = time.perf_counter()
    out = agent.system_one(state, questions)
    return (time.perf_counter() - t0) * 1000, out


def compare(a, b):
    """(probability changes, decisions changed) between two answers to the same questions."""
    deltas, changed = [], 0
    for qid, x in a.items():
        y = b[qid]
        if x["type"] == "noul":
            deltas.append(abs(x["noul"] - y["noul"]))
            changed += (x["noul"] >= 0.5) != (y["noul"] >= 0.5)
        else:
            deltas += [abs(p - y["probabilities"][k]) for k, p in x["probabilities"].items()]
            changed += max(x["probabilities"], key=x["probabilities"].get) != max(
                y["probabilities"], key=y["probabilities"].get)
    return deltas, changed


def cell(agent, states, rows, repeats, warmup):
    questions = {"q%d" % i: q for i, q in enumerate(QUESTIONS[:rows])}
    fp32, fp16 = rows + 1, 1
    for state in states[:warmup]:  # both paths, on this shape
        call(agent, state, questions, fp32)
        call(agent, state, questions, fp16)
    ms32, ms16, deltas, changed, decisions, tokens = [], [], [], 0, 0, 0
    for rep in range(repeats):
        for i, state in enumerate(states):
            first, second = (fp32, fp16) if (i + rep) % 2 else (fp16, fp32)
            results = {first: call(agent, state, questions, first), second: call(agent, state, questions, second)}
            ms32.append(results[fp32][0])
            ms16.append(results[fp16][0])
            tokens = max(tokens, results[fp32][1]["usage"]["input_tokens"])
            if rep == 0:
                d, c = compare(results[fp32][1]["answers"], results[fp16][1]["answers"])
                deltas += d
                changed += c
                decisions += rows
    deltas.sort()
    return {"rows": rows, "pairs": len(ms32), "max_input_tokens": tokens,
            "fp32_ms": round(statistics.median(ms32), 1), "fp16_ms": round(statistics.median(ms16), 1),
            "paired_ms": round(statistics.median(b - a for a, b in zip(ms32, ms16)), 1),
            "max_delta": round(deltas[-1], 4), "p95_delta": round(deltas[int(0.95 * (len(deltas) - 1))], 4),
            "decisions_changed": changed, "decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="english", choices=["english", "multilingual", "typed-decisions"])
    parser.add_argument("--rows", default="1,2,4,5,6,8", help="question rows per request, comma separated")
    parser.add_argument("--repeats", type=int, default=3, help="passes over the %d states" % len(SHORT))
    parser.add_argument("--warmup", type=int, default=3, help="states run in both modes before timing")
    parser.add_argument("--out", help="write the results as JSON")
    args = parser.parse_args()

    import torch
    import transformers

    import laya

    if not torch.backends.mps.is_available():
        raise SystemExit("this benchmark needs an MPS device")
    agent = laya.load(BUNDLE, device="mps", subfolder=None if args.checkpoint == "english" else args.checkpoint)
    default_rows = agent.mps_amp_min_rows
    result = {"checkpoint": args.checkpoint, "chip": chip(), "os": "macOS " + platform.mac_ver()[0],
              "torch": torch.__version__, "transformers": transformers.__version__, "laya": laya.__version__,
              "default_mps_amp_min_rows": default_rows, "load_1m": round(os.getloadavg()[0], 1), "cells": []}
    print("%s on %s, laya %s, torch %s (1-min load %.1f)"
          % (args.checkpoint, result["chip"], laya.__version__, torch.__version__, result["load_1m"]))
    print("| state | rows | input tokens | fp32 ms | fp16 ms | fp16 - fp32 (paired) | max delta | p95 delta "
          "| decisions changed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, states in (("short", SHORT), ("long", LONG)):
        for rows in [int(r) for r in args.rows.split(",")]:
            row = {"state": name, **cell(agent, states, rows, args.repeats, args.warmup)}
            result["cells"].append(row)
            print("| %s | %d | %d | %.1f | %.1f | %+.1f | %.4f | %.4f | %d / %d |"
                  % (name, rows, row["max_input_tokens"], row["fp32_ms"], row["fp16_ms"], row["paired_ms"],
                     row["max_delta"], row["p95_delta"], row["decisions_changed"], row["decisions"]))
    agent.mps_amp_min_rows = default_rows
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
