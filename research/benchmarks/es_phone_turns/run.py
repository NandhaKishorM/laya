"""Runs the frozen cases against one Laya checkpoint and archives every decision.

    python run.py --model convaiinnovations/laya-multilingual --rung es_criteria
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import metrics
import prompts

HERE = Path(__file__).parent


def load_cases(dataset):
    path = HERE / "data" / f"{dataset}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--rung", default="es_criteria", choices=sorted(prompts.RUNGS))
    parser.add_argument("--dataset", default="cases", choices=sorted(prompts.MENUS))
    parser.add_argument("--lang", default="es")
    parser.add_argument("--tag", default=None, help="suffix for the result file")
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()

    import laya
    import torch

    started = time.perf_counter()
    agent = laya.load(args.model, subfolder=args.subfolder, device="cpu",
                      calibration=args.calibration)
    load_ms = (time.perf_counter() - started) * 1000

    cases = load_cases(args.dataset)

    # One throwaway pass: the first forward pays for lazy initialisation and would
    # otherwise land in the latency of whichever case happens to come first.
    state, questions = prompts.build(args.rung, cases[0]["text"], args.dataset)
    agent.system_one(state, questions, lang=args.lang)

    records = []
    for case in cases:
        state, questions = prompts.build(args.rung, case["text"], args.dataset)
        t0 = time.perf_counter()
        result = agent.system_one(state, questions, lang=args.lang)
        latency = (time.perf_counter() - t0) * 1000
        answer = result["answers"]["accion"]
        records.append({
            "id": case["id"], "text": case["text"], "gold": case["gold"], "tags": case["tags"],
            "predicted": answer["choice"],
            "confidence": float(answer["answer_confidence"]),
            "probabilities": {k: float(v) for k, v in answer["probabilities"].items()},
            "latency_ms": latency,
        })

    report = {
        "meta": {
            "model": args.model, "subfolder": args.subfolder, "rung": args.rung,
            "dataset": args.dataset,
            "lang": args.lang, "calibration": args.calibration,
            "laya": getattr(laya, "__version__", None), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "device": "cpu",
            "machine": platform.machine(), "system": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "load_ms": load_ms,
        },
        "summary": metrics.summarize(records, prompts.labels(args.dataset)),
        "records": records,
    }

    name = args.tag or (
        args.model.rstrip("/").split("/")[-1] + (f"-{args.subfolder}" if args.subfolder else ""))
    out = HERE / "results" / f"{name}__{args.dataset}__{args.rung}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    s = report["summary"]
    print(f"{name} / {args.dataset} / {args.rung}: accuracy {s['accuracy']:.4f}  macro-F1 {s['macro_f1']:.4f}  "
          f"ECE {s['ece']:.4f}  wrong actions {s['wrong_actions']}  "
          f"p50 {s['latency_ms']['p50']:.0f} ms  (majority {s['majority_baseline']:.4f})")


if __name__ == "__main__":
    main()
