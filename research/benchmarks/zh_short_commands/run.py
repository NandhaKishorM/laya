"""Run the Chinese short-command benchmark against one Laya checkpoint.

    PYTHONPATH=. python research/benchmarks/zh_short_commands/run.py \
        --checkpoint convaiinnovations/laya --subfolder multilingual \
        --out research/benchmarks/zh_short_commands/results/v1/laya-multilingual

Writes ``<out>/report.json`` with the same four parts as ``research/eval/laya_eval.py``:
``config``, ``report``, ``summary`` and ``cases``. Every number in ``report`` can be
re-derived from ``cases`` alone, with no model and no network.

``--stub`` scores with a fixed pseudo-random vector instead of loading a checkpoint,
which exercises the whole pipeline offline; ``tests/test_benchmark.py`` uses it.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prompts  # noqa: E402

from research.eval.laya_eval import (  # noqa: E402
    ece, macro_f1, score_cases, softmax_t, summarise, temperature_for,
)

CASES_PATH = os.path.join(HERE, "data", "cases.jsonl")


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_cases(path: str = CASES_PATH):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    unknown = sorted({c["gold"] for c in cases} - set(prompts.LABELS))
    if unknown:
        raise ValueError("gold labels outside the policy: %r" % unknown)
    return cases


def stub_logits(cases):
    """Deterministic pseudo-random marker logits, so the pipeline can run offline."""
    import numpy as np
    rng = np.random.RandomState(13)
    out = []
    for _state, questions in cases:
        for _qid, qdef in questions.items():
            k = len(qdef.get("criteria") or (0, 1))
            out.append(rng.randn(k))
    return out


def choice_records(config, cases, logits, agent, unclamped):
    """One record per case, in the shape laya_eval.py emits."""
    import numpy as np
    from laya.common import QTYPES

    gold_index = [prompts.LABELS.index(c["gold"]) for c in cases]
    confidences, corrects, preds, records = [], [], [], []
    for i, z in enumerate(logits):
        k = len(z)
        temperature = 1.0 if agent is None else temperature_for(
            agent, QTYPES["choice"], k, unclamped)
        probs = softmax_t(z, temperature)
        pred = int(np.argmax(probs))
        correct = int(pred == gold_index[i])
        confidences.append(float(probs.max()))
        corrects.append(float(correct))
        preds.append(pred)
        records.append({
            "case_id": cases[i]["id"],
            "config": config,
            "text": cases[i]["text"],
            "state": prompts.state_for(config, cases[i]["text"]),
            "family": cases[i]["family"],
            "instructions": prompts.choice_question(config)["intent"]["instructions"],
            "options": list(prompts.CHOICE_CRITERIA),
            "gold_index": gold_index[i],
            "gold_label": cases[i]["gold"],
            "pred_index": pred,
            "pred_label": prompts.LABELS[pred],
            "probability": round(float(probs[pred]), 6),
            "p_gold": round(float(probs[gold_index[i]]), 6),
            "confidence": round(float(probs.max()), 6),
            "correct": correct,
            "temperature": round(float(temperature), 6),
        })
    report = summarise(confidences, corrects, gold_index, preds)
    report["temperature"] = records[0]["temperature"]
    report["mean_confidence"] = round(
        sum(confidences) / len(confidences), 6) if confidences else None
    # Same population std the noul block records, so both shapes read the same way.
    report["std_confidence"] = round(float(np.std(confidences)), 6) if confidences else None
    return {"report": report, "cases": records}


def noul_records(config, cases, logits, agent, unclamped):
    """One record per (case, dimension); accuracy is per dimension, not top-1."""
    import numpy as np
    from laya.common import QTYPES

    dims = prompts.NOUL_DIMENSIONS
    questions = prompts.noul_questions(config)
    records, per_dim = [], {d: [] for d in dims}
    for i, case in enumerate(cases):
        for j, dim in enumerate(dims):
            z = logits[i * len(dims) + j]
            temperature = 1.0 if agent is None else temperature_for(
                agent, QTYPES["noul"], len(z), unclamped)
            probs = softmax_t(z, temperature)
            p_true = float(probs[1])          # noul scores [false, true] in that order
            # Store the rounded value and derive everything else from it, so the record
            # stays self-consistent: confidence must equal max(p_true, 1 - p_true) of
            # the number that was actually written down, not of the full-precision one.
            p_true = round(p_true, 6)
            gold_bool = bool(prompts.NOUL_GOLD[dim](case["gold"]))
            pred_bool = p_true >= 0.5
            correct = int(pred_bool == gold_bool)
            per_dim[dim].append(correct)
            records.append({
                "case_id": case["id"],
                "config": config,
                "text": case["text"],
                "state": prompts.state_for(config, case["text"]),
                "family": case["family"],
                "dimension": dim,
                "instructions": questions[dim]["instructions"],
                "options": ["false", "true"],
                "gold_bool": gold_bool,
                "gold_label": case["gold"],
                "pred_bool": pred_bool,
                "p_true": p_true,
                "confidence": round(max(p_true, 1.0 - p_true), 6),
                "correct": correct,
                "temperature": round(float(temperature), 6),
            })
    confidences = [r["confidence"] for r in records]
    report = {
        "n_cases": len(cases),
        "n_dimensions": len(dims),
        "accuracy_by_dimension": {
            d: round(sum(v) / len(v), 6) for d, v in per_dim.items() if v
        },
        "accuracy": round(sum(r["correct"] for r in records) / len(records), 6)
        if records else None,
        "mean_confidence": round(sum(confidences) / len(confidences), 6)
        if confidences else None,
        "std_confidence": round(float(np.std(confidences)), 6) if confidences else None,
        "temperature": records[0]["temperature"] if records else None,
    }
    return {"report": report, "cases": records}


def checkpoint_provenance(args, agent):
    """Best-effort record of what was actually scored: repo, snapshot, weight hash."""
    info = {"checkpoint": args.checkpoint, "subfolder": args.subfolder,
            "device": "stub" if args.stub else str(getattr(agent, "device", None)),
            "stub": bool(args.stub)}
    if agent is None:
        return info
    try:
        import torch
        import transformers
        import laya
        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["laya"] = laya.__version__
    except Exception:                              # pragma: no cover - provenance only
        pass
    subfolder = (args.subfolder + "/") if args.subfolder else ""
    try:
        if os.path.isdir(args.checkpoint):
            weight = os.path.join(args.checkpoint, subfolder, "model.safetensors")
            if os.path.exists(weight):
                info["weight_sha256"] = sha256_file(weight)
                info["weight_bytes"] = os.path.getsize(weight)
            else:
                info["provenance_error"] = "no model.safetensors under %s" % args.checkpoint
        else:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(args.checkpoint, allow_patterns=["*.safetensors"],
                                     local_files_only=True)
            info["snapshot"] = os.path.basename(snap)
            weight = os.path.join(snap, subfolder + "model.safetensors")
            if os.path.exists(weight):
                info["weight_sha256"] = sha256_file(weight)
                info["weight_bytes"] = os.path.getsize(weight)
    except Exception as exc:                       # pragma: no cover - provenance only
        info["provenance_error"] = "%s: %s" % (type(exc).__name__, exc)
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="zh-short-commands")
    parser.add_argument("--checkpoint", default="convaiinnovations/laya",
                        help="checkpoint repo id or local path")
    parser.add_argument("--subfolder", default="multilingual")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps (default: auto)")
    parser.add_argument("--configs", default=",".join(prompts.CONFIGS),
                        help="comma list of configs to run")
    parser.add_argument("--out", default=None, help="output directory for report.json")
    parser.add_argument("--unclamped", action="store_true",
                        help="score with the checkpoint's raw bucket temperatures")
    parser.add_argument("--stub", action="store_true",
                        help="skip the checkpoint and score with a fixed pseudo-random vector")
    args = parser.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for config in configs:
        if config not in prompts.CONFIGS:
            parser.error("unknown config %r; known: %s" % (config, ", ".join(prompts.CONFIGS)))

    cases = load_cases()
    print("cases: %d  configs: %s" % (len(cases), ", ".join(configs)))

    agent = None
    if not args.stub:
        import laya
        agent = laya.load(args.checkpoint, subfolder=args.subfolder, device=args.device)

    report, summary = {}, {}
    for config in configs:
        pairs = prompts.cases_for(config, [c["text"] for c in cases])
        logits = stub_logits(pairs) if args.stub else score_cases(agent, pairs)
        if config in prompts.NOUL_CONFIGS:
            out = noul_records(config, cases, logits, agent, args.unclamped)
        else:
            out = choice_records(config, cases, logits, agent, args.unclamped)
        report[config] = out["report"]
        summary[config] = {"accuracy": out["report"].get("accuracy"),
                           "mean_confidence": out["report"].get("mean_confidence")}
        line = "  %-18s acc=%s" % (config, out["report"].get("accuracy"))
        extras = []
        if out["report"].get("macro_f1") is not None:
            extras.append("macro_f1=%s" % out["report"]["macro_f1"])
        if out["report"].get("ece") is not None:
            extras.append("ece=%s" % out["report"]["ece"])
        if out["report"].get("std_confidence") is not None:
            extras.append("std_conf=%s" % out["report"]["std_confidence"])
        extras.append("mean_conf=%s" % out["report"].get("mean_confidence"))
        print("%s  %s" % (line, " ".join(extras)))
        report[config]["cases"] = out["cases"]

    document = {
        "config": {
            "benchmark": "zh_short_commands",
            "cases_file": os.path.relpath(CASES_PATH, ROOT).replace(os.sep, "/"),
            "cases_sha256": sha256_file(CASES_PATH),
            "prompts_sha256": sha256_file(os.path.join(HERE, "prompts.py")),
            "configs": configs,
            "unclamped": bool(args.unclamped),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "checkpoint": checkpoint_provenance(args, agent),
        },
        "report": {c: {k: v for k, v in report[c].items() if k != "cases"} for c in configs},
        "summary": summary,
        "cases": [record for config in configs for record in report[config]["cases"]],
    }

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        target = os.path.join(args.out, "report.json")
        # newline="\n" so the archive is the same bytes on Windows as on Linux; a report
        # written with CRLF makes every later run on another platform look like a rewrite.
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print("wrote %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
