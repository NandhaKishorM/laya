"""Evaluate a fine-tuned checkpoint on the typed-decisions test split.

Mirrors cell 14 of the published notebook, restricted to the three metrics the Step-4 contract
names: **accuracy**, **ECE** and **Brier**. The 20-option order-flip surface is deliberately not
here: it belongs to `massive_intent.en`, not typed-decisions (see README).

Calibration is not byte-comparable across contracts. `laya.train` fits temperatures with
`fit_temperature_map` (per type, and per option-count bucket above a floor) under the runtime
clamp `[0.5, 5.0]`; the old notebook fitted one LBFGS temperature per type under `[0.1, 10]`.
So ECE and Brier support either path, one per invocation:

- ``--temperatures shipped`` (default) — the checkpoint's own fitted `temperature` and
  `temperature_by_options`, which is what the notebook's number also used;
- ``--temperatures raw`` — temperature 1.0 everywhere, isolating the training from the
  calibration contract.

Each run evaluates exactly one of the two and records which in the output; the committed Gate D
table is the shipped path.

Accuracy is unaffected by temperature (it only rescales the logits), so it is the clean
reproduction signal.

Usage:
    python evaluate.py --checkpoint runs/step4/rlcd-seed0 \
        --test-meta data/typed_decisions/test_meta.json \
        --out runs/step4/rlcd-seed0/eval_shipped.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (  # noqa: E402
    accuracy,
    brier_score,
    expected_calibration_error,
    soft_accuracy,
)


def _normalise(values: List[float]) -> List[float]:
    total = sum(values)
    if total > 0:
        return [v / total for v in values]
    return values


def evaluate_checkpoint(checkpoint_dir: str, test_meta: List[Dict[str, Any]],
                        device: str = "auto", temperatures: str = "shipped") -> Dict[str, Any]:
    import laya

    agent = laya.load(checkpoint_dir, device=device)
    if temperatures == "raw":
        agent.temperature = [1.0, 1.0, 1.0]
        agent.temperature_by_options = {}
    elif temperatures != "shipped":
        raise ValueError("temperatures must be 'shipped' or 'raw', got %r" % (temperatures,))

    corrects: List[float] = []
    confidences: List[float] = []
    briers: List[float] = []
    softs: List[float] = []
    per_type: Dict[str, List[float]] = {"choice": [], "noul": [], "score": []}
    score_maes: List[float] = []
    within_one: List[float] = []

    for i, case in enumerate(test_meta):
        state = case["state"]
        questions = case["questions"]
        gold = case["gold"]
        answers = agent.predict(state, questions)["answers"]

        for qid, qdef in questions.items():
            if qid not in gold:
                continue
            g = gold[qid]
            qtype = qdef["type"]
            p_ans = answers[qid]

            if qtype == "choice":
                keys = list(qdef["criteria"].keys())
                pred = p_ans["choice"]
                gold_label = str(g["label"])
                is_corr = float(pred == gold_label)

                p_probs = _normalise([p_ans["probabilities"].get(k, 1e-6) for k in keys])
                g_probs = _normalise([g["probabilities"].get(k, 1e-6) for k in keys])

                corrects.append(is_corr)
                confidences.append(max(p_probs))
                briers.append(brier_score(p_probs, g_probs))
                softs.append(soft_accuracy(p_probs, g_probs))
                per_type["choice"].append(is_corr)

            elif qtype == "noul":
                p_val = float(p_ans["noul"])
                g_val = float(g.get("noul", g.get("probabilities", {}).get("true", 0.5)))
                gold_label = str(g["label"]).lower()
                pred_label = "true" if p_val >= 0.5 else "false"
                is_corr = float(pred_label == gold_label)

                p_dist = [1.0 - p_val, p_val]
                g_dist = [1.0 - g_val, g_val]

                corrects.append(is_corr)
                confidences.append(max(p_val, 1.0 - p_val))
                briers.append(brier_score(p_dist, g_dist))
                softs.append(soft_accuracy(p_dist, g_dist))
                per_type["noul"].append(is_corr)

            elif qtype == "score":
                p_score = float(p_ans["score"])
                g_score = float(g.get("score", 0.0))
                score_maes.append(abs(p_score - g_score))
                within_one.append(float(abs(p_score - g_score) <= 1.0))

                n_levels = len(qdef.get("criteria", []))
                p_probs = [p_ans["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
                if sum(p_probs) > 0:
                    p_probs = _normalise(p_probs)
                    p_lvl = int(max(range(len(p_probs)), key=lambda j: p_probs[j]))
                    confidences.append(max(p_probs))
                else:
                    p_lvl = int(round(p_score))
                    confidences.append(0.5)

                g_lvl = int(g.get("label", int(round(g_score))))
                is_corr = float(p_lvl == g_lvl)
                corrects.append(is_corr)
                per_type["score"].append(is_corr)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(test_meta)} cases", flush=True)

    result = {
        "checkpoint": checkpoint_dir,
        "temperatures": temperatures,
        "n_cases": len(test_meta),
        "n_decisions": len(corrects),
        "accuracy": round(accuracy(corrects), 4),
        "ece": round(expected_calibration_error(confidences, corrects), 4),
        "brier": round(float(sum(briers) / len(briers)) if briers else float("nan"), 4),
        "soft_accuracy": round(float(sum(softs) / len(softs)) if softs else float("nan"), 4),
        "score_mae": round(float(sum(score_maes) / len(score_maes)) if score_maes else float("nan"), 4),
        "within_one_level": round(float(sum(within_one) / len(within_one)) if within_one else float("nan"), 4),
        "per_type_accuracy": {k: round(accuracy(v), 4) for k, v in per_type.items() if v},
    }
    # `brier`/`soft_accuracy`/`ece` mix question types; record the counts they rest on.
    result["n_brier_decisions"] = len(briers)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-meta", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperatures", choices=["shipped", "raw"], default="shipped")
    parser.add_argument("--out", default=None,
                        help="Output JSON (default: <checkpoint>/eval_<temperatures>.json)")
    args = parser.parse_args()

    with open(args.test_meta, encoding="utf-8") as f:
        test_meta = json.load(f)
    print(f"loaded {len(test_meta)} test cases", flush=True)

    result = evaluate_checkpoint(args.checkpoint, test_meta, device=args.device,
                                 temperatures=args.temperatures)

    print("\n" + "=" * 60)
    print(f"evaluation: {args.checkpoint} ({args.temperatures} temperatures)")
    print("=" * 60)
    print(f"  accuracy:     {result['accuracy']:.4f}")
    print(f"  ECE:          {result['ece']:.4f}")
    print(f"  Brier:        {result['brier']:.4f}")
    print(f"  soft acc:     {result['soft_accuracy']:.4f}")
    print(f"  per type:     {result['per_type_accuracy']}")
    print("=" * 60)

    out_path = args.out or os.path.join(args.checkpoint, f"eval_{args.temperatures}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
