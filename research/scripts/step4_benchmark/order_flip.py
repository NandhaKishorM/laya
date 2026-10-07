"""Measure the 20-option order-flip rate on `massive_intent.en` for a checkpoint.

This is the *other* evidence surface of Step 4, and it is measurement-only on purpose. The
original issue names `massive_intent.en` at 20 options as the order-stability surface; the
typed-decisions benchmark is 4-option `choice` and must not be reused for a 20-option claim.

GuilhermeFusari already posted seed-0/1 numbers for this surface, produced by `laya.train` at
`8a6e132` (soft-CE, 4 epochs, effective batch 64, `max_len` 512 / `head_max_len` 256, shuffle vs
no shuffle). The option-order code path has not changed semantically since then (only the
`parallel` layout argument was threaded through, and `shuffle_options` still defaults to off), so
Step 4 **references those numbers instead of starting a second training campaign**. This script
exists so the referenced number can be reproduced inference-only on the exact checkpoint a
reviewer hands over, without retraining.

Construction matches `research/scripts/build_benchmark_nb.py`: each question is the gold intent
plus 19 distractors drawn from the label space, in random order (seed 13). A case is answered
with its options in the original order and in `--permutations` random permutations, in one call
each; the flip rate is the share of permutations whose answer differs from the original order.

Usage:
    python order_flip.py --checkpoint <dir-or-hub-id> --out flip.json
    python order_flip.py --checkpoint convaiinnovations/laya --per-lang 200 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def build_cases(n: int, n_opts: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from datasets import load_dataset

    d = load_dataset("mteb/amazon_massive_intent", "en", split="test")
    labels = sorted(set(d["label_text"]))
    rng = random.Random(seed)
    cases, golds = [], []
    for r in list(d)[:n]:
        gold = r["label_text"]
        pool = [x for x in labels if x != gold]
        keys = [gold] + rng.sample(pool, min(n_opts - 1, len(pool)))
        rng.shuffle(keys)
        crit = {k: k.replace("_", " ").replace(".", ": ") for k in keys}
        cases.append({"utterance": r["text"]})
        golds.append({"label": {"type": "choice",
                                "instructions": "What is the user asking for in `utterance`?",
                                "criteria": crit}, "gold_label": gold})
    return cases, golds


def permute(qdef: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    keys = list(qdef["criteria"].keys())
    order = list(range(len(keys)))
    rng.shuffle(order)
    return {**qdef, "criteria": {keys[i]: qdef["criteria"][keys[i]] for i in order}}


def measure(checkpoint: str, n: int, n_opts: int, permutations: int, seed: int,
            device: str) -> Dict[str, Any]:
    import laya

    agent = laya.load(checkpoint, device=device)
    cases, golds = build_cases(n, n_opts, seed)
    rng = random.Random(seed + 1)
    flips = compared = 0
    for case, gold in zip(cases, golds):
        base_q = gold["label"]
        base_answer = agent.predict(case, {"label": base_q})["answers"]["label"]["choice"]
        for _ in range(permutations):
            perm_q = permute(base_q, rng)
            perm_answer = agent.predict(case, {"label": perm_q})["answers"]["label"]["choice"]
            compared += 1
            if perm_answer != base_answer:
                flips += 1
    return {
        "checkpoint": checkpoint,
        "suite": "massive_intent.en",
        "n_options": n_opts,
        "n_cases": len(cases),
        "permutations_per_case": permutations,
        "n_comparisons": compared,
        "flip_count": flips,
        "flip_rate": round(flips / max(1, compared), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--per-lang", type=int, default=300, help="cases to sample")
    parser.add_argument("--n-options", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = measure(args.checkpoint, args.per_lang, args.n_options, args.permutations,
                     args.seed, args.device)
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
