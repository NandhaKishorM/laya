"""laya-eval: an independent, reproducible evaluation harness for Laya checkpoints.

Why this exists
---------------
The repository's own benchmark scripts are research code: they load everything,
run all parts, and print tables. There is no small, reproducible harness a third
party can point at a Laya checkpoint to get a per-language accuracy and ECE
report plus machine-readable per-case output. This is that harness.

Provenance
----------
The prompt format, sampling and metrics follow ``research/scripts/bench_local.py``
so that results are comparable with the published tables:

* dataset      ``mteb/amazon_massive_intent``, split ``test``
* sampling     first ``--per-lang`` rows, ``random.Random(SEED)`` fresh per language
* options      ``--n-opts`` labels: the gold one plus ``rng.sample`` of the rest
* prompt       "What is the user asking for in `utterance`?"
* rendering    option keys with ``_`` -> `` `` and ``.`` -> ``: ``
* metrics      accuracy, macro-F1, ECE (15 bins) on the temperature-scaled softmax

Verified against ``research/results/cpu_51_language_sweep.json``: with the raw
pre-clamp temperatures, ``en`` reproduces n=100 / accuracy 0.82 / macro_f1 0.7876
/ ece 0.1789 / mean_confidence 0.9989 exactly.

Usage
-----
    python -m laya_eval --model convaiinnovations/laya --langs en,de,ro
    python -m laya_eval --model ./local-checkpoint --langs all --out report.json
    python -m laya_eval --model convaiinnovations/laya --subfolder multilingual --langs all

Output is a JSON document with a ``config`` block, a per-language ``report`` and
a ``cases`` list carrying every individual decision, so a number in the report can
be re-derived without re-running the model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Running this file directly puts research/eval/ on sys.path, not the repo root, so
# `import laya` would fail. Allow both `python research/eval/laya_eval.py` and
# `python -m research.eval.laya_eval` to find the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SEED = 13
N_OPTS = 20
PER_LANG = 100
INSTRUCTIONS = "What is the user asking for in `utterance`?"
DATASET = "mteb/amazon_massive_intent"
ECE_BINS = 15


# --------------------------------------------------------------------- sampling
def render_label(key: str) -> str:
    """Option text for one intent label, matching bench_local.py:176."""
    return key.replace("_", " ").replace(".", ": ")


def build_case(text: str, gold: str, pool: Sequence[str], rng: random.Random,
               n_opts: int) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """One case: a state, a 20-option choice question, and the gold option index."""
    keys = [gold] + rng.sample(list(pool), min(n_opts - 1, len(pool)))
    rng.shuffle(keys)
    state = {"utterance": text}
    question = {
        "intent": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {k: render_label(k) for k in keys},
        }
    }
    return state, question, keys.index(gold)


def build_suite(rows: Sequence[Dict[str, Any]], labels: Sequence[str],
                per_lang: int, n_opts: int, seed: int = SEED):
    """Deterministic suite for one language. Returns (cases, gold, option_keys)."""
    labels = sorted(labels)
    rng = random.Random(seed)                      # fresh per language, as upstream
    cases, gold, option_keys = [], [], []
    for row in list(rows)[:per_lang]:
        pool = [x for x in labels if x != row["label_text"]]
        state, question, gold_idx = build_case(row["text"], row["label_text"],
                                               pool, rng, n_opts)
        keys = list(question["intent"]["criteria"])
        cases.append((state, question))
        gold.append(gold_idx)
        option_keys.append(keys)
    return cases, gold, option_keys


def load_language(lang: str, split: str = "test"):
    """Load one language config. Raises with a readable message if unavailable."""
    from datasets import load_dataset
    try:
        ds = load_dataset(DATASET, lang, split=split)
    except Exception as exc:                       # pragma: no cover - network path
        raise RuntimeError(
            "could not load %s config %r: %s" % (DATASET, lang, exc)
        ) from exc
    return [{"text": r["text"], "label_text": r["label_text"]} for r in ds]


def available_languages() -> List[str]:
    """Language configs the dataset exposes, excluding the aggregate 'default'."""
    from datasets import get_dataset_config_names
    names = get_dataset_config_names(DATASET)
    return sorted(n for n in names if n != "default")


# ---------------------------------------------------------------------- scoring
def score_cases(agent, cases) -> List[Any]:
    """Raw marker logits per case, from one collated forward pass.

    Mirrors bench_local.py:54-97: build every sequence, collate, run the model
    once, and slice each row to its own marker count. No temperature is applied
    here so the caller can score under more than one regime from one pass.
    """
    import torch
    from laya.common import QTYPES, build_sequence, collate_items, render_options

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = []
    for state, questions in cases:
        for _qid, qdef in questions.items():
            q = {"t": qdef["type"],
                 "ins": qdef["instructions"],
                 "crit": qdef.get("criteria")}
            ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("marker/option count mismatch; question exceeds head_max_len")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    with torch.no_grad():
        logits, _ = agent.model(
            batch["input_ids"].to(agent.device),
            batch["attention_mask"].to(agent.device),
            batch["marker_pos"].to(agent.device),
            batch["marker_mask"].to(agent.device),
            batch["qtype"].to(agent.device),
        )
    logits = logits.float().cpu().numpy()
    return [logits[i, :len(it["markers"])] for i, it in enumerate(items)]


def softmax_t(z, temperature: float = 1.0):
    import numpy as np
    z = np.asarray(z, dtype=float) / max(1e-3, float(temperature))
    e = np.exp(z - z.max())
    return e / e.sum()


def temperature_for(agent, qtype: int, k: int, unclamped: bool = False) -> float:
    """The temperature to score this bucket with.

    By default this is what ``Agent`` applies, i.e. the checkpoint's bucket clamped
    to ``[TEMP_MIN, TEMP_MAX]``. With ``unclamped=True`` it is the checkpoint's raw
    value, which is what the committed pre-#42 sweep was produced with.
    """
    from laya.common import temp_bucket
    bucket = temp_bucket(qtype, k)
    if unclamped:
        return float(agent.temperature_by_options_raw.get(
            bucket, agent.temperature_raw[qtype]))
    return float(agent.temperature_by_options.get(bucket, agent.temperature[qtype]))


def ece(confidence, correct, bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-width confidence bins.

    The first bin is closed at the bottom (`>= lo`), so `confidence == 0.0` is counted.
    That is the boundary #39 settled in `laya.common.ece_score`,
    `research/scripts/bench_local.py` and `research/scripts/build_benchmark_nb.py`. This
    harness kept the pre-#39 test until the divergence was found, which made it the only
    one of the four implementations that binned differently.
    """
    import numpy as np
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if not len(confidence):
        return float("nan")
    total, edges = 0.0, np.linspace(0.0, 1.0, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (confidence >= lo if i == 0 else confidence > lo) & (confidence <= hi)
        if sel.any():
            total += sel.mean() * abs(confidence[sel].mean() - correct[sel].mean())
    return float(total)


def macro_f1(gold, pred) -> float:
    import numpy as np
    gold, pred = np.asarray(gold), np.asarray(pred)
    scores = []
    for cls in sorted(set(gold.tolist()) | set(pred.tolist())):
        tp = int(((pred == cls) & (gold == cls)).sum())
        fp = int(((pred == cls) & (gold != cls)).sum())
        fn = int(((pred != cls) & (gold == cls)).sum())
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def summarise(confidences, corrects, golds, preds) -> Dict[str, float]:
    """The metric block, matching bench_local.py:132-144."""
    import numpy as np
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    n = len(confidences)
    if not n:
        return {"n": 0}
    half = max(1, n // 2)
    order = np.argsort(-confidences)[:half]
    return {
        "n": n,
        "accuracy": round(float(corrects.mean()), 4),
        "macro_f1": round(macro_f1(golds, preds), 4),
        "ece": round(ece(confidences, corrects), 4),
        "mean_confidence": round(float(confidences.mean()), 4),
        "acc_at_50_coverage": round(float(corrects[order].mean()), 4),
    }


# ------------------------------------------------------------------------ runner
def run_language(agent, lang: str, per_lang: int, n_opts: int, seed: int = SEED,
                 unclamped: bool = False) -> Dict[str, Any]:
    """Evaluate one language and return its report plus per-case records."""
    import numpy as np
    from laya.common import QTYPES

    rows = load_language(lang)
    cases, gold, option_keys = build_suite(
        rows, sorted({r["label_text"] for r in rows}), per_lang, n_opts, seed)
    logits = score_cases(agent, cases)

    confidences, corrects, preds, records = [], [], [], []
    for i, z in enumerate(logits):
        k = len(z)
        temperature = temperature_for(agent, QTYPES["choice"], k, unclamped)
        probs = softmax_t(z, temperature)
        pred = int(np.argmax(probs))
        correct = int(pred == gold[i])
        confidences.append(float(probs.max()))
        corrects.append(float(correct))
        preds.append(pred)
        records.append({
            "lang": lang,
            "index": i,
            "state": cases[i][0],
            "instructions": INSTRUCTIONS,
            "options": option_keys[i],
            "gold_index": int(gold[i]),
            "gold_label": option_keys[i][gold[i]],
            "pred_index": pred,
            "pred_label": option_keys[i][pred],
            "probability": round(float(probs[pred]), 6),
            "p_gold": round(float(probs[gold[i]]), 6),
            "confidence": round(float(probs.max()), 6),
            "correct": correct,
            "temperature": round(float(temperature), 6),
        })

    report = summarise(confidences, corrects, gold, preds)
    report["temperature"] = round(
        float(temperature_for(agent, QTYPES["choice"], n_opts, unclamped)), 6)
    return {"report": report, "cases": records}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="laya-eval",
        description="Per-language accuracy and calibration report for a Laya checkpoint.")
    parser.add_argument("--model", default="convaiinnovations/laya",
                        help="checkpoint repo id or local path")
    parser.add_argument("--subfolder", default=None,
                        help="checkpoint subfolder, e.g. multilingual")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps (default: auto)")
    parser.add_argument("--langs", default="en",
                        help="comma-separated configs, or 'all'")
    parser.add_argument("--per-lang", type=int, default=PER_LANG)
    parser.add_argument("--n-opts", type=int, default=N_OPTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--no-cases", action="store_true",
                        help="omit per-case records (smaller file)")
    parser.add_argument("--unclamped", action="store_true",
                        help="score with the checkpoint's RAW bucket temperatures instead "
                             "of the clamped ones Agent applies. This is what reproduces "
                             "the committed pre-#42 sweep")
    args = parser.parse_args(argv)

    if args.langs.strip().lower() == "all":
        try:
            langs = available_languages()
        except Exception as exc:
            print("could not list dataset configs: %s" % exc, file=sys.stderr)
            return 2
    else:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        print("no languages selected", file=sys.stderr)
        return 2

    import laya

    started = time.time()
    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder)
    agent.model.eval()
    payload: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "subfolder": args.subfolder,
            "device": str(agent.device),
            "max_len": agent.cfg.get("max_len"),
            "head_max_len": agent.cfg.get("head_max_len"),
            "dataset": DATASET,
            "split": "test",
            "per_lang": args.per_lang,
            "n_opts": args.n_opts,
            "seed": args.seed,
            "instructions": INSTRUCTIONS,
            "temperatures": dict(agent.temperature_by_options_raw) if args.unclamped
            else dict(agent.temperature_by_options),
            "laya_version": getattr(laya, "__version__", "unknown"),
        },
        "report": {},
        "cases": [],
    }

    for lang in langs:
        t0 = time.time()
        try:
            out = run_language(agent, lang, args.per_lang, args.n_opts,
                               args.seed, args.unclamped)
        except Exception as exc:
            print("  %-8s FAILED: %s" % (lang, str(exc)[:110]), file=sys.stderr)
            payload["report"][lang] = {"error": str(exc)[:200]}
            continue
        payload["report"][lang] = out["report"]
        if not args.no_cases:
            payload["cases"].extend(out["cases"])
        r = out["report"]
        print("  %-8s n=%-4d acc=%.4f macro_f1=%.4f ece=%.4f conf=%.4f  (%.1fs)"
              % (lang, r.get("n", 0), r.get("accuracy", float("nan")),
                 r.get("macro_f1", float("nan")), r.get("ece", float("nan")),
                 r.get("mean_confidence", float("nan")), time.time() - t0),
              flush=True)

    scored = {k: v for k, v in payload["report"].items() if "error" not in v and v.get("n")}
    if scored:
        payload["summary"] = {
            "languages": len(scored),
            "macro_accuracy": round(sum(v["accuracy"] for v in scored.values()) / len(scored), 4),
            "macro_ece": round(sum(v["ece"] for v in scored.values()) / len(scored), 4),
            "macro_f1": round(sum(v["macro_f1"] for v in scored.values()) / len(scored), 4),
            "seconds": round(time.time() - started, 1),
        }
        print("\n  macro over %d languages: acc=%.4f  ece=%.4f  f1=%.4f"
              % (len(scored), payload["summary"]["macro_accuracy"],
                 payload["summary"]["macro_ece"], payload["summary"]["macro_f1"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
