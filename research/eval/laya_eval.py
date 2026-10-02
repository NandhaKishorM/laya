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
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
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


def load_language(lang: str, split: str = "test", revision: Optional[str] = None):
    """Load one language config. Raises with a readable message if unavailable."""
    try:
        if revision is not None:
            from huggingface_hub import hf_hub_download
            # datasets may fall back to its latest prepared cache when a pin is unavailable.
            # Hub file lookup keeps the requested revision part of the cache identity.
            path = hf_hub_download(repo_id=DATASET, filename="%s/%s.json.gz" % (split, lang),
                                   repo_type="dataset", revision=revision)
            with gzip.open(path, "rt", encoding="utf-8") as source:
                ds = [json.loads(line) for line in source]
        else:
            from datasets import load_dataset
            ds = load_dataset(DATASET, lang, split=split)
        return [{"text": r["text"], "label_text": r["label_text"]} for r in ds]
    except Exception as exc:                       # pragma: no cover - network path
        raise RuntimeError(
            "could not load %s config %r: %s" % (DATASET, lang, exc)
        ) from exc


def available_languages(revision: Optional[str] = None) -> List[str]:
    """Language configs the dataset exposes, excluding the aggregate 'default'."""
    from datasets import get_dataset_config_names
    kw = {"revision": revision} if revision else {}
    names = get_dataset_config_names(DATASET, **kw)
    return sorted(n for n in names if n != "default")


# ---------------------------------------------------------------------- scoring
def internal_question(qdef: Dict[str, Any]) -> Dict[str, Any]:
    """The `build_sequence` question dict for one harness question definition.

    Shared with the metamorphic budget probe so a diagnostic measures exactly the
    question that gets scored.
    """
    return {"t": qdef["type"], "ins": qdef["instructions"], "crit": qdef.get("criteria")}


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
            q = internal_question(qdef)
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
def input_fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    """Hash ordered, model-facing inputs without predictions or timing fields."""
    inputs = [{key: row[key] for key in (
        "lang", "index", "state", "instructions", "options", "option_texts", "gold_index")}
        for row in records]
    blob = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def source_revision() -> Tuple[Optional[str], Optional[bool]]:
    """Identify the checked-out code, including tracked local edits."""
    try:
        head = subprocess.run(["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", _REPO_ROOT, "status", "--porcelain",
                                 "--untracked-files=no"], capture_output=True, text=True,
                                check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(status.strip())


def package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and (isinstance(value, int) or math.isfinite(value)))


def validate_release_report(payload: Dict[str, Any], langs: Sequence[str]) -> List[str]:
    """Refuse a release-candidate artifact whose inputs or cases cannot be audited."""
    if not isinstance(payload, dict):
        return ["release report must be an object"]
    config, report, cases = (payload.get(key) for key in ("config", "report", "cases"))
    if (not isinstance(config, dict) or not isinstance(report, dict)
            or not isinstance(cases, list) or any(not isinstance(row, dict) for row in cases)):
        return ["release report needs config, report and per-case records"]
    errors = []
    for key in ("requested_model_revision", "model_revision", "dataset_revision", "source_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(config.get(key) or "")):
            errors.append("%s must be a full commit SHA" % key)
    if config.get("model_revision") != config.get("requested_model_revision"):
        errors.append("loaded model revision differs from the requested revision")
    if config.get("source_dirty") is not False:
        errors.append("source checkout has tracked changes or cannot be inspected")
    environment = config.get("environment")
    if not isinstance(environment, dict) or any(not environment.get(name) for name in
           ("python", "torch", "transformers", "datasets")):
        errors.append("runtime versions are incomplete")
    if len(set(langs)) != len(langs):
        errors.append("requested languages contain duplicates")
    per_lang, n_opts = config.get("per_lang"), config.get("n_opts")
    if not isinstance(per_lang, int) or isinstance(per_lang, bool) or per_lang < 1:
        errors.append("per_lang must be positive")
        return errors
    if not isinstance(n_opts, int) or isinstance(n_opts, bool) or n_opts < 2:
        errors.append("n_opts must be an integer of at least two")
        return errors
    required = {
        "lang", "index", "state", "instructions", "options", "option_texts",
        "gold_index", "gold_label", "pred_index", "pred_label", "probability",
        "p_gold", "confidence", "correct", "temperature",
    }
    for lang in langs:
        result = report.get(lang, {})
        records = [row for row in cases if row.get("lang") == lang]
        if not isinstance(result, dict):
            errors.append("%s has no metric object" % lang)
            continue
        if "error" in result:
            errors.append("%s failed: %s" % (lang, result["error"]))
            continue
        if result.get("n") != per_lang or len(records) != per_lang:
            errors.append("%s has incomplete metrics or per-case records" % lang)
            continue
        valid = True
        for i, row in enumerate(records):
            missing = required - set(row)
            if missing:
                errors.append("%s case %d is missing fields: %s" % (lang, i, ", ".join(sorted(missing))))
                valid = False
                continue
            options, texts = row.get("options"), row.get("option_texts")
            gold, pred = row.get("gold_index"), row.get("pred_index")
            if (type(row["index"]) is not int or row["index"] != i
                    or not isinstance(row["instructions"], str) or not isinstance(options, list)
                    or not isinstance(texts, list) or len(options) != n_opts
                    or any(not isinstance(option, str) for option in options)
                    or any(not isinstance(text, str) for text in texts)
                    or len(set(options)) != len(options) or len(texts) != len(options)
                    or type(gold) is not int or not 0 <= gold < len(options) or type(pred) is not int
                    or not 0 <= pred < len(options) or row.get("gold_label") != options[gold]
                    or row.get("pred_label") != options[pred] or type(row["correct"]) is not int
                    or row["correct"] != int(pred == gold)):
                errors.append("%s case %d has inconsistent inputs or decision labels" % (lang, i))
                valid = False
                continue
            confidence, probability, p_gold = (row[key] for key in ("confidence", "probability", "p_gold"))
            if (any(not _finite_number(value) or not 0 <= value <= 1
                    for value in (confidence, probability, p_gold))
                    or not _finite_number(row["temperature"]) or row["temperature"] <= 0):
                errors.append("%s case %d has invalid probabilities or temperature" % (lang, i))
                valid = False
                continue
            # The artifact carries the chosen and gold probabilities, not the whole vector.
            # These are the argmax facts those two numbers can establish without inventing it.
            if (abs(probability - confidence) > 1e-12 or probability < 1.0 / n_opts - 1e-12
                    or p_gold > probability or (pred == gold and abs(p_gold - probability) > 1e-12)
                    or (pred != gold and p_gold + probability > 1.0 + 1e-12)
                    or (pred != gold and 1.0 - probability - p_gold > (n_opts - 2) * probability + 1e-12)
                    or (pred != gold and p_gold == probability and gold < pred)):
                errors.append("%s case %d probabilities disagree with its decision" % (lang, i))
                valid = False
        if not valid:
            continue
        try:
            fingerprint = input_fingerprint(records)
        except (TypeError, ValueError):
            errors.append("%s case inputs cannot be serialized" % lang)
            continue
        if result.get("input_sha256") != fingerprint:
            errors.append("%s input fingerprint does not match its cases" % lang)
        recomputed = summarise([row["confidence"] for row in records],
                               [row["correct"] for row in records],
                               [row["gold_index"] for row in records],
                               [row["pred_index"] for row in records])
        for metric, value in recomputed.items():
            recorded = result.get(metric)
            if (not _finite_number(recorded) or (metric != "n" and not 0 <= recorded <= 1)
                    or abs(recorded - value) > 1e-12):
                errors.append("%s %s does not match its cases" % (lang, metric))
        if (not _finite_number(result.get("temperature")) or result["temperature"] <= 0
                or any(row["temperature"] != result["temperature"] for row in records)):
            errors.append("%s temperature does not match its cases" % lang)
    if any(row.get("lang") not in langs for row in cases):
        errors.append("cases include an unrequested language")
    return errors


def run_language(agent, lang: str, per_lang: int, n_opts: int, seed: int = SEED,
                 unclamped: bool = False, dataset_revision: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate one language and return its report plus per-case records."""
    import numpy as np
    from laya.common import QTYPES

    rows = load_language(lang, revision=dataset_revision) if dataset_revision else load_language(lang)
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
            "option_texts": [render_label(key) for key in option_keys[i]],
            "gold_index": int(gold[i]),
            "gold_label": option_keys[i][gold[i]],
            "pred_index": pred,
            "pred_label": option_keys[i][pred],
            # Keep the scored precision: rounding here can move an ECE bin or reorder
            # near-tied cases at the 50% coverage cutoff when a report is audited later.
            "probability": float(probs[pred]),
            "p_gold": float(probs[gold[i]]),
            "confidence": float(probs.max()),
            "correct": correct,
            "temperature": float(temperature),
        })

    report = summarise(confidences, corrects, gold, preds)
    report["input_sha256"] = input_fingerprint(records)
    report["temperature"] = float(temperature_for(agent, QTYPES["choice"], n_opts, unclamped))
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
    parser.add_argument("--model-revision", default=None,
                        help="checkpoint commit, branch or tag (release runs require a full SHA)")
    parser.add_argument("--dataset-revision", default=None,
                        help="dataset commit, branch or tag (release runs require a full SHA)")
    parser.add_argument("--langs", default="en",
                        help="comma-separated configs, or 'all'")
    parser.add_argument("--per-lang", type=int, default=PER_LANG)
    parser.add_argument("--n-opts", type=int, default=N_OPTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--no-cases", action="store_true",
                        help="omit per-case records (smaller file)")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail unless a pinned, complete, auditable report is produced")
    parser.add_argument("--unclamped", action="store_true",
                        help="score with the checkpoint's RAW bucket temperatures instead "
                             "of the clamped ones Agent applies. This is what reproduces "
                             "the committed pre-#42 sweep")
    args = parser.parse_args(argv)

    if args.require_complete and (not args.out or args.no_cases):
        parser.error("--require-complete needs --out and per-case records")
    if args.require_complete:
        if args.per_lang < 1 or args.n_opts < 2:
            parser.error("--require-complete needs positive --per-lang and at least two options")
        for name, revision in (("model", args.model_revision),
                               ("dataset", args.dataset_revision)):
            if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
                parser.error("--%s-revision must be a full commit SHA" % name)

    code_sha, code_dirty = source_revision()
    if args.require_complete and (not code_sha or code_dirty):
        parser.error("--require-complete needs a Git checkout without tracked changes")

    if args.langs.strip().lower() == "all":
        try:
            langs = available_languages(args.dataset_revision)
        except Exception as exc:
            print("could not list dataset configs: %s" % exc, file=sys.stderr)
            return 2
    else:
        langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        print("no languages selected", file=sys.stderr)
        return 2

    import laya

    if args.require_complete:
        expected = os.path.normcase(os.path.realpath(os.path.join(_REPO_ROOT, "laya", "__init__.py")))
        loaded = os.path.normcase(os.path.realpath(getattr(laya, "__file__", None) or ""))
        if loaded != expected:
            parser.error("--require-complete needs laya imported from the same checkout as this harness")

    started = time.time()
    load_kw = {"revision": args.model_revision} if args.model_revision else {}
    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder, **load_kw)
    agent.model.eval()
    if args.require_complete and getattr(agent, "revision", None) != args.model_revision:
        parser.error("loaded checkpoint revision differs from --model-revision")
    payload: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "requested_model_revision": args.model_revision,
            "model_revision": getattr(agent, "revision", None),
            "dataset_revision": args.dataset_revision,
            "source_revision": code_sha,
            "source_dirty": code_dirty,
            "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or None,
                "logical_cpus": os.cpu_count(),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "datasets": package_version("datasets"),
            },
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
                               args.seed, args.unclamped, args.dataset_revision)
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

    errors = validate_release_report(payload, langs) if args.require_complete else []
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("  wrote %s" % args.out)
    for error in errors:
        print("  release report invalid: %s" % error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
