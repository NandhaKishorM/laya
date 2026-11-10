"""Answer stability under option rename and reorder, as a predict hook (#635).

`StabilityHook` adds, in `on_predict_start`, re-asked copies of every `choice` question and
folds them back in `on_predict_end`, so the caller gets the questions it asked plus a
`reliability` field on each choice answer. Laya's own answer is unchanged and no core code is
touched. The copies are extra question rows in the same forward pass.

Which probes run depends on the checkpoint's `option_layout` (#951):

* rename probes, always: the same options in the same slots, under opaque keys `A`, `B`,
  `C`... assigned forwards, backwards and shuffled. Skipped above `rename_max_options`, where a
  rename can leave only a fragment of each description (#543).
* reorder probes, only on a sequential checkpoint: the reversed order and seeded shuffles,
  through `option_order`. On a parallel checkpoint the logits permute with the options, so a
  reorder probe always agrees and would only add cost.

`reliability["probes"]` names the probe kinds that actually ran. When none could run, the
signals are `None`, never a perfect score, so a caller can tell an absence from a zero.

Run from the repository root after installing Laya and `datasets`:

    python -m research.eval.stability --langs en --per-lang 300 --n-opts 20 --out stability.json
    python -m unittest research.eval.test_stability -v
"""
from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
import weakref
import zlib
from typing import Any, Dict, List, Optional, Sequence

from laya.hooks import BaseHook

from . import laya_eval as harness
from .metamorphic import auroc, opaque_labels

SEP = "::stab"
PROBES = ("rename", "reorder")
LAYOUTS = ("auto", "sequential", "parallel")
# Above this many options the head budget cuts option text, and a rename can leave only a
# fragment of each description (#543), so rename probes are skipped there.
RENAME_MAX_OPTIONS = 20


class StabilityHook(BaseHook):
    """Adds a `reliability` field to every choice answer from re-asked copies of the question.

    reliability = {
        "layout":             "sequential" or "parallel", the layout the probes were chosen for,
        "probes":             probe kinds that ran, e.g. ["rename", "reorder"]; [] if none could,
        "soft_stability":     mean probability of the chosen option over the original and every copy,
        "stability":          share of copies whose answer is the chosen option,
        "by_probe":           the same two numbers, and the copy count, per probe kind,
        "n_variants":         rows used for the question, the original included,
        "variant_choices":    each copy's answer, mapped back to the caller's option keys,
        "distinct_choices":   the different answers seen,
        "variants_collapsed": copies whose options lost their distinct token spans,
    }

    `layout="auto"` reads the layout from the agent running the call (`ctx.agent`), defaulting
    to sequential, the layout of every published checkpoint. `probes` restricts the probe kinds,
    e.g. `probes=("rename",)` to measure what a parallel checkpoint would get. Copies are seeded
    from the question id, so every state in a batch, and every call, gets the same copies.
    """

    def __init__(self, shuffles: int = 3, renames: int = 3,
                 rename_max_options: int = RENAME_MAX_OPTIONS, layout: str = "auto",
                 probes: Sequence[str] = PROBES, seed: int = 13):
        if shuffles < 0 or renames < 0:
            raise ValueError("shuffles and renames must be >= 0")
        if layout not in LAYOUTS:
            raise ValueError("layout must be one of %s, got %r" % (", ".join(LAYOUTS), layout))
        unknown = set(probes) - set(PROBES)
        if unknown:
            raise ValueError("unknown probes %s; use %s" % (sorted(unknown), ", ".join(PROBES)))
        self.shuffles = shuffles
        self.renames = renames
        self.rename_max_options = rename_max_options
        self.layout = layout
        self.probes = tuple(p for p in PROBES if p in probes)
        self.seed = seed
        self._plans: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def layout_of(self, ctx) -> str:
        if self.layout != "auto":
            return self.layout
        return "parallel" if getattr(ctx.agent, "parallel_options", False) else "sequential"

    def variants(self, qid: str, question: Dict[str, Any], layout: str = "sequential") -> List[tuple]:
        """(probe, name, question, renamed key -> caller key or None) for each copy of one question."""
        keys = list(question["criteria"])
        n = len(keys)
        rng = random.Random(zlib.crc32(("%s|%s" % (self.seed, qid)).encode("utf-8")))
        base = {k: v for k, v in question.items() if k != "option_order"}
        out = []
        if "rename" in self.probes and n <= self.rename_max_options and self.renames:
            labels = opaque_labels(n)
            assignments = [("rename", labels), ("rename_reversed", labels[::-1])]
            for i in range(max(0, self.renames - 2)):
                shuffled = labels[:]
                rng.shuffle(shuffled)
                assignments.append(("rename_shuffle%d" % (i + 1), shuffled))
            seen = set()
            for name, assigned in assignments[:self.renames]:
                if tuple(assigned) in seen:
                    continue
                seen.add(tuple(assigned))
                copy = dict(base)
                # same slots and descriptions; only the model-facing keys change
                copy["criteria"] = dict(zip(assigned, question["criteria"].values()))
                out.append(("rename", name, copy, dict(zip(assigned, keys))))
        if "reorder" in self.probes and layout == "sequential":
            identity = list(range(n))
            seen = {tuple(identity)}  # the original; a repeat would always agree with it
            orders = [("reversed", identity[::-1])]
            for i in range(self.shuffles):
                order = identity[:]
                rng.shuffle(order)
                orders.append(("shuffle%d" % (i + 1), order))
            for name, order in orders:
                if tuple(order) in seen:
                    continue
                seen.add(tuple(order))
                copy = dict(base)
                copy["option_order"] = list(order)
                out.append(("reorder", name, copy, None))
        return out

    def on_predict_start(self, ctx) -> None:
        if ctx.results is not None or not isinstance(ctx.questions, dict):
            return
        layout = self.layout_of(ctx)
        packed = dict(ctx.questions)
        plan: Dict[str, List[tuple]] = {}
        for qid, question in ctx.questions.items():
            if SEP in str(qid):
                raise ValueError("question id %r must not contain %r" % (qid, SEP))
            if (not isinstance(question, dict) or question.get("type") != "choice"
                    or not isinstance(question.get("criteria"), dict) or len(question["criteria"]) < 2):
                continue
            plan[qid] = []  # kept even when empty, so the answer still reports that no probe ran
            for i, (probe, name, copy, back) in enumerate(self.variants(qid, question, layout), 1):
                key = "%s%s%d" % (qid, SEP, i)
                packed[key] = copy
                plan[qid].append((probe, name, key, back))
        if not plan:
            return
        with self._lock:
            self._plans[ctx] = (ctx.questions, plan, layout)
        ctx.questions = packed

    def on_predict_end(self, ctx) -> None:
        with self._lock:
            entry = self._plans.pop(ctx, None)
        if entry is None:
            return
        ctx.questions, plan, layout = entry
        if not ctx.results:
            return
        copies = {key for steps in plan.values() for _, _, key, _ in steps}
        for result in ctx.results:
            answers = result.get("answers") or {}
            usage = result.get("usage") or {}
            collapsed = usage.get("options") or {}
            for qid, steps in plan.items():
                answer = answers.get(qid)
                if isinstance(answer, dict) and answer.get("choice") is not None:
                    answer["reliability"] = _fold(answer, steps, answers, collapsed, layout)
            for key in copies:
                answers.pop(key, None)
            if "truncated_questions" in usage:
                usage["truncated_questions"] = [q for q in usage["truncated_questions"] if q not in copies]
            if collapsed:
                kept = {q: v for q, v in collapsed.items() if q not in copies}
                if kept:
                    usage["options"] = kept
                else:
                    usage.pop("options", None)


def _summary(winner: str, original: Optional[float], copies: List[tuple]) -> Dict[str, Any]:
    if not copies:
        return {"n": 0, "soft_stability": None, "stability": None}
    support = [p for _, _, p in copies] + ([original] if original is not None else [])
    return {"n": len(copies), "soft_stability": round(sum(support) / len(support), 4),
            "stability": round(sum(c == winner for _, c, _ in copies) / len(copies), 4)}


def _fold(answer: Dict[str, Any], steps: List[tuple], answers: Dict[str, Any],
          collapsed: Dict[str, Any], layout: str) -> Dict[str, Any]:
    winner = answer["choice"]
    original = float((answer.get("probabilities") or {}).get(winner, 0.0))
    seen = []  # (probe, name, choice, support)
    for probe, name, key, back in steps:
        copy = answers.get(key)
        if not isinstance(copy, dict) or copy.get("choice") is None:
            continue
        back = back or {}
        probabilities = {back.get(k, k): v for k, v in (copy.get("probabilities") or {}).items()}
        seen.append((probe, name, back.get(copy["choice"], copy["choice"]), float(probabilities.get(winner, 0.0))))
    ran = [p for p in PROBES if any(s[0] == p for s in seen)]
    overall = _summary(winner, original, [(n, c, p) for _, n, c, p in seen])
    return {
        "layout": layout,
        "probes": ran,
        "soft_stability": overall["soft_stability"],
        "stability": overall["stability"],
        # per kind the original is left out, so each number reflects that probe alone
        "by_probe": {p: _summary(winner, None, [(n, c, s) for k, n, c, s in seen if k == p]) for p in ran},
        "n_variants": 1 + len(seen),
        "variant_choices": {name: choice for _, name, choice, _ in seen},
        "distinct_choices": sorted({winner, *(choice for _, _, choice, _ in seen)}),
        "variants_collapsed": sum(key in collapsed for _, _, key, _ in steps),
    }


def signal_metrics(scores: Sequence[float], correct: Sequence[bool]) -> Dict[str, Any]:
    """AUROC plus the selective-classification metrics from laya.evals (#854) for one signal."""
    from laya import evals

    scores, correct = list(scores), list(correct)
    out = {"auroc": auroc(scores, correct), "aurc": evals.aurc(scores, correct),
           "brier": evals.brier(scores, correct), "distinct_values": len(set(scores))}
    for coverage in evals.SELECTIVE_COVERAGES + (0.7,):
        out["selective_accuracy@%d" % round(coverage * 100)] = evals.selective_accuracy(scores, correct, coverage)
    return out


def bootstrap_difference(rows: List[Dict[str, Any]], signal: str, baseline: str, metric: str,
                         n_boot: int = 2000, seed: int = 0) -> Optional[Dict[str, float]]:
    """Mean and 95% interval of metric(signal) - metric(baseline) over cases resampled with replacement."""
    from laya import evals

    fn = auroc if metric == "auroc" else getattr(evals, metric)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [rng.choice(rows) for _ in rows]
        y = [r["correct"] for r in sample]
        a, b = fn([r[signal] for r in sample], y), fn([r[baseline] for r in sample], y)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return None
    diffs.sort()
    return {"mean": sum(diffs) / len(diffs), "low": diffs[int(0.025 * len(diffs))],
            "high": diffs[int(0.975 * len(diffs)) - 1]}


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def evaluate_language(agent, hook: StabilityHook, cases, gold, keys, batch_states: int,
                      batch_size: int) -> Dict[str, Any]:
    """Per-case plain vs hooked predictions, signal metrics, and the measured cost."""
    rows = []
    agent.predict(cases[0][0], cases[0][1], hooks=[hook])  # warm-up, so one-time setup is not timed
    for i, (state, questions) in enumerate(cases):
        qid = next(iter(questions))
        t0 = time.perf_counter()
        plain = agent.predict(state, questions)
        t1 = time.perf_counter()
        hooked = agent.predict(state, questions, hooks=[hook])
        t2 = time.perf_counter()
        p, h = plain["answers"][qid], hooked["answers"][qid]
        rel = h["reliability"]
        rows.append({
            "index": i, "state": state, "gold": keys[i][gold[i]], "choice": p["choice"],
            "correct": p["choice"] == keys[i][gold[i]],
            "answer_confidence": p["answer_confidence"],
            "soft_stability": rel["soft_stability"], "stability": rel["stability"],
            "soft_rename": (rel["by_probe"].get("rename") or {}).get("soft_stability"),
            "soft_reorder": (rel["by_probe"].get("reorder") or {}).get("soft_stability"),
            "probes": rel["probes"],
            "same_answer": h["choice"] == p["choice"]
                           and abs(h["answer_confidence"] - p["answer_confidence"]) <= 1e-4,
            "rows": rel["n_variants"], "variants_collapsed": rel["variants_collapsed"],
            "original_collapsed": qid in ((plain.get("usage") or {}).get("options") or {}),
            "tokens_plain": (plain.get("usage") or {}).get("input_tokens"),
            "tokens_hooked": (hooked.get("usage") or {}).get("input_tokens"),
            "ms_plain": 1000 * (t1 - t0), "ms_hooked": 1000 * (t2 - t1),
            "variant_choices": rel["variant_choices"],
        })
        if (i + 1) % 25 == 0:
            print("  %d/%d" % (i + 1, len(cases)), flush=True)

    correct = [r["correct"] for r in rows]
    present = [s for s in ("answer_confidence", "soft_stability", "stability", "soft_rename", "soft_reorder")
               if all(r[s] is not None for r in rows)]
    signals = {s: signal_metrics([r[s] for r in rows], correct) for s in present}
    differences = {}
    if 0 < sum(correct) < len(correct):
        for signal in ("soft_stability", "soft_rename", "soft_reorder"):
            if signal not in present:
                continue
            for metric in ("auroc", "aurc"):
                differences["%s - answer_confidence, %s" % (signal, metric)] = bootstrap_difference(
                    rows, signal, "answer_confidence", metric)
    cost = {
        "probes_run": sorted({p for r in rows for p in r["probes"]}),
        "rows_per_question": _mean(r["rows"] for r in rows),
        "tokens_per_case_plain": _mean(r["tokens_plain"] for r in rows),
        "tokens_per_case_hooked": _mean(r["tokens_hooked"] for r in rows),
        "ms_per_case_plain": _mean(r["ms_plain"] for r in rows),
        "ms_per_case_hooked": _mean(r["ms_hooked"] for r in rows),
        "cases_original_collapsed": sum(r["original_collapsed"] for r in rows),
        "cases_with_collapsed_variants": sum(r["variants_collapsed"] > 0 for r in rows),
        "same_answer_as_plain": sum(r["same_answer"] for r in rows),
    }
    report = {"n": len(rows), "n_wrong": len(rows) - sum(correct), "accuracy": _mean(correct),
              "signals": signals, "differences": differences, "cost": cost}
    if batch_states > 0:
        report["batch_cost"] = batch_cost(agent, hook, cases, batch_states, batch_size)
    return {"report": report, "cases": rows}


def batch_cost(agent, hook: StabilityHook, cases, n_states: int, batch_size: int) -> Dict[str, Any]:
    """Plain vs hooked predict_batch on one shared question over many states."""
    questions = cases[0][1]
    qid = next(iter(questions))
    states = [state for state, _ in cases[:n_states]]
    agent.predict_batch(states[:2], questions, batch_size=batch_size)  # warm-up
    t0 = time.perf_counter()
    plain = agent.predict_batch(states, questions, batch_size=batch_size)
    t1 = time.perf_counter()
    hooked = agent.predict_batch(states, questions, batch_size=batch_size, hooks=[hook])
    t2 = time.perf_counter()
    same = sum(h["answers"][qid]["choice"] == p["answers"][qid]["choice"] for p, h in zip(plain, hooked))
    tokens = lambda results: sum((r.get("usage") or {}).get("input_tokens", 0) for r in results)
    return {"states": len(states), "batch_size": batch_size,
            "options": len(questions[qid]["criteria"]),
            "rows_per_state_hooked": hooked[0]["answers"][qid]["reliability"]["n_variants"],
            "ms_per_state_plain": 1000 * (t1 - t0) / len(states),
            "ms_per_state_hooked": 1000 * (t2 - t1) / len(states),
            "tokens_plain": tokens(plain), "tokens_hooked": tokens(hooked),
            "same_answer_as_plain": same}


def _print_report(lang: str, report: Dict[str, Any]) -> None:
    print("\n%s: %d cases, %d wrong, accuracy %.3f" % (lang, report["n"], report["n_wrong"], report["accuracy"]))
    names = ["auroc", "aurc", "brier", "selective_accuracy@50", "selective_accuracy@70",
             "selective_accuracy@80", "distinct_values"]
    print("  %-22s" % "" + "".join("%13s" % n.replace("selective_accuracy", "sel_acc").replace("distinct_values", "distinct")
                                    for n in names))
    for signal, metrics in report["signals"].items():
        print("  %-22s" % signal + "".join(
            "%13s" % ("-" if metrics[n] is None else ("%d" % metrics[n] if n == "distinct_values"
                                                      else "%.3f" % metrics[n])) for n in names))
    for name, diff in report["differences"].items():
        if diff:
            print("  %s: %+.3f (%+.3f to %+.3f)" % (name, diff["mean"], diff["low"], diff["high"]))
    for name, value in report["cost"].items():
        print("  %s: %s" % (name, "%.1f" % value if isinstance(value, float) else value))
    for name, value in (report.get("batch_cost") or {}).items():
        print("  batch %s: %s" % (name, "%.1f" % value if isinstance(value, float) else value))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--langs", default="en", help="comma-separated MASSIVE configs")
    parser.add_argument("--per-lang", type=int, default=harness.PER_LANG)
    parser.add_argument("--n-opts", type=int, default=harness.N_OPTS)
    parser.add_argument("--seed", type=int, default=harness.SEED)
    parser.add_argument("--shuffles", type=int, default=3)
    parser.add_argument("--renames", type=int, default=3)
    parser.add_argument("--rename-max-options", type=int, default=RENAME_MAX_OPTIONS)
    parser.add_argument("--layout", choices=LAYOUTS, default="auto",
                        help="probe set to choose: auto reads the checkpoint config")
    parser.add_argument("--probes", default=",".join(PROBES),
                        help="comma-separated probe kinds; 'rename' alone is what a parallel checkpoint gets")
    parser.add_argument("--batch-states", type=int, default=64, help="states for the predict_batch timing; 0 skips it")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", required=True, help="JSON report path")
    args = parser.parse_args(argv)

    import laya

    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder)
    hook = StabilityHook(shuffles=args.shuffles, renames=args.renames,
                         rename_max_options=args.rename_max_options, layout=args.layout,
                         probes=[p.strip() for p in args.probes.split(",") if p.strip()], seed=args.seed)
    payload: Dict[str, Any] = {
        "config": {**vars(args), "dataset": harness.DATASET, "split": "test",
                   "device": str(agent.device), "laya_version": laya.__version__,
                   "max_len": agent.cfg.get("max_len"), "head_max_len": agent.cfg.get("head_max_len"),
                   "binning_map": bool(getattr(agent, "binning_map", None)),
                   "option_layout": "parallel" if getattr(agent, "parallel_options", False) else "sequential"},
        "report": {}, "cases": {},
    }
    failed = False
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        try:
            data = harness.load_language(lang)
            cases, gold, keys = harness.build_suite(
                data, sorted({r["label_text"] for r in data}), args.per_lang, args.n_opts, args.seed)
            result = evaluate_language(agent, hook, cases, gold, keys, args.batch_states, args.batch_size)
            payload["report"][lang] = result["report"]
            payload["cases"][lang] = result["cases"]
            _print_report(lang, result["report"])
        except Exception as exc:  # keep the other languages, report the failure
            failed = True
            payload["report"][lang] = {"error": str(exc)}
            print("%s FAILED: %s" % (lang, exc), flush=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2,
                  default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
