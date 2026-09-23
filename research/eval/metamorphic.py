"""Small, reproducible choice-invariance diagnostics (issue #244).

Run from the checkout: python -m research.eval.metamorphic --help.
Pure transformations/metrics can also be used without a model or gold labels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import random
from typing import Any, Callable, Sequence

from . import laya_eval as harness


@dataclass
class MetamorphicCase:
    """One canonical choice decision in the existing harness input format."""

    state: Any
    questions: dict
    gold_index: int | None = None

    def __post_init__(self):
        if len(self.questions) != 1:
            raise ValueError("expected exactly one choice question per case")
        question = next(iter(self.questions.values()))
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, dict):
            raise ValueError("expected choice criteria as a dictionary")
        if len(criteria) < 2:
            raise ValueError("at least two options are required")
        if any(not isinstance(k, str) or not k.strip() for k in criteria):
            raise ValueError("option keys must be nonempty strings")
        if self.gold_index is not None and (
                type(self.gold_index) is not int or not 0 <= self.gold_index < len(criteria)):
            raise ValueError("gold index outside canonical options")

    @property
    def option_keys(self):
        return list(next(iter(self.questions.values()))["criteria"])

    def as_pair(self):
        return self.state, self.questions


@dataclass
class MetamorphicVariant:
    """Transformed input plus an explicit bidirectional semantic label mapping."""

    kind: str
    case: MetamorphicCase
    canonical_to_transformed: dict[str, str]

    @property
    def transformed_to_canonical(self):
        return {v: k for k, v in self.canonical_to_transformed.items()}

    def as_record(self):
        mapping = self.canonical_to_transformed
        inverse = self.transformed_to_canonical
        keys = list(mapping)
        if len(inverse) != len(mapping) or set(inverse) != set(self.case.option_keys):
            raise ValueError("label mapping must be a bijection over transformed options")
        return {"kind": self.kind, "case": self.case.as_pair(),
                "canonical_to_transformed": dict(mapping),
                "transformed_to_canonical": inverse,
                "canonical_indices": [keys.index(inverse[k]) for k in self.case.option_keys]}


def _transform(case, kind, order, labels):
    questions = deepcopy(case.questions)
    question = next(iter(questions.values()))
    criteria = question["criteria"]
    keys = case.option_keys
    mapping = dict(zip(keys, labels))
    question["criteria"] = {mapping[keys[i]]: deepcopy(criteria[keys[i]]) for i in order}
    gold = None if case.gold_index is None else order.index(case.gold_index)
    return MetamorphicVariant(kind, MetamorphicCase(deepcopy(case.state), questions, gold), mapping)


def permute_options(case: MetamorphicCase, seed: int = harness.SEED):
    """One deterministic nonidentity shuffle; identity falls back to rotation."""
    order = list(range(len(case.option_keys)))
    random.Random(seed).shuffle(order)
    if order == list(range(len(order))):
        order = order[1:] + order[:1]
    return _transform(case, "option_order", order, case.option_keys)


def _baseline(case):
    return _transform(case, "baseline", list(range(len(case.option_keys))), case.option_keys)


def make_variants(case, rng: random.Random):
    """Adapt the harness tuple format to baseline plus option permutation.

    Label renaming, paraphrases, and other metamorphic transforms are intentionally deferred.
    """
    canonical = MetamorphicCase(*case)
    return [v.as_record() for v in (
        _baseline(canonical), permute_options(canonical, rng.getrandbits(64)))]


def canonicalize(probabilities, canonical_indices):
    """Validate a probability vector and restore canonical semantic ordering."""
    values = [float(p) for p in probabilities]
    n = len(values)
    if not n or any(type(i) is not int for i in canonical_indices) or sorted(canonical_indices) != list(range(n)):
        raise ValueError("probabilities and canonical mapping must be a bijection")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("probabilities must sum to one")
    restored = [0.0] * n
    for p, index in zip(values, canonical_indices):
        restored[index] = p
    return restored


def distribution_metrics(baseline, variant):
    """Drift in canonical space; JS uses natural logs (range 0..ln(2)).

    Ties choose the first canonical option, so reordering tied slots does not
    itself count as disagreement. Confidence is the maximum probability;
    confidence drift is signed (variant minus baseline).
    """
    p = canonicalize(baseline, range(len(baseline)))
    q = canonicalize(variant, range(len(baseline)))
    middle = [(a + b) / 2 for a, b in zip(p, q)]
    js = sum(0.5 * x * math.log(x / m)
             for distribution in (p, q)
             for x, m in zip(distribution, middle) if x > 0)
    drift = [abs(a - b) for a, b in zip(p, q)]
    return {
        "semantic_agreement": max(range(len(p)), key=p.__getitem__) ==
                              max(range(len(q)), key=q.__getitem__),
        "mean_probability_drift": sum(drift) / len(drift),
        "max_probability_drift": max(drift),
        "js_divergence": max(0.0, js),
        "confidence_drift": max(q) - max(p),
    }


def summarise_pairs(pairs):
    """Pair-weighted summary; absent observations are not perfect agreement."""
    if not pairs:
        return {"n": 0}
    return {
        "n": len(pairs),
        "semantic_agreement_rate": sum(p["semantic_agreement"] for p in pairs) / len(pairs),
        "mean_probability_drift": sum(p["mean_probability_drift"] for p in pairs) / len(pairs),
        "max_probability_drift": max(p["max_probability_drift"] for p in pairs),
        "mean_js_divergence": sum(p["js_divergence"] for p in pairs) / len(pairs),
        "mean_confidence_drift": sum(p["confidence_drift"] for p in pairs) / len(pairs),
        "mean_absolute_confidence_drift": sum(abs(p["confidence_drift"]) for p in pairs) / len(pairs),
        "worst_confidence_increase_on_disagreement": max(
            [0.0] + [p["confidence_drift"] for p in pairs if not p["semantic_agreement"]]),
    }


def evaluate(cases, score: Callable, gold_indices=None, seed=harness.SEED,
             batch_size=16):
    """Score cases via ``score(batch) -> probability vectors in presented order``.

    Optional gold indices refer to original option order; entries may be None.
    Batches bound inference memory. RNG state and variants do not depend on batch
    size. Returned records retain inputs, mappings and full precision vectors.
    """
    cases = list(cases)
    rng = random.Random(seed)
    generated = [make_variants(case, rng) for case in cases]
    return _evaluate_generated(cases, generated, score, gold_indices, batch_size)


def _evaluate_generated(cases, generated_by_case, score, gold_indices, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cases = list(cases)
    golds = [None] * len(cases) if gold_indices is None else list(gold_indices)
    if len(golds) != len(cases):
        raise ValueError("one gold index is required per case")
    variants, records = [], []
    for index, (case, gold) in enumerate(zip(cases, golds)):
        generated = generated_by_case[index]
        keys = list(next(iter(case[1].values()))["criteria"])
        if gold is not None and (type(gold) is not int or not 0 <= gold < len(keys)):
            raise ValueError("gold index outside canonical options")
        records.append({"index": index, "state": deepcopy(case[0]),
                        "questions": deepcopy(case[1]), "options": keys,
                        "gold_index": gold, "variants": []})
        variants.extend((index, v) for v in generated)
    for start in range(0, len(variants), batch_size):
        batch = variants[start:start + batch_size]
        vectors = list(score([v["case"] for _, v in batch]))
        if len(vectors) != len(batch):
            raise ValueError("scorer returned the wrong number of probability vectors")
        for (index, v), raw in zip(batch, vectors):
            probs = canonicalize(raw, v["canonical_indices"])
            pred = max(range(len(probs)), key=probs.__getitem__)
            records[index]["variants"].append({
                "kind": v["kind"],
                "presented_options": list(next(iter(v["case"][1].values()))["criteria"]),
                "canonical_indices": v["canonical_indices"],
                "canonical_to_transformed": v["canonical_to_transformed"],
                "transformed_to_canonical": v["transformed_to_canonical"],
                "probabilities": probs, "pred_index": pred,
                "pred_label": records[index]["options"][pred],
                "confidence": max(probs),
                "correct": None if golds[index] is None else pred == golds[index],
            })
    return {"report": _report(records), "cases": records}


def _report(records):
    groups = {"option_order": []}
    for record in records:
        baseline = record["variants"][0]["probabilities"]
        for variant in record["variants"][1:]:
            variant["comparison"] = distribution_metrics(baseline, variant["probabilities"])
            groups[variant["kind"]].append(variant["comparison"])
    report = {kind: summarise_pairs(pairs) for kind, pairs in groups.items()}
    report["overall"] = summarise_pairs([p for pairs in groups.values() for p in pairs])
    report["quality"] = {}
    for kind in ("baseline", "option_order"):
        labelled = [v for r in records for v in r["variants"]
                    if v["kind"] == kind and v["correct"] is not None]
        quality = {"n_labelled": len(labelled)}
        if labelled:
            quality.update(accuracy=sum(v["correct"] for v in labelled) / len(labelled),
                           ece=harness.ece([v["confidence"] for v in labelled],
                                           [v["correct"] for v in labelled]))
        report["quality"][kind] = quality
    return report


def model_scorer(agent, unclamped=False):
    """Reuse the harness's raw logits and per-option-count temperatures."""
    from laya.common import QTYPES

    def score(cases):
        return [harness.softmax_t(z, harness.temperature_for(
            agent, QTYPES["choice"], len(z), unclamped))
                for z in harness.score_cases(agent, cases)]
    return score


@dataclass
class MetamorphicResults:
    baseline: dict
    variants: list[dict]


def evaluate_variants(agent, case: MetamorphicCase, variants, *, score=None,
                      unclamped=False, batch_size=16):
    """Evaluate explicit transforms; inject ``score`` for offline stub models.

    With a real agent, call ``agent.model.eval()`` first, as with the harness.
    All output vectors are restored to the original case's canonical ordering.
    """
    generated = [_baseline(case).as_record()]
    for variant in variants:
        if variant.kind not in ("option_order",):
            raise ValueError("unsupported transformation kind")
        if list(variant.canonical_to_transformed) != case.option_keys:
            raise ValueError("variant mapping does not match the canonical case")
        generated.append(variant.as_record())
    result = _evaluate_generated([case.as_pair()], [generated],
                                 score if score is not None else model_scorer(agent, unclamped),
                                 [case.gold_index], batch_size)
    predictions = result["cases"][0]["variants"]
    return MetamorphicResults(predictions[0], predictions[1:])


def compare_predictions(*, baseline, variants):
    """Report semantic agreement separately from continuous stability metrics.

    No drift threshold or automatic bug classification is applied.
    """
    return _report([{"variants": [deepcopy(baseline), *deepcopy(variants)]}])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--langs", default="en", help="comma-separated MASSIVE configs, or all")
    parser.add_argument("--per-lang", type=int, default=harness.PER_LANG)
    parser.add_argument("--n-opts", type=int, default=harness.N_OPTS)
    parser.add_argument("--seed", type=int, default=harness.SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unclamped", action="store_true")
    parser.add_argument("--out", required=True, help="JSON report path")
    args = parser.parse_args(argv)
    if args.per_lang < 1 or args.n_opts < 2 or args.batch_size < 1:
        parser.error("per-lang/batch-size must be positive and n-opts must be at least 2")
    langs = (harness.available_languages() if args.langs.strip().lower() == "all"
             else list(dict.fromkeys(x.strip() for x in args.langs.split(",") if x.strip())))
    if not langs:
        parser.error("no languages selected")
    import laya

    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder)
    agent.model.eval()
    payload: dict[str, Any] = {
        "config": {**vars(args), "dataset": harness.DATASET, "split": "test",
                   "device": str(agent.device), "laya_version": laya.__version__,
                   "max_len": agent.cfg.get("max_len"),
                   "head_max_len": agent.cfg.get("head_max_len"),
                   "temperature": list(agent.temperature_raw if args.unclamped else agent.temperature),
                   "temperature_by_options": dict(agent.temperature_by_options_raw if args.unclamped
                                                  else agent.temperature_by_options),
                   "permutations_per_case": 1, "js_log_base": "e"},
        "report": {}, "cases": [],
    }
    failed = False
    for lang in langs:
        try:
            rows = harness.load_language(lang)
            cases, gold, _ = harness.build_suite(
                rows, sorted({r["label_text"] for r in rows}), args.per_lang, args.n_opts, args.seed)
            if not cases:
                raise ValueError("dataset returned no cases")
            result = evaluate(cases, model_scorer(agent, args.unclamped), gold,
                              args.seed, args.batch_size)
            payload["report"][lang] = result["report"]
            payload["cases"].extend({"lang": lang, **r} for r in result["cases"])
            print(lang, json.dumps(result["report"]), flush=True)
        except Exception as exc:
            failed = True
            payload["report"][lang] = {"error": str(exc)}
            print("%s FAILED: %s" % (lang, exc), flush=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
