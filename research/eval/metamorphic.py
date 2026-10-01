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


def opaque_labels(count: int, alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> list[str]:
    """Deterministic neutral labels: A, B, ... Z, then key_26, key_27, ..."""
    if count < 1:
        raise ValueError("at least one label is required")
    if len(set(alphabet)) < count:
        return [f"key_{i}" for i in range(count)]
    return list(alphabet[:count])


def rename_labels(case: MetamorphicCase, alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Replace model-facing labels with neutral opaque ones; semantics and order unchanged.

    The option at canonical position i is renamed to opaque_labels[i], so the
    presented order is identical and only the label tokens differ (issue #512).
    """
    labels = opaque_labels(len(case.option_keys), alphabet)
    return _transform(case, "label_rename", list(range(len(case.option_keys))), labels)


def _baseline(case):
    return _transform(case, "baseline", list(range(len(case.option_keys))), case.option_keys)


def make_variants(case, rng: random.Random):
    """Adapt the harness tuple format to baseline, option permutation, label rename.

    Paraphrases and other metamorphic transforms are intentionally deferred.
    """
    canonical = MetamorphicCase(*case)
    return [v.as_record() for v in (
        _baseline(canonical), permute_options(canonical, rng.getrandbits(64)),
        rename_labels(canonical))]


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
             batch_size=16, budget: BudgetProbe | None = None):
    """Score cases via ``score(batch) -> probability vectors in presented order``.

    Optional gold indices refer to original option order; entries may be None.
    Batches bound inference memory. RNG state and variants do not depend on batch
    size. Returned records retain inputs, mappings and full precision vectors.
    An optional ``budget`` probe checks whether both sides of each comparison still
    carry the same rendered option information (#543/#517/#569).
    """
    cases = list(cases)
    rng = random.Random(seed)
    generated = [make_variants(case, rng) for case in cases]
    return _evaluate_generated(cases, generated, score, gold_indices, batch_size, budget)


def _evaluate_generated(cases, generated_by_case, score, gold_indices, batch_size,
                        budget=None):
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
            record = {
                "kind": v["kind"],
                "presented_options": list(next(iter(v["case"][1].values()))["criteria"]),
                "canonical_indices": v["canonical_indices"],
                "canonical_to_transformed": v["canonical_to_transformed"],
                "transformed_to_canonical": v["transformed_to_canonical"],
                "probabilities": probs, "pred_index": pred,
                "pred_label": records[index]["options"][pred],
                "confidence": max(probs),
                "correct": None if golds[index] is None else pred == golds[index],
            }
            if budget is not None:
                record["budget"] = budget.measure(*v["case"])
            records[index]["variants"].append(record)
    return {"report": _report(records), "cases": records}


def auroc(scores, labels):
    """Chance that a random correct case scores above a random wrong one.

    Ties count half. Returns None unless both correct and wrong cases exist.
    """
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    ranks, start = [0.0] * len(pairs), 0
    while start < len(pairs):
        end = start
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[start][0]:
            end += 1
        for i in range(start, end + 1):
            ranks[i] = (start + end) / 2 + 1
        start = end + 1
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def selective_prediction(records, coverages=(0.5, 0.7, 0.8, 0.9)):
    """Whether disagreement under the transforms predicts which baseline answers are wrong.

    Per labelled case, with w the baseline winner:
      confidence          baseline probability of w
      agreement_<kind>    1 if the <kind> variant(s) also pick w, else the share that do
      support_<kind>      mean probability the <kind> variant(s) give w
      support_all         mean probability of w over the baseline and every variant
    Reports AUROC per signal and, for the continuous signals, the accuracy of the
    most-trusted fraction of cases at each coverage (ties broken by case order).
    """
    rows = []
    for record in records:
        baseline, variants = record["variants"][0], record["variants"][1:]
        if baseline.get("correct") is None:
            continue
        winner = baseline["pred_index"]
        row = {"correct": bool(baseline["correct"]), "confidence": baseline["probabilities"][winner]}
        for kind in dict.fromkeys(v["kind"] for v in variants):
            same = [v for v in variants if v["kind"] == kind]
            row["agreement_" + kind] = sum(v["pred_index"] == winner for v in same) / len(same)
            row["support_" + kind] = sum(v["probabilities"][winner] for v in same) / len(same)
        row["support_all"] = sum(v["probabilities"][winner] for v in record["variants"]) / len(record["variants"])
        rows.append(row)
    wrong = sum(not r["correct"] for r in rows)
    if not rows:
        return {"n_labelled": 0}
    signals = [key for key in rows[0] if key != "correct" and all(key in r for r in rows)]
    labels = [r["correct"] for r in rows]
    result = {"n_labelled": len(rows), "n_wrong": wrong,
              "auroc": {key: auroc([r[key] for r in rows], labels) for key in signals},
              "accuracy_at_coverage": {}}
    for key in signals:
        if key.startswith("agreement_"):
            continue  # near-binary: the coverage cut would mostly depend on tie order
        ranked = sorted(range(len(rows)), key=lambda i: -rows[i][key])
        result["accuracy_at_coverage"][key] = {
            str(c): sum(rows[i]["correct"] for i in ranked[:max(1, round(len(rows) * c))])
            / max(1, round(len(rows) * c)) for c in coverages}
    return result


def _report(records):
    groups = {"option_order": [], "label_rename": []}
    budget_groups = {"option_order": [], "label_rename": []}
    for record in records:
        baseline = record["variants"][0]
        for variant in record["variants"][1:]:
            variant["comparison"] = distribution_metrics(
                baseline["probabilities"], variant["probabilities"])
            groups[variant["kind"]].append(variant["comparison"])
            if "budget" in baseline and "budget" in variant:
                variant["budget_comparison"] = compare_budgets(
                    baseline["budget"], variant["budget"], variant["canonical_indices"],
                    kind=variant["kind"])
                budget_groups[variant["kind"]].append(variant["budget_comparison"])
    report = {kind: summarise_pairs(pairs) for kind, pairs in groups.items()}
    report["overall"] = summarise_pairs([p for pairs in groups.values() for p in pairs])
    if any(budget_groups.values()):
        report["budget"] = {kind: summarise_budgets(pairs)
                            for kind, pairs in budget_groups.items()}
        report["budget"]["overall"] = summarise_budgets(
            [p for pairs in budget_groups.values() for p in pairs])
    report["quality"] = {}
    for kind in ("baseline", "option_order", "label_rename"):
        labelled = [v for r in records for v in r["variants"]
                    if v["kind"] == kind and v["correct"] is not None]
        quality = {"n_labelled": len(labelled)}
        if labelled:
            quality.update(accuracy=sum(v["correct"] for v in labelled) / len(labelled),
                           ece=harness.ece([v["confidence"] for v in labelled],
                                           [v["correct"] for v in labelled]))
        report["quality"][kind] = quality
    report["selective_prediction"] = selective_prediction(records)
    return report


def model_scorer(agent, unclamped=False):
    """Reuse the harness's raw logits and per-option-count temperatures."""
    from laya.common import QTYPES

    def score(cases):
        return [harness.softmax_t(z, harness.temperature_for(
            agent, QTYPES["choice"], len(z), unclamped))
                for z in harness.score_cases(agent, cases)]
    return score


_ATTRIBUTION_LIMIT = 4096
"""Longest option text (characters) whose surviving span is attributed at all."""


def _retained_prefix(encode, text, kept, full):
    """Character prefix of `text` the surviving token span `kept` stands for, or None.

    The only sound claim is an exact witness: a prefix whose encoding equals `kept`
    token for token. Prefix token counts are not guaranteed to be non-decreasing --
    a BPE merge can encode a longer prefix to fewer tokens, and lossy normalization
    can drop characters -- so no bisection point identifies the witness, witnesses
    need not be adjacent, and a single exact witness is required: with several, which
    characters the model kept is undecidable. Every prefix is therefore checked, and
    anything above `_ATTRIBUTION_LIMIT` characters is left unattributed rather than
    searched.

    None means the span could not be attributed to a character boundary, which the
    caller must not read as "nothing was lost".
    """
    if full[:len(kept)] != kept:
        return None
    if full == kept:
        return text
    if len(text) > _ATTRIBUTION_LIMIT:
        return None
    witnesses = [size for size in range(len(text)) if encode(text[:size]) == kept]
    return text[:witnesses[0]] if len(witnesses) == 1 else None


def _retained_description(retained, label, rendered):
    """Description characters a truncated option kept, or None when unverifiable."""
    if retained is None or not rendered.startswith(retained):
        return None
    marker = label + ": "
    start = len(marker) if rendered.startswith(marker) else len(rendered)
    return rendered[start:len(retained)]


def _description_present(value):
    """Whether a criterion value carries description text at all.

    None and blank strings mean "no description" -- whitespace renders as blanks, so
    it is missing too. Anything structured (dict, list, number, boolean) renders as
    JSON text and counts as present; `0` and `False` are legitimate criterion values
    (see `render_options`), not absences.
    """
    if value is None:
        return False
    from laya.common import render_criterion

    return render_criterion(value).strip() != ""


def _to_canonical(values, canonical_indices):
    """Restore presented-order values to canonical order.

    `canonical_indices[p]` is the canonical slot of the option presented at position
    `p` (the same direction `canonicalize` uses for probabilities), so the inverse
    assignment is a scatter, not a gather: gathering is only correct when the
    permutation is its own inverse.
    """
    restored = [None] * len(values)
    for presented, canonical in enumerate(canonical_indices):
        restored[canonical] = values[presented]
    return restored


def _canonical_partition(classes, canonical_indices):
    """Span classes relabelled by first canonical occurrence, so permutations compare equal."""
    relabel = {}
    return [relabel.setdefault(class_, len(relabel))
            for class_ in _to_canonical(classes, canonical_indices)]


class BudgetProbe:
    """Renders options exactly as the model sees them and reports what survived.

    A metamorphic comparison is only information-equivalent if both sides still present
    the same option content after rendering and budgeting; this probe measures that per
    pair. It follows the option-collapse question of #543 (closed; addressed at runtime
    in v0.3.21 via #569/#542) on the evaluation side, where #569's per-inference report
    cannot answer whether baseline and transformed variants agree.

    `build_sequence` caps every option and, when the head budget runs out, re-caps them
    all to the same length, so a shorter label leaves more room for its description. The
    probe reads the option spans back from one `build_sequence` call with an empty state,
    which closes the last option's span -- in a scored sequence it runs on into the
    serialized state and always looks distinguishable (#538) -- and checks them against
    the stats that same call returns.
    """

    def __init__(self, tok, max_len=512, head_max_len=192):
        self.tok = tok
        self.max_len = max_len
        self.head_max_len = head_max_len

    @classmethod
    def from_agent(cls, agent):
        cfg = getattr(agent, "cfg", None) or {}
        return cls(agent.tok, cfg.get("max_len", 512), cfg.get("head_max_len", 192))

    def measure(self, state, questions):
        """One case's option spans, span classes and retained descriptions.

        Returns {"error": ...} instead of raising: a tokenizer the probe cannot drive,
        or a question the budget cuts apart, has to leave the comparison unverifiable
        rather than decide it.
        """
        try:
            return self._measure(state, questions)
        except Exception as exc:
            return {"error": "%s: %s" % (type(exc).__name__, exc)}

    def _measure(self, state, questions):
        from laya.common import build_sequence, encode_text, render_options

        qdef = next(iter(questions.values()))
        q = harness.internal_question(qdef)
        ids, markers, usage = build_sequence(
            self.tok, state, q, self.max_len, self.head_max_len, state_ids=[], return_stats=True)
        rendered = render_options(q)
        if len(markers) != len(rendered):
            # score_cases raises on the same condition: options the head budget accepted
            # were dropped again by max_len, so these are not the spans that get scored.
            raise ValueError("markers %d != options %d" % (len(markers), len(rendered)))
        sep = self.tok.sep_token_id
        # With an empty state the last two ids are the option-block and state separators;
        # anything else at max_len means max_len truncated the option block itself.
        tail_truncated = len(ids) == self.max_len and ids[-2:] != [sep, sep]
        spans = [ids[markers[i] + 1:markers[i + 1]] for i in range(len(markers) - 1)]
        spans.append(ids[markers[-1] + 1:] if tail_truncated else ids[markers[-1] + 1:-2])
        distinct = len({tuple(span) for span in spans})
        if not tail_truncated and distinct != usage["options_distinct"]:
            raise ValueError("recovered spans disagree with build_sequence stats")

        def encode(part):
            return encode_text(self.tok, " " + part, add_special_tokens=False)["input_ids"]

        labels = list(qdef["criteria"])
        description_present = [_description_present(value) for value in qdef["criteria"].values()]
        classes, class_of, retained_descriptions = [], {}, []
        retained_characters = []
        truncated = False
        for position, span in enumerate(spans):
            text = rendered[position].replace(self.tok.mask_token, " ")
            full = encode(text)
            truncated = truncated or len(span) < len(full)
            retained = _retained_prefix(encode, text, list(span), full)
            retained_characters.append(retained)
            retained_descriptions.append(_retained_description(retained, labels[position], text))
            key = tuple(span)
            if key not in class_of:
                class_of[key] = len(class_of)
            classes.append(class_of[key])
        return {
            "options": len(spans),
            "distinct_spans": distinct,
            "tokens_per_option": usage["tokens_per_option"],
            "instruction_tokens": markers[0] - 2,
            "span_classes": classes,
            "retained_text": retained_characters,
            "retained_descriptions": retained_descriptions,
            "description_present": description_present,
            "truncated": truncated,
            "tail_truncated": tail_truncated,
        }


def compare_budgets(baseline, variant, canonical_indices, *, kind):
    """Whether both sides of a comparison still carry the same option information.

    `baseline` and `variant` are `BudgetProbe.measure` results; the baseline must be in
    canonical order, and `canonical_indices` maps the variant's presented slots back to
    canonical ones (the direction `canonicalize` uses). `kind` is the comparison's
    transformation: a `label_rename` over options whose descriptions are missing removes
    the only semantic content there was, so its verdict is unknown regardless of what the
    rendering did; an `option_order` permutation moves every key/value pair together and
    stays valid. `budget_confounded` is True (the rendering no longer carries the same
    content, so drift cannot be read as lexical sensitivity alone), False, or None when
    the rendering could not be measured or attributed. `tokens_per_option` is reported
    but never confounds by itself: when the re-cap bites it changes the spans reported
    here, and when it does not the model sees the same sequence either way.
    """
    comparison = {key: {"baseline": baseline.get(key), "variant": variant.get(key)}
                  for key in ("options", "distinct_spans", "tokens_per_option",
                              "instruction_tokens")}
    errors = {side: measurement["error"] for side, measurement in
              (("baseline", baseline), ("variant", variant)) if "error" in measurement}
    if errors:
        return {**comparison, "retained_descriptions": None, "budget_confounded": None,
                "reasons": [{"code": "measurement_error", "errors": errors}]}
    baseline_retained = baseline["retained_descriptions"]
    variant_retained = _to_canonical(variant["retained_descriptions"], canonical_indices)
    comparison["retained_descriptions"] = {"baseline": baseline_retained,
                                           "variant": variant_retained}
    reasons = []
    if baseline["options"] != variant["options"]:
        reasons.append({"code": "option_count"})
    if baseline["distinct_spans"] != variant["distinct_spans"]:
        reasons.append({"code": "distinct_spans"})
    if baseline["instruction_tokens"] != variant["instruction_tokens"]:
        reasons.append({"code": "instruction_tokens"})
    if baseline["tail_truncated"] or variant["tail_truncated"]:
        reasons.append({"code": "tail_truncation"})
    if baseline["options"] == variant["options"] and _canonical_partition(
            baseline["span_classes"], range(len(baseline["span_classes"]))) != (
            _canonical_partition(variant["span_classes"], canonical_indices)):
        reasons.append({"code": "collision_partition"})
    pairs = list(zip(baseline_retained, variant_retained))
    unverified = [slot for slot, pair in enumerate(pairs) if None in pair]
    differing = [slot for slot, pair in enumerate(pairs)
                 if None not in pair and pair[0] != pair[1]]
    if differing:
        reasons.append({"code": "retained_description", "slots": differing})
    if kind == "label_rename":
        # Descriptions are moved to the same canonical slots by the rename, so the two
        # sides' presence vectors are compared as-is once the variant is restored. The
        # model observation is kept; only the experiment's validity is unknown.
        presence = _to_canonical(variant["description_present"], canonical_indices)
        comparison["description_present"] = {"baseline": baseline["description_present"],
                                             "variant": presence}
        missing = [slot for slot, (base, var) in enumerate(
            zip(baseline["description_present"], presence)) if not base or not var]
        if missing:
            return {**comparison, "budget_confounded": None,
                    "reasons": [{"code": "missing_semantic_description", "slots": missing}]}
    if reasons:
        return {**comparison, "budget_confounded": True, "reasons": reasons}
    if unverified:
        return {**comparison, "budget_confounded": None,
                "reasons": [{"code": "unverified_retained_text", "slots": unverified}]}
    return {**comparison, "budget_confounded": False, "reasons": [{"code": "clean"}]}


def summarise_budgets(pairs):
    """Budget verdict counts per group; `confounded_rate` is over verifiable pairs only."""
    if not pairs:
        return {"n": 0}
    verifiable = [pair for pair in pairs if pair["budget_confounded"] is not None]
    return {
        "n": len(pairs),
        "clean": sum(pair["budget_confounded"] is False for pair in pairs),
        "confounded": sum(pair["budget_confounded"] is True for pair in pairs),
        "unknown": sum(pair["budget_confounded"] is None for pair in pairs),
        "confounded_rate": (sum(pair["budget_confounded"] is True for pair in verifiable)
                            / len(verifiable) if verifiable else None),
    }


@dataclass
class MetamorphicResults:
    baseline: dict
    variants: list[dict]


def evaluate_variants(agent, case: MetamorphicCase, variants, *, score=None,
                      unclamped=False, batch_size=16, budget=None):
    """Evaluate explicit transforms; inject ``score`` for offline stub models.

    With a real agent, call ``agent.model.eval()`` first, as with the harness.
    All output vectors are restored to the original case's canonical ordering.
    A real agent is probed for the option budget by default; a stub ``score``
    is left unprobed unless ``budget`` is passed explicitly.
    """
    probe = budget if budget is not None else (BudgetProbe.from_agent(agent) if score is None else None)
    generated = [_baseline(case).as_record()]
    for variant in variants:
        if variant.kind not in ("option_order", "label_rename"):
            raise ValueError("unsupported transformation kind")
        if list(variant.canonical_to_transformed) != case.option_keys:
            raise ValueError("variant mapping does not match the canonical case")
        generated.append(variant.as_record())
    result = _evaluate_generated([case.as_pair()], [generated],
                                 score if score is not None else model_scorer(agent, unclamped),
                                 [case.gold_index], batch_size, probe)
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
    probe = BudgetProbe.from_agent(agent)
    payload: dict[str, Any] = {
        "config": {**vars(args), "dataset": harness.DATASET, "split": "test",
                   "device": str(agent.device), "laya_version": laya.__version__,
                   "max_len": agent.cfg.get("max_len"),
                   "head_max_len": agent.cfg.get("head_max_len"),
                   "temperature": list(agent.temperature_raw if args.unclamped else agent.temperature),
                   "temperature_by_options": dict(agent.temperature_by_options_raw if args.unclamped
                                                  else agent.temperature_by_options),
                   "permutations_per_case": 1, "label_renames_per_case": 1, "js_log_base": "e"},
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
                              args.seed, args.batch_size, probe)
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
