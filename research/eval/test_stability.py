"""Offline tests: python -m unittest research.eval.test_stability -v."""
from copy import deepcopy
import unittest

from laya.hooks import PredictContext

from research.eval import stability as s

QUESTIONS = {"intent": {
    "type": "choice", "instructions": "Choose the request category.",
    "criteria": {"billing": "payment issues", "technical": "software help", "sales": "new purchases"},
}}
DESCRIPTION_WEIGHTS = {"payment issues": 0.7, "software help": 0.2, "new purchases": 0.1}


class FakeAgent:
    def __init__(self, parallel=False):
        self.parallel_options = parallel


def run(hook, states, questions, slot_scorer, collapse=(), agent=None):
    """Mimic Agent.predict_batch: start hooks, one answer per question row, end hooks.

    `slot_scorer(state, keys_by_slot, texts_by_slot)` returns weights in slot order. As in Laya,
    slot s shows option `option_order[s]`, and probabilities come back in the question's own
    option order. Question ids in `collapse` report collapsed options in `usage["options"]`.
    """
    ctx = PredictContext(states=list(states), questions=questions, agent=agent or FakeAgent())
    hook.on_predict_start(ctx)
    results = []
    for state in ctx.states:
        answers = {}
        for qid, q in ctx.questions.items():
            keys, texts = list(q["criteria"]), list(q["criteria"].values())
            order = q.get("option_order", list(range(len(keys))))
            weights = slot_scorer(state, [keys[i] for i in order], [texts[i] for i in order])
            total = sum(weights)
            probs = [0.0] * len(keys)
            for slot, option in enumerate(order):
                probs[option] = weights[slot] / total
            best = max(range(len(keys)), key=lambda i: (probs[i], -i))
            answers[qid] = {"type": "choice", "choice": keys[best],
                            "probabilities": dict(zip(keys, probs)),
                            "answer_confidence": probs[best], "low_confidence": False}
        usage = {"input_tokens": 10 * len(answers), "output_tokens": 0,
                 "truncated_questions": list(answers)}
        collapsed = {qid: {"total": 3, "distinct": 2, "tokens_per_option": 4}
                     for qid in answers if any(qid.startswith(c) for c in collapse)}
        if collapsed:
            usage["options"] = collapsed
        results.append({"model": "laya-rl-agent", "answers": answers, "usage": usage})
    ctx.results = results
    hook.on_predict_end(ctx)
    return ctx


def reliability(ctx, qid="intent", i=0):
    return ctx.results[i]["answers"][qid]["reliability"]


def by_description(state, keys, texts):
    return [DESCRIPTION_WEIGHTS[t] for t in texts]


def by_first_slot(state, keys, texts):
    return [0.9] + [0.1 / (len(keys) - 1)] * (len(keys) - 1)


def by_label(state, keys, texts):
    weights = {"billing": 0.9, "technical": 0.05, "sales": 0.05, "A": 0.2, "B": 0.6, "C": 0.2}
    return [weights[k] for k in keys]


class VariantTests(unittest.TestCase):
    def test_variants_are_deterministic_unique_and_valid(self):
        hook = s.StabilityHook()
        first = hook.variants("intent", QUESTIONS["intent"])
        self.assertEqual(first, s.StabilityHook().variants("intent", QUESTIONS["intent"]))
        shapes = [(tuple(q.get("option_order", (0, 1, 2))), tuple(q["criteria"])) for _, _, q, _ in first]
        self.assertEqual(len(shapes), len(set(shapes)))
        self.assertNotIn(((0, 1, 2), ("billing", "technical", "sales")), shapes)  # never the original
        descriptions = list(QUESTIONS["intent"]["criteria"].values())
        for probe, _, q, back in first:
            if probe == "rename":
                self.assertNotIn("option_order", q)  # renames keep every option in its slot
                self.assertEqual(list(q["criteria"].values()), descriptions)
                self.assertEqual(sorted(q["criteria"]), ["A", "B", "C"])
                self.assertEqual(sorted(back.values()), sorted(QUESTIONS["intent"]["criteria"]))
            else:
                self.assertEqual(list(q["criteria"]), list(QUESTIONS["intent"]["criteria"]))
                self.assertEqual(sorted(q["option_order"]), [0, 1, 2])

    def test_layout_chooses_the_probes(self):
        hook = s.StabilityHook()
        sequential = {p for p, _, _, _ in hook.variants("intent", QUESTIONS["intent"], "sequential")}
        parallel = {p for p, _, _, _ in hook.variants("intent", QUESTIONS["intent"], "parallel")}
        self.assertEqual(sequential, {"rename", "reorder"})
        self.assertEqual(parallel, {"rename"})  # a reorder probe always agrees on a parallel checkpoint

    def test_two_options_and_rename_limit(self):
        two = {"type": "choice", "criteria": {"yes_please": "accept", "no_thanks": "decline"}}
        names = [n for _, n, _, _ in s.StabilityHook(shuffles=5).variants("q", two)]
        self.assertEqual(sorted(names), ["rename", "rename_reversed", "reversed"])
        many = {"type": "choice", "criteria": {"k%d" % i: "option %d" % i for i in range(25)}}
        probes = {p for p, _, _, _ in s.StabilityHook().variants("q", many)}
        self.assertEqual(probes, {"reorder"})
        self.assertEqual(s.StabilityHook().variants("q", many, "parallel"), [])
        probes = {p for p, _, _, _ in s.StabilityHook(probes=("reorder",)).variants("intent", QUESTIONS["intent"])}
        self.assertEqual(probes, {"reorder"})

    def test_own_option_order_is_not_copied(self):
        question = dict(QUESTIONS["intent"], option_order=[2, 0, 1])
        copies = {name: q for _, name, q, _ in s.StabilityHook().variants("intent", question)}
        self.assertNotIn("option_order", copies["rename"])
        self.assertEqual(copies["reversed"]["option_order"], [2, 1, 0])


class HookTests(unittest.TestCase):
    def test_stable_answers_and_clean_output(self):
        questions = deepcopy(QUESTIONS)
        ctx = run(s.StabilityHook(), ["a", "b"], questions, by_description, collapse=("intent::stab",))
        self.assertIs(ctx.questions, questions)
        self.assertEqual(questions, QUESTIONS)
        for result in ctx.results:
            self.assertEqual(list(result["answers"]), ["intent"])
            answer = result["answers"]["intent"]
            self.assertEqual(answer["choice"], "billing")
            rel = answer["reliability"]
            self.assertEqual((rel["layout"], rel["probes"]), ("sequential", ["rename", "reorder"]))
            self.assertEqual(rel["stability"], 1.0)
            self.assertAlmostEqual(rel["soft_stability"], 0.7)
            self.assertEqual(rel["by_probe"]["rename"]["stability"], 1.0)
            self.assertEqual(rel["distinct_choices"], ["billing"])
            self.assertEqual(rel["n_variants"], len(rel["variant_choices"]) + 1)
            self.assertEqual(rel["variants_collapsed"], rel["n_variants"] - 1)
            self.assertEqual(result["usage"]["truncated_questions"], ["intent"])
            self.assertNotIn("options", result["usage"])

    def test_position_sensitive_answers_are_flagged_by_reorder_only(self):
        rel = reliability(run(s.StabilityHook(), ["a"], deepcopy(QUESTIONS), by_first_slot))
        self.assertLess(rel["stability"], 1.0)
        self.assertLess(rel["by_probe"]["reorder"]["stability"], 1.0)
        self.assertEqual(rel["by_probe"]["rename"]["stability"], 1.0)  # renames keep the slots

    def test_label_sensitive_answers_are_flagged_by_rename_only(self):
        rel = reliability(run(s.StabilityHook(), ["a"], deepcopy(QUESTIONS), by_label))
        self.assertEqual(rel["by_probe"]["reorder"]["stability"], 1.0)
        self.assertLess(rel["by_probe"]["rename"]["stability"], 1.0)
        for choice in rel["variant_choices"].values():
            self.assertIn(choice, QUESTIONS["intent"]["criteria"])  # mapped back, never A/B/C

    def test_parallel_checkpoint_runs_rename_probes_only(self):
        ctx = run(s.StabilityHook(), ["a"], deepcopy(QUESTIONS), by_description, agent=FakeAgent(parallel=True))
        rel = reliability(ctx)
        self.assertEqual((rel["layout"], rel["probes"]), ("parallel", ["rename"]))
        self.assertTrue(all(name.startswith("rename") for name in rel["variant_choices"]))

    def test_no_probe_reports_an_absence_not_a_zero(self):
        questions = {"intent": {"type": "choice", "criteria": {"k%d" % i: "option %d" % i for i in range(25)}}}
        rel = reliability(run(s.StabilityHook(), ["a"], questions, by_first_slot, agent=FakeAgent(parallel=True)))
        self.assertEqual(rel["probes"], [])
        self.assertIsNone(rel["soft_stability"])
        self.assertIsNone(rel["stability"])
        self.assertEqual((rel["n_variants"], rel["by_probe"]), (1, {}))

    def test_explicit_layout_overrides_the_agent(self):
        rel = reliability(run(s.StabilityHook(layout="parallel"), ["a"], deepcopy(QUESTIONS), by_description))
        self.assertEqual(rel["probes"], ["rename"])

    def test_same_copies_for_every_state(self):
        ctx = run(s.StabilityHook(), ["a", "b", "c"], deepcopy(QUESTIONS), by_first_slot)
        rels = [r["answers"]["intent"]["reliability"] for r in ctx.results]
        self.assertTrue(all(r == rels[0] for r in rels))

    def test_other_questions_untouched(self):
        questions = deepcopy(QUESTIONS)
        questions["urgent"] = {"type": "noul", "instructions": "Urgent?"}
        questions["single"] = {"type": "choice", "criteria": {"only": "the one option"}}
        ctx = PredictContext(states=["a"], questions=questions, agent=FakeAgent())
        s.StabilityHook().on_predict_start(ctx)
        self.assertNotIn("urgent::stab1", ctx.questions)
        self.assertNotIn("single::stab1", ctx.questions)
        self.assertIn("intent::stab1", ctx.questions)

    def test_failed_or_skipped_calls(self):
        hook = s.StabilityHook()
        questions = deepcopy(QUESTIONS)
        ctx = PredictContext(states=["a"], questions=questions, agent=FakeAgent())
        hook.on_predict_start(ctx)
        hook.on_predict_end(ctx)  # inference failed: no results
        self.assertIs(ctx.questions, questions)
        skipped = PredictContext(states=["a"], questions=questions, agent=FakeAgent())
        skipped.skip([{"answers": {}}])
        hook.on_predict_start(skipped)
        self.assertIs(skipped.questions, questions)
        hook.on_predict_end(skipped)

    def test_invalid_arguments(self):
        ctx = PredictContext(states=["a"], questions={"bad::stab1": QUESTIONS["intent"]}, agent=FakeAgent())
        with self.assertRaises(ValueError):
            s.StabilityHook().on_predict_start(ctx)
        for kwargs in ({"shuffles": -1}, {"renames": -1}, {"layout": "diagonal"}, {"probes": ("paraphrase",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                s.StabilityHook(**kwargs)


class MetricTests(unittest.TestCase):
    def test_signal_metrics_and_bootstrap(self):
        scores, correct = [0.9, 0.8, 0.3, 0.2], [True, True, False, False]
        metrics = s.signal_metrics(scores, correct)
        self.assertEqual(metrics["auroc"], 1)
        self.assertEqual(metrics["selective_accuracy@50"], 1)
        self.assertEqual(metrics["distinct_values"], 4)
        rows = [{"good": g, "bad": 1 - g, "correct": c} for g, c in zip(scores, correct)]
        diff = s.bootstrap_difference(rows, "good", "bad", "auroc", n_boot=200)
        self.assertGreater(diff["low"], 0)
        self.assertIsNone(s.bootstrap_difference([{"good": 1, "bad": 0, "correct": True}],
                                                 "good", "bad", "auroc", n_boot=20))


if __name__ == "__main__":
    unittest.main()
