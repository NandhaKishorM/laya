"""Question encoding reuse: run with python tests/test_question_token_reuse.py (no weights)."""
import copy
import contextvars
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent  # noqa: E402
from laya.common import _reuse_question_tokens, build_sequence  # noqa: E402


class Tokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __init__(self, offset=0):
        self.calls = []
        self.offset = offset

    def __call__(self, text, **kwargs):
        self.calls.append((threading.get_ident(), text, kwargs))
        ids = [ord(c) + 4 + self.offset for c in text]
        if kwargs.get("truncation"):
            ids = ids[:kwargs["max_length"]]
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 4 - self.offset) for i in ids)


QUESTION = {"t": "choice", "ins": "Which team?", "crit": {"billing": "charges", "tech": "errors"}}
QUESTIONS = {"team": {"type": "choice", "instructions": QUESTION["ins"], "criteria": QUESTION["crit"]}}


def agent_with(tok):
    agent = Agent.__new__(Agent)
    agent.tok = tok
    agent.cfg = {"max_len": 512, "head_max_len": 192}
    agent.batches = []

    def forward(batch):
        agent.batches.append({key: value.clone() for key, value in batch.items() if hasattr(value, "clone")})
        shape = batch["marker_pos"].shape
        return np.zeros(shape, dtype=np.float32), np.full((shape[0], 2), 0.5, dtype=np.float32)

    def decode(logits, act, items, ids, internal, offset, **kwargs):
        return {qid: {"ids": item["ids"], "markers": item["markers"], "answer_confidence": 0.5}
                for qid, item in zip(ids, items)}

    agent._forward = forward
    agent._decode_answers = decode
    return agent


class QuestionTokenReuse(unittest.TestCase):
    def test_reuse_across_chunks_preserves_all_collated_tensors(self):
        states = ["state %d" % i for i in range(19)]
        tok = Tokenizer()
        agent = agent_with(tok)
        for grouped in (False, True):
            for batch_size in (None, 1, 3):
                with self.subTest(grouped=grouped, batch_size=batch_size):
                    tok.calls.clear()
                    agent.batches.clear()
                    got = agent.predict_batch(states, QUESTIONS, batch_size=batch_size, sort_by_length=grouped)
                    self.assertEqual(len(tok.calls), len(states) + 3)
                    cached = agent.batches[:]
                    agent.batches.clear()
                    # The real undecorated method is the same orchestration without reuse.
                    plain = Agent.predict_batch.__wrapped__.__wrapped__(
                        agent, states, QUESTIONS, batch_size=batch_size, sort_by_length=grouped)
                    self.assertEqual(got, plain)
                    self.assertEqual(len(cached), len(agent.batches))
                    for left, right in zip(cached, agent.batches):
                        self.assertEqual(set(left), set(right))
                        for key in left:
                            self.assertTrue(left[key].equal(right[key]), key)

    def test_predict_long_uses_the_same_cache_for_all_windows(self):
        tok = Tokenizer()
        agent = agent_with(tok)
        scanned = agent.predict_long("x" * 1000, QUESTIONS, window=80, stride=50, batch_size=3)
        head = "choice question: " + QUESTION["ins"]
        # Twice, not once per window: predict_long assembles each question's head once on its own
        # to measure the room the question leaves for the state -- the cap on the window -- and the
        # scan that follows then shares one cache across every window it scores. The point of this
        # test is the second number: the head is encoded a fixed number of times, not once per
        # window, which the assertion below states as such.
        self.assertEqual(sum(text == head for _, text, _ in tok.calls), 2)
        self.assertEqual(sum(text == " billing: charges" for _, text, _ in tok.calls), 2)
        self.assertGreater(scanned["usage"]["windows"], 2)
        self.assertGreater(len(agent.batches), 1)

    def test_sequence_parity_for_rendering_order_and_budgets(self):
        cases = [
            QUESTION,
            {"t": "choice", "ins": "a [MASK] b", "crit": {7: None, "zero": 0, "false": False}},
            {"t": "choice", "ins": "中文", "crit": {"a": {"desc": [1, "二"]}, "b": "z" * 200}},
            {"t": "score", "ins": "level", "crit": [{"desc": "one"}, ["two"], 0]},
            {"t": "noul", "ins": "holds?", "crit": {"false": {"reason": "no"}, "true": ["yes"]},
             "labels": {"false": "否", "true": "是"}},
        ]
        tok = Tokenizer()
        for q in cases:
            for max_len, head_len in ((512, 192), (128, 32), (16, 8)):
                for left in (False, True):
                    for order in (None, list(reversed(range(len(q.get("crit", {})))))):
                        with self.subTest(q=q, max_len=max_len, left=left, order=order):
                            kwargs = dict(max_len=max_len, head_max_len=head_len,
                                          option_order=order, truncate_left=left)
                            expected = build_sequence(tok, [{"text": "x" * 600}], q, **kwargs)

                            @_reuse_question_tokens
                            def repeated():
                                first = build_sequence(tok, [{"text": "x" * 600}], q, **kwargs)
                                second = build_sequence(tok, [{"text": "x" * 600}], q, **kwargs)
                                return first, second

                            self.assertEqual(repeated(), (expected, expected))

    def test_state_encodings_are_not_cached(self):
        tok = Tokenizer()
        agent_with(tok).predict_batch(["same"] * 20, QUESTIONS, batch_size=2)
        self.assertEqual(sum(text == "same" for _, text, _ in tok.calls), 20)
        self.assertEqual(len(tok.calls), 23)

    def test_single_state_skips_cache_but_a_hook_can_expand_it(self):
        tok = Tokenizer()
        agent = agent_with(tok)
        duplicate = dict(QUESTIONS, another=QUESTIONS["team"])
        agent.predict_batch(["state"], duplicate)
        self.assertEqual(len(tok.calls), 7)  # one state plus both uncached three-tokenizer-call prefixes
        tok.calls.clear()

        def expand(ctx):
            ctx.states = ["state", "other"]

        agent.predict_batch(["state"], duplicate, on_predict_start=expand)
        self.assertEqual(len(tok.calls), 5)  # two states plus one shared prefix

    def test_new_call_reencodes_and_sees_question_changes(self):
        tok = Tokenizer()
        agent = agent_with(tok)
        questions = copy.deepcopy(QUESTIONS)
        before = agent.predict_batch(["state"] * 2, questions)
        first_count = len(tok.calls)
        self.assertEqual(agent.predict_batch(["state"] * 2, questions), before)
        self.assertEqual(len(tok.calls), first_count * 2)
        questions["team"]["criteria"]["billing"] = {"rubric": "refunds"}
        after = agent.predict_batch(["state"] * 2, questions)
        self.assertNotEqual(before, after)
        self.assertTrue(any("refunds" in text for _, text, _ in tok.calls))

    def test_start_hook_question_rewrite_is_encoded(self):
        tok = Tokenizer()
        agent = agent_with(tok)

        def rewrite(ctx):
            ctx.questions = copy.deepcopy(QUESTIONS)
            ctx.questions["team"]["instructions"] = "Replacement"

        agent.predict_batch(["state"] * 2, QUESTIONS, on_predict_start=rewrite)
        self.assertEqual(sum(text == "choice question: Replacement" for _, text, _ in tok.calls), 1)
        self.assertFalse(any(text == "choice question: Which team?" for _, text, _ in tok.calls))

    def test_nested_call_restores_outer_cache_and_exception_discards_it(self):
        tok = Tokenizer()

        @_reuse_question_tokens
        def inner():
            build_sequence(tok, "state", QUESTION, state_ids=[])

        @_reuse_question_tokens
        def outer(fail=False):
            build_sequence(tok, "state", QUESTION, state_ids=[])
            inner()
            build_sequence(tok, "state", QUESTION, state_ids=[])
            if fail:
                raise RuntimeError("failure")

        outer()
        self.assertEqual(len(tok.calls), 6)
        with self.assertRaisesRegex(RuntimeError, "failure"):
            outer(fail=True)
        self.assertEqual(len(tok.calls), 12)
        build_sequence(tok, "state", QUESTION, state_ids=[])
        self.assertEqual(len(tok.calls), 15)

    def test_threads_sharing_a_tokenizer_have_separate_call_caches(self):
        tok = Tokenizer()
        barrier = threading.Barrier(2)

        @_reuse_question_tokens
        def run():
            first = build_sequence(tok, "state", QUESTION, state_ids=[])
            barrier.wait(timeout=10)
            second = build_sequence(tok, "state", QUESTION, state_ids=[])
            return first, second

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: run(), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(tok.calls), 6)
        self.assertEqual(len({tid for tid, _, _ in tok.calls}), 2)

    def test_tokenizer_identity_and_mutated_sequences_do_not_leak(self):
        first, second = Tokenizer(), Tokenizer(offset=100)
        expected = build_sequence(first, "state", QUESTION, state_ids=[])

        @_reuse_question_tokens
        def run():
            ids, markers = build_sequence(first, "state", QUESTION, state_ids=[])
            ids[:] = [999]
            markers[:] = []
            again = build_sequence(first, "state", QUESTION, state_ids=[])
            other = build_sequence(second, "state", QUESTION, state_ids=[])
            return again, other

        again, other = run()
        self.assertEqual(again, expected)
        self.assertNotEqual(other, expected)
        self.assertEqual(len(second.calls), 3)

    def test_copied_hook_context_cannot_keep_a_finished_call_cache(self):
        tok = Tokenizer()

        @_reuse_question_tokens
        def run():
            build_sequence(tok, "state", QUESTION, state_ids=[])
            return contextvars.copy_context()

        copied = run()
        copied.run(build_sequence, tok, "state", QUESTION, state_ids=[])
        self.assertEqual(len(tok.calls), 6)

    def test_copied_hook_context_does_not_share_tokens_with_a_worker(self):
        tok = Tokenizer()

        @_reuse_question_tokens
        def run():
            build_sequence(tok, "state", QUESTION, state_ids=[])
            copied = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(copied.run, build_sequence, tok, "state", QUESTION, state_ids=[]).result()
            build_sequence(tok, "state", QUESTION, state_ids=[])

        run()
        self.assertEqual(len(tok.calls), 6)


if __name__ == "__main__":
    unittest.main()
