"""Empty-question inference regressions; tiny CPU fixture, no training or downloads.

Run: python tests/test_empty_questions.py
"""
import copy
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
# `laya/onnx_agent.py` imports onnxruntime inside a function, not at module scope, so this
# import is safe in an environment without it -- which is the case for the Windows CI job that
# runs this file. ONNXAgent.__new__ skips __init__, so no session and no checkpoint are needed.
from laya.onnx_agent import ONNXAgent  # noqa: E402


class _FakeOnnxTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"          # _encode_state does serialize_state(...).replace(mask_token, " ")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(c) % 40) for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _StubOnnxSession:
    """Row logits from the row's own tokens. No onnxruntime, no weights, no network."""

    def run(self, names, inputs):
        n = len(inputs["input_ids"])
        w = np.full(n, 0.6, dtype=np.float32)
        logits = np.stack([w, np.ones_like(w)], axis=1).astype(np.float32)
        act = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (n, 1))
        return [logits, act]


def _bare_onnx_agent():
    a = ONNXAgent.__new__(ONNXAgent)
    a.model_id = "laya-rl-agent-onnx"
    a.cfg = {"max_len": 64, "head_max_len": 32}
    a.tok = _FakeOnnxTok()
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    a.session = _StubOnnxSession()
    a.hooks = ()
    a.hooks_raise = True
    a.hooks_concurrent = True
    a.hooks_timeout = None
    a._hooks_lock = None
    a._hooks_mutex = threading.Lock()
    a.lang_temperatures = {}
    return a


class EmptyQuestionsTests(unittest.TestCase):
    def setUp(self):
        # Supply the loaded runtime attributes in memory; no checkpoint is needed.
        self.agent = Agent.__new__(Agent)
        self.agent.cfg = {"max_len": 64, "head_max_len": 32}
        self.agent.device = torch.device("cpu")
        self.agent.dtype = torch.float32
        self.agent.temperature = [1.0, 1.0, 1.0]
        self.agent.temperature_by_options = {}
        self.agent.tok = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]",
        )
        config = BertConfig(vocab_size=6, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=1, intermediate_size=32)
        self.agent.model = DecisionModel(BertModel(config), head_layers=0).eval()
        self.empty = {"model": "laya-rl-agent", "answers": {},
                      "usage": {"input_tokens": 0, "output_tokens": 0}}
        # The same runtime surface, built for the ONNX backend. Both are exercised by the
        # parity tests below, which is the point: a guard that exists on only one backend is
        # invisible to a suite that looks at one.
        self.onnx = _bare_onnx_agent()

    def test_both_methods_return_empty_response_for_supported_states(self):
        for name in ("predict", "system_one"):
            method = getattr(self.agent, name)
            for state in ("hello", {"text": "hello"}, [{"role": "user", "content": "hello"}], "", {}, []):
                with self.subTest(method=name, state=state):
                    original = copy.deepcopy(state)
                    questions = {}
                    self.assertEqual(method(state, questions), self.empty)
                    self.assertEqual(state, original)
                    self.assertEqual(questions, {})

    def test_empty_questions_skip_tokenization_batching_and_forward(self):
        self.agent.tok = Mock(side_effect=AssertionError("unexpected tokenization"))
        self.agent.model = Mock(side_effect=AssertionError("unexpected forward pass"))
        with patch("laya.agent.build_sequence", side_effect=AssertionError("unexpected encoding")) as encode, \
                patch("laya.agent.collate_items", side_effect=AssertionError("unexpected batching")) as collate:
            self.assertEqual(self.agent.predict("hello", {}), self.empty)
            self.assertEqual(self.agent.system_one("hello", {}), self.empty)
        encode.assert_not_called()
        collate.assert_not_called()
        self.agent.tok.assert_not_called()
        self.agent.model.assert_not_called()

    def test_empty_responses_do_not_share_mutable_containers(self):
        result = self.agent.predict("hello", {})
        result["answers"]["changed"] = True
        result["usage"]["input_tokens"] = 7
        self.assertEqual(self.agent.system_one("hello", {}), self.empty)

    def test_non_dict_questions_raise_a_clear_type_error(self):
        # `list(questions.keys())` used to raise `AttributeError: 'NoneType' object has no
        # attribute 'keys'` (or the list/str equivalent) from three frames down, naming neither
        # the argument nor the fix. The core API is the one surface that did not validate this;
        # serve.py, shortlist.py and evals.py all already do.
        for name in ("predict", "system_one"):
            method = getattr(self.agent, name)
            for bad in (None, [], "not-a-dict"):
                with self.subTest(method=name, questions=bad):
                    with self.assertRaises(TypeError) as cm:
                        method("hello", bad)
                    self.assertIn("questions must be a dict", str(cm.exception))

    def test_none_state_raises_instead_of_answering_the_literal_null(self):
        # `serialize_state(None)` is `json.dumps(None)` == "null", so a missing state was
        # answered as a decision about the literal text "null" -- byte-identical to passing
        # `"null"` -- at full confidence. Reject it before serialization.
        for name in ("predict", "system_one"):
            method = getattr(self.agent, name)
            with self.subTest(method=name):
                with self.assertRaises(TypeError) as cm:
                    method(None, {"q": {"type": "noul", "instructions": "Is it true?"}})
                self.assertIn("state must not be None", str(cm.exception))

    def _both_backends(self, state, questions):
        """(torch exception, onnx exception) as (class name, message) pairs."""
        def call(agent):
            try:
                agent.predict(state, questions)
            except BaseException as exc:  # noqa: BLE001
                return type(exc).__name__, str(exc)
            return "<none>", ""
        return call(self.agent), call(self.onnx)

    def test_onnx_rejects_exactly_what_torch_rejects(self):
        # The two guards above were added in 86fe2a1 to `Agent.predict_batch` only.
        # `git log -S "state must not be None" -- laya/` returns that single commit, and
        # laya/onnx_agent.py was never updated: it kept the first of the three checks and a
        # comment claiming parity with Agent.predict_batch. Two consequences, both measured
        # before the fix:
        #
        #   ONNXAgent.predict(None, Q)  -> a confident answer byte-identical to
        #                                   ONNXAgent.predict("null", Q), because
        #                                   serialize_state(None) is json.dumps(None)
        #   ONNXAgent.predict("hi", []) -> AttributeError: 'list' object has no attribute
        #                                   'keys', three frames down, naming neither the
        #                                   argument nor the fix
        #
        # The second matters over HTTP as well as in-process: laya/serve.py classifies only
        # ValueError as a caller error, so the request torch answers with 422 became a 500
        # "inference failed" whenever the ONNX backend was attached.
        #
        # Both backends must raise the same class with the same message, so the assertion is
        # parity rather than "ONNX now raises something".
        for label, (state, questions) in {
            "state=None": (None, {"q": {"type": "noul", "instructions": "Is it true?"}}),
            "questions=None": ("hello", None),
            "questions=[]": ("hello", []),
            "questions=str": ("hello", "not-a-dict"),
        }.items():
            with self.subTest(case=label):
                (t_kind, t_msg), (o_kind, o_msg) = self._both_backends(state, questions)
                self.assertEqual(t_kind, "TypeError", "torch regressed: %s" % t_kind)
                self.assertEqual(o_kind, t_kind,
                                 "ONNX raised %s: %s where torch raised %s" % (o_kind, o_msg, t_kind))
                self.assertEqual(o_msg, t_msg)

    def test_onnx_still_answers_valid_input_and_honours_a_start_hook(self):
        # Negative space: the guards are in the `if ctx.results is None` branch, exactly where
        # Agent's are, so a start hook that answers still short-circuits them and a valid
        # request is untouched. Without this, a fix that validated too eagerly would pass the
        # test above and break both of these.
        questions = {"q": {"type": "noul", "instructions": "Is it true?"}}
        answer = self.onnx.predict("hello", questions)
        self.assertEqual(answer["model"], "laya-rl-agent-onnx")
        self.assertIn("q", answer["answers"])
        self.assertGreater(answer["usage"]["input_tokens"], 0)
        self.assertEqual(answer["usage"]["output_tokens"], 0)

        def answered(ctx):
            ctx.results = [self.empty]

        for agent in (self.agent, self.onnx):
            with self.subTest(model=agent.model_id):
                # state=None is invalid, but the hook answers before validation is reached.
                # predict/system_one return the single result, so the hook's one-element list
                # is unwrapped by the same path on both backends.
                self.assertEqual(agent.predict(None, questions, on_predict_start=answered),
                                 self.empty)

    def test_nonempty_predictions_are_unchanged_after_empty_call(self):
        questions = {
            "choice": {"type": "choice", "instructions": "Pick one", "criteria": ["yes", "no"]},
            "score": {"type": "score", "instructions": "Rate it", "criteria": ["low", "medium", "high"]},
            "noul": {"type": "noul", "instructions": "Is it true?"},
        }
        original = copy.deepcopy(questions)
        with patch.object(self.agent.model, "forward", wraps=self.agent.model.forward) as forward:
            before = self.agent.predict("hello", questions)
            self.assertEqual(self.agent.predict("hello", {}), self.empty)
            after = self.agent.system_one("hello", questions)
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(before, after)
        self.assertEqual(questions, original)
        self.assertEqual(before["model"], "laya-rl-agent")
        self.assertGreater(before["usage"]["input_tokens"], 0)
        self.assertEqual(before["usage"]["output_tokens"], 0)
        answers = before["answers"]
        self.assertEqual(set(answers), set(questions))
        for qid in ("choice", "score"):
            self.assertEqual(answers[qid]["type"], qid)
            self.assertAlmostEqual(sum(answers[qid]["probabilities"].values()), 1.0, delta=0.0002)
        self.assertIn(answers["choice"]["choice"], ("yes", "no"))
        self.assertTrue(0 <= answers["score"]["score"] <= 2)
        self.assertEqual(answers["noul"]["type"], "noul")
        self.assertTrue(0 <= answers["noul"]["noul"] <= 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
