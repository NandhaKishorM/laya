"""Empty-question inference regressions; tiny CPU fixture, no training or downloads.

Run: python tests/test_empty_questions.py
"""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel  # noqa: E402


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
