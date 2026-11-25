"""Consistency tests for predict_batch: batched results must match single-request results.

Covers Agent.predict_batch and the merged Router.predict_batch (#138), including the
grouping edge that matters for the merge bar (batched == single): a question schema
that arrives with a different key insertion order must be scored with its own order.

Run: python -m pytest tests/test_predict_batch.py
Tiny local weights, no network required.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya import load, Router  # noqa: E402
from laya.common import DecisionModel  # noqa: E402


def build_tiny_repo(repo: Path):
    """A tiny but valid checkpoint: Bert encoder + decision head, WordLevel tokenizer."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    repo.mkdir(parents=True)
    config = BertConfig(vocab_size=6, hidden_size=64, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=128)
    config.save_pretrained(repo / "encoder")
    tok_obj = Tokenizer(WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
        unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    tokenizer.save_pretrained(repo / "tokenizer")
    model = DecisionModel(BertModel(config), head_layers=0)
    save_file(model.state_dict(), repo / "model.safetensors")
    (repo / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
        "max_len": 64, "head_max_len": 32,
    }))


QUESTIONS = {
    "intent": {"type": "choice", "instructions": "Which intent?",
               "criteria": {"greet": "a greeting", "farewell": "a goodbye"}},
    "polarity": {"type": "score", "instructions": "How positive?",
                 "criteria": ["negative", "neutral", "positive"]},
    "mentions_hello": {"type": "noul", "instructions": "Does the state say hello?"},
}
STATES = ["hello", "goodbye", {"text": "hello there"}, "goodbye and hello"]


class PredictBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        build_tiny_repo(Path(cls.tmp.name) / "repo")
        cls.agent = load(str(Path(cls.tmp.name) / "repo"), device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _flatten(result):
        """(qid -> (type, probabilities-or-value, confidence)) rounded, for comparison.

        The argmax label is intentionally excluded: on the untrained tiny model the
        choice probabilities sit near 0.5, and fp batch-shape noise may flip that
        coin flip. The distribution is the invariant; the label is derivable from it.
        """
        out = {}
        for qid, ans in result["answers"].items():
            if ans["type"] in ("choice", "score"):
                probs = tuple(round(v, 5) for v in ans["probabilities"].values())
            else:
                probs = round(ans["noul"], 4)
            out[qid] = (ans["type"], probs, ans["confidence"])
        return out

    def test_batch_matches_single(self):
        """Every batched answer must equal the single-request answer (fp32 CPU:
        identical up to float tail noise, never a different decision)."""
        batched = self.agent.predict_batch(list(STATES), QUESTIONS)
        self.assertEqual(len(batched), len(STATES))
        for state, result in zip(STATES, batched):
            single = self.agent.system_one(state, QUESTIONS)
            self.assertEqual(self._flatten(single), self._flatten(result),
                             "mismatch for state %r" % (state,))
            self.assertEqual(result["usage"]["input_tokens"],
                             single["usage"]["input_tokens"])

    def test_batch_sizes_are_consistent(self):
        """Adding more states to a batch must not change earlier answers."""
        small = self.agent.predict_batch(list(STATES[:2]), QUESTIONS)
        full = self.agent.predict_batch(list(STATES), QUESTIONS)
        for a, b in zip(small, full):
            self.assertEqual(self._flatten(a), self._flatten(b))

    def test_empty_batch(self):
        self.assertEqual(self.agent.predict_batch([], QUESTIONS), [])

    def test_rejects_single_state(self):
        with self.assertRaises(TypeError):
            self.agent.predict_batch("hello", QUESTIONS)


class RouterPredictBatchTests(unittest.TestCase):
    """Router.predict_batch: route per request, group by (checkpoint, schema), share passes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        repo = Path(cls.tmp.name) / "repo"
        build_tiny_repo(repo)
        for sub in ("multilingual",):
            (repo / sub).mkdir()
            for f in ("rl_agent_config.json", "model.safetensors"):
                shutil.copyfile(repo / f, repo / sub / f)
            shutil.copytree(repo / "tokenizer", repo / sub / "tokenizer")
            shutil.copytree(repo / "encoder", repo / sub / "encoder")
        cls.router = Router(models={"english": str(repo), "multilingual": str(repo / "multilingual")},
                            device="cpu", max_loaded=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _requests(states, questions=QUESTIONS):
        return [{"state": s, "questions": questions} for s in states]

    def test_mixed_language_batch_routes_and_preserves_order(self):
        # Sentences rather than single words: a bare "hello" names no language, so since 0.4.0 it
        # follows `default` and all three rows would group together, which would stop this
        # testing that a mixed batch routes per item at all.
        states = ["Please refund the duplicate charge on my invoice",   # identified en -> english
                  "नमस्ते",                                               # devanagari -> multilingual
                  "I would like to close my account today please"]     # identified en -> english
        results = self.router.predict_batch(self._requests(states))
        self.assertEqual(len(results), 3)
        self.assertEqual([r["routing"]["model"] for r in results],
                         ["english", "multilingual", "english"])
        for result in results:
            self.assertIn("answers", result)
            self.assertEqual(set(result["answers"]), set(QUESTIONS))
        # order preserved: same input -> same answer as calling predict() one by one
        for result, state in zip(results, states):
            single = self.router.predict(state, QUESTIONS)
            self.assertEqual(PredictBatchTests._flatten(single), PredictBatchTests._flatten(result))

    def test_equal_questions_with_different_option_order_match_single(self):
        """Regression for the grouping key: two requests whose questions are equal but
        keyed in a different order must NOT share a group. Options are positional in
        the rendered sequence, so the second request would otherwise be scored with
        the first caller's option order and disagree with its own single-request
        answer -- the failure mode reviewed out of the pre-#138 grouping keys that
        serialised questions to JSON with sort_keys=True and read them back."""
        ordered = {"intent": {"type": "choice", "instructions": "Which one?",
                              "criteria": {"zulu": "last alphabetically",
                                           "mike": "in the middle",
                                           "alpha": "first alphabetically"}}}
        reordered = {"intent": {"type": "choice", "instructions": "Which one?",
                                "criteria": {"alpha": "first alphabetically",
                                             "mike": "in the middle",
                                             "zulu": "last alphabetically"}}}
        requests = [{"state": "hello", "questions": ordered},
                    {"state": "hello", "questions": reordered}]
        batched = self.router.predict_batch(requests)
        self.assertEqual(len(batched), 2)
        for req, result in zip(requests, batched):
            single = self.router.predict(req["state"], req["questions"])
            self.assertEqual(PredictBatchTests._flatten(single), PredictBatchTests._flatten(result))
            # each label keeps the probability value its own option order produced
            for qid in single["answers"]:
                self.assertEqual(single["answers"][qid]["probabilities"],
                                 result["answers"][qid]["probabilities"])

    def test_interleaved_schemas_chunking_and_length_sort_order_match_single(self):
        """Interleaved schemas, multi-chunk execution, and length-sort order in Router.predict_batch.

        Verifies that when heterogeneous schemas are interleaved across requests:
        1. Router groups requests by schema, giving each group enough states (> batch_size)
           to trigger multiple forward-pass chunks and within-group length sorting.
        2. Within each group, states arrive in non-sorted length order. With sort_by_length=True,
           states are reordered by length internally and restored to the exact original input index.
        3. Every batched result matches its single-request predict() call in schema keys,
           token sequence usage (state_tokens, input_tokens), and answer probability distributions.
        """
        schema_a = {"intent": QUESTIONS["intent"]}
        schema_b = {"mentions_hello": QUESTIONS["mentions_hello"]}

        # Group A lengths: [12, 2, 6] (non-sorted)
        states_a = [
            "hello goodbye hello goodbye hello goodbye hello goodbye hello goodbye hello goodbye",
            "hello goodbye",
            "hello goodbye hello goodbye hello goodbye",
        ]
        # Group B lengths: [5, 14, 1] (non-sorted)
        states_b = [
            "goodbye hello goodbye hello goodbye",
            "hello goodbye hello goodbye hello goodbye hello goodbye hello goodbye hello goodbye hello goodbye",
            "hello",
        ]

        # Interleave schemas so neither group forms a contiguous batch
        requests = [
            {"state": states_a[0], "questions": schema_a},  # index 0: schema_a, 12 tokens
            {"state": states_b[0], "questions": schema_b},  # index 1: schema_b, 5 tokens
            {"state": states_a[1], "questions": schema_a},  # index 2: schema_a, 2 tokens
            {"state": states_b[1], "questions": schema_b},  # index 3: schema_b, 14 tokens
            {"state": states_a[2], "questions": schema_a},  # index 4: schema_a, 6 tokens
            {"state": states_b[2], "questions": schema_b},  # index 5: schema_b, 1 token
        ]

        for sort_by_length in (False, True):
            diag = {}
            batched = self.router.predict_batch(requests, batch_size=2, sort_by_length=sort_by_length,
                                                diagnostics=diag)
            self.assertEqual(len(batched), len(requests))

            if sort_by_length:
                # Both 3-item groups cleared 1 < batch_size < len(group) and reordered internally
                self.assertEqual(diag.get("sort_by_length", {}).get("groups"), 2)
                self.assertEqual(diag.get("sort_by_length", {}).get("reordered"), 2)
                self.assertEqual(diag.get("sort_by_length", {}).get("smallest"), 3)
                self.assertEqual(diag.get("sort_by_length", {}).get("largest"), 3)
            else:
                self.assertNotIn("sort_by_length", diag)

            for i, (req, result) in enumerate(zip(requests, batched)):
                single = self.router.predict(req["state"], req["questions"])
                # Exact schema keys match request
                self.assertEqual(set(result["answers"]), set(req["questions"]),
                                 f"Schema mismatch at index {i} (sort_by_length={sort_by_length})")
                # Token counts are distinct per state within each group, detecting same-schema swaps
                self.assertEqual(result["usage"]["state_tokens"], single["usage"]["state_tokens"],
                                 f"State token mismatch at index {i} (sort_by_length={sort_by_length})")
                self.assertEqual(result["usage"]["input_tokens"], single["usage"]["input_tokens"],
                                 f"Input token mismatch at index {i} (sort_by_length={sort_by_length})")
                # Probability distribution and confidence match single predict() call
                self.assertEqual(PredictBatchTests._flatten(single), PredictBatchTests._flatten(result),
                                 f"Prediction mismatch at index {i} (sort_by_length={sort_by_length})")
                for qid in single["answers"]:
                    if "probabilities" in single["answers"][qid]:
                        self.assertEqual(single["answers"][qid]["probabilities"],
                                         result["answers"][qid]["probabilities"])

    def test_empty_router_batch(self):
        self.assertEqual(self.router.predict_batch([]), [])


if __name__ == "__main__":
    unittest.main()
