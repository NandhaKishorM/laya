"""Temperature loading regressions using a tiny local checkpoint; no training or downloads.

Run: python tests/test_temperature_loading.py
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import load  # noqa: E402
from laya.common import TEMP_MAX, TEMP_MIN, DecisionModel, temperature_rejected  # noqa: E402


class TemperatureLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.repo = Path(cls.tmp.name)
        config = BertConfig(vocab_size=6, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=1, intermediate_size=32)
        config.save_pretrained(cls.repo / "encoder")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]",
        )
        tokenizer.save_pretrained(cls.repo / "tokenizer")
        model = DecisionModel(BertModel(config), head_layers=0)
        save_file(model.state_dict(), cls.repo / "model.safetensors")

    def load_config(self, **temperatures):
        cfg = {"encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
               "max_len": 64, "head_max_len": 32, **temperatures}
        path = self.repo / "rl_agent_config.json"
        path.write_text(json.dumps(cfg))
        original = path.read_bytes()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
                agent = load(str(self.repo), device="cpu")
        self.assertEqual(agent.device.type, "cpu")
        self.assertEqual(path.read_bytes(), original)
        # JSON comparison also handles NaN, which is unequal to itself.
        self.assertEqual(json.dumps(agent.cfg), json.dumps(cfg))
        self.assertEqual(json.dumps(agent.temperature_raw), json.dumps(cfg.get("temperature", [1.0] * 3)))
        self.assertEqual(json.dumps(agent.temperature_by_options_raw), json.dumps(cfg.get("temperature_by_options", {})))
        messages = [w for w in caught if str(w.message).startswith("laya:")]
        self.assertTrue(all(issubclass(w.category, RuntimeWarning) for w in messages))
        return agent, [str(w.message) for w in messages]

    def assert_predictions(self, agent, choice, score, noul):
        questions = {
            "choice": {"type": "choice", "instructions": "Pick one", "criteria": ["a", "b"]},
            "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]},
            "noul": {"type": "noul", "instructions": "True?"},
        }
        # Loading is real; fixed logits make the expected scaling independent of random weights.
        logits = torch.tensor([[0.0, 1.0, -1e4], [0.0, 1.0, 2.0], [0.0, 1.0, -1e4]])
        with patch.object(agent.model, "forward", return_value=(logits, torch.zeros(3, 2))):
            answers = agent.predict("hello", questions)["answers"]
        for name, temperature, k in (("choice", choice, 2), ("score", score, 3), ("noul", noul, 2)):
            p = torch.softmax(torch.arange(k, dtype=torch.float32) / temperature, -1)
            if name == "noul":
                self.assertAlmostEqual(answers[name]["noul"], p[1].item(), delta=0.0001)
            else:
                actual = list(answers[name]["probabilities"].values())
                self.assertEqual(len(actual), k)
                for got, want in zip(actual, p.tolist()):
                    self.assertAlmostEqual(got, want, delta=0.0001)

    def test_invalid_type_entries_use_neutral_fallback_and_warn(self):
        for value in (None, "invalid", "", [], {}):
            with self.subTest(value=value):
                agent, messages = self.load_config(temperature=[value, 2.0, 3.0])
                self.assertEqual(agent.temperature, [1.0, 2.0, 3.0])
                self.assertEqual(len(messages), 1)
                self.assertIn("temperature[0]", messages[0])
                self.assertIn(repr(value), messages[0])
                self.assertIn("uncalibrated", messages[0])
                self.assert_predictions(agent, choice=1.0, score=2.0, noul=3.0)

    def test_invalid_bucket_entries_use_neutral_fallback_and_keep_precedence(self):
        for value in (None, "invalid", "", [], {}):
            with self.subTest(value=value):
                agent, messages = self.load_config(temperature=[2.0, 3.0, 4.0],
                                                  temperature_by_options={"choice:2": value})
                self.assertEqual(agent.temperature_by_options, {"choice:2": 1.0})
                self.assertEqual(len(messages), 1)
                self.assertIn("choice:2", messages[0])
                self.assertIn(repr(value), messages[0])
                self.assert_predictions(agent, choice=1.0, score=3.0, noul=4.0)

    def test_boolean_entries_use_neutral_fallback_and_warn(self):
        # A JSON `true`/`false` in a checkpoint config is not a temperature: `clamp_temperature`
        # refuses a bool and installs the neutral 1.0. It loaded in silence for `true`, though,
        # because the diagnostic compared `float(raw)` with the applied value and `float(True)` is
        # exactly that 1.0 -- while `False` floated to 0.0, mismatched, and was reported.
        for value in (True, False):
            with self.subTest(value=value):
                agent, messages = self.load_config(temperature=[value, 2.0, 3.0],
                                                  temperature_by_options={"choice:2": value})
                self.assertEqual(agent.temperature, [1.0, 2.0, 3.0])
                self.assertEqual(agent.temperature_by_options, {"choice:2": 1.0})
                self.assertEqual(len(messages), 1)
                self.assertIn("temperature[0]", messages[0])
                self.assertIn("choice:2", messages[0])
                self.assertIn(repr(value), messages[0])
                self.assertIn("uncalibrated", messages[0])
                self.assert_predictions(agent, choice=1.0, score=2.0, noul=3.0)

    def test_nonfinite_values_keep_neutral_fallback_and_warn(self):
        for value in (float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                agent, messages = self.load_config(temperature=[value, 2.0, 3.0],
                                                  temperature_by_options={"noul:2": value})
                self.assertEqual(agent.temperature, [1.0, 2.0, 3.0])
                self.assertEqual(agent.temperature_by_options, {"noul:2": 1.0})
                self.assertEqual(len(messages), 1)
                self.assertIn("temperature[0]", messages[0])
                self.assertIn("noul:2", messages[0])
                self.assert_predictions(agent, choice=1.0, score=2.0, noul=1.0)

    def test_overflowing_integer_entries_use_neutral_fallback_and_warn(self):
        # JSON has no double limit, so a checkpoint file can carry a temperature that is an exact
        # integer and still not a number `float()` can produce. `clamp_temperature`'s docstring
        # promises the neutral 1.0 for anything that is not a number and `calibrate.load_calibration`
        # promises "a non-numeric or out-of-range entry cannot crash" the load; both held for a
        # string, a list or an infinity, and broke on this one value.
        for value in (10 ** 400, -(10 ** 400)):
            with self.subTest(value=value):
                agent, messages = self.load_config(temperature=[value, 2.0, 3.0],
                                                   temperature_by_options={"choice:2": value})
                self.assertEqual(agent.temperature, [1.0, 2.0, 3.0])
                self.assertEqual(agent.temperature_by_options, {"choice:2": 1.0})
                self.assertEqual(len(messages), 1)
                self.assertIn("temperature[0]", messages[0])
                self.assertIn("choice:2", messages[0])
                self.assertIn(repr(value), messages[0])
                self.assert_predictions(agent, choice=1.0, score=2.0, noul=3.0)

    def test_out_of_range_values_keep_existing_clamps(self):
        agent, messages = self.load_config(temperature=[0.1, 10, -1],
                                          temperature_by_options={"choice:2": 9, "score:3-5": 0})
        self.assertEqual(agent.temperature, [0.5, 5.0, 0.5])
        self.assertEqual(agent.temperature_by_options, {"choice:2": 5.0, "score:3-5": 0.5})
        self.assertEqual(len(messages), 1)
        for entry in ("temperature[0]", "temperature[1]", "temperature[2]", "choice:2", "score:3-5"):
            self.assertIn(entry, messages[0])
        self.assert_predictions(agent, choice=5.0, score=0.5, noul=0.5)

    def test_valid_numbers_and_numeric_strings_do_not_warn(self):
        agent, messages = self.load_config(temperature=[0.5, "2", 5],
                                          temperature_by_options={"choice:2": "1.5", "score:3-5": 2.5, "noul:2": "5.0"})
        self.assertEqual(agent.temperature, [0.5, 2.0, 5.0])
        self.assertEqual(agent.temperature_by_options, {"choice:2": 1.5, "score:3-5": 2.5, "noul:2": 5.0})
        self.assertEqual(messages, [])
        self.assert_predictions(agent, choice=1.5, score=2.5, noul=5.0)

    def test_missing_bucket_uses_corresponding_type(self):
        agent, messages = self.load_config(temperature=[2.0, 3.0, 4.0], temperature_by_options={"choice:2": 1.5})
        self.assertEqual(messages, [])
        self.assert_predictions(agent, choice=1.5, score=3.0, noul=4.0)

    def test_missing_temperature_fields_keep_defaults(self):
        agent, messages = self.load_config()
        self.assertEqual(agent.temperature, [1.0, 1.0, 1.0])
        self.assertEqual(agent.temperature_by_options, {})
        self.assertEqual(messages, [])
        self.assert_predictions(agent, choice=1.0, score=1.0, noul=1.0)


class TemperatureRejectionPredicateTests(unittest.TestCase):
    """The diagnostic every loader shares, without a checkpoint to build.

    `Agent.__init__`, `ONNXAgent.__init__` and `calibrate._rejected_temperatures` all ask this one
    question, so the table below is the ONNX site's coverage too -- that suite needs `onnxruntime`,
    which is not installed for a plain checkout.
    """

    def test_rejection_matches_what_the_clamp_actually_accepted(self):
        cases = [
            # A bool is not a number to `clamp_temperature`, and `float(True)` is exactly the 1.0
            # it installs instead, so an equality check alone called both of them accepted.
            (True, 1.0, True),
            (False, 1.0, True),
            (1.5, 1.5, False),
            (2, 2.0, False),
            ("1.5", 1.5, False),
            (0.1, TEMP_MIN, True),
            (9.0, TEMP_MAX, True),
            (None, 1.0, True),
            ("x", 1.0, True),
            ([], 1.0, True),
            ({}, 1.0, True),
            (float("nan"), 1.0, True),
            (float("inf"), 1.0, True),
            # Not a number `float()` can make either: JSON's arbitrary-precision integer literal
            # overflows the conversion, which the diagnostic used to raise instead of report.
            (10 ** 400, 1.0, True),
            (-(10 ** 400), 1.0, True),
        ]
        for raw, applied, want in cases:
            with self.subTest(raw=raw, applied=applied):
                self.assertIs(temperature_rejected(raw, applied), want)

    def test_no_loader_rederives_the_comparison(self):
        # Three loaders each carried their own copy of the predicate, and one copy drifting from
        # `clamp_temperature` is how a `true` temperature loaded unreported. Walk the package
        # rather than naming files, so a re-derived fourth copy is caught the same way.
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "laya").rglob("*.py"))
        scanned = [p.relative_to(root).as_posix() for p in paths]
        # Name one file the walk has to find, or a walk that silently matches nothing would report
        # "no offenders" and guard nothing.
        self.assertIn("laya/agent.py", scanned)
        offenders = [name for name, p in zip(scanned, paths)
                     # An explicit encoding is not decoration: `laya/email.py` and `laya/lang.py`
                     # carry non-ASCII bytes, so a locale-default read dies on Windows' cp1252
                     # before the scan can say anything about them.
                     if "if float(raw) == applied:" in p.read_text(encoding="utf-8")]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
