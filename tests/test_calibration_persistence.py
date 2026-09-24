"""Notebook export and inference regression tests; CPU only, no training or downloads.

Run: python tests/test_calibration_persistence.py
"""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

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
from laya.common import DecisionModel, QTYPES  # noqa: E402


def export_notebook_config(cfg, fitted_temps, output_dir):
    """Execute the notebook's actual config export, without its GPU/training code."""
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / (
        "laya_finetune_typed_decisions_2xT4_kaggle.ipynb"
    )
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    script, = ["".join(c["source"]) for c in cells
               if "".join(c["source"]).startswith("%%writefile ")]
    # This contiguous tail includes the temperature update AND the JSON write.
    # Do not reproduce the export logic here: that would miss notebook regressions.
    start = script.index('        cfg["fine_tuned"] = True')
    end = script.index("\n    dist.destroy_process_group()", start)
    export = ast.parse(textwrap.dedent(script[start:end]))
    exec(compile(export, str(notebook), "exec"), {
        "cfg": cfg, "fitted_temps": fitted_temps, "output_dir": str(output_dir),
        "os": os, "json": json,
    })
    return json.loads((output_dir / "rl_agent_config.json").read_text())


class CalibrationPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.checkpoint = Path(cls.tmp.name)
        config = BertConfig(vocab_size=6, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=1, intermediate_size=32)
        config.save_pretrained(cls.checkpoint / "encoder")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]",
        )
        tokenizer.save_pretrained(cls.checkpoint / "tokenizer")
        model = DecisionModel(BertModel(config), head_layers=0)
        save_file(model.state_dict(), cls.checkpoint / "model.safetensors")

    def setUp(self):
        self.cfg = {
            "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
            "max_len": 128, "head_max_len": 96,
            "temperature": [1.0, 1.0, 1.0], "training": {"updates": 7},
        }
        self.fitted = [2.0, 3.0, 4.0]

    def assert_inference_temperatures(self, expected):
        # Use actual local config/tokenizer/weight loading; replace only logits so
        # the expected scaling is deterministic and independent of random weights.
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
            agent = load(str(self.checkpoint), device="cpu")
        self.assertEqual(agent.device.type, "cpu")
        questions = {}
        for qtype, k in expected:
            q = {"type": qtype, "instructions": "Pick one"}
            if qtype != "noul":
                q["criteria"] = [str(i) for i in range(k)]
            questions[f"{qtype}_{k}"] = q
        kmax = max(k for _, k in expected)
        logits = torch.arange(kmax, dtype=torch.float32).repeat(len(questions), 1)
        with patch.object(agent.model, "forward", return_value=(logits, torch.zeros(len(questions), 2))):
            answers = agent.predict("hello", questions)["answers"]
        for (qtype, k), temperature in expected.items():
            with self.subTest(qtype=qtype, k=k, temperature=temperature):
                p = torch.softmax(torch.arange(k, dtype=torch.float32) / temperature, -1)
                answer = answers[f"{qtype}_{k}"]
                if qtype == "noul":
                    self.assertAlmostEqual(answer["noul"], p[1].item(), delta=0.0001)
                else:
                    for i in range(k):
                        self.assertAlmostEqual(answer["probabilities"][str(i)], p[i].item(), delta=0.0001)
                    if qtype == "score":
                        self.assertAlmostEqual(answer["score"], (torch.arange(k) * p).sum().item(), delta=0.0001)

    def write_config(self):
        (self.checkpoint / "rl_agent_config.json").write_text(json.dumps(self.cfg))

    def test_new_export_removes_inherited_buckets_and_uses_fitted_types(self):
        self.cfg["temperature_by_options"] = {
            f"{qtype}:{bucket}": 1.5
            for qtype in QTYPES for bucket in ("2", "3-5", "6-10", "11+")
        }
        original = copy.deepcopy(self.cfg)
        saved = export_notebook_config(self.cfg, self.fitted, self.checkpoint)
        self.assert_inference_temperatures({
            (qtype, k): self.fitted[qt]
            for qtype, qt in QTYPES.items()
            for k in ((2,) if qtype == "noul" else (1, 2, 3, 5, 6, 10, 11))
        })
        self.assertFalse(saved.get("temperature_by_options"))
        self.assertEqual(saved["temperature"], self.fitted)
        self.assertTrue(saved["fine_tuned"])
        self.assertEqual(saved["model_name"], "laya-typed-decisions")
        for key in original.keys() - {"temperature", "temperature_by_options"}:
            self.assertEqual(saved[key], original[key])

    def test_export_accepts_absent_or_empty_bucket_map(self):
        for buckets in (None, {}):
            with self.subTest(buckets=buckets):
                if buckets is not None:
                    self.cfg["temperature_by_options"] = buckets
                saved = export_notebook_config(self.cfg, self.fitted, self.checkpoint)
                self.assertFalse(saved.get("temperature_by_options"))
                self.assertEqual(saved["temperature"], self.fitted)
                self.assert_inference_temperatures({(t, 2): self.fitted[qt] for t, qt in QTYPES.items()})

    def test_existing_bucket_calibration_keeps_precedence(self):
        self.cfg["temperature"] = self.fitted
        buckets = {"2": 0.75, "3-5": 1.25, "6-10": 1.5, "11+": 1.75}
        self.cfg["temperature_by_options"] = {
            f"{qtype}:{bucket}": value for qtype in QTYPES for bucket, value in buckets.items()
        }
        self.write_config()
        original = (self.checkpoint / "rl_agent_config.json").read_bytes()
        self.assert_inference_temperatures({
            (qtype, k): value for qtype in QTYPES
            for k, value in (((2, 0.75),) if qtype == "noul" else
                             ((1, 0.75), (2, 0.75), (3, 1.25), (5, 1.25),
                              (6, 1.5), (10, 1.5), (11, 1.75)))
        })
        self.assertEqual((self.checkpoint / "rl_agent_config.json").read_bytes(), original)

    def test_missing_bucket_falls_back_to_corresponding_type(self):
        self.cfg["temperature"] = self.fitted
        self.cfg["temperature_by_options"] = {"choice:2": 0.75}
        self.write_config()
        self.assert_inference_temperatures({("choice", 2): 0.75, ("choice", 3): 2.0,
                                            ("score", 2): 3.0, ("noul", 2): 4.0})

    def test_missing_temperatures_keep_unit_defaults(self):
        self.cfg.pop("temperature")
        self.write_config()
        self.assert_inference_temperatures({(t, 2): 1.0 for t in QTYPES})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
