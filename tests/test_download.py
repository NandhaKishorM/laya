"""Checkpoint download regression tests; tiny local weights, no network required.

Run: python tests/test_download.py
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from huggingface_hub.utils import filter_repo_objects  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import load  # noqa: E402
from laya.common import DecisionModel  # noqa: E402


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tmp.name) / "repo"
        cls.repo.mkdir()
        config = BertConfig(vocab_size=6, hidden_size=64, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=128)
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
        (cls.repo / "rl_agent_config.json").write_text(json.dumps({
            "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
            "max_len": 64, "head_max_len": 32,
        }))
        cls.runtime_files = {str(p.relative_to(cls.repo)) for p in cls.repo.rglob("*") if p.is_file()}
        for subfolder in ("multilingual", "typed-decisions", "variants/english"):
            for filename in cls.runtime_files:
                target = cls.repo / subfolder / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cls.repo / filename, target)
        (cls.repo / "README.md").write_text("An unrelated model card")
        (cls.repo / "eval").mkdir()
        (cls.repo / "eval" / "results.json").write_text("{}")
        cls.questions = {"q": {"type": "choice", "instructions": "Pick one",
                               "criteria": ["yes", "no"]}}
        cls.expected = load(str(cls.repo), device="cpu").predict("hello", cls.questions)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_download(self, repo_id, subfolder=None):
        # Use the Hub's real file-filter semantics, then load and run the selected files.
        # Only transport is replaced; tokenizer, weights, model construction and inference
        # use the same code as a real checkpoint download.
        with tempfile.TemporaryDirectory() as destination:
            downloaded = []

            def snapshot(repo_id, **kwargs):
                files = [p.relative_to(self.repo).as_posix()
                         for p in self.repo.rglob("*") if p.is_file()]
                selected = filter_repo_objects(files, allow_patterns=kwargs.get("allow_patterns"),
                                               ignore_patterns=kwargs.get("ignore_patterns"))
                for filename in selected:
                    target = Path(destination) / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.repo / filename, target)
                    downloaded.append(filename)
                return destination

            with patch("huggingface_hub.snapshot_download", side_effect=snapshot) as download:
                agent = load(repo_id, device="cpu", subfolder=subfolder, token="test-token")
            self.assertEqual(download.call_count, 1)
            self.assertEqual(download.call_args.args[0], repo_id)
            self.assertEqual(download.call_args.kwargs["token"], "test-token")
            self.assertEqual(agent.predict("hello", self.questions), self.expected)
            prefix = subfolder + "/" if subfolder else ""
            self.assertEqual(set(downloaded), {prefix + name for name in self.runtime_files})

    def test_default_english_does_not_download_sibling_checkpoints(self):
        self.check_download("convaiinnovations/laya")

    def test_custom_root_checkpoint(self):
        self.check_download("test/custom-model")

    def test_each_subfolder_loads_independently(self):
        for subfolder in ("multilingual", "typed-decisions", "variants/english"):
            with self.subTest(subfolder=subfolder):
                self.check_download("test/bundled-models", subfolder)

    def test_local_paths_do_not_download(self):
        with patch("huggingface_hub.snapshot_download") as download:
            for subfolder in (None, "multilingual", "variants/english"):
                with self.subTest(subfolder=subfolder):
                    agent = load(str(self.repo), device="cpu", subfolder=subfolder)
                    self.assertEqual(agent.predict("hello", self.questions), self.expected)
            download.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
