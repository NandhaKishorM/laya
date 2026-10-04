"""Tiny random local models verify the portable specialist, without a Hub download."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from laya.comparison import ComparisonSpecialist  # noqa: E402
from laya._comparison_data import collate, encode, validate_state  # noqa: E402
from laya._comparison_model import OperandNetwork, OperatorNetwork  # noqa: E402


def example():
    return {"task": "numeric_comparison", "assertion": "reading exceeds limit",
            "fields": [{"name": "reading", "value": "2", "unit": "g"},
                       {"name": "limit", "value": "1", "unit": "g"}]}


def tiny(variant="local_program"):
    def base():
        cfg = BertConfig(vocab_size=32, hidden_size=64, num_hidden_layers=1,
                         num_attention_heads=4, intermediate_size=128, max_position_embeddings=1024,
                         hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        return DecisionModel(BertModel(cfg), head_layers=1)
    tok = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
                               "reading": 5, "limit": 6, "exceeds": 7}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                       cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    return ComparisonSpecialist(OperatorNetwork(base(), variant, width=8), OperandNetwork(base(), width=8),
                                tokenizer, device="cpu", operand_precision="fp32")


class PortableComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_architectures_roundtrip_offline(self):
        for variant in ("flat", "local_program"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as parent:
                model = tiny(variant)
                before = model.predict(example())
                path = Path(parent) / "bundle"
                model.save_pretrained(path)
                rng = torch.get_rng_state()
                restored = ComparisonSpecialist.from_pretrained(path, device="cpu")
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                after = restored.predict(example())
                torch.testing.assert_close(torch.tensor(list(before["answers"]["verdict"]["probabilities"].values())),
                                           torch.tensor(list(after["answers"]["verdict"]["probabilities"].values())),
                                           atol=1e-6, rtol=1e-6)
                self.assertEqual(after["specialist"]["backend"], variant)
                self.assertFalse(after["answers"]["verdict"]["calibrated"])
                with self.assertRaises(FileExistsError):
                    model.save_pretrained(path)
                (path / "comparison_config.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    ComparisonSpecialist.from_pretrained(path, device="cpu")

    def test_inputs_are_label_free_and_lossless(self):
        model = tiny()
        request = example()
        before = copy.deepcopy(request)
        normalized = validate_state(request)
        row = encode(normalized, model.tokenizer)
        op, operand = collate([row], model.tokenizer.pad_token_id)
        self.assertEqual(set(op), {"ids", "mask", "claim_mask", "positions"})
        self.assertTrue(torch.equal(operand["identity_links"].sum(1), torch.ones_like(operand["mask"], dtype=torch.float32)))
        self.assertEqual(op["claim_mask"].sum().item(), 3)
        self.assertEqual(len(model.predict_batch([request, request])), 2)
        self.assertEqual(request, before)
        self.assertEqual(model.predict_batch([]), [])
        for assertion in ("[MASK]", "reading " * 600):
            with self.assertRaises(ValueError):
                model.predict({**request, "assertion": assertion})

    def test_mixed_length_batches_match_single_requests(self):
        requests = [example(), {**example(), "assertion": "the quantity reading exceeds the quantity limit"},
                    {**example(), "fields": []}]
        for variant in ("flat", "local_program"):
            with self.subTest(variant=variant):
                model = tiny(variant)
                singles = model.predict_batch(requests, batch_size=1)
                for size in (2, 3):
                    together = model.predict_batch(requests, batch_size=size)
                    for expected, actual in zip(singles, together):
                        self.assertEqual(expected["usage"], actual["usage"])
                        for key in ("operator_probabilities", "operand_probabilities", "gate"):
                            torch.testing.assert_close(torch.tensor(expected["specialist"][key]),
                                                       torch.tensor(actual["specialist"][key]),
                                                       atol=1e-6, rtol=1e-5)
                        torch.testing.assert_close(
                            torch.tensor(list(expected["answers"]["verdict"]["probabilities"].values())),
                            torch.tensor(list(actual["answers"]["verdict"]["probabilities"].values())),
                            atol=1e-6, rtol=1e-5)

    def test_complete_batch_is_validated_before_inference(self):
        model = tiny()
        calls = []
        handle = model.operator.register_forward_pre_hook(lambda m, args: calls.append(True))
        try:
            for invalid in ({**example(), "fields": [{"name": "x", "value": True}]},
                            {**example(), "assertion": "reading " * 600}):
                with self.assertRaises(ValueError):
                    model.predict_batch([example(), invalid], batch_size=1)
                self.assertEqual(calls, [])
        finally:
            handle.remove()

    def test_json_escaping_and_identifier_boundaries(self):
        model = tiny()
        request = {"fields": [{"name": 'x"y', "value": 1, "unit": "g"},
                              {"name": "x", "value": 2, "unit": "g"}],
                   "task": "numeric_comparison", "assertion": 'the first x"y exceeds x'}
        # Quoted names overlap the short x: refuse an ambiguous token link.
        with self.assertRaisesRegex(ValueError, "overlaps"):
            encode(validate_state(request), model.tokenizer)
        request["fields"][1]["name"] = "limit"
        request["assertion"] = 'the first x"y exceeds limit'
        row = encode(validate_state(request), model.tokenizer)
        self.assertTrue(row["links"][0] and row["links"][1])
        self.assertTrue(set(row["links"][0]).isdisjoint(row["links"][1]))

    def test_invalid_network_results_fail_closed(self):
        model = tiny()
        with self.assertRaises(FloatingPointError):
            model._probabilities(torch.full((1, 6), float("nan")), (1, 6))
        with self.assertRaises(ValueError):
            model._probabilities(torch.zeros(1, 3), (1, 6))
        handle = model.operands.register_forward_hook(lambda m, args, out: (out[0], out[1], torch.full_like(out[2], 1.1)))
        try:
            with self.assertRaises(FloatingPointError):
                model.predict(example())
        finally:
            handle.remove()

    def test_local_only_and_manifest_paths(self):
        with tempfile.TemporaryDirectory() as parent:
            missing = Path(parent) / "does-not-exist"
            with self.assertRaises(FileNotFoundError):
                ComparisonSpecialist.from_pretrained(missing)
            path = Path(parent) / "bundle"
            tiny().save_pretrained(path)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["../escape"] = "0" * 64
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"):
                ComparisonSpecialist.from_pretrained(path)

    def test_unmanifested_tokenizer_override_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            path = Path(parent) / "bundle"
            tiny().save_pretrained(path)
            (path / "tokenizer/added_tokens.json").write_text('{"surprise": 9}')
            with self.assertRaisesRegex(ValueError, "Unmanifested"):
                ComparisonSpecialist.from_pretrained(path)

    def test_training_mode_refused(self):
        model = tiny()
        model.operator.train()
        with self.assertRaises(RuntimeError):
            model.predict(example())


if __name__ == "__main__":
    unittest.main()
