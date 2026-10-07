"""laya.train: data prep, option-order augmentation, losses and an end-to-end run.

CPU only and weight-free: the checkpoint is a one-layer BERT with a word-level tokenizer built
in a temp directory, so nothing is downloaded.

Run: python tests/test_train.py
"""
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import load  # noqa: E402
from laya.common import (  # noqa: E402
    DecisionModel,
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_sequence,
    proper_reward,
    render_options,
    unpermute_probs,
)
from laya.train import (  # noqa: E402
    TrainConfig,
    draw_option_order,
    encode_item,
    encode_state,
    evaluate_records,
    finetune,
    items_from_rows,
    make_item,
    read_data,
    rows_from_csv,
    rlcd_loss,
    soft_ce_loss,
    split_calibration,
    target_from_expected,
    target_from_gold,
    to_internal,
    train_model,
)
from laya.train_cli import build_parser as build_train_parser, main as train_cli_main  # noqa: E402

WORDS = ["refund", "invoice", "charged", "twice", "crash", "error", "app", "screen", "please",
         "billing", "technical", "other", "is", "it", "urgent", "low", "high", "level", "question",
         "choice", "score", "noul", "which", "team", "yes", "no", "the", "statement", "does",
         "not", "hold", "holds", "one", "two", "three", "four", "five", "six", "seven", ":", ",", "?"]

DEPARTMENT = {"type": "choice", "instructions": "which team ?",
              "criteria": {"billing": "", "technical": "", "other": ""}}
URGENT = {"type": "noul", "instructions": "is it urgent ?"}
LEVEL = {"type": "score", "instructions": "which level ?", "criteria": ["low", "high"]}

BILLING_STATES = ["please refund invoice", "charged twice invoice", "refund charged twice", "invoice refund please"]
TECH_STATES = ["app crash error", "screen error crash", "crash app screen", "error screen app"]


def make_tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for w in WORDS:
        vocab[w] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def make_checkpoint(path: Path, tok):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    config.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")
    model = DecisionModel(BertModel(config), head_layers=1, n_act=2)
    save_file(model.state_dict(), path / "model.safetensors")
    cfg = {"encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
           "max_len": 64, "head_max_len": 40, "temperature": [1.0, 1.0, 1.0],
           "temperature_by_options": {"choice:3-5": 0.7}}
    (path / "rl_agent_config.json").write_text(json.dumps(cfg))


def rows(n_repeats=6):
    out = []
    for _ in range(n_repeats):
        for s in BILLING_STATES:
            out.append({"state": s, "questions": {"department": DEPARTMENT, "urgent": URGENT, "level": LEVEL},
                        "gold": {"department": {"probabilities": {"billing": 0.9, "technical": 0.05, "other": 0.05}},
                                 "urgent": {"probabilities": {"false": 0.8, "true": 0.2}},
                                 "level": {"probabilities": {"0": 0.7, "1": 0.3}}}})
        for s in TECH_STATES:
            out.append({"state": s, "questions": {"department": DEPARTMENT, "urgent": URGENT, "level": LEVEL},
                        "gold": {"department": {"probabilities": {"billing": 0.05, "technical": 0.9, "other": 0.05}},
                                 "urgent": {"probabilities": {"false": 0.2, "true": 0.8}},
                                 "level": {"probabilities": {"0": 0.2, "1": 0.8}}}})
    return out


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = make_tokenizer()

    def item(self, question, target, state="app crash", head_max_len=40):
        item, reason = make_item(self.tok, to_internal("q", question), target,
                                 encode_state(self.tok, state, 64), head_max_len)
        self.assertIsNone(reason)
        return item

    def test_target_from_gold_orders_and_normalises(self):
        def target(question, probs):
            return target_from_gold(to_internal("q", question), {"probabilities": probs})

        self.assertEqual(target(DEPARTMENT, {"technical": 2.0, "billing": 2.0}), [0.5, 0.5, 0.0])
        self.assertEqual(target(URGENT, {"true": 0.75, "false": 0.25}), [0.25, 0.75])
        self.assertEqual(target(LEVEL, {"1": 1.0}), [0.0, 1.0])
        self.assertEqual(target(DEPARTMENT, {}), [1 / 3] * 3)
        # list-form choice and non-string labels, which inference accepts
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": ["app", "screen"]},
                                {"screen": 1.0}), [0.0, 1.0])
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": {1: "", 2: ""}},
                                {"2": 3.0, "1": 1.0}), [0.25, 0.75])
        for bad in ({"billing": -1.0}, {"billing": float("nan")}, {"billing": "high"}):
            with self.assertRaises(ValueError):
                target(DEPARTMENT, bad)

    def test_target_from_expected_choice(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(to_internal("q", question), exp, label_smoothing=smoothing)

        self.assertEqual(target(DEPARTMENT, "billing"), [1.0, 0.0, 0.0])
        self.assertEqual(target(DEPARTMENT, "technical"), [0.0, 1.0, 0.0])
        self.assertEqual(target(DEPARTMENT, "other"), [0.0, 0.0, 1.0])
        # Non-string labels matching keys
        self.assertEqual(target({"type": "choice", "instructions": "which ?", "criteria": {1: "a", 2: "b"}}, 1),
                         [1.0, 0.0])

    def test_target_from_expected_noul(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(to_internal("q", question), exp, label_smoothing=smoothing)

        self.assertEqual(target(URGENT, True), [0.0, 1.0])
        self.assertEqual(target(URGENT, False), [1.0, 0.0])
        self.assertEqual(target(URGENT, "true"), [0.0, 1.0])
        self.assertEqual(target(URGENT, "false"), [1.0, 0.0])

    def test_target_from_expected_score(self):
        def target(question, exp, smoothing=0.0):
            return target_from_expected(to_internal("q", question), exp, label_smoothing=smoothing)

        self.assertEqual(target(LEVEL, 0), [1.0, 0.0])
        self.assertEqual(target(LEVEL, 1), [0.0, 1.0])
        self.assertEqual(target(LEVEL, "1"), [0.0, 1.0])

        # Fractional score answers (e.g. from laya-evals ScoreMAE / ScoreWithin)
        three_level = {"type": "score", "instructions": "which level ?", "criteria": ["low", "mid", "high"]}
        self.assertEqual(target(three_level, 1.5), [0.0, 0.5, 0.5])
        res_frac2 = target(three_level, 0.2)
        self.assertAlmostEqual(res_frac2[0], 0.8)
        self.assertAlmostEqual(res_frac2[1], 0.2)
        self.assertAlmostEqual(res_frac2[2], 0.0)

    def test_target_from_expected_label_smoothing(self):
        res = target_from_expected(to_internal("q", DEPARTMENT), "billing", label_smoothing=0.1)
        self.assertAlmostEqual(res[0], 0.9 + 0.1 / 3)
        self.assertAlmostEqual(res[1], 0.1 / 3)
        self.assertAlmostEqual(res[2], 0.1 / 3)
        self.assertAlmostEqual(sum(res), 1.0)

        res_noul = target_from_expected(to_internal("q", URGENT), True, label_smoothing=0.2)
        self.assertAlmostEqual(res_noul[0], 0.1)
        self.assertAlmostEqual(res_noul[1], 0.9)
        self.assertAlmostEqual(sum(res_noul), 1.0)

    def test_target_from_expected_rejects_invalid(self):
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", DEPARTMENT), "unknown_choice")
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", URGENT), "not_a_bool")
        for bad_num in (-3, 7, 0.5, -0.1, 2):
            with self.assertRaises(ValueError):
                target_from_expected(to_internal("q", URGENT), bad_num)
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", LEVEL), 99)
        with self.assertRaises(ValueError):
            target_from_expected(to_internal("q", LEVEL), -0.5)
        for bad_smooth in (-0.1, 1.0, 1.5, True):
            with self.assertRaises(ValueError):
                target_from_expected(to_internal("q", DEPARTMENT), "billing", label_smoothing=bad_smooth)

    def test_resolve_checkpoint_dir(self):
        from laya.train import resolve_checkpoint_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "rl_agent_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_checkpoint_dir(str(p)), str(p))
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_dir(str(p / "nonexistent"))

    def test_rows_from_csv_and_read_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "tickets.csv"
            csv_path.write_text("body,department\nrefund my invoice,billing\napp crash on startup,technical\n",
                                encoding="utf-8")
            data = read_data(str(csv_path), text_column="body", label_column="department")
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["state"], "refund my invoice")
            self.assertEqual(data[0]["expected"]["label"], "billing")
            q = data[0]["questions"]["label"]
            self.assertEqual(q["type"], "choice")
            self.assertEqual(q["criteria"], ["billing", "technical"])

            # Test Excel UTF-8 BOM encoding
            bom_csv = Path(tmpdir) / "bom.csv"
            bom_csv.write_bytes("\ufeffbody,department\nrefund invoice,billing\ncrash,technical\n".encode("utf-8"))
            bom_data = read_data(str(bom_csv), text_column="body", label_column="department")
            self.assertEqual(len(bom_data), 2)
            self.assertEqual(bom_data[0]["state"], "refund invoice")

    def test_rows_from_csv_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad.csv"
            p.write_text("text,dept\nhi,billing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                rows_from_csv(str(p), text_column="body", label_column="dept")
            p.write_text("text,dept\nhi,billing\nhello,billing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                rows_from_csv(str(p), text_column="text", label_column="dept")

    def test_items_from_rows_supports_expected(self):
        exp_rows = [
            {"state": "please refund invoice", "questions": {"department": DEPARTMENT},
             "expected": {"department": "billing"}},
            {"state": "app crash", "questions": {"department": DEPARTMENT},
             "expected": {"department": "technical"}},
        ]
        items, skipped = items_from_rows(self.tok, exp_rows, 64, 40)
        self.assertEqual(len(items), 2)
        self.assertEqual(skipped, {})
        self.assertEqual(items[0]["target"], [1.0, 0.0, 0.0])
        self.assertEqual(items[1]["target"], [0.0, 1.0, 0.0])

    def test_items_from_rows_skip_tracking(self):
        exp_rows_with_empty = [
            {"state": "", "questions": {"department": DEPARTMENT}, "expected": {"department": "billing"}},
            {"state": "valid text", "questions": {"department": DEPARTMENT}, "expected": {"department": ""}},
            {"state": "valid text", "questions": {"department": DEPARTMENT}, "expected": {"department": "billing"}},
        ]
        items, skipped = items_from_rows(self.tok, exp_rows_with_empty, 64, 40)
        self.assertEqual(len(items), 1)
        self.assertEqual(skipped.get("empty_text"), 1)
        self.assertEqual(skipped.get("empty_label"), 1)

    def test_questions_are_rendered_exactly_as_inference_renders_them(self):
        # The sequence a training item produces must be the one `Agent` builds for the same
        # question, or the model is trained on option texts it never sees.
        from laya.agent import Agent

        questions = [
            DEPARTMENT, URGENT, LEVEL,
            {"type": "choice", "instructions": "which ?", "criteria": ["app", "screen"]},
            {"type": "noul", "instructions": "is it ?", "labels": {"false": "no", "true": "yes"},
             "criteria": {True: "holds", False: "does not hold"}},
            {"type": "noul", "instructions": {"which": "team"}},
        ]
        for question in questions:
            with self.subTest(question=question):
                k = len(render_options(Agent._to_internal(question)))
                item = self.item(question, [1.0 / k] * k)
                expected = build_sequence(self.tok, "app crash", Agent._to_internal(question), 64, 40)
                enc = encode_item(self.tok, item, 64, 40)
                self.assertEqual((enc["ids"], enc["markers"]), expected)

    def test_rows_share_one_tokenized_state(self):
        items, skipped = items_from_rows(self.tok, rows(1)[:1], 64, 40)
        self.assertEqual((len(items), skipped), (3, {}))
        self.assertTrue(items[0]["state_ids"] is items[1]["state_ids"] is items[2]["state_ids"])
        self.assertEqual([it["qtype"] for it in items], [QTYPES["choice"], QTYPES["noul"], QTYPES["score"]])

    def test_question_without_gold_is_not_an_item_or_a_skip(self):
        row = dict(rows(1)[0])
        row["gold"] = {"department": row["gold"]["department"]}
        items, skipped = items_from_rows(self.tok, [row], 64, 40)
        self.assertEqual((len(items), skipped), (1, {}))

    def test_bad_rows_are_skipped_with_a_reason(self):
        row = dict(rows(1)[0])
        row["questions"] = dict(row["questions"],
                                broken={"type": "choice", "instructions": "which ?", "criteria": []},
                                negative=DEPARTMENT)
        row["gold"] = dict(row["gold"], broken={"probabilities": {}},
                           negative={"probabilities": {"billing": -2.0}})
        items, skipped = items_from_rows(self.tok, [row], 64, 40)
        self.assertEqual((len(items), skipped), (3, {"invalid_question": 1, "invalid_target": 1}))

    def test_options_the_head_budget_collapses_are_skipped(self):
        # Two options that only differ after the per-option cap come out as the same span (#538).
        q = to_internal("q", {"type": "choice", "instructions": "which ?",
                              "criteria": {"one two three four five six": "", "one two three four five seven": ""}})
        state_ids = encode_state(self.tok, "app", 64)
        self.assertEqual(make_item(self.tok, q, [0.5, 0.5], state_ids, head_max_len=20),
                         (None, "options_collapsed"))
        self.assertIsNotNone(make_item(self.tok, q, [0.5, 0.5], state_ids, head_max_len=64)[0])

    def test_options_past_max_len_are_skipped_like_inference_refuses_them(self):
        # With a head longer than max_len, build_sequence drops the last options' markers and the
        # batch would crash on a target longer than its markers. Agent refuses the same question.
        from laya.common import build_head

        question = {"type": "choice", "instructions": "which ?",
                    "criteria": ["one", "two", "three", "four", "five", "six", "seven"]}
        q = to_internal("q", question)
        _ids, markers, _stats = build_head(self.tok, q, 64)
        state_ids = encode_state(self.tok, "app", 64)
        target = [1.0 / 7] * 7
        self.assertEqual(make_item(self.tok, q, target, state_ids, 64, max_len=markers[-1]),
                         (None, "options_beyond_max_len"))
        self.assertIsNotNone(make_item(self.tok, q, target, state_ids, 64, max_len=markers[-1] + 1)[0])
        self.assertIsNotNone(make_item(self.tok, q, target, state_ids, 64)[0])   # no max_len: not checked
        row = {"state": "app", "questions": {"q": question},
               "gold": {"q": {"probabilities": {"one": 1.0}}}}
        self.assertEqual(items_from_rows(self.tok, [row], markers[-1], 64), ([], {"options_beyond_max_len": 1}))
        # the same budget really does cut a marker in the sequence training would build
        _seq, kept = build_sequence(self.tok, "app", q, markers[-1], 64)
        self.assertEqual(len(kept), 6)

    def test_target_length_must_match_options(self):
        state_ids = encode_state(self.tok, "app", 64)
        self.assertEqual(make_item(self.tok, to_internal("q", DEPARTMENT), [0.5, 0.5], state_ids, 40),
                         (None, "target_mismatch"))

    def test_canonical_encoding_matches_build_sequence(self):
        item = self.item(DEPARTMENT, [0.8, 0.1, 0.1])
        enc = encode_item(self.tok, item, 64, 40)
        ids, markers = build_sequence(self.tok, "app crash", item["q"], 64, 40)
        self.assertEqual((enc["ids"], enc["markers"], enc["target"]), (ids, markers, [0.8, 0.1, 0.1]))

    def test_permuted_item_carries_a_permuted_target(self):
        item = self.item(DEPARTMENT, [0.7, 0.2, 0.1])
        order = [2, 0, 1]
        enc = encode_item(self.tok, item, 64, 40, option_order=order)
        self.assertEqual(enc["target"], [0.1, 0.7, 0.2])
        labels = list(DEPARTMENT["criteria"])
        # The token right after slot s's [MASK] is the label of option order[s].
        for slot, marker in enumerate(enc["markers"]):
            self.assertEqual(enc["ids"][marker], self.tok.mask_token_id)
            self.assertEqual(enc["ids"][marker + 1], self.tok.convert_tokens_to_ids(labels[order[slot]]))

    def test_permuted_training_input_is_what_inference_scores_for_that_order(self):
        # Inference with option_order builds the same tokens in the same slots, and
        # unpermute_probs maps the slot-ordered target back to the canonical one.
        from laya.agent import Agent

        order = [1, 2, 0]
        item = self.item(DEPARTMENT, [0.6, 0.3, 0.1])
        enc = encode_item(self.tok, item, 64, 40, option_order=order)
        internal = Agent._to_internal(dict(DEPARTMENT, option_order=order))
        expected_ids, _ = build_sequence(self.tok, "app crash", internal, 64, 40,
                                         option_order=internal["option_order"])
        self.assertEqual(enc["ids"], expected_ids)
        self.assertEqual(list(unpermute_probs(np.array(enc["target"]), order)), item["target"])

    def test_only_configured_types_are_shuffled(self):
        state_ids = encode_state(self.tok, "app", 64)
        choice, _ = make_item(self.tok, to_internal("q", DEPARTMENT), [1.0, 0.0, 0.0], state_ids, 40)
        noul, _ = make_item(self.tok, to_internal("q", URGENT), [0.5, 0.5], state_ids, 40)
        self.assertIsNone(draw_option_order(noul, random.Random(0), ("choice",)))
        orders = {tuple(draw_option_order(choice, random.Random(s), ("choice",))) for s in range(20)}
        self.assertTrue(all(sorted(o) == [0, 1, 2] for o in orders))
        self.assertGreater(len(orders), 1)
        self.assertEqual(draw_option_order(choice, random.Random(3), ("choice",)),
                         draw_option_order(choice, random.Random(3), ("choice",)))
        self.assertIsNone(draw_option_order(choice, random.Random(0), ()))

    def test_calibration_split_is_disjoint_and_seeded(self):
        items = list(range(50))
        train, calib = split_calibration(items, calib_max=400, calib_frac=0.1, seed=7)
        self.assertEqual((len(train), len(calib)), (45, 5))
        self.assertEqual(sorted(train + calib), items)
        self.assertEqual(split_calibration(items, 400, 0.1, 7), (train, calib))
        self.assertEqual(len(split_calibration(items, 3, 0.5, 7)[1]), 3)


class LossTests(unittest.TestCase):
    def test_soft_ce_ignores_padded_options(self):
        logits = torch.tensor([[2.0, 0.5, 9.0], [1.0, -1.0, 0.0]])
        target = torch.tensor([[0.75, 0.25, 0.0], [0.2, 0.3, 0.5]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        manual = -(torch.tensor([0.75, 0.25]) * torch.log_softmax(torch.tensor([2.0, 0.5]), -1)).sum()
        manual = (manual - (target[1] * torch.log_softmax(logits[1], -1)).sum()) / 2
        self.assertAlmostEqual(soft_ce_loss(logits, target, mask).item(), manual.item(), places=5)

    def test_rlcd_is_finite_and_differentiable(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.softmax(torch.randn(4, 3), -1)
        mask = torch.ones(4, 3, dtype=torch.bool)
        loss = rlcd_loss(logits, target, mask, torch.tensor([0, 1, 2, 0]), sigma=0.4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rlcd_matches_the_notebook_objective(self):
        # The loss as written in research/scripts/finetune_single_device.py, inlined: with the
        # same RNG state, rlcd_loss must give the same value and the same gradient.
        def script_loss(logits, target, mask, qtype, sigma):
            k = mask.sum(-1, keepdim=True).float()
            eps = torch.randn((4,) + logits.shape) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            ce = (target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            return -(adv * logp).mean() - ce

        torch.manual_seed(1)
        base = torch.randn(5, 4)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]],
                            dtype=torch.bool)
        target = torch.softmax(torch.randn(5, 4), -1) * mask
        target = target / target.sum(-1, keepdim=True)
        qtype = torch.tensor([0, 2, 1, 0, 2])
        results = []
        for fn in (lambda lg: script_loss(lg, target, mask, qtype, 0.3),
                   lambda lg: rlcd_loss(lg, target, mask, qtype, 0.3)):
            logits = base.masked_fill(~mask, -1e4).clone().requires_grad_(True)
            torch.manual_seed(5)
            loss = fn(logits)
            loss.backward()
            results.append((loss.item(), logits.grad.clone()))
        self.assertAlmostEqual(results[0][0], results[1][0], places=5)
        self.assertTrue(torch.allclose(results[0][1], results[1][1], atol=1e-6))

    def test_defaults_train_like_the_notebook(self):
        config = TrainConfig()
        self.assertEqual((config.loss, config.shuffle_options), ("rlcd", ()))
        self.assertEqual(config.micro_batch * config.grad_accum, 64)

    def test_config_rejects_unknown_values(self):
        for bad in (TrainConfig(loss="mse"), TrainConfig(shuffle_options=("choise",)),
                    TrainConfig(epochs=0), TrainConfig(calib_frac=1.0)):
            with self.assertRaises(ValueError):
                bad.validate()


class TrainingLoopTests(unittest.TestCase):
    def test_partial_accumulation_window_averages_its_own_steps(self):
        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Identity()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

        model = ScalarModel()
        tok = type("Tokenizer", (), {"pad_token_id": 0})()
        item = {"q": {"t": "choice"}, "k": 2}
        batch = {
            "marker_mask": torch.ones((1, 1), dtype=torch.bool),
            "target": torch.ones((1, 1)),
        }
        config = TrainConfig(
            epochs=1,
            micro_batch=1,
            grad_accum=2,
            head_lr=1.0,
            min_lr=1.0,
            weight_decay=0.0,
            grad_clip=100.0,
            loss="soft-ce",
            freeze_encoder=True,
            amp=False,
            log_every=0,
        )

        with patch("laya.train.encode_item", return_value={}), \
                patch("laya.train.collate_items", return_value=batch), \
                patch("laya.train._forward", side_effect=lambda current, *_args: current.value), \
                patch("laya.train.soft_ce_loss", side_effect=lambda logits, *_args: logits), \
                patch("laya.train.torch.optim.AdamW",
                      side_effect=lambda groups, weight_decay: torch.optim.SGD(
                          groups, weight_decay=weight_decay)):
            train_model(model, tok, [item] * 3, config, torch.device("cpu"), 1, 1)

        # The full two-step window and the final one-step window each contribute one update.
        self.assertAlmostEqual(model.value.item(), -2.0)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.tok = make_tokenizer()
        make_checkpoint(cls.root / "base", cls.tok)
        cls.data = cls.root / "train.jsonl"
        cls.data.write_text("\n".join(json.dumps(r) for r in rows()), encoding="utf-8")

    def run_finetune(self, out, **overrides):
        # Options stay in order here: a randomly initialised one-layer encoder cannot learn to read
        # option labels in a handful of epochs, only their slots. The permutation itself is
        # covered by DataTests; test_shuffled_run_saves_a_loadable_checkpoint runs it end to end.
        kwargs = dict(epochs=12, micro_batch=8, grad_accum=1, encoder_lr=1e-3, head_lr=1e-3,
                      calib_frac=0.2, log_every=0, loss="soft-ce", shuffle_options=())
        kwargs.update(overrides)
        config = TrainConfig(**kwargs)
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
            return finetune(str(self.data), str(self.root / "base"), str(self.root / out), config, device="cpu")

    def test_finetune_learns_and_saves_a_loadable_checkpoint(self):
        report = self.run_finetune("out")
        self.assertEqual(report["train_items"] + report["calibration_items"], len(rows()) * 3)
        self.assertLess(report["epoch_loss"][-1], report["epoch_loss"][0])

        saved = json.loads((self.root / "out" / "rl_agent_config.json").read_text())
        self.assertTrue(saved["fine_tuned"])
        self.assertEqual((saved["max_len"], saved["head_max_len"]), (64, 40))
        # The inherited bucket would override the new fit at inference; too few items to refit it.
        self.assertNotIn("temperature_by_options", saved)
        self.assertTrue(all(TEMP_MIN <= t <= TEMP_MAX for t in saved["temperature"]))
        self.assertEqual(saved["training"]["laya_train"]["loss"], "soft-ce")
        self.assertTrue((self.root / "out" / "checkpoint_latest" / "model.safetensors").exists())

        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
            agent = load(str(self.root / "out"), device="cpu")
        answers = agent.predict("please refund invoice", {"department": DEPARTMENT})["answers"]
        self.assertEqual(answers["department"]["choice"], "billing")
        answers = agent.predict("app crash error", {"department": DEPARTMENT})["answers"]
        self.assertEqual(answers["department"]["choice"], "technical")

    def test_shuffled_run_saves_a_loadable_checkpoint(self):
        report = self.run_finetune("out_shuffled", shuffle_options=("choice",), epochs=1)
        saved = json.loads((self.root / "out_shuffled" / "rl_agent_config.json").read_text())
        self.assertEqual(saved["training"]["laya_train"]["shuffle_options"], ["choice"])
        self.assertEqual(len(report["epoch_loss"]), 1)
        agent = load(str(self.root / "out_shuffled"), device="cpu")
        self.assertIn(agent.predict("app", {"department": DEPARTMENT})["answers"]["department"]["choice"],
                      DEPARTMENT["criteria"])

    def test_rlcd_objective_still_trains(self):
        report = self.run_finetune("out_rlcd", loss="rlcd")
        self.assertLess(report["epoch_loss"][-1], report["epoch_loss"][0])

    def test_frozen_encoder_is_left_untouched(self):
        self.run_finetune("out_frozen", freeze_encoder=True, epochs=2)
        before = load_file(str(self.root / "base" / "model.safetensors"))
        after = load_file(str(self.root / "out_frozen" / "model.safetensors"))
        for name in before:
            if name.startswith("encoder."):
                self.assertTrue(torch.allclose(before[name].half().float(), after[name].float()), name)
        self.assertFalse(torch.allclose(before["scorer.1.weight"].half().float(), after["scorer.1.weight"].float()))

    def test_frozen_encoder_runs_without_dropout(self):
        from laya.train import load_checkpoint

        model, tok, _cfg = load_checkpoint(str(self.root / "base"))
        modes = []
        model.encoder.register_forward_pre_hook(lambda module, _args: modes.append(module.training))
        items, _ = items_from_rows(tok, rows(1)[:4], 64, 40)
        train_model(model, tok, items, TrainConfig(epochs=1, freeze_encoder=True, log_every=0),
                    torch.device("cpu"), 64, 40)
        self.assertTrue(modes)
        self.assertFalse(any(modes))

    def test_train_model_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            train_model(None, self.tok, [], TrainConfig(), torch.device("cpu"), 64, 40)

    def test_laya_train_cli_parser_and_validation(self):
        parser = build_train_parser()
        args = parser.parse_args(["--data", "train.csv", "--eval", "eval.csv", "--target-error", "0.05",
                                  "--min-abstain-n", "5", "--out", "./out", "--label-smoothing", "0.1",
                                  "--shuffle-options", "--freeze-encoder"])
        self.assertEqual(args.data, "train.csv")
        self.assertEqual(args.eval_data, "eval.csv")
        self.assertEqual(args.target_error, 0.05)
        self.assertEqual(args.min_abstain_n, 5)
        self.assertEqual(args.output_dir, "./out")
        self.assertEqual(args.label_smoothing, 0.1)
        self.assertTrue(args.shuffle_options)
        self.assertTrue(args.freeze_encoder)

        # --out is required when --dry-run is not set
        code_missing_out = train_cli_main(["--data", "train.csv"])
        self.assertEqual(code_missing_out, 2)

        # --dry-run without --out succeeds parsing
        dry_args = parser.parse_args(["--data", "train.csv", "--dry-run"])
        self.assertIsNone(dry_args.output_dir)
        self.assertTrue(dry_args.dry_run)

    def test_train_config_eval_validation(self):
        cfg = TrainConfig(eval_data="val.csv", target_error=0.15, min_abstain_n=20)
        cfg.validate()

        with self.assertRaises(ValueError):
            TrainConfig(target_error=-0.1).validate()
        with self.assertRaises(ValueError):
            TrainConfig(target_error=1.5).validate()
        with self.assertRaises(ValueError):
            TrainConfig(target_error=True).validate()
        with self.assertRaises(ValueError):
            TrainConfig(min_abstain_n=0).validate()
        with self.assertRaises(ValueError):
            TrainConfig(min_abstain_n=-5).validate()
        with self.assertRaises(ValueError):
            TrainConfig(eval_data=123).validate()

    def test_evaluate_records(self):
        # Empty records returns default dict
        empty = evaluate_records([])
        self.assertEqual(empty["items"], 0)
        self.assertIsNone(empty["ece"])
        self.assertIsNone(empty["brier"])
        self.assertIsNone(empty["brier_top1"])

        # Distinct Brier values for known input:
        # 3 options, uniform prediction (logits [0, 0, 0] -> p = [1/3, 1/3, 1/3]), one-hot target [1, 0, 0].
        # Multiclass Brier = (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 0.6667
        # Top-1 Confidence Brier = (1/3 - 1)^2 = 4/9 = 0.4444
        uniform_recs = [(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 3)]
        uniform_metrics = evaluate_records(uniform_recs)
        self.assertEqual(uniform_metrics["brier"], 0.6667)
        self.assertEqual(uniform_metrics["brier_top1"], 0.4444)
        self.assertNotEqual(uniform_metrics["brier"], uniform_metrics["brier_top1"])

        # Synthetic records: 10 choice items, 10 score items
        # Choice: 3 options, logits favor option 0 strongly, gold is option 0
        choice_recs = [(0, [5.0, -2.0, -2.0], [1.0, 0.0, 0.0], 3) for _ in range(10)]
        # Score: 5 levels, logits favor level 2, gold is level 2
        score_recs = [(1, [-2.0, -2.0, 5.0, -2.0, -2.0], [0.0, 0.0, 1.0, 0.0, 0.0], 5) for _ in range(10)]
        metrics = evaluate_records(choice_recs + score_recs)
        self.assertEqual(metrics["items"], 20)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["mean_confidence"], 0.9)
        self.assertIsNotNone(metrics["ece"])
        self.assertIsNotNone(metrics["brier"])
        self.assertIsNotNone(metrics["brier_top1"])
        self.assertIn("choice", metrics["by_type"])
        self.assertIn("score", metrics["by_type"])
        self.assertEqual(metrics["by_type"]["choice"]["items"], 10)
        self.assertEqual(metrics["by_type"]["score"]["items"], 10)
        self.assertIn("brier_top1", metrics["by_type"]["choice"])

        # Temperature scaling softens confidences
        warm = evaluate_records(choice_recs, temperature=[5.0, 5.0, 5.0])
        self.assertLess(warm["mean_confidence"], metrics["by_type"]["choice"]["mean_confidence"])

    def test_laya_train_cli_runs_on_csv(self):
        csv_path = self.root / "tickets.csv"
        lines = ["text,department"]
        for s in BILLING_STATES:
            lines.append(f"{s},billing")
        for s in TECH_STATES:
            lines.append(f"{s},technical")
        csv_path.write_text("\n".join(lines), encoding="utf-8")

        out_dir = self.root / "out_cli"
        code = train_cli_main([
            "--data", str(csv_path),
            "--base", str(self.root / "base"),
            "--out", str(out_dir),
            "--text-column", "text",
            "--label-column", "department",
            "--epochs", "1",
            "--micro-batch", "4",
            "--grad-accum", "1",
            "--loss", "soft-ce",
            "--label-smoothing", "0.05",
            "--device", "cpu",
        ])
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "model.safetensors").exists())
        self.assertTrue((out_dir / "rl_agent_config.json").exists())
        self.assertTrue((out_dir / "questions.json").exists())
        self.assertTrue((out_dir / "train_report.json").exists())
        self.assertTrue((out_dir / "checkpoint_latest" / "train_report.json").exists())
        self.assertTrue((out_dir / "checkpoint_latest" / "rl_agent_config.json").exists())

        out_cfg = json.loads((out_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
        latest_cfg = json.loads((out_dir / "checkpoint_latest" / "rl_agent_config.json").read_text(encoding="utf-8"))
        # Top-level min_confidence ghost field must not be written to checkpoint config
        self.assertNotIn("min_confidence", out_cfg)
        # checkpoint_latest must be synchronized with final calibrated out_cfg
        self.assertEqual(latest_cfg["temperature"], out_cfg["temperature"])
        self.assertEqual(latest_cfg["training"]["train_report"], out_cfg["training"]["train_report"])

        saved_q = json.loads((out_dir / "questions.json").read_text(encoding="utf-8"))
        self.assertIn("label", saved_q)
        self.assertEqual(saved_q["label"]["criteria"], ["billing", "technical"])

        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertIsNone(report["eval_source"])
        self.assertIsNone(report["eval_mode"])
        self.assertFalse(report["is_held_out"])
        self.assertEqual(report["eval_items"], 0)
        self.assertIn("No evaluation performed", report["note"])
        self.assertIsNone(report["before"])
        self.assertIsNone(report["after"])
        self.assertIsNone(report["comparison"])

        agent = load(str(out_dir), device="cpu")
        ans = agent.predict("please refund invoice", {"label": {"type": "choice", "instructions": "Classify",
                                                               "criteria": ["billing", "technical"]}})
        self.assertIn(ans["answers"]["label"]["choice"], ["billing", "technical"])

        # Test --dry-run
        dry_code = train_cli_main([
            "--data", str(csv_path),
            "--base", str(self.root / "base"),
            "--text-column", "text",
            "--label-column", "department",
            "--dry-run",
        ])
        self.assertEqual(dry_code, 0)

    def test_laya_train_cli_with_eval_file(self):
        train_path = self.root / "train_eval_test.csv"
        eval_path = self.root / "eval_eval_test.csv"
        train_lines = ["text,department"]
        for s in BILLING_STATES:
            train_lines.append(f"{s},billing")
        for s in TECH_STATES:
            train_lines.append(f"{s},technical")
        train_path.write_text("\n".join(train_lines), encoding="utf-8")

        # Independent, non-overlapping evaluation items
        eval_lines = [
            "text,department",
            "please dispute my credit card charge,billing",
            "why was my card debited twice,billing",
            "the API server returns 502 bad gateway,technical",
            "the database socket connection timed out,technical",
        ]
        eval_path.write_text("\n".join(eval_lines), encoding="utf-8")

        out_dir = self.root / "out_with_eval"
        code = train_cli_main([
            "--data", str(train_path),
            "--eval", str(eval_path),
            "--base", str(self.root / "base"),
            "--out", str(out_dir),
            "--text-column", "text",
            "--label-column", "department",
            "--epochs", "1",
            "--micro-batch", "4",
            "--grad-accum", "1",
            "--loss", "soft-ce",
            "--target-error", "0.08",
            "--min-abstain-n", "2",
            "--device", "cpu",
        ])
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "train_report.json").exists())
        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["eval_source"], str(eval_path))
        self.assertEqual(report["eval_mode"], "held_out")
        self.assertTrue(report["is_held_out"])
        self.assertIn("independent held-out evaluation", report["note"])
        self.assertEqual(report["eval_items"], 4)
        self.assertIn("before", report)
        self.assertIn("after", report)
        self.assertIn("comparison", report)
        self.assertIn("brier", report["after"])
        self.assertIn("brier_top1", report["after"])
        self.assertIn("delta_brier", report["comparison"])
        self.assertIn("delta_brier_top1", report["comparison"])

    def test_laya_train_cli_with_overlapping_eval_file(self):
        train_path = self.root / "train_overlap.csv"
        eval_path = self.root / "eval_overlap.csv"
        train_lines = ["text,department"]
        eval_lines = ["text,department"]
        for s in BILLING_STATES:
            train_lines.append(f"{s},billing")
            eval_lines.append(f"{s},billing")
        for s in TECH_STATES:
            train_lines.append(f"{s},technical")
            eval_lines.append(f"{s},technical")
        train_path.write_text("\n".join(train_lines), encoding="utf-8")
        eval_path.write_text("\n".join(eval_lines), encoding="utf-8")

        out_dir = self.root / "out_overlap"
        code = train_cli_main([
            "--data", str(train_path),
            "--eval", str(eval_path),
            "--base", str(self.root / "base"),
            "--out", str(out_dir),
            "--text-column", "text",
            "--label-column", "department",
            "--epochs", "1",
            "--micro-batch", "4",
            "--grad-accum", "1",
            "--loss", "soft-ce",
            "--device", "cpu",
        ])
        self.assertEqual(code, 0)
        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["eval_source"], str(eval_path))
        self.assertEqual(report["eval_mode"], "overlapping_eval")
        self.assertFalse(report["is_held_out"])
        self.assertIn("overlap training data", report["note"])
        self.assertIn("brier", report["after"])
        self.assertIn("brier_top1", report["after"])
        self.assertIn("delta_brier", report["comparison"])
        self.assertIn("delta_brier_top1", report["comparison"])

    def test_laya_train_cli_with_empty_or_skipped_eval_file(self):
        train_path = self.root / "train_norm.csv"
        eval_path = self.root / "eval_empty.jsonl"
        train_lines = ["text,department"]
        for s in BILLING_STATES:
            train_lines.append(f"{s},billing")
        for s in TECH_STATES:
            train_lines.append(f"{s},technical")
        train_path.write_text("\n".join(train_lines), encoding="utf-8")
        eval_path.write_text("", encoding="utf-8")

        out_dir = self.root / "out_empty_eval"
        code = train_cli_main([
            "--data", str(train_path),
            "--eval", str(eval_path),
            "--base", str(self.root / "base"),
            "--out", str(out_dir),
            "--text-column", "text",
            "--label-column", "department",
            "--epochs", "1",
            "--micro-batch", "4",
            "--grad-accum", "1",
            "--loss", "soft-ce",
            "--device", "cpu",
        ])
        self.assertEqual(code, 0)
        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["eval_source"], str(eval_path))
        self.assertIsNone(report["eval_mode"])
        self.assertFalse(report["is_held_out"])
        self.assertEqual(report["eval_items"], 0)
        self.assertIsNone(report["before"])
        self.assertIsNone(report["after"])
        self.assertIn("0 usable items", report["note"])

    def test_overlap_detection_across_gold_and_expected_formats(self):
        # Training row uses 'gold', eval row uses 'expected'
        train_file = self.root / "train_gold.jsonl"
        eval_file = self.root / "eval_expected.jsonl"
        q_def = {"type": "choice", "instructions": "Classify", "criteria": ["billing", "technical"]}
        train_row = {"state": "please refund payment", "questions": {"label": q_def}, "gold": {"label": {"probabilities": {"billing": 1.0}}}}
        eval_row = {"state": "please refund payment", "questions": {"label": q_def}, "expected": {"label": "billing"}}
        train_file.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
        eval_file.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

        out_dir = self.root / "out_gold_expected_overlap"
        cfg = TrainConfig(epochs=1, micro_batch=1, grad_accum=1, eval_data=str(eval_file))
        finetune(str(train_file), str(self.root / "base"), str(out_dir), cfg, device="cpu")
        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["eval_mode"], "overlapping_eval")
        self.assertFalse(report["is_held_out"])
        self.assertIn("overlap training data", report["note"])

    def test_overlap_detection_with_different_targets(self):
        # Training and eval have identical input text and question, but different target label
        train_file = self.root / "train_diff_tgt.jsonl"
        eval_file = self.root / "eval_diff_tgt.jsonl"
        q_def = {"type": "choice", "instructions": "Classify", "criteria": ["billing", "technical"]}
        train_row = {"state": "please refund payment", "questions": {"label": q_def}, "expected": {"label": "billing"}}
        eval_row = {"state": "please refund payment", "questions": {"label": q_def}, "expected": {"label": "technical"}}
        train_file.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
        eval_file.write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

        out_dir = self.root / "out_diff_tgt_overlap"
        cfg = TrainConfig(epochs=1, micro_batch=1, grad_accum=1, eval_data=str(eval_file))
        finetune(str(train_file), str(self.root / "base"), str(out_dir), cfg, device="cpu")
        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["eval_mode"], "overlapping_eval")
        self.assertFalse(report["is_held_out"])
        self.assertIn("overlap training data", report["note"])

    def test_overlap_detection_with_calibration_slice(self):
        # Training set has multiple items split into train and calib.
        # Eval set only contains an item that ended up in the calibration slice, not train_items.
        train_file = self.root / "train_calib_split.jsonl"
        eval_file = self.root / "eval_calib_split.jsonl"
        q_def = {"type": "choice", "instructions": "Classify", "criteria": ["billing", "technical"]}

        states = ["refund please", "charged twice", "crash app", "screen error"]
        rows = [
            {"state": s, "questions": {"label": q_def}, "expected": {"label": "billing" if "refund" in s or "charged" in s else "technical"}}
            for s in states
        ]
        train_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        import random
        order = list(range(len(rows)))
        random.Random(42).shuffle(order)
        calib_row = rows[order[0]]  # in calib set, not in train set
        eval_file.write_text(json.dumps(calib_row) + "\n", encoding="utf-8")

        out_dir = self.root / "out_calib_overlap"
        cfg = TrainConfig(epochs=1, micro_batch=1, grad_accum=1, calib_frac=0.5, calib_max=2, calib_seed=42, eval_data=str(eval_file))
        finetune(str(train_file), str(self.root / "base"), str(out_dir), cfg, device="cpu")
        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["eval_mode"], "overlapping_eval")
        self.assertFalse(report["is_held_out"])
        self.assertIn("overlap calibration data", report["note"])

    def test_laya_train_in_sample_calibration_mode(self):
        csv_path = self.root / "calib_tickets.csv"
        lines = ["text,department"]
        for s in BILLING_STATES:
            lines.append(f"{s},billing")
        for s in TECH_STATES:
            lines.append(f"{s},technical")
        csv_path.write_text("\n".join(lines), encoding="utf-8")

        out_dir = self.root / "out_in_sample"
        cfg = TrainConfig(
            text_column="text",
            label_column="department",
            epochs=1,
            micro_batch=4,
            grad_accum=1,
            calib_frac=0.2,
            calib_max=10,
            target_error=0.10,
            min_abstain_n=2,
        )
        finetune(str(csv_path), str(self.root / "base"), str(out_dir), cfg, device="cpu")
        report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["eval_source"], "calibration_slice")
        self.assertEqual(report["eval_mode"], "in_sample_calibration")
        self.assertFalse(report["is_held_out"])
        self.assertIn("in-sample calibration fit", report["note"])
        self.assertGreater(report["eval_items"], 0)
        self.assertIn("before", report)
        self.assertIn("after", report)
        self.assertIn("comparison", report)


class CalibrationReportTests(unittest.TestCase):
    @staticmethod
    def records(qtype, n, k=3, scale=1.0):
        # Logits that favour the target option by `scale`; target one-hot on option 0.
        return [(qtype, [scale] + [0.0] * (k - 1), [1.0] + [0.0] * (k - 1), k) for _ in range(n)]

    def test_each_kind_of_weak_fit_is_named(self):
        from laya.train import CALIB_WARN_N, calibration_report

        records = (self.records(QTYPES["choice"], 5) + self.records(QTYPES["noul"], 30, k=2)
                   + self.records(QTYPES["score"], CALIB_WARN_N + 10))
        report = calibration_report(records, [1.0, 1.0, 2.0])   # choice, score, noul
        self.assertEqual(report["choice"]["items"], 5)
        self.assertIn("not fitted", report["choice"]["issues"][0])
        self.assertEqual(len(report["noul"]["issues"]), 1)
        self.assertIn("only 30", report["noul"]["issues"][0])
        self.assertEqual(len(report["score"]["issues"]), 1)
        self.assertIn("unchanged at 1.0", report["score"]["issues"][0])
        self.assertAlmostEqual(report["score"]["accuracy"], 1.0)

    def test_clamped_and_healthy_fits(self):
        from laya.train import CALIB_WARN_N, calibration_report

        n = CALIB_WARN_N
        records = self.records(QTYPES["choice"], n) + self.records(QTYPES["score"], n)
        report = calibration_report(records, [TEMP_MIN, 1.3, 1.0])   # choice, score, noul
        self.assertIn("clamp", report["choice"]["issues"][0])
        self.assertEqual(report["score"]["issues"], [])
        # a type absent from the data is reported, but not as a problem
        self.assertEqual((report["noul"]["items"], report["noul"]["issues"]), (0, []))
        self.assertNotIn("accuracy", report["noul"])
        p = torch.softmax(torch.tensor([1.0, 0.0, 0.0]) / 1.3, -1)
        self.assertAlmostEqual(report["score"]["mean_confidence"], p.max().item(), places=6)

    def test_finetune_warns_and_records_a_weak_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_checkpoint(root / "base", make_tokenizer())
            config = TrainConfig(epochs=1, micro_batch=8, grad_accum=1, calib_frac=0.2, log_every=0)
            # The rows go straight to finetune through read_jsonl rather than through a file: nothing
            # here needs the JSONL parsing, which the end-to-end tests above already cover.
            # warnings.catch_warnings, not assertWarns: assertWarns walks every loaded module, and
            # transformers' lazy modules then try to import optional packages such as torchvision.
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")), \
                    patch("laya.train.read_jsonl", return_value=rows(2)), \
                    warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = finetune("rows.jsonl", str(root / "base"), str(root / "out"), config, device="cpu")
            saved = json.loads((root / "out" / "rl_agent_config.json").read_text())
        messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(any(re.match(r"laya\.train: \w+ calibration: not fitted", m) for m in messages), messages)
        self.assertEqual(saved["training"]["laya_train_calibration"], report["calibration"])
        self.assertEqual(sum(e["items"] for e in report["calibration"].values()), report["calibration_items"])


class PriorCollapseTests(unittest.TestCase):
    """#963: a collapsed head is chance-level and silent unless finetune warns."""

    @staticmethod
    def rows_of(logits, target, n=8, qtype=None, k=None):
        qtype = QTYPES["choice"] if qtype is None else qtype
        k = len(logits) if k is None else k
        return [(qtype, list(logits), list(target), k) for _ in range(n)]

    def test_constant_logits_match_the_prior_and_warn(self):
        from laya.train import COLLAPSE_LOGIT_RANGE, collapse_stats, prior_collapse_message

        # Tiny 4-way set: every row is the same logit, so the predicted distribution is the
        # uniform prior. This is the measured failure in #963 (range 0.01, CE at ln 4).
        records = self.rows_of([0.002, 0.0, -0.001, 0.001], [1.0, 0.0, 0.0, 0.0])
        stats = collapse_stats(records)
        self.assertIsNotNone(stats)
        self.assertLess(stats["mean_logit_range"], COLLAPSE_LOGIT_RANGE)
        self.assertAlmostEqual(stats["mean_ce"], math.log(4), places=2)
        msg = prior_collapse_message(records)
        self.assertIsNotNone(msg)
        self.assertIn("collapsed to the class prior", msg)
        self.assertIn("more epochs", msg)
        self.assertIn("more data", msg)
        self.assertIn("another seed", msg)

    def test_peaked_logits_on_a_learnable_set_do_not_warn(self):
        from laya.train import prior_collapse_message

        records = self.rows_of([5.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        self.assertIsNone(prior_collapse_message(records))
        self.assertIsNone(prior_collapse_message([]))
        self.assertIsNone(prior_collapse_message(self.rows_of([0.0], [1.0], k=1)))

    def test_ce_stuck_at_ln_k_warns_even_when_logit_range_is_not_tiny(self):
        from laya.train import COLLAPSE_LOGIT_RANGE, collapse_stats, prior_collapse_message

        # Softmax is still near-uniform, so CE sits on ln K, but the within-row range is
        # above the constant-logit cutoff -- the other #963 trigger.
        records = self.rows_of([0.2, 0.05, -0.05, -0.2], [0.25, 0.25, 0.25, 0.25])
        stats = collapse_stats(records)
        self.assertGreaterEqual(stats["mean_logit_range"], COLLAPSE_LOGIT_RANGE)
        self.assertAlmostEqual(stats["mean_ce"], stats["mean_ln_k"], delta=0.03)
        msg = prior_collapse_message(records)
        self.assertIsNotNone(msg)
        self.assertIn("cross-entropy", msg)

    def test_finetune_emits_the_collapse_warning_only_when_logits_collapse(self):
        from laya.train import prior_collapse_message

        collapsed = self.rows_of([0.01, 0.0, -0.01], [1.0, 0.0, 0.0], n=12)
        peaked = self.rows_of([4.0, 0.0, 0.0], [1.0, 0.0, 0.0], n=12)
        self.assertIsNotNone(prior_collapse_message(collapsed))
        self.assertIsNone(prior_collapse_message(peaked))

        def run(records):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                make_checkpoint(root / "base", make_tokenizer())
                config = TrainConfig(epochs=1, micro_batch=8, grad_accum=1, calib_frac=0.2, log_every=0)
                with patch("huggingface_hub.snapshot_download",
                           side_effect=AssertionError("unexpected download")), \
                        patch("laya.train.read_jsonl", return_value=rows(2)), \
                        patch("laya.train.calibration_records", return_value=records), \
                        warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    finetune("rows.jsonl", str(root / "base"), str(root / "out"), config, device="cpu")
            return [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

        collapsed_msgs = run(collapsed)
        self.assertTrue(any("collapsed to the class prior" in m for m in collapsed_msgs), collapsed_msgs)
        peaked_msgs = run(peaked)
        self.assertFalse(any("collapsed to the class prior" in m for m in peaked_msgs), peaked_msgs)


if __name__ == "__main__":
    unittest.main()
