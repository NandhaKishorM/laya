"""laya.train: data prep, option-order augmentation, losses and an end-to-end run.

CPU only and weight-free: the checkpoint is a one-layer BERT with a word-level tokenizer built
in a temp directory, so nothing is downloaded.

Run: python tests/test_train.py
"""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
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
    finetune,
    items_from_rows,
    make_item,
    rlcd_loss,
    soft_ce_loss,
    split_calibration,
    target_from_gold,
    to_internal,
    train_model,
)

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


if __name__ == "__main__":
    unittest.main()
