"""Decision-head checkpointing regressions with a tiny BERT; no downloads.

Run: python tests/test_head_checkpointing.py
"""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import BertConfig, BertModel  # noqa: E402

from laya.common import DecisionModel  # noqa: E402


def tiny_model(head_layers=2):
    torch.manual_seed(41)
    config = BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=1, intermediate_size=32)
    model = DecisionModel(BertModel(config), head_layers=head_layers, dropout=0.2)
    if model.head is not None:
        # TransformerEncoder clones its layers. Distinct weights catch accidentally
        # recomputing the last layer for every iteration of the forward loop.
        with torch.no_grad():
            for i, layer in enumerate(model.head.layers):
                layer.linear1.weight.add_(0.1 * i)
    return model


def inputs():
    torch.manual_seed(42)
    ids = torch.randint(0, 64, (3, 12))
    attention = torch.ones_like(ids)
    attention[0, 9:] = 0
    attention[1, 11:] = 0
    positions = torch.tensor([[2, 0, 0], [2, 5, 0], [2, 5, 8]])
    markers = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    return ids, attention, positions, markers, torch.tensor([0, 2, 1])


def training_step(model, *, detach_encoder=False):
    batch = inputs()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    logits, act = model(*batch, detach_encoder=detach_encoder)
    loss = logits.masked_select(batch[3]).square().mean() + act.square().mean()
    loss.backward()
    gradients = {name: p.grad.detach().clone() if p.grad is not None else None
                 for name, p in model.named_parameters()}
    optimizer.step()
    return logits.detach(), act.detach(), loss.detach(), gradients, torch.get_rng_state()


class HeadCheckpointingTests(unittest.TestCase):
    def assert_same_step(self, model, **kwargs):
        ordinary = copy.deepcopy(model).train()
        checkpointed = copy.deepcopy(model).train()
        ordinary.head_checkpointing = False
        checkpointed.head_checkpointing = True
        expected = training_step(ordinary, **kwargs)
        actual = training_step(checkpointed, **kwargs)
        for before, after in zip(expected[:3], actual[:3]):
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        for name, gradient in expected[3].items():
            other = actual[3][name]
            with self.subTest(parameter=name):
                if gradient is None:
                    self.assertIsNone(other)
                else:
                    self.assertIsNotNone(other)
                    torch.testing.assert_close(gradient, other, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(expected[4], actual[4]), "dropout RNG state changed")
        for name, value in ordinary.state_dict().items():
            torch.testing.assert_close(value, checkpointed.state_dict()[name], rtol=1e-5, atol=1e-6)
        return actual[3]

    def test_enabled_head_recomputes_each_layer_during_backward(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = tiny_model().train()
                model.head_checkpointing = enabled
                calls = [0, 0]
                handles = []
                for i, layer in enumerate(model.head.layers):
                    def count(_module, _args, index=i):
                        calls[index] += 1
                    handles.append(layer.register_forward_pre_hook(count))
                try:
                    training_step(model)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(calls, [2, 2] if enabled else [1, 1])

    def test_training_outputs_gradients_and_updates_match_with_dropout(self):
        self.assert_same_step(tiny_model())

    def test_encoder_and_head_checkpointing_work_together(self):
        model = tiny_model()
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.assert_same_step(model)

    def test_detached_encoder_keeps_head_gradients(self):
        gradients = self.assert_same_step(tiny_model(), detach_encoder=True)
        self.assertTrue(all(g is None for n, g in gradients.items() if n.startswith("encoder.")))
        self.assertIsNotNone(gradients["head.layers.0.linear1.weight"])

    def test_frozen_head_inputs_still_train_head_parameters(self):
        model = tiny_model()
        for parameter in list(model.encoder.parameters()) + list(model.type_emb.parameters()):
            parameter.requires_grad_(False)
        gradients = self.assert_same_step(model)
        for i in range(2):
            gradient = gradients["head.layers.%d.linear1.weight" % i]
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_eval_and_no_grad_bypass_checkpointing(self):
        for training, grad_enabled in ((False, True), (False, False), (True, False)):
            with self.subTest(training=training, grad_enabled=grad_enabled):
                model = tiny_model().train(training)
                batch = inputs()
                with torch.set_grad_enabled(grad_enabled):
                    torch.manual_seed(123)
                    expected = model(*batch)
                    model.head_checkpointing = True
                    with patch("laya.common.checkpoint", side_effect=AssertionError("unexpected checkpoint")):
                        torch.manual_seed(123)
                        actual = model(*batch)
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_zero_head_layers_remain_supported(self):
        self.assert_same_step(tiny_model(head_layers=0))

    def test_default_is_disabled(self):
        model = tiny_model().train()
        self.assertFalse(model.head_checkpointing)
        with patch("laya.common.checkpoint", side_effect=AssertionError("unexpected checkpoint")):
            training_step(model)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
