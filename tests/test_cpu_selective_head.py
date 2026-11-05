"""CPU final-head parity and fallback coverage; no checkpoint or network required.

    python tests/test_cpu_selective_head.py
"""
import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.nn import functional as F

from laya.common import DecisionModel, _can_select_head, _selected_head


class Encoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.embedding = nn.Embedding(256, dim)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def inputs(length=65, markers=4):
    ids = torch.randint(0, 256, (3, length))
    mask = torch.ones_like(ids)
    mask[1, length // 2:] = 0
    pos = torch.arange(1, markers + 1)[None].expand(3, -1).clone()
    valid = torch.ones_like(pos, dtype=torch.bool)
    if markers > 1:
        valid[1, -1] = False
        pos[1, -1] = -1
        pos[2, -1] = pos[2, 0]  # repeated markers are still separate output slots
    return ids, mask, pos, valid, torch.tensor([0, 1, 2])


class SelectiveHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DecisionModel(Encoder(), head_layers=2).eval()

    def reference(self, model, batch):
        with patch("laya.common._can_select_head", return_value=False):
            return model(*batch)

    def test_logits_action_and_probabilities_match_full_forward(self):
        for layers in (0, 1, 2, 3):
            model = DecisionModel(Encoder(), head_layers=layers).eval()
            for length in (17, 65, 257):
                for markers in (1, 4, 8):
                    with self.subTest(layers=layers, length=length, markers=markers), torch.inference_mode():
                        batch = inputs(length, markers)
                        expected = self.reference(model, batch)
                        with patch("laya.common._selected_head", wraps=_selected_head) as selected:
                            actual = model(*batch)
                        self.assertEqual(selected.call_count, int(layers > 0))
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
                        torch.testing.assert_close(actual[0].softmax(-1), expected[0].softmax(-1),
                                                   atol=2e-6, rtol=1e-5)
                        if markers > 1:
                            # Repeated markers (and repeated token IDs in this position-free
                            # encoder) create exact ties; roundoff can choose either tied slot.
                            top = expected[0].topk(2, dim=-1).values
                            clear = (top[:, 0] - top[:, 1]) > 4e-6
                            self.assertTrue(torch.equal(actual[0].argmax(-1)[clear],
                                                        expected[0].argmax(-1)[clear]))

    def test_layer_retains_all_keys_and_only_requested_queries(self):
        for bias in (False, True):
            for activation in ("relu", "gelu"):
                layer = nn.TransformerEncoderLayer(128, 4, 256, batch_first=True,
                                                  norm_first=True, bias=bias, activation=activation).eval()
                h = torch.randn(2, 65, 128)
                pad = torch.zeros(2, 65, dtype=torch.bool)
                pad[1, 45:] = True
                positions = torch.tensor([[0, 3, 64], [0, 4, 4]])
                with torch.no_grad():
                    expected = layer(h, src_key_padding_mask=pad).gather(
                        1, positions[:, :, None].expand(-1, -1, 128))
                    with patch("laya.common.F.scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa:
                        actual = _selected_head(layer, h, positions, pad)
                    self.assertEqual(sdpa.call_args.args[0].shape[-2], 3)
                    self.assertEqual(sdpa.call_args.args[1].shape[-2], 65)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
                    changed = h.clone()
                    changed[:, 30] += torch.randn(128) * 10
                    self.assertGreater((_selected_head(layer, changed, positions, pad) - actual).abs().max(), 1e-4)

    def test_training_and_gradients_use_original_forward(self):
        for training, gradients in ((True, True), (True, False), (False, True)):
            model = copy.deepcopy(self.model).train(training)
            batch = inputs()
            with torch.set_grad_enabled(gradients):
                torch.manual_seed(7)
                expected = self.reference(model, batch)
                torch.manual_seed(7)
                with patch("laya.common._selected_head", side_effect=AssertionError("selected during training")):
                    actual = model(*batch)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                if gradients:
                    sum(t.sum() for t in actual).backward()
                    self.assertIsNotNone(model.head.layers[-1].linear1.weight.grad)

    def test_layer_and_child_hooks_see_full_sequence(self):
        for target in (self.model.head.layers[-1], self.model.head.layers[-1].linear1):
            calls = []
            handle = target.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
            try:
                with torch.no_grad(), patch("laya.common._selected_head", side_effect=AssertionError("bypassed hook")):
                    self.model(*inputs())
                self.assertEqual(calls, [65])
            finally:
                handle.remove()

    def test_global_hooks_use_original_forward(self):
        handle = nn.modules.module.register_module_forward_hook(lambda module, args, result: None)
        try:
            with torch.no_grad(), patch("laya.common._selected_head", side_effect=AssertionError("bypassed global hook")):
                self.model(*inputs())
        finally:
            handle.remove()

    def test_unsupported_layers_and_execution_modes_fall_back(self):
        layer = self.model.head.layers[-1]
        h = torch.randn(3, 65, 64)
        markers = inputs()[2]
        with torch.no_grad():
            self.assertTrue(_can_select_head(layer, h, markers))
            self.assertFalse(_can_select_head(layer, h.double(), markers))
            self.assertFalse(_can_select_head(layer, h.to("meta"), markers))
            self.assertFalse(_can_select_head(layer, h[:, :5], markers))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertFalse(_can_select_head(layer, h, markers))
            for predicate in ("torch._dynamo.is_compiling", "torch.jit.is_tracing", "torch.jit.is_scripting"):
                with patch(predicate, return_value=True):
                    self.assertFalse(_can_select_head(layer, h, markers))
            for key, value in (("norm_first", False), ("activation", lambda x: x.mean(1, keepdim=True)),
                               ("linear1", nn.Sequential(layer.linear1)), ("dropout1", nn.Identity()),
                               ("forward", lambda x, **kwargs: x)):
                custom = copy.deepcopy(layer)
                setattr(custom, key, value)
                self.assertFalse(_can_select_head(custom, h, markers))
            layer.train()
            self.assertFalse(_can_select_head(layer, h, markers))

    def test_state_dict_and_parameter_identity_are_preserved(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        parameters = [(id(p), p._version) for p in self.model.parameters()]
        with torch.inference_mode():
            self.model(*inputs())
        self.assertEqual(parameters, [(id(p), p._version) for p in self.model.parameters()])
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        copy.deepcopy(self.model).load_state_dict(before, strict=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
