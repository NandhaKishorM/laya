"""Exact CPU window attention and combined model parity; no weights or network.

    python tests/test_cpu_window_attention.py
"""
import copy
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import ModernBertConfig, ModernBertModel, __version__ as transformers_version

from laya.common import DecisionModel, _can_use_cpu_windows, _cpu_window_attention, _cpu_window_encoder


class WindowKernelTests(unittest.TestCase):
    def test_boundaries_padding_and_overlapping_blocks(self):
        torch.manual_seed(31)
        with torch.inference_mode():
            for length in (1, 31, 64, 65, 129, 257):
                for radius in (1, 8, 64):
                    for block in (32, 64):
                        with self.subTest(length=length, radius=radius, block=block):
                            q, k, v = (torch.randn(3, 2, length, 16) for _ in range(3))
                            valid = torch.ones(3, length, dtype=torch.bool)
                            valid[1, length // 2:] = False
                            valid[2] = False
                            pos = torch.arange(length)
                            mask = (pos[:, None] - pos[None]).abs() <= radius
                            mask = mask[None, None] & valid[:, None, None, :]
                            expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                            actual = _cpu_window_attention(q, k, v, valid, radius, block)
                            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
                            self.assertTrue(torch.isfinite(actual).all())


class WindowModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        cfg = ModernBertConfig(vocab_size=256, hidden_size=64, intermediate_size=96,
                               num_hidden_layers=3, num_attention_heads=4, local_attention=16,
                               global_attn_every_n_layers=2, max_position_embeddings=1024,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
        if transformers_version.split(".")[0] != "5" or not hasattr(cfg, "layer_types"):
            self.skipTest("Transformers 4.x keeps the stock encoder forward")
        cfg._attn_implementation = "sdpa"
        self.encoder = ModernBertModel(cfg).eval()

    def inputs(self, length=513):
        ids = torch.randint(3, 256, (2, length))
        mask = torch.ones_like(ids)
        mask[1, length // 2:] = 0
        return ids, mask

    def test_encoder_and_combined_decision_outputs(self):
        model = DecisionModel(self.encoder, head_layers=2).eval()
        with torch.inference_mode():
            for length in (512, 513, 769):
                ids, mask = self.inputs(length)
                self.assertTrue(_can_use_cpu_windows(self.encoder, ids, mask))
                expected = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                actual = _cpu_window_encoder(self.encoder, ids, mask)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
                pos = torch.tensor([[1, 4, 7], [1, 4, -1]])
                valid = pos >= 0
                qtype = torch.tensor([0, 1])
                with patch("laya.common._can_use_cpu_windows", return_value=False), \
                        patch("laya.common._can_use_cpu_head", return_value=False):
                    expected = model(ids, mask, pos, valid, qtype)
                with patch("laya.common._cpu_window_encoder", wraps=_cpu_window_encoder) as call:
                    actual = model(ids, mask, pos, valid, qtype)
                self.assertEqual(call.call_count, 1)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-5)

    def test_fallbacks_and_hook_observability(self):
        ids, mask = self.inputs()
        with torch.no_grad():
            self.assertFalse(_can_use_cpu_windows(self.encoder, ids[:, :128], mask[:, :128]))
            self.assertFalse(_can_use_cpu_windows(self.encoder.double(), ids, mask))
            self.encoder.float()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertFalse(_can_use_cpu_windows(self.encoder, ids, mask))
            self.encoder.train()
            self.assertFalse(_can_use_cpu_windows(self.encoder, ids, mask))
            self.encoder.eval()
            for target in (self.encoder, self.encoder.layers[1], self.encoder.layers[1].attn):
                calls = []
                handle = target.register_forward_hook(lambda module, args, out: calls.append(True))
                try:
                    self.assertFalse(_can_use_cpu_windows(self.encoder, ids, mask))
                    model = DecisionModel(self.encoder).eval()
                    with patch("laya.common._cpu_window_encoder", side_effect=AssertionError("bypassed hook")):
                        model(ids, mask, torch.ones(2, 1, dtype=torch.long),
                              torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
                    self.assertEqual(calls, [True])
                finally:
                    handle.remove()
            for predicate in ("torch._dynamo.is_compiling", "torch.jit.is_tracing", "torch.jit.is_scripting"):
                with patch(predicate, return_value=True):
                    self.assertFalse(_can_use_cpu_windows(self.encoder, ids, mask))
            custom = copy.deepcopy(self.encoder)
            custom.layers[1].forward = lambda h, **kwargs: h
            self.assertFalse(_can_use_cpu_windows(custom, ids, mask))
            custom.layers[1] = nn.Identity()
            self.assertFalse(_can_use_cpu_windows(custom, ids, mask))
            self.assertFalse(_can_use_cpu_windows(self.encoder, ids, None))
        self.assertFalse(_can_use_cpu_windows(self.encoder, ids, mask))  # gradients enabled

    def test_parallel_layout_keeps_the_original_mask_path(self):
        model = DecisionModel(self.encoder).eval()
        ids, mask = self.inputs()
        positions = torch.arange(ids.shape[1])[None].expand_as(ids)
        options = torch.zeros_like(ids)
        options[:, 1:4] = 1
        markers = torch.tensor([[1, 4], [1, 4]])
        with torch.inference_mode(), patch("laya.common._cpu_window_encoder", side_effect=AssertionError("parallel mask")):
            logits, action = model(ids, mask, markers, torch.ones_like(markers, dtype=torch.bool),
                                   torch.zeros(2, dtype=torch.long), position_ids=positions, option_ids=options)
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(action).all())

    def test_registered_attention_and_mask_backends_are_not_bypassed(self):
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
        model = DecisionModel(self.encoder).eval()
        ids, mask = self.inputs()
        for registry, expected_calls in ((AttentionInterface, 3), (AttentionMaskInterface, 2)):
            original = registry._global_mapping["sdpa"]
            with patch.dict(registry._global_mapping, {"sdpa": unittest.mock.Mock(wraps=original)}):
                registered = registry._global_mapping["sdpa"]
                with torch.inference_mode(), patch("laya.common._cpu_window_encoder", side_effect=AssertionError("custom backend")):
                    model(ids, mask, torch.ones(2, 1, dtype=torch.long),
                          torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
                self.assertEqual(registered.call_count, expected_calls)

    def test_parameters_and_serialization_unchanged(self):
        state = {k: v.clone() for k, v in self.encoder.state_dict().items()}
        versions = [(id(p), p._version) for p in self.encoder.parameters()]
        with torch.inference_mode():
            _cpu_window_encoder(self.encoder, *self.inputs())
        self.assertEqual(versions, [(id(p), p._version) for p in self.encoder.parameters()])
        for k, v in self.encoder.state_dict().items():
            torch.testing.assert_close(v, state[k], atol=0, rtol=0)
        copy.deepcopy(self.encoder).load_state_dict(state, strict=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
