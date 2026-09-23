"""Regression tests for head_checkpointing in DecisionModel.forward.

The flag was being set by the fine-tuning notebook but never consulted in the
forward pass, so the head's activations were always retained for backward even
when the caller asked for them to be recomputed. These tests run on CPU with a
tiny BERT config - no checkpoint download needed.
"""
import os
import sys

import pytest
import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import DecisionModel  # noqa: E402


def _tiny_model(head_layers: int = 2) -> DecisionModel:
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        vocab_size=50,
    )
    encoder = AutoModel.from_config(cfg)
    return DecisionModel(encoder, head_layers=head_layers, n_act=2)


def _inputs(batch: int = 2, seq: int = 12, n_markers: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 50, (batch, seq), generator=g)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    qtype = torch.arange(batch) % 3
    marker_pos = (torch.arange(n_markers) + 1).unsqueeze(0).expand(batch, -1).clone()
    marker_mask = torch.ones(batch, n_markers, dtype=torch.bool)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype


def _run_train_step(model, inputs, use_ckpt, drop_seed=None):
    model.train()
    model.head_checkpointing = use_ckpt
    model.zero_grad(set_to_none=True)
    if drop_seed is not None:
        # Pin the global RNG so both runs draw the same dropout masks. The
        # checkpointed path restores the RNG per layer, so it must agree with the
        # uncheckpointed path when both start from the same seed.
        torch.manual_seed(drop_seed)
    input_ids, attention_mask, marker_pos, marker_mask, qtype = inputs
    logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    loss = logits.square().mean() + act.square().mean()
    loss.backward()
    return loss.item()


def test_off_is_one_call_per_layer():
    # Sanity: with the flag off, each head layer runs exactly once per step.
    torch.manual_seed(0)
    model = _tiny_model(head_layers=1)
    calls = []
    hook = model.head.layers[0].register_forward_pre_hook(lambda m, a, kw=None: calls.append(1))
    inputs = _inputs(batch=1, seq=8, n_markers=2)
    _run_train_step(model, inputs, use_ckpt=False)
    hook.remove()
    assert len(calls) == 1


def test_on_recomputes_each_layer_during_backward():
    # With the flag on during training, each checkpointed layer runs once on
    # forward and again inside backward - the 2nd call is the recompute.
    torch.manual_seed(0)
    model = _tiny_model(head_layers=1)
    calls = []
    hook = model.head.layers[0].register_forward_pre_hook(lambda m, a, kw=None: calls.append(1))
    inputs = _inputs(batch=1, seq=8, n_markers=2)
    _run_train_step(model, inputs, use_ckpt=True)
    hook.remove()
    assert len(calls) == 2


def test_gradients_match_with_and_without_checkpointing():
    # Checkpointing must not change the numerics. Same init, same inputs, and
    # the same dropout seed on both runs: the loss and every parameter gradient
    # must match the uncheckpointed run.
    torch.manual_seed(7)
    ref = _tiny_model(head_layers=2)
    torch.manual_seed(7)
    new = _tiny_model(head_layers=2)
    inputs = _inputs(batch=2, seq=10, n_markers=3, seed=11)

    loss_ref = _run_train_step(ref, inputs, use_ckpt=False, drop_seed=99)
    loss_new = _run_train_step(new, inputs, use_ckpt=True,  drop_seed=99)
    assert loss_ref == pytest.approx(loss_new, rel=1e-5, abs=1e-7)

    named_ref = dict(ref.named_parameters())
    named_new = dict(new.named_parameters())
    for name, p_ref in named_ref.items():
        p_new = named_new[name]
        g_ref = p_ref.grad
        g_new = p_new.grad
        assert (g_ref is None) == (g_new is None), "param %r has gradient on one side only" % name
        if g_ref is not None:
            assert torch.allclose(g_ref, g_new, rtol=1e-5, atol=1e-7), (
                "grad mismatch on %r: max |diff| = %.3g" % (name, (g_ref - g_new).abs().max()))


def test_eval_mode_does_not_checkpoint():
    # During eval there is no graph to release, so the flag must be a no-op
    # and each layer still runs exactly once.
    torch.manual_seed(0)
    model = _tiny_model(head_layers=1).eval()
    model.head_checkpointing = True
    calls = []
    hook = model.head.layers[0].register_forward_pre_hook(lambda m, a, kw=None: calls.append(1))
    inputs = _inputs(batch=1, seq=8, n_markers=2)
    input_ids, attention_mask, marker_pos, marker_mask, qtype = inputs
    model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    hook.remove()
    assert len(calls) == 1


def test_no_grad_does_not_checkpoint():
    # no_grad disables grad tracking outside training too; flag must be a no-op.
    torch.manual_seed(0)
    model = _tiny_model(head_layers=1).train()
    model.head_checkpointing = True
    calls = []
    hook = model.head.layers[0].register_forward_pre_hook(lambda m, a, kw=None: calls.append(1))
    inputs = _inputs(batch=1, seq=8, n_markers=2)
    input_ids, attention_mask, marker_pos, marker_mask, qtype = inputs
    with torch.no_grad():
        model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    hook.remove()
    assert len(calls) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
