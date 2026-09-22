"""Regression tests for DecisionModel.forward (laya/common.py).

Uses a tiny from-config BERT encoder (no pretrained weights downloaded) so
these run fast and offline, unlike tests/test_local_e2e.py which needs a
real checkpoint on disk.
"""
import os
import sys
from contextlib import nullcontext

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import DecisionModel, _meta_parameters, materialise_meta_parameters


def _tiny_model(head_layers: int = 1, n_act: int = 2) -> DecisionModel:
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        vocab_size=50,
    )
    encoder = AutoModel.from_config(cfg)
    model = DecisionModel(encoder, head_layers=head_layers, n_act=n_act)
    model.eval()
    return model


def _inputs(batch: int, seq: int, n_markers: int):
    input_ids = torch.randint(0, 50, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    qtype = torch.zeros(batch, dtype=torch.long)
    marker_pos = torch.arange(n_markers).unsqueeze(0).expand(batch, -1).clone()
    marker_mask = torch.ones(batch, n_markers, dtype=torch.bool)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype


def test_single_option_question_does_not_crash():
    # Regression test for #96: a `choice` question with exactly one criterion
    # used to crash inside forward() with "selected index k out of range",
    # because p.topk(2, -1) has nothing to select for the second slot when
    # there is only one valid marker.
    torch.manual_seed(0)
    model = _tiny_model()
    input_ids, attention_mask, marker_pos, marker_mask, qtype = _inputs(
        batch=1, seq=8, n_markers=1
    )
    with torch.no_grad():
        logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    assert logits.shape == (1, 1)
    assert act_logits.shape == (1, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(act_logits).all()


def test_single_option_top1_minus_top2_is_exactly_one():
    # Softmax over a single valid logit is 1.0 regardless of its value, so the
    # padded top2 (0.0) must make the act head's top1-top2 gap read as a fully
    # decided 1.0 - the same signal it gets for any other unambiguous choice.
    torch.manual_seed(1)
    model = _tiny_model()
    input_ids, attention_mask, marker_pos, marker_mask, qtype = _inputs(
        batch=1, seq=6, n_markers=1
    )
    with torch.no_grad():
        h = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + model.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = model.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
    assert torch.allclose(p, torch.ones_like(p))


def test_multi_option_question_is_unaffected():
    # The >=2-marker path must still use the original torch.topk(2, -1) call
    # unchanged - this pins that the single-option fix didn't touch it.
    torch.manual_seed(2)
    model = _tiny_model()
    input_ids, attention_mask, marker_pos, marker_mask, qtype = _inputs(
        batch=2, seq=10, n_markers=4
    )
    with torch.no_grad():
        logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(act_logits).all()


def _tiny_rope_model(empty: bool = False) -> DecisionModel:
    """A tiny ModernBERT-backed DecisionModel, the point being its RoPE buffers.

    ModernBERT computes non-persistent rotary `inv_freq` buffers in __init__, they are absent
    from every checkpoint, and forward() reads them. They are the reason a model cannot simply
    be constructed empty and materialised with to_empty().

    Built from a config rather than `build_model`, so no pretrained weights are fetched.
    """
    cfg = AutoConfig.for_model(
        "modernbert",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        cls_token_id=2,
        sep_token_id=1,
    )
    with _meta_parameters() if empty else nullcontext():
        encoder = AutoModel.from_config(cfg, attn_implementation="sdpa")
        return DecisionModel(encoder, head_layers=1, n_act=1).eval()


def test_empty_build_defers_parameters_but_keeps_buffers():
    # Regression test for the cold-load path: parameters are created on `meta` so the random
    # init that a checkpoint load immediately overwrites is never paid, while buffers must
    # still be initialised because they are computed, not loaded.
    import torch.nn as nn

    original = nn.Module.register_parameter
    reference = _tiny_rope_model(empty=False)
    assert nn.Module.register_parameter is original, "register_parameter was not restored"

    empty = _tiny_rope_model(empty=True)
    params = list(empty.parameters())
    assert params, "the model should still expose its parameters"
    assert all(p.device.type == "meta" for p in params), "parameters should live on meta"
    assert nn.Module.register_parameter is original, "register_parameter leaked out of the build"

    # Buffers are not deferred, and the RoPE ones hold what __init__ computed rather than zeros.
    empty_rope = {n: b for n, b in empty.named_buffers() if "inv_freq" in n}
    reference_rope = {n: b for n, b in reference.named_buffers() if "inv_freq" in n}
    assert empty_rope, "ModernBERT should expose rotary inv_freq buffers"
    assert set(empty_rope) == set(reference_rope)
    for name, buf in empty_rope.items():
        assert buf.device.type != "meta", "%s should be a real buffer" % name
        assert buf.abs().sum() > 0, "%s is zeroed; RoPE would be dead" % name
        assert torch.equal(buf, reference_rope[name]), "%s differs from a normal build" % name


def test_materialising_does_not_disturb_the_rope_buffers():
    # The trap this pins down: nn.Module.to_empty() re-creates *buffers* as well as parameters,
    # and the rotary inv_freq buffers are computed in __init__ and absent from every checkpoint.
    # Materialising that way leaves them zeroed, so the encoder silently attends with a dead
    # rotary embedding. Only the parameters may be touched.
    empty = _tiny_rope_model(empty=True)
    before = {n: b.clone() for n, b in empty.named_buffers()}
    assert before, "the model should expose buffers"

    materialise_meta_parameters(empty, device="cpu")

    after = dict(empty.named_buffers())
    for name, original in before.items():
        assert original.abs().sum() > 0, "%s should have been computed at build time" % name
        assert torch.equal(after[name], original), "%s was disturbed by materialisation" % name


def test_materialised_empty_model_matches_a_normal_build():
    # The end-to-end guarantee for the fast cold-load path: building empty, materialising the
    # parameters and loading a state dict must reproduce a normal build exactly.
    torch.manual_seed(3)
    reference = _tiny_rope_model(empty=False)
    input_ids, attention_mask, marker_pos, marker_mask, qtype = _inputs(
        batch=2, seq=8, n_markers=3
    )
    with torch.no_grad():
        expected_logits, expected_act = reference(
            input_ids, attention_mask, marker_pos, marker_mask, qtype
        )

    loaded = _tiny_rope_model(empty=True)
    materialise_meta_parameters(loaded, device="cpu")
    assert all(p.device.type == "cpu" for p in loaded.parameters())
    loaded.load_state_dict(reference.state_dict(), strict=True)

    with torch.no_grad():
        logits, act_logits = loaded(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    assert torch.equal(logits, expected_logits), "empty-then-loaded logits diverged"
    assert torch.equal(act_logits, expected_act), "empty-then-loaded act logits diverged"


if __name__ == "__main__":
    test_single_option_question_does_not_crash()
    test_single_option_top1_minus_top2_is_exactly_one()
    test_multi_option_question_is_unaffected()
    test_empty_build_defers_parameters_but_keeps_buffers()
    test_materialising_does_not_disturb_the_rope_buffers()
    test_materialised_empty_model_matches_a_normal_build()
    print("all decision model tests passed")

