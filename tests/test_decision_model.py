"""Regression tests for DecisionModel.forward (laya/common.py).

Uses a tiny from-config BERT encoder (no pretrained weights downloaded) so
these run fast and offline, unlike tests/test_local_e2e.py which needs a
real checkpoint on disk.
"""
import os
import sys
import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import DecisionModel


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


if __name__ == "__main__":
    test_single_option_question_does_not_crash()
    test_single_option_top1_minus_top2_is_exactly_one()
    test_multi_option_question_is_unaffected()
    print("all decision model tests passed")

