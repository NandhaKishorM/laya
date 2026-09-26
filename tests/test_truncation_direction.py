"""Regression: Agent.system_one preserves the newest conversation turn.

The public Agent API advertises `state` as accepting a chronological
conversation-turn list. build_sequence defaults to truncate_left=False
(st[:room]), which preserves the head and drops the tail — for a chronological
list that means the newest turn is silently lost. For list-shaped state the agent
now truncates from the left so the most recent intent survives. Strings and
dicts are unaffected (backward compatible).
"""
import json

import torch
from transformers import AutoConfig, AutoModel

from laya.agent import Agent
from laya.common import DecisionModel, build_sequence, serialize_state


class _FakeTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _tiny_agent():
    agent = object.__new__(Agent)
    cfg = AutoConfig.for_model("bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
                               intermediate_size=32, vocab_size=64)
    agent.cfg = {"max_len": 30, "head_max_len": 12, "encoder": "tiny"}
    agent.tok = _FakeTok()
    agent.model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2).eval()
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    return agent


def _state_prefix_len(tok, q, max_len, head_max_len):
    ref, _ = build_sequence(tok, "", q, max_len, head_max_len)
    return len(ref) - 1


Q = {"t": "choice", "ins": "What action?", "crit": {"refund": "money back", "escalate": "manager", "hold": "wait"}}


def test_agent_list_vs_string_truncation_direction():
    """Agent must truncate left for lists, right for strings (via mock inspection)."""
    from unittest.mock import patch

    agent = _tiny_agent()
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}

    list_capture = {}
    string_capture = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, **kwargs):
        if isinstance(state, list):
            list_capture["truncate_left"] = truncate_left
        else:
            string_capture["truncate_left"] = truncate_left
        return build_sequence(tok, state, q, max_len, head_max_len,
                              truncate_left=truncate_left, **kwargs)

    with patch("laya.agent.build_sequence", side_effect=fake_build):
        agent.system_one([{"role": "user", "content": "hi"}, {"role": "user", "content": "newest"}], questions)
        agent.system_one("string state", questions)

    assert list_capture["truncate_left"] is True
    assert string_capture["truncate_left"] is False


def test_agent_system_one_passes_truncate_left_for_list():
    """Agent.system_one must pass truncate_left=True for list state."""
    from unittest.mock import patch

    agent = _tiny_agent()
    conversation = [{"role": "user", "content": "hello"}, {"role": "user", "content": "NEWEST"}]
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}

    captured = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, **kwargs):
        captured["truncate_left"] = truncate_left
        captured["state"] = state
        return build_sequence(tok, state, q, max_len, head_max_len,
                              truncate_left=truncate_left, **kwargs)

    with patch("laya.agent.build_sequence", side_effect=fake_build):
        agent.system_one(conversation, questions)

    assert captured["truncate_left"] is True, "list state must trigger truncate_left=True"
    assert isinstance(captured["state"], list)


def test_agent_system_one_string_state_default_truncation():
    """Agent.system_one with string state keeps default (truncate_left=False)."""
    from unittest.mock import patch

    agent = _tiny_agent()
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}

    captured = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, **kwargs):
        captured["truncate_left"] = truncate_left
        return build_sequence(tok, state, q, max_len, head_max_len,
                              truncate_left=truncate_left, **kwargs)

    with patch("laya.agent.build_sequence", side_effect=fake_build):
        agent.system_one("just a string state", questions)

    assert captured["truncate_left"] is False, "string state must keep default truncation"


def test_string_state_preserves_head():
    """A string state must still preserve the head (backward compatible)."""
    tok = _FakeTok()
    state = "HEADMARKERWORD " + "filler " * 20 + " TAILMARKER"
    state_ids = tok(state.replace("[MASK]", " "), add_special_tokens=False)["input_ids"]
    head_id = state_ids[0]
    tail_id = state_ids[-1]
    assert head_id != tail_id, "head and tail must have different ids"

    seq, _ = build_sequence(tok, state, Q, 30, 12)
    prefix = _state_prefix_len(tok, Q, 30, 12)
    kept = seq[prefix:-1]

    assert head_id in kept, "string state must preserve the head (default mode)"
    assert tail_id not in kept, "string state must drop the tail (default mode)"


def test_build_sequence_default_unchanged():
    """build_sequence's default behavior is unchanged for non-list callers."""
    tok = _FakeTok()
    state = "OLDFRONT " + "filler " * 20 + " NEWBACK"
    seq_default, _ = build_sequence(tok, state, Q, 30, 12)
    seq_explicit, _ = build_sequence(tok, state, Q, 30, 12, truncate_left=False)
    assert seq_default == seq_explicit, "default must remain truncate_left=False"
