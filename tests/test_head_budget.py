"""Auto-allocate head_max_len so high-cardinality choice keeps 4 tokens per option.

Arithmetic only for head_budget_for / Agent._resolve_head_budget (no Hub download).
The predict path is a hand-built Agent: encoding is real, the forward pass is stubbed.
system_one delegates to predict_batch, so the opt-in lives there and still composes with
per-call budgets, start hooks, truncation direction, and predict_shortlist.
"""
import inspect
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import (  # noqa: E402
    HEAD_OPTION_SLACK,
    MIN_OPTION_TOKENS,
    build_sequence,
    head_budget_for,
    render_options,
)
from laya.shortlist import predict_shortlist  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc as e:
        PASS.append(name)
        return e
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: raised %r, expected %s" % (name, e, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))
    return None


# --------------------------------------------------------------- named floor
check("MIN_OPTION_TOKENS is 4", MIN_OPTION_TOKENS, 4)
check("HEAD_OPTION_SLACK is 16", HEAD_OPTION_SLACK, 16)
_bs = inspect.getsource(build_sequence)
check_true("build_sequence uses MIN_OPTION_TOKENS", "MIN_OPTION_TOKENS" in _bs)
check_true("build_sequence uses HEAD_OPTION_SLACK", "HEAD_OPTION_SLACK" in _bs)


# --------------------------------------------------------------- k=4 stays put
b = head_budget_for(4, 192, 512, 512)
check("k4/tokens_per_option >= 4", b.tokens_per_option >= 4, True)
check("k4/raised False", b.raised, False)
check("k4/ok", b.ok, True)
check("k4/head unchanged", b.head_max_len, 192)
check("k4/max_len unchanged", b.max_len, 512)
check("k4/tpo formula", b.tokens_per_option, max(1, (192 - 16) // 4))


# --------------------------------------------------------------- Banking77 multilingual defaults
b = head_budget_for(77, 256, 1024, 1024)
check("k77/ok", b.ok, True)
check("k77/raised True", b.raised, True)
check_true("k77/tokens_per_option >= 4", b.tokens_per_option >= 4, "got %r" % b.tokens_per_option)
check_true("k77/head >= 16+4*77", b.head_max_len >= 16 + 4 * 77, "got %r" % b.head_max_len)
check("k77/smallest head", b.head_max_len, 16 + 4 * 77)
check("k77/tokens_per_option == 4", b.tokens_per_option, 4)
check("k77/max_len stays in encoder", b.max_len, 1024)


# --------------------------------------------------------------- 255 options cannot fit in 512
b = head_budget_for(255, 192, 512, 512)
check("k255/ok is False", b.ok, False)
check("k255/raised False", b.raised, False)
check("k255/keeps default head", b.head_max_len, 192)
check("k255/keeps default max_len", b.max_len, 512)


# --------------------------------------------------------------- k < 2 never raises
b = head_budget_for(1, 16, 512, 512)
check("k1/ok even if tpo < 4", b.ok, True)
check("k1/raised False", b.raised, False)
check("k0/ok", head_budget_for(0, 192, 512, 512).ok, True)
check("k0/raised", head_budget_for(0, 192, 512, 512).raised, False)


# --------------------------------------------------------------- english 77-way on a 1024 encoder
b = head_budget_for(77, 192, 512, 1024)
check("en77/ok", b.ok, True)
check("en77/raised", b.raised, True)
check("en77/head", b.head_max_len, 16 + 4 * 77)
check_true("en77/max_len grew with doc room", b.max_len >= 324 + 64, "got %r" % b.max_len)
check_true("en77/max_len <= encoder", b.max_len <= 1024, "got %r" % b.max_len)
check("en77/max_len", b.max_len, 644)


# --------------------------------------------------------------- Agent._resolve_head_budget (no weights)
cfg = {"head_max_len": 256, "max_len": 1024}
low = {
    "dept": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"a": None, "b": None, "c": None, "d": None},
    }
}
head, mx, report = Agent._resolve_head_budget(low, cfg)
check("resolve/k4 head unchanged", head, 256)
check("resolve/k4 max unchanged", mx, 1024)
check("resolve/k4 raised", report["dept"]["raised"], False)
check("resolve/k4 k", report["dept"]["k"], 4)

labels = ["l%02d" % i for i in range(77)]
high = {
    "intent": {
        "type": "choice",
        "instructions": "Which Banking77 intent?",
        "criteria": labels,
    }
}
head, mx, report = Agent._resolve_head_budget(high, cfg)
check("resolve/k77 raised", report["intent"]["raised"], True)
check("resolve/k77 k", report["intent"]["k"], 77)
check_true("resolve/k77 tpo >= 4", report["intent"]["tokens_per_option"] >= 4,
           "got %r" % report["intent"]["tokens_per_option"])
check_true("resolve/k77 head >= 16+4*77", head >= 16 + 4 * 77, "got %r" % head)
check("resolve/k77 report head matches", report["intent"]["head_max_len"], head)
check("resolve/k77 head", head, 324)
check("resolve/k77 max_len", mx, 1024)
check("resolve/k77 tpo", report["intent"]["tokens_per_option"], 4)
check("resolve/k77 report", report["intent"], {
    "k": 77, "tokens_per_option": 4, "head_max_len": 324, "max_len": 1024, "raised": True,
})

# one forward, one cfg: mixed low+high takes the max required head
mixed = dict(low)
mixed.update(high)
head, mx, report = Agent._resolve_head_budget(mixed, cfg)
check("resolve/mixed shares raised head", report["dept"]["head_max_len"], report["intent"]["head_max_len"])
check("resolve/mixed raised on both", report["dept"]["raised"] and report["intent"]["raised"], True)
check("resolve/mixed k preserved", (report["dept"]["k"], report["intent"]["k"]), (4, 77))

# overflow stays at defaults so predict can still raise
overflow = {
    "huge": {
        "type": "choice",
        "instructions": "pick",
        "criteria": ["c%d" % i for i in range(255)],
    }
}
head, mx, report = Agent._resolve_head_budget(overflow, {"head_max_len": 192, "max_len": 512, "encoder_max": 512})
check("resolve/255 keeps default head", head, 192)
check("resolve/255 report raised", report["huge"]["raised"], False)
check("resolve/255 k", report["huge"]["k"], 255)

# noul is 2 options; never needs a raise at defaults
noul = {"flag": {"type": "noul", "instructions": "is this true?"}}
head, mx, report = Agent._resolve_head_budget(noul, {"head_max_len": 192, "max_len": 512})
check("resolve/noul k", report["flag"]["k"], 2)
check("resolve/noul raised", report["flag"]["raised"], False)


# --------------------------------------------------------------- predict wiring (batching + hooks)
import laya.agent as _agent  # noqa: E402

_src = inspect.getsource(_agent.Agent.system_one)
_batch = inspect.getsource(_agent.Agent.predict_batch)
check_true("system_one/auto_head_budget defaults False",
           "auto_head_budget: bool = False" in _src)
check_true("system_one/persist defaults False", "persist: bool = False" in _src)
check_true("system_one/forwards auto_head_budget", "auto_head_budget=auto_head_budget" in _src)
check_true("system_one/forwards persist", "persist=persist" in _src)
check_true("system_one/still delegates to predict_batch", "self.predict_batch(" in _src)
check_true("predict_batch/auto_head_budget defaults False",
           "auto_head_budget: bool = False" in _batch)
check_true("predict_batch/persist defaults False", "persist: bool = False" in _batch)
check_true("predict_batch/auto is a local flag",
           "auto = bool(auto_head_budget or cfg_auto)" in _batch)
check_true("predict_batch/resolve only when auto",
           "if auto:" in _batch and "_resolve_head_budget" in _batch)
check_true("predict_batch/persist writes cfg", 'self.cfg["head_max_len"] = head_max_len' in _batch)
check_true("predict_batch/report only on auto path",
           'usage["head_budget"] = head_budget_report' in _batch)
check_true("system_one/no two_stage_choice", "two_stage_choice" not in _src)
check_true("predict_batch/no two_stage_choice", "two_stage_choice" not in _batch)
check_true("helper/no two_stage_choice", "two_stage_choice" not in inspect.getsource(head_budget_for))
check_true("predict is system_one", _agent.Agent.predict is _agent.Agent.system_one)

# staticmethod: callable without an instance / without Hub
check_true("_resolve_head_budget is static",
           isinstance(inspect.getattr_static(_agent.Agent, "_resolve_head_budget"), staticmethod))


class _FakeTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in str(text).split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _bare(cfg):
    """Real predict path, no weights: encoding runs, the forward pass does not."""
    agent = Agent.__new__(Agent)
    agent.cfg = dict(cfg)
    agent.tok = _FakeTok()
    agent.model_id = "test"

    def _forward(b):
        n = b["input_ids"].shape[0]
        k = b["marker_mask"].shape[1]
        return np.zeros((n, max(k, 1)), dtype=np.float32), np.full((n, 2), 0.5, dtype=np.float32)

    def _decode(logits, act, items, ids, internal, offset, lang=None):
        return {qid: {"ok": True} for qid in ids}

    agent._forward = _forward
    agent._decode_answers = _decode
    return agent


def _choice(k, qid="intent"):
    return {qid: {"type": "choice", "instructions": "Which?", "criteria": ["l%02d" % i for i in range(k)]}}


_captured = []


def _record_build(tok, state, q, max_len, head_max_len, option_order=None, truncate_left=False, state_ids=None):
    _captured.append({
        "max_len": max_len,
        "head_max_len": head_max_len,
        "truncate_left": truncate_left,
        "k": len(render_options(q)),
        "list_state": isinstance(state, list),
    })
    n = len(render_options(q))
    return list(range(n + 2)), list(range(n))


def _run(agent, questions, **kwargs):
    _captured.clear()
    with patch("laya.agent.build_sequence", side_effect=_record_build):
        return agent.system_one("state", questions, **kwargs)


ML = {"head_max_len": 256, "max_len": 1024, "encoder_max_len": 1024}
q77 = _choice(77)
agent = _bare(ML)
out = _run(agent, q77)
check("default/head stays cfg", _captured[-1]["head_max_len"], 256)
check("default/max stays cfg", _captured[-1]["max_len"], 1024)
check_true("default/no head_budget key", "head_budget" not in out["usage"], str(out["usage"]))
check("default/cfg untouched", (agent.cfg["head_max_len"], agent.cfg["max_len"]), (256, 1024))
check("default/caller criteria intact", len(q77["intent"]["criteria"]), 77)

agent = _bare(ML)
out = _run(agent, q77, auto_head_budget=True)
check("auto/head raised", _captured[-1]["head_max_len"], 324)
check("auto/max stays encoder", _captured[-1]["max_len"], 1024)
check("auto/report", out["usage"]["head_budget"]["intent"], {
    "k": 77, "tokens_per_option": 4, "head_max_len": 324, "max_len": 1024, "raised": True,
})
check("auto/cfg not persisted", (agent.cfg["head_max_len"], agent.cfg["max_len"]), (256, 1024))

agent = _bare(ML)
_run(agent, q77, auto_head_budget=True, persist=True)
check("persist/writes head", agent.cfg["head_max_len"], 324)
check("persist/writes max", agent.cfg["max_len"], 1024)
check_true("persist/keeps encoder cap", agent.cfg.get("encoder_max_len") == 1024)

agent = _bare(ML)
_run(agent, q77, persist=True)
check("persist/ignored when auto is off", (agent.cfg["head_max_len"], agent.cfg["max_len"]), (256, 1024))

agent = _bare(dict(ML, auto_head_budget=True))
out = _run(agent, q77)
check("cfg flag/raises without the kwarg", out["usage"]["head_budget"]["intent"]["head_max_len"], 324)
check("cfg flag/build head", _captured[-1]["head_max_len"], 324)

# an explicit budget that already clears the floor is the baseline, not a ceiling to replace
agent = _bare(ML)
out = _run(agent, q77, auto_head_budget=True, head_max_len=512)
check("explicit/already enough stays", _captured[-1]["head_max_len"], 512)
check("explicit/not marked raised", out["usage"]["head_budget"]["intent"]["raised"], False)

# a too-small explicit budget is what gets raised, and max_len grows inside the encoder
agent = _bare(ML)
out = _run(agent, q77, auto_head_budget=True, head_max_len=192, max_len=512)
check("explicit/small head raised", _captured[-1]["head_max_len"], 324)
check("explicit/max grew with doc room", _captured[-1]["max_len"], 644)
check("explicit/report max", out["usage"]["head_budget"]["intent"]["max_len"], 644)

# start hook runs first; its budget is the baseline auto raises from
agent = _bare(ML)
seen = {}


def _hook_small(ctx):
    ctx.head_max_len = 128
    ctx.max_len = 512


def _hook_wide(ctx):
    ctx.head_max_len = 512


def _see_end(ctx):
    seen["budget"] = ctx.results[0]["usage"].get("head_budget")
    seen["ctx_head"] = ctx.head_max_len


out = _run(agent, q77, auto_head_budget=True, on_predict_start=_hook_small, on_predict_end=_see_end)
check("hook/small baseline raised", _captured[-1]["head_max_len"], 324)
check("hook/doc room from hook max", _captured[-1]["max_len"], 708)
check("hook/end sees report", seen["budget"]["intent"]["raised"], True)
check("hook/end sees applied head", seen["ctx_head"], 324)

agent = _bare(ML)
out = _run(agent, q77, auto_head_budget=True, on_predict_start=_hook_wide)
check("hook/wide enough is kept", _captured[-1]["head_max_len"], 512)
check("hook/wide not raised", out["usage"]["head_budget"]["intent"]["raised"], False)

# list state still truncates from the left while the head is raised (#181)
agent = _bare(ML)
_captured.clear()
with patch("laya.agent.build_sequence", side_effect=_record_build):
    agent.system_one([{"role": "user", "content": "old"}, {"role": "user", "content": "new"}],
                     q77, auto_head_budget=True)
check("truncation/list still left", _captured[-1]["truncate_left"], True)
check("truncation/list still raised head", _captured[-1]["head_max_len"], 324)

agent = _bare(ML)
_captured.clear()
with patch("laya.agent.build_sequence", side_effect=_record_build):
    agent.system_one("plain text", q77)
check("truncation/string stays right", _captured[-1]["truncate_left"], False)
check("truncation/string default head", _captured[-1]["head_max_len"], 256)

# one forward, every state shares the raised budget
agent = _bare(ML)
_captured.clear()
with patch("laya.agent.build_sequence", side_effect=_record_build):
    batch = agent.predict_batch(["a", "b"], q77, auto_head_budget=True)
check("batch/both raised", [c["head_max_len"] for c in _captured], [324, 324])
check("batch/both report the same head",
      (batch[0]["usage"]["head_budget"]["intent"]["head_max_len"],
       batch[1]["usage"]["head_budget"]["intent"]["head_max_len"]),
      (324, 324))

# cannot fit: keep the default and still raise, even if persist was asked
agent = _bare({"head_max_len": 192, "max_len": 512, "encoder_max": 512})
err = check_raises(
    "overflow/still raises",
    ValueError,
    lambda: agent.system_one("state", _choice(255, "huge"), auto_head_budget=True, persist=True),
)
check_true("overflow/names the kept head",
           err is not None and "head_max_len=192" in str(err), str(err))
check("overflow/cfg unchanged", (agent.cfg["head_max_len"], agent.cfg["max_len"]), (192, 512))

# validation still runs before allocation, so a bad question names itself
err = check_raises(
    "validate/before allocate",
    ValueError,
    lambda: _bare(ML).system_one("state", {"q": {"type": "choice", "instructions": "x"}},
                                 auto_head_budget=True),
)
check_true("validate/names criteria", err is not None and "criteria" in str(err), str(err))


# --------------------------------------------------------------- shortlist runs first (#106)
def _embed(texts):
    arr = np.zeros((len(texts), 2), dtype=np.float64)
    arr[0, 0] = 1.0
    for i in range(1, len(texts)):
        arr[i, 0] = 1.0 / i
    return arr


agent = _bare(ML)
caller = _choice(77)
_captured.clear()
with patch("laya.agent.build_sequence", side_effect=_record_build):
    short = predict_shortlist(agent, "transfer fee", caller, _embed, k=4, auto_head_budget=True)
check("shortlist/allocator sees kept labels", _captured[-1]["k"], 4)
check("shortlist/no raise after narrowing", _captured[-1]["head_max_len"], 256)
check("shortlist/report k", short["usage"]["head_budget"]["intent"]["k"], 4)
check("shortlist/report not raised", short["usage"]["head_budget"]["intent"]["raised"], False)
check("shortlist/meta n", short["shortlist"]["intent"]["n"], 77)
check("shortlist/caller criteria intact", len(caller["intent"]["criteria"]), 77)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
