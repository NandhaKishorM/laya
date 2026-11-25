"""Offline tests for the embedding shortlist. No checkpoint and no Hub download.

A fake ``embed_fn`` supplies vectors. ``predict`` / ``system_one`` are mocks, so the
decision model is never constructed.
"""
import inspect
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import laya  # noqa: E402
from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel, render_options  # noqa: E402
from laya.shortlist import (  # noqa: E402
    cached_embed_fn,
    embed_fn_from_agent,
    predict_shortlist,
    predict_tournament,
    shortlist_choice,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def check_raises(name, fn, exc=ValueError):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:
        FAIL.append("%s: raised %s (%s), want %s" % (name, type(e).__name__, e, exc.__name__))
    else:
        FAIL.append("%s: no exception" % name)


class TableEmbed:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        missing = [t for t in texts if t not in self.vectors]
        if missing:
            raise AssertionError("unexpected texts %r" % (missing,))
        return [self.vectors[t] for t in texts]


class BoomEmbed:
    def __call__(self, texts):
        raise AssertionError("embed_fn should not run when k >= n")


class Recorder:
    """Stand-in for Agent. Records the questions handed to predict."""

    def __init__(self):
        self.calls = []
        self.system_one_calls = 0

    def predict(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {"model": "fake", "answers": _answers(questions)}

    def system_one(self, state, questions, **kwargs):
        self.system_one_calls += 1
        self.calls.append((state, questions, kwargs))
        return {"model": "fake", "answers": _answers(questions)}


def _answers(questions):
    answers = {}
    for qid, qdef in questions.items():
        if isinstance(qdef, dict) and qdef.get("type") == "choice":
            crit = qdef["criteria"]
            keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            answers[qid] = {"type": "choice", "choice": keys[0]}
    return answers


# Vectors: query [1, 0]. alpha and delta tie at cosine 1; gamma is 0.6; beta is 0.
# Stable order must keep alpha ahead of delta.
VEC = {
    "pay me": [1.0, 0.0],
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "gamma": [0.6, 0.8],
    "delta": [1.0, 0.0],
    "zero": [0.0, 0.0],
}
CRITERIA = {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"}
# option texts follow render_options: bare key when value is None or ""
OPTION_TEXTS = {
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "gamma: mid": [0.6, 0.8],
    "delta: same": [1.0, 0.0],
}


def _embed_for(query_text, option_vectors):
    vectors = {query_text: [1.0, 0.0]}
    vectors.update(option_vectors)
    return TableEmbed(vectors)


# ---------------------------------------------------------------- exports and default path
check_true("export/shortlist_choice", laya.shortlist_choice is shortlist_choice)
check_true("export/predict_shortlist", laya.predict_shortlist is predict_shortlist)
check_true("export/embed_fn_from_agent", laya.embed_fn_from_agent is embed_fn_from_agent)
check_true("export/cached_embed_fn", laya.cached_embed_fn is cached_embed_fn)
for _name in ("shortlist_choice", "predict_shortlist", "embed_fn_from_agent", "cached_embed_fn"):
    check_true("all/%s" % _name, _name in laya.__all__)

_predict_src = inspect.getsource(Agent.system_one)
check_true("default/predict is system_one", Agent.predict is Agent.system_one)
check_true("default/system_one has no shortlist", "shortlist" not in _predict_src and "embed_fn" not in _predict_src)
_forward_src = inspect.getsource(DecisionModel.forward)
check_true("default/forward has no shortlist", "shortlist" not in _forward_src and "embed_fn" not in _forward_src)


# ---------------------------------------------------------------- deterministic top-k
embed = _embed_for("pay me", OPTION_TEXTS)
labels = shortlist_choice("pay me", CRITERIA, embed, k=2)
check("topk/k=2 keeps the cosine tie in input order", labels, ["alpha", "delta"])
check("topk/one embed call", len(embed.calls), 1)
check("topk/query is the first text", embed.calls[0][0], "pay me")
check(
    "topk/option texts match render_options",
    embed.calls[0][1:],
    render_options({"t": "choice", "ins": "", "crit": CRITERIA}),
)
check("topk/k=1 is the earliest max", shortlist_choice("pay me", CRITERIA, embed, k=1), ["alpha"])
check(
    "topk/k=3 appends the next cosine",
    shortlist_choice("pay me", CRITERIA, embed, k=3),
    ["alpha", "delta", "gamma"],
)

# all-zero query: every cosine is 0, so the earliest labels win
zero_q = TableEmbed({
    "pay me": [0.0, 0.0],
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "gamma: mid": [0.6, 0.8],
    "delta: same": [3.0, 4.0],
})
check(
    "topk/zero query keeps original order",
    shortlist_choice("pay me", CRITERIA, zero_q, k=2),
    ["alpha", "beta"],
)

# non-finite option vector is treated as 0 and loses to a real match
nan_embed = TableEmbed({
    "pay me": [1.0, 0.0],
    "alpha": [float("nan"), float("nan")],
    "beta": [1.0, 0.0],
})
check(
    "topk/nan vector sorts behind a finite match",
    shortlist_choice("pay me", {"alpha": None, "beta": None}, nan_embed, k=1),
    ["beta"],
)


# ---------------------------------------------------------------- list criteria and instructions
list_embed = TableEmbed({
    "Classify\npay me": [0.0, 1.0],
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "gamma": [0.0, 0.2],
})
check(
    "list/instructions change the query and the winner",
    shortlist_choice("pay me", ["alpha", "beta", "gamma"], list_embed, k=2, instructions="Classify"),
    ["beta", "gamma"],
)
check("list/query text includes instructions", list_embed.calls[0][0], "Classify\npay me")
check("list/option texts are the labels", list_embed.calls[0][1:], ["alpha", "beta", "gamma"])

state = {"text": "hi"}
dict_embed = TableEmbed({
    'Classify\n{"text": "hi"}': [1.0, 0.0],
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
})
check(
    "query/dict state is serialized",
    shortlist_choice(state, ["alpha", "beta"], dict_embed, k=1, instructions="Classify"),
    ["alpha"],
)
check("query/json matches serialize_state", dict_embed.calls[0][0], 'Classify\n{"text": "hi"}')

# 0 and False are real criterion values, so they are part of the embedded text
rich = {"zero": 0, "no": False, "bare": None, "named": {"desc": "payments"}}
rich_rendered = render_options({"t": "choice", "ins": "", "crit": rich})
rich_embed = TableEmbed({"pay me": [1.0, 0.0], **{text: [1.0, 0.0] for text in rich_rendered}})
shortlist_choice("pay me", rich, rich_embed, k=1)
check("render/0 and False stay in the option text", rich_embed.calls[0][1:], rich_rendered)


# ---------------------------------------------------------------- k >= n pass-through
boom = BoomEmbed()
check("pass/k == n returns every label in order", shortlist_choice("pay me", CRITERIA, boom, k=4), list(CRITERIA))
check("pass/k > n returns every label in order", shortlist_choice("pay me", CRITERIA, boom, k=20), list(CRITERIA))


# ---------------------------------------------------------------- return_scores
score_embed = _embed_for("pay me", OPTION_TEXTS)
pair = shortlist_choice("pay me", CRITERIA, score_embed, k=2, return_scores=True)
check("scores/labels match the bare call", pair[0], ["alpha", "delta"])
check("scores/cosines are in rank order", pair[1], [1.0, 1.0])
trio = shortlist_choice("pay me", CRITERIA, _embed_for("pay me", OPTION_TEXTS), k=3, return_scores=True)
check("scores/third label is gamma", trio[0], ["alpha", "delta", "gamma"])
check("scores/third cosine rounds to gamma", [round(s, 9) for s in trio[1]], [1.0, 1.0, 0.6])
bare = shortlist_choice("pay me", CRITERIA, _embed_for("pay me", OPTION_TEXTS), k=2)
check("scores/default still returns bare labels", bare, ["alpha", "delta"])
pass_pair = shortlist_choice("pay me", CRITERIA, BoomEmbed(), k=20, return_scores=True)
check("scores/passthrough keeps every label", pass_pair[0], list(CRITERIA))
check("scores/passthrough reports no scores", pass_pair[1], None)
meta_agent = Recorder()
meta_out = predict_shortlist(
    meta_agent, "pay me", {"intent": {"type": "choice", "criteria": CRITERIA}},
    _embed_for("pay me", OPTION_TEXTS), k=2,
)
check("scores/agree with predict_shortlist metadata",
      pair[1], meta_out["shortlist"]["intent"]["scores"])


# ---------------------------------------------------------------- mock predict sees only k criteria
sentinel = {"desc": "payments"}
full = {"billing": sentinel, "tech": "bugs", "sales": None, "other": "misc"}
full_vectors = {
    "Which desk?\nI was charged twice": [1.0, 0.0],
    'billing: {"desc": "payments"}': [0.0, 1.0],
    "tech: bugs": [1.0, 0.0],
    "sales": [0.2, 0.2],
    "other: misc": [0.0, 1.0],
}
# cosine vs [1, 0]: tech=1, sales=0.707, billing=0, other=0. k=2 -> tech, sales.
agent = Recorder()
score_q = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high", "now"]}
noul_q = {"type": "noul", "instructions": "Is a refund requested?"}
questions = {
    "intent": {
        "type": "choice",
        "instructions": "Which desk?",
        "criteria": full,
    },
    "urgency": score_q,
    "refund": noul_q,
    "note": "leave me alone",
}
state = "I was charged twice"
result = predict_shortlist(agent, state, questions, TableEmbed(full_vectors), k=2, model="english")

check("predict/called once", len(agent.calls), 1)
check("predict/system_one not used when predict exists", agent.system_one_calls, 0)
got_state, got_questions, got_kwargs = agent.calls[0]
check_true("predict/state is the same object", got_state is state)
check("predict/kwargs forwarded", got_kwargs, {"model": "english"})
check("predict/choice criteria are only the top 2", list(got_questions["intent"]["criteria"]), ["tech", "sales"])
check_true(
    "predict/kept description is the original object",
    got_questions["intent"]["criteria"]["tech"] == "bugs",
)
check_true("predict/score question is the same object", got_questions["urgency"] is score_q)
check_true("predict/noul question is the same object", got_questions["refund"] is noul_q)
check_true("predict/non-dict question is the same object", got_questions["note"] is questions["note"])
check("caller/criteria unchanged", questions["intent"]["criteria"], full)
check_true("caller/criteria object unchanged", questions["intent"]["criteria"] is full)
check("caller/sentinel intact", full["billing"], sentinel)

internal = Agent._to_internal(got_questions["intent"])
check("predict/internal choice keys are the shortlist", list(internal["crit"]), ["tech", "sales"])
check("result/choice is the mock's first shortlisted label", result["answers"]["intent"]["choice"], "tech")
check("result/shortlist labels", result["shortlist"]["intent"]["labels"], ["tech", "sales"])
check_true(
    "result/scores descend",
    result["shortlist"]["intent"]["scores"][0] > result["shortlist"]["intent"]["scores"][1] > 0,
)
check("result/shortlist k and n", (result["shortlist"]["intent"]["k"], result["shortlist"]["intent"]["n"]), (2, 4))
check("result/not a passthrough", result["shortlist"]["intent"]["passthrough"], False)
check_true("result/non-choice questions are absent from shortlist meta", "urgency" not in result["shortlist"])
# The predict return is copied before shortlist is attached.
held = {}


class Holding:
    def predict(self, state, questions, **kwargs):
        held["questions"] = questions
        held["result"] = {"model": "fake", "answers": {}}
        return held["result"]


out = predict_shortlist(Holding(), "pay me", {"intent": {"type": "choice", "criteria": CRITERIA}}, BoomEmbed(), k=4)
check_true("result/shortlist key is on the copy", "shortlist" in out and "shortlist" not in held["result"])


# ---------------------------------------------------------------- k >= n pass-through reaches predict unchanged
passthrough_agent = Recorder()
original_q = {"type": "choice", "instructions": "Which desk?", "criteria": full}
out = predict_shortlist(
    passthrough_agent,
    "I was charged twice",
    {"intent": original_q},
    BoomEmbed(),
    k=4,
)
check_true("pass/predict received the original question", passthrough_agent.calls[0][1]["intent"] is original_q)
check("pass/metadata labels are the full set", out["shortlist"]["intent"]["labels"], list(full))
check("pass/scores omitted", out["shortlist"]["intent"]["scores"], None)
check("pass/flag", out["shortlist"]["intent"]["passthrough"], True)
check("pass/k > n flag", predict_shortlist(
    Recorder(), "x", {"intent": original_q}, BoomEmbed(), k=99
)["shortlist"]["intent"]["passthrough"], True)

# list criteria stay a list, in rank order, and Agent._to_internal accepts them
list_agent = Recorder()
list_q_embed = TableEmbed({
    "Which?\nhello": [1.0, 0.0],
    "alpha": [0.0, 1.0],
    "beta": [1.0, 0.0],
    "gamma": [0.0, 0.0],
})
list_questions = {"intent": {"type": "choice", "instructions": "Which?", "criteria": ["alpha", "beta", "gamma"]}}
predict_shortlist(list_agent, "hello", list_questions, list_q_embed, k=2)
received = list_agent.calls[0][1]["intent"]["criteria"]
check("list/predict receives a list of k labels", received, ["beta", "alpha"])
check("list/caller criteria unchanged", list_questions["intent"]["criteria"], ["alpha", "beta", "gamma"])
check(
    "list/internal keys follow the shortlist",
    list(Agent._to_internal(list_agent.calls[0][1]["intent"])["crit"]),
    ["beta", "alpha"],
)


# ---------------------------------------------------------------- system_one-only agent
class SystemOneOnly:
    def __init__(self):
        self.questions = None

    def system_one(self, state, questions):
        self.questions = questions
        return {"answers": {"intent": {"choice": "beta"}}}


only = SystemOneOnly()
out = predict_shortlist(only, "hello", list_questions, list_q_embed, k=1)
check("system_one/used when predict is absent", out["answers"]["intent"]["choice"], "beta")
check("system_one/criteria length is k", len(only.questions["intent"]["criteria"]), 1)


# ---------------------------------------------------------------- errors
check_raises("err/k=0", lambda: shortlist_choice("pay me", CRITERIA, embed, k=0))
check_raises("err/k negative", lambda: shortlist_choice("pay me", CRITERIA, embed, k=-3))
check_raises("err/k bool", lambda: shortlist_choice("pay me", CRITERIA, embed, k=True))
check_raises("err/k float", lambda: shortlist_choice("pay me", CRITERIA, embed, k=1.5))
check_raises("err/k str", lambda: shortlist_choice("pay me", CRITERIA, embed, k="2"))
check_raises("err/empty dict", lambda: shortlist_choice("pay me", {}, embed, k=1))
check_raises("err/empty list", lambda: shortlist_choice("pay me", [], embed, k=1))
check_raises("err/criteria tuple", lambda: shortlist_choice("pay me", ("alpha", "beta"), embed, k=1), TypeError)
check_raises("err/duplicate label", lambda: shortlist_choice("pay me", ["alpha", "alpha"], embed, k=1))
check_raises(
    "err/missing criteria",
    lambda: predict_shortlist(Recorder(), "pay me", {"intent": {"type": "choice"}}, embed, k=1),
)
check_raises("err/questions not a dict", lambda: predict_shortlist(Recorder(), "pay me", [], embed, k=1), TypeError)


def _bad_shape(texts):
    return np.zeros((1, 4))


bad_agent = Recorder()
check_raises(
    "err/bad embed shape",
    lambda: predict_shortlist(bad_agent, "pay me", {"intent": original_q}, _bad_shape, k=2),
)
check("err/bad shape does not call predict", len(bad_agent.calls), 0)


def _torch_rows(texts):
    import torch
    rows = []
    for i, _text in enumerate(texts):
        rows.append([1.0, 0.0] if i == 0 or i == 1 else [0.0, 1.0])
    return torch.tensor(rows)


check(
    "embed/torch tensor return is accepted",
    shortlist_choice("pay me", {"alpha": None, "beta": None}, _torch_rows, k=1),
    ["alpha"],
)


# ---------------------------------------------------------------- encoder mean-pool, no Hub
import torch  # noqa: E402


class _Out:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": 2})()
        self.forwards = 0

    def forward(self, input_ids, attention_mask):
        self.forwards += 1
        # Channel 0 is the token id; channel 1 is 1. Padding must drop out of the mean.
        ids = input_ids.float()
        hidden = torch.stack([ids, torch.ones_like(ids)], dim=-1)
        return _Out(hidden)


class TinyTok:
    def __init__(self):
        self.kwargs = None
        self.batches = []

    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        self.kwargs = {"padding": padding, "truncation": truncation, "max_length": max_length}
        self.batches.append(list(texts))
        rows = [[(ord(ch) % 5) + 1 for ch in text][:max_length] or [1] for text in texts]
        width = max(len(row) for row in rows)
        input_ids, mask = [], []
        for row in rows:
            pad = width - len(row)
            input_ids.append(row + [0] * pad)
            mask.append([1] * len(row) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class TinyAgent:
    def __init__(self):
        self.tok = TinyTok()
        self.encoder = TinyEncoder()
        self.model = type("M", (), {"encoder": self.encoder})()
        self.device = torch.device("cpu")
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("decision forward must not run during embedding")


tiny = TinyAgent()
fn = embed_fn_from_agent(tiny, max_length=32, batch_size=2)
first = fn(["ab", "a"])
second = fn(["ab", "a"])
check("encoder/shape", first.shape, (2, 2))
check_true("encoder/deterministic", np.array_equal(first, second))
check("encoder/one batch for both strings", tiny.tok.batches, [["ab", "a"], ["ab", "a"]])
check("encoder/max_length forwarded", tiny.tok.kwargs["max_length"], 32)
check("encoder/decision forward not used", tiny.forward_calls, 0)

# "a" is one real token (id = ord('a') % 5 + 1) plus padding. Masked mean keeps that id.
token_id = (ord("a") % 5) + 1
check("encoder/padding excluded from mean", first[1].tolist(), [float(token_id), 1.0])
# "ab" has two real tokens and no reliance on padding once both are kept.
ids = [(ord(ch) % 5) + 1 for ch in "ab"]
check("encoder/mean of real tokens", first[0].tolist(), [sum(ids) / 2.0, 1.0])

empty = fn([])
check("encoder/empty input shape", empty.shape, (0, 2))
check_raises("encoder/bad max_length", lambda: embed_fn_from_agent(tiny, max_length=0))
check_raises("encoder/bad batch_size", lambda: embed_fn_from_agent(tiny, batch_size=True))


# The two attributes the docstring names -- `agent.tok`, `agent.model.encoder` -- were read with no
# check, so an object that is not a loaded checkpoint answered with the interpreter's own
# AttributeError instead of naming what the helper needs. `_call_predict` refuses the same mistake
# for the agent *it* uses, which is the sibling this now matches.
class NoAttrs:
    pass


class TokOnly:
    def __init__(self):
        self.tok = TinyTok()


class ModelWithoutEncoder:
    def __init__(self):
        self.tok = TinyTok()
        self.model = type("M", (), {})()


for _bad_agent in (NoAttrs(), TokOnly(), ModelWithoutEncoder()):
    check_raises("encoder/agent/%s refused" % type(_bad_agent).__name__,
                 lambda _a=_bad_agent: embed_fn_from_agent(_a), TypeError)

try:
    embed_fn_from_agent(NoAttrs())
    FAIL.append("encoder/agent refusal message: no TypeError")
except TypeError as _exc:
    check_true("encoder/agent refusal names both attributes",
               str(_exc) == "agent must provide tok and model.encoder", repr(str(_exc)))
except AttributeError as _exc:
    FAIL.append("encoder/agent refusal message: leaked AttributeError (%s)" % _exc)


# ---------------------------------------------------------------- device changes, simulated entirely on CPU
class DeviceRecordingTensor:
    def __init__(self, tensor, requests):
        self.tensor = tensor
        self.requests = requests

    def to(self, device):
        # Record placement requests, but never allocate a CUDA tensor in these tests.
        self.requests.append(str(device))
        return self.tensor


class DeviceRecordingTok(TinyTok):
    def __init__(self):
        super().__init__()
        self.requests = []

    def __call__(self, *args, **kwargs):
        encoded = super().__call__(*args, **kwargs)
        return {key: DeviceRecordingTensor(value, self.requests) for key, value in encoded.items()}


moving = TinyAgent()
moving.tok = DeviceRecordingTok()
moving.device = torch.device("cuda")
moving_fn = embed_fn_from_agent(moving, batch_size=2)
# A callback can outlive Agent.system_one's GPU-to-CPU fallback, even before its
# first use. Later invocations must also follow the current device, not hardcode CPU.
for call, target in enumerate(("cpu", "cuda:1", "cpu")):
    moving.device = torch.device(target)
    moving.encoder.train(call == 1)
    moving.tok.requests.clear()
    pooled = moving_fn(["ab", "a", "abc"])
    check("device/call %d places both tensors in every batch" % call, moving.tok.requests, [target] * 4)
    check("device/call %d keeps batch boundaries" % call, moving.tok.batches[-2:], [["ab", "a"], ["abc"]])
    check("device/call %d keeps masked pooling" % call, pooled.tolist(), [[3.5, 1.0], [3.0, 1.0], [4.0, 1.0]])
    check("device/call %d keeps output dtype" % call, pooled.dtype, np.dtype("float32"))
    check("device/call %d preserves encoder mode" % call, moving.encoder.training, call == 1)

before_empty = (len(moving.tok.batches), moving.encoder.forwards, len(moving.tok.requests))
check("device/empty input shape", moving_fn([]).shape, (0, 2))
check("device/empty input skips tokenization, forward and placement",
      (len(moving.tok.batches), moving.encoder.forwards, len(moving.tok.requests)), before_empty)


# ---------------------------------------------------------------- end-to-end helper: shortlist then the real question normalizer only
# Confirms a reduced choice still has one criterion per kept label, which is what
# system_one would score. No weights, no forward.
pipe_agent = Recorder()
pipe_q = {
    "intent": {"type": "choice", "instructions": "Which desk?", "criteria": full},
    "urgency": score_q,
}
piped = predict_shortlist(pipe_agent, "I was charged twice", pipe_q, TableEmbed(full_vectors), k=2)
scored = Agent._to_internal(pipe_agent.calls[0][1]["intent"])
check("pipe/marker count equals k", len(scored["crit"]), 2)
check("pipe/answer choice is inside the shortlist", piped["answers"]["intent"]["choice"] in piped["shortlist"]["intent"]["labels"], True)


# ---------------------------------------------------------------- cached_embed_fn
CACHE_VEC = {"pay me": [1.0, 0.0], "refund please": [0.0, 1.0]}
CACHE_VEC.update(OPTION_TEXTS)
CACHE_TEXTS = ["pay me", "alpha", "beta", "gamma: mid", "delta: same"]

check_raises("cache/embed_fn must be callable", lambda: cached_embed_fn(None), TypeError)
check_raises("cache/maxsize bool rejected", lambda: cached_embed_fn(TableEmbed({}), True))
check_raises("cache/maxsize zero rejected", lambda: cached_embed_fn(TableEmbed({}), 0))
check_raises("cache/maxsize negative rejected", lambda: cached_embed_fn(TableEmbed({}), -3))
check_raises("cache/maxsize must be an int", lambda: cached_embed_fn(TableEmbed({}), "8"))

cache_table = TableEmbed(dict(CACHE_VEC))
wrapped = cached_embed_fn(cache_table)
first = wrapped(CACHE_TEXTS)
check("cache/cold call embeds every text once", cache_table.calls, [CACHE_TEXTS])
check("cache/cold output shape", first.shape, (5, 2))
check("cache/cold output dtype is float32", first.dtype == np.float32, True)
check_true("cache/cold values match unwrapped", np.allclose(first, [CACHE_VEC[t] for t in CACHE_TEXTS]))
second = wrapped(CACHE_TEXTS)
check("cache/warm call makes no embed call", len(cache_table.calls), 1)
check_true("cache/warm returns identical values", np.array_equal(first, second))
third = wrapped(["refund please", "alpha", "beta"])
check("cache/repeat embeds only the new text", cache_table.calls[-1], ["refund please"])
check("cache/underlying calls total", len(cache_table.calls), 2)
check_true(
    "cache/partial rows keep request order",
    np.allclose(third, [CACHE_VEC["refund please"], CACHE_VEC["alpha"], CACHE_VEC["beta"]]),
)

dup_table = TableEmbed(dict(CACHE_VEC))
dup_cached = cached_embed_fn(dup_table)
dup = dup_cached(["alpha", "beta", "alpha"])
check("cache/duplicate text embedded once per call", dup_table.calls, [["alpha", "beta"]])
check("cache/duplicate output keeps request length", len(dup), 3)
check_true("cache/duplicate rows repeat the vector", np.array_equal(dup[0], dup[2]))

lru_table = TableEmbed(dict(CACHE_VEC))
lru_cached = cached_embed_fn(lru_table, maxsize=2)
lru_cached(["alpha", "beta"])           # cache: alpha, beta
lru_cached(["alpha"])                   # hit; alpha now newest, beta is LRU
check("cache/lru touch needs no embed", len(lru_table.calls), 1)
lru_cached(["gamma: mid"])              # inserts gamma, evicts beta
check("cache/lru insert embeds the new text", lru_table.calls[-1], ["gamma: mid"])
lru_cached(["alpha", "gamma: mid"])     # both hits
check("cache/lru survivors are both cached", len(lru_table.calls), 2)
lru_cached(["beta"])                    # beta was evicted
check("cache/lru evicted entry is re-embedded", lru_table.calls[-1], ["beta"])
check("cache/lru size stays at the bound", lru_cached.cache_info()["size"], 2)
lru_cached(["gamma: mid", "beta"])      # both hits
check("cache/lru most recent pair survives", len(lru_table.calls), 3)
lru_cached(["alpha"])                   # alpha was the oldest of the three
check("cache/lru oldest of three was evicted", lru_table.calls[-1], ["alpha"])

info_table = TableEmbed(dict(CACHE_VEC))
info_cached = cached_embed_fn(info_table, maxsize=8)
info_cached(["alpha", "beta"])          # 2 misses
info_cached(["alpha", "gamma: mid"])    # 1 hit + 1 miss
info = info_cached.cache_info()
check("cache/info size", info["size"], 3)
check("cache/info maxsize", info["maxsize"], 8)
check("cache/info hits", info["hits"], 1)
check("cache/info misses", info["misses"], 3)
info_cached.cache_clear()
check("cache/clear resets info", info_cached.cache_info(), {"size": 0, "maxsize": 8, "hits": 0, "misses": 0})
info_cached(["alpha"])
check("cache/clear forces re-embed", info_table.calls[-1], ["alpha"])


class FlakyEmbed:
    """Fails on the first call, succeeds after."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == 1:
            raise RuntimeError("boom")
        return [[1.0, 0.0] for _text in texts]


flaky = FlakyEmbed()
flaky_cached = cached_embed_fn(flaky)
check_raises("cache/embed error propagates", lambda: flaky_cached(["alpha"]), RuntimeError)
check("cache/failed call caches nothing", flaky_cached.cache_info()["size"], 0)
retry = flaky_cached(["alpha"])
check("cache/retry calls embed again", len(flaky.calls), 2)
check_true(
    "cache/retried row is cached",
    flaky_cached.cache_info()["size"] == 1 and np.allclose(retry[0], [1.0, 0.0]),
)


class BadShapeEmbed:
    def __call__(self, texts):
        return [[1.0, 0.0]]  # one row no matter how many texts arrive


bad_cached = cached_embed_fn(BadShapeEmbed())
check_raises("cache/bad shape raises", lambda: bad_cached(["alpha", "beta"]))
check("cache/bad shape caches nothing", bad_cached.cache_info()["size"], 0)


class NonFiniteEmbed:
    """`alpha` embeds NaN on the first call and the query vector on every later one.

    A transient NaN is the case the "bad shape" pair above does not cover: the row is
    shape-valid, so it is not refused -- `nan_to_num` rewrites it to zeros on the way in,
    exactly as `_embeddings` does for an unwrapped embedder. The difference is that the
    wrapper then *stores* the repaired row, so the rewrite outlives the call that caused
    it, while the unwrapped path re-embeds and recovers on its own.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        first = len(self.calls) == 1
        return [[float("nan"), float("nan")] if (first and text == "alpha")
                else ([1.0, 0.0] if text in ("pay me", "alpha") else [0.8, 0.2])
                for text in texts]


nf_embed = NonFiniteEmbed()
nf_cached = cached_embed_fn(nf_embed)
# `cache/nan row stored cleaned` and `cache/cleaned row served from cache`, further down
# this file, pin the rewrite and that the cache serves it. What they do not pin is that the
# second call's *rows* are the repaired ones, which is where the wrapper and the unwrapped
# embedder visibly disagree: a transient NaN costs that option its real embedding forever.
nf_cached(["alpha"])                       # embeds NaN, stores the zeroed row
check_true("cache/sticky row stays the repaired one",
           np.array_equal(nf_cached(["alpha"])[0], [0.0, 0.0]) and len(nf_embed.calls) == 1)

# The consequence a caller actually sees: shortlist membership, not just the numbers.
NF_CRITERIA = {"alpha": None, "beta": None}
raw_embed = NonFiniteEmbed()
raw = (shortlist_choice("pay me", NF_CRITERIA, raw_embed, k=1),
       shortlist_choice("pay me", NF_CRITERIA, raw_embed, k=1))
wrapped_embed = NonFiniteEmbed()
wrapped_fn = cached_embed_fn(wrapped_embed)
sticky = (shortlist_choice("pay me", NF_CRITERIA, wrapped_fn, k=1),
          shortlist_choice("pay me", NF_CRITERIA, wrapped_fn, k=1))
check("cache/unwrapped second call recovers the finite row", raw, (["beta"], ["alpha"]))
check("cache/wrapped second call keeps the zeroed row", sticky, (["beta"], ["beta"]))
wrapped_fn.cache_clear()
check("cache/clearing it restores the unwrapped ranking",
      shortlist_choice("pay me", NF_CRITERIA, wrapped_fn, k=1), ["alpha"])


class FakeTensor:
    """Mimics the detach().float().cpu().numpy() chain of a torch tensor."""

    def __init__(self, rows):
        self._rows = rows

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self._rows, dtype=np.float64)


class TensorEmbed:
    def __call__(self, texts):
        return FakeTensor([[1.0, 0.0] for _text in texts])


tensor_cached = cached_embed_fn(TensorEmbed())
tensor_out = tensor_cached(["alpha"])
check("cache/torch-style return accepted", tensor_out.shape, (1, 2))
check("cache/torch-style rows stored float32", tensor_out.dtype == np.float32, True)

none_table = TableEmbed({"": [1.0, 0.0]})
none_cached = cached_embed_fn(none_table)
none_cached([None])
check("cache/None text normalizes to empty string", none_table.calls, [[""]])
none_cached([""])
check("cache/empty string hits the same entry", len(none_table.calls), 1)
check("cache/empty input embeds nothing", none_cached([]).shape, (0, 0))

nan_table = TableEmbed({"alpha": [float("nan"), 0.0], "beta": [0.0, 1.0]})
nan_cached = cached_embed_fn(nan_table)
nan_out = nan_cached(["alpha", "beta"])
check_true("cache/nan row stored cleaned", bool(np.isfinite(nan_out).all()))
nan_cached(["alpha", "beta"])
check("cache/cleaned row served from cache", len(nan_table.calls), 1)

e2e_table = TableEmbed(dict(CACHE_VEC))
e2e_cached = cached_embed_fn(e2e_table)
run1 = shortlist_choice("pay me", CRITERIA, e2e_cached, k=2)
run2 = shortlist_choice("refund please", CRITERIA, e2e_cached, k=2)
check("cache/e2e first call embeds query and options", e2e_table.calls[0], CACHE_TEXTS)
check("cache/e2e repeat embeds only the new query", e2e_table.calls[1], ["refund please"])
check("cache/e2e two calls total", len(e2e_table.calls), 2)
check("cache/e2e labels match the uncached path", run1, ["alpha", "delta"])
check("cache/e2e second query labels", run2, ["beta", "gamma"])

pipe_agent2 = Recorder()
pipe_vectors2 = dict(CACHE_VEC)
pipe_vectors2["category\npay me"] = [1.0, 0.0]
pipe_vectors2["category\nrefund please"] = [0.0, 1.0]
pipe_table2 = TableEmbed(pipe_vectors2)
pipe_cached2 = cached_embed_fn(pipe_table2)
pipe_q2 = {"intent": {"type": "choice", "instructions": "category", "criteria": CRITERIA}}
predict_shortlist(pipe_agent2, "pay me", pipe_q2, pipe_cached2, k=2)
pipe_b = predict_shortlist(pipe_agent2, "refund please", pipe_q2, pipe_cached2, k=2)
check("cache/predict repeat embeds query only", pipe_table2.calls[1], ["category\nrefund please"])
check("cache/predict keeps shortlist metadata", pipe_b["shortlist"]["intent"]["labels"], ["beta", "gamma"])
check("cache/predict answer comes from the shortlist", pipe_b["answers"]["intent"]["choice"], "beta")

mt_table = TableEmbed(dict(CACHE_VEC))
mt_cached = cached_embed_fn(mt_table)


def _mt_work(i):
    texts = ["alpha", "beta"] if i % 2 else ["beta", "gamma: mid"]
    return texts, mt_cached(texts)


with ThreadPoolExecutor(max_workers=4) as pool:
    mt_results = list(pool.map(_mt_work, range(16)))
check_true(
    "cache/concurrent calls return correct rows",
    all(np.allclose(rows, [CACHE_VEC[t] for t in texts]) for texts, rows in mt_results),
)
check_true("cache/concurrent size stays bounded", mt_cached.cache_info()["size"] <= 4096)
check_true(
    "cache/concurrent counters consistent",
    mt_cached.cache_info()["hits"] + mt_cached.cache_info()["misses"] == 32,
)

# --------------------------------------------------------------- cache dimensionality
# A row stored under one dimensionality cannot stack against a row of another: the
# docstring asks the caller to clear the cache when the model changes, and the cache
# refuses the call instead of returning a matrix that silently mixes both.


class SwapEmbed:
    """One callable whose width changes between calls, like a swapped model."""

    def __init__(self):
        self.dim = 2
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[1.0] * self.dim for _text in texts]


swap_fn = SwapEmbed()
swap_cached = cached_embed_fn(swap_fn)
swap_cached(["alpha"])
check("cache/dim first width cached", swap_cached.cache_info()["size"], 1)
swap_fn.dim = 3
try:
    swap_cached(["beta"])
    check_true("cache/dim drift refused", False, "no error raised")
except ValueError as exc:
    check_true("cache/dim drift refused", "cache_clear()" in str(exc), str(exc))
check("cache/dim drift caches nothing new", swap_cached.cache_info()["size"], 1)
swap_cached.cache_clear()
swap_cached(["beta"])
check("cache/dim clear then re-embed works", swap_cached.cache_info()["size"], 1)


# ------------------------------------------------------- the dimensionality guard is a read
# That guard reads `rows_by_text`, so it needs the lock every other read holds. Measured on the
# unlocked version: 3 callers plus a `cache_clear()` churner over 6,000 calls (12,000 texts) let
# 42, 46, 55, 65 and 67 exceptions escape on five runs in a row, each one either a RuntimeError
# ("OrderedDict mutated during iteration") or a StopIteration out of `next(iter(...))`, raised at
# the guard line and handed to the caller instead of a matrix. The switch interval is compressed
# because at CPython's default 5 ms the same workload escaped 0 twice over -- the window between
# the truthiness test and the iterator is two bytecodes wide, and a thread only lands in it when
# the interpreter chooses to switch there. The locked build escapes 0 at 1e-6.
def _race_parse(text):
    tid, n = text.split("-")
    return float(int(tid[1:]) * 100000 + int(n))


def _race_embed(texts):
    # Every text embeds to a row of its own, so a call that gets back another text's row is
    # caught below and not just a call that raises.
    return np.array([[_race_parse(t)] * 8 for t in texts], dtype=np.float32)


def _race_texts(tid, n):
    # Two texts per call, because a one-text call cannot tell a row bound to the wrong key from
    # a correct one: there is nothing to compare the position against.
    return ["t%d-%d" % (tid, n), "t%d-%d" % (tid, n + 5000)]


race_errors = []
race_bad_rows = []
race_stop = threading.Event()
race_fn = cached_embed_fn(_race_embed, maxsize=4)


def _race_caller(tid):
    for n in range(2000):
        texts = _race_texts(tid, n)
        want = [_race_parse(t) for t in texts]
        try:
            rows = race_fn(texts)
            got = [float(row[0]) for row in rows]
            if rows.shape != (2, 8) or got != want:
                race_bad_rows.append((tuple(rows.shape), got, want))
        except BaseException as exc:
            race_errors.append(type(exc).__name__)


def _race_churner():
    while not race_stop.is_set():
        race_fn.cache_clear()


_switch_interval = sys.getswitchinterval()
sys.setswitchinterval(1e-6)
try:
    _churner = threading.Thread(target=_race_churner)
    _churner.start()
    _callers = [threading.Thread(target=_race_caller, args=(t,)) for t in range(3)]
    for _caller in _callers:
        _caller.start()
    for _caller in _callers:
        _caller.join()
    race_stop.set()
    _churner.join()
finally:
    sys.setswitchinterval(_switch_interval)

check_true("cache/concurrent dimensionality guard does not iterate a mutating cache",
           not race_errors,
           "%d escaped: %s" % (len(race_errors), sorted(set(race_errors))))
# The arm above is the one the unlocked build fails; it crashed calls rather than mis-serving
# them, so this arm measured 0 there too. This one is what catches a wrong row: serving every
# position from `found[keys[0]]` instead of `found[key]` trips it on all 6,000 calls.
check_true("cache/concurrent guarded calls return the row for their own text",
           not race_bad_rows, "%d calls returned %s" % (len(race_bad_rows), race_bad_rows[:3]))


# --------------------------------------------------------------- tournament
class Ranker:
    """Stand-in for Agent that always answers the label it ranks highest, in any group."""

    def __init__(self, order):
        self.rank = {label: i for i, label in enumerate(order)}
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((questions, kwargs))
        return {"model": "fake", "answers": {
            qid: {"type": "choice", "choice": min(qdef["criteria"], key=self.rank.get)}
            for qid, qdef in questions.items() if qdef["type"] == "choice"}}


check_true("export/predict_tournament", laya.predict_tournament is predict_tournament)
check_true("all/predict_tournament", "predict_tournament" in laya.__all__)

# A ranker that is consistent across groups must see its best label win from any position.
for n, calls in ((77, 2), (257, 3)):
    labels = ["l%03d" % i for i in range(n)]
    for best in (labels[0], labels[n // 2], labels[-1]):
        ranker = Ranker([best] + [label for label in labels if label != best])
        out = predict_tournament(ranker, "s", {"q": {"type": "choice", "instructions": "pick",
                                                     "criteria": list(labels)}})
        check("tournament/%d labels: %s wins" % (n, best), out["answers"]["q"]["choice"], best)
        check("tournament/%d labels: %s calls" % (n, best), len(ranker.calls), calls)
    if n == 77:
        check("tournament/77 labels in balanced groups",
              [len(qdef["criteria"]) for qdef in ranker.calls[0][0].values()], [15, 15, 16, 15, 16])

# Only the choice over the group size plays rounds; the final call carries the whole request.
labels = ["l%02d" % i for i in range(20)]
big = {"type": "choice", "instructions": "pick", "criteria": {label: "about " + label for label in labels}}
small = {"type": "choice", "instructions": "pick", "criteria": ["x", "y"]}
level = {"type": "score", "instructions": "rate", "criteria": ["low", "high"]}
questions = {"big": big, "small": small, "level": level}
before = repr(questions)
ranker = Ranker(labels[::-1] + ["y", "x"])
out = predict_tournament(ranker, "s", questions, group_size=8, model="english")
check("tournament/caller questions unchanged", repr(questions), before)
check("tournament/two calls", len(ranker.calls), 2)
check("tournament/kwargs reach every call", [kwargs for _q, kwargs in ranker.calls], [{"model": "english"}] * 2)
check("tournament/round holds only the big choice's groups",
      [q["criteria"] for q in ranker.calls[0][0].values()],
      [{label: "about " + label for label in part} for part in (labels[:6], labels[6:13], labels[13:])])
final = ranker.calls[1][0]
check("tournament/final keeps every question id", sorted(final), ["big", "level", "small"])
check("tournament/final cuts the big choice to its finalists",
      final["big"], dict(big, criteria={label: "about " + label for label in ("l05", "l12", "l19")}))
check_true("tournament/final passes the rest unchanged", final["small"] is small and final["level"] is level)
check("tournament/answers come from the final call", out["answers"]["big"]["choice"], "l19")
check("tournament/meta", out["tournament"], {
    "big": {"labels": ["l05", "l12", "l19"], "n": 20, "rounds": 1},
    "small": {"labels": ["x", "y"], "n": 2, "rounds": 0}})

ranker = Ranker(["y", "x"])
predict_tournament(ranker, "s", {"small": small, "level": level})
check("tournament/nothing to narrow: one call with the request itself",
      ranker.calls, [({"small": small, "level": level}, {})])

for bad in (1, 0, -3, True, 2.5, "16"):
    check_raises("tournament/group_size %r refused" % (bad,),
                 lambda bad=bad: predict_tournament(Ranker([]), "s", {}, group_size=bad))
check_raises("tournament/questions must be a dict", lambda: predict_tournament(Ranker([]), "s", []), TypeError)
check_raises("tournament/choice without criteria",
             lambda: predict_tournament(Ranker([]), "s", {"q": {"type": "choice", "instructions": "pick"}}))


# --------------------------------------------- every call's reply is held to the same contract
# `predict_shortlist` and the tournament's final call both refused a non-dict reply by naming that
# contract; the tournament's round call indexed the reply instead, so the same agent fault raised
# Python's indexing error from inside the module. These checks hold the shape guards that close
# that gap, and pin the bare `KeyError`s the narrowing deliberately leaves where they are.
class Reply:
    """Answers every call with ``shape``, so only the reply's shape varies between these stubs.

    A callable ``shape`` is handed the questions, which is how a stub replies to the caller's own
    request and to the invented group questions differently.
    """

    def __init__(self, shape):
        self.shape = shape
        self.seen = []

    def predict(self, state, questions, **kwargs):
        self.seen.append(sorted(questions))
        return self.shape(questions) if callable(self.shape) else self.shape


def _message(fn):
    try:
        fn()
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    return "no exception"


def _outcome(fn):
    """What a call returns, or the exception it raised.

    A guard that fires where it should not has to fail the check that owns that call site, not
    abort the suite before the checks below it run.
    """
    try:
        return ("returned", fn())
    except Exception as exc:
        return ("raised", "%s: %s" % (type(exc).__name__, exc))


# 20 labels at group_size 4 is two rounds -- five groups, then two -- and then the final call over
# the finalists. `_round` hands one reply shape to every call of such a tournament, as a thunk.
_TOUR_Q = {"q": {"type": "choice", "criteria": ["l%02d" % i for i in range(20)]}}
_DICT_MSG = "TypeError: predict/system_one must return a dict, got list"


def _round(shape):
    return lambda: predict_tournament(Reply(shape), "s", _TOUR_Q, group_size=4)


def _group_answers(questions):
    return {"model": "fake", "answers": {qid: {"choice": next(iter(qdef["criteria"]))}
                                         for qid, qdef in questions.items()}}


check("tournament/round reply must be a dict, said by name", _message(_round(["not", "a", "dict"])),
      _DICT_MSG)
check("shortlist/reply must be a dict, said by name", _message(
    lambda: predict_shortlist(Reply(["not", "a", "dict"]), "s", _TOUR_Q,
                              embed_fn=lambda texts: [[float(i)] for i in range(len(texts))], k=5)),
    _DICT_MSG)
check("tournament/round reply without an answers mapping is named too",
      _message(_round({"model": "fake"})),
      "TypeError: predict/system_one must return an answers dict, got NoneType")
# What is deliberately not changed: an answer this module indexes is still looked up directly, so a
# round that skips a group raises the bare `KeyError` for that group's number, and an answer with no
# choice raises `KeyError: 'choice'`. `laya-java/fixtures/tournament.json` records both messages as
# the contract the Java and TypeScript ports are tested against, so naming them here would move
# three ports and their committed fixtures rather than one module.
check("tournament/a round that leaves groups out keeps the bare KeyError",
      _message(_round(lambda q: {"model": "fake", "answers": {"0": {"choice": "l00"}}})),
      "KeyError: '1'")
check("tournament/an answer with no choice keeps the bare KeyError",
      _message(_round(lambda q: {"model": "fake",
                                 "answers": {qid: {"score": 1.0} for qid in q}})),
      "KeyError: 'choice'")
# Each call site is guarded on its own: an agent can answer the caller's request and hand back a
# list for the invented groups, or the other way round. Either way the round has already run, so a
# guard only on the last call would not catch the first of these.
check("tournament/a bad round reply is refused before the final call",
      _message(_round(lambda q: ["not", "a", "dict"] if "q" not in q else _group_answers(q))),
      _DICT_MSG)
_bad_final = Reply(lambda q: ["not", "a", "dict"] if "q" in q else _group_answers(q))
check("tournament/a bad final reply is refused after the rounds ran",
      (_message(lambda: predict_tournament(_bad_final, "s", _TOUR_Q, group_size=4)), _bad_final.seen),
      (_DICT_MSG, [["0", "1", "2", "3", "4"], ["0", "1"], ["q"]]))
# And a reply that satisfies the contract reaches the same result it did before: the checks are on
# the shape this module indexes, not on an answer's contents, which the final dict passes through.
_clean = Reply(_group_answers)
check("tournament/a well-shaped reply still plays its rounds",
      _outcome(lambda: (predict_tournament(_clean, "s", _TOUR_Q, group_size=4)["tournament"]["q"]["rounds"],
                       _clean.seen)),
      ("returned", (2, [["0", "1", "2", "3", "4"], ["0", "1"], ["q"]])))
check("tournament/an answer-less final reply is still returned as given",
      _outcome(lambda: predict_tournament(Reply({"model": "fake"}), "s",
                                          {"q": {"type": "choice", "criteria": ["a", "b"]}})),
      ("returned", {"model": "fake",
                    "tournament": {"q": {"labels": ["a", "b"], "n": 2, "rounds": 0}}}))
# The contract belongs in the docstring, because a caller picking an agent for this function reads
# it there -- and it has to say the rounds are covered, since that is the call site that used to be
# unchecked.
_doc = " ".join(predict_tournament.__doc__.split())
_contract = (_doc.split("Every call's reply")[1].split("Extra keyword")[0]
             if "Every call's reply" in _doc else "")
check_true("tournament/docstring states the contract for every call",
           _contract.startswith(" is checked against the same shape, the elimination rounds included")
           and "answers" in _contract,
           "the docstring says: " + (_contract or "(no such paragraph)")[:160])


# ------------------------------------------------- narrowing drops a stale option_order
# Nandha on #958: narrowing `criteria` while forwarding a full-length `option_order`
# "is a core contract question"; his preferred remedy drops the order with a warning,
# because remapping it onto the kept labels changes the request silently.


def _run_catching_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _check_validates(name, qid, qdef):
    try:
        Agent._check_question(qid, qdef)
        check_true(name, True)
    except ValueError as e:
        check_true(name, False, str(e))


# --- shortlist: a narrowed choice loses its order, once, and says so
order_q = {
    "type": "choice",
    "instructions": "Which desk?",
    "criteria": dict(full),
    "option_order": [3, 2, 1, 0],
}
order_agent = Recorder()
out_order, order_warns = _run_catching_warnings(
    lambda: predict_shortlist(order_agent, state, {"intent": order_q}, TableEmbed(full_vectors), k=2)
)
narrowed = order_agent.calls[0][1]["intent"]
check_true("order/narrowed question carries no option_order", "option_order" not in narrowed)
check("order/narrowed criteria stay the top 2", list(narrowed["criteria"]), ["tech", "sales"])
check("order/exactly one RuntimeWarning", len(order_warns), 1)
check_true(
    "order/warning names the question and the key",
    bool(order_warns) and "intent" in str(order_warns[0].message) and "option_order" in str(order_warns[0].message),
)
_check_validates("order/narrowed question passes Agent._check_question", "intent", narrowed)
plain_agent = Recorder()
out_plain = predict_shortlist(
    plain_agent, state,
    {"intent": {k: v for k, v in order_q.items() if k != "option_order"}},
    TableEmbed(full_vectors), k=2,
)
check("order/result equals the call without an order", out_order, out_plain)
check_true("order/caller question keeps its option_order", "option_order" in order_q)

# --- shortlist: passthrough and non-choice questions are untouched
passthrough_order = {"type": "choice", "instructions": "pick", "criteria": ["x", "y"], "option_order": [1, 0]}
pt_agent = Recorder()
_out, pt_warns = _run_catching_warnings(
    lambda: predict_shortlist(pt_agent, "s", {"pick": passthrough_order}, TableEmbed({}), k=2)
)
check_true("order/passthrough keeps its order", pt_agent.calls[0][1]["pick"] is passthrough_order)
check("order/passthrough emits no warning", len(pt_warns), 0)
_check_validates("order/passthrough order still validates", "pick", passthrough_order)
score_order = dict(score_q, option_order=[3, 2, 1, 0])
nc_agent = Recorder()
_out, nc_warns = _run_catching_warnings(
    lambda: predict_shortlist(nc_agent, "s", {"urgency": score_order}, TableEmbed({}), k=2)
)
check_true("order/non-choice keeps its order", nc_agent.calls[0][1]["urgency"] is score_order)
check("order/non-choice emits no warning", len(nc_warns), 0)

# --- shortlist with list criteria behaves the same way
list_order = {"type": "choice", "instructions": "pick", "criteria": ["a", "b", "c", "d"],
              "option_order": [3, 2, 1, 0]}
list_agent = Recorder()
_out, list_warns = _run_catching_warnings(
    lambda: predict_shortlist(
        list_agent, "a", {"pick": list_order},
        TableEmbed({"pick\na": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.1, 0.9], "d": [0.0, 1.0]}),
        k=2,
    )
)
list_narrowed = list_agent.calls[0][1]["pick"]
check_true("order/list-criteria narrowed carries no option_order", "option_order" not in list_narrowed)
check("order/list-criteria one warning", len(list_warns), 1)
_check_validates("order/list-criteria narrowed validates", "pick", list_narrowed)

# --- tournament: groups and finalists lose the order, once per question
t_labels = ["l%02d" % i for i in range(8)]
t_big = {"type": "choice", "instructions": "pick",
         "criteria": {label: "about " + label for label in t_labels},
         "option_order": [7, 6, 5, 4, 3, 2, 1, 0]}
t_small = {"type": "choice", "instructions": "pick", "criteria": ["x", "y"], "option_order": [1, 0]}
t_ranker = Ranker(t_labels[::-1] + ["y", "x"])
out_t, t_warns = _run_catching_warnings(
    lambda: predict_tournament(t_ranker, "s", {"big": t_big, "small": t_small}, group_size=4)
)
_bad = []
for _call_questions, _kwargs in t_ranker.calls:
    for _qid, _qdef in _call_questions.items():
        if _qdef.get("type") != "choice":
            continue
        if "option_order" in _qdef and _qid == "big":
            _bad.append("%s kept its order" % _qid)
        try:
            Agent._check_question(_qid, _qdef)
        except ValueError as e:
            _bad.append("%s: %s" % (_qid, e))
check_true("order/tournament questions all validate without the order", not _bad, "; ".join(_bad))
check("order/tournament warns once for the narrowed choice", len(t_warns), 1)
check_true("order/tournament warning names the question", bool(t_warns) and "big" in str(t_warns[0].message))
final_t = t_ranker.calls[-1][0]
check_true("order/tournament passthrough keeps its order", final_t["small"] is t_small)
check_true(
    "order/tournament caller keeps its order",
    t_big["option_order"] == [7, 6, 5, 4, 3, 2, 1, 0],
)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
