"""Offline tests for the embedding shortlist. No checkpoint and no Hub download.

A fake ``embed_fn`` supplies vectors. ``predict`` / ``system_one`` are mocks, so the
decision model is never constructed.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import laya  # noqa: E402
from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel, render_options  # noqa: E402
from laya.shortlist import (  # noqa: E402
    embed_fn_from_agent,
    predict_shortlist,
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
for _name in ("shortlist_choice", "predict_shortlist", "embed_fn_from_agent"):
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


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
