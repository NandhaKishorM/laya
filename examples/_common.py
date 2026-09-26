"""Shared helpers for the Laya examples.

Each example is a standalone script:

    .venv/bin/python examples/01_hello_world_decision.py

Run it from anywhere -- paths resolve from this file, and Python puts the examples/ directory
on `sys.path` automatically, so `from _common import ...` just works.

Checkpoints come from `models/` when `setup_laya.sh` has downloaded them, and from the Hub bundle
(`convaiinnovations/laya`, with a subfolder per checkpoint) otherwise, so the examples run after a
plain `pip install laya`. Loading one takes a few seconds -- plus the first download when there is
no local copy -- and the first MPS call pays Metal kernel compilation, so anything that reports
latency warms up first.

This module also imports `laya` for you -- after setting `USE_TF=0`, because a stray TensorFlow
install can deadlock model loading. Examples therefore do `from _common import laya, ...` rather
than importing it themselves first, which would defeat the guard.
"""
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")          # torch-only: a stray TF install can deadlock loading
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # repository root; <root>/laya/email.py must not shadow stdlib email

import laya  # noqa: E402
from laya.router import Router  # noqa: E402

BUNDLE_REPO = "convaiinnovations/laya"
_SUBFOLDER = {
    "english": None,                 # the bundle's root holds the English checkpoint
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

LOCAL_MODELS = {
    "english": os.path.join(ROOT, "models", "laya"),
    "multilingual": os.path.join(ROOT, "models", "laya-multilingual"),
    "typed-decisions": os.path.join(ROOT, "models", "laya-typed-decisions"),
}


def has_local(name="english"):
    """True when setup_laya.sh has already downloaded this checkpoint into `models/`."""
    return os.path.exists(os.path.join(LOCAL_MODELS[name], "model.safetensors"))


def checkpoint(name="english"):
    """A Router model spec: the local directory when there is one, else the Hub repo + subfolder."""
    if has_local(name):
        return LOCAL_MODELS[name]
    return (BUNDLE_REPO, _SUBFOLDER[name])


# What `Router(models=...)` takes: absolute local paths in a checkout that ran setup_laya.sh,
# Hub specs otherwise.
MODELS = {name: checkpoint(name) for name in LOCAL_MODELS}

# ---------------------------------------------------------------- sample inputs
STATE_EN = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan.",
}
STATE_HI = {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}
STATE_DE = {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung."}

# the quickstart question set: every primitive, one forward pass
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}


def banner(number, title, blurb=""):
    """Print the header every example starts with."""
    line = "=" * 74
    print("\n%s\n  Example %s -- %s\n%s" % (line, number, title, line))
    if blurb:
        for para in blurb.strip().split("\n"):
            print("  " + para.strip())
        print()


def load(name="english", device=None):
    """Load one checkpoint: from `models/` when it is there, else from the Hub.

    `device=None` lets Laya pick: CUDA, then MPS, then CPU.
    """
    if has_local(name):
        return laya.load(LOCAL_MODELS[name], device=device)
    return laya.load(BUNDLE_REPO, subfolder=_SUBFOLDER[name], device=device)


def router(device=None, **kwargs):
    """A Router over the local checkpoints when they are there, else over the Hub bundle."""
    return Router(models=MODELS, device=device, **kwargs)


def timed(fn, repeat=3):
    """(result, median_ms) for a call, after one untimed warm-up."""
    fn()
    samples = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return result, statistics.median(samples)


def top(probabilities, n=3):
    """The n highest-probability labels, as a printable string."""
    items = sorted(probabilities.items(), key=lambda kv: -kv[1])[:n]
    return "  ".join("%s=%.3f" % (k, v) for k, v in items)


def describe(answers, indent="   "):
    """One readable line per question, whatever primitive it used."""
    for qid, a in answers.items():
        if a["type"] == "choice":
            best = max(a["probabilities"].items(), key=lambda kv: kv[1])
            detail = "%s (p=%.3f, conf=%.3f)" % (best[0], best[1], a["confidence"])
            if len(a["probabilities"]) > 2:
                detail += "   next: %s" % top({k: v for k, v in a["probabilities"].items()
                                               if k != best[0]}, 2)
        elif a["type"] == "score":
            levels = len(a["probabilities"]) - 1
            detail = "%.2f / %d (conf=%.3f)" % (a["score"], levels, a["confidence"])
        else:
            detail = "%.3f (%s)  conf=%.3f" % (a["noul"], "true" if a["noul"] > 0.5 else "false",
                                               a["confidence"])
        print("%s%-18s %-7s %s" % (indent, qid, a["type"], detail))


def heading(text):
    print("\n   -- %s --" % text)


def device_line(agent):
    print("   device: %s   (torch %s)" % (agent.device, __import__("torch").__version__))
