"""Numerics check: is the forward pass on this machine trustworthy?

For every checkpoint:
  1. The RoPE base transformers 4.x actually uses == the one the checkpoint was trained with
     (transformers 5 records it as `rope_parameters`).
  2. Run-to-run determinism (same model, same input, twice) -- the float32 noise floor.
  3. SDPA (what Laya asks transformers for) vs eager (reference math), on the user-visible
     answers: choice labels, probabilities, confidence, noul, score.

Run:  python verify/numerics_check.py
"""
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # repository root; <root>/laya/email.py must not shadow stdlib email

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

import laya  # noqa: E402
from laya.common import _apply_rope_config  # noqa: E402

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}
STATES = [{"body": "I was charged twice for invoice 4411, please refund it today."},
          {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung."},
          {"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}]


def flatten(node, prefix=""):
    """Every float/number in an answers payload, keyed by path."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flatten(v, "%s.%s" % (prefix, k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(flatten(v, "%s[%d]" % (prefix, i)))
    elif isinstance(node, bool):
        out[prefix] = float(node)
    elif isinstance(node, (int, float)):
        out[prefix] = float(node)
    return out


def run(agent):
    return [flatten(agent.predict(state, QUESTIONS)["answers"]) for state in STATES]


def compare(label, a, b):
    keys = sorted(set(a) & set(b))
    worst = max((abs(a[k] - b[k]), k) for k in keys)
    return worst


def main():
    print("torch %s" % torch.__version__)
    failures = []
    for name, rel in [("english", "models/laya"), ("multilingual", "models/laya-multilingual"),
                      ("typed-decisions", "models/laya-typed-decisions")]:
        path = os.path.join(ROOT, rel)
        raw = json.load(open(os.path.join(path, "encoder", "config.json")))
        ecfg = AutoConfig.from_pretrained(os.path.join(path, "encoder"))
        _apply_rope_config(ecfg)
        want = raw.get("rope_parameters") or {}
        rope_ok = (float(want["full_attention"]["rope_theta"]) == ecfg.global_rope_theta and
                   float(want["sliding_attention"]["rope_theta"]) == ecfg.local_rope_theta)
        print("\n   %-16s rope full=%-8s sliding=%-8s (trained %s / %s)  %s"
              % (name, ecfg.global_rope_theta, ecfg.local_rope_theta,
                 want["full_attention"]["rope_theta"], want["sliding_attention"]["rope_theta"],
                 "OK" if rope_ok else "MISMATCH"))
        if not rope_ok:
            failures.append("%s rope theta" % name)

        agent = laya.load(path, device="cpu")
        first = run(agent)
        second = run(agent)                      # noise floor: identical input, same weights
        floor = max(compare("", a, b)[0] for a, b in zip(first, second))

        impl = agent.model.encoder.config._attn_implementation
        agent.model.encoder.config._attn_implementation = "eager"
        eager = run(agent)
        agent.model.encoder.config._attn_implementation = impl
        del agent

        backend = max(compare("", a, b)[0] for a, b in zip(first, eager))
        _, worst_key = max(compare("", a, b) for a, b in zip(first, eager))
        ok = backend <= max(1e-3, floor * 10)
        print("   %-16s float32 noise floor %.2e | sdpa-vs-eager %.2e (worst: %s)  %s"
              % (name, floor, backend, worst_key.strip("."), "OK" if ok else "DIFFERS"))
        if not ok:
            failures.append("%s sdpa/eager" % name)

    print("\n%s" % ("FAILURES: %s" % failures if failures else "all numerical checks passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
