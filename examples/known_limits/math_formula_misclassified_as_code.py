"""Known limit: math formula mis-scored as coding by the English checkpoint (#622).

The English checkpoint (laya / ModernBERT) conflates formula-like syntax -- superscripts,
f(x) notation, Greek letters -- with code, so a calculus prompt such as
"Compute the derivative of f(x) = 3x² + 2x" can score ~0.88 on a user-defined `is_coding`
noul question even though no code is involved.  laya-multilingual does not share this bias
and scores the same prompt near 0.01.

The fix: use `router_questions()` instead of a homegrown `is_coding` noul.  Its `domain`
question names `code` and `math_or_logic` as separate criteria, so the English checkpoint
has a math bucket to route into and the false positive disappears.
"""
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# examples/ is the parent directory; its _common.py holds the shared checkpoint-loading helpers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import banner, heading, load, laya  # noqa: E402

banner("known_limits/math_formula", "Math formula mis-scored as code (#622)", """
    The English checkpoint may score calculus prompts as coding-related (~0.88) because
    ModernBERT conflates formula syntax with code syntax.  laya-multilingual is not
    affected and scores the same prompt near 0.01.

    The fix: replace a homegrown `is_coding` noul with `router_questions()`.  Its `domain`
    choice names `code` and `math_or_logic` as distinct criteria, so the model has a math
    bucket to route calculus into.
    """)

CALCULUS = "Compute the derivative of f(x) = 3x² + 2x"
CODE = "Refactor this Python function to use list comprehensions instead of for-loops"

agent = load("english")

heading("false positive: homegrown is_coding noul")

# A noul with no named math option leaves the model with no math bucket.
# Formula syntax looks like code to ModernBERT, so it routes calculus here.
IS_CODING = {
    "is_coding": {
        "type": "noul",
        "instructions": "Is `request` a software engineering or programming task?",
    },
}

results = {}
for label, text in (("calculus", CALCULUS), ("coding", CODE)):
    a = agent.predict({"request": text}, IS_CODING)["answers"]["is_coding"]
    results[label] = a
    flag = "  <-- FALSE POSITIVE" if label == "calculus" and a["noul"] > 0.5 else ""
    print("   %-10s is_coding=%.4f  confidence=%.4f%s" % (label, a["noul"], a["confidence"], flag))

print()
print("   The calculus prompt scores %.4f -- above the 0.5 threshold -- because the model has"
      % results["calculus"]["noul"])
print("   no math option to route into.  Formula syntax leaks into the coding score.")

heading("workaround: named domain labels via router_questions()")

# router_questions().domain names both `code` and `math_or_logic` as explicit criteria.
# With a math bucket available, the English checkpoint routes calculus correctly.
domain_question = {"domain": laya.router_questions()["domain"]}

for label, text in (("calculus", CALCULUS), ("coding", CODE)):
    a = agent.predict({"request": text}, domain_question)["answers"]["domain"]
    best = max(a["probabilities"].items(), key=lambda kv: kv[1])
    print("   %-10s domain=%-16s p=%.4f  confidence=%.4f" % (label, best[0], best[1], a["confidence"]))

print()
print("   Once `math_or_logic` is a named option, the English checkpoint routes calculus")
print("   there instead of into `code`.  Replace a homegrown `is_coding` noul with")
print("   `router_questions()`, or write a `choice` question with both `code` and `math`")
print("   as explicit options.")
