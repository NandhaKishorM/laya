"""`confidence` carries two definitions, and only one of them is the calibrated quantity.

    choice / score  ->  confidence_from_probs(p, k) == 1 - H(p) / log(k)
    noul            ->  max(p_true, 1 - p_true)     == max(p)

They are not on the same scale. The same two-option distribution comes back as 0.90 from a
`noul` and 0.53 from an equivalent two-option `choice`, and every shipped preset mixes the two
types in one call -- `moderation_questions()` is four `noul` and one `score`. The README's
"Automated Confidence Gating" section gates all of them on a single threshold.

Both benchmark harnesses in research/scripts take `conf = max(probs)` before calling
`ece_score`, so every ECE figure in the README describes max(p) and not normalized entropy.
`answer_confidence` reports that quantity on every question type, additively: `confidence` is
untouched, so nothing a caller gates on today moves.

No weights are loaded: the confidence helpers are pure.
"""
import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import answer_confidence, confidence_from_probs, ece_score  # noqa: E402

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


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --------------------------------------------------------------- answer_confidence is max(p)
for probs in ([0.5, 0.5], [0.1, 0.9], [0.7, 0.2, 0.1], [0.25] * 4, [1.0, 0.0, 0.0]):
    p = np.array(probs)
    check_true("answer/max of %s" % (probs,), close(answer_confidence(p, len(probs)), max(probs)))

check("answer/only the first k entries count", answer_confidence(np.array([0.4, 0.6, 0.99]), 2), 0.6)
check("answer/k=1 is certain", answer_confidence(np.array([1.0]), 1), 1.0)
check("answer/k=0 does not crash", answer_confidence(np.array([]), 0), 1.0)

# --------------------------------------------------------------- it matches noul's definition
# `noul` reports max(p_true, 1 - p_true), which over two options is exactly max(p). So for a
# noul answer the new field equals the existing one, and the two agree by construction.
for p_true in (0.0, 0.05, 0.3, 0.5, 0.62, 0.9, 1.0):
    p = np.array([1.0 - p_true, p_true])
    check_true("noul/answer_confidence equals the shipped noul confidence at p=%.2f" % p_true,
               close(answer_confidence(p, 2), max(p_true, 1.0 - p_true)))

# --------------------------------------------------------------- the two scales really differ
# Same distribution, two question types. 0.85 is the threshold the README's gating section uses.
disagree = []
for p_true in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
    p = np.array([1.0 - p_true, p_true])
    if (answer_confidence(p, 2) >= 0.85) != (confidence_from_probs(p, 2) >= 0.85):
        disagree.append(p_true)
check_true("scales/the two disagree across the documented threshold", disagree == [0.85, 0.90, 0.95],
           "disagreed at %s" % (disagree,))
check_true("scales/entropy sits far below max(p) on the same distribution",
           confidence_from_probs(np.array([0.1, 0.9]), 2) < 0.55 < answer_confidence(np.array([0.1, 0.9]), 2))
# max(p) over two options has a floor of 0.5; entropy reads 0.0 for the same coin flip.
check("scales/a coin flip is 0.5 on max(p)", answer_confidence(np.array([0.5, 0.5]), 2), 0.5)
check("scales/and 0.0 on entropy", confidence_from_probs(np.array([0.5, 0.5]), 2), 0.0)

# --------------------------------------------------------------- only one of them is calibrated
# Perfectly calibrated predictions: an answer reported at top probability c is right exactly c
# of the time. ECE on max(p) must be near zero. ECE on normalized entropy must not be, which is
# why it cannot be compared against a probability threshold.
rng = np.random.default_rng(0)
tops, ents, correct = [], [], []
for c in np.linspace(0.30, 0.99, 24):
    rest = (1.0 - c) / 2.0
    p = np.array([c, rest, rest])
    for _ in range(400):
        tops.append(answer_confidence(p, 3))
        ents.append(confidence_from_probs(p, 3))
        correct.append(1.0 if rng.random() < c else 0.0)

tops, ents, correct = np.array(tops), np.array(ents), np.array(correct)
ece_top, ece_ent = ece_score(tops, correct), ece_score(ents, correct)
check_true("calibration/max(p) is calibrated on calibrated data (ECE %.4f)" % ece_top,
           ece_top < 0.03, "ECE %.4f" % ece_top)
check_true("calibration/entropy is not (ECE %.4f)" % ece_ent, ece_ent > 0.20, "ECE %.4f" % ece_ent)
check_true("calibration/entropy is worse by a wide margin", ece_ent > 5 * ece_top,
           "top %.4f vs entropy %.4f" % (ece_top, ece_ent))

# --------------------------------------------------------------- the entropy helper is untouched
for probs, k in (([0.1, 0.9], 2), ([0.25] * 4, 4), ([0.7, 0.2, 0.1], 3)):
    p = np.array(probs)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    check_true("entropy/formula for k=%d" % k,
               close(confidence_from_probs(p, k), float(np.clip(1 - ent / math.log(k), 0.0, 1.0))))

# --------------------------------------------------------------- min_confidence validation & abstention (#361)
from laya.confidence import check_min_confidence, flag_low_confidence  # noqa: E402

# check_min_confidence validates [0.0, 1.0] and rejects non-floats and bools
for valid in (0.0, 0.5, 1.0, 0, 1, 0.85):
    check("min_confidence/valid %.2f" % valid, check_min_confidence(valid), float(valid))

for invalid in (True, False, -0.01, 1.01, -1.0, 2.0, float("nan"), float("inf"), float("-inf"), "0.5", None, [0.5]):
    try:
        check_min_confidence(invalid)
        FAIL.append("min_confidence/should reject %r" % (invalid,))
    except ValueError:
        PASS.append("min_confidence/rejected %r" % (invalid,))

# flag_low_confidence marks answers where answer_confidence < threshold
sample_res = [{
    "answers": {
        "q_high": {"type": "choice", "choice": "a", "answer_confidence": 0.92, "confidence": 0.8},
        "q_low": {"type": "score", "score": 1, "answer_confidence": 0.45, "confidence": 0.4},
        "q_fallback": {"type": "choice", "choice": "b", "confidence": 0.3},
        "q_exact": {"type": "choice", "choice": "c", "answer_confidence": 0.70},
    }
}]

# min_confidence=0.0 is a no-op (default behavior untouched)
flag_low_confidence(sample_res, 0.0)
check_true("flag/0.0 is no-op", not any("low_confidence" in a for a in sample_res[0]["answers"].values()))

# min_confidence=0.70: q_low (0.45) and q_fallback (0.3) flagged; q_high (0.92) and q_exact (0.70) NOT flagged
flag_low_confidence(sample_res, 0.70)
ans = sample_res[0]["answers"]
check_true("flag/q_low flagged", ans["q_low"].get("low_confidence") is True)
check_true("flag/q_fallback flagged", ans["q_fallback"].get("low_confidence") is True)
check_true("flag/q_high unflagged", "low_confidence" not in ans["q_high"])
check_true("flag/q_exact unflagged", "low_confidence" not in ans["q_exact"])
check("flag/answer_confidence intact", ans["q_low"]["answer_confidence"], 0.45)
check("flag/raw choice intact", ans["q_high"]["choice"], "a")

# --------------------------------------------------------------- the gate's state is reported
# `low_confidence` is written only when the gate fires, so its absence cannot tell a caller
# "a gate ran and this answer cleared it" from "no gate ran at all". An operator with 10,000
# logged decisions cannot compute an abstention rate, cannot tell whether a run was gated, and
# cannot re-split a batch that used different thresholds per request class -- the threshold is
# consumed and dropped. `docs/staged-adoption.md` asks a shadow record to carry "errors and any
# fallback or review decision"; there was no field to put that in.
#
# The contract: where a gate ran, its state is an explicit value, so it is never inferred from a
# missing field. Where no gate was configured, nothing is written at all -- an ungated call returns
# the payload it always returned, rather than growing a field every existing caller would then have
# to read. The two cases are told apart by the presence of `abstention`, which is why the vocabulary
# has no "not configured" member to read out of the field.
from laya.confidence import (  # noqa: E402
    GATE_ABSTAINED,
    GATE_PASSED,
    GATE_STATES,
    GATE_UNEVALUATED,
    apply_confidence_gate,
)

check("gate/the vocabulary is closed",
      list(GATE_STATES), ["passed", "abstained", "unevaluated"])

fresh = lambda: [{  # noqa: E731
    "answers": {
        "q_high": {"type": "choice", "choice": "a", "answer_confidence": 0.92},
        "q_low": {"type": "choice", "choice": "b", "answer_confidence": 0.45},
    }
}]

# No gate: the payload is untouched. Not a sentinel, not a flag -- byte-for-byte what the caller
# got before this helper existed, so no existing caller has to learn a field to keep working.
off = fresh()
before = copy.deepcopy(off)
apply_confidence_gate(off, None)
check("gate/an ungated call writes nothing at all", off, before)
for qid, a in off[0]["answers"].items():
    check_true("gate/ungated %s reports no state" % qid, "abstention" not in a)
    check_true("gate/ungated %s carries no threshold" % qid, "abstention_threshold" not in a)
    check_true("gate/ungated %s is not a flag" % qid, "low_confidence" not in a)

# A gate that ran: the answer that cleared it says `passed`, which is the state the boolean
# could not express. This is the whole point -- before, this answer was byte-identical to the
# ungated one above.
on = fresh()
apply_confidence_gate(on, 0.80)
check("gate/cleared answer says passed", on[0]["answers"]["q_high"]["abstention"], GATE_PASSED)
check("gate/cleared answer is still unflagged", "low_confidence" in on[0]["answers"]["q_high"], False)
check("gate/cleared answer echoes the threshold", on[0]["answers"]["q_high"]["abstention_threshold"], 0.80)
check("gate/below-threshold answer says abstained", on[0]["answers"]["q_low"]["abstention"], GATE_ABSTAINED)
check("gate/below-threshold answer is flagged", on[0]["answers"]["q_low"]["low_confidence"], True)
check("gate/below-threshold echoes the threshold", on[0]["answers"]["q_low"]["abstention_threshold"], 0.80)
check_true("gate/the raw answer is untouched",
           on[0]["answers"]["q_high"]["choice"] == "a" and on[0]["answers"]["q_low"]["choice"] == "b")

# The two states a caller used to be unable to tell apart are now told apart by a value, and the
# distinction survives a round trip through JSON, which is how a shadow record reaches a log.
import json as _json  # noqa: E402

round_tripped = _json.loads(_json.dumps(on[0]["answers"]))
check("gate/survives a JSON round trip", round_tripped["q_high"]["abstention"], GATE_PASSED)

# One rule, one implementation: the flag stays `flag_low_confidence`'s, not a second copy of it.
# `apply_confidence_gate(results, 0.0)` is a gate that ran and nothing can fail, which is why
# `flag_low_confidence`'s 0.0 no-op above is still correct and still asserted.
zero = fresh()
apply_confidence_gate(zero, 0.0)
check("gate/0.0 reports passed", zero[0]["answers"]["q_low"]["abstention"], GATE_PASSED)
check_true("gate/0.0 still flags nothing", "low_confidence" not in zero[0]["answers"]["q_low"])

# A non-finite or missing confidence is skipped by `flag_low_confidence`, so it is not an
# abstention. It is also not a pass: the gate ran and could not decide, which is the state Argo
# Rollouts ships as `Inconclusive` and the one collapsing it into success would be the same lie
# `low_confidence`'s absence already tells.
odd = [{"answers": {
    "q_nan": {"type": "choice", "answer_confidence": float("nan")},
    "q_none": {"type": "choice"},
    "q_inf": {"type": "choice", "answer_confidence": float("inf")},
    "q_bool": {"type": "choice", "answer_confidence": True},
    # The fallback path, which is where the number used to leak. With no usable
    # `answer_confidence` the gate reads the entropy `confidence`, and an entropy NaN has to come
    # back as "nothing to gate on" -- not as the NaN itself, which is not None and would
    # therefore read as a pass. `flag_low_confidence` never noticed (NaN < x is False, and
    # None is not None is also False), so nothing else pinned this down.
    "q_nan_entropy": {"type": "choice", "confidence": float("nan")},
    "q_bool_entropy": {"type": "choice", "confidence": True},
    "q_fine": {"type": "choice", "answer_confidence": 0.99},
}}]
apply_confidence_gate(odd, 0.80)
for qid, a in odd[0]["answers"].items():
    if qid == "q_fine":
        check("gate/a usable confidence is evaluated %s" % qid, a["abstention"], GATE_PASSED)
    else:
        check("gate/unusable confidence is unevaluated %s" % qid, a["abstention"], GATE_UNEVALUATED)
    check_true("gate/unusable confidence is not flagged %s" % qid, "low_confidence" not in a)

# `answer_confidence` is the quantity the gate reads, so a `noul`-style entropy `confidence`
# below threshold does not abstain while `answer_confidence` above it does. The state follows
# the same field the flag follows.
pref = [{"answers": {
    "q_a": {"type": "noul", "answer_confidence": 0.91, "confidence": 0.10},
    "q_b": {"type": "choice", "answer_confidence": 0.10, "confidence": 0.99},
}}]
apply_confidence_gate(pref, 0.80)
check("gate/gate reads answer_confidence (a)", pref[0]["answers"]["q_a"]["abstention"], GATE_PASSED)
check("gate/gate reads answer_confidence (b)", pref[0]["answers"]["q_b"]["abstention"], GATE_ABSTAINED)

# The guards the six call sites rely on: a result that is not a dict, answers that are not a
# dict, and answers that are not dicts. Every call site passes a list it has already partly
apply_confidence_gate([None, {"answers": None}, {"answers": [1, 2]}], 0.5)
apply_confidence_gate([], None)
PASS.append("gate/non-dict results are skipped without raising")

# --------------------------------------------------------------- BUG-005 regression: stale low_confidence across reuse (#910)
# `flag_low_confidence` and `apply_confidence_gate` used to only set `low_confidence: True` and
# never clear it. When a result dict was re-evaluated with a more permissive threshold (or 0.0),
# the stale flag remained, causing downstream callers to treat the answer as permanently abstained.

# 1. flag_low_confidence clears stale flag on re-evaluation
reuse_res = [{
    "answers": {
        "intent": {"choice": "refund", "answer_confidence": 0.8},
    }
}]
reuse_ans = reuse_res[0]["answers"]["intent"]
flag_low_confidence(reuse_res, 0.9)
check_true("stale/flag strict threshold sets flag", reuse_ans.get("low_confidence") is True)
flag_low_confidence(reuse_res, 0.5)
check_true("stale/flag permissive threshold clears flag", "low_confidence" not in reuse_ans)

clean_res = [{"answers": {"q": {"answer_confidence": 0.95}}}]
flag_low_confidence(clean_res, 0.5)
check_true("stale/unflagged answer stays clean", "low_confidence" not in clean_res[0]["answers"]["q"])

# 2. apply_confidence_gate clears stale flag and updates abstention state on re-evaluation
gate_reuse = [{
    "answers": {
        "intent": {"choice": "refund", "answer_confidence": 0.8},
    }
}]
g_ans = gate_reuse[0]["answers"]["intent"]
apply_confidence_gate(gate_reuse, 0.9)
check("stale/gate strict threshold abstains", g_ans["abstention"], GATE_ABSTAINED)
check_true("stale/gate strict threshold sets low_conf", g_ans.get("low_confidence") is True)

apply_confidence_gate(gate_reuse, 0.5)
check("stale/gate permissive threshold passes", g_ans["abstention"], GATE_PASSED)
check_true("stale/gate permissive threshold clears low_conf", "low_confidence" not in g_ans)
check("stale/gate echoes updated threshold", g_ans["abstention_threshold"], 0.5)

apply_confidence_gate(gate_reuse, 0.0)
check("stale/gate 0.0 passes", g_ans["abstention"], GATE_PASSED)
check_true("stale/gate 0.0 clears low_conf", "low_confidence" not in g_ans)

# 3. unusable confidence clears stale low_confidence and marks unevaluated
odd_reuse = [{"answers": {"q": {"choice": "x", "answer_confidence": 0.2, "low_confidence": True}}}]
odd_reuse[0]["answers"]["q"]["answer_confidence"] = float("nan")
apply_confidence_gate(odd_reuse, 0.5)
check("stale/unusable conf marks unevaluated", odd_reuse[0]["answers"]["q"]["abstention"], GATE_UNEVALUATED)
check_true("stale/unusable conf clears stale low_conf", "low_confidence" not in odd_reuse[0]["answers"]["q"])

# --------------------------------------------------------------- the operator's three questions
# End to end through a real call site, weight-free. `decide` forwards to any runner with a
# `predict`, so this exercises `laya/structured.py`'s own gate call site rather than the helper
# in isolation -- the six call sites are where the state used to be skipped entirely.
from laya.structured import decide  # noqa: E402

_GATE_Q = {"type": "choice", "instructions": "Pick one.",
           "criteria": {"a": "alpha", "b": "beta"}}
CONFIDENCES = {"high": 0.95, "low": 0.40}
QUESTIONS = {qid: dict(_GATE_Q) for qid in CONFIDENCES}


class _StubRunner:
    def __init__(self, conf):
        self.conf = conf

    def predict(self, state, questions, **kwargs):
        return {"model": "stub", "answers": {
            qid: {"type": "choice", "choice": "a", "answer_confidence": self.conf[qid]}
            for qid in questions}}


stub = _StubRunner(CONFIDENCES)

# 1. Was the gate in effect at all? Before, both answers were byte-identical to the gated pair
#    below, so the answer was "no way to tell". Now the ungated run simply has no `abstention`
#    key, and the gated one does -- which is how a caller tells them apart.
ungated = decide(stub, "state", questions=QUESTIONS)
for qid in CONFIDENCES:
    check_true("workflow/ungated %s reports no state" % qid, "abstention" not in ungated[qid])

gated = decide(stub, "state", questions=QUESTIONS, min_confidence=0.80)
check("workflow/the cleared answer reports passed", gated["high"]["abstention"], GATE_PASSED)
check("workflow/the low answer reports abstained", gated["low"]["abstention"], GATE_ABSTAINED)

# 2. What fraction abstained, over whatever decisions the operator kept? Computable from the
#    artifact, with no re-run and no out-of-band record of which threshold was used.
log = [gated, decide(stub, "other", questions=QUESTIONS, min_confidence=0.80)]
answers = [a for d in log for a in d.values()]
abstained = sum(a["abstention"] == GATE_ABSTAINED for a in answers)
check("workflow/abstention rate over the log", abstained, 2)
check("workflow/…out of", len(answers), 4)
check_true("workflow/…and no answer is left unlabelled", all("abstention" in a for a in answers))

# 3. What threshold produced these? `flag_low_confidence` consumes it and drops it, so before
#    this a batch that gated different request classes differently could not be re-split.
check("workflow/the threshold is recoverable", gated["low"]["abstention_threshold"], 0.80)
check_true("workflow/…and a per-class mix is re-splittable",
           {a["abstention_threshold"] for a in gated.values()} == {0.80})

# The schema projection is unchanged -- an abstained field is still `null` -- so the flat
# `values` mapping still cannot say why. `return_details=True` is where the state is readable,
# and it was unreadable before.
SCHEMA = {"type": "object",
          "properties": {"high": {"type": "string", "enum": ["a", "b"]},
                         "low": {"type": "string", "enum": ["a", "b"]}}}
projected = decide(stub, "state", schema=SCHEMA, min_confidence=0.80)
check("workflow/the schema projection is unchanged", projected["low"], None)
check("workflow/…and still reports the cleared field", projected["high"], "a")
detailed = decide(stub, "state", schema=SCHEMA, min_confidence=0.80, return_details=True)
check("workflow/return_details carries the state",
      detailed.answers["low"]["abstention"], GATE_ABSTAINED)

# --------------------------------------------------------------- exported
import laya  # noqa: E402

check_true("export/answer_confidence is importable from laya", hasattr(laya, "answer_confidence"))
check_true("export/answer_confidence is in __all__", "answer_confidence" in laya.__all__)
check_true("export/answer_confidence is in dir()", "answer_confidence" in dir(laya))
check_true("export/check_min_confidence is importable from laya", hasattr(laya, "check_min_confidence"))
check_true("export/check_min_confidence is in __all__", "check_min_confidence" in laya.__all__)
check_true("export/check_min_confidence is in dir()", "check_min_confidence" in dir(laya))
check_true("export/flag_low_confidence is importable from laya", hasattr(laya, "flag_low_confidence"))
check_true("export/flag_low_confidence is in __all__", "flag_low_confidence" in laya.__all__)
check_true("export/flag_low_confidence is in dir()", "flag_low_confidence" in dir(laya))
check_true("export/apply_confidence_gate is importable from laya", hasattr(laya, "apply_confidence_gate"))
check_true("export/apply_confidence_gate is in __all__", "apply_confidence_gate" in laya.__all__)
check_true("export/apply_confidence_gate is in dir()", "apply_confidence_gate" in dir(laya))
check_true("export/GATE_STATES is importable from laya", hasattr(laya, "GATE_STATES"))
check_true("export/GATE_STATES is in __all__", "GATE_STATES" in laya.__all__)
check_true("export/GATE_STATES is in dir()", "GATE_STATES" in dir(laya))
# The three values are reached through GATE_STATES rather than exported one by one: iterating the
# vocabulary is the whole use, and every `laya.__all__` name has to earn a documented entry
# (`tests/test_packaging.py` enforces that).
import laya.confidence as _confidence  # noqa: E402

check("export/GATE_STATES is the vocabulary a caller iterates", list(GATE_STATES),
      [_confidence.GATE_PASSED, _confidence.GATE_ABSTAINED, _confidence.GATE_UNEVALUATED])
check("export/laya re-exports the same tuple", laya.GATE_STATES, _confidence.GATE_STATES)

# --------------------------------- the two doc pages must attribute confidence per question type
# `docs/structured.md` and `docs/questions-and-answers.md` each opened their confidence section with
# "`confidence` is normalized entropy". Neither agent writes that for every type:
# `Agent._decode_answers` (`laya/agent.py:1367-1404`) and `OnnxAgent._decode_answers`
# (`laya/onnx_agent.py:682-727`) put `confidence_from_probs(p, k)` = `1 - H(p) / log(k)` in the
# `choice` and `score` answers and `max(p_true, 1 - p_true)` in the `noul` one. Each page printed a
# `noul` in the very block its sentence introduces, so the sentence was falsifiable from the page
# alone: the Q&A page's sample answer is `{"type": "noul", "noul": 0.8727, "confidence": 0.8727}`,
# where entropy over two options reads 0.45, and structured.md's `Ticket` carries `needs_human: bool`
# beside the `department` choice. structured.md also quoted `0.71` for the field whose probabilities it
# prints one line below as `{"billing": 0.94, "support": 0.06, "sales": 0.0}` -- 0.79 under the formula
# the page names. So both pages are read as data here: every number they print is recomputed from the
# distribution printed beside it, the option count from the schema's own labels, and each formula has
# to appear in a sentence naming exactly the types the code uses it for. No weights, no pydantic.
import ast as _ast  # noqa: E402  (`ast` itself is imported again further down)
import re as _re  # noqa: E402  (same reason: this section must not rebind `re`)

from laya.structured import questions_from_json_schema  # noqa: E402

DOC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_STRUCTURED = os.path.join(DOC_ROOT, "docs", "structured.md")
DOC_QA = os.path.join(DOC_ROOT, "docs", "questions-and-answers.md")
DOC_AGENT_PY = os.path.join(DOC_ROOT, "laya", "agent.py")
DOC_ONNX_PY = os.path.join(DOC_ROOT, "laya", "onnx_agent.py")

ATTRIBUTED = {"entropy": ["choice", "score"], "maxp": ["noul"]}

# A type has to be named in the same sentence as the formula, so `choice`/`score` cannot inherit the
# `noul` formula and back. Only backticked type tokens count: the pages use "the choice is explicit" as
# English, and a gate that read that as a question type would pass the wrong page.
ENTROPY_MENTION = _re.compile(r"normalized\s+entropy")
MAXP_MENTION = _re.compile(r"max\(\s*p(?:_true)?\s*,\s*1\s*-\s*p(?:_true)?\s*\)")
TYPE_TOKEN = _re.compile(r"`(choice|score|noul)`")

# The blanket claims, and the one number that was simply wrong.
BLANKET_ENTROPY = _re.compile(r"`?confidence`?\s+is\s+normalized\s+entropy", _re.I)
BLANKET_SAMPLE_NOTE = _re.compile(r"normalized\s+entropy,\s+which\s+depends\s+on\s+label\s+count")
MAXP_IN_CODE = _re.compile(r"max\(\s*float\(\s*p\[1\]\s*\)\s*,\s*1\.0\s*-\s*float\(\s*p\[1\]\s*\)\s*\)")

# The pages as they ship on main, sentence for sentence. Every ban below fires on this text and every
# attribution rule below shows this text failing it, so no rule here is a guess about what changed.
OLD_BLANKETS = (
    ("docs/structured.md",
     "`confidence`. `confidence` is normalized entropy, which depends on how many options the "
     "question had: `tests/test_confidence.py` pins that a two-option distribution comes back as "
     "0.90 on a `noul` and 0.53 on an equivalent `choice`, so it does not compare against a "
     "threshold."),
    ("docs/questions-and-answers.md",
     "`confidence` is normalized entropy: high when the distribution is peaked, low when it is "
     "spread out, regardless of whether the top answer is correct."),
)
OLD_SAMPLE_LINE = ('result.confidence["department"]        '
                   '# 0.71  normalized entropy, which depends on label count')


def _read_page(path):
    # newline="" plus an explicit encoding: the pages carry em-dashes, and a CRLF checkout or a
    # non-UTF-8 locale must not change what the rules below see.
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def _flatten(text):
    return " ".join(text.split())


def _fenced_blocks(text):
    """Every fenced block on a page, in order, as (language, body)."""
    blocks, lang, body = [], None, []
    for line in text.split("\n"):
        stripped = line.strip()
        if lang is None and stripped.startswith("```"):
            lang = stripped[3:].strip()
            body = []
        elif lang is not None and stripped.startswith("```"):
            blocks.append((lang, "\n".join(body)))
            lang = None
        elif lang is not None:
            body.append(line)
    assert lang is None, "unbalanced fence in %s" % text[:40]
    return blocks


def _prose(text):
    """The page's prose: fenced blocks dropped, line wraps undone, whitespace collapsed."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return _flatten("\n".join(out))


def _attributions(prose):
    """{formula: sorted question types named in the same sentence as it}."""
    got = {}
    for sentence in _re.split(r"(?<=[.!?])\s+", prose):
        types = set(TYPE_TOKEN.findall(sentence))
        if not types:
            continue
        if ENTROPY_MENTION.search(sentence):
            got.setdefault("entropy", set()).update(types)
        if MAXP_MENTION.search(sentence):
            got.setdefault("maxp", set()).update(types)
    return {key: sorted(value) for key, value in sorted(got.items())}


def _decode_conf_formulas(path):
    """{question type: the formula its `confidence` is built from}, read out of that agent's own
    `_decode_answers` with `ast`: `entropy` where the answer dict calls `confidence_from_probs`,
    `maxp` where it takes `max(float(p[1]), 1.0 - float(p[1]))`. A bare name resolves one hop to its
    assignment in the same function, because the ONNX builder binds `conf_score` once and uses it for
    both of its entropy types."""
    src = _read_page(path)
    fn = next((node for node in _ast.walk(_ast.parse(src))
               if isinstance(node, _ast.FunctionDef) and node.name == "_decode_answers"), None)
    assert fn is not None, "%s no longer defines _decode_answers" % path
    bound = {}
    for node in _ast.walk(fn):
        if (isinstance(node, _ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], _ast.Name)):
            bound[node.targets[0].id] = _ast.get_source_segment(src, node.value) or ""
    got = {}
    for node in _ast.walk(fn):
        if not isinstance(node, _ast.Dict):
            continue
        pairs = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, _ast.Constant) and isinstance(key.value, str):
                pairs[key.value] = value
        if "confidence" not in pairs or "type" not in pairs:
            continue
        qtype = pairs["type"].value
        expr = pairs["confidence"]
        text = bound.get(expr.id, "") if isinstance(expr, _ast.Name) else (
            _ast.get_source_segment(src, expr) or "")
        if "confidence_from_probs" in text:
            formula = "entropy"
        elif MAXP_IN_CODE.search(text):
            formula = "maxp"
        else:
            raise AssertionError("%s builds `%s`'s confidence from %r" % (path, qtype, text))
        assert qtype not in got, "%s decodes `%s` twice" % (path, qtype)
        got[qtype] = formula
    return got


def _structured_sample():
    """The `return_details=True` block as data: the number printed for each field, the probabilities
    printed for the same field, and the option count named beside the confidence number."""
    src = _read_page(DOC_STRUCTURED)

    def printed(key):
        m = _re.search(r'result\.' + key + r'\["department"\]\s*#\s*([0-9.]+)', src)
        assert m, 'the sample block no longer prints result.%s["department"]' % key
        return float(m.group(1))

    m = _re.search(r'result\.probabilities\["department"\]\s*#\s*(\{[^}]*\})', src)
    assert m, "the sample block no longer prints the probabilities the two numbers come from"
    probs = _ast.literal_eval(m.group(1))
    note = _re.search(r'result\.confidence\["department"\]\s*#\s*[0-9.]+\s+([^\n]*)', src)
    assert note, "the confidence line lost its note"
    k = _re.search(r"(\d+)-option", note.group(1))
    assert k, "the confidence line must name the option count its formula depends on: %r" % note.group(1)
    return printed("confidence"), printed("answer_confidence"), probs, int(k.group(1)), note.group(1)


def test_no_page_calls_confidence_one_formula():
    for path in (DOC_STRUCTURED, DOC_QA):
        prose = _prose(_read_page(path))
        assert not BLANKET_ENTROPY.search(prose), "%s still says `confidence` is entropy" % path
    src = _read_page(DOC_STRUCTURED)
    assert not BLANKET_SAMPLE_NOTE.search(src), "the sample note still blames the label count alone"
    for name, old in OLD_BLANKETS:
        assert BLANKET_ENTROPY.search(_flatten(old)), "the ban does not fire on %s's old sentence" % name
    assert BLANKET_SAMPLE_NOTE.search(OLD_SAMPLE_LINE), "the sample-note ban does not fire on main's line"


def test_each_page_attributes_both_formulas_to_named_types():
    for path in (DOC_STRUCTURED, DOC_QA):
        got = _attributions(_prose(_read_page(path)))
        assert got == ATTRIBUTED, "%s attributes %s, want %s" % (path, got, ATTRIBUTED)
    # main's prose satisfies neither half, so the rule could not have passed before the fix.
    assert _attributions(_flatten(OLD_BLANKETS[1][1])) == {}, "main's Q&A sentence passes the rule"
    assert _attributions(_flatten(OLD_BLANKETS[0][1])) == {"entropy": ["choice", "noul"]}, (
        "main's structured.md sentence passes the rule")


def test_the_agents_build_what_the_pages_attribute():
    want = {}
    for path in (DOC_AGENT_PY, DOC_ONNX_PY):
        got = _decode_conf_formulas(path)
        assert sorted(got) == ["choice", "noul", "score"], "%s decodes %s" % (path, sorted(got))
        assert got == {"choice": "entropy", "score": "entropy", "noul": "maxp"}, got
        want = got
    derived = {"entropy": sorted(t for t, f in want.items() if f == "entropy"),
               "maxp": sorted(t for t, f in want.items() if f == "maxp")}
    assert derived == ATTRIBUTED, "the pages are held to a transcription, not to the agents: %s" % derived
    for path in (DOC_STRUCTURED, DOC_QA):
        got = _attributions(_prose(_read_page(path)))
        assert got == derived, "%s attributes %s; the agents build %s" % (path, got, derived)


def test_structured_page_sample_numbers_recompute():
    conf, answer_conf, probs, k, note = _structured_sample()
    p = np.array([probs[label] for label in probs])
    assert k == len(p), "the page says %d-option and prints %d labels" % (k, len(p))
    assert conf == round(confidence_from_probs(p, k), 2), (
        "the page prints confidence %s for %s over %d options; that formula gives %s"
        % (conf, probs, k, round(confidence_from_probs(p, k), 2)))
    assert abs(answer_conf - float(p.max())) < 5e-3, (
        "answer_confidence %s is not max(p)=%s of the probabilities printed beside it"
        % (answer_conf, round(float(p.max()), 4)))
    assert "`choice`" in note, "the confidence line must say which type the field is: %r" % note
    # and main's number was not this formula's either way: the witness stays true while the
    # probabilities on the page stay these.
    assert abs(0.71 - round(confidence_from_probs(p, k), 2)) > 5e-3, (
        "0.71 now recomputes, so this witness has gone stale: %s" % probs)


def test_structured_page_claims_a_schema_that_mixes_the_two_types():
    """The page says its own `Ticket` holds one `choice` field and one `noul` field, which is what
    makes one `confidence` dict carry two scales. The compiler that builds those questions has to
    agree, and the probabilities in the sample have to be that field's labels."""
    src = _read_page(DOC_STRUCTURED)
    ticket = next((body for _lang, body in _fenced_blocks(src) if "class Ticket" in body), None)
    assert ticket is not None, "the page lost the Ticket schema the confidence sample reads"
    fields = dict(_re.findall(r"^\s*(\w+)\s*:\s*([^\n]+?)\s*$", ticket, _re.M))
    assert sorted(fields) == ["department", "needs_human", "urgency"], fields
    labels = [s.strip().strip("'\"")
              for s in _re.search(r"Literal\[(.*?)\]", fields["department"]).group(1).split(",")]
    assert fields["needs_human"] == "bool", fields
    _conf, _ac, probs, _k, _note = _structured_sample()
    assert labels == list(probs), "the sample's probabilities are not the schema's labels"
    compiled = questions_from_json_schema({
        "type": "object",
        "properties": {
            "department": {"enum": labels},
            "needs_human": {"type": "boolean"},
        },
    })
    got = {qid: q["type"] for qid, q in compiled.items()}
    assert got == {"department": "choice", "needs_human": "noul"}, got


def _qa_answers():
    """{question type: the sample answer the page prints for it}. Each block is a dict literal, so the
    numbers are read as data and not matched as prose."""
    out = {}
    for _lang, body in _fenced_blocks(_read_page(DOC_QA)):
        try:
            node = _ast.literal_eval(body.strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(node, dict) and "confidence" in node and "answer_confidence" in node:
            out[node["type"]] = node
    return out


def test_questions_page_sample_answers_recompute():
    answers = _qa_answers()
    assert sorted(answers) == ["choice", "noul", "score"], (
        "the page must show one sample answer per question type, it shows %s" % sorted(answers))
    for qtype in ("choice", "score"):
        ans = answers[qtype]
        p = np.array(list(ans["probabilities"].values()))
        entropy = confidence_from_probs(p, len(p))
        assert abs(ans["confidence"] - entropy) <= 1e-3, (
            "%s: printed confidence %s, entropy of %s is %s"
            % (qtype, ans["confidence"], list(ans["probabilities"]), round(entropy, 4)))
        assert ans["answer_confidence"] == round(float(p.max()), 4), (
            qtype, ans["answer_confidence"], round(float(p.max()), 4))
    ans = answers["noul"]
    p_true = ans["noul"]
    assert ans["confidence"] == round(max(p_true, 1.0 - p_true), 4), ans
    assert ans["confidence"] == ans["answer_confidence"], (
        "over two options the two are the same number, so a mismatch means the sample changed shape")
    entropy2 = round(confidence_from_probs(np.array([1.0 - p_true, p_true]), 2), 4)
    assert ans["confidence"] != entropy2, (
        "the noul sample now reads as entropy, so the blanket sentence would be defensible")
    m = _re.search(r"rather than the ([0-9.]+)", _prose(_read_page(DOC_QA)))
    assert m, "the page must keep the counterfactual that proves its noul row is not entropy"
    assert abs(float(m.group(1)) - entropy2) < 0.005, (m.group(1), entropy2)


def test_questions_page_row_table_matches_its_samples():
    body = next((b for _l, b in _fenced_blocks(_read_page(DOC_QA)) if "confidence 0." in b), None)
    assert body is not None, "the page lost the three-row confidence/answer_confidence comparison"
    rows = _re.findall(r"^(\w+)\s+confidence\s+([0-9.]+)\s+answer_confidence\s+([0-9.]+)", body, _re.M)
    answers = _qa_answers()
    named = {"dept": "choice", "urgent": "noul", "severity": "score"}
    assert [row[0] for row in rows] == list(named), rows
    for field, conf, ac in rows:
        ans = answers[named[field]]
        assert abs(float(conf) - ans["confidence"]) <= 1e-3, (field, conf, ans["confidence"])
        assert abs(float(ac) - ans["answer_confidence"]) <= 1e-3, (field, ac, ans["answer_confidence"])
    assert sorted(named.values()) == sorted(_decode_conf_formulas(DOC_AGENT_PY)), (
        "the rows do not cover the three types the agents build `confidence` for")


for _fn in (test_no_page_calls_confidence_one_formula,
            test_each_page_attributes_both_formulas_to_named_types,
            test_the_agents_build_what_the_pages_attribute,
            test_structured_page_sample_numbers_recompute,
            test_structured_page_claims_a_schema_that_mixes_the_two_types,
            test_questions_page_sample_answers_recompute,
            test_questions_page_row_table_matches_its_samples):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("doc-pages/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("doc-pages/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("doc-pages/%s" % _fn.__name__)

# ------------------------------------------------- a preset page's conclusions follow their numbers
# `examples/28_presets_moderation.py` prints a summary over `laya.moderation_questions()` answers,
# and the version this section replaces hardcoded three of its conclusions:
#   * `benign -> every flag 0.000, severity 0.24 / 3` printed `toxic` alone, while the same page's
#     table showed the benign post reading 0.066 on `spam`;
#   * ``severity orders the set correctly (1.34 > 1.05 > 0.99 > 0.24)`` put the `>` signs in the
#     format string, so the sentence stayed "correct" whatever the scores came back as;
#   * `the questions that separate them are harassment and threat` named two of four flags, while
#     the gaps measured on that same run were harassment 0.518, spam 0.193, threat 0.054 -- threat
#     was third, and spam was never mentioned.
# The page now builds those sentences from pure functions over its answers. No weights are loaded:
# the functions are extracted from the example's own AST and exec'd here on fabricated answers, so
# the gate drives the code the page actually runs rather than re-reading its prose.
import ast  # noqa: E402
import builtins  # noqa: E402
import re  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_28 = os.path.join(ROOT, "examples", "28_presets_moderation.py")
HELPERS_28 = ("noul_flags", "score_ceiling", "flag_line", "past_half", "gaps", "rank_by_severity")

# The page as it ships on main: the banner's universality claim and the three conclusions, as
# literal source lines. Every ban below is witnessed against this text, and every positive rule
# below is witnessed by showing this text does not satisfy it.
OLD_PAGE = '''
    lights up harassment, a generic one only toxic, spam only spam, and a normal post
    nothing at all.
        label, a["toxic"]["noul"], a["harassment"]["noul"], a["threat"]["noul"],
print("   benign          -> every flag %.3f, severity %.2f / 3"
      % (ben["toxic"]["noul"], ben["severity"]["score"]))
print("   `severity` orders the set correctly (%.2f > %.2f > %.2f > %.2f) but is a coarse 0-3"
print("   posts differ by only %.3f on `toxic`: the questions that separate them are")
print("   `harassment` and `threat`, not `toxic` on its own.")
'''

UNIVERSAL_ONE_FLOAT = re.compile(r"every (?:flag|score|level)[^.]{0,12}%\.\d+f", re.I)
ASSERTED_CHAIN = re.compile(r"%\.2f\s*>\s*%\.2f")
NAMED_SEPARATORS = re.compile(r"separat\w+[^.]{0,60}`(?:toxic|harassment|threat|spam)`"
                              r"[^.]{0,30}`(?:toxic|harassment|threat|spam)`", re.I)
HARDCODED_FLAG_READ = re.compile(r"\[[\"'](?:toxic|harassment|threat|spam)[\"']\]\[[\"']noul[\"']\]")
NOTHING_QUALIFIED = re.compile(r"nothing past 0\.5[^,]{0,10}, not zero", re.I)
DERIVED_FLAGS = re.compile(r"q\[.type.\]\s*==\s*[\"']noul[\"']")
DERIVED_CEILING = re.compile(r"len\(q\[.criteria.\]\)\s*-\s*1")


def _src28():
    with open(EXAMPLE_28, encoding="utf-8") as fh:
        return fh.read()


def _helpers28():
    """Exec only the example's top-level helper functions -- its `load()` call needs weights."""
    tree = ast.parse(_src28())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS_28]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), EXAMPLE_28, "exec"), ns)  # noqa: S102
    return ns


def _called28():
    """Names the example's top-level *statements* use, so a helper cannot pass by being dead."""
    tree = ast.parse(_src28())
    top = [n for n in tree.body if not isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    return {node.id for stmt in top for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _a(toxic, harassment, threat, spam, severity):
    return {"toxic": {"noul": toxic}, "harassment": {"noul": harassment},
            "threat": {"noul": threat}, "spam": {"noul": spam},
            "severity": {"score": severity}}


def test_preset_is_four_flags_and_one_rubric():
    """The docstring and the example's banner both claim this shape; nothing asserted it."""
    mq = laya.moderation_questions()
    assert list(mq) == ["toxic", "harassment", "threat", "spam", "severity"], list(mq)
    assert {k: q["type"] for k, q in mq.items()} == {
        "toxic": "noul", "harassment": "noul", "threat": "noul", "spam": "noul",
        "severity": "score"}
    assert len(mq["severity"]["criteria"]) == 4, len(mq["severity"]["criteria"])


def _free_names(node):
    """Names a function reads that it neither binds locally nor takes as an argument."""
    bound = {a.arg for fn in [node] + [k for k in ast.walk(node) if isinstance(k, ast.Lambda)]
             for a in fn.args.args}
    bound |= {n.id for n in ast.walk(node)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound}


def test_helpers_are_live_and_pure():
    """Each helper is called by the page, takes its whole world as arguments, and returns."""
    ns = _helpers28()
    got = sorted(k for k in ns if not k.startswith("__"))
    assert got == sorted(HELPERS_28), "example 28 lost a helper: %s" % got
    unused = set(HELPERS_28) - _called28()
    assert not unused, "defined but never called at module level: %s" % sorted(unused)
    for name in HELPERS_28:
        node = _fn_node(name)
        outside = sorted(n for n in _free_names(node) if not hasattr(builtins, n))
        assert not outside, "%s reads %s from the page, so this gate cannot drive it" % (name, outside)
        prints = [c for c in ast.walk(node)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "print"]
        assert not prints, "%s prints instead of returning, so the summary is not checkable" % name


def _fn_node(name):
    for n in ast.parse(_src28()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def test_flag_line_prints_every_flag():
    ns = _helpers28()
    flags = ns["noul_flags"](laya.moderation_questions())
    assert flags == ["toxic", "harassment", "threat", "spam"], flags
    line = ns["flag_line"](_a(0.0, 0.0, 0.0, 0.066, 0.24), flags)
    assert "0.066" in line, "the non-zero flag vanished: %r" % line
    assert re.findall(r"`([a-z]+)`", line) == flags, line
    assert len(re.findall(r"\d\.\d{3}", line)) == 4, "one number per flag: %r" % line


def test_past_half_uses_describes_cut():
    """`_common.describe` labels a noul true at `> 0.5`; the page must agree with it exactly."""
    ns = _helpers28()
    flags = ns["noul_flags"](laya.moderation_questions())
    assert ns["past_half"](_a(0.0, 0.0, 0.0, 0.066, 0.24), flags) == []
    assert ns["past_half"](_a(0.587, 0.662, 0.142, 0.049, 1.34), flags) == ["toxic", "harassment"]
    assert ns["past_half"](_a(0.5, 0.5, 0.5, 0.5, 1.0), flags) == [], "0.5 reads false, not true"


def test_gaps_names_the_widest_separators():
    """The old page said harassment and threat; on these numbers it is harassment and spam."""
    ns = _helpers28()
    flags = ns["noul_flags"](laya.moderation_questions())
    tgt, gen = _a(0.587, 0.662, 0.142, 0.049, 1.34), _a(0.584, 0.144, 0.088, 0.242, 1.05)
    g = ns["gaps"](tgt, gen, flags)
    assert [k for k, _ in g] == ["harassment", "spam", "threat", "toxic"], g
    assert abs(dict(g)["harassment"] - 0.518) < 1e-9 and dict(g)["spam"] < 0, g
    # and the name follows the data: move `threat` to the top and it is the separator.
    assert ns["gaps"](_a(0.1, 0.1, 0.9, 0.1, 1), _a(0.1, 0.1, 0.1, 0.1, 1), flags)[0][0] == "threat"


def test_rank_cannot_assert_its_own_order():
    ns = _helpers28()
    labels = ["a", "b", "c", "d"]
    scores = {"a": 1.34, "b": 1.05, "c": 0.99, "d": 0.24}
    results = {k: _a(0, 0, 0, 0, v) for k, v in scores.items()}
    chain, in_order = ns["rank_by_severity"](results, labels)
    assert in_order is True, chain
    assert [float(x) for x in re.findall(r"\d+\.\d{2}", chain)] == [1.34, 1.05, 0.99, 0.24], chain

    shuffled = {k: _a(0, 0, 0, 0, scores[j]) for k, j in
                zip(labels, ["d", "a", "c", "b"])}
    chain2, in_order2 = ns["rank_by_severity"](shuffled, labels)
    assert in_order2 is False, "the verdict must flip when the ranking does: %s" % chain2
    assert [float(x) for x in re.findall(r"\d+\.\d{2}", chain2)] == [1.34, 1.05, 0.99, 0.24], (
        "the printed chain must descend, whatever the order it lists: %s" % chain2)
    assert chain2.index("b") < chain2.index("a"), chain2


def test_page_drops_the_hardcoded_conclusions():
    """Each ban fires on main's page and not on this one -- a ban with no witness is a guess."""
    src = _src28()
    for name, rule in (("every flag %.3f", UNIVERSAL_ONE_FLOAT),
                       ("%.2f > %.2f in a format string", ASSERTED_CHAIN),
                       ("separate them are `harassment` and `threat`", NAMED_SEPARATORS),
                       ('a["toxic"]["noul"] in the summary', HARDCODED_FLAG_READ)):
        assert rule.search(OLD_PAGE), "%s does not fire on the wording it bans" % name
        assert not rule.search(src), "%s is still in example 28" % name


def test_page_qualifies_the_banner_and_derives_its_numbers():
    """The positive half: the page must read its flag list and rubric size from the preset."""
    src = _src28()
    assert NOTHING_QUALIFIED.search(src), "the banner must say that `nothing` is not zero"
    assert not re.search(r"nothing at all", src, re.I), "the unqualified claim is back"
    assert DERIVED_FLAGS.search(src), "flags must be read off the preset, not typed"
    assert DERIVED_CEILING.search(src), "the rubric ceiling must be read off the criteria"
    assert not re.search(r"/ 3\b", src), "a hardcoded severity divisor is back"
    # and main's page satisfies none of that, so these rules could not have passed before.
    assert not NOTHING_QUALIFIED.search(OLD_PAGE)
    assert re.search(r"nothing at all", OLD_PAGE, re.I)
    assert not DERIVED_FLAGS.search(OLD_PAGE) and not DERIVED_CEILING.search(OLD_PAGE)


# ------------------------------- example 03 must give `confidence` separately per question type
# `examples/03_reading_the_result.py` is the page a reader comes back to "when writing your own
# glue code", and its field tour closed with "`confidence` is 1 minus normalised entropy -- a scale
# that moves with the option count on the same answer". `Agent._decode_answers` runs two formulas
# and picks by question type (`laya/agent.py:1366-1402`):
#   * `choice` and `score`  ->  `round(confidence_from_probs(p, k), 4)` = `1 - H(p) / log(k)`
#   * `noul`                ->  `round(max(float(p[1]), 1.0 - float(p[1])), 4)`
# The page calls all three types in one `predict()`, so no single sentence covers `confidence` --
# and the proof is in the run the page itself prints: a `noul` answer reports the same number in
# `confidence` and `answer_confidence`, which a normalized entropy of a two-option distribution
# never equals (0.9/0.1 reads 0.531 as entropy, 0.900 as max(p)). Same defect class as examples
# 40/18/30 and as `DecisionResult`'s docstring, on the page a beginner reads first. The two pages
# that still carry the blanket sentence (`docs/structured.md`, `docs/questions-and-answers.md`) are
# left for a follow-up.
EXAMPLE_03 = os.path.join(ROOT, "examples", "03_reading_the_result.py")
AGENT_PY = os.path.join(ROOT, "laya", "agent.py")
ONNX_PY = os.path.join(ROOT, "laya", "onnx_agent.py")

# The wording this section bans, as it ships on main. Every ban below is witnessed against it, and
# every positive rule below is witnessed by showing that it does not satisfy the rule.
OLD_PAGE_03 = """
   `confidence` is 1 minus normalised entropy -- a scale that moves with the option
   count on the same answer. `answer_confidence` is max(p), the probability mass on
   the answer being reported, and it is the field to gate on (example 18 routes at a
   threshold on it).
"""

BAN_UNIFORM_ENTROPY = re.compile(r"`confidence` is 1 minus normalised entropy -- a scale", re.I)
NAMES_ENTROPY_TYPES = re.compile(r"normalised entropy on a `choice` or `score` answer", re.I)
NAMES_NOUL_FORMULA = re.compile(r"max\(p\[1\],\s*1\s*-\s*p\[1\]\)")
ATTRIBUTES_NOUL = re.compile(r"-- on a `noul`\.")
CALLS_SCALES_INCOMPARABLE = re.compile(r"scales are not comparable", re.I)
NAMES_ANSWER_MAXP = re.compile(r"`answer_confidence` is max\(p\),")
GATES_ON_ANSWER = re.compile(r"field to gate on \(example 18", re.I)
NAMES_BINNING_SCOPE = re.compile(
    r"`binning_map` remaps `answer_confidence` and leaves `confidence` alone", re.I)


def _src03(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _printed_text(path):
    """The page's output as one flattened string.

    Every string literal handed to `print()` anywhere in the file, joined and whitespace-collapsed,
    so a sentence the page wraps across four `print()` calls is still one sentence to the rules
    below -- and a claim that survives only because it is split across lines cannot hide from them.
    """
    parts = []
    for node in ast.walk(ast.parse(_src03(path))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            parts.extend(a.value for a in ast.walk(node)
                         if isinstance(a, ast.Constant) and isinstance(a.value, str))
    return " ".join(" ".join(parts).split())


def _confidence_exprs(path):
    """{answer type: source of the expression that becomes its `confidence`}.

    Read off `_decode_answers` through AST, one entry per answer dict the builder constructs, so
    the page is held to the code that produces the field rather than to a comment about it. A value
    that is a local name (the ONNX path pre-computes `conf_score`) is resolved to what it is bound
    to inside the same function.
    """
    src = _src03(path)
    func = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "_decode_answers")
    bound = {}
    for node in ast.walk(func):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            bound[node.targets[0].id] = ast.get_source_segment(src, node.value)
    found = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Dict):
            continue
        literal = {k.value: v for k, v in zip(node.keys, node.values)
                   if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if "type" not in literal or "confidence" not in literal:
            continue
        if not isinstance(literal["type"], ast.Constant):
            continue
        expr = ast.get_source_segment(src, literal["confidence"])
        found[literal["type"].value] = bound.get(expr, expr)
    return found


def test_page_drops_the_single_entropy_formula():
    text = _printed_text(EXAMPLE_03)
    assert BAN_UNIFORM_ENTROPY.search(OLD_PAGE_03), "the ban does not fire on the wording it bans"
    assert not BAN_UNIFORM_ENTROPY.search(
        text), "example 03 again gives `confidence` one formula for all three question types"
    for name, rule in (("the entropy formula is attributed to `choice` and `score`",
                        NAMES_ENTROPY_TYPES),
                       ("the `noul` formula is written out", NAMES_NOUL_FORMULA),
                       ("that formula is attributed to `noul`", ATTRIBUTES_NOUL),
                       ("the two scales are called incomparable", CALLS_SCALES_INCOMPARABLE),
                       ("`answer_confidence` is max(p)", NAMES_ANSWER_MAXP),
                       ("the gate field is named with its page", GATES_ON_ANSWER),
                       ("the binning map remaps only `answer_confidence`", NAMES_BINNING_SCOPE)):
        assert rule.search(text), "example 03 no longer states that %s" % name
    # and main's page satisfies none of the new rules, so they could not have passed before.
    old = " ".join(OLD_PAGE_03.split())
    assert not NAMES_ENTROPY_TYPES.search(old) and not NAMES_NOUL_FORMULA.search(old)
    assert not CALLS_SCALES_INCOMPARABLE.search(old) and not NAMES_BINNING_SCOPE.search(old)


def test_page_names_the_formulas_the_agents_build():
    exprs = _confidence_exprs(AGENT_PY)
    assert set(exprs) == {"choice", "score", "noul"}, (
        "the scan reached %s, not all three answer dicts" % sorted(exprs))
    assert all(exprs.values()), "a `confidence` value could not be read: %s" % exprs
    assert "confidence_from_probs" in exprs["choice"], exprs["choice"]
    assert "confidence_from_probs" in exprs["score"], exprs["score"]
    assert "1.0 - float(p[1])" in exprs["noul"], exprs["noul"]
    assert "confidence_from_probs" not in exprs["noul"], (
        "`noul` has switched to entropy -- the page's per-type split is now the stale claim: %s"
        % exprs["noul"])
    text = _printed_text(EXAMPLE_03)
    assert NAMES_ENTROPY_TYPES.search(text) and NAMES_NOUL_FORMULA.search(text), (
        "the page must name both expressions the builder uses")


def test_page_cites_the_numbers_the_repo_functions_produce():
    """The earned-numbers arm: the pairs the page prints must be what `laya.common` returns."""
    text = _printed_text(EXAMPLE_03)
    for p_true in (0.60, 0.90):
        p = np.array([1.0 - p_true, p_true])
        noul = "%.3f" % max(float(p[1]), 1.0 - float(p[1]))
        entropy = "%.3f" % round(float(confidence_from_probs(p, 2)), 3)
        rule = re.compile(r"%s reads %s[^.]*%s" % (re.escape(noul), re.escape(noul),
                                                   re.escape(entropy)))
        assert rule.search(text), (
            "the page must cite p(true)=%s as %s on a `noul` and %s as entropy; both come from "
            "confidence_from_probs above, so a number that drifts fails here" % (noul, noul,
                                                                                entropy))
    # the reason one sentence could not work: on a `noul` the shipped `confidence` *is* max(p), the
    # same quantity `answer_confidence` reports, which an entropy cannot be.
    for p_true in (0.60, 0.90):
        p = np.array([1.0 - p_true, p_true])
        assert close(max(float(p[1]), 1.0 - float(p[1])), answer_confidence(p, 2)), (
            "the page's `noul` answer would no longer carry the same number in both fields")


def _binning_targets(path):
    """What the installed binning map is applied to inside `_decode_answers`.

    The page closes its tour with "`binning_map` remaps `answer_confidence` and leaves `confidence`
    alone", which is a claim about the builder, not about prose: the map's input is the raw
    `answer_confidence`, and the two `confidence` expressions sit outside that branch.
    """
    src = _src03(path)
    func = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "_decode_answers")
    return [ast.get_source_segment(src, node.args[0]) for node in ast.walk(func)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "apply_binning_map" and node.args)]


def test_the_remap_reaches_only_answer_confidence():
    for path in (AGENT_PY, ONNX_PY):
        targets = _binning_targets(path)
        assert targets, "%s no longer applies a binning map inside the answer builder" % path
        assert all(t == "ans_raw" for t in targets), (
            "the page says a calibration payload remaps `answer_confidence` and leaves "
            "`confidence` alone, but %s remaps %s" % (path, sorted(set(targets))))


def test_the_two_agents_answer_the_same_way():
    """Parity arm: the ONNX path builds the same two expressions, so one page can be true of both."""
    agent_exprs = _confidence_exprs(AGENT_PY)
    onnx_exprs = _confidence_exprs(ONNX_PY)
    assert set(onnx_exprs) == {"choice", "score", "noul"}, (
        "the ONNX answer builder no longer yields all three types: %s" % sorted(onnx_exprs))
    assert "confidence_from_probs" in onnx_exprs["choice"], onnx_exprs["choice"]
    assert "confidence_from_probs" in onnx_exprs["score"], onnx_exprs["score"]
    assert onnx_exprs["noul"] == agent_exprs["noul"], (
        "the two agents now disagree on what a `noul` confidence is (%r vs %r), and the page "
        "describes only one of them" % (onnx_exprs["noul"], agent_exprs["noul"]))


for _fn in (test_page_drops_the_single_entropy_formula,
            test_page_names_the_formulas_the_agents_build,
            test_page_cites_the_numbers_the_repo_functions_produce,
            test_the_remap_reaches_only_answer_confidence,
            test_the_two_agents_answer_the_same_way):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-03/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("page-03/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-03/%s" % _fn.__name__)


for _fn in (test_preset_is_four_flags_and_one_rubric, test_helpers_are_live_and_pure,
            test_flag_line_prints_every_flag, test_past_half_uses_describes_cut,
            test_gaps_names_the_widest_separators, test_rank_cannot_assert_its_own_order,
            test_page_drops_the_hardcoded_conclusions,
            test_page_qualifies_the_banner_and_derives_its_numbers):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-28/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("page-28/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-28/%s" % _fn.__name__)


# --------------------------------------------- example 40 must not call a raw softmax calibrated
# `examples/40_caching_and_monitoring.py` gates 15 tickets on `answer_confidence`. On main it calls
# that field "the calibrated probability of the answer Laya reports" in both its banner and its
# closing paragraph, and its `bucket()` docstring reads "application policy over a calibrated
# number". `answer_confidence` is max(p) -- the quantity temperature scaling fits and ECE measures
# -- but the README's own Calibration section says that reading holds *only* after temperatures are
# fitted and validated, the shipped checkpoints are over-confident, and the loader emits at load
# time `RuntimeWarning: ... choice:11+=0.10058... -> 0.5. Treat confidence from the affected
# entries as uncalibrated.` The page now reads the temperature each shape was actually scaled by off
# the loaded agent and states the calibration as conditional.
#
# No weights are loaded here: the four helpers are pulled from the example's AST and exec'd against
# a fabricated agent, and `scale_for`'s lookup is checked against core's own `temp_bucket`, so the
# gate drives the code the page runs instead of re-reading its prose.
from laya.common import QTYPES as _QTYPES40, temp_bucket as _temp_bucket  # noqa: E402

EXAMPLE_40 = os.path.join(ROOT, "examples", "40_caching_and_monitoring.py")
HELPERS_40 = ("option_count", "scale_for", "clamped_buckets", "entropy_confidence")
# Names a helper may read from the page's world: the two core symbols it must defer to, and `math`
# for the entropy. Any other non-builtin free name means this gate cannot drive that helper.
CORE_GLOBALS_40 = {"temp_bucket", "QTYPES", "math"}

# main's page, as literal source lines. Every ban is witnessed against this text; every positive
# rule below is witnessed by showing this text does NOT satisfy it.
OLD_PAGE_40 = '''
    the triage preset and gates on `answer_confidence`, the calibrated probability of the
    """Our thresholds, not the model's: application policy over a calibrated number."""
   The monitor gates on `answer_confidence`: the calibrated probability of the answer Laya
   distribution has H/log(k) = %.2f, so `confidence` is %.2f while
   """ % (entropy, 1 - entropy, vague["intent"]["answer_confidence"]))
'''

UNCONDITIONAL_CALIBRATION = re.compile(r"the calibrated probability", re.I)
POLICY_OVER_CALIBRATED = re.compile(r"over a calibrated number", re.I)
SUBSTITUTED_CONFIDENCE = re.compile(r"1\s*-\s*entropy")
DERIVES_SCALING = re.compile(r"temp_bucket\(")
READS_REPORTED_CONFIDENCE = re.compile(r"\[[\"']confidence[\"']\]")
HARDCODED_BUCKET = re.compile(r"[\"'](choice|score|noul):[0-9]")


def _src40():
    with open(EXAMPLE_40, encoding="utf-8") as fh:
        return fh.read()


def _helpers40():
    """Exec only the example's top-level helpers -- its `load()` call needs weights."""
    tree = ast.parse(_src40())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS_40]
    ns = {"temp_bucket": _temp_bucket, "QTYPES": _QTYPES40, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), EXAMPLE_40, "exec"), ns)  # noqa: S102
    return ns


def _called40():
    tree = ast.parse(_src40())
    top = [n for n in tree.body if not isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    return {node.id for stmt in top for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _fn40(name):
    for n in ast.parse(_src40()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


class _FakeAgent(object):
    """Only the three temperature attributes `scale_for` and `clamped_buckets` read."""

    def __init__(self, by_options, per_type, raw=None):
        self.temperature_by_options = by_options
        self.temperature = per_type
        self.temperature_by_options_raw = dict(by_options) if raw is None else raw


def test_page40_helpers_are_live_defer_to_core():
    defined = sorted(n.name for n in ast.parse(_src40()).body
                     if isinstance(n, ast.FunctionDef) and n.name in HELPERS_40)
    assert defined == sorted(HELPERS_40), "example 40 lost a helper: %s" % defined
    ns = _helpers40()
    unused = set(HELPERS_40) - _called40()
    assert not unused, "defined but never called at module level: %s" % sorted(unused)
    for name in HELPERS_40:
        node = _fn40(name)
        outside = sorted(n for n in _free_names(node)
                         if not hasattr(builtins, n) and n not in CORE_GLOBALS_40)
        assert not outside, "%s reads %s from the page, so this gate cannot drive it" % (name, outside)
        prints = [c for c in ast.walk(node)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "print"]
        assert not prints, "%s prints instead of returning" % name
    # `scale_for` must resolve the bucket through core, not carry a typed-in table.
    assert not HARDCODED_BUCKET.search(ast.dump(_fn40("scale_for"))), (
        "scale_for hardcodes a bucket name instead of calling temp_bucket")


def test_scale_for_replays_core_lookup():
    """For every shape, scale_for returns exactly `temperature_by_options.get(bucket, per_type)`."""
    ns = _helpers40()
    per_type = [7.0, 8.0, 9.0]                       # choice, score, noul fallbacks
    by_options = {"choice:6-10": 1.0, "noul:2": 1.9834, "score:3-5": 1.2514}
    fake = _FakeAgent(by_options, per_type)
    for qtype, k in (("choice", 6), ("noul", 2), ("score", 4), ("noul", 2)):
        name, applied, in_map = ns["scale_for"](fake, qtype, k)
        bucket = _temp_bucket(_QTYPES40[qtype], k)
        assert name == bucket, (qtype, k, name, bucket)
        assert in_map == (bucket in by_options), (bucket, in_map)
        assert abs(applied - by_options.get(bucket, per_type[_QTYPES40[qtype]])) < 1e-9, (
            "%s: %r not core's get(...)" % (bucket, applied))
    # the fallback arm: an unmapped shape must fall to the per-type temperature, flagged not-in-map.
    name, applied, in_map = ns["scale_for"](_FakeAgent({}, [0.5, 0.6, 0.7]), "choice", 12)
    assert (name, applied, in_map) == ("choice:11+", 0.5, False), (name, applied, in_map)
    # a mapped temperature of exactly 1.0 must be reported as 1.0, not nudged.
    assert ns["scale_for"](_FakeAgent({"choice:6-10": 1.0}, [1.0, 1.0, 1.0]), "choice", 6) == (
        "choice:6-10", 1.0, True)


def test_option_count_matches_the_shipped_preset():
    ns = _helpers40()
    counts = {qid: ns["option_count"](q) for qid, q in laya.triage_questions().items()}
    assert counts == {"intent": 6, "is_urgent": 2, "frustration": 4,
                      "refund_requested": 2, "churn_risk": 2}, counts


def test_clamped_buckets_names_only_a_shipped_value_the_runtime_refused():
    ns = _helpers40()
    # `choice:11+` ships below TEMP_MIN and is clamped to 0.5; everything else ships unchanged.
    applied = {"choice:11+": 0.5, "noul:2": 1.9834}
    raw = {"choice:11+": 0.10058280825614929, "noul:2": 1.9834}
    out = ns["clamped_buckets"](_FakeAgent(applied, [0.5, 0.5, 0.5], raw))
    assert out == ["choice:11+=0.1006 -> 0.5000"], out
    # and an honest checkpoint reports nothing.
    same = {"noul:2": 1.9834}
    assert ns["clamped_buckets"](_FakeAgent(same, [0.5, 0.5, 0.5], dict(same))) == []


def test_entropy_confidence_recomputes_core_exactly():
    ns = _helpers40()
    for dist in ([0.5, 0.3, 0.2], [0.9, 0.1], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0]):
        p = np.array(dist, dtype=float)
        got = ns["entropy_confidence"](dist)
        want = confidence_from_probs(p, len(dist))
        assert close(got, want, 1e-9), (dist, got, want)
    assert ns["entropy_confidence"]([1.0]) == 1.0, "k<2 is 1.0 by definition"


def test_page40_drops_the_unconditional_calibration_claim():
    """Each ban fires on main's page and not on this one -- a ban with no witness is a guess."""
    src = _src40()
    for name, rule in (("'the calibrated probability'", UNCONDITIONAL_CALIBRATION),
                       ("'over a calibrated number'", POLICY_OVER_CALIBRATED),
                       ("prints `1 - entropy` as the confidence", SUBSTITUTED_CONFIDENCE)):
        assert rule.search(OLD_PAGE_40), "%s does not fire on the wording it bans" % name
        assert not rule.search(src), "%s is still in example 40" % name


def test_page40_reads_the_scaling_and_the_reported_field():
    """The positive half: the page must derive the temperature and print the reported field."""
    src = _src40()
    assert DERIVES_SCALING.search(src), "the bucket must be resolved through core's temp_bucket"
    assert READS_REPORTED_CONFIDENCE.search(src), "the page must read the reported `confidence`"
    # and main's page satisfies none of that, so these rules could not have passed before.
    assert not DERIVES_SCALING.search(OLD_PAGE_40)
    assert not READS_REPORTED_CONFIDENCE.search(OLD_PAGE_40)


for _fn in (test_page40_helpers_are_live_defer_to_core, test_scale_for_replays_core_lookup,
            test_option_count_matches_the_shipped_preset,
            test_clamped_buckets_names_only_a_shipped_value_the_runtime_refused,
            test_entropy_confidence_recomputes_core_exactly,
            test_page40_drops_the_unconditional_calibration_claim,
            test_page40_reads_the_scaling_and_the_reported_field):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-40/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("page-40/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-40/%s" % _fn.__name__)
# Gate: the two predict_long call sites' comments must describe the None-min_confidence
# contract truthfully. The pre-fix wording claimed every answer "reports that it ran ungated"
# / "say so on every answer"; confidence.py:196-197 documents and :211-212 implements the
# opposite -- the gate writes nothing at all, so the payload carries no `abstention` field.
import ast as _gate_ast  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gate_gate_comment(rel_path):
    """Return the run of comment lines that immediately precede the
    apply_confidence_gate([result], None) call inside predict_long."""
    with open(os.path.join(_REPO, rel_path)) as f:
        lines = f.readlines()
    tree = _gate_ast.parse("".join(lines))
    for node in _gate_ast.walk(tree):
        if isinstance(node, _gate_ast.FunctionDef) and node.name == "predict_long":
            call_line = None
            for sub in _gate_ast.walk(node):
                if (isinstance(sub, _gate_ast.Call)
                        and getattr(sub.func, "id", getattr(sub.func, "attr", None)) == "apply_confidence_gate"
                        and len(sub.args) == 2
                        and isinstance(sub.args[1], _gate_ast.Constant)
                        and sub.args[1].value is None):
                    call_line = sub.lineno - 1
                    break
            if call_line is None:
                continue
            out = []
            i = call_line - 1
            while i >= 0 and lines[i].lstrip().startswith("#"):
                out.insert(0, lines[i].strip().lstrip("#").strip())
                i -= 1
            return " ".join(out)
    return ""


_agent_comment = _gate_gate_comment(os.path.join("laya", "agent.py"))
_onnx_comment = _gate_gate_comment(os.path.join("laya", "onnx_agent.py"))

for _label, _comment in (("agent.py", _agent_comment), ("onnx_agent.py", _onnx_comment)):
    check_true("gate/comment/%s exists" % _label, len(_comment) > 0, "no comment found")
    check_true("gate/comment/%s drops the ungated-reports claim" % _label,
               "reports that it ran ungated" not in _comment
               and "say so on every answer" not in _comment,
               "pre-fix wording still on the call site: %r" % _comment)
    check_true("gate/comment/%s names writes nothing" % _label,
               "writes nothing" in _comment,
               "gate comment must name the None-contract: %r" % _comment)
    check_true("gate/comment/%s names the min_confidence argument" % _label,
               "min_confidence" in _comment,
               "gate comment must refer to the argument it is about: %r" % _comment)

# Live witness: apply_confidence_gate([payload], None) leaves the payload byte-identical,
# so the corrected comment is not itself a claim the code contradicts.
_witness = {"model": "w", "answers": {"q": {"choice": "a", "confidence": 0.42}}, "usage": {}}
_witness_before = copy.deepcopy(_witness)
apply_confidence_gate([_witness], None)
check("gate/live None writes nothing", _witness, _witness_before)
check("gate/live None adds no abstention key", "abstention" in _witness["answers"]["q"], False)


# --------------------------------------------- example 18 must not call the threshold a measurement
# `examples/18_confidence_gating.py` is the intro-level gating pattern: it picks a 0.85 cutoff on
# `answer_confidence` and acts on it. On main it calls the field "the calibrated probability Laya
# puts on the answer it reports" and its module docstring says it "branches on Laya's calibrated
# confidence". The README's Calibration section (line 1586) says the opposite about the shipped
# state: mean ECE on `laya` is 0.466 as shipped, dropping to 0.081 only after refitting one
# temperature per (question type, option-count) bucket on held-out data, and both shipped
# checkpoints are over-confident. `answer_confidence` is the field temperature scaling fits and
# the abstention gate reads -- but "calibrated" is a property a fit earns, not a property of the
# field, and the loader emits `RuntimeWarning: ... Treat confidence from the affected entries as
# uncalibrated.` on this very checkpoint. The page now reads back the temperature it actually
# applies to each of its two questions and states the calibration claim as conditional.
from laya.common import QTYPES as _QTYPES18, temp_bucket as _temp_bucket18  # noqa: E402

EXAMPLE_18 = os.path.join(ROOT, "examples", "18_confidence_gating.py")
HELPERS_18 = ("option_count", "scale_for")
CORE_GLOBALS_18 = {"temp_bucket", "QTYPES", "math"}

OLD_PAGE_18 = '''
Answers a batch of support emails and branches on Laya\'s calibrated confidence: act
    The production pattern from the README. Gate on `answer_confidence`: the calibrated
    probability Laya puts on the answer it reports, defined the same way on every question
'''

UNQUALIFIED_CALIBRATION_18 = re.compile(
    r"the\s+calibrated\s+probability|Laya['’]s\s+calibrated\s+confidence", re.I)
DERIVES_SCALING_18 = re.compile(r"temp_bucket\(")
NAMES_CONDITION_18 = re.compile(r"over-confident", re.I)
HARDCODED_BUCKET_18 = re.compile(r"[\"'](choice|score|noul):[0-9]")


def _src18():
    with open(EXAMPLE_18, encoding="utf-8") as fh:
        return fh.read()


def _helpers18():
    tree = ast.parse(_src18())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS_18]
    ns = {"temp_bucket": _temp_bucket18, "QTYPES": _QTYPES18, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), EXAMPLE_18, "exec"), ns)  # noqa: S102
    return ns


class _FakeAgent18(object):
    """Only the two temperature attributes `scale_for` reads."""

    def __init__(self, by_options, per_type):
        self.temperature_by_options = by_options
        self.temperature = per_type


def _top_literal18(name):
    for n in ast.parse(_src18()).body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)


def _fn18(name):
    for n in ast.parse(_src18()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def test_page18_helpers_are_live_and_defer_to_core():
    defined = sorted(n.name for n in ast.parse(_src18()).body
                     if isinstance(n, ast.FunctionDef) and n.name in HELPERS_18)
    assert defined == sorted(HELPERS_18), "example 18 lost a helper: %s" % defined
    tree = ast.parse(_src18())
    top_loads = {node.id for stmt in tree.body if not isinstance(
        stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    unused = set(HELPERS_18) - top_loads
    assert not unused, "defined but never called at module level: %s" % sorted(unused)
    for name in HELPERS_18:
        node = _fn18(name)
        outside = sorted(n for n in _free_names(node)
                         if not hasattr(builtins, n) and n not in CORE_GLOBALS_18)
        assert not outside, "%s reads %s from the page" % (name, outside)
        prints = [c for c in ast.walk(node)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "print"]
        assert not prints, "%s prints instead of returning" % name
    assert not HARDCODED_BUCKET_18.search(ast.dump(_fn18("scale_for"))), (
        "scale_for hardcodes a bucket name instead of calling temp_bucket")


def test_page18_questions_reach_core_bucket_names():
    """Replay the page's own QUESTIONS through core's temp_bucket and through the example's helper."""
    ns = _helpers18()
    questions = _top_literal18("QUESTIONS")
    per_type = [7.0, 8.0, 9.0]
    by_options = {"choice:3-5": 1.7601518630981445, "noul:2": 1.983399510383606}
    fake = _FakeAgent18(by_options, per_type)
    shapes = {}
    for qid, q in questions.items():
        k = ns["option_count"](q)
        name, applied, in_map = ns["scale_for"](fake, q["type"], k)
        assert name == _temp_bucket18(_QTYPES18[q["type"]], k), (qid, name)
        assert abs(applied - by_options[name]) < 1e-9, (qid, applied)
        assert in_map is True, "%s: %s missing from map" % (qid, name)
        shapes[qid] = (q["type"], k, name, applied)
    assert shapes == {
        "department": ("choice", 4, "choice:3-5", 1.7601518630981445),
        "refund_requested": ("noul", 2, "noul:2", 1.983399510383606),
    }, shapes


def test_page18_drops_the_unconditional_calibration_claim():
    """The two phrases main uses fire on the ban and are gone here; main has no temp_bucket call."""
    src = _src18()
    assert UNQUALIFIED_CALIBRATION_18.search(OLD_PAGE_18), "the ban must fire on main's wording"
    assert not UNQUALIFIED_CALIBRATION_18.search(src), (
        "the page still calls `answer_confidence` calibrated without condition")
    assert not DERIVES_SCALING_18.search(OLD_PAGE_18), "main's page has no scaling lookup"
    assert not NAMES_CONDITION_18.search(OLD_PAGE_18), "main's page states no calibration condition"
    assert DERIVES_SCALING_18.search(src), "the page must resolve the bucket via core's temp_bucket"
    assert NAMES_CONDITION_18.search(src), (
        "the page must carry the README's Calibration condition (mean ECE 0.466 -> 0.081, "
        "both shipped checkpoints over-confident)")
    # The README's exact language lives at line 1586; the page must not misquote it.
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "Both checkpoints are over-confident as shipped" in readme
    assert "0.466 -> 0.081" in readme


for _fn in (test_page18_helpers_are_live_and_defer_to_core,
            test_page18_questions_reach_core_bucket_names,
            test_page18_drops_the_unconditional_calibration_claim):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-18/%s: %s" % (_fn.__name__, e))
    except Exception as e:
        FAIL.append("page-18/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-18/%s" % _fn.__name__)

# ------------------------------------------ example 30's score confidence recomputation
# `examples/30_custom_schema_design.py` teaches the `score` answer type on the page. Main's
# version asserted -- in prose, with nothing to check it against -- that "`confidence` on a
# score is normalised entropy, so a wide-but-ordered distribution looks unconfident". That is
# the inverse of the repo's own definition: `laya.common.confidence_from_probs` returns
# `1 - H(p) / log(k)`, examples 18/28/40 all describe it as "1 minus normalised entropy", and
# the sentence was self-contradictory in place -- if confidence were the entropy itself, a wide
# distribution would read as HIGH confidence, not "unconfident".
#
# The fix rewrites the sentence to the correct definition AND recomputes the field on the spot:
# `entropy_confidence(list(probabilities.values()))` is printed next to
# `answer["severity"]["confidence"]` for both records, so the reader sees 1 - H/log(k) produce
# the reported 0.2314 / 0.2019 to within rounding. No weights are loaded here; the gate execs
# the helper from the example's own AST and drives it against `confidence_from_probs`.
EXAMPLE_30 = os.path.join(ROOT, "examples", "30_custom_schema_design.py")
HELPERS_30 = ("entropy_confidence",)

# The page as it ships on main: the sentence this PR replaces, verbatim. The ban below must
# fire on this text and not on the current example; the positive rules must fire on the
# current example and not on this text.
OLD_PAGE_30 = '''
print("   them, and a choice when the labels are unordered. Read the score itself: `confidence`")
print("   on a score is normalised entropy, so a wide-but-ordered distribution looks")
print("   unconfident even when the expected level is informative.")
'''

# The definition stated as the entropy itself rather than 1 - H/log(k).
INVERTED_SCORE_DEF = re.compile(
    r"`confidence`[^.]{0,120}?\bon a score is (?:the )?normali[sz]ed entropy", re.I | re.S)
# The correct shape, either symbolic or spelled out.
SCORE_DEFINES_1_MINUS_H = re.compile(
    r"1\s*-\s*H\s*/\s*log\(k\)|one\s+minus\s+the\s+normali[sz]ed\s+entropy", re.I)
# The recomputation call must be in the source, not just asserted. Two parts: the page must
# read the reported probabilities off the answer, and it must feed them into the helper.
PROBS_READ = re.compile(
    r"list\(\s*ans\[\"severity\"\]\[\"probabilities\"\]\s*\.\s*values\(\)\s*\)")
HELPER_CALLED = re.compile(r"entropy_confidence\(")
# Reported and recomputed values printed side by side (either literal digits or a %.Nf
# conversion the page formats them with).
PRINTS_REPORTED_AND_RECOMPUTED = re.compile(
    r"reported\s+(?:%?\.\d+f|[\d.]+)\s*,\s*recomputed", re.I)


def _src30():
    with open(EXAMPLE_30, encoding="utf-8") as fh:
        return fh.read()


def _helpers30():
    """Exec the example's pure top-level helpers. `entropy_confidence` reads `math`."""
    tree = ast.parse(_src30())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS_30]
    ns = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), EXAMPLE_30, "exec"), ns)  # noqa: S102
    return ns, {n.name for n in nodes}


def _called30():
    """Names the example's top-level *statements* use, so a helper cannot pass by being dead."""
    tree = ast.parse(_src30())
    top = [n for n in tree.body if not isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    return {node.id for stmt in top for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def test_page30_helpers_are_live_and_pure():
    """`entropy_confidence` is defined, called at top level, and returns rather than prints."""
    _, defined = _helpers30()
    assert defined == set(HELPERS_30), "example 30 lost a helper: %s" % sorted(defined)
    unused = set(HELPERS_30) - _called30()
    assert not unused, "defined but never called at module level: %s" % sorted(unused)
    tree = ast.parse(_src30())
    for name in HELPERS_30:
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        prints = [c for c in ast.walk(node)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                  and c.func.id == "print"]
        assert not prints, "%s prints instead of returning" % name


def test_entropy_confidence_matches_core():
    """Recomputing 1 - H/log(k) by hand must equal `confidence_from_probs` on the same list."""
    ns = _helpers30()[0]
    for probs in ([0.5, 0.5], [0.1, 0.9], [0.25, 0.25, 0.25, 0.25],
                  [0.7, 0.2, 0.05, 0.05], [1e-9, 0.999999999, 0.0],
                  [0.4, 0.3, 0.2, 0.1], [0.05, 0.15, 0.3, 0.35, 0.1, 0.05]):
        got = ns["entropy_confidence"](list(probs))
        want = confidence_from_probs(np.array(probs, dtype=float), len(probs))
        assert abs(got - want) < 1e-9, (probs, got, want)
    assert ns["entropy_confidence"]([1.0]) == 1.0, "k=1 is certain"
    assert ns["entropy_confidence"]([]) == 1.0, "k=0 does not crash"


def test_page30_drops_the_inverted_definition():
    """The ban fires on main's sentence and not on this one; the positive rule inverts."""
    src = _src30()
    assert INVERTED_SCORE_DEF.search(OLD_PAGE_30), \
        "the ban does not fire on main's wording"
    assert not INVERTED_SCORE_DEF.search(src), \
        "the inverted definition is still in example 30"
    assert SCORE_DEFINES_1_MINUS_H.search(src), \
        "the correct 1 - H/log(k) definition is missing"
    assert not SCORE_DEFINES_1_MINUS_H.search(OLD_PAGE_30), \
        "the positive rule passes on main's page too"


def test_page30_recomputes_in_place():
    """The reader sees the field cross-checked against the probabilities that produced it."""
    src = _src30()
    assert PROBS_READ.search(src), \
        "the score section no longer reads the reported `probabilities` list"
    assert HELPER_CALLED.search(src), \
        "the score section no longer calls `entropy_confidence`"
    assert PRINTS_REPORTED_AND_RECOMPUTED.search(src), \
        "reported and recomputed values are not printed side by side"
    assert not PROBS_READ.search(OLD_PAGE_30)
    assert not HELPER_CALLED.search(OLD_PAGE_30)
    assert not PRINTS_REPORTED_AND_RECOMPUTED.search(OLD_PAGE_30)


for _fn in (test_page30_helpers_are_live_and_pure,
            test_entropy_confidence_matches_core,
            test_page30_drops_the_inverted_definition,
            test_page30_recomputes_in_place):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-30/%s: %s" % (_fn.__name__, e))
    except Exception as e:
        FAIL.append("page-30/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-30/%s" % _fn.__name__)


# -------------------------------- evals' --min-confidence wording must match the gate's effect
#
# `laya/evals_cli.py`, `laya/evals.py`, `docs/evals.md`, `tests/test_evals.py` and
# `tests/test_evals_api.py` each describe the abstention gate's role in a scored run. Their
# previous wording claimed the threshold changes what scores: "answers below it come back
# abstained", "an abstention overwrites a low-confidence choice", "the run scores the policy
# at that threshold, not the raw argmax", "a `precision@coverage` figure for a policy that
# never ran". But `apply_confidence_gate` in this module writes `low_confidence` and
# `abstention` state fields and leaves `answer["choice"]/["noul"]/["score"]` untouched, and
# every `Evaluator.score` and `_correct` in `laya/evals.py` reads only those value keys. So the
# metrics -- accuracy, ece, brier, aurc, selective_accuracy@NN -- are identical at every
# threshold; what changes is `report.config["timing"]["min_confidence"]` and `min_confidence_sent`,
# which the wording below must describe as claims about the run, not claims about the answers.

_EVALS_CLI_PATH = os.path.join(ROOT, "laya", "evals_cli.py")
_EVALS_PY_PATH = os.path.join(ROOT, "laya", "evals.py")
_EVALS_MD_PATH = os.path.join(ROOT, "docs", "evals.md")
_TEST_EVALS_PATH = os.path.join(ROOT, "tests", "test_evals.py")
_TEST_EVALS_API_PATH = os.path.join(ROOT, "tests", "test_evals_api.py")


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _docstring_of(path, func_name):
    """A function's docstring pulled out of the file, so a ban can point at one site not the file."""
    tree = ast.parse(_read_text(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_docstring(node) or ""
    return ""


def _cli_help_of(path, flag):
    """The concatenated `help=` literal on a `parser.add_argument(flag, ...)` call."""
    tree = ast.parse(_read_text(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args \
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag:
            for kw in node.keywords:
                if kw.arg == "help":
                    try:
                        return ast.literal_eval(kw.value) or ""
                    except (ValueError, SyntaxError):
                        return ""
    return ""


# The wrong-language patterns: each is a claim that the threshold changes the scored numbers.
_BAN_COME_BACK_ABSTAINED = re.compile(r"come back\s+abstained", re.I)
_BAN_OVERWRITES_A_LOW = re.compile(r"abstention\s+overwrites\s+a\s+low", re.I)
_BAN_SCORES_POLICY = re.compile(r"scores the policy at (?:the|that) threshold", re.I)
_BAN_VS_RAW_ARGMAX = re.compile(r"(?:rather than|not)\s+the\s+raw\s+argmax", re.I)
_BAN_MINCONFIDENCE_CHANGES_ANSWER = re.compile(
    r"`min_confidence`[^.\n]{0,80}?changes the answ", re.I)
_BAN_UNLIKE_GROUPING_CHANGES = re.compile(
    r"[Uu]nlike\s+`?grouping`?[^.]{0,40}?changes the answ", re.I)
_BAN_PRECISION_NUMBER_FOR_POLICY = re.compile(
    r"precision@coverage`?\s+(?:number|figure)\s+for a policy that never ran", re.I)
_BAN_WOULD_REPORT_PRECISION = re.compile(
    r"would report `?precision@coverage`?", re.I)
_BAN_OTHERWISE_SCORE_POLICY = re.compile(
    r"otherwise score a policy that never ran", re.I)
_BAN_NO_WAY_TO_MEASURE_PATCOV = re.compile(
    r"no way to measure `?precision@coverage", re.I)
_BAN_A_SCORING_CONTROL = re.compile(r"(?:it is|that is|dropping) a\b[^.\n]{0,10}scoring control", re.I)
_BAN_ANSWER_NOT_SAME_DECISION = re.compile(
    r"an answer below the threshold is not the same decision", re.I)
_BAN_T0_T07_DIFFERENT_EXPERIMENT = re.compile(
    r"same run at `T=0` and `T=0\.7` is a different experiment", re.I)
_BAN_SWEEP_IS_SERIES = re.compile(
    r"`precision@coverage`\s+sweep is a series of these", re.I)
_BAN_CHANGES_WHICH_ANSWERS_SCORE = re.compile(
    r"(?:abstention threshold|min_confidence).{0,30}?changes which answers score", re.I | re.S)

_EVALS_BANS = (
    _BAN_COME_BACK_ABSTAINED, _BAN_OVERWRITES_A_LOW, _BAN_SCORES_POLICY,
    _BAN_VS_RAW_ARGMAX, _BAN_MINCONFIDENCE_CHANGES_ANSWER,
    _BAN_UNLIKE_GROUPING_CHANGES, _BAN_PRECISION_NUMBER_FOR_POLICY,
    _BAN_WOULD_REPORT_PRECISION, _BAN_OTHERWISE_SCORE_POLICY,
    _BAN_NO_WAY_TO_MEASURE_PATCOV, _BAN_A_SCORING_CONTROL,
    _BAN_ANSWER_NOT_SAME_DECISION, _BAN_T0_T07_DIFFERENT_EXPERIMENT,
    _BAN_SWEEP_IS_SERIES, _BAN_CHANGES_WHICH_ANSWERS_SCORE,
)

# Positive claims the wording must make: name the two state fields `apply_confidence_gate`
# writes. A wording that only says what the gate does NOT do is a diff, not a fix.
_SAYS_LOW_CONFIDENCE = re.compile(r"low_confidence")
_SAYS_ABSTENTION_FIELD = re.compile(
    r"abstention[^a-zA-Z]{0,3}[:=]?[^a-zA-Z]{0,3}[\"']abstained[^a-zA-Z]{0,3}[\"']")

# The pre-fix wording at each of the ten sites we touched, so every ban above has a witness
# it fires on. A ban that never fires is a rule the code cannot check.
_OLD_E1 = ("abstention threshold on `answer_confidence` (#361): answers below it come back "
           "abstained, so the run scores the policy at that threshold rather than the raw "
           "argmax.")
_OLD_E2 = ("`min_confidence` changes the answer (an abstention overwrites a low-confidence "
           "choice), so it is a scoring control, not an optimisation. silently dropping the "
           "threshold and reporting the same run would give a `precision@coverage` number for "
           "a policy that never ran")
_OLD_E3 = ("answers below it come back abstained, so the run scores the policy at that "
           "threshold, not the raw argmax. Unlike `sort_by_length` this changes the answers")
_OLD_E4 = ("because an abstention threshold changes which answers score as correct. Silently "
           "dropping it would publish a `precision@coverage` number for a policy that never ran")
_OLD_E5 = ("or drop the threshold -- the report would otherwise score a policy that never ran")
_OLD_MD1 = ("Unlike grouping, this changes the answers that score: the same run at `T=0` and "
            "`T=0.7` is a different experiment, and a `precision@coverage` sweep is a series "
            "of these, not a single baseline drifting.")
_OLD_MD2 = ("Silently dropping a scoring control is the class of lie this harness exists to "
            "prevent: the report would publish a `precision@coverage` figure for a policy "
            "that never ran.")
_OLD_T1 = ("That is a scoring control: an answer below the threshold is not the same decision "
           "as one above. A `laya-evals run` that could not pass it through had no way to "
           "measure `precision@coverage` at any threshold")
_OLD_T2 = ("Silently dropping a scoring control would report `precision@coverage` for a "
           "policy that never ran.")
_OLD_T3 = ("a `--min-confidence` run that silently dropped the argument would publish a "
           "`precision@coverage` figure for a policy that never ran")

_OLD_EVALS_SITES = (_OLD_E1, _OLD_E2, _OLD_E3, _OLD_E4, _OLD_E5,
                    _OLD_MD1, _OLD_MD2, _OLD_T1, _OLD_T2, _OLD_T3)


def test_abstention_wording_bans_have_teeth():
    """Every ban fires on a pre-fix site, and every pre-fix site is caught by at least one ban."""
    matched = [[bool(rule.search(old)) for rule in _EVALS_BANS] for old in _OLD_EVALS_SITES]
    for i, old in enumerate(_OLD_EVALS_SITES):
        assert any(matched[i]), "site %d's pre-fix wording is not caught by any ban:\n%s" % (i, old)
    for j, rule in enumerate(_EVALS_BANS):
        assert any(matched[i][j] for i in range(len(_OLD_EVALS_SITES))), \
            "ban /%s/ never fires on any pre-fix wording: a rule without a witness" % rule.pattern


def test_abstention_wording_is_gone_from_every_site():
    """None of the scoring-change claims may appear in the sources they were removed from."""
    for path in (_EVALS_CLI_PATH, _EVALS_PY_PATH, _EVALS_MD_PATH,
                 _TEST_EVALS_PATH, _TEST_EVALS_API_PATH):
        src = _read_text(path)
        for rule in _EVALS_BANS:
            assert not rule.search(src), \
                "%s still carries the scoring-change claim: /%s/" % (path, rule.pattern)


def test_abstention_wording_names_the_state_fields():
    """Each site individually must name what the gate writes, not just somewhere in the file.

    Per-docstring rather than per-file, so a mutation that drops `low_confidence` from the
    `evaluate()` docstring alone is caught even though the sibling `_takes_min_confidence`
    docstring above it still names the field.
    """
    help_text = _cli_help_of(_EVALS_CLI_PATH, "--min-confidence")
    assert help_text, "no `help=` string on the --min-confidence argument in laya/evals_cli.py"
    assert _SAYS_LOW_CONFIDENCE.search(help_text), \
        "evals_cli's --min-confidence help does not name the `low_confidence` field"
    assert _SAYS_ABSTENTION_FIELD.search(help_text), \
        "evals_cli's --min-confidence help does not name `abstention: \"abstained\"`"

    for func_name in ("_takes_min_confidence", "evaluate"):
        doc = _docstring_of(_EVALS_PY_PATH, func_name)
        assert doc, "no docstring on laya/evals.py::%s" % func_name
        assert _SAYS_LOW_CONFIDENCE.search(doc), \
            "laya/evals.py::%s docstring does not name the `low_confidence` field" % func_name
        assert _SAYS_ABSTENTION_FIELD.search(doc), \
            "laya/evals.py::%s docstring does not name `abstention: \"abstained\"`" % func_name

    md = _read_text(_EVALS_MD_PATH)
    assert _SAYS_LOW_CONFIDENCE.search(md), \
        "docs/evals.md does not name the `low_confidence` field"
    assert _SAYS_ABSTENTION_FIELD.search(md), \
        "docs/evals.md does not name `abstention: \"abstained\"`"


def test_gate_writes_the_fields_the_docs_name():
    """The witness that ties the wording above to `apply_confidence_gate`'s actual effect.

    Drive the gate at 0.7 over one high-confidence and one low-confidence `choice` answer, both
    carrying `choice: "keep"`. The low one must come back flagged; both must keep their raw
    argmax in `answer["choice"]`. If the gate ever starts overwriting the value, this assertion
    breaks and the wording claim breaks with it.
    """
    results = [{"answers": {
        "q_high": {"type": "choice", "choice": "keep",
                    "answer_confidence": 0.90,
                    "probabilities": {"keep": 0.90, "drop": 0.10}},
        "q_low": {"type": "choice", "choice": "keep",
                   "answer_confidence": 0.40,
                   "probabilities": {"keep": 0.40, "drop": 0.60}},
    }}]
    apply_confidence_gate(results, 0.7)
    high = results[0]["answers"]["q_high"]
    low = results[0]["answers"]["q_low"]
    check("evals-wording/high clears the gate", high["abstention"], GATE_PASSED)
    check("evals-wording/low falls below", low["abstention"], GATE_ABSTAINED)
    check_true("evals-wording/low is flagged", low.get("low_confidence") is True)
    check_true("evals-wording/high is not flagged", "low_confidence" not in high)
    # The claim the wording makes about the gate: the value key stays at the raw argmax.
    check("evals-wording/high's choice is unchanged", high["choice"], "keep")
    check("evals-wording/low's choice is unchanged even when abstained", low["choice"], "keep")
    # Both thresholds echoed onto the answer, so `abstention_threshold` matches the wording.
    check("evals-wording/high echoes the threshold", high["abstention_threshold"], 0.7)
    check("evals-wording/low echoes the threshold", low["abstention_threshold"], 0.7)


for _fn in (test_abstention_wording_bans_have_teeth,
            test_abstention_wording_is_gone_from_every_site,
            test_abstention_wording_names_the_state_fields,
            test_gate_writes_the_fields_the_docs_name):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("evals-abstention/%s: %s" % (_fn.__name__, e))
    except Exception as e:
        FAIL.append("evals-abstention/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("evals-abstention/%s" % _fn.__name__)

# ------------------------------------------ example 26's banner must count the primitives triage_questions() returns
# `examples/26_presets_triage.py` opened with "`triage_questions()` is a ready-made schema for
# inbound support: five questions ... a multi-way choice, two yes/no probabilities, and an ordinal
# score". `laya.triage_questions()` ships three `noul` fields -- `is_urgent`, `refund_requested`,
# `churn_risk` -- plus the choice and the score, so the banner's own arithmetic read 1 + 2 + 1 = 4
# against its stated "five questions", and the example's own print block already prints all three
# `noul` values two paragraphs later. A reader counting primitives against the schema could not
# reconcile them.
EXAMPLE_26 = os.path.join(ROOT, "examples", "26_presets_triage.py")
OLD_BANNER_26 = '''
    `triage_questions()` is a ready-made schema for inbound support: five questions, one
    forward pass. It mixes all three primitives -- a multi-way choice, two yes/no
    probabilities, and an ordinal score -- so a single call fills a whole triage record.
'''
_BAN_TWO_YESNO = re.compile(r"\btwo\s+yes/no\s+probabilit", re.I)
# The banner's own stated counts. `_WORDS` maps each spelling that could stand for one primitive.
_STATED_TOTAL_26 = re.compile(r"\b(one|two|three|four|five|six)\s+questions", re.I)
_STATED_YESNO_26 = re.compile(r"\b(one|two|three|four|five|six)\s+yes/no\s+probabilit", re.I)
_STATED_CHOICE_26 = re.compile(r"\b(a|one)\s+multi-way\s+choice", re.I)
_STATED_SCORE_26 = re.compile(r"\b(a|an)\s+ordinal\s+score", re.I)
_WORDS = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _src26():
    with open(EXAMPLE_26, encoding="utf-8") as fh:
        return fh.read()


def _triage_type_counts():
    from laya.presets import triage_questions
    q = triage_questions()
    by_type = {}
    for spec in q.values():
        by_type[spec["type"]] = by_type.get(spec["type"], 0) + 1
    return len(q), by_type


def test_triage_preset_is_one_choice_three_noul_one_score():
    # The gate's positive anchor: this is the schema the banner must describe. If `triage_questions()`
    # ever moves, the banner test fails here first, and the drift is loud rather than silent.
    n_total, by_type = _triage_type_counts()
    assert n_total == 5, "triage_questions() no longer returns five fields: %r" % (by_type,)
    assert by_type == {"choice": 1, "noul": 3, "score": 1}, \
        "triage_questions()'s type mix moved: %r" % (by_type,)


def test_example_26_banner_counts_match_the_preset():
    n_total, by_type = _triage_type_counts()
    src = _src26()
    m_total = _STATED_TOTAL_26.search(src)
    m_yesno = _STATED_YESNO_26.search(src)
    m_choice = _STATED_CHOICE_26.search(src)
    m_score = _STATED_SCORE_26.search(src)
    assert m_total and m_yesno and m_choice and m_score, (
        "the banner no longer spells out its primitive counts; the gate needs them readable")
    assert _WORDS[m_total.group(1).lower()] == n_total, \
        "banner says %r questions, triage_questions() returns %d" % (m_total.group(1), n_total)
    assert _WORDS[m_yesno.group(1).lower()] == by_type["noul"], \
        "banner says %r yes/no probabilities, triage_questions() returns %d noul fields" % \
        (m_yesno.group(1), by_type["noul"])
    assert _WORDS[m_choice.group(1).lower()] == by_type["choice"]
    assert _WORDS[m_score.group(1).lower()] == by_type["score"]
    # Parts must add up to the whole the same paragraph asserts.
    parts = (_WORDS[m_choice.group(1).lower()] + _WORDS[m_yesno.group(1).lower()]
             + _WORDS[m_score.group(1).lower()])
    assert parts == _WORDS[m_total.group(1).lower()], (
        "banner arithmetic is wrong: %d + %d + %d = %d but it says %d questions" % (
            _WORDS[m_choice.group(1).lower()], _WORDS[m_yesno.group(1).lower()],
            _WORDS[m_score.group(1).lower()], parts, _WORDS[m_total.group(1).lower()]))


def test_example_26_drops_the_wrong_yes_no_count():
    # The ban has teeth: it fires on the pre-fix wording shipped on main.
    assert _BAN_TWO_YESNO.search(OLD_BANNER_26), "the ban must fire on the pre-fix wording"
    # ...and every count regex still reads the OLD banner as its pre-fix numbers, so the positive
    # rule was genuinely falsifiable.
    m = _STATED_YESNO_26.search(OLD_BANNER_26)
    assert m is not None and _WORDS[m.group(1).lower()] == 2
    # The current example satisfies the ban.
    assert not _BAN_TWO_YESNO.search(_src26()), "the wrong yes/no count is back in the shipped example"


for _fn in (test_triage_preset_is_one_choice_three_noul_one_score,
            test_example_26_banner_counts_match_the_preset,
            test_example_26_drops_the_wrong_yes_no_count):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("page-26/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("page-26/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("page-26/%s" % _fn.__name__)

# ------------------------------------------- tl_kernels' docstring must name the caller's real padding policy
# `laya/tl_kernels.py`'s module docstring said "M must be a multiple of 16 (the caller pads);
# out-of-bounds rows are predicated by TileLang". The caller pads two dimensions, not one:
# `laya/backends/base.py::bucket_rows` pads the batch dim to the next power of two (1, 2, 4, 8,
# 16, 32, ...) and `bucket_tokens` pads the sequence dim to a 16-token bucket up to
# `DYNAMIC_MAX_L` and a 64-token bucket beyond. M is the flattened `rows * tokens`, so the
# multiple-of-16 came from the token dim, never from the row dim. A reader who took the
# docstring as a row constraint would add `assert M % 16 == 0` at the call site and break
# every batch of 1, 2, 4, or 8 questions -- and the shipped `pad_batch` tests already pin those
# shapes: `tests/test_backends.py:86` "pad_batch/one row stays one row" is (1, 16), and
# `:89` "pad_batch/five rows pad to eight" is (8, 16).
TL_KERNELS = os.path.join(ROOT, "laya", "tl_kernels.py")
OLD_TL_DOCSTRING = (
    "GPU kernels take 16-bit activations (bf16 by default, fp16 with dtype=\"float16\"), accumulate in fp32.  Row count M is a runtime\n"
    "symbol so one compiled kernel serves every batch/sequence bucket; M must be a\n"
    "multiple of 16 (the caller pads); out-of-bounds rows are predicated by TileLang.\n"
    "Use compile_cpu for an explicit fp32 CPU specialization; the GPU defaults are unchanged."
)
_BAN_ROW_MULT_16 = re.compile(r"M must be a\s*\n?\s*multiple of 16 \(the caller pads\)", re.I)
_BAN_MULT_16_FROM_ROW = re.compile(r"multiple[- ]of[- ]16[^.\n]{0,80}from the row dim", re.I)
_NAMED_POWER_OF_TWO = re.compile(r"power of two", re.I)
_NAMED_16_TOKEN_BUCKET = re.compile(r"16[- ]token bucket", re.I)
_NAMED_NOT_ROW_DIM = re.compile(r"not the row dim", re.I)
_NAMED_BUCKET_ROWS = re.compile(r"\bbucket_rows\b")
_NAMED_BUCKET_TOKENS = re.compile(r"\bbucket_tokens\b")


def _tl_docstring():
    """Return `laya/tl_kernels.py`'s module docstring via `ast`, so the gate reads what a
    reader reads -- not whatever happens to appear in a comment."""
    with open(TL_KERNELS, encoding="utf-8") as fh:
        src = fh.read()
    return ast.get_docstring(ast.parse(src)) or ""


def test_tl_kernels_drops_the_wrong_row_claim():
    # The ban fires on the pre-fix docstring...
    assert _BAN_ROW_MULT_16.search(OLD_TL_DOCSTRING), "the ban must fire on the pre-fix wording"
    # ...and does not fire on the shipped one.
    doc = _tl_docstring()
    assert doc, "the tl_kernels module docstring was removed -- the gate has nothing to read"
    assert not _BAN_ROW_MULT_16.search(doc), (
        "`laya/tl_kernels.py` again says M must be a multiple of 16 because the caller pads; "
        "the caller pads rows to a power of two (1, 2, 4, 8, ...). A reader who trusts the "
        "claim asserts `M % 16 == 0` at the call site and breaks every small batch.")
    # The second ban catches the mis-attribution (claiming the 16-multiple comes from rows).
    assert not _BAN_MULT_16_FROM_ROW.search(doc), (
        "the docstring attributes the multiple-of-16 to the row dim; `bucket_rows` pads to a "
        "power of two, so the 16-multiple is a token-dim artefact.")


def test_tl_kernels_names_the_real_padding_policy():
    doc = _tl_docstring()
    assert _NAMED_POWER_OF_TWO.search(doc), (
        "the docstring must name the row pad as a power of two, matching `bucket_rows`")
    assert _NAMED_16_TOKEN_BUCKET.search(doc), (
        "the docstring must name the token pad as a 16-token bucket, matching `bucket_tokens`")
    assert _NAMED_BUCKET_ROWS.search(doc), (
        "the docstring must attribute the row pad to `bucket_rows`, the function that does it")
    assert _NAMED_BUCKET_TOKENS.search(doc), (
        "the docstring must attribute the token pad to `bucket_tokens`, the function that does it")
    assert _NAMED_NOT_ROW_DIM.search(doc), (
        "the docstring must name the row dim as the wrong source of the 16-multiple, so a "
        "reader cannot re-attribute it back")


def test_bucket_rows_stays_a_power_of_two_below_16():
    # The witness: rows below 16 really do stay at 1, 2, 4, 8, so the pre-fix claim
    # ("M must be a multiple of 16 (the caller pads)") was false for them. If `bucket_rows`
    # ever starts rounding to 16, the docstring fix needs review and this test says so.
    from laya.backends.base import bucket_rows
    for n, want in ((1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (7, 8), (8, 8), (9, 16),
                    (15, 16), (17, 32), (24, 32), (33, 64)):
        got = bucket_rows(n)
        assert got == want, (
            "bucket_rows(%d) returned %d, expected %d -- the docstring's row contract needs "
            "review" % (n, got, want))
        assert got & (got - 1) == 0, (
            "bucket_rows(%d) = %d is not a power of two; the docstring's `power of two` claim "
            "is stale" % (n, got))
    # Rows below 16 must stay below 16 -- that is the case the pre-fix docstring got wrong.
    for n in (1, 2, 4, 8):
        assert bucket_rows(n) == n, (
            "bucket_rows(%d) = %d; a caller that trusts the old multiple-of-16 claim would "
            "expect 16 here" % (n, bucket_rows(n)))


for _fn in (test_tl_kernels_drops_the_wrong_row_claim,
            test_tl_kernels_names_the_real_padding_policy,
            test_bucket_rows_stays_a_power_of_two_below_16):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("tl-kernels/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("tl-kernels/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("tl-kernels/%s" % _fn.__name__)

# --------------------------------------- DecisionResult's docstring must split confidence by question type
# `laya/structured.py`'s `DecisionResult` closed with "confidence keeps the **normalized-entropy
# value** it has always had, because that is a different quantity on a scale that depends on the
# label count". `Agent._decode_answers` (`laya/agent.py:1349-1402`) uses two different formulas:
#   * `choice` and `score`  ->  `confidence_from_probs(p, k)` = `1 - H(p) / log(k)`
#   * `noul`                ->  `max(p_true, 1 - p_true)`  (its own comment: "identical here:
#                               over two options `max(p_true, 1 - p_true)` is `max(p)`")
# The `noul` branch is deliberately not entropy: with k=2, entropy 1 - H/log(2) hits its floor
# (0.0) exactly when the answer is a coin flip, which is inverted from what a boolean reporter
# wants. `structured._details` (`:303`) copies `answer["confidence"]` verbatim, so
# `DecisionResult.confidence` is per-type -- the docstring's blanket "normalized-entropy"
# claim is false for every boolean field a caller projects. Same defect class as examples 18/40
# (#972, #973) but in the public docstring rather than in an example banner.
STRUCTURED = os.path.join(ROOT, "laya", "structured.py")
OLD_DECISIONRESULT_DOC = """The detailed result of `decide(..., return_details=True)`.

    `confidence` keeps the normalized-entropy value it has always had, because that is a
    different quantity on a scale that depends on the label count. A field that reported no
    usable `answer_confidence` maps to `None`, which is not the same as a reported `0.0`.
"""
_BAN_ENTROPY_UNIFORM = re.compile(r"`confidence` keeps the normalized-entropy value", re.I)
_NAMED_MAX_PT = re.compile(r"max\(p_true,\s*1 - p_true\)")
_NAMED_ENTROPY_FORMULA = re.compile(r"1 - H\(p\) / log\(k\)")
_NAMED_CHOICE_SCORE = re.compile(r"`choice` and\s+`score`", re.I)
_NAMED_NOUL = re.compile(r"for\s+`noul`", re.I)
_NAMED_NOT_ENTROPY = re.compile(r"not an\s+entropy", re.I)


def _decisionresult_doc():
    """Return the `DecisionResult` class docstring via `ast`, so the gate reads what a caller
    reads at `help(laya.structured.DecisionResult)` time."""
    with open(STRUCTURED, encoding="utf-8") as fh:
        src = fh.read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "DecisionResult":
            return ast.get_docstring(node) or ""
    return ""


def test_decisionresult_drops_the_uniform_entropy_claim():
    assert _BAN_ENTROPY_UNIFORM.search(OLD_DECISIONRESULT_DOC), (
        "the ban must fire on the pre-fix wording")
    doc = _decisionresult_doc()
    assert doc, "the DecisionResult class docstring was removed -- the gate has nothing to read"
    assert not _BAN_ENTROPY_UNIFORM.search(doc), (
        "`laya/structured.py`'s `DecisionResult` again claims `confidence` is uniformly the "
        "normalized entropy. `noul` fields carry `max(p_true, 1 - p_true)`, a probability, "
        "which is not an entropy and does not scale with the label count.")


def test_decisionresult_names_both_confidence_formulas():
    doc = _decisionresult_doc()
    assert _NAMED_ENTROPY_FORMULA.search(doc), (
        "the docstring must name the entropy formula `1 - H(p) / log(k)`")
    assert _NAMED_MAX_PT.search(doc), (
        "the docstring must name `max(p_true, 1 - p_true)` for the noul branch")
    assert _NAMED_CHOICE_SCORE.search(doc), (
        "the docstring must attribute the entropy formula specifically to `choice` and `score`")
    assert _NAMED_NOUL.search(doc), (
        "the docstring must attribute the probability formula specifically to `noul`")
    assert _NAMED_NOT_ENTROPY.search(doc), (
        "the docstring must state that the `noul` value is not an entropy, so a caller cannot "
        "silently re-collapse the two again")


def test_noul_and_entropy_confidence_are_different_quantities():
    # The witness: on the same 2-class distribution `confidence_from_probs` and the shipped
    # `noul` formula disagree sharply, so a docstring that says "confidence is normalized
    # entropy" and cites label count is wrong by construction. If they ever converge (say,
    # if `noul` switched to entropy), the docstring fix would need review and this test
    # tells the reader.
    uniform = np.array([0.5, 0.5])
    entropy_val = confidence_from_probs(uniform, 2)
    noul_val = max(float(uniform[1]), 1.0 - float(uniform[1]))
    assert entropy_val == 0.0 and noul_val == 0.5, (
        "the two formulas must disagree at the uniform 2-class: entropy=%r, noul=%r; if "
        "they've converged, the docstring's per-type split is stale" % (entropy_val, noul_val))
    # And on a peaked 2-class, entropy-confidence stays near 1 but is not the reported probability.
    peak = np.array([0.01, 0.99])
    entropy_peak = confidence_from_probs(peak, 2)
    noul_peak = max(float(peak[1]), 1.0 - float(peak[1]))
    assert abs(noul_peak - 0.99) < 1e-6 and abs(entropy_peak - 0.9192) < 1e-3, (
        "at (0.01, 0.99): entropy=%r (expect ~0.919) noul=%r (expect 0.99) -- if the values "
        "moved, the docstring's formula names need re-checking" % (entropy_peak, noul_peak))


for _fn in (test_decisionresult_drops_the_uniform_entropy_claim,
            test_decisionresult_names_both_confidence_formulas,
            test_noul_and_entropy_confidence_are_different_quantities):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("structured-doc/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("structured-doc/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("structured-doc/%s" % _fn.__name__)

# --------------------------------------- `docs/questions-and-answers.md` must count what `laya` exports
# The Presets section opens with "Three ready-made question sets" and its example imports three
# names, but `laya/__init__.py` re-exports five `*_questions` helpers -- `triage_questions`,
# `email_questions`, `guard_questions`, `moderation_questions`, `router_questions` -- and every one
# is in `laya.__all__`. A caller who reads "three" and picks from the shipped list misses two of
# the five; the doc drifts silently as presets get added because nothing ties the number to the
# exports. This gate reads the actual `laya.__all__` inventory so any future preset -- e.g. the
# Swedish sets on #729 -- fails the doc until the doc is updated alongside the export.
DOC_QA = os.path.join(ROOT, "docs", "questions-and-answers.md")
OLD_PRESET_INTRO = "Three ready-made question sets, so the common cases do not need hand-written criteria:"
_NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def _preset_exports():
    """The names `laya` publicly exports that end in `_questions`.

    Read via `laya.__all__`, not `dir(laya)` -- a helper that exists but is not in the export
    tuple is an internal name and does not need a doc mention.
    """
    import laya
    return sorted(n for n in laya.__all__ if n.endswith("_questions"))


def _doc_preset_section():
    with open(DOC_QA, encoding="utf-8") as fh:
        src = fh.read()
    idx = src.find("## Presets")
    if idx == -1:
        return ""
    nxt = src.find("\n## ", idx + len("## Presets"))
    return src[idx:] if nxt == -1 else src[idx:nxt]


def _doc_stated_count_word(section):
    m = re.search(r"\b(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s+ready-made\s+question", section, re.I)
    return m.group(1) if m else None


def test_presets_page_drops_the_three_set_claim():
    # The witness: the pre-fix wording is banned, so if the doc ever regresses to "Three"
    # while the exports are still five, the gate fires.
    assert "Three ready-made" in OLD_PRESET_INTRO, (
        "the ban's literal must itself carry the wrong count, otherwise the ban is vacuous")
    section = _doc_preset_section()
    assert section, "the Presets section was removed -- the gate has nothing to read"
    word = _doc_stated_count_word(section)
    actual = len(_preset_exports())
    expected_word = _NUMBER_WORDS.get(actual)
    assert word == expected_word, (
        "docs/questions-and-answers.md's Presets section says %r but `laya.__all__` exports "
        "%d `*_questions` helpers (%s). Update the doc's number word to %r (or, if it has "
        "grown past the mapped words, extend `_NUMBER_WORDS`)." %
        (word, actual, ", ".join(_preset_exports()), expected_word))


def test_presets_page_names_every_exported_preset():
    section = _doc_preset_section()
    exports = _preset_exports()
    assert exports, "`laya.__all__` has no `*_questions` names -- either the exports moved or " \
                    "this gate is checking the wrong thing"
    missing = [n for n in exports if ("`" + n + "`") not in section]
    assert not missing, (
        "docs/questions-and-answers.md's Presets section does not name these exported helpers "
        "in backticks: %s. A caller who reads only this page cannot discover them." % missing)


def test_presets_page_no_three_claim_witness():
    # Direct ban on the pre-fix wording, independent of the count-regex, so a future PR that
    # quietly reverts to "Three" fails with an explicit "this phrase is banned" message rather
    # than a numeric-mismatch one.
    section = _doc_preset_section()
    assert "Three ready-made question sets" not in section, (
        "`docs/questions-and-answers.md` again says \"Three ready-made question sets\". As of "
        "this PR `laya.__all__` exports five `*_questions` helpers; if the number has since "
        "changed, update the doc's number word AND this ban's literal together.")


for _fn in (test_presets_page_drops_the_three_set_claim,
            test_presets_page_names_every_exported_preset,
            test_presets_page_no_three_claim_witness):
    try:
        _fn()
    except AssertionError as e:
        FAIL.append("docs-presets/%s: %s" % (_fn.__name__, e))
    except Exception as e:                      # a crash is a failure, never a silent pass
        FAIL.append("docs-presets/%s raised %s: %s" % (_fn.__name__, type(e).__name__, e))
    else:
        PASS.append("docs-presets/%s" % _fn.__name__)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all confidence tests passed")
sys.exit(1 if FAIL else 0)
