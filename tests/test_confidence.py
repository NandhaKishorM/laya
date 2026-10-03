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

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all confidence tests passed")
sys.exit(1 if FAIL else 0)
