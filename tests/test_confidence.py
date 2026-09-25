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

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all confidence tests passed")
sys.exit(1 if FAIL else 0)
