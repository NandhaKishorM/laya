"""Confidence validation, calibrated thresholds, and abstention gating helpers (#361).

Pure Python: safe to import without PyTorch so that Router and structured
decisions stay lightweight and free of torch import overhead.
"""
import math
from typing import Any, Dict, List, Optional


def answer_confidence_value(answer: Dict[str, Any]) -> Optional[float]:
    """The `answer_confidence` of `answer`, or None if it did not report a usable one.

    `answer_confidence` is `max(p)` -- the probability mass on the answer being reported. It is
    the quantity temperature scaling fits, the quantity every calibration figure in this
    repository is computed on, and the quantity the `min_confidence` gate is defined against
    (#361). That makes it the right number for a threshold a caller compares against; it is
    not on its own a claim that the number is right. Reading it as "about c of the answers
    returned at c are correct" holds only after temperatures have been fitted and validated on
    held-out data for that checkpoint and question shape -- the shipped checkpoints are
    over-confident as shipped, and `laya-multilingual` ships with no fitted temperatures at all
    (README, Calibration). `common.answer_confidence` states the quantity; this states when it
    carries that property.

    `confidence` is a different quantity on a different scale -- normalized entropy, which
    `common.confidence_from_probs` calls "not calibrated" and #394 says does not transfer across
    option counts. So this deliberately does **not** fall back to it: reporting the entropy
    number under the `answer_confidence` name would be worse than reporting nothing, because the
    name is what a caller filters on.

    A `bool` is not a confidence and a NaN is not a decision, so both give None rather than 0.0.
    """
    conf = answer.get("answer_confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf):
        return float(conf)
    return None


def _gate_confidence(answer: Dict[str, Any]) -> Optional[float]:
    """The number the abstention gate compares against `min_confidence`, or None if there is none.

    `answer_confidence` first, falling back to the entropy `confidence` so an answer that carries
    only the older field is still gated rather than silently passed.

    None means *no usable number*, never an unusable one. An earlier version returned whatever
    `confidence` held when the validity test failed, so an answer with no `answer_confidence` and a
    NaN `confidence` returned NaN. `flag_low_confidence` was indifferent to that -- `NaN < x` is
    False and `None is not None` is also False, so it flagged either way -- but a caller that
    *reads* the number could not tell "nothing to gate on" from "a gate ran on a NaN", and
    :func:`apply_confidence_gate` reports those two differently on purpose.
    """
    conf = answer_confidence_value(answer)
    if conf is not None:
        return conf
    fallback = answer.get("confidence")
    if isinstance(fallback, (int, float)) and not isinstance(fallback, bool) and math.isfinite(fallback):
        return float(fallback)
    return None


def check_min_confidence(v: Any) -> float:
    """Validate opt-in abstention threshold `min_confidence` (#361).

    Must be a real number in [0.0, 1.0]. Booleans are rejected (even though `isinstance(True, int)`).
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError("min_confidence must be a float in [0.0, 1.0], got %r" % (v,))
    return float(v)


def flag_low_confidence(results: List[Dict[str, Any]], min_confidence: float) -> None:
    """Opt-in abstention marker (#361): flag answers whose confidence falls below `min_confidence`.

    Reads `answer_confidence` (`max(p)`, the quantity the calibration figures describe and the one
    that does not drift with the number of options), falling back to `confidence` if
    `answer_confidence` is absent.
    The raw answer and confidence stay intact; `low_confidence: True` is added.
    """
    if min_confidence == 0.0:
        return
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            conf = _gate_confidence(a)
            if conf is not None and conf < min_confidence:
                a["low_confidence"] = True


#: The states :func:`apply_confidence_gate` reports, and the only ones. There is deliberately no
#: "no gate was configured" member: when `min_confidence` is not set the function writes nothing at
#: all, so a caller tells "this run had no gate" from "this answer cleared the gate" by whether
#: `abstention` is present. A sentinel for the unconfigured case would put that same information
#: back into the payload for every caller, which is the thing the gate is meant to avoid.
GATE_PASSED = "passed"            # a gate ran and this answer's confidence cleared it
GATE_ABSTAINED = "abstained"      # a gate ran and this answer's confidence fell below it
GATE_UNEVALUATED = "unevaluated"  # a gate ran and this answer carried no usable confidence

GATE_STATES = (GATE_PASSED, GATE_ABSTAINED, GATE_UNEVALUATED)


def apply_confidence_gate(results: List[Dict[str, Any]], min_confidence: Optional[float] = None) -> None:
    """Report the confidence gate's state, on the answers a gate was actually applied to.

    A gate is a policy, and a policy whose application cannot be observed is not one. With
    `min_confidence` set, this writes ``abstention`` -- one of :data:`GATE_STATES` -- onto every
    answer, plus ``abstention_threshold``, so a caller can answer three questions it otherwise
    cannot:

    * what fraction of decisions abstained, rather than inferring it from whether
      ``low_confidence`` happened to be set;
    * how many answers the gate could not decide on, which a boolean cannot express at all;
    * what threshold produced these results -- ``flag_low_confidence`` consumes the threshold and
      drops it, so without this a batch run with per-class thresholds cannot be re-split.

    ``GATE_UNEVALUATED`` is the case a boolean cannot express: the gate ran and the answer carried
    no usable confidence, so the gate could not decide. Reporting that as a pass is the same lie
    as reporting it as a flag.

    **With `min_confidence` unset, this writes nothing.** No ``abstention``, no
    ``abstention_threshold``, no flag. That is the whole contract: an ungated call returns exactly
    the payload it returned before, and the presence of the field -- not a fourth value read out
    of it -- is what tells a caller the gate ran. Call it unconditionally, once per call, in place
    of an `if min_confidence is not None:` guard: that guard is what leaves a path reporting
    nothing at all, which is the state this function exists to distinguish.

    The flag itself stays :func:`flag_low_confidence`'s -- this delegates rather than
    re-implementing the rule, so the boolean and the reported state cannot drift apart.

    A `min_confidence` of exactly ``0.0`` *was* set, so states are reported, and
    :func:`flag_low_confidence` treats ``0.0`` as a no-op because nothing can fall below it. Every
    answer carrying a usable confidence therefore reads ``passed``, and the threshold echo is what
    distinguishes that from a real pass at a real threshold.
    """
    if min_confidence is None:
        return
    flag_low_confidence(results, min_confidence)
    for res in results:
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            continue
        for a in answers.values():
            if not isinstance(a, dict):
                continue
            if a.get("low_confidence"):
                a["abstention"] = GATE_ABSTAINED
            elif _gate_confidence(a) is None:
                a["abstention"] = GATE_UNEVALUATED
            else:
                a["abstention"] = GATE_PASSED
            a["abstention_threshold"] = float(min_confidence)
