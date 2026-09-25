"""Confidence validation, calibrated thresholds, and abstention gating helpers (#361).

Pure Python: safe to import without PyTorch so that Router and structured
decisions stay lightweight and free of torch import overhead.
"""
import math
from typing import Any, Dict, List


def check_min_confidence(v: Any) -> float:
    """Validate opt-in abstention threshold `min_confidence` (#361).

    Must be a real number in [0.0, 1.0]. Booleans are rejected (even though `isinstance(True, int)`).
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError("min_confidence must be a float in [0.0, 1.0], got %r" % (v,))
    return float(v)


def flag_low_confidence(results: List[Dict[str, Any]], min_confidence: float) -> None:
    """Opt-in abstention marker (#361): flag answers whose confidence falls below `min_confidence`.

    Reads `answer_confidence` (the calibrated max(p) confidence, invariant to label count k),
    falling back to `confidence` if `answer_confidence` is absent.
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
            conf = a.get("answer_confidence")
            if conf is None:
                conf = a.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf) and conf < min_confidence:
                a["low_confidence"] = True
