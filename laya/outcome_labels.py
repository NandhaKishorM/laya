"""Weak labels from logged decisions: stability-filtered self-labels and review queues.

An outcome log records what the model answered, not whether it was right. This module
derives the first labels without a human in the loop, from signals the log already holds:

* `export_stability_labels` -- a decision that answers identically across
  meaning-preserving perturbations is likely correct. When an annotating hook (such as
  `OptionStabilityHook`, installed *before* the log hook so its `reliability` block is in
  the stored answer) reports `stability` at or above `min_stability`, the logged answer
  itself becomes a weak label with the stability as its weight. The default requires
  unanimity: precision matters more than recall here, because a wrong weak label trains
  the fitter to trust a wrong answer.
* `export_review_queue` -- the complement: decisions the gate abstained on, could not
  evaluate, or answered with low confidence. A worklist for human review, not labels.

Neither function overwrites an existing outcome: a human or shadow-eval label always
wins over a weak one, and re-running an export only labels new rows.
"""

import json
import math

ABSTAINED = "abstained"
UNEVALUATED = "unevaluated"


def _existing_outcome_cells(hook) -> set:
    return {(row["run_id"], row["qid"]) for row in hook.outcomes()}


def export_stability_labels(hook, *, min_stability: float = 1.0,
                            source: str = "stability") -> int:
    """Write weak labels for unanimously stable decisions. Returns rows written.

    Reads `hook.decisions()`, and for every answered row whose stored answer carries a
    `reliability` block with a numeric `stability` in [0, 1] at or above `min_stability`,
    records the logged answer as the label with that stability as its weight. Rows that
    already have an outcome -- from any source -- are left alone. Anything unparseable,
    out of range, or missing a usable answer value is skipped, never guessed.
    """
    if isinstance(min_stability, bool) or not isinstance(min_stability, (int, float)) \
            or not math.isfinite(min_stability):
        raise ValueError("min_stability must be a finite number, got %r" % (min_stability,))
    existing = _existing_outcome_cells(hook)
    written = 0
    for row in hook.decisions():
        key = (row.get("run_id"), row.get("qid"))
        if row.get("qid") is None or key in existing:
            continue
        try:
            answer = json.loads(row.get("answer_json")) if row.get("answer_json") else None
        except (TypeError, ValueError):
            continue
        if not isinstance(answer, dict):
            continue
        reliability = answer.get("reliability")
        if not isinstance(reliability, dict):
            continue
        stability = reliability.get("stability")
        if isinstance(stability, bool) or not isinstance(stability, (int, float)) \
                or not 0 <= stability <= 1 or stability < min_stability:
            continue
        value = row.get("value_text")
        if value is None:
            value = row.get("value_real")
        if value is None:
            continue
        hook.record_outcome(row["run_id"], row["qid"], {"value": value}, source,
                            weight=float(stability))
        existing.add(key)
        written += 1
    return written


def export_review_queue(hook, *, min_confidence: float = None) -> list:
    """Decisions a human should look at, oldest first, as plain dicts.

    A row qualifies when the abstention gate did not pass it (`abstained` or
    `unevaluated`), or -- when `min_confidence` is given -- when its confidence falls
    below it. Rows that already carry an outcome are done being reviewed and are
    excluded. Each entry names its `reason` so the reviewer knows what question to ask:
    `abstained` (the model declined), `unevaluated` (no usable confidence to gate on),
    or `low_confidence`.
    """
    if min_confidence is not None and (
            isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float))
            or not math.isfinite(min_confidence)):
        raise ValueError("min_confidence must be a finite number or None, got %r"
                         % (min_confidence,))
    existing = _existing_outcome_cells(hook)
    queue = []
    for row in hook.decisions():
        if row.get("qid") is None or (row.get("run_id"), row.get("qid")) in existing:
            continue
        abstention = row.get("abstention")
        confidence = row.get("confidence")
        if abstention == ABSTAINED:
            reason = "abstained"
        elif abstention == UNEVALUATED:
            reason = "unevaluated"
        elif min_confidence is not None and isinstance(confidence, (int, float)) \
                and not isinstance(confidence, bool) and confidence < min_confidence:
            reason = "low_confidence"
        else:
            continue
        queue.append({"run_id": row["run_id"], "qid": row["qid"], "reason": reason,
                      "confidence": confidence, "abstention": abstention,
                      "model": row.get("model"), "value_text": row.get("value_text"),
                      "value_real": row.get("value_real")})
    return queue
