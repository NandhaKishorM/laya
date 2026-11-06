"""From joined outcome labels to a calibration payload, without touching serving.

Slice 3 of outcome-logged calibration closes the loop the log opened:

1. `build_fit_pairs` joins the `outcomes` table with operator-supplied items
   (`{"state": ..., "questions": {...}}`, matched by state hash) and turns each label
   into the sum-to-1 target vector `laya.calibrate` fits on. Pure Python, no model.
2. The operator runs `records_from_labeled(agent, pairs)` for one line per model group
   -- the only step that needs the checkpoint -- exactly as `Agent.fit_temperatures`
   does in-process.
3. `fit_payload_from_records` fits temperatures and abstention thresholds on the
   records and returns the `calibration_payload` JSON a deployment applies at load
   with `Agent.load_calibration`.

The log never stores raw states, so step 1 cannot replay them: the deployment owns its
traffic and re-supplies the items. The stored state hash is the join key.
"""

import json
import math
import os
import tempfile

from .outcome_log import OutcomeLogHook, _canonical_state_hash

#: Outcome sources `build_fit_pairs` accepts, and what each label shape means. A scalar is
#: the asserted value directly; `{"value": ...}` is what `export_stability_labels` writes;
#: `{"correct": true}` adopts the logged answer (and `false` is skipped -- it says what
#: was wrong, not what was right, so no target can be built from it).
_LABEL_SKIPS = ("no_decision_row", "no_matching_item", "label_unusable",
                "weight_below_floor", "model_filtered", "unknown_qtype")


def _label_value(label):
    """The asserted value inside a label, or None when no target can be built from it."""
    if isinstance(label, dict):
        if "value" in label:
            return label["value"]
        if label.get("correct") is True:
            return True  # sentinel: adopt the logged answer below
        return None
    return label


def _choice_target(qdef, value):
    from .shortlist import _criteria_items

    labels = [key for key, _ in _criteria_items(qdef.get("criteria"))]
    if isinstance(value, bool) or value not in labels:
        return None
    return [1.0 if key == value else 0.0 for key in labels]


def _score_target(qdef, value):
    criteria = qdef.get("criteria")
    if not isinstance(criteria, list):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        index = value
    elif isinstance(value, str):
        if value not in criteria:
            return None
        index = criteria.index(value)
    else:
        return None
    if not 0 <= index < len(criteria):
        return None
    return [1.0 if i == index else 0.0 for i in range(len(criteria))]


def _noul_target(value):
    if isinstance(value, bool):
        return [0.0, 1.0] if value else [1.0, 0.0]
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and 0.0 <= value <= 1.0:
        return [1.0 - float(value), float(value)]
    return None


def build_fit_pairs(hook, items, *, source=None, min_weight=0.0, models=None):
    """Join outcomes with operator items into per-model `records_from_labeled` input.

    `hook` is an `OutcomeLogHook` (or a path to its database). `items` is an iterable
    of `{"state": ..., "questions": {...}}` mappings -- the deployment's own copy of
    what it asked. Each outcome is matched to the logged decision for its
    `(run_id, qid)` (for the state hash, the model, and -- for `{"correct": true}`
    labels -- the logged answer), and to the item with the same state hash holding
    that question (for the criteria the target vector is built against).

    `source` keeps only one provenance (a string) or several (a collection); `None`
    keeps every source, with the never-overwrite rule from the log doing the
    deduplication: one outcome cell per `(run_id, qid)` exists at most. Outcomes below
    `min_weight` are skipped, so a stricter cut needs no re-export. `models` keeps only
    the named checkpoints -- a fit is per checkpoint, and a mixed log usually wants one
    payload per model group anyway. A `{"correct": true}` label adopts the logged
    answer, which is a valid target for `choice` (the logged choice string names a
    label); for `score` and `noul` the logged value is a continuous prediction, not a
    level, so record the explicit level or probability there instead.

    Returns `(pairs_by_model, report)` where `pairs_by_model` maps a model name to a
    list of `(state, questions, targets)` ready for `records_from_labeled`, and
    `report` counts outcomes seen, pairs built, and every skip by reason.
    """
    if isinstance(hook, str):
        hook = OutcomeLogHook(hook)
    if isinstance(min_weight, bool) or not isinstance(min_weight, (int, float)) \
            or not math.isfinite(min_weight):
        raise ValueError("min_weight must be a finite number, got %r" % (min_weight,))
    if isinstance(source, str):
        sources = {source}
    elif source is None:
        sources = None
    else:
        sources = set(source)
    if isinstance(models, str):
        models = {models}
    elif models is not None:
        models = set(models)

    index = {}
    for pos, item in enumerate(items):
        if not isinstance(item, dict) or "state" not in item \
                or not isinstance(item.get("questions"), dict):
            raise ValueError("item %d must be a {'state': ..., 'questions': {...}} mapping, "
                             "got %r" % (pos, item))
        try:
            digest = _canonical_state_hash(item["state"])
        except Exception:
            continue
        for qid, qdef in item["questions"].items():
            index[(digest, qid)] = (item["state"], qdef)

    decisions = {(row["run_id"], row["qid"]): row for row in hook.decisions()
                 if row.get("qid") is not None}
    pairs = {}
    report = {"n_outcomes": 0, "n_pairs": 0, "skipped": {reason: 0 for reason in _LABEL_SKIPS}}
    for outcome in hook.outcomes():
        if sources is not None and outcome["source"] not in sources:
            continue
        report["n_outcomes"] += 1
        weight = outcome.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) \
                or weight < min_weight:
            report["skipped"]["weight_below_floor"] += 1
            continue
        decision = decisions.get((outcome["run_id"], outcome["qid"]))
        if decision is None:
            report["skipped"]["no_decision_row"] += 1
            continue
        model = decision.get("model") or "unknown"
        if models is not None and model not in models:
            report["skipped"]["model_filtered"] += 1
            continue
        found = index.get((decision.get("state_hash"), outcome["qid"]))
        if found is None:
            report["skipped"]["no_matching_item"] += 1
            continue
        state, qdef = found
        try:
            label = json.loads(outcome["label_json"])
        except (TypeError, ValueError):
            report["skipped"]["label_unusable"] += 1
            continue
        value = _label_value(label)
        if value is True:  # {"correct": true}: adopt what the model actually answered
            value = decision.get("value_text")
            if value is None:
                value = decision.get("value_real")
        qtype = qdef.get("type") if isinstance(qdef, dict) else None
        if qtype == "choice":
            target = _choice_target(qdef, value)
        elif qtype == "score":
            target = _score_target(qdef, value)
        elif qtype == "noul":
            target = _noul_target(value)
        else:
            report["skipped"]["unknown_qtype"] += 1
            continue
        if target is None or sum(target) != 1.0:
            report["skipped"]["label_unusable"] += 1
            continue
        group = pairs.setdefault(model, {})
        entry = group.setdefault(decision.get("state_hash"),
                                 [state, {}, {}])
        entry[1][outcome["qid"]] = qdef
        entry[2][outcome["qid"]] = target
    pairs_by_model = {model: [(state, questions, targets)
                              for _digest, (state, questions, targets) in sorted(group.items())]
                      for model, group in pairs.items()}
    report["n_pairs"] = sum(len(group) for group in pairs_by_model.values())
    return pairs_by_model, report


def fit_payload_from_records(records, *, compute_ece=False, seed=0, target_error=0.10,
                             model_id_or_path=None, subfolder=None, config=None):
    """Fit temperatures and abstention thresholds on records, returning the payload.

    `records` are the `(qtype, logits, target, k)` tuples `records_from_labeled`
    builds -- in tests, synthetic ones in exactly that shape, which is how
    `laya.calibrate` stays weight-free too. Empty input is a `ValueError`, not an
    empty fit: a payload fitted on nothing would install with a straight face.

    Returns `{"payload", "temperature", "abstention_thresholds", "n_records"}` where
    `payload` is the `calibration_payload` JSON a deployment writes to disk and
    applies at load with `Agent.load_calibration`. `abstention_thresholds` may be
    `{}` on a small set -- buckets below the sample floor are omitted, and an empty
    map gates nothing, so pass no gate in that case rather than an empty one.
    """
    from .calibrate import (calibration_payload, fit_abstention_thresholds,
                            fit_temperature_map)

    recs = list(records)
    if not recs:
        raise ValueError("fit_payload_from_records needs at least one record")
    temperature = fit_temperature_map(recs, compute_ece=compute_ece, seed=seed)
    thresholds = fit_abstention_thresholds(recs, temperature["temperature"],
                                           temperature["temperature_by_options"],
                                           target_error=target_error)
    payload = calibration_payload(temperature["temperature"],
                                  temperature["temperature_by_options"],
                                  model_id_or_path=model_id_or_path, subfolder=subfolder,
                                  config=config)
    return {"payload": payload, "temperature": temperature,
            "abstention_thresholds": thresholds, "n_records": len(recs)}


def write_payload(payload, path):
    """Write a payload atomically, so a nightly job never leaves a half-written file."""
    destination = os.path.realpath(path)
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(destination) or ".",
                                     prefix=".calibration.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            mode = os.stat(destination).st_mode & 0o777
        except FileNotFoundError:
            pass
        else:
            os.chmod(temporary, mode)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
