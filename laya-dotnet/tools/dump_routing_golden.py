"""Record golden vectors for the router/language/email/shortlist port (Phase A).

Pure Python, no model weights and no ONNX session: everything this script touches --
`laya.lang`, `laya.router.Router.route` (never `.load`/`.predict`), `laya.email` and
`laya.shortlist.shortlist_choice` -- is dependency-free Python plus NumPy. It reads its input
corpus from `laya-dotnet/tools/routing_cases.py` and writes one JSON file per probe to
`laya-dotnet/tests/Laya.Tests/golden/routing/`:

    lang_probe.json              lang.analyse / detect_script / script_profile /
                                  latin_profile / state_text, for every LANG_STATES entry.
    route_probe.json             Router.route() decisions (model, reason, detection,
                                  workflow -- `repo` is intentionally omitted: the ported
                                  RouteDecision has no Repo field) for every ROUTE_CASES entry.
    route_batch_probe.json       Router.route_batch() decisions for every ROUTE_BATCH_CASES entry, and
                                  the exception type/message for every ROUTE_BATCH_ERROR_CASES entry
                                  (route_batch loads nothing, so no model is needed).
    email_probe.json             clean_email_body and email_state for every EMAIL_CLEAN_CASES /
                                  EMAIL_STATE_CASES entry.
    shortlist_probe.json         shortlist_choice(labels, scores, passthrough) for every
                                  SHORTLIST_CASES entry, with the exact positional embedding
                                  matrix each case used (so a port needs no embedder to check
                                  its ranking/cosine/tie-break logic against this file). NaN/Inf
                                  values in a matrix are written as the strings "nan"/"inf"/
                                  "-inf" (see meta.json) because JSON has no literal for them.
    confidence_probe.json        laya.confidence's own answers over the synthetic inputs of
                                  tools/confidence_cases.py: threshold validation (check_min_confidence and
                                  _map, with the error text), _option_bucket, resolve_min_confidence,
                                  apply_confidence_gate (incl. reused payloads, #910) and jev_confidence.
                                  Non-finite floats are written as "nan"/"inf"/"-inf" strings.
    calibration_probe.json       laya.calibrate.apply_calibration_payload and apply_binning_map over the synthetic
                                  inputs of tools/calibration_cases.py: what a payload installs (clamped temperatures,
                                  binning map) or the exception (type and text), every warning raised (text and
                                  category), and the recalibrated confidence per lookup (bin edges, 0.0, 1.0, a
                                  bucket the map lacks). No model.
    lang_temperatures_probe.json laya.common.resolve_lang_temperatures over the synthetic inputs of
                                  tools/lang_temperature_cases.py: the resolved table (key normalisation, last
                                  duplicate wins, inheriting the checkpoint's temperature, clamping) or the
                                  exception (type and ValueError text), plus every warning raised, per case.
                                  Non-finite floats are written as "nan"/"inf"/"-inf" strings.
    router_forwarding_probe.json What a Router hands the agent it routed to: Router.predict / predict_long / predict_batch over
                                  stub agents that record the keyword arguments they received (the language, the
                                  token budgets, the batch grouping), from tools/lang_temperature_cases.py. No model.
    hashing_embedder_probe.json  Exact laya-dotnet/tools/hashing_embedder.embed() vectors for a fixed probe
                                  list of strings, including non-ASCII and astral input.
    sample_inputs.json           The combined routing sample's fixed inputs (detection states,
                                  routing-decision examples, support questions/emails, a raw
                                  email, and the banking shortlist demo), plus the
                                  Python-computed outputs a sample or test can check itself
                                  against with no model loaded.
    meta.json                    Generator/version info and per-file entry counts.

Usage:
    laya-dotnet/tools/dump_routing_golden.py                # writes routing/ next to the checkpoint goldens
    laya-dotnet/tools/dump_routing_golden.py --force         # overwrite files whose content differs
    laya-dotnet/tools/dump_routing_golden.py --out DIR       # write elsewhere (rare; testing only)

Refuses to overwrite an existing golden file whose content would change unless `--force` is
given -- a content diff, not a mtime check, so re-running with no corpus changes is always a
silent no-op.
"""
import argparse
import copy
import json
import math
import os
import sys
import types
import warnings

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, REPO)
sys.path.insert(0, TOOLS)

from routing_cases import (  # noqa: E402
    EMAIL_CLEAN_CASES,
    EMAIL_STATE_CASES,
    LANG_STATES,
    ROUTE_BATCH_CASES,
    ROUTE_BATCH_ERROR_CASES,
    ROUTE_CASES,
    SAMPLE_BANKING_CRITERIA,
    SAMPLE_BANKING_INSTRUCTIONS,
    SAMPLE_BANKING_STATE,
    SAMPLE_DETECTION_STATES,
    SAMPLE_RAW_EMAIL_BODY,
    SAMPLE_RAW_EMAIL_SENDER,
    SAMPLE_RAW_EMAIL_SUBJECT,
    SAMPLE_ROUTING_DECISIONS,
    SAMPLE_SUPPORT_EMAIL_EN,
    SAMPLE_SUPPORT_EMAIL_HI,
    SAMPLE_SUPPORT_QUESTIONS,
    SHORTLIST_CASES,
)

import calibration_cases as calc  # noqa: E402
import confidence_cases as cc  # noqa: E402
import lang_temperature_cases as ltc  # noqa: E402
import hashing_embedder  # noqa: E402

from laya import confidence  # noqa: E402
from laya.calibrate import CALIBRATION_VERSION, apply_binning_map, apply_calibration_payload  # noqa: E402
from laya.common import TEMP_MAX, TEMP_MIN, resolve_lang_temperatures  # noqa: E402
from laya.email import clean_email_body, email_state  # noqa: E402
from laya.lang import analyse, detect_script, latin_profile, script_profile, state_text  # noqa: E402
from laya.router import Router  # noqa: E402
from laya.shortlist import DEFAULT_SHORTLIST_K, shortlist_choice  # noqa: E402

OUT_DIR_DEFAULT = os.path.join(REPO, "laya-dotnet", "tests", "Laya.Tests", "golden", "routing")

# ~20+ strings for hashing_embedder_probe.json: ASCII, accents, full case mapping, CJK,
# Cyrillic, Arabic, Devanagari, astral (emoji and a script with no astral-BMP equivalent),
# whitespace-only, and a long repeated run to exercise bucket collisions.
HASHING_EMBEDDER_PROBE_STRINGS = [
    "",
    "a",
    "hello",
    "Hello World",
    "HELLO WORLD",
    "café",
    "naïve résumé Müller",
    "İstanbul",
    "ıstanbul",
    "我这个月被重复扣款了两次",
    "こんにちは世界",
    "안녕하세요",
    "тест русского текста",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "\U0001F600\U0001F680",
    "refund \U0001F600 today",
    "\U00010400\U00010401",
    "   ",
    "a" * 50,
    "The quick brown fox jumps over the lazy dog",
    "refund!! invoice #4411 @ $99.99 (urgent)",
]


def _encode_floats(value):
    """Recursively replace non-finite floats with the sentinel strings "nan"/"inf"/"-inf",
    since JSON has no literal for them (Python's json module would otherwise emit the
    non-standard bareword `NaN`/`Infinity`, which most other languages' JSON parsers reject).
    Everything else passes through unchanged."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, list):
        return [_encode_floats(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode_floats(v) for k, v in value.items()}
    return value


def dump(obj, path: str, force: bool) -> str:
    """Write `obj` as pretty JSON to `path`. Refuses to overwrite a file whose content would
    change unless `force` is set. Returns "written", "unchanged" or "refused"."""
    new_text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n"
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="\n") as f:
            old_text = f.read()
        if old_text == new_text:
            return "unchanged"
        if not force:
            raise SystemExit(
                "refusing to overwrite %s: its content would change and --force was not given.\n"
                "Re-run with --force if this change is intended." % path
            )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
    print("  wrote %-28s %8.1f KB" % (os.path.basename(path), os.path.getsize(path) / 1024.0))
    return "written"


# ============================================================================ lang_probe
def build_lang_probe():
    out = []
    for label, state in LANG_STATES:
        text = state_text(state)
        out.append({
            "label": label,
            "state": state,
            "state_text": text,
            "detect_script": detect_script(text),
            "script_profile": script_profile(text),
            "latin_profile": latin_profile(text),
            "analyse": analyse(state),
        })
    return out


# ============================================================================ route_probe
def _decision_payload(decision):
    """RouteDecision -> {model, reason, detection, workflow}. `repo` is intentionally
    dropped: the ported RouteDecision type has no Repo field (see the porting plan)."""
    return {
        "model": decision["model"],
        "reason": decision["reason"],
        "detection": decision["detection"],
        "workflow": decision["workflow"],
    }


def build_route_probe():
    out = []
    for label, router_kwargs, route_kwargs in ROUTE_CASES:
        router = Router(**router_kwargs)
        decision = router.route(**route_kwargs)
        out.append({
            "label": label,
            "router_kwargs": router_kwargs,
            "route_kwargs": route_kwargs,
            "decision": _decision_payload(decision),
        })
    return out


# ============================================================================ route_batch_probe
def build_route_batch_probe():
    cases = []
    for label, router_kwargs, requests in ROUTE_BATCH_CASES:
        decisions = Router(**router_kwargs).route_batch(requests)
        cases.append({
            "label": label,
            "router_kwargs": router_kwargs,
            "requests": requests,
            "decisions": [_decision_payload(d) for d in decisions],
        })
    errors = []
    for label, router_kwargs, requests in ROUTE_BATCH_ERROR_CASES:
        try:
            Router(**router_kwargs).route_batch(requests)
        except (TypeError, ValueError) as e:
            errors.append({
                "label": label,
                "router_kwargs": router_kwargs,
                "requests": requests,
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
        else:
            raise SystemExit("route_batch case %r was expected to raise but did not" % label)
    return {"cases": cases, "errors": errors}


# ============================================================================ confidence_probe
def _checked(fn, arg):
    """`{"valid": True, "result": ...}` or `{"valid": False, "error_type", "error_message"}`."""
    try:
        return {"valid": True, "result": fn(arg)}
    except ValueError as e:
        return {"valid": False, "error_type": type(e).__name__, "error_message": str(e)}


def _gate_verdict(answers):
    """What the gate wrote on each answer: only the keys that are present."""
    return {
        qid: {k: a[k] for k in ("low_confidence", "abstention", "abstention_threshold") if k in a}
        for qid, a in answers.items()
    }


def _gate(answers, threshold):
    """apply_confidence_gate exactly as ONNXAgent.predict_batch calls it: validate, then gate."""
    mc = confidence.check_min_confidence(threshold) if threshold is not None else None
    confidence.apply_confidence_gate([{"answers": answers}], mc)


def build_confidence_probe():
    # jev_confidence sums floats; CPython 3.12 made sum() compensated, and the SDKs reproduce that.
    if sys.version_info < (3, 12):
        sys.exit("confidence_probe.json must be recorded with Python 3.12 or later "
                 "(sum() of floats is compensated from 3.12 on), got %d.%d" % sys.version_info[:2])
    scalars = [dict(value=v, **_checked(confidence.check_min_confidence, v)) for v in cc.SCALAR_THRESHOLDS]

    maps = []
    for pairs in cc.MAP_THRESHOLDS:
        # Python's dict keeps one entry per key, at its first position, holding its last value.
        checked = _checked(confidence.check_min_confidence, dict(pairs))
        if checked["valid"]:
            checked["result"] = [list(item) for item in checked["result"].items()]
        maps.append(dict(pairs=[list(pair) for pair in pairs], **checked))

    buckets = [{"answer": a, "bucket": confidence._option_bucket(a)} for a in cc.BUCKET_ANSWERS]

    resolve = [
        {"thresholds": m, "answer": a, "result": confidence.resolve_min_confidence(a, m)}
        for m in cc.RESOLVE_MAPS for a in cc.RESOLVE_ANSWERS
    ]

    gate_cases = []
    for threshold in cc.GATE_THRESHOLDS:
        answers = copy.deepcopy(cc.GATE_ANSWERS)
        _gate(answers, threshold)
        gate_cases.append({"threshold": threshold, "expected": _gate_verdict(answers)})

    reuse = []
    for steps in cc.REUSE_SEQUENCES:
        answers = copy.deepcopy(cc.GATE_ANSWERS)
        recorded = []
        for threshold in steps:
            _gate(answers, threshold)
            recorded.append({"threshold": threshold, "expected": _gate_verdict(answers)})
        reuse.append({"steps": recorded})

    jev = [dict(case, result=confidence.jev_confidence(case["answer"])) for case in cc.JEV_CASES]

    return {
        "bucket_keys": sorted(confidence.bucket_keys()),
        "gate_states": list(confidence.GATE_STATES),
        "scalars": scalars,
        "maps": maps,
        "buckets": buckets,
        "resolve": resolve,
        "gate_answers": cc.GATE_ANSWERS,
        "gate": gate_cases,
        "reuse": reuse,
        "jev": jev,
    }


# ============================================================================ lang_temperatures_probe
def _resolve_recorded(pairs, base):
    """`resolve_lang_temperatures(dict(pairs), base)` with its warnings: `{"result"|"error", "warnings"}`."""
    raw = None if pairs is None else dict(pairs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            outcome = {"result": resolve_lang_temperatures(raw, base)}
        except Exception as e:  # the helper promises ValueError; anything else is a finding, so it is recorded too
            outcome = {"error_type": type(e).__name__, "error_message": str(e)}
    outcome["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    return outcome


def build_lang_temperatures_probe():
    cases = []
    for label, pairs, base, typed in ltc.PROBE_CASES:
        cases.append(dict(
            label=label,
            typed=typed,
            pairs=None if pairs is None else [[key, entry] for key, entry in pairs],
            base=base,
            **_resolve_recorded(pairs, base),
        ))
    return {"temp_min": TEMP_MIN, "temp_max": TEMP_MAX, "cases": cases}


# ============================================================================ calibration_probe
def _recorded_warnings(caught):
    return [{"category": w.category.__name__, "message": str(w.message)} for w in caught]


def _apply_recorded(text, identity):
    """`apply_calibration_payload` on `json.loads(text)` for an agent with `identity`: what it installed, or the error."""
    agent = types.SimpleNamespace(**identity)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            apply_calibration_payload(agent, json.loads(text))
            outcome = {"result": {
                "temperature": agent.temperature,
                "temperature_by_options": agent.temperature_by_options,
                "binning_map": agent.binning_map,
            }}
        except Exception as e:  # ValueError by contract; an OverflowError is a finding, so everything is recorded
            outcome = {"error_type": type(e).__name__, "error_message": str(e)}
    outcome["warnings"] = _recorded_warnings(caught)
    return outcome


def _lookup_recorded(binning_map, bucket, confidence_value):
    try:
        return {"result": apply_binning_map(confidence_value, bucket, binning_map)}
    except Exception as e:
        return {"error_type": type(e).__name__, "error_message": str(e)}


def build_calibration_probe():
    payloads = [
        dict(label=label, json=text, identity=calc.IDENTITIES[identity], **_apply_recorded(text, calc.IDENTITIES[identity]))
        for label, text, identity in calc.PAYLOAD_CASES
    ]
    lookups = [
        dict(label=label, map=name, bucket=bucket, confidence=value,
             **_lookup_recorded(calc.MAPS[name], bucket, value))
        for label, name, bucket, value in calc.LOOKUP_CASES
    ]
    return {"temp_min": TEMP_MIN, "temp_max": TEMP_MAX, "version": CALIBRATION_VERSION,
            "maps": calc.MAPS, "payloads": payloads, "lookups": lookups}


# ============================================================================ router_forwarding_probe
class _StubAgent:
    """An agent that answers nothing and records the keyword arguments each entry point received.

    `lang_temperatures` is what a Router tests (`getattr(agent, "lang_temperatures", None)`) before it splits a
    batch by language. `takes_lang=False` swaps `predict_long` for one with no `lang` parameter.
    """

    def __init__(self, name, log, has_lang_temperatures=False, takes_lang=True):
        self.name, self.log = name, log
        self.lang_temperatures = {"de": {}} if has_lang_temperatures else {}
        if not takes_lang:
            self.predict_long = self._predict_long_without_lang

    def _answer(self, state):
        return {"model": self.name, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def system_one(self, state, questions, **kwargs):
        self.log.append({"model": self.name, "entry": "system_one", "states": [state], "kwargs": kwargs})
        return self._answer(state)

    def predict_batch(self, states, questions, batch_size=None, **kwargs):
        self.log.append({"model": self.name, "entry": "predict_batch", "states": list(states), "kwargs": kwargs})
        return [self._answer(s) for s in states]

    def predict_long(self, state, questions, window=None, stride=None, aggregate="auto", batch_size=None, **kwargs):
        self.log.append({"model": self.name, "entry": "predict_long", "states": [state], "kwargs": kwargs})
        return self._answer(state)

    def _predict_long_without_lang(self, state, questions, window=None, stride=None, aggregate="auto",
                                   batch_size=None):
        self.log.append({"model": self.name, "entry": "predict_long", "states": [state], "kwargs": {}})
        return self._answer(state)


def _stub_router(log, has_lang_temperatures=None, takes_lang=True):
    has = has_lang_temperatures or {}
    router = Router()
    for name in ("english", "multilingual"):
        router.attach(name, _StubAgent(name, log, has.get(name, False), takes_lang))
    return router


def _warned(caught):
    return [{"category": w.category.__name__, "message": str(w.message)} for w in caught]


def build_router_forwarding_probe():
    predict = []
    for label, args in ltc.ROUTER_PREDICT_CASES:
        log = []
        kwargs = dict(args)
        result = _stub_router(log).predict(kwargs.pop("state"), ltc.ROUTER_QUESTIONS, **kwargs)
        (call,) = log
        predict.append({"label": label, "args": args, "model": result["routing"]["model"],
                        "agent": call["model"], "received": call["kwargs"]})

    long = []
    for label, args in ltc.ROUTER_LONG_CASES:
        log = []
        kwargs = dict(args)
        takes_lang = kwargs.pop("stub_takes_lang", True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _stub_router(log, takes_lang=takes_lang).predict_long(
                kwargs.pop("state"), ltc.ROUTER_QUESTIONS, **kwargs)
        (call,) = log
        long.append({"label": label, "args": args, "model": result["routing"]["model"],
                     "received": call["kwargs"], "warnings": _warned(caught)})

    batch = []
    for label, has, requests in ltc.ROUTER_BATCH_CASES:
        log = []
        results = _stub_router(log, has).predict_batch(copy.deepcopy(requests))
        batch.append({
            "label": label,
            "has_lang_temperatures": has,
            "requests": requests,
            "models": [r["routing"]["model"] for r in results],
            "calls": [{"model": c["model"], "states": c["states"], "kwargs": c["kwargs"]} for c in log],
        })
    return {"predict": predict, "long": long, "batch": batch}


# ============================================================================ email_probe
def build_email_probe():
    clean = []
    for label, body, max_chars in EMAIL_CLEAN_CASES:
        clean.append({
            "label": label,
            "body": body,
            "max_chars": max_chars,
            "result": clean_email_body(body, max_chars),
        })
    state = []
    for label, subject, body, sender, clean_flag, extra in EMAIL_STATE_CASES:
        state.append({
            "label": label,
            "subject": subject,
            "body": body,
            "sender": sender,
            "clean": clean_flag,
            "extra": extra,
            "result": email_state(subject, body, sender, clean_flag, **extra),
        })
    return {"clean": clean, "state": state}


# ============================================================================ shortlist_probe
class _Boom:
    """embed_fn stand-in for a passthrough case: raising proves shortlist_choice never calls
    it when k >= n, exactly like laya-dotnet/tools/routing_cases.py's SHORTLIST_CASES expects."""

    def __call__(self, texts):
        raise AssertionError("embed_fn must not be called when k >= n (passthrough)")


def _matrix_embed_fn(vectors):
    import numpy as np

    def embed_fn(texts):
        if len(texts) != len(vectors):
            raise AssertionError(
                "case's embed_fn called with %d texts, expected %d (vectors length)"
                % (len(texts), len(vectors))
            )
        return np.array(vectors, dtype=np.float64)

    return embed_fn


def build_shortlist_probe():
    out = []
    for case in SHORTLIST_CASES:
        vectors = case["vectors"]
        embed_fn = _Boom() if vectors is None else _matrix_embed_fn(vectors)
        labels, scores, passthrough, n = _rank_with_metadata(
            case["state"], case["criteria"], embed_fn, case["k"], case["instructions"]
        )
        out.append({
            "label": case["label"],
            "state": case["state"],
            "criteria": case["criteria"],
            "k": case["k"],
            "instructions": case["instructions"],
            "vectors": _encode_floats(vectors) if vectors is not None else None,
            "labels": labels,
            "scores": scores,
            "passthrough": passthrough,
            "n": n,
        })
    return out


def _rank_with_metadata(state, criteria, embed_fn, k, instructions):
    """shortlist_choice only returns labels; re-derive passthrough/n/scores the same way
    predict_shortlist does, by calling the private _rank directly (still the real code path,
    just the one that also reports the metadata the golden needs)."""
    from laya.shortlist import _rank

    labels, scores, passthrough, n = _rank(state, criteria, embed_fn, k, instructions)
    # Cross-check against the public function so the golden can never silently diverge from
    # what a caller of shortlist_choice actually gets.
    public_labels = shortlist_choice(state, criteria, embed_fn, k=k, instructions=instructions)
    if list(labels) != list(public_labels):
        raise SystemExit(
            "internal inconsistency: _rank labels %r != shortlist_choice labels %r for case"
            % (labels, public_labels)
        )
    return list(labels), scores, passthrough, n


# ============================================================================ hashing_embedder_probe
def build_hashing_embedder_probe():
    vecs = hashing_embedder.embed(HASHING_EMBEDDER_PROBE_STRINGS)
    return [
        {"text": s, "vector": [float(x) for x in vecs[i]]}
        for i, s in enumerate(HASHING_EMBEDDER_PROBE_STRINGS)
    ]


# ============================================================================ sample_inputs
def build_sample_inputs():
    detection = []
    for label, state in SAMPLE_DETECTION_STATES:
        detection.append({"label": label, "state": state, "analyse": analyse(state)})

    routing = []
    for label, router_kwargs, route_kwargs in SAMPLE_ROUTING_DECISIONS:
        router = Router(**router_kwargs)
        decision = router.route(**route_kwargs)
        routing.append({
            "label": label,
            "router_kwargs": router_kwargs,
            "route_kwargs": route_kwargs,
            "decision": _decision_payload(decision),
        })

    support_email = {
        "questions": SAMPLE_SUPPORT_QUESTIONS,
        "english_state": SAMPLE_SUPPORT_EMAIL_EN,
        "hindi_state": SAMPLE_SUPPORT_EMAIL_HI,
        "english_routing": _decision_payload(Router().route(SAMPLE_SUPPORT_EMAIL_EN, SAMPLE_SUPPORT_QUESTIONS)),
        "hindi_routing": _decision_payload(Router().route(SAMPLE_SUPPORT_EMAIL_HI, SAMPLE_SUPPORT_QUESTIONS)),
    }

    cleaned_body = clean_email_body(SAMPLE_RAW_EMAIL_BODY)
    raw_email = {
        "subject": SAMPLE_RAW_EMAIL_SUBJECT,
        "sender": SAMPLE_RAW_EMAIL_SENDER,
        "body": SAMPLE_RAW_EMAIL_BODY,
        "cleaned_body": cleaned_body,
        "state": email_state(SAMPLE_RAW_EMAIL_SUBJECT, SAMPLE_RAW_EMAIL_BODY, SAMPLE_RAW_EMAIL_SENDER),
        "routing": _decision_payload(Router().route(
            email_state(SAMPLE_RAW_EMAIL_SUBJECT, SAMPLE_RAW_EMAIL_BODY, SAMPLE_RAW_EMAIL_SENDER)
        )),
    }

    banking_k = 20
    banking_labels, banking_scores, banking_passthrough, banking_n = _rank_with_metadata(
        SAMPLE_BANKING_STATE, SAMPLE_BANKING_CRITERIA, hashing_embedder.embed,
        banking_k, SAMPLE_BANKING_INSTRUCTIONS,
    )
    banking = {
        "state": SAMPLE_BANKING_STATE,
        "instructions": SAMPLE_BANKING_INSTRUCTIONS,
        "criteria": SAMPLE_BANKING_CRITERIA,
        "k": banking_k,
        "n": banking_n,
        "passthrough": banking_passthrough,
        "labels": banking_labels,
        "scores": banking_scores,
    }

    return {
        "detection_states": detection,
        "routing_decisions": routing,
        "support_email": support_email,
        "raw_email": raw_email,
        "banking_shortlist": banking,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DIR_DEFAULT, help="output directory (default: %s)" % OUT_DIR_DEFAULT)
    ap.add_argument("--force", action="store_true", help="overwrite files whose content would change")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)

    lang_probe = build_lang_probe()
    route_probe = build_route_probe()
    route_batch_probe = build_route_batch_probe()
    confidence_probe = build_confidence_probe()
    lang_temperatures_probe = build_lang_temperatures_probe()
    calibration_probe = build_calibration_probe()
    router_forwarding_probe = build_router_forwarding_probe()
    email_probe = build_email_probe()
    shortlist_probe = build_shortlist_probe()
    hashing_probe = build_hashing_embedder_probe()
    sample_inputs = build_sample_inputs()

    meta = {
        "generator": "tools/dump_routing_golden.py",
        "laya_version": __import__("laya").__version__,
        "default_shortlist_k": DEFAULT_SHORTLIST_K,
        "notes": {
            "route_probe": "RouteDecision is serialized as {model, reason, detection, workflow}; "
                            "'repo' is intentionally omitted (not part of the ported RouteDecision).",
            "shortlist_probe": "Non-finite floats (NaN, +Inf, -Inf) inside a case's 'vectors' matrix "
                                "are encoded as the JSON strings \"nan\", \"inf\", \"-inf\" because JSON "
                                "has no native literal for them; a case with vectors=null is a k>=n "
                                "passthrough case where embed_fn must never be called.",
        },
        "counts": {
            "lang_probe": len(lang_probe),
            "route_probe": len(route_probe),
            "route_batch_probe": len(route_batch_probe["cases"]),
            "route_batch_probe_errors": len(route_batch_probe["errors"]),
            "confidence_probe_scalars": len(confidence_probe["scalars"]),
            "confidence_probe_maps": len(confidence_probe["maps"]),
            "confidence_probe_buckets": len(confidence_probe["buckets"]),
            "confidence_probe_resolve": len(confidence_probe["resolve"]),
            "confidence_probe_gate": len(confidence_probe["gate"]),
            "confidence_probe_reuse": len(confidence_probe["reuse"]),
            "confidence_probe_jev": len(confidence_probe["jev"]),
            "lang_temperatures_probe": len(lang_temperatures_probe["cases"]),
            "calibration_probe_payloads": len(calibration_probe["payloads"]),
            "calibration_probe_lookups": len(calibration_probe["lookups"]),
            "router_forwarding_probe_predict": len(router_forwarding_probe["predict"]),
            "router_forwarding_probe_long": len(router_forwarding_probe["long"]),
            "router_forwarding_probe_batch": len(router_forwarding_probe["batch"]),
            "email_probe_clean": len(email_probe["clean"]),
            "email_probe_state": len(email_probe["state"]),
            "shortlist_probe": len(shortlist_probe),
            "hashing_embedder_probe": len(hashing_probe),
        },
    }

    results = {}
    results["lang_probe.json"] = dump(lang_probe, os.path.join(out_dir, "lang_probe.json"), args.force)
    results["route_probe.json"] = dump(route_probe, os.path.join(out_dir, "route_probe.json"), args.force)
    results["route_batch_probe.json"] = dump(
        route_batch_probe, os.path.join(out_dir, "route_batch_probe.json"), args.force
    )
    results["confidence_probe.json"] = dump(
        _encode_floats(confidence_probe), os.path.join(out_dir, "confidence_probe.json"), args.force
    )
    results["lang_temperatures_probe.json"] = dump(
        _encode_floats(lang_temperatures_probe), os.path.join(out_dir, "lang_temperatures_probe.json"), args.force
    )
    results["calibration_probe.json"] = dump(
        _encode_floats(calibration_probe), os.path.join(out_dir, "calibration_probe.json"), args.force
    )
    results["router_forwarding_probe.json"] = dump(
        router_forwarding_probe, os.path.join(out_dir, "router_forwarding_probe.json"), args.force
    )
    results["email_probe.json"] = dump(email_probe, os.path.join(out_dir, "email_probe.json"), args.force)
    results["shortlist_probe.json"] = dump(shortlist_probe, os.path.join(out_dir, "shortlist_probe.json"), args.force)
    results["hashing_embedder_probe.json"] = dump(
        hashing_probe, os.path.join(out_dir, "hashing_embedder_probe.json"), args.force
    )
    results["sample_inputs.json"] = dump(sample_inputs, os.path.join(out_dir, "sample_inputs.json"), args.force)
    results["meta.json"] = dump(meta, os.path.join(out_dir, "meta.json"), args.force)

    print("\nrouting golden vectors -> %s" % out_dir)
    for name, status in results.items():
        print("  %-28s %s" % (name, status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
