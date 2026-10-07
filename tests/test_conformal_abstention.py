"""Per-bucket abstention thresholds (#394): fit one cut per option-count bucket so a single
confidence threshold's failure to transfer across option counts is fixed.

Weight-free: synthetic (qtype, logits, target, k) records with deliberately different confidence
regimes per bucket, and synthetic answer dicts for the gate. No model, no download.

Run: python tests/test_conformal_abstention.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.calibrate import (  # noqa: E402
    apply_binning_map,
    fit_abstention_thresholds,
    fit_binning_map,
)
from laya.confidence import (  # noqa: E402
    check_min_confidence,
    check_min_confidence_map,
    resolve_min_confidence,
    flag_low_confidence,
    apply_confidence_gate,
    GATE_ABSTAINED,
    GATE_PASSED,
)

PASS, FAIL = [], []


def check(name, got, want):
    (PASS if got == want else FAIL).append(name if got == want else "%s: got %r want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else "%s %s" % (name, detail))


def check_raises(name, exc, fn):
    try:
        fn()
    except exc:
        PASS.append(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append("%s: raised %r not %s" % (name, e, exc.__name__))
    else:
        FAIL.append("%s: did not raise %s" % (name, exc.__name__))


# ---- build a (qtype, logits, target, k) record realising a given (k, confidence, correct) ----
def record(k, conf, correct, qtype=0):
    conf = min(max(conf, 1e-3), 1 - 1e-3)
    L = math.log(conf * (k - 1) / (1 - conf))   # logit on index 0 so softmax(z).max() == conf
    z = [0.0] * k
    z[0] = L
    target = [0.0] * k
    target[0 if correct else 1] = 1.0           # argmax(z)==0; correct iff gold is 0
    return (qtype, z, target, k)


# Two choice buckets with DIFFERENT confidence regimes:
#   k=2  : overconfident -- a 0.85-confidence cohort is only half right, so a low cut lets errors in.
#   k=12 : well separated -- a 0.50-confidence cohort is already right, so a low cut is safe.
records = []
for _ in range(120):
    records.append(record(2, 0.97, True))        # k=2 confident & correct
for i in range(120):
    records.append(record(2, 0.85, i % 2 == 0))  # k=2 confident but 50% wrong
for _ in range(120):
    records.append(record(12, 0.55, True))       # k=12 modest conf, correct
for i in range(120):
    records.append(record(12, 0.33, i % 5 == 0)) # k=12 low conf, 80% wrong

TARGET = 0.10
thr = fit_abstention_thresholds(records, [1.0, 1.0, 1.0], {}, target_error=TARGET, min_bucket_n=50)
check_true("fit/keyed by temp_bucket", set(thr) == {"choice:2", "choice:11+"}, thr)
check_true("fit/k=2 needs a high cut", thr["choice:2"] > 0.9, thr)
check_true("fit/k=12 accepts a lower cut", thr["choice:11+"] < 0.6, thr)
check_true("fit/the two cuts genuinely differ", thr["choice:2"] - thr["choice:11+"] > 0.3, thr)


# ---- apply per-bucket vs one global cut, measure error among accepted per bucket ----
def err_and_cov(records, cut_fn):
    out = {}
    for qt, z, t, k in records:
        import numpy as np
        p = np.exp(np.array(z) - max(z)); p = p / p.sum()
        conf = float(p.max()); correct = int(p.argmax()) == int(np.argmax(t))
        key = "choice:2" if k <= 2 else "choice:11+"
        if conf >= cut_fn(key):
            a, e = out.get(key, (0, 0))
            out[key] = (a + 1, e + (0 if correct else 1))
    return out

per_bucket = err_and_cov(records, lambda key: thr[key])
for key, (acc, err) in per_bucket.items():
    rate = err / acc if acc else 0.0
    check_true("apply/per-bucket holds target in %s (err %.0f%%, n=%d)" % (key, rate * 100, acc),
               acc > 0 and rate <= TARGET + 0.02, (acc, err))

# one global cut = the per-bucket cut for k=2 (0.9+); it over-abstains k=12 to near-zero coverage,
# which is exactly the #394 failure: a cut tuned for one option count does not serve the other.
GLOBAL = thr["choice:2"]
glob = err_and_cov(records, lambda key: GLOBAL)
k12_acc = glob.get("choice:11+", (0, 0))[0]
check_true("transfer/global k=2 cut starves k=12 coverage (#394)", k12_acc == 0, glob)
# and the mirror: the k=12 cut lets k=2 errors through
GLOBAL2 = thr["choice:11+"]
glob2 = err_and_cov(records, lambda key: GLOBAL2)
k2_acc, k2_err = glob2.get("choice:2", (0, 0))
check_true("transfer/global k=12 cut exceeds target error on k=2 (#394)",
           k2_acc > 0 and (k2_err / k2_acc) > TARGET, (k2_acc, k2_err))


# ---- confidence.py: resolve + gate with a map ----
def ans(qtype, k, conf):
    probs = {str(i): 0.0 for i in range(k)}
    return {"type": qtype, "probabilities": probs, "answer_confidence": conf}

MAP = {"choice:2": 0.9, "choice:11+": 0.5, "noul:2": 0.8, "default": 0.3}
check("resolve/choice k=2 bucket", resolve_min_confidence(ans("choice", 2, 0.5), MAP), 0.9)
check("resolve/choice k=12 bucket (11+)", resolve_min_confidence(ans("choice", 12, 0.5), MAP), 0.5)
check("resolve/choice k=4 falls to default", resolve_min_confidence(ans("choice", 4, 0.5), MAP), 0.3)
check("resolve/noul bucket", resolve_min_confidence({"type": "noul", "answer_confidence": 0.5}, MAP), 0.8)
check("resolve/unknown + no default -> 0.0 (gate nothing)",
      resolve_min_confidence(ans("score", 7, 0.5), {"choice:2": 0.9}), 0.0)

# gate a batch with a per-bucket map: a 0.85-confident k=2 answer abstains (cut 0.9), a 0.6-confident
# k=12 answer passes (cut 0.5) -- a single float could not do both.
results = [{"answers": {"a": ans("choice", 2, 0.85), "b": ans("choice", 12, 0.60)}}]
apply_confidence_gate(results, MAP)
check("gate/k=2 low-conf abstains under its bucket cut", results[0]["answers"]["a"]["abstention"], GATE_ABSTAINED)
check("gate/k=12 passes under its lower bucket cut", results[0]["answers"]["b"]["abstention"], GATE_PASSED)
check("gate/echoes the per-bucket threshold (k=2)", results[0]["answers"]["a"]["abstention_threshold"], 0.9)
check("gate/echoes the per-bucket threshold (k=12)", results[0]["answers"]["b"]["abstention_threshold"], 0.5)

# the float path is unchanged
res_f = [{"answers": {"a": ans("choice", 3, 0.4)}}]
flag_low_confidence(res_f, 0.7)
check("float/below a scalar threshold still flags", res_f[0]["answers"]["a"].get("low_confidence"), True)
check("validate/float still accepted", check_min_confidence(0.7), 0.7)
check("validate/map accepted and normalised", check_min_confidence({"choice:2": 0.9}), {"choice:2": 0.9})
check_raises("validate/map value out of range rejected", ValueError,
             lambda: check_min_confidence_map({"choice:2": 1.5}))
check_raises("validate/map non-string key rejected", ValueError,
             lambda: check_min_confidence_map({2: 0.9}))

# ---- the key vocabulary: a bucket a map may name is a bucket an answer can produce ----
# Before this, `check_min_confidence_map` accepted any string key, so {"choice:2-5": 0.9} validated,
# `_option_bucket` never produced it for any answer, and `resolve_min_confidence` fell through to
# "default" and then 0.0. The caller was told two-option choices were gated at 0.9 and nothing gated
# them -- an abstention gate disabled by a typo, which is the one failure this library exists to
# avoid reporting as success. The refusal is the fix; everything below keeps the vocabulary honest on
# both sides, because it is deliberately copied: `confidence.py` must import without PyTorch, so it
# cannot read `common.QTYPES`.
import inspect  # noqa: E402
from laya import confidence as confidence_module  # noqa: E402
from laya.common import QTYPES, temp_bucket  # noqa: E402

check_true("vocabulary/the copy has something to be honest about",
           len(confidence_module.bucket_keys()) == 12, sorted(confidence_module.bucket_keys()))

# The mirror, both directions: `common.temp_bucket` is the spelling a checkpoint's
# `temperature_by_options` and `fit_abstention_thresholds` use; `bucket_keys()` is what a
# `min_confidence` map may name. A band added to one side only either rejects a map the fitter
# produced or accepts a key no answer can ever produce.
check("vocabulary/bucket_keys() is exactly what common.temp_bucket produces",
      confidence_module.bucket_keys(),
      {temp_bucket(QTYPES[qt], k) for qt in ("choice", "score", "noul") for k in range(2, 25)})

# An accepted key that no answer resolves to is the same silent hole as a rejected one that some
# answer does resolve to, so the reachability has to be driven, not asserted from the spelling. And
# it has to *report* an unknown band: a band in the vocabulary that no answer produces is exactly
# the case under test, so this must not be a lookup that dies on it.
BAND_K = {"2": 2, "3-5": 4, "6-10": 8, "11+": 12}

unreachable = []
for _key in sorted(confidence_module.bucket_keys()):
    _qt, _, _size = _key.partition(":")
    _a = ans(_qt, BAND_K[_size], 0.5) if _size in BAND_K else None
    if _a is None or confidence_module._option_bucket(_a) != _key:
        unreachable.append(_key)
check("vocabulary/every accepted bucket is reachable by some answer", unreachable, [])


def _accepts(m):
    """The validated map, or the refusal's text -- so a widened or narrowed vocabulary reports which
    key it stopped taking, rather than a `default` rejection ending the suite on a traceback."""
    try:
        return check_min_confidence_map(m)
    except ValueError as _e:
        return "raised: %s" % _e


FULL = {k: 0.5 for k in sorted(confidence_module.bucket_keys())}
check("validate/a complete bucket map is accepted unchanged", _accepts(dict(FULL)), FULL)
check("validate/'default' is still accepted beside the buckets",
      _accepts({**FULL, "default": 0.2}), {**FULL, "default": 0.2})
# The producer and the validator have to agree, or the documented path
# (`fit_abstention_thresholds` -> `min_confidence=`) raises on its own output.
check("validate/the fitted map this suite ran on passes the validator", _accepts(dict(thr)), dict(thr))

BOGUS = {
    "choice:2-5": "an invented band",
    "choice:12": "an option count, not a band",
    "choice:11": "the open band is spelled 11+",
    "Choice:2": "the type name is lower case",
    "choise:2": "a transposed type name",
    "noul:3": "noul answers carry two labels",
    "choice": "no band",
    "choice:": "an empty band",
    ":2": "an empty type name",
    "": "the empty key",
    "default:2": "'default' takes no band",
    "score:3-5 ": "a trailing space",
}
for _key, _why in sorted(BOGUS.items()):
    check_raises("refuse/%r is %s" % (_key, _why), ValueError,
                 lambda k=_key: check_min_confidence_map({k: 0.5}))
try:
    check_min_confidence_map({"choice:2-5": 0.9})
    _msg = ""
except ValueError as _e:
    _msg = str(_e)
check_true("refuse/the message names both accepted shapes",
           "choice:3-5" in _msg and "default" in _msg, _msg)

# The prose carries the same contract now, and points at the refusal rather than implying a
# suggestion: a reader of the page is the one writing the key.
_doc = " ".join(inspect.getdoc(confidence_module.check_min_confidence_map).split())
check_true("doc/the docstring states that a bad key raises", "Anything else raises" in _doc, _doc)
check_true("doc/the docstring names the fall-through it refuses to leave silent",
           "resolve_min_confidence" in _doc, _doc)
check_true("doc/the docstring does not present the buckets as a spelling suggestion",
           "plus an optional" not in _doc, _doc)
_resolve_doc = " ".join(inspect.getdoc(confidence_module.resolve_min_confidence).split())
check_true("doc/resolve says a validated map can only miss by omitting a bucket",
           "refused at validation" in _resolve_doc, _resolve_doc)
# `_option_bucket` must read the one vocabulary list, not keep a second copy of the type names.
_bucket_src = inspect.getsource(confidence_module._option_bucket)
check_true("vocabulary/_option_bucket reads BUCKET_QTYPES",
           "BUCKET_QTYPES" in _bucket_src and '("choice", "score", "noul")' not in _bucket_src,
           _bucket_src)
_lines = inspect.getsource(confidence_module).splitlines()
_at = next((i for i, l in enumerate(_lines) if l.startswith("BUCKET_QTYPES")), None)
check_true("vocabulary/the comment above BUCKET_QTYPES names the suite that holds the copy",
           _at is not None and any("test_conformal_abstention" in l for l in _lines[max(0, _at - 5):_at]),
           "no pointer within five lines above the vocabulary")



# --------------------------------------------------------------- thresholds vs a binning map
# A histogram-binning map recalibrates `answer_confidence` at runtime (`Agent._decode_answers`),
# so a cut fitted on the temperature-scaled scale gates a quantity the runtime no longer reports:
# the same number admits far more than its target error. Fitting with the map puts both on one
# scale, and then the order the two were fitted in stops mattering.
import numpy as np  # noqa: E402

_rng = np.random.default_rng(7)
_binning_records = []
for _ in range(1200):
    _true = int(_rng.integers(12))
    _z = _rng.normal(0.0, 1.0, 12)
    _z[_true] += 2.2                        # usually right, sometimes not: a continuous spread
    _t = [0.0] * 12
    _t[_true] = 1.0
    _binning_records.append((0, _z.tolist(), _t, 12))

_temps, _tbo = [1.0, 1.0, 1.0], {}
_bmap = fit_binning_map(_binning_records, _temps, _tbo)
_cut_raw = fit_abstention_thresholds(_binning_records, _temps, _tbo, target_error=0.10)["choice:11+"]
_cut_binned = fit_abstention_thresholds(_binning_records, _temps, _tbo, binning_map=_bmap,
                                        target_error=0.10)["choice:11+"]


def _gate_error(cut, binned):
    """Coverage and error among accepted, when the runtime reports binned (or raw) confidences."""
    accepted = wrong = 0
    for qt, z, t, k in _binning_records:
        p = np.exp(np.asarray(z) - max(z))
        p = p / p.sum()
        conf = float(p.max())
        if binned:
            conf = apply_binning_map(conf, "choice:11+", _bmap)
        if conf >= cut:
            accepted += 1
            wrong += int(int(p.argmax()) != int(np.argmax(t)))
    return accepted, (wrong / accepted if accepted else 0.0)


_acc_raw, _err_raw = _gate_error(_cut_raw, binned=False)
_acc_mismatch, _err_mismatch = _gate_error(_cut_raw, binned=True)
_acc_fit, _err_fit = _gate_error(_cut_binned, binned=True)

check_true("binning/an un-binned cut holds its target on its own scale", _err_raw <= 0.12,
           "cut=%.4f accepted=%d error=%.3f" % (_cut_raw, _acc_raw, _err_raw))
check_true("binning/the same cut stops gating once the runtime bins", _err_mismatch > 0.2,
           "cut=%.4f accepted=%d error=%.3f" % (_cut_raw, _acc_mismatch, _err_mismatch))
check_true("binning/fitting with the map restores the target", _err_fit <= 0.12,
           "cut=%.4f accepted=%d error=%.3f" % (_cut_binned, _acc_fit, _err_fit))
check_true("binning/the two cuts genuinely differ", abs(_cut_binned - _cut_raw) > 1e-9,
           "raw=%.4f binned=%.4f" % (_cut_raw, _cut_binned))
check("binning/no map leaves the fit unchanged",
      fit_abstention_thresholds(_binning_records, _temps, _tbo, binning_map=None,
                                target_error=0.10)["choice:11+"], _cut_raw)


# --------------------------------------------------------------- thresholds vs 4-decimal reporting
# `_decode_answers` reports `answer_confidence` rounded to 4 decimals and the gate compares that, so
# a cut chosen at full precision can sit above every value the runtime reports for its own cohort:
# seed 2's cut is the bin value 5/6 = 0.83333..., the runtime reports that bin as 0.8333, and the
# whole bin abstains. Gated through the real decoder, the fitted cut must keep the coverage and
# error the fitter judged it on.
from laya.agent import Agent  # noqa: E402

_rng = np.random.default_rng(2)
_round_records = []
for _ in range(1200):
    _true = int(_rng.integers(12))
    _z = _rng.normal(0.0, 1.0, 12)
    _z[_true] += 2.2
    _t = [0.0] * 12
    _t[_true] = 1.0
    _round_records.append((0, _z.tolist(), _t, 12))

_round_bmap = fit_binning_map(_round_records, _temps, _tbo)
_round_cut = fit_abstention_thresholds(_round_records, _temps, _tbo, binning_map=_round_bmap,
                                       target_error=0.10)["choice:11+"]

_dec = Agent.__new__(Agent)
_dec.temperature, _dec.temperature_by_options, _dec.lang_temperatures = list(_temps), {}, {}
_dec.binning_map = _round_bmap
_round_ids = ["q%d" % i for i in range(len(_round_records))]
_q = Agent._to_internal({"type": "choice", "instructions": "?", "criteria": ["o%d" % i for i in range(12)]})
_decoded = _dec._decode_answers(np.asarray([z for _, z, _, _ in _round_records], dtype=np.float64),
                                np.zeros((len(_round_records), 2)), [{"markers": list(range(12))}] * len(_round_ids),
                                _round_ids, {qid: _q for qid in _round_ids}, 0)
_round_results = [{"answers": _decoded}]
apply_confidence_gate(_round_results, {"choice:11+": _round_cut})

_fit_acc = _gate_acc = _gate_wrong = 0
for qid, (_qt, z, t, k) in zip(_round_ids, _round_records):
    p = np.exp(np.asarray(z) - max(z))
    p = p / p.sum()
    _fit_acc += apply_binning_map(float(p.max()), "choice:11+", _round_bmap) >= _round_cut
    if _decoded[qid]["abstention"] == GATE_PASSED:
        _gate_acc += 1
        _gate_wrong += int(int(p.argmax()) != int(np.argmax(t)))

check_true("rounding/the cut is a value the runtime can report", round(_round_cut, 4) == _round_cut,
           "cut=%r" % (_round_cut,))
check("rounding/the gate keeps every answer of the cut bin", _gate_acc, _fit_acc)
check_true("rounding/the gate holds the target error", _gate_acc > 0 and _gate_wrong / _gate_acc <= 0.10,
           "accepted=%d wrong=%d" % (_gate_acc, _gate_wrong))
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
