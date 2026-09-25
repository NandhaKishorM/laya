"""Per-bucket temperature fitting. No model weights are loaded.

Records are CPU tuples (qtype, logits, target, k). The fitter is the notebook's NLL+LBFGS
on log T, grouped by `temp_bucket`. Bounds are `TEMP_MIN` / `TEMP_MAX` via `clamp_temperature`.
Logits for a labeled agent come from `_encode_state` and the raw rows `_decode_answers` scales.
"""
import inspect
import json
import os
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent, load  # noqa: E402
from laya.calibrate import (  # noqa: E402
    CALIBRATION_VERSION,
    ECE_HOLDOUT_FRAC,
    MIN_BUCKET_N,
    MIN_TYPE_N,
    _ece_report,
    _ece_split,
    _iter_records,
    apply_calibration_payload,
    calibration_payload,
    fit_one_temperature,
    fit_temperature_map,
    fit_temperatures,
    records_from_labeled,
)
from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, temp_bucket  # noqa: E402
import laya.calibrate as _calibrate  # noqa: E402

PASS, FAIL = [], []

# Large enough that ECE_HOLDOUT_FRAC still leaves MIN_BUCKET_N rows to fit.
BUCKET_N = 2500


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def peaked(k, idx, mag):
    z = np.full(k, -mag, dtype=np.float32)
    z[idx] = mag
    t = np.zeros(k, dtype=np.float32)
    t[idx] = 1.0
    return z, t


def overconfident_records(n, k, qtype, seed=0, acc=0.6, mag=8.0):
    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n):
        y = int(rng.randint(0, k))
        pred = y if rng.rand() < acc else int((y + 1) % k)
        z = np.full(k, -mag, dtype=np.float32)
        z[pred] = mag
        t = np.zeros(k, dtype=np.float32)
        t[y] = 1.0
        recs.append((qtype, z, t, k))
    return recs


check("floor/MIN_BUCKET_N", MIN_BUCKET_N, 2000)
check("floor/MIN_TYPE_N", MIN_TYPE_N, 10)
check("clamp/TEMP_MIN", TEMP_MIN, 0.5)
check("clamp/TEMP_MAX", TEMP_MAX, 5.0)
check_true("clamp/no local TEMP_LO", not hasattr(_calibrate, "TEMP_LO"))
check_true("clamp/no local TEMP_HI", not hasattr(_calibrate, "TEMP_HI"))
check_true(
    "setup/holdout leaves a fittable bucket",
    BUCKET_N - int(round(BUCKET_N * ECE_HOLDOUT_FRAC)) >= MIN_BUCKET_N,
)


# --------------------------------------------------------------- overconfident -> T > 1
choice_k2 = overconfident_records(BUCKET_N, 2, QTYPES["choice"], seed=1, acc=0.6, mag=8.0)
t_choice = fit_one_temperature([(z, t) for _, z, t, _ in choice_k2])
check_true("overconfident/T>1", t_choice > 1.0, "T=%r" % t_choice)
check_true("overconfident/T<=TEMP_MAX", t_choice <= TEMP_MAX, "T=%r" % t_choice)


# --------------------------------------------------------------- bucket keys + omit n<MIN_BUCKET_N
noul_k2 = overconfident_records(BUCKET_N, 2, QTYPES["noul"], seed=2, acc=0.6, mag=8.0)
choice_k12 = overconfident_records(BUCKET_N, 12, QTYPES["choice"], seed=3, acc=0.6, mag=8.0)
# Exactly the per-bucket floor: fitted, but too small to hold any rows out for ECE.
choice_k4 = overconfident_records(MIN_BUCKET_N, 4, QTYPES["choice"], seed=6, acc=0.6, mag=8.0)
tiny_score = overconfident_records(5, 4, QTYPES["score"], seed=4, acc=0.6, mag=8.0)
mixed = choice_k2 + noul_k2 + choice_k12 + choice_k4 + tiny_score
fitted = fit_temperature_map(mixed, compute_ece=True)
keys = set(fitted["temperature_by_options"])
check_true("keys/choice:2", "choice:2" in keys, keys)
check_true("keys/noul:2", "noul:2" in keys, keys)
check_true("keys/choice:11+", "choice:11+" in keys, keys)
check_true("keys/choice:3-5 at floor", "choice:3-5" in keys, keys)
check("keys/temp_bucket choice:2", temp_bucket(QTYPES["choice"], 2), "choice:2")
check("keys/temp_bucket noul:2", temp_bucket(QTYPES["noul"], 2), "noul:2")
check("keys/temp_bucket choice:11+", temp_bucket(QTYPES["choice"], 12), "choice:11+")
check_true("omit/n<MIN_BUCKET_N score:3-5 not in map", "score:3-5" not in keys, keys)
check("omit/n_by_bucket still counts tiny", fitted["n_by_bucket"].get("score:3-5"), 5)
check("omit/n_by_bucket counts floor bucket", fitted["n_by_bucket"].get("choice:3-5"), MIN_BUCKET_N)
check_true("omit/not NaN", all(np.isfinite(v) for v in fitted["temperature_by_options"].values()))
check_true("alias/fit_temperatures is fit_temperature_map", fit_temperatures is fit_temperature_map)


# --------------------------------------------------------------- type floor stays below the bucket floor
# 50 is above MIN_TYPE_N and below MIN_BUCKET_N, so the scalar moves and the bucket is omitted.
medium_score = overconfident_records(50, 4, QTYPES["score"], seed=5, acc=0.6, mag=8.0)
med = fit_temperature_map(medium_score, compute_ece=False, seed=0)
med_other_seed = fit_temperature_map(medium_score, compute_ece=False, seed=1)
check_true(
    "type-floor/fitted below bucket floor",
    med["temperature"][QTYPES["score"]] > 1.0,
    med["temperature"],
)
check_true("type-floor/bucket omitted", "score:3-5" not in med["temperature_by_options"])
check_true("ece/off has no report", "report" not in med)
check("ece/off ignores seed", med["temperature"], med_other_seed["temperature"])


# --------------------------------------------------------------- clamp 0.5-5
# 50% accurate, extreme logits: NLL wants a large T, clamped at TEMP_MAX.
rng = np.random.RandomState(7)
hi_pairs = []
for i in range(MIN_BUCKET_N):
    y = int(rng.randint(0, 2))
    pred = y if i % 2 == 0 else 1 - y
    z, t = peaked(2, pred, 80.0)
    t = np.zeros(2, dtype=np.float32)
    t[y] = 1.0
    hi_pairs.append((z, t))
t_hi = fit_one_temperature(hi_pairs)
check_true("clamp/high in [0.5, 5]", TEMP_MIN <= t_hi <= TEMP_MAX, "T=%r" % t_hi)
check_true("clamp/high at TEMP_MAX", abs(t_hi - TEMP_MAX) < 1e-4, "T=%r" % t_hi)

# Always-correct, mild logits: NLL wants a small T, clamped at TEMP_MIN.
lo_pairs = []
for i in range(MIN_BUCKET_N):
    z, t = peaked(2, i % 2, 0.3)
    lo_pairs.append((z, t))
t_lo = fit_one_temperature(lo_pairs)
check_true("clamp/low in [0.5, 5]", TEMP_MIN <= t_lo <= TEMP_MAX, "T=%r" % t_lo)
check_true("clamp/low at TEMP_MIN", abs(t_lo - TEMP_MIN) < 1e-4, "T=%r" % t_lo)

src = inspect.getsource(fit_one_temperature)
check_true("clamp/source uses clamp_temperature", "clamp_temperature" in src)
check_true("clamp/source uses TEMP_MIN", "TEMP_MIN" in src)
check_true("clamp/source uses TEMP_MAX", "TEMP_MAX" in src)
check_true("clamp/n<MIN_BUCKET_N returns 1.0", fit_one_temperature(lo_pairs[:5]) == 1.0)
check_true(
    "clamp/n=MIN_BUCKET_N-1 returns 1.0",
    fit_one_temperature(lo_pairs[: MIN_BUCKET_N - 1]) == 1.0,
)


# --------------------------------------------------------------- ECE on a held-out split, not the fit rows
report = fitted["report"]
check_true("ece/after < before", report["ece_after"] < report["ece_before"],
           "before=%r after=%r" % (report["ece_before"], report["ece_after"]))
check_true("ece/before finite", np.isfinite(report["ece_before"]))
check_true("ece/after finite", np.isfinite(report["ece_after"]))
check("ece/n is the full input", report["n"], float(len(mixed)))

parsed = _iter_records(mixed)
fit_recs, eval_recs, excluded = _ece_split(parsed, 0)
fit_ids = [id(r) for r in fit_recs]
eval_ids = [id(r) for r in eval_recs]
check_true("split/disjoint", set(fit_ids).isdisjoint(eval_ids))
check("split/partition", len(fit_ids) + len(eval_ids), len(parsed))
again_fit, again_eval, again_ex = _ece_split(parsed, 0)
check("split/deterministic eval", [id(r) for r in again_eval], eval_ids)
check("split/deterministic excluded", again_ex, excluded)
other_fit, other_eval, _other_ex = _ece_split(parsed, 1)
check_true("split/seed changes holdout", [id(r) for r in other_eval] != eval_ids)
check_true("split/floor bucket excluded", "choice:3-5" in excluded, excluded)
check_true("split/tiny excluded", "score:3-5" in excluded, excluded)
check_true("split/big bucket held out", "choice:2" not in excluded, excluded)
check("ece/n_eval", report["n_eval"], float(len(eval_recs)))
check_true("ece/n_eval < n", report["n_eval"] < report["n"])
check("ece/excluded noted", report["buckets_excluded_from_eval"], excluded)
# The map itself is the one fit on the non-held-out rows.
refit = fit_temperature_map(fit_recs, compute_ece=False)
check("split/fit ignores eval buckets", fitted["temperature_by_options"], refit["temperature_by_options"])
check("split/fit ignores eval types", fitted["temperature"], refit["temperature"])
holdout_ece = _ece_report(eval_recs, fitted["temperature"], fitted["temperature_by_options"])
check("ece/after is the holdout", report["ece_after"], holdout_ece["ece_after"])
check("ece/before is the holdout", report["ece_before"], holdout_ece["ece_before"])
full_ece = _ece_report(parsed, fitted["temperature"], fitted["temperature_by_options"])
check_true(
    "ece/not scored on the fit pool",
    full_ece["ece_after"] != report["ece_after"],
    "full=%r holdout=%r" % (full_ece["ece_after"], report["ece_after"]),
)
again_fit_map = fit_temperature_map(mixed, compute_ece=True, seed=0)
check("ece/fit deterministic", again_fit_map["report"], fitted["report"])


# --------------------------------------------------------------- live Agent map + JSON round-trip
_CFG = {
    "encoder": "answerdotai/ModernBERT-large",
    "head_layers": 2,
    "max_len": 512,
    "model_name": "laya",
    "temperature": [9.0, 9.0, 9.0],
    "temperature_by_options": {"choice:2": 9.0},
}
agent = Agent.__new__(Agent)
agent.temperature = [1.0, 1.0, 1.0]
agent.temperature_by_options = {}
agent.model_id_or_path = "convaiinnovations/laya"
agent.subfolder = None
agent.cfg = dict(_CFG)
live = agent.fit_temperatures(mixed, compute_ece=True)
check_true("agent/live temperature_by_options updates",
           "choice:2" in agent.temperature_by_options, agent.temperature_by_options)
check("agent/live matches result", agent.temperature_by_options, live["temperature_by_options"])
check_true("agent/type-level choice T>1", agent.temperature[0] > 1.0, agent.temperature)
check_true("agent/forwards seed", "seed=seed" in inspect.getsource(Agent.fit_temperatures))

td = tempfile.mkdtemp()
calib_path = os.path.join(td, "calibration.json")
agent.save_calibration(calib_path)
check_true("save/no model.safetensors", not os.path.exists(os.path.join(td, "model.safetensors")))
check_true("save/only the json file", os.listdir(td) == ["calibration.json"], os.listdir(td))
with open(calib_path) as f:
    payload = json.load(f)
check(
    "save/keys",
    sorted(payload.keys()),
    ["config", "model_id_or_path", "subfolder", "temperature", "temperature_by_options", "version"],
)
check("save/version", payload["version"], CALIBRATION_VERSION)
check("save/model_id", payload["model_id_or_path"], "convaiinnovations/laya")
check("save/subfolder", payload["subfolder"], None)
check("save/config encoder", payload["config"]["encoder"], "answerdotai/ModernBERT-large")
check("save/config model_name", payload["config"]["model_name"], "laya")
check_true("save/config omits temperature", "temperature" not in payload["config"])
check_true(
    "save/config omits temperature_by_options",
    "temperature_by_options" not in payload["config"],
)
check_true("save/no weights key", "model.safetensors" not in json.dumps(payload))

other = Agent.__new__(Agent)
other.model_id_or_path = agent.model_id_or_path
other.subfolder = agent.subfolder
other.cfg = dict(_CFG)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    other.load_calibration(calib_path)
check_true("load/matching identity is silent", not caught, [str(w.message) for w in caught])
check("load/temperature", other.temperature, agent.temperature)
check("load/by_options", other.temperature_by_options, agent.temperature_by_options)

mismatch = Agent.__new__(Agent)
mismatch.model_id_or_path = "convaiinnovations/laya-multilingual"
mismatch.subfolder = "multilingual"
mismatch.cfg = {"encoder": "jhu-clsp/mmBERT-base", "head_layers": 2}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    mismatch.load_calibration(calib_path)
check_true(
    "load/mismatch warns",
    any(issubclass(w.category, UserWarning) for w in caught),
    [str(w.message) for w in caught],
)
check("load/mismatch still applies temperature", mismatch.temperature, agent.temperature)
check("load/mismatch still applies by_options", mismatch.temperature_by_options, agent.temperature_by_options)

cfg_only = Agent.__new__(Agent)
cfg_only.model_id_or_path = agent.model_id_or_path
cfg_only.subfolder = agent.subfolder
cfg_only.cfg = {"encoder": "some-other-encoder", "head_layers": 4, "temperature": [1.0, 1.0, 1.0]}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    cfg_only.load_calibration(calib_path)
check_true(
    "load/config mismatch warns",
    any("config identity differs" in str(w.message) for w in caught),
    [str(w.message) for w in caught],
)
check("load/config mismatch still applies", cfg_only.temperature, agent.temperature)

# module helpers round-trip on a stub
stub = type("Stub", (), {})()
apply_calibration_payload(stub, calibration_payload([1.2, 1.1, 1.3], {"choice:2": 1.4}))
check("stub/temperature", stub.temperature, [1.2, 1.1, 1.3])
check("stub/by_options", stub.temperature_by_options, {"choice:2": 1.4})

# A payload with no version is the original schema and must still load, even onto an
# agent that has its own identity. No warning: there is no recorded checkpoint to disagree with.
legacy = {"temperature": [1.4, 1.2, 1.1], "temperature_by_options": {"noul:2": 1.5}}
legacy_agent = Agent.__new__(Agent)
legacy_agent.model_id_or_path = "convaiinnovations/laya"
legacy_agent.subfolder = None
legacy_agent.cfg = {"encoder": "answerdotai/ModernBERT-large"}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    apply_calibration_payload(legacy_agent, legacy)
check("version/fallback temperature", legacy_agent.temperature, [1.4, 1.2, 1.1])
check("version/fallback by_options", legacy_agent.temperature_by_options, {"noul:2": 1.5})
check_true("version/missing is silent", not caught, [str(w.message) for w in caught])

# Invalid entries clamp instead of crashing the load (#142, same guard as a checkpoint).
clamped = Agent.__new__(Agent)
clamped.model_id_or_path = "convaiinnovations/laya"
clamped.subfolder = None
clamped.cfg = {"encoder": "answerdotai/ModernBERT-large"}
bad_payload = {
    "version": CALIBRATION_VERSION,
    "temperature": [0.1, "nope", 9],
    "temperature_by_options": {"choice:2": 0, "noul:2": None},
    "model_id_or_path": "convaiinnovations/laya",
    "subfolder": None,
    "config": {"encoder": "answerdotai/ModernBERT-large"},
}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    apply_calibration_payload(clamped, bad_payload)
check("clamp-load/temperature", clamped.temperature, [TEMP_MIN, 1.0, TEMP_MAX])
check("clamp-load/by_options", clamped.temperature_by_options, {"choice:2": TEMP_MIN, "noul:2": 1.0})
check_true(
    "clamp-load/warns",
    any(issubclass(w.category, RuntimeWarning) and "choice:2" in str(w.message) for w in caught),
    [str(w.message) for w in caught],
)
bad_path = os.path.join(td, "bad.json")
Agent.save_calibration(clamped, bad_path)
with open(bad_path) as f:
    bad_saved = json.load(f)
check("clamp-load/saved temperature is applied", bad_saved["temperature"], [TEMP_MIN, 1.0, TEMP_MAX])
check(
    "clamp-load/saved buckets are applied",
    bad_saved["temperature_by_options"],
    {"choice:2": TEMP_MIN, "noul:2": 1.0},
)


# --------------------------------------------------------------- constructor wiring (no Hub download)
init_src = inspect.getsource(Agent.__init__)
check_true("init/calibration kwarg", "calibration: Optional[str] = None" in init_src)
check_true("init/load_calibration call", "self.load_calibration(calibration)" in init_src)
check_true("init/does not write safetensors", "save_file" not in init_src)
check_true("init/stores model_id_or_path", "self.model_id_or_path = model_id_or_path" in init_src)
check_true("init/stores subfolder", "self.subfolder = subfolder" in init_src)
load_src = inspect.getsource(load)
check_true("load/calibration kwarg", "calibration" in load_src)
check_true("load/forwards calibration", "calibration=calibration" in load_src)

rec_src = inspect.getsource(records_from_labeled)
check_true("records/_encode_state", "_encode_state" in rec_src)
check_true("records/_decode_answers path", "_decode_answers" in rec_src)
check_true("records/no temp_bucket divide", "temp_bucket" not in rec_src)
check_true("records/_option_logits", "_option_logits" in rec_src)
check_true("decode/uses _option_logits", "_option_logits" in inspect.getsource(Agent._decode_answers))
sys_one = inspect.getsource(Agent.system_one)
check_true("system_one/uses predict_batch", "predict_batch" in sys_one)
check_true("system_one/does not fit", "fit_temperature" not in sys_one)
batch_src = inspect.getsource(Agent.predict_batch)
check_true("predict_batch/_encode_state", "_encode_state" in batch_src)
check_true("predict_batch/_decode_answers", "_decode_answers" in batch_src)


# --------------------------------------------------------------- labeled records use raw logits
class _RecAgent:
    """Stand-in encoder: real question checks, fake forward, temperatures that must not scale logits."""

    temperature = [9.0, 9.0, 9.0]
    temperature_by_options = {"choice:2": 9.0}
    tok = type("Tok", (), {"pad_token_id": 0})()
    calls = []

    def _check_question(self, qid, qdef):
        Agent._check_question(qid, qdef)

    def _to_internal(self, qdef):
        return Agent._to_internal(qdef)

    def _encode_state(self, state, ids, internal):
        self.calls.append(state)
        items = []
        for qid in ids:
            q = internal[qid]
            if q["t"] == "noul":
                k = 2
            else:
                k = len(q["crit"])
            items.append({"ids": [4, 5, 6], "markers": list(range(k)), "qtype": QTYPES[q["t"]]})
        return items

    def _forward(self, batch):
        n = int(batch["input_ids"].shape[0])
        kmax = int(batch["marker_pos"].shape[1])
        logits = np.full((n, kmax), 4.0, dtype=np.float32)
        for i in range(n):
            logits[i, 0] = 1.0 + i
        act = np.zeros((n, 2), dtype=np.float32)
        return logits, act


rec_agent = _RecAgent()
questions = {
    "c": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]},
    "n": {"type": "noul", "instructions": "yes?"},
}
targets = {
    "c": np.array([1.0, 0.0], dtype=np.float32),
    "n": np.array([0.0, 1.0], dtype=np.float32),
}
labeled = records_from_labeled(rec_agent, [("hello", questions, targets)])
check("records/state reaches _encode_state", rec_agent.calls, ["hello"])
check("records/choice qtype", labeled[0][0], QTYPES["choice"])
check("records/choice k", labeled[0][3], 2)
check("records/choice logits unscaled", labeled[0][1].tolist(), [1.0, 4.0])
check("records/choice target", labeled[0][2].tolist(), [1.0, 0.0])
check("records/noul qtype", labeled[1][0], QTYPES["noul"])
check("records/noul k", labeled[1][3], 2)
check("records/noul logits unscaled", labeled[1][1].tolist(), [2.0, 4.0])


# public exports stay lazy: importing the package must not import torch for these names
import laya as _laya  # noqa: E402
for name in ("fit_temperatures", "fit_one_temperature", "fit_temperature_map"):
    check_true("export/%s in __all__" % name, name in _laya.__all__)
    check_true("export/%s attr" % name, hasattr(_laya, name))
    check(
        "export/%s lazy table" % name,
        _laya._LAZY_ATTRS.get(name),
        (".calibrate", name),
    )


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
if not FAIL:
    print("synthetic ECE before=%.4f after=%.4f  choice:2 T=%.3f  noul:2 T=%.3f  choice:11+ T=%.3f" % (
        report["ece_before"], report["ece_after"],
        fitted["temperature_by_options"]["choice:2"],
        fitted["temperature_by_options"]["noul:2"],
        fitted["temperature_by_options"]["choice:11+"],
    ))
sys.exit(1 if FAIL else 0)
