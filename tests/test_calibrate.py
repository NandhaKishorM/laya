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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import Agent, load  # noqa: E402
from laya.calibrate import (  # noqa: E402
    CALIBRATION_VERSION,
    ECE_HOLDOUT_FRAC,
    MIN_BUCKET_N,
    MIN_TYPE_N,
    MIN_BINNING_BUCKET_N,
    _ece_report,
    _ece_split,
    _iter_records,
    _softmax,
    apply_binning_map,
    apply_calibration_payload,
    fit_binning_map,
    calibration_payload,
    fit_one_temperature,
    fit_temperature_map,
    fit_temperatures,
    records_from_labeled,
)
from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, ece_score, temp_bucket  # noqa: E402
import laya.calibrate as _calibrate  # noqa: E402
import laya.agent as _agent_module  # noqa: E402

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

def _refuses_records(name, records, fragment):
    try:
        fit_temperature_map(records)
    except ValueError as exc:
        check_true("records/" + name, fragment in str(exc), str(exc))
    except Exception as exc:
        FAIL.append("records/%s: raised %r, want ValueError" % (name, exc))
    else:
        FAIL.append("records/%s: invalid input was accepted" % name)


for name, record, field in (
    ("scalar record", 3, "record"),
    ("wrong record length", (0, [1, 2]), "record"),
    ("bool type", (True, [1, 2], [1, 0]), "qtype"),
    ("fractional type", (0.9, [1, 2], [1, 0]), "qtype"),
    ("unknown type", (3, [1, 2], [1, 0]), "qtype"),
    ("negative type", (-1, [1, 2], [1, 0]), "qtype"),
    ("bool width", (0, [1, 2], [1, 0], True), "k"),
    ("null width", (0, [1, 2], [1, 0], None), "k"),
    ("fractional width", (0, [1, 2], [1, 0], 1.9), "k"),
    ("zero width", (0, [1, 2], [1, 0], 0), "k"),
    ("oversized width", (0, [1, 2], [1, 0], 3), "k"),
    ("empty vectors", (0, [], []), "k"),
    ("matrix logits", (0, [[1, 2]], [1, 0]), "logits"),
    ("matrix target", (0, [1, 2], [[1, 0]]), "target"),
    ("text logits", (0, ["1", "2"], [1, 0]), "logits"),
    ("nan logits", (0, [np.nan, 2], [1, 0]), "logits"),
    ("infinite target", (0, [1, 2], [np.inf, 0]), "target"),
    ("unequal vectors", (0, [1, 2], [1]), "same length"),
    ("negative target", (0, [1, 2], [-0.1, 1.1]), "probability"),
    ("unnormalized target", (0, [1, 2], [1, 1]), "probability"),
):
    _refuses_records(name, [record], field)

soft_record = _iter_records([(np.int64(0), [1, 2, 0], [0.3, 0.7, 0], np.int64(2))])[0]
check("records/numpy integer and explicit padding", soft_record[3], 2)
check_true("records/soft target kept", np.allclose(soft_record[2], [0.3, 0.7]))
check_true("records/soft target fits", np.isfinite(fit_one_temperature([(soft_record[1], soft_record[2])], min_n=1)))
check("records/empty dataset still neutral", fit_temperature_map([])["temperature"], [1.0] * 3)
for name, pairs, min_n in (
    ("mismatched lengths", [([1, 2], [1])], None),
    ("invalid below sample floor", [([np.nan, 2], [1, 0])], None),
    ("invalid min_n", [], 0),
):
    try:
        fit_one_temperature(pairs, min_n=min_n)
    except ValueError:
        PASS.append("pairs/" + name)
    else:
        FAIL.append("pairs/%s: invalid input was accepted" % name)

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


# --------------------------------------------------------------- failed saves preserve the old map
with tempfile.TemporaryDirectory() as atomic_dir:
    atomic_path = Path(atomic_dir) / "calibration.json"
    agent.save_calibration(atomic_path)
    original_bytes = atomic_path.read_bytes()
    original_replace = os.replace

    def _partial_dump(body, stream, **kwargs):
        stream.write('{"temperature": [')
        raise OSError("injected write failure")

    for failure, context in (
        ("write", patch.object(_agent_module.json, "dump", _partial_dump)),
        ("sync", patch.object(_agent_module.os, "fsync", side_effect=OSError("injected sync failure"))),
        ("replace", patch.object(_agent_module.os, "replace", side_effect=OSError("injected replace failure"))),
    ):
        with context:
            try:
                agent.save_calibration(atomic_path)
            except OSError as exc:
                check_true("atomic/%s error propagates" % failure, "injected" in str(exc))
            else:
                FAIL.append("atomic/%s error did not propagate" % failure)
        check("atomic/%s keeps prior bytes" % failure, atomic_path.read_bytes(), original_bytes)
        check("atomic/%s cleans temp" % failure, sorted(os.listdir(atomic_dir)), ["calibration.json"])

    missing_path = Path(atomic_dir) / "new.json"
    with patch.object(_agent_module.json, "dump", _partial_dump):
        try:
            agent.save_calibration(missing_path)
        except OSError:
            pass
        else:
            FAIL.append("atomic/first-save error did not propagate")
    check_true("atomic/failed first save leaves no destination", not missing_path.exists())
    check("atomic/failed first save cleans temp", sorted(os.listdir(atomic_dir)), ["calibration.json"])

    # Successful updates keep the bytes/format and the existing readers' permissions.
    agent.save_calibration(atomic_path)
    check("atomic/success preserves JSON format", atomic_path.read_bytes(), original_bytes)
    if os.name != "nt":
        os.chmod(atomic_path, 0o640)
        agent.save_calibration(atomic_path)
        check("atomic/preserves mode", atomic_path.stat().st_mode & 0o777, 0o640)
        link_path = Path(atomic_dir) / "linked.json"
        link_path.symlink_to(atomic_path.name)
        agent.save_calibration(link_path)
        check_true("atomic/keeps symlink", link_path.is_symlink())
        check("atomic/writes symlink target", atomic_path.read_bytes(), original_bytes)
        link_path.unlink()

    rival = Agent.__new__(Agent)
    rival.temperature = [1.1, 1.2, 1.3]
    rival.temperature_by_options = {"choice:2": 1.4}
    for simulate_lost_race in (False, True):
        barrier = Barrier(2)

        def _together_replace(src, dst):
            rank = barrier.wait(timeout=10)
            if simulate_lost_race and rank == 0:
                error = PermissionError("injected NTFS replacement race")
                error.winerror = 5
                raise error
            original_replace(src, dst)

        succeeded = 0
        denied = 0
        with patch.object(_agent_module.os, "replace", _together_replace):
            with ThreadPoolExecutor(max_workers=2) as pool:
                writes = [pool.submit(writer.save_calibration, atomic_path) for writer in (agent, rival)]
                for write in writes:
                    try:
                        write.result()
                    except PermissionError as exc:
                        # NTFS may deny one simultaneous replacement; other errors still fail.
                        if getattr(exc, "winerror", None) != 5:
                            raise
                        denied += 1
                    else:
                        succeeded += 1
        label = "injected race" if simulate_lost_race else "concurrent writers"
        check_true("atomic/%s has a successful writer" % label, succeeded >= 1)
        if simulate_lost_race:
            check("atomic/injected race exercises denial", denied, 1)
        concurrent_payload = json.loads(atomic_path.read_text())
        check_true("atomic/%s leave one complete payload" % label,
                   concurrent_payload in (payload, calibration_payload(rival.temperature, rival.temperature_by_options)))
        check("atomic/%s clean temps" % label, sorted(os.listdir(atomic_dir)), ["calibration.json"])

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

# Optional histogram-binning map round-trips through the payload and lands on the agent.
bm = {"choice:2": {"bins": 2, "values": [0.9, 0.2]}}
with_bm = calibration_payload([1.2, 1.1, 1.3], {"choice:2": 1.4}, binning_map=bm)
check("payload/no-binning omission", "binning_map" not in calibration_payload([1.2, 1.1, 1.3], {}), True)
check_true("payload/binning key present", "binning_map" in with_bm, with_bm.keys())
apply_calibration_payload(stub, with_bm)
check("stub/binning_map", stub.binning_map, bm)
stub_no_bm = type("Stub", (), {})()
apply_calibration_payload(stub_no_bm, {"temperature": [1.0, 1.0, 1.0]})
check("stub/binning_map cleared when absent", stub_no_bm.binning_map, None)

# The same `_decode_answers` stub test_batch uses proves the map is selectable: no map,
# the temperature-scaled confidence; a map on this bucket, the remapped one.
bin_decoder = Agent.__new__(Agent)
bin_decoder.temperature = [1.0, 1.0, 1.0]
bin_decoder.temperature_by_options = {}
bin_ids = ["pick"]
bin_internal = {"pick": {"t": "choice", "crit": {"left": "left", "right": "right"}}}
bin_items = [{"markers": [0, 1]}]
bin_logits = np.log([[0.5, 0.5]]) * 1.0
bin_act = np.array([[0.2, 0.8]])
plain = bin_decoder._decode_answers(bin_logits, bin_act, bin_items, bin_ids, bin_internal, 0)
check("decode/no binning keeps scaled confidence", plain["pick"]["answer_confidence"], 0.5)
bin_decoder.binning_map = bm
remapped = bin_decoder._decode_answers(bin_logits, bin_act, bin_items, bin_ids, bin_internal, 0)
check("decode/binning remaps confidence", remapped["pick"]["answer_confidence"], 0.2)

# Q1: the fitter bins full-precision answer_confidence; the runtime must bin the same
# value and only round the final public field. Boundary case: a max(p) just below a
# histogram boundary rounds to the boundary, and the rounded value would pick the wrong bin.
boundary_bins = {"noul:2": {"bins": 20, "values": [round(0.05 * i, 2) for i in range(20)]}}
bounce_target = 0.9499995  # just below bin 19's start at 0.95
boundary_logits = np.log([[1.0 - bounce_target, bounce_target]]) * 1.0
boundary_act = np.array([[0.01, 0.99]])
boundary_ids = ["flag"]
boundary_internal = {"flag": {"t": "noul", "crit": None}}
boundary_items = [{"markers": [0, 1]}]

agent_boundary = Agent.__new__(Agent)
agent_boundary.temperature = [1.0, 1.0, 1.0]
agent_boundary.temperature_by_options = {}
agent_boundary.binning_map = boundary_bins
agent_decoded = agent_boundary._decode_answers(
    boundary_logits, boundary_act, boundary_items, boundary_ids, boundary_internal, 0
)
check(
    "decode/binning uses unrounded value (agent)",
    agent_decoded["flag"]["answer_confidence"],
    boundary_bins["noul:2"]["values"][18],
)

from laya.onnx_agent import ONNXAgent  # noqa: E402

onnx_boundary = ONNXAgent.__new__(ONNXAgent)
onnx_boundary.temperature = [1.0, 1.0, 1.0]
onnx_boundary.temperature_by_options = {}
onnx_boundary.binning_map = boundary_bins
onnx_decoded = onnx_boundary._decode_answers(
    boundary_logits, boundary_act, boundary_items, boundary_ids, boundary_internal, 0
)
check(
    "decode/binning uses unrounded value (onnx)",
    onnx_decoded["flag"]["answer_confidence"],
    agent_decoded["flag"]["answer_confidence"],
)

# Q3: cross-backend selectability — the same installed map must move both decodes to the same value.
onnx_select = ONNXAgent.__new__(ONNXAgent)
onnx_select.temperature = [1.0, 1.0, 1.0]
onnx_select.temperature_by_options = {}
onnx_select_ids = ["flag"]
onnx_select_internal = {"flag": {"t": "noul", "crit": None}}
select_logits = np.log([[0.8, 0.2]]) * 1.0
onnx_no_map = onnx_select._decode_answers(select_logits, bin_act, bin_items, onnx_select_ids, onnx_select_internal, 0)
check("onnx/no binning keeps scaled confidence", onnx_no_map["flag"]["answer_confidence"], 0.8)
onnx_select.binning_map = {"noul:2": {"bins": 2, "values": [0.95, 0.05]}}
onnx_with_map = onnx_select._decode_answers(select_logits, bin_act, bin_items, onnx_select_ids, onnx_select_internal, 0)
check("onnx/binning remaps confidence", onnx_with_map["flag"]["answer_confidence"], 0.05)

agent_bin = Agent.__new__(Agent)
agent_bin.temperature = [1.0, 1.0, 1.0]
agent_bin.temperature_by_options = {}
agent_bin.binning_map = {"noul:2": {"bins": 2, "values": [0.95, 0.05]}}
agent_binned = agent_bin._decode_answers(select_logits, bin_act, bin_items, onnx_select_ids, onnx_select_internal, 0)
check("agent/binning remaps confidence", agent_binned["flag"]["answer_confidence"], 0.05)
agent_bin.lang_temperatures = {"zh": {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}}
agent_lang = agent_bin._decode_answers(select_logits, bin_act, bin_items, onnx_select_ids, onnx_select_internal, 0, lang="zh")
check("agent/lang override skips binning", agent_lang["flag"]["answer_confidence"], 0.8)
onnx_lang = ONNXAgent.__new__(ONNXAgent)
onnx_lang.temperature = [1.0, 1.0, 1.0]
onnx_lang.temperature_by_options = {}
onnx_lang.binning_map = {"noul:2": {"bins": 2, "values": [0.95, 0.05]}}
onnx_lang.lang_temperatures = {"zh": {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}}
onnx_runtime = onnx_lang._decode_answers(select_logits, bin_act, bin_items, onnx_select_ids, onnx_select_internal, 0, lang="zh")
check("onnx/lang override skips binning", onnx_runtime["flag"]["answer_confidence"], 0.8)

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


# --------------------------------------------------------------- payload shape
# The values above are tolerated and clamped; the *shape* is not negotiable, and the
# check is the line above the tolerance: "[3 floats]" means a list of three, not
# anything with a length of three. JSON hands back int, str, list and dict for the
# mistakes below just as happily as the right shapes -- a dict or a string of length 3
# used to pass the length check and install its *keys* as temperatures, and a scalar
# temperature or bucket map crashed with a raw TypeError instead of the ValueError
# the function's own message promises.


def _refuses(name, payload, fragment=None):
    try:
        apply_calibration_payload(stub, payload)
    except ValueError as exc:
        check_true(name, fragment is None or fragment in str(exc), str(exc))
        return
    except BaseException as exc:  # noqa: BLE001 -- the old TypeError/AttributeError
        FAIL.append("%s: raised %r, want ValueError" % (name, exc))
        return
    FAIL.append("%s: accepted %r" % (name, payload))


_refuses("shape/non-object payload", ["temperature", [1.0, 1.0, 1.0]], "must be an object")
_refuses("shape/binning not object", {"temperature": [1.0, 1.0, 1.0], "binning_map": [1, 2]}, "binning_map")
_refuses("shape/binning wrong entry", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": 1}}, "bins")
_refuses("shape/binning values length", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 2, "values": [0.5]}}}, "values")
_refuses("shape/binning value above 1", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [1.5]}}}, "[0, 1]")
_refuses("shape/binning value below 0", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [-0.1]}}}, "[0, 1]")
_refuses("shape/binning NaN", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [float("nan")]}}}, "finite")
_refuses("shape/binning Infinity", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [float("inf")]}}}, "finite")
_refuses("shape/binning bool value", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [True]}}}, "number")
_refuses("shape/binning string value", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": ["0.5"]}}}, "number")
_refuses("shape/binning null value", {"temperature": [1.0, 1.0, 1.0], "binning_map": {"a": {"bins": 1, "values": [None]}}}, "number")
_refuses("shape/scalar temperature", {"temperature": 5}, "[3 floats]")
_refuses("shape/string temperature", {"temperature": "abc"}, "[3 floats]")
_refuses("shape/dict temperature", {"temperature": {"a": 1, "b": 2, "c": 3}}, "[3 floats]")
_refuses("shape/wrong-length temperature", {"temperature": [1.0, 1.0]}, "[3 floats]")
_refuses("shape/string version", {"temperature": [1.0] * 3, "version": "x"},
         "version must be an integer")
_refuses("shape/bool version", {"temperature": [1.0] * 3, "version": True},
         "version must be an integer")
_refuses("shape/object version", {"temperature": [1.0] * 3, "version": {"v": 2}},
         "version must be an integer")
_refuses("shape/scalar by_options",
         {"temperature": [1.0] * 3, "temperature_by_options": 5}, "temperature_by_options")
_refuses("shape/list by_options",
         {"temperature": [1.0] * 3, "temperature_by_options": [["choice:2", 1.5]]},
         "temperature_by_options")
# and a numeric string version still parses, as it did before the check existed
version_ok = type("Stub", (), {})()
apply_calibration_payload(version_ok, {"temperature": [1.1, 1.2, 1.3], "version": "1"})
check("shape/numeric string version still loads", version_ok.temperature, [1.1, 1.2, 1.3])


# --------------------------------------------------------------- the load contract, from the code
# The page used to promise two installed fields and "the three mistakes below" while the function
# installs three (`binning_map` included) and refuses a dozen shapes. The prose is the only place a
# reader learns that a bad *temperature* is clamped and a bad *binning value* is refused, so it is
# held to the code here: the field vocabulary is read out of the `raise ValueError` messages by AST
# and compared, both ways, to the fields the docstring names.
import ast  # noqa: E402

FIELD_WORDS = ("binning_map", "temperature_by_options", "version", "temperature", "bins", "values")


def _fields_in(text):
    """Which of the refused fields `text` names. Longest first, and a matched span is removed, so
    `temperature_by_options` counts once and never also reports `temperature`."""
    found, rest = set(), text
    for word in sorted(FIELD_WORDS, key=len, reverse=True):
        if word in rest:
            found.add(word)
            rest = rest.replace(word, "\x00")
    return found


_calibrate_tree = ast.parse(inspect.getsource(_calibrate))
_apply_fn = next(n for n in ast.walk(_calibrate_tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "apply_calibration_payload")


def _refusal_message(node):
    """The literal text of a raised `ValueError`, ignoring its `%` arguments."""
    exc = node.exc
    arg = exc.args[0] if isinstance(exc, ast.Call) and exc.args else exc
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
        arg = arg.left
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


_refusal_texts = [m for m in (_refusal_message(n) for n in ast.walk(_apply_fn) if isinstance(n, ast.Raise))
                  if m and m.startswith("calibration JSON")]
check_true("contract/the AST scan reaches every refusal (_refuses drives these)",
           len(_refusal_texts) >= 11, "only %d refusal messages found" % len(_refusal_texts))
refused_fields = set().union(*[_fields_in(m) for m in _refusal_texts]) if _refusal_texts else set()
check("contract/fields the code refuses on", sorted(refused_fields), sorted(FIELD_WORDS))

_apply_doc = " ".join(apply_calibration_payload.__doc__.split())
check("contract/the docstring names every field the code refuses on",
      sorted(refused_fields - _fields_in(_apply_doc)), [])
check_true("contract/the docstring names no field the code does not refuse on",
           not _fields_in(_apply_doc) - refused_fields,
           sorted(_fields_in(_apply_doc) - refused_fields))
_summary_line = apply_calibration_payload.__doc__.strip().splitlines()[0]
check("contract/the summary line names all three installed fields",
      sorted(w for w in ("temperature", "temperature_by_options", "binning_map") if w in _summary_line),
      ["binning_map", "temperature", "temperature_by_options"])
check_true("contract/the summary line calls it a copy of three, not two",
           _summary_line.startswith("Copy `temperature`, `temperature_by_options` and `binning_map`"),
           _summary_line)
# The split has to be stated *as* the split: "clamp" and "refuse" both appear elsewhere in this
# page whatever it says, so the check reads the window that follows the contrast it is claiming.
_contrast = "opposite value policies"
_policy_window = (_apply_doc.split(_contrast)[-1][:500] if _contrast in _apply_doc else "")
check_true("contract/the docstring states the clamp-vs-refuse split, not one policy for both",
           "clamp" in _policy_window and "refus" in _policy_window, _apply_doc[:200])
check_true("contract/the docstring says a keyless file clears the map, not leaves it",
           "installs `None`" in _apply_doc and "clears" in _apply_doc, _apply_doc[:300])
# The pre-fix wording, banned so the count claim cannot come back.
check_true("contract/the docstring makes no count claim about the refusals",
           "three mistakes" not in _apply_doc and " two fields" not in _apply_doc, _apply_doc[:200])

# The same file, described by the agent that reads it: `Agent.save_calibration` says what it writes
# and `Agent.load_calibration` says what it installs, and both named only temperatures. Read from the
# module's own source through AST, so the claim is checked against the words on the page rather than
# against a docstring a `-O` run has stripped out.
def _method_doc(tree, classname, method):
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == classname), None)
    fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method), None) if cls else None
    return " ".join(ast.get_docstring(fn).split()) if fn and ast.get_docstring(fn) else ""


for _name, _doc in (
    ("Agent.load_calibration", _method_doc(ast.parse(inspect.getsource(_agent_module)), "Agent", "load_calibration")),
    ("Agent.save_calibration", _method_doc(ast.parse(inspect.getsource(_agent_module)), "Agent", "save_calibration")),
):
    check_true("contract/%s names the binning map it installs" % _name, "binning_map" in _doc, _doc[:160])
check_true("contract/Agent.load_calibration says the file is the whole state",
           "whole calibration state" in _method_doc(
               ast.parse(inspect.getsource(_agent_module)), "Agent", "load_calibration"), "")

# And the semantics those words claim, driven.
_binned = type("Stub", (), {})()
apply_calibration_payload(_binned, {"temperature": [1.0] * 3,
                                    "binning_map": {"choice:2": {"bins": 2, "values": [0.25, 0.75]}}})
check("install/binning_map is installed onto the object", _binned.binning_map,
      {"choice:2": {"bins": 2, "values": [0.25, 0.75]}})

_cleared = type("Stub", (), {})()
_cleared.binning_map = {"choice:2": {"bins": 1, "values": [0.5]}}
apply_calibration_payload(_cleared, {"temperature": [1.0] * 3})
check("install/a file with no binning_map key clears the installed map", _cleared.binning_map, None)

_explicit_null = type("Stub", (), {})()
_explicit_null.binning_map = {"choice:2": {"bins": 1, "values": [0.5]}}
apply_calibration_payload(_explicit_null, {"temperature": [1.0] * 3, "binning_map": None})
check("install/an explicit null binning_map clears it too", _explicit_null.binning_map, None)

# `None` is "no map"; `{}` is "a map that recalibrates nothing" and survives the round trip as
# `{}`, which is what `Agent.fit_binning` returns when no bucket reaches the floor.
check("install/no map omits the key", "binning_map" in calibration_payload([1.0] * 3, {}, None), False)
_empty_saved = calibration_payload([1.0] * 3, {}, binning_map={})
check("install/an empty map is written, not omitted", _empty_saved.get("binning_map"), {})
_empty_back = type("Stub", (), {})()
apply_calibration_payload(_empty_back, _empty_saved)
check("install/an empty map round-trips as empty, not as None", _empty_back.binning_map, {})


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


# --------------------------------------------------------------- histogram-binning recalibration
check("binning/floor", MIN_BINNING_BUCKET_N, 200)
bmap = fit_binning_map(mixed, fitted["temperature"], fitted["temperature_by_options"], bins=10)
check_true("binning/keyed by temp_bucket", {"choice:2", "noul:2", "choice:11+", "choice:3-5"} <= set(bmap), set(bmap))
check_true("binning/omits sub-floor bucket (score:3-5, n=5)", "score:3-5" not in bmap, set(bmap))
_entry = bmap["choice:2"]
check("binning/records bins", _entry["bins"], 10)
check("binning/one value per bin", len(_entry["values"]), 10)
check_true("binning/values are probabilities", all(0.0 <= v <= 1.0 for v in _entry["values"]), _entry["values"])

# apply: a bucket with no map returns the confidence unchanged; edges clamp into range.
check("binning/apply unknown bucket is identity", apply_binning_map(0.73, "choice:6-10", bmap), 0.73)
check_true("binning/apply in range for conf=1.0", 0.0 <= apply_binning_map(1.0, "choice:2", bmap) <= 1.0)
check_true("binning/apply in range for conf=0.0", 0.0 <= apply_binning_map(0.0, "choice:2", bmap) <= 1.0)

# Binning reduces ECE where temperature alone cannot: recalibrate the temperature-scaled
# confidences of a bucket and compare ECE on the same data.
_qt = QTYPES["choice"]
_tscale = fitted["temperature_by_options"].get("choice:2", fitted["temperature"][_qt])
_cal_conf, _corr = [], []
for qt, z, t, k in _iter_records(choice_k2):
    p = _softmax(z[:k], _tscale)
    _cal_conf.append(float(p.max()))
    _corr.append(bool(int(p.argmax()) == int(np.argmax(t[:k]))))
_binned = [apply_binning_map(c, "choice:2", bmap) for c in _cal_conf]
_ece_temp = ece_score(np.asarray(_cal_conf), np.asarray(_corr))
_ece_binned = ece_score(np.asarray(_binned), np.asarray(_corr))
check_true("binning/lowers ECE vs temperature on the bucket",
           _ece_binned < _ece_temp, "temp=%.4f binned=%.4f" % (_ece_temp, _ece_binned))

try:
    fit_binning_map(mixed, fitted["temperature"], fitted["temperature_by_options"], bins=0)
    check_true("binning/bins must be >= 1", False, "no ValueError")
except ValueError:
    check_true("binning/bins must be >= 1", True)


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
