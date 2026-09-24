"""Offline tests for verify_results.py.

No checkpoint and no network: every document is a small synthetic dict written
to a temp file. Covers format detection, cell comparison, tolerance and exit
codes, macro reporting, and language-set warnings.

Run: python research/eval/test_verify_results.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.eval import verify_results as vr  # noqa: E402

PASS, FAIL = [], []


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


def row(acc, f1, ece, conf, cov):
    return {"accuracy": acc, "macro_f1": f1, "ece": ece,
            "mean_confidence": conf, "acc_at_50_coverage": cov}


CELLS = {"en": row(0.82, 0.7876, 0.1382, 0.9582, 0.98),
         "de": row(0.70, 0.66, 0.20, 0.90, 0.90)}


def copy_cells(cells):
    return {lang: dict(r) for lang, r in cells.items()}


def eval_doc(cells):
    return {"config": {"device": "cpu"}, "report": copy_cells(cells)}


def sweep_doc(cells, key="english"):
    return {"part_a": {"by_model": {key: {"per_language": copy_cells(cells)}}}}


def clamped_doc(cells, branch="clamped_rerun"):
    return {"per_language": {lang: {branch: dict(r)} for lang, r in cells.items()}}


def write(doc):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh)
    return path


# --- format detection -------------------------------------------------------

cells, label = vr.extract(eval_doc(CELLS))
check("extract eval langs", sorted(cells), ["de", "en"])
check_true("extract eval label", "cpu" in label, label)

cells, label = vr.extract(sweep_doc(CELLS), model_key="english")
check("extract sweep langs", sorted(cells), ["de", "en"])
check_true("extract sweep label", "english" in label, label)

cells, label = vr.extract(clamped_doc(CELLS))
check("extract clamped default branch", sorted(cells), ["de", "en"])
check_true("extract clamped label", "clamped_rerun" in label, label)

both = {"per_language": {"en": {"committed": row(1, 1, 1, 1, 1),
                                "clamped_rerun": row(2, 2, 2, 2, 2)}}}
cells, _ = vr.extract(both, branch="committed")
check("extract clamped picks branch", cells["en"]["accuracy"], 1)


# --- comparison -------------------------------------------------------------

rows, misses, only_run, only_ref, langs = vr.compare(eval_doc(CELLS)["report"], CELLS, vr.METRICS, vr.DEFAULT_TOL)
check("identical: no misses", misses, 0)
check("identical: all exact", [r[1] for r in rows], [2] * len(vr.METRICS))
check("identical: langs", langs, ["de", "en"])

moved = eval_doc(CELLS)["report"]
moved["en"]["accuracy"] = 0.81
rows, misses, _, _, _ = vr.compare(moved, CELLS, vr.METRICS, vr.DEFAULT_TOL)
check("one drift: misses", misses, 1)
worst = {r[0]: (r[3], r[4]) for r in rows}
check("one drift: worst cell", worst["accuracy"], ("en", abs(0.82 - 0.81)))

within = eval_doc(CELLS)["report"]
within["de"]["ece"] = CELLS["de"]["ece"] + 4e-5  # below 5e-5
_, misses, _, _, _ = vr.compare(within, CELLS, vr.METRICS, vr.DEFAULT_TOL)
check("sub-tolerance diff passes", misses, 0)

run_extra = eval_doc(CELLS)["report"]
run_extra["fr"] = row(0.5, 0.5, 0.5, 0.5, 0.5)
_, _, only_run, only_ref, langs = vr.compare(run_extra, CELLS, vr.METRICS, vr.DEFAULT_TOL)
check("extra lang reported", only_run, ["fr"])
check("extra lang excluded", langs, ["de", "en"])
check("missing side empty", only_ref, [])


# --- CLI / exit codes -------------------------------------------------------

ok_run, ok_ref = write(eval_doc(CELLS)), write(sweep_doc(CELLS))
check("cli exit 0 on match", vr.main([ok_run, ok_ref]), 0)

bad_run = write(eval_doc({"en": row(0.50, 0.7876, 0.1382, 0.9582, 0.98),
                          "de": row(0.70, 0.66, 0.20, 0.90, 0.90)}))
check("cli exit 1 on drift", vr.main([bad_run, ok_ref]), 1)

clamped_path = write(clamped_doc(CELLS))
check("cli clamped branch", vr.main([ok_run, clamped_path]), 0)
check("cli clamped other branch drifts",
      vr.main([ok_run, write(clamped_doc({"en": row(0.82, 0.7876, 0.1789, 0.9989, 0.98),
                                          "de": row(0.70, 0.66, 0.20, 0.90, 0.90)}))]), 1)

multi = write(sweep_doc(CELLS, key="multilingual"))
check("cli model-key multilingual", vr.main([ok_run, multi, "--model-key", "multilingual"]), 0)

for p in (ok_run, ok_ref, bad_run, clamped_path, multi):
    os.unlink(p)


# --- summary ----------------------------------------------------------------

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("FAIL", f)
sys.exit(1 if FAIL else 0)
