"""API-stability guard for laya.evals: an accidental rename or a torch import fails here.

Run: python tests/test_evals_api.py
"""
import dataclasses
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import evals  # noqa: E402

PASS, FAIL = [], []


def check(name, got, expected):
    ok = got == expected
    (PASS if ok else FAIL).append(name)
    print(("  ok   " if ok else "  FAIL ") + name, "" if ok else "(got %r, expected %r)" % (got, expected))


def check_true(name, condition):
    check(name, bool(condition), True)


# --------------------------------------------------------------- fields
check("Example fields",
      [f.name for f in dataclasses.fields(evals.Example)],
      ["state", "questions", "expected", "tags", "language", "model"])
check("EvalReport fields",
      [f.name for f in dataclasses.fields(evals.EvalReport)],
      ["config", "overall", "slices", "cases"])

# --------------------------------------------------------------- evaluators
check("default evaluator names",
      sorted(e.name for e in evals.default_evaluators()),
      ["choice_accuracy", "mean_confidence", "noul_accuracy", "score_mae"])
check_true("ScoreWithin names its tolerance", evals.ScoreWithin(0.25).name == "score_within_0.25")

# --------------------------------------------------------------- exports / callables
for name in ("Dataset", "Example", "EvalError", "EvalReport", "evaluate", "ece", "assert_regression"):
    check_true("laya.evals.%s exists" % name, hasattr(evals, name))

# --------------------------------------------------------------- torch stays out of import
probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; import laya.evals; sys.exit(1 if 'torch' in sys.modules else 0)"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
check("import laya.evals does not import torch", probe.returncode, 0)

# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for name in FAIL:
    print("  FAIL " + name)
if not FAIL:
    print("all eval API checks passed")
sys.exit(1 if FAIL else 0)
