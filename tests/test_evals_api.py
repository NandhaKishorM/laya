"""API-stability guard for laya.evals: an accidental rename or a torch import fails here.

Run: python tests/test_evals_api.py
"""
import argparse
import dataclasses
import inspect
import os
import re
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
for name in ("Dataset", "Example", "EvalError", "EvalReport", "evaluate", "ece", "assert_regression",
             "REPORT_SCHEMA", "questions_fingerprint", "file_fingerprint",
             "brier", "aurc", "selective_accuracy", "is_confidence_metric",
             # `_eval_policy` imports these three, so they are a contract between two modules in
             # this package, not internals -- a rename would break the release gate, not just a
             # caller.
             "is_coverage_metric", "coverage_definition_conflict",
             "COVERAGE_METRIC_DEFINITION"):
    check_true("laya.evals.%s exists" % name, hasattr(evals, name))

check("REPORT_SCHEMA", evals.REPORT_SCHEMA, "laya-evals-report/1")
check("the coverage-cut definition this code computes", evals.COVERAGE_METRIC_DEFINITION, 2)
check_true("EvalReport.comparable_to is part of the report contract",
           callable(getattr(evals.EvalReport, "comparable_to", None)))
check("EvalReport fields are unchanged", [f.name for f in dataclasses.fields(evals.EvalReport)],
      ["config", "overall", "slices", "cases"])

# --------------------------------------------------------------- evaluate's call signature
# `evaluate` is the harness entry point callers extend from their own scripts, so its parameter
# names and order are the contract. `sort_by_length` is core's #294 forward-pass grouping knob
# reaching a scored run through `laya-evals run --sort-by-length`; it is appended, never inserted,
# because a caller that passed `on_error` or `config` positionally must still mean the same thing.
# `min_confidence` extends the same rule: it is appended after `sort_by_length`, defaults to None
# (an unset abstention threshold means "score the raw argmax", which is what a pre-#361 run did),
# and sits after the batch size the calls that carry it can group inside.
_sig = inspect.signature(evals.evaluate)
check("evaluate signature", [p.name for p in _sig.parameters.values()],
      ["runner", "dataset", "evaluators", "batch_size", "on_error", "config",
       "sort_by_length", "min_confidence"])
check("evaluate: grouping defaults to off", _sig.parameters["sort_by_length"].default, False)
check("evaluate: abstention threshold defaults to unset",
      _sig.parameters["min_confidence"].default, None)
check_true("evaluate: the knob sits after the batch size it groups inside",
           list(_sig.parameters).index("batch_size")
           < list(_sig.parameters).index("sort_by_length"))
check_true("evaluate: the abstention knob sits after the batch size it may apply within",
           list(_sig.parameters).index("batch_size")
           < list(_sig.parameters).index("min_confidence"))

# The two batch entry points the CLI wires up: both have to take the knob by the same name, or a
# `--sort-by-length` run reports `sort_by_length_sent: false` for the surface it ships. The same
# is true for the abstention threshold; a `--min-confidence` run that silently dropped the
# argument would publish `report.config["timing"]["min_confidence"]` naming a threshold that was
# never applied -- the metrics stay identical (the gate flags, it does not overwrite the argmax),
# but the report itself would lie -- so the CLI runner shapes have to accept the kwarg under the
# same name the guard on `evaluate` checks.
from laya import evals_cli  # noqa: E402

for label, fn in (("RouterRunner.predict_batch", evals_cli.RouterRunner.predict_batch),
                  ("OnnxRunner.predict_batch", evals_cli.OnnxRunner.predict_batch)):
    params = inspect.signature(fn).parameters
    check_true("%s takes the grouping knob" % label, "sort_by_length" in params)
    check("%s defaults it to off" % label, params["sort_by_length"].default, False)
    check_true("%s takes the abstention threshold" % label, "min_confidence" in params)
    check("%s defaults the threshold to unset" % label,
          params["min_confidence"].default, None)

# The single-predict fallback path takes the threshold under the same name on both runner shapes,
# so a run without --batch-size still scores the gate rather than silently skipping it.
for label, fn in (("RouterRunner.predict", evals_cli.RouterRunner.predict),
                  ("OnnxRunner.predict", evals_cli.OnnxRunner.predict)):
    params = inspect.signature(fn).parameters
    check_true("%s takes the abstention threshold" % label, "min_confidence" in params)
    check("%s defaults the threshold to unset" % label,
          params["min_confidence"].default, None)

# A signature is not a wiring: a runner that declares the knob and drops it on the way through
# would pass both checks above. `OnnxRunner`'s call is gated in test_evals_onnx.py against a
# recording agent; this is the same witness for the router-shaped runner the CLI uses by default.
class _RecordingRouter:
    def __init__(self):
        self.calls = []

    def predict_batch(self, requests, batch_size=None, sort_by_length=False,
                      min_confidence=None):
        self.calls.append({"batch_size": batch_size, "sort_by_length": sort_by_length,
                           "min_confidence": min_confidence})
        return [{"model": "m", "answers": {}} for _ in requests]


_router = _RecordingRouter()

# --------------------------------------------------------------- torch stays out of import
probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; import laya.evals; sys.exit(1 if 'torch' in sys.modules else 0)"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
check("import laya.evals does not import torch", probe.returncode, 0)

# --------------------------------------------------------------- argument documentation
# `_build_parser` is what a caller reads before running anything: argparse prints each
# declaration's help into `--help`, and docs/evals.md is the page that explains the command. On
# `main` five of its 22 declarations carried no help at all -- `run --on-error` printed as bare
# `--on-error {fail,skip}`, `compare --baseline` printed bare while being the one argument that
# call requires, and `validate`/`run`'s `dataset` and `compare`'s `report` had nothing under them
# -- while the page described `on_error=skip`'s effect on the batch counters without ever naming
# the flag that sets it. These read the parser the CLI really builds, so a flag added tomorrow is
# held to the same three rules as `--on-error` is today.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Backslash continuations are folded first, so a flag on the second line of a multi-line example is
# still attributed to the subcommand that started it.
_PAGE = re.sub(r"\\\n\s*", " ",
               open(os.path.join(_ROOT, "docs", "evals.md"), encoding="utf-8").read())
# Greedy, so `--min-accuracy` is one token and never also yields `--min`.
_PAGE_FLAGS = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", _PAGE))


def _subparsers(parser):
    """{subcommand name: its parser}, read off the subparsers action the parser holds."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


_subs = _subparsers(evals_cli._build_parser())
check("laya-evals subcommands", sorted(_subs), ["compare", "evidence", "run", "validate"])

_unhelped, _unstated, _options = [], [], {}
for _sub, _parser in sorted(_subs.items()):
    for action in _parser._actions:
        if action.dest == "help":
            continue
        site = "%s %s" % (_sub, " ".join(action.option_strings) or action.dest)
        _options.setdefault(_sub, set()).update(action.option_strings)
        if not (action.help or "").strip():
            _unhelped.append(site)
        elif action.default not in (None, False):
            # argparse prints a flag's default nowhere unless its own help text does, so a default
            # that matters has to be said: `--on-error` is documented as a choice, and the run it
            # aborts rather than reports is the half a caller cannot guess. The word `default` and
            # the quoted value must share a clause, because a help string that simply names both
            # choices -- which is what `--on-error`'s does -- would otherwise still read as if the
            # other one were the fallback after a flip.
            value = re.escape(str(action.default))
            if not re.search(r"'%s'[^;:]*\bdefault|\bdefault\b[^;:]*'%s'" % (value, value),
                             action.help):
                _unstated.append("%s (default %r)" % (site, action.default))

check("every laya-evals argument carries a help string", sorted(_unhelped), [])
check("every laya-evals default is stated in its own help", sorted(_unstated), [])

_documented = sorted("%s %s" % (sub, opt) for sub, opts in _options.items() for opt in opts
                     if opt not in _PAGE_FLAGS)
check("every laya-evals option is named in docs/evals.md", _documented, [])

_advertised = []
for line in _PAGE.splitlines():
    match = re.search(r"laya-evals\s+([a-z]+)", line)
    if not match or match.group(1) not in _options:
        continue
    _advertised += ["%s %s" % (match.group(1), flag)
                    for flag in re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", line)
                    if flag not in _options[match.group(1)]]
check("docs/evals.md advertises only options the parser defines", sorted(_advertised), [])

# --------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for name in FAIL:
    print("  FAIL " + name)
if not FAIL:
    print("all eval API checks passed")
sys.exit(1 if FAIL else 0)
