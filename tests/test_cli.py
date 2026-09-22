"""CLI tests: argument parsing and output formatting only, no model weights loaded."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.cli import (  # noqa: E402
    _split_criteria_options,
    _split_levels,
    build_question,
    format_answer,
    format_result,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def ns(**kwargs):
    """Build an argparse-like namespace with CLI defaults filled in."""
    defaults = {
        "type": "choice",
        "instructions": None,
        "options": None,
        "levels": None,
        "model": None,
        "lang": None,
        "file": None,
        "text": None,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ------------------------------------------------------------------ option parsing
check("split/options with desc",
      _split_criteria_options("billing:invoices,payments;technical:bugs;sales:pricing"),
      {"billing": "invoices,payments", "technical": "bugs", "sales": "pricing"})
check("split/options without desc",
      _split_criteria_options("billing;technical;sales"),
      {"billing": None, "technical": None, "sales": None})
check("split/options ignores empties",
      _split_criteria_options("billing:x;;technical:y;"),
      {"billing": "x", "technical": "y"})
check("split/options strips whitespace",
      _split_criteria_options(" a : one ; b "),
      {"a": "one", "b": None})

check("split/levels", _split_levels("not urgent, soon, critical"),
      ["not urgent", "soon", "critical"])
check("split/levels drops empties", _split_levels("a,, b ,"),
      ["a", "b"])

# ------------------------------------------------------------------ question building
check("build/choice",
      build_question(ns(type="choice", instructions="which?",
                        options="a:one;b:two")),
      {"q1": {"type": "choice", "instructions": "which?", "criteria": {"a": "one", "b": "two"}}})

check("build/score",
      build_question(ns(type="score", instructions="how much?", levels="low,high")),
      {"q1": {"type": "score", "instructions": "how much?", "criteria": ["low", "high"]}})

check("build/noul default instruction",
      build_question(ns(type="noul")),
      {"q1": {"type": "noul", "instructions": "Is this true?"}})

check("build/noul keeps instruction",
      build_question(ns(type="noul", instructions="does it threaten?")),
      {"q1": {"type": "noul", "instructions": "does it threaten?"}})


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


check("build/choice needs options",
      _raises_value_error(lambda: build_question(ns(type="choice", instructions="x"))), True)
check("build/choice needs two options",
      _raises_value_error(lambda: build_question(ns(type="choice", instructions="x", options="a"))), True)
check("build/score needs levels",
      _raises_value_error(lambda: build_question(ns(type="score", instructions="x"))), True)

# ------------------------------------------------------------------ formatting
def _contains(name, haystack, *needles):
    missing = [n for n in needles if n not in haystack]
    if missing:
        FAIL.append("%s: missing %r in %r" % (name, missing, haystack))
    else:
        PASS.append(name)


_choice_out = format_answer("q1", {"type": "choice", "choice": "billing",
                                   "probabilities": {"billing": 0.9, "sales": 0.1},
                                   "confidence": 0.9})
_contains("format/choice marks winner", _choice_out, "q1 -> billing *", "confidence 90.0%")
_contains("format/choice shows both percentages", _choice_out, "90.0%", "10.0%")

_score_out = format_answer("q1", {"type": "score", "score": 1.5,
                                  "legend": {"0": "low", "1": "high"},
                                  "probabilities": {"0": 0.5, "1": 0.5},
                                  "confidence": 0.5})
_contains("format/score shows expected score", _score_out, "expected score 1.50", "confidence 50.0%")
_contains("format/score shows levels", _score_out, "low", "high")

_noul_out = format_answer("q1", {"type": "noul", "noul": 0.7, "confidence": 0.7})
_contains("format/noul shows probability", _noul_out, "P(true) = 70.0%", "confidence 70.0%")

check("format/result includes routing",
      format_result({"routing": {"model": "english", "reason": "English Latin text"},
                     "answers": {"q1": {"type": "noul", "noul": 0.5, "confidence": 0.5}}}),
      "route: english (English Latin text)\nq1 -> P(true) = 50.0%  (confidence 50.0%)")

# ------------------------------------------------------------------ report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all CLI tests passed")
sys.exit(1 if FAIL else 0)
