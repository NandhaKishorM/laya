"""Offline tests for the layered evaluation harness.

No checkpoint and no network: every function under test is pure, so this runs in
CI next to the other suites. The model-dependent paths (`score_cases`,
`run_language`) are exercised by `tests/test_local_e2e.py` when a checkpoint is
present.

Run: python research/eval/test_laya_eval.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.eval.laya_eval import (  # noqa: E402
    ECE_BINS, INSTRUCTIONS, N_OPTS, SEED, build_case, build_suite, ece, macro_f1,
    render_label, summarise, temperature_for,
)

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


# --------------------------------------------------------------- label rendering
check("render/underscores become spaces", render_label("alarm_set"), "alarm set")
check("render/dots become colon-space", render_label("news.»"), "news: »".replace("»", "»"))
check("render/query_definition", render_label("qa_definition"), "qa definition")
check("render/no underscores left", "_" in render_label("a_b_c"), False)
# must match bench_local.py:176 exactly
check("render/upstream parity", render_label("datetime_query"), "datetime query")


# ------------------------------------------------------------------ one case
_rng = random.Random(SEED)
_state, _q, _gold_idx = build_case("wake me up at five am", "alarm_set",
                                   ["a", "b", "c"], _rng, 4)
check("case/state key is utterance", list(_state), ["utterance"])
check("case/single question id", list(_q), ["intent"])
check("case/type is choice", _q["intent"]["type"], "choice")
check("case/instructions fixed", _q["intent"]["instructions"], INSTRUCTIONS)
check("case/option count", len(_q["intent"]["criteria"]), 4)
check("case/gold is present", "alarm_set" in _q["intent"]["criteria"], True)
check("case/gold index points at gold",
      list(_q["intent"]["criteria"])[_gold_idx], "alarm_set")


# ------------------------------------------------- determinism and seed parity
ROWS = [{"text": "t%d" % i, "label_text": "lab%d" % (i % 7)} for i in range(30)]
LABELS = sorted({r["label_text"] for r in ROWS})

a, b, c = build_suite(ROWS, LABELS, 10, 5, SEED)
d, e, f = build_suite(ROWS, LABELS, 10, 5, SEED)
check("suite/same seed same cases", a == d, True)
check("suite/same seed same gold", b == e, True)
check("suite/same seed same options", c == f, True)

g, h, i = build_suite(ROWS, LABELS, 10, 5, SEED + 1)
check_true("suite/different seed different options", c != i)

# a fresh Random(SEED) per language is what upstream does; two languages with the
# same row content must therefore produce identical option sets
ROWS_B = [dict(r) for r in ROWS]
_b1 = build_suite(ROWS, LABELS, 10, 5, SEED)[2]
_b2 = build_suite(ROWS_B, LABELS, 10, 5, SEED)[2]
check("suite/rng reset per language", _b1, _b2)

check("suite/per_lang caps cases", len(a), 10)
check("suite/gold within options",
      all(0 <= gi < len(opts) for gi, opts in zip(b, c)), True)
check("suite/every gold is in its option set",
      all(opts[gi] == ROWS[i]["label_text"] for i, (gi, opts) in enumerate(zip(b, c))), True)
check("suite/options are distinct",
      all(len(set(o)) == len(o) for o in c), True)
check("suite/n_opts respected", all(len(o) == 5 for o in c), True)

# more options requested than labels available: must not crash or duplicate
tiny = [{"text": "x", "label_text": "only"}]
_tc, _tg, _to = build_suite(tiny, ["only"], 1, 20, SEED)
check("suite/fewer labels than n_opts", len(_to[0]), 1)


# --------------------------------------------------------------------- metrics
import math  # noqa: E402

check_true("ece/empty is nan", math.isnan(ece([], [])))
check("ece/perfect on one bin", round(ece([0.95] * 50, [1.0] * 50), 4), 0.05)
check("ece/all wrong and confident", round(ece([0.95] * 50, [0.0] * 50), 4), 0.95)
check("ece/normalised by count", round(ece([0.9] * 100, [1.0] * 100), 4),
      round(ece([0.9] * 1000, [1.0] * 1000), 4))
check("ece/one sample", round(ece([1.0], [1.0]), 4), 0.0)

# Known boundary: the bin test is `conf > lo`, so conf == 0.0 falls in no bin and
# contributes nothing. `laya.common.ece_score` and `research/scripts/bench_local.py`
# bin identically, so this harness deliberately matches them rather than diverging --
# comparability with the published tables is the point. PR #39 addresses the same
# boundary in `laya.common`; if it lands, this harness should follow it.
check("ece/conf==0.0 is not binned (matches upstream)",
      round(ece([0.0, 0.0], [1.0, 1.0]), 4), 0.0)
check_true("ece/conf slightly above 0 IS binned",
           ece([1e-9, 1e-9], [1.0, 1.0]) > 0.0)

check_true("f1/empty is nan", math.isnan(macro_f1([], [])))
check("f1/perfect", macro_f1([0, 1, 2], [0, 1, 2]), 1.0)
check("f1/all wrong", round(macro_f1([0, 1], [1, 0]), 4), 0.0)
check("f1/disjoint labels", round(macro_f1([0], [1]), 4), 0.0)

_s = summarise([0.9, 0.8, 0.7, 0.6], [1.0, 1.0, 0.0, 0.0], [0, 1, 2, 3], [0, 1, 3, 2])
check("summary/n", _s["n"], 4)
check("summary/accuracy", _s["accuracy"], 0.5)
check("summary/mean_confidence", _s["mean_confidence"], 0.75)
check("summary/acc_at_50_coverage takes the confident half",
      _s["acc_at_50_coverage"], 1.0)
check("summary/empty", summarise([], [], [], []), {"n": 0})


# ------------------------------------------------------- temperature selection
class _FakeAgent:
    temperature_by_options = {"choice:11+": 0.5}
    temperature = [1.0, 1.0, 1.0]
    temperature_by_options_raw = {"choice:11+": 0.10058280825614929}
    temperature_raw = [1.0, 1.0, 1.0]


_fa = _FakeAgent()
from laya.common import QTYPES  # noqa: E402

check("temp/clamped path is used by default",
      temperature_for(_fa, QTYPES["choice"], 20), 0.5)
check("temp/raw path under unclamped",
      round(temperature_for(_fa, QTYPES["choice"], 20, unclamped=True), 6), 0.100583)
check("temp/falls back to the per-type list",
      temperature_for(_fa, QTYPES["choice"], 3), 1.0)
check("temp/raw falls back too",
      temperature_for(_fa, QTYPES["choice"], 3, unclamped=True), 1.0)


# ------------------------------------------------------------------- constants
check("const/seed matches upstream", SEED, 13)
check("const/n_opts matches upstream", N_OPTS, 20)
check("const/bins", ECE_BINS, 15)
check("const/instructions match bench_local.py",
      INSTRUCTIONS, "What is the user asking for in `utterance`?")


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
