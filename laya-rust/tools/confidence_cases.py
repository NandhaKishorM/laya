"""Synthetic inputs for the model-free abstention-gate probe, `routing/confidence_probe.json`.

`dump_routing_golden.build_confidence_probe` feeds each of these to Python's own
`laya/confidence.py` and records what it returned, so a port is checked against the shipping
code on the same inputs without loading a model. This list is SDK-neutral: it is the shared
corpus for the .NET and Rust ports, and each port keeps a verbatim copy.

Everything here is representable by a typed SDK answer (a choice with ordered probabilities, a
score with ordered probabilities, a noul with none), except the jev cases flagged `typed: false`,
which exist to pin Python's refusal of shapes a typed answer cannot hold (a boolean or text
"probability", a non-integer score level). A threshold is always a float: JSON `0.5`, never `1`.

Non-finite floats are written by the dumper as the strings "nan", "inf" and "-inf"; a missing
`answer_confidence` or `confidence` key is a field the answer did not report, and a typed port
reads it the same way as NaN (no usable number).
"""
import random

NAN = float("nan")
INF = float("inf")

# ---------------------------------------------------------------------------------------------
# check_min_confidence: scalar thresholds. Only floats, since a typed SDK takes a float.
# ---------------------------------------------------------------------------------------------
SCALAR_THRESHOLDS = [
    0.0, 1.0, 0.5, 0.35, 1e-09, 0.9999999999, 1e-05,
    -0.1, 1.5, -1e-05, 1.0000000001, 2.0, 100.0, 1e16, 1.2345678901234567e-07,
    NAN, INF, float("-inf"),
]

# ---------------------------------------------------------------------------------------------
# check_min_confidence_map: ordered (key, value) pairs, so an error names the first offender.
# ---------------------------------------------------------------------------------------------
_ALL_BUCKETS = ["%s:%s" % (qt, size) for qt in ("choice", "score", "noul") for size in ("2", "3-5", "6-10", "11+")]

MAP_THRESHOLDS = [
    [],
    [("default", 0.4)],
    [("choice:2", 0.7), ("default", 0.4)],
    [("choice:3-5", 0.5)],
    [("score:11+", 0.0), ("noul:2", 1.0)],
    [(key, round(0.05 * (i + 1), 2)) for i, key in enumerate(_ALL_BUCKETS)] + [("default", 0.5)],
    [("choice:2", 1.5)],
    [("choice:2", NAN)],
    [("choice:2", -0.1)],
    [("choice:2", INF)],
    [("bogus", 0.5)],
    [("Choice:2", 0.5)],
    [("choice:3", 0.5)],
    [("choice:2", 0.5), ("bogus", 0.5)],
    [("choice:2", 1.5), ("bogus", 0.1)],
    [("bogus", 1.5), ("choice:2", 0.1)],
    [("choice:2", 0.5), ("choice:3-5", 2.0), ("nope", 0.1)],
    [("it's", 0.5)],
    [("say \"hi\"", 0.5)],
    [("it's \"x\"", 0.5)],
    [("tab\there", 0.5)],
    [("café", 0.5)],
    [("", 0.5)],
    [("default", 0.5), ("default", 0.25)],
]


def _choice(*probs, ac=None, conf=None):
    out = {"type": "choice", "probabilities": {chr(ord("a") + i): p for i, p in enumerate(probs)}}
    _fields(out, ac, conf)
    return out


def _score(*probs, ac=None, conf=None):
    out = {"type": "score", "probabilities": {str(i): p for i, p in enumerate(probs)}}
    _fields(out, ac, conf)
    return out


def _noul(ac=None, conf=None, p=0.9):
    out = {"type": "noul", "noul": p}
    _fields(out, ac, conf)
    return out


def _fields(out, ac, conf):
    if ac is not None:
        out["answer_confidence"] = ac
    if conf is not None:
        out["confidence"] = conf


def _uniform(kind, k, ac=None, conf=None):
    return (_choice if kind == "choice" else _score)(*([round(1.0 / k, 4)] * k), ac=ac, conf=conf)


# ---------------------------------------------------------------------------------------------
# _option_bucket: every type at every option count the bands distinguish.
# ---------------------------------------------------------------------------------------------
BUCKET_ANSWERS = (
    [_uniform("choice", k) for k in (1, 2, 3, 5, 6, 10, 11, 13)]
    + [_uniform("score", k) for k in (1, 2, 3, 5, 6, 10, 11, 13)]
    + [_noul()]
    + [{"type": "choice", "probabilities": {}}, {"type": "score", "probabilities": {}}]
)

# ---------------------------------------------------------------------------------------------
# resolve_min_confidence: a validated map against answers of each bucket, and an unnamed one.
# ---------------------------------------------------------------------------------------------
RESOLVE_MAPS = [
    {"choice:2": 0.7, "default": 0.4},
    {"choice:3-5": 0.5},
    {"default": 0.25},
    {"score:6-10": 0.8, "noul:2": 0.3, "choice:11+": 0.6},
]
RESOLVE_ANSWERS = [_uniform("choice", 2), _uniform("choice", 4), _uniform("choice", 8), _uniform("choice", 12),
                   _uniform("score", 2), _uniform("score", 4), _uniform("score", 8), _uniform("score", 12),
                   _noul(), {"type": "choice", "probabilities": {}}]

# ---------------------------------------------------------------------------------------------
# apply_confidence_gate: a result with these answers, gated at each threshold. A threshold is
# None (unset: nothing is written), a float, or a map.
# ---------------------------------------------------------------------------------------------
GATE_ANSWERS = {
    "choice2": _choice(0.6, 0.4, ac=0.6, conf=0.029),
    "choice4": _choice(0.45, 0.25, 0.2, 0.1, ac=0.45, conf=0.12),
    "choice8": _uniform("choice", 8, ac=0.3, conf=0.05),
    "choice12": _uniform("choice", 12, ac=0.15, conf=0.02),
    "score2": _score(0.7, 0.3, ac=0.7, conf=0.1),
    "score4": _score(0.1, 0.5, 0.3, 0.1, ac=0.5, conf=0.2),
    "score8": _uniform("score", 8, ac=0.125, conf=0.01),
    "noul": _noul(ac=0.8, conf=0.8, p=0.8),
    "at_half": _choice(0.5, 0.5, ac=0.5, conf=0.0),
    "fallback_conf": _choice(0.5, 0.5, conf=0.3),
    "nan_answer_conf": _choice(0.5, 0.5, ac=NAN, conf=0.7),
    "inf_answer_conf": _choice(0.5, 0.5, ac=INF, conf=0.2),
    "unusable_both": _choice(0.5, 0.5, ac=NAN, conf=INF),
    "no_confidence": _choice(0.5, 0.5),
    "empty_choice": {"type": "choice", "probabilities": {}, "answer_confidence": 0.4},
}

GATE_THRESHOLDS = [
    None,
    0.0,
    1.0,
    0.5,
    0.35,
    0.2,
    {"default": 0.4},
    {"choice:2": 0.7, "default": 0.4},
    {"choice:3-5": 0.5},
    {"choice:2": 0.7, "choice:3-5": 0.5, "choice:6-10": 0.4, "choice:11+": 0.1,
     "score:2": 0.8, "score:3-5": 0.6, "score:6-10": 0.2, "noul:2": 0.9},
    {"noul:2": 0.0, "score:3-5": 1.0},
]

# The same answers gated again and again, as one reused payload is (#910): each step's verdict must
# depend only on that step's threshold, never on an earlier one.
REUSE_SEQUENCES = [
    [0.9, 0.0, 0.2, 1.0, 0.5],
    [{"choice:2": 0.9, "default": 0.5}, 0.0, {"choice:3-5": 0.9}, 0.35],
]


# ---------------------------------------------------------------------------------------------
# jev_confidence
# ---------------------------------------------------------------------------------------------
def _jev_corpus():
    cases = [
        ("choice-confident", _choice(0.9, 0.1)),
        ("choice-even", _choice(0.5, 0.5)),
        ("choice-three", _choice(0.7, 0.2, 0.1)),
        ("choice-differs-from-entropy", _choice(0.7, 0.1, 0.1, 0.1)),
        ("choice-unnormalised", _choice(2.0, 1.0, 1.0)),
        ("choice-zero-total", _choice(0.0, 0.0, 0.0)),
        ("choice-single", _choice(0.3)),
        ("choice-single-zero", _choice(0.0)),
        ("choice-zero-among-others", _choice(0.0, 1.0)),
        ("score-peaked", _score(0.0, 0.95, 0.05)),
        ("score-jevcompat-example", _score(0.0, 0.95, 0.05)),
        ("score-even", _score(0.25, 0.25, 0.25, 0.25)),
        ("score-edge-mode", _score(0.8, 0.1, 0.1)),
        ("score-zero-total", _score(0.0, 0.0, 0.0, 0.0)),
        ("score-single", _score(0.6)),
        ("score-two", _score(0.3, 0.7)),
        ("score-bimodal", _score(0.45, 0.05, 0.05, 0.45)),
        ("score-tied-modes", _score(0.4, 0.4, 0.2)),
        ("score-levels-in-numeric-order",
         {"type": "score", "probabilities": {str(i): (1.0 if i == 10 else 0.0) for i in range(11)}}),
        ("noul", {"type": "noul", "noul": 0.9}),
        ("empty-choice-probabilities", {"type": "choice", "probabilities": {}}),
        ("empty-score-probabilities", {"type": "score", "probabilities": {}}),
        ("nan-probability", _choice(NAN, 0.5)),
        ("infinite-probability", _choice(INF, 0.5)),
        ("negative-infinite-probability", _choice(float("-inf"), 0.5)),
        ("negative-probability", _choice(2.0, -1.0)),
        ("mixed-sign-argmax-flips", _choice(0.5, -1.5)),
        ("all-negative-probabilities", _choice(-0.9, -0.1)),
        ("negative-total-near-zero", _choice(-0.61, 0.6)),
        ("negative-score-level", _score(1.5, -0.5, 0.0)),
        ("nan-score-level", _score(0.5, NAN)),
        ("negative-single", _choice(-0.5)),
        ("nan-single", _choice(NAN)),
    ]
    out = [{"name": name, "answer": answer, "typed": True} for name, answer in cases]
    # Shapes a typed answer cannot hold: kept so the corpus records that Python refuses them.
    out += [
        {"name": "unknown-type", "answer": {"type": "weird", "probabilities": {"a": 1.0}}, "typed": False},
        {"name": "missing-probabilities", "answer": {"type": "choice"}, "typed": False},
        {"name": "non-numeric-probability", "answer": _choice(0.5, "high"), "typed": False},
        {"name": "bool-probability", "answer": _choice(True, False), "typed": False},
        {"name": "bool-among-numbers", "answer": _choice(True, 0.4), "typed": False},
        {"name": "non-integer-score-level", "answer": {"type": "score", "probabilities": {"low": 0.5, "high": 0.5}},
         "typed": False},
    ]
    # CPython 3.12's float sum() is compensated (Neumaier), so a plain left-to-right sum differs here
    # in the last bits of the result; the first two are the textbook case, [0.1] * 10.
    sums = [
        ("sum-differs-choice-tenths", "choice", [0.1] * 10),
        ("sum-differs-score-tenths", "score", [0.1] * 10),
        ("sum-differs-choice-hundredths", "choice", [0.01] * 100),
        ("sum-differs-choice-7", "choice", [0.6946, 0.03, 0.3634, 0.67, 0.2649, 0.72, 0.6497]),
        ("sum-differs-choice-5", "choice", [0.7133, 0.3416, 0.7, 0.8634, 0.6]),
        ("sum-differs-choice-10", "choice", [0.6, 0.1, 0.3, 0.7918, 0.94, 0.7, 0.9983, 0.5, 0.8, 0.6]),
        ("sum-differs-choice-16", "choice", [0.9, 0.11, 0.4, 0.84, 0.1412, 0.8523, 0.45, 0.69, 0.48, 0.5, 0.1126,
                                              0.9598, 0.94, 0.18, 0.2723, 0.87]),
        ("sum-differs-score-7", "score", [0.4, 0.4528, 0.2, 0.1, 0.93, 0.34, 0.89]),
        ("sum-differs-score-3", "score", [0.453, 0.4, 0.0333]),
    ]
    for name, kind, probs in sums:
        out.append({"name": name, "answer": (_choice if kind == "choice" else _score)(*probs), "typed": True})
    # A deterministic spread of distributions, to pin the arithmetic beyond the hand-picked ones.
    rng = random.Random(1361)
    for i in range(24):
        k = rng.choice([2, 3, 4, 5, 8, 12])
        probs = [round(rng.random(), 4) for _ in range(k)]
        out.append({"name": "random-choice-%02d" % i, "answer": _choice(*probs), "typed": True})
        out.append({"name": "random-score-%02d" % i, "answer": _score(*probs), "typed": True})
    return out


JEV_CASES = _jev_corpus()
