"""Options that collapse into the same token span must be reported (issue #538).

`build_sequence` caps every option at 48 tokens, and re-caps all of them to
`max(4, (head_max_len - 16) // n)` once they overflow the head budget. Two options that share a
prefix can survive that cut as the *same* span: the marker count still equals the option count,
so the guard in `Agent._encode_state` passes and the request is answered normally -- from a
question that can no longer name those options apart. On a 58-label question at the default
budget only 42 of 58 spans differ, which is an accuracy ceiling of 72% that nothing in the
response mentions.

No weights and no network: a character-level stub tokenizer, so a shared prefix behaves the way
a real tokenizer's does.

Run: python tests/test_option_collapse.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import build_sequence, collapsed_options  # noqa: E402

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


class PrefixTok:
    """Character-level ids, so options that share a prefix share their leading tokens."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(c) % 90) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


TOK = PrefixTok()
# Four labels that agree for their first fourteen characters, as MASSIVE's iot_hue_light* do.
HUE = ["iot_hue_lightup", "iot_hue_lightoff", "iot_hue_lightdim", "iot_hue_lightchange"]


def q_choice(labels):
    return {"t": "choice", "ins": "which one?", "crit": {k: k.replace("_", " ") for k in labels}}


# ---------------------------------------------------------------- the return shape
ret = build_sequence(TOK, "a state", q_choice(["yes", "no"]), 128, 64)
check("shape/default is unchanged", len(ret), 2)
ret = build_sequence(TOK, "a state", q_choice(["yes", "no"]), 128, 64, return_stats=True)
check("shape/return_stats adds one value", len(ret), 3)
ids_a, markers_a = build_sequence(TOK, "a state", q_choice(["yes", "no"]), 128, 64)
check("shape/ids are identical either way", ret[0], ids_a)
check("shape/markers are identical either way", ret[1], markers_a)
check("shape/stat keys", sorted(ret[2]), ["options", "options_distinct", "tokens_per_option"])

# ---------------------------------------------------------------- nothing collapses
_, _, roomy = build_sequence(TOK, "a state", q_choice(HUE), 512, 256, return_stats=True)
check("roomy/every option is its own span", roomy["options_distinct"], 4)
check("roomy/counts every option", roomy["options"], 4)
check("roomy/no cap was applied", roomy["tokens_per_option"], None)

# ---------------------------------------------------------------- the cut collapses them
_, _, tight = build_sequence(TOK, "a state", q_choice(HUE), 128, 24, return_stats=True)
check("tight/a cap was applied", tight["tokens_per_option"], 4)
check("tight/still one marker per option", tight["options"], 4)
check_true("tight/but not one span per option", tight["options_distinct"] < tight["options"],
           "distinct=%r" % tight["options_distinct"])

# The number is the question's, not the state's: the same question collapses identically
# whatever it is asked about, which is what makes it computable once per question set.
_, _, other_state = build_sequence(TOK, {"body": "a completely different request"} , q_choice(HUE),
                                   128, 24, return_stats=True)
check("tight/independent of the state", other_state["options_distinct"], tight["options_distinct"])

# A distinct set of labels under the same budget keeps its spans, so the report is about the
# option texts and not merely about the option count.
_, _, spread = build_sequence(TOK, "a state", q_choice(["alpha", "bravo", "charlie", "delta"]),
                              128, 24, return_stats=True)
check("tight/distinct labels survive the same cap", spread["options_distinct"], 4)

# ---------------------------------------------------------------- `total` is the question's
# Markers past `max_len` are dropped from the sequence; `options` must still count the options
# the question defines, or a report reads "2 of 2 distinct" about a question with 40 of them.
MANY = ["label_%02d" % i for i in range(40)]      # distinct keys: a dict would fold repeats
_, dropped_markers, dropped = build_sequence(TOK, "a state", q_choice(MANY), 40, 400,
                                             return_stats=True)
check_true("markers/some were dropped by max_len", len(dropped_markers) < 40,
           "kept=%d" % len(dropped_markers))
check("markers/total still counts the question's options", dropped["options"], 40)

# ---------------------------------------------------------------- the filter
items = [{"options": {"options": 58, "options_distinct": 42, "tokens_per_option": 4}},
         {"options": {"options": 4, "options_distinct": 4, "tokens_per_option": None}},
         {"qtype": 0}]
out = collapsed_options(["intent", "dept", "legacy"], items)
check("filter/only the collapsed question", sorted(out), ["intent"])
check("filter/reports the question's own numbers", out["intent"],
      {"total": 58, "distinct": 42, "tokens_per_option": 4})
check("filter/an item without stats is skipped", "legacy" in out, False)
check("filter/nothing collapsed is an empty dict", collapsed_options(["dept"], [items[1]]), {})

# ---------------------------------------------------------------- what the agents publish
_agent_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "laya", "agent.py"),
                  encoding="utf-8").read()
_onnx_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "laya", "onnx_agent.py"),
                 encoding="utf-8").read()
for name, src in (("agent", _agent_src), ("onnx", _onnx_src)):
    check_true("usage/%s publishes it only when something collapsed" % name,
               'collapsed = collapsed_options(ids, items)' in src and
               'if collapsed:' in src and 'usage["options"] = collapsed' in src)
    check_true("usage/%s asks build_sequence for the stats" % name, "return_stats=True" in src)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
