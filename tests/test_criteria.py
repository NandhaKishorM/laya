"""Criteria rendering: structured values must not crash or leak Python reprs.

Regression tests for the bug reported in PR #2 (trocker): a `noul` question whose criteria
values were dicts raised `TypeError: can only concatenate str (not "dict") to str`, and
`choice`/`score` stringified dicts as Python reprs instead of JSON.

A question whose *shape* is wrong is the other half of the same promise and lives at the bottom of
this file: `Agent.system_one` rejects it by name before anything is rendered or tokenized, instead
of raising `AttributeError: 'NoneType' object has no attribute 'items'` from `render_options` or a
`selected index k out of range` from inside the decision head (#182).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import ece_score, render_criterion, render_options  # noqa: E402

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


def check_raises(name, fn, message):
    try:
        fn()
        FAIL.append("%s: should have raised ValueError" % name)
    except ValueError as e:
        check(name, str(e), message)


# --------------------------------------------------------------- render_criterion
check("criterion/str passes through", render_criterion("phishing or scam"), "phishing or scam")
check("criterion/dict -> json", render_criterion({"desc": "phishing"}), '{"desc": "phishing"}')
check("criterion/list -> json", render_criterion(["a", "b"]), '["a", "b"]')
check("criterion/int -> json", render_criterion(3), "3")
check("criterion/bool -> json", render_criterion(False), "false")
check("criterion/non-ascii kept", render_criterion({"d": "münchen"}), '{"d": "münchen"}')
check_true("criterion/unserialisable falls back to str",
           isinstance(render_criterion({"o": object()}), str))


# --------------------------------------------------------------- the reported crash
q = {"t": "noul", "ins": "Is this phishing?",
     "crit": {"true": {"desc": "phishing, scam or fraud"}, "false": {"desc": "legitimate"}}}
try:
    out = render_options(q)
    check("noul/dict criteria does not crash", len(out), 2)
    check_true("noul/false renders as json", out[0] == 'false: {"desc": "legitimate"}', out[0])
    check_true("noul/true renders as json", out[1] == 'true: {"desc": "phishing, scam or fraud"}', out[1])
    check_true("noul/no python repr leaked", "'" not in "".join(out), out)
except TypeError as e:
    FAIL.append("noul/dict criteria CRASHED: %s" % e)


# --------------------------------------------------------------- choice and score
out = render_options({"t": "choice", "ins": "x",
                      "crit": {"billing": {"desc": "payments"}, "tech": None, "sales": ""}})
check("choice/dict -> json", out[0], 'billing: {"desc": "payments"}')
check("choice/None -> bare key", out[1], "tech")
check("choice/empty string -> bare key", out[2], "sales")
check_true("choice/no python repr", "{'" not in "".join(out), out)

# 0 and False are real criterion values, not "missing"
out = render_options({"t": "choice", "ins": "x", "crit": {"zero": 0, "no": False}})
check("choice/0 is kept", out[0], "zero: 0")
check("choice/False is kept", out[1], "no: false")

out = render_options({"t": "score", "ins": "x", "crit": [{"d": "low"}, "high", 2]})
check("score/dict level -> json", out[0], 'level 0: {"d": "low"}')
check("score/str level unchanged", out[1], "level 1: high")
check("score/int level -> json", out[2], "level 2: 2")


# --------------------------------------------------------------- calibration boundaries
check("ece/zero confidence is included",
      ece_score(np.array([0.0]), np.array([1.0])), 1.0)
check("ece/zero confidence has its proper weight",
      ece_score(np.array([0.0, 1.0]), np.array([1.0, 1.0])), 0.5)


# --------------------------------------------------------------- noul labels
check("noul/default false text", render_options({"t": "noul", "ins": "x", "crit": None})[0],
      "false: no, the statement does not hold")
check("noul/default true text", render_options({"t": "noul", "ins": "x", "crit": None})[1],
      "true: yes, the statement holds")
check("noul/explicit None labels use defaults",
      render_options({"t": "noul", "ins": "x", "crit": None, "labels": None}),
      ["false: no, the statement does not hold", "true: yes, the statement holds"])
check("noul/string criteria still work",
      render_options({"t": "noul", "ins": "x", "crit": {"true": "yes it is", "false": "no"}}),
      ["false: no", "true: yes it is"])
custom_labels = {"true": " A ", "false": " B "}
custom_question = {"t": "noul", "ins": "x", "crit": None, "labels": custom_labels}
check("noul/custom labels preserve false-then-true semantics", render_options(custom_question),
      ["B: no, the statement does not hold", "A: yes, the statement holds"])
check("noul/custom label input is not mutated", custom_labels, {"true": " A ", "false": " B "})

_label_error = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
for name, labels in [
    ("not a dict", ["negative", "positive"]),
    ("missing true", {"false": "negative"}),
    ("extra key", {"false": "negative", "true": "positive", "other": "x"}),
    ("blank value", {"false": " ", "true": "positive"}),
    ("duplicate values", {"false": "same", "true": "same"}),
    ("non-string value", {"false": 0, "true": "positive"}),
]:
    check_raises("noul/invalid labels " + name,
                 lambda labels=labels: render_options({"t": "noul", "ins": "x", "labels": labels}),
                 _label_error)

for qtype, crit in [("choice", {"a": None, "b": None}), ("score", ["low", "high"])]:
    check_raises("%s/rejects labels" % qtype,
                 lambda qtype=qtype, crit=crit: render_options(
                     {"t": qtype, "ins": "x", "crit": crit,
                      "labels": {"false": "B", "true": "A"}}),
                 "labels is only supported for noul questions")

# --------------------------------------------------------------- unchanged choice and score behaviour
check("choice/string criteria still work",
      render_options({"t": "choice", "ins": "x", "crit": {"a": "first", "b": None}}),
      ["a: first", "b"])
check("choice/boolean-word labels are not rewritten",
      render_options({"t": "choice", "ins": "x", "crit": {"true": "yes", "false": "no"}}),
      ["true: yes", "false: no"])
check("score/string criteria still work",
      render_options({"t": "score", "ins": "x", "crit": ["low", "high"]}),
      ["level 0: low", "level 1: high"])

# every rendered option must be a str, whatever went in.
# The original set varied the criterion *values* only, and left every label a string, which is
# why it passed while `render_options({1: None})` returned `[1]` from a function annotated
# `-> List[str]`. `build_sequence` then called `.replace` on that int, and the error named
# neither the question nor the label. Labels are varied here for the same reason the values were.
for label, qq in [
    ("choice/objects", {"t": "choice", "ins": "x", "crit": {"a": {"n": 1}, "b": [1, 2], "c": 3.5}}),
    ("score/objects", {"t": "score", "ins": "x", "crit": [{"a": 1}, [2], None]}),
    ("noul/objects", {"t": "noul", "ins": "x", "crit": {"true": [1], "false": {"z": 0}}}),
    ("choice/int labels, no description", {"t": "choice", "ins": "x", "crit": {1: None, 2: None, 3: None}}),
    ("choice/float labels, no description", {"t": "choice", "ins": "x", "crit": {1.5: None}}),
    ("choice/None label, no description", {"t": "choice", "ins": "x", "crit": {None: None, "billing": None}}),
    ("choice/bool labels, no description", {"t": "choice", "ins": "x", "crit": {True: None, False: None}}),
    ("choice/int labels with descriptions", {"t": "choice", "ins": "x", "crit": {1: "one", 2: "two"}}),
]:
    check_true("all options are str (%s)" % label,
               all(isinstance(o, str) for o in render_options(qq)), render_options(qq))

# the rendered text of a non-string label is its string form, not its repr
check("choice/int label renders as its str", render_options({"t": "choice", "ins": "x", "crit": {1: None}}), ["1"])
check("choice/None label renders as the word none",
      render_options({"t": "choice", "ins": "x", "crit": {None: None}}), ["None"])
check("choice/int label keeps its description form",
      render_options({"t": "choice", "ins": "x", "crit": {1: "one"}}), ["1: one"])

# the JSON we emit is parseable back
parsed = json.loads(render_options(
    {"t": "noul", "ins": "x", "crit": {"true": {"a": 1}, "false": {"b": 2}}})[1].split("true: ", 1)[1])
check("emitted json round-trips", parsed, {"a": 1})

# public labels reach the renderer through Agent._to_internal without changing caller data
from laya.agent import Agent  # noqa: E402

public_labels = {"true": "A", "false": "B"}
public_question = {"type": "noul", "instructions": "Is this true?", "labels": public_labels}
internal = Agent._to_internal(public_question)
check("agent/forwards noul labels", internal["labels"], public_labels)
check("agent/forwarded labels reach renderer", render_options(internal),
      ["B: no, the statement does not hold", "A: yes, the statement holds"])
check("agent/leaves public question unchanged", public_question,
      {"type": "noul", "instructions": "Is this true?", "labels": {"true": "A", "false": "B"}})

boolean_criteria = {True: "yes", False: "no"}
boolean_question = {"type": "noul", "instructions": "Is this true?", "criteria": boolean_criteria,
                    "labels": {"false": "B", "true": "A"}}
boolean_internal = Agent._to_internal(boolean_question)
check("agent/custom labels keep boolean criteria normalization", boolean_internal["crit"],
      {"true": "yes", "false": "no"})
check("agent/boolean criteria render with custom labels", render_options(boolean_internal), ["B: no", "A: yes"])
check("agent/leaves boolean criteria unchanged", boolean_criteria, {True: "yes", False: "no"})


# --------------------------------------------------------------- CPU-fallback warning (#9 follow-up)
# The warning must fire only when a fallback actually happened -- not merely because the machine
# has CUDA. `laya.load(path, device="cpu")` on a GPU box is a deliberate choice, not a problem.
import inspect  # noqa: E402

from laya import agent as _agent  # noqa: E402

_src = inspect.getsource(_agent.Agent.__init__)
check_true("fallback/flag is initialised", "fell_back_from = fell_back_why = None" in _src)
check_true("fallback/warns only on a real fallback", "if fell_back_from is not None:" in _src)
check_true("fallback/reports the underlying reason", "Reason: %s" in _src)
check_true("fallback/keeps the actionable advice", "download.pytorch.org/whl/nightly" in _src)
check_true("fallback/no bare cuda probe for the warning",
           "torch.cuda.is_available() or getattr(torch.version" not in _src)


# --------------------------------------------------------------- malformed question shapes (#182)
# A question that cannot be answered used to fail three frames down, as an exception that named
# neither the question nor the fix: `AttributeError: 'NoneType' object has no attribute 'items'`
# from `render_options` for a `choice` without criteria, `KeyError: 'bool'` from the type table,
# and -- for a question that ended up with no options at all -- a `selected index k out of range`
# raised inside `DecisionModel.forward`, which reads like a bug in laya rather than in the caller's
# definition. The inference path below is real: a tiny from-config encoder, no checkpoint
# downloaded (tests/test_local_e2e.py covers the real weights).
import torch  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from laya.router import Router  # noqa: E402


# `_to_internal` serialises non-string `instructions` with json.dumps. The default
# `ensure_ascii=True` escaped non-ASCII to literal `\uXXXX`, which the tokenizer then
# read as escape text: on the English checkpoint one German question answered noul=0.1652
# as a dict and noul=0.2650 as the identical plain string. Every other text path keeps
# its characters -- see the `criterion/non-ascii kept` case above.
_internal = Agent._to_internal(
    {"type": "noul", "instructions": {"frage": "Bittet um eine R\u00fcckerstattung?"},
     "criteria": None})
check("instructions/non-ascii kept as a dict",
      _internal["ins"], '{"frage": "Bittet um eine R\u00fcckerstattung?"}')
check_true("instructions/no escape sequences in the prompt",
           "\\u" not in _internal["ins"], repr(_internal["ins"]))
check("instructions/ascii is unchanged",
      Agent._to_internal({"type": "noul", "instructions": {"asks": "for a refund"},
                          "criteria": None})["ins"],
      '{"asks": "for a refund"}')
check("instructions/plain string is untouched",
      Agent._to_internal({"type": "noul", "instructions": "Bittet der Kunde um eine "
                          "R\u00fcckerstattung?", "criteria": None})["ins"],
      "Bittet der Kunde um eine R\u00fcckerstattung?")
check("instructions/non-string still renders as json",
      Agent._to_internal({"type": "noul", "instructions": ["a", "b"],
                          "criteria": None})["ins"], '["a", "b"]')


class _FakeTok:
    """The tokenizer surface `build_sequence` uses, with predictable ids."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _tiny_agent():
    """An `Agent` with a tiny random encoder: the inference path, without the download."""
    agent = object.__new__(Agent)
    cfg = AutoConfig.for_model("bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
                               intermediate_size=32, vocab_size=64)
    agent.cfg = {"max_len": 256, "head_max_len": 96, "encoder": "tiny"}
    agent.tok = _FakeTok()
    agent.model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2).eval()
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    return agent


STATE = {"body": "I was charged twice for invoice 4411 and want a refund"}
agent = _tiny_agent()
for label, qdef in [
    ("choice without criteria", {"type": "choice", "instructions": "Which team?"}),
    ("choice with criteria None", {"type": "choice", "instructions": "Which team?", "criteria": None}),
    ("choice with empty criteria", {"type": "choice", "instructions": "Which team?", "criteria": {}}),
    ("choice with a tuple of labels", {"type": "choice", "instructions": "Which team?",
                                       "criteria": ("billing", "tech")}),
    ("score without criteria", {"type": "score", "instructions": "How urgent?"}),
    ("score with an empty list", {"type": "score", "instructions": "How urgent?", "criteria": []}),
    ("score with a dict of levels", {"type": "score", "instructions": "How urgent?",
                                     "criteria": {"low": "no pressure", "high": "blocking"}}),
    # a null level reached the model as the text "level 1: null" and came back as a null legend
    # value, which a Jev client refuses to parse (#302)
    ("score with a null level", {"type": "score", "instructions": "How urgent?",
                                 "criteria": ["low", None, "high"]}),
    ("choice with labels", {"type": "choice", "instructions": "Which team?",
                            "criteria": ["billing", "tech"],
                            "labels": {"false": "B", "true": "A"}}),
    ("score with labels", {"type": "score", "instructions": "How urgent?",
                           "criteria": ["low", "high"],
                           "labels": {"false": "B", "true": "A"}}),
    ("noul with list criteria", {"type": "noul", "instructions": "Is it spam?", "criteria": ["a", "b"]}),
    ("noul with string criteria", {"type": "noul", "instructions": "Is it spam?", "criteria": "spam?"}),
    ("noul with incomplete labels", {"type": "noul", "instructions": "Is it spam?",
                                     "labels": {"true": "A"}}),
    ("noul with duplicate labels", {"type": "noul", "instructions": "Is it spam?",
                                    "labels": {"false": "A", "true": "A"}}),
    # `render_options` reads the two noul descriptions by name, so any other key used to be
    # dropped and replaced with the defaults without a word (#156). These are the shapes a
    # caller reaches for when they want to word the two options themselves.
    ("noul with yes/no criteria", {"type": "noul", "instructions": "Is it spam?",
                                   "criteria": {"yes": "it is spam", "no": "it is not"}}),
    ("noul with neutral keys", {"type": "noul", "instructions": "Is it spam?",
                                "criteria": {"spam": "it is spam", "ham": "it is not"}}),
    ("noul with alpha/beta criteria", {"type": "noul", "instructions": "Is it spam?",
                                       "criteria": {"alpha": "yes", "beta": "no"}}),
    ("noul with a typo'd key", {"type": "noul", "instructions": "Is it spam?",
                                "criteria": {"ture": "yes", "false": "no"}}),
    ("noul with an extra key", {"type": "noul", "instructions": "Is it spam?",
                                "criteria": {"true": "y", "false": "n", "maybe": "?"}}),
    ("unknown type", {"type": "bool", "instructions": "Is it spam?"}),
    ("missing type", {"instructions": "Is it spam?"}),
    ("no instructions", {"type": "noul"}),
]:
    try:
        agent.system_one(STATE, {"q": qdef})
        FAIL.append("rejected/%s: no error raised" % label)
    except ValueError as e:
        # the message must name the question: a caller with twenty of them needs to know which
        check_true("rejected/%s names the question" % label, "'q'" in str(e), str(e))
        check_true("rejected/%s says what to fix" % label, len(str(e)) > 40, str(e))
    except Exception as e:
        FAIL.append("rejected/%s: %s instead of ValueError: %s" % (label, type(e).__name__, e))

try:
    agent.system_one(STATE, {"q": {"type": "score", "instructions": "How urgent?", "criteria": ["low", None]}})
    FAIL.append("rejected/score null level names the level: no error raised")
except ValueError as e:
    check_true("rejected/score null level names the level", "level 1" in str(e), str(e))

# the same questions through the public entry point, not only the method under it
router = Router()
router.attach("english", agent)
for label, qdef in (("choice without criteria", {"type": "choice", "instructions": "x"}),):
    try:
        router.predict(STATE, {"q": qdef}, model="english")
        FAIL.append("rejected/router %s: no error raised" % label)
    except ValueError as e:
        check_true("rejected/router %s names the question" % label, "'q'" in str(e), str(e))
    except Exception as e:
        FAIL.append("rejected/router %s: %s instead of ValueError: %s" % (label, type(e).__name__, e))

# every question id is validated, not only the first one put in the dict
try:
    agent.system_one(STATE, {"ok": {"type": "noul", "instructions": "Is it urgent?"},
                             "broken": {"type": "choice", "instructions": "Which team?"}})
    FAIL.append("rejected/second question: no error raised")
except ValueError as e:
    check_true("rejected/second question names it", "'broken'" in str(e), str(e))
except Exception as e:
    FAIL.append("rejected/second question: %s instead of ValueError: %s" % (type(e).__name__, e))

# ...and the shapes that are valid still answer, so this is not validation-only coverage
GOOD = {
    "choice": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "invoices and refunds", "tech": "bugs"}},
    "choice as a list": {"type": "choice", "instructions": "Which team?",
                         "criteria": ["billing", "tech"]},
    "score": {"type": "score", "instructions": "How urgent?",
              "criteria": ["no pressure", "soon", "blocking"]},
    "noul": {"type": "noul", "instructions": "Does the sender want a reply?"},
    "noul with criteria": {"type": "noul", "instructions": "Is it phishing?",
                           "criteria": {"true": "phishing", "false": "legitimate"}},
    "noul with labels": {"type": "noul", "instructions": "Is it phishing?",
                         "criteria": {"true": "phishing", "false": "legitimate"},
                         "labels": {"false": "B", "true": "A"}},
    "non-string instructions": {"type": "noul", "instructions": {"asks": "for a refund"}},
}
out = agent.system_one(STATE, GOOD)
check("good/one answer per question", sorted(out["answers"]), sorted(GOOD))
check_true("good/choice label", out["answers"]["choice"]["choice"] in ("billing", "tech"),
           str(out["answers"]["choice"]))
check_true("good/choice from a list",
           out["answers"]["choice as a list"]["choice"] in ("billing", "tech"),
           str(out["answers"]["choice as a list"]))
check("good/choice probabilities sum", round(sum(out["answers"]["choice"]["probabilities"].values()), 3), 1.0)
check_true("good/score is in range", 0.0 <= out["answers"]["score"]["score"] <= 2.0,
           str(out["answers"]["score"]))
check("good/score legend", out["answers"]["score"]["legend"],
      {"0": "no pressure", "1": "soon", "2": "blocking"})
check_true("good/noul is a probability", 0.0 <= out["answers"]["noul"]["noul"] <= 1.0,
           str(out["answers"]["noul"]))
check_true("good/noul with criteria is a probability",
           0.0 <= out["answers"]["noul with criteria"]["noul"] <= 1.0,
           str(out["answers"]["noul with criteria"]))
check_true("good/noul with labels is a probability",
           0.0 <= out["answers"]["noul with labels"]["noul"] <= 1.0,
           str(out["answers"]["noul with labels"]))
check("good/usage has no output tokens", out["usage"]["output_tokens"], 0)


# ------------------------------------------------- noul criteria keys that must keep working
# The guard above rejects a key it cannot use. These are the spellings it must still accept,
# and the check is on the rendered text rather than on "no exception", because the whole point
# is that the caller's descriptions reach the model. Before the guard, the yes/no row below
# would have rendered the defaults instead and the caller had no way to tell (#156).
_DEFAULT_FALSE_TEXT = "false: no, the statement does not hold"
_DEFAULT_TRUE_TEXT = "true: yes, the statement holds"

for label, crit, want in [
    ("true/false use the caller's text",
     {"true": "the review is positive", "false": "the review is negative"},
     ["false: the review is negative", "true: the review is positive"]),
    ("uppercase keys work, via the .lower() in _to_internal",
     {"TRUE": "the review is positive", "FALSE": "the review is negative"},
     ["false: the review is negative", "true: the review is positive"]),
    ("Python bool keys work, which is how JSON true/false arrive",
     {True: "the review is positive", False: "the review is negative"},
     ["false: the review is negative", "true: the review is positive"]),
    ("one key is enough",
     {"true": "the review is positive"},
     [_DEFAULT_FALSE_TEXT, "true: the review is positive"]),
    ("an empty dict falls back to both defaults", {},
     [_DEFAULT_FALSE_TEXT, _DEFAULT_TRUE_TEXT]),
    ("omitting criteria falls back to both defaults", None,
     [_DEFAULT_FALSE_TEXT, _DEFAULT_TRUE_TEXT]),
    ("a description equal to the default wording still counts as given",
     {"true": "yes, the statement holds", "false": "no, the statement does not hold"},
     [_DEFAULT_FALSE_TEXT, _DEFAULT_TRUE_TEXT]),
]:
    qdef = {"type": "noul", "instructions": "Is the review positive?"}
    if crit is not None:
        qdef["criteria"] = crit
    try:
        check("noul keys/%s" % label, render_options(Agent._to_internal(qdef)), want)
    except Exception as exc:  # noqa: BLE001
        FAIL.append("noul keys/%s raised %s: %s" % (label, type(exc).__name__, exc))

# `labels` is the supported way to word the answer without touching the option text, so the
# message the guard raises points at it. It replaces the `false:`/`true:` prefixes the model
# reads; the criteria text after them is unchanged, and the result stays P(true).
_mixed = Agent._to_internal({"type": "noul", "instructions": "Is the review positive?",
                             "criteria": {"true": "the review is positive",
                                          "false": "the review is negative"},
                             "labels": {"true": "positive", "false": "negative"}})
check("noul keys/criteria text survives alongside labels",
      render_options(_mixed),
      ["negative: the review is negative", "positive: the review is positive"])
check("noul keys/labels are carried through", _mixed["labels"],
      {"true": "positive", "false": "negative"})
check("noul keys/labels keep the false/true slot order",
      render_options(_mixed)[0].startswith("negative:"),
      True)
# ...and the criteria descriptions are all `labels` changes -- the same pair without labels
# is the same text behind the default prefixes.
check("noul keys/labels change only the prefix",
      [o.split(": ", 1)[1] for o in render_options(_mixed)],
      [o.split(": ", 1)[1] for o in render_options(Agent._to_internal(
          {"type": "noul", "instructions": "Is the review positive?",
           "criteria": {"true": "the review is positive", "false": "the review is negative"}}))])
check_true("good/usage counted input tokens", out["usage"]["input_tokens"] > 0, str(out["usage"]))

# --------------------------------------------------------------- build_sequence left truncation
# With no room left for the state, `st[-0:]` kept all of it: the closing [SEP] was replaced by the
# *first* state token, i.e. the wrong end of the state and an unterminated sequence.
from laya.common import build_sequence  # noqa: E402


class _SeqTok:
    mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 1, 2, 3

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [self.vocab.setdefault(w, 100 + len(self.vocab)) for w in text.split()]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


_tok, _q = _SeqTok(), {"t": "noul", "ins": "Is it urgent?", "crit": None}
_full = len(build_sequence(_tok, "", _q, 10 ** 6)[0])     # prompt + closing [SEP], no state
for room, kept in [(0, []), (2, ["two", "three"]), (10, ["one", "two", "three"])]:
    ids = build_sequence(_tok, "one two three", _q, _full + room, truncate_left=True)[0]
    check("truncate_left/room=%d keeps the tail" % room, ids[_full - 1:],
          [_tok.vocab[w] for w in kept] + [_tok.sep_token_id])


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
