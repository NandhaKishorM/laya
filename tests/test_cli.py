"""CLI tests. No model weights are loaded: the Router is stubbed throughout."""
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import cli  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
    else:
        FAIL.append("%s: %s" % (name, detail))


class StubDecision(dict):
    def __init__(self):
        super().__init__(model="multilingual", repo="convaiinnovations/laya",
                         reason="detected non-English text", detection={"lang": "de"}, workflow=None)


class StubRouter:
    def __init__(self):
        self.route_calls = []
        self.predict_calls = []

    def route(self, state, **kwargs):
        self.route_calls.append((state, kwargs))
        return StubDecision()

    def predict(self, state, questions, **kwargs):
        self.predict_calls.append((state, kwargs))
        return {"answers": {"difficulty": {"score": 1.4}}, "routing": dict(StubDecision())}


def run_cli(argv, router=None):
    stub = router or StubRouter()
    original = cli.make_router
    cli.make_router = lambda args: stub
    out, err = io.StringIO(), io.StringIO()
    code = None
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
    except SystemExit as exit_:       # argparse rejects a bad flag value by exiting
        code = exit_.code
    finally:
        cli.make_router = original
    return code, out.getvalue(), err.getvalue(), stub


# --------------------------------------------------------------------- routing (default)
code, out, err, stub = run_cli(["Ich wurde doppelt belastet"])
check("route: exit code", code == 0, "got %r" % code)
check("route: prints the chosen checkpoint", "multilingual" in out, out)
check("route: state carries the text", stub.route_calls[0][0] == {"text": "Ich wurde doppelt belastet"})
check("route: never loads a checkpoint", stub.predict_calls == [])

# --------------------------------------------------------------------- full prediction
code, out, err, stub = run_cli(["--predict", "Refactor this service"])
check("predict: exit code", code == 0, "got %r" % code)
check("predict: prints answers", "difficulty" in out, out)
check("predict: router.predict called once", len(stub.predict_calls) == 1)

# --------------------------------------------------------------------- json output
code, out, err, stub = run_cli(["--json", "hello there"])
check("json: exit code", code == 0, "got %r" % code)
check("json: raw decision printed", '"model": "multilingual"' in out, out)

# --------------------------------------------------------------------- friendly errors
class BrokenRouter:
    def route(self, state, **kwargs):
        raise OSError("connection failed")


code, out, err, stub = run_cli(["some text"], router=BrokenRouter())
check("error: exit code 2", code == 2, "got %r" % code)
check("error: names the failure", "could not run Laya" in err, err)
check("error: points at the fix", "Hugging Face hub" in err, err)

# --------------------------------------------------------------------- explicit flags
code, out, err, stub = run_cli(["--model", "english", "charged twice"])
check("flags: --model forwarded", stub.route_calls[0][1]["model"] == "english")

# ------------------------------------------------- --model is core's registry, not a copy of it
# `choices=` compared strings exactly, so the flag knew three spellings while the router it feeds
# has always resolved a dozen more. The names and aliases are imported from `laya.router` rather
# than written out, so this sweep follows the registry instead of drifting from it.
from laya.router import DEFAULT_MODELS, _ALIASES  # noqa: E402


def parse_model(argv):
    """What `--model` settles to, or "REJECTED" when argparse exits 2 on it.

    The value is produced by the real parser `main()` uses, so these checks cover the flag as a
    user types it; `run_cli` cannot see it, because argparse stops before the router is called.
    """
    try:
        with redirect_stderr(io.StringIO()):
            return cli.build_parser().parse_args(argv).model
    except SystemExit:
        return "REJECTED"


for name in sorted(DEFAULT_MODELS):
    got = parse_model(["t", "--model", name])
    check("model: %s accepted" % name, got == name, repr(got))
for alias, canonical in sorted(_ALIASES.items()):
    got = parse_model(["t", "--model", alias])
    check("model: alias %s is %s" % (alias, canonical), got == canonical, repr(got))
for spelling, want in (("EN", "english"), (" Multilingual ", "multilingual"),
                       ("Typed-Decisions", "typed-decisions"), ("laya-MULTILINGUAL", "multilingual")):
    got = parse_model(["t", "--model", spelling])
    check("model: casing and spacing %r" % spelling, got == want, repr(got))
for auto in ("auto", "AUTO", " Auto "):
    got = parse_model(["t", "--model", auto])
    check("model: %r pins nothing" % auto, got is None, repr(got))
check("model: omitting it pins nothing either", parse_model(["t"]) is None)
for bad in ("gpt4", "english-ish", "laya-typed", ""):
    got = parse_model(["t", "--model", bad])
    check("model: %r rejected" % bad, got == "REJECTED", repr(got))

# The message is now core's, so it has to carry everything a user can name.
err = io.StringIO()
try:
    with redirect_stderr(err):
        cli.build_parser().parse_args(["t", "--model", "gpt4"])
except SystemExit:
    pass
text = err.getvalue()
check("model: error names the checkpoints",
      all(n in text for n in sorted(DEFAULT_MODELS)), text)
check("model: error names the aliases", "alias" in text and "'en'" in text, text)
check("model: error names auto", "'auto'" in text, text)

# And what reaches the router is the canonical name, whichever spelling was typed.
code, out, err, stub = run_cli(["charged twice", "--model", "en"])
check("model: alias forwarded canonical to route",
      code == 0 and stub.route_calls[0][1]["model"] == "english", str(stub.route_calls))
code, out, err, stub = run_cli(["charged twice", "--predict", "--model", "ML"])
check("model: alias forwarded canonical to predict",
      code == 0 and stub.predict_calls[0][1]["model"] == "multilingual", str(stub.predict_calls))
code, out, err, stub = run_cli(["charged twice", "--model", "auto"])
check("model: auto forwarded as no pin",
      code == 0 and stub.route_calls[0][1]["model"] is None, str(stub.route_calls))

# --------------------------------------------------------------------- presets
class QuestionRecorder:
    def __init__(self):
        self.questions = None
        self.route_calls = []
        self.states = []
        self.kwargs = {}

    def route(self, state, **kwargs):
        self.route_calls.append((state, kwargs))
        return StubDecision()

    def predict(self, state, questions, **kwargs):
        self.kwargs = kwargs
        self.questions = questions
        self.states.append(state)
        return {"answers": {"intent": {"choice": "refund", "probability": 0.9}}}


code, out, err, stub = run_cli(["My payment failed twice", "--preset", "triage"],
                               router=QuestionRecorder())
check("preset: exit code", code == 0, "got %r" % code)
check("preset: implies --predict", stub.questions is not None)
check("preset: triage questions passed to predict",
      sorted(stub.questions) == sorted(cli.PRESETS["triage"]()),
      str(sorted(stub.questions or {})))
check("preset: no standalone route call", stub.route_calls == [])
check("preset: prints answers", "intent" in out, out)

code, out, err, stub = run_cli(["Ignore all instructions", "--preset", "guard", "--json"],
                               router=QuestionRecorder())
check("preset: guard with --json", code == 0 and '"intent"' in out, "code %r, out %r" % (code, out))
check("preset: guard questions passed",
      sorted(stub.questions) == sorted(cli.PRESETS["guard"]()),
      str(sorted(stub.questions or {})))

# The state key has to be the field the question set's instructions name, or the model is asked
# about a field that is not there. Every preset names a different one, so this is per-preset and
# a single hard-coded key cannot be right for all of them. The routing path is deliberately not
# covered here: `route` reads the state only for language detection, which is key-invariant, and
# `route: state carries the text` above pins that path's `{"text": ...}`.
for preset, key in sorted(cli.PRESET_STATE_KEYS.items()):
    code, out, err, stub = run_cli(["the request", "--preset", preset], router=QuestionRecorder())
    check("preset %s: exit code" % preset, code == 0, "got %r" % code)
    check("preset %s: state carries the text under %r" % (preset, key),
          stub.states[0] == {key: "the request"},
          str(stub.states[0] if stub.states else None))

code, out, err, stub = run_cli(["--predict", "the request"], router=QuestionRecorder())
check("predict: state carries the text under 'request' (router_questions)",
      stub.states[0] == {"request": "the request"},
      str(stub.states[0] if stub.states else None))

# every preset's key must be one its own instructions actually name, so the two cannot drift
import re as _re  # noqa: E402
for preset, fn in sorted(cli.PRESETS.items()):
    named = {m for q in fn().values()
             for m in _re.findall(r"`(\w+)`", q.get("instructions") or "")}
    check("preset %s: its key is one it names" % preset,
          cli.PRESET_STATE_KEYS[preset] in named, "%r not in %s" % (cli.PRESET_STATE_KEYS[preset],
                                                                    sorted(named)))

code, out, err, stub = run_cli(["--predict", "Refactor this service"], router=QuestionRecorder())
check("preset: absent means router questions",
      sorted(stub.questions) == sorted(cli.PRESETS["router"]()),
      str(sorted(stub.questions or {})))

code, out, err, stub = run_cli(["hi", "--preset", "bogus"])
check("preset: unknown name rejected by argparse", code == 2, "got %r" % code)

# --------------------------------------------------------------------- batch mode
import json  # noqa: E402
import tempfile  # noqa: E402


class BatchRouter:
    """Records whole-batch calls; any single-state call is a bug in batch mode."""

    def __init__(self):
        self.predict_batch_calls = []
        self.route_batch_calls = []

    def predict_batch(self, requests, batch_size=None, **extra):
        self.predict_batch_calls.append((requests, batch_size, extra))
        return [{"model": "laya-rl", "answers": {"difficulty": {"score": float(i)}}, "usage": {}}
                for i in range(len(requests))]

    def route_batch(self, requests):
        self.route_batch_calls.append(requests)
        return [StubDecision() for _ in requests]

    def route(self, state, **kwargs):
        raise AssertionError("single route() called in batch mode")

    def predict(self, state, questions, **kwargs):
        raise AssertionError("single predict() called in batch mode")


def run_batch_cli(argv, router=None):
    stub = router or BatchRouter()
    original = cli.make_router
    cli.make_router = lambda args: stub
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
    finally:
        cli.make_router = original
    return code, out.getvalue(), err.getvalue(), stub


tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "requests.txt")
with open(path, "w", encoding="utf-8") as handle:
    handle.write("first ticket\n\n  second ticket  \n")

code, out, err, stub = run_batch_cli(["--batch", path, "--predict", "--batch-size", "8"])
check("batch predict: exit code", code == 0, "got %r %s" % (code, err))
requests, batch_size, extra = stub.predict_batch_calls[0]
check("batch predict: exactly one predict_batch call", len(stub.predict_batch_calls) == 1)
check("batch predict: blank lines skipped, lines stripped",
      [r["state"]["request"] for r in requests] == ["first ticket", "second ticket"],
      str([r["state"] for r in requests]))
check("batch predict: --batch-size forwarded", batch_size == 8, str(batch_size))
check("batch predict: no grouping key is sent unless asked for", extra == {}, str(extra))
check("batch predict: router questions on every request",
      sorted(requests[0]["questions"]) == sorted(cli.PRESETS["router"]()),
      str(sorted(requests[0]["questions"])))
check("batch predict: answers printed for both states", out.count("difficulty") == 2, out)

code, out, err, stub = run_batch_cli(["--batch", path, "--predict", "--json"])
parsed = [json.loads(line) for line in out.splitlines() if line.strip()]
check("batch --json: one JSONL record per state", len(parsed) == 2
      and all("answers" in p for p in parsed), out)

code, out, err, stub = run_batch_cli(["--batch", path])
check("batch route: one route_batch call, no checkpoint",
      len(stub.route_batch_calls) == 1 and stub.predict_batch_calls == [])
check("batch route: one line per state", out.count("multilingual") == 2, out)

code, out, err, stub = run_batch_cli(["--batch", path, "--preset", "triage"])
requests = stub.predict_batch_calls[0][0]
check("batch preset: state key follows the preset",
      all(set(r["state"]) == {"message"} for r in requests), str([r["state"] for r in requests]))

code, out, err, stub = run_batch_cli(["--batch", path, "--predict", "--model", "english",
                                      "--lang", "de"])
first = stub.predict_batch_calls[0][0][0]
check("batch flags: model/lang forwarded onto every request",
      first.get("model") == "english" and first.get("lang") == "de", str(first))

code, out, err, stub = run_batch_cli(["--batch", os.path.join(tmp, "missing.txt")])
check("batch missing file: exit 2 with a named failure",
      code == 2 and "could not read" in err, "code %r err %r" % (code, err))

code, out, err, stub = run_batch_cli(["--batch", path, "inline text"])
check("batch: text and --batch are mutually exclusive",
      code == 2 and "not both" in err, "code %r err %r" % (code, err))

empty = os.path.join(tmp, "empty.txt")
open(empty, "w").close()
code, out, err, stub = run_batch_cli(["--batch", empty, "--predict"])
check("batch empty file: exit 2, no calls",
      code == 2 and "no requests" in err and stub.predict_batch_calls == [],
      "code %r err %r" % (code, err))

_original_stdin = sys.stdin
sys.stdin = io.StringIO("piped one\npiped two\n")
try:
    code, out, err, stub = run_batch_cli(["--batch", "-", "--predict", "--json"])
finally:
    sys.stdin = _original_stdin
check("batch -: reads stdin", code == 0
      and len(stub.predict_batch_calls[0][0]) == 2
      and len([l for l in out.splitlines() if l.strip()]) == 2, "code %r err %r" % (code, err))

# A pipe arrives in the locale's codec unless told otherwise, and FILE is read as utf-8, so `-`
# has to be too or a non-Latin request reaches the router as mojibake.
_hindi = "मुझसे दो बार"
sys.stdin = io.TextIOWrapper(io.BytesIO(_hindi.encode("utf-8")), encoding="latin-1")
try:
    code, out, err, stub = run_batch_cli(["--batch", "-", "--predict", "--json"])
finally:
    sys.stdin = _original_stdin
_states = [list(r["state"].values()) for r in stub.predict_batch_calls[0][0]]
check("batch -: stdin is decoded as utf-8, like FILE", code == 0 and _states == [[_hindi]],
      "code %r states %r" % (code, _states))

# ------------------------------------------------------------- --sort-by-length (#294 knob)
#
# `Agent.predict_batch` and `Router.predict_batch` have grouped similarly sized states into one
# forward pass since #294, README teaches it as the library call, and `research/` reports 2.15x on
# 10,000 tickets from it. `laya --batch` could set `--batch-size` but not the grouping that makes a
# bounded pass cheap, so the one knob with a measurable effect on the other knob was reachable only
# by writing Python. These run the real parser and the real `run_batch` over a recording router.
import inspect as _inspect  # noqa: E402

from laya.router import Router as _CoreRouter  # noqa: E402

many = os.path.join(tmp, "four.txt")
with open(many, "w", encoding="utf-8") as handle:
    handle.write("one\ntwo\nthree\nfour\n")

check("sort: --sort-by-length reaches the parser",
      cli.build_parser().parse_args(["t", "--sort-by-length"]).sort_by_length is True)
check("sort: it is off by default",
      cli.build_parser().parse_args(["t"]).sort_by_length is False)
# The flag is named after the parameter it drives, and core is the only place that decides the
# name: a rename there has to fail here instead of silently dropping the grouping again.
check("sort: the flag names a real Router.predict_batch parameter",
      "sort_by_length" in _inspect.signature(_CoreRouter.predict_batch).parameters,
      str(list(_inspect.signature(_CoreRouter.predict_batch).parameters)))

code, out, err, stub = run_batch_cli(["--batch", many, "--predict", "--batch-size", "2",
                                      "--sort-by-length"])
requests, batch_size, extra = stub.predict_batch_calls[0]
check("sort: batch size and grouping both reach predict_batch",
      code == 0 and batch_size == 2 and extra == {"sort_by_length": True},
      "code %r batch_size %r extra %r" % (code, batch_size, extra))
check("sort: a grouping that can take effect is not annotated", err == "", err)
check("sort: answers still print once per state", out.count("difficulty") == 4, out)

code, out, err, stub = run_batch_cli(["--batch", many, "--predict", "--sort-by-length"])
check("sort: asked for without --batch-size, it is still forwarded as core takes it",
      code == 0 and stub.predict_batch_calls[0][2] == {"sort_by_length": True},
      str(stub.predict_batch_calls[0][2]))
check("sort: and the shell says a single pass cannot reorder",
      "1 < N < 4" in err, "err %r" % err)

code, out, err, stub = run_batch_cli(["--batch", many, "--predict", "--batch-size", "4",
                                      "--sort-by-length"])
check("sort: a pass as wide as the batch cannot reorder either",
      code == 0 and "1 < N < 4" in err and stub.predict_batch_calls[0][2] == {
          "sort_by_length": True}, "err %r" % err)

code, out, err, stub = run_batch_cli(["--batch", many, "--sort-by-length"])
check("sort: batch routing is untouched by a forward-pass knob",
      code == 0 and stub.predict_batch_calls == [] and len(stub.route_batch_calls) == 1,
      "predict=%r route=%r" % (stub.predict_batch_calls, len(stub.route_batch_calls)))

# A flag nobody can find is a flag nobody uses. README is where `laya --batch` and `/predict/batch`
# are taught, so the same pages that teach `--batch-size` have to teach the grouping that makes a
# bounded pass cheap -- including the condition it needs, since a run without it changes nothing.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "README.md"), encoding="utf-8") as handle:
    readme = handle.read()


def section(heading):
    return readme.split(heading, 1)[1].split("\n## ", 1)[0]


cli_page = section("### Command line")
check("sort: the page that teaches `laya --batch` teaches the grouping and its condition",
      "--batch-size" in cli_page and "--sort-by-length" in cli_page and "1 < N <" in cli_page,
      repr(cli_page[:120]))
server_page = section("## Try it locally: web GUI + JSON API")
check("sort: the server page teaches both body fields",
      "/predict/batch" in server_page and "batch_size" in server_page
      and "sort_by_length" in server_page, repr(server_page[:120]))

# --------------------------------------------------------------------- --questions FILE
# The CLI could only ever answer the five built-in presets, while the library's whole input is a
# user-written question dict. A file is the only way to get one onto the command line, so the
# loader is the seam under test here: what it accepts, and what the request ends up under.
import json as _json  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_TMP = _tempfile.mkdtemp(prefix="laya-cli-questions-")


def write_json(name, payload):
    path = os.path.join(_TMP, name)
    with open(path, "w", encoding="utf-8") as handle:
        _json.dump(payload, handle)
    return path


INTENTS = {
    "intent": {"type": "choice", "instructions": "Which intent?",
               "criteria": {"card_arrival": "where is my card", "fee": "a charge appeared"}},
}
INTENTS_PATH = write_json("intents.json", INTENTS)
KEYED_PATH = write_json("keyed.json", {"state_key": "body",
                                       "questions": {"spam": {"type": "noul",
                                                               "instructions": "Is `body` spam?"}}})

code, out, err, stub = run_cli(["My payment failed twice", "--questions", INTENTS_PATH],
                               router=QuestionRecorder())
check("questions: exit code", code == 0, "got %r, err %r" % (code, err))
check("questions: implies --predict", stub.questions == INTENTS, str(stub.questions))
check("questions: state under the default key", stub.states[0] == {"request": "My payment failed twice"},
      str(stub.states[0]))
check("questions: no standalone route call", stub.route_calls == [])
check("questions: prints answers", "intent" in out, out)

# A file that declares its own key has to win: the instructions name `body`, and asking about a
# field that is not there is the #426 failure this path keeps reproducing.
code, out, err, stub = run_cli(["hello", "--questions", KEYED_PATH], router=QuestionRecorder())
check("questions: declared state_key used", stub.states[0] == {"body": "hello"}, str(stub.states[0]))
check("questions: questions unwrapped",
      sorted(stub.questions) == ["spam"], str(sorted(stub.questions or {})))

# The CLI validates the file's shape and defers question semantics to core, so a question set is
# handed over unchanged rather than normalised twice.
code, out, err, stub = run_cli(["hi", "--questions", write_json("odd.json", {"a": {"type": "bogus"}})],
                               router=QuestionRecorder())
check("questions: nonsense type reaches predict unchanged",
      stub.questions == {"a": {"type": "bogus"}}, str(stub.questions))

for name, payload, fragment in (
        ("empty.json", {}, "holds no questions"),
        ("list.json", [1], "must be a JSON object"),
        ("scalar.json", {"a": 5}, "must map to an object"),
        ("badwrap.json", {"questions": [1]}, "'questions' field"),
        ("badkey.json", {"state_key": "", "questions": {"a": {"type": "noul"}}}, "'state_key'"),
):
    code, out, err, stub = run_cli(["hi", "--questions", write_json(name, payload)],
                                   router=QuestionRecorder())
    check("questions %s: exit code 2" % name, code == 2, "got %r" % code)
    check("questions %s: names the problem" % name, fragment in err, err)

code, out, err, stub = run_cli(["hi", "--questions", os.path.join(_TMP, "missing.json")],
                               router=QuestionRecorder())
check("questions: missing file is a handled error",
      code == 2 and "no such --questions file" in err, "code %r, err %r" % (code, err))

code, out, err, stub = run_cli(["hi", "--questions", INTENTS_PATH, "--preset", "triage"],
                               router=QuestionRecorder())
check("questions: two sources refuse each other",
      code == 2 and "pass one" in err, "code %r, err %r" % (code, err))

code, out, err, stub = run_cli(["hi", "--questions", INTENTS_PATH, "--json"],
                               router=QuestionRecorder())
check("questions: --json prints the raw answers", code == 0 and '"intent"' in out,
      "code %r, out %r" % (code, out))

# The same file answers every preset's questions too, so --questions replaces none of them.
code, out, err, stub = run_cli(["hi", "--questions", INTENTS_PATH, "--model", "english"],
                               router=QuestionRecorder())
check("questions: --model still forwarded", stub.kwargs.get("model") == "english", str(stub.kwargs))

# --------------------------------------------------------------------- the token budget
# `--questions` is what makes a many-label question reachable from the shell, and a many-label
# question is exactly what overflows the shared option budget. Both halves have to be here: on
# `laya` (head budget 192) the 58-label Massive-Intent question scores 24/58 by default and
# 34/58 at --head-max-len 384, so the CLI that can ask the question must also be able to fit it.
# Nothing set has to mean nothing sent: the checkpoint's own budget stays in charge, and a
# `max_len=None` in the call would be a different promise from the one `predict` documents.
code, out, err, stub = run_cli(["hi", "--questions", INTENTS_PATH], router=QuestionRecorder())
check("budget: default sends no override", "max_len" not in stub.kwargs and
      "head_max_len" not in stub.kwargs, str(stub.kwargs))

for argv, expected in (
        (["--max-len", "1024"], {"max_len": 1024}),
        (["--head-max-len", "384"], {"head_max_len": 384}),
        (["--max-len", "1024", "--head-max-len", "384"], {"max_len": 1024, "head_max_len": 384}),
        # A zero budget is a real value, not an absent one.
        (["--head-max-len", "0"], {"head_max_len": 0}),
):
    code, out, err, stub = run_cli(["hi", "--questions", INTENTS_PATH] + argv, router=QuestionRecorder())
    sent = {k: v for k, v in stub.kwargs.items() if k in ("max_len", "head_max_len")}
    check("budget %s: forwarded" % " ".join(argv), sent == expected, "%r != %r" % (sent, expected))

# The budget belongs to the request, not to how the questions arrived.
for argv in (["--predict"], ["--preset", "triage"], ["--questions", INTENTS_PATH]):
    code, out, err, stub = run_cli(["hi"] + argv + ["--head-max-len", "512"], router=QuestionRecorder())
    check("budget on %s: forwarded" % " ".join(argv), stub.kwargs.get("head_max_len") == 512,
          str(stub.kwargs))
    check("budget on %s: questions unaffected" % " ".join(argv), stub.questions is not None)

# A head budget cannot exceed the request budget it lives inside; that is core's rule, so the CLI
# passes the pair through and reports core's words rather than inventing its own.
class BudgetRejectingRouter(QuestionRecorder):
    def predict(self, state, questions, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("head_max_len", 0) > kwargs.get("max_len", 1 << 30):
            raise ValueError("head_max_len must not exceed max_len")
        return {"answers": {}}


code, out, err, stub = run_cli(["hi", "--questions", INTENTS_PATH, "--max-len", "128",
                               "--head-max-len", "384"], router=BudgetRejectingRouter())
check("budget: a rejected pair exits 2", code == 2, "got %r" % code)
check("budget: core's message reaches the shell", "head_max_len must not exceed max_len" in err, err)
check("budget: both flags reached predict together",
      (stub.kwargs.get("max_len"), stub.kwargs.get("head_max_len")) == (128, 384), str(stub.kwargs))

for value in ("abc", ""):
    code, out, err, stub = run_cli(["hi", "--questions", INTENTS_PATH, "--head-max-len=%s" % value],
                                   router=QuestionRecorder())
    check("budget %r: rejected by argparse" % value, code == 2 and "invalid int value" in err,
          "code %r, err %r" % (code, err))

# --------------------------------------------------------------- --min-confidence (#361's gate)
# The gate itself is core's: `Router.predict` and `Router.predict_batch` both run the argument
# through `laya.confidence.check_min_confidence` and then `flag_low_confidence`, which marks an
# answer and keeps it. What this section covers is the CLI's half -- the flag reaches the call, an
# unset flag is not sent as `None`, a value outside core's range is refused in core's words,
# routing refuses the flag instead of exiting 0 having ignored it, and the mark a gated run
# produces is printed to a terminal rather than only reachable through --json.
import inspect  # noqa: E402

from laya import Router as _Router  # noqa: E402
from laya.confidence import check_min_confidence as _core_check  # noqa: E402


class GatedRouter(QuestionRecorder):
    """Returns one answer, optionally with the mark `flag_low_confidence` adds."""

    def __init__(self, mark=True, kind="choice"):
        super().__init__()
        self.mark = mark
        self.kind = kind

    def predict(self, state, questions, **kwargs):
        self.kwargs = kwargs
        answer = {"probabilities": {"refund": 0.31, "billing": 0.4}}
        if self.kind == "choice":
            answer["choice"] = "refund"
        else:
            answer["score"] = 1.0
        answer["answer_confidence"] = 0.31
        if self.mark:
            answer["low_confidence"] = True
        return {"answers": {"intent": answer}, "routing": dict(StubDecision())}


code, out, err, stub = run_cli(["--predict", "--min-confidence", "0.5", "hello"],
                               router=GatedRouter(mark=False))
check("threshold: forwarded to predict", stub.kwargs.get("min_confidence") == 0.5, str(stub.kwargs))
check("threshold: exit code", code == 0, "got %r %s" % (code, err))

code, out, err, stub = run_cli(["--predict", "hello"], router=GatedRouter(mark=False))
check("threshold: unset is not sent as None", "min_confidence" not in stub.kwargs, str(stub.kwargs))

# `0.0` is the one threshold core itself treats as a no-op (`flag_low_confidence` returns early on
# it), but it is still a value a caller chose rather than an absence; `0` and `1` are core's
# inclusive bounds.
for raw, want in (("0", 0.0), ("0.0", 0.0), ("1", 1.0)):
    code, out, err, stub = run_cli(["--predict", "--min-confidence", raw, "hello"],
                                   router=GatedRouter(mark=False))
    check("threshold %r: still a threshold, not an absence" % raw,
          code == 0 and stub.kwargs.get("min_confidence") == want,
          "code %r, got %r" % (code, stub.kwargs.get("min_confidence", "<absent>")))

# The accepted range is core's, so the CLI's verdict on a value has to match core's own -- the
# same check `predict` runs, not a copy of its bounds.
for raw in ("0", "0.5", "1", "0.9", "1.5", "-0.1", "nan", "inf", "1e999", "true", "abc", ""):
    try:
        _core_check(float(raw))
        accepted_by_core = True
    except (TypeError, ValueError):
        accepted_by_core = False
    code, out, err, stub = run_cli(["--predict", "--min-confidence", raw, "hello"],
                                   router=GatedRouter(mark=False))
    check("threshold %r: same verdict as core" % raw,
          (code == 0) == accepted_by_core,
          "code %r, core accepts %r, err %r" % (code, accepted_by_core, err))
    check("threshold %r: never a traceback" % raw, "Traceback" not in err, err)

code, out, err, stub = run_cli(["--predict", "--min-confidence", "1.5", "hello"],
                               router=GatedRouter(mark=False))
check("threshold: an out-of-range value is refused in core's words",
      code == 2 and "min_confidence must be a float in [0.0, 1.0], got 1.5" in err, err)
code, out, err, stub = run_cli(["--predict", "--min-confidence", "inf", "hello"],
                               router=GatedRouter(mark=False))
check("threshold: infinity is refused, not accepted as a gate",
      code == 2 and "min_confidence must be a float in [0.0, 1.0]" in err, err)

# Routing answers nothing, so there is no confidence to gate: `route` does not take the argument.
# Exiting 0 here would be the silent-ignore this flag exists to avoid.
code, out, err, stub = run_cli(["--min-confidence", "0.5", "hello"], router=GatedRouter())
check("threshold: routing mode refuses it", code == 2, "got %r" % code)
check("threshold: the refusal names the flag and an answering mode",
      "--min-confidence" in err and "--predict" in err, err)
check("threshold: nothing was routed", stub.route_calls == [] and stub.kwargs == {}, str(stub.kwargs))

# The refusal is about routing, not about the flag being unwelcome: every answering mode takes it,
# including the two that reach answering by implying `--predict` rather than being told.
for argv, label in ((["--predict", "hello"], "--predict"),
                    (["--preset", "triage", "hello"], "--preset"),
                    (["--questions", INTENTS_PATH, "hello"], "--questions")):
    code, out, err, stub = run_cli(argv + ["--min-confidence", "0.5"], router=GatedRouter(mark=False))
    check("threshold: %s mode takes it" % label,
          code == 0 and stub.kwargs.get("min_confidence") == 0.5,
          "code %r, kwargs %r, err %r" % (code, stub.kwargs, err))

code, out, err, stub = run_batch_cli(["--batch", path, "--min-confidence", "0.5"])
check("threshold: batch routing refuses it too", code == 2 and stub.predict_batch_calls == [],
      "code %r, calls %r" % (code, stub.predict_batch_calls))


class ThresholdBatchRouter(BatchRouter):
    """Records the call-level keyword arguments of `predict_batch` separately from the requests."""

    def __init__(self):
        super().__init__()
        self.call_kwargs = []

    def predict_batch(self, requests, batch_size=None, **call_kwargs):
        self.call_kwargs.append(call_kwargs)
        return super().predict_batch(requests, batch_size=batch_size)


code, out, err, stub = run_batch_cli(["--batch", path, "--predict", "--min-confidence", "0.5"],
                                     router=ThresholdBatchRouter())
requests = stub.predict_batch_calls[0][0]
check("threshold batch: sent to predict_batch as a call argument",
      code == 0 and stub.call_kwargs == [{"min_confidence": 0.5}],
      "code %r, kwargs %r" % (code, stub.call_kwargs))
# `predict_batch` validates one threshold for the whole batch and has no per-request form, so a
# key inside the request dicts would be read by nothing.
check("threshold batch: not stuffed into each request",
      all("min_confidence" not in r for r in requests), str(requests))

code, out, err, stub = run_batch_cli(["--batch", path, "--predict"], router=ThresholdBatchRouter())
check("threshold batch: unset sends nothing", stub.call_kwargs == [{}], str(stub.call_kwargs))

# Rendering: the mark has to reach the terminal, or --json is the only way to learn the model was
# unsure -- which is how the gate reads as a no-op to anyone not parsing JSON.
code, out, err, stub = run_cli(["--predict", "--min-confidence", "0.5", "hello"],
                               router=GatedRouter(mark=True))
check("threshold: the printed line carries the mark", "[low-confidence]" in out, out)
check("threshold: the answer itself is still printed", "refund" in out, out)
code, out, err, stub = run_cli(["--predict", "hello"], router=GatedRouter(mark=False))
check("threshold: an unmarked answer prints no mark", "[low-confidence]" not in out, out)
code, out, err, stub = run_cli(["--predict", "--min-confidence", "0.5", "hello"],
                               router=GatedRouter(mark=True, kind="score"))
check("threshold: a score answer is marked too", "[low-confidence]" in out and "1.00" in out, out)
code, out, err, stub = run_cli(["--predict", "--min-confidence", "0.5", "--json", "hello"],
                               router=GatedRouter(mark=True))
check("threshold: --json still carries the raw key",
      '"low_confidence": true' in out, out)

# Drift pin, the other direction: every control `predict` takes that the parser can name must
# arrive in the call, and the ones it cannot name must be the accepted three. A control added to
# `predict` lands in the second check and has to be placed -- flagged and forwarded, or written
# down as something this command cannot mean -- before the suite passes again.
PREDICT_CONTROLS = {p for p in inspect.signature(_Router.predict).parameters
                    if p not in ("self", "state", "questions")}
NAMED = set(vars(cli.build_parser().parse_args(["hi"])))
code, out, err, stub = run_cli(["--predict", "--model", "english", "--task", "multi",
                                "--lang", "de", "--max-len", "1024", "--head-max-len", "512",
                                "--min-confidence", "0.5", "hello"], router=GatedRouter(mark=False))
check("drift: every control the parser names a flag for arrives in the call",
      all(control in stub.kwargs for control in (PREDICT_CONTROLS & NAMED)),
      "missing %s" % sorted((PREDICT_CONTROLS & NAMED) - set(stub.kwargs)))
# `lang_guess` has had its own flag since #795, so what is left is the hook arguments:
# `hooks_raise` / `hooks_timeout` govern hooks and `make_router` installs none, so there is
# nothing in this process for them to change, and the three callables cannot come off a shell.
check("drift: the controls with no flag are exactly the accepted five",
      sorted(PREDICT_CONTROLS - NAMED)
      == ["hooks", "hooks_raise", "hooks_timeout",
          "on_predict_end", "on_predict_start"],
      sorted(PREDICT_CONTROLS - NAMED))

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all CLI tests passed")
sys.exit(1 if FAIL else 0)
