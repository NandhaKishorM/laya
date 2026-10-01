"""The CLI's `--lang-guess` must reach core's soft language hint. No weights are loaded."""
import io
import os
import sys
import inspect
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
    """Records the kwargs of every route/predict call the CLI makes."""

    def __init__(self):
        self.route_calls = []
        self.predict_calls = []

    def route(self, state, **kwargs):
        self.route_calls.append((state, kwargs))
        return StubDecision()

    def predict(self, state, questions, **kwargs):
        self.predict_calls.append((state, kwargs))
        return {"answers": {"difficulty": {"score": 1.4}}, "routing": dict(StubDecision())}


class BatchRouter:
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


def run_cli(argv, router=None):
    stub = router or StubRouter()
    original = cli.make_router
    cli.make_router = lambda args: stub
    out, err = io.StringIO(), io.StringIO()
    code = None
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    except Exception as error:  # noqa: BLE001 - a forwarding bug should fail a named check, not
        code = "raised: %s" % error.__class__.__name__  # crash the suite before the report prints
    finally:
        cli.make_router = original
    return code, out.getvalue(), err.getvalue(), stub


def first_kwargs(calls):
    """The kwargs dict of the first recorded call, or {} when none ran.

    Read defensively so a broken flag fails its own named check instead of raising IndexError
    and aborting the run before the report prints -- a red that names the check is the witness.
    """
    return calls[0][1] if calls else {}


def parse_value(argv):
    """Parse argv through the real parser, or None if argparse rejects it (an unknown flag)."""
    try:
        with redirect_stderr(io.StringIO()):
            return cli.build_parser().parse_args(argv)
    except SystemExit:
        return None


# ------------------------------------------------------------------ the parser knows the flag
ns = parse_value(["t", "--lang-guess", "de"])
check("parse: --lang-guess reaches the namespace",
      ns is not None and getattr(ns, "lang_guess", None) == "de", repr(ns))
check("parse: omitted means no hint (None), not '' or 'auto'",
      getattr(parse_value(["t"]), "lang_guess", "MISSING") is None,
      repr(getattr(parse_value(["t"]), "lang_guess", "MISSING")))

# `--lang` forces a language and skips detection; `--lang-guess` is the soft sibling that
# participates in routing. Two different parameters with two different contracts, so the flag
# must drive the one core names `lang_guess` -- and the names are core's, not a copy.
from laya.router import Router as _CoreRouter  # noqa: E402

check("core: Router.route really takes lang_guess",
      "lang_guess" in inspect.signature(_CoreRouter.route).parameters,
      str(list(inspect.signature(_CoreRouter.route).parameters)))
check("core: Router.predict really takes lang_guess",
      "lang_guess" in inspect.signature(_CoreRouter.predict).parameters,
      str(list(inspect.signature(_CoreRouter.predict).parameters)))

# ------------------------------------------------------------------ single-request forwarding
code, out, err, stub = run_cli(["Ich wurde doppelt belastet", "--lang-guess", "de"])
check("route: exit code", code == 0, "got %r, err %r" % (code, err))
check("route: --lang-guess reaches Router.route",
      first_kwargs(stub.route_calls).get("lang_guess") == "de", str(first_kwargs(stub.route_calls)))

code, out, err, stub = run_cli(["My payment failed twice", "--predict", "--lang-guess", "en"])
check("predict: exit code", code == 0, "got %r, err %r" % (code, err))
check("predict: --lang-guess reaches Router.predict",
      first_kwargs(stub.predict_calls).get("lang_guess") == "en", str(first_kwargs(stub.predict_calls)))

# Unset is a real state, not a missing one: core reads `lang_guess=None` as "no per-call hint,
# fall through to Router(lang_guess=...) and detection", so the CLI must send None -- not omit the
# keyword and not send the empty string, which would name a language that is not there.
code, out, err, stub = run_cli(["hello", "--predict"])
sent = first_kwargs(stub.predict_calls)
check("predict: unset sends lang_guess=None (a fall-through, not a blank hint)",
      "lang_guess" in sent and sent["lang_guess"] is None, str(sent))
code, out, err, stub = run_cli(["hello"])
sent = first_kwargs(stub.route_calls)
check("route: unset sends lang_guess=None",
      "lang_guess" in sent and sent["lang_guess"] is None, str(sent))

# --lang (decisive) and --lang-guess (soft) are separate parameters; setting both must forward
# both under their own keys rather than one clobbering the other.
code, out, err, stub = run_cli(["hi", "--predict", "--lang", "de", "--lang-guess", "fr"])
sent = first_kwargs(stub.predict_calls)
check("both flags forwarded under distinct keys",
      sent.get("lang") == "de" and sent.get("lang_guess") == "fr", str(sent))

# ------------------------------------------------------------------ batch forwarding
import tempfile  # noqa: E402

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "requests.txt")
with open(path, "w", encoding="utf-8") as handle:
    handle.write("first ticket\nsecond ticket\n")


def first_batch_requests(calls):
    return calls[0][0] if calls else []


code, out, err, stub = run_cli(["--batch", path, "--predict", "--lang-guess", "de"],
                               router=BatchRouter())
requests = first_batch_requests(stub.predict_batch_calls)
check("batch predict: exit code", code == 0, "got %r, err %r" % (code, err))
check("batch predict: every request carries lang_guess",
      requests and all(r.get("lang_guess") == "de" for r in requests),
      str([sorted(r) for r in requests]))

requests = run_cli(["--batch", path, "--lang-guess", "de"], router=BatchRouter())[3].route_batch_calls
requests = requests[0] if requests else []
check("batch route: every request carries lang_guess",
      requests and all(r.get("lang_guess") == "de" for r in requests),
      str([sorted(r) for r in requests]))

# Batch shares the same `is not None` rule the model/task/lang overrides already use: an unset
# hint is absent from the request dict, so route()/predict() see their own default, not a forced
# None that could shadow a Router(lang_guess=...) the caller installed.
code, out, err, stub = run_cli(["--batch", path, "--predict"], router=BatchRouter())
requests = first_batch_requests(stub.predict_batch_calls)
check("batch predict: unset hint is absent, not sent as None",
      requests and all("lang_guess" not in r for r in requests),
      str([sorted(r) for r in requests]))

# ------------------------------------------------------------------ drift guard vs the serve surface
# `laya.serve` forwards lang_guess as a per-call body control. The CLI is the same wrapper class,
# so the routing hints it can send should cover the ones the HTTP surface can send. If core ever
# renames or drops lang_guess from that set, this fails rather than the CLI quietly going stale.
from laya import serve as _serve  # noqa: E402

check("parity: lang_guess is a laya.serve body control",
      "lang_guess" in _serve.BODY_CONTROLS, str(_serve.BODY_CONTROLS))
parsed = parse_value(["t"])
namespace_keys = set(vars(parsed)) if parsed is not None else set()
check("parity: the CLI flag set covers the routing controls serve forwards",
      {"model", "task", "lang", "lang_guess"} <= namespace_keys, str(sorted(namespace_keys)))

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all lang-guess CLI tests passed")
sys.exit(1 if FAIL else 0)
