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
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
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

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all CLI tests passed")
sys.exit(1 if FAIL else 0)
