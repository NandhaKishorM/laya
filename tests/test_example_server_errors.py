"""Regression: examples/server.py must not echo exception text to a caller.

`laya/serve.py` answers an inference failure with a fixed 500 (`"inference failed"`) and
keeps the cause in the log -- see `test_inference_failure_is_logged_and_not_leaked` in
tests/test_serve.py, which is where that policy is asserted. The demo server did the
opposite on all three surfaces: `f"{type(exc).__name__}: {exc}"` was returned by
`/predict`, by `/predict/batch` in its per-item `error` field, and rendered into the
`/gui` error page. Exception text names paths, libraries and memory sizes -- it describes
the deployment, not the request -- and code scanning flags it as stack-trace exposure
(py/stack-trace-exposure).

Scope: an unexpected failure. A 4xx stays as it was: those messages name the question or
the field to fix and are meant for the caller (`/predict` answers 422 with the reason, and
the GUI lists one line per validation error).

The router is replaced with one that raises, so no weights are needed and the failure is
deterministic. The exception carries marker text that must reach the log and must not
appear in any response.

Run: python tests/test_example_server_errors.py
"""
import json
import logging
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LAYA_PRELOAD", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, ("  -- " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail and not cond else ""), flush=True)


# The shape of the real failure this policy exists for (#365): triton's JIT with no
# compiler, naming a virtualenv path. Nothing here may reach a response.
SECRET = ("Failed to find C compiler. Please specify via CC environment variable or set "
          "triton.knobs.build.impl (/opt/venv/lib/python3.11/site-packages/triton)")
LEAKS = ("C compiler", "/opt/venv", "site-packages", "triton", "RuntimeError")


class ExplodingRouter:
    """Only what the demo calls: predict, plus the `loaded` list health reports."""

    loaded = ["english"]

    def predict(self, state, questions, **kw):
        raise RuntimeError(SECRET)


class Collect(logging.Handler):
    """Records, for the assertion that the cause reached the log."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def main():
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        if (getattr(exc, "name", None) or "").split(".")[0] not in ("fastapi", "httpx", "starlette"):
            raise
        print("SKIP: fastapi/httpx not installed -- pip install laya[serve] httpx")
        return 0
    try:
        import server as demo
    except ImportError as exc:
        if (getattr(exc, "name", None) or "").split(".")[0] not in ("fastapi", "httpx", "starlette", "multipart"):
            raise
        print("SKIP: examples/server.py needs the serve extra -- pip install laya[serve]")
        return 0

    # The lifespan (which builds a real Router) is not entered: TestClient without a
    # `with` block leaves the app unstarted, so the stub below is what `_predict` finds.
    demo.ROUTER = ExplodingRouter()
    client = TestClient(demo.app, raise_server_exceptions=False)

    handler = Collect()
    logger = logging.getLogger("laya.example-server")
    logger.addHandler(handler)
    # WARNING, not ERROR: the parse-path fallback answers the caller and logs at warning level --
    # a client's bad request is not a server error, but its cause still belongs in the log.
    logger.setLevel(logging.WARNING)

    one = {"a": {"type": "noul", "instructions": "x"}}

    # --- /predict: a bare 500, and no exception text in it ------------------
    r = client.post("/predict", json={"state": "hi", "questions": one})
    ok("a failed /predict is a 500", r.status_code == 500, str(r.status_code))
    ok("it answers the fixed message", r.json() == {"detail": "prediction failed"}, r.text[:200])
    ok("no exception text reaches the caller",
       not any(s in r.text for s in LEAKS), r.text[:200])

    # --- /predict/batch: per-item, the index identifies it -------------------
    r = client.post("/predict/batch", json={"states": ["hi", "ho"], "questions": one})
    errors = [item.get("error") for item in (r.json().get("results") or [])]
    ok("a failed batch item reports the fixed message",
       r.status_code == 200 and errors == ["prediction failed", "prediction failed"], repr(errors))
    ok("no exception text reaches a batch caller",
       not any(s in r.text for s in LEAKS), r.text[:200])

    # --- /gui: the page says what happened, not what raised ------------------
    r = client.post("/gui", json={"state": "hi", "questions": one})
    ok("a failed /gui is an error page", r.status_code == 200 and "Prediction failed" in r.text)
    ok("the page carries no exception text",
       not any(s in r.text for s in LEAKS), r.text[:400])

    # --- the cause is in the log, traceback included -------------------------
    causes = [r.exc_info[1] for r in handler.records if r.exc_info]
    ok("each failure logged its cause with a traceback",
       len(causes) == 4, "%d records, %d with exc_info" % (len(handler.records), len(causes)))
    ok("the logged cause is the real exception",
       all(str(c) == SECRET for c in causes), str(causes[:1])[:200])

    # --- a 4xx is the caller's mistake and keeps its message -----------------
    # The fix must not swallow what a caller needs to correct a request.
    r = client.post("/predict", json={"state": "hi",
                                      "questions": {"a": {"type": "bogus", "instructions": "x"}}})
    ok("a validation error is still 422 with its reason",
       r.status_code == 422 and "bogus" in r.text, "%s %s" % (r.status_code, r.text[:200]))

    # --- a limit the /gui page reports is built from the request -------------
    # The page used to read its line back off the raised HTTPException. It names the limit and
    # the number that broke it either way -- what changed is where the words come from.
    one_q = {"a": {"type": "noul", "instructions": "x"}}
    r = client.post("/gui", json={"state": "x" * (demo.MAX_STATE_CHARS + 1), "questions": one_q})
    ok("an oversized /gui names the limit",
       r.status_code == 200 and "state too large" in r.text, r.text[:300])
    ok("the limit page carries no exception text",
       not any(s in r.text for s in LEAKS), r.text[:300])

    # --- an unclassified failure on the parse path keeps its cause in the log -
    before = len(handler.records)
    lines = demo._error_lines(ValueError(SECRET))
    ok("an unreadable request answers a fixed line",
       lines == ["The request could not be read."], repr(lines))
    logged = [rec.exc_info[1] for rec in handler.records[before:] if rec.exc_info]
    ok("and its cause still reaches the log", [str(c) for c in logged] == [SECRET], str(logged)[:200])

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
