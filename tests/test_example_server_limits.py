"""Regression: examples/server.py must bound a request the way laya.serve does.

`laya/serve.py` refuses a request carrying more than MAX_QUESTIONS questions, a state
over MAX_STATE_CHARS, or a body over its own cap, because Laya encodes the state once
per question -- cost is questions x state size, collated into one tensor.

examples/server.py bounded `states` to 64 and left the rest open: 20 000 questions and
a 5 MB state were both accepted where the shipped server answers 413. The bounds are
read from laya.serve rather than restated, so the two cannot drift.

Scope: the question count and the state size, answered 413 as laya.serve answers them.
`Question.instructions` and `criteria` still carry unbounded text that no per-field
bound can see; capping the request body is the backstop for those and is left out
deliberately -- see the PR description.

Driven over HTTP through TestClient. No weights are loaded; the router stays unbuilt, so
a request that passes validation answers 503, which is the assertion for "accepted".

Run: python tests/test_example_server_limits.py
"""
import json
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


def main():
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        # Only the optional serving stack may be missing. An ImportError naming anything
        # else -- in particular `cannot import name MAX_QUESTIONS from laya.serve`, the
        # drift this test exists to catch -- must fail rather than skip.
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

    from laya.serve import MAX_QUESTIONS, MAX_STATE_CHARS

    client = TestClient(demo.app, raise_server_exceptions=False)
    one = {"a": {"type": "noul", "instructions": "x"}}

    def questions(n):
        return {"q%d" % i: {"type": "noul", "instructions": "x"} for i in range(n)}

    def code(**kw):
        return client.post("/predict", **kw).status_code

    # The bounds must come from laya.serve, not a local copy, or the two drift.
    demo_q = getattr(demo, "MAX_QUESTIONS", None)
    demo_s = getattr(demo, "MAX_STATE_CHARS", None)
    ok("the per-request bounds are laya.serve's",
       demo_q == MAX_QUESTIONS and demo_s == MAX_STATE_CHARS,
       "demo %r/%r vs laya.serve %r/%r" % (demo_q, demo_s, MAX_QUESTIONS, MAX_STATE_CHARS))

    # --- too many questions, too large a state: 413, as laya.serve answers ---
    ok("more than MAX_QUESTIONS questions is 413",
       code(json={"state": "hi", "questions": questions(MAX_QUESTIONS + 1)}) == 413)
    # A noul question carries no answer options, so an options-based budget cannot see
    # it; the count is what has to be bounded.
    ok("a noul-only flood is 413 (it carries no answer options)",
       code(json={"state": "hi", "questions": questions(20_000)}) == 413)
    ok("a state over MAX_STATE_CHARS is 413",
       code(json={"state": "A" * (MAX_STATE_CHARS + 1), "questions": one}) == 413)
    ok("an oversized dict state is 413",
       code(json={"state": {"body": "A" * (MAX_STATE_CHARS + 1)}, "questions": one}) == 413)
    ok("an oversized state inside a batch is 413",
       client.post("/predict/batch",
                   json={"states": ["hi", "A" * (MAX_STATE_CHARS + 1)], "questions": one}
                   ).status_code == 413)


    # --- the refusal must not echo the rejected payload back ----------------
    # Declaring these as Field(max_length=...) instead would report 422 *and* include
    # the offending `input` in FastAPI's validation-error body, so refusing a 5 MB
    # state would write 5 MB back to the caller -- a size limit that amplifies.
    big = {"state": "A" * 5_000_000, "questions": one}
    sent = len(json.dumps(big).encode())
    resp = client.post("/predict", json=big)
    ok("a rejected 5 MB state answers 413 without echoing it",
       resp.status_code == 413 and len(resp.content) < 1_000,
       "%s, %d bytes returned for %d sent" % (resp.status_code, len(resp.content), sent))

    # --- a genuine schema error is still 422, not 413 -----------------------
    ok("an unknown question type is 422",
       code(json={"state": "hi", "questions": {"a": {"type": "bogus", "instructions": "x"}}}) == 422)
    ok("a choice question with no criteria is 422",
       code(json={"state": "hi", "questions": {"a": {"type": "choice", "instructions": "x"}}}) == 422)
    ok("an empty state is 422", code(json={"state": "", "questions": one}) == 422)
    ok("zero questions is 422", code(json={"state": "hi", "questions": {}}) == 422)

    # --- the limits are limits, not walls -----------------------------------
    # No router is built, so anything that passes validation answers 503.
    ok("an ordinary request passes validation",
       code(json={"state": {"body": "billed twice, please refund"}, "questions": one}) == 503)
    ok("exactly MAX_QUESTIONS passes",
       code(json={"state": "hi", "questions": questions(MAX_QUESTIONS)}) == 503)
    ok("a state of exactly MAX_STATE_CHARS passes",
       code(json={"state": "A" * MAX_STATE_CHARS, "questions": one}) == 503)
    ok("a list state passes", code(json={"state": ["a", "b"], "questions": one}) == 503)
    ok("a small chunked body passes",
       code(content=(lambda: (yield json.dumps({"state": "hi", "questions": one}).encode()))(),
            headers={"content-type": "application/json"}) == 503)
    # /predict/batch reports per-item failures inside a 200 envelope, so an accepted
    # batch is a 200 here; over the state bound it is refused outright, and 413 rather
    # than 422 because "too many states" is a size violation like the others.
    ok("a 64-state batch is accepted",
       client.post("/predict/batch", json={"states": ["hi"] * 64, "questions": one}).status_code == 200)
    ok("a 65-state batch is still refused by the existing bound",
       client.post("/predict/batch", json={"states": ["hi"] * 65, "questions": one}).status_code == 422)
    ok("the page and health endpoints are unaffected",
       client.get("/").status_code == 200 and client.get("/health").status_code == 200)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
