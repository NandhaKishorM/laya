"""Regression: examples/server.py must bound a request the way laya.serve does.

`laya/serve.py` refuses a request carrying more than MAX_QUESTIONS questions, a state
over MAX_STATE_CHARS, or a body over its own cap, because Laya encodes the state once
per question -- cost is questions x state size, collated into one tensor.

examples/server.py bounded `states` to 64 and left the rest open: 20 000 questions and
a 5 MB state were both accepted where the shipped server answers 413. The bounds are
read from laya.serve rather than restated, so the two cannot drift.

Scope: the question count, the state size, and the per-question and total option counts,
answered 413 as laya.serve answers them. The option budgets matter for the same reason the
other two do -- a choice or score question encodes one sequence per option, and those
sequences share the head budget -- so a request laya.serve refuses with 413 must not be
answered here. The bounds are read from laya.serve, never restated.
`Question.instructions` and `criteria` still carry unbounded *text* that no per-field
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

    from laya.serve import (
        MAX_CHOICE_OPTIONS,
        MAX_QUESTIONS,
        MAX_SCORE_LEVELS,
        MAX_STATE_CHARS,
        MAX_TOTAL_OPTIONS,
    )

    client = TestClient(demo.app, raise_server_exceptions=False)
    one = {"a": {"type": "noul", "instructions": "x"}}

    def questions(n):
        return {"q%d" % i: {"type": "noul", "instructions": "x"} for i in range(n)}

    def choice(n_opts, qid="a"):
        return {qid: {"type": "choice", "instructions": "Which?",
                      "criteria": {"opt%d" % i: "desc %d" % i for i in range(n_opts)}}}

    def score(n_levels, qid="a"):
        return {qid: {"type": "score", "instructions": "How bad?",
                      "criteria": ["level %d" % i for i in range(n_levels)]}}

    def code(**kw):
        return client.post("/predict", **kw).status_code

    # The bounds must come from laya.serve, not a local copy, or the two drift.
    demo_q = getattr(demo, "MAX_QUESTIONS", None)
    demo_s = getattr(demo, "MAX_STATE_CHARS", None)
    ok("the per-request bounds are laya.serve's",
       demo_q == MAX_QUESTIONS and demo_s == MAX_STATE_CHARS,
       "demo %r/%r vs laya.serve %r/%r" % (demo_q, demo_s, MAX_QUESTIONS, MAX_STATE_CHARS))
    demo_opts = (getattr(demo, "MAX_CHOICE_OPTIONS", None),
                 getattr(demo, "MAX_SCORE_LEVELS", None),
                 getattr(demo, "MAX_TOTAL_OPTIONS", None))
    ok("the option budgets are laya.serve's too",
       demo_opts == (MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS, MAX_TOTAL_OPTIONS),
       "demo %r vs laya.serve %r" % (demo_opts, (MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS,
                                                  MAX_TOTAL_OPTIONS)))

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

    # --- the option budgets, which laya.serve also answers 413 -----------------
    # A choice or score question encodes one sequence per option against a shared head
    # budget, so the option counts are size limits for the same reason the state is.
    ok("more than MAX_CHOICE_OPTIONS options in one question is 413",
       code(json={"state": "hi", "questions": choice(MAX_CHOICE_OPTIONS + 1)}) == 413)
    ok("more than MAX_SCORE_LEVELS levels in one question is 413",
       code(json={"state": "hi", "questions": score(MAX_SCORE_LEVELS + 1)}) == 413)
    # Under the per-question caps but over the shared total: the case a per-question check
    # alone would still accept.
    per_q = MAX_CHOICE_OPTIONS
    n_questions = MAX_TOTAL_OPTIONS // per_q + 1
    ok("over MAX_TOTAL_OPTIONS across questions is 413",
       code(json={"state": "hi",
                  "questions": {"q%d" % i: choice(per_q, "q%d" % i)[ "q%d" % i]
                                for i in range(n_questions)}}) == 413,
       "%d questions x %d options" % (n_questions, per_q))
    # A noul question contributes no options, so it cannot move the total either way.
    ok("a noul flood is still bounded by the question count, not the option total",
       code(json={"state": "hi", "questions": questions(MAX_QUESTIONS)}) == 503)
    # ...including when it carries the ordinary false/true criteria, which are option
    # *texts* but not answer options. laya.serve adds them nowhere; a total that counted
    # them would refuse a request serve accepts, which is the drift this file exists to
    # catch -- in the direction of refusing something legal.
    counted = 5 * MAX_CHOICE_OPTIONS + 12  # exactly MAX_TOTAL_OPTIONS by laya.serve's count
    mixed = {"q%d" % i: choice(MAX_CHOICE_OPTIONS, "q%d" % i)["q%d" % i] for i in range(5)}
    mixed["last"] = choice(12, "last")["last"]
    for i in range(30):  # 30 noul questions, 2 criteria each, 60 if wrongly counted
        mixed["n%d" % i] = {"type": "noul", "instructions": "True?",
                            "criteria": {"false": "no", "true": "yes"}}
    ok("noul criteria do not count toward the shared option total",
       len(mixed) <= MAX_QUESTIONS and counted == MAX_TOTAL_OPTIONS
       and code(json={"state": "hi", "questions": mixed}) == 503,
       "%d questions, %d options by laya.serve's count" % (len(mixed), counted))


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
    ok("exactly MAX_CHOICE_OPTIONS options passes",
       code(json={"state": "hi", "questions": choice(MAX_CHOICE_OPTIONS)}) == 503)
    ok("exactly MAX_SCORE_LEVELS levels passes",
       code(json={"state": "hi", "questions": score(MAX_SCORE_LEVELS)}) == 503)
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

    # --- the two surfaces must reach the same verdict on the same numbers ---------
    # The demo reads `Question` instances where serve reads plain dicts, so the counting
    # lives behind a shape branch that could drift from serve's. Pin the verdicts against
    # each other rather than restating either one: serve's validator is called with the
    # plain dicts it actually receives, the demo through HTTP with the models it actually
    # receives, and the two have to agree.
    import laya.serve as _serve_mod  # noqa: E402

    def serve_verdict(state, questions):
        try:
            _serve_mod._check_request_limits(state, questions)
        except Exception:  # noqa: BLE001 -- HTTPException carries the 413
            return "refused"
        return "accepted"

    def demo_verdict(state, questions):
        return "refused" if client.post("/predict", json={"state": state,
                                                          "questions": questions}).status_code == 413 else "accepted"

    noul_crit = {"type": "noul", "instructions": "True?",
                 "criteria": {"false": "no", "true": "yes"}}
    parity_cases = [
        ("101 choice options", "hi", choice(MAX_CHOICE_OPTIONS + 1)),
        ("33 score levels", "hi", score(MAX_SCORE_LEVELS + 1)),
        ("600 options total", "hi",
         dict(("q%d" % i, choice(20, "q%d" % i)["q%d" % i]) for i in range(30))),
        ("exactly 100 choice options", "hi", choice(MAX_CHOICE_OPTIONS)),
        ("exactly 32 score levels", "hi", score(MAX_SCORE_LEVELS)),
        ("noul criteria only, near the total", "hi",
         dict([("c%d" % i, choice(MAX_CHOICE_OPTIONS, "c%d" % i)["c%d" % i]) for i in range(5)]
               + [("last", choice(12, "last")["last"])]
               + [("n%d" % i, noul_crit) for i in range(30)])),
        ("noul only", "hi", {"n%d" % i: noul_crit for i in range(10)}),
        ("an ordinary request", {"body": "billed twice"}, one),
    ]
    for label, st, qs in parity_cases:
        s, d = serve_verdict(st, qs), demo_verdict(st, qs)
        ok("parity with laya.serve: %s" % label, s == d, "serve=%s demo=%s" % (s, d))

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
