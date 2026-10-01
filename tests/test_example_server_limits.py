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
a request that passes validation answers 503, which is the assertion for "accepted". The
one arm that needs a Router enters the lifespan, which is also the arm that checks the cap.

On `main` this file is 19 checks, all of them about request size. It is now 38: the other
nineteen follow the cap from the environment to the constructor, to the running Router, to
`/health` in both JSON and HTML, and back out through the `--reload` push. The first
nineteen are untouched.

FAILS_ON_MAIN -- this file, against `main`'s `examples/server.py` at 4066d5d:

    PASS the page and health endpoints are unaffected
    FAIL an unset LAYA_MAX_LOADED means 'not asked for', not a number of this file's own  1
    AttributeError: module 'server' has no attribute '_router_kwargs'

One check names the number, then the run aborts because the helper it drives does not exist
there. So the same claims are checked on `main` through nothing but the public surface -- its
own lifespan, the Router that builds, `/health` in both representations -- which prints:

    Router()'s own default = 2      built Router's cap = 1      /health JSON = 1
    /health HTML says      = 'Checkpoints are loaded on demand; up to 1 kept in memory.'
    preload() lifts the cap to -> 3, while the page prints 1

and `False` for "the cap came from the environment", "the Router holds laya's default", and
"that default is more than one"; on this branch all three read `True` and the cap reads 2.
That last line is the half that is not about the default at all: `Router.preload()` raises the
cap to fit what it preloads (`laya/router.py:275` -> `:372`), so the demo's own default mode ran
with three checkpoints resident while `/health` reported one. Nothing in the file ever asked the
running Router what it was holding.

`main` is at 66 checks now, and this file is at 81. The fifteen new ones are the request `model`
field. `laya.router.normalise_name` is core's one resolver -- trim, lower-case, alias table,
registry -- and `laya --model` and `laya/serve.py` both hand it the caller's string, but this demo
compared it against a tuple of the three checkpoint names. So on `main` the aliases, the checkpoint
ids with the `laya-` prefix, `typed_decisions`, and the upper-cased and space-padded form of every
one of them answer 422 here: 60 of the 67 spellings core resolves, which
`laya-bench/model_alias_witness_main.log` counts against `main` at 9d95567 through
`_check_model` and through `POST /predict` on a TestClient -- `{"model": "english"}` answers 503,
`{"model": "laya"}` answers 422, while `laya --model laya` pins the checkpoint. The new checks take
the probe from `laya.router`'s own tables rather than listing the aliases here, require the
canonical name to reach the Router, keep blank meaning "do not pin" as the playground sends it,
read the alias list back out of `/models`, its HTML page, the published request schema and the
README section for this server, and then add a fourth checkpoint to core at runtime, reload the
demo, and require it to be pinnable by name and by alias and to appear in that text with no edit to
this file.

Run: python tests/test_example_server_limits.py
"""
import importlib
import json
import os
import sys
import types

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
        # examples/server.py reads the environment at import, so the cap under test has to
        # be absent/present *before* the import, never patched after it.
        os.environ.pop("LAYA_MAX_LOADED", None)
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
        # A state whose `str()` and whose JSON serialization are different lengths (50000 vs 99988).
        # This case catches the two surfaces DIVERGING -- one measuring the serialized text and the
        # other `str(state)`. It cannot catch them being wrong together: `examples/server.py` binds
        # `_state_length` from `laya.serve` by `getattr`, so a regression inside that helper moves
        # both and parity stays green. Measured: regressing `_state_length` fails 0 of these parity
        # checks and both of the absolute ones below. Those are the real coverage -- do not prune
        # them as redundant.
        ("a state whose repr is half its JSON", {"body": '"' * 49988}, one),
    ]
    for label, st, qs in parity_cases:
        s, d = serve_verdict(st, qs), demo_verdict(st, qs)
        ok("parity with laya.serve: %s" % label, s == d, "serve=%s demo=%s" % (s, d))

    # Parity alone cannot see a bug both surfaces share, and `getattr` guarantees they share one:
    # with `_state_length` measuring `str(state)` again, both agree on accepting a state that
    # serializes to 99 988 characters and every parity check above stays green (measured: 0 of them
    # fail, both of these do). So the verdict itself is asserted, not just the agreement.
    quote_heavy = {"body": '"' * 49988}
    # Recorded through `ok` like everything else in this file: a bare `assert` here would raise out
    # of `main()` and abandon the ~30 checks that follow instead of recording one failure.
    ok("the quote-heavy fixture passes a str()-based gate",
       len(str(quote_heavy)) <= MAX_STATE_CHARS, len(str(quote_heavy)))
    ok("the quote-heavy fixture fails a serialization-based gate",
       len(json.dumps(quote_heavy, ensure_ascii=False)) > MAX_STATE_CHARS,
       len(json.dumps(quote_heavy, ensure_ascii=False)))
    for who, verdict in (("laya.serve", serve_verdict(quote_heavy, one)),
                         ("the demo server", demo_verdict(quote_heavy, one))):
        ok("%s refuses a state whose JSON is twice its repr" % who, verdict == "refused", verdict)

    # --- the resident-checkpoint cap: derived, not copied -------------------
    # examples/server.py used to build its Router with `max_loaded=1`, a copy of a default
    # laya/router.py retired in #180. A cap of one cannot hold both english and
    # multilingual, so the demo ran the churn #172 measured and fixed -- one checkpoint
    # rebuilt per alternating-language request -- while the library and laya.serve did not.
    # What is asserted here is that the demo asks for nothing unless asked to, and that
    # what it then reports is the number the running Router holds.
    from laya.router import Router

    ok("an unset LAYA_MAX_LOADED means 'not asked for', not a number of this file's own",
       demo._CFG["max_loaded"] is None, repr(demo._CFG["max_loaded"]))
    ok("so the key never reaches the constructor",
       "max_loaded" not in demo._router_kwargs(demo._CFG),
       repr(demo._router_kwargs(demo._CFG)))
    # Before a Router exists the page has no resident count to state. This is the check
    # that fails if the page goes back to guessing one (`cfg.get('max_loaded', 1)`).
    ok("and while loading, the page states no resident count",
       "kept in memory" not in client.get("/health", headers={"accept": "text/html"}).text,
       client.get("/health", headers={"accept": "text/html"}).text[:200])

    with client:                       # the lifespan builds the Router; preload is off
        # The witness that this arm needed no weights: LAYA_PRELOAD=0 above means the
        # Router the app built holds no checkpoint yet, so the cap is the only thing here
        # that could have been loaded.
        ok("building it downloaded nothing",
           not demo.ROUTER._agents, repr(sorted(demo.ROUTER._agents)))
        cap = demo.ROUTER.max_loaded
        ok("the running Router holds laya's own default",
           cap == Router().max_loaded, "demo %r vs laya %r" % (cap, Router().max_loaded))
        ok("and laya's own default is more than one checkpoint",
           cap > 1, repr(cap))
        ok("/health reports the cap the Router really holds",
           client.get("/health").json()["config"]["max_loaded"] == cap, repr(cap))
        ok("and the HTML page prints that same number",
           ("up to %d kept in memory" % cap)
           in client.get("/health", headers={"accept": "text/html"}).text, repr(cap))
        # The number cannot come from the requested config, because the config and the
        # Router are allowed to disagree: `Router(preload=True)` raises its own cap to fit
        # what it preloads (laya/router.py:275 -> :372), so the demo's default mode has run
        # with three resident while its env said one. Move the Router and the page must move.
        demo.ROUTER.max_loaded = cap + 1
        moved = client.get("/health")
        ok("and both follow the Router when the Router raises its own cap",
           moved.json()["config"]["max_loaded"] == cap + 1
           and ("up to %d kept in memory" % (cap + 1))
           in client.get("/health", headers={"accept": "text/html"}).text,
           repr(moved.json()["config"]))

    def with_cap(value):
        """Re-import the demo the way uvicorn starts it, with LAYA_MAX_LOADED set to `value`."""
        if value is None:
            os.environ.pop("LAYA_MAX_LOADED", None)
        else:
            os.environ["LAYA_MAX_LOADED"] = value
        return importlib.reload(demo)

    # The default moving must not take the operator's own number away with it.
    for raw, want in (("3", 3), ("1", 1), (" 4 ", 4)):
        fresh = with_cap(raw)
        with TestClient(fresh.app) as c:
            ok("LAYA_MAX_LOADED=%r still reaches the Router" % raw,
               fresh.ROUTER.max_loaded == want, repr(fresh.ROUTER.max_loaded))
            ok("LAYA_MAX_LOADED=%r still shows up in /health" % raw,
               c.get("/health").json()["config"]["max_loaded"] == want, raw)
    fresh = with_cap(None)
    from laya.serve import build_router

    ok("the demo and laya.serve leave the cap to the same place",
       fresh._router_kwargs(fresh._CFG).get("max_loaded", Router().max_loaded)
       == build_router().max_loaded,
       "%r vs %r" % (fresh._router_kwargs(fresh._CFG), build_router().max_loaded))

    # --- the --reload env push has to survive the round trip ----------------
    # With reload=True uvicorn re-imports `server:app` in a child process and only the
    # environment crosses over. Writing str(None) for "not asked for" would land on the
    # int() that reads LAYA_MAX_LOADED in that child and stop the server at import, so the
    # unset case must push nothing at all. uvicorn is replaced so main() never binds a port.
    real_uvicorn = sys.modules.get("uvicorn")
    started = {}
    fake = types.ModuleType("uvicorn")
    fake.run = lambda target, **kw: started.update(target=target)
    sys.modules["uvicorn"] = fake
    argv = sys.argv

    def start(args):
        with_cap(None)
        sys.argv = ["server.py"] + args
        demo.main()
        return os.environ.get("LAYA_MAX_LOADED", "(absent)")

    try:
        ok("--reload with no flag pushes no cap, so the reimport can read it",
           start(["--reload"]) == "(absent)", repr(started))
        ok("--reload --max-loaded 3 pushes 3",
           start(["--reload", "--max-loaded", "3"]) == "3", repr(started))
        ok("without --reload nothing is pushed at all",
           start([]) == "(absent)", repr(started))
    finally:
        sys.argv = argv
        if real_uvicorn is not None:
            sys.modules["uvicorn"] = real_uvicorn
        else:
            sys.modules.pop("uvicorn", None)
    with_cap(None)                     # leave the module as the rest of the file found it

    # --- /predict/batch must make ONE Router call, not one per state ---------
    #
    # README teaches `Router.predict_batch` as the way to answer many states with shared forward
    # passes ("routes the full workload first, groups requests by checkpoint ... results are
    # restored to the original request order"), and this endpoint is the batch-shaped surface the
    # README points at for trying Laya without writing code. The handler used to call
    # `Router.predict` once per state anyway, so 64 states were 64 forwards. These drive the real
    # app -- validation, `_questions()`, the handler body, the JSON envelope -- over a recording
    # stand-in at `demo.ROUTER`, so no weights are needed to count the calls.

    import inspect

    from laya.router import Router as CoreRouter

    class RecordingRouter:
        """Answers like the Router does, and remembers how it was asked."""

        def __init__(self, fail_on=None):
            self.predict_calls = []
            self.batch_calls = []
            self.fail_on = fail_on

        def predict(self, state, questions, **kw):
            self.predict_calls.append((state, dict(kw)))
            return self._answer(state, questions)

        def predict_batch(self, requests, **kw):
            self.batch_calls.append((list(requests), dict(kw)))
            return [self._answer(r["state"], r["questions"]) for r in requests]

        def _answer(self, state, questions):
            if self.fail_on and self.fail_on in str(state):
                raise ValueError("simulated failure for " + self.fail_on)
            return {"answers": {"a": {"type": "noul", "choice": False}}, "state": state,
                    "questions": questions}

    states = ["ticket %d" % i for i in range(8)]

    class LegacyRouter(RecordingRouter):
        """A Router-like object whose `predict_batch` predates the batch path entirely."""

        predict_batch = None

    # A Router that predates `predict_batch` must still be usable through the fallback.
    older = LegacyRouter()
    demo.ROUTER = older
    legacy = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one})
    ok("a router without predict_batch still answers every state",
       legacy.status_code == 200 and len(legacy.json()["results"]) == 8
       and not [r for r in legacy.json()["results"] if "error" in r],
       "%s / %s" % (legacy.status_code, json.dumps(legacy.json())[:200]))

    router = RecordingRouter()
    demo.ROUTER = router
    batched = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one})
    body = batched.json()
    ok("an 8-state batch is ONE predict_batch call and zero predict calls",
       len(router.batch_calls) == 1 and not router.predict_calls,
       "predict_batch=%d predict=%d" % (len(router.batch_calls), len(router.predict_calls)))
    ok("the batch envelope is unchanged: count and one result per state, in order",
       body.get("count") == 8 and [r.get("state") for r in body.get("results", [])] == states)
    # Read defensively: an endpoint that never batches has nothing to inspect, and the checks below
    # say so by name instead of letting this script die at the unpack.
    requests_sent, call_kwargs = (router.batch_calls[0] if router.batch_calls else ([], {}))
    ok("each request carries state + questions, and the questions map is the same one",
       len(requests_sent) == 8
       and all(sorted(r) == ["questions", "state"] for r in requests_sent)
       and all(r["questions"] == requests_sent[0]["questions"] for r in requests_sent))
    controls_sent = [k for r in requests_sent for k in r if k in ("model", "task", "lang")]
    ok("unset controls are absent, not sent as null",
       len(requests_sent) == 8 and not call_kwargs and not controls_sent,
       "call kwargs=%r controls=%r" % (call_kwargs, controls_sent))
    # Every key the endpoint puts in a request must be one core actually reads out of it, and the
    # set of those keys is derived from `Router.route_batch`'s own source, so a rename or a new
    # override in core shows up here rather than silently stopping reaching the router.
    import re

    route_src = inspect.getsource(CoreRouter.route_batch)
    read = ({"state", "questions"}
            | set(re.findall(r'request\["(\w+)"\]', route_src))
            | set(re.findall(r'request\.get\("(\w+)"', route_src)))
    sent_keys = {k for r in requests_sent for k in r}
    ok("the request keys the endpoint sends are ones core reads",
       len(requests_sent) == 8 and read >= sent_keys,
       "core reads %r, endpoint sends %r" % (sorted(read), sorted(sent_keys)))
    ok("predict_batch is still a one-positional-list call",
       list(inspect.signature(CoreRouter.predict_batch).parameters)[1] == "requests")

    pinned = RecordingRouter()
    demo.ROUTER = pinned
    TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one,
                                "model": "multilingual", "lang": "de"})
    sent = pinned.batch_calls[0][0] if pinned.batch_calls else []
    ok("a pinned model/lang travels with every request",
       len(sent) == 8
       and all(r["model"] == "multilingual" and r["lang"] == "de" for r in sent)
       and not any("task" in r for r in sent),
       json.dumps(sent[:1])[:200])

    # --- how the batch is packed is askable ---------------------------------------------
    #
    # `Router.predict_batch` has taken `batch_size` and `sort_by_length` since #294, and the README
    # teaches `router.predict_batch(requests, batch_size=8, sort_by_length=True)`. The endpoint that
    # answers up to 64 states in one call forwarded neither, so every batch was one forward pass
    # padded to its longest state: a caller who wanted smaller passes had no way to ask, in code or
    # over HTTP. These drive the real request model, so a body that named the keys and had them
    # dropped by validation fails here rather than returning a silently slower 200.
    shape_params = set(inspect.signature(CoreRouter.predict_batch).parameters) - {"self", "requests"}
    ok("core still takes the shape keys this endpoint forwards",
       {"batch_size", "sort_by_length"} <= shape_params, str(sorted(shape_params)))

    shaped = RecordingRouter()
    demo.ROUTER = shaped
    body = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one,
                                "batch_size": 4, "sort_by_length": True})
    ok("a shape request is still one batch call, in order",
       len(shaped.batch_calls) == 1 and not shaped.predict_calls
       and body.json().get("count") == 8
       and [r.get("state") for r in body.json()["results"]] == states,
       "predict_batch=%d predict=%d" % (len(shaped.batch_calls), len(shaped.predict_calls)))
    ok("batch_size and sort_by_length reach predict_batch",
       shaped.batch_calls[0][1] == {"batch_size": 4, "sort_by_length": True},
       str(shaped.batch_calls[0][1]))

    sized = RecordingRouter()
    demo.ROUTER = sized
    TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one, "batch_size": 2})
    ok("batch_size alone does not invent a sort request",
       sized.batch_calls[0][1] == {"batch_size": 2}, str(sized.batch_calls[0][1]))

    # Core documents `sort_by_length` without a split batch as a no-op, not an error; the endpoint
    # must not be the surface that turns a valid call into a 422.
    unsized = RecordingRouter()
    demo.ROUTER = unsized
    no_split = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one, "sort_by_length": True})
    ok("sort_by_length alone is accepted and forwarded, as core takes it",
       no_split.status_code == 200 and unsized.batch_calls[0][1] == {"sort_by_length": True},
       "%s %s" % (no_split.status_code, str(unsized.batch_calls[0][1])))

    junk = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one, "batch_size": 0})
    ok("a zero-size forward pass is a 422 before any router call",
       junk.status_code == 422, "%s %s" % (junk.status_code, str(junk.json())[:120]))

    # The shape belongs to the batch call only. When a state fails, the endpoint retries per state
    # through `Router.predict`, which takes neither key: leaking them there would turn one bad
    # state into 8 TypeErrors and an empty batch.
    leaked = RecordingRouter(fail_on="ticket 3")
    demo.ROUTER = leaked
    retried = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one,
                                "batch_size": 4, "sort_by_length": True})
    ok("the per-state fallback carries no batch shape",
       leaked.batch_calls[0][1] == {"batch_size": 4, "sort_by_length": True}
       and len(leaked.predict_calls) == 8
       and not any("batch_size" in kw or "sort_by_length" in kw
                   for _, kw in leaked.predict_calls)
       and [r for r in retried.json()["results"] if "error" in r][0]["index"] == 3,
       str([kw for _, kw in leaked.predict_calls][:1])[:200])

    legacy_shaped = LegacyRouter()
    demo.ROUTER = legacy_shaped
    old = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one,
                                "batch_size": 4, "sort_by_length": True})
    ok("a router without predict_batch still answers a shaped request",
       old.status_code == 200 and len(old.json()["results"]) == 8
       and not [r for r in old.json()["results"] if "error" in r],
       "%s %s" % (old.status_code, json.dumps(old.json())[:200]))

    # One state failing must not cost its neighbours their answer: the endpoint's published
    # contract is per-item errors inside a 200.
    partial = RecordingRouter(fail_on="ticket 3")
    demo.ROUTER = partial
    poison = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one}).json()
    errors = [r for r in poison["results"] if "error" in r]
    ok("one failing state yields one error entry, not an empty batch",
       len(poison["results"]) == 8 and [r["index"] for r in errors] == [3]
       # #625: the item names the failure without its exception text, which stays in the log
       and errors[0]["error"] == "prediction failed",
       json.dumps(poison)[:240])
    ok("the failing batch retried per state, so its neighbours still answered",
       len(partial.batch_calls) == 1 and len(partial.predict_calls) == 8)

    # The single-state surface must keep going through predict(), and `/predict/batch` with one
    # state must still batch -- otherwise the two endpoints diverge on where hooks fire.
    single = RecordingRouter()
    demo.ROUTER = single
    one_state = TestClient(demo.app, raise_server_exceptions=False)
    one_state.post("/predict", json={"state": "ticket 0", "questions": one})
    ok("/predict still calls Router.predict once",
       len(single.predict_calls) == 1 and not single.batch_calls)

    demo.ROUTER = None
    not_ready = TestClient(demo.app, raise_server_exceptions=False).post(
        "/predict/batch", json={"states": states, "questions": one})
    ok("an unready router keeps answering 200 with per-item 503s",
       not_ready.status_code == 200
       and len(not_ready.json()["results"]) == 8
       and all("503" in r.get("error", "") for r in not_ready.json()["results"]),
       json.dumps(not_ready.json())[:200])
    demo.ROUTER = None

    # ---- the `model` field resolves the way core resolves one ----------------
    # `laya.router.normalise_name` trims, lower-cases and maps its alias table before it checks
    # the checkpoint registry, and `laya --model laya` and `POST /v1/systemone {"model":"laya"}`
    # both go through it. This demo compared the string against a tuple of the three names, so the
    # alias worked on the CLI and answered 422 here. The probe below is built from laya.router's
    # own tables, so it covers spellings this file has never heard of and cannot fall behind.
    from laya import router as _router_mod

    names = sorted(_router_mod.DEFAULT_MODELS)
    aliases = sorted(a for a, target in _router_mod._ALIASES.items()
                     if target in _router_mod.DEFAULT_MODELS)
    spellings = [spelling for value in names + aliases
                 for spelling in (value, value.upper(), " %s " % value)]

    refused, not_canonical = [], []
    for value in spellings:
        want = _router_mod.normalise_name(value)
        try:
            got = demo._check_model(value)
        except Exception as exc:
            refused.append("%r -> %s" % (value, exc))
            continue
        if got != want:
            not_canonical.append("%r -> %r, core says %r" % (value, got, want))
    ok("every one of the %d spellings core resolves, the demo resolves too" % len(spellings),
       not refused, "; ".join(refused[:3]))
    ok("and what reaches the Router is the checkpoint, not the spelling that named it",
       not not_canonical, "; ".join(not_canonical[:3]))

    # The negatives, derived from the same tables so one of them cannot accidentally be real.
    unknown = [v + "-nope" for v in names + aliases]

    def _refuses(fn, value):
        try:
            fn(value)
        except ValueError:
            return True
        return False

    ok("a name neither core nor the demo knows still refuses on both surfaces",
       all(_refuses(_router_mod.normalise_name, v) and _refuses(demo._check_model, v)
           for v in unknown),
       repr([v for v in unknown if not _refuses(demo._check_model, v)][:3]))

    # Blank is the demo's own convenience, not core's: the playground posts an empty field, and
    # core's resolver raises on one. Parity must not take that away.
    def _resolves(v):
        """The value the field validator accepts, or the refusal it gave -- never a raised error."""
        try:
            return demo._check_model(v)
        except Exception as exc:
            return "refused: %s" % exc

    ok("a blank or whitespace-only model still means 'do not pin'",
       [_resolves(v) for v in (None, "", "   ")] == [None, None, None],
       repr([(v, _resolves(v)) for v in (None, "", "   ")])[:180])

    regressed = demo.ROUTER
    demo.ROUTER = RecordingRouter()
    http = TestClient(demo.app, raise_server_exceptions=False)
    single = http.post("/predict", json={"state": "ticket 0", "questions": one, "model": "ml"})
    ok("an alias is accepted by /predict and the Router is asked for its checkpoint",
       single.status_code == 200 and demo.ROUTER.predict_calls
       and demo.ROUTER.predict_calls[-1][1].get("model") == "multilingual",
       "%s %r" % (single.status_code, demo.ROUTER.predict_calls[-1:]))
    batch = http.post("/predict/batch", json={"states": states, "questions": one, "model": "LAYA"})
    ok("the same alias on /predict/batch pins english for every state in the batch",
       batch.status_code == 200 and demo.ROUTER.batch_calls
       and [r.get("model") for r in demo.ROUTER.batch_calls[-1][0]] == ["english"] * len(states),
       "%s %r" % (batch.status_code, [r.get("model")
                                      for r in (demo.ROUTER.batch_calls[-1][0]
                                                if demo.ROUTER.batch_calls else [])][:3]))
    bad = http.post("/predict", json={"state": "ticket 0", "questions": one,
                                      "model": names[0] + "-nope"})
    bad_text = json.dumps(bad.json())
    ok("an unknown name is still a 422, in core's words plus this server's own hint",
       bad.status_code == 422 and "unknown model" in bad_text and "choose one of" in bad_text
       and "(or omit it)" in bad_text, bad_text[:220])

    listing = http.get("/models").json()
    ok("/models publishes the aliases a caller may use, and every one it publishes validates",
       listing["allowed"] == names and listing.get("aliases") == {a: _router_mod._ALIASES[a]
                                                                  for a in aliases}
       and all(_resolves(a) == c for a, c in (listing.get("aliases") or {}).items()),
       json.dumps(listing)[:220])
    page = http.get("/models", headers={"accept": "text/html"}).text
    ok("the page that lists the checkpoints names the count and the spellings it accepts",
       ("%d checkpoints, one router" % len(names)) in page
       and all("<code>%s</code>" % a in page for a in aliases), page[:160])
    demo.ROUTER = regressed

    def _rendered(desc):
        """The checkpoint list and the alias list, read back out of the field's own description.

        That sentence is generated from `MODELS`/`MODEL_ALIASES`, so the honest gate is to parse
        the lists out of it and compare them with what `laya.router` accepts -- a substring check
        would also pass a sentence somebody typed beside the tables. A render this cannot read
        comes back as two empty lists, which fails by name instead of raising and aborting the run.
        """
        try:
            names_part, alias_part = desc.split("one of ", 1)[1].split(
                ", or an alias core resolves (", 1)
            alias_part = alias_part.split("), in any casing", 1)[0]
        except (IndexError, ValueError):
            return [], []
        return ([p.strip() for p in names_part.split(" | ")] if names_part.strip() else [],
                [p.strip() for p in alias_part.split(", ")] if alias_part.strip() else [])

    schema = http.get("/openapi.json").json()["components"]["schemas"]
    description = schema["PredictRequest"]["properties"]["model"].get("description", "")
    ok("the published request schema lists every name and alias the endpoint takes, in neither "
       "more nor less", _rendered(description) == (names, aliases),
       "rendered %r %r from tables %r %r" % (_rendered(description)[0], _rendered(description)[1],
                                             names, aliases))
    ok("and both request models describe the one field identically",
       description and description
       == schema["BatchRequest"]["properties"]["model"].get("description", ""),
       schema["BatchRequest"]["properties"]["model"].get("description", ""))

    # The section of the README that teaches this server has to say the field exists and where its
    # spellings come from, or the only way to learn it is to get a 422.
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
        readme = handle.read()
    demo_page = readme.split("## Try it locally: web GUI + JSON API", 1)[1].split("\n---", 1)[0]
    ok("the README section for this server teaches `model` and where its names come from",
       "`model`" in demo_page and "laya.router" in demo_page and "/models" in demo_page,
       demo_page[-200:])

    # The registry is read, not copied. A checkpoint added to laya.router has to become pinnable
    # here -- by name, by alias, in the schema text and on /models -- with no edit to this file.
    # `laya.DEFAULT_MODELS` is the same dict object, so one insert reaches both reads.
    _router_mod.DEFAULT_MODELS["spanish"] = (_router_mod.BUNDLE_REPO, "spanish")
    _router_mod._ALIASES["es"] = "spanish"
    try:
        grown = importlib.reload(demo)
        grown_names = sorted(_router_mod.DEFAULT_MODELS)
        ok("a checkpoint core gains is pinnable here without editing this file",
           "spanish" in grown.MODELS and _resolves("spanish") == "spanish"
           and _resolves(" ES ") == "spanish"
           and grown.MODEL_ALIASES.get("es") == "spanish",
           "%r %r" % (sorted(grown.MODELS), _resolves(" es ")))
        grown_schema = TestClient(grown.app, raise_server_exceptions=False).get(
            "/openapi.json").json()["components"]["schemas"]
        grown_description = grown_schema["PredictRequest"]["properties"]["model"].get("description", "")
        ok("and the text that teaches the list is rendered from it, not typed beside it",
           _rendered(grown_description) == (grown_names, sorted(grown.MODEL_ALIASES))
           and grown_description
           == grown_schema["BatchRequest"]["properties"]["model"].get("description", ""),
           "rendered %r from %r -- %s" % (_rendered(grown_description), grown_names,
                                          grown_description[:120]))
        grown_listing = TestClient(grown.app, raise_server_exceptions=False).get("/models").json()
        ok("/models reports the new checkpoint and its new alias",
           grown_listing["allowed"] == grown_names
           and (grown_listing.get("aliases") or {}).get("es") == "spanish",
           json.dumps(grown_listing)[:220])
    finally:
        _router_mod.DEFAULT_MODELS.pop("spanish", None)
        _router_mod._ALIASES.pop("es", None)
        demo = importlib.reload(demo)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
