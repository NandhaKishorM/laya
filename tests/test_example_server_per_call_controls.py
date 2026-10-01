"""Regression: examples/server.py must forward the same per-call controls laya.serve does.

`laya/serve.py::BODY_CONTROLS` is the set of caller-side knobs the shipped `/v1/systemone`
validates and forwards to the Router -- `model`, `max_len`, `head_max_len`, `task`, `lang`,
`lang_guess`, `min_confidence` -- and it refuses `BODY_REFUSALS` (the hook surface that cannot
cross an HTTP body) rather than silently dropping it. `laya/cli.py`, `laya/mcp/tools.py` and the
`laya.integrations.*` wrappers each forward that same set.

The demo server did not. Its `PredictRequest` took only `model`/`task`/`lang`, and `BatchRequest`
added the batch shape but nothing else, so a caller who asked for a per-call token budget, a soft
language hint, or an abstention threshold got the *same* answer as one who asked for nothing. On a
deployment that runs `Router(lang_guess=...)` or `Router(min_confidence=...)`, the demo could not
override it per call; a client who asked for `max_len` past `LAYA_MAX_TOKEN_BUDGET` was served
rather than refused; and a body that named `hooks` -- the surface this file has no way to receive
-- was silently ignored instead of answered with the same 422 the shipped server gives.

Scope: what the *handlers* forward on `Router.predict` and `Router.predict_batch`, and what the
*validators* refuse before the Router is touched. The two are pinned together: a body that names
`max_len` and is refused (past the cap) never reaches the recording Router, while one that names
`max_len=256` does, and the call the handler made carries the value -- not `None`, since core
reads an absent argument as "inherit what the deployment built with" and a null would override a
Router's own `lang_guess`/`min_confidence` with the demo's default.

`min_confidence` is the one asymmetry, and it has a reason: `Router.predict_batch` takes it as a
call argument, not a per-request key (`laya/router.py:961`), so on `/predict/batch` it rides on the
call shape rather than the request dicts. The per-state fallback in that endpoint goes through
`Router.predict`, which does take it as an argument, so on the retry the same value rejoins the
controls -- the fallback is one call per state and each carries the ask.

Driven over HTTP through TestClient, on a recording stand-in at `demo.ROUTER`. No weights load.

Run: python tests/test_example_server_per_call_controls.py
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LAYA_PRELOAD", "0")
os.environ.pop("LAYA_MAX_TOKEN_BUDGET", None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(
        "%s%s" % (name, ("  -- " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name,
                          ("  " + detail) if detail and not cond else ""), flush=True)


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
        if (getattr(exc, "name", None) or "").split(".")[0] not in (
                "fastapi", "httpx", "starlette", "multipart"):
            raise
        print("SKIP: examples/server.py needs the serve extra -- pip install laya[serve]")
        return 0

    from laya.serve import BODY_CONTROLS, BODY_REFUSALS

    class RecordingRouter:
        """Answers like the Router does, and remembers how it was asked."""

        def __init__(self):
            self.predict_calls = []
            self.batch_calls = []

        def predict(self, state, questions, **kw):
            self.predict_calls.append((state, dict(kw)))
            return {"answers": {"a": {"type": "noul", "choice": False}},
                    "model": "english", "state": state, "questions": questions}

        def predict_batch(self, requests, **kw):
            self.batch_calls.append((list(requests), dict(kw)))
            return [{"answers": {"a": {"type": "noul", "choice": False}},
                     "model": "english", "state": r["state"], "questions": r["questions"]}
                    for r in requests]

    def new_recorder():
        r = RecordingRouter()
        demo.ROUTER = r
        return r

    client = TestClient(demo.app, raise_server_exceptions=False)
    one = {"a": {"type": "noul", "instructions": "x"}}

    # --- what the two surfaces forward is the same set, drawn from laya.serve ----------
    # PredictRequest's public fields must cover BODY_CONTROLS exactly: this file's own claim
    # to be "the demo of the shipped server" fails if it stops accepting a control core takes.
    per_call_set = set(BODY_CONTROLS)
    predict_fields = set(demo.PredictRequest.model_fields) - {"state", "questions"}
    batch_fields = set(demo.BatchRequest.model_fields) - {"states", "questions",
                                                          "batch_size", "sort_by_length"}
    ok("PredictRequest accepts every key in laya.serve.BODY_CONTROLS",
       per_call_set <= predict_fields,
       "BODY_CONTROLS=%r, PredictRequest fields=%r" % (sorted(per_call_set),
                                                        sorted(predict_fields)))
    ok("BatchRequest accepts the same set, not a subset that drifts from the single path",
       per_call_set <= batch_fields,
       "missing from BatchRequest: %r" % sorted(per_call_set - batch_fields))
    ok("and the two models expose the same per-call surface, so a client sees no drift",
       predict_fields == batch_fields,
       "predict=%r vs batch=%r" % (sorted(predict_fields), sorted(batch_fields)))

    # --- /predict forwards each control when the caller set it -----------------------
    r = new_recorder()
    client.post("/predict", json={"state": "hello", "questions": one,
                                   "max_len": 256})
    ok("max_len reaches Router.predict when set",
       r.predict_calls and r.predict_calls[0][1].get("max_len") == 256,
       str(r.predict_calls[0][1] if r.predict_calls else "no call"))

    r = new_recorder()
    client.post("/predict", json={"state": "hello", "questions": one,
                                   "head_max_len": 64})
    ok("head_max_len reaches Router.predict when set",
       r.predict_calls and r.predict_calls[0][1].get("head_max_len") == 64)

    r = new_recorder()
    client.post("/predict", json={"state": "hello", "questions": one,
                                   "lang_guess": "de"})
    ok("lang_guess reaches Router.predict when set",
       r.predict_calls and r.predict_calls[0][1].get("lang_guess") == "de")

    r = new_recorder()
    client.post("/predict", json={"state": "hello", "questions": one,
                                   "min_confidence": 0.85})
    ok("min_confidence reaches Router.predict when set",
       r.predict_calls and r.predict_calls[0][1].get("min_confidence") == 0.85)

    r = new_recorder()
    client.post("/predict", json={"state": "hello", "questions": one,
                                   "model": "english", "task": "routing", "lang": "fr",
                                   "lang_guess": "de", "max_len": 512, "head_max_len": 32,
                                   "min_confidence": 0.5})
    ok("and every control on a single body arrives together, none overwriting another",
       r.predict_calls
       and r.predict_calls[0][1] == {"model": "english", "task": "routing", "lang": "fr",
                                     "lang_guess": "de", "max_len": 512, "head_max_len": 32,
                                     "min_confidence": 0.5},
       str(r.predict_calls[0][1] if r.predict_calls else "no call"))

    # --- unset controls are absent, never sent as None ------------------------------
    # `Router.predict` reads an absent argument as "inherit what the deployment built with",
    # so passing None here would override a `Router(lang_guess=...)` or a
    # `Router(min_confidence=...)` with the demo's own default.
    r = new_recorder()
    client.post("/predict", json={"state": "hello", "questions": one})
    kw = r.predict_calls[0][1] if r.predict_calls else None
    ok("an unset control is absent from the call, not sent as None",
       kw is not None and not any(kw.get(k, "__sentinel__") is None
                                   for k in ("max_len", "head_max_len", "lang_guess",
                                             "min_confidence", "task", "lang", "model")),
       str(kw))

    # `min_confidence=0.0` is falsy but a real ask -- it means "gate at zero", which is not
    # the same as "no gate". Any `if value:` guard would drop it; only `if value is not None`
    # is correct here.
    r = new_recorder()
    client.post("/predict", json={"state": "hello", "questions": one,
                                   "min_confidence": 0.0})
    ok("min_confidence=0.0 is still forwarded -- falsy is not absent",
       r.predict_calls and "min_confidence" in r.predict_calls[0][1]
       and r.predict_calls[0][1]["min_confidence"] == 0.0,
       str(r.predict_calls[0][1] if r.predict_calls else "no call"))

    # --- the shipped server's refusals are also refused here -----------------------
    # A body that names `hooks` cannot cross this surface: there is no receiver on the
    # demo, so a caller who asks for one today gets an answer that ignores their ask. The
    # shipped server 422s; the demo must too, on both endpoints.
    for key in BODY_REFUSALS:
        r = new_recorder()
        resp = client.post("/predict", json={"state": "hello", "questions": one, key: []})
        ok("a body that names %r is 422, not silently ignored" % key,
           resp.status_code == 422 and not r.predict_calls,
           "%s, router calls=%d" % (resp.status_code, len(r.predict_calls)))

    r = new_recorder()
    resp = client.post("/predict/batch", json={"states": ["a", "b"], "questions": one,
                                                "hooks_timeout": 0.5})
    ok("the refusal covers /predict/batch too",
       resp.status_code == 422 and not r.batch_calls,
       "%s, batch calls=%d" % (resp.status_code, len(r.batch_calls)))

    # --- max_len is capped by LAYA_MAX_TOKEN_BUDGET, refused before inference ------
    cap = demo._resolve_max_token_budget()
    r = new_recorder()
    resp = client.post("/predict", json={"state": "hello", "questions": one,
                                          "max_len": cap + 1})
    ok("a max_len past the server cap is a 422, and the Router is never asked",
       resp.status_code == 422 and not r.predict_calls,
       "%s, calls=%d" % (resp.status_code, len(r.predict_calls)))

    r = new_recorder()
    resp = client.post("/predict", json={"state": "hello", "questions": one,
                                          "head_max_len": cap + 1})
    ok("the same cap covers head_max_len",
       resp.status_code == 422 and not r.predict_calls)

    # The cap is read at call time, so LAYA_MAX_TOKEN_BUDGET actually moves it: the demo
    # does not carry a copy of the number.
    os.environ["LAYA_MAX_TOKEN_BUDGET"] = "2048"
    try:
        r = new_recorder()
        resp = client.post("/predict", json={"state": "hello", "questions": one,
                                              "max_len": 4096})
        ok("LAYA_MAX_TOKEN_BUDGET is what caps the request, not a number typed in this file",
           resp.status_code == 422 and not r.predict_calls,
           "%s" % resp.status_code)
        r = new_recorder()
        ok("and the same value under the raised cap is accepted and forwarded",
           client.post("/predict", json={"state": "hello", "questions": one,
                                          "max_len": 1024}).status_code == 200
           and r.predict_calls[0][1]["max_len"] == 1024)
    finally:
        os.environ.pop("LAYA_MAX_TOKEN_BUDGET", None)

    # --- type refusals, mirroring laya.serve's validators -------------------------
    # A JSON `true` where the Router expects a language code would become `"true"` through
    # core's `_english_from_code` -- a real, non-English code that decides the checkpoint.
    # `Optional[str]` in the field type also refuses a bool, so a plain status check cannot
    # tell the two paths apart; the assertion below requires laya.serve's own wording in
    # the 422 body, which only fires when this file actually calls `_validate_language_param`.
    for key in ("lang", "lang_guess"):
        r = new_recorder()
        resp = client.post("/predict", json={"state": "hello", "questions": one, key: True})
        detail = str(resp.json().get("detail", ""))
        ok("%s: a JSON bool is refused before the Router is asked" % key,
           resp.status_code == 422 and not r.predict_calls,
           "%s, calls=%d" % (resp.status_code, len(r.predict_calls)))
        ok("%s: the 422 detail is laya.serve's own wording, not pydantic's field error" % key,
           "language code string" in detail, detail[:180])

    r = new_recorder()
    resp = client.post("/predict", json={"state": "hello", "questions": one,
                                          "min_confidence": True})
    ok("min_confidence: a JSON bool is refused -- it is not a threshold",
       resp.status_code == 422 and not r.predict_calls,
       "%s, calls=%d" % (resp.status_code, len(r.predict_calls)))

    r = new_recorder()
    resp = client.post("/predict", json={"state": "hello", "questions": one,
                                          "min_confidence": 1.5})
    ok("min_confidence outside core's [0, 1] is refused before an inference slot",
       resp.status_code == 422 and not r.predict_calls)

    r = new_recorder()
    resp = client.post("/predict", json={"state": "hello", "questions": one,
                                          "max_len": "abc"})
    ok("a non-integer max_len is refused, not coerced",
       resp.status_code == 422 and not r.predict_calls)

    r = new_recorder()
    resp = client.post("/predict", json={"state": "hello", "questions": one,
                                          "max_len": -1})
    ok("a non-positive max_len is refused, matching laya.serve._validate_budget_param",
       resp.status_code == 422 and not r.predict_calls)

    # --- /predict/batch forwards the same set -------------------------------------
    # On the batch path the per-request controls ride in each request dict (so `route_batch`
    # and `predict_batch` read them per state), while `min_confidence` is a call-level
    # argument to `Router.predict_batch` -- see laya/router.py:961.
    r = new_recorder()
    states = ["s1", "s2", "s3"]
    client.post("/predict/batch", json={"states": states, "questions": one,
                                         "lang": "fr", "lang_guess": "de",
                                         "max_len": 512, "head_max_len": 64,
                                         "min_confidence": 0.75,
                                         "batch_size": 2, "sort_by_length": True})
    reqs, shape = r.batch_calls[0] if r.batch_calls else ([], {})
    ok("each request on the batch carries the caller's max_len/head_max_len/lang/lang_guess",
       all(rr.get("max_len") == 512 and rr.get("head_max_len") == 64
           and rr.get("lang") == "fr" and rr.get("lang_guess") == "de" for rr in reqs),
       str(reqs[:1]))
    ok("min_confidence rides on the batch call as a shape argument, not per request",
       shape.get("min_confidence") == 0.75
       and not any("min_confidence" in rr for rr in reqs),
       "shape=%r" % shape)
    ok("batch_size and sort_by_length still reach the Router alongside it",
       shape.get("batch_size") == 2 and shape.get("sort_by_length") is True)

    # Unset controls on the batch path stay absent too, for the same "inherit" reason.
    r = new_recorder()
    client.post("/predict/batch", json={"states": states, "questions": one})
    reqs, shape = r.batch_calls[0] if r.batch_calls else ([], {})
    ok("a plain batch sends no null controls and no shape",
       reqs and not any(k in rr for rr in reqs for k in
                        ("max_len", "head_max_len", "lang", "lang_guess", "min_confidence",
                         "task", "model"))
       and not shape,
       "reqs=%r shape=%r" % (reqs[:1], shape))

    # --- the fallback keeps the ask on every per-state retry ----------------------
    # When Router.predict_batch fails, the endpoint retries each state through Router.predict,
    # which *does* take min_confidence as a call argument. If the retry dropped it, one bad
    # state would silently cost the neighbours their abstention gate.
    class HalfRouter(RecordingRouter):
        def predict_batch(self, requests, **kw):
            self.batch_calls.append((list(requests), dict(kw)))
            raise ValueError("batch forward failed")

    r = HalfRouter()
    demo.ROUTER = r
    body = client.post("/predict/batch", json={"states": states, "questions": one,
                                                "max_len": 256, "min_confidence": 0.75}).json()
    ok("the fallback retried every state",
       len(body.get("results", [])) == 3 and len(r.predict_calls) == 3)
    ok("and each retry carries the caller's max_len",
       all(c[1].get("max_len") == 256 for c in r.predict_calls),
       str([c[1] for c in r.predict_calls][:1]))
    ok("and each retry carries the abstention threshold, this time as a call argument",
       all(c[1].get("min_confidence") == 0.75 for c in r.predict_calls),
       str([c[1] for c in r.predict_calls][:1]))
    # batch_size and sort_by_length belong to the forward-pass shape only; the fallback is one
    # state per call, so leaking them into `Router.predict(...)` would TypeError every retry.
    ok("and the fallback does not leak batch shape into Router.predict",
       not any("batch_size" in c[1] or "sort_by_length" in c[1] for c in r.predict_calls),
       str([c[1] for c in r.predict_calls][:1]))

    # --- parity with laya.serve's refusal wording ---------------------------------
    # The detail must be the shipped server's, not a paraphrase: a client that reads the
    # 422 body gets the same "install them where laya-serve runs" guidance from both.
    resp = client.post("/predict", json={"state": "hello", "questions": one, "hooks": []})
    ok("the 422 detail is laya.serve's own wording, not a copy",
       resp.status_code == 422
       and "run inside the server process" in str(resp.json().get("detail", "")),
       str(resp.json())[:220])

    # --- the two endpoints refuse identically -------------------------------------
    # A client who POSTs `{"hooks_timeout": 0.5}` to the batch endpoint must not get a
    # different answer than one who POSTs the same field to the single endpoint.
    for key in BODY_REFUSALS:
        single = client.post("/predict", json={"state": "hi", "questions": one, key: 1})
        batch = client.post("/predict/batch", json={"states": ["a"], "questions": one, key: 1})
        ok("%s refused the same way on both endpoints" % key,
           single.status_code == batch.status_code == 422,
           "single=%s batch=%s" % (single.status_code, batch.status_code))

    demo.ROUTER = None
    print()
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
