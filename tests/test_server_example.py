"""Smoke test for examples/server.py: does the demo API wiring still match laya's
public surface (Router, DEFAULT_MODELS, QTYPES)?

Loads real weights for the `english` checkpoint only (skips multilingual/typed-decisions
to keep this fast) and drives the FastAPI app in-process via TestClient -- no network
socket, no subprocess.

Run:  python3 tests/test_server_example.py
"""
import base64
import json
import os
import re
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
# server._CFG["preload"] defaults to true (LAYA_PRELOAD unset), which preloads all three
# checkpoints in the app's lifespan before the first request -- overriding it here is what
# keeps this an "english only" test: every payload below is English, so lazy loading only
# ever touches the one checkpoint this test actually needs.
os.environ.setdefault("LAYA_PRELOAD", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, (" -- " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""), flush=True)


def errs(r):
    """The error lines of a /gui error page, for a short failure message."""
    return " | ".join(re.findall(r"<li>(.*?)</li>", r.text)) or r.text[-300:]


def main():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("SKIP: fastapi/httpx not installed -- pip install laya[serve] httpx")
        return 0

    import server  # examples/server.py

    with TestClient(server.app) as client:
        r = client.get("/health")
        ok("GET /health -> 200", r.status_code == 200, str(r.status_code))
        ok("GET /health -> status ok", r.json().get("status") == "ok", str(r.json()))

        r = client.get("/qtypes")
        ok("GET /qtypes -> 200", r.status_code == 200, str(r.status_code))
        ok("GET /qtypes has choice/score/noul", set(r.json().get("types", [])) >= {"choice", "score", "noul"})

        r = client.get("/models")
        ok("GET /models -> 200", r.status_code == 200, str(r.status_code))
        ok("GET /models stays JSON for API clients", set(r.json()) == {"default", "allowed", "models"}, r.text[:200])

        r = client.get("/presets")
        ok("GET /presets -> 200", r.status_code == 200, str(r.status_code))
        presets = r.json()
        expected_presets = {"triage", "email", "guard", "moderation", "router"}
        ok("GET /presets has all five laya.presets workflows", set(presets) == expected_presets, str(set(presets)))
        ok(
            "GET /presets entries have label/state/questions",
            all({"label", "state", "questions"} <= set(p) for p in presets.values()),
        )

        r = client.get("/")
        ok("GET / (playground page) -> 200", r.status_code == 200, str(r.status_code))
        ok("GET / is html", "<html" in r.text.lower())
        ok("GET / embeds PRESETS for the playground JS", "const PRESETS = " in r.text)
        ok("GET / has a preset option per workflow", r.text.count("data-preset=") == len(expected_presets))
        ok("GET / sets the theme in <head>, before first paint", r.text.index("laya.theme") < r.text.index("<body"))
        ok("GET / names the page in an h1", "<h1 class='crumb-t'>Playground</h1>" in r.text)
        ok("GET / keeps a no-JavaScript form that posts to /gui",
           "<noscript><form class='nojs' method='post' action='/gui'>" in r.text)
        ok("GET / loads nothing from the network",
           not re.search(r"""(src|href)=["']?(https?:)?//|url\(\s*["']?(https?:)?//""", r.text))
        r = client.get("/?preset=guard")
        ok("GET /?preset=guard prefills the no-JavaScript form", "&quot;jailbreak&quot;" in r.text)
        r = client.get("/gui", follow_redirects=False)
        ok("GET /gui redirects to the playground", r.status_code == 307 and r.headers.get("location") == "/",
           str(r.status_code))

        r = client.get("/models", headers={"accept": "text/html"})
        ok("GET /models (browser) renders every checkpoint",
           all(f"href='/?model={m}'" in r.text for m in server.MODELS), r.text[:200])
        r = client.get("/health", headers={"accept": "text/html"})
        ok("GET /health (browser) renders the status page", "<h1>Health</h1>" in r.text and "Ready" in r.text)

        payload = {
            "state": "We were billed twice for March. Please refund it today.",
            "questions": {
                "refund_requested": {
                    "type": "noul",
                    "instructions": "Does the user explicitly request a refund?",
                }
            },
        }
        r = client.post("/predict", json=payload)
        ok("POST /predict -> 200", r.status_code == 200, str(r.status_code) + " " + r.text[:200])
        if r.status_code == 200:
            body = r.json()
            ans = body.get("answers", {}).get("refund_requested", {})
            ok("POST /predict answer has noul type", ans.get("type") == "noul", str(ans))
            ok("POST /predict noul in [0,1]", 0.0 <= ans.get("noul", -1) <= 1.0, str(ans))

        r = client.post("/predict", json={"state": "hi", "questions": {"x": {"type": "bogus", "instructions": "?"}}})
        ok("POST /predict rejects unknown qtype with 422", r.status_code == 422, str(r.status_code))

        # A choice/score question with no criteria used to reach laya.render_options and raise
        # there (AttributeError on crit.items()), surfacing as a 500 instead of a validation
        # error -- this is what the criteria-required model_validator on Question now catches.
        r = client.post("/predict", json={"state": "hi", "questions": {"x": {"type": "choice", "instructions": "?"}}})
        ok("POST /predict rejects choice with no criteria as 422, not 500", r.status_code == 422, str(r.status_code))
        r = client.post("/predict", json={"state": "hi", "questions": {"x": {"type": "score", "instructions": "?"}}})
        ok("POST /predict rejects score with no criteria as 422, not 500", r.status_code == 422, str(r.status_code))
        # a null level used to be scored as the text "level 1: null" and echoed back in the legend (#302)
        r = client.post("/predict", json={"state": "hi", "questions": {"x": {
            "type": "score", "instructions": "?", "criteria": ["low", None, "high"]}}})
        ok("POST /predict rejects a null score level as 422", r.status_code == 422, str(r.status_code) + " " + r.text[:200])

        # /gui is the form-post path the no-JavaScript fallback uses -- this is what caught the
        # missing python-multipart dependency during manual verification.
        form = {
            "state": '{"body": "We were billed twice for March. Please refund it."}',
            "questions": '{"refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"}}',
            "model": "",
        }
        r = client.post("/gui", data=form)
        ok("POST /gui (form) -> 200", r.status_code == 200, str(r.status_code) + " " + r.text[:200])
        ok("POST /gui is html", "<html" in r.text.lower())
        ok("POST /gui renders one answer row with a winning bar",
           r.text.count("<details class='ans'") == 1 and r.text.count("class='dr win'") == 1)
        link = re.search(r"href='/#r=([A-Za-z0-9_-]+)'", r.text)
        shared = json.loads(base64.urlsafe_b64decode(link.group(1) + "=" * (-len(link.group(1)) % 4))) if link else {}
        ok("POST /gui links back to the playground with the same request",
           shared == {"state": json.loads(form["state"]), "questions": json.loads(form["questions"])}, str(shared))
        ok("POST /gui links to a fresh playground too", "href='/'>Open the playground</a>" in r.text)

        r = client.post("/gui", data={"state": '"hi"', "questions": '{"x": {"type": "choice", "instructions": "?"}}'})
        ok("POST /gui explains a validation error per location",
           r.status_code == 200 and "questions \u2192 x" in r.text and "criteria" in r.text, errs(r))
        r = client.post("/gui", data={"state": '"hi"', "questions": '{"x": {"type": "noul", "instructions": "?"}}',
                                      "model": "gpt-5"})
        ok("POST /gui drops pydantic's 'Value error, ' prefix",
           "model: unknown model" in r.text and "Value error," not in r.text, errs(r))
        r = client.post("/gui", data={"state": "5", "questions": '{"x": {"type": "choice", "instructions": "?", '
                                                                 '"criteria": "billing"}}'})
        ok("POST /gui folds a union's per-branch errors into one line",
           "state: Input should be a valid string, dictionary or list" in r.text
           and "criteria: Input should be a valid dictionary or list" in r.text and "list[any]" not in r.text,
           errs(r))
        r = client.post("/gui", data={"state": '"hi"', "questions": '{"int": {"type": "choice", "instructions": "a"}, '
                                                                   '"bool": {"type": "choice", "instructions": "b"}}'})
        ok("POST /gui keeps question keys named like union branches apart",
           "questions \u2192 int" in r.text and "questions \u2192 bool" in r.text, errs(r))
        r = client.post("/gui", data={"state": "{not json", "questions": "{}"})
        ok("POST /gui explains invalid JSON", r.status_code == 200 and "Invalid JSON" in r.text, errs(r))

        # laya.guard_questions()'s "topic" criteria uses None as a placeholder for "no
        # description" ({"coding": None, ...}) -- this is what caught two bugs: the
        # Question model rejecting None criteria values (fixed: Dict[str, Optional[str]]),
        # and _criteria_legend rendering the literal string "None" in the UI.
        guard = presets["guard"]
        r = client.post(
            "/gui",
            data={
                "state": json.dumps(guard["state"]),
                "questions": json.dumps(guard["questions"]),
                "model": "",
            },
        )
        ok("POST /gui with guard preset (None criteria) -> 200", r.status_code == 200, str(r.status_code) + " " + r.text[:300])
        ok("POST /gui with guard preset renders real answers",
           r.text.count("<details class='ans'") == len(guard["questions"]), r.text[:300])
        ok("POST /gui with guard preset does not leak literal 'None'",
           ">None<" not in r.text and ">null<" not in r.text and "&mdash; None" not in r.text)
        ok("POST /gui renders `backtick` spans in instructions as code", "<code>prompt</code>" in r.text)
        ok("POST /gui reports server time in the playground's units",
           re.search(r"Server time</dt><dd>(\d+ ms|\d+\.\d\d s)</dd>", r.text) is not None)

        # Every user string on the server-rendered page is escaped, keys and criteria included.
        evil = "<img src=x onerror=alert(1)>"
        hostile = {evil: {"type": "choice", "instructions": "</script><script>alert(2)</script> `" + evil + "`",
                          "criteria": {evil: evil, "other": None}}}
        r = client.post("/gui", data={"state": json.dumps({"body": evil}), "questions": json.dumps(hostile)})
        ok("POST /gui escapes hostile keys, instructions and criteria",
           r.status_code == 200 and evil not in r.text and "<script>alert" not in r.text
           and "&lt;img src=x onerror=alert(1)&gt;" in r.text, str(r.status_code))

        # The no-JS path refuses what /predict refuses, and the page knows the limits for its own check.
        many = {"q%d" % i: {"type": "noul", "instructions": "?"} for i in range(server.MAX_QUESTIONS + 1)}
        r = client.post("/gui", data={"state": json.dumps({"body": "hi"}), "questions": json.dumps(many)})
        ok("POST /gui refuses more questions than /predict takes", r.status_code == 200
           and "Request too large" in r.text and "too many questions" in r.text, r.text[:300])
        r = client.get("/")
        ok("GET / tells the playground the server's request limits",
           "const LIMITS = " + json.dumps({"questions": server.MAX_QUESTIONS, "stateChars": server.MAX_STATE_CHARS})
           in r.text)

    ok("_pct never rounds a near-certainty to 100% or a long shot to 0%",
       (server._pct(0.0004), server._pct(0.9996), server._pct(0.5), server._pct(0.0), server._pct(1.0))
       == ("<0.1%", ">99.9%", "50.0%", "0.0%", "100.0%"))
    ans = {"type": "choice", "choice": "a", "probabilities": {"a": 0.9999, "b": 0.0001}, "answer_confidence": 0.9999,
           "confidence": 0.9986}
    row = server._answer_row("x", ans, {"type": "choice", "instructions": "?"}, 0)
    ok("_answer_row prints laya's 4 decimals, so a near-certainty does not read 1.000",
       "<b>0.9999</b> calibrated" in row and "<b>0.9986</b>" in row, row)
    ok("_answer_row's tooltip shows field names as code", "<code>answer_confidence</code> is the probability" in row)
    row = server._answer_row("x", None, {"type": "choice", "instructions": "?"}, 0)
    ok("_answer_row shows a non-object answer as raw JSON instead of failing", "class='a-raw'>null</pre>" in row, row)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
