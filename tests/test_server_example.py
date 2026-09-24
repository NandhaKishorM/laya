"""Smoke test for examples/server.py: does the demo API wiring still match laya's
public surface (Router, DEFAULT_MODELS, QTYPES)?

Loads real weights for the `english` checkpoint only (skips multilingual/typed-decisions
to keep this fast) and drives the FastAPI app in-process via TestClient -- no network
socket, no subprocess.

Run:  python3 tests/test_server_example.py
"""
import json
import os
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
        ok("GET / (builder page) -> 200", r.status_code == 200, str(r.status_code))
        ok("GET / is html", "<html" in r.text.lower())
        ok("GET / embeds PRESETS for the builder JS", "const PRESETS = " in r.text)
        ok("GET / has a preset button per workflow", r.text.count("data-preset=") == len(expected_presets))

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

        # /gui is the form-post path the builder JS uses -- this is what caught the
        # missing python-multipart dependency during manual verification.
        r = client.post(
            "/gui",
            data={
                "state": '{"body": "We were billed twice for March. Please refund it."}',
                "questions": '{"refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"}}',
                "model": "",
            },
        )
        ok("POST /gui (form) -> 200", r.status_code == 200, str(r.status_code) + " " + r.text[:200])
        ok("POST /gui is html", "<html" in r.text.lower())

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
        ok("POST /gui with guard preset renders real answers", "<h1>Answers." in r.text, r.text[:300])
        ok("POST /gui with guard preset does not leak literal 'None'", "&mdash; None" not in r.text)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print("  -", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
