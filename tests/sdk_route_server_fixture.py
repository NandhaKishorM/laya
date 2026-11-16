"""Model-free SDK routing fixture using the real Router and HTTP application."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402
from laya import BaseHook, Router  # noqa: E402
from laya.serve import create_app  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-router", action="store_true")
    args = parser.parse_args()
    counts = dict(route=0, hook=0, load=0, predict=0)

    class ObserveRoute(BaseHook):
        def on_route(self, ctx):
            counts["hook"] += 1
            state = ctx.states[0]
            if isinstance(state, dict) and "operator_diagnostic" in state:
                ctx.decision = dict(ctx.decision, model="support", repo=None, reason="operator override",
                                    detection={"diagnostic": state["operator_diagnostic"],
                                               "mixed_segment": "private input value"})

    router = Router(default="english", hooks=[ObserveRoute()])
    oracle = Router(default="english")
    for instance in (router, oracle):
        instance.register("support", "acme/support")
        instance.attach("resident", object())

    def forbid_load(*args, **kwargs):
        counts["load"] += 1
        raise AssertionError("routing must not load checkpoints")

    def forbid_predict(*args, **kwargs):
        counts["predict"] += 1
        raise AssertionError("routing must not predict")

    route = router.route

    def counted_route(*args, **kwargs):
        counts["route"] += 1
        return route(*args, **kwargs)

    def legacy_route(state, questions, model=None, task=None, lang=None, lang_guess=None):
        return counted_route(state, questions, model=model, task=task, lang=lang, lang_guess=lang_guess)

    router.load = forbid_load
    router.predict = router.predict_batch = forbid_predict
    router.route = legacy_route if args.legacy_router else counted_route
    cases = []
    for state, questions, options in (
        ("Turn lights off", {}, {}),
        ("Turn lights off", {}, {"explain": True}),
        ({"customer": "Ich kann mich nicht anmelden und brauche Hilfe mit der Rechnung"}, {}, {"explain": True}),
        ("hello", {}, {"model": "support", "explain": True}),
        ("hello", {}, {"model": "resident"}),
        ("hello", {}, {"task": "support"}),
        ("hello", {}, {"lang": "de"}),
        ("hello", {}, {"lang_guess": "fr"}),
        ("hello", {"intent": {"instructions": "Routing does not require a complete inference question"}}, {}),
    ):
        decision = oracle.route(state, questions, **options)
        expected = {key: decision[key] for key in ("model", "repo", "reason")}
        if options.get("explain"):
            detection = decision.get("detection")
            expected["diagnostic"] = detection.get("diagnostic") if isinstance(detection, dict) else None
        cases.append(dict(state=state, questions=questions, options=options, expected=expected))

    os.environ["LAYA_API_KEY"] = "integration-test"
    os.environ["LAYA_PRELOAD"] = "0"
    app = create_app(router)

    @app.get("/fixture-counts")
    def fixture_counts():
        return counts

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        print("LAYA_TEST_SERVER=" + json.dumps({
            "baseURL": "http://127.0.0.1:%d" % listener.getsockname()[1], "route_cases": cases,
        }), flush=True)
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[listener])


if __name__ == "__main__":
    main()
