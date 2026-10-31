"""Compare local routing with the deployed Router, without loading checkpoints.

From a checkout with the serve extra and its HTTP test client installed:
    python -m pip install -e ".[serve]" httpx
    python examples/http_route.py

TestClient exercises the real HTTP handler in process; it opens no listening socket.
"""
import json
import os

from _common import Router
from fastapi.testclient import TestClient

from laya.serve import create_app


def main():
    state = {"message": "Turn lights off"}
    body = {"state": state, "questions": {}}
    local = Router(default="multilingual")
    deployed = Router(default="english")
    local_decision = local.route(state, explain=True)

    # Injecting a Router avoids the normal server's startup preload. Honour a key
    # from the environment without printing it or sending it over a network.
    api_key = os.environ.get("LAYA_API_KEY")
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    with TestClient(create_app(deployed), headers=headers) as client:
        plain = client.post("/v1/route", json=body)
        plain.raise_for_status()
        assert set(plain.json()) == {"model", "repo", "reason"}

        response = client.post("/v1/route", json={**body, "explain": True})
        response.raise_for_status()
        snapshot = response.json()
        assert local_decision["model"] == "multilingual"
        assert snapshot["model"] == "english"
        assert snapshot["diagnostic"] == local_decision["detection"]["diagnostic"]
        assert {key: snapshot[key] for key in plain.json()} == plain.json()

        # A new request with a model pin has a new decision. The earlier snapshot
        # neither pins a later prediction nor manufactures language evidence.
        response = client.post("/v1/route", json={**body, "model": "multilingual", "explain": True})
        response.raise_for_status()
        pinned = response.json()
        assert pinned["model"] == "multilingual" and pinned["diagnostic"] is None

    assert local.loaded == [] and deployed.loaded == []
    print(json.dumps({
        "local_model": local_decision["model"],
        "server_snapshot": snapshot,
        "server_snapshot_with_pin": pinned,
        "resident_checkpoints": deployed.loaded,
    }, indent=2))


if __name__ == "__main__":
    main()
