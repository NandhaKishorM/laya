"""Run the installed SDK against a separate Python process with real offline inference."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

from laya_client import LayaAPIError, LayaClient, LayaConnectionError


def check_batch(result, expected, strict=False):
    assert len(result["results"]) == len(expected)
    for item, direct in zip(result["results"], expected):
        if strict:
            fields = {"choice": ("type", "choice", "probabilities", "confidence"),
                      "score": ("type", "score", "probabilities", "confidence", "legend"),
                      "noul": ("type", "noul")}
            projected = {
                "model": direct["model"],
                "answers": {qid: {k: answer[k] for k in fields[answer["type"]]}
                            for qid, answer in direct["answers"].items()},
                "usage": {k: direct["usage"][k] for k in ("input_tokens", "output_tokens")},
            }
            assert item == projected, (item, projected)
        else:
            assert item["model"] == direct["model"]
            for qid, answer in direct["answers"].items():
                for key, value in answer.items():
                    assert item["answers"][qid][key] == value, (qid, key)
            for key, value in direct["usage"].items():
                assert item["usage"][key] == value, key
            assert item["routing"] == direct["routing"]
    for key in ("input_tokens", "output_tokens"):
        assert result["total_usage"][key] == sum(item["usage"][key] for item in expected), key


def check_batch_http(client, anonymous, fixture, strict, legacy):
    questions = fixture["questions"]
    if legacy:
        # v0.4.7 has a batch endpoint. One-state passes give its unchanged fixture a parity oracle.
        states = ["hello", fixture["state"], ["hello"]]
        expected = [client.predict(state, questions, model="english") for state in states]
        check_batch(client.predict_batch(states, questions, model="english", batch_size=1), expected)
        check_batch(client.predict_batch(states[:1], questions, model="english"), expected[:1])
    else:
        for case in fixture["batch_cases"]:
            result = client.predict_batch(case["states"], questions, model="english", **case["controls"])
            check_batch(result, case["expected"], strict)
        labels = {str(i): "Option %d" % i for i in range(5)}
        large = {"large": {"type": "choice", "instructions": "Pick one", "criteria": labels}}
        states = fixture["batch_cases"][0]["states"]
        # Tournament execution is per state on the server, even when batch shape controls are set.
        expected = [client.predict(state, large, model="english") for state in states]
        check_batch(client.predict_batch(states, large, model="english",
                                         batch_size=2, sort_by_length=True), expected, strict)

    small = {"refund": questions["refund"]}
    result = client.predict_batch(["hello"] * 64, small, model="english", batch_size=8)
    assert len(result["results"]) == 64
    for key in ("input_tokens", "output_tokens"):
        assert result["total_usage"][key] == sum(item["usage"][key] for item in result["results"])
    for caller, states, controls, status in (
        (client, ["hello"] * 65, {}, 413),
        (client, ["hello", "x" * 50_001], {}, 413),
        (client, ["hello"], {"model": "unpublished/checkpoint"}, 422),
        (anonymous, ["hello"], {}, 401),
    ):
        try:
            caller.predict_batch(states, small, **controls)
        except LayaAPIError as error:
            assert error.status_code == status, (error.status_code, status, error.detail)
        else:
            raise AssertionError("Batch request should have failed with HTTP %d" % status)


def run(server_python, fixture_path, strict=False, legacy=False):
    with tempfile.TemporaryDirectory(prefix="laya-sdk-live-") as directory:
        root = Path(directory)
        command = [str(server_python), "-u", str(fixture_path)]
        if not legacy:
            command.extend(["--tournament", "--batch"])
            if strict:
                command.append("--strict")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LAYA_TOURNAMENT_GROUP="0",
                   LAYA_JEV_STRICT="0", LAYA_PRELOAD="0", LAYA_IDLE_UNLOAD_SECONDS="0")
        with (root / "stderr.txt").open("w+", encoding="utf-8", errors="replace") as stderr:
            server = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                      stderr=stderr, text=True, encoding="utf-8")
            lines = queue.Queue()

            def read_lines():
                for line in server.stdout:
                    if line.startswith("LAYA_TEST_SERVER="):
                        lines.put(json.loads(line.split("=", 1)[1]))
                        return
                lines.put(None)

            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            try:
                fixture = lines.get(timeout=90)
                if fixture is None:
                    raise RuntimeError("Server exited before producing its fixture")
                client = LayaClient(base_url=fixture["baseURL"], api_key="integration-test", timeout=10)
                deadline = time.monotonic() + 15
                while True:
                    try:
                        client.health(timeout=0.5)
                        break
                    except LayaConnectionError:
                        if time.monotonic() >= deadline or server.poll() is not None:
                            raise
                        time.sleep(0.05)
                result = client.predict(fixture["state"], fixture["questions"], model="english")
                expected = fixture["expected"]
                assert result["model"] == expected["model"]
                if strict:
                    fields = {"choice": ("type", "choice", "probabilities", "confidence"),
                              "score": ("type", "score", "probabilities", "confidence", "legend"),
                              "noul": ("type", "noul")}
                    projected = {
                        "model": expected["model"],
                        "answers": {qid: {k: answer[k] for k in fields[answer["type"]]}
                                    for qid, answer in expected["answers"].items()},
                        "usage": {k: expected["usage"][k] for k in ("input_tokens", "output_tokens")},
                    }
                    assert result == projected, (result, projected)
                    gated = client.predict(fixture["state"], fixture["questions"], min_confidence=0.99)
                    assert all("abstention" not in a for a in gated["answers"].values())
                else:
                    for qid, answer in expected["answers"].items():
                        for key, value in answer.items():
                            assert result["answers"][qid][key] == value, (qid, key)
                    for key, value in expected["usage"].items():
                        assert result["usage"][key] == value, key
                    assert result["routing"]["model"] == "english"
                    auto = client.predict(fixture["state"], fixture["questions"])
                    assert auto["routing"]["model"] == "multilingual"
                    gated = client.predict(fixture["state"], fixture["questions"], lang="en",
                                           max_len=128, head_max_len=64, min_confidence=0.0)
                    assert gated["routing"]["model"] == "english"
                    assert all(a["abstention"] == "passed" for a in gated["answers"].values())

                if not legacy:
                    labels = {str(i): "Option %d" % i for i in range(5)}
                    answer = client.predict("hello", {"large": {
                        "type": "choice", "instructions": "Pick one", "criteria": labels,
                    }})["answers"]["large"]
                    assert set(answer["probabilities"]) == set(labels)
                    assert answer["choice"] in labels
                    if strict:
                        assert "x_tournament" not in answer
                    else:
                        finalists = answer["x_tournament"]["finalists"]
                        assert 0 < len(finalists) <= 2
                        assert all(p == 0.0 for k, p in answer["probabilities"].items() if k not in finalists)

                anonymous = LayaClient(base_url=fixture["baseURL"], api_key="wrong")
                assert anonymous.health() == {"status": "ok"}
                try:
                    anonymous.predict("hello", fixture["questions"])
                except LayaAPIError as error:
                    assert error.status_code == 401
                else:
                    raise AssertionError("Unauthenticated prediction succeeded")
                try:
                    client.predict("x" * 50_001, fixture["questions"])
                except LayaAPIError as error:
                    assert error.status_code == 413
                else:
                    raise AssertionError("Server request limit was bypassed")
                check_batch_http(client, anonymous, fixture, strict, legacy)
                print("PASS real HTTP single and batch inference, order, controls, auth and limits: "
                      + ("legacy fixture" if legacy else "strict" if strict else "extended/tournament"), flush=True)
            except BaseException:
                stderr.flush()
                stderr.seek(0)
                print(stderr.read(), file=sys.stderr)
                raise
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                server.stdout.close()
                reader.join(timeout=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-python", required=True, type=Path)
    parser.add_argument("--server-fixture", type=Path,
                        default=Path(__file__).resolve().parents[3] / "tests" / "sdk_server_fixture.py")
    parser.add_argument("--legacy-fixture", action="store_true")
    args = parser.parse_args()
    run(args.server_python.resolve(), args.server_fixture.resolve(), legacy=args.legacy_fixture)
    if not args.legacy_fixture:
        run(args.server_python.resolve(), args.server_fixture.resolve(), strict=True)
