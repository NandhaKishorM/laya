"""Measure #888 with a local checkpoint; never download weights.

Run each mode in a fresh process. See README.md beside this file for commands.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_OFFLINE"] = "1"

STATE = {"message": "I was billed twice. Please refund the duplicate payment."}
QUESTIONS = {"dept": {"type": "choice", "instructions": "Which team handles `message`?",
                      "criteria": {"billing": "payments and refunds", "technical": "bugs and outages"}}}
SCHEMA = {"type": "object", "properties": {"dept": {"type": "string", "enum": ["billing", "technical"]}}}


def rss_mb():
    value = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return round(int(value.strip()) / 1024, 3)


def memory():
    import torch
    torch.mps.synchronize()
    return {"rss_mb": rss_mb(),
            "metal_mb": round(torch.mps.current_allocated_memory() / 1024**2, 3),
            "metal_driver_mb": round(torch.mps.driver_allocated_memory() / 1024**2, 3)}


def make_router(checkpoint):
    from laya.router import Router
    router = Router(models={"english": str(checkpoint)}, device="mps")
    router.preload(["english"])
    return router


def idle_benchmark(args):
    from fastapi.testclient import TestClient
    from laya.serve import create_app

    enabled = args.mode == "idle-on"
    os.environ["LAYA_IDLE_UNLOAD_SECONDS"] = str(args.idle_seconds if enabled else 0)
    started = time.perf_counter()
    router = make_router(args.checkpoint)
    load_seconds = time.perf_counter() - started
    with TestClient(create_app(router)) as client:
        body = {"state": STATE, "questions": QUESTIONS}

        def predict():
            start = time.perf_counter()
            response = client.post("/v1/systemone", json=body)
            response.raise_for_status()
            return response.json()["answers"], (time.perf_counter() - start) * 1000

        predict()  # warm up Metal kernels
        answers, warm_ms = predict()
        resident = memory()
        if enabled:
            deadline = time.monotonic() + args.idle_seconds + 30
            while client.get("/health").json()["loaded"]:
                assert time.monotonic() < deadline, "idle unload did not complete"
                time.sleep(0.05)
        else:
            time.sleep(args.idle_seconds + 0.5)
        after_idle = memory()
        assert bool(router.loaded) != enabled
        reloaded_answers, next_ms = predict()
        assert answers == reloaded_answers, "reload changed the answer"
        assert router.loaded == ["english"]
        result = {"mode": args.mode, "load_seconds": round(load_seconds, 3),
                  "warm_ms": round(warm_ms, 3), "next_request_ms": round(next_ms, 3),
                  "resident": resident, "after_idle": after_idle, "after_next_request": memory(),
                  "answers": answers, "answer_parity": True}
    router.unload()
    return result


def mcp_benchmark(args):
    if args.mode == "mcp-remote":
        assert args.base_url, "--base-url is required in remote mode"
        os.environ["LAYA_BASE_URL"] = args.base_url
    else:
        os.environ.pop("LAYA_BASE_URL", None)
    from laya.mcp import server

    imported_rss = rss_mb()
    if args.mode == "mcp-local":
        server._ROUTER = make_router(args.checkpoint)
    outputs = {}
    outputs["status"] = json.loads(server.laya_status_tool())
    outputs["predict"] = json.loads(server.laya_predict_tool(STATE, QUESTIONS))
    items = [{"state": STATE, "questions": QUESTIONS}]
    outputs["predict_batch"] = json.loads(server.laya_predict_batch_tool(items))
    outputs["decide"] = json.loads(server.laya_decide_tool(STATE, SCHEMA))
    outputs["preset"] = json.loads(server.laya_preset_tool("triage", STATE))
    outputs["route"] = json.loads(server.laya_route_tool(STATE, QUESTIONS))
    outputs["route_batch"] = json.loads(server.laya_route_batch_tool(items))
    assert all("error" not in value for value in outputs.values()), outputs
    loaded = server._ensure_router().loaded
    result = {"mode": args.mode, "imported_rss_mb": imported_rss, "after_tools_rss_mb": rss_mb(),
              "torch_imported": "torch" in sys.modules, "loaded": loaded,
              "answers": outputs["predict"]["answers"],
              "batch_answers": outputs["predict_batch"]["requests"][0]["answers"],
              "decided_values": outputs["decide"]["values"], "preset_answers": outputs["preset"]["answers"]}
    assert result["torch_imported"] == (args.mode == "mcp-local")
    if args.mode == "mcp-local":
        server._ROUTER.unload()
    else:
        try:
            server.laya_shortlist_tool(STATE, QUESTIONS)
        except Exception as exc:
            assert "unsupported_remote" in str(exc), exc
        else:
            raise AssertionError("remote shortlist did not refuse the call")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["idle-off", "idle-on", "mcp-local", "mcp-remote", "serve"], required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    args = parser.parse_args()
    assert (args.checkpoint / "model.safetensors").is_file(), "local checkpoint is incomplete"
    os.environ.pop("LAYA_API_KEY", None)
    if args.mode == "serve":
        import uvicorn
        from laya.serve import create_app
        os.environ["LAYA_IDLE_UNLOAD_SECONDS"] = "0"
        router = make_router(args.checkpoint)
        try:
            uvicorn.run(create_app(router), host="127.0.0.1", port=args.port, log_level="warning")
        finally:
            router.unload()
        return
    result = idle_benchmark(args) if args.mode.startswith("idle-") else mcp_benchmark(args)
    result["checkpoint_revision"] = args.checkpoint.name
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
