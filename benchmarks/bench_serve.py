"""Throughput of `laya.serve` at 1 / 8 / 32 concurrent clients, with and without dynamic batching.

    python benchmarks/bench_serve.py                       # fake agent with a launch-bound cost model (CPU only)
    python benchmarks/bench_serve.py --real [--backend tilelang] [--subfolder multilingual]   # a real Router on the GPU

The app is driven in-process over `httpx.ASGITransport`, so what is measured is the server's own
scheduling (event loop, executor, batcher) plus inference, without a network. Each cell sends
`--requests` requests from N concurrent clients and reports requests/s and p50/p95 latency.

The fake router models a GPU forward the way benchmarks/parity.py's hardware behaves: a fixed launch
cost per forward plus a small per-request cost (`--fixed-ms`, `--per-request-ms`), so merging eight
requests costs about one. The `--real` run loads the real checkpoints; on a shared GPU label the
numbers as measured under contention (the JSON records `nvidia-smi` before the run).
"""
import argparse, asyncio, json, os, statistics, subprocess, sys, time
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQ = {"state": {"subject": "Duplicate charge on invoice 4411",
                 "body": "We were billed twice for March. Please refund the duplicate or we're moving to a competitor."},
       "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                    "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "pricing"}},
                     "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
                     "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"}}}


class FakeRouter:
    """A launch-bound forward: `fixed_ms` per forward + `per_request_ms` per request in it."""
    loaded = ["english"]

    def __init__(self, fixed_ms, per_request_ms):
        self.fixed, self.per = fixed_ms / 1000.0, per_request_ms / 1000.0
        self.batch_sizes = []

    def _answer(self):
        return {"model": "laya-rl-agent", "answers": {"department": {"type": "choice", "choice": "billing"}},
                "usage": {"input_tokens": 40, "output_tokens": 0}, "routing": {"model": "english"}}

    def predict(self, state, questions, model=None):
        self.batch_sizes.append(1); time.sleep(self.fixed + self.per); return self._answer()

    def predict_batch(self, requests, batch_size=None):
        self.batch_sizes.append(len(requests)); time.sleep(self.fixed + self.per * len(requests))
        return [self._answer() for _ in requests]


def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "n/a"


def run_cell(app, clients, requests):
    import httpx

    async def go():
        lat = []
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=120) as client:
            per_client = max(1, requests // clients)

            async def worker():
                for _ in range(per_client):
                    t = time.perf_counter()
                    r = await client.post("/v1/systemone", json=REQ)
                    assert r.status_code == 200, r.text
                    lat.append((time.perf_counter() - t) * 1000)
            for _ in range(2):                         # warm-up
                await client.post("/v1/systemone", json=REQ)
            t0 = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(clients)))
            wall = time.perf_counter() - t0
        lat.sort()
        return {"clients": clients, "requests": len(lat), "req_per_s": round(len(lat) / wall, 1),
                "p50_ms": round(statistics.median(lat), 2), "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 2)}
    return asyncio.run(go())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true"); ap.add_argument("--backend", default=None); ap.add_argument("--subfolder", default=None)
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--requests", type=int, default=256); ap.add_argument("--clients", default="1,8,32")
    ap.add_argument("--windows", default="0,2", help="LAYA_BATCH_WINDOW_MS values to compare")
    ap.add_argument("--fixed-ms", type=float, default=4.0); ap.add_argument("--per-request-ms", type=float, default=0.3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from laya.serve import create_app

    report = {"real": a.real, "gpu_state_before": gpu_state(), "cells": []}
    if a.real:
        from laya.router import Router
        # every request routes to "english"; point that name at the checkpoint under test
        router = Router(backend=a.backend, models={"english": (a.model, a.subfolder)})
        router.preload(["english"])
        report.update({"backend": router.load("english").backend, "model": a.model, "subfolder": a.subfolder})
    else:
        router = FakeRouter(a.fixed_ms, a.per_request_ms)
        report.update({"fixed_ms": a.fixed_ms, "per_request_ms": a.per_request_ms})
    print(report, flush=True)
    for window in a.windows.split(","):
        os.environ["LAYA_BATCH_WINDOW_MS"] = window
        for clients in (int(c) for c in a.clients.split(",")):
            router.batch_sizes = []
            # one app per cell: each cell runs its own event loop, and the app's gate binds to one loop
            cell = run_cell(create_app(router=router), clients, a.requests)
            cell["window_ms"] = float(window)
            sizes = router.batch_sizes
            cell["mean_batch"] = round(sum(sizes) / max(1, len(sizes)), 2)
            report["cells"].append(cell); print(cell, flush=True)
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1); print("saved", a.json)


if __name__ == "__main__":
    main()
