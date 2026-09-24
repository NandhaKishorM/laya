#!/usr/bin/env python3
"""Throughput and latency of a running laya-serve, at several concurrency levels.

Compares batching on vs off by restarting the server with `LAYA_BATCH_WINDOW_MS=0`
(the "off" side of the table below). Stdlib only; no dependencies.

    python scripts/bench_serve.py --url http://127.0.0.1:8000 --state "hello"
"""
import argparse
import json
import threading
import time
import urllib.request

BODY = {"state": "hello", "questions": {"q": {"type": "noul", "instructions": "how is it?"}}}


def one_post(url, body, api_key, latencies, errors):
    request = urllib.request.Request(
        url + "/v1/systemone", data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": "Bearer " + api_key} if api_key else {})})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response.read()
    except Exception as exc:  # noqa: BLE001 - a load test counts failures, it does not stop on them
        errors.append(str(exc))
    finally:
        latencies.append((time.perf_counter() - start) * 1000.0)


def run_round(url, body, api_key, concurrency, requests):
    latencies, errors = [], []
    per_thread = max(1, requests // concurrency)
    start = time.perf_counter()

    def worker():
        for _ in range(per_thread):
            one_post(url, body, api_key, latencies, errors)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
    total = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": total,
        "ok": total - len(errors),
        "errors": len(errors),
        "throughput": total / elapsed if elapsed else 0.0,
        "p50_ms": p50,
        "p99_ms": p99,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--state", default="hello", help="value for the request's `state`")
    parser.add_argument("--concurrency", default="1,8,32")
    parser.add_argument("--requests", type=int, default=200, help="requests per concurrency level")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    body = dict(BODY, state=args.state)
    print("%11s %8s %7s %9s %9s %9s" % ("concurrency", "requests", "errors", "req/s", "p50(ms)", "p99(ms)"))
    for level in (int(item) for item in args.concurrency.split(",")):
        row = run_round(args.url, body, args.api_key, level, args.requests)
        print("%11d %8d %7d %9.1f %9.1f %9.1f" % (
            row["concurrency"], row["requests"], row["errors"],
            row["throughput"], row["p50_ms"], row["p99_ms"]))


if __name__ == "__main__":
    main()
