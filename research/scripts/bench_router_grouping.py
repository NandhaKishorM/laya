"""Synthetic Router orchestration benchmark; no weights or model inference.

Run with ``python research/scripts/bench_router_grouping.py``. Cold loads have a fixed
 delay to illustrate cache churn; elapsed times depend on the host and are not a model
throughput or latency measurement.
"""
import argparse
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from laya.router import Router  # noqa: E402


Q = {"intent": {"type": "noul", "instructions": "Relevant?"}}


def requests(count):
    return [{"state": "request %d" % i, "questions": Q,
             "model": "english" if i % 2 == 0 else "multilingual"}
            for i in range(count)]


def measure(items, capacity, delay, grouped):
    constructions = []
    evictions = []

    class StubAgent:
        def __init__(self, repo, *, device, token, subfolder):
            key = subfolder or "english"
            constructions.append(key)
            time.sleep(delay)

        def system_one(self, state, questions):
            return {"model": "stub", "answers": {"seen": state}, "usage": {}}

    with patch("laya.agent.Agent", StubAgent):
        router = Router(max_loaded=capacity)
        original_evict = router._evict

        def count_evictions():
            before = set(router.loaded)
            original_evict()
            evictions.extend(before - set(router.loaded))

        router._evict = count_evictions
        start = time.perf_counter()
        if grouped:
            outputs = router.predict_batch(items)
        else:
            outputs = [router.predict(item["state"], item["questions"], model=item["model"])
                       for item in items]
        elapsed = time.perf_counter() - start

    expected = [item["state"] for item in items]
    correct = [output["answers"]["seen"] for output in outputs] == expected
    correct &= [output["routing"]["model"] for output in outputs] == [item["model"] for item in items]
    return len(constructions), len(evictions), elapsed, correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--load-delay", type=float, default=0.01,
                        help="seconds per synthetic cold load")
    args = parser.parse_args()
    if args.requests < 0 or args.load_delay < 0:
        parser.error("requests and load-delay must be non-negative")
    items = requests(args.requests)
    print("synthetic orchestration only; requests=%d; cold-load delay=%.3f s" %
          (len(items), args.load_delay))
    print("max_loaded method constructions evictions elapsed_s ordered")
    for capacity in (1, 2):
        for name, grouped in (("repeated_predict", False), ("predict_batch", True)):
            loads, evictions, elapsed, correct = measure(items, capacity, args.load_delay, grouped)
            print("%d %s %d %d %.6f %s" % (capacity, name, loads, evictions, elapsed, correct))
            if not correct:
                raise AssertionError("result ordering or routing differs from input")


if __name__ == "__main__":
    main()
