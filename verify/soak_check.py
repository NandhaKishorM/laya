"""Soak, memory and concurrency check -- the things a short correctness run never shows.

    .venv/bin/python verify/soak_check.py [--models ./models] [--device auto|cpu|mps]
                                          [--calls 200] [--cycles 4] [--threads 4]

Three questions this answers, none of which the other suites ask:

1. Does repeated inference stay correct and stay flat? N calls on one agent, checking that every
   answer matches the first one and that no Python object or traced heap allocation is retained
   (RSS is reported rather than asserted: allocators keep freed pages, so it grows without a leak).
2. Does the router leak across reloads? Alternate checkpoints through the LRU and watch RSS.
3. Is a loaded agent safe to call from several threads, and what does that buy in throughput?

Exits non-zero if a check fails. Concurrency is attempted on the selected device and retried on
CPU if the device rejects it; that outcome is reported rather than hidden.
"""
import argparse
import gc
import json
import os
import statistics
import sys
import threading
import time
import tracemalloc

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # repository root; <root>/laya/email.py must not shadow stdlib email

import torch  # noqa: E402

import laya  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from laya.router import Router  # noqa: E402

FAILS = []
STATE = {"body": "I was charged twice for invoice 4411. Please refund the duplicate today."}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages", "sales": "pricing", "other": "rest"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["not urgent", "soon", "critical"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}


# Resident memory of this process. psutil reads it through sysctl/mach; the fallback is the
# kernel's peak RSS, which never falls, so it still catches an upward trend. (`ps` would be a
# third option but spawning processes can be blocked by a sandbox.)
try:
    import psutil

    def rss_mb():
        return psutil.Process().memory_info().rss / 1e6

    RSS_SOURCE = "psutil"
except ImportError:  # pragma: no cover - only when psutil is absent
    import resource

    def rss_mb():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3   # bytes on macOS, KB on Linux

    RSS_SOURCE = "resource.getrusage (peak RSS)"


def _live(cls):
    """How many instances of `cls` the garbage collector can still see."""
    return sum(1 for o in gc.get_objects() if isinstance(o, cls))


def report(name, ok, detail=""):
    print("   %s %-52s %s" % ("PASS" if ok else "FAIL", name, detail if not ok else ""))
    if not ok:
        FAILS.append("%s (%s)" % (name, detail))


def head(title):
    print("\n" + "=" * 74 + "\n  " + title + "\n" + "=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    models = {"english": os.path.join(args.models, "laya"),
              "multilingual": os.path.join(args.models, "laya-multilingual"),
              "typed-decisions": os.path.join(args.models, "laya-typed-decisions")}
    for name, path in models.items():
        if not os.path.exists(os.path.join(path, "model.safetensors")):
            sys.exit("missing checkpoint %r -- run ./setup_laya.sh" % name)
    device = None if args.device == "auto" else args.device

    print("torch %s | pid %d | RSS via %s | %d calls, %d cycles, %d threads"
          % (torch.__version__, os.getpid(), RSS_SOURCE, args.calls, args.cycles, args.threads))

    # ------------------------------------------------------------ 1. steady state
    head("1. Repeated inference on one agent")
    agent = laya.load(models["multilingual"], device=device)
    print("   device: %s   baseline RSS %.0f MB" % (agent.device, rss_mb()))
    first = agent.predict(STATE, QUESTIONS)
    reference = json.dumps(first["answers"], sort_keys=True)

    latencies, mismatches = [], 0
    gc.collect()
    objects_before = len(gc.get_objects())
    after_warm = rss_mb()
    t0 = time.perf_counter()
    for _ in range(args.calls):
        t1 = time.perf_counter()
        out = agent.predict(STATE, QUESTIONS)
        latencies.append((time.perf_counter() - t1) * 1000)
        if json.dumps(out["answers"], sort_keys=True) != reference:
            mismatches += 1
    elapsed = time.perf_counter() - t0
    grew = rss_mb() - after_warm
    gc.collect()
    objects_grew = len(gc.get_objects()) - objects_before

    report("every answer identical to the first", mismatches == 0, "%d mismatches" % mismatches)
    # Resident size cannot separate a leak from an allocator that keeps freed pages: on macOS
    # this exact loop grew RSS ~280 MB while retaining nothing. Section 2 makes the same point.
    # So retention is measured where it is unambiguous -- live Python objects and the traced
    # heap -- and RSS is reported, guarded only against runaway growth.
    report("no Python objects retained across %d calls" % args.calls,
           objects_grew <= 500, "gc objects grew %d" % objects_grew)

    # Traced heap in its own pass, so the tracking overhead does not land in the latency numbers.
    heap_calls = max(20, args.calls // 4)
    gc.collect()
    tracemalloc.start()
    heap_before = tracemalloc.take_snapshot()
    for _ in range(heap_calls):
        agent.predict(STATE, QUESTIONS)
    gc.collect()
    heap_after = tracemalloc.take_snapshot()
    heap_grew = sum(st.size_diff for st in heap_after.compare_to(heap_before, "filename"))
    tracemalloc.stop()
    report("Python heap flat across %d further calls" % heap_calls,
           heap_grew < 5e6, "traced heap grew %.1f MB" % (heap_grew / 1e6))
    report("RSS growth stays sub-gigabyte (freed pages are kept, not leaked)",
           grew < 1024.0, "grew %.0f MB" % grew)
    print("   latency p50 %.0f ms  p95 %.0f ms  |  %.1f calls/s  |  RSS %.0f -> %.0f MB"
          % (statistics.median(latencies), sorted(latencies)[int(len(latencies) * 0.95)],
             args.calls / elapsed, after_warm, after_warm + grew))
    del agent

    # ------------------------------------------------------------ 2. reload / eviction
    # RSS alone cannot tell a leak from an allocator that keeps freed pages (macOS malloc does
    # not hand them back), and repeated checkpoint loads always push RSS up. So the leak check
    # counts live objects -- Agent / DecisionModel instances -- which is the property that
    # actually matters; RSS is only reported, with a generous envelope around it.
    head("2. Reload and eviction cycles (max_loaded=1)")
    r = Router(models=models, device=device, max_loaded=1)
    r.load("english")
    one_model_rss = rss_mb()
    print("   one checkpoint resident: RSS %.0f MB" % one_model_rss)
    live_counts = []
    for i in range(args.cycles):
        name = "multilingual" if i % 2 == 0 else "english"
        r.load(name)                       # evicts the other checkpoint
        r.predict(STATE, QUESTIONS, model=name)
        agents = _live(laya.Agent)
        models_live = _live(DecisionModel)
        live_counts.append((i + 1, agents, models_live, name, rss_mb()))
        print("   cycle %d: loaded %-14s RSS %7.0f MB   live agents %d  live models %d"
              % (i + 1, name, rss_mb(), agents, models_live))

    report("%d reload cycles completed" % args.cycles, r.loaded in (["english"], ["multilingual"]),
           "loaded=%s" % r.loaded)
    report("evicted checkpoints are really released (1 live agent, 1 live model)",
           all(a == 1 and m == 1 for _, a, m, _, _ in live_counts),
           "counts=%s" % [(a, m) for _, a, m, _, _ in live_counts])
    peak = max(row[4] for row in live_counts)
    envelope = 3.0 * one_model_rss        # one resident checkpoint, plus allocator retention
    report("RSS stays inside a bounded envelope across reloads", peak < envelope,
           "peak %.0f MB, envelope %.0f MB (freed pages are kept by the allocator, not leaked)"
           % (peak, envelope))
    r.unload()
    gc.collect()
    report("unload() clears the router", r.loaded == [], "loaded=%s" % r.loaded)
    report("nothing survives unload()",
           _live(laya.Agent) == 0 and _live(DecisionModel) == 0,
           "live agents %d, live models %d" % (_live(laya.Agent), _live(DecisionModel)))
    print("   after unload()+gc: RSS %.0f MB (allocator retains freed pages; objects are gone)"
          % rss_mb())

    # ------------------------------------------------------------ 3. concurrency
    head("3. Concurrent calls from %d threads" % args.threads)
    per_thread = max(1, args.calls // (args.threads * 4))
    total = args.threads * per_thread

    def worker(ag, results, errors):
        """Every worker asks the same questions about the same state, so any cross-talk or
        corruption shows up as an answer that differs from the single-threaded reference.
        The agent is passed in rather than closed over, so a device retry cannot change it
        underneath running threads."""
        try:
            for _ in range(per_thread):
                out = ag.predict(STATE, QUESTIONS)
                results.append(json.dumps(out["answers"], sort_keys=True))
        except Exception as e:  # noqa: BLE001
            errors.append("%s: %s" % (type(e).__name__, str(e)[:120]))

    primary = laya.load(models["multilingual"], device=device)
    ref = json.dumps(primary.predict(STATE, QUESTIONS)["answers"], sort_keys=True)
    attempts = [primary.device.type] + (["cpu"] if primary.device.type != "cpu" else [])
    for attempt_device in attempts:
        if primary.device.type != attempt_device:
            print("   the %s pass failed; retrying on cpu" % primary.device.type)
            primary = laya.load(models["multilingual"], device="cpu")
        results, errors = [], []
        threads = [threading.Thread(target=worker, args=(primary, results, errors))
                   for _ in range(args.threads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t0
        if not errors:
            report("all %d concurrent calls returned (%s)" % (total, attempt_device), len(results) == total,
                   "expected %d, got %d" % (total, len(results)))
            report("concurrent answers match the single-threaded reference",
                   all(r == ref for r in results),
                   "%d of %d differ" % (sum(1 for r in results if r != ref), len(results)))
            print("   %d calls in %.1f s across %d threads (%.0f calls/s)"
                  % (total, elapsed, args.threads, total / elapsed))
            if attempt_device != primary.device.type:
                print("   note: the %s pass is what succeeded, not the default device"
                      % attempt_device)
            break
        report("concurrency on %s" % attempt_device, False, "%d error(s), first: %s" % (len(errors), errors[0]))
        if attempt_device == "cpu":
            break

    head("RESULT")
    if FAILS:
        print("   %d check(s) failed:" % len(FAILS))
        for f in FAILS:
            print("     - " + f)
        return 1
    print("   soak check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
