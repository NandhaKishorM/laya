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
CPU if the device rejects it; a device-level crash (SIGABRT/SIGSEGV from a GPU driver) is
isolated in a short-lived helper process so it becomes a reported FAIL with a CPU retry,
not an aborted run with no verdict.
"""
import argparse
import gc
import json
import os
import statistics
import subprocess
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


def _concurrency_child(models_dir, call_device, ref, n_threads, per_thread):
    """The threading burst from section 3, running in a disposable helper process.

    Same script, same agent loader, same threaded `predict` storm, same single-threaded
    reference as the in-process loop it replaces; only the outcome travels back (one JSON
    verdict line on stdout). A device-level crash (SIGABRT/SIGSEGV from a GPU driver)
    then kills this helper, not the process owed the leak verdicts -- the parent sees a
    bad exit code and runs the same CPU retry the script always had.
    """
    agent = _concurrency_load(models_dir, call_device)
    here_ref = json.dumps(agent.predict(STATE, QUESTIONS)["answers"], sort_keys=True)
    if here_ref != ref:
        print(json.dumps({"status": "reference-mismatch",
                          "detail": "child reference differs from the parent one"}), flush=True)
        return 3

    def worker(results, errors):
        """Every worker asks the same questions about the same state, so any cross-talk or
        corruption shows up as an answer that differs from the single-threaded reference.
        The agent is closed over because there is exactly one burst in exactly one process,
        and no retry can happen underneath running threads."""
        try:
            for _ in range(per_thread):
                out = agent.predict(STATE, QUESTIONS)
                results.append(json.dumps(out["answers"], sort_keys=True))
        except Exception as e:  # noqa: BLE001
            errors.append("%s: %s" % (type(e).__name__, str(e)[:120]))

    results, errors = [], []
    threads = [threading.Thread(target=worker, args=(results, errors)) for _ in range(n_threads)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    if errors:
        print(json.dumps({"status": "errors", "errors": errors[:5], "elapsed": elapsed,
                          "done": len(results)}), flush=True)
        return 4
    mismatches = sum(1 for r in results if r != here_ref)
    print(json.dumps({"status": "ok", "total": len(results), "mismatches": mismatches,
                      "elapsed": elapsed}), flush=True)
    return 0 if mismatches == 0 else 5


def _concurrency_load(models_dir, call_device):
    """The multilingual agent section 3 bursts against, loaded one way in both processes."""
    return laya.load(os.path.join(models_dir, "laya-multilingual"), device=call_device)


def _concurrency_shim():
    """`python verify/soak_check.py --concurrency-child ...`: section 3's burst, on its
    own argv. Prints one JSON verdict line on stdout and exits with a code the parent
    reads; argparse learns the extra flag through `parse_known_args`, so `main` is
    untouched."""
    if "--concurrency-child" not in sys.argv:
        return None
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--per-thread", type=int, required=True)
    known, _ = ap.parse_known_args()
    sys.exit(_concurrency_child(known.models_dir, known.device,
                                known.ref, known.threads, known.per_thread))


_CONCURRENCY_CHILD_RC = _concurrency_shim()
if _CONCURRENCY_CHILD_RC is not None:  # pragma: no cover - child entry only
    sys.exit(_CONCURRENCY_CHILD_RC)


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
            sys.exit("missing checkpoint %r -- fetch it with "
                     "python verify/checkpoints.py --fetch" % name)
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
    # ------------------------------------------------------------ 3. concurrency
    # Threads hitting an MPS (or CUDA/XPU) model at once are exactly what SIGABRTed this
    # script's interpreter on Apple silicon (torch 2.14, Metal assertion
    # `_status < MTLCommandBufferStatusCommitted`): the driver aborts below Python, so a
    # worker pool's try/except never sees it and a threaded runner that mimics the parent
    # loop dies with no verdict either. Run the burst in a disposable helper process
    # instead -- same loader, same threads, same reference, one JSON verdict -- whose
    # death the parent converts into the CPU retry the script already promised. Leak
    # sections 1-2 stay in-process: they measure the serving process, not a throwaway
    # helper.
    head("3. Concurrent calls from %d threads" % args.threads)
    per_thread = max(1, args.calls // (args.threads * 4))
    total = args.threads * per_thread

    def _report_verdict(verdict, device_name, expected_total):
        status = verdict["status"]
        if status == "reference-mismatch":
            report("concurrency on %s" % device_name, False, verdict["detail"])
            return device_name == "cpu"
        if status == "errors":
            errs = verdict["errors"] or ["no detail"]
            report("concurrency on %s" % device_name, False,
                   "%d error(s), first: %s" % (len(errs), errs[0]))
            return device_name == "cpu"
        done = verdict.get("total", 0)
        report("all %d concurrent calls returned (%s)" % (expected_total, device_name),
               done == expected_total,
               "expected %d, got %d" % (expected_total, done))
        mismatches = verdict.get("mismatches", 1)
        report("concurrent answers match the single-threaded reference",
               mismatches == 0,
               "%d of %d differ" % (mismatches, done))
        elapsed = verdict.get("elapsed") or 0.0
        print("   %d calls in %.1f s across %d threads (%.0f calls/s)"
              % (done, elapsed, args.threads, done / elapsed if elapsed else 0.0))
        return True

    def _last_verdict(lines):
        """The child's one JSON line, read back-to-front past warnings and progress."""
        for line in reversed(lines):
            try:
                verdict = json.loads(line)
            except ValueError:
                continue
            if isinstance(verdict, dict) and "status" in verdict:
                return verdict
        return None

    def _child_timeout():
        """Bound the helper's life: 300 s floor, rising with --calls like the work does."""
        return max(300.0, 300.0 * args.calls / 200.0 if args.calls else 0.0)

    def burst_in_child(call_device, expected_total):
        """Run section 3's burst for one device inside the helper and report its verdict.

        Returns True when the parent should stop retrying -- either the burst succeeded,
        or this was the CPU pass itself and there is nothing left to fall back to. The
        burst keeps the original report names, so a passing machine prints exactly what
        the in-process loop used to, including the note when the CPU fallback is the pass
        that won.
        """
        argv = [sys.executable, os.path.join(ROOT, "verify", "soak_check.py"),
                "--concurrency-child", "--models-dir", args.models, "--device", call_device,
                "--ref", ref, "--threads", str(args.threads), "--per-thread", str(per_thread)]
        try:
            child = subprocess.run(argv, capture_output=True, text=True, timeout=_child_timeout())
        except subprocess.TimeoutExpired:
            report("concurrency on %s" % call_device, False,
                   "helper timed out after %ds" % _child_timeout())
            return call_device == "cpu"
        verdict = _last_verdict((child.stdout or "").splitlines())
        if verdict is None or child.returncode != 0:
            report("concurrency on %s" % call_device, False,
                   "helper died without a verdict (exit %s)" % child.returncode)
            return call_device == "cpu"
        return _report_verdict(verdict, call_device, expected_total)

    primary = _concurrency_load(args.models, device)
    ref = json.dumps(primary.predict(STATE, QUESTIONS)["answers"], sort_keys=True)
    attempts = [primary.device.type] + (["cpu"] if primary.device.type != "cpu" else [])
    for attempt_device in attempts:
        if primary.device.type != attempt_device:
            print("   the %s pass failed; retrying on cpu" % primary.device.type)
            primary = _concurrency_load(args.models, "cpu")
        done = burst_in_child(attempt_device, total)
        if done:
            if attempt_device != primary.device.type:
                print("   note: the %s pass is what succeeded, not the default device"
                      % attempt_device)
            break
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
