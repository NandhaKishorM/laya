"""Latency of a real Laya call on this machine: CPU (Xeon) vs MPS (AMD Radeon), plus CPU thread scaling.

Run:  python verify/bench_devices.py
"""
import os
import statistics
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # repository root; <root>/laya/email.py must not shadow stdlib email

import torch  # noqa: E402

import laya  # noqa: E402

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}
STATE = {"body": "I was charged twice for invoice 4411. Please refund the duplicate today or we cancel."}
REPEAT = 10


def bench(agent, repeat=REPEAT):
    agent.predict(STATE, QUESTIONS)                      # warm-up (MPS compiles kernels here)
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        agent.predict(STATE, QUESTIONS)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), min(samples)


def main():
    print("torch %s | cpu threads %d | mps available %s\n"
          % (torch.__version__, torch.get_num_threads(), torch.backends.mps.is_available()))
    print("%-16s %-6s %10s %10s %12s" % ("checkpoint", "device", "median ms", "min ms", "ms/question"))
    for device in ("cpu", "mps"):
        for name, rel in [("english", "models/laya"), ("multilingual", "models/laya-multilingual"),
                          ("typed-decisions", "models/laya-typed-decisions")]:
            agent = laya.load(os.path.join(ROOT, rel), device=device)
            med, lo = bench(agent)
            print("%-16s %-6s %10.0f %10.0f %12.0f"
                  % (name, agent.device, med, lo, med / len(QUESTIONS)))
            del agent
            sys.stdout.flush()

    # CPU thread scaling on the heaviest checkpoint
    print("\nCPU thread scaling (english, median of %d)" % REPEAT)
    agent = laya.load(os.path.join(ROOT, "models/laya"), device="cpu")
    for threads in (4, 8, 16, 28, 56):
        torch.set_num_threads(threads)
        med, lo = bench(agent, repeat=5)
        print("   threads=%-3d median %6.0f ms  min %6.0f ms" % (threads, med, lo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
