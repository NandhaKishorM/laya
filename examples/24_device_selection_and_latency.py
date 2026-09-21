"""Example 24 -- device selection, fp32 off CUDA, and device latency.

Laya picks CUDA, then MPS, then CPU when `device=None`. This example times the multilingual
checkpoint on CPU and on MPS (if present), prints median ms and ms/question, and checks that the
two devices return the same answers.
"""
import json

import torch

from _common import STATE_HI, banner, describe, load, timed

banner("24", "Device selection and latency", """
    Device resolution in `Agent.__init__`: an explicit `device=` wins (with a warning and a
    fallback to CPU if it is unavailable), otherwise CUDA, then MPS, then CPU. Precision follows
    the device -- mixed precision (bf16/fp16) is a CUDA win, while CPU and MPS run **fp32**;
    `torch.autocast` has no MPS backend here, so Laya only wraps the forward pass on CUDA.

    For explicit control pass `device="cpu"`, `"mps"` or `"cuda"`. Laya does not read an env var
    itself, but a common application pattern is to read one (`LAYA_DEVICE`) and pass it through.

    `timed(fn, repeat=3)` discards one warm-up call, so the first MPS call's ~13 s of Metal
    kernel compilation does not pollute the median. Both legs answer the same question set.
    """)

device_line_torch = "   torch %s   cuda_available=%s   mps_available=%s"
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(device_line_torch % (torch.__version__, torch.cuda.is_available(), mps_ok))
print("   explicit selection example: device = os.environ.get('LAYA_DEVICE') or None")

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "account": "login, seats, profile changes",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}
N_QUESTIONS = len(QUESTIONS)


def measure(device):
    agent = load("multilingual", device=device)
    result, median_ms = timed(lambda: agent.predict(STATE_HI, QUESTIONS), repeat=3)
    print("   %-4s -> loaded on %-4s dtype=%-8s median %7.1f ms/call  %.1f ms/question"
          % (device, agent.device, agent.dtype, median_ms, median_ms / N_QUESTIONS))
    return agent, result, median_ms


print("\n   == CPU ==")
cpu_agent, cpu_result, cpu_ms = measure("cpu")

print("\n   == MPS ==")
if mps_ok:
    mps_agent, mps_result, mps_ms = measure("mps")
else:
    print("   MPS not available on this machine; skipping the MPS leg cleanly.")
    mps_agent = mps_result = mps_ms = None

if mps_result is not None:
    print("\n   speed-up: CPU %.1f ms -> MPS %.1f ms  (%.2fx)"
          % (cpu_ms, mps_ms, cpu_ms / mps_ms))
    cpu_json = json.dumps(cpu_result["answers"], sort_keys=True)
    mps_json = json.dumps(mps_result["answers"], sort_keys=True)
    print("   answers agree after JSON serialisation: %s" % (cpu_json == mps_json))
    if cpu_json != mps_json:
        for qid in QUESTIONS:
            a, b = cpu_result["answers"][qid], mps_result["answers"][qid]
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                print("     %s: cpu=%s mps=%s" % (qid, a, b))
    print("   cpu answers : %s" % cpu_json[:100])
    print("   mps answers : %s" % mps_json[:100])

print("\n   device line and one answer from the CPU leg:")
describe(cpu_result["answers"])
print("   both legs used fp32 (device.type in ('cpu','mps') -> torch.float32);")
print("   mixed precision would only be enabled on CUDA.")
