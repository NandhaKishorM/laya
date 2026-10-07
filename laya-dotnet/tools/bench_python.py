"""Benchmark the Python laya SDK across all three checkpoints.

Measures:
  - Cold latency: model load + first Predict call (ms)
  - Warm latency: median and p95 over N_WARM timed calls (ms), after N_WARMUP warmups
  - Peak RSS/working-set (MB), via psutil if available
  - Answer snapshot for each case, for comparison with the .NET results

Outputs JSON to stdout so the caller can parse and tabulate.

By default this benchmarks the public SDK (`laya.load`, PyTorch eager on CPU). Pass
`--runtime onnx` to run the same SDK code path with the exported ONNX graph as the forward
pass, which separates runtime cost from language cost when comparing with the .NET SDK.

Usage:
    python laya-dotnet/tools/bench_python.py --checkpoint multilingual
    python laya-dotnet/tools/bench_python.py --checkpoint multilingual --runtime onnx
    python laya-dotnet/tools/bench_python.py --checkpoint english
    python laya-dotnet/tools/bench_python.py --checkpoint typed-decisions
    python laya-dotnet/tools/bench_python.py --checkpoint all   # runs all three in sequence
    python laya-dotnet/tools/bench_python.py --checkpoint all --output bench_results.json
"""

import argparse
import json
import os
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, REPO)
sys.path.insert(0, TOOLS)

from sample_cases import SAMPLE_ENGLISH_STATE, SAMPLE_HINDI_STATE, SAMPLE_QUESTIONS

N_WARMUP = 5
N_WARM = 30

# The benchmark cases: a subset of the full golden suite chosen to cover real behaviour
# without taking forever. These are the same cases the task brief specifies.
BENCH_CASES = [
    {
        "name": "sample_app_english",
        "state": SAMPLE_ENGLISH_STATE,
        "questions": SAMPLE_QUESTIONS,
    },
    {
        "name": "sample_app_hindi",
        "state": SAMPLE_HINDI_STATE,
        "questions": SAMPLE_QUESTIONS,
    },
]

ONNX_ROOT = os.path.join(REPO, "onnx")

CHECKPOINTS = ["english", "multilingual", "typed-decisions"]


def _onnx_dir(checkpoint: str) -> str:
    return os.path.join(ONNX_ROOT, checkpoint)


def _load_agent(onnx_dir: str):
    """Load an Agent using the ONNX forward pass, exactly as dump_golden.py does."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoTokenizer

    from laya.agent import Agent, _fix_tokenizer_config
    from laya.common import clamp_temperature

    NP_DTYPES = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
        "tensor(float)": np.float32,
    }

    class OnnxForward:
        MIN_MARKERS = 2

        def __init__(self, session):
            self.sess = session
            self.dtypes = {i.name: NP_DTYPES[i.type] for i in session.get_inputs()}

        def _pad_markers(self, marker_pos, marker_mask):
            k = marker_pos.shape[1]
            if k >= self.MIN_MARKERS:
                return marker_pos, marker_mask
            pad = self.MIN_MARKERS - k
            marker_pos = torch.cat(
                [marker_pos, torch.zeros((marker_pos.shape[0], pad), dtype=marker_pos.dtype)], 1
            )
            marker_mask = torch.cat(
                [marker_mask, torch.zeros((marker_mask.shape[0], pad), dtype=torch.bool)], 1
            )
            return marker_pos, marker_mask

        def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            marker_pos, marker_mask = self._pad_markers(marker_pos, marker_mask)
            feeds = {
                n: t.cpu().numpy().astype(self.dtypes[n])
                for n, t in [
                    ("input_ids", input_ids),
                    ("attention_mask", attention_mask),
                    ("marker_pos", marker_pos),
                    ("marker_mask", marker_mask),
                    ("qtype", qtype),
                ]
            }
            logits, act_logits = self.sess.run(["logits", "act_logits"], feeds)
            return torch.from_numpy(logits), torch.from_numpy(act_logits)

    with open(os.path.join(onnx_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    _fix_tokenizer_config(onnx_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(onnx_dir, "tokenizer"))

    sess_opts = ort.SessionOptions()
    # Single-thread to match default Python behaviour: the ONNX runtime defaults to
    # using all available cores; pin to 1 so the number is reproducible.
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        os.path.join(onnx_dir, "model.onnx"),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    agent = Agent.__new__(Agent)
    agent.cfg = cfg
    agent.tok = tok
    import torch as _torch
    agent.device = _torch.device("cpu")
    agent.dtype = _torch.float32
    agent.model = OnnxForward(sess)
    agent.temperature = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
    agent.temperature_by_options = {
        k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()
    }
    return agent


def _load_sdk_agent(checkpoint: str):
    """Load the checkpoint through the public Python SDK (PyTorch eager, CPU)."""
    import laya

    subfolder = None if checkpoint == "english" else checkpoint
    return laya.load("convaiinnovations/laya", device="cpu", subfolder=subfolder)


def _peak_wset_mb():
    """Return the process peak working set (Windows) or peak RSS (other) in MB."""
    try:
        import psutil
    except ImportError:
        return None
    mi = psutil.Process(os.getpid()).memory_info()
    peak = getattr(mi, "peak_wset", None)
    if peak is None:
        import resource  # POSIX: ru_maxrss is KiB on Linux
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return round(peak / 1024 / 1024, 1)


def bench_checkpoint(checkpoint: str, runtime: str) -> dict:
    """Run the full benchmark for one checkpoint. Returns a result dict."""
    print(f"[bench_python] {checkpoint} ({runtime}): loading...", flush=True)

    onnx_dir = _onnx_dir(checkpoint)
    if runtime == "onnx" and not os.path.isdir(onnx_dir):
        return {"checkpoint": checkpoint, "error": f"ONNX dir not found: {onnx_dir}"}

    # Imports are timed separately so cold latency matches the .NET number
    # (engine creation + first call, excluding runtime start-up).
    t_import = time.perf_counter()
    import torch  # noqa: F401
    import laya  # noqa: F401
    import_ms = (time.perf_counter() - t_import) * 1000.0

    # ── cold latency ─────────────────────────────────────────────────────────
    # Measure load + first predict together as the "cold" number.
    t_cold_start = time.perf_counter()
    agent = _load_sdk_agent(checkpoint) if runtime == "torch" else _load_agent(onnx_dir)

    # Use the first benchmark case for the cold call.
    first_case = BENCH_CASES[0]
    _ = agent.system_one(first_case["state"], first_case["questions"])
    cold_ms = (time.perf_counter() - t_cold_start) * 1000.0

    print(f"[bench_python] {checkpoint}: cold={cold_ms:.0f} ms, running warm-up...", flush=True)

    # ── warm latency ──────────────────────────────────────────────────────────
    # Cycle through all bench cases for variety; this also exercises both states.
    call_idx = 0

    def _next_call():
        nonlocal call_idx
        c = BENCH_CASES[call_idx % len(BENCH_CASES)]
        call_idx += 1
        return c["state"], c["questions"]

    # Warmup
    for _ in range(N_WARMUP):
        s, q = _next_call()
        agent.system_one(s, q)

    # Timed calls
    times_ms = []
    for _ in range(N_WARM):
        s, q = _next_call()
        t0 = time.perf_counter()
        agent.system_one(s, q)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    p95_ms = times_ms[int(len(times_ms) * 0.95)]

    # ── peak memory ───────────────────────────────────────────────────────────
    peak_mb = _peak_wset_mb()

    print(
        f"[bench_python] {checkpoint}: warm median={median_ms:.1f} ms  p95={p95_ms:.1f} ms"
        + (f"  peak={peak_mb:.0f} MB" if peak_mb else ""),
        flush=True,
    )

    # ── answer snapshot ───────────────────────────────────────────────────────
    answers = {}
    for case in BENCH_CASES:
        result = agent.system_one(case["state"], case["questions"])
        answers[case["name"]] = result["answers"]

    import torch

    return {
        "checkpoint": checkpoint,
        "runtime": "Python SDK + PyTorch (CPU)" if runtime == "torch" else "Python + ONNX Runtime",
        "threads": f"torch default ({torch.get_num_threads()})" if runtime == "torch" else "intra=1 inter=1",
        "import_ms": round(import_ms, 1),
        "cold_ms": round(cold_ms, 1),
        "warm_times_ms": [round(t, 2) for t in times_ms],
        "warm_median_ms": round(median_ms, 1),
        "warm_p95_ms": round(p95_ms, 1),
        "n_warmup": N_WARMUP,
        "n_warm": N_WARM,
        "peak_wset_mb": peak_mb,
        "answers": answers,
    }


def get_versions() -> dict:
    info = {}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        info["onnxruntime"] = None
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = None
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except ImportError:
        info["numpy"] = None
    try:
        import laya
        info["laya"] = laya.__version__
    except Exception:
        info["laya"] = None
    info["python"] = sys.version
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        choices=CHECKPOINTS + ["all"],
        default="multilingual",
        metavar="{multilingual,english,typed-decisions,all}",
    )
    ap.add_argument(
        "--runtime",
        choices=["torch", "onnx"],
        default="torch",
        help="torch: the public SDK (laya.load, PyTorch eager). onnx: the SDK code path with "
             "the exported ONNX graph as the forward pass, single-threaded.",
    )
    ap.add_argument("--output", default=None, help="Write JSON results to this file (default: stdout only)")
    args = ap.parse_args()

    checkpoints = CHECKPOINTS if args.checkpoint == "all" else [args.checkpoint]

    results = {"versions": None, "checkpoints": {}}

    # Note: "all" shares one process, so peak memory and cold latency of later checkpoints
    # are inflated. Run one checkpoint per process for publishable numbers.
    for cp in checkpoints:
        results["checkpoints"][cp] = bench_checkpoint(cp, args.runtime)
    # Collected after the runs so the imports it triggers are not charged to import_ms.
    results["versions"] = get_versions()

    out = json.dumps(results, indent=2, ensure_ascii=False)
    print("\n" + out)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\nResults written to {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
