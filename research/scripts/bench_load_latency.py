"""Cold-load cost of a Laya checkpoint, measured as `Agent()` constructions after a warmup.

This is the number the README's "Production Preload & Memory" section talks about: with the
Router's default `max_loaded=1`, alternating languages rebuilds an Agent on every request.

Each checkpoint is copied to a scratch directory first so the timed runs never touch the hub, and
a warmup construction absorbs the one-time import/config costs before the timed runs.

Run:  python3 research/scripts/bench_load_latency.py [reps] [model_root]

`model_root` defaults to the cached hub snapshot, which is where `laya.load()` puts it. It must
contain `model.safetensors` (English at the root) plus the `multilingual/` and
`typed-decisions/` subfolders, the same layout tests/test_local_e2e.py expects.
"""
import gc
import os
import shutil
import statistics
import sys
import time

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_TORCH="1",
                  TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from laya.agent import Agent  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CHECKPOINTS = (("english", None), ("multilingual", "multilingual"),
               ("typed-decisions", "typed-decisions"))
SCRATCH = "/tmp/laya_bench"


def cached_snapshot() -> str:
    """Newest cached snapshot of the bundled repo, avoiding a network round trip.

    `snapshot_download` is deliberately not used: this script sets offline mode, and a partial
    snapshot (the hub client only fetches the files a checkpoint needs) makes it raise
    IncompleteSnapshotError even though every file we are about to load is present.
    """
    root = os.path.expanduser("~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots")
    candidates = [os.path.join(root, d) for d in sorted(os.listdir(root))]
    candidates = [c for c in candidates if os.path.exists(os.path.join(c, "model.safetensors"))]
    if not candidates:
        raise SystemExit("no cached laya snapshot under %s; run `laya.load('convaiinnovations/laya')` "
                         "first or pass an explicit model_root" % root)
    return candidates[-1]


def main():
    snap = sys.argv[2] if len(sys.argv) > 2 else cached_snapshot()
    print("cold Agent() construction, %d reps each, CPU, from %s" % (REPS, snap))
    for label, sub in CHECKPOINTS:
        work = os.path.join(SCRATCH, sub or "english")
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(os.path.join(snap, sub) if sub else snap, work)
        Agent(work, device="cpu")            # warm: imports, module registration, page cache
        timings = []
        for _ in range(REPS):
            gc.collect()
            started = time.perf_counter()
            agent = Agent(work, device="cpu")
            timings.append(time.perf_counter() - started)
            del agent
            gc.collect()
        print("  %-18s median %6.2f s   (%s)"
              % (label, statistics.median(timings), ", ".join("%.2f" % t for t in timings)))


if __name__ == "__main__":
    main()
