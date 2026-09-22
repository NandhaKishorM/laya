"""Cold-load cost of a Laya checkpoint, measured as `Agent()` constructions after a warmup.

This is the number the README's "Production Preload & Memory" section talks about: with the
Router's default `max_loaded=1`, alternating languages rebuilds an Agent on every request.

Run:  python3 research/scripts/bench_load_latency.py
"""
import os, sys, time, gc, statistics, shutil
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_TORCH="1",
                  TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya.agent import Agent
snap = os.path.expanduser("~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982")
for label, sub in (("english",""),("multilingual","multilingual"),("typed-decisions","typed-decisions")):
    work = "/tmp/bench_%s" % (sub or "english")
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(os.path.join(snap, sub) if sub else snap, work)
    Agent(work, device="cpu")                      # warm: imports, module registration, page cache
    ts = []
    for _ in range(3):
        gc.collect(); t = time.perf_counter(); ag = Agent(work, device="cpu")
        ts.append(time.perf_counter() - t); del ag; gc.collect()
    print("  %-18s median %6.2f s   (%s)" % (label, statistics.median(ts),
          ", ".join("%.2f" % x for x in ts)))
