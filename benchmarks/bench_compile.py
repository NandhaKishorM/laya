"""Graphs built and per-call cost under `compile=True`: this tree vs the stock `torch.compile(model)`.

    python benchmarks/bench_compile.py [--subfolder multilingual] [--device cuda] [--stock]
    python benchmarks/bench_compile.py --device cpu --dynamo-backend eager     # graph count only, no inductor

Runs a fixed sequence of `predict()` calls whose questions, options and state length change on
every call, the way real traffic does, and prints each call's wall time and how many graphs dynamo
has built so far. A call that builds a graph is a (re)compile: tens of seconds on a GPU with
inductor. `--stock` compiles the way `compile=True` did before (`torch.compile(model)`, duck sizing
on); `--dynamo-backend eager` swaps inductor out so the graph count is cheap to check on CPU.
"""
import argparse, os, sys, time
from contextlib import nullcontext
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (questions, options per question, state words): rows, markers and tokens all move, and the first
# call has rows == markers, the case duck sizing ties together
CALLS = [(4, 4, 20), (4, 4, 60), (3, 4, 30), (5, 2, 80), (2, 3, 40), (8, 5, 120), (6, 6, 50),
         (1, 3, 20), (7, 3, 200), (4, 4, 90)]
WORDS = "the customer reports a duplicate charge on the March invoice and asks for a refund today".split()


def questions(n, k):
    return {"q%d" % i: {"type": "choice", "instructions": "Which team should handle this (%d)?" % i,
                        "criteria": {"team%d" % j: "handles case %d" % j for j in range(k)}} for i in range(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya"); ap.add_argument("--subfolder", default=None)
    ap.add_argument("--device", default=None); ap.add_argument("--stock", action="store_true")
    ap.add_argument("--dynamo-backend", default=None, help="torch.compile backend (default inductor)")
    args = ap.parse_args()
    import torch
    import laya.agent as A
    from laya import _compile
    from torch._dynamo.utils import counters
    kw = {"backend": args.dynamo_backend} if args.dynamo_backend else {}
    if args.stock:
        A.compile_model = lambda m: torch.compile(m, **kw)
        A.independent_dims = nullcontext
    elif kw:
        A.compile_model = lambda m: _compile.compile_model(m, **kw)
    agent = A.Agent(args.model, subfolder=args.subfolder, device=args.device, compile=True)
    print("%s compile, device %s, torch %s" % ("stock" if args.stock else "this tree", agent.device, torch.__version__))
    total = time.perf_counter()
    for n, k, words in CALLS:
        state = " ".join(WORDS[i % len(WORDS)] for i in range(words))
        t = time.perf_counter()
        agent.predict(state, questions(n, k))
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
        print("  %d questions x %d options, %3d words: %8.2f s, graphs so far %d"
              % (n, k, words, time.perf_counter() - t, counters["stats"]["unique_graphs"]))
    print("total %.1f s, %d graphs" % (time.perf_counter() - total, counters["stats"]["unique_graphs"]))


if __name__ == "__main__":
    main()
