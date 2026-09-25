"""Cold start of the `compile` backend, and an AOTInductor trial.

    python benchmarks/bench_compile.py [--subfolder multilingual] [--cache-dir DIR] [--aoti] [--json out.json]

Measures, in this order:

1. eager `predict()` latency (short: 1 state x 3 questions; long: 30 questions, ~1000 tokens);
2. the compile backend's warm-up at load with an EMPTY inductor cache (a fresh `--cache-dir`), then its
   latency on the same two cases and the cost of the first call at a length the warm-up did not cover;
3. the same warm-up in a SECOND process reading the cache the first one wrote (`--second-process` is the
   flag that process is started with);
4. with `--aoti`: `torch.export` + `torch._inductor.aoti_compile_and_package` of the decision model with
   dynamic rows/tokens, the package size, its load time and its latency, so the "ship a precompiled
   artifact per checkpoint + GPU arch" option is answered with numbers rather than an opinion.

The GPU is shared on the machine this was written on; the JSON records `nvidia-smi` utilisation before the run.
"""
import argparse, json, os, subprocess, sys, tempfile, time
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Q = {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "pricing", "shipping": "delivery"}},
     "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
     "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"}}
SHORT = {"subject": "Duplicate charge on invoice 4411", "body": "We were billed twice for March. Please refund the duplicate or we're moving to a competitor."}
LONG = {"subject": "Outage report", "body": "Since yesterday our whole team cannot log in, the dashboard returns 502 errors and our release is blocked. " * 40}
MID = {"subject": "Thread", "body": "Thanks for the quick turnaround on the invoice issue, the credit note arrived this morning. " * 6}
Q30 = {f"{k}{i}": v for i in range(10) for k, v in Q.items()}


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out
    except Exception:
        return "n/a"


def timeit(fn, iters=30):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1000


def measure(agent):
    import torch
    out = {"short_ms": round(timeit(lambda: agent.predict(SHORT, Q)), 2)}
    # The long case is 32 x 1024 rows x tokens after padding. Under `compile` the expanded attention
    # mask is materialised (rows x heads x L x L bf16, ~0.8 GB), and on a GPU that is already nearly
    # full the caching allocator thrashes to tens of seconds per call; that is not the number the
    # table wants, so one probe call decides whether the timed loop is meaningful.
    t = time.perf_counter(); agent.predict(LONG, Q30); torch.cuda.synchronize(); probe = time.perf_counter() - t
    if probe > 5:
        out["long_ms"] = None; out["long_note"] = "probe call took %.1f s: GPU memory contention, not timed" % probe
    else:
        out["long_ms"] = round(timeit(lambda: agent.predict(LONG, Q30), 10), 1)
    out["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def warmup_run(args):
    """One process: load the compile backend against args.cache_dir and report the warm-up."""
    import torch, laya
    os.environ["LAYA_INDUCTOR_CACHE_DIR"] = args.cache_dir
    t0 = time.perf_counter()
    agent = laya.load(args.model, subfolder=args.subfolder, backend="compile")
    load_s = time.perf_counter() - t0
    be = agent.backend_object
    rep = {"backend": agent.backend, "load_s": round(load_s, 1), "warmup_s": round(be.warmup_s, 1), "cache_dir": be.cache_dir}
    rep.update(measure(agent))
    t = time.perf_counter(); agent.predict(MID, Q); torch.cuda.synchronize()
    rep["first_unseen_length_s"] = round(time.perf_counter() - t, 3)
    return rep


def aoti_trial(agent, out_dir):
    """Export the decision model once with dynamic rows/tokens and package it with AOTInductor."""
    import torch
    from torch.export import Dim
    m = agent.model
    dev = agent.device
    ex = {}
    rows, tokens = 4, 128
    ids = torch.full((rows, tokens), agent.tok.pad_token_id or 0, dtype=torch.long, device=dev); ids[:, 0] = agent.tok.cls_token_id or 0
    att = torch.ones((rows, tokens), dtype=torch.long, device=dev)
    mpos = torch.zeros((rows, 2), dtype=torch.long, device=dev); mmask = torch.ones((rows, 2), dtype=torch.bool, device=dev)
    qt = torch.zeros((rows,), dtype=torch.long, device=dev)
    # tokens as a multiple of 16: that is what the bucket policy feeds every backend, and the
    # attention-mask alignment guard (`L % 8`) rejects a plain range of lengths at export time
    R, K = Dim("rows", min=1, max=64), Dim("markers", min=1, max=64)
    L = 16 * Dim("tokens16", min=1, max=int(agent.cfg.get("max_len", 512)) // 16)
    dyn = ({0: R, 1: L}, {0: R, 1: L}, {0: R, 1: K}, {0: R, 1: K}, {0: R})
    t0 = time.perf_counter()
    try:
        # Exported under autocast, the program carries dtype asserts that AOTI then trips outside it
        # ("Tensor dtype mismatch! Expected: torch.bfloat16, Got: torch.float32"), so the trial exports
        # a copy of the model cast to the agent's dtype instead: what a shipped artifact would be.
        import copy
        m = copy.deepcopy(m).to(agent.dtype) if agent.amp_enabled else m
        with torch.no_grad():
            ep = torch.export.export(m, (ids, att, mpos, mmask, qt), dynamic_shapes=dyn, strict=False)
        ex["export_s"] = round(time.perf_counter() - t0, 1)
        path = os.path.join(out_dir, "laya_decision_model.pt2")
        t0 = time.perf_counter()
        pkg = torch._inductor.aoti_compile_and_package(ep, package_path=path)
        ex["compile_s"] = round(time.perf_counter() - t0, 1)
        ex["package_mb"] = round(os.path.getsize(pkg) / 1e6, 1)
        t0 = time.perf_counter()
        runner = torch._inductor.aoti_load_package(pkg)
        ex["load_s"] = round(time.perf_counter() - t0, 2)
        with torch.no_grad(), torch.autocast("cuda", dtype=agent.dtype, enabled=agent.amp_enabled):
            ref = agent.model(ids, att, mpos, mmask, qt)[0].float()
        with torch.no_grad():
            out = runner(ids, att, mpos, mmask, qt)[0].float()
        ex["max_abs_logit_diff"] = round((ref - out).abs().max().item(), 4)
        # latency through the runner, same shapes as the eager/compile rows: swap it in as the forward
        from laya.backends.base import Backend, pad_batch

        class AOTI(Backend):
            name = "aoti"

            def forward(self, i, a, mp, mm, q, detach_encoder=False):
                i, a, mp, mm, q, n, _ = pad_batch(i, a, mp, mm, q, int(agent.cfg.get("max_len", 512)),
                                                  pad_id=agent.tok.pad_token_id or 0, keep_one_token=True)
                lg, ac = runner(i, a, mp, mm, q)
                return lg[:n], ac[:n]
        agent.set_backend("eager"); b = AOTI(agent); b.install()
        ex.update(measure(agent)); b.uninstall()
    except Exception as e:
        ex["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
    return ex


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="convaiinnovations/laya"); ap.add_argument("--subfolder", default=None)
    ap.add_argument("--cache-dir", default=None); ap.add_argument("--aoti", action="store_true"); ap.add_argument("--json", default=None)
    ap.add_argument("--second-process", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--aoti-only", action="store_true", help="run just the AOTInductor trial")
    args = ap.parse_args()
    if args.second_process:
        print(json.dumps(warmup_run(args))); return
    if args.aoti_only:
        import laya
        agent = laya.load(args.model, subfolder=args.subfolder)
        out = aoti_trial(agent, args.cache_dir or tempfile.mkdtemp(prefix="laya-aoti-"))
        print("aoti", out)
        if args.json:
            doc = json.load(open(args.json)) if os.path.exists(args.json) else {}
            doc["aoti"] = out; json.dump(doc, open(args.json, "w"), indent=1); print("saved", args.json)
        return
    import torch, laya
    args.cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="laya-inductor-")
    report = {"model": args.model, "subfolder": args.subfolder, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "gpu_state_before": gpu_util(), "cache_dir": args.cache_dir}
    agent = laya.load(args.model, subfolder=args.subfolder)
    report["eager"] = measure(agent)
    print("eager", report["eager"], flush=True)
    if args.aoti:
        report["aoti"] = aoti_trial(agent, args.cache_dir); print("aoti", report["aoti"], flush=True)
    del agent; torch.cuda.empty_cache()
    env = dict(os.environ, LAYA_INDUCTOR_CACHE_DIR=args.cache_dir)
    cmd = [sys.executable, os.path.abspath(__file__), "--model", args.model, "--cache-dir", args.cache_dir, "--second-process"]
    if args.subfolder:
        cmd += ["--subfolder", args.subfolder]
    for label in ("compile_cold", "compile_cached"):
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        try:
            report[label] = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            report[label] = {"error": out.stderr[-2000:]}
        print(label, report[label], flush=True)
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1); print("saved", args.json)


if __name__ == "__main__":
    main()
