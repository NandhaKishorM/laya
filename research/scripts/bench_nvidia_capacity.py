"""Capacity of one NVIDIA GPU serving `laya` under a p99 SLO: dynamic-batch HTTP server + open-loop load.

The laya-only part of the independent study in BENCHMARKS.md ("Independent NVIDIA CUDA capacity study"),
condensed from https://github.com/bhushankinge/laya-cuda-bench at 6cf4148 (harness/server.py, loadgen.py,
sweep.py, export_onnx.py). Result: research/results/nvidia_capacity_20260925.json.

Workload: the frozen SAM.gov mix and the 3-question bundle from that repo's fixtures/workload/ at 6cf4148
(states.jsonl, questions.json, diurnal.csv). Every request = one notice + the same three questions.

  USE_TF=0 python research/scripts/bench_nvidia_capacity.py sweep --model-dir ~/laya_models/laya \
      --workload fixtures/workload --backend eager-fp16 --max-batch 128 --start-qps 50 --out sweep.json
  USE_TF=0 python research/scripts/bench_nvidia_capacity.py export --model-dir ~/laya_models/laya
  LAYA_TRT_MAX_BATCH=256 USE_TF=0 python research/scripts/bench_nvidia_capacity.py sweep ... --backend trt-fp16
  USE_TF=0 python research/scripts/bench_nvidia_capacity.py replay ... --curve fixtures/workload/diurnal.csv \
      --total-decisions 10000000 --compress 24 --out day.jsonl

Backends: `eager-fp16` (torch autocast FP16) and `trt-fp16`. TensorRT is not a laya backend: `trt-fp16` runs
ONNX Runtime's TensorRT execution provider (onnxruntime-gpu + tensorrt, installed separately) over an FP16
ONNX export made by `export`, with LayerNorm kept in FP32.

Sweep: geometric steps from --start-qps (Poisson arrivals, --warmup s excluded, then --duration s measured),
up while the server keeps up, down first if the first rate is already saturated. Saturated = errors, p99 > 4x
the loosest SLO, achieved < 80% of target, or latency trend > 2. It then extends downward until the tightest
SLO is met, and adds one step per SLO at the geometric mean of the best passing and lowest failing rate.
Sustained = achieved >= 90% of target, p99 <= SLO and latency trend <= 1.5; decisions/s = 3 x requests/s.
Latency = completion minus *scheduled* send time, so client lag counts against the server.
"""
import argparse
import asyncio
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# "eager" means stock aten kernels: torch 2.14 otherwise routes some eager ops to Triton via torch._native
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

INPUTS = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]


# ---- workload ---------------------------------------------------------------------------------------------

def load_workload(d):
    """Notices at most 472 `laya` tokens (749 of the 1,000), plus the 3-question bundle."""
    d = Path(d)
    states = [json.loads(l) for l in (d / "states.jsonl").read_text().splitlines() if l.strip()]
    return [s["text"] for s in states if s["tokens"]["laya"] <= 472], json.loads((d / "questions.json").read_text())


# ---- server -----------------------------------------------------------------------------------------------

def make_forward(agent, backend, model_dir):
    """fn(batch of host tensors) -> (logits[n,k], act_probs[n,2]) as float32 numpy."""
    import torch
    if backend == "eager-fp16":
        agent.model.to(agent.device)

        @torch.inference_mode()
        def fn(b):
            b = {k: v.to(agent.device, non_blocking=True) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, act = agent.model(*(b[k] for k in INPUTS))
            return logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()
        return fn
    if backend == "trt-fp16":
        import ctypes
        import importlib.util

        import onnxruntime as ort
        agent.model.to("cpu")  # ORT holds its own weights
        libdir = Path(importlib.util.find_spec("tensorrt_libs").origin).parent  # pip TensorRT is not on the loader path
        for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            ctypes.CDLL(str(libdir / lib), mode=ctypes.RTLD_GLOBAL)
        mb, L = int(os.environ.get("LAYA_TRT_MAX_BATCH", 256)), agent.cfg.get("max_len", 512)
        shp = lambda b, l, k: f"input_ids:{b}x{l},attention_mask:{b}x{l},marker_pos:{b}x{k},marker_mask:{b}x{k},qtype:{b}"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts = {"device_id": 0, "trt_fp16_enable": True, "trt_layer_norm_fp32_fallback": True,
                "trt_engine_cache_enable": True, "trt_engine_cache_path": str(Path(model_dir) / "trt_cache"),
                "trt_timing_cache_enable": True, "trt_profile_min_shapes": shp(1, 8, 2),
                "trt_profile_opt_shapes": shp(16, 256, 4), "trt_profile_max_shapes": shp(mb, L, 20)}
        sess = ort.InferenceSession(str(Path(model_dir) / "onnx" / "laya-fp16.onnx"), so,
                                    providers=[("TensorrtExecutionProvider", opts), ("CUDAExecutionProvider", {"device_id": 0})])
        if sess.get_providers()[0] != "TensorrtExecutionProvider":  # ORT falls back silently; never mislabel a row
            raise RuntimeError(f"TensorRT EP unavailable, session runs on {sess.get_providers()}")

        def fn(b):
            logits, act = sess.run(None, {k: b[k].numpy() for k in INPUTS})
            return np.asarray(logits, np.float32), np.asarray(act, np.float32)
        return fn
    raise ValueError(backend)


def forward_requests(agent, fn, reqs):
    """Every (state, question) pair of every request becomes one row, built by upstream build_sequence;
    calibration and answer schema mirror Agent.predict, so only the forward pass differs."""
    from laya.common import (
        QTYPES,
        build_sequence,
        collate_items,
        confidence_from_probs,
        render_options,
        temp_bucket,
    )
    max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    items, meta = [], []
    for r, (state, questions) in enumerate(reqs):
        for qid, qdef in questions.items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            meta.append((r, qid, q, len(markers), QTYPES[q["t"]]))
    b = collate_items([items], agent.tok.pad_token_id)
    logits, act = fn({k: b[k] for k in INPUTS})
    answers, tokens = [dict() for _ in reqs], [0] * len(reqs)
    for i, ((r, qid, q, k, qt), t) in enumerate(zip(meta, b["attention_mask"].sum(1).tolist())):
        z = np.asarray(logits[i, :k], np.float64) / max(1e-3, float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])))
        p = np.exp(z - z.max())
        p /= p.sum()
        conf, ext = round(confidence_from_probs(p, k), 4), {"act_probability": round(float(act[i, 0]), 4)}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            a = {"type": "choice", "choice": keys[int(p.argmax())], "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                 "confidence": conf, "action": ext}
        elif q["t"] == "score":
            a = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                 "legend": {str(j): c for j, c in enumerate(q["crit"])},
                 "probabilities": {str(j): round(float(v), 4) for j, v in enumerate(p)}, "confidence": conf, "action": ext}
        else:
            a = {"type": "noul", "noul": round(float(p[1]), 4), "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
        answers[r][qid] = a
        tokens[r] += int(t)
    return answers, tokens


def serve(args):
    """Triton-style dynamic batcher: take the first waiting request, collect until max_batch rows or
    max_delay, run one forward on a single worker thread, answer everyone."""
    from aiohttp import web

    from laya.agent import Agent

    agent = Agent(args.model_dir, device="cuda")
    fn = make_forward(agent, args.backend, args.model_dir)
    states, questions = load_workload(args.workload)
    for n in (1, min(16, args.max_batch), args.max_batch):  # compile / engine build before accepting traffic
        forward_requests(agent, fn, [(states[i % len(states)], questions) for i in range(max(1, n // 3))])
    q, pool = asyncio.Queue(), ThreadPoolExecutor(max_workers=1)

    async def batcher():
        loop = asyncio.get_running_loop()
        while True:
            pending = [await q.get()]
            rows, deadline = len(pending[0][0][1]), loop.time() + args.max_delay_ms / 1e3
            while rows < args.max_batch and (timeout := deadline - loop.time()) > 0:
                try:
                    pending.append(await asyncio.wait_for(q.get(), timeout))
                except asyncio.TimeoutError:
                    break
                rows += len(pending[-1][0][1])
            try:
                answers, toks = await loop.run_in_executor(pool, forward_requests, agent, fn, [p[0] for p in pending])
                for (_, fut), a, t in zip(pending, answers, toks):
                    if not fut.done():
                        fut.set_result({"answers": a, "usage": {"input_tokens": t}, "served_batch": rows})
            except Exception as e:
                for _, fut in pending:
                    if not fut.done():
                        fut.set_exception(e)

    async def predict(request):
        body = await request.json()
        if not isinstance(body.get("questions"), dict):
            raise web.HTTPBadRequest(text="need {state, questions}")
        fut = asyncio.get_running_loop().create_future()
        await q.put(((body.get("state", ""), body["questions"]), fut))
        try:
            return web.json_response(await fut)
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))

    async def healthz(_):
        return web.json_response({"ok": True})

    async def start(app):
        app["task"] = asyncio.create_task(batcher())

    app = web.Application(client_max_size=4 * 2**20)
    app.on_startup.append(start)
    app.router.add_post("/predict", predict)
    app.router.add_get("/healthz", healthz)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


def start_server(args, timeout=900):
    cmd = [sys.executable, __file__, "serve", "--model-dir", args.model_dir, "--workload", args.workload,
           "--backend", args.backend, "--max-batch", str(args.max_batch), "--max-delay-ms", str(args.max_delay_ms), "--port", str(args.port)]
    proc, t0 = subprocess.Popen(cmd), time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("server exited early")
            time.sleep(1)
    proc.kill()
    raise TimeoutError("server did not become healthy")


# ---- open-loop load ---------------------------------------------------------------------------------------

def schedule_poisson(qps, duration, rng):
    t, out = 0.0, []
    while t < duration:
        t += rng.expovariate(qps)
        out.append(t)
    return out


def schedule_curve(curve_csv, total_decisions, compress, rng):
    """curve_csv: hour,weight (24 rows); total_decisions is the day's volume. Each hour is replayed for
    3600/compress s at that hour's real arrival rate, so the run walks the day's shape without inflating load."""
    weights = [float(l.split(",")[1]) for l in Path(curve_csv).read_text().splitlines()[1:] if l.strip()]
    total_req, hour_s, out = total_decisions / 3 / compress, 3600.0 / compress, []
    for h, w in enumerate(weights):
        qps, t = total_req * w / sum(weights) / hour_s, h * hour_s
        while t < (h + 1) * hour_s:
            t += rng.expovariate(qps)
            out.append(t)
    return out


def summarize(rows):
    ok = [r for r in rows if r["status"] == 200]
    if not ok:
        return {"requests": len(rows), "ok": 0}
    lat = sorted(r["latency_ms"] for r in ok)
    span = max(r["t_done"] for r in ok) - min(r["t_sched"] for r in ok)
    pct = lambda p: lat[min(len(lat) - 1, int(math.ceil(p * len(lat))) - 1)]
    by_time, med = sorted(ok, key=lambda r: r["t_sched"]), lambda rs: sorted(r["latency_ms"] for r in rs)[len(rs) // 2]
    third = max(1, len(by_time) // 3)
    return {"requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok), "achieved_qps": len(ok) / span,
            "decisions_per_s": sum(r["n_decisions"] for r in ok) / span, "p50_ms": pct(0.5), "p95_ms": pct(0.95),
            "p99_ms": pct(0.99), "max_ms": lat[-1], "mean_served_batch": sum(r["served_batch"] for r in ok) / len(ok),
            # queue growth inside a step: the last third is slower than the first
            "latency_trend": med(by_time[-third:]) / max(1e-9, med(by_time[:third]))}


async def run_load(url, states, questions, times, warmup_s, seed, out_path=None):
    import aiohttp
    rng, rows = random.Random(seed), []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=30)) as sess:
        start = time.perf_counter()

        async def one(t_sched):
            body, now = {"state": rng.choice(states), "questions": questions}, time.perf_counter() - start
            try:
                async with sess.post(url, json=body) as resp:
                    js, status = (await resp.json() if resp.status == 200 else {}), resp.status
            except Exception:
                js, status = {}, 599
            done = time.perf_counter() - start
            rows.append({"t_sched": t_sched, "t_send": now, "t_done": done, "latency_ms": (done - t_sched) * 1e3,
                         "status": status, "served_batch": js.get("served_batch", 0), "n_decisions": len(js.get("answers", {}))})

        tasks = []
        for t in times:  # open loop: send on schedule regardless of responses, so queueing shows up as latency
            if (delay := t - (time.perf_counter() - start)) > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(t)))
        await asyncio.gather(*tasks)
    if out_path:
        Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return summarize([r for r in rows if r["t_sched"] >= warmup_s])


# ---- commands ---------------------------------------------------------------------------------------------

def meets(s, slo):
    return bool(s.get("ok")) and s["p99_ms"] <= slo and s["latency_trend"] <= 1.5 and s["achieved_qps"] >= 0.9 * s["target_qps"]


def sweep(args):
    states, questions = load_workload(args.workload)
    url, steps, qps = f"http://127.0.0.1:{args.port}/predict", [], args.start_qps
    proc = start_server(args)
    try:
        def step(qps):
            times = schedule_poisson(qps, args.duration + args.warmup, random.Random(len(steps)))
            s = asyncio.run(run_load(url, states, questions, times, args.warmup, len(steps)))
            s["target_qps"] = qps
            s["saturated"] = s.get("ok", 0) == 0 or s.get("errors", 0) > 0 or s.get("p99_ms", 1e9) > 4 * max(args.slo_ms) \
                or s.get("achieved_qps", 0) < 0.8 * qps or s.get("latency_trend", 9) > 2.0
            steps.append(s)
            print(f"target {qps:7.1f} q/s -> achieved {s.get('achieved_qps', 0):7.1f}  p50 {s.get('p50_ms', 0):6.1f}  "
                  f"p99 {s.get('p99_ms', 0):7.1f} ms{'  [saturated]' if s['saturated'] else ''}", flush=True)
            return s

        for _ in range(args.max_steps):
            if step(qps)["saturated"]:
                if any(not x["saturated"] for x in steps[:-1]):
                    break
                qps /= args.factor
            else:
                qps *= args.factor
        for _ in range(4):
            if any(meets(s, min(args.slo_ms)) for s in steps):
                break
            step(min(s["target_qps"] for s in steps) / args.factor)
        for slo in args.slo_ms:
            good = [s["target_qps"] for s in steps if meets(s, slo)]
            bad = [s["target_qps"] for s in steps if good and s["target_qps"] > max(good) and not meets(s, slo)]
            if good and bad:
                step((max(good) * min(bad)) ** 0.5)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    sustained = {}
    for slo in args.slo_ms:
        good = [s for s in steps if meets(s, slo)]
        sustained[str(int(slo))] = max(good, key=lambda s: s["target_qps"]) if good else None
        print(f"p99 <= {slo:g} ms: " + (f"{sustained[str(int(slo))]['decisions_per_s']:.0f} decisions/s" if good else "not met"))
    report = {"backend": args.backend, "max_batch": args.max_batch, "max_delay_ms": args.max_delay_ms, "warmup_s": args.warmup,
              "step_s": args.duration, "slo_ms": args.slo_ms, "sustained": sustained,
              "steps": sorted(steps, key=lambda s: s["target_qps"])}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", args.out)


def replay(args):
    states, questions = load_workload(args.workload)
    times = schedule_curve(args.curve, args.total_decisions, args.compress, random.Random(0))
    proc = start_server(args)
    try:
        s = asyncio.run(run_load(f"http://127.0.0.1:{args.port}/predict", states, questions, times, 0, 0, args.out))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    print(json.dumps(s, indent=1))


def export(args):
    """FP32 ONNX (dynamo exporter, opset 18, dynamic batch/sequence/options) and an FP16 copy for TensorRT."""
    import onnx
    import torch
    from onnxruntime.transformers.float16 import convert_float_to_float16

    from laya.agent import Agent

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            logits, act = self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            return logits, torch.softmax(act.float(), -1)

    w = Wrapper(Agent(args.model_dir, device="cpu").model.float().eval())
    out = Path(args.model_dir) / "onnx"
    out.mkdir(exist_ok=True)
    ex = (torch.randint(5, 1000, (2, 40)), torch.ones(2, 40, dtype=torch.long), torch.tensor([[3, 9, 15, 21], [3, 9, 0, 0]]),
          torch.tensor([[True] * 4, [True, True, False, False]]), torch.tensor([0, 2]))
    ex[1][1, 30:] = 0
    batch, seq, opts = torch.export.Dim("batch"), torch.export.Dim("seq", min=8), torch.export.Dim("options", min=2)
    # grad stays enabled: nn.TransformerEncoderLayer's fused fast path (not exportable) is only taken under no_grad
    prog = torch.onnx.export(w, ex, opset_version=18, dynamo=True, optimize=True, input_names=INPUTS, output_names=["logits", "act_probs"],
                             dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq},
                                             "marker_pos": {0: batch, 1: opts}, "marker_mask": {0: batch, 1: opts}, "qtype": {0: batch}})
    prog.save(str(out / "laya-fp32.onnx"), external_data=False)
    onnx.save(convert_float_to_float16(onnx.load(str(out / "laya-fp32.onnx")), keep_io_types=True), str(out / "laya-fp16.onnx"))
    print("wrote", out / "laya-fp16.onnx")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sweep", "replay", "serve", "export"):
        p = sub.add_parser(name)
        p.add_argument("--model-dir", required=True, help="laya checkpoint directory (the English `laya` checkpoint)")
        if name == "export":
            continue
        p.add_argument("--workload", required=True, help="directory with states.jsonl and questions.json")
        p.add_argument("--backend", choices=["eager-fp16", "trt-fp16"], default="eager-fp16")
        p.add_argument("--max-batch", type=int, default=64, help="rows per forward (requests x 3 questions)")
        p.add_argument("--max-delay-ms", type=float, default=2.0)
        p.add_argument("--port", type=int, default=8080)
        p.add_argument("--out")
        if name == "sweep":
            p.add_argument("--start-qps", type=float, default=5)
            p.add_argument("--factor", type=float, default=1.5)
            p.add_argument("--max-steps", type=int, default=20)
            p.add_argument("--warmup", type=float, default=20)
            p.add_argument("--duration", type=float, default=60)
            p.add_argument("--slo-ms", nargs="+", type=float, default=[50, 130])
        if name == "replay":
            p.add_argument("--curve", required=True)
            p.add_argument("--total-decisions", type=float, required=True)
            p.add_argument("--compress", type=float, default=24)
    args = ap.parse_args()
    if args.cmd == "sweep" and not args.out:
        ap.error("sweep needs --out")
    {"sweep": sweep, "replay": replay, "serve": serve, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()
