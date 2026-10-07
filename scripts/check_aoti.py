"""Offline AOTInductor packaging/parity check and CUDA forward benchmark.

Run with --model pointing at a local Laya checkpoint. Without it, use a tiny
random ModernBERT (no downloads). Outputs and compiler caches belong outside git.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="Local checkpoint directory; never downloads weights")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expect-export-failure", action="store_true", help="Measure the unpatched baseline")
    ap.add_argument("--compare-eager", help="Baseline eager.pt to check bit identity")
    args = ap.parse_args()
    try:
        import torch
        from torch.export import Dim
        from torch._inductor import aoti_compile_and_package, aoti_load_package
    except ImportError:
        print("SKIP: torch.export / AOTInductor unavailable")
        return
    if not torch.cuda.is_available() or not shutil.which("g++"):
        print("SKIP: CUDA and a C++ compiler are required")
        return
    from laya.common import DecisionModel, QTYPES, build_sequence

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    report = {"torch": torch.__version__, "model": args.model or "tiny random ModernBERT",
              "gpu": torch.cuda.get_device_name(), "cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
              "gpu_state_before": subprocess.check_output(["nvidia-smi"], text=True)}
    if args.model:
        if not Path(args.model).is_dir():
            ap.error("--model must be an existing local checkpoint directory")
        from laya.agent import Agent
        agent = Agent(args.model, device="cuda", compile=False)
        model = agent.model.eval()
    else:
        from transformers import ModernBertConfig, ModernBertModel
        cfg = ModernBertConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=512,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
                               reference_compile=False, attn_implementation="sdpa")
        model = DecisionModel(ModernBertModel(cfg), head_layers=1, dropout=0).cuda().eval()

    cases = []
    for rows, tokens, markers in ((2, 128, 3), (1, 256, 3), (4, 160, 5)):
        ids = torch.zeros((rows, tokens), dtype=torch.long)
        att = torch.zeros_like(ids)
        pos = torch.zeros((rows, markers), dtype=torch.long)
        mask = torch.zeros_like(pos, dtype=torch.bool)
        qt = torch.arange(rows) % 3
        for row in range(rows):
            count = markers if row == 0 else max(1, markers - row)
            if args.model:
                kind = ("choice", "score", "noul")[row % 3]
                question = {"type": kind, "instructions": "Which team should handle this request?"}
                if kind == "choice":
                    question["criteria"] = dict(list({"billing": "refunds", "tech": "bugs", "sales": "pricing",
                                                      "account": "login", "support": "other"}.items())[:count])
                elif kind == "score":
                    question["criteria"] = ["low", "medium", "high"][:count]
                else:
                    question["instructions"] = "Is the customer asking for a refund?"
                qq = agent._to_internal(question)
                seq, mp = build_sequence(agent.tok, {"body": "We were billed twice. Please refund the duplicate charge."},
                                         qq, tokens, min(tokens, agent.cfg["head_max_len"]))
                qt[row] = QTYPES[qq["t"]]
                ids[row].fill_(agent.tok.pad_token_id or 0)
            else:
                seq = torch.randint(1, 128, (tokens - 16 * row,)).tolist()
                mp = list(range(1, count + 1))
            assert len(seq) <= tokens and len(mp) <= markers
            ids[row, :len(seq)] = torch.tensor(seq)
            att[row, :len(seq)] = 1
            pos[row, :len(mp)] = torch.tensor(mp)
            mask[row, :len(mp)] = True
        cases.append(tuple(t.cuda() for t in (ids, att, pos, mask, qt)))
    torch.save([[t.cpu() for t in case] for case in cases], outdir / "inputs.pt")

    def latency(fn, inputs):
        for _ in range(10):
            fn(*inputs)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(30):
                fn(*inputs)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 30)
        return round(statistics.median(samples), 4)

    def delta(ref, got):
        result = {}
        for name, a, b in zip(("decision", "action"), ref, got):
            assert torch.isfinite(b).all()
            result[name + "_logits"] = (a.float() - b.float()).abs().max().item()
            result[name + "_probabilities"] = (a.float().softmax(-1) - b.float().softmax(-1)).abs().max().item()
            result[name + "_answers_equal"] = bool(torch.equal(a.argmax(-1), b.argmax(-1)))
        return result

    with torch.no_grad():
        # Pin the existing fp32-parameter path, both without AMP and with fp16/bf16 AMP.
        eager = {}
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                eager[str(dtype)] = [[x.cpu() for x in model(*case)] for case in cases]
        torch.save(eager, outdir / "eager.pt")
        if args.compare_eager:
            baseline = torch.load(args.compare_eager, weights_only=True)
            for key in eager:
                for old, new in zip(baseline[key], eager[key]):
                    for a, b in zip(old, new):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
            report["existing_eager_max_delta"] = 0.0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            report["amp_eager_ms"] = [latency(model, case) for case in cases]
            compiled = torch.compile(model, dynamic=True)
            report["amp_compiled_ms"] = [latency(compiled, case) for case in cases]
        print(json.dumps(report), flush=True)
        del compiled
        candidate = copy.deepcopy(model).to(torch.bfloat16)
        rows, markers = Dim("rows", min=1, max=8), Dim("markers", min=2, max=8)
        tokens = 16 * Dim("tokens16", min=2, max=32)
        dynamic = ({0: rows, 1: tokens}, {0: rows, 1: tokens}, {0: rows, 1: markers},
                   {0: rows, 1: markers}, {0: rows})
        start = time.perf_counter()
        stage = "export"
        try:
            exported = torch.export.export(candidate, cases[0], dynamic_shapes=dynamic, strict=False)
            report["export_s"] = time.perf_counter() - start
            print("Export complete; packaging", flush=True)
            stage = "package"
            start = time.perf_counter()
            path = aoti_compile_and_package(exported, package_path=str(outdir / "decision.pt2"))
            report["package_s"] = time.perf_counter() - start
            report["package_bytes"] = Path(path).stat().st_size
        except Exception as exc:
            if not args.expect_export_failure or not any(s in str(exc) for s in ("same dtype", "dtype mismatch")):
                raise
            report.update(failure_stage=stage, failure_s=time.perf_counter() - start, error=str(exc),
                          package_s=None, package_bytes=None, aoti_ms=None)
        else:
            assert not args.expect_export_failure, "Expected the baseline dtype failure"
            start = time.perf_counter()
            runner = aoti_load_package(path)
            report["load_s"] = time.perf_counter() - start
            report["cases"] = []
            for i, case in enumerate(cases):
                reference = candidate(*case)
                actual = runner(*case)
                diff = delta(reference, actual)
                amp_diff = delta([x.cuda() for x in eager["torch.bfloat16"][i]], actual)
                for comparison in (diff, amp_diff):
                    for name in ("decision", "action"):
                        assert comparison[name + "_probabilities"] < 0.02, comparison
                        assert comparison[name + "_answers_equal"], comparison
                item = {"shape": [*case[0].shape, case[2].shape[1]], "aoti_vs_bf16_eager": diff,
                        "aoti_vs_amp_eager": amp_diff,
                        "bf16_eager_ms": latency(candidate, case), "aoti_ms": latency(runner, case)}
                report["cases"].append(item)
            compiled = torch.compile(candidate, dynamic=True)
            for item, case in zip(report["cases"], cases):
                item["bf16_compiled_ms"] = latency(compiled, case)
                item["compiled_vs_bf16_eager"] = delta(candidate(*case), compiled(*case))
    report["gpu_state_after"] = subprocess.check_output(["nvidia-smi"], text=True)
    (outdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("PASS: expected baseline failure" if args.expect_export_failure else "PASS: AOTI package, parity and latency")


if __name__ == "__main__":
    main()
