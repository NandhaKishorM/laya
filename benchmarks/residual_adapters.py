"""Local checkpoint parity, latency and memory check for zero-initialized adapters.

Run from the repository: python benchmarks/residual_adapters.py --model /path/to/checkpoint
This measures integration overhead, not trained decision quality. It downloads nothing.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers

from laya import Agent
from laya.common import DecisionModel


def measure(model, tensors, device, iterations):
    model.eval().to(device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                               enabled=device.type == "cuda"):
        for _ in range(3):
            model(*tensors)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            logits, actions = model(*tensors)
            sync()
            samples.append((time.perf_counter() - start) * 1000)
        result = {"median_ms": statistics.median(samples),
                  "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                  "peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                           if device.type == "cuda" else None)}
        outputs = (logits.float().cpu(), actions.float().cpu())
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=0)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be an existing local checkpoint directory")
    if args.rows < 1 or args.iterations < 1 or args.train_steps < 0 or min(args.lengths) < 8:
        parser.error("rows/iterations must be positive, lengths >= 8 and train-steps >= 0")
    torch.set_num_threads(2)
    torch.manual_seed(41)
    device = torch.device(args.device)
    agent = Agent(str(args.model), device="cpu", compile=False)
    stock = agent.model.eval()
    config = {"version": 1, "layers": list(range(len(stock.encoder.layers))),
              "rank": args.rank, "scale": args.scale}
    candidate = DecisionModel(copy.deepcopy(stock.encoder), head_layers=agent.cfg["head_layers"],
                              n_act=stock.act_head[-1].out_features, residual_adapters=config).eval()
    missing, unexpected = candidate.load_state_dict(stock.state_dict(), strict=False)
    assert missing and all(k.startswith("residual_adapters.") for k in missing) and not unexpected
    report = {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "iterations": args.iterations, "seed": 41, "cpu_threads": torch.get_num_threads(),
              "residual_config": config,
              "device_type": device.type, "autocast": "bfloat16" if device.type == "cuda" else None,
              "added_parameters": sum(p.numel() for p in candidate.residual_adapters.parameters()),
              "stock_parameters": sum(p.numel() for p in stock.parameters()), "measurements": []}
    for length in args.lengths:
        if length > agent.cfg.get("max_len", 512):
            parser.error("a requested length exceeds the checkpoint's configured limit")
        tensors = (torch.randint(4, stock.encoder.config.vocab_size, (args.rows, length), device=device),
                   torch.ones(args.rows, length, dtype=torch.long, device=device),
                   torch.tensor([[2, 4, 6]], device=device).expand(args.rows, -1),
                   torch.ones(args.rows, 3, dtype=torch.bool, device=device),
                   torch.zeros(args.rows, dtype=torch.long, device=device))
        baseline, reference = measure(stock, tensors, device, args.iterations)
        adapted, actual = measure(candidate, tensors, device, args.iterations)
        delta = max((a - b).abs().max().item() for a, b in zip(reference, actual))
        assert delta == 0.0, "zero-init parity failed: %g" % delta
        report["measurements"].append({"rows": args.rows, "length": length, "stock": baseline,
                                       "residual": adapted, "max_output_difference": delta})
    if args.train_steps:
        candidate.to(device).train()
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-5)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.train_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = candidate(*tensors)
                loss = torch.nn.functional.cross_entropy(logits, torch.zeros(args.rows, dtype=torch.long,
                                                                             device=device))
            loss.backward()
            assert torch.isfinite(loss)
            for branch in candidate.residual_adapters.branches.values():
                assert branch.up.weight.grad is not None and torch.isfinite(branch.up.weight.grad).all()
                assert branch.up.weight.grad.abs().sum() > 0
            optimizer.step()
        report["training"] = {"completed_steps": args.train_steps, "last_loss": loss.item(),
                              "peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                       if device.type == "cuda" else None)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
