"""Run the Step-4 benchmark through the `laya-train` CLI.

Two gates, kept separate on purpose:

* **Gate R — reproduction.** Seed 0, the notebook's hyperparameters, `--loss rlcd`. It has to land
  on the published checkpoint / the maintainer's target within noise; that verifies acceptance
  criterion 1. Nothing else is claimed from one seed.
* **Gate D — default-loss decision.** `rlcd` vs `soft-ce` over more than one seed. The existing
  single-seed soft-ce win is not enough to recommend changing a shipped default, so Gate D is
  only designed and run after Gate R passes.

This calls the CLI as a subprocess: it does not import `laya.train`, so it cannot depend on the
training internals #967/#968/#982 are still changing.

Usage:
    # Gate R: one run
    python run_benchmark.py --gate R --data data/typed_decisions/train.jsonl \
        --out-dir runs/step4 --device cuda

    # Gate D: rlcd + soft-ce over seeds 0/1/2
    python run_benchmark.py --gate D --data data/typed_decisions/train.jsonl \
        --out-dir runs/step4 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

# The notebook's recipe, single-device form. 8 micro-batch x 2 GPUs x 4 accumulation = 64
# effective; on one device that is 8 x 8.
NOTEBOOK_HYPERPARAMS = {
    "epochs": 4,
    "micro_batch": 8,
    "grad_accum": 8,
    "encoder_lr": 2.5e-5,
    "head_lr": 1e-4,
    "max_len": 1024,
    "head_max_len": 256,
}

GATES: Dict[str, List[Tuple[str, int]]] = {
    "R": [("rlcd", 0)],
    "D": [("rlcd", 0), ("rlcd", 1), ("rlcd", 2),
          ("soft-ce", 0), ("soft-ce", 1), ("soft-ce", 2)],
}


def build_command(data: str, out_dir: str, loss: str, seed: int, base: str, device: str,
                  hyper: Dict[str, Any]) -> List[str]:
    return [
        sys.executable, "-m", "laya.train_cli",
        "--data", data,
        "--out", out_dir,
        "--base", base,
        "--loss", loss,
        "--seed", str(seed),
        "--epochs", str(hyper["epochs"]),
        "--micro-batch", str(hyper["micro_batch"]),
        "--grad-accum", str(hyper["grad_accum"]),
        "--encoder-lr", str(hyper["encoder_lr"]),
        "--head-lr", str(hyper["head_lr"]),
        "--max-len", str(hyper["max_len"]),
        "--head-max-len", str(hyper["head_max_len"]),
        "--device", device,
    ]


def run_one(data: str, out_root: str, loss: str, seed: int, base: str, device: str,
            hyper: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = os.path.join(out_root, f"{loss.replace('-', '_')}-seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    cmd = build_command(data, run_dir, loss, seed, base, device, hyper)
    print("\n" + "=" * 72)
    print(" ".join(shlex.quote(c) for c in cmd))
    print("=" * 72, flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = round(time.time() - t0, 1)
    if result.returncode != 0:
        raise SystemExit(f"laya-train failed for loss={loss} seed={seed} "
                         f"(exit {result.returncode})")

    cfg_path = os.path.join(run_dir, "rl_agent_config.json")
    cfg: Dict[str, Any] = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

    return {
        "loss": loss,
        "seed": seed,
        "run_dir": run_dir,
        "elapsed_seconds": elapsed,
        "temperature": cfg.get("temperature"),
        "temperature_by_options": cfg.get("temperature_by_options", {}),
        "calibration": (cfg.get("training") or {}).get("laya_train_calibration"),
        "max_len": cfg.get("max_len"),
        "head_max_len": cfg.get("head_max_len"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=sorted(GATES), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="runs/step4")
    parser.add_argument("--base", default="convaiinnovations/laya")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--losses", nargs="+", default=None,
                        help="Override the gate's losses.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override the gate's seeds.")
    args = parser.parse_args()

    matrix = GATES[args.gate]
    if args.losses:
        matrix = [(loss, seed) for loss, seed in matrix if loss in args.losses]
    if args.seeds:
        matrix = [(loss, seed) for loss, seed in matrix if seed in args.seeds]
    if not matrix:
        parser.error("the loss/seed filters left the gate empty")

    os.makedirs(args.out_dir, exist_ok=True)
    runs = [run_one(args.data, args.out_dir, loss, seed, args.base, args.device,
                    NOTEBOOK_HYPERPARAMS)
            for loss, seed in matrix]

    manifest = {
        "gate": args.gate,
        "base": args.base,
        "data": args.data,
        "hyperparameters": NOTEBOOK_HYPERPARAMS,
        "runs": runs,
    }
    manifest_path = os.path.join(args.out_dir, f"manifest_{args.gate}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
