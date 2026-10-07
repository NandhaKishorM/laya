"""Fine-tune a Laya checkpoint on a single device (CPU or GPU).

Reproduces the 2xT4 notebook's training loop without DDP, so any checkpoint
(including the multilingual one) can be fine-tuned on one GPU or on a CPU.
Preprocesses a JSONL dataset into tokenized items, trains with RLCD, fits one
calibration temperature per question type, and saves a loadable checkpoint.

Each JSONL line is one case: {"state": "...", "questions": {...}, "gold": {...}}
where questions use the choice / score / noul primitives and gold carries the
teacher probabilities plus a label, matching the schema in docs/finetune.md.

Usage:
  python research/scripts/finetune_single_device.py \
      --data dataset.jsonl --model-dir /path/to/multilingual --output-dir out/
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from huggingface_hub import snapshot_download

from laya.agent import _fix_tokenizer_config
from laya.train import TrainConfig, finetune


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Laya checkpoint on one device."
    )
    parser.add_argument(
        "--data", required=True, help="JSONL dataset of {state, questions, gold} cases."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Checkpoint directory; defaults to the multilingual subfolder.",
    )
    parser.add_argument("--output-dir", default="laya_finetuned")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join(
            snapshot_download(
                "convaiinnovations/laya",
                allow_patterns=["multilingual/*", "multilingual/*/*"],
            ),
            "multilingual",
        )
    _fix_tokenizer_config(model_dir)

    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=8,
        grad_accum=1,
        seed=args.seed,
        calib_seed=args.seed,
        max_len=1024,
        head_max_len=256,
    )
    summary = finetune(
        data=args.data,
        model_dir=model_dir,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )
    # Preserve the old wrapper contract verbatim: a qtype that never appeared in the
    # calibration split did not get a fitted 1.0 from the shared fit path; it kept the
    # pre-fit 1.2 no-data default. Low-count present types still report 1.0 via #933.
    if isinstance(summary.get("calibration"), dict):
        for rel in [Path(args.output_dir) / "rl_agent_config.json"]:
            if rel.exists():
                cfg = json.loads(rel.read_text(encoding="utf-8"))
                for name, entry in summary["calibration"].items():
                    if isinstance(entry, dict) and entry.get("items") == 0:
                        idx = {"choice": 0, "score": 1, "noul": 2}.get(name)
                        if idx is not None:
                            cfg["temperature"][idx] = 1.2
                cfg.pop("temperature_by_options", None)
                rel.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Training items: {} ({} held out)".format(
        summary["train_items"], summary["calibration_items"]
    ))
    print("Fitted temperatures (choice, score, noul):", [round(t, 3) for t in summary["temperature"]])
    print("Saved checkpoint to", summary["output_dir"])


if __name__ == "__main__":
    main()
