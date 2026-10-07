"""`laya-train`: fine-tune a Laya checkpoint on labelled data (CSV or JSONL).

Usage:
    laya-train --data tickets.csv --text-column body --label-column department \\
               --base convaiinnovations/laya --out ./my-checkpoint

    laya-train --data train.jsonl --base multilingual --out ./my-checkpoint

`laya train ...` dispatches here from the main CLI, so both spellings work.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-train",
        description="Fine-tune a Laya checkpoint on labelled decisions (JSONL or CSV).",
    )
    # Required data and output options
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training data file (.jsonl or .csv).",
    )
    parser.add_argument(
        "--eval",
        "--eval-data",
        dest="eval_data",
        default=None,
        help="Path to evaluation dataset (.jsonl or .csv) for before/after assessment.",
    )
    parser.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        required=False,
        default=None,
        help="Output directory where the fine-tuned checkpoint will be saved.",
    )
    parser.add_argument(
        "--base",
        "--model",
        dest="model_dir",
        default="convaiinnovations/laya",
        help="Base checkpoint path or model ID (default: 'convaiinnovations/laya').",
    )

    # CSV mapping options
    parser.add_argument(
        "--text-column",
        default="text",
        help="CSV column containing the input text/state (default: 'text').",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="CSV column containing the target label (default: 'label').",
    )
    parser.add_argument(
        "--question-id",
        default="label",
        help="Question identifier when synthesizing a question from CSV (default: 'label').",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Instructions string for synthesized choice question.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs (default: 4).",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=8,
        help="Micro-batch size per forward pass (default: 8).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8).",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=2.5e-5,
        help="Learning rate for encoder (default: 2.5e-5).",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="Learning rate for classification heads (default: 1e-4).",
    )
    parser.add_argument(
        "--loss",
        choices=["soft-ce", "rlcd"],
        default="rlcd",
        help="Training objective: 'rlcd' or 'soft-ce' (default: 'rlcd').",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon for hard labels in [0, 1) (default: 0.0).",
    )
    parser.add_argument(
        "--shuffle-options",
        action="store_true",
        help="Enable random option-order augmentation during training for choice questions.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights, training classification heads only.",
    )
    parser.add_argument(
        "--target-error",
        type=float,
        default=0.10,
        help="Target error rate for fitting abstention thresholds in [0, 1] (default: 0.10).",
    )
    parser.add_argument(
        "--min-abstain-n",
        type=int,
        default=10,
        help="Minimum items in a bucket required to fit an abstention threshold (default: 10).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device ('auto', 'cpu', 'cuda', 'mps'; default: 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Override maximum state token budget.",
    )
    parser.add_argument(
        "--head-max-len",
        dest="head_max_len",
        type=int,
        default=None,
        help="Override maximum question head token budget.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect dataset, build items, and print stats without loading model weights or training.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if not args.dry_run and not args.output_dir:
            parser.error("--out is required unless --dry-run is set")
    except SystemExit as exc:
        return exc.code

    try:
        from .train import TrainConfig, dry_run, finetune
    except ImportError as err:
        if "torch" in str(err).lower():
            print("laya-train: error: PyTorch is required for training. Please install torch.",
                  file=sys.stderr)
            return 1
        raise

    shuffle_opts = ("choice",) if args.shuffle_options else ()
    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        loss=args.loss,
        label_smoothing=args.label_smoothing,
        shuffle_options=shuffle_opts,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        text_column=args.text_column,
        label_column=args.label_column,
        question_id=args.question_id,
        instructions=args.instructions,
        eval_data=args.eval_data,
        target_error=args.target_error,
        min_abstain_n=args.min_abstain_n,
    )

    try:
        config.validate()
        if args.dry_run:
            dry_run(
                data=args.data,
                model_dir=args.model_dir,
                config=config,
            )
            return 0
        summary = finetune(
            data=args.data,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            config=config,
            device=args.device,
        )
        print("Training complete! Model saved to %s" % summary["output_dir"])
        return 0
    except (ValueError, FileNotFoundError, TypeError) as err:
        print("laya-train: error: %s" % (err,), file=sys.stderr)
        return 1
    except Exception as err:
        print("laya-train: unexpected error: %s" % (err,), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
