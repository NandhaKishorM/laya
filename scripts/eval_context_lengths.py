"""Measure accuracy and calibration vs. context length for a laya checkpoint.

Answers "how far past its trained window can this checkpoint be pushed?" by running the
same needle-in-a-document task at increasing lengths and reporting accuracy plus ECE.
"""
import argparse
import json
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/laya")
from train_long_context import build_batch, evaluate  # noqa: E402

from laya.common import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--out", default="context_results.json")
    args = ap.parse_args()

    import os

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    cfg = json.load(open(os.path.join(args.model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    enc_dir = os.path.join(args.model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    model.load_state_dict(load_file(os.path.join(args.model_dir, "model.safetensors")), strict=True)
    model.encoder.config.max_position_embeddings = 32768
    model.to(device).eval()

    rows = []
    for L in [int(x) for x in args.lengths.split(",")]:
        acc, ece = evaluate(
            model, tok, random.Random(1), L, cfg.get("head_max_len", 192),
            int(L * 3.6), device, n=args.n, bs=4,
        )
        row = {"max_len": L, "accuracy": round(acc, 4), "ece": round(ece, 4)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    json.dump({"model_dir": args.model_dir, "results": rows}, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
