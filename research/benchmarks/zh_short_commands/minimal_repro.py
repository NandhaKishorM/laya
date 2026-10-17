"""Small SDK reproduction using the existing Chinese prompt definitions."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("USE_TF", "0")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "models/laya-multilingual"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import torch
    import laya
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13)
    torch.use_deterministic_algorithms(True)
    agent = laya.load(args.checkpoint, device="cpu")
    agent.model.float().eval()
    agent.amp_enabled = False
    rows = []
    for text in ("太快了", "太慢了", "不要右转，请左转", "不要左转，请右转"):
        for config in ("choice_criteria", "noul_plain", "noul_criteria"):
            result = agent.predict(prompts.state_for(config, text), prompts.questions_for(config))
            rows.append({"text": text, "config": config, "answers": result["answers"]})
    payload = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(payload)
    if args.out:
        args.out.write_text(payload, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
