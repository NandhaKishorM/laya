"""Run one SDK request using the adjacent example input."""
import json
import os
from pathlib import Path

import torch

from laya import Router, load


def main():
    device = os.environ.get("LAYA_DEVICE", "cpu")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Check the GPU override, host driver and NVIDIA Container Toolkit."
        )
    request_file = os.environ.get("LAYA_REQUEST_FILE") or Path(__file__).with_name("request.json")
    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    model_path = os.environ.get("LAYA_MODEL_PATH")
    model = os.environ.get("LAYA_MODEL", "auto") or "auto"
    if model_path:
        if model != "auto":
            raise ValueError("Set either LAYA_MODEL or LAYA_MODEL_PATH, not both")
        result = load(model_path, device=device).predict(request["state"], request["questions"])
    else:
        result = Router(device=device).predict(
            request["state"], request["questions"], model=None if model == "auto" else model
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
