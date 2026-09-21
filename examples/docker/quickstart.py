"""Run one SDK request using the adjacent example input."""
import json
import os
from pathlib import Path

import torch

from laya import Router, load


def main():
    device = os.environ.get("LAYA_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Check the GPU override, host driver and NVIDIA Container Toolkit."
        )
    request = json.loads(Path(__file__).with_name("request.json").read_text(encoding="utf-8"))
    model_path = os.environ.get("LAYA_MODEL_PATH")
    engine = load(model_path, device=device) if model_path else Router(device=device)
    result = engine.predict(request["state"], request["questions"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
