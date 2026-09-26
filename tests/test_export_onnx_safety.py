"""The shipped ONNX exporter must refuse checkpoints without safe weights."""

import importlib.util
import tempfile
from pathlib import Path


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "laya_ts_export_onnx", root / "laya-ts" / "scripts" / "export_onnx.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


with tempfile.TemporaryDirectory() as directory:
    model_dir = Path(directory)
    (model_dir / "pytorch_model.bin").write_bytes(b"legacy weights must never be loaded")
    try:
        exporter._checkpoint_weights_path(str(model_dir))
    except SystemExit as error:
        assert "model.safetensors" in str(error)
    else:
        raise AssertionError("a legacy .bin checkpoint must be rejected")

    weights = model_dir / "model.safetensors"
    weights.touch()
    assert exporter._checkpoint_weights_path(str(model_dir)) == str(weights)
