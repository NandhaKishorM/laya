"""The shipped ONNX exporters must refuse unsafe checkpoints and declare dynamic shapes.

The first half covers `laya-ts/scripts/export_onnx.py`'s weights check. The second covers
`scripts/export_onnx.py`'s shape declarations: torch 2.2 -- the macOS/Intel pin in
LOCAL_SETUP.md -- has no `dynamic_shapes` keyword, so the exporter raised
`TypeError: export() got an unexpected keyword argument 'dynamic_shapes'` there until it
learned the TorchScript `dynamic_axes` spelling.

The security workflow runs this file on the runner's bare Python, which has neither torch nor
laya, so the second half is guarded and skips there; the tests job, which installs the package
and torch, is where those checks execute.
"""

import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = _load("laya_ts_export_onnx", root / "laya-ts" / "scripts" / "export_onnx.py")


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


# ------------------------------------------------------- the Python exporter's shape spelling
try:
    import torch

    py_exporter = _load("laya_export_onnx", root / "scripts" / "export_onnx.py")
except ImportError as missing:          # the supply-chain job's Python has neither torch nor laya
    torch = None
    py_exporter = None
    _skipped = str(missing)

if py_exporter is None:
    print("note: the Python exporter's shape checks were skipped (%s)" % _skipped)
else:
    AXES = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype",
            "logits", "act_logits"}

    legacy = py_exporter._shape_kwargs(False)
    assert "dynamic_axes" in legacy and "dynamic_shapes" not in legacy, legacy
    assert legacy["opset_version"] <= 17, legacy["opset_version"]
    assert set(legacy["dynamic_axes"]) == AXES, sorted(legacy["dynamic_axes"])
    assert all(axes for axes in legacy["dynamic_axes"].values()), legacy["dynamic_axes"]

    if hasattr(torch, "export") and hasattr(torch.export, "Dim"):
        modern = py_exporter._shape_kwargs(True)
        assert "dynamic_shapes" in modern and "dynamic_axes" not in modern, modern
        assert modern["opset_version"] == 18, modern["opset_version"]

    # The default has to follow the installed torch: a hard-coded choice is the TypeError this
    # branch exists for, and it would be right on only one of the two stacks.
    installed = "dynamic_shapes" in inspect.signature(torch.onnx.export).parameters
    auto = py_exporter._shape_kwargs()
    assert ("dynamic_shapes" in auto) == installed, (sorted(auto), installed)
