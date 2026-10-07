import argparse
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import research.scripts.finetune_single_device as single  # noqa: E402

mps_spec = importlib.util.spec_from_file_location(
    "laya_finetune_typed_decisions_mps",
    ROOT / "notebooks" / "laya_finetune_typed_decisions_mps.py",
)
mps = importlib.util.module_from_spec(mps_spec)
sys.modules["laya_finetune_typed_decisions_mps"] = mps
mps_spec.loader.exec_module(mps)


def test_single_device_main_delegates_to_laya_train(tmp_path):
    calls = {}
    def fake_finetune(**kwargs):
        calls.update(kwargs)
        return {
            "train_items": 1,
            "calibration_items": 0,
            "temperature": [1.0, 1.0, 1.0],
            "output_dir": str(tmp_path),
        }
    argv = [
        "prog",
        "--data",
        str(tmp_path / "data.jsonl"),
        "--model-dir",
        str(tmp_path / "model"),
        "--output-dir",
        str(tmp_path / "out"),
        "--epochs",
        "2",
        "--seed",
        "7",
        "--device",
        "cpu",
    ]
    with mock.patch.object(single, "finetune", side_effect=fake_finetune):
        with mock.patch.object(sys, "argv", argv):
            single.main()
    assert calls["data"].endswith("data.jsonl")
    assert calls["model_dir"].endswith("model")
    assert calls["output_dir"].endswith("out")
    cfg = calls["config"]
    assert cfg.epochs == 2
    assert cfg.micro_batch == 8
    assert cfg.grad_accum == 1
    assert cfg.max_len == 1024
    assert cfg.head_max_len == 256
    assert calls["device"] == "cpu"


def test_mps_train_delegates_to_laya_train(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "rl_agent_config.json").write_text(json.dumps({"max_len": 1024}), encoding="utf-8")
    calls = {}
    def fake_finetune(**kwargs):
        calls.update(kwargs)
        return {
            "train_items": 1,
            "calibration_items": 0,
            "temperature": [1.0, 1.0, 1.0],
            "output_dir": str(out),
        }
    args = argparse.Namespace(
        epochs=3,
        micro_batch=2,
        grad_accum=16,
        calib_max=400,
        output_dir=str(out),
        no_checkpointing=False,
        device="cpu",
    )
    with mock.patch("laya.train.finetune", side_effect=fake_finetune):
        mps.train(args, str(tmp_path / "model"), str(tmp_path / "rows.jsonl"), None)
    cfg = calls["config"]
    assert cfg.epochs == 3
    assert cfg.micro_batch == 2
    assert cfg.grad_accum == 16
    assert cfg.calib_seed == 20260922
    assert cfg.gradient_checkpointing is True
    assert json.loads((out / "rl_agent_config.json").read_text())["model_name"] == "laya-typed-decisions"


def test_single_wrapper_zero_vs_low_count_temperature_and_pops_bucket_map(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    def fake_finetune(**kwargs):
        (out / "rl_agent_config.json").write_text(json.dumps({
            "temperature": [1.0, 1.0, 1.0],
            "temperature_by_options": {"choice:2": 1.3},
            "training": {"laya_train_calibration": {"choice": {"items": 0, "issues": []}}},
        }), encoding="utf-8")
        return {
            "train_items": 1,
            "calibration_items": 0,
            "temperature": [1.0, 1.0, 1.0],
            "output_dir": str(out),
            "calibration": {"choice": {"items": 0, "issues": []}, "score": {"items": 5, "issues": ["not fitted"]}, "noul": {"items": 0, "issues": []}},
        }
    argv = ["prog", "--data", str(tmp_path/"d.jsonl"), "--model-dir", str(tmp_path/"m"), "--output-dir", str(out), "--device", "cpu"]
    with mock.patch.object(single, "finetune", side_effect=fake_finetune):
        with mock.patch.object(sys, "argv", argv):
            single.main()
    cfg = json.loads((out / "rl_agent_config.json").read_text())
    assert cfg["temperature"] == [1.2, 1.0, 1.2]
    assert cfg["temperature_by_options"] if "temperature_by_options" in cfg else True
    assert "temperature_by_options" not in cfg
    assert cfg["training"]["laya_train_calibration"]["choice"]["items"] == 0


def test_mps_train_wrapper_config_parity_and_bucket_map_removed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    ckpt = out / "checkpoint_latest"
    ckpt.mkdir()
    def make_cfg():
        return {"fine_tuned": True, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {"score:4": 1.1}, "training": {"laya_train_calibration": {"score": {"items": 0, "issues": []}}}}
    (out / "rl_agent_config.json").write_text(json.dumps(make_cfg()), encoding="utf-8")
    (ckpt / "rl_agent_config.json").write_text(json.dumps(make_cfg()), encoding="utf-8")
    def fake_finetune(**kwargs):
        (out / "rl_agent_config.json").write_text(json.dumps(make_cfg()), encoding="utf-8")
        (ckpt / "rl_agent_config.json").write_text(json.dumps(make_cfg()), encoding="utf-8")
        return {"train_items": 1, "calibration_items": 0, "temperature": [1.0, 1.0, 1.0], "output_dir": str(out), "calibration": {"choice": {"items": 0}, "score": {"items": 0}, "noul": {"items": 12}}}
    args = argparse.Namespace(epochs=1, micro_batch=2, grad_accum=16, calib_max=400, output_dir=str(out), no_checkpointing=False, device="cpu")
    with mock.patch("laya.train.finetune", side_effect=fake_finetune):
        mps.train(args, str(tmp_path/"model"), str(tmp_path/"rows.jsonl"), None)
    for path in [out / "rl_agent_config.json", ckpt / "rl_agent_config.json"]:
        cfg = json.loads(path.read_text())
        assert cfg["temperature"] == [1.2, 1.2, 1.0]
        assert "temperature_by_options" not in cfg
        assert cfg["training"]["laya_train_calibration"]["score"]["items"] == 0
        assert cfg["model_name"] == "laya-typed-decisions"
        assert cfg["max_tokens_per_batch"] == 2048


def test_no_duplicated_fitter_or_training_loop_remains():
    single_text = (ROOT / "research" / "scripts" / "finetune_single_device.py").read_text(encoding="utf-8")
    mps_text = (ROOT / "notebooks" / "laya_finetune_typed_decisions_mps.py").read_text(encoding="utf-8")
    assert "fit_one_temp" not in single_text
    assert "fit_temperature(" not in mps_text
    assert "def preprocess" not in single_text
    assert "while n_batches % args.grad_accum" not in mps_text
