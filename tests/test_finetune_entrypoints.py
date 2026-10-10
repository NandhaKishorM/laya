import argparse
import importlib.util
import json
import re
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


# ------------------------------------------------- `laya-train`'s options table, against the parser

# `docs/finetune.md`'s Options table is where a reader learns what `laya-train` accepts and what each
# flag does to the run; `--help` gives one line per flag and no consequence. Three flags were in
# neither the table nor any other page under `docs/` or the README -- `--eval`, `--target-error` and
# `--min-abstain-n` -- although the last two decide the per-bucket abstention map a fine-tuned
# checkpoint ships with, which is the number `Router` and `laya-serve` later gate answers on.
#
# Not a completeness gate over the parser, on purpose: #982 is adding six flags to `build_parser` as
# this lands, and a check demanding a row for every flag would go red on that PR instead of on the
# gap it exists to catch. What is checked is derived from the parser and applied to whatever the page
# claims, in both directions: a flag the table names has to exist, and a default the table prints has
# to be the parser's. The three flags this table gained stay named by name.

TABLE_ROW = re.compile(r"^\|\s*((?:`--[a-z0-9-]+`)(?:\s*,\s*`--[a-z0-9-]+`)*)\s*\|\s*([^|]*?)\s*\|")
FLAG_IN_CELL = re.compile(r"`(--[a-z0-9-]+)`")
FINETUNE_PAGE = ROOT / "docs" / "finetune.md"


def _parser_defaults():
    """`{option string: default}` for every flag `laya-train` accepts, aliases included."""
    from laya.train_cli import build_parser

    out = {}
    for action in build_parser()._actions:
        for option in action.option_strings:
            out[option] = action.default
    return out


def _option_table():
    """The Options table's rows as `(flags, default cell)`, read out of `docs/finetune.md`."""
    lines = FINETUNE_PAGE.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "### Options"), None)
    assert start is not None, "%s no longer has an `### Options` section" % FINETUNE_PAGE
    rows = []
    for line in lines[start:start + 40]:
        if line.startswith("###") and rows:
            break
        match = TABLE_ROW.match(line)
        if match:
            rows.append((FLAG_IN_CELL.findall(match.group(1)), match.group(2)))
    assert rows, "`### Options` in %s holds no table rows this sweep can read" % FINETUNE_PAGE
    return rows


def test_every_flag_in_the_options_table_exists():
    """The table cannot promise a flag `laya-train` does not take."""
    defaults = _parser_defaults()
    rows = _option_table()
    unknown = sorted({flag for flags, _cell in rows for flag in flags if flag not in defaults})
    assert not unknown, (
        "docs/finetune.md's Options table names %s, which `laya-train` does not accept; the flags it "
        "takes are %s" % (unknown, sorted(defaults)))


def test_the_options_table_prints_the_parser_defaults():
    """Where the table states numbers for a row of flags, they are the parser's, in that order."""
    defaults = _parser_defaults()
    checked = 0
    stale = []
    for flags, cell in _option_table():
        parts = [part.strip().strip("`") for part in cell.split(",")]
        if len(parts) != len(flags):
            continue        # a shared row stating one thing for several flags, or prose
        try:
            stated = [float(part) for part in parts]
        except ValueError:
            continue        # `rlcd`, `off`, `auto`: not a number this check can compare
        for flag, value in zip(flags, stated):
            got = defaults.get(flag)
            if isinstance(got, bool) or not isinstance(got, (int, float)):
                continue        # unknown flags are the other check's business
            checked += 1
            if float(got) != value:
                stale.append("%s: table says %r, `build_parser` defaults to %r" % (flag, value, got))
    assert not stale, "docs/finetune.md prints a default the parser does not have: %s" % stale
    assert checked >= 6, (
        "fixture: this check compared %d defaults, and the table's numeric rows have to keep it "
        "meaningful -- the rows it reads are no longer the ones it was written against" % checked)


def test_the_abstention_and_eval_flags_are_documented():
    """The three flags that decide what a fine-tuned checkpoint ships with stay in the table."""
    rows = _option_table()
    named = {flag for flags, _cell in rows for flag in flags}
    for flag in ("--eval", "--target-error", "--min-abstain-n"):
        assert flag in named, (
            "%s is what decides a fine-tuned checkpoint's held-out numbers and its abstention map, "
            "and it is no longer in docs/finetune.md's Options table" % flag)
