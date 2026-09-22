"""Offline tests for the Windows + Python 3.14 init guard. No Hub download.

Run: python tests/test_win_patch.py
"""
import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transformers.modeling_utils as modeling_utils  # noqa: E402
from transformers import AutoModel, ModernBertConfig  # noqa: E402

from laya.common import DecisionModel, build_model  # noqa: E402
from laya.win_patch import (  # noqa: E402
    _affected,
    _noop_initialize_weights,
    guard,
    install,
    installed,
    needs_guard,
    restore,
)

_REAL = modeling_utils.PreTrainedModel.initialize_weights


def _reset():
    while installed():
        restore()
    modeling_utils.PreTrainedModel.initialize_weights = _REAL


def _method():
    return modeling_utils.PreTrainedModel.initialize_weights


def test_needs_guard_matches_windows_and_python_314():
    assert _affected("Windows", (3, 14)) is True
    assert _affected("Windows", (3, 14, 0)) is True
    assert _affected("Windows", (3, 15, 1)) is True
    assert _affected("Windows", (3, 13, 9)) is False
    assert _affected("Windows", (3, 10)) is False
    assert _affected("Linux", (3, 14)) is False
    assert _affected("Darwin", (3, 14)) is False
    assert needs_guard() is _affected(platform.system(), sys.version_info)


def test_live_platform_installs_or_is_a_noop():
    """On this machine, install applies only when needs_guard() is true, and restore undoes it."""
    try:
        if needs_guard():
            assert install() is True
            assert installed() is True
            assert _method() is _noop_initialize_weights
            assert _method()(object()) is None
            assert restore() is True
            assert installed() is False
            assert _method() is _REAL
            assert restore() is False
        else:
            assert install() is False
            assert installed() is False
            assert _method() is _REAL
            assert restore() is False
            assert _method() is _REAL
    finally:
        _reset()


def test_force_install_nests_and_restore_puts_the_original_back():
    calls = []

    def spy(self, *args, **kwargs):
        calls.append("spy")
        return "spy"

    try:
        modeling_utils.PreTrainedModel.initialize_weights = spy
        assert install(force=True) is True
        assert install(force=True) is True
        assert installed() is True
        assert _method() is _noop_initialize_weights
        assert _method()(object()) is None
        assert calls == []
        assert restore() is True
        assert installed() is True
        assert _method() is _noop_initialize_weights
        assert restore() is True
        assert installed() is False
        assert _method() is spy
        assert restore() is False
    finally:
        _reset()


def test_guard_restores_after_success_and_after_an_error():
    try:
        with guard(force=True) as held:
            assert held is True
            assert _method() is _noop_initialize_weights
        assert installed() is False
        assert _method() is _REAL

        try:
            with guard(force=True):
                assert _method() is _noop_initialize_weights
                raise RuntimeError("build failed")
        except RuntimeError as exc:
            assert str(exc) == "build failed"
        assert installed() is False
        assert _method() is _REAL
    finally:
        _reset()


def test_guard_is_a_noop_when_the_platform_is_unaffected():
    try:
        with patch("laya.win_patch.needs_guard", return_value=False):
            with guard() as held:
                assert held is False
                assert _method() is _REAL
            assert install() is False
        assert _method() is _REAL
    finally:
        _reset()


def test_install_does_nothing_when_initialize_weights_is_missing():
    try:
        with patch("laya.win_patch.needs_guard", return_value=True), patch.object(
            modeling_utils.PreTrainedModel, "initialize_weights", None
        ):
            assert install() is False
            assert installed() is False
    finally:
        _reset()


def test_explicit_install_outlives_a_nested_guard():
    try:
        assert install(force=True) is True
        with guard(force=True) as held:
            assert held is True
            assert _method() is _noop_initialize_weights
        assert installed() is True
        assert _method() is _noop_initialize_weights
        assert restore() is True
        assert _method() is _REAL
    finally:
        _reset()


def _tiny_modernbert_dir() -> str:
    tmp = tempfile.mkdtemp(prefix="laya-win-patch-")
    cfg = ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        reference_compile=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    cfg.save_pretrained(tmp)
    return tmp


def _spy_from_config(seen):
    original = AutoModel.from_config

    def spy(config, *args, **kwargs):
        seen["during"] = _method()
        return original(config, *args, **kwargs)

    return spy


def test_build_model_holds_the_noop_only_during_construction_when_affected():
    seen = {}
    encoder_dir = _tiny_modernbert_dir()
    try:
        with patch("laya.win_patch.needs_guard", return_value=True), patch.object(
            AutoModel, "from_config", _spy_from_config(seen)
        ):
            model = build_model({"head_layers": 0, "act_costs": {}}, encoder_dir=encoder_dir)
        assert isinstance(model, DecisionModel)
        assert type(model.encoder).__name__ == "ModernBertModel"
        assert seen["during"] is _noop_initialize_weights
        assert installed() is False
        assert _method() is _REAL
    finally:
        _reset()


def test_build_model_leaves_initialize_weights_alone_when_unaffected():
    seen = {}
    encoder_dir = _tiny_modernbert_dir()
    try:
        with patch("laya.win_patch.needs_guard", return_value=False), patch.object(
            AutoModel, "from_config", _spy_from_config(seen)
        ):
            model = build_model({"head_layers": 0, "act_costs": {}}, encoder_dir=encoder_dir)
        assert isinstance(model, DecisionModel)
        assert seen["during"] is _REAL
        assert installed() is False
        assert _method() is _REAL
    finally:
        _reset()


def test_build_model_restores_when_construction_raises():
    encoder_dir = _tiny_modernbert_dir()

    def boom(config, *args, **kwargs):
        assert _method() is _noop_initialize_weights
        raise RuntimeError("encoder failed")

    try:
        with patch("laya.win_patch.needs_guard", return_value=True), patch.object(AutoModel, "from_config", boom):
            try:
                build_model({"head_layers": 0, "act_costs": {}}, encoder_dir=encoder_dir)
            except RuntimeError as exc:
                assert str(exc) == "encoder failed"
            else:
                raise AssertionError("build_model should have raised")
        assert installed() is False
        assert _method() is _REAL
    finally:
        _reset()


if __name__ == "__main__":
    test_needs_guard_matches_windows_and_python_314()
    test_live_platform_installs_or_is_a_noop()
    test_force_install_nests_and_restore_puts_the_original_back()
    test_guard_restores_after_success_and_after_an_error()
    test_guard_is_a_noop_when_the_platform_is_unaffected()
    test_install_does_nothing_when_initialize_weights_is_missing()
    test_explicit_install_outlives_a_nested_guard()
    test_build_model_holds_the_noop_only_during_construction_when_affected()
    test_build_model_leaves_initialize_weights_alone_when_unaffected()
    test_build_model_restores_when_construction_raises()
    _reset()
    print("all win_patch tests passed")
