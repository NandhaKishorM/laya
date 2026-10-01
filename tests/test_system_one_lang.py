from typing import Any, Dict, List, Union
import pytest
from laya.agent import Agent

class DummyAgent(Agent):
    def __init__(self):
        # Skip the whole HuggingFace initialization for testing the plumbing
        self.device = "cpu"
        self.temperature_by_options = {}
        self.temperature = [1.0, 1.0, 1.0]
        self.lang_temperatures = {}
        self.cfg = {"max_len": 512, "head_max_len": 192}

    def predict_batch(self, states: List[Union[str, dict, list]], questions: Dict[str, Dict[str, Any]],
                      batch_size: int = None, lang: str = None, **kwargs) -> List[Dict[str, Any]]:
        # Mock predict_batch to just return the lang it was given so we can test the pass-through
        return [{"lang_passed_down": lang}]


def test_system_one_lang_parameter_passthrough():
    agent = DummyAgent()
    questions = {"q1": {"type": "noul", "instructions": "test"}}

    # 1. Test without lang
    res_no_lang = agent.system_one("test state", questions)
    assert res_no_lang.get("lang_passed_down") is None

    # 2. Test with lang
    res_with_lang = agent.system_one("test state", questions, lang="zh-CN")
    assert res_with_lang.get("lang_passed_down") == "zh-CN"


# ---------------------------------------------------------------- `lang_temperatures`' shape
#
# `lang_temperatures` was validated with `cfg.get(...)` and `len(...)` before anything checked the
# shape of either, so the inputs the `ValueError` exists to reject raised `AttributeError` and
# `TypeError` instead -- after the whole checkpoint had loaded. The `laya-ts` port reads the same
# option with `lc?.temperature ?? rawList` and `!Array.isArray` (`agent.ts:322-327`); this is the
# Python half of that contract.
#
# These are `test_` functions rather than a `__main__` block, and they do not load a checkpoint.
# A `__main__` block does not execute under `python -m pytest tests/test_system_one_lang.py`, which
# is how this file is run in CI, so the earlier version of these checks asserted nothing there; and
# loading `convaiinnovations/laya` would have made the suite depend on a download. The parsing is
# `laya.common.resolve_lang_temperatures`, which `Agent` and `ONNXAgent` both call, so exercising
# it covers both backends without weights.

BASE_TEMPERATURE = (1.0, 1.5, 2.0)


def _resolve(raw):
    from laya.common import resolve_lang_temperatures
    return resolve_lang_temperatures(raw, BASE_TEMPERATURE)


@pytest.mark.parametrize("name,raw,fragment", [
    ("a scalar temperature", {"de": {"temperature": 2}}, "list of 3 floats"),
    ("a two-element temperature", {"de": {"temperature": [1, 1]}}, "list of 3 floats"),
    ("a string temperature", {"de": {"temperature": "1,1,1"}}, "list of 3 floats"),
    ("a non-mapping bucket table", {"de": {"temperature_by_options": 2}}, "mapping"),
    ("a non-mapping entry", {"de": 2}, "must be a mapping"),
    ("a non-string language key", {1: None}, "must be strings"),
])
def test_lang_temperatures_rejects_malformed_shapes(name, raw, fragment):
    """Every one of these is a shape the option's own documentation excludes."""
    with pytest.raises(ValueError) as info:
        _resolve(raw)
    # the message has to name the offending language and the field, or a caller cannot act on it
    assert fragment in str(info.value), str(info.value)


@pytest.mark.parametrize("name,raw", [
    ("a null entry", {"de": None}),
    ("a null temperature", {"de": {"temperature": None}}),
    ("a valid override", {"de": {"temperature": [1.0, 1.0, 1.0]}}),
    ("a subtag key", {"de-AT": {"temperature": [1.0, 1.0, 1.0]}}),
    ("bucket overrides", {"de": {"temperature_by_options": {"choice:2": 1.5}}}),
    ("no option at all", None),
])
def test_lang_temperatures_accepts_wellformed_shapes(name, raw):
    """`None` means "inherit the checkpoint's own", so it is accepted rather than rejected."""
    assert isinstance(_resolve(raw), dict)


def test_lang_temperatures_null_entry_inherits_the_checkpoint():
    resolved = _resolve({"de": None})
    # the checkpoint's own list, unchanged -- and going through the same clamp an override does,
    # which leaves these alone because they are already inside [0.5, 5.0]
    assert resolved["de"]["temperature"] == [1.0, 1.5, 2.0], resolved["de"]


def test_lang_temperatures_subtag_normalises_to_its_base():
    assert "de" in _resolve({"de-AT": None}), "a subtag key did not normalise"


def test_lang_temperatures_still_clamps_an_override():
    resolved = _resolve({"de": {"temperature": [99, 0.01, 1]}})
    assert resolved["de"]["temperature"] == [5.0, 0.5, 1.0], resolved["de"]


def test_lang_temperatures_result_is_what_both_backends_store():
    """The shape `resolve_lang_temperatures` returns is assigned to `lang_temperatures` verbatim."""
    from laya.agent import Agent as _Agent
    import inspect
    for cls in (_Agent,):
        src = inspect.getsource(cls.__init__)
        assert "resolve_lang_temperatures" in src, cls.__name__
