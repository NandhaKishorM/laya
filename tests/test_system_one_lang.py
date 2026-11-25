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


def _resolve(raw, base_temperature=BASE_TEMPERATURE):
    from laya.common import resolve_lang_temperatures
    return resolve_lang_temperatures(raw, base_temperature)


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


# ------------------------------------------------------------------ what an override reports
#
# The clamp is the whole of what `lang_temperatures` does to a value, and until now it was the only
# part with no voice: `{"de": {"temperature": [True, 1.5, 2.0]}}` answered `[1.0, 1.5, 2.0]` and
# `[99, 0.01, 1]` answered `[5.0, 0.5, 1.0]` with nothing written anywhere, while the checkpoint's own
# `temperature` list reported the same class of replacement. A per-language override is the one a user
# hand-edits, so the silent case was the more likely one. These tests are at the shared helper rather
# than behind a checkpoint because `Agent` and `ONNXAgent` both assign its return value verbatim.


def _resolve_catching(raw, base_temperature=BASE_TEMPERATURE):
    """Resolve `raw`, returning the result and every warning the load path emitted."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve(raw, base_temperature)
    return resolved, [str(w.message) for w in caught]


@pytest.mark.parametrize("value", [True, False])
def test_lang_temperatures_reports_a_replaced_bool(value):
    """`clamp_temperature` refuses a bool, so the entry is a substitution and has to be named."""
    resolved, messages = _resolve_catching(
        {"de": {"temperature": [value, 1.5, 2.0], "temperature_by_options": {"choice:2": value}}})
    assert resolved["de"]["temperature"] == [1.0, 1.5, 2.0], resolved["de"]
    assert resolved["de"]["temperature_by_options"] == {"choice:2": 1.0}, resolved["de"]
    assert len(messages) == 1, messages
    assert "'de'" in messages[0], messages[0]
    # both fields named: a reader has to know which override key to go fix
    assert "temperature[0]=%r -> 1" % value in messages[0], messages[0]
    assert "choice:2=%r -> 1" % value in messages[0], messages[0]
    assert "uncalibrated" in messages[0], messages[0]


def test_lang_temperatures_reports_an_out_of_range_entry():
    resolved, messages = _resolve_catching({"de": {"temperature": [99, 0.01, 1]}})
    assert resolved["de"]["temperature"] == [5.0, 0.5, 1.0], resolved["de"]
    assert len(messages) == 1, messages
    for fragment in ("temperature[0]=99 -> 5", "temperature[1]=0.01 -> 0.5"):
        assert fragment in messages[0], messages[0]


def test_lang_temperatures_reports_a_value_that_is_not_a_number():
    resolved, messages = _resolve_catching({"de": {"temperature": [None, 1.5, 2.0]}})
    assert resolved["de"]["temperature"] == [1.0, 1.5, 2.0], resolved["de"]
    assert len(messages) == 1, messages
    assert "temperature[0]=None -> 1" in messages[0], messages[0]


@pytest.mark.parametrize("name,raw", [
    ("an in-range override", {"de": {"temperature": [1.0, 1.5, 2.0]}}),
    ("a numeric string that equals the applied value", {"de": {"temperature": ["1.5", 1.5, 2.0]}}),
    ("a valid bucket table", {"de": {"temperature_by_options": {"choice:2": 1.5}}}),
    ("an entry that inherits the checkpoint", {"de": None}),
    ("no option at all", None),
])
def test_lang_temperatures_stays_silent_for_what_it_accepted(name, raw):
    """A warning for a value the clamp left alone would train readers to ignore the warning."""
    _, messages = _resolve_catching(raw)
    assert messages == [], (name, messages)


def test_lang_temperatures_names_the_language_as_the_user_wrote_it():
    """The table keys on the base subtag, but the fix has to be made in the key that was typed."""
    resolved, messages = _resolve_catching({"de-AT": {"temperature": [True, 1.5, 2.0]}})
    assert "de" in resolved, resolved
    assert len(messages) == 1, messages
    assert "'de-AT'" in messages[0], messages[0]


def test_lang_temperatures_warns_once_per_bad_language():
    resolved, messages = _resolve_catching(
        {"de": {"temperature": [True, 1.5, 2.0]},
         "fr": {"temperature": [99, 1.5, 2.0]},
         "es": {"temperature": [2.0, 2.0, 2.0]}})
    assert resolved["es"]["temperature"] == [2.0, 2.0, 2.0], resolved["es"]
    assert len(messages) == 2, messages
    de = [m for m in messages if "'de'" in m]
    fr = [m for m in messages if "'fr'" in m]
    assert len(de) == 1 and len(fr) == 1, messages
    # each line carries only its own language's entries
    assert "temperature[0]=True" in de[0] and "temperature[0]=99" not in de[0], de[0]
    assert "temperature[0]=99 -> 5" in fr[0] and "True" not in fr[0], fr[0]
    assert not [m for m in messages if "'es'" in m], messages


def test_lang_temperatures_does_not_repeat_the_base_diagnostic():
    """An override that omits `temperature` inherits the checkpoint's own list.

    The loader reporting its own replaced entries already said so once; saying it again per language
    would bury the overrides a reader is actually holding. The inherited values still clamp.
    """
    resolved, messages = _resolve_catching({"de": None, "fr": None}, base_temperature=(True, 1.5, 2.0))
    assert resolved["de"]["temperature"] == [1.0, 1.5, 2.0], resolved["de"]
    assert resolved["fr"]["temperature"] == [1.0, 1.5, 2.0], resolved["fr"]
    assert messages == [], messages
