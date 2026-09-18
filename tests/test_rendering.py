"""P0 #1: structured JSON criteria must render as JSON, and must never crash."""
import json

import pytest

from laya.common import render_criterion, render_options


class TestRenderCriterion:
    def test_string_passes_through(self):
        assert render_criterion("invoices and refunds") == "invoices and refunds"

    def test_dict_renders_as_json_not_python_repr(self):
        out = render_criterion({"what": "invoices", "not_for": ["refunds"]})
        assert "'" not in out, "must be JSON, not a Python repr"
        assert json.loads(out) == {"what": "invoices", "not_for": ["refunds"]}

    def test_list_renders_as_json(self):
        assert json.loads(render_criterion(["a", "b"])) == ["a", "b"]

    def test_non_ascii_is_preserved(self):
        assert "café" in render_criterion({"place": "café"})

    def test_unserializable_value_does_not_crash(self):
        assert render_criterion({"fn": object()})  # default=str


class TestChoice:
    def test_dict_criteria_render_as_json(self):
        opts = render_options({"t": "choice", "crit": {"billing": {"what": "invoices"}}})
        assert opts == ['billing: {"what": "invoices"}']

    def test_string_criteria_unchanged(self):
        assert render_options({"t": "choice", "crit": {"billing": "invoices"}}) == ["billing: invoices"]

    def test_none_criteria_renders_bare_label(self):
        assert render_options({"t": "choice", "crit": {"a": None, "b": "d"}}) == ["a", "b: d"]

    def test_empty_string_criteria_renders_bare_label(self):
        assert render_options({"t": "choice", "crit": {"a": ""}}) == ["a"]

    def test_order_is_preserved(self):
        crit = {"z": None, "a": None, "m": None}
        assert render_options({"t": "choice", "crit": crit}) == ["z", "a", "m"]


class TestScore:
    def test_dict_levels_render_as_json(self):
        opts = render_options({"t": "score", "crit": [{"d": "low"}, {"d": "high"}]})
        assert opts == ['level 0: {"d": "low"}', 'level 1: {"d": "high"}']

    def test_string_levels_unchanged(self):
        assert render_options({"t": "score", "crit": ["calm", "angry"]}) == [
            "level 0: calm",
            "level 1: angry",
        ]


class TestNoul:
    def test_dict_criteria_do_not_raise(self):
        """The P0 regression: this raised TypeError before the fix."""
        opts = render_options({"t": "noul", "crit": {"true": {"what": "holds"}, "false": "nope"}})
        assert opts == ['false: nope', 'true: {"what": "holds"}']

    def test_both_sides_structured(self):
        opts = render_options({"t": "noul", "crit": {"true": {"a": 1}, "false": {"b": 2}}})
        assert opts == ['false: {"b": 2}', 'true: {"a": 1}']

    def test_missing_criteria_uses_defaults(self):
        opts = render_options({"t": "noul", "crit": None})
        assert opts[0].startswith("false: no,") and opts[1].startswith("true: yes,")

    def test_partial_criteria_fills_the_other_side(self):
        opts = render_options({"t": "noul", "crit": {"true": "it holds"}})
        assert opts == ["false: no, the statement does not hold", "true: it holds"]

    def test_false_index_is_zero_and_true_is_one(self):
        """Label order is load-bearing: p[1] is read as P(true) in the agent."""
        opts = render_options({"t": "noul", "crit": {"true": "T", "false": "F"}})
        assert opts[0] == "false: F" and opts[1] == "true: T"


@pytest.mark.parametrize(
    "q",
    [
        {"t": "choice", "crit": {"a": {"x": 1}}},
        {"t": "score", "crit": [{"x": 1}, {"y": 2}]},
        {"t": "noul", "crit": {"true": {"x": 1}, "false": {"y": 2}}},
    ],
)
def test_all_types_accept_structured_criteria(q):
    out = render_options(q)
    assert all(isinstance(o, str) for o in out)
    assert all("'" not in o for o in out), "no Python reprs anywhere"
