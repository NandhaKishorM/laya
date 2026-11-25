"""Opt-in diagnostics explain the executed detector, without changing its decisions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya import Router, BaseHook  # noqa: E402
from laya import lang  # noqa: E402

EN = "This is an English note about the order. "
DE = "Ich kann mich nicht anmelden und brauche Hilfe"
PT = "Quero cancelar a minha conta porque não consigo entrar"


def explained(state):
    plain = lang.analyse(state)
    result = lang.analyse(state, explain=True)
    diagnostic = result.pop("diagnostic")
    assert result == plain
    assert set(diagnostic) == {"stage", "reason", "source"}
    json.dumps(diagnostic, ensure_ascii=False, allow_nan=False)
    return diagnostic


@pytest.mark.parametrize(("state", "reason"), [
    ("Turn lights off", "insufficient_words"),
    ("xyzzy frobnitz quux blorb", "no_language_evidence"),
    ("I would like to book a flight to Berlin tomorrow", "language_evidence"),
    (DE, "language_evidence"),
    ("glömt lösenord", "short_swedish"),
    ("MON DES EST LA", "caps_veto"),
    ("This is my ticket. MON DES EST LA", "caps_filtered"),
    ("åå øø äa", "non_english_markers"),
    ("请帮我取消订单", "non_latin_script"),
    ("1234", "no_letters"),
    ("", "no_letters"),
    (None, "no_letters"),
    (b"\xff", "no_letters"),
])
def test_whole_state_reason(state, reason):
    assert explained(state) == {"stage": "whole_state", "reason": reason, "source": None}


def test_original_fields_stay_fixed():
    assert lang.analyse("Turn lights off") == {
        "script": "latin", "script_profile": {"latin": 1.0}, "language": None,
        "is_english": True, "language_undecided": True, "diacritic_rate": 0.0,
        "non_latin_fraction": 0.0, "mixed_segment": None,
    }
    assert set(lang.latin_profile("Turn lights off")) == {
        "language", "english_hits", "diacritic_rate", "looks_non_english",
    }


def test_segment_keeps_its_original_text_and_identifies_the_first_matching_line():
    state = {"note": EN * 20, "messages": [{"customer": "status\n" + DE + "\n" + PT}]}
    d = explained(state)
    assert d == {"stage": "segment", "reason": "named_segment",
                 "source": {"leaf_index": 1, "path": ["messages", 0, "customer"], "line": 2}}
    assert lang.analyse(state)["mixed_segment"] == DE
    # Equal text in another field must not replace the source by searching the state afterwards.
    state["duplicate"] = DE
    assert explained(state)["source"] == d["source"]


def test_root_string_source_is_a_line_not_a_fabricated_field():
    d = explained(EN * 20 + "\n" + DE)
    assert d["stage"] == "segment"
    assert d["source"] == {"leaf_index": 0, "path": [], "line": 2}


@pytest.mark.parametrize("wrap", [dict, UserDict, MappingProxyType])
def test_late_value_records_nested_path_without_changing_mixed_segment(wrap):
    state = wrap({"note": EN * 140, "messages": ({"customer": "status\n" + DE},)})
    d = explained(state)
    assert d == {"stage": "value", "reason": "named_value",
                 "source": {"leaf_index": 1, "path": ["messages", 0, "customer"], "line": 2}}
    assert lang.analyse(state)["mixed_segment"] is None
    assert DE not in json.dumps(d)


@pytest.mark.parametrize(("message", "reason"), [
    ("请帮我取消这个订单并且退款给我", "non_latin_value"),
    ("WIE SPÄT IST ES IN KÖLN", "diacritic_value"),
])
def test_value_eligibility_reason(message, reason):
    d = explained({"note": EN * 140, "customer": message})
    assert d["stage"] == "value"
    assert d["reason"] == reason
    assert d["source"]["path"] == ["customer"]
    assert lang.analyse({"note": EN * 140, "customer": message})["script_profile"] == {"latin": 1.0}


def test_value_selection_keeps_both_existing_winner_rules():
    # Inner scan chooses the longest qualifying line; outer scan weighs the entire leaf window.
    state = {"note": EN * 140, "short": DE,
             "long": "status\n" + DE + "\n" + PT * 3}
    d = explained(state)
    assert d["source"] == {"leaf_index": 2, "path": ["long"], "line": 3}
    tie = {"note": EN * 140, "first": DE, "second": DE}
    assert explained(tie)["source"]["path"] == ["first"]
    lines_tie = {"note": EN * 140, "first": DE + "\n" + DE}
    assert explained(lines_tie)["source"]["line"] == 1


class UnprintableKey:
    def __repr__(self):
        raise AssertionError("diagnostics must not stringify arbitrary keys")


@pytest.mark.parametrize("key", [UnprintableKey(), ("tuple",), "k" * 129, 2 ** 80, True])
def test_unrepresentable_path_retains_leaf_index(key):
    d = explained({"note": EN * 140, key: {"customer": DE}})
    assert d["source"] == {"leaf_index": 1, "path": None, "line": 1}


@pytest.mark.parametrize("key", ["k" * 128, -10, "a/b~c\nfield"])
def test_bounded_keys_are_preserved_as_path_components(key):
    d = explained({"note": EN * 140, key: DE})
    assert d["source"]["path"] == [key]


def test_decode_depth_and_window_limits_are_unchanged():
    d = explained({"note": EN * 140, "bad": b"\xff", "good": DE.encode()})
    assert d["source"] == {"leaf_index": 1, "path": ["good"], "line": 1}
    at_limit = DE
    for _ in range(5):
        at_limit = [at_limit]
    d = explained({"note": EN * 140, "deep": at_limit})
    assert d["source"]["path"] == ["deep", 0, 0, 0, 0, 0]
    beyond = {"note": EN * 140, "deep": [at_limit]}
    assert explained(beyond)["source"] is None
    # The fallback reads each line's first 4000 characters, not an unbounded field.
    assert explained({"note": EN * 140, "hidden": EN * 140 + DE})["source"] is None


def test_diagnostics_are_per_call_and_do_not_mutate_state():
    states = [{"note": EN * 140, "customer": DE}, "Turn lights off", "MON DES EST LA"]
    original = copy.deepcopy(states)
    expected = [lang.analyse(s, explain=True) for s in states]
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(lambda s: lang.analyse(s, explain=True), states * 8))
    assert actual == expected * 8
    assert states == original
    actual[0]["diagnostic"]["source"]["path"].append("changed")
    assert lang.analyse(states[0], explain=True) == expected[0]


def test_explain_does_not_run_a_second_detection_pass():
    state = {"note": EN * 140, "customer": DE}
    with patch.object(lang, "_analyse_text", wraps=lang._analyse_text) as detector:
        lang.analyse(state)
        plain_calls = detector.call_args_list
    with patch.object(lang, "_analyse_text", wraps=lang._analyse_text) as detector:
        lang.analyse(state, explain=True)
        assert [c.args[0] for c in detector.call_args_list] == [c.args[0] for c in plain_calls]


@pytest.mark.parametrize("default", ["english", "multilingual"])
def test_router_keeps_fallback_and_never_loads_models(default):
    router = Router(default=default)
    with patch.object(router, "load", side_effect=AssertionError("must not load")):
        plain = router.route("Turn lights off")
        result = router.route("Turn lights off", explain=True)
    assert result["model"] == default
    assert result["detection"].pop("diagnostic")["reason"] == "insufficient_words"
    assert result == plain


@pytest.mark.parametrize("override", [
    {"model": "english"}, {"task": "typed-decisions"}, {"lang": "de"}, {"lang_guess": "en"},
])
def test_explicit_routes_do_not_invent_detector_evidence(override):
    router = Router()
    with patch("laya.router.analyse", side_effect=AssertionError("detector must be skipped")):
        result = router.route(DE, explain=True, **override)
    assert result["detection"] is None


def test_abstaining_hint_runs_detection_once():
    calls = []
    def hint(state):
        calls.append(state)
        return None
    result = Router(lang_guess=hint).route("Turn lights off", explain=True)
    assert calls == ["Turn lights off"]
    assert result["detection"]["diagnostic"]["reason"] == "insufficient_words"


def test_automatic_workflow_selection_skips_detection():
    questions = {key: {"type": "bool"} for key in
                 ("action", "category", "churn_risk", "needs_human", "urgency")}
    with patch("laya.router.analyse", side_effect=AssertionError("detector must be skipped")):
        result = Router(auto_task_detection=True).route(DE, questions, explain=True)
    assert result["workflow"] == "customer_service"
    assert result["detection"] is None


def test_hooks_observe_and_may_replace_the_actual_decision():
    seen = []
    class Hook(BaseHook):
        def on_route(self, ctx):
            seen.append(ctx.decision["detection"]["diagnostic"]["reason"])
            ctx.decision = {"model": "custom", "reason": "hook"}
    assert Router().route("Turn lights off", explain=True, hooks=Hook()) == {
        "model": "custom", "reason": "hook"}
    assert seen == ["insufficient_words"]


def test_batch_uses_same_explanation_and_order():
    states = ["Turn lights off", {"note": EN * 140, "customer": DE}]
    router = Router()
    batch = router.route_batch([{"state": s, "questions": {}} for s in states], explain=True)
    assert batch == [router.route(s, explain=True) for s in states]
    assert all("diagnostic" not in r["detection"] for r in router.route_batch(
        [{"state": s, "questions": {}} for s in states]))


def cli(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "laya.cli", *args],
        cwd=Path(__file__).resolve().parents[1], input=stdin, text=True, encoding="utf-8",
        capture_output=True, timeout=45,
        env={**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})


def test_cli_real_process_json_text_batch_and_interactive():
    result = cli("Turn lights off", "--explain", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["detection"]["diagnostic"]["reason"] == "insufficient_words"
    result = cli("Turn lights off", "--explain")
    assert result.returncode == 0, result.stderr
    assert "Detected" in result.stdout and "insufficient_words" in result.stdout
    result = cli("--batch", "-", "--explain", "--json", stdin="Turn lights off\n" + DE + "\n")
    assert result.returncode == 0, result.stderr
    assert len([json.loads(line)["detection"]["diagnostic"] for line in result.stdout.splitlines()]) == 2
    result = cli("--batch", "-", "--explain", stdin="Turn lights off\n")
    assert result.returncode == 0 and "insufficient_words" in result.stdout
    result = cli("--explain", stdin="Turn lights off\nquit\n")
    assert result.returncode == 0 and "insufficient_words" in result.stdout


@pytest.mark.parametrize("args", [("--predict",), ("--preset", "email"), ("--questions", "absent.json")])
def test_cli_rejects_prediction_modes_before_any_loading(args):
    result = cli("Turn lights off", "--explain", *args)
    assert result.returncode == 2
    assert "--explain is for routing only" in result.stderr
    assert result.stdout == ""

