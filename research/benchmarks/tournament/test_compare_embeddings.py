"""Offline behavioral checks for the research comparison; no Hub weights required."""
import copy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from laya.evals import Dataset, Example
from laya.shortlist import predict_shortlist, predict_tournament
from research.benchmarks.tournament import compare_embeddings as benchmark
from research.benchmarks.tournament.diagnostics import (
    fingerprint, outcome, paired_comparison, stage_metrics, token_diagnostics,
)
from research.benchmarks.tournament.embedders import EmbeddingSpec, RoleEmbedder, pool


def question(n):
    return {"q": {"type": "choice", "instructions": "Which?", "criteria": {str(i): None for i in range(n)}}}


class Ranker:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append((copy.deepcopy(questions), dict(kwargs)))
        answers = {}
        for qid, q in questions.items():
            if q.get("type") == "choice":
                chosen = max(q["criteria"], key=int)
                answers[qid] = {"type": "choice", "choice": chosen, "answer_confidence": .7}
            else:
                answers[qid] = {"type": "noul", "value": True}
        return {"answers": answers, "usage": {"input_tokens": 17}}


def embedding(texts):
    return np.array([[1., 0.]] + [[float(t), 1.] for t in texts[1:]])


@pytest.mark.parametrize("n,k,calls", [(8, 20, 1), (77, 16, 1), (77, 20, 2), (257, 257, 3)])
def test_composition_uses_exact_rounds_and_keeps_both_metadata(n, k, calls):
    agent = Ranker()
    runner = benchmark.Predictor(agent, "combined")
    qs = question(n)
    before = copy.deepcopy(qs)
    result = benchmark.predict_strategy(runner, "request", qs, embedding, k)
    assert result["answers"]["q"]["choice"] == str(n - 1)
    assert len(agent.calls) == calls
    assert result["shortlist"]["q"]["n"] == n
    assert result["tournament"]["q"]["n"] == min(n, k)
    assert qs == before


def test_composition_matches_direct_helpers_and_preserves_kwargs():
    qs = question(77)
    qs["other"] = {"type": "noul", "instructions": "Safe?"}
    original = copy.deepcopy(qs)
    agent = Ranker()
    runner = benchmark.Predictor(agent, "combined")
    result = predict_shortlist(runner, "x", qs, embedding, k=20, max_len=777)
    assert len(agent.calls) == 2
    assert all(kwargs == {"max_len": 777} for _, kwargs in agent.calls)
    assert agent.calls[-1][0]["other"] == qs["other"]
    reference = predict_tournament(Ranker(), "x", {"q": dict(qs["q"], criteria=dict.fromkeys(result["shortlist"]["q"]["labels"])),
                                                   "other": qs["other"]}, max_len=777)
    assert result["answers"] == reference["answers"]
    assert qs == original


def test_stale_order_uses_the_core_warning_and_passthrough_skips_embedding():
    qs = question(77)
    qs["q"]["option_order"] = list(reversed(range(77)))
    with pytest.warns(RuntimeWarning, match="option_order"):
        result = benchmark.predict_strategy(benchmark.Predictor(Ranker(), "combined"), "x", qs, embedding)
    assert len(qs["q"]["option_order"]) == 77
    assert result["shortlist"]["q"]["k"] == 20
    small = question(2)
    small["q"]["option_order"] = [1, 0]
    agent = Ranker()
    def forbidden(_):
        raise AssertionError("passthrough must not embed")
    benchmark.predict_strategy(benchmark.Predictor(agent, "combined"), "x", small, forbidden)
    assert agent.calls[0][0] == small


def test_role_prefixes_survive_deduplication_partial_hits_and_shared_text():
    spec = EmbeddingSpec("test", "a" * 40, "mean", 2, 20, "Q:", "D:")
    calls = []
    def encode(texts):
        calls.append(texts)
        return [[1., 0.] if t.startswith("Q:") else [0., 1.] for t in texts]
    embed = RoleEmbedder(encode, spec)
    first = embed(["same", "same", "other"])
    second = embed(["same", "new", "same"])
    assert calls == [["Q:same", "D:same", "D:other"], ["D:new"]]
    np.testing.assert_array_equal(first, [[1., 0.], [0., 1.], [0., 1.]])
    np.testing.assert_array_equal(first[0], second[0])
    assert embed([]).shape == (0, 2)
    embed.cache_clear()
    np.testing.assert_array_equal(embed(["same", "same", "other"]), first)


def test_separate_model_caches_and_option_warming():
    spec = EmbeddingSpec("test", "b" * 40, "mean", 2, 20, "Q:", "D:")
    calls = []
    def encode(texts):
        calls.append(texts)
        return np.ones((len(texts), 2))
    one, two = RoleEmbedder(encode, spec), RoleEmbedder(encode, spec)
    one.warm_options(["a", "b"])
    one(["q", "a", "b"])
    two(["q", "a", "b"])
    assert calls == [["D:a", "D:b"], ["Q:q"], ["Q:q", "D:a", "D:b"]]


def test_pooling_masks_padding_normalizes_and_rejects_invalid_vectors():
    import torch
    hidden = torch.tensor([[[1., 0.], [0., 1.], [1000., -1000.]]])
    mask = torch.tensor([[1, 1, 0]])
    np.testing.assert_allclose(pool(hidden, mask, "mean").numpy(), [[2**-.5, 2**-.5]], atol=1e-6)
    np.testing.assert_array_equal(pool(hidden, mask, "cls").numpy(), [[1., 0.]])
    for bad in (torch.zeros_like(hidden), torch.full_like(hidden, float("nan"))):
        with pytest.raises(ValueError, match="non-finite or zero"):
            pool(bad, mask, "mean")


def test_first_irreversible_loss_and_count_conservation():
    args = [("a", "b", ["a", "b", "c"], ["b", "c"], ["b"]),
            ("a", "b", ["a", "b", "c"], ["a", "b"], ["b"]),
            ("a", "b", ["a", "b", "c"], ["a", "b"], ["a", "b"]),
            ("a", "a", ["a", "b", "c"], ["a", "b"], ["a", "b"])]
    results = [outcome(*a) for a in args]
    assert results == ["retrieval_miss", "elimination_miss", "final_miss", "correct"]
    metrics = stage_metrics([{"outcome": r} for r in results])
    assert sum(metrics["counts"].values()) == 4
    assert metrics["recall_at_k"] == .75
    assert metrics["survival_on_recalled"] == pytest.approx(2/3)
    assert metrics["accuracy_on_finalists"] == .5
    assert metrics["accuracy"] == .25
    zero = stage_metrics([{"outcome": "retrieval_miss"}])
    assert zero["survival_on_recalled"] is None
    assert zero["accuracy_on_finalists"] is None
    with pytest.raises(ValueError):
        outcome("a", "a", ["a", "b"], ["b"], ["b"])


def test_quality_reuses_shortlist_metrics_and_pairs_trace_by_id():
    qs = question(77)
    cases = [{"id": "one", "state": "one", "state_sha256": fingerprint("one"), "expected": "76"},
             {"id": "two", "state": "two", "state_sha256": fingerprint("two"), "expected": "0"}]
    dataset = Dataset([Example(c["state"], qs, {"q": c["expected"]}, tags=(c["id"],)) for c in cases])
    ident = {"checkpoint_revision": "a" * 40, "embedder_id": "test"}
    result = benchmark.quality(Ranker(), dataset, cases, ident, "combined", embedding, diagnose=False)
    assert result["overall"]["shortlist_recall_at_k"] == .5
    assert result["stages"]["accuracy"] == .5
    assert [c["outcome"] for c in result["cases"]] == ["correct", "retrieval_miss"]
    assert [c["id"] for c in result["cases"]] == ["one", "two"]
    assert all(c["decision_calls"] == 2 for c in result["cases"])
    with pytest.raises(ValueError, match="next planned case"):
        benchmark.quality(Ranker(), dataset, list(reversed(cases)), ident, "combined", embedding, diagnose=False)


def test_invalid_decision_fails_before_it_can_win_another_round():
    class Bad(Ranker):
        def predict(self, state, questions, **kwargs):
            return {"answers": {qid: {"type": "choice", "choice": "outside"} for qid in questions}}
    with pytest.raises(ValueError, match="out-of-set"):
        predict_tournament(benchmark.CallRecorder(Bad()), "x", question(77))


def test_timing_contains_embedding_and_every_decision_round(monkeypatch):
    clock = [0.]
    class Timed(Ranker):
        def predict(self, *args, **kwargs):
            clock[0] += 2
            return super().predict(*args, **kwargs)
    class Embed:
        def cache_clear(self):
            pass
        def warm_options(self, _):
            pass
        def cache_info(self):
            return {}
        def __call__(self, texts):
            clock[0] += 3
            return embedding(texts)
    monkeypatch.setattr(benchmark, "synchronize", lambda _: None)
    result = benchmark.latency(Timed(), [{"state": "request"}], question(77), "combined", Embed(),
                               passes=2, clock=lambda: clock[0])
    assert result["passes_ms"] == [[7000.], [7000.]]
    assert (result["strategy"], result["k"], result["group_size"]) == ("combined", 20, 16)


def test_timing_rotates_strategies_before_repeating_a_pass():
    strategies = ["shortlist", "combined", "tournament"]
    schedule = benchmark.timing_schedule(strategies, 3)
    assert [s for i, s in schedule if i == 0] == strategies
    for position in range(3):
        assert {schedule[i * 3 + position][1] for i in range(3)} == set(strategies)
    assert strategies == ["shortlist", "combined", "tournament"]


class Tokenizer:
    mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 1, 2, 3
    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(c) + 10 for c in text]
        return {"input_ids": ids[:max_length] if truncation else ids}


def test_budget_diagnosis_reads_actual_spans_and_keeps_state_out_of_last_option():
    q = {"type": "choice", "instructions": "Which?",
         "criteria": {"shared label %d" % i: None for i in range(77)}}
    low = token_diagnostics(Tokenizer(), "request", q, 1024, 192)
    same = token_diagnostics(Tokenizer(), "request", q, 1024, 324)
    higher = token_diagnostics(Tokenizer(), "request", q, 2048, 1556)
    assert low["tokens_per_option"] == same["tokens_per_option"] == 4
    assert set(low["option_content_tokens"]) == {3}
    assert low["option_span_sha256"] == same["option_span_sha256"]
    assert higher["options_distinct"] == 77
    assert low["options_distinct"] == 1
    changed_state = token_diagnostics(Tokenizer(), "other", q, 1024, 192)
    assert changed_state["option_span_sha256"] == low["option_span_sha256"]
    assert changed_state["input_sha256"] != low["input_sha256"]


def test_infeasible_head_is_not_reported_as_a_measured_zero():
    agent = Ranker()
    agent.cfg, agent.tok = {"max_len": 32, "head_max_len": 192}, Tokenizer()
    result = benchmark.infeasible_budget(agent, "request", question(77), "single")
    assert result["status"] == "infeasible" and "accuracy" not in result
    assert result["diagnostics"]["markers_kept"] < 77
    assert benchmark.infeasible_budget(agent, "request", question(77), "wide") is None
    assert benchmark.infeasible_budget(agent, "request", question(77), "combined") is None


def test_surviving_marker_does_not_imply_surviving_option_body():
    q = {"type": "choice", "instructions": "Which?", "criteria": dict.fromkeys(["long abc", "long xyz"])}
    info = token_diagnostics(Tokenizer(), "request", q, 36, 192)
    assert info["markers_kept"] == 2
    assert info["head_option_content_tokens"] == [9, 9]
    assert info["option_content_tokens"] == [9, 0]
    assert info["empty_option_bodies"] == 1
    assert info["option_span_sha256"]["long xyz"] == fingerprint([])
    assert info["head_option_span_sha256"]["long xyz"] != fingerprint([])


def test_warmups_exclude_duplicate_content_not_just_row_ids(monkeypatch):
    rows = [{"id": str(i), "state_sha256": str(i // 2)} for i in range(12)]
    monkeypatch.setattr(benchmark, "load_cases", lambda *args: (rows, None, None))
    measured = [{"id": "different", "state_sha256": "0"}]
    selected = benchmark.warmup_cases("banking77", measured, 0)
    assert len(selected) == 3 and all(r["state_sha256"] != "0" for r in selected)
    assert benchmark.warmup_cases("banking77", measured, 0) == selected


def reports():
    identity = {"dataset": "test", "split": "test", "dataset_revision": "a",
                "questions_sha256": "b", "checkpoint_revision": "c"}
    report = {"status": "complete", "identity": identity, "cases": [
        {"id": "1", "state_sha256": "one", "expected": "a", "correct": True},
        {"id": "2", "state_sha256": "two", "expected": "b", "correct": False}]}
    return report, copy.deepcopy(report)


def test_pairing_checks_id_content_protocol_and_does_not_depend_on_row_order():
    left, right = reports()
    right["cases"].reverse()
    result = paired_comparison(left, right, samples=100)
    assert result["accuracy_delta"] == 0 and result["interval"] == [0, 0]
    right["cases"][0]["state_sha256"] = "other"
    with pytest.raises(ValueError, match="contents"):
        paired_comparison(left, right, samples=100)
    left, right = reports()
    right["cases"].append(right["cases"][0])
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison(left, right, samples=100)
    left, right = reports()
    right["identity"]["questions_sha256"] = "changed"
    with pytest.raises(ValueError, match="questions_sha256"):
        paired_comparison(left, right, samples=100)


def test_fingerprint_keeps_criteria_order():
    assert fingerprint({"a": None, "b": None}) != fingerprint({"b": None, "a": None})


def test_case_sampling_keeps_source_identity_and_dev_split(monkeypatch):
    seen = []
    def load(split):
        seen.append(split)
        return ["a", "b"], [("row %d" % i, "a") for i in range(20)]
    monkeypatch.setitem(benchmark.DATASETS, "banking77", (load, "Which?"))
    cases, dataset, ident = benchmark.load_cases("banking77", "dev", 5, 42)
    again, _, _ = benchmark.load_cases("banking77", "dev", 5, 42)
    assert cases == again and seen == ["train", "train"]
    assert len(set(c["id"] for c in cases)) == len(cases) == len(dataset.examples) == 5
    assert ident["split"] == "train"
    larger, _, _ = benchmark.load_cases("banking77", "dev", 16, 42)
    assert cases == larger[:5]


def test_narrowed_order_changes_groups_but_not_candidate_membership():
    agent = Ranker()
    runner = benchmark.Predictor(agent, "combined")
    benchmark.predict_strategy(runner, "x", question(77), embedding)
    groups = agent.calls[0][0]
    assert [len(q["criteria"]) for q in groups.values()] == [10, 10]
    ranked = list(groups["0"]["criteria"]) + list(groups["1"]["criteria"])
    assert ranked == [str(i) for i in reversed(range(57, 77))]


def test_recording_does_not_change_decisions():
    qs = question(77)
    one = benchmark.Predictor(Ranker(), "combined", record=False)
    two = benchmark.Predictor(Ranker(), "combined", record=True)
    assert benchmark.predict_strategy(one, "x", qs, embedding) == benchmark.predict_strategy(two, "x", qs, embedding)


def test_json_report_is_atomic_and_rejects_nonfinite(tmp_path):
    path = tmp_path / "report.json"
    benchmark.write_report(path, {"status": "complete"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        benchmark.write_report(path, {"accuracy": float("nan")})
    assert path.read_bytes() == before
def exercise_main(fail_on=None, mode="latency"):
    questions = {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}}
    agent = SimpleNamespace(
        device="cpu", cfg={"temperature": [1., 1., 1.], "temperature_by_options": {"choice:11+": .1}},
        parallel_options=False, temperature=[1., 1., 1.], temperature_raw=[1., 1., 1.],
        temperature_by_options={"choice:11+": .25}, temperature_by_options_raw={"choice:11+": .1})
    calls, saves, progress = [], [], {"retrieval_done": False}

    def latency(agent, cases, questions, strategy, embed, k, group_size, passes):
        assert passes == 1  # main must interleave strategies, not finish all passes at once
        calls.append(strategy)
        if len(calls) == fail_on:
            raise RuntimeError("injected pass failure")
        value = float(len(calls))
        return {"status": "complete", "strategy": strategy, "k": k, "group_size": group_size,
                "passes_ms": [[value]], "cache": [None], "p50_ms": value, "p95_ms": value,
                "rows_per_pass": 1}

    def retrieval(*args):
        progress["retrieval_done"] = True
        return {"accuracy": 1.0, "cases": []}

    argv = ["compare_embeddings.py", "--mode", mode, "--dataset", "banking77", "--split", "test",
            "--strategies", "single,wide,tournament", "--passes", "3", "--out", "unused.json"]
    embedder = SimpleNamespace(identity=lambda: {"repo": "fixture", "revision": "a" * 40})
    with patch.object(sys, "argv", argv), patch.multiple(
            benchmark,
            load_cases=lambda *args: ([{"id": "one", "state": "s", "state_sha256": "s"}],
                                      SimpleNamespace(examples=[SimpleNamespace(questions=questions)]),
                                      {"dataset": "banking77", "split": "test"}),
            environment=lambda: {}, file_fingerprint=lambda _: "hash",
            memory_snapshot=lambda _: {"after_retrieval": progress["retrieval_done"]},
            warmup_cases=lambda *args: [], infeasible_budget=lambda *args: None,
            TextEncoder=lambda *args, **kwargs: object(), RoleEmbedder=lambda *args: embedder,
            truncation_summary=lambda *args: {}, reset_cache=lambda *args: None,
            retrieval=retrieval, latency=latency,
            write_report=lambda path, report: saves.append(copy.deepcopy(report))), \
            patch.object(benchmark.laya, "load", lambda *args, **kwargs: agent), \
            patch("huggingface_hub.snapshot_download", lambda *args, **kwargs: "."):
        try:
            benchmark.main()
        except RuntimeError:
            if fail_on is None:
                raise
    return calls, saves


def test_rotated_passes_and_temperature_provenance():
    calls, saves = exercise_main()
    final = saves[-1]
    assert calls == ["single", "wide", "tournament", "wide", "tournament", "single",
                     "tournament", "single", "wide"]
    assert final["status"] == "complete"
    for strategy in ("single", "wide", "tournament"):
        arm = final["strategies"][strategy]
        assert len(arm["passes_ms"]) == len(arm["cache"]) == 3
        assert arm["status"] == "complete"
    assert final["strategies"]["single"]["passes_ms"] == [[1.], [6.], [8.]]
    assert final["strategies"]["single"]["p50_ms"] == 6.
    assert final["checkpoint"]["config"]["temperature_by_options"] == {"choice:11+": .1}
    assert final["checkpoint"]["effective_temperature_by_options"] == {"choice:11+": .25}
    assert final["checkpoint"]["temperature_clamped"] is True
    for snapshot in saves:
        for arm in snapshot["strategies"].values():
            if len(arm["passes_ms"]) < 3:
                assert arm["status"] == "running"


def test_mid_schedule_failure_preserves_partial_results():
    _, saves = exercise_main(fail_on=5)
    failed = saves[-1]
    assert failed["status"] == "failed"
    assert "injected pass failure" in failed["error"]
    assert {key: len(value["passes_ms"]) for key, value in failed["strategies"].items()} == {
        "single": 1, "wide": 2, "tournament": 1}
    assert all(value["status"] == "running" for value in failed["strategies"].values())


def test_retrieval_memory_is_captured_after_inference():
    _, saves = exercise_main(mode="recall")
    final = saves[-1]
    assert final["status"] == "complete"
    assert final["memory_after_embedding"]["after_retrieval"] is False
    assert final["memory"]["after_retrieval"] is True
