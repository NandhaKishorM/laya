"""Offline contracts for lossless evidence and matched result analysis."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("report_tools", Path(__file__).with_name("report_tools.py"))
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


def report():
    identity = dict(dataset="example", split="test", dataset_revision="data-pin",
                    questions_sha256="question-pin", checkpoint_revision="checkpoint-pin")
    cases = [
        dict(id="a", state_sha256="hash-a", expected="left", choice="left", correct=True,
             shortlist=["right", "left"], finalists=["left"], decision_order=["left", "right"],
             decision_calls=2, outcome="correct", confidence=None),
        dict(id="b", state_sha256="hash-b", expected="right", choice="left", correct=False,
             shortlist=["right", "left"], finalists=["left"], decision_calls=2, outcome="elimination_miss"),
    ]
    return dict(schema="raw", status="complete", environment={}, identity=identity,
                strategies={"combined": dict(status="complete", identity=identity, cases=cases)})


def test_compact_keeps_candidate_order_and_absence_separate_from_null():
    raw = report()
    restored = tools.expand(tools.compact(raw, "source-hash"))
    assert restored["strategies"]["combined"]["cases"] == raw["strategies"]["combined"]["cases"]
    assert restored["strategies"]["combined"]["cases"][0]["shortlist"] == ["right", "left"]
    assert "confidence" not in restored["strategies"]["combined"]["cases"][1]


def test_compact_trace_omission_keeps_original_measurements_untouched():
    raw = report()
    raw["strategies"]["combined"]["cases"][0]["calls"] = [
        {"questions": {"q": {"truncated": True, "state_tokens_dropped": 3}}}]
    original = copy.deepcopy(raw)
    evidence = tools.compact(raw, "source-hash")
    assert raw == original
    arm = tools.expand(evidence)["strategies"]["combined"]
    assert arm["trace_counts"] == {"questions": 1, "truncated_questions": 1, "state_tokens_dropped": 3}
    assert "calls" not in arm["cases"][0]


@pytest.mark.parametrize("index", [-1, 500, True])
def test_corrupt_candidate_references_rejected(index):
    evidence = tools.compact(report(), "source-hash")
    evidence["orders"][0][0] = index
    with pytest.raises(ValueError, match="invalid evidence reference"):
        tools.expand(evidence)


def test_unknown_case_data_is_not_silently_dropped():
    raw = report()
    raw["strategies"]["combined"]["cases"][0]["state"] = {"private_text": "not for publishing"}
    with pytest.raises(ValueError, match="unrecognized case"):
        tools.compact(raw, "source-hash")


def test_pair_recomputes_outcomes_and_rejects_changed_content():
    left, right = report(), report()
    right["strategies"]["combined"]["cases"][0].update(
        choice="right", correct=False, finalists=["right"], outcome="elimination_miss")
    result = tools.pair(left, right, "combined", "combined")
    assert (result["gained"], result["lost"], result["accuracy_delta"]) == (1, 0, .5)
    right["strategies"]["combined"]["cases"][0]["state_sha256"] = "different-content"
    with pytest.raises(ValueError, match="case contents"):
        tools.pair(left, right, "combined", "combined")


def test_pair_checks_boolean_correctness_and_duplicate_ids():
    raw = report()
    raw["strategies"]["combined"]["cases"][0]["correct"] = 1
    with pytest.raises(ValueError, match="correctness"):
        tools.summary(raw)
    raw = report()
    raw["strategies"]["combined"]["cases"][1]["id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        tools.summary(raw)


def test_mixed_query_requires_explicit_opt_in():
    left, right = report(), report()
    right["query_format"] = "bare message"
    with pytest.raises(ValueError, match="--mixed-query"):
        tools.pair(left, right, "combined", "combined")
    assert tools.pair(left, right, "combined", "combined", mixed_query=True)["interpretation"] == "mixed-query pipeline comparison"


def timing():
    raw = report()
    raw.pop("strategies")
    raw["timing"] = dict(status="complete", passes=3, case_ids=["a", "b"], arms={"top1": {}},
                         scope="cold query; warm labels",
                         samples=[dict(id=i, arm="top1", **{"pass": p}, ms=float(p + 1))
                                  for p in range(3) for i in ["a", "b"]])
    return raw


def test_timing_counts_cases_separately_from_repeated_passes():
    result = tools.summary(timing())["timing"]["top1"]
    assert result == {"p50_ms": 2.0, "p95_ms": 3.0, "distinct_cases": 2, "passes": 3, "observations": 6}


def test_duplicate_or_invalid_timing_does_not_become_a_distribution():
    raw = timing()
    raw["timing"]["samples"][-1] = raw["timing"]["samples"][0]
    with pytest.raises(ValueError, match="duplicate timing"):
        tools.summary(raw)
    raw = timing()
    raw["timing"]["samples"][0]["ms"] = float("nan")
    with pytest.raises(ValueError, match="invalid timing"):
        tools.summary(raw)


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "evidence.json"
    tools.write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        tools.write_new(path, {"replacement": True})


def test_compact_timing_retains_content_manifest_and_provenance(tmp_path):
    raw = timing()
    raw["rows"] = [dict(id="a", state_sha256="hash-a"), dict(id="b", state_sha256="hash-b")]
    path = tmp_path / "compact.json"
    tools.write_new(path, tools.compact(raw, "original-report-sha"))
    restored = tools.read(path)
    assert restored["rows"] == raw["rows"]
    assert restored["timing"]["samples"] == raw["timing"]["samples"]
    result = tools.summary(restored)
    assert result["evidence_provenance"]["source_report_sha256"] == "original-report-sha"
    assert "token-level" in result["evidence_provenance"]["omitted"]


@pytest.mark.parametrize("outcome", ["correct", "other", None])
def test_stage_summary_rejects_false_or_missing_outcome(outcome):
    raw = report()
    row = raw["strategies"]["combined"]["cases"][1]
    if outcome is None:
        row.pop("outcome")
    else:
        row["outcome"] = outcome
    with pytest.raises(ValueError, match="outcome differs"):
        tools.summary(raw)


def test_stage_summary_rejects_choice_outside_finalists():
    raw = report()
    raw["strategies"]["combined"]["cases"][0]["finalists"] = ["right"]
    with pytest.raises(ValueError, match="candidate sets"):
        tools.summary(raw)


def test_retrieval_arm_status_and_identity_cannot_be_replaced_by_outer_metadata():
    raw = report()
    raw["retrieval"] = raw.pop("strategies")["combined"]
    raw["retrieval"]["status"] = "running"
    with pytest.raises(ValueError, match="arm is incomplete"):
        tools.summary(raw)
    raw["retrieval"]["status"] = "complete"
    raw["retrieval"]["identity"] = dict(raw["identity"], dataset="other")
    with pytest.raises(ValueError, match="arm identity"):
        tools.summary(raw)


def test_arm_cases_must_match_report_content_manifest():
    raw = report()
    raw["rows"] = [dict(id="a", state_sha256="different"), dict(id="b", state_sha256="hash-b")]
    with pytest.raises(ValueError, match="row manifest"):
        tools.summary(raw)


@pytest.mark.parametrize("field", ["rows_per_pass", "independent_requests"])
def test_timing_rejects_false_independent_sample_count(field):
    raw = timing()
    raw["timing"][field] = 6  # Two case identities repeated three times are not six cases.
    with pytest.raises(ValueError, match="request count"):
        tools.summary(raw)


def test_pair_retains_both_source_fingerprints(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    tools.write_new(left, tools.compact(report(), "left-source"))
    tools.write_new(right, tools.compact(report(), "right-source"))
    result = tools.pair(tools.read(left), tools.read(right), "combined", "combined")
    assert result["candidate_provenance"]["source_report_sha256"] == "left-source"
    assert result["baseline_provenance"]["source_report_sha256"] == "right-source"


def test_compact_keeps_explicit_single_pass_alias():
    raw = report()
    raw["strategies"]["combined"].update(equivalent_to="shortlist", degenerate_single_pass=True)
    restored = tools.expand(tools.compact(raw, "source-hash"))
    assert restored["strategies"]["combined"]["equivalent_to"] == "shortlist"
    assert restored["strategies"]["combined"]["degenerate_single_pass"] is True


def grid_report(variant, k, correct=100):
    import select_configurations as selector

    rows = [dict(id=f"banking77:train:{i}", state_sha256=selector.fingerprint(i)) for i in range(200)]
    identity = dict(dataset="banking77", split="train", dataset_revision=selector.DATASET_REVISION,
                    checkpoint_revision=selector.CHECKPOINT,
                    case_ids_sha256=selector.fingerprint([r["id"] for r in rows]))
    result = dict(schema="laya-dedicated-shortlist/amendment-v2", status="complete", environment={},
                  source_sha256={}, checkpoint={}, parameters=selector.parameters(variant=variant, k=k),
                  identity=identity, rows=rows, embedder=copy.deepcopy(selector.SPECS[variant]),
                  ordering=dict(seed=None, global_labels=None, policy="natural"), strategies={})
    labels = [f"label{i}" for i in range(k)]
    for method in selector.METHODS:
        calls = 2 if method == "combined" and k > 16 else 1
        final = labels[:2] if calls == 2 else labels
        cases = [dict(row, expected=labels[0], choice=labels[0] if i < correct else labels[1],
                      correct=i < correct, shortlist=labels, finalists=final, decision_order=labels,
                      decision_calls=calls, outcome="correct" if i < correct else "final_miss")
                 for i, row in enumerate(rows)]
        result["strategies"][method] = dict(status="complete", k=k, group_size=16, cases=cases,
            identity=dict(identity, embedder_id=selector.fingerprint(result["embedder"])))
    if k <= 16:
        result["strategies"]["combined"].update(equivalent_to="shortlist", degenerate_single_pass=True)
    return result


def save_bundle(directory, documents):
    files = []
    for name, value in documents.items():
        data = (json.dumps(value, allow_nan=False) + "\n").encode("utf-8")
        (directory / name).write_bytes(data)
        item = dict(file=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if value.get("schema") == tools.SCHEMA:
            item["source_report_sha256"] = value["source_report_sha256"]
            item["phase"] = "fixed-study" if name.startswith("dev200-") else "amendment"
        files.append(item)
    (directory / "manifest.json").write_text(json.dumps(dict(schema="laya-study-manifest/1", files=files)),
                                               encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    import select_configurations as selector

    documents, grid, hashes = {}, {}, {}
    for variant in selector.VARIANTS:
        for k in selector.KS:
            cell = grid[variant, k] = grid_report(variant, k)
            name = f"grid-dev200-{variant}-k{k}.json"
            source = hashes[f"{variant}/k{k}"] = selector.fingerprint(cell)
            documents[name] = tools.compact(cell, source)
    documents["selected-configurations.json"] = dict(selected=selector.choose(grid), grid_sha256=hashes)
    documents["latency-example.json"] = tools.compact(timing(), "f" * 64)
    documents["selected-example.json"] = tools.compact(report(), "d" * 64)
    documents["order-example.json"] = tools.compact(report(), "c" * 64)
    documents["measurement.json"] = dict(schema="laya-study-measurement/1", complete_report_files=21,
        timing_observations=6, completed_matrix_jobs=21, timing_unique_requests_per_cell=2)
    for dataset in ("banking77", "clinc150", "massive"):
        for model, correct in (("bge", 80), ("minilm", 100), ("nomic", 120)):
            raw = grid_report("minilm", 20, correct)
            raw["parameters"].update(dataset=dataset, embedder=model)
            raw["identity"].update(dataset=dataset, split="train" if dataset == "banking77" else "validation")
            for arm in raw["strategies"].values():
                arm["identity"].update(raw["identity"])
            documents[f"dev200-{dataset}-{model}.json"] = tools.compact(raw, "e" * 64)
    documents["model-selection.json"] = dict(macro_shortlist_accuracy=dict(bge=.4, minilm=.5),
        based_only_on="development", tie_break=None, selected="minilm")
    documents["measurement.json"]["complete_report_files"] += 9
    documents["planned-samples.json"] = {}
    documents["frozen-protocol.json"] = {}
    save_bundle(tmp_path, documents)
    return tmp_path, documents


def test_verify_bundle_recounts_grid_and_repeated_timing(bundle):
    directory, _documents = bundle
    result = tools.verify(directory)
    assert (result["manifest_files"], result["compact_reports"], result["quality_reports"],
            result["timing_reports"], result["quality_case_records"], result["timing_observations"]) == (
                35, 30, 29, 1, 10804, 6)
    assert result["grid_selection"]["selected"]["nomic"]["combined"] == dict(variant="nomic-qd", k=5, correct=100)
    assert any("cannot be re-audited" in limit for limit in result["limits"])
    assert result["measurement_inventory_matches"] is True


@pytest.mark.parametrize("name", ["latency-example.json", "selected-example.json", "order-example.json"])
def test_verify_rejects_incomplete_bundle_even_when_manifest_hashes_match(bundle, name):
    directory, documents = bundle
    documents.pop(name)
    (directory / name).unlink()
    save_bundle(directory, documents)
    with pytest.raises(ValueError, match="measurement inventory differs"):
        tools.verify(directory)


@pytest.mark.parametrize("name", ["model-selection.json", "planned-samples.json", "frozen-protocol.json"])
def test_verify_rejects_missing_metadata_even_when_manifest_hashes_match(bundle, name):
    directory, documents = bundle
    documents.pop(name)
    (directory / name).unlink()
    save_bundle(directory, documents)
    with pytest.raises(ValueError, match="missing required metadata: " + name):
        tools.verify(directory)


def test_verify_detects_modified_file_before_reading_outcomes(bundle):
    directory, _documents = bundle
    path = directory / "grid-dev200-minilm-k5.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file SHA256"):
        tools.verify(directory)


def test_verify_checks_source_hash_independently_of_compact_hash(bundle):
    directory, documents = bundle
    documents["grid-dev200-minilm-k5.json"]["source_report_sha256"] = "0" * 64
    save_bundle(directory, documents)
    with pytest.raises(ValueError, match="selection grid source hashes"):
        tools.verify(directory)


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r["parameters"].update(split="test"), "grid parameters"),
    (lambda r: r["embedder"].update(query_prefix="changed"), "model contract"),
    (lambda r: r["strategies"]["combined"]["cases"][0].update(expected="label1", choice="label1"), "gold labels"),
])
def test_verify_grid_rejects_mismatched_contracts_and_golds(bundle, mutation, message):
    directory, documents = bundle
    name = "grid-dev200-minilm-k20.json"
    source = documents[name]["source_report_sha256"]
    raw = tools.expand(documents[name])
    mutation(raw)
    documents[name] = tools.compact(raw, source)
    save_bundle(directory, documents)
    with pytest.raises(ValueError, match=message):
        tools.verify(directory)


def test_verify_recomputes_selection_instead_of_trusting_saved_scores(bundle):
    directory, documents = bundle
    documents["selected-configurations.json"]["selected"]["minilm"]["shortlist"]["correct"] += 1
    save_bundle(directory, documents)
    with pytest.raises(ValueError, match="saved selection differs"):
        tools.verify(directory)


@pytest.mark.parametrize("name", ["../outside.json", "C:\\outside.json", "manifest.json"])
def test_verify_rejects_unsafe_manifest_paths(bundle, name):
    directory, _documents = bundle
    path = directory / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["files"][0]["file"] = name
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid manifest filename"):
        tools.verify(directory)


def test_verify_recounts_three_dataset_pilot_and_excludes_reference_from_choice(bundle):
    directory, documents = bundle
    result = tools.verify(directory)["pilot_selection"]
    assert result["selected"] == "minilm"
    assert result["macro_shortlist_accuracy"]["nomic"] == pytest.approx(.6)
    documents["model-selection.json"]["macro_shortlist_accuracy"]["minilm"] = .7
    save_bundle(directory, documents)
    with pytest.raises(ValueError, match="pilot macro accuracy"):
        tools.verify(directory)
