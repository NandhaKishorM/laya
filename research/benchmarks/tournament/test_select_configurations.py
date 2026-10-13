"""Offline contracts for the development-only selection artifact."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_configurations as selection  # noqa: E402


def report(variant="minilm", k=5, correct=100):
    rows = [dict(id=f"banking77:train:{i}", state_sha256=selection.fingerprint(i)) for i in range(200)]
    identity = dict(dataset="banking77", split="train", dataset_revision=selection.DATASET_REVISION,
                    questions_sha256="a" * 64, case_ids_sha256=selection.fingerprint([r["id"] for r in rows]),
                    seed=0, sampling="seeded permutation prefix", offset=0, checkpoint_revision=selection.CHECKPOINT)
    placement = dict(requested_device="cpu", cpu_fallback_count=0, amp_failures=0,
                     autocast_dtype="torch.float32", amp_enabled=False,
                     laya=dict(device="cpu", parameter_dtypes=["torch.float32"], buffer_dtypes=["torch.float32"]),
                     encoder=dict(device="cpu", parameter_dtypes=["torch.float32"],
                                  buffer_dtypes=["torch.int64" if variant == "minilm" else "torch.float32"]))
    source = {name: selection.fingerprint(name) for name in selection.SOURCES}
    result = dict(schema="laya-dedicated-shortlist/amendment-v2", status="complete",
                  parameters=selection.parameters(variant=variant, k=k), source_sha256=source,
                  environment=dict(code_commit="b" * 40, python="3.11", platform="offline fixture",
                                   versions=dict(torch="test", transformers="test", numpy="test"),
                                   runner_sha256={name: source[name] for name in selection.SOURCES[2:]}),
                  checkpoint=dict(revision=selection.CHECKPOINT,
                                  sha256={name: selection.fingerprint(name) for name in
                                          ("model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json")}),
                  embedder=copy.deepcopy(selection.SPECS[variant]), rows=rows, identity=identity,
                  runtime=dict(requested_device="cpu", actual_laya_device="cpu", actual_embedder_device="cpu",
                               torch_threads=4, torch_interop_threads=8,
                               placement_before=placement, placement_after=copy.deepcopy(placement)),
                  ordering=dict(seed=None, global_labels=None, policy="natural"), strategies={})
    labels = [f"label{i}" for i in range(k)]
    for method in selection.METHODS:
        cases = []
        for i, row in enumerate(rows):
            if method == "combined" and k > 16:
                n = (k + 15) // 16
                groups = [labels[j * k // n:(j + 1) * k // n] for j in range(n)]
                final = [group[0] for group in groups]
                all_groups = [groups, [final]]
            else:
                final, all_groups = labels, [[labels]]
            choice = final[0] if i < correct else final[-1]
            cases.append(dict(row, expected="label0", choice=choice, correct=choice == "label0",
                              confidence=0.7, shortlist=labels, decision_order=labels, finalists=final,
                              gold_rank=1, decision_calls=len(all_groups),
                              outcome="correct" if choice == "label0" else "final_miss",
                              groups=all_groups,
                              calls=[dict(questions={f"q{j}": dict(labels=group) for j, group in enumerate(groups)})
                                     for groups in all_groups]))
        result["strategies"][method] = dict(status="complete", strategy=method, k=k, group_size=16,
                                            identity=dict(identity, embedder_id=selection.fingerprint(result["embedder"])),
                                            runtime=copy.deepcopy(placement), stages=selection.stage_summary(cases), cases=cases)
    if k <= 16:
        result["strategies"]["combined"].update(equivalent_to="shortlist", degenerate_single_pass=True)
    return result


def write_grid(directory):
    for variant in selection.VARIANTS:
        for k in selection.KS:
            (directory / f"grid-dev200-{variant}-k{k}.json").write_text(
                json.dumps(report(variant, k)), encoding="utf-8")


@pytest.fixture
def grid_dir(tmp_path):
    write_grid(tmp_path)
    return tmp_path


def test_complete_grid_recomputes_scores_and_ties(grid_dir):
    grid, hashes, _common = selection.load_grid(grid_dir)
    assert len(hashes) == 18
    chosen = selection.choose(grid)
    assert chosen == {
        "minilm": {m: dict(variant="minilm", k=5, correct=100) for m in selection.METHODS},
        "nomic": {m: dict(variant="nomic-qd", k=5, correct=100) for m in selection.METHODS}}
    # A better score dominates size; size wins before Q/D when the scores tie.
    grid["nomic-qq", 12] = report("nomic-qq", 12, correct=101)
    grid["nomic-qd", 20] = report("nomic-qd", 20, correct=101)
    assert selection.choose(grid)["nomic"]["combined"] == dict(variant="nomic-qq", k=12, correct=101)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(status="running"),
    lambda r: r["parameters"].update(split="test"),
    lambda r: r["parameters"].update(offset=200),
    lambda r: r["parameters"].update(seed=False),
    lambda r: r["embedder"].update(query_prefix="search_document: "),
    lambda r: r["source_sha256"].pop("interleaved.py"),
    lambda r: r["checkpoint"].update(revision="a" * 40),
    lambda r: r["identity"].update(case_ids_sha256="f" * 64),
    lambda r: r["runtime"].update(actual_embedder_device="cuda:0"),
    lambda r: r["runtime"]["placement_after"].update(cpu_fallback_count=1),
    lambda r: r["strategies"]["shortlist"]["identity"].update(embedder_id="f" * 64),
    lambda r: r["strategies"]["shortlist"]["cases"][0].update(correct=1),
    lambda r: r["strategies"]["shortlist"]["cases"][0].update(choice="outside"),
    lambda r: r["strategies"]["shortlist"]["cases"][0].update(outcome="retrieval_miss"),
    lambda r: r["strategies"]["shortlist"]["cases"][0].update(gold_rank=True),
    lambda r: r["strategies"]["shortlist"]["stages"].update(accuracy=1),
    lambda r: r["strategies"]["combined"].pop("equivalent_to"),
    lambda r: r["strategies"]["combined"]["cases"][0].update(confidence=0.1),
    lambda r: r["strategies"]["shortlist"]["cases"][0]["calls"][0]["questions"]["q0"].update(labels=["other"]),
])
def test_reject_corrupt_or_mismatched_report(mutation):
    cell = report()
    mutation(cell)
    with pytest.raises((ValueError, KeyError)):
        selection.validate_report(cell, "minilm", 5)


def test_balanced_combined_rounds_and_alias():
    cell = report(k=48)
    selection.validate_report(cell, "minilm", 48)
    cell["strategies"]["combined"]["cases"][0]["finalists"] = ["label0", "label0", "label32"]
    with pytest.raises(ValueError, match="finalists"):
        selection.validate_report(cell, "minilm", 48)


def test_missing_or_extra_cell_rejected(grid_dir):
    path = grid_dir / "grid-dev200-minilm-k5.json"
    path.rename(grid_dir / "grid-dev200-minilm-k6.json")
    with pytest.raises(ValueError, match="grid"):
        selection.load_grid(grid_dir)


def test_same_identity_required_across_cells(grid_dir):
    path = grid_dir / "grid-dev200-nomic-qq-k48.json"
    cell = selection.read(path)
    cell["environment"]["code_commit"] = "c" * 40
    path.write_text(json.dumps(cell), encoding="utf-8")
    with pytest.raises(ValueError, match="execution identity"):
        selection.load_grid(grid_dir)


def test_freeze_refuses_overwrite_and_preserves_timestamp(grid_dir):
    path = grid_dir / "selection.json"
    artifact = selection.freeze(grid_dir, path)
    original = path.read_bytes()
    assert selection.read(path)["saved_before_new_test_utc"] == artifact["saved_before_new_test_utc"]
    with pytest.raises(ValueError, match="overwrite"):
        selection.freeze(grid_dir, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("filename", ["selected-test500-minilm-k5.json", "selected-test500-partial.json.tmp",
                                    "latency-selected-banking77-minilm.json"])
def test_no_first_selection_after_test_evidence(grid_dir, filename):
    (grid_dir / filename).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="original selection timestamp"):
        selection.freeze(grid_dir, grid_dir / "selection.json")
    assert not (grid_dir / "selection.json").exists()


def test_evidence_in_output_directory_also_blocks(grid_dir, tmp_path_factory):
    output = tmp_path_factory.mktemp("selection")
    (output / "selected-test500-existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="original selection timestamp"):
        selection.freeze(grid_dir, output / "selection.json")


def test_test_evidence_appearing_during_validation_blocks(grid_dir):
    real_loader = selection.load_grid
    def load_and_add_test(directory):
        result = real_loader(directory)
        (directory / "selected-test500-late.json").write_text("{}", encoding="utf-8")
        return result
    with mock.patch.object(selection, "load_grid", side_effect=load_and_add_test):
        with pytest.raises(ValueError, match="original selection timestamp"):
            selection.freeze(grid_dir, grid_dir / "selection.json")
    assert not (grid_dir / "selection.json").exists()


def test_commands_are_readonly_and_use_matched_top1(grid_dir):
    path = grid_dir / "selection.json"
    frozen = selection.freeze(grid_dir, path)
    original = path.read_bytes()
    # Existing results never prevent reading the historical selection, only creating one.
    (grid_dir / "selected-test500-existing.json").write_text("{}", encoding="utf-8")
    plan = selection.command_plan(grid_dir, frozen)
    assert len(plan["commands"]) == 6  # Two dev, two test, two timing.
    assert len(plan["arms_files"]) == 2
    assert plan["selection_time"] == frozen["saved_before_new_test_utc"]
    assert path.read_bytes() == original
    for command in plan["commands"]:
        args = command["argv"]
        assert args[:2] == ["python", "research/benchmarks/tournament/amended_comparison.py"]
        if command["report"].startswith("selected-dev"):
            assert args[args.index("--offset") + 1] == "200"
            assert args[args.index("--limit") + 1] == "400"
        if command["report"].startswith("selected-test"):
            assert args[args.index("--strategies") + 1] == "top1,shortlist,combined"
    for arms in plan["arms_files"].values():
        assert len(arms) == 4 and arms[1]["strategy"] == "tournament"
        assert arms[-1]["strategy"] == "top1" and arms[-1]["k"] == 1


def test_commands_reject_tampered_selection(grid_dir):
    frozen = selection.freeze(grid_dir, grid_dir / "selection.json")
    frozen["selected"]["minilm"]["shortlist"]["k"] = 48
    with pytest.raises(ValueError, match="selection differs"):
        selection.command_plan(grid_dir, frozen)


def test_json_duplicate_keys_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for contents in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError):
            selection.read(path)


def test_cli_invalid_modes_are_errors(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(selection.__file__)),
                             "--reports", str(tmp_path), "--commands"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "--selection" in result.stderr


def test_grid_change_during_validation_cannot_be_frozen(grid_dir):
    real_loader = selection.load_grid
    def load_and_change_grid(directory):
        result = real_loader(directory)
        with (directory / "grid-dev200-minilm-k5.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        return result
    with mock.patch.object(selection, "load_grid", side_effect=load_and_change_grid):
        with pytest.raises(ValueError, match="grid changed"):
            selection.freeze(grid_dir, grid_dir / "selection.json")
    assert not (grid_dir / "selection.json").exists()
