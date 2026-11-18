"""Model-free checks for the MASSIVE Chinese routing data recipe."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from research.scripts import build_massive_zh_routing as recipe


QUESTION = {"type": "choice", "instructions": "Route this command", "criteria": {
    label: scenario for scenario, label in recipe.SCENARIO_LABELS.items()}}


def source_row(source_id, split, scenario, text, *, clean=True):
    return {"id": source_id, "locale": "zh-CN", "partition": split, "scenario": scenario, "utt": text,
            "judgments": [{"intent_score": 1 if clean else 0}]}


def sample_rows():
    return [
        source_row("1", "train", "calendar", "Book a meeting"),
        source_row("2", "train", "alarm", "Set an alarm"),
        source_row("3", "test", "alarm", "set  an  alarm"),
        source_row("4", "train", "music", "Play a song"),
        source_row("5", "dev", "weather", "Will it rain"),
        source_row("6", "test", "transport", "Find a bus"),
        source_row("7", "train", "audio", "Turn it up", clean=False),
        source_row("8", "dev", "weather", "Is it sunny"),
    ]


def test_original_benchmark_contract_is_reused():
    question, held_out = recipe.benchmark_contract(recipe.BENCHMARK)
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(recipe.SCENARIO_LABELS.values())
    assert held_out


def test_benchmark_overlap_and_cross_split_duplicates_are_excluded():
    rows, audit = recipe.build_rows(sample_rows(), QUESTION,
                                    {"book a meeting", "will it rain", "find a bus"})
    assert [r["tags"][-1] for r in rows["train"]] == ["source_id:4"]
    assert [r["tags"][-1] for r in rows["dev"]] == ["source_id:8"]
    assert [r["tags"][-1] for r in rows["test"]] == ["source_id:3"]
    assert audit["benchmark_overlap_excluded_train"] == 1
    assert audit["benchmark_overlap_excluded_dev"] == 1
    assert audit["benchmark_overlap_excluded_test"] == 1
    assert audit["duplicate_text_excluded_train"] == 1
    assert audit["quality_excluded"] == 1
    targets = list(recipe.hard_target_rows(rows["train"]))
    assert targets[0]["target_source"] == "hard_label_one_hot"
    assert targets[0]["source_id"] == "4"
    assert targets[0]["license"] == "CC BY 4.0"
    assert targets[0]["gold"]["route"]["label"] == "音乐点播"
    assert sum(targets[0]["gold"]["route"]["probabilities"].values()) == 1.0


def test_conflicts_and_duplicate_ids_fail_loudly():
    rows = sample_rows()
    rows.append(source_row("9", "train", "weather", "Play a song"))
    with pytest.raises(ValueError, match="conflicting labels"):
        recipe.build_rows(rows, QUESTION, set())
    rows = sample_rows()
    rows.append(source_row("4", "train", "music", "Play another song"))
    with pytest.raises(ValueError, match="duplicate MASSIVE source id"):
        recipe.build_rows(rows, QUESTION, set())


def test_archive_hash_is_pinned(tmp_path):
    archive = tmp_path / "wrong.tar.gz"
    archive.write_bytes(b"not MASSIVE")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        recipe.obtain_archive(archive, download=False)


def make_archive(path):
    payload = "".join(json.dumps(row) + "\n" for row in sample_rows()).encode("utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo(recipe.SOURCE_MEMBER)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))


def test_build_is_deterministic_and_hard_targets_are_opt_in(tmp_path):
    archive = tmp_path / "source.tar.gz"
    make_archive(archive)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest_a = recipe.build(archive, first, export_hard_targets=True)
    manifest_b = recipe.build(archive, second, export_hard_targets=True)
    assert manifest_a == manifest_b
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "train_hard_targets.jsonl").read_bytes() == (second / "train_hard_targets.jsonl").read_bytes()
    assert manifest_a["files"]["train_hard_targets.jsonl"]["target_source"].startswith("hard_label_one_hot")
    with pytest.raises(ValueError, match="must be empty"):
        recipe.build(archive, first)
    only_eval = tmp_path / "only_eval"
    recipe.build(archive, only_eval)
    assert not (only_eval / "train_hard_targets.jsonl").exists()
    assert sorted(path.name for path in only_eval.glob("*.jsonl")) == [
        "dev_eval.jsonl", "test_eval.jsonl", "train_eval.jsonl"]


def test_exported_labels_match_choice_criteria(tmp_path):
    archive = tmp_path / "source.tar.gz"
    make_archive(archive)
    output = tmp_path / "out"
    recipe.build(archive, output)
    for path in Path(output).glob("*_eval.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["expected"]["route"] in row["questions"]["route"]["criteria"]
