"""Bare-query treatment and reference-identity checks without model downloads."""
import copy
from dataclasses import replace
import hashlib
import json
import sys

import numpy as np
import pytest
import torch

from laya.evals import Dataset, Example, questions_fingerprint
from research.benchmarks.tournament import query_text_diagnostic as diagnostic
from research.benchmarks.tournament.diagnostics import fingerprint
from research.benchmarks.tournament.embedders import EmbeddingSpec, RoleEmbedder


SPEC = EmbeddingSpec("fake", "a" * 40, "mean", 2, 64, "Q:", "D:")


class Encoder:
    def __init__(self, spec=SPEC, *args, **kwargs):
        self.spec = spec
        self.device = "cpu"
        self.model = torch.nn.Linear(1, 1)
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            if self.spec.query_prefix and text.startswith(self.spec.query_prefix):
                vectors.append([1., 0.])
            else:
                value = text.removeprefix(self.spec.option_prefix).split(":")[0]
                vectors.append([float(value), 1.] if value.isdigit() else [1., 0.])
        return np.asarray(vectors, dtype=np.float32)


def data():
    qs = {"q": {"type": "choice", "instructions": "SECRET INSTRUCTION must not enter the bare query",
                 "criteria": dict.fromkeys(map(str, range(25)))}}
    cases = [{"id": "test:0", "state": {"message": "  raw user message\n原文  "}, "expected": "24"},
             {"id": "test:1", "state": {"message": "another request"}, "expected": "0"}]
    for case in cases:
        case["state_sha256"] = fingerprint(case["state"])
    dataset = Dataset([Example(row["state"], qs, {"q": row["expected"]}) for row in cases])
    identity = {"case_ids_sha256": fingerprint([row["id"] for row in cases]),
                "questions_sha256": questions_fingerprint(dataset), "checkpoint_revision": "a" * 40}
    return cases, qs, identity


def test_query_is_exact_bare_message_with_prefix_and_canonical_label_rendering():
    cases, qs, identity = data()
    encoder = Encoder()
    original = copy.deepcopy(cases)
    result = diagnostic.retrieval(cases, qs, identity, RoleEmbedder(encoder, SPEC))
    assert encoder.calls == [["D:" + str(i) for i in range(25)],
                             ["Q:" + cases[0]["state"]["message"]], ["Q:another request"]]
    assert cases == original
    assert result["accuracy"] == .5
    assert result["recall"]["20"] == .5
    assert result["cases"][0]["gold_rank"] == 1
    assert result["cases"][1]["gold_rank"] is None
    assert result["cases"][0]["state_sha256"] == fingerprint(cases[0]["state"])
    assert result["cases"][0]["query_text_sha256"] == hashlib.sha256(cases[0]["state"]["message"].encode()).hexdigest()
    assert result["cases"][0]["prefixed_query_text_sha256"] == hashlib.sha256(("Q:" + cases[0]["state"]["message"]).encode()).hexdigest()
    assert cases[0]["state"]["message"] not in json.dumps(result, ensure_ascii=False)
    assert "SECRET INSTRUCTION" not in str(encoder.calls)


def test_query_equal_to_label_keeps_query_and_option_roles_separate():
    cases, qs, identity = data()
    cases[0]["state"] = {"message": "24"}
    cases[0]["state_sha256"] = fingerprint(cases[0]["state"])
    encoder = Encoder()
    diagnostic.retrieval(cases, qs, identity, RoleEmbedder(encoder, SPEC))
    assert encoder.calls[0][-1] == "D:24"
    assert encoder.calls[1] == ["Q:24"]


def test_minilm_has_no_unrequested_prefix_and_gold_changes_do_not_change_encoder_inputs():
    cases, qs, identity = data()
    spec = replace(SPEC, query_prefix="", option_prefix="")
    first, second = Encoder(spec), Encoder(spec)
    diagnostic.retrieval(cases, qs, identity, RoleEmbedder(first, spec))
    altered = copy.deepcopy(cases)
    altered[0]["expected"] = "12"
    diagnostic.retrieval(altered, qs, identity, RoleEmbedder(second, spec))
    assert first.calls == second.calls
    assert first.calls[1] == [cases[0]["state"]["message"]]


def test_described_labels_keep_existing_rendering():
    cases, qs, identity = data()
    qs["q"]["criteria"]["0"] = "description"
    identity["questions_sha256"] = questions_fingerprint(Dataset([Example(cases[0]["state"], qs, {})]))
    encoder = Encoder()
    diagnostic.retrieval(cases, qs, identity, RoleEmbedder(encoder, SPEC))
    assert encoder.calls[0][0] == "D:0: description"


@pytest.mark.parametrize("corruption", ["state", "hash", "case_ids", "duplicate", "gold", "question", "small"])
def test_invalid_data_fails_before_any_encoding(corruption):
    cases, qs, identity = data()
    if corruption == "state":
        cases[0]["state"]["message"] = 123
    elif corruption == "hash":
        cases[0]["state_sha256"] = "invalid"
    elif corruption == "case_ids":
        cases.reverse()
    elif corruption == "duplicate":
        cases[1]["id"] = cases[0]["id"]
    elif corruption == "gold":
        cases[0]["expected"] = "unknown"
    elif corruption == "question":
        qs["q"]["instructions"] = "changed reference"
    else:
        qs["q"]["criteria"] = dict.fromkeys(map(str, range(20)))
    encoder = Encoder()
    with pytest.raises(ValueError):
        diagnostic.retrieval(cases, qs, identity, RoleEmbedder(encoder, SPEC))
    assert encoder.calls == []


@pytest.mark.parametrize("change", ["device", "dtype"])
def test_actual_tensor_placement_must_match_declared_fp32_device(change):
    cases, qs, identity = data()
    encoder = Encoder()
    encoder.model = encoder.model.to("meta") if change == "device" else encoder.model.double()
    with pytest.raises(ValueError):
        diagnostic.retrieval(cases, qs, identity, RoleEmbedder(encoder, SPEC))
    assert encoder.calls == []


def test_offset_slices_exact_pinned_sample_prefix(monkeypatch):
    cases, qs, identity = data()
    calls = []
    def load(name, split, limit, seed):
        calls.append((name, split, limit, seed))
        return cases, Dataset([Example(row["state"], qs, {}) for row in cases]), identity
    monkeypatch.setattr(diagnostic.fixed, "load_cases", load)
    chosen, dataset, updated = diagnostic.selected_cases("banking77", "dev", 1, 1, 0)
    assert calls == [("banking77", "dev", 2, 0)]
    assert chosen == cases[1:] and dataset.examples[0].state == cases[1]["state"]
    assert updated["case_ids_sha256"] == fingerprint([cases[1]["id"]])
    assert updated["offset"] == 1


def test_cli_completes_without_loading_laya_and_refuses_report_overwrite(monkeypatch, tmp_path):
    cases, qs, identity = data()
    dataset = Dataset([Example(row["state"], qs, {}) for row in cases])
    monkeypatch.setattr(diagnostic, "selected_cases", lambda *args: (cases, dataset, identity))
    monkeypatch.setattr(diagnostic, "TextEncoder", Encoder)
    monkeypatch.setattr(diagnostic.fixed, "environment", lambda: {"test": True})
    monkeypatch.setattr(diagnostic.fixed, "memory_snapshot", lambda device: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("retrieval diagnostic must never load decision weights")
    monkeypatch.setattr(diagnostic.fixed.laya, "load", forbidden)
    path = tmp_path / "bare.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--dataset", "banking77", "--variant", "minilm",
                                    "--split", "test", "--limit", "2", "--out", str(path)])
    diagnostic.main()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["retrieval"]["status"] == "complete"
    assert report["decision_model_loaded"] is False
    assert report["schema"] == diagnostic.SCHEMA
    assert report["reference_checkpoint"]["role"].endswith("no decision weights loaded")
    assert report["retrieval"]["cases"][0]["id"] == cases[0]["id"]
    before = path.read_bytes()
    with pytest.raises(SystemExit) as failure:
        diagnostic.main()
    assert failure.value.code == 2 and path.read_bytes() == before
