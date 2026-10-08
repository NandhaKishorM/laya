# Changelog

All notable changes to the `laya-onnx` crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `laya-rust-v<version>`.

## [0.1.0] - unreleased

First release: a port of the Laya Python SDK, checked against regenerated laya (Python) 0.4.0
goldens for the supported paths (see the known gaps below).

### Added

- Advisory parity CI on every push and PR: regenerate fixtures from the current Python checkout,
  test all checkpoints, and reject skipped tests. No golden cache or crate publication.
- Python float-repr parity probes, including exact half-even decimal ties and random bit patterns.
- `LayaResult::state_usage()` reports state-token truncation across question budgets, matching
  Python 0.4.0 without changing the existing `Usage` totals or result constructor.
- Engine score legends render structured criteria as strings, matching Python 0.4.0.
- `NoMarkers` errors include actual/requested marker counts and the total sequence budget,
  matching Python's actionable truncation diagnostic.
- State usage also reports option-span collisions after clipping, including the applied cap.
- `LayaEngine`: `choice`, `score` and `noul` predictions over the exported ONNX checkpoints
  (english, multilingual, typed-decisions); all questions for a state run in one batched forward pass.
- Loads both ONNX layouts: the fused `model.onnx` and the split `encoder.onnx` + `head.onnx`.
- Answers report `confidence` and `answer_confidence` (added in Python 0.3.21).
- Python-compatible sequence building, JSON serialization and calibration, checked against goldens
  recorded from the Python SDK.
- Checkpoint router, language detection, email state and option shortlisting.
- Opt-in model download from Hugging Face (Cargo feature `download`).
- Samples: quickstart, routing and benchmark.

### Known gaps

- Routing, script counting and fused-line email disclaimers follow the regenerated Python 0.4.0
  probes. New mixed-field language heuristics are not yet ported.
- Not yet ported: abstention (`min_confidence`, including per-bucket thresholds),
  histogram-binning calibration, `predict_long` and `lang_temperatures`.
