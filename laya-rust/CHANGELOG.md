# Changelog

All notable changes to the `laya-onnx` crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases are tagged `laya-rust-v<version>`.

## [0.1.0] - unreleased

First release: a port of the Laya Python SDK, tested against laya (Python) 0.3.21.

### Added

- Advisory parity CI on every push and PR: regenerate fixtures from the current Python checkout,
  test all checkpoints, and reject skipped tests. No golden cache or crate publication.
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
- Not yet ported from Python 0.3.21: abstention (`min_confidence`), `predict_long` and
  `lang_temperatures`.
