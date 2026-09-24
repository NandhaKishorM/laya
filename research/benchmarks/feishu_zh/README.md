# Chinese workplace decisions (Feishu-style)

[简体中文](README.zh-CN.md) · [Original project](https://github.com/Adkid-Zephyr/chinese-workflow-decision-bench)

A small, self-contained contribution following [#154](https://github.com/NandhaKishorM/laya/issues/154): **64 synthetic Chinese scenarios, 8 families, 4 balanced labels**, with frozen prompts and real per-request Laya/Jev responses. It tests ownership, cancellation, urgency, knowledge sharing, conditions, thread context, cross-chat interference, and quoted instructions.

**The scenarios and reference labels are AI-assisted synthetic fixtures, not extracted private Feishu chats. The outputs and timings are actual recorded model calls.** This is a diagnostic, not a general model ranking, an official Feishu evaluation, or an independently annotated blind test. No fine-tuned models or training data are included.

<img src="assets/xiaohongshu-scorecard-3x4-en.png" width="420" alt="Archived results, 64 Chinese synthetic cases">

## Start here: verify offline

From the repository root, using Python 3.10+:

```bash
python research/benchmarks/feishu_zh/audit.py
python -m unittest discover -s research/benchmarks/feishu_zh/tests -v
```

No model downloads, API keys, or third-party packages are needed. The audit checks dataset/prompt hashes, every case ID and reference label, request equality, repeated-run completeness, raw-response decoding, finite probabilities, and failure accounting before recomputing metrics.

| First-repeat label matches | Choice | Four independent yes/no questions |
|---|---:|---:|
| Laya multilingual | 20/64 | 18/64 |
| Jev 1.13.0 | 64/64 | 63/64 |

These are **archived 2026-09-21 results, not scores for the current main branch**. Three repetitions are preserved (384 requests per backend). Quality uses the preselected first repeat, not the best repeat. Jev's choice score was 63/64 on one later repeat. Failures would remain in the denominator; this archive has none.

- Laya: checkpoint `convaiinnovations/laya/multilingual` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982`; source `ef7d7d269e2e34c763c228144dc10fe7d421acf9` (download filtering only; no inference changes from upstream `42626c3`); M4 MPS, float32 execution, PyTorch 2.14.0, Transformers 5.17.0.
- Jev: actual API responses identify `jev-1.13.0`; persistent HTTP client. Its timing includes network/server latency, unlike the local deployment path. The systems were not measured simultaneously or on identical hardware.
- The archived CPU sanity check matched MPS on 16/16 sampled labels and rounded probabilities; this does not rule out shared software errors. No state/instruction/option truncation was observed in the archive.
- The multilingual base checkpoint was tested without task-specific fine-tuning or threshold fitting. This does not characterize the English or typed-decisions checkpoints.

## Run Laya on your machine

Use the repo's normal installation instructions. Download only the multilingual runtime files if you do not already have them (about 678 MB):

```python
from huggingface_hub import snapshot_download
root = snapshot_download(
    "convaiinnovations/laya",
    revision="1c5edc17a7acd8701df6fc341c0d179f1c62c982",
    allow_patterns=["multilingual/rl_agent_config.json", "multilingual/model.safetensors",
                    "multilingual/encoder/*", "multilingual/tokenizer/*"],
)
print(root + "/multilingual")
```

Set `CHECKPOINT` to the printed local directory. Run a smoke test first:

```bash
PYTHONPATH=. python research/benchmarks/feishu_zh/run.py --backend laya \
  --checkpoint "$CHECKPOINT" --device cpu --limit 2 --repeats 1 --modes choice \
  --output /tmp/feishu-laya-smoke
python research/benchmarks/feishu_zh/audit.py --run-dir /tmp/feishu-laya-smoke
```

For the full protocol, omit `--limit`, `--repeats` and `--modes` (defaults: 64 cases, 3 repeats, both modes). Select `--device mps` or `cuda` when available. Use a new output directory for every run. The runner records the actual source commit/file hashes, weight hash, device, versions and length diagnostics; it does not claim the archived version was used. `--checkpoint-revision` is an optional user-supplied provenance hint, not a substitute for the file hash. Loading/inference runs offline after local checkpoint preparation.

## Optional Jev comparison

```bash
pip install httpx
python research/benchmarks/feishu_zh/run.py --backend jev --model jev-1.13.0 \
  --output /tmp/feishu-jev-run
python research/benchmarks/feishu_zh/audit.py --run-dir /tmp/feishu-jev-run
```

This makes billable API calls. The key is read from `TYPESAFE_API_KEY` or a hidden prompt; it is never saved. No keys or remote calls are required by CI. Requests are serial, use one excluded warmup per mode, and are not automatically retried. Raw exception messages are not written to public records.

## Files and design choices

| Path | Purpose |
|---|---|
| `data/cases.jsonl` | Frozen original Chinese text, reference labels, rationale, target message and chat IDs |
| `SOURCE.json` | Original source commit and byte hashes for archived artifacts |
| `data/manifest.json`, `prompts.py` | Hashes, label policy and exact choice/four-question prompts |
| `run.py` | Opt-in local/API runner, fresh output directories and explicit smoke-test metadata |
| `audit.py`, `metrics.py`, `tests/` | Model-free validation, scoring and corruption/failure tests |
| `results/v1/{laya,jev}/` | Archived metadata and all 768 timed raw responses |
| `results/v1/summary.json` | Original machine-readable score summary |
| `results/v1/environment_check.json` | Small historical CPU/MPS and weight-digest check |
| `assets/`, `render_cards.py` | English/Chinese mobile scorecards, Chinese table and plotting source |

`choice` selects `urgent / todo / valuable / noise`. `four_noul` separately asks relevance, personal action, urgency and useful information, then applies frozen thresholds. They are different workflows and must not be blended into one score. A completion/cancellation notice without substantive new information is `noise` under this product policy, not under every possible product policy.

We highlight **false task assignments and missed tasks**, not accuracy alone. Context and distractors are retained in the input, while reference labels/rationales/family names stay outside the model request. Independent requests do not inherit conversational state.

The runner is intentionally small; the [companion project](https://github.com/Adkid-Zephyr/chinese-workflow-decision-bench) provides additional classifier adapters and the longer analysis. To regenerate cards, install `matplotlib` and run `render_cards.py --lang en` (or `zh`, requiring Hiragino Sans GB/Noto Sans CJK). Figures are generated from saved results, not manually entered scores.

## Limitations and attribution

Only 64 related synthetic examples, with no independent multi-annotator agreement. An earlier 12-case pilot informed prompt design. These public fixtures are regression diagnostics; do not tune on them and claim held-out generalization. Exact label semantics, checkpoint, calibration, and prompt format can materially change results. Chinese post-training remains a separate research question, not a result established by this benchmark.

Contributed by Adkid-Zephyr with OpenAI Codex assistance. Imported from companion commit `b694dc6dbcba12c5bcf8d51b60ec48ff6ab87d57`, with the runner/audit adapted for this repository. The bundled contribution retains its [MIT license](LICENSE); it does not alter Laya's license. No credentials, real user chats, local account paths or model weights are included.
