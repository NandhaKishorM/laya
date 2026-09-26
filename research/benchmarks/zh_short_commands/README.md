# Chinese short-command routing

[简体中文](README.zh-CN.md) · [Sibling benchmark: Chinese workplace decisions](../feishu_zh/README.md) · [Issue #218](https://github.com/NandhaKishorM/laya/issues/218)

18 frozen Chinese voice commands for a cleaning robot, six labels, and a seven-rung ablation over the repository's own prompt guidance. One checkpoint, one run, every decision archived — the point is to turn [#218](https://github.com/NandhaKishorM/laya/issues/218)'s one-off report ("adding criteria, a scenario and a structured JSON state made Chinese decisions *worse*") into an artifact anyone can re-derive, and to say which primitive it is actually true of.

**These are 18 hand-written fixtures with a label policy fixed before any model was run. They are not a held-out test set, not an independently annotated corpus, and not an official evaluation.** The numbers below are one checkpoint's decisions on those fixtures.

## Start here: verify offline

From the repository root, using Python 3.10+:

```bash
python research/benchmarks/zh_short_commands/audit.py
python -m unittest discover -s research/benchmarks/zh_short_commands/tests -v
```

No model download, no network. The audit needs nothing but the standard library, and CI runs it in a job where nothing is installed. The test suite drives `run.py --stub`, so it needs numpy — which every environment that can run Laya already has.

The audit re-checks the hashes of the frozen cases and prompts, rebuilds every request (state, instructions, options, gold) instead of trusting the record, and recomputes accuracy, macro-F1, ECE and confidence spread from the per-case records alone. The tests corrupt a copy of the archive in 18 different ways and require the audit to refuse each one; those exist to show the audit can fail.

## The ladder

`choice` cannot exist without criteria — the criteria keys *are* its options — so task A starts one rung higher than task B.

**Task A, `choice`: one six-way question per command.**

| Rung | Correct | Accuracy | macro-F1 | ECE | Mean conf | SD conf |
|---|---:|---:|---:|---:|---:|---:|
| `choice_criteria` (six options only) | 13/18 | 0.7222 | 0.7149 | 0.2659 | 0.8008 | 0.1728 |
| `choice_scenario` (+ scenario sentence) | 14/18 | 0.7778 | 0.7942 | 0.1827 | 0.8257 | 0.1725 |
| `choice_json_state` (+ structured state) | 12/18 | 0.6667 | 0.6817 | 0.2484 | 0.7961 | 0.1238 |

The most frequent label covers 5/18 cases, so the majority-class baseline is 0.2778.

**Task B, `noul`: four independent yes/no questions per command, 72 decisions.**

| Rung | Correct | Accuracy | Mean conf | SD conf |
|---|---:|---:|---:|---:|
| `noul_plain` (no criteria, one question) | 48/72 | 0.6667 | 0.8128 | 0.1786 |
| `noul_criteria` | 33/72 | 0.4583 | 0.8920 | 0.0944 |
| `noul_scenario` | 34/72 | 0.4722 | 0.8944 | 0.0807 |
| `noul_json_state` | 30/72 | 0.4167 | 0.9437 | 0.0418 |

Answering "true" to all 72 questions scores 30/72 = 0.4167, because only 30 of the 72 gold answers are true. That is exactly what `noul_json_state` scored.

## What the ladder shows

The documented guidance does not make the `noul` path read Chinese differently; it makes the four questions stop discriminating. Count how many of the 18 commands each dimension answered "true" for:

| Rung | `wants_faster` (4 truly true) | `wants_slower` (5) | `wants_stop` (4) | `is_command` (17) |
|---|---|---|---|---|
| `noul_plain` | 9 — acc 0.611 | 14 — 0.389 | 5 — 0.944 | 12 — 0.722 |
| `noul_criteria` | 17 — 0.278 | 17 — 0.333 | **18** — 0.222 | 17 — **1.000** |
| `noul_scenario` | 17 — 0.278 | 17 — 0.333 | 17 — 0.278 | 17 — **1.000** |
| `noul_json_state` | **18** — 0.222 | **18** — 0.278 | **18** — 0.222 | **18** — 0.944 |

Once criteria are present, `wants_stop` answers "true" for all 18 commands — including the 14 that are not stop commands, with p(true) ≥ 0.794 in every case. At the top rung every dimension answers "true" for every case. `is_command` reaches 1.000 under criteria only because "true" is the common answer there (17/18), and it drops back to the always-true baseline (0.944) once the JSON state is added — at which point it also says "yes, a movement command" about 今天天气不错 ("the weather is nice today") with p(true) = 0.938, where `noul_plain` gave 0.005.

So the accuracy loss is a property of the answer distribution, not of the guidance teaching the model anything about Chinese. The confidence spread says the same thing from the other side: the top rung's 72 confidences sit in 0.750–0.995 while `noul_plain` spans 0.504–1.000.

**Task A does not show this.** No choice rung collapses: 0.7222 → 0.7778 → 0.6667, with the scenario helping by one case and the JSON state costing two. A `choice` question is not asked to return "yes" or "no" — the criteria keys are its options, so the answer cannot degenerate into a constant. That is the difference [#218](https://github.com/NandhaKishorM/laya/issues/218) could not see: the guidance it tested is harmful on the `noul` path and not on the `choice` path, and the two must not be averaged into one "Chinese accuracy".

## What the ladder does not change

Six of the failures are identical across all three choice rungs, so they are properties of the checkpoint on these inputs, not of the prompt format:

| # | Command | Gold | All three rungs | Confidence |
|---|---|---|---|---|
| 001 | 快一点 | `faster` | `slower` | 0.652 / 0.690 / 0.703 |
| 003 | 太慢了 | `faster` | `slower` | 0.990 / 0.993 / 0.898 |
| 006 | 太快了 | `slower` | `faster` | 0.795 / 0.901 / 0.767 |
| 016 | 往右边靠一下 | `right` | `left` | 0.395 / 0.399 / 0.556 |
| 010 | 别动了 | `stop` | `none` (2 of 3 rungs) | 0.607 / – / 0.796 |
| 017 | 别太快了 | `slower` | `faster` (JSON state only) | – / – / 0.556 |

Reading down the speed family: 太慢了 means "too slow" (go faster) and 太快了 means "too fast" (slow down) — the model picks the direction the adjective points to, not the correction the phrase asks for, and does it with 0.90–0.99 confidence. Negation is handled: 别太快了 and 别动了 are both read correctly in at least one rung, and 别太快了 is even corrected by the scenario sentence. So the pattern is specific to the 太 ("excessively") construction inverting the action, not to Chinese generally.

Direction routing is not uniformly broken either: 左转, 往左一点 and 向右转 are all correct in all three rungs, and the single failure (往右边靠一下 → `left`) is the least confident decision in the archive. Chit-chat is rejected: 今天天气不错 → `none` in all three rungs. Each of these rows is **one fixture** — treat them as cheap-to-falsify hypotheses about where to put the next 100 cases, not as established morphology claims.

**The two shapes disagree about what a command is at all.** For 6 of the 18 commands, `choice_criteria` and `noul_plain`'s `is_command` contradict each other:

| # | Command | `choice` says | `is_command` p(true) |
|---|---|---|---|
| 001 | 快一点 | `slower` | 0.254 |
| 005 | 慢一点 | `slower` | 0.496 |
| 010 | 别动了 | `none` | 0.681 |
| 013 | 左转 | `left` | 0.115 |
| 014 | 往左一点 | `left` | 0.226 |
| 017 | 别太快了 | `slower` | 0.462 |

A product has to pick one workflow and calibrate it, not consume both. This is the same conclusion the [sibling Feishu benchmark](../feishu_zh/README.md) reaches for its `choice`/`four_noul` pair.

## Provenance

Recorded by `run.py` in the archive's `config.checkpoint` block; the audit refuses an archive that does not pin the weights.

- Checkpoint: `multilingual/` subfolder of `convaiinnovations/laya`, read from a local directory (`E:/home/laya-models`). The bundled copy and the standalone `convaiinnovations/laya-multilingual` repo currently report the same LFS digest for the weights, so this is the published multilingual checkpoint.
- `model.safetensors`: 643835514 bytes, `sha256 9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`.
- Run: 2026-09-24T07:56:04Z, CPU, float32, `laya 0.3.20`, `torch 2.14.0+cpu`, `transformers 5.17.0`, Python 3.12.
- Scoring: the checkpoint's own per-bucket temperatures, clamped exactly as `Agent` applies them (`unclamped: false` in the archive), the same code path as [`research/eval/laya_eval.py`](../../eval/laya_eval.py). No thresholds were fitted, and nothing here is fine-tuned.
- The whole seven-configuration sweep takes about 20 seconds on CPU.

`research/eval/README.md` records a different value for the same file — `sha256 b99c8bea…`. That string is the git blob id (`SHA-1`) of the *LFS pointer* stored in the Hub repository, not the SHA-256 of the weights: hashing the pointer text `version https://git-lfs.github.com/spec/v1\noid sha256:9d628fd9…\nsize 643835514\n` as a git blob reproduces `b99c8bea239c53f6f6bce734557dc6c403fa6b3e` exactly. The same file's `rl_agent_config.json` entry (`sha256 00e35f88…`) is likewise the blob id of the JSON itself. Anyone who checks these with `sha256sum` will see a mismatch and conclude the checkpoint changed when it did not. Flagged here rather than silently fixed, since it is a separate change to a file this contribution does not own.

## Run it on your machine

Download only the multilingual runtime files if you do not already have them (about 614 MiB):

```python
from huggingface_hub import snapshot_download
root = snapshot_download(
    "convaiinnovations/laya",
    allow_patterns=["multilingual/*.json", "multilingual/model.safetensors",
                    "multilingual/encoder/*", "multilingual/tokenizer/*"],
)
print(root + "/multilingual")
```

Set `CHECKPOINT` to that directory, check the file, then run the sweep. `--stub` scores with a fixed pseudo-random vector instead of a checkpoint and exercises the whole pipeline offline; `--configs` runs a subset:

```bash
python research/benchmarks/zh_short_commands/run.py --checkpoint "$CHECKPOINT" \
  --device cpu --out /tmp/zh-short-commands
python research/benchmarks/zh_short_commands/audit.py --run-dir /tmp/zh-short-commands
```

The runner hashes the cases, `prompts.py` and the checkpoint's `model.safetensors` into the report, so an archive from another machine can be compared with the committed one field by field. Use a new output directory for every run. `--unclamped` reproduces the raw pre-clamp temperatures the older committed sweeps used; the archived numbers above use the clamped ones.

## Files

| Path | Purpose |
|---|---|
| `data/cases.jsonl` | The 18 frozen commands: id, text, family, gold label, notes |
| `data/manifest.json` | Label/family counts, the ladder, the policy, and the hashes of both frozen inputs |
| `prompts.py` | The seven rungs: instructions, six criteria, four `noul` dimensions and their criteria |
| `run.py` | The runner: loads a checkpoint, scores the ladder, writes `report.json` |
| `audit.py`, `tests/` | Offline re-derivation, 18 archive-corruption cases, and the metric pins against `laya_eval` |
| `results/v1/laya-multilingual/report.json` | The archived run: 7 report blocks and all 342 per-case records |

The three parts of the archive are the same three parts `research/eval/laya_eval.py` emits — `config`, `report`, `cases` — plus a `summary`. Nothing in `report` needs the model to be re-derived.

## Limitations and attribution

- 18 cases, hand-written by the contributor, no independent annotation and no inter-annotator agreement. Families are unbalanced (9 speed, 4 stop, 4 direction, 1 chit-chat) and so are the labels (1 to 5 per label), which is why the `noul` base rate matters more than the rung ordering.
- One checkpoint, one language, one device, fixed temperatures, no sampling. Nothing here characterizes the English or typed-decisions checkpoints.
- The comparison reproduces the *shape* of [#218](https://github.com/NandhaKishorM/laya/issues/218), not its prompts: that report's exact text was never published, so its ~50% figure is not directly comparable with the 0.4167–0.6667 above.
- These fixtures are a regression diagnostic. Do not tune on them and then report them as held-out.
- Chinese post-training remains an open research question; 中文 short-command routing is not solved by this file, it is measured by it.

Contributed by GaotianJin, with AI assistance for the harness, the audit and the tests. The case text is the contributor's own. No user data, recordings, credentials or model weights are included; the checkpoint stays under its own license. This directory is Apache-2.0-licensed like the rest of the repository.
