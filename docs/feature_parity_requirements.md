# Feature Parity Requirements: Laya vs. TypeSafe Jev

Gap analysis against [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (launched 2026-09-15), the hosted System One decision model Laya reimplements.

> **Status (2026-09-18, v0.2.0).** P0 #1, P1 and P2 are implemented and tested. P0 #2 and #3
> were validated on an L40S GPU — see the [GPU validation report](gpu_validation_report.md).
> P0 #3 is resolved; P0 #2 is proven feasible and now needs real training data rather than
> compute. See [Implementation status](#implementation-status) for the full picture.
> Three measurements in the original analysis were corrected against real hardware — the option
> ceiling, the notebook `max_len` range, and the value of two-stage choice — and are annotated inline.

**The concept is already covered.** Both expose the same three primitives (`choice`/`score`/`noul`), the same `{state, questions} -> {answers}` shape, and the same RLCD calibration story. Nearly every gap below is surface area and limits, not architecture.

Ordered by importance for parity.

---

## P0 — Blocking

### 1. Structured JSON criteria (one path crashes)

Jev accepts JSON objects in `instructions`, choice option values, score levels, and noul `true`/`false` boundaries ([docs](https://docs.typesafe.ai/primitives/advanced.md)). This enables rubric-style options (`{"what": ..., "not_for": ..., "examples": [...]}`) and taxonomy walking via nested child options.

`render_options` (`laya/common.py:22`) assumes strings throughout:

- **choice / score** — dict criteria stringify as Python reprs (`billing: {'what': 'invoices'}`, single-quoted, not JSON)
- **noul** — dict criteria raise `TypeError: can only concatenate str (not "dict") to str` at `laya/common.py:30`

`instructions` is fine — `_to_internal` already json-dumps non-strings (`laya/agent.py:180`). Only `criteria` is affected.

**Fix:** `json.dumps` non-string criteria values in all three branches. ~1 hour; it is currently a hard crash.

### 2. Context window: 512 vs ~32,000 tokens

`max_len` defaults to 512 (`laya/agent.py:206`); notebooks train at 512 (Colab) and 1024 (Kaggle). [^1] Jev handles ~32k (~150k chars) — a 60x gap, and the most consequential one.

Jev's flagship cookbook runs 13 questions over a long Wikipedia article; Laya cannot hold the document.

[^1]: Corrected. The original text said "192–1024"; 192 is the default `head_max_len`, not a trained `max_len`.

State is also truncated **from the right** silently: `truncate_left` defaults to `False` (`laya/common.py:44`) and is never plumbed through `predict`, so long inputs lose their tail with no warning.

**Fix:** ModernBERT (the encoder) supports 8k natively, so this is a config/training choice, not an architectural limit. Raise `max_len`, expose `truncate_left`, and warn on truncation.

### 3. Choice option ceiling: ~126 vs 255

All options must fit `head_max_len=192` tokens alongside instructions, each capped at 48 tokens (`laya/common.py:54`). Past that, `predict` raises `"options exceed head_max_len"` (`laya/agent.py:213`).

**Measured, not estimated:** with the real ModernBERT tokenizer and realistic descriptive labels, **~126 options fit**, not the 27–38 originally estimated. [^2] The gap to Jev's 255 is real but roughly half as wide as first assessed.

[^2]: Corrected. The original estimate assumed each option consumed its full token cost, but `build_sequence` shrinks the per-option budget when options do not fit (`laya/common.py:57-60`), so far more short options fit than a naive count suggests.

This rules out hierarchical classification and skill-suggestion use cases outright.

**Fix:** needs a two-stage or hierarchical scoring path. The only item here that is genuine design work.

**Measured outcome (GPU, 2026-09-18):** two-stage works, but *only with semantic grouping*. With
default chunking it scored 0.229 against a single call's 0.750 at 100 options, because stage 1
cannot discriminate between arbitrarily-named groups (stage-1 accuracy 0.042, stage-2 1.000).
With `group_of` supplying real categories it reaches 0.646. Guidance is now: single call below
~100 options, `two_stage_choice(..., group_of=...)` above.

---

## P1 — SDK ergonomics

Cheap, mechanical, and what makes the library feel production-ready.

| Feature | Jev | Laya |
|---|---|---|
| Async client | yes | missing (sync only) |
| Configurable retry policy | yes | missing |
| Typed error hierarchy | 12 classes (`RateLimitError`, `APITimeoutError`, …) | raw `ValueError` / `RuntimeError` |
| Typed answer objects | `ChoiceAnswer.confidence` | raw dicts |
| `request_id` + structured logging | yes | missing |
| JS/TS SDK | `@typesafe-ai/sdk` | Python only |

Typed answer objects and a proper exception hierarchy matter most — they are what callers write code against.

---

## P2 — Patterns library

Jev ships **18 cookbooks**; Laya has none. These are library-level compositions over the same primitives, not model features — the cheapest gap to close and the one that most changes day-to-day usability.

Highest value first: speculative fan-out, confidence-gated routing, composite scoring, self-consistency (run a question several ways, route disagreement to review), re-ranking, function calling, citation checking, multi-stage extraction cascades.

**Fix:** port as `laya.patterns` helpers on top of the existing `predict`.

---

## P3 — Tooling

Playground / console, hosted docs site, per-version known-issues page, agent skill for Claude Code.

---

## Where Laya already wins

Worth protecting — none of this exists in Jev:

- **Self-hosted, Apache-2.0 weights.** Zero egress, zero marginal cost, air-gap capable. Jev is waitlisted cloud-only.
- **Exposed calibration internals.** Per-bucket temperature scaling, `ece_score` for measuring your own calibration, `proper_reward` and `td_lambda_targets` for retraining. Jev hides all of it.
- **`act_head`** — an explicit act/escalate signal alongside the answer.
- **TD(λ) multi-turn trajectory modeling.** Jev has no documented multi-turn support at all.
- **Email-specific cleaning** (`laya/email.py`) and fine-tuning notebooks. Jev offers no fine-tuning.

---

## Benchmark caveat

The README's comparison table uses Jev's *published* numbers, not a head-to-head run. Jev's 70–500 ms is end-to-end network latency; Laya's ~38 ms is local GPU compute. Those are not the same measurement and the table should say so.

---

## Suggested order

1. Fix structured-criteria rendering (crash)
2. Raise `max_len`; expose `truncate_left`; warn on truncation
3. Typed answers + exception hierarchy
4. Async client + retries
5. Port 4–6 cookbooks as `laya.patterns`
6. Two-stage scoring for high-cardinality choice

---

## Implementation status

Landed in v0.2.0. 182 tests run without a model; 17 more run against the real checkpoint
under `LAYA_INTEGRATION=1`.

| Item | Status | Where |
|---|---|---|
| P0 #1 structured criteria | **Done** | `render_criterion` in `laya/common.py`; the noul `TypeError` is gone and all three branches emit JSON |
| P0 #2 truncation reporting | **Done** | `truncate_left` + `max_len` are constructor args; truncation is logged and returned as `Response.truncated` |
| P0 #2 long context | **Validated on GPU** | 8k fine-tuning measured: 0.594 → 1.000 accuracy, 0.350 → 0.000 ECE in 50 steps (~2.5 min, 9.7GB). Needs real data for a production checkpoint. 32k still not recommended. See [GPU validation report](gpu_validation_report.md) |
| P0 #3 option ceiling | **Resolved on GPU** | Measured: single call wins below ~100 options (0.750 vs 0.229); above that use `two_stage_choice` **with `group_of`** (0.646 vs 0.229 for default chunking). Two bugs found and fixed. See [GPU validation report](gpu_validation_report.md) |
| P1 typed errors | **Done** | `laya/errors.py`, 10 classes under `LayaError`, keeping builtin bases for compatibility |
| P1 typed answers | **Done** | `laya/types.py`; dict-compatible so existing callers and the notebooks keep working |
| P1 async + retries | **Done** | `laya/aio.py`: `AsyncAgent`, `RetryPolicy`, `batch(max_concurrency=…)` |
| P1 request_id + logging | **Done** | `Response.request_id`; warnings go through the `laya` logger instead of `print` |
| P1 JS/TS SDK | **Not started** | Out of scope for this pass |
| P2 patterns | **Done** | `laya/patterns.py`, 14 helpers — including `select_tool` (function calling) and `check_citation` (citation checking), the two the priority list named that were initially missed |
| P3 tooling | **Not started** | Playground, hosted docs, known-issues page |
| Benchmark caveat | **Done** | README now states the latency and accuracy comparisons are not head-to-head |

### GPU work: done

Both items were validated on an L40S on 2026-09-18 — see the
[GPU validation report](gpu_validation_report.md) for full numbers.

1. **Context window.** 8k fine-tuning works and is cheap (2.9 s/step, 9.7GB peak, converged in
   50 steps). Accuracy and ECE both resolved on synthetic data. What remains is *data*, not
   compute: a production 8k checkpoint needs real long-document examples.
2. **High-cardinality choice.** Resolved, with a negative result: `two_stage_choice` as
   originally shipped was substantially *worse* than a single call, because stage 1 could not
   discriminate between arbitrarily-chunked groups (stage-1 accuracy 0.042, stage-2 1.000).
   Fixed two bugs and documented semantic grouping as the recommended path.

---

### Sources

[Docs index](https://docs.typesafe.ai/llms.txt) · [Advanced structure](https://docs.typesafe.ai/primitives/advanced.md) · [Question types](https://docs.typesafe.ai/sdk/python/api/types/questions.md) · [Response types](https://docs.typesafe.ai/sdk/python/api/types/responses.md) · [Confidence](https://docs.typesafe.ai/confidence.md) · [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions)
