# GPU Validation Report

**Date:** 2026-09-18 · **Hardware:** 1× NVIDIA L40S (46GB), 128 vCPU, driver 595.91.07, CUDA 13.2
**Checkpoint under test:** `convaiinnovations/laya` (ModernBERT-large backbone, trained at `max_len=512`)
**Library version:** 0.2.0 · **Raw artifacts:** [`docs/results/`](results/)

Closes the two items from [feature_parity_requirements.md](feature_parity_requirements.md)
that could not be settled on CPU: whether the context window can be extended past 512, and
whether `two_stage_choice` is a viable answer to the option ceiling.

**Headline:** the context-extension question resolved favourably — 8k training works, is cheap,
and fits in 9.7GB. The high-cardinality question resolved **unfavourably**, and produced a
genuine negative result plus two bugs. Both are detailed below.

---

## 1. Context window: 8k training works

### Does it train at 8k?

Yes, comfortably, and far cheaper than estimated.

| Metric | Measured |
|---|---|
| Step time (8192 tokens, batch 1, accum 8) | **~2.9 s** |
| Peak GPU memory | **9.65 GB** of 46GB |
| Time to converge | **50 steps / ~2.5 minutes** |
| GPU utilization | 100% |

Memory is the notable figure: at 9.7GB, 8k fine-tuning fits on a 16GB card. The L40S I
specced was oversized for this job — an A10G (24GB) or even a 16GB T4-class card would do.

### Does it help?

On a needle-in-a-long-document task (one decisive sentence buried at a random depth in an
8k-token document, no keyword shortcut distinguishing the classes):

| | Accuracy | ECE |
|---|---|---|
| Base checkpoint (trained at 512), evaluated at 8k | 0.594 | 0.350 |
| After 50 steps of 8k fine-tuning | **1.000** | **0.000** |

The base checkpoint is barely above chance at 8k with badly miscalibrated confidence — the
predicted failure mode, confirmed. Fine-tuning fixes it almost immediately.

**Caveat, and it is a real one.** Both the training and evaluation data are synthetic, drawn
from the same generator, with only 8 evidence sentences and 8 filler sentences. A score of
1.000/0.000 means the task was learnable, not that the model is now good at long documents
generally. This establishes **feasibility and cost**, not production readiness. A real
deployment needs real long-document data; the numbers here should not be published as
benchmark results.

### How far can the base checkpoint be pushed without retraining?

| `max_len` | Accuracy | ECE |
|---|---|---|
| 512 | 0.531 | 0.348 |
| 1,024 | 0.542 | 0.334 |
| 2,048 | 0.479 | 0.410 |
| 4,096 | 0.552 | 0.357 |
| 8,192 | 0.646 | 0.296 |
| 16,384 | 0.479 | 0.378 |

**This table does not show what it was designed to show, and should not be read as a
degradation curve.** Accuracy sits near chance (~0.5) at *every* length including 512, so what
it actually measures is that the base checkpoint cannot do this task at all — not that longer
contexts are worse. The non-monotonic bump at 8,192 is noise on 96 samples, not a finding.

The useful conclusion is narrow but real: the checkpoint is **not usable out of the box for
long-document decisions at any length**, which is consistent with the 0.594 baseline above and
is precisely what fine-tuning fixes.

### Does 32k work?

Mechanically yes, but it remains unvalidated, and I did not train at 32k. From the earlier
CPU investigation: forward passes run cleanly at 8k/16k/32k after raising
`max_position_embeddings`, but needle representations drift (cosine similarity to a
short-context reference falls from ~0.95 at 1.6k to ~0.81 at 26k), and ModernBERT's model card
states it was *"pre-trained up to 1,024 tokens, then extended to 8,192"* — 8,192 is where
deliberate context-extension training stopped, not a config ceiling. RoPE is configured
`rope_type: "default"` with no scaling.

**Recommendation unchanged: target 8k.** It is inside the encoder's trained range, it is a 16x
improvement over today's 512, and it now has a measured cost of minutes. 32k would need RoPE
scaling plus its own context-extension phase.

---

## 2. High cardinality: `two_stage_choice` underperforms as shipped

This is the negative result, and it reverses the optimistic framing in the requirements doc.

### Measured accuracy (96 cases, synthetic support-ticket routing)

| Options | Single call | ECE | Two-stage (default chunking) | ECE |
|---|---|---|---|---|
| 16 | 1.000 | 0.003 | 1.000 | 0.003 |
| 50 | **0.677** | 0.305 | 0.323 | 0.251 |
| 100 | **0.750** | 0.250 | 0.229 | 0.135 |
| 255 | *does not fit* | — | 0.073 | 0.859 |
| 400 | *does not fit* | — | 0.000 | 0.002 |

Wherever both approaches fit, **single-call wins decisively** — 0.750 vs 0.229 at 100 options.
Two-stage removes the ceiling but pays for it heavily, and at 255+ it is close to useless.

### Why: the failure is in stage 1, not stage 2

Decomposing the 100-option case:

| Stage | Accuracy |
|---|---|
| Stage 1 (pick the right group) | **0.042** |
| Stage 2 (pick within a correct group) | **1.000** |
| End to end | 0.042 |

Stage 2 is perfect. Stage 1 is catastrophic. The cause is that default chunking creates groups
named `group_0`, `group_1`, … whose descriptions are arbitrary lists of member names — there is
no semantic signal to choose between them. The model is being asked an unanswerable question.

### The fix: semantic grouping

Passing `group_of` so that groups correspond to real categories (`billing_*`, `outage_*`, …):

| Approach (100 options) | Accuracy |
|---|---|
| Two-stage, default chunking | 0.229 |
| **Two-stage, semantic grouping** | **0.646** |
| Single call (where it fits) | 0.750 |

Semantic grouping is **2.8x better** than default chunking and approaches single-call accuracy
while removing the ceiling entirely.

### Two bugs found and fixed

1. **Group descriptions crowded out the state.** Stage-1 descriptions listed every group member,
   which at 400 options consumed the whole head budget — the logs show `kept 0 of 19 tokens`,
   i.e. the document was entirely discarded and accuracy was 0.000. Descriptions are now
   length-bounded with an "and N more" suffix.
2. **More groups than fit one head.** 400 options at `group_size=16` produces 25 groups, more
   than a single call can hold. `two_stage_choice` now recurses so the group question is itself
   staged.

Both are covered by new regression tests in `tests/test_scaling.py`.

### Recommendation

- **Below ~100 options: use a single call.** It fits and it is more accurate.
- **Above that: use `two_stage_choice` with an explicit `group_of`.** Default chunking is now a
  documented fallback that trades substantial accuracy for capacity, not a recommended path.
- The docstring and README now say this. The requirements doc's claim that two-stage "lifts the
  ceiling" is true but was incomplete: it lifts the ceiling *at a cost that depends entirely on
  grouping quality*.

---

## 3. What changed in the code

| Change | File | Reason |
|---|---|---|
| Bounded group descriptions | `laya/patterns.py` | Stage-1 descriptions starved the state of window |
| Recursive group selection | `laya/patterns.py` | >`group_size` groups overflowed a single head |
| Confidence threaded through recursion | `laya/patterns.py` | Joint confidence across nested stages |
| `group_of` guidance in docstring | `laya/patterns.py` | Records the 0.65-vs-0.23 finding at the call site |
| 3 regression tests | `tests/test_scaling.py` | Lock in both bug fixes and the semantic-grouping path |

Scripts used, committed for reproducibility: `scripts/eval_high_cardinality.py`,
`scripts/train_long_context.py`, `scripts/eval_context_lengths.py`.

Test suite: **182 passing** without a model, 17 more against the real checkpoint under
`LAYA_INTEGRATION=1` (199 total).

---

## 4. Addendum: citation checking (CPU, post-GPU)

Completing P2 added `select_tool` and `check_citation`. The latter surfaced two findings on
the published checkpoint, both measured against a small hand-built set of claim/source pairs.

**State shape matters more than expected.** The first implementation passed
`{"claim": ..., "source": ...}` as a JSON state and referenced the fields from the
instructions with backticks, mirroring the style of the built-in presets. On a genuinely
supported pair it scored **0.048**. The same content as inline text
(`"Premise: ...\n\nHypothesis: ..."`) scored **0.865** — a 18x difference on identical
information. The checkpoint was not trained to resolve backtick field references into a JSON
state for this kind of question. `check_citation` now uses the inline framing.

| Framing | P(supported) on a supported pair |
|---|---|
| `{"claim":…, "source":…}` + backtick refs | 0.048 |
| Inline text | 0.865 |
| Premise/hypothesis | 0.869 |

**The contradiction signal is not trustworthy.** After the fix, support is reliable (0.937 and
0.994 on supported pairs, ≤0.023 elsewhere). Contradiction is not: an *unrelated* source
("the cafeteria menu changed" vs. a claim about a service outage) scored **0.996** contradiction
— higher than a genuine refutation at 0.965. The model conflates "does not support" with
"contradicts".

Adding a third relevance question separates most such cases (0.000 for the unrelated pair) and
is used to downgrade borderline contradictions. It does not separate all of them: one true
contradiction also scored 0.015 relevance, so a decisive contradiction is deliberately allowed
to stand on its own. Final accuracy on the 6-case set is 5/6, with the single miss being a pair
that is genuinely inseparable on both signals.

This is documented in the docstring and README rather than tuned away — with 6 examples, any
threshold that fixed the last case would be overfitting. The honest guidance is that a
`"supported"` verdict is dependable and everything else means "route to a human".

---

## 5. Remaining work

| Item | Status | Needs |
|---|---|---|
| 8k context | Feasibility proven; needs a real training run | Real long-document data, ~1 GPU-hour |
| High-cardinality choice | **Resolved** — guidance is "single call under 100, semantic grouping above" | Nothing |
| 32k context | Not recommended | RoPE scaling + context-extension training |
| Production 8k checkpoint | Not started | Real data, then publish alongside the 512 checkpoint |

The single most valuable next step is assembling **real** long-document training data. The
compute is no longer the bottleneck — 2.5 minutes of L40S time was enough to move 0.594 → 1.000
on synthetic data. Data quality is now the limiting factor.
