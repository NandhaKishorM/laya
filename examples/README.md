# Laya examples — a learning path

41 runnable scripts arranged as eight stages. Each stage assumes the one before it and adds one
idea, so you can stop wherever you already have what you need. Every script runs against the real
checkpoints and prints real output — nothing here is a mock, and the numbers you see come from the
model on your machine. `_common.py` prefers `../models` when `setup_laya.sh` has downloaded them
and otherwise loads the same checkpoints from the Hub (`convaiinnovations/laya`) on first use, so
these run after a plain `pip install laya`.

```bash
cd ..                                            # repository root
./setup_laya.sh                                  # optional: venv + local checkpoints
.venv/bin/python examples/01_first_call_minimal.py
./examples/run_all.sh                            # all of them, pass/fail (non-zero exit on failure)
./examples/run_all.sh 2[0-4]                     # just stage 5, by basename glob
```

**Before you start**

* Each example is its own process, so each loads its own checkpoints: 30-60 s on CPU, faster on
  MPS, and the first MPS call pays ~13 s of Metal kernel compilation. Without a local `../models`
  copy, the first run also downloads the checkpoint it needs into `HF_HOME`.
* `device=None` lets Laya pick CUDA → MPS → CPU. `_common.load(name, device="cpu")` forces CPU.
* `_common.py` holds the shared states, question sets and printing helpers. Examples import it, so
  you can copy any example out and it keeps working.
* Where a weak spot is visible in an example's output, the example says so rather than hiding it:
  `score` is the weakest primitive, high-cardinality choice sets degrade, and Latin-script
  language detection is a heuristic.

---

## Stage 1 — first contact (01-03)

| | what you'll learn |
|---|---|
| `01_first_call_minimal.py` | the smallest possible program: load a checkpoint, ask one question, read the answer. Deliberately standalone, no shared helpers, so you can copy it into your own project. |
| `02_hello_world_decision.py` | the same call through the shared helpers, as every later example does it. |
| `03_reading_the_result.py` | what `predict()` actually returns: `answers` per question, the three answer shapes, `usage`, and the `action` payload. |

**You can now** load a checkpoint and read a typed answer.

## Stage 2 — the three decision primitives (04-10)

| | what you'll learn |
|---|---|
| `04_choice_question.py` | `choice`: a closed label set with descriptions, returning the winner and the full distribution. |
| `05_choice_criteria_design.py` | the criteria text is part of the prompt, so wording is part of the job: vague, specific and confusable rubrics on the same input. |
| `06_score_question.py` | `score`: an ordinal rubric, returning the expected level on the rubric's own scale. |
| `07_score_levels.py` | the same judgement at 3, 5 and 7 levels, and what granularity does to the distribution. |
| `08_noul_question.py` | `noul`: a calibrated P(true) you threshold yourself. |
| `09_noul_criteria_wording.py` | spelling out what true and false mean, and how loaded wording moves the probabilities. |
| `10_mixed_primitives_one_pass.py` | all three primitives in a single forward pass. |

**You can now** ask any of the three question types and read the answers as probabilities.

## Stage 3 — state shapes (11-15)

| | what you'll learn |
|---|---|
| `11_json_state.py` | a dict state, serialised to JSON for the model — no schema lookup, the text is the input. |
| `12_state_wording_matters.py` | the same facts as a ticket, an email and a JSON record, and which answers move. |
| `13_email_state_and_cleaning.py` | `email_state` / `clean_email_body`: stripping quoted replies, signatures and disclaimers, and what that buys. |
| `14_conversation_turns.py` | a list of turns as the state, so a whole exchange can be judged at once. |
| `15_truncation_basics.py` | what happens when the state does not fit `max_len`, and how to see it in `usage`. |

**You can now** feed Laya text, JSON, email or a conversation, and know what happens when it is too long.

## Stage 4 — cost, confidence and batching (16-19)

| | what you'll learn |
|---|---|
| `16_many_questions_one_pass.py` | many questions, one pass: `input_tokens` is a batch total across the questions in the call. |
| `17_timing_a_call.py` | measuring honestly: load time vs inference, warm-up, and medians over several calls. |
| `18_confidence_gating.py` | the production pattern: gate on `answer_confidence`, the calibrated probability of the reported answer, and escalate on low. |
| `19_many_states_loop.py` | many states in your own loop — and why batching *questions* is the model's job while batching *states* is yours. |

**You can now** size a workload, and branch on confidence instead of always acting.

## Stage 5 — the three checkpoints and routing (20-24)

| | what you'll learn |
|---|---|
| `20_which_checkpoint.py` | what `english`, `multilingual` and `typed-decisions` each are good at, on the same input. |
| `21_routing_without_running.py` | `route()` alone: a checkpoint decision in microseconds, with the reason string, before any forward pass. |
| `22_route_and_predict_multilingual.py` | `predict()` routing and answering in one call, in several languages. |
| `23_local_offline_models.py` | where the weights come from: local `../models` with no network at all when they are there, the Hub otherwise, and checking them against the manifest. |
| `24_preload_and_memory.py` | what stays resident: `preload`, `max_loaded`, LRU eviction, `attach`, `unload`. |

**You can now** let the router choose, and control what is held in memory.

## Stage 6 — the built-in presets (25-29)

| | what you'll learn |
|---|---|
| `25_preset_email_phishing.py` | `email_questions()`: phishing, spam, category and urgency from a raw email. |
| `26_presets_triage.py` | `triage_questions()`: intent, urgency, frustration, refund, churn in one pass. |
| `27_presets_guardrails.py` | `guard_questions()`: jailbreak, injection and sensitive-data signals. |
| `28_presets_moderation.py` | `moderation_questions()`: toxicity, harassment, threat and spam. |
| `29_presets_model_router.py` | `router_questions()`: difficulty, domain and sensitivity, and a small-vs-frontier policy. |

**You can now** skip schema design for the common application shapes.

## Stage 7 — design your own schema, then tune it (30-34)

| | what you'll learn |
|---|---|
| `30_custom_schema_design.py` | building a question schema from scratch for your own domain, with the design decisions spelled out. |
| `31_structured_criteria.py` | criteria as dicts and lists — rendered as compact JSON for the model. |
| `32_typed_decisions_workflow.py` | the fine-tuned checkpoint's four real workflows, and `auto_task_detection`. |
| `33_high_cardinality_choice.py` | 20, 77 and 120 options: the `head_max_len` option budget, and when to raise it. |
| `34_calibration_temperature.py` | the shipped temperature buckets, and what they do to the logits. |

**You can now** write a schema for your own problem and know where its limits are.

## Stage 8 — devices, scale and production shape (35-41)

| | what you'll learn |
|---|---|
| `35_device_selection_and_latency.py` | CPU vs MPS latency, and that both give the same answers. |
| `36_batch_throughput.py` | questions per second, and the trade-off between calls and questions per call. |
| `37_long_document.py` | raising the context window at runtime, and what that costs. |
| `38_multi_checkpoint_pipeline.py` | composing route → triage → specialist decision in one pipeline. |
| `39_error_handling.py` | the real behaviour of bad inputs — including the cases that do *not* raise. |
| `40_caching_and_monitoring.py` | an in-memory answer cache, and a confidence monitor over a batch. |
| `41_production_triage_service.py` | the capstone: route, triage, apply policy, emit a structured decision record. |

**You can now** run Laya as a component of a real system.

---

## Where to look things up

| you want to… | go to |
|---|---|
| understand the result dict | 03 |
| write criteria that work | 05, 09, 30 |
| choose `choice` vs `score` vs `noul` | 04, 06, 08, 30 |
| handle email | 13, 25 |
| handle conversations | 14 |
| handle long inputs | 15, 37 |
| know what a call costs | 17, 36 |
| branch on confidence | 18, 40 |
| pick between checkpoints | 20, 21, 22 |
| keep models resident / free memory | 24 |
| see a preset schema | 25-29 |
| design my own schema | 30 |
| use the fine-tuned checkpoint | 32 |
| many options in one question | 33 |
| tune probabilities | 34 |
| choose a device | 35 |
| compose checkpoints | 38 |
| handle failures | 39 |
| build a service | 41 |

`run_all.sh` exits non-zero if any example fails, so it doubles as a smoke test over the whole API
surface. It needs the checkpoints, so it is not part of CI; `python -m compileall examples/` is.
