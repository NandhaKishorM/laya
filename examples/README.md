# Laya examples

27 runnable scripts, ordered from a single call to a small production service. Every one runs
against the checkpoints in `../models` and prints real output — nothing here is a mock, and the
numbers in the output come from the model on your machine.

```bash
cd ..                                            # repository root
./setup_laya.sh                                  # once: venv + checkpoints
.venv/bin/python examples/01_hello_world_decision.py
./examples/run_all.sh                            # or run all of them and report pass/fail
./examples/run_all.sh 1[0-9]                     # just the 10s (basename glob)
```

`_common.py` holds the shared state, question sets and printing helpers; each example imports it,
so you can copy any example out and it keeps working.

Notes before you start:

* Each example is its own process, so it loads its own checkpoints — 30-60 s each on CPU, faster
  on MPS, and the first MPS call pays ~13 s of Metal kernel compilation.
* `device=None` lets Laya pick CUDA → MPS → CPU. Pass `device="cpu"` to `_common.load()` to force
  the conservative path (see example 24).
* The examples state Laya's weak spots where they are visible instead of hiding them: `score` is
  the weakest primitive, 50+ options in one question degrade, and Latin-script language detection
  is a heuristic.

## 01-09 — fundamentals

| | shows |
|---|---|
| `01_hello_world_decision.py` | one checkpoint, one `noul` question, one forward pass |
| `02_choice_question.py` | a closed label set with descriptions; probabilities and confidence |
| `03_score_question.py` | an ordinal rubric; expected level, distribution, legend |
| `04_noul_question.py` | calibrated P(true), your own threshold, true vs false |
| `05_mixed_primitives_one_pass.py` | `choice` + `score` + `noul` in a single call |
| `06_many_questions_one_pass.py` | 13 questions in one call; `input_tokens` is a batch total |
| `07_json_state.py` | a dict state, serialised to JSON for the model |
| `08_email_state_and_cleaning.py` | `email_state` / `clean_email_body`, raw vs cleaned |
| `09_conversation_turns.py` | a list of turns as the state; calm vs escalating |

## 10-18 — routing and the built-in presets

| | shows |
|---|---|
| `10_confidence_gating.py` | automate above a confidence threshold, escalate below |
| `11_routing_without_running.py` | `route()` alone: microseconds, no weights, with `model=` / `lang=` overrides |
| `12_route_and_predict_multilingual.py` | the router switching checkpoints for EN / HI / DE |
| `13_local_offline_models.py` | local checkpoint paths and a size check against `verify/checkpoints.json` |
| `14_preload_and_memory.py` | `max_loaded` eviction, measured reload cost, LRU, `preload`, `attach`, `unload` |
| `15_presets_triage.py` | `triage_questions()`: intent, urgency, frustration, refund, churn |
| `16_presets_guardrails.py` | `guard_questions()`: jailbreak, injection, sensitive data |
| `17_presets_moderation.py` | `moderation_questions()`: toxicity, harassment, threat, spam |
| `18_presets_model_router.py` | `router_questions()` plus a small-vs-frontier policy |

## 19-27 — the harder edges, and production shape

| | shows |
|---|---|
| `19_typed_decisions_workflow.py` | the fine-tuned checkpoint's real workflows, and `auto_task_detection` |
| `20_high_cardinality_choice.py` | 20, 77 and 120 options; why the `head_max_len` budget matters |
| `21_long_document.py` | truncation at `max_len`, and raising the window at runtime |
| `22_structured_criteria.py` | dict/list criteria, rendered as compact JSON via `render_options` |
| `23_calibration_temperature.py` | the shipped temperature buckets and what they do to logits |
| `24_device_selection_and_latency.py` | CPU vs MPS latency, and that both give the same answers |
| `25_batch_throughput.py` | questions/second, and why batching questions (not states) is the win |
| `26_error_handling.py` | the real behaviour of bad inputs — including what does *not* raise |
| `27_production_triage_service.py` | route → triage → policy → structured decision record |

`run_all.sh` exits non-zero if any example fails, so it doubles as a smoke test of the whole API
surface. It needs the checkpoints, so it is not part of CI; `python -m compileall examples/` is.
