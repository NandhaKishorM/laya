# Helpers

## Language detection

`laya.detect_language` is `laya.lang.analyse`.

::: laya.lang.analyse

::: laya.lang.detect_script

::: laya.lang.is_english

## Email

::: laya.email.clean_email_body

::: laya.email.email_state

## Question presets

::: laya.presets.triage_questions

::: laya.presets.email_questions

::: laya.presets.guard_questions

::: laya.presets.moderation_questions

::: laya.presets.router_questions

## Pre-run request sizing

What a request will fit before any forward pass. These are the helpers `predict_long` itself
reads, so a question can be measured without holding an Agent: `build_head` is the question half
of a sequence, `state_room` is what is left for the state after it, and `build_sequence` reports
what the head budget did to the options and to the state.

```python
from transformers import AutoTokenizer
from laya.common import build_sequence, collapsed_options, state_room, window_budget

tok = AutoTokenizer.from_pretrained("convaiinnovations/laya", subfolder="tokenizer")
max_len, head_max_len = 512, 192      # from the checkpoint's rl_agent_config.json

question = {"type": "choice", "instructions": "What does the customer want?",
            "criteria": {"refund": "money back", "cancel": "stop the service"}}
# The short-key internal shape the sizing helpers read; inline the mapping as
# questions-and-answers.md shows rather than reaching into Agent._to_internal.
q = {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}
state = "I was charged twice for the same invoice."

state_room(tok, q, max_len, head_max_len)   # state tokens this question leaves room for
seq, markers, stats, trunc = build_sequence(
    tok, state, q, max_len, head_max_len, return_stats=True, return_truncation_stats=True)
stats       # how many options the head budget kept distinct, and the cap it applied
trunc       # state tokens, used, dropped, and whether the state was truncated
collapsed_options(["q"], [{"options": stats}])   # {} until an option loses its own span
window_budget(tok, [q], max_len, head_max_len)   # (window, stride, room) a scan may use
```

The same numbers come back after a call in `usage`: `state_tokens`, `state_tokens_dropped`,
`truncated`, `truncated_questions`, and `options` when an option lost its span. That block is
the record of what a run did; the helpers above are how to find out before paying for one.
`predict_long` hands a state straight to a single pass when its tokens fit the room its
questions leave, and a scan window is capped at that room with a 50%-of-window stride -- a cap
large enough to multiply the passes warns before the scan runs.

None of this picks a strategy: tournament rounds come back as `tournament[qid]["rounds"]`
after `predict_tournament`, and shortlist ranking quality stays with whatever embedder
supplies it.

::: laya.common.state_room

::: laya.common.window_budget

::: laya.common.collapsed_options

## Shortlisting

::: laya.shortlist.shortlist_choice

::: laya.shortlist.predict_shortlist

::: laya.shortlist.predict_tournament

::: laya.shortlist.embed_fn_from_agent

::: laya.shortlist.cached_embed_fn

## Abstention

::: laya.confidence.check_min_confidence

::: laya.confidence.check_min_confidence_map

::: laya.confidence.resolve_min_confidence

::: laya.confidence.flag_low_confidence

::: laya.confidence.apply_confidence_gate

::: laya.confidence.GATE_STATES

## Calibration and training

::: laya.common.answer_confidence

::: laya.common.confidence_from_probs

::: laya.confidence.jev_confidence

::: laya.common.ece_score

::: laya.calibrate.fit_temperatures

::: laya.calibrate.fit_one_temperature

::: laya.calibrate.fit_temperature_map

::: laya.calibrate.fit_abstention_thresholds
::: laya.calibrate.fit_binning_map

::: laya.calibrate.apply_binning_map

::: laya.agent.Agent.fit_binning

::: laya.common.render_options

::: laya.common.proper_reward

::: laya.common.td_lambda_targets

::: laya.common.QTYPES

::: laya.common.QTYPE_NAMES
