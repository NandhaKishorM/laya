# vegaml inside laya: `Agent(vegaml=True)`

One argument swaps the engine under laya's surface for [vegaml](https://github.com/NandhaKishorM/vegaml)'s:
a frozen 0.8B universal reader with a small physics engine on top. Every method keeps its laya
name, and the answers gain what that engine knows that an encoder does not. The class behind it
is `laya.VegaAgent`, in `laya/vega.py`.

```bash
pip install -U laya               # vegaml ships with laya 0.4.23 and newer
```

```python
import laya

agent = laya.Agent(vegaml=True)                 # the 0.8B universal reader (nandakishorm/vega-U), the default

# Switching the model underneath is one string. Every option keeps the same engine, questions and fields:
#   laya.Agent(vegaml="4b")                                # the Qwen3.5-4B reader from the same Hub repo
#   laya.Agent(vegaml="gemma-1b")                          # the gemma-3-1b-it reader, same repo
#   laya.Agent(vegaml="nandakishorm/vega-U")               # any Hub repo id holding a vega-U checkpoint
#   laya.Agent(vegaml="./my-reader/")                      # a local dir from VegaAgent.attach(...), your own LLM ported in
#   laya.Agent(vegaml=True, backbone="Qwen/Qwen3.5-0.8B")  # name the reader weights explicitly, e.g. a mirror
# Without the argument, laya's own encoders, also by Hub name:
#   laya.Agent("convaiinnovations/laya")                   # English, 33 ms on a T4
#   laya.Agent("convaiinnovations/laya-multilingual")      # 100+ languages, up to 8,192 tokens with max_len=8192
#   laya.Agent("convaiinnovations/laya-typed-decisions")   # fine-tuned for the typed-decisions workflows

out = agent.predict(state, questions)
ans = out["answers"]["department"]
ans["choice"]                                    # as always
ans["probabilities"], ans["answer_confidence"]   # as always: the calibrated one is max p
ans["prediction_set"]                            # the conformal set at alpha (0.1)
ans["abstain"]                                   # the engine's own refusal
ans["unbound"]                                   # far from every well: likely off-distribution
ans["route"], ans["route_reason"]                # "auto" or "review", decided for you
```

## What changes, feature by feature

| | laya checkpoints | `Agent(vegaml=True)` |
|---|---|---|
| reader | ModernBERT encoder, ~150M | frozen decoder reader, 0.8B (or the 4B) |
| context | 512 or 1,024 tokens, `predict_long` windows past that | 73k tokens, read whole |
| learning from your labels | gradient fine-tune | `agent.fit(rows)`: a head per question over the frozen features, seconds |
| calibration | a shipped temperature map | refit on your rows: temperature, conformal threshold, escalation band |
| refusal | `min_confidence` you choose | `abstain`, `unbound`, the conformal set, the fitted band |
| images, speech, video | no | `predict(image=...)`, `predict(audio=...)`, `predict(video=...)` |
| speed on text (T4) | ~16 ms a decision batched | ~72 ms a decision; each served head adds a read |
| memory | under 1 GB | 2 to 4 GB |

## Fit and calibrate

```python
rows = [{"state": "...", "questions": questions, "gold": {"department": "billing"}}, ...]
report = agent.fit(rows)            # heads, then temperature + conformal + band, one call
report["serving"]                   # the questions whose head actually beat the engine
agent.predict(state, questions)     # heads serve where they earned it; route from the band
agent.save_calibration("fit/")      # heads.npz + calibration.json
agent.load_calibration("fit/")      # on a fresh VegaAgent, identical answers and routes
```

Rows are the same `{"state", "questions", "gold" or "expected"}` shape laya's fine-tune reads,
so a corpus prepared for one fits the other unchanged. A path to a CSV works too: a `state`
column and one column per question id holding the gold label, as in
[`examples/data/routing_labels.csv`](https://github.com/NandhaKishorM/laya/blob/main/examples/data/routing_labels.csv),
60 messages routed to billing, HR, PR, sales or technical with an urgency level:
`agent.fit("examples/data/routing_labels.csv", questions=questions)`. The fewest rows `fit`
accepts is `laya.VegaAgent.min_rows(questions)`: five per option of the widest question for the
heads, plus a quarter held out inside `fit` to calibrate on, never fewer than ten, so 35 for a
five-way choice and 20 for a yes/no question. A quarter held out is the rule because a head is
right on its own rows: calibrating there rails the temperature to its floor and puts every
confidence at 1.000. When the calibration rows cannot certify the review line at
`target_precision`, the band is reported `None` for that question and the conformal set routes
its answers, with `route_reason` saying so. The full measured loop, before and after on held-out
rows with an ablation over the number of fitting rows, is `examples/42_fit_and_calibrate_vegaml.py`
and its output is in the README. A head is kept only if it beats chance
under cross-validation and served only if it also beats the engine on the same rows:
`report["heads"][qid]["used_by_auto"]` is the fact to read, not `fitted`.

`agent.calibrate(rows)` (also reachable as `fit_temperatures`) does the calibration half alone.
The argmax is invariant under a temperature, so it moves confidence, sets, `abstain` and
`route`, never which option wins.

## Automatic thresholding

Nothing here needs a confidence number from you. `route` is decided per answer, in this order:

1. A fitted band, if `fit` or `calibrate` has run: its line for that question was placed where
   precision holds at `target_precision` (0.95) on your rows, with a Wilson bound to spare.
2. Otherwise the engine's own machinery, with no labels at all: `review` when it abstains,
   when the answer is `unbound`, or when the conformal set at `alpha` (0.1) holds more than
   one option; `auto` only when all three agree.

`alpha` and `target_precision` are targets with defaults, not thresholds to guess at.
`agent.thresholds()` prints what is deciding right now. laya's `min_confidence` is still
accepted and applied on top when given.

Gate on `route`, not on `confidence`. A fitted head can report 1.000 on an answer it gets wrong;
`route` on the same answer is still right, because it comes from the set and the band rather
than the top probability.

## Images, speech, video, long documents

```python
# text: up to 73,728 tokens in one pass; usage["truncated"] says if a state hit the limit
agent.predict_long(document, questions)
# image: PIL image, path or HxWx3 array, any format Pillow opens; scaled to ~1 megapixel, no file-size cap
agent.predict(image=img, questions=questions)
# audio: path or bytes (wav, flac, ogg, mp3; needs soundfile) or a 1-D waveform with sampling_rate=;
#        any length, resampled to 16 kHz, transcribed in 30 s windows, transcript counts toward the 73,728 tokens
agent.predict(audio=wave, questions=questions, sampling_rate=16000)
# video: path or bytes, any container ffmpeg opens, any length and resolution (needs the ffmpeg binary);
#        speech is transcribed and decided as text, a silent clip is decided on 4 sampled frames as one image
agent.predict(video="call.mp4", questions=questions)
```


The modality is chosen by which argument you passed; the questions and the answer shape are the
same. An audio answer carries `transcript` and `asr`, because a decision made on words you never
saw cannot be audited. `predict_image_text` answers the same questions with the image withheld,
which measures what the picture itself contributed.

## Model switching

```python
from laya.vega import Router

router = Router(fast=laya.Agent(), vega=agent)
out = router.predict(state, questions)
out["routing"]                     # {"model": "fast" | "vega", "reason": "..."}
```

The rule, in order: an image or audio goes to vega; a state longer than the fast model's context
goes to vega; non-English or non-Latin text goes to vega; a question with a fitted head on vega
goes to vega; everything else goes to the fast encoder. `model=` forces one. The result carries
`routing` in the shape `laya.Router` already returns.

## Your own reader

```python
laya.VegaAgent.attach("Qwen/Qwen3.5-4B", texts, "my-reader/")   # no labels needed
agent = laya.Agent(vegaml="my-reader/")
```

## What stays the same, and what is not here

`predict`, `system_one`, `predict_batch`, `predict_long`, `fit_temperatures`,
`save_calibration`, `load_calibration` and the hooks all keep their names. `predict_batch` is a
loop: vegaml packs a state's questions into one pass, not many states. `fit_binning` raises,
because histogram binning is laya's own calibration and on vegaml the calibrated probabilities
come from `calibrate()`. Without the `vegaml` package installed, `laya.vega` is never imported
and `Agent(vegaml=True)` says what to install.
