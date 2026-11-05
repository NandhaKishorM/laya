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

agent = laya.Agent(vegaml=True)                 # the 0.8B universal reader
agent = laya.load(vegaml="4b")                  # a larger reader; or a Hub id, or a local dir

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
so a corpus prepared for one fits the other unchanged. A head is kept only if it beats chance
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
agent.predict(image=img, questions=questions)                        # PIL image, path, or array
agent.predict(audio=wave, questions=questions, sampling_rate=16000)   # transcribed, then decided
agent.predict(video="call.mp4", questions=questions)                  # speech if any, else frames
agent.predict_long(document, questions)                               # one pass up to 73k tokens
```

Image, audio and video, in one paragraph: an image is a PIL image, a path or an HxWx3 array in any format Pillow opens, and it is scaled to about one megapixel (1,048,576 pixels) before the reader sees it, so a larger or sharper file costs nothing more and gains nothing; there is no file-size cap of its own, memory is the bound. Audio is a path or raw file bytes (wav, flac, ogg, mp3, with `pip install soundfile`) or a 1-D numpy waveform with `sampling_rate=`; any length works, stereo is averaged, the wave is resampled to 16 kHz and transcribed in 30-second windows, and the transcript then counts toward the 73,728-token context. The default transcriber is English-only; pass `asr="openai/whisper-tiny", language="hi"` for other languages. Video is a path or raw file bytes in any container ffmpeg opens (mp4, mov, webm, mkv), decoded through the `ffmpeg` binary on your PATH, with no Python dependency; the sound track is extracted at 16 kHz and transcribed, and if the transcript holds real words the decision is made on it through the text path, with heads, calibration, set and route all applying. A silent clip, or one whose sound is music or noise, has `frames` (4 by default) evenly spaced frames tiled into one contact sheet and decided through the image path, which is zero-shot. Any length and resolution work, since only the frames sampled and the transcript reach the reader; `out["video"]` says which path decided, the frame times or the transcript, and the duration.

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
