# Spanish phone turns: routing a caller's sentence to an action

217 frozen Spanish sentences said to a telephone voice agent, a fine-tuning recipe that takes
`laya-multilingual` from **0.396 to 0.912** on them, and the language model that agent uses today
measured on the same sentences as the bar to clear.

The task is the one a voice agent faces on every turn. The caller said something; the agent can
keep talking, transfer the call to one of a closed list of destinations, or end the call. It is
one six- or seven-way `choice` question over one short sentence.

**These are hand-written and hand-labelled fixtures with a label policy fixed before any model
was run on them. They are not an independently annotated corpus and not an official
evaluation.** Read [What this does not show](#what-this-does-not-show) before quoting a number.

## Start here: verify offline

From the repository root, Python 3.10+:

```bash
python research/benchmarks/es_phone_turns/audit.py
python -m unittest discover -s research/benchmarks/es_phone_turns/tests -v
```

No model download and no network. The audit needs only the standard library. It re-checks the
hashes of the five frozen files, checks every record against the case it answers, and recomputes
every summary from the per-case records. The tests break a copy of the archive five different ways
and require the audit to refuse each one.

## The five frozen sets

| Set | Cases | Menu | Who wrote the sentences |
|---|---:|---|---|
| `cases` | 112 (+4 borderline) | telephone company: `ventas`, `soporte`, `facturacion`, `operadora` | the author of this benchmark |
| `clinic` | 45 | clinic: `citas`, `laboratorio`, `farmacia`, `emergencias`, `recepcion` | the author of this benchmark |
| `independent_cases` | 31 | telephone company | `glm-4.7-flash`, then labelled by hand |
| `independent_clinic` | 20 | clinic | `glm-4.7-flash`, then labelled by hand |
| `frustration` | 9 | telephone company | a native Honduran Spanish speaker |

Every menu also has `colgar` (end the call) and `ninguno` (keep talking). The label policy is at
the top of [`build_cases.py`](build_cases.py). Two rules carry most of the weight:

- A question the agent can answer by itself (opening hours, address) is `ninguno`.
- A request that **negates** a transfer or a hang-up ("no me pase con ventas") is `ninguno`.
  Acting on the destination that was named is the exact mistake a phone agent cannot make.

The independent sets exist because the first two share an author with part of the training
data's design. A different language model wrote 96 sentences from briefs about negation; they
were read one by one, labelled by hand under the same policy, and the ones two readers could
label differently were dropped. 51 were kept.

## Results

Accuracy, and after the slash the number of **wrong actions**: turns where the model transferred
or hung up and should not have, or transferred to the wrong place. Answering `ninguno` by mistake
costs a slower turn; a wrong action sends a caller somewhere they did not ask to go.

| Model | `cases` | `clinic` | `independent_cases` | `independent_clinic` | `frustration` | Pooled (217) |
|---|---:|---:|---:|---:|---:|---:|
| `laya-multilingual`, zero-shot | 0.384 / 64 | 0.489 / 21 | 0.419 / 18 | 0.400 / 12 | 0.000 / 9 | **0.396 / 124** |
| fine-tuned, this recipe (`v5`) | 0.875 / 9 | 0.978 / 1 | 0.968 / 0 | 0.900 / 1 | 0.889 / 1 | **0.912 / 12** |
| `qwen2.5:7b-instruct`, tool calling | 0.920 / 6 | 0.911 / 3 | 0.871 / 4 | 0.850 / 2 | 1.000 / 0 | **0.908 / 15** |
| always the most frequent label | 0.313 | 0.289 | 0.452 | 0.250 | 1.000 | |

The language model is not deterministic (temperature 0.3, as the agent runs it). Over repeated
runs it scored 0.911 to 0.938 on `cases` and 0.867 to 0.933 on `clinic`; the table has the last one. With sets this size a
difference of two or three points between two rows is noise. What the table supports is that the
fine-tuned checkpoint and the language model are level, and that the zero-shot checkpoint is not
usable for this.

### Where each one fails

Pooled over the five sets:

| Kind of sentence | fine-tuned `v5` | `qwen2.5:7b-instruct` |
|---|---:|---:|
| names an area and refuses it (`refused`) | 16 / 17 | 11 / 17 |
| refusal written by the benchmark author (`negation`) | 20 / 22 | 18 / 22 |
| a real need stated with a negation (`need-with-no`) | 25 / 26 | 25 / 26 |
| refuses one area and asks for another (`redirected`) | 5 / 6 | 6 / 6 |
| names an area that is not the one that must act (`hard`) | 7 / 9 | 9 / 9 |
| angry at the agent (`frustration`) | 8 / 9 | 9 / 9 |

The mistakes are complementary. The language model acts on refusals: "No me pasen con soporte
técnico, llámeme directo al celular" ended the call. The fine-tuned checkpoint is weaker when a
sentence mentions two areas.

### Zero-shot: the option descriptions matter more than the sentence

| Checkpoint | criteria in Spanish | criteria in English |
|---|---:|---:|
| `laya-multilingual` on `cases` | 0.384 | 0.589 |
| `laya-typed-decisions` on `cases` | 0.384 | 0.616 |
| `laya` on `cases` | 0.313 | 0.634 |

The sentences are Spanish in both columns. What changes is the language of the instructions and
of the option descriptions. Zero-shot, the checkpoints read a Spanish sentence against English
options far better than against Spanish ones. Negation is the floor either way:
`laya-multilingual` with English criteria got 3 of 15 refusals right and answered "No me pase con
ventas, solo quiero saber el horario" with `ventas` at confidence 1.00.

### Latency

| | p50 | Where |
|---|---:|---|
| fine-tuned, PyTorch | 60 ms | Apple M2 Ultra, CPU, 4 threads |
| fine-tuned, PyTorch | ~130 ms | same machine, inside the repository's Docker image (linux/arm64) |
| fine-tuned, ONNX fp32 | 60 ms | Apple M2 Ultra, CPU, 4 threads |
| fine-tuned, ONNX int8 | 39 ms | Apple M2 Ultra, CPU, 4 threads |
| `qwen2.5:7b-instruct`, first token | ~405 ms | same machine, GPU, through Ollama |

The language model also reads about 380 tokens and writes about 25 on every turn.

### INT8 changes decisions on this checkpoint

| | same decision as PyTorch | max confidence drift | accuracy |
|---|---:|---:|---:|
| ONNX fp32 | 221 / 221 | 0.000 | 198 / 217 |
| ONNX int8, per-channel (`export_onnx.py --quantize`) | 197 / 221 | 0.459 | 188 / 217 |

The exporter's note reports no flipped decision on 20 English states. On this fine-tuned
multilingual checkpoint it flips 24 of 221 and costs ten correct answers. Measure before
shipping the int8 graph of a fine-tuned checkpoint.

### Confidence is a weak gate

Pooled, `answer_confidence` separates right from wrong with AUROC 0.72 on `v5`. If only actions
at or above a threshold skip the language model:

| Threshold | Turns that take the fast path | Wrong actions among them |
|---:|---:|---:|
| none | 140 (65%) | 12 (8.6%) |
| 0.70 | 133 (61%) | 9 (6.8%) |
| 0.80 | 129 (59%) | 7 (5.4%) |
| 0.84 | 112 (52%) | 6 (5.4%) |
| 0.88 | 24 (11%) | 2 (8.3%) |

Confidence sits between 0.83 and 0.89 for almost every answer, right or wrong. Two things push
it there: the soft targets (0.94 and 0.98 here; with 0.90 the ceiling was 0.85) and the fitted
temperature, which came out above 1 in every run (1.18 to 1.37). A threshold moves the trade-off;
it does not remove the wrong actions.

## The training recipe

```bash
cd research/benchmarks/es_phone_turns/train
python generate.py          # needs a local Ollama server; cached answers are in raw/
python build_train.py       # 4,740 cases -> train.jsonl, seeded, no network
cd ../../../..
python research/scripts/finetune_single_device.py \
    --data research/benchmarks/es_phone_turns/train/train.jsonl \
    --output-dir out/es-phone --device mps --epochs 4
```

Four epochs take about half an hour on an Apple M2 Ultra (`--device mps`). `train.jsonl` is not
committed: `build_train.py` rebuilds it, byte for byte, from the cached answers in
[`train/raw/`](train/raw).

What is in the 4,740 cases:

| Source | Cases | Answer |
|---|---:|---|
| `indirect`: a need, the area not named | 984 | the area |
| `direct`: asks for the area by name | 374 | the area |
| `needs_with_no`: a need stated with a negation | 387 | the area |
| `redirected`: refuses one area, asks for another | 568 | the area asked for |
| `negated`: refuses an area, written by the language model | 323 | keep talking |
| `negated_named`: refuses an area, written by hand as templates | 239 | keep talking |
| `open`: a question the agent answers | 191 | keep talking |
| `greeting`, `frustration`, `negated_hangup`, `negated_human` | 786 | keep talking |
| `human` | 666 | a person |
| `farewell` | 222 | end the call |

Decisions that mattered, in the order they were found:

1. **Thirteen trades, 49 destinations, none of them in the tests.** No training destination
   shares a label with a test destination, and the three answers every menu has appear under
   several names (`terminar`, `despedida`, `cortar`...). Each case draws its menu at random, in
   random order, and one in eight has no descriptions. The tests ask the model to read a menu it
   has never seen.
2. **The language model writes sentences and never labels one.** Each request is for sentences of
   a kind known beforehand, so the label comes from the request. `qwen3.6:35b` wrote the training
   sentences; a different model wrote the independent tests.
3. **Negation has to point both ways.** With refusals only, every "no" pointed at "keep talking".
   `needs_with_no` ("no me funciona", "nunca me llegó") are requests that use the same words.
4. **Hand-written refusal templates are not what makes negation work.** Without the 239
   `negated_named` cases (`v3`) refusals were still 15 of 15 and 7 of 7.
5. **Inside one trade no two destinations may claim the same sentence.** An early menu had both
   `tarjetas` and `reclamos`; "a charge I do not recognise on my card" fits both, and its label
   depended on which one the generator had been asked for.
6. **A refusal that also asks for someone is dropped from the refusals.** "No me conecten con
   cuentas, necesito hablar con tarjetas" is a request, and the generator cannot know for whom.
7. **Thirty percent of the cases are rewritten the way a speech recogniser hands them over**: no
   capitals, no punctuation, sometimes no accents, a filler or a repeated word at the start.
8. **Anything close to a test sentence is dropped** (token-set Jaccard of 0.7 or more): 12 cases.

### The variants

| | What changed | Pooled accuracy | Wrong actions |
|---|---|---:|---:|
| `v1` | first recipe, soft targets at 0.90 | `cases` 0.902, `clinic` 0.933 | 7, 3 |
| `v2` | targets at 0.94 / 0.98 | 0.894 | 13 |
| `v3` | `v2` without the hand-written refusal templates | not run on `frustration` | |
| `v4` | `v2` plus `redirected`, refusals that ask for someone dropped | 0.894 | 16 |
| `v5` | `v4` plus `frustration` | 0.912 | 12 |

Only `v5` can be rebuilt from the code as committed. The earlier ones are kept because the
results are what motivated each change.

## What this does not show

- **`v4` and `v5` were designed after looking at test results.** `redirected` was added because
  the independent sets showed 1 of 6, and `frustration` because that set showed 7 of 9. The six
  `redirected` cases and the nine `frustration` cases measured a known gap being closed, not an
  unseen one. The training sentences for both were generated separately and none is close to a
  test sentence, but the *kinds* were chosen with the tests in view.
- **The sets are small.** 217 cases put roughly ±4 points of sampling error on the pooled number
  and far more on any single row of the per-kind table.
- **One person labelled everything.** There is no second annotator and no agreement figure. Four
  cases in `cases` are marked `borderline` and reported apart for that reason.
- **The sentences are written, not recorded.** None went through a telephone line and a speech
  recogniser. The `asr` cases imitate that text; they are not that text.
- **Most of it is one author's Spanish**, Central American, addressed as *usted*. The
  `frustration` set is the only one written by someone who was not also designing the benchmark.
- **The language model baseline is one model with one prompt.** It mirrors how a production voice
  agent calls its model: two tools, a one-line system prompt, streaming, 120 tokens. A different
  prompt would move it, and a larger model likely scores higher.
- **Only `choice` was trained and measured.** What fine-tuning did to `score` and `noul` on this
  checkpoint was not checked.

## Files

| | |
|---|---|
| [`build_cases.py`](build_cases.py) | the frozen cases and the label policy; writes `data/*.jsonl` |
| [`prompts.py`](prompts.py) | the menus and the three renderings of the question |
| [`run.py`](run.py) | runs one checkpoint on one set, archives every decision |
| [`baseline_llm.py`](baseline_llm.py) | the language model baseline; needs a local Ollama server |
| [`metrics.py`](metrics.py) | every metric, from per-case records, standard library only |
| [`audit.py`](audit.py) | re-derives every number from the archive |
| [`train/`](train) | the trades, the generator, the builder and the cached sentences |
| [`results/`](results) | one file per model, set and rendering |
