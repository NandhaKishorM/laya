# Benchmarks and known limits

Laya's results depend on the checkpoint, task, question wording, option count and hardware. Use
the [full benchmark tables](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
to find a comparable run before choosing a checkpoint or a confidence threshold. The
[research directory](https://github.com/NandhaKishorM/laya/tree/main/research) holds the scripts
and result files behind the main tables.

## Find the relevant measurement

| If you need to assess | Start with | Check before applying the result |
|---|---|---|
| Decisions across languages | The MASSIVE and XNLI tables in `BENCHMARKS.md` | Language, task, number of options and checkpoint |
| A workflow such as triage or moderation | The application-workflow table in `BENCHMARKS.md` | Whether the dataset was in the training mix or held out |
| The `laya-typed-decisions` checkpoint | The typed-decisions table in `BENCHMARKS.md` | It was fine-tuned on that benchmark's training split; the base checkpoints have separate rows |
| Response time | The T4, GB10, laptop CPU and server CPU sections in `BENCHMARKS.md` | Device, batch size, number of questions, warm-up and whether HTTP time is included |

The published Jev figures alongside the original Laya suites come from third-party studies
with different prompts and sample sizes. They are useful context, but are not a controlled
head-to-head run. See the comparison notes in
[`research/README.md`](https://github.com/NandhaKishorM/laya/blob/main/research/README.md).

Read accuracy together with the baseline and data split. For example, the typed-decisions
benchmark reports 0.766 accuracy for the fine-tuned checkpoint, while both base checkpoints
score below its 0.461 majority-class baseline. That result supports fine-tuning for a similar
task; it does not establish 0.766 accuracy for an untrained checkpoint or a new domain.

Read confidence separately from accuracy. Expected calibration error (ECE) measures how well
reported probabilities match observed correctness; lower is better. The 51-language sweep's
original ECE and mean-confidence columns predate the temperature clamp in #42. Its accuracy
columns still apply, but use the clamped rerun in `research/results/` when comparing current
confidence values. Even a lower ECE on one suite does not set a safe threshold for another task
or option count.

## Limits to check on your own data

- **Language routing:** The English checkpoint can be confident on text it handles badly outside
  English. Use `Router` for mixed-language inputs and check routing decisions on the languages
  you serve. The multilingual checkpoint also scores below the English checkpoint on the English
  MASSIVE and XNLI slices.
- **Many options:** Choice descriptions share a fixed token budget. The 77-label Banking77 run
  performs poorly at the default budget. Keep a single choice question to roughly 20 options,
  or evaluate a shortlist and a larger head budget on your own labels.
- **Calibration:** Both base checkpoints are over-confident on the published suites as shipped,
  yet a separate routing task was under-confident. Fit and evaluate temperatures on separate,
  held-out examples from your workflow before using a confidence gate.
- **Task transfer:** Held-out moderation is weak in the application benchmark, and ordinal
  `score` is the weakest primitive in the reported English suites. The multilingual checkpoint
  also has a measured bias against the first `score` level. Test the actual question type and
  data distribution you intend to serve.
- **Wording and order:** Option order can change a `choice` answer. Boolean-word choice labels
  and negated requests have also failed in documented examples; `noul` can follow its option
  labels instead of the state. Check alternate option orders and wording, especially when a wrong
  decision is costly.
- **Long documents:** The multilingual encoder can read up to 8,192 tokens when configured for
  that limit, but the [long-context benchmark](https://github.com/NandhaKishorM/laya/blob/main/research/results/long_context_multilingual.json)
  reports less reliable answers beyond about 4,000 tokens of preceding text. Measure accuracy at
  the lengths you expect in use.
- **Latency:** The T4 figures do not predict CPU or cold-load time. Measure warm and cold calls
  with your own checkpoint, device, input lengths and number of questions.

The [README's Honest limits section](https://github.com/NandhaKishorM/laya#honest-limits)
has examples and current workarounds. The benchmark tables give the dataset and hardware
behind each of the limits above.

## Reproduce or extend a result

Start with the [script and result map](https://github.com/NandhaKishorM/laya/blob/main/research/README.md)
and the run index at the top of `BENCHMARKS.md`. `research/scripts/bench_local.py` runs the
51-language CPU sweep, `bench_apps.py` covers application workflows, and
`bench_latency.py` measures routing and inference speed. The T4 notebook is generated from
`research/scripts/build_benchmark_nb.py`; edit the generator when changing that benchmark.

For a new deployment, keep a held-out set with the same states, questions and expected answers
for every checkpoint you compare. Record the checkpoint revision, Laya and library versions,
device, question count, option count and token budget with each run. Include a simple baseline
for accuracy and report latency after warm-up as well as first-use load time. This makes your
result comparable to the published runs and lets you revisit it after an upgrade.
