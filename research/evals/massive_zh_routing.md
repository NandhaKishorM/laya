# MASSIVE zh-CN routing data recipe

This is a narrow data contribution for [#709](https://github.com/NandhaKishorM/laya/issues/709):
build choice-format data for the six-way Chinese voice-routing task already present in
[`zh_decision_bench.jsonl`](zh_decision_bench.jsonl). It adds a reproducible training-data
*candidate* and separate evaluation files. It does not claim a better checkpoint or a Jev result.

## Source and task

- Source: [Amazon MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive),
  `zh-CN`, CC BY 4.0. The script downloads Amazon's version 1.1 archive on request and
  verifies its pinned SHA256. The manifest also records the source archive hash and the
  committed benchmark's hash. Generated MASSIVE text retains Amazon's CC BY 4.0 terms;
  the code in this repository remains under its own license.
- Task: the six `scenario` values `calendar`, `alarm`, `audio`, `music`, `weather` and
  `transport` map to the six option labels used by the accepted community evaluation
  in [#557](https://github.com/NandhaKishorM/laya/pull/557). The question text and
  option descriptions are read from that committed evaluation, so the training and
  evaluation prompts cannot quietly drift apart. This extends the data preparation
  used by [zh-decision-bench](https://github.com/CodyQin/zh-decision-bench) to the
  source's official train/dev/test partitions; it does not copy that benchmark's
  sample or reported results.
- Quality rule: keep utterances of 4 to 60 characters only when all available
  MASSIVE judgments mark the intent correct, matching the community evaluation's
  filter. Source, license and original ID travel with every row; ID and scenario
  also appear in `tags` for the existing evaluation slices.

## Build and inspect

From the repository root, using Python 3.10 or later:

```bash
python research/scripts/build_massive_zh_routing.py --download
laya-evals validate .cache/massive-zh-routing/train_eval.jsonl
laya-evals validate .cache/massive-zh-routing/dev_eval.jsonl
laya-evals validate .cache/massive-zh-routing/test_eval.jsonl
```

The first run downloads a 40 MB archive to `.cache/`; later runs can use an already
downloaded archive with `--archive PATH`. The output directory must be empty, so two
runs cannot mix files from different source or option versions. Choose `--output-dir`
to retain two builds for comparison. The full source data and model weights are not
committed to this repository.

The files named `*_eval.jsonl` use the existing `{state, questions, expected}` format.
The optional command below also writes `train_hard_targets.jsonl` for the existing
single-device trainer's `{state, questions, gold}` input shape:

```bash
python research/scripts/build_massive_zh_routing.py --archive .cache/amazon-massive-dataset-1.1.tar.gz \
    --output-dir .cache/massive-zh-routing-hard --export-hard-targets
```

**Target semantics:** MASSIVE provides hard scenario labels, not teacher probability
distributions. The optional file encodes each hard label as an exact one-hot target;
every row has `target_source: hard_label_one_hot`, and the manifest repeats this fact.
The current RLCD training recipe was developed for soft teacher targets. This optional
representation needs a separate training and calibration experiment before it is
recommended as a replacement for that recipe. No soft confidences are inferred.

## Leakage audit and current counts

The builder preserves MASSIVE's official `train`, `dev` and `test` assignment and
does not move an item between them. It normalizes Unicode and whitespace, then keeps
the test copy first, the dev copy next and the train copy last when identical text
appears more than once. It also removes any utterance that matches the
committed voice-routing evaluation in #557. Conflicting labels for identical
normalized text fail the build.

For the pinned archive and current committed evaluation, 5,436 source rows use one
of the six scenarios; 4,912 pass the quality rule. The resulting files contain
3,290 train, 424 dev and 816 test rows. The audit excludes 32 train, 179 dev and 3
test rows that match the existing evaluation, plus 146 train, 10 dev and 12 test
rows that duplicate other selected text. `manifest.json` records the exact counts,
hashes and per-label distribution. The training split is imbalanced. The remaining
dev split has no `audio` example, so it cannot measure all six classes; the test
split still covers all six classes. Keep the test split untouched during tuning.

This audit detects normalized exact matches. It does not prove that translations,
paraphrases or source artifacts cannot leak across splits. The existing #557
benchmark includes MASSIVE examples; all exact matches are removed from these
exports, but using that benchmark to choose a model or prompt still affects claims
about independent generalization.

## Check

```bash
python -m pytest tests/test_massive_zh_routing.py -q
ruff check research/scripts/build_massive_zh_routing.py tests/test_massive_zh_routing.py \
    --select=E9,F63,F7,F82,F401,F811 --line-length=120
```

The tests need no download or model weights. They check source pinning, label
conflicts, train/eval overlap, one-hot target provenance and byte-identical
outputs from identical inputs.
