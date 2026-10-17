# Paired context reference example

This archive exercises the evaluator on 12 synthetic **English** support requests.
It is a demonstration of paired analysis, not a production benchmark, an estimate
of attack prevalence, or a reproduction of the unpublished probe in
[#1229](https://github.com/NandhaKishorM/laya/issues/1229).

## Fixed protocol

- Fixture: [context_robustness_choice.jsonl](../../eval/fixtures/context_robustness_choice.jsonl).
  Four teams, three requests per team, two benign/adversarial control pairs per request.
- Checkpoint: convaiinnovations/laya at 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851,
  root and multilingual subfolder. Both receive the same English inputs.
- CPU eager, float32, AMP off; max_len=512, head_max_len=192.
- min_confidence=0.7 is a fixed illustrative policy, **not** a calibrated recommendation.
- Measured source commit: aba2c6cf51ce5777d4a7b875393b0fd03881648a,
  based on ca5045a3f1fac06fc6daa5cf3371110f322f5255 (Laya 0.4.9).
  Later archive/documentation changes do not alter the measured evaluator or core.
- Python 3.11.13, Windows, torch 2.14.0, transformers 5.17.0. Exact loaded
  temperatures, artifact hashes, package versions and source identities are in each manifest.

The fixture was written before observing model responses. A predetermined first
clean-case compatibility smoke was inspected before freezing [protocol.json](protocol.json);
the fixture and model selection were not changed. This is not a blind benchmark.
The first multilingual attempt failed before any predictions because the offline
cache lacked its weight file. After fetching that same pinned file, collection
started in a fresh directory with the unchanged protocol; no partial runs were combined.

## Observed counts

Each model made 60 calls: 12 clean, 24 benign, 24 adversarial. The clean answer is
shared within a request; the 24 pairs are not 24 independent requests.

| Measure | English checkpoint | Multilingual checkpoint |
| --- | ---: | ---: |
| Unique clean correct | 12/12 | 9/12 |
| Benign correct | 24/24 | 21/24 |
| Adversarial correct | 16/24 | 13/24 |
| Clean/adversarial CC, CW, WC, WW | 16, 8, 0, 0 | 12, 6, 1, 5 |
| New error given clean correct | 8/24 | 6/18 |
| New wrong-and-accepted given clean correct | 5/24 | 6/18 |
| Gate miss among new errors | 5/8 | 6/6 |
| Matched error delta, on the same clean-correct triplets | 8/24 | 5/18 |
| Truncated inputs | 0/60 | 0/60 |

The multilingual matched delta is 5/18, not 6/18: one of its six new errors is
also wrong under the linked benign control. Its benign/adversarial table on the
18 clean-correct triplets is CC=12, CW=5, WC=0, WW=1. This is why a clean/attack
comparison alone does not answer the matched-control question.

State token counts range from 21 to 56 for English and 19 to 54 for multilingual;
no state tokens were dropped. Controls share the field and style of their linked
variants but are not token-length matched. The report does not isolate a causal
effect of malicious wording. Twelve hand-written requests cannot establish which
checkpoint is safer, and the multilingual checkpoint here is not a multilingual
language evaluation.

## Rebuild and inspect

Each directory contains the full manifest, raw observations, derived report and
three direct-API parity observations. For each checkpoint the first request's
clean, policy_control and policy inputs were separately passed to Agent.predict
with the manifest's model, revision, subfolder and settings, without the collector
or its runner. All six complete responses exactly matched the corresponding saved
responses, including usage and gate fields. These checks test collection fidelity,
not correctness of the gold labels.

From the repository root, rebuilding reports requires no model load:

~~~bash
python -m research.eval.context_robustness summarize research/results/context_robustness/english --out english-rebuilt.json
python -m research.eval.context_robustness summarize research/results/context_robustness/multilingual --out multilingual-rebuilt.json
~~~

The output files must not already exist. Compare parsed JSON with the archived
report.json. Use the recorded evaluator/semantic sources; a later incompatible
scoring or gating implementation is intentionally refused.

To recollect, use new directories and the same pinned model:

~~~bash
python -m research.eval.context_robustness run research/eval/fixtures/context_robustness_choice.jsonl \
  --model convaiinnovations/laya --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
  --max-len 512 --head-max-len 192 --min-confidence 0.7 --out fresh-context-english
python -m research.eval.context_robustness run research/eval/fixtures/context_robustness_choice.jsonl \
  --model convaiinnovations/laya --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
  --subfolder multilingual --max-len 512 --head-max-len 192 --min-confidence 0.7 \
  --out fresh-context-multilingual
~~~

Checkpoint files can be prepared in the normal Hub cache before enabling offline
mode. Numerical parity on other environments is not guaranteed; record their own
provenance instead of treating these saved responses as new measurements.
