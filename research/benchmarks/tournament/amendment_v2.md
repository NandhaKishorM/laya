# Review-driven evaluation amendment

This extends the [fixed-k20 study](dedicated_embeddings.md) following
[Dylan's review](https://github.com/NandhaKishorM/laya/issues/102#issuecomment-6120484926).
It asks two separate questions: does Laya add value over embedding top-1, and does
an additional tournament earn its cost over one decision on retrieved candidates?
The answer may favor the simpler pipeline.
See the [recorded study](study_results.md) for completed results and portable verification/selection commands.

The fixed study's test outcomes were already available when these rules were set.
All new analyses are exploratory. A frozen amendment does not turn them into an
independent confirmation. Preserve the original outputs and protocol; write new
reports to a separate directory. The entry point refuses to overwrite a report.

## Versions and evidence

| component | version / rule |
|---|---|
| original quality | unchanged fixed k20, group size 16, model screening on development rows |
| top-1 supplement | paired predictions from the original reports; 95% exploratory intervals |
| BANKING77 development selection | `amendment-v2`; fixed grid, two stages, no test reselection |
| BANKING77 label-order sensitivity | `amendment-v2`; MiniLM/k20, seeds 11 and 29 |
| online latency | `latency-v2`; cold queries, resident label vectors, row-interleaved arms |

Original BANKING77 primary comparisons retain their 97.5% intervals. Supplementary
top-1 comparisons use gained/lost counts and 10,000 paired bootstrap replicates with
seed 0 and 95% intervals. These intervals do not control a new family-wise error rate.
Pair only matching dataset/checkpoint/question identities, embedding prefixes, row
IDs, content hashes and gold labels. Never compare a paired result to a historical
scalar. Repeated timing passes and order permutations are not extra independent rows.

## Bounded development selection

Only BANKING77 is tuned. Keep MiniLM from the original lightweight model screening
and retain Nomic as the reference. Do not select the model family again. Test each
method (`shortlist` and `combined`) at k in **5, 8, 12, 20, 32, 48**, group size 16.
MiniLM has no prefixes. Nomic uses `search_query: ` for queries and either
`search_document: ` (`nomic-qd`) or `search_query: ` (`nomic-qq`) for labels.
Weights, pooling, dtype, token caps, state and label rendering remain fixed.

Select on the first 200 of the original 600 development rows. Maximize the exact correct count;
exact ties prefer smaller k, then QD over QQ. Both methods see the full candidate
grid. At k<=16 the combined method performs the same single decision as shortlist;
record an explicit alias, eligible for selection, without claiming a tournament gain.
Any failed or missing cell blocks selection.

Persist each selected configuration before evaluating the remaining 400 development
rows. These rows check stability without reselection; they have some prior v1
observations and are not an untouched holdout. Evaluate the selected settings on the
original 500 BANKING77 test rows, including each configuration's matching top-1.
Keep every grid cell, including losses. No centering, new training, group-size search
or test-driven expansion to the full split is part of this amendment.

Independently selected pipelines can have different k or prefixes. Comparing them
does not isolate the causal effect of adding tournament. The retained v1 comparison
at the same prefix and k20 provides the matched-configuration mechanism comparison.

```bash
python research/benchmarks/tournament/amended_comparison.py \
    --dataset banking77 --split dev --limit 200 --variant nomic-qq \
    --strategies shortlist,combined --k 8 --diagnose \
    --device cpu --threads 4 --local-files-only --out grid-nomic-qq-k8.json
```

Run all grid cells before selection. For the selected stability run use
`--split dev --offset 200 --limit 400`; for selected test use `--split test --limit 500`.
Use `HF_HUB_OFFLINE=1` with the prepared model and dataset cache. Data/model revisions
are inherited from the fixed runner; the output additionally records source hashes,
actual devices and explicit rows. `--variant` is never inferred from a filename.

## Order sensitivity

Use MiniLM, k20, group size 16 and the original 500 BANKING77 rows. Shuffle the full
canonical 77-label list once per seed using Python's `Random(seed).shuffle`, with
exactly seeds 11 and 29. Run full tournament, shortlist and combined for each seed.
The shortlisted methods restrict the global order to the original retrieved top-20
set. Keep retrieval rank separate from decision presentation order and preserve
candidate membership. The shortlist arm controls for presentation order effects.

```bash
python research/benchmarks/tournament/amended_comparison.py \
    --dataset banking77 --split test --limit 500 --variant minilm \
    --strategies tournament,shortlist,combined --k 20 --permutation-seed 11 \
    --diagnose --device cpu --threads 4 --local-files-only --out order-11.json
```

Publish each seed against its corresponding natural-order result, including changed
predictions and gained/lost counts. Do not select a better seed. Any across-seed
uncertainty must resample case IDs together. This measures overall order sensitivity,
not an isolated causal effect of grouping.

## Online timing

`interleaved.py` implements a separate timing protocol. The fixed comparison uses
all eight original arms, MiniLM and Nomic-QD, and three datasets. Use the original 50
latency rows selected with seed 1, three passes, one process and four CPU threads.
For pass p and row i, rotate arm order by `(p + i) % number_of_arms`. Save the complete
schedule and per-request observations; do not concatenate the unfinished v1 passes.

```bash
python research/benchmarks/tournament/amended_comparison.py \
    --mode latency --dataset banking77 --split test --limit 50 --seed 1 \
    --variant minilm --strategies single,wide,builtin,tournament,top1,shortlist,combined,shortlist-wide \
    --k 20 --passes 3 --device cpu --threads 4 --local-files-only --out timing-fixed-minilm.json
```

Each arm has its own cold-query adapter. Resident encoder weights and immutable
label vectors may be shared; query results are never shared. Encode labels once
during setup and queries exactly once inside every measured request, even when a
query text equals a label. This deliberately differs from v1's bounded cache, which
can hit for repeated query text. The built-in embedder follows the same rule.

Warm each arm with three disjoint development requests using its actual call shape.
Time query encoding, ranking, every decision round and completion synchronization.
Exclude loading, label setup, diagnostics, gold-label analysis and report writing.
The measured wrapper includes lightweight per-round answer/fallback guards; tensor
placement inventories run outside the clock. No validation overhead is subtracted.
Record both models' actual tensor devices/dtypes, effective inference dtype, thread
counts, whole-process memory and p50/p95. Reject device/precision changes and CPU
fallback. Failed requests stop the run; partial samples remain failure evidence,
not a completed latency distribution. Run without competing inference jobs.

After development selection, run a separate BANKING77 comparison with selected
shortlist, selected combined, full tournament and matching top-1 for every distinct
selected prefix (four or five arms). An `--arms-file` JSON list specifies each arm's
`id`, `strategy`, `k`, `embedder` variant and `group_size`. QD and QQ share one Nomic
model's weights. Different model families require separate processes. Never pool
fixed and selected configurations or different devices into one latency table.

## Historical attribution and limits

Dylan's original measurements use a pooled/resplit 2,500-row protocol, body state,
different question/label rendering, Q/Q prefixes, batch-centered similarities and
predict-only timing. Cite those results separately, with their original source.
Q/Q in this amendment still uses our official split and independent cosine retrieval;
it is not a reproduction of that historical pipeline.

Same seed does not guarantee the same rows: a full-permutation prefix and a direct
500-item sample can use different `Random.sample` algorithms. BANKING77 and MASSIVE
indices happen to match for these fixed sizes; CLINC150 differs. Explicit manifests
are the comparison contract, and algorithm reconstruction alone does not validate
the original author's per-row manifest. A CPU result makes no claim about MPS/CUDA.

## Query format diagnostic (v2.1)

A [subsequent review](https://github.com/NandhaKishorM/laya/issues/102#issuecomment-6121525060)
identified another distinction: top-1 on the shortlist helper's instructions plus
serialized state can differ substantially from top-1 on the bare message. The
same-query comparison measures the value of a decision stage under the former
contract, not its advantage over every simpler embedding classifier.

Keep v1 and the v2 selection unchanged. Separately run bare-message retrieval for
MiniLM and Nomic-QD on the original 500 test rows of each dataset. Change only the
query to the exact `state["message"]` string before the model's existing prefix.
Keep the label text/prefix, weights, pooling, normalization and cap fixed. The
diagnostic does not load Laya, fit centering or select a new model/configuration.

```bash
python research/benchmarks/tournament/query_text_diagnostic.py \
    --dataset banking77 --split test --limit 500 --variant minilm \
    --seed 0 --device cpu --threads 4 --local-files-only --out bare-message-minilm.json
```

Reports identify the query contract and store hashes of actual encoded queries,
without distributing dataset text. Pair these predictions with the same original
rows, verifying content and gold identities. Show gained/lost counts and 95%
exploratory paired bootstrap intervals (10,000 replicates, seed 0) against the
formatted-query top-1. Comparisons against existing shortlist/combined are explicitly
**mixed-query pipeline comparisons**; they cannot isolate the effect of adding Laya.
The original model screening remains conditional on its query format. No raw-message
model screening, extra k search or raw-message latency claim is included here.

## Checks

```bash
python -m pytest research/benchmarks/tournament/test_amended_comparison.py \
    research/benchmarks/tournament/test_interleaved.py -q
```

Before a long run, verify all frozen identities and execute a small real-model smoke
check on disjoint development rows. Keep smoke outputs separate from study data.
Complete independent source, numerical and report review before publishing findings.
