# Dedicated embeddings and tournament on the same requests

Issue [#102](https://github.com/NandhaKishorM/laya/issues/102#issuecomment-5999814797)
asks whether a dedicated sentence embedder followed by a tournament can beat either
retrieval followed by one decision, or a tournament over every label. `compare_embeddings.py`
runs those paths through the existing helpers. It also measures simpler alternatives;
adding a second model or a second decision round is not assumed to help.

This is a research entry point. It does not change Laya's API, defaults, dependencies,
or the earlier `benchmark.py` and `results.json`.
The [recorded study](study_results.md) includes the completed results, evidence and reproduction steps.

The fixed-k20 protocol below is retained as v1. The subsequent
[review-driven amendment](amendment_v2.md) separately specifies development k/prefix
selection, order sensitivity and row-interleaved timing. Its measurements must not
replace the v1 results or be described as the original test-blind protocol.

## Reproduce a comparison

Use a checkout with the ordinary Laya dependencies and `pyarrow` for the parquet files.
All inference is local. The first run downloads public model and dataset artifacts;
their immutable revisions are in `embedders.py`, `benchmark.py` and the report.

```bash
python research/benchmarks/tournament/compare_embeddings.py \
    --dataset banking77 --split dev --embedder bge --strategies shortlist \
    --device cpu --threads 4 --out dev-banking77-bge.json

python research/benchmarks/tournament/compare_embeddings.py \
    --dataset banking77 --split test --embedder bge \
    --device cpu --threads 4 --out test-banking77-bge.json
```

Without `--limit`, development means 600 seeded rows; test means the full official
test split. `--limit 100` explicitly selects a reproducible subset, whose row IDs and
content hashes appear in the report. Do not label that result as the full split.
Dataset names are `banking77`, `clinc150` and `massive`; see the existing
[benchmark description](README.md) for their question text, labels and splits.
Criteria order, state serialization and label humanization match that benchmark.

After preparing the cache, run offline with `HF_HUB_OFFLINE=1` and
`--local-files-only`. The flag alone restricts model loading; the environment variable
also prevents dataset network requests. Store output outside the checkout unless it
is an intentionally selected result artifact. Reports contain predictions and gold
label names, but not dataset text or model weights.

## What is compared

| strategy | candidate selection | decision budget / rounds |
|---|---|---|
| `single` | all labels | checkpoint defaults |
| `wide` | all labels | head 512, sequence 1024 |
| `builtin` | decision encoder, top 20 | one default decision |
| `tournament` | all labels | groups of at most 16, then winners |
| `top1` | dedicated embedding, top 1 | no decision |
| `shortlist` | dedicated embedding, top 20 | one default decision |
| `combined` | dedicated embedding, top 20 | groups of at most 16, then winners |
| `shortlist-wide` | dedicated embedding, top 20 | one decision, head 512, sequence 1024 |

`--k` and `--group-size` are recorded, not inferred from a filename. In particular,
top 20 followed by group size 16 means two groups of 10 and one final decision, using
the existing balanced contiguous grouping. It does not mean a 16/4 split. The ordering
is cosine rank order; this can affect which labels compete in the same group.

The combination supplies a tournament-backed predictor to `predict_shortlist` and
`evaluate_shortlist`. It never makes an extra ordinary prediction to collect metrics.
The returned tournament and shortlist metadata describe the same actual call chain.

For an application's already loaded `agent`, raw `state` and `questions`, the same
path can be called directly. It receives no gold labels or precomputed query vectors:

```python
from research.benchmarks.tournament.compare_embeddings import Predictor, predict_strategy
from research.benchmarks.tournament.embedders import RoleEmbedder, SPECS, TextEncoder

spec = SPECS["bge"]
embed = RoleEmbedder(TextEncoder(spec, device="cpu"), spec)
decision = Predictor(agent, "combined", group_size=16)
result = predict_strategy(decision, state, questions, embed, k=20)
```

Keep the encoder and its bounded cache alive across requests. Use the quality and
latency commands to decide whether this path is appropriate for the application's
label set before adopting it.

## Embedding contracts

| name | pinned model | pooling | dimensions | token cap | query / label prefix |
|---|---|---|---:|---:|---|
| `nomic` | `nomic-ai/modernbert-embed-base` | masked mean | 768 | 512 | `search_query: ` / `search_document: ` |
| `bge` | `BAAI/bge-small-en-v1.5` | CLS | 384 | 512 | neither |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | masked mean | 384 | 256 | neither |

These are fp32 `AutoModel` adapters with normalized vectors and `trust_remote_code=False`.
Pooling and prompts follow the official model cards:
[Nomic](https://huggingface.co/nomic-ai/modernbert-embed-base),
[BGE](https://huggingface.co/BAAI/bge-small-en-v1.5),
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
BGE's query instruction for short-query/long-passage retrieval is omitted for this
short-label task. Nomic's 512-token cap is an experiment setting, not its maximum
supported context. The report separately counts query and label truncations.

The query is the instruction followed by the serialized state. Prefixes are added
**before** the existing per-string cache: partial cache hits and deduplication must not
turn a document into a query. Each model owns a cache, reset between strategies and
timing passes. Label vectors are warmed outside the timed request; new query vectors
are computed inside it. Repeated query text within a pass can hit the bounded cache,
and the resulting hit/miss counts are reported.

The original #102 source URL was unavailable when this study started. A
[public checkout with matching figures](https://github.com/alex-shekhter/open-system-one/blob/3543ef3b350cf930e1a779eb141117b07c09a07c/laya_shortlist.py)
uses ModernBERT with both roles prefixed as queries, similarities centered using the
whole query batch, and precomputed embeddings excluded from decision timing.
Dylan subsequently confirmed the source relationship in the issue discussion and
restored [the original repository](https://github.com/zhlei07/open-system-one/tree/e2f50d0e6101dfa0f86c44d8060d2e4b120fe40f).
This comparison uses
independent online cosine retrieval and official splits; it is not a reproduction of
the original 2,500-row BANKING77 number or its latency claim.

## Development and held-out evaluation

Start with the first 200 of the fixed 600 development rows per dataset, using
`shortlist` accuracy averaged equally over the three datasets to select BGE or MiniLM.
Confirm the selected model and Nomic on all 600 rows, without reselecting from test
outcomes. Samples use prefixes of a seeded full permutation, so increasing `--limit`
preserves pilot membership. Keep Nomic as a source reference.
If the lightweight models differ by less than one percentage point, use measured
development p95 and process memory as an engineering tie-break, not a claim of
statistical equivalence. Save that selection and the run commands before scoring test
rows. A resource-limited subset must be fixed before its test outcomes are inspected.

```bash
python research/benchmarks/tournament/compare_embeddings.py \
    --mode recall --dataset banking77 --split dev --embedder bge --out recall-bge.json
```

The retrieval-only pass gives top-1 and recall at 16/20/32/48, without loading Laya.
The main comparison keeps k=20 and group size=16 fixed. Do not search test configurations
until the combination wins. Large search spaces, adaptive fallback, training and new
grouping algorithms are outside this experiment.

For the development-selected lightweight model, `--primary` on BANKING77 test uses
97.5% paired bootstrap intervals for `combined` versus `shortlist` and `tournament`.
This is a Bonferroni adjustment for those two preselected comparisons; bootstrap
coverage is approximate. Other runs use 95% exploratory intervals. Pairing checks
dataset revision, split, question order, checkpoint, row IDs, state hashes and gold
labels. Neither unpaired rows nor incomplete results can be silently compared.

## Interpret a result

Every example belongs to exactly one outcome:

1. `retrieval_miss`: retrieval removed the gold label.
2. `elimination_miss`: retrieval kept it, but a tournament round removed it.
3. `final_miss`: the final candidate set contained it and the final answer was wrong.
4. `correct`: the final answer was right.

The sum is the original row count. Conditional rates use their own denominators and
are `null` when undefined. Retrieval misses remain in total accuracy and calibration.
Confidence and ECE/Brier are the existing decision evaluator's outputs; cosine scores
and products of round probabilities are never presented as calibrated confidence.
`top1` reports accuracy and recall only.

For every actual decision call the quality report records its candidate labels,
budget, model usage, precision and token diagnostics. Diagnostics replay the same
core sequence builders after inference. They record state loss, surviving markers,
option token lengths, distinct spans and ordered input hashes. Distinct truncated
spans do not prove that all label meaning survived. A full-set strategy that cannot
fit its option markers is reported as `infeasible`, without an invented accuracy.
Unexpected errors fail the run and leave `status: failed`; completed arms stay visible.

## Measure costs separately

```bash
python research/benchmarks/tournament/compare_embeddings.py \
    --mode latency --dataset banking77 --split test --limit 100 --seed 1 \
    --embedder bge --strategies shortlist,combined,tournament --passes 3 \
    --device cpu --threads 4 --local-files-only --out latency-bge.json
```

The latency mode collects no quality trace. Its clock covers query embedding, ranking,
and every real decision round, with device synchronization before and after each
request. Three development warmups exclude all measured state hashes, and their IDs
are saved. Label cache warming and model loading are outside request latency. Run one
benchmark process at a time. The runner rotates strategy order between passes and
records the schedule; partial multi-pass results retain `status: running`.

Model acquisition/loading time is recorded separately and may include network time
unless the cache is already populated. Peak RSS describes the whole process lifetime,
including all resident models; it is not the incremental memory of one strategy.
Raw and effective checkpoint temperatures, including whether they were clamped by
the runtime, are saved. A clamped checkpoint's confidence must be treated as uncalibrated
in the affected option-count buckets; lower observed ECE alone does not repair that.
The current quality process loads Laya even when `top1` is among its arms, so use
retrieval-only mode to measure the embedding-only process. GPU allocation is recorded
when CUDA is used. No CPU run implies a GPU speedup or a production latency guarantee.

## Checks

```bash
python -m pytest research/benchmarks/tournament/test_compare_embeddings.py -q
python tests/test_shortlist.py
python -m pytest tests/test_evals_shortlist.py -q
```

The new suite also runs through `scripts/test_suites.py`. It checks composition,
passthrough, invalid answers, role-aware partial cache hits, timing boundaries, option
budgets, paired identity and failure accounting without downloading weights.

Dataset attribution follows the original providers: BANKING77 by Casanueva et al.
([PolyAI, CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)),
CLINC150 by Larson et al.
([dataset card, CC BY 3.0](https://huggingface.co/datasets/clinc/clinc_oos)), and MASSIVE by
FitzGerald et al.
([Amazon NOTICE, CC BY 4.0](https://github.com/alexa/massive/blob/main/NOTICE)).
The BANKING77 mirror's MIT metadata does not replace the original dataset attribution.
No dataset text or model weight files are included here.
