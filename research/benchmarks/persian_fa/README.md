# Persian support-triage decisions

[Original project](https://github.com/alipyth/laya-persian-benchmark) · [Issue #703](https://github.com/NandhaKishorM/laya/issues/703)

64 frozen synthetic Persian support messages, eight families and four balanced labels:
`billing`, `technical`, `cancel`, `other`. Two prompt modes ask the same question with Persian
or English instructions. This diagnostic covers formal and colloquial language, Finglish,
Arabic/Persian character variants and ZWNJ, code mixing, negation, sarcasm/taarof and digits.
Each family contains two messages per label. Trap tags identify particular failure mechanisms.

**The messages and labels were drafted with AI assistance and are synthetic, not customer
data.** The source manifest attributes their review to Ali Jahani. This is a small regression
diagnostic with one attributed reviewer, not an independently annotated blind test or a model
ranking. It measures routing to a department; it never closes an account or takes any action.

## Verify the archive offline

From the Laya repository root, with Python 3.10+:

```bash
python research/benchmarks/persian_fa/audit.py
python -m unittest discover -s research/benchmarks/persian_fa/tests -v
```

Both commands require only the standard library. They load no checkpoint and need no network
or API key. The default audit verifies the original source hashes, all 1,152 answer records
and every field of the three recomputed JSON summaries. CI runs these same commands.
Frozen artifacts keep LF line endings on Windows as well as Linux.

## Historical results

These are the original author's **laya 0.3.21 CPU results**, recorded on 2026-09-28 UTC,
using Python 3.10.11, PyTorch 2.7.1+cpu and Transformers 4.51.3 on Windows. They are not
scores for the current Laya source. The recorded requested checkpoint revision is
`convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.

| First-repeat result | Multilingual | Multilingual + normalization | English |
|---|---:|---:|---:|
| Persian instructions: correct | 36/64 | 37/64 | 21/64 |
| Persian instructions: false `cancel` | 4 | 3 | 38 |
| English instructions: correct | 41/64 | 40/64 | 29/64 |
| English instructions: false `cancel` | 2 | 3 | 16 |

Each configuration contains 64 cases × two prompt modes × three repeats (384 records).
Quality uses the preselected first repeat; all three repeats produced the same decisions in
the archive. Failures remain in the denominator as wrong answers; the archive has none.
Do not combine the two instruction languages into a single score.

The archive keeps the author's answer projections: choice, all four probabilities, confidence,
latency and error. It did not record the complete SDK response, a resolved checkpoint revision
or weight digest. Those missing fields have not been reconstructed or invented. `SOURCE.json`
pins the original Git artifacts; it proves what was imported, not what hardware executed them.

## Run against a checkpoint

Install Laya using the repository's normal instructions. This runner imports the current
checkout. Start with a short CPU run, using a new output directory:

```bash
python research/benchmarks/persian_fa/run.py --checkpoint multilingual \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --device cpu \
  --limit 2 --repeats 1 --modes choice_fa --output /tmp/persian-smoke
python research/benchmarks/persian_fa/audit.py --run-dir /tmp/persian-smoke --write
```

Named checkpoints (`multilingual`, `english`, `typed-decisions`) and other Hub repository IDs
may download weights. `--revision` takes a full Hub commit SHA. A local exported checkpoint
directory is also accepted; omit `--revision` for local files, whose weights and configuration
are hashed instead. The runtime records the requested and resolved revision separately.
Use `--device cuda` or `mps` when available. No hosted inference service is used.

The full three-configuration protocol is:

```bash
python research/benchmarks/persian_fa/run.py --checkpoint multilingual \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --device cpu --output /tmp/fa-multi
python research/benchmarks/persian_fa/run.py --checkpoint multilingual --normalize \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --device cpu --output /tmp/fa-norm
python research/benchmarks/persian_fa/run.py --checkpoint english \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --device cpu --output /tmp/fa-english
python research/benchmarks/persian_fa/audit.py --run-dir /tmp/fa-multi --write
python research/benchmarks/persian_fa/audit.py --run-dir /tmp/fa-norm --write
python research/benchmarks/persian_fa/audit.py --run-dir /tmp/fa-english --write
python research/benchmarks/persian_fa/compare.py /tmp/fa-multi /tmp/fa-norm /tmp/fa-english --mode choice_fa
python research/benchmarks/persian_fa/compare.py /tmp/fa-multi /tmp/fa-norm /tmp/fa-english --mode choice_en
```

`--normalize` uses the frozen character/digit/ZWNJ transformations. It is an explicit ablation,
not preprocessing silently applied to every run. Labels, rationale, family and trap tags never
enter a model request. One warm-up per mode is excluded. Requests are serial and never retried.
The scripts refuse to overwrite existing runs or write into the historical archive.

Fresh format-2 runs save the exact request and complete SDK response alongside the answer
projection, source hashes, package versions and device. The audit checks that the recorded
request equals the frozen input and that the scored fields agree with the raw response.
Incomplete runs fail the audit. Explicit failed requests remain auditable and count as wrong;
the runner exits nonzero when a request fails and retains its metadata and available records.

`--limit`, fewer/more repeats or a subset of modes mark a run as a partial protocol. The
`--backend keyword` adapter exists only for model-free pipeline tests: it was written with
these cases in view, is explicitly marked as not a model result, and comparison tables refuse it.

## Metrics and interpretation

The original `metrics.py` is retained unchanged. Reports include accuracy, macro-F1,
per-family/per-label/per-trap counts, confusion matrix, false/missed `cancel`, mean answer
confidence, ECE, latency and repeat agreement. A false `cancel` means a non-cancellation
message was routed to cancellation. A missed `cancel` includes a failed request on a true
cancellation case. Confidence and ECE use successful answers only; quality and F1 retain
failures. ECE uses ten equal-width bins over the answer's calibrated probability.
For current checkpoints with binning calibration, `answer_confidence` can differ from the
winning option probability. Fresh records preserve the SDK value and the checkpoint's binning
map; the audit verifies that the scored confidence agrees with that raw response.

Latency summaries use successful requests from all repeats. The source was measured on a
shared desktop, with large timing variation; this archive cannot establish a speed ranking or
a normalization speedup. Repeat agreement counts identical outcomes, including repeated
failure, and must be read together with the failure count.

The five-case instruction-language gap and one-case normalization differences are observations
on this fixture set. They do not establish population-level effects. Do not tune prompts,
temperatures or checkpoints on these public cases and then describe them as held-out evidence.
Persian fine-tuning and broader real-traffic evaluation remain separate work.

## Provenance and maintenance

Adapted from Ali Jahani's (`alipyth`) [source snapshot](https://github.com/alipyth/laya-persian-benchmark/tree/e307e54029da7d5c9da72b54b4211378490cf34c).
The data, prompts, normalization, metric definitions, license and `results/v1/` files retain
their original bytes. `SOURCE.json` lists their SHA-256 values. The runner, audit and tests
were adapted for repository-root execution, complete fresh-run records and offline CI.
The retained [MIT license](LICENSE) applies to the imported material; it does not change
Laya's Apache-2.0 license. Research files are not imported by the installed package.

Keep historical runs intact. Any deliberate fixture/prompt change needs a new dataset/results
version and explicit review; updating hashes to make an unexplained audit failure pass would
break the evidence chain. The companion project retains additional orchestration and Persian
walkthroughs; this directory contains the runnable diagnostic and its audit.
