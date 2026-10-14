# Dedicated retrieval and tournament: recorded study

On these requests, adding a tournament after retrieval does not consistently improve the simpler shortlisted decision. The development-selected BANKING77 pipelines tie on test accuracy, while their costs and errors differ. This supports comparing retrieval, decision and elimination errors separately before adding another stage; it does not establish a universally best configuration or justify changing library defaults.

This report accompanies [issue #102](https://github.com/NandhaKishorM/laya/issues/102). The original [fixed-k20 protocol](dedicated_embeddings.md) remains intact. The [amendment](amendment_v2.md) was designed **after those test results were visible**: its k selection, order checks, timing and query diagnostic are exploratory, not a fresh test-blind confirmation.

## Evidence and comparison contract

[evidence/2026-10-13/](evidence/2026-10-13/manifest.json) contains 63 compact reports: nine original development screening reports, six development confirmations, six fixed test reports and 42 amendment reports. The final amendment attempt completed 26 jobs and reused 16 hash-verified complete artifacts. These counts describe artifacts, not independent experiments or additional test samples.

[model-selection.json](evidence/2026-10-13/model-selection.json), [selected-configurations.json](evidence/2026-10-13/selected-configurations.json) and [planned-samples.json](evidence/2026-10-13/planned-samples.json) preserve the recorded choices and samples. The k-selection record retains its original timestamp, 2026-10-12T15:49:46Z; portable tooling added for publication does not backdate a new selection.

All test-quality comparisons use the same 500 official test rows within each dataset, with matched dataset revision, question, checkpoint, row ID, state hash and gold label. They are subsets, not full-split scores. Accuracy is percent; deltas and confidence intervals are percentage points. Gained/lost means the candidate correct and baseline wrong, or the reverse. Paired bootstrap resamples the same sorted IDs 10,000 times, seed 0. Intervals are exploratory 95% unless marked 97.5%; repeated passes or permutations never add independent rows.

The decision checkpoint is `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. MiniLM is `sentence-transformers/all-MiniLM-L6-v2`; Nomic is `nomic-ai/modernbert-embed-base`. [measurement.json](evidence/2026-10-13/measurement.json) pins model/data-related contracts, revisions, prefixes and execution environment. Q/D means Nomic's `search_query: ` query prefix and `search_document: ` label prefix; Q/Q uses `search_query: ` for both. The original model screening is conditional on the formatted query described below.

## Fixed k20: does the extra stage help?

`shortlist` makes one decision on 20 retrieved labels; `combined` runs a group-size-16 tournament on the same retrieved labels. `top1` uses their same embedding ranking. These three arms use instructions plus the serialized `{"message": ...}` state as the embedding query, with the model prefix retained.

| Dataset / embedder | Formatted top1 | Shortlist | Combined | Combined minus shortlist | Gained / lost | Paired interval |
|---|---:|---:|---:|---:|---:|---|
| BANKING77 / MiniLM | 48.0 | 62.8 | 63.2 | +0.4 | 27 / 25 | [-2.8, 3.6], 97.5% |
| BANKING77 / Nomic Q/D | 43.2 | 62.0 | 61.2 | -0.8 | 21 / 25 | [-3.4, 1.8] |
| CLINC150 / MiniLM | 38.0 | 81.2 | 82.0 | +0.8 | 12 / 8 | [-1.0, 2.6] |
| CLINC150 / Nomic Q/D | 20.0 | 74.4 | 74.4 | 0.0 | 8 / 8 | [-1.6, 1.6] |
| MASSIVE en-US / MiniLM | 27.6 | 60.2 | 59.0 | -1.2 | 24 / 30 | [-4.0, 1.6] |
| MASSIVE en-US / Nomic Q/D | 28.2 | 53.4 | 57.4 | +4.0 | 41 / 21 | [1.0, 7.0] |

Nomic/MASSIVE is a positive exploratory counterexample to "tournament never helps"; its 57.4% still trails the same-row wider-budget single decision at 62.2%. Conversely, CLINC150's full-label tournament gets 91.0%, ahead of either retrieved pipeline. BANKING77's full-label tournament is 58.4%, versus 63.2% for MiniLM combined: 51 gained / 27 lost, +4.8 points, original 97.5% interval [0.8, 8.8]. The remedy ranking depends on the dataset and baseline.

The BANKING77 gain over **formatted** top1 is substantial for both fixed methods: MiniLM shortlist +14.8 [10.2, 19.2] (110/36), combined +15.2 [10.6, 19.6] (112/36); Nomic shortlist +18.8 [13.8, 23.6] (134/40), combined +18.0 [13.4, 22.6] (126/36). The bare-query diagnostic below changes the practical interpretation.

Checkpoint confidence has a separate limitation: for questions with 11 or more options, the recorded temperature near 0.10058 is clamped to 0.5 by the runtime. Confidence in those buckets is treated as uncalibrated; lower observed ECE is not evidence of a calibration advantage. See the [checkpoint and confidence caveat](dedicated_embeddings.md). The accuracy and paired correctness comparisons above do not make a confidence-calibration claim.

## Development-selected BANKING77 configurations

The complete 18-cell grid uses the first 200 frozen development rows: MiniLM and Nomic Q/D or Q/Q, each at k=5,8,12,20,32,48. Each method selects maximum correct, breaking ties by smaller k, then Q/D. The remaining 400 development rows check stability without reselection; they had prior v1 observations and are not an untouched holdout.

| Family / method | Selected variant, k | Dev200 correct | Dev400 correct | Test500 correct | Decision calls |
|---|---|---:|---:|---:|---:|
| MiniLM shortlist | MiniLM, 32 | 119 | 236 | 311 (62.2%) | 1 |
| MiniLM combined | MiniLM, 12 | 118 | 247 | 311 (62.2%) | 1 |
| Nomic shortlist | Q/D, 20 | 117 | 239 | 310 (62.0%) | 1 |
| Nomic combined | Q/D, 48 | 119 | 241 | 310 (62.0%) | 2 |

Both families' combined-minus-shortlist comparisons have 26 gained / 26 lost, delta 0.0, interval [-2.8, 2.8]. This is not proof of equivalence or non-inferiority. MiniLM k12 is explicitly `equivalent_to: "shortlist"` and `degenerate_single_pass: true`: it executes no tournament round. Because the selected k differs between methods, these are pipeline comparisons, not isolated effects of adding tournament.

Nomic illustrates the decision bottleneck. Its k20 shortlist has 57 retrieval misses and 133 final-decision misses. At k48, combined reduces retrieval misses to 15 but loses 145 gold labels during elimination and 30 at the final decision. Both finish with 190 errors. Retrieving more correct labels is useful evidence, but does not itself deliver better final choices.

## Query contract and order sensitivity

The separate `query-text-diagnostic/1` changes only the embedding query to the exact `state["message"]` string. Model/label configurations stay fixed; exact bare and prefixed query hashes are recorded. It loads no decision model and makes no bare-query latency claim.

| Dataset / embedder | Formatted top1 | Bare top1 | Bare recall@20 |
|---|---:|---:|---:|
| BANKING77 / MiniLM | 48.0 | 59.6 | 97.0 |
| BANKING77 / Nomic Q/D | 43.2 | 63.4 | 97.0 |
| CLINC150 / MiniLM | 38.0 | 70.4 | 96.4 |
| CLINC150 / Nomic Q/D | 20.0 | 68.8 | 97.6 |
| MASSIVE en-US / MiniLM | 27.6 | 51.6 | 92.4 |
| MASSIVE en-US / Nomic Q/D | 28.2 | 45.0 | 86.4 |

Against bare BANKING77 top1, selected MiniLM shortlist gains only +2.6 points (60/47; [-1.6, 6.6]); selected combined also +2.6 (58/45; [-1.4, 6.6]). Selected Nomic shortlist is -1.4 (54/61; [-5.6, 2.8]), combined -1.4 (60/67; [-5.8, 3.2]). These compare **different query pipelines**, not the causal effect of adding Laya. No bare-message shortlist/combined run or new model-family selection was performed. The large gains over formatted top1 must not be advertised as gains over a simple bare-message classifier.

The order supplement keeps MiniLM/k20 and candidate membership fixed, applying global label permutations with seeds 11 and 29. Natural/seed11/seed29 accuracies are 58.4/63.6/62.0 for full tournament, 62.8/63.6/62.0 for shortlist, and 63.2/61.2/64.0 for combined. Full tournament gains 51/25 and 43/25 over natural order, with deltas +5.2 [1.8, 8.6] and +3.6 [0.4, 6.8]. Yet combined-minus-shortlist changes sign: -2.4 [-5.4, 0.6] and +2.0 [-1.2, 5.4]. These are the same 500 rows, not 1,500 observations. Two fixed permutations show sensitivity; they neither choose a winning seed nor separate grouping from presentation/finalist-order effects.

## Request timing

`latency-v2` uses resident model weights and label vectors, cold queries, and end-to-end request timing including each request's retrieval and decision calls. Setup/loading is excluded. Arms rotate within each row/pass; each report has 50 unique seed-1 requests, three passes. All eight reports retain 8,400 raw observations. The measured environment is Windows CPU, float32, four intra-op/eight inter-op threads; Python 3.11.13 and library versions are recorded in `measurement.json`.

| Selected BANKING77 process | Shortlist p50 / p95 ms | Combined p50 / p95 ms | Full tournament p50 / p95 ms |
|---|---:|---:|---:|
| MiniLM: k32 vs k12 alias | 969.01 / 1082.49 | 535.90 / 658.55 | 3077.16 / 3892.31 |
| Nomic Q/D: k20 vs k48 | 818.21 / 980.14 | 2111.60 / 2631.71 | 3120.37 / 3924.20 |

Compare arms **within each process**. MiniLM's faster selected "combined" is a smaller single decision, not a faster tournament. Nomic's two-call path costs about 2.58 times its shortlist p50 without a measured accuracy gain here. These are descriptive quantiles over repeated requests, not 150 independent samples per arm.

Cross-process variation is retained: the identical BANKING77 full tournament has p50 4820.40 ms in fixed-MiniLM, 3610.89 in fixed-Nomic, 3077.16 in selected-MiniLM and 3120.37 in selected-Nomic. Do not pool those times, attribute them to the embedder, or silently discard an inconvenient process. This CPU study makes no MPS/CUDA replication claim.

Three extreme observations remain in the published samples and quantiles: CLINC150/Nomic Q/D, pass 0, `clinc150:test:990`, `shortlist-wide` 66764.1249 ms and `combined` 13370.9267 ms; CLINC150/MiniLM, pass 2, `clinc150:test:3648`, `top1` 1142.0342 ms. No contemporaneous Windows sleep/wake event evidence was found. Their specific cause is unknown; these are measurements on a shared desktop, and the anomalies are retained rather than attributed to a model or removed.

## Inspect and reproduce

Compact evidence retains original raw-report SHA256, identities, choices/golds, state hashes, candidates/stages, aliases and timing samples. It omits token-level call traces; it permits numerical recomputation, not a full tokenizer/trace audit. From the repository root, public evidence can be checked without model inference:

```bash
python research/benchmarks/tournament/report_tools.py verify research/benchmarks/tournament/evidence/2026-10-13
python research/benchmarks/tournament/report_tools.py summarize research/benchmarks/tournament/evidence/2026-10-13/latency-selected-banking77-nomic.json
python research/benchmarks/tournament/report_tools.py pair research/benchmarks/tournament/evidence/2026-10-13/selected-test500-nomic-qd-k48.json research/benchmarks/tournament/evidence/2026-10-13/selected-test500-nomic-qd-k20.json --candidate-arm combined --baseline-arm shortlist
```

Use `--confidence 0.975` for the preserved primary BANKING77 comparisons; differing query contracts require `--mixed-query`. The portable selector requires newly generated **full raw reports**, not compact evidence. Its strict identity/configuration/alias checks reject missing grid cells, overwritten selections and first-time selection after selected-test/timing evidence exists. Rechecking an existing timestamp is not proof of historical execution order; the original freeze record is retained.

The measurements used core base `68804629e8ccd9d616d48a40e87de9fabbeae069`, whose `laya` tree is `772f948710467beb921da4d14cebddb3340397d7`, and local measurement commit `ecb4bbbfb37921a592f869dc533af546c02c6a95`. Publication was rebased onto main `c73908d9` with core tree `35d167c376736ed02366475fbfaaef39c0fe7eee`; research source content is unchanged. **These numbers do not validate the rebased core.** A historical local commit need not be publicly reachable to reconstruct the measured code:

```bash
# PR_HEAD is the fetched commit containing this study, not a moving branch name.
git worktree add --detach ../laya-study-measured 68804629e8ccd9d616d48a40e87de9fabbeae069
git -C ../laya-study-measured restore --source=PR_HEAD -- research/benchmarks/tournament
cd ../laya-study-measured
git rev-parse HEAD:laya
```

Confirm the printed core tree matches the measured tree above. Verify the seven inference sources against `measurement.json`'s `source_lf_sha256` after CRLF-to-LF normalization; `source_sha256` separately preserves exact measured bytes. Prepare an environment with the recorded dependencies plus `pyarrow`, install this worktree's package, and cache the pinned models and datasets before offline runs. Do not install another checkout's Laya into this environment.

The following portable Python driver, run from that measured worktree's root, reconstructs the amendment's 42 serial jobs for the recorded selections. It creates a new output directory, validates source content, freezes selection after all 18 grid cells, then runs the emitted dev400/test500 jobs, order/query/fixed-timing supplements, and finally selected timing. A rerun that selects different configurations may produce a different number of grouped jobs; preserve its own selection rather than forcing the published outcome.

```python
import hashlib, json, os, subprocess, sys
from pathlib import Path
src = Path("research/benchmarks/tournament")
measurement = json.loads((src / "evidence/2026-10-13/measurement.json").read_text())
for name, digest in measurement["source_lf_sha256"].items():
    assert hashlib.sha256((src / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, name
out = Path("../study-run-new").resolve()
out.mkdir()  # Never reuse an existing results directory.
os.environ["HF_HUB_OFFLINE"] = "1"
common = ["--device", "cpu", "--threads", "4", "--local-files-only"]
def run(script, *args):
    subprocess.run([sys.executable, str(src / script), *map(str, args)], check=True)
def quality(name, *args):
    run("amended_comparison.py", "--mode", "quality", "--group-size", "16",
        "--diagnose", *common, *args, "--out", out / (name + ".json"))
for variant in ("minilm", "nomic-qd", "nomic-qq"):
    for k in (5, 8, 12, 20, 32, 48):
        quality(f"grid-dev200-{variant}-k{k}", "--dataset", "banking77",
                "--split", "dev", "--limit", 200, "--seed", 0, "--variant", variant,
                "--strategies", "shortlist,combined", "--k", k)
selection = out / "selected-configurations.json"
run("select_configurations.py", "--reports", out, "--out", selection)
plan = json.loads(subprocess.check_output([sys.executable, str(src / "select_configurations.py"),
    "--reports", str(out), "--selection", str(selection), "--commands"], text=True))
for name, arms in plan["arms_files"].items():
    with (out / name).open("x", encoding="utf-8") as handle:
        json.dump(arms, handle)
def selected_jobs(timing):
    for job in plan["commands"]:
        if job["report"].startswith("latency-") == timing:
            subprocess.run([sys.executable, *job["argv"][1:]], check=True)
selected_jobs(False)  # Frozen dev400 stability and test500; no reselection.
for seed in (11, 29):
    quality(f"order-test500-minilm-seed{seed}", "--dataset", "banking77",
            "--split", "test", "--limit", 500, "--variant", "minilm",
            "--strategies", "tournament,shortlist,combined", "--k", 20,
            "--permutation-seed", seed)
for dataset in ("banking77", "clinc150", "massive"):
    for variant in ("minilm", "nomic-qd"):
        run("query_text_diagnostic.py", "--dataset", dataset, "--variant", variant,
            "--split", "test", "--limit", 500, "--seed", 0, *common,
            "--out", out / f"query-bare-test500-{dataset}-{variant}.json")
for dataset in ("banking77", "clinc150", "massive"):
    for variant in ("minilm", "nomic-qd"):
        run("amended_comparison.py", "--mode", "latency", "--dataset", dataset,
            "--variant", variant, "--split", "test", "--limit", 50, "--seed", 1,
            "--k", 20, "--group-size", 16, "--passes", 3, "--diagnose", *common,
            "--strategies", "single,wide,builtin,tournament,top1,shortlist,combined,shortlist-wide",
            "--out", out / f"latency-fixed-{dataset}-{variant}.json")
selected_jobs(True)
```

`--commands` prints a JSON plan and arms-file definitions; it neither runs models nor rewrites the frozen choice. The driver materializes those arms files before invoking their commands. Keep all raw outputs. `report_tools.py compact RAW_REPORT --out NEW_FILE` creates a separate compact artifact; `summarize` and `pair` work with raw or compact reports. Earlier fixed/model-screening commands remain in [dedicated_embeddings.md](dedicated_embeddings.md); they are not newly invented amendment jobs.

## Attribution and limits

Dylan's [original experiment and protocol clarification](https://github.com/NandhaKishorM/laya/issues/102#issuecomment-6120484926), including its reported 54.3% to 60.8% and timing improvement, remain his historical results with a different split/query/scoring/timing contract. His [later bare-query/MPS comparison](https://github.com/NandhaKishorM/laya/issues/102#issuecomment-6121525060) motivated the bounded query diagnostic. Neither is relabeled as this study's replication; the earlier benchmark README and its MPS numbers remain separately attributed.

Dylan subsequently [compared our recorded `a9c4344` outputs with his existing MPS/CPU files](https://github.com/NandhaKishorM/laya/issues/102#issuecomment-6158752377). Rechecking his [pinned per-row files and manifest](https://github.com/zhlei07/open-system-one/tree/4cb01b19403f21745ee81508cdd062874878cec9/out/laya102) against our evidence confirms 13,999 matching choices in 14,000 comparisons: 28 arm comparisons reuse 500 BANKING77 and 500 MASSIVE rows, **1,000 unique items**, not 14,000 independent samples. The sole choice difference is full tournament on `banking77:test:1345`: his MPS run chooses `top up failed`, ours `pending top up`; both miss `topping up by card`. All four formatted top-20 candidate sets match on every row. Candidate order differs once: Nomic Q/D on `banking77:test:693` swaps ranks 15 and 16, without changing the recorded choices. CLINC150 is excluded from this cross-check; Dylan reports only 388 overlapping rows. This checks shared recorded outputs; neither comparison reran this runner or validated its timings. His additional bare-message shortlist/combined results remain external MPS evidence, separate from our retrieval-only diagnostic and frozen matrix.

This contribution supplies a reproducible comparison and evidence about where errors arise. It does not establish cross-device speed, general accuracy improvement, the best bare-query model, maintainer approval, or a new default. The numerical review recomputed recorded predictions and verified identity/hash consistency; it was not an independent model rerun. The next useful experiment would need its own untouched data and an explicit query contract, not another selection based on these 500 test rows.
