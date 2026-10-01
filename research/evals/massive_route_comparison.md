# Swedish routing: MASSIVE full-test comparison

This report compares the `laya.lang` detector on all 51 locale files in the pinned
`mteb/amazon_massive_intent` test split (dataset revision
`940fd47a81eaa7f2cc7b129674d945d618ac38c2`). Each locale contributes 2,974 examples,
for 151,674 examples total. The comparison is model-free: it evaluates the same text
against the baseline `main` detector and the proposed detector without loading either
checkpoint.

## Method

The expected route is `english` for the `en` locale and `multilingual` for every other
locale. The script calls the repository's real `Router._route` for both detectors, so
all other routing rules stay the same, including the configured `english` default when
Latin text is undecided. Route accuracy measures only that checkpoint choice. Exact
named-language accuracy is reported separately: a detector can safely choose
`multilingual` while leaving the specific language undecided. Results for every locale,
including guess counts, are in
[`../results/massive_route_comparison_swedish.json`](../results/massive_route_comparison_swedish.json).

Reproduce from the repository root, passing baseline and candidate commits:

```sh
python research/evals/massive_route_comparison.py \
  --before-ref 6d942c92081fbc139e736bbd9ac0023223c29b7f \
  --after-ref ee499bb095587c8af17f2747dacd89a4c76c1ff1 \
  --out research/results/massive_route_comparison_swedish.json
```

The report records the source Git revisions and pinned dataset revision. The script
downloads the split from Hugging Face on first run and needs `huggingface_hub` installed.

## Results

| Measure | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Overall checkpoint route accuracy | 76.831% | 77.428% | +0.597 pp |
| Exact named-language accuracy | 8.243% | 8.837% | +0.594 pp |
| Swedish checkpoint route accuracy | 78.379% | 86.954% | +8.574 pp |
| Swedish named-language accuracy | 0.000% | 30.296% | +30.296 pp |
| Danish checkpoint route accuracy | 44.586% | 53.430% | +8.843 pp |
| Norwegian Bokmål route accuracy | 43.073% | 49.630% | +6.557 pp |

No locale's checkpoint-routing accuracy decreased. Both detectors are loaded directly
from committed Git objects, with commit IDs and SHA-256 source hashes recorded in the
artifact. The local router implementation is also fingerprinted. This prevents modified
worktree files from being mislabeled as a baseline commit.

## Interpretation and limits

MASSIVE contains intent utterances, not customer-support transcripts. Routing accuracy
measures checkpoint selection, not model answer quality. Exact language identification
is reported separately; the detector does not name all 51 languages.

Four Norwegian Bokmål utterances and five Icelandic utterances are labeled Swedish by
the candidate. They still select the expected multilingual checkpoint. Shared Nordic
words therefore remain a limitation for downstream uses of the language label.

The benchmark was used during development and is not an untouched holdout. Real Swedish
support examples and end-to-end model evaluation are still needed for support-quality
claims.
