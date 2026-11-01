# Results: english

## choice_fa

- accuracy **21/64 (32.8%)**, macro-F1 0.261, failures 0
- false `cancel` (would wrongly close an account): **38**; missed `cancel`: 1
- mean answer confidence 41.1%, ECE 0.105
- latency p50 602.9 ms, repeat agreement 100.0%

| family | correct |
|---|---|
| formal | 3/8 |
| colloquial | 2/8 |
| finglish | 4/8 |
| orthography | 2/8 |
| code_mixed | 2/8 |
| negation | 2/8 |
| sarcasm_taarof | 3/8 |
| digits | 3/8 |

| gold \ predicted | billing | technical | cancel | other | failed |
|---|---|---|---|---|---|
| billing | 3 | 1 | 12 | 0 | 0 |
| technical | 1 | 1 | 13 | 1 | 0 |
| cancel | 0 | 1 | 15 | 0 | 0 |
| other | 0 | 1 | 13 | 2 | 0 |

## choice_en

- accuracy **29/64 (45.3%)**, macro-F1 0.396, failures 0
- false `cancel` (would wrongly close an account): **16**; missed `cancel`: 3
- mean answer confidence 47.1%, ECE 0.084
- latency p50 1467.2 ms, repeat agreement 100.0%

| family | correct |
|---|---|
| formal | 4/8 |
| colloquial | 3/8 |
| finglish | 5/8 |
| orthography | 4/8 |
| code_mixed | 5/8 |
| negation | 2/8 |
| sarcasm_taarof | 2/8 |
| digits | 4/8 |

| gold \ predicted | billing | technical | cancel | other | failed |
|---|---|---|---|---|---|
| billing | 11 | 2 | 2 | 1 | 0 |
| technical | 5 | 3 | 8 | 0 | 0 |
| cancel | 2 | 1 | 13 | 0 | 0 |
| other | 7 | 1 | 6 | 2 | 0 |

Quality is scored on the first repeat (preselected, not the best). Failures count as wrong.
