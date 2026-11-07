# Results: multilingual + normalize

## choice_fa

- accuracy **37/64 (57.8%)**, macro-F1 0.513, failures 0
- false `cancel` (would wrongly close an account): **3**; missed `cancel`: 8
- mean answer confidence 83.5%, ECE 0.257
- latency p50 695.8 ms, repeat agreement 100.0%

| family | correct |
|---|---|
| formal | 5/8 |
| colloquial | 5/8 |
| finglish | 4/8 |
| orthography | 5/8 |
| code_mixed | 5/8 |
| negation | 3/8 |
| sarcasm_taarof | 5/8 |
| digits | 5/8 |

| gold \ predicted | billing | technical | cancel | other | failed |
|---|---|---|---|---|---|
| billing | 13 | 2 | 1 | 0 | 0 |
| technical | 1 | 15 | 0 | 0 | 0 |
| cancel | 0 | 8 | 8 | 0 | 0 |
| other | 7 | 6 | 2 | 1 | 0 |

## choice_en

- accuracy **40/64 (62.5%)**, macro-F1 0.619, failures 0
- false `cancel` (would wrongly close an account): **3**; missed `cancel`: 5
- mean answer confidence 82.0%, ECE 0.202
- latency p50 543.4 ms, repeat agreement 100.0%

| family | correct |
|---|---|
| formal | 5/8 |
| colloquial | 6/8 |
| finglish | 4/8 |
| orthography | 5/8 |
| code_mixed | 6/8 |
| negation | 3/8 |
| sarcasm_taarof | 6/8 |
| digits | 5/8 |

| gold \ predicted | billing | technical | cancel | other | failed |
|---|---|---|---|---|---|
| billing | 9 | 4 | 2 | 1 | 0 |
| technical | 1 | 14 | 1 | 0 | 0 |
| cancel | 0 | 5 | 11 | 0 | 0 |
| other | 4 | 6 | 0 | 6 | 0 |

Quality is scored on the first repeat (preselected, not the best). Failures count as wrong.
