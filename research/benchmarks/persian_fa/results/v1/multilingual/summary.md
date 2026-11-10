# Results: multilingual

## choice_fa

- accuracy **36/64 (56.2%)**, macro-F1 0.513, failures 0
- false `cancel` (would wrongly close an account): **4**; missed `cancel`: 9
- mean answer confidence 83.5%, ECE 0.273
- latency p50 218.4 ms, repeat agreement 100.0%

| family | correct |
|---|---|
| formal | 6/8 |
| colloquial | 5/8 |
| finglish | 4/8 |
| orthography | 4/8 |
| code_mixed | 5/8 |
| negation | 3/8 |
| sarcasm_taarof | 5/8 |
| digits | 4/8 |

| gold \ predicted | billing | technical | cancel | other | failed |
|---|---|---|---|---|---|
| billing | 13 | 2 | 1 | 0 | 0 |
| technical | 1 | 14 | 1 | 0 | 0 |
| cancel | 1 | 8 | 7 | 0 | 0 |
| other | 6 | 6 | 2 | 2 | 0 |

## choice_en

- accuracy **41/64 (64.1%)**, macro-F1 0.633, failures 0
- false `cancel` (would wrongly close an account): **2**; missed `cancel`: 6
- mean answer confidence 81.8%, ECE 0.250
- latency p50 170.2 ms, repeat agreement 100.0%

| family | correct |
|---|---|
| formal | 5/8 |
| colloquial | 7/8 |
| finglish | 4/8 |
| orthography | 5/8 |
| code_mixed | 6/8 |
| negation | 3/8 |
| sarcasm_taarof | 6/8 |
| digits | 5/8 |

| gold \ predicted | billing | technical | cancel | other | failed |
|---|---|---|---|---|---|
| billing | 10 | 3 | 2 | 1 | 0 |
| technical | 1 | 15 | 0 | 0 | 0 |
| cancel | 0 | 6 | 10 | 0 | 0 |
| other | 4 | 6 | 0 | 6 | 0 |

Quality is scored on the first repeat (preselected, not the best). Failures count as wrong.
