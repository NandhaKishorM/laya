# laya × Quilt — the decision ledger

Every laya decision can be booked as an **attested observation**: content
(state + questions + answers), a witness-stake (the operator and the exact
checkpoint that served the decision), a time-context (a monotonic tick —
partial order, never false finality), and a content-binding (a hash chain
over canonical rows, fnv1a-32 — the same chain algorithm the reference
quilt kernel uses, so any kernel port can verify rows written here).

`laya/quilt.py` is **stdlib-only**. Torch and checkpoint weights load
lazily through `QuiltLayaBridge`, never at import — the ledger verifies in
environments where the decision engine cannot run at all.

## The five opcodes, as practiced here

| opcode | ledger operation |
|---|---|
| **BIND** | `decide(state, questions, decide_fn)` — book one decision under one engine identity |
| **LINK** | `ensemble(group_id, state, questions, engines)` — the same question set under ≥2 engine identities, linked into one disagreement group |
| **EFFECT** | `resolve(group_id, member_hash, rationale)` — a staked meta-attestation choosing one ensemble member; derived state, recomputable from rows |
| **VIEW** | `view("json" \| "canon")` — a booked, receipted projection; the chain head is captured before the VIEW row so a view describes the ledger as it was |
| **TICK** | the monotonic `tick` on every row; the hash chain is the partial order |
| **REFUSED** | a named refusal row — malformed engine results, missing resolve targets, solo ensembles. Refusals are visible, never silent drops |

## Doctrine

- **Disagreement is preserved, not deleted.** Ensembles keep every rival
  answer verbatim; resolution cites the rivals it passed over and deletes
  nothing. An erased rival destroys the audit trail that makes the winner
  worth anything.
- **Refusal visibility.** Every failure mode books a REFUSED row naming
  the reason. A ledger that can refuse silently can censor silently.
- **Booked history is immutable by construction.** Answers are deep-copied
  through the canonical form at booking time; a later mutation of any
  object the engine handed back cannot rewrite a booked row.
- **Integrity, not security.** fnv1a-32 catches accidental mutation and
  pins tamper at its own row; payloads also carry sha-256 bindings for
  state and questions. Swap the chain function by editing one constant —
  the checker plurality instinct: never one implementation, one algorithm,
  or one steward in sole custody.
- **A ledger is one actor's stake stream.** Concurrency lives between
  ledgers, not within one.

## Use with a real checkpoint

```python
from laya import QuiltLedger, QuiltLayaBridge

ledger = QuiltLedger(actor="ops-crab", engine="english")
bridge = QuiltLayaBridge(route="english")   # torch loads on first decision
ledger.decide(email_state(raw), triage_questions(), bridge.decide_fn())

ok, bad_row = ledger.verify()
canon = ledger.view("canon")   # CANON-stub-shaped digest for pipelines
```

`QuiltLayaBridge` records the serving checkpoint in the row's
`engine_model`, so routing reality — not routing intention — is what the
ledger attests to.

## Verification across substrates

A row's `row_hash` is `fnv1a32(canonical(row_without_row_hash))` where
`canonical` is `json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`. The chain input of the first row is `"0"*8`.
Any substrate that can JSON-encode and run fnv1a can verify a Python
ledger — that is the point.

Run it yourself:

```bash
python3 tests/test_quilt.py            # 54 checks, stdlib only
python3 examples/quilt_decisions.py --check
```
