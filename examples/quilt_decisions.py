"""Quilt-ledger decision session — deterministic, no model weights.

Two fake checkpoints disagree on a triage question; the ledger books both
answers, a resolution attests to the choice, and the whole chain verifies.
Run with --check to assert chain integrity and exit non-zero on tamper.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.quilt import QuiltLedger  # noqa: E402

QUESTIONS = {
    "triage": {"type": "choice", "instructions": "Route this customer message.", "criteria": {"refund": None, "escalate": None}},
    "abuse": {"type": "noul", "instructions": "Is this abuse?"},
}

BASE = {
    "triage": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.62, "escalate": 0.38}, "confidence": 0.48},
    "abuse": {"type": "noul", "noul": 0.03, "confidence": 0.97},
}


def checkpoint_a(state, questions):
    return {"model": "demo-checkpoint-a", "answers": BASE, "usage": {"input_tokens": 33, "output_tokens": 0}}


def checkpoint_b(state, questions):
    answers = dict(BASE)
    answers["triage"] = dict(BASE["triage"], choice="escalate", probabilities={"refund": 0.41, "escalate": 0.59})
    return {"model": "demo-checkpoint-b", "answers": answers, "usage": {"input_tokens": 33, "output_tokens": 0}}


def main():
    ledger = QuiltLedger(actor="example-operator", engine="demo-checkpoint-a")
    state = "Customer reports a double charge and wants money back today."

    # A normal single-engine decision.
    ledger.decide(state, QUESTIONS, checkpoint_a)

    # The same question set under two checkpoints: both answers preserved.
    group = ledger.ensemble("triage-dispute-1", state, QUESTIONS,
                            {"demo-checkpoint-a": checkpoint_a, "demo-checkpoint-b": checkpoint_b})

    # A resolution: a staked meta-attestation citing the rival it passed over.
    ledger.resolve("triage-dispute-1", group["rows"][1],
                   "checkpoint-b reports higher calibrated confidence; escalate wins under dispute")

    ok, bad = ledger.verify()
    print("rows booked :", len(ledger.rows))
    for row in ledger.rows:
        print("  t%-2d %-7s %s" % (row["tick"], row["op"], row["row_hash"]))
    print("chain verify:", ok, "" if ok else "(first bad row: %s)" % bad)

    if "--check" in sys.argv:
        if not ok:
            print("CHECK FAILED: tamper pinned at row %s" % bad)
            return 1
        print("CHECK OK: %d-row chain verifies" % len(ledger.rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
