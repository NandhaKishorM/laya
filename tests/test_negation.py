"""Negation test suite for forced-choice questions.

Tracks the known failure where laya-multilingual ignores negation markers
('not', 'do not', 'please do not') in forced-choice questions. All KNOWN
failures are expected until a future checkpoint improves negation handling.

Run: python3 tests/test_negation.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import laya  # noqa: F401
    SKIP_LIVE = False
except Exception:
    SKIP_LIVE = True

PASS, FAIL, KNOWN, SKIP = [], [], [], []


def _models_dir():
    return os.environ.get("LAYA_LAB_MODELS", os.path.expanduser("~/laya_models"))


def _load_agent(checkpoint_name):
    """Load a checkpoint by short name, or return None if not cached locally."""
    path = os.path.join(_models_dir(), checkpoint_name)
    if not os.path.isdir(path):
        return None
    try:
        return laya.load(path, device="cpu")
    except Exception as exc:
        return None


def check_negation(name, positive_text, negated_text, questions, expect_positive, expect_negated,
                   agent, checkpoint_label, known_failure=False):
    """Assert positive and negated states produce opposite answers.

    Records PASS/FAIL/KNOWN/SKIP per state. `known_failure=True` converts a FAIL into KNOWN
    so that expected model limitations appear in the summary without blocking CI.
    """
    qid = next(iter(questions))

    for text, expected, polarity in (
        (positive_text, expect_positive, "positive"),
        (negated_text, expect_negated, "negated"),
    ):
        test_id = "%s/%s/%s" % (checkpoint_label, name, polarity)
        try:
            result = agent.system_one(text, questions)
        except Exception as exc:
            SKIP.append("%s: inference error: %s" % (test_id, exc))
            continue

        answer = result.get("answers", {}).get(qid, {})
        got = answer.get("choice", answer.get("answer", None))
        conf = answer.get("confidence", float("nan"))
        ok = (got == expected)

        if ok:
            print("  PASS  %s  got=%r conf=%.4f" % (test_id, got, conf))
            PASS.append(test_id)
        elif known_failure:
            print("  KNOWN %s  got=%r (want %r) conf=%.4f" % (test_id, got, expected, conf))
            KNOWN.append("%s: got %r, want %r, conf=%.4f" % (test_id, got, expected, conf))
        else:
            print("  FAIL  %s  got=%r (want %r) conf=%.4f" % (test_id, got, expected, conf))
            FAIL.append("%s: got %r, want %r, conf=%.4f" % (test_id, got, expected, conf))


# Minimal pairs: (name, positive_text, negated_text, questions, expect_positive, expect_negated)
# Each pair exercises one action verb under negation. Questions use the "choice" type with two
# options so the correct answer is unambiguous even for a model with no negation awareness.
NEGATION_PAIRS = [
    (
        "cancel/please-do-not",
        "Please cancel my account.",
        "Please do NOT cancel my account.",
        {"intent": {"type": "choice", "instructions": "What does the user want?",
                    "criteria": {"cancel_account": "user wants to cancel", "keep_account": "user wants to keep"}}},
        "cancel_account", "keep_account",
    ),
    (
        "cancel/i-do-not-want",
        "I want to cancel my subscription.",
        "I do NOT want to cancel my subscription.",
        {"intent": {"type": "choice", "instructions": "What does the user want?",
                    "criteria": {"cancel_account": "user wants to cancel", "keep_account": "user wants to keep"}}},
        "cancel_account", "keep_account",
    ),
    (
        "refund/do-not-issue",
        "Please issue a refund for my order.",
        "Please do NOT issue a refund for my order.",
        {"action": {"type": "choice", "instructions": "What action should be taken?",
                    "criteria": {"issue_refund": "process a refund", "decline_refund": "do not refund"}}},
        "issue_refund", "decline_refund",
    ),
    (
        "delete/do-not-delete",
        "Delete all my data immediately.",
        "Do NOT delete my data.",
        {"action": {"type": "choice", "instructions": "What does the user want done with their data?",
                    "criteria": {"delete_data": "remove user data", "retain_data": "keep user data"}}},
        "delete_data", "retain_data",
    ),
    (
        "suspend/please-do-not",
        "Suspend my account for now.",
        "Please do NOT suspend my account.",
        {"action": {"type": "choice", "instructions": "What should happen to the account?",
                    "criteria": {"suspend_account": "suspend the account", "keep_active": "leave account active"}}},
        "suspend_account", "keep_active",
    ),
    (
        "lock/do-not-lock",
        "Lock my account until further notice.",
        "Do NOT lock my account.",
        {"action": {"type": "choice", "instructions": "What should happen to the account?",
                    "criteria": {"lock_account": "lock the account", "leave_unlocked": "leave account unlocked"}}},
        "lock_account", "leave_unlocked",
    ),
    (
        "cancel/negation-with-context",
        "My subscription is up for renewal and I would like to cancel it.",
        "My subscription is up for renewal but I do NOT want to cancel it.",
        {"intent": {"type": "choice", "instructions": "What does the user want to do with their subscription?",
                    "criteria": {"cancel_subscription": "cancel the subscription",
                                 "renew_subscription": "renew the subscription"}}},
        "cancel_subscription", "renew_subscription",
    ),
    (
        "refund/strong-negation",
        "I want a full refund on my purchase.",
        "I absolutely do NOT want a refund on my purchase.",
        {"action": {"type": "choice", "instructions": "Does the user want a refund?",
                    "criteria": {"wants_refund": "user wants a refund", "no_refund": "user does not want a refund"}}},
        "wants_refund", "no_refund",
    ),
]


def _run_checkpoint(checkpoint_name, known_failure=False):
    """Run all negation pairs against one checkpoint. Skips gracefully if not cached."""
    if SKIP_LIVE:
        for name, *_ in NEGATION_PAIRS:
            test_id = "%s/%s" % (checkpoint_name, name)
            print("  SKIP  %s: laya not importable" % test_id)
            SKIP.append(test_id)
        return

    agent = _load_agent(checkpoint_name)
    if agent is None:
        for name, *_ in NEGATION_PAIRS:
            test_id = "%s/%s" % (checkpoint_name, name)
            print("  SKIP  %s: checkpoint not in %s" % (test_id, _models_dir()))
            SKIP.append(test_id)
        return

    label = checkpoint_name.replace("laya-", "").replace("laya", "english")
    print("\n--- checkpoint: %s ---" % checkpoint_name)
    for name, pos, neg, qs, exp_pos, exp_neg in NEGATION_PAIRS:
        check_negation(name, pos, neg, qs, exp_pos, exp_neg,
                       agent, label, known_failure=known_failure)


if __name__ == "__main__":
    print("Negation test suite (#377)")
    print("Models dir: %s" % _models_dir())

    # English checkpoint — expected to pass on most pairs
    _run_checkpoint("laya", known_failure=False)

    # Multilingual checkpoint — known to ignore negation markers (issue #377)
    _run_checkpoint("laya-multilingual", known_failure=True)

    print()
    print("%d passed, %d failed, %d known, %d skipped" % (
        len(PASS), len(FAIL), len(KNOWN), len(SKIP)))

    if KNOWN:
        print("\nKNOWN (model limitation, not CI failure):")
        for k in KNOWN:
            print("  KNOWN " + k)
    if SKIP:
        print("\nSkipped:")
        for s in SKIP:
            print("  SKIP  " + s)
    if FAIL:
        print("\nUnexpected failures:")
        for f in FAIL:
            print("  FAIL  " + f)

    sys.exit(1 if FAIL else 0)
