"""Interpret actual candidate sets and token budgets without changing prediction."""
import hashlib
import json

import numpy as np


def fingerprint(value):
    # Criteria insertion order is part of a question, including in a dict.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def outcome(gold, choice, original, shortlist=None, finalists=None):
    """The first irreversible loss of the gold label, or a correct final answer."""
    kept = list(original if shortlist is None else shortlist)
    final = list(kept if finalists is None else finalists)
    if (gold not in original or len(set(kept)) != len(kept) or len(set(final)) != len(final)
            or not kept or not final or not set(kept) <= set(original)
            or not set(final) <= set(kept) or choice not in final):
        raise ValueError("inconsistent candidate sets or answer")
    if gold not in kept:
        return "retrieval_miss"
    if gold not in final:
        return "elimination_miss"
    return "correct" if choice == gold else "final_miss"


def stage_metrics(cases):
    names = ("correct", "retrieval_miss", "elimination_miss", "final_miss")
    counts = {key: sum(c["outcome"] == key for c in cases) for key in names}
    n = len(cases)
    if not n or sum(counts.values()) != n:
        raise ValueError("stage counts do not cover the planned cases")
    recalled = n - counts["retrieval_miss"]
    survived = recalled - counts["elimination_miss"]
    return {"counts": counts, "rows": n, "accuracy": counts["correct"] / n,
            "recall_at_k": recalled / n,
            "survival_on_recalled": survived / recalled if recalled else None,
            "accuracy_on_finalists": counts["correct"] / survived if survived else None}


def token_diagnostics(tok, state, question, max_len, head_max_len):
    """Replay the documented sizing helpers, outside the measured prediction."""
    from laya.common import build_head, build_sequence

    if question.get("type") != "choice" or not isinstance(question["instructions"], str):
        raise ValueError("token diagnosis supports the benchmark's string-instruction choices")
    criteria = question["criteria"]
    criteria = criteria if isinstance(criteria, dict) else dict.fromkeys(criteria)
    q = {"t": "choice", "ins": question["instructions"], "crit": criteria}
    order = question.get("option_order")
    head, positions, _ = build_head(tok, q, head_max_len, option_order=order)
    seq, markers, stats, trunc = build_sequence(
        tok, state, q, max_len, head_max_len, option_order=order,
        truncate_left=isinstance(state, list), return_stats=True, return_truncation_stats=True)
    ends = positions[1:] + [len(head) - 1]
    head_spans = [head[a + 1:b] for a, b in zip(positions, ends)]
    spans = [seq[a + 1:min(b, len(seq))] for a, b in zip(positions, ends)]
    labels = list(criteria)
    labels = [labels[i] for i in order] if order is not None else labels
    # The last span ends at the head SEP, before state. A surviving marker can still
    # have its entire body cut by max_len; preserve both the reference and actual spans.
    stats["head_options_distinct"] = stats.pop("options_distinct")
    stats["options_distinct"] = len({tuple(span) for span in spans if span})
    return dict(stats, **trunc, head_tokens=len(head), input_tokens=len(seq),
                markers_kept=len(markers), input_sha256=fingerprint(seq),
                empty_option_bodies=sum(not span for span in spans),
                head_option_content_tokens=[len(span) for span in head_spans],
                head_option_span_sha256={label: fingerprint(span) for label, span in zip(labels, head_spans)},
                option_content_tokens=[len(span) for span in spans],
                option_span_sha256={label: fingerprint(span) for label, span in zip(labels, spans)})


def paired_comparison(candidate, baseline, confidence=0.95, samples=10000, seed=0):
    """Pair by immutable case identity; different model/strategy is the treatment."""
    if not 0 < confidence < 1 or not isinstance(samples, int) or samples < 100:
        raise ValueError("invalid bootstrap parameters")
    for key in ("dataset", "split", "dataset_revision", "questions_sha256", "checkpoint_revision"):
        if candidate["identity"].get(key) is None or candidate["identity"].get(key) != baseline["identity"].get(key):
            raise ValueError("cannot pair reports: %s differs or is missing" % key)

    def index(report):
        rows = report["cases"]
        out = {row["id"]: row for row in rows}
        if not rows or len(rows) != len(out) or report.get("status") != "complete":
            raise ValueError("cannot pair incomplete or duplicate cases")
        return out

    left, right = index(candidate), index(baseline)
    if left.keys() != right.keys():
        raise ValueError("cannot pair different case ids")
    deltas = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if (a["state_sha256"], a["expected"]) != (b["state_sha256"], b["expected"]):
            raise ValueError("cannot pair different case contents")
        deltas.append(int(a["correct"]) - int(b["correct"]))
    d = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        boot[start:start + count] = d[rng.integers(len(d), size=(count, len(d)))].mean(axis=1)
    tail = (1 - confidence) / 2
    return {"rows": len(d), "gained": int((d > 0).sum()), "lost": int((d < 0).sum()),
            "accuracy_delta": float(d.mean()), "confidence": confidence,
            "interval": np.quantile(boot, [tail, 1 - tail]).tolist(), "samples": samples, "seed": seed}
