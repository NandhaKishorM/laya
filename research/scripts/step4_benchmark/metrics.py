"""Metric definitions for the Step-4 benchmark, mirroring the published notebook.

Reference: cell 14 of `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`. The published
0.766 checkpoint and the maintainer's reference table are scored with these definitions, so a
reproduction has to use them too, not a differently-defined library metric:

- **accuracy** — top reported option against the gold label, over every decision (choice, score
  and noul).
- **ECE** — `laya.common.ece_score` of the top probability against correctness, 15 equal-width
  bins, over every decision.
- **brier** — mean over **choice and noul** decisions of `sum((p - gold)**2)`. The notebook does
  not append score decisions to this mean, so a score change never moves Brier; kept as-is so the
  number compares with the reference.
- **soft_accuracy** — mean over **choice and noul** of `sum(p * gold)`.

`gold` is the teacher distribution (`gold.probabilities`), not a one-hot label, so Brier and soft
accuracy measure the distribution the run was trained to match.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from laya.common import ece_score


def accuracy(correct: Sequence[float]) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[float],
                               bins: int = 15) -> float:
    """Thin wrapper over the repository's `common.ece_score`, as the notebook calls it."""
    if not len(confidences):
        return float("nan")
    return float(ece_score(np.asarray(confidences, dtype=float),
                           np.asarray(correct, dtype=float), bins=bins))


def brier_score(probs: Sequence[float], gold: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=float)
    g = np.asarray(gold, dtype=float)
    return float(((p - g) ** 2).sum())


def soft_accuracy(probs: Sequence[float], gold: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=float)
    g = np.asarray(gold, dtype=float)
    return float((p * g).sum())
