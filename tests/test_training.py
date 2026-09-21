"""Training-side helpers: the scoring rules Laya is trained against, and the metrics around them.

`proper_reward` is the reward the model is optimised for, so its *properness* is a correctness
property, not a detail: if the reward were maximised by something other than the true
distribution, the calibration claims in the README would not hold. `td_lambda_targets` builds
bootstrapped targets for multi-turn episodes, and `ece_score` / `confidence_from_probs` are the
metrics used to report calibration.

No model weights are loaded, so this runs anywhere torch does.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from laya.common import (  # noqa: E402
    QTYPES,
    QTYPE_NAMES,
    amp_dtype,
    build_sequence,
    collate_items,
    confidence_from_probs,
    ece_score,
    proper_reward,
    temp_bucket,
    td_lambda_targets,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def close(name, got, want, tol=1e-6):
    check_true(name, abs(got - want) <= tol, "got %r want %r" % (got, want))


def one_hot(i, k):
    row = [0.0] * k
    row[i] = 1.0
    return row


# =============================================================== proper_reward
# The reward is a weighted sum of a log score and a spherical score, minus a ranked probability
# score for ordinal (`score`) questions only.
Q = torch.tensor([[0.7, 0.3]])
T = torch.tensor([[1.0, 0.0]])
M = torch.tensor([[1.0, 1.0]])
QT = torch.tensor([QTYPES["choice"]])

r_perfect = proper_reward(torch.tensor([[0.7, 0.3]]), T, QT, M).item()
r_wrong = proper_reward(torch.tensor([[0.3, 0.7]]), T, QT, M).item()
check_true("reward/perfect beats wrong", r_perfect > r_wrong, "%.4f vs %.4f" % (r_perfect, r_wrong))

# Expected reward under the target must be maximised at q == target (strict properness).
# E[r(q, y)] = sum_i p_i * r(q, one_hot(i)) -- the property that makes the reported
# probabilities meaningful.
def expected_reward(q_vec, target, qtype=QTYPES["choice"], w_sph=0.5, w_rps=1.0):
    q = torch.tensor([q_vec], dtype=torch.float32)
    qt = torch.tensor([qtype])
    mask = torch.ones_like(q)
    total = 0.0
    for i, p in enumerate(target):
        if p == 0:
            continue
        tgt = torch.tensor([one_hot(i, len(target))], dtype=torch.float32)
        total += p * proper_reward(q, tgt, qt, mask, w_sph=w_sph, w_rps=w_rps).item()
    return total


target = [0.7, 0.3]
grid = [0.5, 0.6, 0.7, 0.8, 0.9]
rewards = [(q, expected_reward([q, 1 - q], target)) for q in grid]
best = max(rewards, key=lambda kv: kv[1])[0]
close("reward/strictly proper: argmax at q == target (choice)", best, 0.7)

# same property for the ordinal (score) rule, where the RPS term also applies
rewards_score = [(q, expected_reward([q, 1 - q], target, qtype=QTYPES["score"])) for q in grid]
close("reward/strictly proper: argmax at q == target (score)",
      max(rewards_score, key=lambda kv: kv[1])[0], 0.7)

# the RPS term is score-only: the same inputs must score differently for the two types
r_choice = proper_reward(Q, T, torch.tensor([QTYPES["choice"]]), M).item()
r_score = proper_reward(Q, T, torch.tensor([QTYPES["score"]]), M).item()
check_true("reward/rps applies to score but not choice", r_score < r_choice,
           "score %.4f vs choice %.4f" % (r_score, r_choice))

# a masked-out option cannot influence the reward
r_masked_a = proper_reward(torch.tensor([[0.9, 0.1]]), T, QT, torch.tensor([[1.0, 0.0]])).item()
r_masked_b = proper_reward(torch.tensor([[0.9, 0.9]]), T, QT, torch.tensor([[1.0, 0.0]])).item()
close("reward/masked option is ignored", r_masked_a, r_masked_b)

# zero probability is clamped by log_floor rather than producing -inf
r_zero = proper_reward(torch.tensor([[1.0, 0.0]]), T, QT, M).item()
check_true("reward/zero probability is finite", math.isfinite(r_zero), "%r" % r_zero)

# permuting options consistently leaves the reward unchanged
perm = proper_reward(torch.tensor([[0.3, 0.7]]), torch.tensor([[0.0, 1.0]]), QT, M).item()
close("reward/invariant to consistent permutation", perm, proper_reward(Q, T, QT, M).item())

# the spherical weight is a real knob
hi = proper_reward(Q, T, QT, M, w_sph=2.0).item()
lo = proper_reward(Q, T, QT, M, w_sph=0.0).item()
check_true("reward/w_sph changes the reward", hi > lo, "%.4f vs %.4f" % (hi, lo))

# multi-row batches are handled independently
batch_r = proper_reward(torch.tensor([[0.7, 0.3], [0.1, 0.9]]),
                        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                        torch.tensor([QTYPES["choice"], QTYPES["choice"]]),
                        torch.ones(2, 2))
check("reward/batch shape", tuple(batch_r.shape), (2,))


# =============================================================== td_lambda_targets
p_true = torch.tensor([0.1, 0.4, 0.9])

# without episode grouping the targets are returned untouched
batch = {"target": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])}
out = td_lambda_targets(p_true, batch, lam=1.0)
check_true("td_lambda/no groups -> unchanged targets",
           torch.equal(out, batch["target"]))

# grouped: rows are ordered by ep_step, and the terminal row's truth bootstraps backwards
batch = {"target": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
         "ep_group": torch.tensor([0, 0, 0]),
         "ep_step": torch.tensor([0, 1, 2])}
lam1 = td_lambda_targets(p_true, {"target": batch["target"].clone(),
                                  "ep_group": batch["ep_group"],
                                  "ep_step": batch["ep_step"]}, lam=1.0)
close("td_lambda/lam=1 propagates terminal truth to every step", float(lam1[0, 1]), 0.0)
close("td_lambda/lam=1 terminal step unchanged", float(lam1[2, 1]), 0.0)
check_true("td_lambda/lam=1 all steps equal the terminal target",
           torch.allclose(lam1[:, 0], 1 - lam1[:, 1]))

lam0 = td_lambda_targets(p_true, {"target": batch["target"].clone(),
                                  "ep_group": batch["ep_group"],
                                  "ep_step": batch["ep_step"]}, lam=0.0)
close("td_lambda/lam=0 step0 uses the next step's p_true", float(lam0[0, 1]), 0.4)
close("td_lambda/lam=0 step1 uses the next step's p_true", float(lam0[1, 1]), 0.9)
close("td_lambda/lam=0 terminal keeps the terminal target", float(lam0[2, 1]), 0.0)

# row order must not matter: steps are sorted by ep_step inside each group, and `p_true` is
# indexed by row, so the same logical episode written out of step order must agree per step.
shuffled = td_lambda_targets(torch.tensor([0.9, 0.1, 0.4]),          # rows are [step2, step0, step1]
                             {"target": torch.tensor([[1.0, 0.0]] * 3),
                              "ep_group": torch.tensor([0, 0, 0]),
                              "ep_step": torch.tensor([2, 0, 1])}, lam=0.0)
ordered = td_lambda_targets(torch.tensor([0.1, 0.4, 0.9]),           # the same episode in step order
                            {"target": torch.tensor([[1.0, 0.0]] * 3),
                             "ep_group": torch.tensor([0, 0, 0]),
                             "ep_step": torch.tensor([0, 1, 2])}, lam=0.0)
check_true("td_lambda/rows are processed in ep_step order",
           float(ordered[0, 1]) == float(shuffled[1, 1]) and
           float(ordered[1, 1]) == float(shuffled[2, 1]) and
           float(ordered[2, 1]) == float(shuffled[0, 1]),
           "ordered=%s shuffled=%s" % (ordered[:, 1].tolist(), shuffled[:, 1].tolist()))

# groups are independent: each episode bootstraps from its own terminal row
two = td_lambda_targets(torch.tensor([0.1, 0.2, 0.8, 0.9]),
                        {"target": torch.tensor([[1.0, 0.0], [0.2, 0.8], [1.0, 0.0], [0.6, 0.4]]),
                         "ep_group": torch.tensor([0, 0, 1, 1]),
                         "ep_step": torch.tensor([0, 1, 0, 1])}, lam=1.0)
check("td_lambda/group 0 uses its own terminal", [round(float(v), 4) for v in two[0:2, 1]], [0.8, 0.8])
check("td_lambda/group 1 uses its own terminal", [round(float(v), 4) for v in two[2:4, 1]], [0.4, 0.4])


# =============================================================== ece_score
# perfectly calibrated: 90% confident, right 90% of the time
conf = np.full(100, 0.9)
correct = np.array([1] * 90 + [0] * 10)
close("ece/perfectly calibrated is ~0", ece_score(conf, correct), 0.0, tol=1e-9)

# maximally overconfident: certain and always wrong
close("ece/certain and wrong is 1.0", ece_score(np.ones(10), np.zeros(10, dtype=int)), 1.0, tol=1e-9)

# empty input is nan rather than an exception
check_true("ece/empty is nan", math.isnan(ece_score(np.array([]), np.array([]))))

# overconfidence costs: same accuracy, higher confidence -> higher ECE
base_correct = np.array([1] * 80 + [0] * 20)
conf_a = np.full(100, 0.8)     # calibrated
conf_b = np.full(100, 0.99)    # overconfident
check_true("ece/overconfidence raises ECE", ece_score(conf_b, base_correct) > ece_score(conf_a, base_correct),
           "%.4f vs %.4f" % (ece_score(conf_b, base_correct), ece_score(conf_a, base_correct)))

# bins argument changes granularity, not correctness of the perfect case
close("ece/bins=5 perfectly calibrated", ece_score(conf, correct, bins=5), 0.0, tol=1e-9)


# =============================================================== confidence_from_probs
close("confidence/uniform over 2 is 0", confidence_from_probs(np.array([0.5, 0.5]), 2), 0.0)
close("confidence/uniform over 5 is 0", confidence_from_probs(np.full(5, 0.2), 5), 0.0)
close("confidence/one-hot is 1", confidence_from_probs(np.array([1.0, 0.0]), 2), 1.0)
close("confidence/single option is 1 by definition", confidence_from_probs(np.array([1.0]), 1), 1.0)
spread = confidence_from_probs(np.array([0.6, 0.3, 0.1]), 3)
peaked = confidence_from_probs(np.array([0.9, 0.05, 0.05]), 3)
check_true("confidence/decreases with entropy", peaked > spread, "%.4f vs %.4f" % (peaked, spread))
close("confidence/only the first k entries count", confidence_from_probs(np.array([0.5, 0.5, 0.0]), 2), 0.0)


# =============================================================== small helpers
check("qtypes/choice", QTYPES["choice"], 0)
check("qtypes/score", QTYPES["score"], 1)
check("qtypes/noul", QTYPES["noul"], 2)
check("qtypes/names round-trip", QTYPE_NAMES[QTYPES["noul"]], "noul")

check("temp_bucket/2 options is noul:2", temp_bucket(QTYPES["noul"], 2), "noul:2")
check("temp_bucket/3-5", temp_bucket(QTYPES["choice"], 4), "choice:3-5")
check("temp_bucket/6-10", temp_bucket(QTYPES["choice"], 10), "choice:6-10")
check("temp_bucket/11+", temp_bucket(QTYPES["choice"], 77), "choice:11+")
check("temp_bucket/score bucket", temp_bucket(QTYPES["score"], 3), "score:3-5")

check("amp_dtype/bf16", amp_dtype("bf16"), torch.bfloat16)
check("amp_dtype/anything else is fp16", amp_dtype(None), torch.float16)


# =============================================================== collate_items
items = [{"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0, "label": 1},
         {"ids": [4, 5], "markers": [1], "qtype": 2, "label": -1}]
b = collate_items([items], pad_id=0)
check("collate/batch size", tuple(b["input_ids"].shape), (2, 3))
check("collate/marker columns = longest marker list", tuple(b["marker_pos"].shape), (2, 2))
check("collate/padding filled with pad_id", b["input_ids"][1].tolist(), [4, 5, 0])
check("collate/attention mask marks real tokens", b["attention_mask"][1].tolist(), [1, 1, 0])
check("collate/marker mask marks real markers", b["marker_mask"][1].tolist(), [True, False])
check("collate/qtype carried", b["qtype"].tolist(), [0, 2])
check("collate/no target unless requested", "target" in b, False)
b2 = collate_items([[{"ids": [1], "markers": [0, 1], "qtype": 1, "target": [0.0, 1.0]}]], pad_id=0)
check("collate/target included when present", tuple(b2["target"].shape), (1, 2))
check("collate/empty batch returns None", collate_items([], pad_id=0), None)

# a target with more entries than the item has options is a data bug; it must say so, not
# surface as a tensor shape error from inside the assignment
try:
    collate_items([[{"ids": [1], "markers": [0], "qtype": 1, "target": [0.0, 1.0]}]], pad_id=0)
    FAIL.append("collate/oversized target raises ValueError (nothing raised)")
except ValueError as e:
    check_true("collate/oversized target raises a clear ValueError",
               "one entry per option" in str(e), str(e)[:90])
except Exception as e:  # noqa: BLE001
    FAIL.append("collate/oversized target raised %s instead of ValueError: %s" % (type(e).__name__, e))


# =============================================================== build_sequence
class _Tok:
    """Minimal stand-in for a fast tokenizer: one id per character."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 50) for c in text]}


tok = _Tok()
q = {"t": "choice", "ins": "Which team?", "crit": {"a": "first", "b": "second", "c": "third"}}
seq, markers = build_sequence(tok, "some state text", q, max_len=128, head_max_len=64)
check("build_sequence/one marker per option", len(markers), 3)
check_true("build_sequence/markers point inside the sequence", all(0 <= m < len(seq) for m in markers))
check_true("build_sequence/starts with CLS", seq[0] == tok.cls_token_id)
check_true("build_sequence/ends with SEP", seq[-1] == tok.sep_token_id)
check_true("build_sequence/respects max_len", len(seq) <= 128)

long_seq, long_markers = build_sequence(tok, "x" * 5000, q, max_len=64, head_max_len=32)
check_true("build_sequence/truncates long state to max_len", len(long_seq) <= 64, "len=%d" % len(long_seq))

noul_seq, noul_markers = build_sequence(tok, "state", {"t": "noul", "ins": "Is it?", "crit": None}, 128, 64)
check("build_sequence/noul always offers two options", len(noul_markers), 2)

score_seq, score_markers = build_sequence(tok, "state", {"t": "score", "ins": "How bad?",
                                                         "crit": ["low", "mid", "high"]}, 128, 64)
check("build_sequence/score has one marker per level", len(score_markers), 3)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
