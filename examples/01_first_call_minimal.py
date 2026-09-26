"""Example 01 -- the absolute minimum Laya call.

No helpers and no project layout: one import, one local checkpoint, one typed question, one
forward pass. Copy this file into your own project and it still runs.
"""
import os

# Set these before importing laya: a stray TensorFlow install can deadlock model loading, and
# without the last one the fast tokenizer forks a thread per core.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya  # noqa: E402  -- must come after the environment variables above

# The checkpoint that ships with this repository, resolved from this file so the script runs
# from any working directory. To use the published model instead, delete these two lines and
# replace the load below with the single line:
#
#     agent = laya.load("convaiinnovations/laya")
#
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(os.path.dirname(HERE), "models", "laya")

agent = laya.load(CHECKPOINT)              # device=None -> CUDA, else MPS, else CPU

result = agent.predict(
    {"body": "Hi, we were billed twice for March. Please refund the duplicate today."},
    {"refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    }},
)

answer = result["answers"]["refund_requested"]
print("P(refund requested) = %.4f  (confidence %.4f)" % (answer["noul"], answer["confidence"]))
