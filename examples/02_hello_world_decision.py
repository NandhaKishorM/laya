"""Example 02 -- the smallest useful Laya call.

One checkpoint, one question, one forward pass. Laya returns a typed answer, not text, so
there is nothing to parse and nothing to hallucinate.
"""
from _common import STATE_EN, banner, describe, device_line, load

banner("02", "Hello, decision", """
    Loads the English checkpoint and asks a single yes/no question about an email.

    The whole API is three steps: load a checkpoint, hand it a state and typed questions,
    read the answers. The first call also loads the weights, so it takes a few seconds;
    the answer itself is milliseconds.
    """)

agent = load("english")          # device=None -> CUDA, else MPS, else CPU
device_line(agent)

result = agent.predict(STATE_EN, {
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    }
})

describe(result["answers"])
print("\n   raw answer: %r" % result["answers"]["refund_requested"])
print("   tokens used: %d in, %d out" % (result["usage"]["input_tokens"], result["usage"]["output_tokens"]))
