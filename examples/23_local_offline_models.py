"""Example 23 -- running against ./models, fully offline.

Shows that the local checkpoints need no network: prints the resolved paths, cross-checks
them against verify/checkpoints.json, and runs a prediction from disk.
"""
import json
import os

from _common import laya, MODELS, ROOT, STATE_EN, banner, describe, heading, load, router

banner("23", "Local, offline checkpoints", """
    Every example here passes absolute paths from `models/` to `Agent`, so the first thing
    `Agent` does is `os.path.exists` -- and it is true. No `snapshot_download`, no hub
    call, no token. That is the difference from a stock `Router()`, which holds Hugging
    Face repo ids and downloads the checkpoints into `HF_HOME` on first use.

    This script proves the point from disk: it resolves each local path, checks the weight
    file against the sizes recorded in `verify/checkpoints.json`, and then answers one
    question without touching the network.
    """)

print("   laya %s" % laya.__version__)

heading("resolved checkpoint paths")
for name, path in MODELS.items():
    weights = os.path.join(path, "model.safetensors")
    size = os.path.getsize(weights) if os.path.exists(weights) else 0
    print("   %-16s %s" % (name, path))
    print("   %-16s exists=%s  model.safetensors=%d bytes" %
          ("", os.path.isdir(path), size))

heading("cross-check against verify/checkpoints.json")
manifest_path = os.path.join(ROOT, "verify", "checkpoints.json")
print("   manifest: %s" % manifest_path)
if not os.path.exists(manifest_path):
    print("   (verify/checkpoints.json is not in this checkout; skipping the size cross-check)")
else:
    with open(manifest_path) as f:
        manifest = json.load(f)
    print("   repo=%s  revision=%s" % (manifest["repo"], manifest["revision"][:12]))
    for cp in manifest["checkpoints"]:
        local = MODELS[cp["name"]]
        found = os.path.getsize(os.path.join(local, "model.safetensors"))
        print("   %-16s expected=%d  local=%d  %s" %
              (cp["name"], cp["bytes"], found, "MATCH" if found == cp["bytes"] else "MISMATCH"))

heading("one prediction, straight from disk")
agent = load("english")
result = agent.predict(STATE_EN, {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email in `body`?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "refund_requested": {"type": "noul",
                         "instructions": "Does the sender ask for a refund?"},
})
describe(result["answers"])
print("   device: %s   tokens: %d in, %d out" %
      (agent.device, result["usage"]["input_tokens"], result["usage"]["output_tokens"]))

heading("Router(models=...) vs the default Router()")
local = router()
print("   local Router().models['english']      %r" % local.models["english"])
print("   default DEFAULT_MODELS['english']     %r" % (laya.DEFAULT_MODELS["english"],))
print("   local Router().models['multilingual'] %r" % local.models["multilingual"])
print("   default DEFAULT_MODELS['multilingual'] %r" % (laya.DEFAULT_MODELS["multilingual"],))
print("""
   The local values are absolute directories, so `Agent` never calls the hub. The default
   values are a repo id plus an optional subfolder; `Agent` would call `snapshot_download`
   and write the weights under `HF_HOME` (unset here, so ~/.cache/huggingface by default).
   Offline inference is therefore not a special mode -- it is what passing local paths does.
   """)
