"""Example 23 -- where the weights come from: ./models, or the Hub.

Laya takes either a local directory (no network, no token) when a copy is in `models/`, or a Hub
repo id. This prints what each name resolves to, cross-checks the local weights against
verify/checkpoints.json when they are there, and runs one prediction.
"""
import json
import os

from _common import (LOCAL_MODELS, MODELS, ROOT, STATE_EN, banner, describe, has_local, heading,
                     laya, load, router)

banner("23", "Where the weights come from: ./models or the Hub", """
    Keep the three checkpoints in `models/` and `_common` prefers them: an absolute directory
    means `Agent` stops at `os.path.exists`, so nothing touches the network and no token is
    needed.

    Without `models/` the same code falls back to the Hub bundle (`convaiinnovations/laya`,
    with a subfolder per checkpoint), which downloads on first use. Same checkpoints either
    way; only where they are read from changes.
    """)

print("   laya %s" % laya.__version__)

heading("what each name resolves to")
for name in MODELS:
    print("   %-16s %r" % (name, MODELS[name]))
    if has_local(name):
        weight = os.path.join(LOCAL_MODELS[name], "model.safetensors")
        print("   %-16s local: %d bytes" % ("", os.path.getsize(weight)))
    else:
        print("   %-16s no local copy: the Hub spec above is used" % "")

heading("cross-check against verify/checkpoints.json")
manifest_path = os.path.join(ROOT, "verify", "checkpoints.json")
print("   manifest: %s" % manifest_path)
if not os.path.exists(manifest_path):
    print("   (verify/checkpoints.json is not in this checkout; skipping the size cross-check)")
elif not all(has_local(name) for name in LOCAL_MODELS):
    print("   (some checkpoints are not local; skipping the size cross-check)")
else:
    with open(manifest_path) as f:
        manifest = json.load(f)
    print("   repo=%s  revision=%s" % (manifest["repo"], manifest["revision"][:12]))
    for cp in manifest["checkpoints"]:
        found = os.path.getsize(os.path.join(LOCAL_MODELS[cp["name"]], "model.safetensors"))
        print("   %-16s expected=%d  local=%d  %s" %
              (cp["name"], cp["bytes"], found, "MATCH" if found == cp["bytes"] else "MISMATCH"))

heading("one prediction")
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
configured = router()
print("   Router().models['english']       %r" % (configured.models["english"],))
print("   DEFAULT_MODELS['english']        %r" % (laya.DEFAULT_MODELS["english"],))
print("   Router().models['multilingual']  %r" % (configured.models["multilingual"],))
print("   DEFAULT_MODELS['multilingual']   %r" % (laya.DEFAULT_MODELS["multilingual"],))
print("""
   With a local copy the configured value is an absolute directory, so `Agent` never calls the
   hub. Without one it is the same Hub spec `DEFAULT_MODELS` uses, and the weights land in
   `HF_HOME` (~/.cache/huggingface by default). Passing a local path is still the only way to
   keep a run fully offline.
   """)
