"""`import laya` must not import torch; torch-backed names stay lazy.

The pure-Python surface (routing, language detection, email cleaning) has to be usable
without torch installed. The torch-backed names must still resolve when it is available.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


# ------------------------------------------------------------------ torch blocked
PROBE = r'''
import sys
sys.path.insert(0, %r)
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked")
        return None
sys.meta_path.insert(0, Blocker())

import laya
assert "torch" not in sys.modules, "import laya pulled in torch"
# `laya.serve` (#31) defers fastapi/uvicorn/torch into its functions, so importing it must
# stay cheap too -- otherwise the lazy package init buys nothing for the server entry point.
import laya.serve  # noqa: F401
assert "torch" not in sys.modules, "import laya.serve pulled in torch"
script = laya.detect_script("The customer was charged twice")
try:
    laya.Agent
    agent = "resolved-without-torch"
except ImportError:
    agent = "blocked"
try:
    laya.shortlist_choice
    sh = "resolved-without-torch"
except ImportError:
    sh = "blocked"
# The reviewed pins must be reachable with torch unavailable: the checkpoint-integrity guide
# tells operators to pass one, and pinning matters most in the offline and on-device
# deployments that may not have torch imported at all.
try:
    pins = "ok" if laya.PINNED_REVISIONS and all(
        isinstance(v, str) for v in laya.PINNED_REVISIONS.values()) else "broken"
except Exception:
    # Guarded like the probes above: an unguarded raise here kills the subprocess before the
    # print, so every check in this file reports None and the failure looks unrelated.
    pins = "missing"
print(script, agent, sh, pins)
''' % ROOT

proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
check("no-torch/import laya succeeds", proc.returncode, 0)
out = proc.stdout.strip().split()
check("no-torch/detect_script works", out[0] if out else None, "latin")
check("no-torch/Agent stays lazy", out[1] if len(out) > 1 else None, "blocked")
check("no-torch/shortlist stays lazy", out[2] if len(out) > 2 else None, "blocked")
check("no-torch/PINNED_REVISIONS resolves", out[3] if len(out) > 3 else None, "ok")


# ------------------------------------------------------------------ torch available
import laya  # noqa: E402

missing = [n for n in laya.__all__ if not hasattr(laya, n)]
check("all __all__ names resolve with torch", missing, [])
check("from-import of a lazy name", laya.Agent.__name__, "Agent")
check("__dir__ lists lazy names", "load" in dir(laya), True)


# ------------------------------------------------------------------ USE_TF guard (#915)
# On transformers 4.x the model path reaches `transformers.modeling_utils` for `no_init_weights`,
# and importing that module runs a chain (`loss_utils` -> `loss_d_fine` ->
# `loss_for_object_detection` -> `image_transforms`) that ends in `import tensorflow`. Where
# TensorFlow is installed but cannot load, that is a native crash, not an exception -- `Fatal
# Python error: Bus error` -- out of a library laya never uses. `laya/__init__.py` now sets
# `USE_TF=0` before anything can import transformers.
#
# The check is done in a subprocess because transformers decides TensorFlow's availability when it
# is first imported: by the time this file is running, transformers is already in `sys.modules`, so
# an in-process assertion could not tell "the guard worked" from "it was too late but harmless".
TF_GUARD = r'''
import os, sys
sys.path.insert(0, %r)
os.environ.pop("USE_TF", None)

# Make TensorFlow unavailable the way a broken install is, but as an ImportError rather than a
# segfault: if laya's import chain reaches for it at all, the import fails loudly here.
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ImportError("tensorflow blocked")
        return None
sys.meta_path.insert(0, Blocker())

import laya
print(os.environ.get("USE_TF"), "tensorflow" in sys.modules)
''' % ROOT

guard = subprocess.run([sys.executable, "-c", TF_GUARD], capture_output=True, text=True)
check("USE_TF/import laya succeeds with tensorflow blocked", guard.returncode, 0)
gout = guard.stdout.strip().split()
check("USE_TF/is set to 0 on import", gout[0] if gout else None, "0")
check("USE_TF/tensorflow is not imported", gout[1] if len(gout) > 1 else None, "False")

# ...and a caller who set it deliberately keeps their value: `setdefault`, not assignment. `1` is
# the interesting one, because it is the value that asks FOR tensorflow and must survive.
RESPECT = r'''
import os, sys
sys.path.insert(0, %r)
import laya
print(os.environ["USE_TF"])
''' % ROOT

for preset in ("1", "0"):
    env = dict(os.environ, USE_TF=preset)
    resp = subprocess.run([sys.executable, "-c", RESPECT], capture_output=True, text=True, env=env)
    check("USE_TF/a caller's %r is preserved" % preset, resp.stdout.strip(), preset)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all lazy-import tests passed")
sys.exit(1 if FAIL else 0)
