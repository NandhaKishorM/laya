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
print(script, agent, sh)
''' % ROOT

proc = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
check("no-torch/import laya succeeds", proc.returncode, 0)
out = proc.stdout.strip().split()
check("no-torch/detect_script works", out[0] if out else None, "latin")
check("no-torch/Agent stays lazy", out[1] if len(out) > 1 else None, "blocked")
check("no-torch/shortlist stays lazy", out[2] if len(out) > 2 else None, "blocked")


# ------------------------------------------------------------------ torch available
import laya  # noqa: E402

missing = [n for n in laya.__all__ if not hasattr(laya, n)]
check("all __all__ names resolve with torch", missing, [])
check("from-import of a lazy name", laya.Agent.__name__, "Agent")
check("__dir__ lists lazy names", "load" in dir(laya), True)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all lazy-import tests passed")
sys.exit(1 if FAIL else 0)
