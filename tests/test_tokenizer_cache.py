"""`_fix_tokenizer_config` must not write through the HuggingFace snapshot symlink.

`snapshot_download` lays snapshots out as symlinks into a shared `blobs/` store, so opening
the snapshot file for writing truncates the shared blob. This test builds that exact layout
and asserts the blob survives and the snapshot is swapped in atomically, with no temporary
file left behind.
"""
import os
import shutil
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.agent import _fix_tokenizer_config  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


root = tempfile.mkdtemp(prefix="laya_cache_")
blob_dir = os.path.join(root, "models--x", "blobs")
snap_dir = os.path.join(root, "models--x", "snapshots", "rev1", "tokenizer")
os.makedirs(blob_dir)
os.makedirs(snap_dir)

ORIGINAL = '{"tokenizer_class": "TokenizersBackend", "backend": "x", "is_local": true}'
blob = os.path.join(blob_dir, "deadbeef")
with open(blob, "w") as f:
    f.write(ORIGINAL)

link = os.path.join(snap_dir, "tokenizer_config.json")
os.symlink(blob, link)

check("setup/snapshot file is a symlink", os.path.islink(link), True)

_fix_tokenizer_config(os.path.dirname(snap_dir))

check("blob/is untouched", open(blob).read(), ORIGINAL)
check("snapshot/is now a regular file", os.path.islink(link), False)
check("snapshot/carries the patch", '"PreTrainedTokenizerFast"' in open(link).read(), True)
check("snapshot/drops backend+is_local as intended", '"backend": "x"' not in open(link).read(), True)
check("snapshot/no temporary file left behind",
      [n for n in os.listdir(snap_dir) if n.startswith(".tokenizer_config.")], [])
check("snapshot/preserves the file mode",
      os.stat(link).st_mode & 0o777, os.stat(blob).st_mode & 0o777)

# a second call is a no-op and must not disturb anything either
_fix_tokenizer_config(os.path.dirname(snap_dir))
check("idempotent/blob still untouched", open(blob).read(), ORIGINAL)

# an unparseable config must warn (not crash, not silently do nothing)
bad_snap = os.path.join(root, "bad", "tokenizer")
os.makedirs(bad_snap)
bad_file = os.path.join(bad_snap, "tokenizer_config.json")
with open(bad_file, "w") as f:
    f.write("{ not json")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _fix_tokenizer_config(os.path.dirname(bad_snap))
check(
    "bad-json/warns instead of raising",
    any(issubclass(w.category, RuntimeWarning) for w in caught),
    True,
)
check("bad-json/file left as written", open(bad_file).read(), "{ not json")

# if the atomic swap fails, the temp file must be cleaned up and the original left intact
fail_snap = os.path.join(root, "fail", "tokenizer")
os.makedirs(fail_snap)
fail_link = os.path.join(fail_snap, "tokenizer_config.json")
os.symlink(blob, fail_link)

real_replace = os.replace


def _boom(*args, **kwargs):
    raise OSError("replace failed")


os.replace = _boom
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fix_tokenizer_config(os.path.dirname(fail_snap))
finally:
    os.replace = real_replace

check("replace-failure/warns instead of raising",
      any(issubclass(w.category, RuntimeWarning) for w in caught), True)
check("replace-failure/no temp file left behind",
      [n for n in os.listdir(fail_snap) if n.startswith(".tokenizer_config.")], [])
check("replace-failure/original symlink intact", os.path.islink(fail_link), True)
check("replace-failure/blob untouched", open(blob).read(), ORIGINAL)

shutil.rmtree(root, ignore_errors=True)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all tokenizer-cache tests passed")
sys.exit(1 if FAIL else 0)
