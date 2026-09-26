"""Supply-chain integrity for published checkpoints: opt-in pins and digest checks.

Runtime loaders keep the Hub default revision unless the caller supplies one. This preserves
compatibility with existing offline caches, including ``HF_HUB_OFFLINE=1`` deployments. The
reviewed commit SHAs below are available for callers that opt in, and every loader accepts an
optional SHA-256 map to verify artifact integrity before weights reach the runtime.
"""
import hashlib
import json
import ntpath
import os
from typing import Dict, Optional

# Opt-in reviewed commit SHAs of the published checkpoints. They are not applied
# implicitly, so existing Hub/offline caches keep working; pass one explicitly to pin a load.
PINNED_REVISIONS: Dict[str, str] = {
    "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "convaiinnovations/laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
}


def resolve_revision(model_id_or_path: str, revision: Optional[str] = None) -> Optional[str]:
    """Pick the revision to download.

    An explicit `revision` is returned unchanged. Otherwise None is returned so
    huggingface_hub applies its normal default, preserving existing online/offline caches.
    """
    return revision or None


def snapshot_revision(path: str) -> Optional[str]:
    """Commit SHA a Hub snapshot directory points at, or None for a plain directory.

    `snapshot_download` returns ``<cache>/snapshots/<sha>``; resolving symlinks keeps this
    correct when the snapshot entry is a link into the blob store.
    """
    real = os.path.realpath(path).rstrip(os.sep)
    parent, base = os.path.split(real)
    if os.path.basename(parent) == "snapshots" and base:
        return base
    return None


def verify_digests(
    model_dir: str,
    expected: Optional[Dict[str, str]] = None,
    onnx_path: Optional[str] = None,
) -> None:
    """Verify SHA-256 digests of files under `model_dir` against {relpath: hexdigest}.

    Raises FileNotFoundError when a listed file is absent and ValueError on a digest
    mismatch or an unsafe (absolute or escaping) relative path. Verification runs before
    any weight is parsed or executed, so a tampered artifact never reaches the runtime.
    """
    if expected is None:
        raw = os.environ.get("LAYA_SHA256_DIGESTS", "").strip()
        if not raw:
            return
        try:
            expected = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LAYA_SHA256_DIGESTS must be a JSON object of artifact->sha256") from exc
    if not isinstance(expected, dict):
        raise ValueError("expected_sha256 must be a mapping of artifact paths to digests")
    for rel, want in expected.items():
        raw_rel = str(rel).replace("\\", "/")
        if rel in ("onnx", "onnx_path") and onnx_path:
            path = onnx_path
        else:
            if raw_rel.startswith("/") or os.path.isabs(raw_rel) or ntpath.isabs(raw_rel):
                raise ValueError("laya: unsafe absolute path in expected digests: %r" % (rel,))
            rel_norm = raw_rel.lstrip("/")
            if not rel_norm or rel_norm == ".." or rel_norm.startswith("../") or "/../" in rel_norm:
                raise ValueError("laya: unsafe path in expected digests: %r" % (rel,))
            path = os.path.join(model_dir, rel_norm)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "laya: cannot verify %r: no such file under %s" % (rel, model_dir))
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        got = digest.hexdigest()
        if got.lower() != str(want).strip().lower():
            raise ValueError(
                "laya: SHA-256 mismatch for %s: expected %s, got %s. The artifact does "
                "not match the reviewed digest; refusing to load it." % (rel, want, got))
