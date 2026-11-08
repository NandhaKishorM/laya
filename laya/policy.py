"""Policies trained in Embedding Studio: read from a folder or straight from the downloaded .zip,
and verified before a single head is installed.

A policy is three files, ``heads.npz`` (the trained layer per question), ``calibration.json`` and
``questions.json``. Studio adds two more:

    manifest.json   the SHA-256 of every file, the reader and the engine the policy was trained
                    on, the project and the fit, and the id of the key that signed it
    manifest.sig    an Ed25519 signature of manifest.json's exact bytes, by Embedding Space

Verification, all of it before anything is loaded:

1. every file the manifest names has exactly the SHA-256 it states (a changed or swapped file
   is refused, naming the file);
2. a manifest that names a signing key must carry a signature that verifies against a key this
   release of laya trusts (``TRUSTED_KEYS``, pinned below; an unknown key or a bad signature is
   refused, so a policy cannot be re-signed by anybody else);
3. the engine the agent loaded has the SHA-256 the policy was trained on (heads trained on one
   reader's features are meaningless on another's), checked by the caller with :func:`check_engine`.

A folder written by laya's own ``save_calibration`` has a manifest with hashes and no key: it is
checked for integrity and reported as unsigned. A folder with no manifest at all (written before
0.4.28) loads as before and is reported as unverified.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import zipfile
from typing import Any, Dict, Mapping, Optional

#: Public keys whose signatures laya accepts, by key id (the first 16 hex characters of the
#: SHA-256 of the raw public key). Embedding Space signs every policy Studio exports with the
#: private half, which never leaves its servers.
TRUSTED_KEYS: Dict[str, str] = {
    "589c4677fa61c61c": "dZMFedqlr5VlF5yt+sT0L86hrmprsLPCF6CLyIaA38U=",
}

FILES = ("heads.npz", "calibration.json", "questions.json", "manifest.json", "manifest.sig")


class PolicyError(ValueError):
    """The policy failed verification; the message names what and why."""


def key_id(public_key_b64: str) -> str:
    return hashlib.sha256(base64.b64decode(public_key_b64)).hexdigest()[:16]


def read(path: str) -> Dict[str, bytes]:
    """The policy's files as bytes, from a folder or a .zip (files at the root of the archive or
    inside one top-level folder, as Studio's download has them). Nothing is extracted to disk."""
    out: Dict[str, bytes] = {}
    if os.path.isdir(path):
        for name in FILES:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    out[name] = f.read()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = info.filename.rsplit("/", 1)[-1]
                if name in FILES and info.filename.count("/") <= 1:
                    if name in out:
                        raise PolicyError("%s holds more than one %s" % (path, name))
                    out[name] = z.read(info)
    else:
        raise PolicyError("%s is neither a policy folder nor a .zip" % path)
    if "heads.npz" not in out:
        raise PolicyError("%s has no heads.npz; it is not a trained policy" % path)
    return out


def manifest_for(files: Mapping[str, bytes], **fields: Any) -> bytes:
    """A manifest naming the SHA-256 of each file, plus whatever fields are given. Serialised once
    and kept as bytes: the signature covers these exact bytes, so nobody re-serialises it."""
    body = dict(fields)
    body["files"] = {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())
                     if n not in ("manifest.json", "manifest.sig")}
    return json.dumps(body, indent=2, sort_keys=True).encode("utf-8")


def verify(files: Mapping[str, bytes], trusted: Optional[Mapping[str, str]] = None,
           require_signature: bool = False) -> Dict[str, Any]:
    """Check the manifest's hashes and signature. Returns what was verified; raises PolicyError."""
    trusted = TRUSTED_KEYS if trusted is None else trusted
    raw = files.get("manifest.json")
    if raw is None:
        if require_signature:
            raise PolicyError("this policy has no manifest, so its origin cannot be verified")
        return {"status": "unverified", "reason": "no manifest (a policy from before laya 0.4.28)"}
    try:
        manifest = json.loads(raw)
    except ValueError as e:
        raise PolicyError("manifest.json is not valid JSON: %s" % e) from e
    if not isinstance(manifest, dict):
        raise PolicyError("manifest.json is not a JSON object")
    hashes = manifest.get("files")
    if hashes is None:
        hashes = {}
    if not isinstance(hashes, dict) or not all(
            isinstance(name, str) and isinstance(want, str) for name, want in hashes.items()):
        raise PolicyError("manifest.json's files is not a map of file names to SHA-256 hashes")
    for name, want in hashes.items():
        have = files.get(name)
        if have is None:
            raise PolicyError("the manifest lists %s but the policy does not contain it" % name)
        got = hashlib.sha256(have).hexdigest()
        if got != want:
            raise PolicyError("%s has been changed: its SHA-256 is %s, the manifest says %s"
                              % (name, got[:16], want[:16]))
    for name in ("heads.npz", "calibration.json"):
        if name in files and name not in hashes:
            raise PolicyError("%s is in the policy but not in its manifest" % name)
    kid = manifest.get("key_id")
    if not kid:
        if require_signature:
            raise PolicyError("this policy is not signed, so its origin cannot be verified")
        return {"status": "integrity", "signed_by": None, "manifest": manifest}
    sig = files.get("manifest.sig")
    if sig is None:
        raise PolicyError("the manifest names signing key %s but manifest.sig is missing" % kid)
    pub = trusted.get(kid)
    if pub is None:
        raise PolicyError("signed with key %s, which this release of laya does not trust; "
                          "upgrade laya, or the policy did not come from Embedding Space" % kid)
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as e:   # a core dependency since 0.4.28; said plainly if it is missing
        raise PolicyError("verifying a signed policy needs the cryptography package: "
                          "pip install cryptography") from e
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(pub)).verify(
            base64.b64decode(sig.strip()), raw)
    except (InvalidSignature, ValueError) as e:
        raise PolicyError("the signature does not match the manifest: the policy was altered "
                          "after Embedding Space signed it, or it was signed by someone else") from e
    return {"status": "signed", "signed_by": "Embedding Space (key %s)" % kid, "manifest": manifest}


def expected_engine(files: Mapping[str, bytes], report: Mapping[str, Any]) -> Optional[str]:
    """The engine SHA-256 the policy was trained on: the manifest's, else the calibration's own."""
    m = report.get("manifest") or {}
    if m.get("engine_sha256"):
        return str(m["engine_sha256"])
    try:
        cal = json.loads(files.get("calibration.json") or b"{}")
    except ValueError:
        return None
    return ((cal.get("provenance") or {}).get("engine_sha256")) or None


def check_engine(expected: Optional[str], actual: Optional[str]) -> None:
    if expected and actual and expected != actual:
        raise PolicyError("this policy was trained on engine %s… but the loaded engine is %s…; "
                          "load the reader it was trained for (the manifest names it)"
                          % (expected[:12], actual[:12]))


def npz(files: Mapping[str, bytes]):
    import numpy as np
    return np.load(io.BytesIO(files["heads.npz"]), allow_pickle=False)
