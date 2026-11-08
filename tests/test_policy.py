"""Policies from Embedding Studio load from a folder or the downloaded .zip, and are verified first.

No model needed: these exercise laya.policy directly with real Ed25519 signatures from a throwaway
key, so a tampered file, a foreign key, a missing signature and a wrong engine are each refused
by name before anything could be installed.
"""
import base64
import hashlib
import io
import json
import zipfile

import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization as S  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from laya import policy  # noqa: E402


def _key():
    k = Ed25519PrivateKey.generate()
    pub = base64.b64encode(k.public_key().public_bytes(S.Encoding.Raw, S.PublicFormat.Raw)).decode()
    return k, pub, policy.key_id(pub)


def _files(engine="e" * 64):
    files = {"heads.npz": b"heads-bytes", "calibration.json": json.dumps(
        {"alpha": 0.1, "provenance": {"engine_sha256": engine}}).encode(),
        "questions.json": b'{"route": {"type": "choice"}}'}
    return files


def _signed(files, k, kid, engine="e" * 64):
    raw = policy.manifest_for(files, format="embedding-studio-policy/1", key_id=kid,
                              engine_sha256=engine)
    return dict(files, **{"manifest.json": raw, "manifest.sig": base64.b64encode(k.sign(raw))})


def _zip(files, folder="acme"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in files.items():
            z.writestr("%s/%s" % (folder, n) if folder else n, b)
    return buf.getvalue()


def test_a_signed_policy_verifies_from_a_zip_without_unpacking(tmp_path):
    k, pub, kid = _key()
    p = tmp_path / "acme-policy.zip"
    p.write_bytes(_zip(_signed(_files(), k, kid)))
    files = policy.read(str(p))
    report = policy.verify(files, trusted={kid: pub})
    assert report["status"] == "signed" and kid in report["signed_by"]
    assert set(files) == set(policy.FILES)


def test_files_at_the_root_of_the_zip_and_a_plain_folder_both_read(tmp_path):
    k, pub, kid = _key()
    signed = _signed(_files(), k, kid)
    z = tmp_path / "root.zip"
    z.write_bytes(_zip(signed, folder=None))
    assert policy.verify(policy.read(str(z)), trusted={kid: pub})["status"] == "signed"
    d = tmp_path / "acme"
    d.mkdir()
    for n, b in signed.items():
        (d / n).write_bytes(b)
    assert policy.verify(policy.read(str(d)), trusted={kid: pub})["status"] == "signed"


def test_a_changed_file_is_refused_by_name():
    k, pub, kid = _key()
    files = _signed(_files(), k, kid)
    files["heads.npz"] = b"someone else's heads"
    with pytest.raises(policy.PolicyError, match="heads.npz has been changed"):
        policy.verify(files, trusted={kid: pub})


def test_a_manifest_rewritten_to_match_new_files_fails_the_signature():
    k, pub, kid = _key()
    files = _signed(_files(), k, kid)
    files["heads.npz"] = b"swapped"
    raw = policy.manifest_for(files, format="embedding-studio-policy/1", key_id=kid,
                              engine_sha256="e" * 64)
    files["manifest.json"] = raw                      # hashes now match; the signature cannot
    with pytest.raises(policy.PolicyError, match="signature does not match"):
        policy.verify(files, trusted={kid: pub})


def test_a_key_laya_does_not_trust_is_refused():
    k, pub, kid = _key()
    files = _signed(_files(), k, kid)
    with pytest.raises(policy.PolicyError, match="does not trust"):
        policy.verify(files, trusted={})


def test_a_signature_by_another_key_under_a_trusted_id_is_refused():
    k, pub, kid = _key()
    other, _, _ = _key()
    files = _signed(_files(), other, kid)             # claims kid, signed by someone else
    with pytest.raises(policy.PolicyError, match="signature does not match"):
        policy.verify(files, trusted={kid: pub})


def test_a_manifest_that_names_a_key_but_has_no_signature_is_refused():
    k, pub, kid = _key()
    files = _signed(_files(), k, kid)
    files.pop("manifest.sig")
    with pytest.raises(policy.PolicyError, match="manifest.sig is missing"):
        policy.verify(files, trusted={kid: pub})


def test_unsigned_and_unverified_policies_are_reported_and_can_be_required_away():
    files = _files()
    assert policy.verify(files)["status"] == "unverified"
    with pytest.raises(policy.PolicyError, match="no manifest"):
        policy.verify(files, require_signature=True)
    files["manifest.json"] = policy.manifest_for(files, format="laya-policy/1")
    assert policy.verify(files)["status"] == "integrity"
    with pytest.raises(policy.PolicyError, match="not signed"):
        policy.verify(files, require_signature=True)


def test_the_engine_a_policy_was_trained_on_must_be_the_one_loaded():
    k, pub, kid = _key()
    files = _signed(_files(engine="a" * 64), k, kid, engine="a" * 64)
    report = policy.verify(files, trusted={kid: pub})
    assert policy.expected_engine(files, report) == "a" * 64
    policy.check_engine("a" * 64, "a" * 64)
    with pytest.raises(policy.PolicyError, match="trained on engine"):
        policy.check_engine("a" * 64, "b" * 64)
    # no manifest: the calibration's own provenance still names the engine
    assert policy.expected_engine(_files(engine="c" * 64), {}) == "c" * 64


def test_the_pinned_key_id_is_the_hash_of_its_public_key():
    for kid, pub in policy.TRUSTED_KEYS.items():
        assert kid == hashlib.sha256(base64.b64decode(pub)).hexdigest()[:16]


def test_a_manifest_that_is_not_a_json_object_is_refused_by_name():
    files = {"heads.npz": b"heads-bytes", "manifest.json": b"[]"}
    with pytest.raises(policy.PolicyError, match="manifest.json is not a JSON object"):
        policy.verify(files)


def test_manifest_files_that_is_not_a_name_to_hash_map_is_refused_by_name():
    files = {"heads.npz": b"heads-bytes", "manifest.json": b'{"files": {"heads.npz": 5}}'}
    with pytest.raises(policy.PolicyError, match="files is not a map"):
        policy.verify(files)
    files["manifest.json"] = b'{"files": "heads.npz"}'
    with pytest.raises(policy.PolicyError, match="files is not a map"):
        policy.verify(files)
