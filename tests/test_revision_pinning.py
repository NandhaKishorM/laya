"""Revision pinning and digest verification tests; no network required.

Run: python tests/test_revision_pinning.py
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.agent import Agent  # noqa: E402
from laya.revisions import (  # noqa: E402
    PINNED_REVISIONS,
    resolve_revision,
    snapshot_revision,
    verify_digests,
)
from laya.router import Router  # noqa: E402


class _StopLoad(Exception):
    """Sentinel raised by the fake snapshot_download once kwargs are captured."""


def _capturing_snapshot(captured):
    def fake_snapshot(repo_id, **kwargs):
        captured.update(kwargs)
        raise _StopLoad
    return fake_snapshot


class ResolveRevisionTests(unittest.TestCase):
    def test_explicit_revision_is_returned(self):
        self.assertEqual(resolve_revision("convaiinnovations/laya", "abc123"), "abc123")

    def test_published_repos_keep_the_hub_default_without_an_explicit_pin(self):
        for repo in PINNED_REVISIONS:
            self.assertIsNone(resolve_revision(repo))

    def test_unknown_repo_keeps_the_hub_default(self):
        self.assertIsNone(resolve_revision("acme/custom-model"))
        self.assertIsNone(resolve_revision("acme/custom-model", ""))


class SnapshotRevisionTests(unittest.TestCase):
    def test_snapshot_layout(self):
        self.assertEqual(snapshot_revision("/cache/models--a--b/snapshots/deadbeef"), "deadbeef")

    def test_plain_directory(self):
        self.assertIsNone(snapshot_revision("/plain/dir"))
        self.assertIsNone(snapshot_revision(""))


class VerifyDigestsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        with open(os.path.join(self.dir, "weights.bin"), "wb") as f:
            f.write(b"weights")
        self.digest = hashlib.sha256(b"weights").hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_matching_digest_passes(self):
        verify_digests(self.dir, {"weights.bin": self.digest})
        verify_digests(self.dir, {"weights.bin": self.digest.upper()})

    def test_mismatch_raises(self):
        with self.assertRaises(ValueError):
            verify_digests(self.dir, {"weights.bin": "0" * 64})

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            verify_digests(self.dir, {"absent.bin": self.digest})

    def test_escaping_paths_rejected(self):
        for rel in ("../evil", "..", "a/../../evil", "\\..\\evil", "/absolute/evil", "C:\\absolute\\evil"):
            with self.assertRaises(ValueError, msg=rel):
                verify_digests(self.dir, {rel: self.digest})

    def test_digest_map_can_come_from_environment(self):
        with patch.dict(os.environ, {"LAYA_SHA256_DIGESTS": json.dumps({"weights.bin": self.digest})}):
            verify_digests(self.dir)

    def test_external_onnx_digest_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            with open(path, "wb") as f:
                f.write(b"onnx")
            digest = hashlib.sha256(b"onnx").hexdigest()
            verify_digests(self.dir, {"onnx": digest}, onnx_path=path)


class AgentPinningTests(unittest.TestCase):
    def test_hub_load_keeps_the_default_revision(self):
        captured = {}
        with patch("huggingface_hub.snapshot_download", _capturing_snapshot(captured)):
            with self.assertRaises(_StopLoad):
                Agent("convaiinnovations/laya")
        self.assertNotIn("revision", captured)

    def test_explicit_revision_overrides_the_pin(self):
        captured = {}
        with patch("huggingface_hub.snapshot_download", _capturing_snapshot(captured)):
            with self.assertRaises(_StopLoad):
                Agent("convaiinnovations/laya", revision="abc123")
        self.assertEqual(captured["revision"], "abc123")

    def test_unpinned_repo_gets_no_revision_kwarg(self):
        captured = {}
        with patch("huggingface_hub.snapshot_download", _capturing_snapshot(captured)):
            with self.assertRaises(_StopLoad):
                Agent("acme/custom-model")
        self.assertNotIn("revision", captured)

    def test_local_path_never_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("huggingface_hub.snapshot_download") as download:
                with self.assertRaises(FileNotFoundError):
                    Agent(tmp)
            download.assert_not_called()

    def test_digest_mismatch_raises_before_weights_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "rl_agent_config.json").write_text(json.dumps({"act_costs": {"a": 0}}))
            (Path(tmp) / "model.safetensors").write_bytes(b"not the reviewed weights")
            with self.assertRaises(ValueError):
                Agent(tmp, expected_sha256={"model.safetensors": "0" * 64})


class RouterRevisionTests(unittest.TestCase):
    def test_router_stores_explicit_revisions(self):
        router = Router(revision="default", revisions={"ml": "multi-sha", "typed": "typed-sha"})
        self.assertEqual(router.revision, "default")
        self.assertEqual(router.revisions["multilingual"], "multi-sha")
        self.assertEqual(router.revisions["typed-decisions"], "typed-sha")
        self.assertIsNone(Router().revision)
        self.assertEqual(Router().revisions, {})

    def test_router_forwards_only_the_selected_per_model_revision(self):
        import laya.agent

        captured = []

        class FakeAgent:
            def __init__(self, repo, **kwargs):
                captured.append((repo, kwargs))

        with patch.object(laya.agent, "Agent", FakeAgent):
            router = Router(revision="default", revisions={"multilingual": "multi-sha"})
            router.load("english")
            router.load("multi")

        self.assertEqual(captured[0][1]["revision"], "default")
        self.assertEqual(captured[1][1]["revision"], "multi-sha")

    def test_router_omits_revision_when_none_is_configured(self):
        import laya.agent

        captured = {}

        class FakeAgent:
            def __init__(self, repo, **kwargs):
                captured.update(kwargs)

        with patch.object(laya.agent, "Agent", FakeAgent):
            Router().load("english")

        self.assertNotIn("revision", captured)

    def test_loaded_revisions_reports_resident_agents(self):
        router = Router()
        router.attach("english", SimpleNamespace(revision="sha-english"))
        self.assertEqual(router.loaded_revisions, {"english": "sha-english"})


class RouterDigestTests(unittest.TestCase):
    """`Router(sha256_digests=...)` is the per-checkpoint sibling of `revisions`."""

    def _capture(self):
        import laya.agent

        captured: list = []

        class FakeAgent:
            def __init__(self, repo, **kwargs):
                captured.append(kwargs)

        return patch.object(laya.agent, "Agent", FakeAgent), captured

    def test_router_normalises_digest_keys_like_revision_keys(self):
        router = Router(sha256_digests={"ml": {"w.bin": "a" * 64}, "typed": None})
        self.assertEqual(sorted(router.sha256_digests), ["multilingual", "typed-decisions"])
        self.assertEqual(Router().sha256_digests, {})

    def test_misspelled_model_fails_at_construction(self):
        with self.assertRaises(ValueError):
            Router(sha256_digests={"engligh": {"w.bin": "a" * 64}})

    def test_each_checkpoint_gets_its_own_map(self):
        capture, captured = self._capture()
        with capture:
            router = Router(sha256_digests={
                "english": {"model.safetensors": "a" * 64},
                "multilingual": {"model.safetensors": "b" * 64},
            })
            router.load("english")
            router.load("multilingual")
        self.assertEqual(captured[0]["expected_sha256"], {"model.safetensors": "a" * 64})
        self.assertEqual(captured[1]["expected_sha256"], {"model.safetensors": "b" * 64})

    def test_unlisted_checkpoint_is_left_to_the_environment(self):
        capture, captured = self._capture()
        with capture:
            Router(sha256_digests={"multilingual": {"model.safetensors": "b" * 64}}).load("english")
        self.assertNotIn("expected_sha256", captured[0])

    def test_none_entry_masks_the_environment_default_for_that_checkpoint(self):
        capture, captured = self._capture()
        with capture:
            Router(sha256_digests={"english": None}).load("english")
        # `{}` and "absent" differ inside verify_digests: only the absent one falls back to env.
        self.assertEqual(captured[0]["expected_sha256"], {})


class RouterDigestEndToEndTests(unittest.TestCase):
    """Two checkpoints, one shared relative filename, two different digests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dirs = {}
        self.digests = {}
        for name in ("english", "multilingual"):
            # `model.safetensors` without a valid config: a matching digest gets as far as the
            # config check, a mismatching one never leaves the digest check.
            path = os.path.join(self.tmp.name, name)
            os.makedirs(path)
            with open(os.path.join(path, "model.safetensors"), "wb") as f:
                f.write(b"weights of " + name.encode())
            self.dirs[name] = path
            self.digests[name] = hashlib.sha256(b"weights of " + name.encode()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def _router(self, digests, **kw):
        return Router(models=dict(self.dirs), sha256_digests=digests, **kw)

    def _assert_past_the_digest_gate(self, router, name):
        """The config check sits just after `verify_digests`, so reaching it proves the digest passed."""
        with self.assertRaises(FileNotFoundError) as cm:
            router.load(name)
        self.assertIn("rl_agent_config.json", str(cm.exception))

    def test_a_matching_digest_lets_the_checkpoint_through(self):
        router = self._router({"english": {"model.safetensors": self.digests["english"]}})
        self._assert_past_the_digest_gate(router, "english")

    def test_one_flat_map_cannot_cover_both_checkpoints(self):
        router = self._router({"english": {"model.safetensors": self.digests["english"]},
                               "multilingual": {"model.safetensors": self.digests["english"]}})
        self._assert_past_the_digest_gate(router, "english")
        with self.assertRaises(ValueError) as cm:
            router.load("multilingual")
        self.assertIn("SHA-256 mismatch", str(cm.exception))

    def test_per_checkpoint_maps_cover_both(self):
        router = self._router({"english": {"model.safetensors": self.digests["english"]},
                               "multilingual": {"model.safetensors": self.digests["multilingual"]}})
        self._assert_past_the_digest_gate(router, "english")
        self._assert_past_the_digest_gate(router, "multilingual")

    def test_environment_digest_still_applies_when_no_map_is_given(self):
        env = {"LAYA_SHA256_DIGESTS": json.dumps({"model.safetensors": self.digests["english"]})}
        router = self._router({})
        with patch.dict(os.environ, env):
            self._assert_past_the_digest_gate(router, "english")
            with self.assertRaises(ValueError):
                router.load("multilingual")

    def test_explicit_none_list_skips_the_environment_digest(self):
        env = {"LAYA_SHA256_DIGESTS": json.dumps({"model.safetensors": self.digests["english"]})}
        router = self._router({"multilingual": None})
        with patch.dict(os.environ, env):
            self._assert_past_the_digest_gate(router, "multilingual")


if __name__ == "__main__":
    unittest.main()
