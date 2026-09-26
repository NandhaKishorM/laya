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


class RouterAgentKwargsTests(unittest.TestCase):
    """`agent_kwargs` is how a Router user reaches the rest of `Agent`'s constructor.

    Before it, `Router.load` built every checkpoint with exactly four arguments, so
    `lang_temperatures`, `expected_sha256`, `fast` and `compile` could only be set by giving up the
    Router and hand-building an `Agent` -- which also left the per-language grouping in
    `Router.predict_batch` unable to fire for any agent the Router owned.
    """

    TABLE = {"de": {"temperature": [2.0, 2.0, 2.0], "temperature_by_options": {"choice:2": 3.0}}}

    @staticmethod
    def _fake_agent():
        """(module holding Agent, the fake, the list each build is appended to).

        `check_agent_kwargs` reads the option names out of whatever `laya.agent.Agent` currently
        is, so the stand-in has to carry the real signature or these tests would be refused before
        they ever reached a build.
        """
        import inspect

        import laya.agent

        captured = []

        class FakeAgent:
            def __init__(self, repo, **kwargs):
                captured.append((repo, kwargs))

            __init__.__signature__ = inspect.signature(Agent.__init__)

        return laya.agent, FakeAgent, captured

    def test_agent_kwargs_reach_the_agent_build(self):
        module, fake, captured = self._fake_agent()
        with patch.object(module, "Agent", fake):
            Router(agent_kwargs={"lang_temperatures": self.TABLE}).load("english")
        self.assertEqual(captured[0][1]["lang_temperatures"], self.TABLE)

    def test_applied_to_every_checkpoint_the_router_builds(self):
        module, fake, captured = self._fake_agent()
        with patch.object(module, "Agent", fake):
            router = Router(agent_kwargs={"expected_sha256": {"model.safetensors": "0" * 64}})
            router.load("english")
            router.load("multi")
        self.assertEqual(len(captured), 2)
        for _repo, kwargs in captured:
            self.assertEqual(kwargs["expected_sha256"], {"model.safetensors": "0" * 64})

    def test_passing_nothing_leaves_the_build_exactly_as_it_was(self):
        module, fake, captured = self._fake_agent()
        with patch.object(module, "Agent", fake):
            Router().load("english")
            Router(revision="sha").load("english")
        self.assertEqual(sorted(captured[0][1]), ["device", "subfolder", "token"])
        self.assertEqual(sorted(captured[1][1]), ["device", "revision", "subfolder", "token"])

    def test_a_router_value_is_never_shadowed_by_an_agent_kwarg(self):
        module, fake, captured = self._fake_agent()
        with patch.object(module, "Agent", fake):
            Router(device="cuda", revisions={"english": "sha"},
                   agent_kwargs={"lang_temperatures": self.TABLE}).load("english")
        self.assertEqual(captured[0][1]["device"], "cuda")
        self.assertEqual(captured[0][1]["revision"], "sha")

    def test_router_owned_names_are_refused(self):
        owned = ("model_id_or_path", "device", "token", "subfolder", "revision", "hooks",
                 "on_predict_start", "on_predict_end", "hooks_raise", "hooks_concurrent",
                 "hooks_timeout")
        for name in owned:
            with self.assertRaises(ValueError) as ctx:
                Router(agent_kwargs={name: "x"})
            self.assertIn(name, str(ctx.exception))
            self.assertIn("Router(...)", str(ctx.exception))

    def test_refusal_happens_at_construction_not_at_the_first_load(self):
        module, fake, captured = self._fake_agent()
        with patch.object(module, "Agent", fake):
            with self.assertRaises(ValueError):
                Router(agent_kwargs={"device": "cpu"}).load("english")
        self.assertEqual(captured, [])

    def test_unknown_option_is_refused_with_the_names_that_do_exist(self):
        with self.assertRaises(ValueError) as ctx:
            Router(agent_kwargs={"lang_tempertaures": self.TABLE})
        message = str(ctx.exception)
        self.assertIn("lang_tempertaures", message)
        # The typo is refused, and the accepted list carries the spelling the caller meant.
        self.assertIn("lang_temperatures", message)

    def test_accepted_names_are_read_from_agent_rather_than_copied_here(self):
        """The anti-drift check: this file must not grow its own list of checkpoint options."""
        import inspect

        from laya.router import _ROUTER_OWNED_AGENT_ARGS, check_agent_kwargs

        owned = set(_ROUTER_OWNED_AGENT_ARGS)
        accepted = set(inspect.signature(Agent.__init__).parameters) - {"self"} - owned
        self.assertTrue(accepted, "expected some Agent options to be reachable")
        for name in ("fast", "compile", "expected_sha256", "lang_temperatures"):
            self.assertIn(name, accepted)
        check_agent_kwargs({name: None for name in accepted})

    def test_the_callers_dict_is_copied_not_aliased(self):
        module, fake, captured = self._fake_agent()
        options = {"lang_temperatures": self.TABLE}
        with patch.object(module, "Agent", fake):
            router = Router(agent_kwargs=options)
            options["compile"] = True
            router.load("english")
        self.assertNotIn("compile", captured[0][1])

    def test_default_is_an_empty_build(self):
        self.assertEqual(Router().agent_kwargs, {})
        self.assertEqual(Router(agent_kwargs={}).agent_kwargs, {})


if __name__ == "__main__":
    unittest.main()
