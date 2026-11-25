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
from laya.router import Router, _digests_from_env, _merge_expected_digests  # noqa: E402

# A Windows environment variable cannot exceed 32 767 characters. That is the whole budget any
# test that delivers a value through `LAYA_SHA256_DIGESTS` has to work inside, and the reason the
# 200 000-character literal that used to be inlined below never reached the code under test on
# Windows: `patch.dict` raised `ValueError: the environment variable is longer than 32767
# characters` first, failing the suite there while passing on Linux and macOS, which have no
# comparable per-variable limit. `tests/test_serve.py` pins the same cap for pytest node ids.
# Files and request bodies have no per-variable limit, which is why the deep-nesting cases in the
# other suites pass far larger documents through those channels instead.
_ENV_VALUE_MAX = 32767


def _deep_json_within(budget):
    """The shallowest array nesting *this* interpreter answers with a `RecursionError`.

    Returns `None` when no document that fits `budget` characters reaches the scanner's limit.

    The depth cannot be predicted, which is the trap this function exists to avoid. `json` scans
    in C, and what bounds that recursion is not `sys.getrecursionlimit()`. Measured: depth 995 on
    CPython 3.11, where the Python limit does gate it; 9998 on 3.12 and 9999 on 3.13, where
    `sys.setrecursionlimit` has no effect on it at all; and past 100 000 on 3.14. So every fixed
    depth is wrong on some version in the matrix. A literal 1100 raised on 3.11 and *parsed* on
    3.12 and 3.13, where `verify_digests` then reported the list it got rather than the
    unparseable value these tests mean to hand it.

    Doubling upward rather than searching down from `budget` keeps the deepest parse attempted
    close to the real threshold, which matters on hosts with a smaller stack than the limit
    implies.
    """
    def raises(depth):
        try:
            json.loads("[" * depth + "]" * depth)
        except RecursionError:
            return True
        except ValueError:
            return False
        return False

    cap = budget // 2
    depth = 256
    while True:
        attempt = min(depth, cap)
        if raises(attempt):
            return "[" * attempt + "]" * attempt
        if attempt == cap:
            return None
        depth *= 2


_DEEP_JSON = _deep_json_within(_ENV_VALUE_MAX)
_NO_DEEP_JSON = (
    "no array nesting that fits a %d character environment variable reaches this interpreter's "
    "JSON scanner limit" % _ENV_VALUE_MAX
)

# `json.loads` made to report the one refusal that is not reachable through the environment on
# every interpreter, so the `except RecursionError` clauses stay covered even where `_DEEP_JSON`
# is None. Keyed on a sentinel rather than replacing `json.loads` outright: both code paths under
# test parse other documents, and a blanket stub would pass for reasons that have nothing to do
# with the clause being asserted.
_RECURSION_SENTINEL = "laya-test-force-recursion-error"


def _json_scanner_refuses_the_sentinel():
    real = json.loads

    def fake(value, *args, **kwargs):
        if isinstance(value, str) and value.strip() == _RECURSION_SENTINEL:
            raise RecursionError("maximum recursion depth exceeded while decoding a JSON object")
        return real(value, *args, **kwargs)

    return patch.object(json, "loads", fake)


class _StopLoad(Exception):
    """Sentinel raised by the fake snapshot_download once kwargs are captured."""


def _capturing_snapshot(captured):
    def fake_snapshot(repo_id, **kwargs):
        captured.update(kwargs)
        raise _StopLoad
    return fake_snapshot


class PublicApiTests(unittest.TestCase):
    def test_pinned_revisions_is_importable_from_the_package_root(self):
        """`from laya import PINNED_REVISIONS` is what the checkpoint-integrity guide tells
        operators to use, and laya-ts already exports its mirror from the package root."""
        import laya

        self.assertIn("PINNED_REVISIONS", laya.__all__)
        self.assertIs(laya.PINNED_REVISIONS, PINNED_REVISIONS)

    def test_pins_are_usable_revision_strings(self):
        # Not asserting a 40-hex shape: `resolve_revision` and the checkpoint-integrity guide both
        # say a revision may be a branch or a tag, so pinning that shape here would invent a policy
        # the library does not state. What must hold is that each pin is something the Hub can be
        # given -- a non-empty string with no surrounding whitespace.
        for repo, rev in PINNED_REVISIONS.items():
            self.assertIsInstance(rev, str, "%s pin is not a string" % repo)
            self.assertTrue(rev, "%s pin is empty" % repo)
            self.assertEqual(rev, rev.strip(), "%s pin has surrounding whitespace" % repo)


class ResolveRevisionTests(unittest.TestCase):
    def test_explicit_revision_is_returned(self):
        self.assertEqual(resolve_revision("convaiinnovations/laya", "abc123"), "abc123")

    def test_published_repos_keep_the_hub_default_without_an_explicit_pin(self):
        for repo in PINNED_REVISIONS:
            self.assertIsNone(resolve_revision(repo))

    def test_unknown_repo_keeps_the_hub_default(self):
        self.assertIsNone(resolve_revision("acme/custom-model"))
        self.assertIsNone(resolve_revision("acme/custom-model", ""))

    def test_reviewed_finds_the_pin_whatever_case_the_repo_is_spelled_in(self):
        """Hugging Face resolves repo ids case-insensitively, so the pin lookup must too.

        `laya.load("ConvaiInnovations/Laya")` under `LAYA_REVISION=reviewed` is the same
        repository as the table's lowercase key. An exact-match lookup called it unpinned and
        refused the load -- reporting a missing reviewed SHA for one that exists, the
        opposite of what this control is for.
        """
        repo = "convaiinnovations/laya"
        sha = PINNED_REVISIONS[repo]
        for spelling in (repo, repo.upper(), repo.title(), repo.capitalize()):
            with patch.dict(os.environ, {"LAYA_REVISION": "reviewed"}):
                self.assertEqual(resolve_revision(spelling), sha, spelling)
            # the explicit-argument form reaches the same branch
            self.assertEqual(resolve_revision(spelling, "reviewed"), sha, spelling)

    def test_reviewed_still_refuses_a_repo_the_table_does_not_have(self):
        for spelling in ("Acme/Some-Model", "ACME/SOME-MODEL"):
            with patch.dict(os.environ, {"LAYA_REVISION": "reviewed"}):
                with self.assertRaises(ValueError, msg=spelling):
                    resolve_revision(spelling)

    def test_the_refusal_blames_whichever_side_asked_for_reviewed(self):
        """One message for both callers sends the argument's caller to a variable they never set.

        `resolve_revision("acme/any", "reviewed")` is a documented call -- `Router(revision=)` and
        `laya.load(revision=)` both hand it a string -- and before it refused by naming
        ``LAYA_REVISION``, so an operator who grepped the deployment environment found nothing and
        had no way to tell which of the two levers was misconfigured.
        """
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as from_argument:
                resolve_revision("acme/custom-checkpoint", "reviewed")
        text = str(from_argument.exception)
        self.assertIn("revision=", text)
        self.assertNotIn("LAYA_REVISION", text)
        self.assertIn("acme/custom-checkpoint", text)
        self.assertIn("PINNED_REVISIONS", text)

    def test_the_refusal_still_blames_the_environment_when_it_set_the_pin(self):
        """The deployment's wording is unchanged: `tests/test_download.py` asserts on it, and the
        env fallback is the path the container guides (`LAYA_REVISION=reviewed laya-serve`) use."""
        with patch.dict(os.environ, {"LAYA_REVISION": "reviewed"}, clear=True):
            with self.assertRaises(ValueError) as from_environment:
                resolve_revision("acme/custom-checkpoint")
        text = str(from_environment.exception)
        self.assertIn("LAYA_REVISION", text)
        self.assertNotIn("revision=", text)

    def test_an_explicit_reviewed_argument_blames_the_argument_not_the_environment(self):
        """Both sides can be live at once, and the argument wins the `or`. So the message must
        name the argument even when ``LAYA_REVISION`` holds a perfectly good value -- unsetting the
        variable would not change what this call does."""
        with patch.dict(os.environ, {"LAYA_REVISION": "a" * 40}, clear=True):
            with self.assertRaises(ValueError) as error:
                resolve_revision("acme/custom-checkpoint", "reviewed")
        self.assertIn("revision=", str(error.exception))
        self.assertNotIn("LAYA_REVISION", str(error.exception))


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
        for rel in ("../evil", "..", "a/../../evil", "\\..\\evil", "/absolute/evil", "C:\\absolute\\evil",
                    # drive-anchored but not ntpath-absolute: ntpath.join("C:\\dir", "a", "b:c") == "b:c",
                    # so on Windows these escape model_dir the same way the absolute ones do
                    "C:evil", "c:evil", "a/b:c", "C:..\\evil"):
            with self.assertRaises(ValueError, msg=rel):
                verify_digests(self.dir, {rel: self.digest})

    def test_digest_map_can_come_from_environment(self):
        with patch.dict(os.environ, {"LAYA_SHA256_DIGESTS": json.dumps({"weights.bin": self.digest})}):
            verify_digests(self.dir)

    def test_an_unreadable_environment_map_is_a_value_error(self):
        # `json` refuses a document two ways: a `ValueError` for text that is not JSON, and -- for a
        # nesting depth past the scanner's own stack -- a `RecursionError`, which is a `RuntimeError`
        # and so was never in the `except` here. The second escaped `load()` as an interpreter
        # traceback instead of the refusal this function's docstring names, which is what
        # `laya/_eval_policy.py`, `laya/evidence.py` and `laya/cli.py` already refuse to let a
        # caller's JSON do.
        for bad in ("{not json",) + ((), (_DEEP_JSON,))[_DEEP_JSON is not None]:
            with self.subTest(value=bad[:20] + ("..." if len(bad) > 20 else "")):
                with patch.dict(os.environ, {"LAYA_SHA256_DIGESTS": bad}):
                    with self.assertRaises(ValueError) as cm:
                        verify_digests(self.dir)
                self.assertIn("LAYA_SHA256_DIGESTS", str(cm.exception))

    def test_a_scanner_recursion_error_is_reported_as_this_variables_refusal(self):
        """The `RecursionError` clause, asserted without depending on a reachable depth.

        `_DEEP_JSON` is the honest end-to-end case and it runs wherever the interpreter's scanner
        limit is reachable inside an environment variable, but that is not every version: on
        CPython 3.14 no document under 32 767 characters gets there. Without this test the clause
        would simply stop being covered on such a version, silently and with the suite green,
        which is the failure mode that let the original defect ship.
        """
        with _json_scanner_refuses_the_sentinel():
            with patch.dict(os.environ, {"LAYA_SHA256_DIGESTS": _RECURSION_SENTINEL}):
                with self.assertRaises(ValueError) as cm:
                    verify_digests(self.dir)
        self.assertIn("LAYA_SHA256_DIGESTS", str(cm.exception))

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

    def test_nested_environment_pins_both_checkpoints(self):
        """The shape a server with two resident checkpoints has to write.

        Before this, `LAYA_SHA256_DIGESTS` was read only by `laya.revisions`, which treats every
        key as a path -- so a model-named map failed with `cannot verify 'english': no such file`
        on the *first* checkpoint and the flat map refused the second one.
        """
        with self._env({"english": {"model.safetensors": self.digests["english"]},
                        "multilingual": {"model.safetensors": self.digests["multilingual"]}}):
            router = self._router({})
            self._assert_past_the_digest_gate(router, "english")
            self._assert_past_the_digest_gate(router, "multilingual")

    def test_nested_environment_fails_only_the_checkpoint_it_calls_out(self):
        with self._env({"english": {"model.safetensors": self.digests["english"]},
                        "multilingual": {"model.safetensors": "0" * 64}}):
            router = self._router({})
            self._assert_past_the_digest_gate(router, "english")
            with self.assertRaises(ValueError) as cm:
                router.load("multilingual")
            self.assertIn("SHA-256 mismatch", str(cm.exception))

    def test_nested_environment_leaves_an_unnamed_checkpoint_unpinned(self):
        # `multilingual` is a resident model that the map does not name. It has to be loaded
        # unverified, not handed the model-named map as if "english" were one of its files.
        with self._env({"english": {"model.safetensors": self.digests["english"]}}):
            router = self._router({})
            self._assert_past_the_digest_gate(router, "english")
            self._assert_past_the_digest_gate(router, "multilingual")

    def _env(self, per_model):
        return patch.dict(os.environ, {"LAYA_SHA256_DIGESTS": json.dumps(per_model)})

    # An `agent_kwargs` map is the third channel, and neither of the other two may cancel it:
    # `expected_sha256` is reachable there on purpose, and a caller that pins a file is asking
    # for it to be verified on every checkpoint this Router builds.
    WRONG = {"model.safetensors": "0" * 64}

    def test_agent_kwargs_still_refuses_a_checkpoint_the_environment_map_skips(self):
        with self._env({"english": {"model.safetensors": self.digests["english"]}}):
            router = self._router({}, agent_kwargs={"expected_sha256": self.WRONG})
            with self.assertRaises(ValueError) as cm:
                router.load("multilingual")
        self.assertIn("SHA-256 mismatch", str(cm.exception))

    def test_agent_kwargs_still_refuses_a_checkpoint_listed_as_none(self):
        router = self._router({"multilingual": None}, agent_kwargs={"expected_sha256": self.WRONG})
        with self.assertRaises(ValueError) as cm:
            router.load("multilingual")
        self.assertIn("SHA-256 mismatch", str(cm.exception))

    def test_a_correct_agent_kwargs_digest_loads_under_the_same_configuration(self):
        with self._env({"english": {"model.safetensors": self.digests["english"]}}):
            router = self._router({}, agent_kwargs={
                "expected_sha256": {"model.safetensors": self.digests["multilingual"]}})
            self._assert_past_the_digest_gate(router, "multilingual")


class RouterEnvDigestTests(unittest.TestCase):
    """How `Router` reads the two shapes of `LAYA_SHA256_DIGESTS`.

    `serve.build_router()` builds its Router from the environment and nothing else, so the
    environment is the only channel a container, a compose stack or a `services.laya-serve`
    host has. These pin down which shape `Router` takes over and which it leaves to
    `laya.revisions`, without weights: the `Agent` builds are captured, not run.
    """

    ENGLISH = {"model.safetensors": "a" * 64}
    MULTI = {"model.safetensors": "b" * 64}

    def _capture(self):
        import laya.agent

        captured: list = []

        class FakeAgent:
            def __init__(self, repo, **kwargs):
                captured.append(kwargs)

        return patch.object(laya.agent, "Agent", FakeAgent), captured

    def _env(self, value):
        return patch.dict(os.environ, {"LAYA_SHA256_DIGESTS": value})

    def test_flat_map_is_left_to_revisions(self):
        with self._env(json.dumps({"model.safetensors": "a" * 64})):
            router = Router()
            self.assertEqual(router.sha256_digests, {})
            capture, captured = self._capture()
            with capture:
                router.load("english")
        self.assertNotIn("expected_sha256", captured[0])

    def test_nested_map_is_split_per_checkpoint(self):
        with self._env(json.dumps({"english": self.ENGLISH, "multilingual": self.MULTI})):
            capture, captured = self._capture()
            with capture:
                router = Router()
                self.assertEqual(sorted(router.sha256_digests),
                                 ["english", "multilingual", "typed-decisions"])
                for name in ("english", "multilingual", "typed-decisions"):
                    router.load(name)
        self.assertEqual(captured[0]["expected_sha256"], self.ENGLISH)
        self.assertEqual(captured[1]["expected_sha256"], self.MULTI)
        # Named by neither the map nor the caller: explicitly unpinned, so `Agent` does not
        # fall back to an environment value keyed by model names.
        self.assertEqual(captured[2]["expected_sha256"], {})

    def test_nested_map_keys_are_normalised(self):
        with self._env(json.dumps({"en": self.ENGLISH})):
            self.assertEqual(Router().sha256_digests["english"], self.ENGLISH)

    def test_argument_wins_for_the_checkpoint_it_names(self):
        with self._env(json.dumps({"english": self.ENGLISH, "multilingual": self.MULTI})):
            router = Router(sha256_digests={"english": {"model.safetensors": "c" * 64}})
            self.assertEqual(router.sha256_digests["english"], {"model.safetensors": "c" * 64})
            self.assertEqual(router.sha256_digests["multilingual"], self.MULTI)

    def test_misspelled_key_in_the_environment_fails_at_construction(self):
        with self._env(json.dumps({"engligh": self.ENGLISH})):
            with self.assertRaises(ValueError):
                Router()

    def test_mixed_artifact_and_model_keys_fail_rather_than_guess(self):
        env = json.dumps({"model.safetensors": "a" * 64, "english": self.ENGLISH})
        with self._env(env):
            with self.assertRaises(ValueError) as cm:
                Router()
        self.assertIn("LAYA_SHA256_DIGESTS", str(cm.exception))

    def test_unparseable_environment_is_not_this_layers_error(self):
        # `laya.revisions.verify_digests` owns the message a malformed value gets; a Router
        # that repeated it would be a second copy of a rule it does not enforce.
        #
        # The last value is the shape `except ValueError` here did not cover: `json` answers a
        # nesting depth past its own stack with a `RecursionError`, a `RuntimeError`, so building a
        # Router on it crashed instead of naming nothing -- the one unparseable input this layer
        # promises to hand on rather than report.
        values = ("", "   ", "{not json", "[]", '"digests"', "{}")
        for value in values + ((), (_DEEP_JSON,))[_DEEP_JSON is not None]:
            with self._env(value):
                self.assertEqual(Router().sha256_digests, {}, value)

    def test_a_scanner_recursion_error_names_nothing_rather_than_raising(self):
        """Same clause on this side of the boundary, equally independent of a reachable depth.

        A Router that let the `RecursionError` out would crash a caller who only misconfigured a
        variable, so this is the behaviour the docstring above promises: name nothing, and leave
        the wording to `laya.revisions`.
        """
        with _json_scanner_refuses_the_sentinel():
            with self._env(_RECURSION_SENTINEL):
                self.assertEqual(Router().sha256_digests, {})


class RouterDigestPrecedenceTests(unittest.TestCase):
    """What reaches `Agent` when more than one channel pins the same checkpoint.

    `expected_sha256` is deliberately *not* router-owned -- `test_accepted_names_are_read_from_
    agent_rather_than_copied_here` asserts that, and the class docstring advertises
    `Router(agent_kwargs={"expected_sha256": ...})` -- so a Router can hold digests from three
    places at once: that argument, `sha256_digests`, and a nested `LAYA_SHA256_DIGESTS`. These
    pin down that none of them silently turns another one off.
    """

    SHARED = {"tokenizer.json": "a" * 64}
    PER_MODEL = {"model.safetensors": "b" * 64}

    def setUp(self):
        """Start every test from an unset `LAYA_SHA256_DIGESTS`.

        Several checks here assert that *no* `expected_sha256` reaches `Agent`, which is only true of
        an unconfigured process. A developer machine or CI runner that exports a **nested** variable --
        the shape this feature exists to support -- seeds a `{}` placeholder for every model, so
        `expected_sha256={}` is passed and those assertions fail with no code change. Tests that want
        the variable set use `_env`.
        """
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        os.environ.pop("LAYA_SHA256_DIGESTS", None)
        self.addCleanup(patcher.stop)

    def _capture(self):
        """Agent stand-in carrying the real signature, so `check_agent_kwargs` still applies."""
        import inspect

        import laya.agent

        captured: list = []

        class FakeAgent:
            def __init__(self, repo, **kwargs):
                captured.append(kwargs)

            __init__.__signature__ = inspect.signature(Agent.__init__)

        return patch.object(laya.agent, "Agent", FakeAgent), captured

    def _env(self, per_model):
        return patch.dict(os.environ, {"LAYA_SHA256_DIGESTS": json.dumps(per_model)})

    def _load(self, name, **router_kwargs):
        capture, captured = self._capture()
        with capture:
            Router(**router_kwargs).load(name)
        return captured[0]

    def test_a_nested_environment_pin_does_not_unverify_the_other_checkpoints(self):
        # The variable names `english` only. `multilingual` is not mentioned by it, which is not
        # the same as being asked for unverified: the caller's own map still has to apply.
        with self._env({"english": {"model.safetensors": "e" * 64}}):
            kwargs = self._load("multilingual", agent_kwargs={"expected_sha256": self.SHARED})
        self.assertEqual(kwargs["expected_sha256"], self.SHARED)

    def test_a_none_entry_does_not_discard_the_agent_kwargs_map(self):
        kwargs = self._load("english", sha256_digests={"english": None},
                            agent_kwargs={"expected_sha256": self.SHARED})
        self.assertEqual(kwargs["expected_sha256"], self.SHARED)

    def test_the_two_maps_are_merged_file_by_file(self):
        kwargs = self._load("english", sha256_digests={"english": self.PER_MODEL},
                            agent_kwargs={"expected_sha256": self.SHARED})
        self.assertEqual(kwargs["expected_sha256"], dict(self.SHARED, **self.PER_MODEL))

    def test_a_per_checkpoint_pin_wins_over_a_shared_one_for_the_same_file(self):
        """The two channels are not equally specific, so the narrower one decides.

        `agent_kwargs["expected_sha256"]` reaches every checkpoint, and `model.safetensors` is the
        one name every checkpoint uses for a *different* file -- so a shared entry for it cannot be
        a correct claim about all of them at once. An earlier version of this fix refused the
        overlap as a contradiction, which broke the ordinary shape it appears in: a shared pin plus
        a per-checkpoint override, which loaded correctly before any of this existed.
        """
        capture, captured = self._capture()
        with capture:
            router = Router(agent_kwargs={"expected_sha256": {"model.safetensors": "c" * 64}},
                            sha256_digests={"multilingual": {"model.safetensors": "b" * 64}})
            router.load("english")
            router.load("multilingual")
        self.assertEqual(captured[0]["expected_sha256"], {"model.safetensors": "c" * 64},
                         "the shared pin applies where nothing overrides it")
        self.assertEqual(captured[1]["expected_sha256"], {"model.safetensors": "b" * 64},
                         "the per-checkpoint pin wins for the file both name")

    def test_an_environment_pin_overrides_the_shared_one_per_checkpoint(self):
        """Same precedence, reached through the variable, and every checkpoint stays loadable.

        This is the shape that matters in a deployment: the variable pins each checkpoint's own
        weight file and the caller passes one shared map. Refusing it would have stopped the server
        from serving the checkpoint the two configurations do not actually disagree about.
        """
        capture, captured = self._capture()
        shared = {"model.safetensors": "c" * 64}
        with self._env({"english": {"model.safetensors": "b" * 64}}), capture:
            router = Router(agent_kwargs={"expected_sha256": shared})
            router.load("multilingual")
            router.load("english")
        self.assertEqual(captured[0]["expected_sha256"], shared,
                         "the checkpoint the variable does not name keeps the shared pin")
        self.assertEqual(captured[1]["expected_sha256"], {"model.safetensors": "b" * 64},
                         "the checkpoint the variable names uses its own digest")

    def test_files_only_one_channel_names_are_still_merged(self):
        """Precedence applies per file, not per map: a file only one channel names still counts.

        This is the half of the merge that is worth having -- the caller pins the tokenizer, the
        deployment pins the weights, and the load verifies both. Ranking whole maps against each
        other, as the code did before this fix, verified only one of them.
        """
        capture, captured = self._capture()
        with self._env({"english": {"model.safetensors": "b" * 64}}), capture:
            router = Router(agent_kwargs={"expected_sha256": {"tokenizer.json": "d" * 64}})
            router.load("english")
        self.assertEqual(captured[0]["expected_sha256"],
                         {"tokenizer.json": "d" * 64, "model.safetensors": "b" * 64},
                         "both channels' files must reach verify_digests")

    def test_digests_added_into_an_environment_placeholder_in_place_are_verified(self):
        """Filling in the placeholder the variable left is a pin, and has to be honoured.

        `_digests_from_env` leaves an empty entry for every checkpoint a nested
        `LAYA_SHA256_DIGESTS` does not name, tagged as "not something anyone asked for" so it cannot
        cancel an `agent_kwargs` map. But `sha256_digests` is public: an operator may add digests to
        that entry in place. Treating it as unpinned for ever dropped them silently -- and only when
        an `agent_kwargs` map happened to be present, so the behaviour was config-dependent.
        """
        capture, captured = self._capture()
        with self._env({"english": {"model.safetensors": "b" * 64}}), capture:
            router = Router(agent_kwargs={"expected_sha256": {"tokenizer.json": "d" * 64}})
            router.sha256_digests["multilingual"]["model.safetensors"] = "e" * 64
            router.load("multilingual")
        self.assertEqual(captured[0]["expected_sha256"],
                         {"tokenizer.json": "d" * 64, "model.safetensors": "e" * 64},
                         "a digest added into the placeholder must reach verify_digests")

    def test_an_untouched_environment_placeholder_still_reaches_agent_as_an_empty_map(self):
        """A placeholder must arrive as `{}`, NOT be omitted -- omitting it is a refusal.

        This looks backwards until you run it: with a *nested* `LAYA_SHA256_DIGESTS` and no
        `expected_sha256` kwarg at all, `verify_digests` reads that nested map as an artifact map and
        raises `cannot verify 'english': no such file` -- on a checkpoint the variable never named.
        The empty map is what masks that fallback. Passing nothing is only correct when no channel
        said anything.
        """
        capture, captured = self._capture()
        with self._env({"english": {"model.safetensors": "b" * 64}}), capture:
            Router().load("multilingual")
        self.assertEqual(captured[0].get("expected_sha256"), {},
                         "the placeholder must mask the nested-variable fallback")

        # And with nothing configured anywhere, no kwarg is passed at all (#332).
        capture2, captured2 = self._capture()
        with capture2:
            Router().load("multilingual")
        self.assertNotIn("expected_sha256", captured2[0])

    def test_an_explicit_none_entry_masks_a_flat_environment_map(self):
        """`sha256_digests={"english": None}` means "load this one unverified", and has to hold.

        A *flat* `LAYA_SHA256_DIGESTS` is applied by `verify_digests` itself, for every checkpoint.
        Saying "not this one" therefore cannot be expressed by passing nothing -- that is exactly
        what lets the flat map through. It has to arrive as an empty map.
        """
        flat = {"model.safetensors": "f" * 64}
        for entry in (None, {}):
            capture, captured = self._capture()
            with self._env(flat), capture:
                Router(sha256_digests={"english": entry}).load("english")
            self.assertEqual(captured[0].get("expected_sha256"), {},
                             "an explicit %r entry must mask the flat variable" % (entry,))
        # And with no entry at all the flat map is left to apply.
        capture, captured = self._capture()
        with self._env(flat), capture:
            Router().load("english")
        self.assertNotIn("expected_sha256", captured[0])

    def test_a_pin_assigned_after_construction_wins_for_its_own_file(self):
        """`sha256_digests` is public and mutable, so a pin assigned later has to take effect."""
        capture, captured = self._capture()
        with capture:
            router = Router(agent_kwargs={"expected_sha256": {"model.safetensors": "c" * 64}})
            router.sha256_digests["english"] = {"model.safetensors": "b" * 64}
            router.load("english")
        self.assertEqual(captured[0]["expected_sha256"], {"model.safetensors": "b" * 64})

    def test_a_pin_added_after_construction_is_verified(self):
        # `sha256_digests` is public and mutable, so what it holds at load time is what has to be
        # verified. Deciding "somebody asked for this checkpoint" from a key set frozen in
        # `__init__` dropped a later per-checkpoint pin -- and only when `agent_kwargs` happened
        # to carry a map, which is the one configuration where the drop mattered.
        capture, captured = self._capture()
        with capture:
            router = Router(agent_kwargs={"expected_sha256": self.SHARED})
            router.sha256_digests["english"] = self.PER_MODEL
            router.load("english")
        self.assertEqual(captured[0]["expected_sha256"], dict(self.SHARED, **self.PER_MODEL))

    def test_an_empty_map_assigned_after_construction_replaces_the_environment_entry(self):
        """Assigning `{}` over an environment entry drops that entry's files.

        What this pins is the *assignment*, not a placeholder/explicit distinction: `_digest_entry`
        collapses a caller's `{}`, a caller's `None` and the placeholder to the same `{}` on purpose,
        so there is nothing here that could tell them apart. The behaviour worth pinning is that the
        entry the variable had put there is gone, which is why the shared map alone is the answer.
        """
        capture, captured = self._capture()
        with self._env({"english": dict(self.PER_MODEL)}), capture:
            router = Router(agent_kwargs={"expected_sha256": dict(self.SHARED)})
            router.sha256_digests["english"] = {}
            router.load("english")
        self.assertEqual(captured[0]["expected_sha256"], self.SHARED,
                         "the assigned empty map must replace the variable's entry")

    def test_the_environment_entries_still_read_as_plain_maps(self):
        # How the placeholder is marked is this module's business: `sha256_digests` is public, and
        # `RouterEnvDigestTests` reads it as plain dicts.
        with self._env({"english": {"model.safetensors": "e" * 64}}):
            digests = Router().sha256_digests
        self.assertEqual(digests, {"english": {"model.safetensors": "e" * 64},
                                   "multilingual": {}, "typed-decisions": {}})

    def test_the_same_digest_from_both_channels_is_not_a_conflict(self):
        both = {"model.safetensors": "b" * 64}
        kwargs = self._load("english", sha256_digests={"english": both},
                            agent_kwargs={"expected_sha256": {"model.safetensors": "B" * 64}})
        self.assertEqual({k: v.lower() for k, v in kwargs["expected_sha256"].items()}, both)

    def test_nothing_configured_still_passes_no_map_at_all(self):
        # #332's opt-in guarantee: an unconfigured Router must leave `verify_digests` its own
        # `LAYA_SHA256_DIGESTS` fallback, so an existing HF_HUB_OFFLINE=1 cache loads as before.
        self.assertNotIn("expected_sha256", self._load("english"))


class RouterRevisionFallbackTests(unittest.TestCase):
    """A `revisions` entry that names no commit must not cost the caller the pin it did ask for.

    `revisions` is typed `Optional[Dict[str, Optional[str]]]`, so `None` values are legal, and
    `{"english": os.environ.get("EN_SHA")}` produces one whenever that variable is unset.
    """

    def _capture(self):
        import inspect

        import laya.agent

        captured: list = []

        class FakeAgent:
            def __init__(self, repo, **kwargs):
                captured.append(kwargs)

            __init__.__signature__ = inspect.signature(Agent.__init__)

        return patch.object(laya.agent, "Agent", FakeAgent), captured

    def _load(self, name="english", env="", **router_kwargs):
        capture, captured = self._capture()
        with patch.dict(os.environ, {"LAYA_REVISION": env}), capture:
            Router(**router_kwargs).load(name)
        return captured[0]

    def _effective(self, kwargs, env=""):
        """What `Agent` would download: `resolve_revision` applied to the kwarg it was handed."""
        with patch.dict(os.environ, {"LAYA_REVISION": env}):
            return resolve_revision("convaiinnovations/laya", kwargs.get("revision"))

    def test_a_none_entry_inherits_the_router_wide_revision(self):
        kwargs = self._load(revision="aaa", revisions={"english": None})
        self.assertEqual(kwargs["revision"], "aaa")

    def test_a_none_entry_does_not_hand_the_checkpoint_to_the_environment(self):
        kwargs = self._load(revision="aaa", revisions={"english": None}, env="bbb")
        self.assertEqual(self._effective(kwargs, env="bbb"), "aaa")

    def test_a_blank_entry_is_read_the_same_way(self):
        for blank in ("", "   "):
            kwargs = self._load(revision="aaa", revisions={"english": blank}, env="bbb")
            self.assertEqual(self._effective(kwargs, env="bbb"), "aaa", blank)

    def test_a_blank_router_wide_revision_is_not_a_pin(self):
        self.assertNotIn("revision", self._load(revision="   "))

    def test_an_explicit_per_model_revision_still_wins(self):
        kwargs = self._load("multilingual", revision="aaa",
                            revisions={"multilingual": "ccc", "english": None}, env="bbb")
        self.assertEqual(kwargs["revision"], "ccc")

    def test_an_unconfigured_router_still_leaves_the_revision_to_the_environment(self):
        # The other half of #332: nothing asked for means the deployment's `LAYA_REVISION` (or
        # huggingface_hub's default) applies, exactly as it does for a hand-built `Agent`.
        kwargs = self._load(env="bbb")
        self.assertNotIn("revision", kwargs)
        self.assertEqual(self._effective(kwargs, env="bbb"), "bbb")


class ReviewGapTests(unittest.TestCase):
    """Guards an adversarial review found unpinned: each mutant below survived every lane."""

    def test_a_non_dict_per_checkpoint_entry_is_handed_on_untouched(self):
        """`_merge_expected_digests` hands a non-dict on so `verify_digests` owns the message.

        Dropping the guard let a list of pairs be `dict.update`-ed into a map and silently accepted
        as digests. Both shapes still fail closed, but only one of them fails in the right words.
        """
        self.assertEqual(_merge_expected_digests({"a": "1" * 64}, "nope"), "nope")
        self.assertEqual(_merge_expected_digests({"a": "1" * 64}, [("a", "1" * 64)]),
                         [("a", "1" * 64)])

    def test_a_non_dict_shared_map_is_handed_on_untouched(self):
        """The same for the shared half, which had the same missing coverage."""
        self.assertEqual(_merge_expected_digests("nope", None), "nope")
        self.assertEqual(_merge_expected_digests(["a"], None), ["a"])

    def test_a_blank_revision_suppresses_the_environment_through_agent(self):
        """`(revision or env).strip()`: a truthy-but-blank argument wins the `or`, then strips away.

        The Router drops a blank before it reaches here, and that is tested. `Agent`/`laya.load` do
        not, so a config line that ends up whitespace silently loses the deployment's pin and gets
        huggingface_hub's default. Undocumented and untested before: the mutant that makes a blank
        behave like an absent value survived every lane. Pinned as the behaviour that exists, so a
        change to it is a decision rather than an accident.
        """
        with patch.dict(os.environ, {"LAYA_REVISION": "deadbeef"}, clear=False):
            self.assertEqual(resolve_revision("some/repo", None), "deadbeef")
            self.assertEqual(resolve_revision("some/repo", ""), "deadbeef")
            self.assertIsNone(resolve_revision("some/repo", "   "))
            self.assertIsNone(resolve_revision("some/repo", "\t"))
            self.assertEqual(resolve_revision("some/repo", "  aaa  "), "aaa")

    def test_a_flat_environment_map_is_masked_by_any_other_pin(self):
        """The documented exception, asserted so the class docstring cannot drift from it again.

        `verify_digests` reads a flat `LAYA_SHA256_DIGESTS` only `if expected is None`, so anything
        reaching `Agent` as `expected_sha256` means the flat variable is not consulted. An earlier
        revision of the class docstring said the two were "merged file by file", which is true of
        the per-checkpoint shape and false of this one.
        """
        with patch.dict(os.environ,
                             {"LAYA_SHA256_DIGESTS": json.dumps({"model.safetensors": "a" * 64})},
                             clear=False):
            self.assertEqual(_digests_from_env(["english"]), {})   # flat: not a layer here
            merged = _merge_expected_digests({"tokenizer.json": "b" * 64}, None)
            self.assertEqual(merged, {"tokenizer.json": "b" * 64})
            self.assertNotIn("model.safetensors", merged)


if __name__ == "__main__":
    unittest.main()
