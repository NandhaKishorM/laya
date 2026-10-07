"""Checkpoint download regression tests; tiny local weights, no network required.

Also pins the `LAYA_REVISION` lever of `laya/revisions.py`: which revision reaches `snapshot_download`,
what the Agent and the Router report back, and the two loads that must not see it (a local
directory, and `common.build_model`'s training-time base encoder).

Run: python tests/test_download.py
"""
import inspect
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from huggingface_hub.utils import filter_repo_objects  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from torch._dynamo.eval_frame import OptimizedModule  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import Agent, load  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from laya.revisions import PINNED_REVISIONS  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402


class _NoRuntime:
    """Stand-in for the onnxruntime module. `ONNXAgent.__init__` binds the name before it
    touches the Hub, and the download asserted on below happens well before a session is
    built, so nothing here is ever called."""


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls.tmp.name) / "repo"
        cls.repo.mkdir()
        config = BertConfig(vocab_size=6, hidden_size=64, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=128)
        config.save_pretrained(cls.repo / "encoder")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(
                {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]",
        )
        tokenizer.save_pretrained(cls.repo / "tokenizer")
        model = DecisionModel(BertModel(config), head_layers=0)
        save_file(model.state_dict(), cls.repo / "model.safetensors")
        (cls.repo / "rl_agent_config.json").write_text(json.dumps({
            "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
            "max_len": 64, "head_max_len": 32,
        }))
        # Hub repository paths use '/' even when the fixture lives on Windows.
        cls.runtime_files = {p.relative_to(cls.repo).as_posix()
                             for p in cls.repo.rglob("*") if p.is_file()}
        for subfolder in ("multilingual", "typed-decisions", "variants/english"):
            for filename in cls.runtime_files:
                target = cls.repo / subfolder / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cls.repo / filename, target)
        (cls.repo / "README.md").write_text("An unrelated model card")
        (cls.repo / "eval").mkdir()
        (cls.repo / "eval" / "results.json").write_text("{}")
        cls.questions = {"q": {"type": "choice", "instructions": "Pick one",
                               "criteria": ["yes", "no"]}}
        cls.expected = load(str(cls.repo), device="cpu").predict("hello", cls.questions)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_download(self, repo_id, subfolder=None):
        # Use the Hub's real file-filter semantics, then load and run the selected files.
        # Only transport is replaced; tokenizer, weights, model construction and inference
        # use the same code as a real checkpoint download.
        with tempfile.TemporaryDirectory() as destination:
            downloaded = []

            def snapshot(repo_id, **kwargs):
                files = [p.relative_to(self.repo).as_posix()
                         for p in self.repo.rglob("*") if p.is_file()]
                selected = filter_repo_objects(files, allow_patterns=kwargs.get("allow_patterns"),
                                               ignore_patterns=kwargs.get("ignore_patterns"))
                for filename in selected:
                    target = Path(destination) / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.repo / filename, target)
                    downloaded.append(filename)
                return destination

            with patch("huggingface_hub.snapshot_download", side_effect=snapshot) as download:
                agent = load(repo_id, device="cpu", subfolder=subfolder, token="test-token")
            self.assertEqual(download.call_count, 1)
            self.assertEqual(download.call_args.args[0], repo_id)
            self.assertEqual(download.call_args.kwargs["token"], "test-token")
            self.assertEqual(agent.predict("hello", self.questions), self.expected)
            prefix = subfolder + "/" if subfolder else ""
            self.assertEqual(set(downloaded), {prefix + name for name in self.runtime_files})

    def test_default_english_does_not_download_sibling_checkpoints(self):
        self.check_download("convaiinnovations/laya")

    def test_custom_root_checkpoint(self):
        self.check_download("test/custom-model")

    def test_each_subfolder_loads_independently(self):
        for subfolder in ("multilingual", "typed-decisions", "variants/english"):
            with self.subTest(subfolder=subfolder):
                self.check_download("test/bundled-models", subfolder)

    def test_a_checkpoint_name_or_alias_reaches_the_same_download_as_a_repo_id(self):
        # #780: `Router` resolved `english` / `typed-decisions` / `ml`, but `load()` forwarded the
        # same word to the Hub as a repo id, so `laya.load("typed-decisions")` 404'd. One registry
        # now serves both, and this asserts the transport the name produces is the one the spelled
        # -out `repo_id` + `subfolder` pair already produces.
        for name, repo_id, subfolder in [("english", "convaiinnovations/laya", None),
                                         ("laya", "convaiinnovations/laya", None),
                                         ("typed-decisions", "convaiinnovations/laya", "typed-decisions"),
                                         ("typed", "convaiinnovations/laya", "typed-decisions"),
                                         ("ml", "convaiinnovations/laya", "multilingual"),
                                         ("ML", "convaiinnovations/laya", "multilingual")]:
            with self.subTest(name=name):
                with patch("huggingface_hub.snapshot_download",
                           return_value=str(self.repo)) as download:
                    agent = load(name, device="cpu", token="test-token")
                self.assertEqual(download.call_args.args[0], repo_id)
                patterns = download.call_args.kwargs["allow_patterns"]
                prefix = subfolder + "/" if subfolder else ""
                self.assertIn(prefix + "model.safetensors", patterns)
                self.assertEqual(agent.predict("hello", self.questions), self.expected)

    def test_local_paths_do_not_download(self):
        with patch("huggingface_hub.snapshot_download") as download:
            for subfolder in (None, "multilingual", "variants/english"):
                with self.subTest(subfolder=subfolder):
                    agent = load(str(self.repo), device="cpu", subfolder=subfolder)
                    self.assertEqual(agent.predict("hello", self.questions), self.expected)
            download.assert_not_called()

    def test_loading_skips_initialization_and_preserves_weights(self):
        weights = load_file(self.repo / "model.safetensors")
        for bundled_encoder in (True, False):
            with self.subTest(bundled_encoder=bundled_encoder), tempfile.TemporaryDirectory() as destination:
                path = Path(destination)
                for filename in self.runtime_files:
                    if not bundled_encoder and filename.startswith("encoder/"):
                        continue
                    target = path / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.repo / filename, target)
                cfg_path = path / "rl_agent_config.json"
                cfg = json.loads(cfg_path.read_text())
                cfg["encoder"] = str(self.repo / "encoder")
                cfg_path.write_text(json.dumps(cfg))
                rng = torch.get_rng_state()
                with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("Unused base weights")):
                    agent = load(str(path), device="cpu")
                self.assertTrue(torch.equal(torch.get_rng_state(), rng), "Loading initialized random weights")
                for name, value in agent.model.state_dict().items():
                    torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
                self.assertEqual(agent.predict("hello", self.questions), self.expected)

    # ------------------------------------------------------------------ LAYA_REVISION
    #
    # `PINNED_REVISIONS` published the reviewed commit of each published checkpoint and nothing in
    # `laya/` read it, while `LAYA_SHA256_DIGESTS` -- the other half of the same opt-in pair, in
    # the same module -- was already reachable from the environment. These replace only the Hub
    # transport, so the revision that reaches `snapshot_download`, the one the Agent reports back,
    # and the one a Router publishes in `loaded_revisions` (what `/health` serves) are all the real
    # thing. They live here rather than in `tests/test_revision_pinning.py` because this file's
    # subject is already "what the download call was given", and because this suite runs in CI
    # (`ci.yml:107`, `release.yml:47`) while that one is in no workflow.

    def hub_load(self, repo, env=None, **load_kwargs):
        """`load(repo)` with the Hub transport replaced. Returns (kwargs sent, agent, error)."""
        with patch.dict(os.environ), patch("huggingface_hub.snapshot_download",
                                           return_value=str(self.repo)) as download:
            if env is None:
                os.environ.pop("LAYA_REVISION", None)
            else:
                os.environ["LAYA_REVISION"] = env
            try:
                agent = load(repo, device="cpu", **load_kwargs)
            except Exception as error:                # the cases below assert on it
                return download, None, error
            return download, agent, None

    def sent_revision(self, download):
        return (download.call_args.kwargs if download.call_args else {}).get("revision")

    def test_no_pin_asked_for_keeps_the_hub_default(self):
        download, agent, error = self.hub_load("convaiinnovations/laya")
        self.assertIsNone(error)
        self.assertNotIn("revision", download.call_args.kwargs)
        self.assertIsNone(agent.revision)
        self.assertEqual(agent.predict("hello", self.questions), self.expected)

    def test_environment_revision_reaches_the_download(self):
        download, agent, error = self.hub_load("convaiinnovations/laya", env="ab" * 20)
        self.assertIsNone(error)
        self.assertEqual(self.sent_revision(download), "ab" * 20)
        self.assertEqual(agent.revision, "ab" * 20)

    def test_reviewed_resolves_each_repositorys_own_pin(self):
        self.assertGreater(len(set(PINNED_REVISIONS.values())), 1,
                           "the table has one pin per repository, so this sweep is meaningful")
        for repo, sha in PINNED_REVISIONS.items():
            with self.subTest(repo=repo):
                download, agent, error = self.hub_load(repo, env="reviewed")
                self.assertIsNone(error)
                self.assertEqual(self.sent_revision(download), sha)
                self.assertEqual(agent.revision, sha)

    def test_reviewed_refuses_a_repository_with_no_pin(self):
        download, agent, error = self.hub_load("acme/custom-checkpoint", env="reviewed")
        self.assertIsInstance(error, ValueError)
        self.assertIn("acme/custom-checkpoint", str(error))
        self.assertIn("LAYA_REVISION", str(error))
        self.assertIsNone(agent)
        download.assert_not_called()

    def test_an_explicit_revision_outranks_the_environment(self):
        download, agent, error = self.hub_load("convaiinnovations/laya", env="reviewed",
                                               revision="cd" * 20)
        self.assertIsNone(error)
        self.assertEqual(self.sent_revision(download), "cd" * 20)
        self.assertNotEqual(agent.revision, PINNED_REVISIONS["convaiinnovations/laya"])

    def test_an_empty_or_blank_environment_value_is_not_a_pin(self):
        for env in ("", "   ", "\t"):
            with self.subTest(repr=repr(env)):
                download, agent, error = self.hub_load("convaiinnovations/laya", env=env)
                self.assertIsNone(error)
                self.assertNotIn("revision", download.call_args.kwargs)
                self.assertIsNone(agent.revision)

    def test_a_local_directory_never_consults_the_environment(self):
        download, agent, error = self.hub_load(str(self.repo), env="reviewed")
        self.assertIsNone(error)
        download.assert_not_called()
        self.assertIsNone(agent.revision)
        self.assertEqual(agent.predict("hello", self.questions), self.expected)

    def test_router_publishes_the_pinned_commit(self):
        """`/health`'s `revisions` block is `Router.loaded_revisions`, so the pin shows up there."""
        from laya import Router

        bundle = "convaiinnovations/laya"
        with patch.dict(os.environ), patch("huggingface_hub.snapshot_download",
                                           return_value=str(self.repo)):
            os.environ.pop("LAYA_REVISION", None)
            unpinned = Router()
            unpinned.load("english")
            self.assertEqual(unpinned.loaded_revisions, {"english": None})

            os.environ["LAYA_REVISION"] = "reviewed"
            pinned = Router()
            pinned.load("english")
            self.assertEqual(pinned.loaded_revisions, {"english": PINNED_REVISIONS[bundle]})

    def test_the_onnx_loader_resolves_through_the_same_function(self):
        """`ONNXAgent` must not grow its own copy of the rule, or the env pin covers one loader."""
        from laya import onnx_agent, revisions

        self.assertIs(onnx_agent.resolve_revision, revisions.resolve_revision)

    def test_the_base_encoder_load_ignores_the_environment(self):
        """`common.build_model`'s training-time load is a different repository, with no reviewed
        SHA to resolve to, so `LAYA_REVISION` deliberately does not reach it."""
        import laya.common as common

        seen = {}

        def from_pretrained(name, **kwargs):
            seen.update(kwargs)
            seen["name"] = name
            raise RuntimeError("stop after the call")

        cfg = {"encoder": "unused/base-encoder", "head_layers": 1, "act_costs": {"a": 0}}
        with patch.dict(os.environ, {"LAYA_REVISION": "reviewed"}), \
                patch("transformers.AutoModel.from_pretrained", side_effect=from_pretrained):
            with self.assertRaises(RuntimeError):
                common.build_model(cfg, pretrained=True)
        self.assertEqual(seen["name"], "unused/base-encoder")
        self.assertNotIn("revision", seen)

    def test_load_signature_names_every_agent_option(self):
        # `load` is a pass-through, so an `Agent` option it does not name is unreachable
        # from the entry point the guides recommend. Read off the signatures rather than
        # written down, so the next option added to Agent has to be forwarded here too.
        unreachable = (set(inspect.signature(Agent.__init__).parameters) - {"self"}
                       - set(inspect.signature(load).parameters))
        self.assertEqual(unreachable, set(),
                         "laya.load cannot set: %s" % ", ".join(sorted(unreachable)))

    def test_compile_true_wraps_the_model_loaded_by_load(self):
        # The observable `Agent.__init__` produces for compile=True, reached through load().
        # Nothing is actually compiled here: torch.compile wraps eagerly and specializes on
        # the first forward, so this stays a weight-free-of-cost check.
        eager = load(str(self.repo), device="cpu")
        self.assertFalse(eager.model.encoder.config.reference_compile)
        self.assertNotIsInstance(eager.model, OptimizedModule)

        compiled = load(str(self.repo), device="cpu", compile=True)
        self.assertTrue(compiled.model.encoder.config.reference_compile)
        self.assertIsInstance(compiled.model, OptimizedModule)
        self.assertIsInstance(compiled.model._orig_mod, DecisionModel)

    def test_backend_selection_preserves_legacy_compile_and_can_replace_it(self):
        compiled = load(str(self.repo), device="cpu", compile=True)
        self.assertEqual(compiled.backend, "compile")
        self.assertIsNone(compiled.backend_object)
        self.assertEqual(compiled.set_backend("eager"), "eager")
        self.assertNotIsInstance(compiled.model, OptimizedModule)
        self.assertFalse(compiled.model.encoder.config.reference_compile)
        self.assertEqual(compiled.predict("hello", self.questions)["answers"], self.expected["answers"])

    def test_explicit_backend_takes_precedence_over_legacy_flags(self):
        agent = load(str(self.repo), device="cpu", backend="eager", fast=True, compile=True)
        self.assertEqual(agent.backend, "eager")
        self.assertIsNotNone(agent.backend_object)
        self.assertNotIsInstance(agent.model, OptimizedModule)
        self.assertFalse(agent.model.encoder.config.reference_compile)
        self.assertIsNone(agent._fast)
        self.assertEqual(agent.predict("hello", self.questions)["answers"], self.expected["answers"])

    def test_onnx_agent_accepts_every_hub_option_the_agent_does(self):
        # Both runtimes download the same checkpoint from the same place, so an option that
        # selects *which* checkpoint, or *how* to authenticate for it, has to exist on both.
        # Read off the signatures so the next Hub-side option added to Agent has to be
        # copied across rather than silently diverging again.
        #   fast/compile -- the TileLang path and torch.compile, neither of which exists
        #                    inside onnxruntime.
        #   backend      -- selects a PyTorch forward; load() routes ONNX to ONNXAgent.
        #   compile_warmup/compile_cache/compile_mode -- configure torch.compile, which
        #                    does not exist inside onnxruntime either.
        #   device       -- ONNXAgent picks an execution provider from what onnxruntime
        #                    reports and takes no override; a different asymmetry, with its
        #                    own fix.
        not_for_onnxruntime = {"fast", "compile", "device", "backend",
                               "compile_warmup", "compile_cache", "compile_mode"}
        agent_side = (set(inspect.signature(Agent.__init__).parameters)
                      - {"self", "model_id_or_path"} - not_for_onnxruntime)
        onnx_side = set(inspect.signature(ONNXAgent.__init__).parameters) - {"self"}
        missing = agent_side - onnx_side
        self.assertEqual(missing, set(),
                         "ONNXAgent cannot set: %s" % ", ".join(sorted(missing)))

    def test_onnx_agent_forwards_the_token_to_the_download(self):
        # `token` is only useful if it reaches snapshot_download. Drive the real constructor
        # with the transport replaced and read the call it makes: the graph file is absent
        # on purpose, so the load stops right after the download it is meant to authenticate.
        # onnxruntime is imported at the top of __init__ and never used before that stop, so
        # an empty stand-in keeps this a no-extra-required check.
        with tempfile.TemporaryDirectory() as snapshot:
            (Path(snapshot) / "rl_agent_config.json").write_text(json.dumps({
                "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
                "max_len": 64, "head_max_len": 32,
            }))
            with patch.dict(sys.modules, {"onnxruntime": _NoRuntime}), \
                    patch("huggingface_hub.snapshot_download", return_value=snapshot) as download:
                with self.assertRaises(FileNotFoundError):
                    ONNXAgent("test/private-model", token="test-token",
                              onnx_path=str(self.repo / "absent.onnx"))
                self.assertEqual(download.call_args.kwargs["token"], "test-token")
                # the allow-list is unchanged: a token must not widen what gets fetched
                self.assertNotIn("model.safetensors", download.call_args.kwargs["allow_patterns"])

    def test_onnx_agent_token_defaults_and_reads_the_environment(self):
        with tempfile.TemporaryDirectory() as snapshot:
            (Path(snapshot) / "rl_agent_config.json").write_text(json.dumps({
                "encoder": "unused/offline", "head_layers": 0, "act_costs": {"act": 0},
                "max_len": 64, "head_max_len": 32,
            }))
            with patch.dict(sys.modules, {"onnxruntime": _NoRuntime}), \
                    patch("huggingface_hub.snapshot_download", return_value=snapshot) as download:
                with self.assertRaises(FileNotFoundError):
                    ONNXAgent("test/private-model", onnx_path=str(self.repo / "absent.onnx"))
                self.assertIsNone(download.call_args.kwargs["token"])

                with patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
                    with self.assertRaises(FileNotFoundError):
                        ONNXAgent("test/private-model", onnx_path=str(self.repo / "absent.onnx"))
                    self.assertEqual(download.call_args.kwargs["token"], "env-token")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
