"""Bake-from-ModelScope has to produce a cache the offline loader accepts -- exactly that.

`docker/prefetch_modelscope.py` writes a Hugging Face snapshot cache during an image build so that
a host that cannot reach huggingface.co can still ship and serve a checkpoint. The whole design
rests on one property, and it is not visible in the script: with `HF_HUB_OFFLINE=1`,
`snapshot_download` returns `<cache>/snapshots/<sha>` whenever `refs/<revision>` names a directory
that already exists, without listing the repository (`huggingface_hub/_snapshot_download.py`).
These tests therefore bake from a stubbed ModelScope API and then ask the real `snapshot_download`
what it resolves, so a layout that merely looks right cannot pass.

The second property worth pinning is the failure direction. A snapshot directory is *served*, not
validated, so anything half-written would be loaded as weights: a download that fails its size or
digest check must leave nothing behind for the loader to find.

No weights, no network, no docker: the file listing and the file bodies are fixtures.
"""
import hashlib
import http.client
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker" / "prefetch_modelscope.py"
spec = importlib.util.spec_from_file_location("prefetch_modelscope", SCRIPT)
prefetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch)

WEIGHTS_COMMIT = "a" * 40
TIP_COMMIT = "b" * 40
CONTENTS = {
    "multilingual/rl_agent_config.json": b'{"encoder": "jhu-clsp/mmBERT-base"}',
    "multilingual/model.safetensors": b"WEIGHTS" * 64,
    "multilingual/tokenizer/tokenizer.json": b'{"model": {}}',
    "multilingual/encoder/config.json": b'{"hidden_size": 768}',
    "multilingual/README.md": b"not a checkpoint file",
    "README.md": b"outside the subfolder",
    # A standalone repository keeps the same checkpoint at its root.
    "model.safetensors": b"STANDALONE" * 8,
    "rl_agent_config.json": b"standalone config",
    "tokenizer/tokenizer.json": b"standalone tokenizer",
    "encoder/config.json": b"standalone encoder",
}


def entry(path, commit):
    body = CONTENTS[path]
    return {"Type": "blob", "Path": path, "Size": len(body),
            "Sha256": hashlib.sha256(body).hexdigest(), "Revision": commit}


def tree(path):
    return {"Type": "tree", "Path": path, "Size": 0, "Revision": WEIGHTS_COMMIT}


# The listing API answers one directory at a time and names subdirectories as `tree` entries, the
# way the mirror does: the checkpoint's tokenizer and encoder are pages of their own, and a sibling
# checkpoint directory is a page this bake must not walk into.
PAGES = {
    "": [
        entry("model.safetensors", WEIGHTS_COMMIT),
        entry("rl_agent_config.json", WEIGHTS_COMMIT),
        entry("README.md", TIP_COMMIT),
        tree("encoder"),
        tree("multilingual"),
    ],
    "encoder": [entry("encoder/config.json", WEIGHTS_COMMIT)],
    "multilingual": [
        entry("multilingual/model.safetensors", WEIGHTS_COMMIT),
        entry("multilingual/rl_agent_config.json", WEIGHTS_COMMIT),
        entry("multilingual/README.md", WEIGHTS_COMMIT),
        entry("README.md", TIP_COMMIT),
        tree("multilingual/tokenizer"),
        tree("multilingual/encoder"),
        tree("typed-decisions"),
    ],
    "multilingual/tokenizer": [entry("multilingual/tokenizer/tokenizer.json", WEIGHTS_COMMIT)],
    "multilingual/encoder": [entry("multilingual/encoder/config.json", WEIGHTS_COMMIT)],
    "typed-decisions": [tree("typed-decisions/tokenizer"), tree("typed-decisions/encoder")],
    "typed-decisions/tokenizer": [],
    "typed-decisions/encoder": [],
}


def page(root):
    listing = {"Success": True,
               "Data": {"Files": PAGES[root], "LatestCommitter": {"Revision": TIP_COMMIT}}}
    return json.dumps(listing).encode("utf-8")


def params(url):
    # keep_blank_values: the root listing carries `Root=`, which parse_qsl otherwise drops.
    pairs = urllib.parse.parse_qsl(urllib.parse.urlparse(url).query, keep_blank_values=True)
    return dict(pairs)


class FakeResponse:
    """Serves the listing API, the file API and HTTP ranges, the way the mirror's CDN does."""

    status = 200

    def __init__(self, request, **_):
        self.headers = {key.lower(): value for key, value in request.headers.items()}
        self.url = request.full_url
        self.served = 0
        if "/repo/files" in self.url:
            self.body = page(params(self.url)["Root"])
            return
        self.body = CONTENTS[params(self.url)["FilePath"]]
        start = 0
        header = self.headers.get("range", "")
        if header.startswith("bytes="):
            start = int(header[len("bytes="):].split("-")[0])
        if start:
            self.body = self.body[start:]
            self.status = 206

    def read(self, size=-1):
        size = size if size and size > 0 else len(self.body)
        chunk = self.body[self.served:self.served + size]
        self.served += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Dropping(FakeResponse):
    """Hands over the first 32 bytes of the weights file, then drops the connection mid-body."""

    dropped = False

    def read(self, size=-1):
        if "FilePath" not in self.url:
            return FakeResponse.read(self, size)
        if not params(self.url)["FilePath"].endswith("model.safetensors"):
            return FakeResponse.read(self, size)
        if self.served < 32:
            self.served = 32
            return self.body[:32]
        if not Dropping.dropped and "range" not in self.headers:
            Dropping.dropped = True
            raise http.client.IncompleteRead(b"dropped mid-body")
        return FakeResponse.read(self, size)


class ShortBody(FakeResponse):
    """Every body is one byte short of the size the listing promises."""

    def __init__(self, request, **_):
        FakeResponse.__init__(self, request, **_)
        if "/repo/files" not in self.url:
            self.body = self.body[:-1]


class WrongDigest(FakeResponse):
    """Every body hashes to something the listing does not report."""

    def __init__(self, request, **_):
        FakeResponse.__init__(self, request, **_)
        if "/repo/files" not in self.url:
            self.body = b"X" + self.body[1:]


class EmptyListing(FakeResponse):
    """A repository with no checkpoint files at all."""

    def __init__(self, request, **_):
        self.status = 200
        self.headers, self.url, self.served = {}, request.full_url, 0
        if "/repo/files" in self.url:
            self.body = json.dumps(
                {"Success": True,
                 "Data": {"Files": [], "LatestCommitter": {"Revision": TIP_COMMIT}}}).encode("utf-8")
            return
        raise AssertionError("nothing should be downloaded")


class TypeTests(unittest.TestCase):
    """One argument selects the checkpoint, and it defaults to the multilingual one."""

    SPEC = "convaiinnovations/laya"

    def test_each_type_is_its_own_path_in_the_bundled_repository(self):
        # The Router and the one-shot quickstart load these paths by default, so a type that meant
        # anything else would bake a snapshot no caller resolves.
        for name, subfolder in (("english", ""), ("multilingual", "multilingual"),
                                ("typed-decisions", "typed-decisions")):
            specs = prefetch.expand_model(name)
            self.assertEqual(specs, ["%s:%s" % (self.SPEC, subfolder)], name)

    def test_default_and_aliases(self):
        expected = ["convaiinnovations/laya:multilingual"]
        for value in ("multilingual", "multi", "ml", "Multilingual", " multilingual "):
            self.assertEqual(prefetch.expand_model(value), expected, value)

    def test_all_is_the_family(self):
        self.assertEqual(prefetch.expand_model("all"),
                         ["convaiinnovations/laya:", "convaiinnovations/laya:multilingual",
                          "convaiinnovations/laya:typed-decisions"])
        self.assertEqual(prefetch.expand_model("ALL"), prefetch.expand_model("all"))

    def test_specs_pass_through_and_mix_with_types(self):
        self.assertEqual(prefetch.expand_model("convaiinnovations/laya-multilingual"),
                         ["convaiinnovations/laya-multilingual"])
        self.assertEqual(prefetch.expand_model("english,convaiinnovations/laya-typed-decisions"),
                         ["convaiinnovations/laya:", "convaiinnovations/laya-typed-decisions"])
        self.assertEqual(prefetch.expand_model("models/custom:typed-decisions models/other"),
                         ["models/custom:typed-decisions", "models/other"])

    def test_main_drives_the_whole_argument(self):
        """`--model` to baked snapshot, through the CLI: the shape main() passes to prefetch()."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "hub"

            def urlopen(target, **_):
                request = target if isinstance(target, urllib.request.Request)                     else urllib.request.Request(target)
                return FakeResponse(request)

            argv = ["prefetch_modelscope.py", "--model", "english multilingual",
                    "--revision", "master", "--cache-dir", str(cache)]
            with patch.object(sys, "argv", argv),                     patch.object(prefetch.urllib.request, "urlopen", urlopen):
                self.assertEqual(prefetch.main(), 0)
            # Both types of one repository land in one merged snapshot, keyed by the revision tip.
            snapshots = sorted(p.name for p in
                               (cache / "models--convaiinnovations--laya" / "snapshots").iterdir())
            self.assertEqual(snapshots, [TIP_COMMIT])
            snapshot = cache / "models--convaiinnovations--laya" / "snapshots" / TIP_COMMIT
            for rel in ("rl_agent_config.json", "model.safetensors", "encoder/config.json",
                        "multilingual/rl_agent_config.json", "multilingual/model.safetensors"):
                self.assertTrue(snapshot.joinpath(rel).is_file(), rel)

    def test_a_bad_value_is_refused_with_the_valid_names(self):
        for value in ("", "french", "multilingual,french", "ml,spanish"):
            with self.assertRaises(SystemExit) as caught:
                prefetch.expand_model(value)
            self.assertIn("multilingual", str(caught.exception), value)
            self.assertIn("english", str(caught.exception), value)


class SelectionTests(unittest.TestCase):
    def test_wanted_keeps_exactly_what_a_load_fetches(self):
        keep = ["rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json",
                "encoder/config.json"]
        drop = ["README.md", ".gitattributes", "configuration.json", "email_utils.py",
                "rl_agent_api.py"]
        for rel in keep:
            self.assertTrue(prefetch.wanted(rel), rel)
        for rel in drop:
            self.assertFalse(prefetch.wanted(rel), rel)

    def test_specs(self):
        self.assertEqual(prefetch.split_spec("convaiinnovations/laya"), ("convaiinnovations/laya", ""))
        self.assertEqual(prefetch.split_spec("convaiinnovations/laya:multilingual"),
                         ("convaiinnovations/laya", "multilingual"))
        self.assertEqual(prefetch.split_spec("convaiinnovations/laya:"),
                         ("convaiinnovations/laya", ""))

    def test_cache_root_prefers_the_argument_then_the_hf_variables(self):
        self.assertEqual(prefetch.cache_root("/explicit"), Path("/explicit"))
        with patch.dict(os.environ, {"HF_HUB_CACHE": "/hub-cache", "HF_HOME": "/home"}):
            self.assertEqual(prefetch.cache_root(None), Path("/hub-cache"))
        with patch.dict(os.environ, {"HF_HUB_CACHE": "", "HF_HOME": "/home"}):
            self.assertEqual(prefetch.cache_root(None), Path("/home/hub"))
        self.assertEqual(prefetch.repo_folder("convaiinnovations/laya"),
                         "models--convaiinnovations--laya")


class BakeTests(unittest.TestCase):
    SPEC = "convaiinnovations/laya:multilingual"

    def setUp(self):
        self.cache = Path(tempfile.mkdtemp()) / "hub"
        self.cache.mkdir(parents=True)
        Dropping.dropped = False

    def folder(self):
        """The cache directory this bake writes, derived from the spec it is baking."""
        repo, _ = prefetch.split_spec(self.SPEC)
        return self.cache / ("models--" + repo.replace("/", "--"))

    def specs(self):
        """The subfolders under test, in the (repo, subfolder) shape prefetch() takes.

        One repository bakes one snapshot, so the default is the single checkpoint this class
        names and a test can add more of the same repository with `bake_subfolders`.
        """
        repo, spec_sub = prefetch.split_spec(self.SPEC)
        subfolders = getattr(self, "bake_subfolders", None) or [spec_sub]
        return [(repo, sub) for sub in subfolders]

    def bake(self, response=FakeResponse):
        def urlopen(target, **_):
            # `list_repo` passes a URL, `download` passes a Request; the mirror answers both.
            request = target if isinstance(target, urllib.request.Request) \
                else urllib.request.Request(target)
            return response(request)

        with patch.object(prefetch.urllib.request, "urlopen", urlopen):
            prefetch.prefetch(self.specs(), "master", self.cache)
        return self.folder()

    def published(self, repo_dir):
        """Snapshot directories a loader could resolve, ignoring an in-progress staging one."""
        snapshots = repo_dir / "snapshots"
        if not snapshots.exists():
            return []
        return sorted(p.name for p in snapshots.iterdir() if not p.name.startswith(".staging-"))

    def test_seeded_layout_matches_what_the_hub_writes(self):
        repo_dir = self.bake()
        snapshot = repo_dir / "snapshots" / TIP_COMMIT
        self.assertTrue(snapshot.is_dir(), "no snapshot for the weights' commit")
        for rel in ("multilingual/rl_agent_config.json", "multilingual/model.safetensors",
                    "multilingual/tokenizer/tokenizer.json", "multilingual/encoder/config.json"):
            self.assertTrue(snapshot.joinpath(rel).is_file(), rel)
        # The repo-relative layout, because `Agent` appends the subfolder itself: a snapshot that
        # drops `multilingual/` resolves and then fails with "Subfolder not found".
        self.assertEqual(snapshot.joinpath("multilingual", "model.safetensors").read_bytes(),
                         CONTENTS["multilingual/model.safetensors"])
        self.assertEqual(snapshot.joinpath("multilingual", "rl_agent_config.json").read_bytes(),
                         CONTENTS["multilingual/rl_agent_config.json"])
        # The runtime asks for `main`, the mirror publishes `master`: both names resolve.
        # Keyed by the revision's tip: one repository bakes one snapshot, because refs/<revision>
        # names a single commit and every checkpoint in that repository resolves through it.
        self.assertEqual((repo_dir / "refs" / "main").read_text(), TIP_COMMIT)
        self.assertEqual((repo_dir / "refs" / "master").read_text(), TIP_COMMIT)
        # Sibling files never enter the snapshot.
        self.assertFalse(snapshot.joinpath("multilingual", "README.md").exists())
        self.assertFalse(snapshot.joinpath("README.md").exists())
        self.assertEqual(self.published(repo_dir), [TIP_COMMIT])

    def test_one_repository_bakes_one_merged_snapshot(self):
        """A bundle bake carries both checkpoints, or the second one is unreachable.

        `refs/main` names a single commit, so baking english and multilingual separately leaves
        only the last one resolvable -- which is exactly what the English request hit.
        """
        self.SPEC = "convaiinnovations/laya"
        self.bake_subfolders = ["", "multilingual"]
        repo_dir = self.bake()
        snapshot = repo_dir / "snapshots" / TIP_COMMIT
        # English sits at the repository root, multilingual under its own subfolder: both loaded
        # by the same `Agent` revision argument, so both have to be in the same directory.
        self.assertTrue(snapshot.joinpath("rl_agent_config.json").is_file(), "english")
        self.assertTrue(snapshot.joinpath("model.safetensors").is_file(), "english weights")
        self.assertTrue(snapshot.joinpath("multilingual", "rl_agent_config.json").is_file(),
                        "multilingual")
        self.assertTrue(snapshot.joinpath("multilingual", "model.safetensors").is_file(),
                        "multilingual weights")
        self.assertEqual(self.published(repo_dir), [TIP_COMMIT])
        self.assertEqual((repo_dir / "refs" / "main").read_text(), TIP_COMMIT)
        # A third checkpoint in the same repository is walked past, not fetched.
        self.assertFalse(snapshot.joinpath("typed-decisions").exists())

    def test_a_standalone_repo_lands_at_the_snapshot_root(self):
        """No subfolder: the checkpoint's own files sit at the snapshot root, as on the Hub."""
        self.SPEC = "convaiinnovations/laya-multilingual"
        self.bake_subfolders = [""]
        pages = {
            "": [
                entry("model.safetensors", WEIGHTS_COMMIT),
                entry("rl_agent_config.json", WEIGHTS_COMMIT),
                entry("README.md", TIP_COMMIT),
                tree("tokenizer"),
                tree("encoder"),
            ],
            "tokenizer": [entry("tokenizer/tokenizer.json", WEIGHTS_COMMIT)],
            "encoder": [entry("encoder/config.json", WEIGHTS_COMMIT)],
        }
        with patch.dict(PAGES, pages, clear=True):
            repo_dir = self.bake()
        snapshot = repo_dir / "snapshots" / TIP_COMMIT
        self.assertTrue(snapshot.joinpath("model.safetensors").is_file())
        self.assertTrue(snapshot.joinpath("rl_agent_config.json").is_file())
        self.assertTrue(snapshot.joinpath("tokenizer", "tokenizer.json").is_file())
        self.assertFalse(snapshot.joinpath("README.md").exists())
        self.assertFalse(snapshot.joinpath("multilingual").exists())

    def test_offline_snapshot_download_resolves_the_baked_checkpoint(self):
        """The load-bearing property, checked against the real hub resolver.

        `local_files_only` is what `HF_HUB_OFFLINE=1` resolves to, and passing it beats patching the
        variable: the hub reads that one at import time, so a later patch would still dial out.
        """
        from huggingface_hub import snapshot_download

        self.bake()
        resolved = snapshot_download(
            "convaiinnovations/laya",
            allow_patterns=["multilingual/rl_agent_config.json", "multilingual/model.safetensors",
                            "multilingual/tokenizer/*", "multilingual/encoder/*"],
            cache_dir=str(self.cache), local_files_only=True,
        )
        expected = self.cache / "models--convaiinnovations--laya" / "snapshots" / TIP_COMMIT
        self.assertEqual(Path(os.path.realpath(resolved)), Path(os.path.realpath(str(expected))))

    def test_the_commit_the_bake_keyed_resolves_as_a_pin(self):
        """`revision=` reaching the revision's tip must hit the same snapshot."""
        from huggingface_hub import snapshot_download

        self.bake()
        resolved = snapshot_download("convaiinnovations/laya", revision=TIP_COMMIT,
                                     cache_dir=str(self.cache), local_files_only=True)
        self.assertEqual(Path(os.path.realpath(resolved)).name, TIP_COMMIT)

    def test_a_commit_the_bake_did_not_key_misses(self):
        """An upload commit the bake did not key must miss, not serve a snapshot it is not in."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.bake()
        with self.assertRaises(LocalEntryNotFoundError):
            snapshot_download("convaiinnovations/laya", revision=WEIGHTS_COMMIT,
                              cache_dir=str(self.cache), local_files_only=True)

    def test_a_dropped_connection_is_retried_and_resumed(self):
        """A mid-body drop must resume byte-aligned, not restart or duplicate the file."""
        repo_dir = self.bake(Dropping)
        snapshot = repo_dir / "snapshots" / TIP_COMMIT
        self.assertEqual(snapshot.joinpath("multilingual", "model.safetensors").read_bytes(),
                         CONTENTS["multilingual/model.safetensors"])
        self.assertEqual(snapshot.joinpath("multilingual", "rl_agent_config.json").read_bytes(),
                         CONTENTS["multilingual/rl_agent_config.json"])

    def test_truncated_download_refuses_to_publish(self):
        with patch.object(prefetch, "RETRIES", 2), patch.object(prefetch, "BACKOFF", 0):
            with self.assertRaises(SystemExit):
                self.bake(ShortBody)
        self.assertEqual(self.published(self.folder()), [])

    def test_digest_mismatch_refuses_to_publish(self):
        with self.assertRaises(SystemExit):
            self.bake(WrongDigest)
        self.assertEqual(self.published(self.folder()), [])

    def test_empty_selection_is_refused(self):
        with self.assertRaises(SystemExit):
            self.bake(EmptyListing)
        self.assertEqual(self.published(self.folder()), [])

    def test_rebaking_the_same_commit_replaces_the_snapshot(self):
        self.bake()
        repo_dir = self.bake()
        self.assertEqual(self.published(repo_dir), [TIP_COMMIT])


if __name__ == "__main__":
    unittest.main()
