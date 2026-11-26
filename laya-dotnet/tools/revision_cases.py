"""Revision-pinning and digest-check corpus for revision_probe.json (SDK-neutral).

Consumed by `dump_routing_golden.py`, which calls `build_revision_probe()` to run every case through
Python's own `laya.revisions.resolve_revision` / `verify_digests` and `laya.router._digests_from_env` /
`_merge_expected_digests`, recording the inputs together with the result or the exception type and text.
No model weights, no network.

The fixture checkpoint lives in `revision_fixtures/checkpoint/` beside this file (bytes pinned by the
`.gitattributes` there, so a checkout cannot change a digest). A port copies this file and that directory
verbatim and replays `revision_probe.json` against its own implementation:

    pinned_revisions   the PINNED_REVISIONS table and the `reviewed` word
    resolve            {label, model, revision, env, result}   env = LAYA_REVISION or null (unset)
    verify             {label, dir, expected, env, onnx_path, result}
                       dir is relative to this directory; expected is null (None) or {relpath: digest};
                       onnx_path is relative to `dir` or null; env = LAYA_SHA256_DIGESTS or null.
                       Digests are literals; "{dir}" in an error text stands for the directory's path.
    digests_from_env   {label, env, models, result}      the nested-or-flat LAYA_SHA256_DIGESTS reading
    merge              {label, shared, per_model, result}  Router's per-file merge, per-checkpoint wins

`result` is {"ok": value} or {"error_type": <Python class name>, "error_message": <str(exception)>}.
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "revision_fixtures")
CHECKPOINT = "revision_fixtures/checkpoint"
CHECKPOINT_DIR = os.path.join(HERE, *CHECKPOINT.split("/"))

# The fixture files and their exact bytes; written once by `ensure_fixtures()` and committed.
FIXTURE_FILES = {
    "model.onnx": b"fused graph\r\nsecond line\n",
    "model.onnx.data": bytes(range(256)) * 3,
    "encoder.onnx": b"encoder graph",
    "rl_agent_config.json": b'{"encoder": "fixture"}\n',
    "tokenizer/tokenizer.json": b'{"version": "1.0"}\r\n',
    "weights/data.bin": b"\x00\xff\x10weights",
    "\u00dcn\u00ef.bin": b"unicode name",
}


def ensure_fixtures():
    """Create the fixture files when absent. Existing files are never rewritten."""
    for rel, data in FIXTURE_FILES.items():
        path = os.path.join(CHECKPOINT_DIR, *rel.split("/"))
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)


def sha(rel):
    with open(os.path.join(CHECKPOINT_DIR, *rel.split("/")), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


ENV_NAMES = ("LAYA_REVISION", "LAYA_SHA256_DIGESTS")

# (label, model_id_or_path, revision, LAYA_REVISION)
RESOLVE_CASES = [
    ("unset", "convaiinnovations/laya", None, None),
    ("argument_commit", "convaiinnovations/laya", "abc123", None),
    ("argument_stripped", "convaiinnovations/laya", "  abc123\t", None),
    ("env_commit", "convaiinnovations/laya", None, "def456"),
    ("env_stripped", "convaiinnovations/laya", None, "  def456\n"),
    ("env_empty", "convaiinnovations/laya", None, ""),
    ("env_blank", "convaiinnovations/laya", None, "   "),
    ("argument_beats_env", "convaiinnovations/laya", "abc123", "def456"),
    ("empty_argument_uses_env", "convaiinnovations/laya", "", "def456"),
    ("blank_argument_suppresses_env", "convaiinnovations/laya", "   ", "def456"),
    ("blank_argument_no_env", "convaiinnovations/laya", "  ", None),
    ("branch_name", "convaiinnovations/laya", "refs/pr/1", None),
    ("tag_name", "convaiinnovations/laya", "v0.4.16", None),
    ("reviewed_argument_english", "convaiinnovations/laya", "reviewed", None),
    ("reviewed_argument_multilingual", "convaiinnovations/laya-multilingual", "reviewed", None),
    ("reviewed_argument_typed", "convaiinnovations/laya-typed-decisions", "reviewed", None),
    ("reviewed_argument_padded", "convaiinnovations/laya", "  reviewed  ", None),
    ("reviewed_env", "convaiinnovations/laya-multilingual", None, "reviewed"),
    ("reviewed_env_padded", "convaiinnovations/laya", None, " reviewed\n"),
    ("reviewed_case_insensitive_repo", "ConvaiInnovations/Laya", "reviewed", None),
    ("reviewed_upper_repo", "CONVAIINNOVATIONS/LAYA-MULTILINGUAL", None, "reviewed"),
    ("reviewed_is_case_sensitive", "convaiinnovations/laya", "Reviewed", None),
    ("reviewed_argument_beats_env", "convaiinnovations/laya", "reviewed", "def456"),
    ("commit_argument_beats_reviewed_env", "convaiinnovations/laya", "abc123", "reviewed"),
    ("blank_argument_beats_reviewed_env", "convaiinnovations/laya", "  ", "reviewed"),
    ("reviewed_unknown_repo_argument", "someone/else", "reviewed", None),
    ("reviewed_unknown_repo_env", "someone/else", None, "reviewed"),
    ("reviewed_unknown_repo_both", "someone/else", "reviewed", "reviewed"),
    ("reviewed_local_path_argument", "C:\\models\\laya", "reviewed", None),
    ("reviewed_quote_in_repo", "it's/repo", None, "reviewed"),
    ("reviewed_unicode_repo", "\u00c4rger/laya", "reviewed", None),
    ("reviewed_dotted_capital_i", "convaiinnovations/LAYA".replace("convai", "CONVA\u0130"), "reviewed", None),
    ("unknown_repo_commit_is_fine", "someone/else", "abc123", None),
]

# (label, dir, expected, LAYA_SHA256_DIGESTS, onnx_path); expected values may be "@sha:<file>" for the
# fixture file's digest, "@SHA:<file>" for it upper-cased and "@sha-padded:<file>" for it upper-cased
# and padded with spaces. They are resolved to literals before they are recorded.
VERIFY_CASES = [
    ("none_no_env", CHECKPOINT, None, None, None),
    ("none_blank_env", CHECKPOINT, None, "   ", None),
    ("empty_map", CHECKPOINT, {}, None, None),
    ("empty_map_masks_bad_env", CHECKPOINT, {}, '{"model.onnx": "0000"}', None),
    ("empty_map_masks_malformed_env", CHECKPOINT, {}, "{", None),
    ("single_ok", CHECKPOINT, {"model.onnx": "@sha:model.onnx"}, None, None),
    ("several_ok", CHECKPOINT, {
        "model.onnx": "@sha:model.onnx", "model.onnx.data": "@sha:model.onnx.data",
        "tokenizer/tokenizer.json": "@sha:tokenizer/tokenizer.json", "weights/data.bin": "@sha:weights/data.bin",
    }, None, None),
    ("uppercase_digest_ok", CHECKPOINT, {"model.onnx": "@SHA:model.onnx"}, None, None),
    ("padded_digest_ok", CHECKPOINT, {"model.onnx": "@sha-padded:model.onnx"}, None, None),
    ("unicode_name_ok", CHECKPOINT, {"\u00dcn\u00ef.bin": "@sha:\u00dcn\u00ef.bin"}, None, None),
    ("backslash_path_ok", CHECKPOINT, {"tokenizer\\tokenizer.json": "@sha:tokenizer/tokenizer.json"}, None, None),
    ("dot_segment_ok", CHECKPOINT, {"./model.onnx": "@sha:model.onnx"}, None, None),
    ("double_slash_inside_ok", CHECKPOINT, {"tokenizer//tokenizer.json": "@sha:tokenizer/tokenizer.json"}, None, None),
    ("mismatch", CHECKPOINT, {"model.onnx": "0" * 64}, None, None),
    ("mismatch_second_file", CHECKPOINT, {"model.onnx": "@sha:model.onnx", "encoder.onnx": "ab" * 32}, None, None),
    ("mismatch_blank_digest", CHECKPOINT, {"model.onnx": ""}, None, None),
    ("mismatch_prefix_digest", CHECKPOINT, {"model.onnx": "sha256:@sha:model.onnx"}, None, None),
    ("missing_file", CHECKPOINT, {"nope.bin": "00"}, None, None),
    ("missing_after_ok_file", CHECKPOINT, {"model.onnx": "@sha:model.onnx", "nope.bin": "00"}, None, None),
    ("directory_is_not_a_file", CHECKPOINT, {"tokenizer": "00"}, None, None),
    ("trailing_dotdot_is_missing", CHECKPOINT, {"tokenizer/..": "00"}, None, None),
    ("single_dot_is_missing", CHECKPOINT, {".": "00"}, None, None),
    ("absolute_slash", CHECKPOINT, {"/model.onnx": "00"}, None, None),
    ("absolute_double_slash", CHECKPOINT, {"//server/share/x": "00"}, None, None),
    ("absolute_drive_slash", CHECKPOINT, {"C:/windows/x": "00"}, None, None),
    ("absolute_drive_backslash", CHECKPOINT, {"C:\\windows\\x": "00"}, None, None),
    ("absolute_unc_backslash", CHECKPOINT, {"\\\\server\\share": "00"}, None, None),
    ("absolute_any_char_drive", CHECKPOINT, {"1:/x": "00"}, None, None),
    ("escape_dotdot", CHECKPOINT, {"..": "00"}, None, None),
    ("escape_dotdot_slash", CHECKPOINT, {"../x": "00"}, None, None),
    ("escape_dotdot_backslash", CHECKPOINT, {"..\\x": "00"}, None, None),
    ("escape_inner_dotdot", CHECKPOINT, {"a/../b": "00"}, None, None),
    ("empty_key", CHECKPOINT, {"": "00"}, None, None),
    ("slash_key_is_absolute", CHECKPOINT, {"/": "00"}, None, None),
    ("unsafe_after_ok_file", CHECKPOINT, {"model.onnx": "@sha:model.onnx", "../x": "00"}, None, None),
    ("onnx_key_with_onnx_path", CHECKPOINT, {"onnx": "@sha:model.onnx"}, None, "model.onnx"),
    ("onnx_path_key_with_onnx_path", CHECKPOINT, {"onnx_path": "@sha:encoder.onnx"}, None, "encoder.onnx"),
    ("onnx_key_mismatch", CHECKPOINT, {"onnx": "0" * 64}, None, "model.onnx"),
    ("onnx_key_without_onnx_path_is_a_file_name", CHECKPOINT, {"onnx": "00"}, None, None),
    ("onnx_key_and_file", CHECKPOINT, {"onnx": "@sha:model.onnx", "model.onnx.data": "@sha:model.onnx.data"}, None, "model.onnx"),
    ("env_ok", CHECKPOINT, None, '{"model.onnx": "@sha:model.onnx"}', None),
    ("env_uppercase_ok", CHECKPOINT, None, '{"model.onnx": "@SHA:model.onnx"}', None),
    ("env_padded_json_ok", CHECKPOINT, None, ' {"model.onnx": "@sha:model.onnx"}\n', None),
    ("env_mismatch", CHECKPOINT, None, '{"model.onnx": "00"}', None),
    ("env_missing_file", CHECKPOINT, None, '{"nope.bin": "00"}', None),
    ("env_unsafe_path", CHECKPOINT, None, '{"../x": "00"}', None),
    ("env_onnx_key_with_onnx_path", CHECKPOINT, None, '{"onnx": "@sha:model.onnx"}', "model.onnx"),
    ("env_malformed", CHECKPOINT, None, "{", None),
    ("env_single_quotes", CHECKPOINT, None, "{'model.onnx': 'x'}", None),
    ("env_list", CHECKPOINT, None, '["model.onnx"]', None),
    ("env_string", CHECKPOINT, None, '"model.onnx"', None),
    ("env_number", CHECKPOINT, None, "5", None),
    ("env_null", CHECKPOINT, None, "null", None),
    ("env_empty_object", CHECKPOINT, None, "{}", None),
    ("env_number_digest", CHECKPOINT, None, '{"model.onnx": 5}', None),
    ("env_float_digest", CHECKPOINT, None, '{"model.onnx": 1.5}', None),
    ("env_null_digest", CHECKPOINT, None, '{"model.onnx": null}', None),
    ("env_true_digest", CHECKPOINT, None, '{"model.onnx": true}', None),
    ("env_nan_digest", CHECKPOINT, None, '{"model.onnx": NaN}', None),
    ("env_list_digest", CHECKPOINT, None, '{"model.onnx": ["a", 1]}', None),
    ("env_object_digest", CHECKPOINT, None, '{"model.onnx": {"a": null}}', None),
    ("env_nested_shape_names_a_file", CHECKPOINT, None, '{"english": {"model.onnx": "00"}}', None),
    ("env_duplicate_key_last_wins", CHECKPOINT, None, '{"model.onnx": "00", "model.onnx": "@sha:model.onnx"}', None),
    ("env_unicode_escape_key", CHECKPOINT, None, '{"\\u00dcn\\u00ef.bin": "@sha:\u00dcn\u00ef.bin"}', None),
]

_NESTED = '{"english": {"model.onnx": "aa"}, "multilingual": {"model.onnx": "bb", "weights/data.bin": "cc"}}'

# (label, LAYA_SHA256_DIGESTS)
DIGESTS_FROM_ENV_CASES = [
    ("unset", None),
    ("empty", ""),
    ("blank", "   "),
    ("malformed", "{"),
    ("not_an_object", '["a"]'),
    ("empty_object", "{}"),
    ("flat", '{"model.onnx": "aa"}'),
    ("flat_two", '{"model.onnx": "aa", "weights/data.bin": "bb"}'),
    ("nested", _NESTED),
    ("nested_one_model", '{"english": {"model.onnx": "aa"}}'),
    ("nested_aliases", '{"en": {"a": "1"}, "ml": {"b": "2"}, "typed": {"c": "3"}}'),
    ("nested_laya_alias_is_english", '{"laya": {"a": "1"}}'),
    ("nested_padded_upper_name", '{" ENGLISH ": {"a": "1"}}'),
    ("nested_alias_and_name_last_wins", '{"en": {"a": "1"}, "english": {"b": "2"}}'),
    ("nested_empty_inner", '{"english": {}}'),
    ("nested_all_empty", '{"english": {}, "multilingual": {}, "typed-decisions": {}}'),
    ("mixed_flat_and_nested", '{"model.onnx": "aa", "english": {"model.onnx": "bb"}}'),
    ("mixed_with_number", '{"english": {"a": "1"}, "n": 5}'),
    ("mixed_with_list", '{"english": {"a": "1"}, "l": ["x"]}'),
    ("mixed_with_null", '{"english": {"a": "1"}, "z": null}'),
    ("all_numbers", '{"a": 1, "b": 2}'),
    ("mixed_sorted_keys_repr", '{"zeta": "x", "it\'s": {"a": "1"}, "alpha": 1}'),
    ("nested_unknown_model", '{"gpt": {"a": "1"}}'),
    ("nested_unknown_model_after_known", '{"english": {"a": "1"}, "gpt": {"a": "1"}}'),
]

MODELS = ["english", "multilingual", "typed-decisions"]

# (label, shared, per_model)
MERGE_CASES = [
    ("both_silent", None, None),
    ("shared_empty", {}, None),
    ("per_model_empty", None, {}),
    ("both_empty", {}, {}),
    ("shared_only", {"a": "1"}, None),
    ("per_model_only", None, {"a": "1"}),
    ("shared_with_empty_placeholder", {"a": "1"}, {}),
    ("disjoint_files_merge", {"a": "1"}, {"b": "2"}),
    ("per_model_wins_overlap", {"model.safetensors": "shared", "x": "1"}, {"model.safetensors": "own"}),
    ("overlap_only", {"a": "1"}, {"a": "2"}),
    ("order_shared_then_per_model", {"b": "1", "a": "2"}, {"c": "3", "a": "4"}),
]


def _resolve_digest(value):
    if not isinstance(value, str):
        return value
    for prefix, fn in (("@sha-padded:", lambda s: "  " + s.upper() + "\t "),
                       ("@SHA:", str.upper), ("@sha:", str)):
        if value.startswith(prefix):
            return fn(sha(value[len(prefix):]))
        if prefix in value:
            head, _, tail = value.partition(prefix)
            return head + fn(sha(tail))
    return value


def _resolve_json_digests(text):
    if text is None:
        return None
    for prefix, fn in (("@sha-padded:", lambda s: "  " + s.upper() + "  "),
                       ("@SHA:", str.upper), ("@sha:", str)):
        while prefix in text:
            head, _, tail = text.partition(prefix)
            name = tail.split('"', 1)[0]
            text = head + fn(sha(name)) + tail[len(name):]
    return text


def _with_env(values, fn):
    saved = {n: os.environ.pop(n, None) for n in ENV_NAMES}
    try:
        for name, value in values.items():
            if value is not None:
                os.environ[name] = value
        return fn()
    finally:
        for name in ENV_NAMES:
            os.environ.pop(name, None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def _result(fn, model_dir=None):
    try:
        return {"ok": fn()}
    except Exception as exc:  # the probe records whatever Python raised
        text = str(exc)
        if model_dir:
            text = text.replace(model_dir, "{dir}")
        return {"error_type": type(exc).__name__, "error_message": text}


def build_revision_probe():
    from laya.revisions import PINNED_REVISIONS, REVIEWED, resolve_revision, verify_digests
    from laya.router import _digests_from_env, _merge_expected_digests

    ensure_fixtures()

    resolve = []
    for label, model, revision, env in RESOLVE_CASES:
        result = _with_env({"LAYA_REVISION": env}, lambda: _result(lambda: resolve_revision(model, revision)))
        resolve.append({"label": label, "model": model, "revision": revision, "env": env, "result": result})

    verify = []
    for label, rel_dir, expected, env, onnx in VERIFY_CASES:
        model_dir = os.path.join(HERE, *rel_dir.split("/"))
        resolved = None if expected is None else {k: _resolve_digest(v) for k, v in expected.items()}
        env_text = _resolve_json_digests(env)
        onnx_path = None if onnx is None else os.path.join(model_dir, onnx)
        result = _with_env(
            {"LAYA_SHA256_DIGESTS": env_text},
            lambda: _result(lambda: verify_digests(model_dir, resolved, onnx_path=onnx_path), model_dir))
        verify.append({"label": label, "dir": rel_dir, "expected": resolved, "env": env_text,
                       "onnx_path": onnx, "result": result})

    from_env = []
    for label, env in DIGESTS_FROM_ENV_CASES:
        result = _with_env(
            {"LAYA_SHA256_DIGESTS": env},
            lambda: _result(lambda: _digests_from_env({name: name for name in MODELS})))
        from_env.append({"label": label, "env": env, "models": MODELS, "result": result})

    merge = []
    for label, shared, per_model in MERGE_CASES:
        merge.append({"label": label, "shared": shared, "per_model": per_model,
                      "result": _result(lambda: _merge_expected_digests(shared, per_model))})

    return {
        "pinned_revisions": dict(PINNED_REVISIONS),
        "reviewed": REVIEWED,
        "resolve": resolve,
        "verify": verify,
        "digests_from_env": from_env,
        "merge": merge,
    }
