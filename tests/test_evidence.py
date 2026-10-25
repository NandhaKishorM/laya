"""Contract guard for laya.evidence: readiness evidence states over existing artifacts.

Run: python tests/test_evidence.py
"""
import inspect
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import evidence  # noqa: E402
from laya import evals_cli  # noqa: E402

PASS, FAIL = [], []


def check(name, got, expected):
    ok = got == expected
    (PASS if ok else FAIL).append(name)
    print(("  ok   " if ok else "  FAIL ") + name, "" if ok else "(got %r, expected %r)" % (got, expected))


def check_true(name, condition):
    check(name, bool(condition), True)


def calib(config):
    return evidence.inspect_calibration(config)


# ------------------------------------------------------ calibration states
legacy = calib({"fine_tuned": True})
check("legacy checkpoint (no training key) -> UNKNOWN", legacy["state"], evidence.UNKNOWN)

legacy2 = calib({"training": {"laya_train": {"base": "x"}}})
check("training but no calibration block -> UNKNOWN", legacy2["state"], evidence.UNKNOWN)

zero = calib({"training": {"laya_train_calibration": {"choice": {"items": 0, "temperature": 1.0, "issues": []}}}})
check("zero items -> MISSING", zero["types"]["choice"]["state"], evidence.MISSING)

low = calib({"training": {"laya_train_calibration": {"choice": {"items": 5, "temperature": 1.0,
        "issues": ["not fitted: 5 calibration items, fewer than 10, so the temperature stays 1.0"]}}}})
check("upstream below-MIN_TYPE_N limitation -> INSUFFICIENT", low["types"]["choice"]["state"], evidence.INSUFFICIENT)

warned = calib({"training": {"laya_train_calibration": {"choice": {"items": 30, "temperature": 1.2,
        "issues": ["fitted on only 30 calibration items"]}}}})
check("upstream below-CALIB_WARN_N limitation -> INSUFFICIENT", warned["types"]["choice"]["state"], evidence.INSUFFICIENT)

clamped = calib({"training": {"laya_train_calibration": {"choice": {"items": 200, "temperature": 0.5,
        "issues": ["the fitted temperature 0.5 is on the [0.5, 3] clamp"]}}}})
check("clamped fit stays a visible limitation -> INSUFFICIENT", clamped["types"]["choice"]["state"], evidence.INSUFFICIENT)

# ---------------------------------------------- the reason text, which nothing pinned at all
# `low`'s issue list says "not fitted", and the prefix used to say "fitted with recorded
# limitations", so one `laya-evals evidence` line asserted the fit and denied it in the same
# breath. The reason's job is to carry the upstream text; it adds no claim of its own.
check("below-MIN_TYPE_N reason is the prefix plus the upstream text, verbatim",
      low["types"]["choice"]["reason"],
      "limitations recorded: not fitted: 5 calibration items, fewer than 10, so the "
      "temperature stays 1.0")
check("below-CALIB_WARN_N reason is the prefix plus the upstream text, verbatim",
      warned["types"]["choice"]["reason"], "limitations recorded: fitted on only 30 calibration items")
check("clamped reason is the prefix plus the upstream text, verbatim",
      clamped["types"]["choice"]["reason"],
      "limitations recorded: the fitted temperature 0.5 is on the [0.5, 3] clamp")
check_true("no limitation reason states a fit the artifact did not record",
           all(not t["reason"].startswith("fitted") for t in
               (low["types"]["choice"], warned["types"]["choice"], clamped["types"]["choice"])))

clean = calib({"training": {"laya_train_calibration": {"choice": {"items": 400, "temperature": 1.1, "issues": []}}}})
check("clean supported fit -> PRESENT", clean["types"]["choice"]["state"], evidence.PRESENT)
check("all-clean overall -> PRESENT", clean["state"], evidence.PRESENT)

mixed = calib({"training": {"laya_train_calibration": {
    "choice": {"items": 400, "temperature": 1.1, "issues": []},
    "score": {"items": 0, "temperature": 1.0, "issues": []}}}})
check("mixed types overall -> INSUFFICIENT", mixed["state"], evidence.INSUFFICIENT)

malformed = calib({"training": {"laya_train_calibration": {"choice": {"temperature": 1.0}}}})
check("malformed entry -> UNKNOWN", malformed["types"]["choice"]["state"], evidence.UNKNOWN)

# A JSON `true` is not a count, but `isinstance(True, int)` makes it one. Before this guard
# `{"items": true}` answered PRESENT with "fitted with no recorded issues on 1 items" and
# `{"items": false}` answered MISSING with "zero calibration items": two verdicts about a fit that
# the artifact never reported, and the second one is the state a genuinely unfitted type gets.
for _flag in (True, False):
    _t = calib({"training": {"laya_train_calibration": {
        "choice": {"items": _flag, "temperature": 1.1, "issues": []}}}})
    check("items=%r is a malformed entry, not a count" % _flag, _t["types"]["choice"]["state"],
          evidence.UNKNOWN)
    check("items=%r keeps the malformed reason" % _flag, _t["types"]["choice"]["reason"],
          "calibration entry is malformed")
    check("items=%r leaves the overall undecided" % _flag, _t["state"], evidence.INSUFFICIENT)

# The guard is about the type of `items`, not its value: a negative count stays reported, because
# #964 deleted the local floor that judged it and nothing here is supposed to re-introduce one.
_neg = calib({"training": {"laya_train_calibration": {
    "choice": {"items": -5, "temperature": 1.1, "issues": []}}}})
check("negative items are still reported, not judged", _neg["types"]["choice"]["state"], evidence.PRESENT)
check("negative items keep their own count in the reason", _neg["types"]["choice"]["reason"],
      "fitted with no recorded issues on -5 items")

# The rest of the same door, which main already refuses but nothing pinned: a string or a float is
# not a count, and an entry that is not an object cannot carry one. All three land on the same
# `UNKNOWN` as a missing key, because none of them states a number the fit can be checked against.
for _bad, _label in (("5", "a string"), (2.0, "a float")):
    _t = calib({"training": {"laya_train_calibration": {
        "choice": {"items": _bad, "temperature": 1.1, "issues": []}}}})
    check("items=%s is not a count -> UNKNOWN" % _label, _t["types"]["choice"]["state"], evidence.UNKNOWN)
_not_entry = calib({"training": {"laya_train_calibration": {"choice": "5"}}})
check("entry that is not an object -> UNKNOWN", _not_entry["types"]["choice"]["state"], evidence.UNKNOWN)
check("entry that is not an object keeps the malformed reason", _not_entry["types"]["choice"]["reason"],
      "calibration entry is malformed")

# ------------------------------------------------------ one numeric source of truth
# `items` is reported here, never judged: #964 deleted the local `_thresholds()` that compared it
# against copies of `MIN_TYPE_N`/`CALIB_WARN_N`, and the comment describing that mechanism stayed
# behind. These are the checks the deleted mechanism never had -- a re-introduced floor has to
# survive all of them, on both sides of both upstream floors.
for _n in (1, 5, 9, 10, 49, 50, 300, 5000):
    _t = calib({"training": {"laya_train_calibration": {
        "choice": {"items": _n, "temperature": 1.1, "issues": []}}}})
    check("items=%d with no recorded issue -> PRESENT" % _n, _t["types"]["choice"]["state"], evidence.PRESENT)

# The same counts with a limitation recorded, and the state flips -- so the loop above shows that
# `issues` decides, not that a large `items` does.
_states = {calib({"training": {"laya_train_calibration": {
    "choice": {"items": _n, "temperature": 1.1,
               "issues": ["fitted on only %d calibration items" % _n]}}}})["types"]["choice"]["state"]
    for _n in (5, 9, 10, 49, 50, 300)}
check("the limitation text alone decides, at every count", _states, {evidence.INSUFFICIENT})

# The deleted mechanism also exported where its numbers came from. Nothing in this module's result
# may claim a threshold provenance, because there is no threshold to have a provenance about.
check("no threshold provenance in the result",
      "threshold_source" in calib({"training": {"laya_train_calibration": {
          "choice": {"items": 400, "temperature": 1.1, "issues": []}}}}), False)

_src = inspect.getsource(evidence)
check_true("the module comments name the artifact as the authority", "second numeric source of truth" in _src)
check_true("no comment still promises the deleted fallback",
           "_FALLBACK" not in _src and "fall back to the documented values" not in _src)

# ------------------------------------------------------ report identity
full_report = {"config": {"schema": "laya-evals-report/1", "dataset_sha256": "a",
                          "questions_sha256": "b", "laya_version": "0.3.28"}, "overall": {}}
rep = evidence.inspect_report(full_report)
check("full identity -> PRESENT", rep["state"], evidence.PRESENT)

old_report = {"config": {"schema": "laya-evals-report/1"}, "overall": {}}
rep2 = evidence.inspect_report(old_report)
check("missing identity fields -> UNKNOWN", rep2["state"], evidence.UNKNOWN)

toplevel = {"schema": "laya-evals-report/1", "dataset_sha256": "a", "questions_sha256": "b",
            "config": {}}
check("identity at top level still read", evidence.inspect_report(toplevel)["state"], evidence.PRESENT)

# ... and with no `config` block at all, which is the shape that used to answer differently.
# `research/evals/act_head_eval.py` tags its document at the top level and writes no `config` key;
# bailing on the missing block made one document read as PRESENT once an empty `config` was added
# beside the same identity, and left a caller to branch on two different result shapes.
_noblock = {"schema": "laya-evals-report/1", "dataset_sha256": "a", "questions_sha256": "b"}
rep3 = evidence.inspect_report(_noblock)
check("identity at top level, no config block -> PRESENT", rep3["state"], evidence.PRESENT)
check("and it reads field for field like the same identity under config",
      rep3["fields"], evidence.inspect_report({"config": _noblock})["fields"])
check("no identity anywhere still reports its whole vocabulary",
      sorted(evidence.inspect_report({"overall": {}})["fields"]),
      sorted(evidence._IDENTITY_KEYS + evidence._PROVENANCE_KEYS))
# Provenance rides through the same read as identity (see `_report_identity`), so a document that
# records its version at the top level is not read as recording nothing.
check("provenance at the top level is read too",
      evidence.inspect_report({"config": {}, "laya_version": "0.3.28"})["fields"]
      ["laya_version"]["state"], evidence.PRESENT)
# ... by copying, not by writing into the caller's `config`: this inspection is read-only, and a
# view that aliased the report would leave the fallback values in the document it was handed.
_untouched = {"config": {}, "schema": "laya-evals-report/1"}
evidence.inspect_report(_untouched)
check("the report document is not mutated", _untouched, {"config": {}, "schema": "laya-evals-report/1"})

# ------------------------------------------------------ identity vocabulary pinned to laya.evals
# `laya/evidence.py` copies the report-identity vocabulary instead of importing `laya.evals` (numpy at
# module scope), and a copy is only safe while it is held against the original -- the same way
# `tests/test_conformal_abstention.py` holds `confidence.BUCKET_QTYPES` against `common.temp_bucket`.
# Unpinned, a key renamed upstream would make every report read as identity-incomplete here, and a key
# added upstream would read as identity-complete; both silently, in the verdict this module exists to
# give. These checks fail in both directions, on either side of the copy.
from laya import evals  # noqa: E402

check("the copied identity vocabulary is evals' own, in order",
      list(evidence._IDENTITY_KEYS), list(evals._IDENTITY_KEYS))
check("the provenance key is recorded upstream but not compared there",
      [k for k in evidence._PROVENANCE_KEYS if k not in evals._IDENTITY_KEYS],
      list(evidence._PROVENANCE_KEYS))
check("checkpoint <-> report conflicts are the identity vocabulary minus the report tag",
      sorted(set(evals._IDENTITY_KEYS) - set(evidence._IDENTITY_CONFLICT_KEYS)), ["schema"])
check("inspect_report reads exactly its declared vocabulary",
      sorted(evidence.inspect_report({"config": {}})["fields"]),
      sorted(evidence._IDENTITY_KEYS + evidence._PROVENANCE_KEYS))

# The copy has to hold for the *read* as well, or the two halves of the toolchain disagree about
# which document says what: `_cmd_compare` only began refusing a candidate whose top-level `schema`
# differed from its baseline once it started reading through `evals._identity_of`. Same documents,
# same answer. `laya_version` is excluded because this module reports it and `evals` never looks
# for it.
_shadow = {"config": {"schema": "in config"}, "schema": "at the top level"}
for _doc in (_noblock, {"config": _noblock}, toplevel, full_report, {"overall": {}},
             {"config": {"dataset_sha256": "a"}, "schema": "laya-evals-report/1"}, _shadow):
    _here = {k: v for k, v in evidence._report_identity(_doc).items() if k in evidence._IDENTITY_KEYS}
    _there = {k: v for k, v in evals._identity_of(_doc).items() if k in evidence._IDENTITY_KEYS}
    check("the identity read is evals' own for %s" % json.dumps(_doc, sort_keys=True), _here, _there)
# Equality alone would still pass if both sides started preferring the document over `config`: the
# fallback is for a key `config` does not carry, not a way to shadow one it does.
check("config still outranks the top level", evidence._report_identity(_shadow)["schema"], "in config")

# ... and `schema` stays out of the conflict set on purpose: two artifacts that agree on the
# deterministic digests but carry different report tags are still the same measurement. Both sides
# have to expose the tag for the check to say anything: with the checkpoint silent, the pair is
# skipped as "one side missing" whatever the conflict set holds.
check("a differing schema is not a checkpoint <-> report conflict",
      evidence.relate({"dataset_sha256": "a"},
                      {"config": {"dataset_sha256": "a", "schema": "laya-evals-report/2"}})["state"],
      evidence.UNKNOWN)
check("not even when both artifacts expose it",
      evidence.relate({"dataset_sha256": "a", "schema": "laya-evals-report/1"},
                      {"config": {"dataset_sha256": "a", "schema": "laya-evals-report/2"}})["state"],
      evidence.UNKNOWN)

# `docs/evals.md` carries the same vocabulary a third time -- it teaches that the identity fields
# "are reported with the same semantics as `run`", and names the pair a conflict is claimed on -- so
# the page is pinned to the code as well, the way `test_docs_and_the_cli_name_the_same_flags` pins
# the flag list. A missing sentence reads as an empty list, which fails the check rather than
# crashing the suite.
import re  # noqa: E402

_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "evals.md")
with open(_DOCS, encoding="utf-8") as _handle:
    _page = " ".join(_handle.read().split())
_named = _page.split("identity fields (", 1)[1].split(")", 1)[0] if "identity fields (" in _page else ""
check("the documented identity fields are the ones the code reads",
      sorted(re.findall(r"`([^`]+)`", _named)),
      sorted(evidence._IDENTITY_KEYS + evidence._PROVENANCE_KEYS))
_claimed = (_page.split("both artifacts expose the same ", 1)[1].split("with different values", 1)[0]
            if "both artifacts expose the same " in _page else "")
check("the documented conflict pair is the one the code compares",
      sorted(re.findall(r"`([^`]+)`", " ".join(_claimed.split("/")))),
      sorted(evidence._IDENTITY_CONFLICT_KEYS))

# ------------------------------------------------------ relationship
config = {"training": {"laya_train": {"laya_version": "0.3.28"}}}
match = evidence.relate(config, {"config": {"laya_version": "0.3.28"}})
check("same laya_version alone must NOT prove PRESENT", match["state"], evidence.UNKNOWN)
conflict = evidence.relate(config, {"config": {"laya_version": "0.3.20"}})
check("different laya_version must NOT become INCOMPARABLE", conflict["state"], evidence.UNKNOWN)
conflict_ds = evidence.relate({"dataset_sha256": "a"}, {"config": {"dataset_sha256": "b"}})
check("deterministic identity conflict -> INCOMPARABLE", conflict_ds["state"], evidence.INCOMPARABLE)
unknown = evidence.relate(config, {"config": {}})
check("no shared identity field -> UNKNOWN", unknown["state"], evidence.UNKNOWN)

# The conflict is the disagreement, not the place it was written. `relate` used to read the report's
# `config` block only, so this document -- the shape `act_head_eval.py` writes -- stated a digest the
# checkpoint contradicts and still answered "no immutable shared checkpoint identity".
check("deterministic identity conflict at the top level -> INCOMPARABLE",
      evidence.relate({"dataset_sha256": "a"}, {"dataset_sha256": "b"})["state"], evidence.INCOMPARABLE)
check("top-level agreement is still never a claimed match",
      evidence.relate({"dataset_sha256": "a"}, {"dataset_sha256": "a"})["state"], evidence.UNKNOWN)
# Provenance rides through the same read because `inspect_report` reports it; it must not widen what
# `relate` can conflict on.
check("a differing laya_version at the top level is still not a conflict",
      evidence.relate({"laya_version": "0.3.28"}, {"laya_version": "0.3.20"})["state"], evidence.UNKNOWN)

# ------------------------------------------------------ CLI on a realistic artifact pair
tmp = tempfile.mkdtemp()
ckpt = os.path.join(tmp, "ckpt")
os.makedirs(ckpt)
with open(os.path.join(ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as f:
    json.dump({"training": {"laya_train": {"laya_version": "0.3.28"}, "laya_train_calibration": {
        "choice": {"items": 400, "temperature": 1.1, "issues": []},
        "score": {"items": 30, "temperature": 1.0, "issues": ["fitted on only 30 calibration items"]}}}}, f)
report_path = os.path.join(tmp, "report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({"config": {"schema": "laya-evals-report/1", "dataset_sha256": "a",
                          "questions_sha256": "b", "laya_version": "0.3.28"}, "overall": {}}, f)

rc = evals_cli.main(["evidence", "--checkpoint", ckpt, "--report", report_path])
check("evidence CLI returns 0 on realistic pair", rc, 0)

rc_bad = evals_cli.main(["evidence", "--checkpoint", os.path.join(tmp, "nope")])
check("missing checkpoint dir -> usage error 2", rc_bad, 2)

# ... but only that one shape of an unreadable path. `--report` pointing at a directory raised
# `IsADirectoryError`, an `OSError` that is not a `FileNotFoundError`, so it walked past this
# handler and out of `main`: a traceback and exit 1, the code `laya-evals` reserves for a quality
# failure, on the run before it printed anything. The other rows were already usage errors and stay
# so -- they are here because widening the catch to `OSError` must not swallow them, and because a
# decode failure named no path, which matters most when the command takes two of them.
import contextlib
import io

_dir_report = os.path.join(tmp, "adir")
os.makedirs(_dir_report)
_not_object = os.path.join(tmp, "not-an-object.json")
with open(_not_object, "w", encoding="utf-8") as f:
    json.dump([1, 2], f)
_not_json = os.path.join(tmp, "not-json.json")
with open(_not_json, "w", encoding="utf-8") as f:
    f.write("{nope")
_not_text = os.path.join(tmp, "not-text.json")
with open(_not_text, "wb") as f:
    f.write(b"\xff\xfe\x00\x01binary")
# `json` reports unclosable depth as a `RecursionError`, which is neither a `ValueError` decode
# failure nor an `OSError`, so it is the one shape of unreadable artifact that used to reach the
# caller as a traceback.
_deep_report = os.path.join(tmp, "deep.json")
with open(_deep_report, "w", encoding="utf-8") as f:
    f.write("[" * 100000 + "]" * 100000)
_bad_config_ckpt = os.path.join(tmp, "ckpt-bad-config")
os.makedirs(_bad_config_ckpt)
with open(os.path.join(_bad_config_ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as f:
    f.write("{nope")

# The last tuple element is the string the refusal must contain. For five rows that is the whole
# path, because `laya/evidence.py` builds those messages from the argument the caller passed. The
# directory row is the exception: its refusal is the operating system's own `OSError` text, and the
# Windows CI runner reports that path in 8.3 form (`C:\Users\RUNNER~1\...` where the test holds
# `C:\Users\runneradmin\...`), so an exact full-path match there tests the runner's temp layout
# rather than the message. Its fixture is named `adir` -- short enough to be its own 8.3 name --
# and the leaf is what the row asserts, because naming the file the caller passed is the property.
for _label, _argv, _expect in (
        ("--report is a directory", ["evidence", "--checkpoint", ckpt, "--report", _dir_report],
         os.path.basename(_dir_report)),
        ("--report is JSON but not an object",
         ["evidence", "--checkpoint", ckpt, "--report", _not_object], _not_object),
        ("--report is not JSON", ["evidence", "--checkpoint", ckpt, "--report", _not_json],
         _not_json),
        ("--report is not UTF-8 text", ["evidence", "--checkpoint", ckpt, "--report", _not_text],
         _not_text),
        ("--report is JSON nested too deep to read",
         ["evidence", "--checkpoint", ckpt, "--report", _deep_report], _deep_report),
        ("the checkpoint's own config is not JSON",
         ["evidence", "--checkpoint", _bad_config_ckpt],
         os.path.join(_bad_config_ckpt, "rl_agent_config.json"))):
    _err = io.StringIO()
    with contextlib.redirect_stderr(_err):
        _rc = evals_cli.main(_argv)
    check("evidence CLI: %s -> usage error 2" % _label, _rc, 2)
    check_true("evidence CLI: %s prints no traceback" % _label, "Traceback" not in _err.getvalue())
    check_true("evidence CLI: %s names the path" % _label, _expect in _err.getvalue())

# The table asserts the path because that is the property all six rows share. For the depth row the
# *kind* has to be pinned separately: a refusal that reused "is not valid JSON" would still exit 2 and
# still name the file, and would tell the caller their artifact is malformed when it is only nested
# deeper than `json` walks -- a different fix. Reusing the decode wording is the mistake this checks.
_err = io.StringIO()
with contextlib.redirect_stderr(_err):
    _rc = evals_cli.main(["evidence", "--checkpoint", ckpt, "--report", _deep_report])
check("evidence CLI: a too-deep --report is still usage error 2", _rc, 2)
check_true("evidence CLI: a too-deep --report is refused as depth",
           "nested deeper than it can be read" in _err.getvalue())
check_true("and not as a document that fails to parse", "is not valid JSON" not in _err.getvalue())

# ... and on the artifact the research harness actually writes: identity at the document's top
# level, no `config` key. Read through `inspect_checkpoint`/`format_summary`, which is the whole of
# what `evidence` prints, so this covers the summary's own shape as well as the verdicts.
_digest_ckpt = os.path.join(tmp, "ckpt-digests")
os.makedirs(_digest_ckpt)
with open(os.path.join(_digest_ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as f:
    json.dump({"training": {"laya_train": {"dataset_sha256": "a"}}}, f)
_toplevel_report = os.path.join(tmp, "act-head-style.json")
with open(_toplevel_report, "w", encoding="utf-8") as f:
    json.dump({"schema": "laya-evals-report/1", "dataset_sha256": "b", "questions_sha256": "c"}, f)
_summary = evidence.format_summary(evidence.inspect_checkpoint(_digest_ckpt, _toplevel_report))
check_true("a top-level report identity reaches the summary as PRESENT",
           "eval report identity: PRESENT" in _summary)
check_true("and its differing digest reaches it as the conflict it names",
           "checkpoint<->report: INCOMPARABLE" in _summary)
check_true("every identity field is printed for a report with no config block",
           all(("  %-18s" % key) in _summary for key in evidence._IDENTITY_KEYS))

# ------------------------------------------------------ no torch required
import subprocess
_BLOCK = (
    "import sys, os, json, tempfile\n"
    "sys.path.insert(0, os.getcwd())\n"
    "class B:\n"
    "    def find_spec(self, name, path, target=None):\n"
    "        if name == 'torch' or name.startswith('torch.'):\n"
    "            raise ImportError('torch blocked')\n"
    "sys.meta_path.insert(0, B())\n"
    "tmp = tempfile.mkdtemp(); ckpt = os.path.join(tmp, 'c'); os.makedirs(ckpt)\n"
    "with open(os.path.join(ckpt, 'rl_agent_config.json'), 'w') as f:\n"
    "    json.dump({'training': {'laya_train_calibration': {'choice': {'items': 5, 'temperature': 1.0, 'issues': []}}}}, f)\n"
    "from laya import evals_cli, evidence\n"
    "rc = evals_cli.main(['evidence', '--checkpoint', ckpt])\n"
    "res = evidence.inspect_checkpoint(ckpt)\n"
    "print('RC', rc, 'TORCH', 'torch' in sys.modules)\n"
)
proc = subprocess.run([sys.executable, "-c", _BLOCK], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      capture_output=True, text=True, timeout=120)
print(proc.stdout.strip(), proc.stderr.strip()[-300:])
out = proc.stdout
check_true("evidence CLI works with torch blocked", "RC 0" in out)
check_true("torch not imported when blocked", "TORCH False" in out)

if FAIL:
    print("\nFAILURES:", FAIL)
    sys.exit(1)
print("\nall %d checks passed" % len(PASS))
