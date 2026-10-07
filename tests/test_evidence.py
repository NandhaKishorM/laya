"""Contract guard for laya.evidence: readiness evidence states over existing artifacts.

Run: python tests/test_evidence.py
"""
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
check("below MIN_TYPE_N -> INSUFFICIENT", low["types"]["choice"]["state"], evidence.INSUFFICIENT)

warned = calib({"training": {"laya_train_calibration": {"choice": {"items": 30, "temperature": 1.2,
        "issues": ["fitted on only 30 calibration items"]}}}})
check("fitted below CALIB_WARN_N -> INSUFFICIENT", warned["types"]["choice"]["state"], evidence.INSUFFICIENT)

clamped = calib({"training": {"laya_train_calibration": {"choice": {"items": 200, "temperature": 0.5,
        "issues": ["the fitted temperature 0.5 is on the [0.5, 3] clamp"]}}}})
check("clamped fit stays a visible limitation -> INSUFFICIENT", clamped["types"]["choice"]["state"], evidence.INSUFFICIENT)

clean = calib({"training": {"laya_train_calibration": {"choice": {"items": 400, "temperature": 1.1, "issues": []}}}})
check("clean supported fit -> PRESENT", clean["types"]["choice"]["state"], evidence.PRESENT)
check("all-clean overall -> PRESENT", clean["state"], evidence.PRESENT)

mixed = calib({"training": {"laya_train_calibration": {
    "choice": {"items": 400, "temperature": 1.1, "issues": []},
    "score": {"items": 0, "temperature": 1.0, "issues": []}}}})
check("mixed types overall -> INSUFFICIENT", mixed["state"], evidence.INSUFFICIENT)

malformed = calib({"training": {"laya_train_calibration": {"choice": {"temperature": 1.0}}}})
check("malformed entry -> UNKNOWN", malformed["types"]["choice"]["state"], evidence.UNKNOWN)

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
