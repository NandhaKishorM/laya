"""LAYA_DEVICE normalisation *and* parseability: the laya.mcp.device env contract.

The variable is read once, in `env_device`, and the same string is handed to torch
(``Router(device=...)``) and reported by `resolve_device` / `laya_status`. Case and
whitespace normalisation was already covered (``CUDA:0 -> cuda:0``); what this suite
adds is the other half of "torch can use what we return": ``cuda:`` (or ``gpu``, or
``cuda: 0``) normalises cleanly and is still refused by torch -- so on a lazy Router it
survived startup, was reported as the configured device, and the refusal landed inside
the first request that built a checkpoint. `env_device` now parses the value it returns
and raises a named ValueError where the environment is read.

No model weights, no network, no GPU: torch is imported for its parser only.
Run: python tests/test_mcp_device.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401  (the parser under test)
except ImportError:
    print("SKIP: torch not importable, so there is no device parser to check")
    sys.exit(0)

from laya.mcp.device import env_device, resolve_device  # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, (": " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""), flush=True)


def env_device_with(raw):
    """`env_device()` with LAYA_DEVICE set to raw (None = unset); the old value is restored."""
    old = os.environ.get("LAYA_DEVICE")
    try:
        if raw is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = raw
        return env_device()
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old


def rejection(raw):
    """The ValueError message `env_device()` raises for raw, or "" if it does not raise."""
    try:
        env_device_with(raw)
    except ValueError as exc:
        return str(exc)
    except BaseException as exc:  # noqa: BLE001 - a wrong exception is a failed case, not an abort
        return "wrong exception %r" % (exc,)
    return ""


def test_unset_and_blank_stay_none():
    ok("unset -> None", env_device_with(None) is None)
    ok("empty -> None", env_device_with("") is None)
    ok("spaces -> None", env_device_with("   ") is None)


def test_normalisation_still_applies_first():
    ok("cpu verbatim", env_device_with("cpu") == "cpu")
    ok("case + spaces + index", env_device_with("  CUDA:0 ") == "cuda:0")
    ok("index preserved", env_device_with("cuda:1") == "cuda:1")


def test_the_parser_is_the_judge_not_an_allowlist():
    # torch.device("meta") parses; a hand-written {cpu, cuda, mps} allowlist would not,
    # and would make a device this module has never heard of impossible to configure.
    ok("meta (valid for torch, never loadable here)", env_device_with("meta") == "meta")


def test_unparseable_values_are_rejected_by_name():
    for raw in ("cuda:", "gpu", "cuda: 0"):
        message = rejection(raw)
        ok("%r -> ValueError" % raw, bool(message) and "wrong exception" not in message, message or "no error")
        ok("%r names LAYA_DEVICE" % raw, "LAYA_DEVICE" in message, message)
        ok("%r quotes the offending value" % raw, raw.strip() in message, message)
    ok("'cuda:' carries torch's own reason",
       "Invalid device string" in rejection("cuda:"), rejection("cuda:"))
    ok("'gpu' carries torch's own list of device types",
       "Expected one of" in rejection("gpu"), rejection("gpu"))


def test_no_reporting_path_labels_a_refused_device():
    old = os.environ.get("LAYA_DEVICE")
    try:
        os.environ["LAYA_DEVICE"] = "cuda:"
        try:
            resolve_device()
        except ValueError:
            ok("resolve_device() refuses it too", True)
        else:
            ok("resolve_device() refuses it too", False, "returned a label for 'cuda:'")
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old
    # force= is the documented tests hook: it is returned verbatim, exactly like the
    # CLI --device flag, which never reads the environment.
    ok("force= hook stays verbatim", resolve_device("cuda:") == "cuda:")


def test_reported_equals_handed_to_torch():
    old = os.environ.get("LAYA_DEVICE")
    try:
        os.environ["LAYA_DEVICE"] = "CUDA:1"
        ok("env_device == resolve_device", env_device() == resolve_device() == "cuda:1")
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old


def main():
    for fn in (test_unset_and_blank_stay_none,
               test_normalisation_still_applies_first,
               test_the_parser_is_the_judge_not_an_allowlist,
               test_unparseable_values_are_rejected_by_name,
               test_no_reporting_path_labels_a_refused_device,
               test_reported_equals_handed_to_torch):
        fn()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for name in FAIL:
        print("  FAIL " + name)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
