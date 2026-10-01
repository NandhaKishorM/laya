"""Load supported file-backed secrets, then replace this process with the command; on Windows,
run the child and propagate its exit code instead."""
import os
from pathlib import Path
import subprocess
import sys

# Both names are read as `<NAME>_FILE` and then moved into `<NAME>`, so a secret can be
# mounted as a file instead of passed in the environment.
SECRET_NAMES = ("HF_TOKEN", "LAYA_API_KEY")


def main():
    for name in SECRET_NAMES:
        filename = os.environ.get(name + "_FILE")
        if not filename:
            continue
        try:
            # `utf-8-sig`, not `utf-8`: a secret file written by a Windows editor starts
            # with a UTF-8 byte-order mark (PowerShell 5.1's `utf8` encoding and
            # Notepad's default both write one -- this file's audience includes the
            # `os.name == "nt"` path below). Read as plain utf-8 that mark is U+FEFF,
            # which `.strip()` does not remove because it is not whitespace, so the
            # token would arrive with one leading character no consumer accepts.
            # `utf-8-sig` reads a BOM when present and plain UTF-8 otherwise, and a
            # genuine UTF-8 error still fails into the `UnicodeError` below.
            value = Path(filename).read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError, ValueError):
            sys.exit(f"Cannot read {name}_FILE")
        if not value or "\0" in value:
            sys.exit(f"Invalid empty or NUL-containing secret in {name}_FILE")
        os.environ[name] = value
        os.environ.pop(name + "_FILE", None)
    if len(sys.argv) < 2:
        sys.exit("A container command is required")
    if os.name == "nt":
        # os.execvp is an emulation here: it never replaces the process, and the C runtime
        # rebuilds the command line by joining argv with spaces, so a quoted argument (a path
        # with a space, a `-c "<script>"`) arrives split and the child dies before it runs.
        # Run the child and propagate its exit code instead. The container itself is Linux,
        # where the exec below keeps the replacement the docstring promises.
        sys.exit(subprocess.run(sys.argv[1:]).returncode)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
