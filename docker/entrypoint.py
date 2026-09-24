"""Load supported file-backed secrets, then replace this process with the command."""
import os
from pathlib import Path
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
            value = Path(filename).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError, ValueError):
            sys.exit(f"Cannot read {name}_FILE")
        if not value or "\0" in value:
            sys.exit(f"Invalid empty or NUL-containing secret in {name}_FILE")
        os.environ[name] = value
        os.environ.pop(name + "_FILE", None)
    if len(sys.argv) < 2:
        sys.exit("A container command is required")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
