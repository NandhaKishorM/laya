#!/usr/bin/env python3
"""Record internal/demo/testdata/email_questions.json: laya.presets.email_questions() minus
is_phishing, as json.dumps(ensure_ascii=False) writes it.

    python laya-go/samples/scripts/gen_email_golden.py          # write
    python laya-go/samples/scripts/gen_email_golden.py --check  # fail if the file differs
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from laya.presets import email_questions  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "internal" / "demo" / "testdata" / "email_questions.json"


def render():
    questions = email_questions()
    del questions["is_phishing"]
    return json.dumps(questions, ensure_ascii=False, indent=1) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit("%s differs from what Python records now; rerun without --check" % OUT)
        print("email_questions.json matches")
        return
    OUT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
