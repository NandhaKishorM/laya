"""Assert decoded parity across benchmark logs that include answers.

python benchmarks/compare_compile_defaults.py reference.log reduced.log restart.log
"""
import json
import sys
from pathlib import Path


def read(path):
    return json.loads(next(line[7:] for line in Path(path).read_text().splitlines()
                           if line.startswith("RESULT ")))


def compare(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max((compare(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list):
        assert len(left) == len(right)
        return max((compare(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, float):
        return abs(left - right)
    assert left == right, (left, right)
    return 0.0


if __name__ == "__main__":
    reference = read(sys.argv[1])["answers"]
    for path in sys.argv[2:]:
        delta = compare(reference, read(path)["answers"])
        assert delta <= 1e-5, delta
        print("%s: categorical parity passed; max numerical delta %.9g" % (path, delta))
