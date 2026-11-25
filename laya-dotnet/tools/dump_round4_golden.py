"""Record golden vectors for `Calibration.Round4` (Python's builtin `round(x, 4)` on floats).

Pure Python, standard library only. Writes
`laya-dotnet/tests/Laya.Tests/golden/round4/round4_python.txt`, one `input_hex output_hex` pair
per line (`float.hex`, so every double is exact): 20,000 seeded inputs made of

    8,000 uniform in [0, 1)           probabilities and confidences
    8,000 uniform in [0, 10)          expected scores
    2,000 exact binary ties           k / 2**j for j in 5..12, either sign
    2,000 decimal near-ties           k / 10**d + a few ulps either way, either sign

Run: python laya-dotnet/tools/dump_round4_golden.py
"""

import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "Laya.Tests" / "golden" / "round4" / "round4_python.txt"


def inputs():
    rng = random.Random(20260519)
    for _ in range(8000):
        yield rng.random()
    for _ in range(8000):
        yield rng.random() * 10.0
    for _ in range(2000):
        j = rng.randint(5, 12)
        yield rng.choice((-1.0, 1.0)) * rng.randint(0, 10 * 2**j) / 2**j
    for _ in range(2000):
        d = rng.choice((4, 5))
        x = rng.randint(0, 10 * 10**d) / 10**d
        for _ in range(rng.randint(0, 3)):
            x = math.nextafter(x, rng.choice((-math.inf, math.inf)))
        yield rng.choice((-1.0, 1.0)) * x


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="\n") as f:
        for x in inputs():
            f.write(f"{x.hex()} {round(x, 4).hex()}\n")


if __name__ == "__main__":
    main()
