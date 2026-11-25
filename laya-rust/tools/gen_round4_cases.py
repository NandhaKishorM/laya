"""Write tests/round4_python_cases.txt: Python's builtin ``round(x, 4)`` on 20,000 seeded doubles.

Each line is ``<input bits> <output bits> <repr(input)> <repr(output)>``; the bits are the
IEEE-754 pattern as 16 hex digits, so the Rust test compares exactly (``-0.0`` included). The
file is an output of this script only; never edit it by hand.

    python laya-rust/tools/gen_round4_cases.py

Writes LF, and ``.gitattributes`` pins the file to LF, so re-running this on any platform does
not report all 20,000 lines as rewritten.

Reproducible except for the ``kind == 4`` magnitudes: ``10 ** rng.uniform(-12, 14)`` goes through
the platform's libm ``pow``, which is allowed to differ by one ulp, and 3 of the 20,000 inputs do
differ between Windows and macOS. That is a corpus the port replays, not a contract it has to
reproduce -- each line already carries both the input and Python's answer for it -- so a re-run
that moves those 3 inputs is not a behaviour change. Do not commit such a re-run without saying
which platform produced it.
"""

import math
import random
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "round4_python_cases.txt"
COUNT = 20_000


def bits(x):
    return format(struct.unpack(">Q", struct.pack(">d", x))[0], "016x")


def values():
    rng = random.Random(20261019)
    out = []
    for d in (32, 64, 128, 160, 1024):  # exact ties and near-ties, both signs
        for k in range(-40, 41):
            out.append(k / d)
    out += [2.00005, 9.99995, 3.03125, 1e-5, 4.9999e-5, 5e-5, -5e-5, 1e-300, 5e-324, 0.0, -0.0]
    while len(out) < COUNT:
        kind = rng.randrange(6)
        if kind == 0:  # probabilities
            x = rng.random()
        elif kind == 1:  # values up to 10
            x = rng.uniform(0, 10)
        elif kind == 2:  # a 4-decimal value plus half a unit, nudged by a few ulps
            x = (rng.randrange(0, 100_000) + 0.5) / 10_000
            for _ in range(rng.randrange(-2, 3)):
                x = math.nextafter(x, math.inf if rng.random() < 0.5 else -math.inf)
        elif kind == 3:  # dyadic ties k / 2^n
            x = rng.randrange(0, 5000) / (1 << rng.randrange(5, 12))
        elif kind == 4:  # wide magnitude
            x = rng.choice((-1, 1)) * 10 ** rng.uniform(-12, 14)
        else:  # arbitrary finite bit patterns
            x = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
            if not math.isfinite(x):
                continue
        out.append(-x if rng.random() < 0.3 else x)
    return out[:COUNT]


def main():
    lines = [f"{bits(x)} {bits(round(x, 4))} {x!r} {round(x, 4)!r}" for x in values()]
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii", newline="")
    print(f"wrote {len(lines)} cases to {OUT}")


if __name__ == "__main__":
    main()
