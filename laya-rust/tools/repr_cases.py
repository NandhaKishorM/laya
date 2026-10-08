"""Deterministic, model-free Python float-repr oracle for Rust's JSON formatter.

Store IEEE-754 bits, not JSON numbers: parsing a decimal must not change the
input float before the differential test reaches PythonJson::repr.
"""

import json
import math
import os
import random
import struct


def build_repr_probe():
    bits = {0x43179085685D83C9}

    def add(value):
        bits.add(struct.unpack(">Q", struct.pack(">d", value))[0])

    for exponent in range(50, 54):
        for quarter in range(-16, 33):
            value = float(2 ** exponent) + quarter / 4
            add(value)
            add(-value)
    for raw in [0, 1, 2, 3, 15, 0x000FFFFFFFFFFFFE, 0x000FFFFFFFFFFFFF,
                0x0010000000000000, 0x0010000000000001, 0x7FEFFFFFFFFFFFFF]:
        bits.add(raw)
        bits.add(raw | (1 << 63))
    for value in [1e-5, 1e-4, 1e16, 1e22]:
        for neighbour in [math.nextafter(value, -math.inf), value,
                          math.nextafter(value, math.inf)]:
            add(neighbour)
            add(-neighbour)

    rng = random.Random("laya-repr-v1")
    accepted = 0
    while accepted < 2000:
        raw = rng.getrandbits(64)
        if (raw >> 52) & 0x7FF == 0x7FF:
            continue
        bits.add(raw)
        accepted += 1

    return [
        {"bits": "%016x" % raw,
         "repr": repr(struct.unpack(">d", struct.pack(">Q", raw))[0])}
        for raw in sorted(bits)
    ]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = build_repr_probe()
    for checkpoint in ["english", "multilingual", "typed-decisions"]:
        out = os.path.join(root, "tests", "golden", checkpoint)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "repr_probe.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(probe, f, indent=2)
            f.write("\n")
    print("recorded %d finite float repr probes per checkpoint" % len(probe))


if __name__ == "__main__":
    main()
