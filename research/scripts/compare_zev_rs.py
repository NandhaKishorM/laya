#!/usr/bin/env python3
"""Render a public Zev RS comparison summary against Laya.

This repo already tracks Jev comparisons in README.md and BENCHMARKS.md. This helper keeps
that comparison style explicit for Zev RS without requiring a local Zev benchmark run.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "results" / "zev_rs_public_comparison.json"


def main() -> int:
    with ARTIFACT.open(encoding="utf-8") as fh:
        data = json.load(fh)

    zev_default = data["zev_default"]["accuracy"]
    zev_apfel = data["zev_apfel"]["accuracy"]
    laya_acc = data["laya"]["accuracy"]
    speedup = data["comparison"]["latency_speedup_vs_laya"]

    print("Zev RS vs Laya (public benchmark comparison)")
    print("Source:", data["source"])
    print()
    print("Metric                Zev RS            Laya")
    print("-------------------  ----------------  ----------------")
    print(f"Accuracy              {zev_apfel:.3%}           {laya_acc:.3%}")
    print(f"Latency (p50)         {data['zev_apfel']['latency_ms']:.3f} ms      {data['laya']['latency_ms']:.1f} ms")
    print(f"Speedup               {speedup:.0f}x")
    print()
    print(f"Public Zev numbers: Default {zev_default:.3%}, Apfel {zev_apfel:.3%}.")
    print("This is a marketing/public comparison, not a fresh local benchmark run in this repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
