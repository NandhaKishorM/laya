#!/usr/bin/env python
"""Render benchmarks/bench-rtx4070.png from the three result JSONs.

    python benchmarks/plot_results.py

Panels: (1) agent-level speedup per checkpoint, (2) router-layer mixed EN/Hindi
states/s, (3) consistency max|dp| per checkpoint with the 2e-2 bar.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def main():
    agent = load("results-rtx4070.json")
    router = load("results-router-rtx4070.json")
    consist = load("results-consistency-rtx4070.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("predict_batch on RTX 4070 — laya %s" % os.environ.get("LAYA_VER", "0.3.11"),
                 fontsize=13, fontweight="bold")

    # panel 1: agent-level speedup
    ax = axes[0]
    for row in agent["bench"]:
        by_states = {}
    per_ckpt = {}
    for row in agent["bench"]:
        per_ckpt.setdefault(row["checkpoint"], []).append(row)
    for ckpt, rows in per_ckpt.items():
        rows.sort(key=lambda r: r["states"])
        ax.plot([r["states"] for r in rows], [r["speedup"] for r in rows],
                marker="o", label=ckpt)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("states per batch")
    ax.set_ylabel("speedup (x)")
    ax.set_title("Agent.predict_batch vs one-by-one\n(4 questions/state, median of 5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # panel 2: router mixed-language throughput
    ax = axes[1]
    rows = sorted(router["router"], key=lambda r: r["states"])
    xs = [r["states"] for r in rows]
    ax.plot(xs, [r["looped_sps"] for r in rows], marker="o", label="one predict() per request")
    ax.plot(xs, [r["batched_sps"] for r in rows], marker="o", label="Router.predict_batch")
    for r in rows:
        ax.annotate("%.1fx" % r["speedup"], (r["states"], r["batched_sps"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("requests (50/50 EN/Hindi)")
    ax.set_ylabel("states/s")
    ax.set_title("Router layer: mixed EN/Hindi\n(regroups by checkpoint)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # panel 3: consistency
    ax = axes[2]
    rc = router.get("router_consistency", {})
    entries = [(r["checkpoint"], r["max_delta"]) for r in consist["consistency"]]
    if rc:
        entries.append(("router\n(mixed EN/Hindi)", rc["max_delta"]))
    names = [e[0] for e in entries]
    deltas = [e[1] for e in entries]
    bars = ax.bar(names, deltas, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][:len(names)])
    ax.axhline(0.02, color="red", ls="--", lw=1, label="2e-2 bar")
    ax.set_ylabel("max |Δp| (label-attached)")
    ax.set_title("batch vs single consistency\n(%d answers, 0 flips)" %
                 (sum(r["compared"] for r in consist["consistency"]) + rc.get("compared", 0)))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, delta, "%.4f" % delta,
                ha="center", va="bottom", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "bench-rtx4070.png")
    plt.savefig(out, dpi=110)
    print("wrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())
