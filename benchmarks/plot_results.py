#!/usr/bin/env python
"""Render benchmarks/bench-rtx4070.png from the three result JSONs.

    python benchmarks/plot_results.py

Panels: (1) agent-level speedup per checkpoint, (2) router-layer mixed EN/Hindi
states/s, (3) consistency max|dp| per checkpoint with the 2e-2 bar.
"""
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def declared_version():
    """The version the figure is labelled with: ``LAYA_VER`` overrides, pyproject.toml is the default.

    The default used to be a literal -- 0.3.11 -- which every release since the figure was first
    rendered has left behind: the bump commit touches laya/__init__.py and pyproject.toml, never
    this script, so a regeneration that does not set LAYA_VER labels a current run with the
    version the script was written in. Read as text, not with tomllib: this script runs on
    whatever Python a benchmarking machine has, and pyproject's version line is a literal.
    """
    override = os.environ.get("LAYA_VER")
    if override:
        return override
    try:
        with open(os.path.join(os.path.dirname(HERE), "pyproject.toml"), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return "unknown"
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else "unknown"


def main():
    agent = load("results-rtx4070.json")
    router = load("results-router-rtx4070.json")
    consist = load("results-consistency-rtx4070.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("predict_batch on RTX 4070 — laya %s" % declared_version(),
                 fontsize=13, fontweight="bold")

    # panel 1: agent-level speedup
    ax = axes[0]
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
    # Both halves of this claim are read from the JSONs being plotted.  "0 flips" used to be a
    # literal in the title, so a regeneration on a run that did register a flip would have
    # shipped a figure asserting the opposite of the data it was drawn from.
    compared = sum(r["compared"] for r in consist["consistency"]) + rc.get("compared", 0)
    flips = sum(r["flips"] for r in consist["consistency"]) + rc.get("flips", 0)
    ax.set_title("batch vs single consistency\n(%d answers, %d flips)" % (compared, flips))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, delta, "%.4f" % delta,
                ha="center", va="bottom", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "bench-rtx4070.png")
    plt.savefig(out, dpi=110)
    print("consistency: %d answers, %d flips -> %s" % (compared, flips, "PASS" if flips == 0 else "FAIL"))
    print("wrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())
