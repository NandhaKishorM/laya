"""Parity evidence for the TileLang fast path: `benchmarks/parity.py --backend tilelang`.

    python benchmarks/parity_fast.py [--subfolder multilingual] [--dtype bf16|fp16] [--json benchmarks/results/parity_<name>.json]

Kept as the documented entry point; the harness that answers the same fixed 60-state / 288-question set for every
backend lives in `parity.py`.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity import main  # noqa: E402

if __name__ == "__main__":
    main(["--backend", "tilelang"] + [x for x in sys.argv[1:] if x != "--backend"])
