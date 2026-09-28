"""Model-free Swedish/Danish/English routing diagnostic over MASSIVE.

Run from the repository root:
    python research/evals/swedish_route_diagnostic.py

The dataset revision is pinned so the counts stay reproducible. No model weights are loaded.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from laya.lang import guess_latin_language
from laya.router import Router


DATASET = "mteb/amazon_massive_intent"
REVISION = "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
LANGUAGES = ("sv", "da", "en")


def main() -> int:
    router = Router()
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": "test",
        "model_free": True,
        "languages": {},
    }

    for language in LANGUAGES:
        path = hf_hub_download(
            repo_id=DATASET,
            filename="test/%s.json.gz" % language,
            repo_type="dataset",
            revision=REVISION,
        )
        guesses: Counter[str] = Counter()
        routes: Counter[str] = Counter()
        guess_to_route: dict[str, Counter[str]] = defaultdict(Counter)
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                guess = guess_latin_language(row["text"]) or "undecided"
                decision = router.route(row["text"])
                guesses[guess] += 1
                routes[decision.model] += 1
                guess_to_route[guess][decision.model] += 1
                count += 1
        report["languages"][language] = {
            "examples": count,
            "language_guesses": dict(sorted(guesses.items())),
            "router_choices": dict(sorted(routes.items())),
            "guess_to_router": {
                guess: dict(sorted(per_route.items()))
                for guess, per_route in sorted(guess_to_route.items())
            },
        }

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
