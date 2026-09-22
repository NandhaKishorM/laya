"""Command-line interface for testing Laya locally without writing a script.

Examples
--------
Pick one of several options (choice)::

    laya "the customer was charged twice" \
        --instructions "which department should handle this?" \
        --options "billing:invoices,payments,refunds;technical:bugs,outages;sales:pricing"

Rate on an ordered scale (score)::

    laya "the server is completely down" --type score \
        --instructions "how urgent is this?" \
        --levels "not urgent,soon,critical"

Answer a yes/no question (noul)::

    laya "cancel my subscription now" --type noul \
        --instructions "does the user threaten to leave?"

The state text can also come from a file or from stdin::

    laya --file state.txt --instructions "department?" --options "billing:...;sales:..."
    echo "refund my duplicate charge" | laya --instructions "department?" --options "billing:...;sales:..."

Pass ``--json`` for the raw response, or ``--model`` / ``--lang`` to pin routing.
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

_QTYPE_CHOICES = ("choice", "score", "noul")


def _split_criteria_options(raw: str) -> Dict[str, Optional[str]]:
    """Parse ``name:desc;name2:desc2`` into an ordered criteria dict."""
    criteria: Dict[str, Optional[str]] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, desc = chunk.split(":", 1)
            name, desc = name.strip(), desc.strip()
        else:
            name, desc = chunk.strip(), None
        if name:
            criteria[name] = desc or None
    return criteria


def _split_levels(raw: str) -> List[str]:
    """Parse a comma-separated ordered list of score levels."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_question(args: argparse.Namespace) -> Dict[str, Any]:
    """Turn parsed CLI arguments into a single typed question definition."""
    qtype = args.type
    if qtype == "choice":
        if not args.options:
            raise ValueError("choice questions need --options 'name:desc;name2:desc2'")
        criteria = _split_criteria_options(args.options)
        if len(criteria) < 2:
            raise ValueError("choice questions need at least two options")
        return {"q1": {"type": "choice", "instructions": args.instructions, "criteria": criteria}}
    if qtype == "score":
        if not args.levels:
            raise ValueError("score questions need --levels 'low,medium,high'")
        levels = _split_levels(args.levels)
        if len(levels) < 2:
            raise ValueError("score questions need at least two levels")
        return {"q1": {"type": "score", "instructions": args.instructions, "criteria": levels}}
    return {"q1": {"type": "noul", "instructions": args.instructions or "Is this true?"}}


def format_answer(qid: str, ans: Dict[str, Any]) -> str:
    """Render a single answer as a compact, human-readable block."""
    lines: List[str] = []
    conf = ans.get("confidence")
    conf_s = "  (confidence %.1f%%)" % (conf * 100) if conf is not None else ""

    if ans.get("type") == "choice":
        choice = ans["choice"]
        lines.append("%s -> %s *%s" % (qid, choice, conf_s))
        probs = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])
        width = max(len(k) for k, _ in probs) if probs else 0
        for name, p in probs:
            bar = "█" * int(round(p * 40))
            lines.append("  %-*s  %5.1f%%  %s" % (width, name, p * 100, bar))
    elif ans.get("type") == "score":
        lines.append("%s -> expected score %.2f%s" % (qid, ans["score"], conf_s))
        legend = ans.get("legend", {})
        for k in sorted(ans.get("probabilities", {}), key=int):
            label = legend.get(k, "level %s" % k)
            p = ans["probabilities"][k]
            bar = "█" * int(round(p * 40))
            lines.append("  %-16s  %5.1f%%  %s" % (label, p * 100, bar))
    else:  # noul
        lines.append("%s -> P(true) = %.1f%%%s" % (qid, ans["noul"] * 100, conf_s))
    return "\n".join(lines)


def format_result(result: Dict[str, Any]) -> str:
    """Render a full ``Router.predict`` payload as text."""
    out: List[str] = []
    routing = result.get("routing")
    if routing:
        out.append("route: %s (%s)" % (routing.get("model"), routing.get("reason", "")))
    for qid, ans in (result.get("answers") or {}).items():
        out.append(format_answer(qid, ans))
    return "\n".join(out)


def _read_state(args: argparse.Namespace) -> str:
    """Resolve the state text from a positional arg, --file, or piped stdin."""
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    if args.text is not None:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _friendly_error(exc: Exception) -> str:
    """Turn a load/inference failure into an actionable message."""
    text = str(exc)
    lowered = text.lower()
    if "offline" in lowered or "connection" in lowered or "couldn't connect" in lowered or "timed out" in lowered:
        return ("could not reach the Hugging Face Hub to download the model; "
                "check your network (or set HF_TOKEN for gated repos). Details: %s" % text)
    if "huggingface" in lowered or "hf_hub" in lowered or "repository" in lowered or "401" in text:
        return ("model download failed; verify the repo name and that any required "
                "access token is set via HF_TOKEN. Details: %s" % text)
    return text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya",
        description="Test Laya locally: route a state through the decision engine and "
                    "print each option's probability.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="the state text to classify (or pipe via stdin, or use --file)",
    )
    parser.add_argument(
        "-t", "--type",
        choices=_QTYPE_CHOICES,
        default="choice",
        help="question type: choice (pick one), score (ordered levels), or noul (yes/no) "
             "(default: choice)",
    )
    parser.add_argument(
        "-i", "--instructions",
        help="the question to answer, e.g. 'which department should handle this?'",
    )
    parser.add_argument(
        "-o", "--options",
        help="choice criteria as 'name:desc;name2:desc2' (desc optional)",
    )
    parser.add_argument(
        "-l", "--levels",
        help="score levels as a comma-separated ordered list, e.g. 'low,medium,high'",
    )
    parser.add_argument(
        "-m", "--model",
        help="pin a checkpoint: english, multilingual, or typed-decisions",
    )
    parser.add_argument(
        "--lang",
        help="hint the router with a language tag, e.g. 'de' or 'zh'",
    )
    parser.add_argument(
        "-f", "--file",
        help="read the state text from this file instead of a positional arg",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the raw JSON response instead of the formatted text",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_laya_version(),
    )
    return parser


def _laya_version() -> str:
    """Best-effort package version without importing the torch-heavy package."""
    try:
        from importlib.metadata import version

        return version("laya")
    except Exception:  # pragma: no cover - fallback for an editable install edge case
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    state = _read_state(args)
    if not state.strip():
        parser.error("no state text given; pass it as an argument, via --file, or pipe it on stdin")

    if args.type in ("choice", "score") and not args.instructions:
        parser.error("--instructions is required for %s questions" % args.type)

    try:
        questions = build_question(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        from .router import Router

        router = Router()
        result = router.predict(state, questions, model=args.model, lang=args.lang)
    except Exception as exc:  # noqa: BLE001 - surface a helpful message for any failure
        print("error: %s" % _friendly_error(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
