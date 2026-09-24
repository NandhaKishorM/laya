"""Command-line interface for testing Laya locally.

    laya "I was charged twice, please refund"           # routing decision only (no model download)
    laya "Refactor this service" --predict              # full answers, loads the checkpoint
    laya                                                # interactive mode
    laya "Mein Konto wurde zweimal belastet" --lang de  # explicit language
    laya "My payment failed twice" --preset triage      # a ready-made question preset

Routing (the default) never downloads a checkpoint, so it works offline and
returns in milliseconds. --predict loads the routed checkpoint on first use,
which needs network access to the Hugging Face hub. --preset answers one of the
ready-made question presets from laya.presets and implies --predict.
"""

import argparse
import json
import sys

import laya

# Ready-made question presets a prediction can run instead of router_questions().
PRESETS = {
    "email": laya.email_questions,
    "guard": laya.guard_questions,
    "moderation": laya.moderation_questions,
    "router": laya.router_questions,
    "triage": laya.triage_questions,
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="laya",
        description="Test Laya locally: route or answer a request from the command line.",
    )
    parser.add_argument("text", nargs="*", help="the request text (omit for interactive mode)")
    parser.add_argument("--predict", action="store_true",
                        help="run the full prediction, not just the routing decision (downloads the checkpoint on first use)")
    parser.add_argument("--model", choices=sorted(laya.DEFAULT_MODELS),
                        help="force a checkpoint instead of auto-routing")
    parser.add_argument("--lang", help="force a language, e.g. en or de, instead of detecting it")
    parser.add_argument("--task", help="force a typed-decisions workflow instead of detecting it")
    parser.add_argument("--preset", choices=sorted(PRESETS), metavar="NAME",
                        help="answer a ready-made question preset (%s) instead of the router questions; implies --predict"
                        % ", ".join(sorted(PRESETS)))
    parser.add_argument("--device", help="torch device, e.g. cpu or cuda")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    return parser


def make_router(args):
    return laya.Router(device=args.device, preload=False)


def show_decision(decision):
    print("Model     :", decision["model"])
    print("Reason    :", decision["reason"])
    detection = decision.get("detection")
    if detection:
        print("Detected  :", json.dumps(detection, ensure_ascii=False))


def show_answers(result):
    routing = result.get("routing")
    if routing:
        show_decision(routing)
        print()
    for qid, answer in result.get("answers", {}).items():
        if "choice" in answer:
            choice = answer["choice"]
            probability = answer.get("probabilities", {}).get(choice, 0.0)
            detail = "%s (p=%.3f)" % (choice, probability)
        elif "score" in answer:
            detail = "%.2f" % answer["score"]
        elif "noul" in answer:
            detail = "%.3f" % answer["noul"]
        else:
            detail = json.dumps(answer, ensure_ascii=False)
        print("%-12s: %s" % (qid, detail))


def run(text, args, router=None):
    """Route or predict one request; returns 0 on success, 2 on a handled error."""
    router = router or make_router(args)
    state = {"text": text}
    try:
        if args.predict or args.preset:
            questions = PRESETS[args.preset]() if args.preset else laya.router_questions()
            result = router.predict(state, questions,
                                    model=args.model, task=args.task, lang=args.lang)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            else:
                show_answers(result)
        else:
            decision = router.route(state, model=args.model, task=args.task, lang=args.lang)
            if args.json:
                print(json.dumps(dict(decision), ensure_ascii=False, indent=2, default=str))
            else:
                show_decision(decision)
    except ValueError as error:
        print("laya: %s" % error, file=sys.stderr)
        return 2
    except (ImportError, OSError, RuntimeError) as error:
        print("laya: could not run Laya (%s)." % error, file=sys.stderr)
        print("Check that the dependencies are installed and the checkpoints can be "
              "downloaded from the Hugging Face hub (network access is needed on first use).",
              file=sys.stderr)
        return 2
    return 0


def interactive(args):
    print("Laya interactive mode. Type a request and press Enter; Ctrl-D or 'quit' to exit.")
    router = make_router(args)
    while True:
        try:
            text = input("laya> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        run(text, args, router)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    text = " ".join(args.text).strip()
    if not text:
        return interactive(args)
    return run(text, args)


if __name__ == "__main__":
    sys.exit(main())
