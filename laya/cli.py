"""Command-line interface for testing Laya locally.

    laya "I was charged twice, please refund"           # routing decision only (no model download)
    laya "Refactor this service" --predict              # full answers, loads the checkpoint
    laya                                                # interactive mode
    laya "Mein Konto wurde zweimal belastet" --lang de  # explicit language
    laya "My payment failed twice" --preset triage      # a ready-made question preset
    laya "Where is my card" --questions intents.json    # your own questions, from a JSON file
    laya "Where is my card" --questions intents.json --head-max-len 384

Routing (the default) never downloads a checkpoint, so it works offline and
returns in milliseconds. --predict loads the routed checkpoint on first use,
which needs network access to the Hugging Face hub. --preset answers one of the
ready-made question presets from laya.presets and implies --predict. --questions
answers the question set in a JSON file instead; --max-len and --head-max-len
raise its token budget for the request, which is what a many-label question needs.
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

# The state field each preset's instructions name, so the CLI puts the text where that question
# set reads it. `router` is also the default for `--predict`, which answers
# `laya.router_questions()`. Kept beside PRESETS so a new preset and its key arrive together.
PRESET_STATE_KEYS = {
    "email": "body",
    "guard": "prompt",
    "moderation": "post",
    "router": "request",
    "triage": "message",
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
    parser.add_argument("--questions", metavar="FILE",
                        help="a JSON file of your own typed questions to answer instead of the router questions"
                             " or a preset; implies --predict")
    parser.add_argument("--max-len", type=int, dest="max_len", metavar="N",
                        help="token budget for the whole request (state plus options); defaults to the"
                             " checkpoint's own budget")
    parser.add_argument("--head-max-len", type=int, dest="head_max_len", metavar="N",
                        help="token budget the choice options share; raise it when a question has many"
                             " labels, so each keeps enough tokens to stay distinct")
    parser.add_argument("--device", help="torch device, e.g. cpu or cuda")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    return parser


def make_router(args):
    return laya.Router(device=args.device, preload=False)


def load_questions(path):
    """Read a ``--questions`` file: returns ``(questions, state_key)``.

    The file is either a bare ``question id -> definition`` mapping, or an object holding one
    under ``"questions"`` plus an optional ``"state_key"`` naming the field the instructions
    refer to. That second field exists because #426 found the CLI sending every request under a
    key no question set named: with a user's own questions the CLI cannot guess the key, so the
    file declares it. Default ``"request"``, matching bare ``--predict``.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise ValueError("no such --questions file: %s" % path)
    if not isinstance(raw, dict):
        raise ValueError("--questions file must be a JSON object of question id -> definition")
    state_key = "request"
    questions = raw
    if "questions" in raw:
        if not isinstance(raw["questions"], dict):
            raise ValueError("the 'questions' field of the --questions file must be an object")
        questions = raw["questions"]
        state_key = raw.get("state_key", "request")
        if not isinstance(state_key, str) or not state_key.strip():
            raise ValueError("'state_key' must be a non-empty string, got %r" % (state_key,))
    if not questions:
        raise ValueError("--questions file holds no questions")
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict):
            raise ValueError("question %r must map to an object, got %s" % (qid, type(qdef).__name__))
    return questions, state_key


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


def budget_overrides(args):
    """The token-budget flags as `predict` keyword arguments, absent when unset.

    Unset has to mean unsent, so the checkpoint's own defaults stay in charge. A zero is a real
    budget, not an absence, so this tests `is not None` rather than truthiness.
    """
    overrides = {}
    for flag in ("max_len", "head_max_len"):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[flag] = value
    return overrides


def resolve_questions(args):
    """The question set to answer and the state field it reads, from the flags given."""
    if args.questions and args.preset:
        raise ValueError("--questions and --preset choose different question sets; pass one")
    if args.questions:
        return load_questions(args.questions)
    if args.preset:
        return PRESETS[args.preset](), PRESET_STATE_KEYS[args.preset]
    return laya.router_questions(), PRESET_STATE_KEYS["router"]


def run(text, args, router=None):
    """Route or predict one request; returns 0 on success, 2 on a handled error."""
    router = router or make_router(args)
    try:
        if args.predict or args.preset or args.questions:
            questions, state_key = resolve_questions(args)
            # Each preset's instructions name the field they read -- `` `message` ``,
            # `` `body` ``, `` `prompt` ``, `` `post` ``, `` `request` `` -- and the CLI used to
            # send every request as `{"text": ...}`, a key none of them names, so the model was
            # asked about a field that was not there. The state key therefore follows the
            # question set being answered: the preset's own key, or the one a --questions file
            # declares. Routing is not affected either way: `route` reads the state only for
            # language detection, which is key-invariant, so the default path keeps
            # `{"text": ...}`.
            state = {state_key: text}
            result = router.predict(state, questions,
                                    model=args.model, task=args.task, lang=args.lang,
                                    **budget_overrides(args))
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            else:
                show_answers(result)
        else:
            state = {"text": text}
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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":       # `laya eval ...` mirrors the `laya-evals` script
        from .evals_cli import main as eval_main
        return eval_main(argv[1:])
    args = build_parser().parse_args(argv)
    text = " ".join(args.text).strip()
    if not text:
        return interactive(args)
    return run(text, args)


if __name__ == "__main__":
    sys.exit(main())
