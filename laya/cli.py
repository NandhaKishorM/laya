"""Command-line interface for testing Laya locally.

    laya "I was charged twice, please refund"           # routing decision only (no model download)
    laya "Refactor this service" --predict              # full answers, loads the checkpoint
    laya                                                # interactive mode
    laya "Mein Konto wurde zweimal belastet" --lang de  # explicit language
    laya "Jag har debiterats två gånger" --preset triage --lang sv-SE  # Swedish triage
    laya "My payment failed twice" --model ml           # pin a checkpoint, by name or alias
    laya "My payment failed twice" --preset triage      # a ready-made question preset
    laya --batch tickets.txt --predict                  # a file of requests, one per line
    cat tickets.txt | laya --batch - --predict --json   # stdin, one JSON line of answers each
    laya "Where is my card" --questions intents.json    # your own questions, from a JSON file
    laya "Where is my card" --questions intents.json --head-max-len 384

Routing (the default) never downloads a checkpoint, so it works offline and
returns in milliseconds. --predict loads the routed checkpoint on first use,
which needs network access to the Hugging Face hub. --preset answers one of the
ready-made question presets from laya.presets and implies --predict.
--batch scores a whole file (or stdin, with `-`) through Router.predict_batch in one
process, so the states share checkpoint loads and forward passes; --json prints JSONL.
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


_SWEDISH_HELP_TRANSLATIONS = (
    ("Test Laya locally: route or answer a request from the command line.",
     "Testa Laya lokalt: låt den välja modell för en text eller besvara frågor från kommandoraden."),
    ("the request text (omit for interactive mode)",
     "texten som ska behandlas (utelämna för interaktivt läge)"),
    ("show this help message and exit", "visa den här hjälpen och avsluta"),
    ("run the full prediction, not just the routing decision (downloads the checkpoint on first use)",
     "kör hela analysen, inte bara modellvalet (hämtar modellen första gången)"),
    ("force a checkpoint instead of auto-routing: a checkpoint name or any of core's aliases, in any casing, or 'auto' to route it (the default)",
     "välj modell i stället för automatisk routning: modellnamn eller alias oavsett skiftläge; ange 'auto' för automatisk routning (standard)"),
    ("force a language, e.g. sv-SE, en or de, instead of detecting it",
     "ange språk, till exempel sv-SE, en eller de, i stället för att identifiera det automatiskt"),
    ("force a typed-decisions workflow instead of detecting it",
     "ange ett arbetsflöde för strukturerade beslut i stället för att identifiera det automatiskt"),
    ("answer a ready-made question preset", "besvara frågor med en inbyggd frågeuppsättning"),
    ("instead of the router questions; implies --predict", "i stället för routerns frågor; innebär --predict"),
    ("a JSON file of your own typed questions to answer instead of the router questions or a preset; implies --predict",
     "en JSON-fil med egna strukturerade frågor att besvara i stället för routerns frågor eller en frågeuppsättning; innebär --predict"),
    ("token budget for the whole request (state plus options); defaults to the checkpoint's own budget",
     "tokenbudget för hela begäran (text och svarsalternativ); standardvärdet hämtas från modellen"),
    ("token budget the choice options share; raise it when a question has many labels, so each keeps enough tokens to stay distinct",
     "tokenbudget som delas mellan svarsalternativen; höj den vid många alternativ så att de förblir tydliga"),
    ("torch device, e.g. cpu or cuda", "beräkningsenhet, till exempel cpu eller cuda"),
    ("print the raw result as JSON", "skriv ut råresultatet som JSON"),
    ("score a file of requests, one per line (use '-' for stdin), instead of a single text; implies neither --predict nor --preset, but the same modes apply: routing by default, answers with --predict/--preset",
     "bearbeta en fil med texter, en per rad (använd '-' för standardindata), i stället för en text. Aktiverar inte --predict eller --preset: standard är val av modell, med flaggorna besvaras frågor"),
    ("states per forward pass in --batch mode; the default sends each routed group in one pass",
     "antal texter per modellkörning i --batch-läge; standardvärdet skickar varje routad grupp i en körning"),
)


class LayaArgumentParser(argparse.ArgumentParser):
    """Translate common argument errors when the caller explicitly selects Swedish."""

    def parse_args(self, args=None, namespace=None):
        tokens = list(sys.argv[1:] if args is None else args)
        self._swedish_locale = False
        for index, token in enumerate(tokens):
            if token == "--":
                break
            if token.startswith("--lang="):
                value = token.split("=", 1)[1]
            elif token == "--lang" and index + 1 < len(tokens):
                value = tokens[index + 1]
            else:
                continue
            language = value.strip().lower().replace("_", "-").split("-", 1)[0]
            self._swedish_locale = language == "sv"
        if self._swedish_locale:
            self._localize_help()
        return super().parse_args(tokens, namespace)

    @staticmethod
    def _translate_help(text):
        for source, target in _SWEDISH_HELP_TRANSLATIONS:
            text = text.replace(source, target)
        return text

    def _localize_help(self):
        if getattr(self, "_help_localized", False):
            return
        if self.description:
            self.description = self._translate_help(self.description)
        for action in self._actions:
            if action.help:
                action.help = self._translate_help(action.help)
        for group in self._action_groups:
            if group.title == "positional arguments":
                group.title = "textargument"
            elif group.title == "options":
                group.title = "flaggor"
        self._help_localized = True

    def error(self, message):
        if not getattr(self, "_swedish_locale", False):
            return super().error(message)
        translations = (
            ("argument --preset: invalid choice:", "argumentet --preset har ogiltigt värde:"),
            ("(choose from ", "(välj mellan "),
            ("argument --preset: expected one argument", "--preset måste följas av ett namn"),
            ("argument --questions: expected one argument", "--questions måste följas av en fil"),
            ("argument --lang: expected one argument", "--lang måste följas av en språkkod"),
            ("argument --", "argumentet --"),
            (": expected one argument", ": saknar ett värde"),
            (": invalid int value: ", ": måste vara ett heltal; angivet värde: "),
            ("unknown model ", "okänd modell "),
            ("; choose one of ", "; välj bland "),
            ("(or an alias: ", "(eller alias: "),
            (", or 'auto'", "; ange 'auto' för automatisk routning"),
            ("unrecognized arguments: ", "okända argument: "),
        )
        for source, target in translations:
            message = message.replace(source, target)
        usage = super().format_usage().replace("usage:", "användning:", 1)
        self._print_message(usage, sys.stderr)
        self.exit(2, "%s: fel: %s\n" % (self.prog, message))

    def format_help(self):
        help_text = super().format_help()
        if not getattr(self, "_swedish_locale", False):
            return help_text
        return help_text.replace("usage:", "användning:", 1)


def model_name(value):
    """Resolve a `--model` argument the way core resolves it: same names, same aliases, same casing.

    ``argparse``'s ``choices=`` compared strings exactly, so ``--model en`` was rejected with
    "invalid choice" even though the call this flag feeds -- ``router.predict(model=...)`` --
    has always accepted it. ``laya.router.normalise_name`` is the one registry of checkpoints and
    aliases, so delegating here cannot fall behind it, and the name that reaches the router is
    already canonical: what a caller prints is the checkpoint, not its spelling. ``auto`` means
    "do not pin one", which is what omitting the flag means, so it maps to ``None``.
    """
    if value.strip().lower() == "auto":
        return None
    from laya.router import normalise_name

    try:
        return normalise_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("%s, or 'auto'" % error) from None


def build_parser():
    parser = LayaArgumentParser(
        prog="laya",
        description="Test Laya locally: route or answer a request from the command line.",
    )
    parser.add_argument("text", nargs="*", help="the request text (omit for interactive mode)")
    parser.add_argument("--predict", action="store_true",
                        help="run the full prediction, not just the routing decision (downloads the checkpoint on first use)")
    parser.add_argument("--model", type=model_name, metavar="NAME",
                        help="force a checkpoint instead of auto-routing: a checkpoint name or any of "
                             "core's aliases, in any casing, or 'auto' to route it (the default)")
    parser.add_argument("--lang", help="force a language, e.g. sv-SE, en or de, instead of detecting it")
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
    parser.add_argument("--batch", metavar="FILE",
                        help="score a file of requests, one per line (use '-' for stdin), instead "
                             "of a single text; implies neither --predict nor --preset, but the "
                             "same modes apply: routing by default, answers with --predict/--preset")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="states per forward pass in --batch mode; the default sends each "
                             "routed group in one pass")
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


def _swedish_route_reason(reason):
    exact = {
        "English Latin text": "Engelsk text med latinskt alfabet",
        "detected non-English text": "detekterad icke-engelsk text",
    }
    if reason in exact:
        return exact[reason]
    prefixes = (
        ("no letters detected in state; using default (", "Ingen text att språkidentifiera; standardmodell: "),
        ("non-Latin script (", "Icke-latinskt skriftsystem ("),
        ("Latin script but language looks like ", "Latinsk text som verkar vara "),
        ("Latin script, language not identified but ", "Latinsk text med oidentifierat språk och "),
        ("Latin script, language not identified and no non-English letters; using default (",
         "Latinsk text utan identifierat språk eller tydliga icke-engelska bokstäver; standardmodell: "),
        ("Latin script, mostly English, but a line or field reads as ",
         "Mest engelsk text med latinskt alfabet, men en rad eller ett fält verkar vara "),
        ("explicit model=", "Modell vald uttryckligen: "),
        ("explicit task=", "Uppgift vald uttryckligen: "),
        ("explicit lang=", "Språk angivet uttryckligen: "),
        ("question ids match the ", "Fråge-ID:n matchar arbetsflödet "),
        ("lang_guess: the caller identified this as English text",
         "Språkhint: anroparen angav engelska"),
        ("lang_guess: the caller identified this as non-English text",
         "Språkhint: anroparen angav ett annat språk än engelska"),
        ("Router(lang_guess=...): the caller identified this as English text",
         "Router(lang_guess=...): anroparen angav engelska"),
        ("Router(lang_guess=...): the caller identified this as non-English text",
         "Router(lang_guess=...): anroparen angav ett annat språk än engelska"),
    )
    for prefix, translated in prefixes:
        if reason.startswith(prefix):
            remainder = reason[len(prefix):]
            for source, target in (
                ("the English checkpoint cannot read it", "den engelska modellen kan inte läsa texten"),
                (", not English", ", inte engelska"),
                ("; not safe for the English checkpoint", "; olämpligt för den engelska modellen"),
                ("; using default (", "; standardmodell: "),
                ("of letters", "av bokstäverna"),
                ("non-English letters", "bokstäver som inte är engelska"),
                (" typed-decisions workflow", " för strukturerade beslut"),
            ):
                remainder = remainder.replace(source, target)
            return translated + remainder.rstrip(")")
    return reason


def show_decision(decision, language=None):
    if language == "sv":
        print("Modell    :", decision["model"])
        print("Orsak     :", _swedish_route_reason(decision["reason"]))
    else:
        print("Model     :", decision["model"])
        print("Reason    :", decision["reason"])
    detection = decision.get("detection")
    if detection:
        print("Språkdetektion:" if language == "sv" else "Detected  :",
              json.dumps(detection, ensure_ascii=False))


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


def show_localized_answers(result, labels):
    """Print localized labels without changing machine-readable result keys."""
    routing = result.get("routing")
    if routing:
        show_decision(routing, language="sv")
        print()
    for qid, answer in result.get("answers", {}).items():
        label = labels.get(qid, qid)
        if "choice" in answer:
            choice = answer["choice"]
            detail = labels.get(choice, choice)
            probability = answer.get("probabilities", {}).get(choice)
            if probability is not None:
                detail += " (p=%.3f)" % probability
        elif "score" in answer:
            detail = "%.2f" % answer["score"]
            probabilities = answer.get("probabilities") or {}
            legend = answer.get("legend") or {}
            if probabilities and legend:
                level = max(probabilities, key=lambda key: probabilities[key])
                level_label = legend.get(str(level), legend.get(level))
                if level_label:
                    detail += " — %s (p=%.3f)" % (level_label, probabilities[level])
        elif "noul" in answer:
            value = answer["noul"]
            detail = "%s (p=%.3f)" % ("Ja" if value >= 0.5 else "Nej", value)
        else:
            detail = json.dumps(answer, ensure_ascii=False)
        print("%-18s: %s" % (label, detail))


def show_swedish_preset_answers(result, preset):
    labels = laya.preset_labels(preset, "sv")
    show_localized_answers(result, labels)


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


def _is_swedish(args):
    language = (getattr(args, "lang", None) or "").strip().lower().replace("_", "-").split("-", 1)[0]
    return language == "sv"


def _print_cli_error(message, args):
    """Print a localized CLI message while retaining technical error details."""
    if _is_swedish(args):
        question_error = "question "
        type_marker = " must map to an object, got "
        if message.startswith(question_error) and type_marker in message:
            question_id, question_type = message[len(question_error):].split(type_marker, 1)
            message = "frågan %s måste vara ett objekt; angiven typ: %s" % (
                question_id, question_type)
        translations = {
            "could not run Laya": "det gick inte att köra Laya",
            "Check that the dependencies are installed and the checkpoints can be downloaded from the Hugging Face hub (network access is needed on first use).":
                "Kontrollera att beroendena är installerade och att modellfilerna kan hämtas från Hugging Face Hub (nätverksåtkomst behövs första gången).",
            "pass either a text or --batch FILE, not both.":
                "ange antingen en text eller --batch FIL, inte båda.",
            "--questions and --preset choose different question sets; pass one":
                "ange antingen --questions eller --preset; de väljer olika frågeuppsättningar",
            "no such --questions file": "filen som anges med --questions finns inte",
            "--questions file must be a JSON object of question id -> definition":
                ("filen som anges med --questions måste innehålla ett JSON-objekt där "
                 "fråge-ID:n kopplas till definitioner"),
            "the 'questions' field of the --questions file must be an object":
                "fältet 'questions' i --questions-filen måste vara ett objekt",
            "--questions file holds no questions":
                "filen som anges med --questions innehåller inga frågor",
            "'state_key' must be a non-empty string, got":
                "'state_key' måste vara en icke-tom sträng; angivet värde",
            "could not read": "det gick inte att läsa",
            "no requests found in": "hittade inga förfrågningar i",
        }
        for source, target in translations.items():
            message = message.replace(source, target)
    print("laya: %s" % message, file=sys.stderr)


def resolve_questions(args):
    """The question set to answer and the state field it reads, from the flags given."""
    if args.questions and args.preset:
        raise ValueError("--questions and --preset choose different question sets; pass one")
    if args.questions:
        return load_questions(args.questions)
    if args.preset:
        if args.lang:
            language = args.lang.strip().lower().replace("_", "-").split("-", 1)[0]
            if language == "sv":
                return PRESETS[args.preset](language=args.lang), PRESET_STATE_KEYS[args.preset]
        return PRESETS[args.preset](), PRESET_STATE_KEYS[args.preset]
    if args.lang:
        language = args.lang.strip().lower().replace("_", "-").split("-", 1)[0]
        if language == "sv":
            return laya.router_questions(language=args.lang), PRESET_STATE_KEYS["router"]
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
                language = (args.lang or "").strip().lower().replace("_", "-").split("-", 1)[0]
                if language == "sv" and not args.questions:
                    show_swedish_preset_answers(result, args.preset or "router")
                else:
                    show_answers(result)
        else:
            state = {"text": text}
            decision = router.route(state, model=args.model, task=args.task, lang=args.lang)
            if args.json:
                print(json.dumps(dict(decision), ensure_ascii=False, indent=2, default=str))
            else:
                language = (args.lang or "").strip().lower().replace("_", "-").split("-", 1)[0]
                show_decision(decision, language=language if language == "sv" else None)
    except ValueError as error:
        _print_cli_error(str(error), args)
        return 2
    except (ImportError, OSError, RuntimeError) as error:
        _print_cli_error("could not run Laya (%s)." % error, args)
        _print_cli_error("Check that the dependencies are installed and the checkpoints can be downloaded from the "
                         "Hugging Face hub (network access is needed on first use).", args)
        return 2
    return 0


def read_batch_lines(source):
    """One request per non-blank line of FILE, or of stdin for '-'."""
    if source == "-":
        lines = sys.stdin.read().splitlines()
    else:
        with open(source, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    return [line.strip() for line in lines if line.strip()]


def run_batch(lines, args, router=None):
    """Route or predict a whole file in one call, so states share checkpoint loads and
    forward passes; returns 0 on success, 2 on a handled error."""
    router = router or make_router(args)
    overrides = {key: value for key, value in (("model", args.model), ("task", args.task),
                                               ("lang", args.lang)) if value is not None}
    try:
        if args.predict or args.preset or getattr(args, "questions", None):
            questions, key = resolve_questions(args)
            # Router.predict_batch reads the token budget per request (#583).
            overrides.update(budget_overrides(args))
            requests = [{"state": {key: line}, "questions": questions, **overrides}
                        for line in lines]
            results = router.predict_batch(requests, batch_size=args.batch_size)
            for line, result in zip(lines, results):
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, default=str))
                else:
                    print("# %s" % (line if len(line) <= 72 else line[:69] + "..."))
                    language = (args.lang or "").strip().lower().replace("_", "-").split("-", 1)[0]
                    if language == "sv" and not getattr(args, "questions", None):
                        show_swedish_preset_answers(result, args.preset or "router")
                    else:
                        show_answers(result)
                    print()
        else:
            decisions = router.route_batch([{"state": {"text": line}, "questions": {},
                                             **overrides} for line in lines])
            for line, decision in zip(lines, decisions):
                if args.json:
                    print(json.dumps(dict(decision), ensure_ascii=False, default=str))
                else:
                    language = (args.lang or "").strip().lower().replace("_", "-").split("-", 1)[0]
                    model, reason = decision["model"], decision["reason"]
                    if language == "sv":
                        model = "Modell: %s" % model
                        reason = "Orsak: %s" % _swedish_route_reason(reason)
                        request = "Indata: %s" % (line if len(line) <= 48 else line[:45] + "...")
                        print("%-22s %-72s <- %s" % (model, reason, request))
                    else:
                        print("%-14s %s  <- %s" % (model, reason,
                                                   line if len(line) <= 48 else line[:45] + "..."))
    except ValueError as error:
        _print_cli_error(str(error), args)
        return 2
    except (ImportError, OSError, RuntimeError) as error:
        _print_cli_error("could not run Laya (%s)." % error, args)
        _print_cli_error("Check that the dependencies are installed and the checkpoints can be downloaded from the "
                         "Hugging Face hub (network access is needed on first use).", args)
        return 2
    return 0


def interactive(args):
    if _is_swedish(args):
        print("Layas interaktiva läge. Skriv in en text och tryck på Retur. "
              "Avsluta med Ctrl-D eller skriv 'avsluta'.")
        prompt = "laya> "
        quit_words = ("quit", "exit", "avsluta")
    else:
        print("Laya interactive mode. Type a request and press Enter; Ctrl-D or 'quit' to exit.")
        prompt = "laya> "
        quit_words = ("quit", "exit")
    router = make_router(args)
    while True:
        try:
            text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in quit_words:
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
    if args.batch:
        if text:
            _print_cli_error("pass either a text or --batch FILE, not both.", args)
            return 2
        try:
            lines = read_batch_lines(args.batch)
        except OSError as error:
            _print_cli_error("could not read %s (%s)." % (args.batch, error), args)
            return 2
        if not lines:
            _print_cli_error("no requests found in %s." % args.batch, args)
            return 2
        return run_batch(lines, args)
    if not text:
        return interactive(args)
    return run(text, args)


if __name__ == "__main__":
    sys.exit(main())
