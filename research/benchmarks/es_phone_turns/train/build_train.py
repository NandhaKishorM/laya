"""Turns the raw phrases into training cases: one utterance, one menu, one answer.

Every case gets a menu drawn from its own trade, in random order, under randomly chosen
names for the three answers all menus share. The point is that the only way to get a case
right is to read the utterance against what each option says.
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import domains

HERE = Path(__file__).parent
RAW = HERE / "raw"
TESTS = sorted((HERE.parent / "data").glob("*.jsonl"))

NEGATION = re.compile(r"\b(no|ni|nunca|tampoco|sin|nada|nadie|ningun[oa]?)\b")

# Negated transfers, written by hand. The destination is NAMED and refused in the same
# breath, which is the sentence a keyword reader gets wrong with full confidence.
NEGATED_LEADS = [
    "No me pase con {d}",
    "No quiero hablar con {d}",
    "No necesito {d}",
    "No es para {d}",
    "No, con {d} no",
    "Todavía no me pase con {d}",
    "No me transfiera a {d}",
    "No me comunique con {d}",
    "Con {d} no, por favor",
    "No hace falta que me pase con {d}",
    "No busco {d}",
    "Mire, no es con {d}",
    "No quiero que me manden a {d}",
    "A {d} no me pase",
    "No, no, con {d} no es",
    "Yo no pedí {d}",
    "Ya hablé con {d}, no me vuelva a pasar",
    "No ocupo {d}",
]
NEGATED_TAILS = [
    ", solo quiero saber el horario.",
    ", solo tengo una pregunta.",
    ", prefiero que me lo diga usted.",
    ", primero explíqueme usted.",
    ", era otra cosa.",
    ", solo quería saber dónde están ubicados.",
    ".",
    ", por favor.",
    ", solo una consulta rápida.",
    ", nada más quería preguntar algo.",
    ", quiero que me atienda usted.",
    ", déjeme terminar de explicarle.",
    ", solo necesito un dato.",
    ", gracias.",
]
# One area turned down and another one asked for, in the same sentence. Without these the
# fine-tuned model learned "a refusal means keep talking" and sent "not sales, I need
# support" nowhere: 1 of 6 on the independent test, where the language model got 6 of 6.
REDIRECT_LEADS = [
    "No me pase con {a}, ", "Con {a} no, ", "No es con {a}, ", "No quiero {a}, ",
    "No me mande a {a}, ", "A {a} no, ", "Ya hablé con {a} y no era ahí, ",
    "No busco {a}, ", "No me comunique con {a}, ", "Yo no pedí {a}, ",
]
REDIRECT_ASKS = [
    "páseme con {b}.", "comuníqueme con {b}, por favor.", "necesito hablar con {b}.",
    "lo que ocupo es {b}.", "es con {b}.", "quiero {b}.", "mejor con {b}.",
]
REDIRECT_AFTER = [
    "Necesito hablar con {b}, no con {a}.", "Páseme con {b}, con {a} no.",
    "Es para {b}, no para {a}.", "Quiero {b}, no {a}.",
]
GREETING_START = re.compile(
    r"^(hola|buen[oa]s|oiga|oye|disculpe|saludos|alo|aló|mire|fíjese|fijese)\b", re.I)
# A refusal that also asks for someone is not a refusal to keep talking: it is a request,
# and the generator cannot know for whom. It stays out of the refusals.
ASKS_FOR_SOMEONE = re.compile(
    r"\b(p[aá]se(me|nme)|comun[ií]que(me|nme)|necesito hablar|quiero hablar|"
    r"transfi[eé]ra(me|nme)|con[eé]cte(me|nme))\b", re.I)

FILLERS = ["eh", "este", "bueno", "mire", "a ver", "si", "alo", "oiga", "fijese que"]


def normalize(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]+", " ", text).split()


def too_close(tokens, references):
    """A training phrase that is a test phrase with two words changed is the test."""
    mine = set(tokens)
    if not mine:
        return True
    for theirs in references:
        if len(mine & theirs) / len(mine | theirs) >= 0.7:
            return True
    return False


def as_recognised(rng, text):
    """What a speech recogniser hands over: no capitals, no punctuation, a false start."""
    out = re.sub(r"[¿?¡!.,;:«»\"()]+", " ", text.lower())
    if rng.random() < 0.5:
        out = "".join(c for c in unicodedata.normalize("NFD", out)
                      if unicodedata.category(c) != "Mn" or c == "̃")
        out = unicodedata.normalize("NFC", out)
    words = out.split()
    if words and rng.random() < 0.3:
        words.insert(0, rng.choice(FILLERS))
    if len(words) > 3 and rng.random() < 0.2:
        at = rng.randrange(0, min(3, len(words)))
        words.insert(at, words[at])
    return " ".join(words)


def menu(rng, domain_key, must_have, gold_kind):
    """Builds the options. `must_have` are destinations the case cannot be asked without."""
    departments = domains.DOMAINS[domain_key]["departments"]
    chosen = list(must_have)
    others = [d for d in departments if d not in chosen]
    rng.shuffle(others)
    chosen += others[: rng.randint(1, len(others))] if others else []

    human = rng.choice(domains.HUMAN)
    hang_up = rng.choice(domains.HANG_UP)
    keep = rng.choice(domains.KEEP_TALKING)

    options = [(d, departments[d]) for d in chosen]
    if gold_kind == "human" or rng.random() < 0.85:
        options.append(human)
    if gold_kind == "hang_up" or rng.random() < 0.85:
        options.append(hang_up)
    options.append(keep)
    rng.shuffle(options)

    # One menu in eight is written without descriptions: some tenants will only give names.
    bare = rng.random() < 0.125
    criteria = {label: (None if bare else text) for label, text in options}
    special = {"human": human[0], "hang_up": hang_up[0], "keep": keep[0]}
    return criteria, special


def case(rng, text, domain_key, gold_kind, department=None, must_have=(), weight=0.9):
    criteria, special = menu(rng, domain_key, must_have, gold_kind)
    gold = department if gold_kind == "department" else special[gold_kind]
    labels = list(criteria)
    rest = (1.0 - weight) / (len(labels) - 1)
    probabilities = {label: (weight if label == gold else rest) for label in labels}
    return {
        "state": text,
        "questions": {"accion": {
            "type": "choice", "instructions": rng.choice(domains.INSTRUCTIONS),
            "criteria": criteria}},
        "gold": {"accion": {"label": gold, "probabilities": probabilities}},
        "meta": {"domain": domain_key, "kind": gold_kind, "department": department},
    }


def load(name):
    path = RAW / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["phrases"] if path.exists() else []


# How sure the target is, by where the sentence came from. A sentence built to name its
# destination leaves no room for doubt; one a language model wrote to a brief can miss the
# brief. Targets of 0.9 across the board capped the fine-tuned model's confidence near
# 0.85 whether it was right or wrong, which makes a confidence gate useless.
CERTAIN, LIKELY = 0.98, 0.94


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="train.jsonl")
    parser.add_argument("--without", default="",
                        help="comma-separated sources to leave out, for ablations")
    args = parser.parse_args()
    without = {name for name in args.without.split(",") if name}

    rng = random.Random(20260929)
    references = []
    for path in TESTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                references.append(set(normalize(json.loads(line)["text"])))

    drafts, dropped = [], {
        "close_to_test": 0, "negation_missing": 0, "duplicate": 0, "refusal_that_asks": 0}
    domain_keys = list(domains.DOMAINS)

    for key, domain in domains.DOMAINS.items():
        for label in domain["departments"]:
            for kind in ("indirect", "direct", "needs_with_no"):
                for phrase in load(f"{key}.{label}.{kind}"):
                    drafts.append((phrase, key, "department", label, (label,),
                                   CERTAIN if kind == "direct" else LIKELY, kind))
            for phrase in load(f"{key}.{label}.negated"):
                if not NEGATION.search(phrase.lower()):
                    dropped["negation_missing"] += 1
                    continue
                if ASKS_FOR_SOMEONE.search(phrase):
                    dropped["refusal_that_asks"] += 1
                    continue
                drafts.append((phrase, key, "keep", None, (label,), LIKELY, "negated"))
            # The hand-written refusals, with this destination's own name in them.
            for lead in rng.sample(NEGATED_LEADS, 5):
                phrase = lead.format(d=label.replace("_", " ")) + rng.choice(NEGATED_TAILS)
                drafts.append((phrase, key, "keep", None, (label,), CERTAIN, "negated_named"))
        # Refuse one area, ask for another of the same trade.
        names = list(domain["departments"])
        for wanted in names:
            needs = [p for p in load(f"{key}.{wanted}.indirect")
                     if not GREETING_START.search(p) and len(p) < 120]
            for _ in range(12):
                refused = rng.choice([n for n in names if n != wanted])
                a, b = refused.replace("_", " "), wanted.replace("_", " ")
                roll = rng.random()
                if roll < 0.4 and needs:
                    need = rng.choice(needs)
                    phrase = rng.choice(REDIRECT_LEADS).format(a=a) + need[0].lower() + need[1:]
                    weight = LIKELY
                elif roll < 0.75:
                    phrase = rng.choice(REDIRECT_LEADS).format(a=a) + rng.choice(
                        REDIRECT_ASKS).format(b=b)
                    weight = CERTAIN
                else:
                    phrase = rng.choice(REDIRECT_AFTER).format(a=a, b=b)
                    weight = CERTAIN
                drafts.append((phrase, key, "department", wanted, (wanted, refused),
                               weight, "redirected"))
        for phrase in load(f"{key}.open"):
            drafts.append((phrase, key, "keep", None, (), LIKELY, "open"))

    for phrase in load("pool.farewell"):
        drafts.append((phrase, rng.choice(domain_keys), "hang_up", None, (), LIKELY, "farewell"))
    for name in ("human", "human_with_no"):
        for phrase in load(f"pool.{name}"):
            drafts.append((phrase, rng.choice(domain_keys), "human", None, (), LIKELY, "human"))
    for phrase in load("pool.frustration_human"):
        drafts.append((phrase, rng.choice(domain_keys), "human", None, (), LIKELY, "human"))
    for phrase in load("pool.frustration"):
        if ASKS_FOR_SOMEONE.search(phrase):
            dropped["refusal_that_asks"] += 1
            continue
        drafts.append((phrase, rng.choice(domain_keys), "keep", None, (), LIKELY, "frustration"))
    for name in ("greeting", "negated_hangup", "negated_human"):
        for phrase in load(f"pool.{name}"):
            if name != "greeting" and not NEGATION.search(phrase.lower()):
                dropped["negation_missing"] += 1
                continue
            drafts.append((phrase, rng.choice(domain_keys), "keep", None, (), LIKELY, name))

    seen, cases = set(), []
    for phrase, key, gold_kind, department, must_have, weight, source in drafts:
        if source in without:
            continue
        tokens = normalize(phrase)
        signature = " ".join(tokens)
        if signature in seen:
            dropped["duplicate"] += 1
            continue
        if too_close(tokens, references):
            dropped["close_to_test"] += 1
            continue
        seen.add(signature)

        text = as_recognised(rng, phrase) if rng.random() < 0.3 else phrase
        item = case(rng, text, key, gold_kind, department, must_have, weight)
        item["meta"]["source"] = source
        cases.append(item)

    # The pools are small next to the destinations. Seen once each, farewells and greetings
    # would be a rounding error in the loss, and they are most of what a real call is.
    pools = [c for c in cases if c["meta"]["source"] in
             ("farewell", "human", "greeting", "negated_hangup", "negated_human",
              "frustration")]
    for original in pools:
        for _ in range(2):
            text = original["state"]
            if rng.random() < 0.5:
                text = as_recognised(rng, text)
            again = case(rng, text, rng.choice(domain_keys), original["meta"]["kind"],
                         None, (), LIKELY)
            again["meta"]["source"] = original["meta"]["source"]
            cases.append(again)

    rng.shuffle(cases)
    out = HERE / args.out
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
                   encoding="utf-8")

    by_source, by_kind = {}, {}
    for c in cases:
        by_source[c["meta"]["source"]] = by_source.get(c["meta"]["source"], 0) + 1
        by_kind[c["meta"]["kind"]] = by_kind.get(c["meta"]["kind"], 0) + 1
    print(f"{len(cases)} cases -> {out.name}")
    print("  by answer:", by_kind)
    print("  by source:", by_source)
    print("  dropped:", dropped)


if __name__ == "__main__":
    main()
