#!/usr/bin/env python3
"""Generate the parity fixtures `laya-java` asserts against, by running the real `laya`.

Nothing in `laya-java` may hand-write an expectation that this script can produce. A port's
tokenizer, its sequence budgets and its decode arithmetic all drift silently, and a hand-typed
expectation drifts with them; a generated one cannot. `sdk/typescript/scripts/sync_presets.py`
sets the same precedent for the TypeScript client's presets, including the `--check` gate CI runs.

    gen_fixtures.py             # write laya-java/fixtures/*.json
    gen_fixtures.py --check     # exit 1 if any committed fixture would change

`--check` is what CI runs: it regenerates into memory and diffs, so a change to `laya/` that moves
the contract fails the Java build instead of being discovered by a user.

Phase 0 families (no model weights needed -- these are the tables and the text the model is shown):

  lang_tables.json    every table `laya/lang.py` routes on, and every threshold constant
  presets.json        the preset question dicts, verbatim, as the model receives them
"""
import argparse
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
# Where the checkpoint-backed families look for weights when LAYA_FIXTURE_CHECKPOINTS is unset.
# Repo-relative on purpose: this was one developer's absolute home directory, which made WHETHER
# a family regenerates at all a property of whose machine ran the script -- and wrote that path
# into a committed skip marker in an open-source repository.
DEFAULT_CHECKPOINT_ROOT = os.path.normpath(
    os.path.join(HERE, "..", ".work", "checkpoints"))
FIXTURES = os.path.normpath(os.path.join(HERE, "..", "fixtures"))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _sorted_set(values):
    """A set as a sorted list: a fixture has to be byte-stable across runs and interpreters."""
    return sorted(values)


def lang_tables():
    """Every table and threshold `laya.lang` decides a route with.

    Read from the imported module, never parsed out of the source: `_SHARED_WORDS`,
    `_EN_ONLY_WORDS` and `_EN_COLLISION_WORDS` are *derived* at import time from `_STOP`, so
    parsing the file would miss the derivation and a port built from it would be subtly wrong.
    """
    from laya import lang

    return {
        "script_ranges": [
            # (name, [(lo, hi), ...]) with the bounds as ints -- a port needs the numbers, not
            # whatever repr the host language gives a range object.
            {"name": name, "ranges": [[int(lo), int(hi)] for lo, hi in ranges]}
            for name, ranges in lang._SCRIPT_RANGES
        ],
        # Sorted so the committed file diffs stably -- but sorting LOSES the declaration order,
        # and that order is data: the language that wins a tied score is the first one iterated.
        # So it is recorded separately rather than inferred from this mapping, which is exactly
        # the mistake `render`'s missing `sort_keys` was already fixed for once.
        "stop_word_order": list(lang._STOP.keys()),
        "stop_words": {code: _sorted_set(words) for code, words in sorted(lang._STOP.items())},
        "short_swedish_words": _sorted_set(lang._SHORT_SWEDISH_WORDS),
        "non_en_diacritics": _sorted_set(lang._NON_EN_DIACRITICS),
        "shared_words": _sorted_set(lang._SHARED_WORDS),
        "nordic_overlap_words": _sorted_set(lang._NORDIC_OVERLAP_WORDS),
        "en_only_words": _sorted_set(lang._EN_ONLY_WORDS),
        "en_collision_words": _sorted_set(lang._EN_COLLISION_WORDS),
        "thresholds": {
            name: getattr(lang, name)
            for name in sorted(n for n in dir(lang)
                               if n.isupper() and isinstance(getattr(lang, n), (int, float))
                               and not isinstance(getattr(lang, n), bool))
        },
        "regexes": {
            # The PATTERN TEXT, not a compiled object: a port must translate these deliberately
            # (Python `\w` under `re.UNICODE` is not JavaScript's `\w`, and neither is Java's).
            name: getattr(lang, name).pattern
            for name in ("_WORD", "_IDENTIFIER", "_CODE_LINE", "_JOINED", "_LETTER_RUN")
            if hasattr(lang, name) and hasattr(getattr(lang, name), "pattern")
        },
    }


def _state_repr(state):
    """A state as JSON, with `bytes` tagged so a port can rebuild the exact leaf."""
    if isinstance(state, (bytes, bytearray)):
        return {"kind": "bytes", "value": list(state)}
    if isinstance(state, str):
        return {"kind": "string", "value": state}
    return {"kind": "json", "value": state}


def _unicode_digests():
    """One sha256 per character property, over every code point.

    A digest rather than a table: the committed Java tables already hold the ranges, and what
    needs proving is that they still agree with CPython over the WHOLE of Unicode and not just
    over the corpus below. Eleven hashes do that in 700 bytes of fixture, and they fail the moment
    a JDK upgrade, a table edit or a CPython bump moves a single code point.

    Surrogates are included for the predicates -- Python reports every one of them false, and so
    must a port -- and excluded from the lowercase digest, which no port can represent.
    """
    import hashlib

    word = re.compile(r"[^\W\d_]", re.UNICODE)
    digit = re.compile(r"\d", re.UNICODE)
    predicates = {
        "alpha": lambda ch: ch.isalpha(),
        "word": lambda ch: word.fullmatch(ch) is not None,
        "combining": lambda ch: unicodedata.combining(ch) != 0,
        "digit": lambda ch: digit.fullmatch(ch) is not None,
        "upper": lambda ch: ch.isupper(),
        "lower": lambda ch: ch.islower(),
        "space": lambda ch: ch.isspace(),
        # `laya.email`'s two. `mark` is the Mn/Mc/Me categories, which the `combining` digest
        # above does NOT cover: that is the canonical combining class, and the two disagree on
        # 1,528 code points. `initial` is Lu/Lt/Lo, which `upper` does not cover either --
        # `str.isupper()` is the Uppercase property, and the two sets are not nested in either
        # direction.
        "mark": lambda ch: unicodedata.category(ch).startswith("M"),
        "initial": lambda ch: unicodedata.category(ch) in ("Lu", "Lt", "Lo"),
    }
    out = {}
    for name, predicate in predicates.items():
        digest = hashlib.sha256()
        for cp in range(0x110000):
            digest.update(b"1" if predicate(chr(cp)) else b"0")
        out[name] = digest.hexdigest()

    lower = hashlib.sha256()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        mapped = ",".join(str(ord(c)) for c in chr(cp).lower())
        lower.update(("%d:%s;" % (cp, mapped)).encode("ascii"))
    out["python_lower"] = lower.hexdigest()

    from laya import lang

    script = hashlib.sha256()
    for cp in range(0x110000):
        script.update(((lang._script_of(chr(cp)) or "") + ";").encode("ascii"))
    out["script_of"] = script.hexdigest()
    return out


def _round_to_int_cases():
    """Python's one-argument `round` on halfway values, and on the products `_analyse_text` forms.

    Pinned at the function level and not through the corpus, because the rounding MODE is
    currently UNOBSERVABLE through `analyse`: the rounded value only ever feeds
    `n_non_latin >= NON_LATIN_MIN_LETTERS`, half-to-even and half-up differ only at an exact
    k + 0.5 with k even, and at an EVEN threshold both land on the same side of it. It becomes
    observable the moment that threshold is odd -- at 9, a product of 8.5 is 8 half-to-even and 9
    half-up. A mutant that swapped the mode survived all of the corpus, so the contract is
    recorded here instead of being left to a coincidence in a table.
    """
    values = [k + 0.5 for k in range(0, 24)]
    values += [-(k + 0.5) for k in range(0, 5)]
    values += [0.0, 1.0, 9.0, 10.0, 0.49999999999999994, 9.499999999999998, 10.000000000000002]
    for fraction in (0.1, 0.1234, 0.2, 0.25, 0.3333, 0.5, 0.75, 0.9999, 0.0312):
        for letters in (0, 1, 9, 10, 17, 20, 33, 40, 85, 100, 4000):
            values.append(round(fraction, 4) * letters)
    out, seen = [], set()
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        out.append([value, round(value)])
    return out


def lang_detect():
    """Every output `laya.lang` produces, over a corpus built to break a port.

    WHY EACH CASE IS HERE. The corpus is not a sample of real traffic; every entry exercises a
    branch that a reimplementation gets wrong by default:

      * the four regexes, which no other language spells the same way -- Java's `\\w` admits
        combining marks that Python's does not, and `String.split` drops trailing empty fields
        where Python's keeps them;
      * `str.lower()`, which is not `String.toLowerCase` (final sigma) and not one-to-one (U+0130);
      * `str.isspace()`, which `Character.isWhitespace` disagrees with on four code points, two of
        them the no-break spaces a pasted ticket is full of;
      * `round(x, 4)` and `round(x)`, both of which are half-to-EVEN on the exact binary value;
      * dict insertion order, which decides the script profile's order and every tie between two
        languages with the same score;
      * and the structured-state scans, where a long English stack trace must not outvote a short
        foreign message, and a name field must not pull an English ticket off the English model.

    Each case records the full output of `state_text`, `detect_script`, `script_profile`,
    `latin_profile`, `analyse` and `is_english`. String cases also record every internal helper,
    so a failure names the step that broke rather than only the verdict.
    """
    from laya import lang

    long_english = ("The customer reports that the dashboard will not load after the update. "
                    "We have asked for a screenshot and the browser console output. ") * 40
    cases = [
        ("empty", ""),
        ("whitespace-only", "   \t\n  "),
        ("nbsp-only", "\u00a0\u00a0"),
        ("plain-english", "I cannot log in to my account and the password reset email never arrives."),
        ("english-short", "reset password"),
        ("english-one-word", "refund"),
        ("english-three-words", "please reset this"),
        ("english-with-cafe", "We met at the cafe near the office and the invoice was paid."),
        ("english-with-accented-loanword", "We met at the caf\u00e9 near the office and the invoice was paid."),
        ("english-with-three-accents", "The caf\u00e9 r\u00e9sum\u00e9 of Jos\u00e9 was attached to the ticket."),
        ("english-rescue-two-en-only-words", "Please the invoice caf\u00e9 and we will have it."),
        ("french", "Je ne peux pas me connecter a mon compte et le mot de passe ne fonctionne pas."),
        ("french-accented", "Je n'arrive pas \u00e0 me connecter \u00e0 mon compte, le mot de passe est refus\u00e9."),
        ("spanish", "No puedo iniciar sesion en mi cuenta y la contrasena no funciona para nada."),
        ("spanish-accented", "No puedo iniciar sesi\u00f3n en mi cuenta y la contrase\u00f1a no funciona."),
        ("portuguese-stripped", "Voce pode me mandar a nota fiscal do pedido que eu fiz ontem?"),
        ("portuguese-accented", "Voc\u00ea pode me mandar a nota fiscal do pedido que eu fiz ontem?"),
        ("portuguese-jargon-english", "Deu erro 500 no endpoint de login depois do update de ontem"),
        ("italian", "Non riesco ad accedere al mio account e la password non funziona piu"),
        ("italian-articulated", "Ho ricevuto la fattura nel mese di marzo ma non trovo il pagamento"),
        ("german", "Ich kann mich nicht in mein Konto einloggen und das Passwort wird nicht akzeptiert."),
        ("german-der-twice", "reduzieren der helligkeit der lichter"),
        ("dutch", "Het is niet mogelijk om in te loggen op mijn account met dit wachtwoord"),
        ("swedish", "Jag kan inte logga in pa mitt konto och jag behover hjalp med losenord"),
        ("swedish-login-phrase", "kan inte logga in"),
        ("swedish-short-fragment", "glomt losenord"),
        ("swedish-one-short-word", "losenord"),
        ("danish-nordic-overlap", "Hej jeg kan ikke komme ind pa min konto"),
        ("romanian", "Nu pot sa intru in contul meu si parola nu functioneaza pentru care am"),
        ("romanian-diacritics", "Nu pot s\u0103 intru \u00een contul meu \u0219i parola nu func\u021bioneaz\u0103"),
        ("banglish", "ami amar account e login korte parchi na, password ta kaj korche na"),
        ("azerbaijani", "M\u0259n hesabima daxil ola bilmir\u0259m v\u0259 \u015fifr\u0259 i\u015fl\u0259mir, bu \u00fc\u00e7\u00fcn k\u00f6m\u0259k"),
        ("polish-no-stoplist", "Nie mog\u0119 si\u0119 zalogowa\u0107 na swoje konto, has\u0142o nie dzia\u0142a"),
        ("turkish-no-stoplist", "Hesab\u0131ma giri\u015f yapam\u0131yorum ve \u015fifre \u00e7al\u0131\u015fm\u0131yor, yard\u0131m"),
        ("turkish-dotted-capital-I", "\u0130stanbul \u0130zmir \u0130ngilizce"),
        ("shared-words-only", "la o un de pe ca e que"),
        ("hindi", "\u092e\u0948\u0902 \u0905\u092a\u0928\u0947 \u0916\u093e\u0924\u0947 \u092e\u0947\u0902 \u0932\u0949\u0917 \u0907\u0928 \u0928\u0939\u0940\u0902 \u0915\u0930 \u092a\u093e \u0930\u0939\u093e \u0939\u0942\u0902"),
        ("korean", "\uacc4\uc815\uc5d0 \ub85c\uadf8\uc778\ud560 \uc218 \uc5c6\uc5b4\uc694"),
        ("japanese", "\u30a2\u30ab\u30a6\u30f3\u30c8\u306b\u30ed\u30b0\u30a4\u30f3\u3067\u304d\u307e\u305b\u3093"),
        ("chinese", "\u6211\u65e0\u6cd5\u767b\u5f55\u6211\u7684\u8d26\u6237"),
        ("arabic", "\u0644\u0627 \u0623\u0633\u062a\u0637\u064a\u0639 \u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u062f\u062e\u0648\u0644"),
        ("hebrew", "\u05d0\u05e0\u05d9 \u05dc\u05d0 \u05d9\u05db\u05d5\u05dc \u05dc\u05d4\u05ea\u05d7\u05d1\u05e8"),
        ("greek", "\u0394\u03b5\u03bd \u03bc\u03c0\u03bf\u03c1\u03ce \u03bd\u03b1 \u03c3\u03c5\u03bd\u03b4\u03b5\u03b8\u03ce"),
        ("cyrillic", "\u042f \u043d\u0435 \u043c\u043e\u0433\u0443 \u0432\u043e\u0439\u0442\u0438 \u0432 \u0441\u0432\u043e\u0439 \u0430\u043a\u043a\u0430\u0443\u043d\u0442"),
        ("thai", "\u0e09\u0e31\u0e19\u0e40\u0e02\u0e49\u0e32\u0e2a\u0e39\u0e48\u0e23\u0e30\u0e1a\u0e1a\u0e44\u0e21\u0e48\u0e44\u0e14\u0e49"),
        ("tamil", "\u0b8e\u0ba9\u0bcd \u0b95\u0ba3\u0b95\u0bcd\u0b95\u0bbf\u0bb2\u0bcd \u0b89\u0bb3\u0bcd\u0ba8\u0bc1\u0bb4\u0bc8\u0ba3 \u0bae\u0bc1\u0b9f\u0bbf\u0baf\u0bb5\u0bbf\u0bb2\u0bcd\u0bb2\u0bc8"),
        ("amharic", "\u12a0\u1230\u120b\u121d \u12a8\u1218\u1308\u1263\u1275 \u12a0\u120d\u127d\u120d\u121d"),
        ("georgian", "\u10d5\u10d4\u10e0 \u10d5\u10d0\u10ee\u10d4\u10e0\u10ee\u10d4 \u10d0\u10dc\u10d2\u10d0\u10e0\u10d8\u10e8\u10d8"),
        ("armenian", "\u0535\u057d \u0579\u0565\u0574 \u056f\u0561\u0580\u0578\u0572\u0561\u0576\u0578\u0582\u0574 \u0574\u057f\u0576\u0565\u056c"),
        ("unlisted-script-kawi", "\U00011f00\U00011f01\U00011f02\U00011f03\U00011f04"),
        ("cjk-extension-b", "\U00020000\U00020001\U00020002"),
        ("fullwidth-latin", "\uff28\uff45\uff4c\uff4c\uff4f \uff37\uff4f\uff52\uff4c\uff44"),
        ("latin-ext-additional", "\u1e9e\u1ebd\u1ec5\u1e0d\u1e25"),
        ("ipa-pronunciation", "The name is pronounced [vl\u0250\u02c8d\u02b2im\u02b2\u0268r] in Russian and that is all"),
        ("greek-symbol-in-english", "Set \u03b1 to 0.05 and then re-run the whole evaluation again"),
        ("cyrillic-proper-name", "The reviewer was \u0414\u043c\u0438\u0442\u0440\u0438\u0439 \u041f\u0435\u0442\u0440\u043e\u0432\u0438\u0447 and he approved the change"),
        ("cyrillic-name-with-combining", "The reviewer was \u0412\u043b\u0430\u0434\u0438\u0301\u043c\u0438\u0440 and he approved the change"),
        ("latin-brand-in-cjk", "ACME-ORDER-99281-XYZ \u6ce8\u6587\u304c\u5c4a\u304d\u307e\u305b\u3093"),
        ("latin-plurality-over-cjk", "Order ACME-99281 shipped but the label is wrong \u6ce8\u6587\u756a\u53f7\u304c\u9055\u3044\u307e\u3059\u3088"),
        ("urls-and-emails", "See github.com/acme/repo and mail user@acme.com about v1.2.3 in the U.S.A."),
        ("url-heavy-portuguese-looking", "github.com e os.path com o.com de.com na.com"),
        ("leading-dot-identifier", ".com .org .net and the rest of the list is here now"),
        ("at-after-dot", "a.@b and the rest of this sentence is plain English prose here"),
        ("sentence-final-period", "Il pacco e arrivato. Non trovo la fattura nel portale adesso."),
        ("code-line", "result = round(el, 2); os.path.join(a, b)"),
        ("acronym-heavy-english", "The MON LA EST COM DES game was moved to the next week entirely"),
        ("all-caps-portuguese", "N\u00c3O CONSIGO ENTRAR NA MINHA CONTA E A SENHA N\u00c3O FUNCIONA"),
        ("joined-compound-names", "Nav/Com and OS/2 and C:\\DOS\\mode were all on the same list"),
        ("nbsp-joined-words", "Je\u00a0ne\u00a0peux\u00a0pas\u00a0me\u00a0connecter\u00a0a\u00a0mon\u00a0compte\u00a0du\u00a0tout"),
        ("final-sigma", "\u0394\u0395\u039d \u039c\u03a0\u039f\u03a1\u03a9 \u039d\u0391 \u03a3\u03a5\u039d\u0394\u0395\u0398\u03a9"),
        ("english-with-portuguese-line",
         "Customer ticket #4471\nNao consigo entrar na minha conta e a senha nao funciona\nAgent: asked for a screenshot"),
        ("english-stack-trace-with-german-line",
         "Traceback (most recent call last):\n  File \"app.py\", line 12, in handler\n    raise ValueError(x)\nIch kann mich nicht in mein Konto einloggen und das Passwort geht nicht"),
        ("single-line-english-only", "I cannot log in to my account at all today"),
        ("short-lines-under-seven", "ok\nno\nyes\nfail\nabcdef"),
        # Each of the six below exists because a mutant survived without it. They are not extra
        # samples of traffic; each one is the single discriminating input for one decision.
        #
        # 32 letters of which one is Greek: a non-Latin share of exactly 1/32 = 0.03125, which at
        # four decimals is a halfway case. Python's round gives 0.0312 and the
        # `Math.round(x * 1e4) / 1e4` spelling laya-ts uses gives 0.0313.
        ("one-greek-letter-in-thirty-two", "Set the threshold values to \u0391 and rerun"),
        # The same halfway case on the other rounded field: one diacritic in exactly 32 characters.
        ("one-diacritic-in-thirty-two", "caf\u00e9 " + "x" * 27),
        # Four scripts in the profile, so its ORDER is observable. With fewer than two non-Latin
        # keys an unordered map passes, which is how a HashMap here survived.
        ("three-non-latin-scripts", "\u0391\u0392 \u0411\u0412 \u6f22\u5b57 and some latin words here"),
        # Three Latin letters against three Greek ones. The counts tie, and the tie-break is that
        # Latin is inserted LAST, so the named script wins -- which only holds while the mapping
        # keeps insertion order.
        ("latin-and-greek-tied", "abc \u0391\u0392\u0393"),
        # `todo` is an English word as well as a Spanish one, so it counts once however often it
        # repeats. Counting both occurrences scores Spanish 2, clears the margin, and sends this
        # to the multilingual checkpoint; counting it once scores 1 and leaves it undecided.
        ("collision-word-twice", "todo todo bueno bueno"),
        # German scores 2 against English's 1, which is a margin of exactly one. The reference
        # requires two, so this is English; a margin of one would call it German.
        ("margin-exactly-one-over-english", "der nicht the house"),
        # The branch carrying the sharpest comment in the reference: English wins the whole-text
        # read, so the line-by-line scan runs, and one line alone names a foreign language. These
        # are the only cases that can produce a `mixed_segment`, and they are plain strings on
        # purpose -- a string skips the per-leaf scan, so the segment scan is the only thing that
        # can set it and a port that skipped the scan entirely would still fail here.
        ("english-note-with-portuguese-line",
         "The customer opened this ticket yesterday and we have asked for a screenshot.\n"
         "The browser console shows no errors and the network tab looks clean to me.\n"
         "Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum\n"
         "We will escalate this to the platform team if it is not resolved today."),
        ("portuguese-line-first-then-english",
         "Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum\n"
         "The customer opened this ticket yesterday and we have asked for a screenshot.\n"
         "The browser console shows no errors and the network tab looks clean to me.\n"
         "We will escalate this to the platform team if it is not resolved today."),
        ("english-note-with-german-line",
         "The customer opened this ticket yesterday and we have asked for a screenshot.\n"
         "The browser console shows no errors and the network tab looks clean to me.\n"
         "Ich kann mich nicht in mein Konto einloggen und das Passwort wird nicht akzeptiert\n"
         "We will escalate this to the platform team if it is not resolved today."),
        ("english-note-with-code-line-and-foreign-line",
         "The customer opened this ticket yesterday and we have asked for a screenshot.\n"
         "result = round(el, 2); os.path.join(a, b)\n"
         "Non riesco ad accedere al mio account nel portale e la fattura non arriva\n"
         "We will escalate this to the platform team if it is not resolved today."),
        ("english-note-with-acronym-line-and-foreign-line",
         "The customer opened this ticket yesterday and we have asked for a screenshot.\n"
         "MON LA EST COM DES\n"
         "Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum\n"
         "We will escalate this to the platform team if it is not resolved today."),
    ]
    structured = [
        ("none-state", None),
        ("number-state", 42),
        ("bool-state", True),
        ("empty-dict", {}),
        ("empty-list", []),
        ("bytes-valid", "N\u00e3o consigo entrar na minha conta e a senha n\u00e3o funciona".encode("utf-8")),
        ("bytes-invalid", b"\xff\xfe not utf-8 at all"),
        ("dict-english-note-and-portuguese-message", {
            "note": long_english,
            "message": "Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum",
        }),
        ("dict-portuguese-past-the-cap", {
            "note": long_english,
            "tail": "Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum",
        }),
        ("dict-english-name-field", {"name": "Jos\u00e9", "body": "Please reset my password for this account today"}),
        ("dict-cyrillic-name-field", {"name": "\u0414\u043c\u0438\u0442\u0440\u0438\u0439", "body": "Please reset my password for this account today"}),
        ("dict-all-english", {"a": "Please reset my password", "b": "The dashboard will not load at all"}),
        ("nested-list-of-dicts", [{"t": "Please reset my password for the account"},
                                  {"t": "Ich kann mich nicht in mein Konto einloggen und das Passwort"}]),
        ("deep-nesting-within-limit", [[[[[["Ich kann mich nicht in mein Konto einloggen und das Passwort"]]]]]]),
        ("deep-nesting-past-limit", [[[[[[["Ich kann mich nicht in mein Konto einloggen und das Passwort"]]]]]]]),
        ("dict-with-korean-value", {"id": "ORDER-1", "msg": "\uacc4\uc815\uc5d0 \ub85c\uadf8\uc778\ud560 \uc218 \uc5c6\uc5b4\uc694 \ub3c4\uc640\uc8fc\uc138\uc694"}),
        ("dict-with-code-leaf", {"trace": "result = round(el, 2);\nos.path.join(a, b)",
                                 "msg": "Please take a look at this when you can"}),
        ("list-of-numbers", [1, 2.5, None, True]),
        ("long-single-leaf-over-budget", long_english),
        ("two-leaves-budget-split", ["x" * 3990, "Nao consigo entrar na minha conta e a senha nao funciona"]),
        # A short English note leaves the segment scan enough budget to reach the second field, so
        # here the segment scan names the language and records the field. Its sibling above, whose
        # note is long enough to exhaust the budget, is answered by the per-leaf scan instead and
        # records no segment -- the two together pin both halves of #384.
        ("dict-short-english-note-and-german-message", {
            "note": ("The customer opened this ticket yesterday and we have asked for a "
                     "screenshot. The browser console shows no errors and the network tab "
                     "looks clean to me. We will escalate this to the platform team today. "),
            "message": "Ich kann mich nicht in mein Konto einloggen und das Passwort geht nicht",
        }),
    ]

    def record(name, state):
        text = lang.state_text(state)
        profile = lang.latin_profile(text)
        analysis = lang.analyse(state)
        row = {
            "name": name,
            "state": _state_repr(state),
            "state_text": text,
            "detect_script": lang.detect_script(text),
            "script_profile": lang.script_profile(text),
            "latin_profile": {
                "language": profile["language"],
                "english_hits": profile["english_hits"],
                "diacritic_rate": profile["diacritic_rate"],
                "looks_non_english": profile["looks_non_english"],
            },
            "analyse": {
                "script": analysis["script"],
                "script_profile": analysis["script_profile"],
                "language": analysis["language"],
                "is_english": analysis["is_english"],
                "language_undecided": analysis["language_undecided"],
                "diacritic_rate": analysis["diacritic_rate"],
                "non_latin_fraction": analysis["non_latin_fraction"],
                "mixed_segment": analysis["mixed_segment"],
            },
            "is_english": lang.is_english(state),
        }
        if isinstance(state, str):
            blanked = lang._LETTER_RUN.sub(
                lambda m: " " if m.group().isupper() else m.group(), state)
            row["helpers"] = {
                "words": lang._WORD.findall(state),
                "substitute_identifiers": lang._IDENTIFIER.sub(" ", state),
                "non_latin_words": lang._non_latin_words(state),
                "named_prose_language": lang._named_prose_language(state),
                "has_code_line": bool(lang._CODE_LINE.search(state)),
                "has_joined_tokens": [bool(lang._JOINED.search(tok)) for tok in state.split()],
                "blank_upper_runs": blanked,
                "python_lower": state.lower(),
                "strip": state.strip(),
                "split_whitespace": state.split(),
                "split_newline": state.split("\n"),
                "count_alpha": sum(ch.isalpha() for ch in state),
                "code_point_length": len(state),
                "english_rescued_by_words": lang._english_rescued_by_words(
                    lang._WORD.findall(lang._IDENTIFIER.sub(" ", state).replace("\u0130", "i").lower()),
                    profile["diacritic_rate"]),
            }
        return row

    return {
        "thresholds": {
            "non_en_diacritic_rate": lang.NON_EN_DIACRITIC_RATE,
            "english_rescue_diacritic_rate": lang.ENGLISH_RESCUE_DIACRITIC_RATE,
            "non_latin_fraction": lang.NON_LATIN_FRACTION,
            "non_latin_min_fraction": lang.NON_LATIN_MIN_FRACTION,
            "non_latin_min_letters": lang.NON_LATIN_MIN_LETTERS,
        },
        "unicode_version": unicodedata.unidata_version,
        "unicode_digests": _unicode_digests(),
        "round_to_int": _round_to_int_cases(),
        "cases": [record(name, state) for name, state in cases]
                 + [record(name, state) for name, state in structured],
    }

def presets():
    """The preset question dicts exactly as a caller receives them, and as the model is shown them.

    `laya.presets` builds these; one changed word is a different question and therefore a
    different answer, so a port must generate them rather than retype them.
    """
    from laya import presets as mod

    out = {}
    for name in sorted(n for n in dir(mod) if n.endswith("_questions") or n.endswith("Questions")):
        value = getattr(mod, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, dict):
            out[name] = value

    # `state_field` is the other half of the module and was not recorded: it reads the backtick
    # convention out of each preset's instructions, so a port that retyped one instruction without
    # its backticks would still match the question dicts above and then silently report that the
    # preset names no field. Recorded per preset, plus the inputs that must answer None -- a spec
    # that names nothing, one that names two different fields, and the empty mapping.
    hostile = {
        "no-backticks": {"q": {"type": "noul", "instructions": "Is this urgent?"}},
        "two-fields": {
            "a": {"type": "noul", "instructions": "Does `message` ask for a refund?"},
            "b": {"type": "noul", "instructions": "Is `body` urgent?"},
        },
        "same-field-twice": {
            "a": {"type": "noul", "instructions": "Does `message` ask for a refund?"},
            "b": {"type": "noul", "instructions": "Is `message` urgent?"},
        },
        "empty": {},
        "missing-instructions": {"q": {"type": "noul"}},
        "null-instructions": {"q": {"type": "noul", "instructions": None}},
        "non-word-backticks": {"q": {"type": "noul", "instructions": "Read `a-b` and `c d`."}},
        "underscored-field": {"q": {"type": "noul", "instructions": "Read `customer_message`."}},
        "backtick-in-criteria-only": {
            "q": {"type": "choice", "instructions": "Pick one.",
                  "criteria": {"x": "look at `message`"}},
        },
    }
    out["state_field"] = {name: mod.state_field(spec) for name, spec in out.items()
                          if isinstance(spec, dict)}
    out["state_field_hostile"] = {name: mod.state_field(spec)
                                  for name, spec in hostile.items()}
    out["hostile_specs"] = hostile
    # `email_questions` is the one preset that takes an argument, and a caller's categories must
    # replace the defaults rather than merge with them.
    out["email_questions_custom"] = mod.email_questions(
        {"ops": "incidents and deploys", "legal": "contracts and compliance"})
    out["email_questions_empty_categories"] = mod.email_questions({})
    return out


# A corpus chosen to break a tokenizer port, not to flatter it. Each entry is (id, text); the id
# is the join key between this fixture and the Java test, so it must stay stable.
TOKENIZER_CORPUS = [
    ("empty", ""),
    ("space", " "),
    ("ascii-prose", "We were billed twice for March and want a refund today."),
    ("ascii-punct", "!!! ??? ... --- *** ((())) [[]] {{}} <<>> ~~~ ///"),
    ("digits", "0 1 42 007 3.14159 1,000,000 -17 1e9 0x1F 2026-10-04"),
    ("leading-space", "   leading spaces"),
    ("trailing-space", "trailing spaces   "),
    ("inner-runs", "a  b\t\tc\n\nd\r\ne   f"),
    ("newlines-only", "\n\n\n"),
    ("cjk-zh", "我们三月份被重复收费了两次，请今天退还重复的金额。"),
    ("cjk-ja", "三月に二重請求されました。至急返金してください。"),
    ("cjk-ko", "3월에 두 번 청구되었습니다. 환불해 주세요."),
    ("arabic", "لقد تم محاسبتنا مرتين في شهر مارس، يرجى رد المبلغ المكرر اليوم."),
    ("hebrew", "חויבנו פעמיים במרץ, אנא החזירו את הסכום הכפול."),
    ("devanagari", "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया राशि वापस करें।"),
    ("thai", "เราถูกเรียกเก็บเงินสองครั้งในเดือนมีนาคม"),
    ("cyrillic", "С нас дважды списали оплату в марте, верните деньги."),
    ("greek", "Μας χρέωσαν δύο φορές τον Μάρτιο, θέλουμε επιστροφή."),
    ("mixed-scripts", "Refund 退款 استرداد वापसी now!"),
    # NFC matters: english normalises NFC, multilingual does not. The same grapheme, composed and
    # decomposed, must tokenize the way the real tokenizer says -- not the way a port assumes.
    ("nfc-composed", "caf\u00e9 na\u00efve \u00fcber"),
    ("nfd-decomposed", "cafe\u0301 nai\u0308ve u\u0308ber"),
    ("combining-stack", "a\u0301\u0302\u0303\u0304\u0305"),
    ("emoji", "\U0001f600 \U0001f389 \u2764\ufe0f"),
    ("emoji-zwj", "\U0001f468\u200d\U0001f469\u200d\U0001f466 \U0001f3f3\ufe0f\u200d\U0001f308"),
    ("emoji-skin-tone", "\U0001f44d\U0001f3fd \U0001f64f\U0001f3ff"),
    # U+2581 is the Metaspace replacement itself: a port that substitutes spaces naively will
    # collide with a caller who sent this character on purpose.
    ("metaspace-char", "price\u2581list\u2581here"),
    # Characters absent from the multilingual vocabulary, so `byte_fallback` must fire and emit
    # `<0xNN>` tokens. The curated corpus above contains NOT ONE such character, and a port whose
    # fallback was disabled outright still passed every entry of it while diverging on 25% of a
    # 30,000-string random sweep. The fixture now carries the case the sweep found.
    ("byte-fallback-rare-scripts", "\U00010A00 \u0CF1 \U00016FE0 \U0001E900"),
    ("byte-fallback-in-prose", "refund \U00010A00 now please"),
    ("byte-fallback-adjacent", "\U00010A00\U00010A00\u0CF1"),
    ("zero-width", "a\u200bb\u200cc\u200dd\ufeffe"),
    ("rtl-marks", "a\u202eb\u202cc \u200f\u200e"),
    ("mask-literal-angle", "please <mask> this and <pad> that"),
    ("mask-literal-square", "please [MASK] this and [PAD] that"),
    ("unspaced-long", "a" * 2000),
    ("unspaced-cjk-long", "中" * 1000),
    ("repeated-word", "refund " * 400),
    ("one-char-lines", "\n".join("a" * 500)),
    ("url-ish", "see https://example.com/a/b?c=d&e=f#g and mail to a.b+c@example.co.uk"),
    ("code-ish", "if (x == 1) { return y[0].z(); } // comment"),
    ("tabs-json", '{"a": [1, 2], "b": {"c": null}}'),
]

# Option text long enough that `build_head`'s `max_length=48` truncation at the tokenizer bites.
LONG_OPTION = ("a refund of the duplicate charge together with written confirmation that the "
               "payment method on file has been removed and will not be charged again under any "
               "circumstances whatsoever, including renewals")

# Route cases, each naming the precedence level or detection branch it exercises. `state` is a
# plain string unless noted; `questions` is a bare id -> spec mapping.
_WORKFLOW_IDS = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order",
                           "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive",
                           "urgency"],
}


def _noul_set(ids):
    """A question mapping with those ids, which is all `match_typed_decisions_workflow` reads."""
    return {qid: {"type": "noul", "instructions": "Is `message` about %s?" % qid} for qid in ids}


def router():
    """Everything `laya.router` decides before a checkpoint is loaded.

    The routing decision is the part of the router that is pure logic, and it is the part that
    decides whether a request reaches a checkpoint that can read it. The English checkpoint does
    not degrade gently off English -- 0.100 on 20-option Hindi intent against 0.050 for random,
    while reporting high confidence -- so a wrong route is a wrong answer delivered confidently.

    The `reason` strings are recorded too, not just the model. They are user-visible, they go into
    API responses, and they are built with two Python formats that Java spells differently:
    `%r`, which is CPython's repr, and `%.0f`, which rounds halves to EVEN where
    `String.format("%.0f", ...)` rounds them up. One of the reasons interpolates the mixed
    segment, a slice of the caller's own text, so both formats see arbitrary input.
    """
    from laya import router as mod

    names = ["english", "multilingual", "typed-decisions"]
    aliases = sorted(mod._ALIASES)
    hostile_names = ["", "  ", "ENGLISH", " english ", "English", "nope", "laya-x", "en-US",
                     "typed decisions", "multilingual2",
                     # Padded with whitespace Python strips and Java's String.trim does NOT:
                     # a no-break space, an ideographic space, a narrow no-break space, NEL.
                     # A config file pasted from a browser, or a CJK input method, produces these.
                     "\u00a0english", "english\u00a0", "\u3000multilingual",
                     "typed-decisions\u202f", "\u0085en"]

    def normalise(name):
        try:
            return {"ok": mod.normalise_name(name)}
        except ValueError as failure:
            return {"error": str(failure)}

    def spec(name):
        resolved = mod.resolve_model_spec(name)
        return None if resolved is None else list(resolved)

    english_codes = [
        None, "", "   ", "en", "EN", "En", "eng", "english", "ENGLISH",
        "en-US", "en_US", "en_US.UTF-8", "en-GB", "engx", "enx",
        "fr", "FR", "pt-BR", "pt_BR.UTF-8", "zh-Hans", "de", "x",
        "c", "C", "POSIX", "C.UTF-8", "und", "zxx", "mul", "UND",
        ".", "-", "_", ".UTF-8", "-US", 1, True, 0,
        # Padded with whitespace Python strips and Java's String.trim does not. The first is the
        # one that matters most: "en" plus an ideographic space must still route to English, and
        # a padded "C" must still abstain rather than pin every request to multilingual.
        "\u00a0en", "en\u00a0", "en\u3000", "\u3000en-US", "C\u00a0", "\u202fund", "fr\u0085",
    ]

    schemas = [
        {},
        {"a": {"type": "noul", "instructions": "x"}},
        # Option ORDER is positional in render_options, so two orders are two schemas and must
        # not share a forward pass.
        {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "1", "b": "2"}}},
        {"q": {"type": "choice", "instructions": "x", "criteria": {"b": "2", "a": "1"}}},
        {"q": {"type": "score", "instructions": "x", "criteria": ["low", "high"]}},
        {"b": {"type": "noul", "instructions": "x"}, "a": {"type": "noul", "instructions": "y"}},
        {"q": {"type": "noul", "instructions": "café 中文"}},
    ]

    long_english = ("The customer opened this ticket yesterday and we have asked for a "
                    "screenshot. The browser console shows no errors. ")
    cases = [
        # ---- explicit model wins over everything
        ("explicit-model-english", {"state": "अपने खाते",
                                    "model": "english"}),
        ("explicit-model-alias-en", {"state": "अपने", "model": "en"}),
        ("explicit-model-alias-multi", {"state": "plain english here", "model": "multi"}),
        ("explicit-model-alias-typed", {"state": "plain english here", "model": "typed"}),
        ("explicit-model-uppercase", {"state": "plain english here", "model": "MULTILINGUAL"}),
        ("explicit-model-padded", {"state": "plain english here", "model": "  english  "}),
        # ---- explicit task next
        ("explicit-task-typed-underscore", {"state": "plain english", "task": "typed_decisions"}),
        ("explicit-task-typed-hyphen", {"state": "plain english", "task": "typed-decisions"}),
        ("explicit-task-english", {"state": "अपने", "task": "english"}),
        # ---- detected workflow, only when opted in
        ("workflow-detected-off", {"state": "plain english",
                                   "questions": _noul_set(_WORKFLOW_IDS["customer_service"])}),
        ("workflow-detected-on", {"state": "plain english",
                                  "questions": _noul_set(_WORKFLOW_IDS["customer_service"]),
                                  "auto_task_detection": True}),
        ("workflow-invoice-on", {"state": "plain english",
                                 "questions": _noul_set(_WORKFLOW_IDS["invoice_processing"]),
                                 "auto_task_detection": True}),
        ("workflow-security-on", {"state": "plain english",
                                  "questions": _noul_set(_WORKFLOW_IDS["security_incidents"]),
                                  "auto_task_detection": True}),
        ("workflow-trace-on", {"state": "plain english",
                               "questions": _noul_set(_WORKFLOW_IDS["agent_trace_observability"]),
                               "auto_task_detection": True}),
        # a superset is not a match: an unrelated schema holding `urgency` must not be captured
        ("workflow-superset-on", {"state": "plain english",
                                  "questions": _noul_set(
                                      _WORKFLOW_IDS["customer_service"] + ["extra"]),
                                  "auto_task_detection": True}),
        ("workflow-subset-on", {"state": "plain english",
                                "questions": _noul_set(
                                    _WORKFLOW_IDS["customer_service"][:-1]),
                                "auto_task_detection": True}),
        # the workflow is still REPORTED when a later branch decides, which is why it is a field
        ("workflow-reported-with-lang", {"state": "plain english",
                                         "questions": _noul_set(_WORKFLOW_IDS["customer_service"]),
                                         "lang": "fr"}),
        # ---- explicit lang
        ("lang-en", {"state": "अपने खाते", "lang": "en"}),
        ("lang-en-us", {"state": "अपने", "lang": "en-US"}),
        ("lang-posix", {"state": "plain english words here now", "lang": "en_US.UTF-8"}),
        ("lang-fr", {"state": "plain english words here now", "lang": "fr"}),
        ("lang-blank-falls-through", {"state": "plain english words here now", "lang": "   "}),
        ("lang-agnostic-C-falls-through", {"state": "plain english words here now", "lang": "C"}),
        ("lang-und-falls-through", {"state": "अपने खाते",
                                    "lang": "und"}),
        # ---- lang_guess, per call and installed
        ("guess-per-call-pt", {"state": "plain english words here now", "lang_guess": "pt"}),
        ("guess-per-call-en", {"state": "अपने", "lang_guess": "en"}),
        ("guess-installed-pt", {"state": "plain english words here now",
                                "router_lang_guess": "pt"}),
        ("guess-per-call-beats-installed", {"state": "plain english words here now",
                                            "lang_guess": "en", "router_lang_guess": "pt"}),
        ("guess-abstains-falls-through", {"state": "अपने", "lang_guess": "C"}),
        ("guess-after-lang", {"state": "plain english words here now", "lang": "fr",
                              "lang_guess": "en"}),
        # ---- detection branches
        ("detect-no-letters", {"state": "12345 !!! ---"}),
        ("detect-no-letters-default-multi", {"state": "12345 !!! ---",
                                             "default": "multilingual"}),
        ("detect-non-latin-hindi", {"state": "मैं अपने "
                                             "खाते में"}),
        ("detect-non-latin-korean", {"state": "계정에 로그인할 "
                                              "수 없어요"}),
        ("detect-non-latin-partial", {"state": "Order ACME-99281 shipped but the label is wrong "
                                               "注文番号が違いま"
                                               "すよ"}),
        ("detect-identified-language", {"state": "Você pode me mandar a nota fiscal do "
                                                 "pedido que eu fiz ontem?"}),
        ("detect-mixed-segment", {"state": long_english
                                  + "\nNao consigo entrar na minha conta e a senha nao funciona"}),
        ("detect-undecided-with-diacritics",
         {"state": "Nie mogę się zalogować na swoje konto, hasło nie działa"}),
        ("detect-undecided-no-diacritics", {"state": "Quero cancelar"}),
        ("detect-undecided-default-multi", {"state": "Quero cancelar",
                                            "default": "multilingual"}),
        ("detect-english", {"state": "I cannot log in to my account and the password reset email "
                                     "never arrives."}),
        ("detect-none-state", {"state": None}),
        ("detect-structured-state", {"state": {"id": "T-1", "msg": "Ich kann mich nicht in mein "
                                                                   "Konto einloggen und das "
                                                                   "Passwort"}}),
        # one eighth of the letters non-Latin: the percentage is exactly 12.5, where half-even
        # prints 12 and half-up prints 13
        ("detect-percent-halfway", {"state": "Set the threshold values to Α and rerun"}),
        # The two reasons that carry a percentage, at a share where the rounding mode is visible.
        # Seventy Latin letters and ten lowercase Greek ones is a non-Latin share of exactly
        # 10/80 = 0.125, so the reason reads "12% of letters" with CPython's half-to-even %.0f and
        # "13%" with `String.format`. Without this case the router could misreport a percentage it
        # puts in front of a user and every other route case would still pass.
        ("detect-percent-halfway-non-latin",
         {"state": " ".join(["abcdefg"] * 10)
                   + " \u03b1\u03b2\u03b3\u03b4\u03b5 \u03b6\u03b7\u03b8\u03b9\u03ba"}),
        # And the same on the other percentage: five accented characters in exactly forty
        # CHARACTERS -- the rate is measured over every character, not every letter -- is 0.125,
        # in a state whose language stays unidentified. The trailing "xx" is what makes it forty;
        # at thirty-eight the rate is 0.1316 and rounds the same way either side, which is how the
        # first attempt at this case passed while proving nothing.
        ("detect-percent-halfway-diacritics",
         {"state": "\u00e9\u00e9\u00e9\u00e9\u00e9 zz ww qq vv bb nn mm hh kk ll jjxx"}),
        # a mixed segment holding a no-break space and a quote, so the reason's repr has to escape
        ("detect-mixed-segment-hostile",
         {"state": long_english + "\nNao consigo entrar na 'minha' conta e a senha nao "
                                  "funciona de jeito nenhum"}),
    ]

    def route_case(options):
        router_kwargs = {}
        if "default" in options:
            router_kwargs["default"] = options["default"]
        if "auto_task_detection" in options:
            router_kwargs["auto_task_detection"] = options["auto_task_detection"]
        if "router_lang_guess" in options:
            router_kwargs["lang_guess"] = options["router_lang_guess"]
        instance = mod.Router(**router_kwargs)
        decision = instance._route(
            options.get("state"),
            options.get("questions"),
            model=options.get("model"),
            task=options.get("task"),
            lang=options.get("lang"),
            lang_guess=options.get("lang_guess"),
        )
        detection = decision.get("detection")
        return {
            "options": {k: v for k, v in options.items() if k != "state"},
            "state": _state_repr(options.get("state")),
            "decision": {
                "model": decision["model"],
                "repo": decision["repo"],
                "reason": decision["reason"],
                "workflow": decision.get("workflow"),
                "detection": None if detection is None else {
                    "script": detection["script"],
                    "language": detection["language"],
                    "is_english": detection["is_english"],
                    "language_undecided": detection["language_undecided"],
                    "non_latin_fraction": detection["non_latin_fraction"],
                    "diacritic_rate": detection["diacritic_rate"],
                    "mixed_segment": detection["mixed_segment"],
                },
            },
        }

    return {
        "bundle_repo": mod.BUNDLE_REPO,
        "default_models": {k: list(mod._split(v)) for k, v in mod.DEFAULT_MODELS.items()},
        "standalone_models": {k: list(mod._split(v)) for k, v in mod.STANDALONE_MODELS.items()},
        "aliases": dict(mod._ALIASES),
        "repo_strings": {k: mod._repo_str(v) for k, v in mod.DEFAULT_MODELS.items()},
        "standalone_repo_strings": {k: mod._repo_str(v)
                                    for k, v in mod.STANDALONE_MODELS.items()},
        "max_loaded_default": 2,
        "normalise_name": {name: normalise(name)
                           for name in names + aliases + hostile_names},
        "resolve_model_spec": {name: spec(name)
                               for name in names + aliases + hostile_names},
        "typed_decision_workflows": {k: sorted(v)
                                     for k, v in mod._TYPED_DECISION_WORKFLOWS.items()},
        "match_workflow": [
            [sorted(ids), mod.match_typed_decisions_workflow(_noul_set(ids))]
            for ids in list(_WORKFLOW_IDS.values())
            + [_WORKFLOW_IDS["customer_service"] + ["extra"],
               _WORKFLOW_IDS["customer_service"][:-1],
               ["urgency"], [], ["action", "category", "churn_risk", "needs_human", "urgency",
                                 "action"]]
        ],
        "english_from_code": [[code, mod._english_from_code(code)] for code in english_codes],
        "question_schema": [[schema, mod._question_schema(schema)] for schema in schemas],
        "routes": [dict(route_case(options), name=name) for name, options in cases],
    }
# A deterministic stand-in for a real embedder, so the ranking can be gated across languages.
# FNV-1a over the UTF-8 bytes of "text:component", mapped into [-1, 1). Chosen because it is
# exactly reproducible in any language with 64-bit integer arithmetic -- no floating point, no
# library, no seeded PRNG whose stream differs between runtimes. A real bi-encoder cannot be put
# in a fixture; what needs gating is the ranking, the tie order and the passthrough rule, and those
# do not care where the vectors came from.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_FNV_MASK = 0xFFFFFFFFFFFFFFFF
_EMBED_DIM = 8


def _fnv1a(data: bytes) -> int:
    value = _FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * _FNV_PRIME) & _FNV_MASK
    return value


def _stub_embed(texts):
    import numpy as np

    rows = []
    for text in texts:
        row = []
        for component in range(_EMBED_DIM):
            key = ("%s:%d" % ("" if text is None else text, component)).encode("utf-8")
            row.append((_fnv1a(key) % 2000) / 1000.0 - 1.0)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class _FakeAgent:
    """Records what `predict_shortlist` actually asked, which is the thing to gate."""

    def __init__(self):
        self.seen = None

    def predict(self, state, questions, **kwargs):
        self.seen = {
            "questions": {qid: dict(qdef) for qid, qdef in questions.items()},
            "kwargs": {k: v for k, v in sorted(kwargs.items())},
        }
        return {"answers": {qid: {"stub": True} for qid in questions}, "model": "stub"}


def shortlist():
    """`laya.shortlist`'s ranking, its edge cases, and what it hands to `predict`.

    Choice options share one `head_max_len`, so a large label set leaves a few tokens per label.
    Shortlisting embeds the state and each option, keeps the top k, and runs ONE predict on the
    reduced set. The parts that need gating are not the embedding -- a caller supplies that -- but
    the ranking rules around it, and each of them is a place a port goes wrong silently:

      * the sort is STABLE and descending, so a tie keeps the earlier label. `argsort` without
        `kind="mergesort"` reorders ties, which changes which labels survive at the cut.
      * the score is a SIGNED cosine, not a similarity floor. A label scoring 0 -- no signal, or a
        non-finite vector treated as none -- outranks one scoring negative, and k drops the
        negatives first.
      * a zero-norm query scores everything 0, so the kept set is the first k in criteria order.
      * non-finite components become 0 rather than propagating or raising.
      * k >= the label count is a passthrough: the labels come back in criteria order, scores are
        None, and the embedder is NEVER called.
    """
    from laya import shortlist as mod

    criteria_small = {"refund": "money back", "technical": "a bug", "billing": "an invoice"}
    criteria_large = {
        "card_arrival": "where is my new card",
        "card_delivery_estimate": "when will the card arrive",
        "card_not_working": "the card is declined",
        "cash_withdrawal_charge": "charged for an ATM withdrawal",
        "declined_card_payment": "a payment was declined",
        "direct_debit_payment_not_recognised": "an unrecognised direct debit",
        "exchange_rate": "what rate was applied",
        "failed_transfer": "a transfer did not arrive",
        "lost_or_stolen_card": "the card is gone",
        "pending_card_payment": "a payment is still pending",
        "refund_not_showing_up": "a refund has not arrived",
        "request_refund": "I want money back",
        "reverted_card_payment": "a payment was reversed",
        "top_up_failed": "a top up did not work",
        "transfer_fee_charged": "charged a fee for a transfer",
        "verify_my_identity": "identity verification",
        "wrong_amount_of_cash_received": "the ATM gave the wrong amount",
    }
    criteria_tied = {"a": None, "b": None, "c": None, "d": None}
    criteria_one = {"only": "the only option"}

    states = {
        "string": "My card was charged twice for the same order last Tuesday.",
        "structured": {"subject": "duplicate charge", "body": "billed twice in March"},
        "empty": "",
        "none": None,
        "non_latin": "注文が届きません",
    }

    cases = []

    def record(name, state, criteria, k, instructions=None):
        labels, scores = mod.shortlist_choice(
            state, criteria, _stub_embed, k, instructions=instructions, return_scores=True)
        items = mod._criteria_items(criteria)
        cases.append({
            "name": name,
            "state": _state_repr(state),
            "criteria": criteria,
            "k": k,
            "instructions": instructions,
            "query_text": mod._query_text(state, instructions),
            "option_texts": mod._option_texts(items),
            "labels": list(labels),
            "scores": None if scores is None else [_round_score(v) for v in scores],
            "passthrough": scores is None,
            "n": len(items),
            "subset_criteria": (None if scores is None
                                else mod._subset_criteria(criteria, labels)),
        })

    # A NaN score, which numpy's argsort sorts to the END. Java's Double.compare ranks NaN as
    # the LARGEST double, so a descending comparator puts it FIRST unless that is handled -- a
    # different set of labels then survives the cut and the model is asked a different question.
    def nan_embed(texts):
        import numpy as np
        rows = []
        for text in texts:
            if text.startswith("a:") or text == "a" or text.endswith(" a"):
                rows.append([1e200, 1e200])
            elif ":" in text and text.split(":")[0] == "b":
                rows.append([1.0, 0.0])
            elif ":" in text and text.split(":")[0] == "c":
                rows.append([0.0, 1.0])
            else:
                rows.append([1e200, 1e200])
        return np.asarray(rows, dtype=np.float64)

    nan_criteria = {"a": "A", "b": "B", "c": "C"}
    nan_labels, nan_scores = mod.shortlist_choice(
        "state", nan_criteria, nan_embed, 2, return_scores=True)
    nan_case = {
        "criteria": nan_criteria,
        "k": 2,
        "option_texts": mod._option_texts(mod._criteria_items(nan_criteria)),
        "query_text": mod._query_text("state", None),
        "sims": [_round_score(v) for v in mod._cosine(
            nan_embed([mod._query_text("state", None)])[0],
            nan_embed(mod._option_texts(mod._criteria_items(nan_criteria))))],
        "labels": list(nan_labels),
        "scores": None if nan_scores is None else [_round_score(v) for v in nan_scores],
    }

    record("large-top-5", states["string"], criteria_large, 5)
    record("large-top-1", states["string"], criteria_large, 1)
    record("large-top-16-of-17", states["string"], criteria_large, 16)
    record("large-k-equals-n", states["string"], criteria_large, 17)
    record("large-k-over-n", states["string"], criteria_large, 99)
    record("small-k-2", states["string"], criteria_small, 2)
    record("one-option-k-1", states["string"], criteria_one, 1)
    record("structured-state", states["structured"], criteria_large, 4)
    record("empty-state", states["empty"], criteria_large, 4)
    record("none-state", states["none"], criteria_large, 4)
    record("non-latin-state", states["non_latin"], criteria_large, 4)
    record("with-instructions", states["string"], criteria_large, 4,
           "What does the customer want in `body`?")
    record("instructions-empty-string", states["string"], criteria_large, 4, "")
    # criteria with no descriptions: the option text is the bare label
    record("no-descriptions", states["string"], criteria_tied, 2)
    # a list of labels is a legal choice criteria in the reference
    record("list-criteria", states["string"], ["alpha", "beta", "gamma", "delta"], 2)

    # --- the cosine itself, on the vectors a port gets wrong
    import numpy as np

    cosine_cases = []
    for name, query, docs in [
        ("ordinary", [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        ("zero-query", [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
        ("zero-doc", [1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]]),
        ("all-zero", [0.0, 0.0], [[0.0, 0.0]]),
        ("negatives", [1.0, 1.0], [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]),
        ("needs-clipping", [1e-200, 1e-200], [[1e-200, 1e-200]]),
        ("large-magnitudes", [1e200, 1e200], [[1e200, 1e200], [-1e200, 1e200]]),
        ("no-docs", [1.0, 1.0], []),
        # A query whose norm UNDERFLOWS to zero while its dot product does not. Without the
        # zero-norm guard the score would be 1.4e-200 instead of 0 -- a difference no rounded
        # comparison can see, which is why the port asserts this one exactly.
        ("underflowing-query", [1e-200, 1e-200], [[1.0, 1.0]]),
        # A vector whose self-cosine computes just ABOVE 1, so the clamp is observable. Also
        # invisible to a rounded comparison: 1.0000000000000002 and 1.0 agree to twelve decimals.
        ("self-cosine-over-one", [1.0 / 3.0, 1.0 / 3.0, 8.0 / 3.0],
         [[1.0 / 3.0, 1.0 / 3.0, 8.0 / 3.0]]),
    ]:
        q = np.asarray(query, dtype=np.float64)
        d = np.asarray(docs, dtype=np.float64).reshape(len(docs), len(query) if docs else 0)
        if not docs:
            d = np.zeros((0, len(query)), dtype=np.float64)
        # Tagged, not written bare: a vector whose norm overflows gives inf/inf, so the
        # reference's cosine really does return NaN here -- and `NaN` is not JSON. Writing it
        # bare produced a fixture that Python reads back happily and a strict reader rejects.
        cosine_cases.append({"name": name, "query": query, "docs": docs,
                             "sims": [_round_score(v) for v in mod._cosine(q, d)]})

    # --- non-finite components must become zero, not propagate
    nonfinite = []
    for name, rows in [
        ("nan", [[float("nan"), 1.0], [1.0, 0.0]]),
        ("posinf", [[float("inf"), 1.0], [1.0, 0.0]]),
        ("neginf", [[float("-inf"), 1.0], [1.0, 0.0]]),
        ("all-nan", [[float("nan"), float("nan")], [1.0, 0.0]]),
    ]:
        cleaned = mod._embeddings(lambda texts, rows=rows: np.asarray(rows, dtype=np.float64),
                                 ["q", "d"])
        nonfinite.append({"name": name, "rows": [[_json_float(v) for v in row] for row in rows],
                          "cleaned": [[float(v) for v in row] for row in cleaned]})

    # --- refusals: every input the reference rejects, and the message it rejects it with
    refusals = []
    for name, call in [
        ("k-zero", lambda: mod.shortlist_choice("s", criteria_small, _stub_embed, 0)),
        ("k-negative", lambda: mod.shortlist_choice("s", criteria_small, _stub_embed, -1)),
        ("k-bool", lambda: mod.shortlist_choice("s", criteria_small, _stub_embed, True)),
        ("criteria-empty-dict", lambda: mod.shortlist_choice("s", {}, _stub_embed, 2)),
        ("criteria-empty-list", lambda: mod.shortlist_choice("s", [], _stub_embed, 2)),
        ("criteria-wrong-type", lambda: mod.shortlist_choice("s", "nope", _stub_embed, 2)),
        ("criteria-duplicate-label",
         lambda: mod.shortlist_choice("s", ["a", "a"], _stub_embed, 1)),
        ("embed-not-callable", lambda: mod.shortlist_choice("s", criteria_small, None, 2)),
        ("embed-wrong-rows",
         lambda: mod.shortlist_choice("s", criteria_small, lambda t: [[1.0]], 2)),
        ("embed-one-dimensional",
         lambda: mod.shortlist_choice("s", criteria_small, lambda t: [1.0] * (len(t)), 2)),
        ("questions-not-dict",
         lambda: mod.predict_shortlist(_FakeAgent(), "s", [], _stub_embed, 2)),
        ("choice-without-criteria",
         lambda: mod.predict_shortlist(_FakeAgent(), "s",
                                       {"q": {"type": "choice"}}, _stub_embed, 2)),
    ]:
        try:
            call()
            refusals.append({"name": name, "error": None})
        except Exception as failure:
            refusals.append({"name": name, "error": type(failure).__name__,
                             "message": str(failure)})

    # --- what predict_shortlist hands to predict, which is the whole point of the module
    predicts = []
    for name, questions, k in [
        ("choice-reduced", {"intent": {"type": "choice",
                                       "instructions": "What does the customer want?",
                                       "criteria": criteria_large}}, 4),
        ("choice-passthrough", {"intent": {"type": "choice", "instructions": "x",
                                           "criteria": criteria_small}}, 20),
        ("mixed-questions", {
            "intent": {"type": "choice", "instructions": "What?", "criteria": criteria_large},
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "severity": {"type": "score", "instructions": "How bad?",
                         "criteria": ["low", "high"]},
        }, 3),
        ("two-choices", {
            "a": {"type": "choice", "instructions": "A?", "criteria": criteria_large},
            "b": {"type": "choice", "instructions": "B?", "criteria": criteria_small},
        }, 2),
    ]:
        agent = _FakeAgent()
        original = {qid: dict(qdef) for qid, qdef in questions.items()}
        result = mod.predict_shortlist(agent, states["string"], questions, _stub_embed, k)
        predicts.append({
            "name": name,
            "k": k,
            "questions": original,
            "asked": agent.seen["questions"],
            "shortlist": _round_shortlist(result["shortlist"]),
            "caller_questions_unmutated": original == {qid: dict(q)
                                                       for qid, q in questions.items()},
        })

    payload = {
        "score_decimals": _SCORE_DECIMALS,
        "default_k": mod.DEFAULT_SHORTLIST_K,
        "embed_dim": _EMBED_DIM,
        "embed_probe": {text: [float(v) for v in _stub_embed([text])[0]]
                        for text in ["", "a", "refund: money back", states["string"]]},
        "cases": cases,
        "nan_ranking": nan_case,
        "cosine": cosine_cases,
        "nonfinite": nonfinite,
        "cache_counters": _cache_counter_cases(),
        "refusals": refusals,
        "predicts": predicts,
    }
    _assert_scores_are_rounded(payload)
    return payload


# How many decimals of a cosine are safe to record. The reference computes its scores with
# `np.dot(matrix, vector)`, a BLAS matrix-vector product, and how BLAS blocks that accumulation
# differs between implementations -- measured here: on Apple Accelerate one row of a four-row
# product differs from the same row computed on its own, in the last bit. So the raw score is not
# reproducible across machines, let alone across languages: recording it verbatim would make this
# fixture disagree with itself between a developer's macOS and CI's Linux, and the drift gate would
# fail for a reason that has nothing to do with any port.
#
# Twelve decimals is about four orders of magnitude tighter than any porting error that has ever
# shown up in this repository, and about four orders LOOSER than the 1e-16 BLAS spread, so it
# separates the two cleanly. What stays exact is the LABEL ORDER, which is the only thing a
# consumer can observe -- and the thing a ranking bug actually changes.
# Nine, measured rather than chosen. A cosine here is a dot product of 8-dimensional unit
# vectors, and BLAS blocking moves the last bit: the worst difference between numpy's `dot` and a
# sequential sum over 4,000 random pairs is 2.22e-16, about one ULP near 1.
#
# A decimal snapshot of a platform-dependent float can always straddle a rounding boundary, so the
# only question is how often. With 194 floats in this file the chance that at least one lands
# within the BLAS error of a boundary is about 8.3% at twelve decimals -- which is what happened:
# the drift gate failed on Linux x64 while reproducing exactly on arm64, with the same numpy,
# torch and transformers pins. At nine it is 0.0085%, and nine is still three orders of magnitude
# finer than any real porting bug: a wrong normalisation, a missing clamp or an inverted tie order
# moves a score in the third decimal, not the tenth.
_SCORE_DECIMALS = 9


def _cache_counter_cases():
    """What `cached_embed_fn`'s counters report, call by call.

    Recorded because the two counters are on the same basis and it is easy to put them on
    different ones: `hits` counts OCCURRENCES and so does `misses`, so `hits + misses` equals the
    number of texts looked up. Counting misses per DISTINCT text instead -- which is the obvious
    reading of "embed each text once" -- breaks that identity on the very first repeated text.
    """
    from laya import shortlist as mod

    calls = []
    embedder = mod.cached_embed_fn(_stub_embed, maxsize=64)
    for texts in (["a", "a", "b"], ["a", "b", "c"], ["q", "a"]):
        embedder(list(texts))
        info = embedder.cache_info()
        calls.append({"texts": list(texts), "size": info["size"], "maxsize": info["maxsize"],
                      "hits": info["hits"], "misses": info["misses"]})
    return calls


def _round_shortlist(shortlist):
    """Round the scores inside a `predict_shortlist` result, as the `cases` section rounds.

    The reference's rankings were being handed through VERBATIM here, so their scores were raw
    float64 -- sixteen and seventeen significant digits -- while the same numbers in `cases` went
    through `_round_score`. That is the drift the build lane caught: three values in this section
    differed in their last bit between arm64 and x64, because the recorded form kept every bit
    there was to differ in.
    """
    out = {}
    for question_id, ranking in shortlist.items():
        copy = dict(ranking)
        scores = copy.get("scores")
        if scores is not None:
            copy["scores"] = [_round_score(v) for v in scores]
        out[question_id] = copy
    return out


def _assert_scores_are_rounded(payload):
    """Refuse to emit a shortlist fixture holding a score finer than the recorded precision.

    The bug this prevents was not the precision, it was a MISSED call site: one section rounded and
    another did not, and nothing said so. A grep for `_round_score` found two call sites and both
    looked right; what was wrong was a third place that needed one and had none. So the check is on
    the OUTPUT rather than on the code -- every float under a "scores" key, anywhere in the family,
    at any depth.
    """
    bad = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "scores" and isinstance(value, list):
                    for i, score in enumerate(value):
                        if not isinstance(score, float):
                            continue        # a tagged non-finite, which has no decimals
                        if score != round(score, _SCORE_DECIMALS):
                            bad.append(("%s/scores[%d]" % (path, i), score))
                else:
                    walk(value, path + "/" + str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, path + "[%d]" % i)

    walk(payload, "")
    if bad:
        lines = "\n".join("    %s = %r" % (where, value) for where, value in bad[:10])
        raise SystemExit(
            "gen_fixtures: %d score(s) are recorded finer than %d decimals, so the fixture cannot "
            "survive a different BLAS:\n%s\n  round them with _round_score at the site that "
            "emits them." % (len(bad), _SCORE_DECIMALS, lines))


def _round_score(value):
    """A cosine rounded to a precision BLAS blocking is unlikely to move, or tagged if non-finite.

    NOT "cannot move" -- that is what this said, and it was wrong. See _SCORE_DECIMALS.
    """
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return _json_float(value)
    return round(value, _SCORE_DECIMALS)


def _json_float(value):
    """NaN and the infinities have no JSON spelling, so they are tagged for a port to rebuild."""
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return value

def tokenizer_ids():
    """Exact token ids per checkpoint for a corpus designed to break a port.

    Ids, not answers: a tokenizer divergence that changes an answer is almost impossible to
    localise from the answer, and trivial from the ids. Each checkpoint records what it IS as well
    -- model type, pre-tokenizer, normaliser, vocab and merge counts, special ids -- so a fixture
    generated against a different checkpoint cannot be mistaken for a matching one.
    """
    import os

    from laya.common import encode_text

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         DEFAULT_CHECKPOINT_ROOT)
    out = {}
    for name in ("english", "multilingual"):
        root = os.path.join(rig, name)
        if not os.path.isdir(root):
            out[name] = {"skipped": "checkpoint not present at %s" % root}
            continue
        # Resolve the tokenizer directory EXACTLY as laya resolves it at run time:
        # `os.path.join(model_dir, "tokenizer")`, falling back to the configured encoder
        # (`onnx_agent.py:156`, `agent.py:235`). Searching for the shortest path containing a
        # `tokenizer.json` is not the same thing and silently picked the wrong directory: the
        # multilingual checkpoint ships `tokenizer.json` at BOTH its root and its `tokenizer/`
        # subdirectory -- byte-identical, same sha256 -- but only the subdirectory carries
        # `tokenizer_config.json`. Loading the root therefore reported cls/sep/mask/pad as all
        # None, and a port built against that fixture would have handled a checkpoint with no
        # special tokens (which does not exist) while missing the real ids: multilingual reuses
        # `<bos>` as CLS and `<eos>` as SEP, which nothing about the tokenizer itself reveals.
        directory = os.path.join(root, "tokenizer")
        if not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            found = [os.path.join(d, "tokenizer.json")
                     for d, _subdirs, files in os.walk(root) if "tokenizer.json" in files]
            if not found:
                out[name] = {"skipped": "no tokenizer.json under %s" % root}
                continue
            # A TOTAL order: `key=len` alone ties on equal-length candidates and the winner
            # then depends on filesystem walk order -- which decides the tokenizer directory, and
            # therefore every recorded id in this family. That is the same class of bug the
            # comment above describes having been hit once already.
            directory = os.path.dirname(sorted(found, key=lambda path: (len(path), path))[0])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(directory)
        raw = json.load(open(os.path.join(directory, "tokenizer.json"), encoding="utf-8"))
        out[name] = {
            "identity": {
                "tokenizer_dir": os.path.relpath(directory, rig),
                "model_type": raw["model"]["type"],
                "vocab_size": len(raw["model"].get("vocab", {})),
                "merges": len(raw["model"].get("merges", [])),
                "pre_tokenizer": (raw.get("pre_tokenizer") or {}).get("type"),
                "normalizer": (raw.get("normalizer") or {}).get("type"),
                "unk_token": raw["model"].get("unk_token"),
                "added_tokens": len(raw.get("added_tokens") or []),
                # The english checkpoint keeps 88 of its added tokens OUT of `model.vocab`, with
                # ids above the vocabulary's own maximum -- `[MASK]` among them. A port that built
                # its id map from `model.vocab` alone would have no id for masked prediction, so
                # the count is part of the checkpoint's identity rather than a detail.
                "added_tokens_outside_vocab": sum(
                    1 for a in (raw.get("added_tokens") or [])
                    if a["content"] not in raw["model"].get("vocab", {})),
                "byte_fallback": raw["model"].get("byte_fallback"),
                "ignore_merges": raw["model"].get("ignore_merges"),
                "fuse_unk": raw["model"].get("fuse_unk"),
            },
            "special_ids": {
                "cls": tok.cls_token_id, "sep": tok.sep_token_id,
                "mask": tok.mask_token_id, "pad": tok.pad_token_id,
                "mask_token": tok.mask_token,
            },
            # add_special_tokens=False everywhere: `build_sequence` adds [CLS]/[SEP] itself, so the
            # contract a port must match is the bare encoding.
            "corpus": {cid: encode_text(tok, text, add_special_tokens=False)["input_ids"]
                       for cid, text in TOKENIZER_CORPUS},
            # the 48-token cap `build_head` applies, and the same text uncapped, so a port cannot
            # pass by slicing after the fact
            "option_truncation": {
                "text": LONG_OPTION,
                "uncapped": encode_text(tok, " " + LONG_OPTION,
                                        add_special_tokens=False)["input_ids"],
                "capped_48": encode_text(tok, " " + LONG_OPTION, add_special_tokens=False,
                                         truncation=True, max_length=48)["input_ids"],
            },
        }
    # The corpus TEXT, once, beside the per-checkpoint ids. A port needs the inputs as well as the
    # expected outputs, and transcribing 37 hostile strings -- 2,000-character runs, lone
    # surrogates' worth of zero-width marks, RTL overrides -- into a second language by hand is
    # exactly how a parity suite ends up asserting against a corpus that is not the one Python
    # measured. Shared across checkpoints because both encode the same inputs.
    out["corpus_text"] = {cid: text for cid, text in TOKENIZER_CORPUS}
    return out


# Sequence cases chosen to exercise every branch of `build_head`'s budget arithmetic and every
# rendering rule, not to read nicely. `MASKS` is substituted per checkpoint so one case can test
# that each model scrubs ITS OWN mask string: the english checkpoint masks with `[MASK]` and the
# multilingual one with `<mask>`, and a port that hard-codes either lets a caller inject option
# markers into the other.
SEQUENCE_CASES = [
    ("choice-basic", "The customer was billed twice in March.",
     {"t": "choice", "ins": "What should we do?",
      "crit": {"refund": "send the money back", "replace": "ship a new unit"}}, {}),
    ("choice-no-description", "Billed twice.",
     {"t": "choice", "ins": "Approve?", "crit": {"yes": None, "no": ""}}, {}),
    ("choice-falsy-criteria", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"zero": 0, "false": False, "none": None}}, {}),
    ("choice-structured-criteria", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {"a": {"desc": "nested", "n": 1}, "b": [1, 2, "x"], "c": 2.5}}, {}),
    ("choice-many-options", "Billed twice.",
     {"t": "choice", "ins": "Pick one of many.",
      "crit": {("opt%02d" % i): ("criterion number %d with some words" % i) for i in range(40)}}, {}),
    ("choice-long-option", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {"short": "ok", "long": " ".join(["verylongcriterion%d" % i for i in range(120)])}}, {}),
    # Two options whose rendered text is identical for the first 48 tokens, so the tokenizer cap
    # collapses them to the SAME span. The marker count still matches the option count, so the
    # guard in `Agent._encode_state` passes and nothing downstream can tell the question lost the
    # ability to name them apart (#538) -- only `options_distinct` records it. The difference has
    # to sit past token 48, which means it must be in the LABEL's tail: differing labels put it at
    # token 1 and the collision never happens, which an earlier version of this case got wrong.
    ("choice-colliding-options", "Billed twice.",
     {"t": "choice", "ins": "Pick.",
      "crit": {("prefix " * 60 + "one"): "d", ("prefix " * 60 + "two"): "d"}}, {}),
    ("score-levels", "The reply was polite but slow.",
     {"t": "score", "ins": "Rate the reply.", "crit": ["terrible", "poor", "fine", "good"]}, {}),
    ("score-structured", "The reply was polite but slow.",
     {"t": "score", "ins": "Rate.", "crit": [{"d": 1}, "ok", 3]}, {}),
    ("noul-default", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "The invoice is from March."}, {}),
    ("noul-criteria", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "Holds?", "crit": {"false": "no it does not", "true": "yes it does"}}, {}),
    ("noul-custom-labels", "The invoice is dated March 2nd.",
     {"t": "noul", "ins": "Holds?", "labels": {"false": "nope", "true": "yep"}}, {}),
    ("mask-in-instruction", "Billed twice.",
     {"t": "choice", "ins": "Pick MASKS one MASKS now", "crit": {"a": "x", "b": "y"}}, {}),
    ("mask-in-option", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x MASKS y", "b": "MASKS"}}, {}),
    ("mask-in-state", "state with MASKS inside it",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("long-instruction", "Billed twice.",
     {"t": "choice", "ins": "instruction words " * 300, "crit": {"a": "x", "b": "y"}}, {}),
    ("long-state", "evidence sentence number one. " * 400,
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("long-state-truncate-left", "evidence sentence number one. " * 400,
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {"truncate_left": True}),
    ("state-dict", {"order": 1, "items": ["a", "b"], "n": 2.5, "ok": True, "missing": None},
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("state-list", [1, "a", {"b": 2}, None, True, 1.5],
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    # `serialize_state` and `render_criterion` both go through `json.dumps`, so a port needs
    # Python's NUMBER formatting, not its own language's. Python writes floats with `repr`
    # (shortest round-trip, scientific outside [1e-4, 1e16) with a two-digit exponent) and
    # integers at arbitrary precision; Java's `Double.toString` disagrees on both the threshold
    # and the spelling, and JDK 17's is not even always shortest. These values make the
    # disagreement a token-id difference instead of a latent one.
    ("state-hostile-numbers",
     {"big": 1e16, "small": 1e-05, "edge": 0.0001, "third": 1.0 / 3.0, "huge": 1e23,
      "whole": 2.0, "negzero": -0.0, "max": 1e308, "denormal": 5e-324,
      "bigint": 123456789012345678901234567890, "negint": -7},
     {"t": "choice", "ins": "Pick.", "crit": {"a": 1e16, "b": 1.0 / 3.0}}, {}),
    ("state-empty", "",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}}, {}),
    ("state-unicode", "\u6211\u4eec\u88ab\u91cd\u590d\u6263\u8d39 \U0001f600 \u0644\u0642\u062f \u062a\u0645",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "\u9000\u6b3e", "b": "\u0627\u0633\u062a\u0631\u062f\u0627\u062f"}}, {}),
    ("option-order-permuted", "Billed twice.",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y", "c": "z"}},
     {"option_order": [2, 0, 1]}),
    ("tiny-budget", "some state here that will not fit at all",
     {"t": "choice", "ins": "a fairly long instruction that cannot fit either",
      "crit": {"a": "alpha", "b": "beta", "c": "gamma"}},
     {"max_len": 24, "head_max_len": 16}),
    ("head-larger-than-max", "state",
     {"t": "choice", "ins": "Pick.", "crit": {"a": "x", "b": "y"}},
     {"max_len": 12, "head_max_len": 192}),
]


def sequences():
    """`build_sequence` output for every rendering and budget branch, per checkpoint.

    Ids AND markers AND stats: a marker that is off by one still produces a plausible answer, and
    the option-collision counter (`options_distinct`) is invisible from the ids alone.
    """
    import copy
    import os

    from laya.common import build_sequence

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         DEFAULT_CHECKPOINT_ROOT)
    out = {}
    for name in ("english", "multilingual"):
        root = os.path.join(rig, name)
        directory = os.path.join(root, "tokenizer")
        if not os.path.isfile(os.path.join(directory, "tokenizer.json")):
            out[name] = {"skipped": "no tokenizer/ under %s" % root}
            continue
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(directory)
        cfg = json.load(open(os.path.join(root, "rl_agent_config.json"), encoding="utf-8"))
        default_max = cfg.get("max_len", 512)
        default_head = cfg.get("head_max_len", 192)
        cases = {}
        for cid, state, question, kwargs in SEQUENCE_CASES:
            q = copy.deepcopy(question)
            st = state
            # one case per checkpoint substitutes that checkpoint's own mask string
            if isinstance(st, str):
                st = st.replace("MASKS", tok.mask_token)
            q["ins"] = q["ins"].replace("MASKS", tok.mask_token)
            if isinstance(q.get("crit"), dict):
                q["crit"] = {k: (v.replace("MASKS", tok.mask_token) if isinstance(v, str) else v)
                             for k, v in q["crit"].items()}
            max_len = kwargs.get("max_len", default_max)
            head_max_len = kwargs.get("head_max_len", default_head)
            ids, markers, stats, trunc = build_sequence(
                tok, st, q, max_len=max_len, head_max_len=head_max_len,
                option_order=kwargs.get("option_order"),
                truncate_left=kwargs.get("truncate_left", False),
                return_stats=True, return_truncation_stats=True)
            cases[cid] = {
                "state": st, "question": q, "max_len": max_len, "head_max_len": head_max_len,
                "option_order": kwargs.get("option_order"),
                "truncate_left": kwargs.get("truncate_left", False),
                "ids": ids, "markers": markers, "stats": stats, "truncation": trunc,
            }
        out[name] = {
            "config": {"max_len": default_max, "head_max_len": default_head,
                       "temperature": cfg.get("temperature"),
                       "temperature_by_options": cfg.get("temperature_by_options"),
                       "encoder": cfg.get("encoder")},
            "special_ids": {"cls": tok.cls_token_id, "sep": tok.sep_token_id,
                            "mask": tok.mask_token_id, "pad": tok.pad_token_id,
                            "mask_token": tok.mask_token},
            "cases": cases,
        }
    return out


def decode_answers():
    """`Agent._decode_answers` output for hostile logit rows, per checkpoint.

    Driven through the REAL method rather than a transcription of it: the expectations have to come
    from the shipped decoder, including its rounding, its argmax tie-breaking and its clamps.

    Logits are synthesised rather than taken from a forward pass so the case list can aim at the
    branches -- every temperature bucket, a tie at the top, a row whose softmax is exactly uniform,
    magnitudes that would overflow a naive `exp`, and the language override -- and so the fixture
    does not need a 1.2 GB graph to regenerate.
    """
    import os

    import numpy as np

    from laya.agent import Agent, QTYPES
    from laya.common import clamp_temperature, resolve_lang_temperatures

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         DEFAULT_CHECKPOINT_ROOT)

    def row(values):
        return np.asarray(values, dtype=np.float32)

    # (id, question, option_order, lang, logits)
    cases = [
        ("choice-2", {"t": "choice", "ins": "i", "crit": {"a": "x", "b": "y"}},
         None, None, [2.0, 1.0]),
        ("choice-2-tied", {"t": "choice", "ins": "i", "crit": {"a": "x", "b": "y"}},
         None, None, [1.0, 1.0]),
        ("choice-4", {"t": "choice", "ins": "i",
                      "crit": {"a": "1", "b": "2", "c": "3", "d": "4"}},
         None, None, [0.5, 2.5, -1.0, 2.5]),
        ("choice-4-permuted", {"t": "choice", "ins": "i",
                               "crit": {"a": "1", "b": "2", "c": "3", "d": "4"}},
         [2, 0, 3, 1], None, [0.5, 2.5, -1.0, 2.5]),
        ("choice-7", {"t": "choice", "ins": "i",
                      "crit": {("o%d" % i): str(i) for i in range(7)}},
         None, None, [0.1 * i for i in range(7)]),
        ("choice-12", {"t": "choice", "ins": "i",
                       "crit": {("o%d" % i): str(i) for i in range(12)}},
         None, None, [(-1.0) ** i * i * 0.3 for i in range(12)]),
        ("choice-20-uniform", {"t": "choice", "ins": "i",
                               "crit": {("o%d" % i): str(i) for i in range(20)}},
         None, None, [0.0] * 20),
        ("choice-huge-magnitude", {"t": "choice", "ins": "i",
                                   "crit": {"a": "1", "b": "2", "c": "3"}},
         None, None, [700.0, 699.0, -700.0]),
        ("choice-tiny-spread", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, None, [1.0, 1.0 + 1e-7]),
        ("choice-lang-override", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "zh-Hans", [2.0, 1.0]),
        ("choice-lang-override-prefix", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "ZH", [2.0, 1.0]),
        ("choice-lang-unknown", {"t": "choice", "ins": "i", "crit": {"a": "1", "b": "2"}},
         None, "xx", [2.0, 1.0]),
        ("score-3", {"t": "score", "ins": "i", "crit": ["bad", "ok", "good"]},
         None, None, [0.2, 1.4, 0.9]),
        ("score-6", {"t": "score", "ins": "i", "crit": [str(i) for i in range(6)]},
         None, None, [0.0, 1.0, 2.0, 1.0, 0.0, -1.0]),
        ("score-structured-legend", {"t": "score", "ins": "i", "crit": [{"d": 1}, "ok", 3]},
         None, None, [0.4, 0.4, 0.4]),
        ("score-permuted", {"t": "score", "ins": "i", "crit": ["bad", "ok", "good"]},
         [1, 2, 0], None, [0.2, 1.4, 0.9]),
        ("noul-true", {"t": "noul", "ins": "i"}, None, None, [0.3, 1.9]),
        ("noul-false", {"t": "noul", "ins": "i"}, None, None, [2.4, 0.1]),
        ("noul-balanced", {"t": "noul", "ins": "i"}, None, None, [1.0, 1.0]),
        ("noul-labels", {"t": "noul", "ins": "i",
                         "labels": {"false": "nope", "true": "yep"}}, None, None, [0.3, 1.9]),
    ]

    out = {}
    for name in ("english", "multilingual"):
        path = os.path.join(rig, name, "rl_agent_config.json")
        if not os.path.isfile(path):
            out[name] = {"skipped": "no rl_agent_config.json for %s" % name}
            continue
        cfg = json.load(open(path, encoding="utf-8"))
        # Clamped exactly as the runtime clamps: `choice:11+` ships at 0.1006 on one checkpoint,
        # a ~10x sharpener, and a port that applied the raw value would publish a coin flip as a
        # certainty. The fixture therefore records the CLAMPED table the decoder actually uses.
        temperature = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        by_options = {k: clamp_temperature(v)
                      for k, v in (cfg.get("temperature_by_options") or {}).items()}
        # Neither shipped checkpoint sets `lang_temperatures`, so the override branch would never
        # be exercised by their configs. One is supplied here, through the same resolver the
        # runtime uses, so a port cannot skip the branch and still pass.
        lang_raw = {"zh": {"temperature": [2.0, 1.5, 3.0],
                           "temperature_by_options": {"choice:2": 2.5}}}
        lang_temperatures = resolve_lang_temperatures(lang_raw, temperature)

        shim = Agent.__new__(Agent)
        shim.temperature = temperature
        shim.temperature_by_options = by_options
        shim.lang_temperatures = lang_temperatures
        shim.binning_map = None

        recorded = {}
        for cid, question, option_order, lang, logits in cases:
            k = len(logits)
            q = dict(question)
            if option_order is not None:
                q["option_order"] = option_order
            items = [{"markers": list(range(k))}]
            # a wider logits block than k, so a port that forgets to slice to k columns is caught
            block_width = k + 3
            padded = row(list(logits) + [99.0] * 3).reshape(1, block_width)
            act = row([0.25, -0.5]).reshape(1, 2)
            answers = Agent._decode_answers(shim, padded, act, items, [cid], {cid: q}, 0, lang)
            recorded[cid] = {
                "question": question, "option_order": option_order, "lang": lang,
                "logits": [float(x) for x in logits], "logits_block_width": block_width,
                "act": [0.25, -0.5], "answer": answers[cid],
            }
        out[name] = {
            "config": {"temperature": temperature, "temperature_by_options": by_options,
                       "lang_temperatures_raw": lang_raw},
            "qtypes": QTYPES,
            "cases": recorded,
        }
    return out


# Strings `repr` is asked for, chosen for the rule each one exercises rather than for realism:
# quote selection in all four combinations, the letter escapes, the three widths of numeric escape,
# and the printable non-ASCII that CPython deliberately does NOT escape.
_REPR_CASES = [
    "", "english", "multilingual", "pt", "typed_decisions",
    "it's here", 'say "hi"', 'both \' and "', "'", '"', "'\"",
    "line\nbreak", "tab\there", "carriage\rreturn", "back\\slash", "bell\a", "null\0",
    "caf\u00e9", "\u4e2d\u6587", "\U0001f600", "\U00011f00",          # printable, left alone
    "a\u00a0b", "a\u00adb", "a\u200bb", "a\u2028b", "a\u0085b",       # unprintable, escaped
    "a\ufeffb", "a\u202eb", "a\ue000b", "a\u0378b", "a\x7fb",
    "a\u0301b",                                                      # a combining mark IS printable
    "Nao consigo entrar na minha conta e a senha nao funciona",
    "Ich kann mich nicht in mein Konto einloggen und das Passwort",
    "MON LA EST COM DES",
    "mixed \u00a0 and \u200b and 'quotes' in one line",
]

# Fractions whose percentage lands on a halfway value, plus ordinary ones. A share of one letter in
# eight is exactly 12.5%, which is where half-up and half-even part company.
_PERCENT_CASES = [
    0.0, 1.0, 0.005, 0.015, 0.025, 0.045, 0.125, 0.135, 0.205, 0.5, 0.625,
    0.0312, 0.0313, 0.1, 0.2222, 0.2941, 0.3333, 0.4, 0.6667, 0.7778, 0.9999,
    1.0 / 3.0, 2.0 / 3.0, 1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0, 1.0 / 16.0,
]


def _repr_digest():
    """One sha256 over `repr` of every code point, so the escape rule is proven exhaustively.

    The hostile list above covers the rules a reader can name. This covers the ones nobody
    enumerated: 148,998 of Unicode's code points are printable and the rest are not, and the
    boundary between them moves with the Unicode version.
    """
    import hashlib

    digest = hashlib.sha256()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        digest.update(repr("a" + chr(cp) + "b").encode("utf-8"))
    return digest.hexdigest()


def python_json():
    """CPython's `json.dumps(ensure_ascii=False)` and `round(v, 4)` on values a port gets wrong.

    Needs no checkpoint, so these are the parity tests CI can run without downloading a model.

    Doubles are carried as raw 64-bit patterns rather than as decimal text: the point of the
    fixture is the SPELLING of a double, so round-tripping the operand through a decimal literal
    would lose the very thing being measured.

    **Nothing here may go through libm.** `math.exp`, and `10 ** n` for a large negative n, are C
    library calls, and those are not bit-identical across platforms: this family regenerated
    differently on Linux than on macOS and CI reported the committed copy as stale. Every value is
    now produced by an exact route -- a decimal literal (correctly rounded by strtod), IEEE
    arithmetic, integer division, or a raw bit pattern -- so the file is a property of the seed
    rather than of the machine that ran the generator.
    """
    import random
    import struct

    def bits(value):
        return struct.unpack("<Q", struct.pack("<d", value))[0]

    # Values where Java's own spelling differs from Python's, plus the boundaries of the
    # fixed/scientific switch, the subnormal floor, and both infinities.
    hostile = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 1.0 / 3, 2.0 / 3, 0.1, 0.2, 0.3,
               1e-5, 1e-4, 0.0001, 1e15, 1e16, 1e17, 1e23, 1e100, 1e-100, 1e308, 5e-324,
               2.2250738585072014e-308, 1.7976931348623157e308, 9007199254740993.0,
               1e7, 1e-3, 123456789.123456789,
               # pi and e as literals rather than `math.pi` / `math.e`: the constants are
               # identical everywhere, but writing them out keeps this family free of any `math`
               # reference at all, which is the rule the docstring states.
               3.141592653589793, 2.718281828459045,
               float("inf"), float("-inf"), float("nan")]
    rng = random.Random(1091)
    doubles = list(hostile)
    # Enough random draws to cover the exponent range and the raw bit patterns, kept small enough
    # that the committed file stays reviewable: the hostile list above is what actually
    # discriminates a wrong implementation, and these are the sweep behind it.
    #
    # `random()` is exact -- it scales Mersenne Twister integers by a power of two -- and a raw bit
    # pattern is exact by construction and reaches the subnormals, the infinities and the NaNs that
    # arithmetic would not. The previous version multiplied by `10 ** rng.randint(-320, 300)`,
    # which is a libm `pow`, and that is what made this file machine-dependent.
    for _ in range(500):
        kind = rng.randrange(3)
        if kind == 0:
            doubles.append(rng.random())
        elif kind == 1:
            doubles.append(-rng.random())
        else:
            doubles.append(struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0])

    quote, back, newline, tab = chr(34), chr(92), chr(10), chr(9)
    structured = [
        {"b": 1, "a": 2, "z": [1, "x", None, True, 2.5]},
        {"nested": {"k": {"deep": [1.5, -0.0]}}},
        ["a", "b" + quote + "c", "d" + back + "e", "f" + newline + "g" + tab + "h",
         chr(0) + chr(31) + chr(1), chr(0x2028) + chr(0x2029), "/",
         chr(0xE9) + chr(0x4E2D) + chr(0x1F600)],
        {chr(0x4E2D) + chr(0x6587): chr(0x503C), "emoji " + chr(0x1F600): [1e16, 1e-05]},
        "a plain string", 123456789012345678901234567890, -7, True, None, [], {},
    ]

    # Rounding: probability-shaped values, and the exactly-representable halfway values a half-up
    # implementation gets wrong. Both groups are kept separate so a failure says which kind broke.
    #
    # The distributions are built by integer division rather than by an actual softmax: `math.exp`
    # is libm and would make the committed file machine-dependent. `weight / total` is one
    # correctly-rounded IEEE division of two exact integers, so it is identical everywhere, and it
    # still produces the normalised, unevenly-spread values a softmax produces -- which is all the
    # rounding rule cares about.
    rounding = {"distribution": [], "halfway": [], "uniform": []}
    for _ in range(100):
        k = rng.randint(2, 20)
        weights = [rng.getrandbits(20) + 1 for _ in range(k)]
        total = sum(weights)
        rounding["distribution"].extend(weight / total for weight in weights)
    # Every 53rd step rather than every 7th: the step size does not matter, only that the values
    # are exact halves at the fourth decimal, and roughly a fifth of them discriminate half-up from
    # half-even -- which is ample at this size.
    rounding["halfway"] = [i / 20000.0 for i in range(0, 20000, 53)]
    rounding["uniform"] = [rng.random() for _ in range(300)]

    return {
        "doubles": [{"bits": bits(v), "repr": json.dumps(v)} for v in doubles],
        "structured": [{"value": v, "dumps": json.dumps(v, ensure_ascii=False)}
                       for v in structured],
        "round4": {name: [{"bits": bits(v), "r4": round(v, 4)} for v in values]
                   for name, values in rounding.items()},
        # `repr` and `%.0f` are the two formats the ROUTER's reason strings are built from, and
        # both are places Java differs by default: `String.format("%.0f", 12.5)` rounds halves UP
        # where CPython rounds to even, and an ASCII-only escape check leaves U+00A0 raw. One of
        # the reasons interpolates a slice of the caller's own text, so neither is theoretical.
        "repr_strings": [[value, repr(value)] for value in _REPR_CASES],
        "repr_digest": _repr_digest(),
        "percent0": [[fraction, "%.0f" % (100.0 * fraction)] for fraction in _PERCENT_CASES],
    }


# States and questions for the model-backed golden. Public question schema, because this one goes
# through `ONNXAgent.predict` rather than straight into `build_sequence`.
PREDICT_CASES = [
    ("billing-mixed", "We were billed twice for March and want a refund today.", None, {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "money back for a duplicate charge",
                                "replace": "a replacement unit", "info": "an explanation only"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "somewhat", "urgent", "critical"]},
        "duplicate": {"type": "noul", "instructions": "The customer was charged more than once."},
    }),
    ("cjk", "\u6211\u4eec\u4e09\u6708\u4efd\u88ab\u91cd\u590d\u6263\u8d39\u4e86\u4e24\u6b21\uff0c\u8bf7\u4eca\u5929\u9000\u8fd8\u3002", "zh", {
        "intent": {"type": "choice", "instructions": "\u5ba2\u6237\u60f3\u8981\u4ec0\u4e48\uff1f",
                   "criteria": {"refund": "\u9000\u6b3e", "replace": "\u66f4\u6362"}},
        "holds": {"type": "noul", "instructions": "\u5ba2\u6237\u88ab\u91cd\u590d\u6263\u8d39\u3002"},
    }),
    ("many-options", "The reply was polite but arrived nine days late.", None, {
        "grade": {"type": "choice", "instructions": "Grade the reply.",
                  "criteria": {("g%02d" % i): ("grade band number %d" % i) for i in range(14)}},
    }),
    ("structured-state", {"order": 1182, "items": ["widget", "case"], "charged": 2,
                          "currency": "NGN", "note": None}, None, {
        "double": {"type": "noul", "instructions": "This order was charged twice."},
        "band": {"type": "score", "instructions": "Rate the severity.",
                 "criteria": [{"d": "none"}, "minor", 3]},
    }),
    ("long-state", "evidence sentence number one about the duplicate charge. " * 120, None, {
        "intent": {"type": "choice", "instructions": "What should we do?",
                   "criteria": {"refund": "send money back", "escalate": "pass to a human"}},
    }),
    ("unicode-mixed", "Refund \u9000\u6b3e \u0627\u0633\u062a\u0631\u062f\u0627\u062f \u0935\u093e\u092a\u0938\u0940 \U0001f600 now!", "ar", {
        "intent": {"type": "choice", "instructions": "Pick one.",
                   "criteria": {"a": "first", "b": "second"}},
    }),
]


def predict_golden():
    """`ONNXAgent.predict` and `predict_batch` output on a real graph, for the end-to-end gate.

    Needs an exported graph, which this repository does not ship, so it records `skipped` when
    `LAYA_ONNX_GRAPH` is unset and the Java test skips in turn. CI exports the checkpoint and
    regenerates this, the way the .NET lane regenerates its goldens, so a Python-side change shows
    up here as drift rather than as silence.

    The numbers are model-derived and therefore platform-sensitive in their last reported digit.
    The Java test compares structure exactly and probabilities within a stated tolerance; it is the
    test's tolerance that is the gate, not byte equality of this file.
    """
    import os

    graph = os.environ.get("LAYA_ONNX_GRAPH")
    model = os.environ.get("LAYA_PREDICT_MODEL", "multilingual")
    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS",
                         DEFAULT_CHECKPOINT_ROOT)
    if not graph or not os.path.exists(graph):
        return {"skipped": "set LAYA_ONNX_GRAPH to an exported laya.onnx to record this family"}
    from laya.onnx_agent import ONNXAgent
    agent = ONNXAgent(os.path.join(rig, model), onnx_path=graph)
    single = {}
    for cid, state, lang, questions in PREDICT_CASES:
        result = agent.predict(state, questions, lang=lang)
        single[cid] = {"state": state, "lang": lang, "questions": questions,
                       "model": result["model"], "answers": result["answers"],
                       "usage": result["usage"]}
    batch_states = [c[1] for c in PREDICT_CASES] + ["short one", "another short state", ""]
    batch_questions = {
        "intent": {"type": "choice", "instructions": "What should we do?",
                   "criteria": {"refund": "send money back", "escalate": "pass to a human",
                                "ignore": "no action"}},
        "urgent": {"type": "noul", "instructions": "This needs a human today."},
    }
    batches = []
    for batch_size, sort_by_length in ((None, False), (2, False), (3, True)):
        kwargs = {"sort_by_length": sort_by_length}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        results = agent.predict_batch(batch_states, batch_questions, **kwargs)
        batches.append({"batch_size": batch_size, "sort_by_length": sort_by_length,
                        "states": batch_states, "questions": batch_questions,
                        "results": [{"model": r["model"], "answers": r["answers"],
                                     "usage": r["usage"]} for r in results]})
    return {"checkpoint": model, "graph": os.path.basename(graph),
            "single": single, "batch": batches}


def email_clean():
    """`laya.email`'s cleaner and state builder, over a corpus built from its own reasoning.

    Every case below is a rule the module's comments argue for, which is what makes this a
    contract rather than a sample: a marker that must fire, a near-miss that must NOT fire because
    it is ordinary prose, and the Unicode-sensitive sign-off rule whose tail is matched
    structurally. The near-misses matter more than the hits -- a cleaner that is too eager deletes
    the sender's actual request, which is worse than leaving boilerplate behind, and the module
    says so in four separate places.
    """
    from laya import email as email_mod

    q = "I cannot log in and need a password reset."
    cases = [
        # -- nothing to do
        ("empty", ""),
        ("plain", q),
        ("crlf", "line one\r\nline two\r\rline three"),
        ("literal-backslash-n", "first\\nsecond"),
        ("collapses-runs", "a    b\t\tc"),
        ("blank-paragraphs", "first\n\n\n\nsecond\n   \n\nthird"),

        # -- quote headers that MUST cut
        ("on-wrote", q + "\n\nOn Tue, 3 Sep 2025 at 10:04, Ana <ana@x.com> wrote:\n> older text"),
        ("em-escreveu", q + "\n\nEm ter., 3 de set. de 2025, Ana escreveu:\n> antigo"),
        ("el-escribio", q + "\n\nEl mar, 3 sept 2025 a las 10:04, Ana escribió:\n> viejo"),
        ("le-a-ecrit", q + "\n\nLe mar. 3 sept. 2025 a écrit :\n> ancien"),
        ("original-message", q + "\n\n-----Original Message-----\nolder"),
        ("mensagem-original", q + "\n\n-----Mensagem original-----\nantigo"),
        ("message-d-origine", q + "\n\n-----Message d'origine-----\nancien"),
        ("underscore-rule", q + "\n\n________________\nolder"),
        ("from-with-address", q + "\n\nFrom: Ana <ana@x.com>\nolder"),
        ("de-with-address", q + "\n\nDe: Ana <ana@x.com>\nantigo"),
        ("de-spaced-colon", q + "\n\nDe : Marie <marie@x.com>\nancien"),

        # -- near-misses that MUST NOT cut: the module names each of these as body text
        ("em-without-date", q + "\n\nEm resposta ao que você escreveu:\nmais detalhes"),
        ("le-without-date", q + "\n\nLe rapport que vous avez écrit :\nplus de détails"),
        ("from-as-prose", "From: my side the integration works, but please refund the charge."),
        ("de-as-date-range", "De: 10/09 a 15/09 estarei fora, por favor adie a cobranca."),
        ("quote-header-first-line", "On Tue, 3 Sep 2025, Ana <ana@x.com> wrote:\nstill the body"),

        # -- the bare-name Outlook header needs its neighbour to be told from prose
        ("from-name-then-sent", q + "\n\nFrom: Maria Souza\nSent: Tuesday\nolder"),
        ("de-name-then-enviado", q + "\n\nDe: Maria Souza\nEnviado: terca\nantigo"),
        ("de-name-then-envoye", q + "\n\nDe : Marie Dupont\nEnvoyé : mardi\nancien"),
        ("from-name-then-date-year", q + "\n\nFrom: Maria Souza\nDate: 3 Sep 2025\nolder"),
        ("de-name-no-neighbour", q + "\n\nDe: Maria Souza\nPara: 15/09\nstill body"),

        # -- Gmail's wrapped attribution: the tail cuts and takes its head with it
        ("wrapped-attribution", q + "\n\nEm ter., 3 de set. de 2025,\nfulano@x.com> escreveu:\n> antigo"),
        ("wrapped-attribution-fr", q + "\n\nLe mar. 3 sept. 2025,\nsupport@x.com> a écrit :\n> ancien"),
        ("attribution-tail-no-head", q + "\n\nfulano@x.com> escreveu:\n> antigo"),

        # -- quoted lines are dropped wherever they appear
        ("angle-quoted", q + "\n> quoted\n  > indented quote\nmore of my message"),

        # -- sign-offs. The search window starts at 60% of the lines, so each needs a body.
        ("signoff-regards-name", "\n".join([q] * 9 + ["Regards, Ana", "x@y.com"])),
        ("signoff-latin-ext-name", "\n".join([q] * 9 + ["Regards, Łukasz", "x@y.com"])),
        # The case INITIAL governs: Lo is a letter a name may begin with, and `str.isupper()`
        # is false for every caseless script, so UPPER here would lose the sign-off.
        ("signoff-caseless-name", "\n".join([q] * 9 + ["Regards, 山田", "x@y.com"])),
        # And the near-miss next to it: a Portuguese closing allows no trailing words at all,
        # so the same name after `Obrigado` is a sentence, not a signature.
        ("not-signoff-pt-with-name", "\n".join([q] * 9 + ["Obrigado, 山田", "x@y.com"])),
        ("signoff-titlecase-name", "\n".join([q] * 9 + ["Regards, ǅarko", "x@y.com"])),
        ("signoff-combining-name", "\n".join([q] * 9 + ["Regards, José", "x@y.com"])),
        ("signoff-dashes", "\n".join([q] * 9 + ["--", "Ana"])),
        ("signoff-warmest", "\n".join([q] * 9 + ["Warmest regards, Ana"])),
        ("signoff-and-regards", "\n".join([q] * 9 + ["Thanks and regards, Ana"])),
        ("signoff-many-thanks", "\n".join([q] * 9 + ["Many thanks, Ana"])),
        # NOT sign-offs: the next word is not a name
        ("not-signoff-sentence", "\n".join([q] * 9 + ["Thanks for the quick reply."])),
        ("not-signoff-lowercase-other-script", "\n".join([q] * 9 + ["Thanks, żaneta"])),
        ("not-signoff-symbol-name", "\n".join([q] * 9 + ["Thanks, Ⓐ"])),
        ("not-signoff-zwnj", "\n".join([q] * 9 + ["Thanks, क्‌ष"])),
        # pt/es/fr closings cut only when they stand alone
        ("signoff-atenciosamente", "\n".join([q] * 9 + ["Atenciosamente,", "Ana"])),
        ("signoff-cordialement", "\n".join([q] * 9 + ["Cordialement,", "Marie"])),
        ("signoff-saudacoes", "\n".join([q] * 9 + ["Saudações!"])),
        ("not-signoff-obrigado-mas", "\n".join([q] * 9 + ["Obrigado pelo retorno, mas preciso do estorno."])),
        ("not-signoff-merci-mais", "\n".join([q] * 9 + ["Merci pour votre aide, mais le probleme persiste."])),
        # a closing before the window is left alone
        ("signoff-too-early", "Regards, Ana\n" + "\n".join([q] * 12)),

        # -- device footers: only a line that is nothing but the footer
        ("footer-iphone", "\n".join([q] * 9 + ["Sent from my iPhone"])),
        ("footer-samsung", "\n".join([q] * 9 + ["Enviado do meu smartphone Samsung Galaxy."])),
        ("footer-envoye-iphone", "\n".join([q] * 9 + ["Envoyé depuis mon iPhone"])),
        ("footer-get-outlook", "\n".join([q] * 9 + ["Get Outlook for iOS"])),
        ("not-footer-with-request", "\n".join([q] * 9 + ["Enviado do meu celular o comprovante ontem."])),

        # -- disclaimers: a paragraph goes whole only when every sentence is boilerplate
        ("disclaimer-en", q + "\n\nThis email and its contents are confidential and intended solely "
                              "for the addressee."),
        ("disclaimer-en-wrapped", q + "\n\nThis email and its contents are\nconfidential and intended "
                                      "solely for the addressee."),
        ("disclaimer-received-in-error", q + "\n\nIf you have received this email in error please "
                                             "delete it."),
        ("disclaimer-pt", q + "\n\nEsta mensagem e seus anexos sao confidenciais e de uso exclusivo "
                              "do destinatario."),
        ("disclaimer-es", q + "\n\nEste mensaje es confidencial y para uso exclusivo del destinatario."),
        ("disclaimer-fr", q + "\n\nCe message est confidentiel et destiné uniquement au destinataire."),
        ("disclaimer-print", q + "\n\nAntes de imprimir pense no meio ambiente."),
        ("disclaimer-mixed-paragraph", "Please refund the charge. This email and its contents are "
                                       "confidential and intended solely for the addressee."),
        ("disclaimer-fused-line", "The account is locked\nThis email and its contents are confidential "
                                  "and intended solely for the addressee."),
        # NOT disclaimers: the bare word is a sender's own
        ("not-disclaimer-question", "Is this confidential?"),
        ("not-disclaimer-colon", "Confidential: I need a refund."),
        ("not-disclaimer-print", "Antes de imprimir o boleto, confira o valor."),
        ("not-disclaimer-contract", "Preciso do contrato confidencial assinado."),
        ("not-disclaimer-fr-contract", "Je voudrais le contrat confidentiel signé."),

        # -- cases that exist because a mutant survived without them. Each one is the smallest
        # input that tells the port's rule apart from the plausible wrong rule next to it.
        #
        # A mark whose canonical combining CLASS is zero. U+034F is category Mn, so it is dropped
        # before the tail is matched and the name reads as one token; the combining-class table
        # would keep it, and a kept non-word character ends the scan short of the line.
        ("signoff-class-zero-mark", "\n".join([q] * 9 + ["Regards, Jo\u034Fse"])),
        # A mark with NO base: it follows a space, so the reference keeps it, and a kept mark is
        # not a token opener -- `[^\W\d_]` excludes marks -- so this is NOT a sign-off.
        ("not-signoff-baseless-mark", "\n".join([q] * 9 + ["Regards, \u0301Ana"])),
        # 44 characters: a sign-off by every other rule, kept only by the 40-character limit.
        ("not-signoff-too-wide",
         "\n".join([q] * 9 + ["Regards, Anastasia Konstantinopolitanopoulos"])),
        # A fourth token: `{0,3}` is the whole rule, so four names are a sentence.
        ("not-signoff-four-tokens", "\n".join([q] * 9 + ["Regards, Ana Maria Souza Lima"])),
        # The wrapped-attribution tail pops the line above it ONLY when that line is the
        # `On/Em/El/Le ...` head it belongs to. Here it is ordinary prose and must survive.
        ("attribution-tail-keeps-prose", q + "\nSee the details below.\nfulano@x.com> escreveu:\n> antigo"),
        # U+0085 is a line terminator to Java and an ordinary character to Python's `.`, so a
        # header spanning one is recognised by the reference and only by a port that says so.
        ("quote-header-across-u0085", q + "\n\nOn Tue, 3 Sep 2025\u0085Ana wrote:\n> older"),
        # U+00A0 is Python `\s` and is not Java's; U+001C is Python `\s` and is not even
        # White_Space, so it is missed by Java's Unicode `\s` as well as by its default one.
        ("quote-header-nbsp-indent", q + "\n\n\u00A0On Tue, 3 Sep 2025, Ana wrote:\n> older"),
        ("quote-header-u001c-indent", q + "\n\n\u001COn Tue, 3 Sep 2025, Ana wrote:\n> older"),
        # Python's `\d` is the Nd category, not ASCII. The date these headers require can be
        # written in any decimal script, and an ASCII-only port keeps the quoted history instead.
        ("quote-header-arabic-indic-date",
         q + "\n\nEm ter., \u0663 de set. de \u0662\u0660\u0662\u0665, Ana escreveu:\n> antigo"),
        ("header-next-devanagari-year",
         q + "\n\nFrom: Maria Souza\nDate: 3 Sep \u0968\u0966\u0968\u096B\nolder"),

        # -- the signature window, `max(1, min(int(len * 0.6), len - 8))`. Every part of that
        # formula was unpinned: a review mutated the ratio to 0.55 and to 0.9, the offset to 9
        # and to 2, and the floor to 2, and all five left the suite green. These five cases are
        # the smallest inputs under 60 lines on which each mutant's window reaches a different
        # line from the real one -- found by searching, not derived, because deriving them by
        # hand gets the `min` the wrong way round.
        ("window-floor-of-one", "\n".join(["Regards, Ana" if i == 1 else q for i in range(2)])),
        ("window-offset-dominates", "\n".join(["Regards, Ana" if i == 1 else q for i in range(4)])),
        ("window-offset-excludes", "\n".join(["Regards, Ana" if i == 1 else q for i in range(10)])),
        ("window-ratio-excludes", "\n".join(["Regards, Ana" if i == 9 else q for i in range(18)])),
        ("window-ratio-includes", "\n".join(["Regards, Ana" if i == 12 else q for i in range(21)])),

        # -- the device-footer width gate, 60 code points against the sign-off's 40. Both
        # directions, because a review found `<= 60` mutable to 41 and to 200 with nothing
        # failing.
        ("footer-at-the-width-limit",
         "\n".join([q] * 9 + ["Sent from my iphone using " + "a" * 34])),
        ("footer-just-over-the-limit",
         "\n".join([q] * 9 + ["Sent from my iphone using " + "a" * 35])),

        # -- the body-side strip is Python's `str.strip()`, which takes U+00A0 and U+2007 where
        # `String.trim()` does not. The subject side was pinned and the body side was not, and
        # swapping it for `trim()` left the suite green while returning a leading space.
        ("body-strip-is-pythons", "\u00a0Refund please\u2007"),
        ("body-strip-both-ends", "\u2007\u00a0Please refund\u00a0"),

        # -- outside the BMP. The first sweep of this port had no astral character in it and
        # passed on 40,000 cases; a corpus of them found four defects at once. Each case below
        # is one of them, because the units the reference counts are CODE POINTS and the units
        # Java reaches for are chars.
        #
        # A word boundary. U+12432 is a letter-number, which is a word character to Python, so
        # there is no boundary after "mensagem" and this paragraph is NOT a disclaimer. An ASCII
        # `\\b` finds a boundary anyway and deletes the sentence whole.
        ("astral-word-char-breaks-no-boundary",
         "Esta mensagem\U00012432 e confidencial."),
        # And the same shape where the reference DOES match, so the fix cannot be "never match".
        ("astral-elsewhere-still-matches",
         "Esta mensagem e confidencial e de uso exclusivo do destinatario\U00012432."),
        # A 40-code-point closing is 41 Java chars with one astral letter in it, so a char count
        # leaves the signature in.
        ("astral-signoff-at-the-width-limit",
         "\n".join([q] * 9 + ["Regards, \U0001D400" + "A" * 30])),
        ("astral-signoff-just-over-the-limit",
         "\n".join([q] * 9 + ["Regards, \U0001D400" + "A" * 31])),
        # A combining mark outside the BMP: Java's word class holds it and Python's does not, so
        # the view substitutes it -- one code point for one code point, which is what keeps the
        # `[^.]{0,80}` window counting the same.
        ("astral-mark-in-a-disclaimer",
         "Esta mensagem e confidencial\U00011001 e de uso exclusivo do destinatario."),

        # -- everything at once
        ("full-stack", "\n".join(
            [q, "Please check the attached receipt.", ""]
            + [q] * 7
            + ["Atenciosamente,", "Ana Souza", "Enviado do meu iPhone", "",
               "Esta mensagem e confidencial e de uso exclusivo do destinatario.", "",
               "Em ter., 3 de set. de 2025, Bruno <bruno@x.com> escreveu:",
               "> mensagem antiga"])),
    ]
    budgets = [
        ("budget-default", "x" * 50 + " " + q, 3000),
        ("budget-tiny", q, 12),
        ("budget-cut-mid-word", "abcdefghij " * 10, 25),
        # the 4x pre-truncation bound: input longer than max_chars*4 is cut before matching
        ("budget-four-x-bound", ("word " * 400) + "This email is confidential and intended solely "
                                                  "for the addressee.", 100),
        ("budget-zero", q, 0),
        # A NEGATIVE budget is not a zero budget. Python slices twice, at `max_chars * 4` and at
        # `max_chars`, and a negative index drops the LAST |n| code points rather than everything
        # -- so the reference answers with a tail-trimmed body where this port used to answer
        # with nothing. Every one of the 17 differing body-and-budget combinations a review found
        # was negative, and the Java test asserted the empty string as correct.
        ("budget-negative-small", q, -5),
        ("budget-negative-one", q, -1),
        ("budget-negative-astral", "\U0001D400" * 20, -2),
        ("budget-negative-empties", "x" * 60, -12),
        # A budget that lands between a high and a low surrogate. Python slices by code point and
        # returns the whole character; a char-index cut returns half of one, which is not a
        # character the reference can produce.
        ("budget-splits-a-surrogate-pair", "ab\U0001D400cd\U0001D400ef", 3),
        ("budget-ends-on-a-surrogate-pair", "ab\U0001D400cd", 4),
        ("budget-all-astral", "\U0001D400\U0001D401\U0001D402\U0001D403", 2),
        # And the pre-truncation bound, which is max_chars * 4 code points.
        # The pre-truncation bound is max_chars * 4 CODE POINTS, and it decides whether the
        # disclaimer below is inside the window at all. Cut by char index the window is half as
        # wide, the disclaimer arrives truncated, the truncated fragment no longer matches, and
        # it survives into the model's input instead of being stripped. The final cut hides this
        # for most inputs, which is why the case is this specific: 3 astral characters and a
        # budget of 11.
        ("budget-four-x-bound-astral",
         "\U0001D400\U0001D400\U0001D400 This email is confidential and intended solely for "
         "the addressee.", 11),
    ]
    states = [
        ("state-plain", "Refund request", q, None, True, 3000, {}),
        ("state-with-sender", "Refund request", q, "ana@x.com", True, 3000, {}),
        ("state-unclean", "Refund request", q + "\n\nSent from my iPhone", None, False, 3000, {}),
        ("state-extra-fields", "Refund", q, "ana@x.com", True, 3000,
         {"priority": "high", "ticket": 42, "dropped": None}),
        ("state-blank-subject", "   ", q, None, True, 3000, {}),
        ("state-none-ish", "", "", None, True, 3000, {}),
        ("state-budget", "S", "y" * 80, None, True, 20, {}),
    ]
    return {
        "cleaned": [{"name": name, "body": body,
                     "result": email_mod.clean_email_body(body)}
                    for name, body in cases],
        "budgets": [{"name": name, "body": body, "max_chars": n,
                     "result": email_mod.clean_email_body(body, max_chars=n)}
                    for name, body, n in budgets],
        "states": [{"name": name, "subject": subject, "body": body, "sender": sender,
                    "clean": clean, "max_chars": n, "extra": extra,
                    "result": email_mod.email_state(subject, body, sender=sender, clean=clean,
                                                    max_chars=n, **extra)}
                   for name, subject, body, sender, clean, n, extra in states],
        # The re-export the module exists to keep working for callers who import it from here.
        "questions_match_presets": (
            email_mod.email_questions() == __import__("laya.presets", fromlist=["x"])
            .email_questions()),
    }


def decode_text():
    """What `Tokenizer.decode` must return, per checkpoint, recorded from the reference.

    Decoding has no Python reference to port -- it lives in the Rust `tokenizers` crate -- so this
    family is the only specification the Java side has, and the three rules it pins were each
    wrong on the first attempt:

      * a character outside the ByteLevel stand-in alphabet is KEPT as its own UTF-8 bytes, not
        dropped. The english vocabulary mixes byte-level tokens like "\\u0120world" with 23 added
        tokens whose content is literal whitespace -- id 50275 is three real spaces -- and dropping
        unmapped characters got 414 of 5,312 texts wrong.
      * the ByteLevel fallback is per TOKEN, not per character: one unmapped character and the
        whole token uses its raw bytes.
      * ByteFallback does NOT decode lossily. It tries strict UTF-8 and, on failure, emits one
        U+FFFD per BYTE of the run. Lossy decoding emits one per maximal subpart, which differs
        the moment a window boundary cuts a multi-byte character -- 629 of 201,279 window slices.

    The SLICES are the point of this family. `predictLong` cuts a tokenized state into windows at
    arbitrary offsets and decodes each one, so a boundary lands inside a multi-byte character
    routinely. Every whole-text decode agreed while 629 slices did not, so a fixture of whole
    texts would have shipped the bug.
    """
    import os

    rig = os.environ.get("LAYA_FIXTURE_CHECKPOINTS", DEFAULT_CHECKPOINT_ROOT)

    # Chosen for where a decoder breaks, not for coverage of prose.
    texts = [
        "Hello world",
        "",
        " ",
        "   ",                                  # an added token whose content is literal spaces
        "\t",
        "a\tb\nc",                              # multilingual inserts a space after the newline
        "café résumé",
        "注文をキャンセル",     # 3-byte characters, split often
        "مرحبا",
        "क्‌ष",             # ZWNJ, which is Cf and not a mark
        "\U0001f600",                           # 4-byte, the shortest ByteFallback run
        "emoji \U0001f600 here",
        "\U0001f1ec\U0001f1e7",                 # a flag: two regional indicators
        "▁literal marker",                 # the Metaspace marker as ORDINARY TEXT
        "pre▁fix",
        " nbsp figure",
        "<0x41>",                               # a byte token spelled out in ordinary text
        "a<0xFF>b",
        "",
        "",                               # private use
        "\U000f0000",                           # unassigned: four ByteFallback tokens
        "I cannot log in and need a password reset.",
        "Esta mensagem e confidencial.",
    ]

    out = {"texts": texts, "by": {}}
    for name in ("english", "multilingual"):
        root = os.path.join(rig, name)
        tokenizer_json = os.path.join(root, "tokenizer", "tokenizer.json")
        if not os.path.isfile(tokenizer_json):
            out["by"][name] = {"skipped": "checkpoint not present at %s" % root}
            continue
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(tokenizer_json)
        cases = []
        for text in texts:
            ids = tok.encode(text, add_special_tokens=False).ids
            # Every cut point at several widths: this is the predictLong shape, and the only
            # place the ByteFallback rule is observable.
            slices = []
            for size in (1, 2, 3, 5):
                for start in range(0, max(1, len(ids)), max(1, size)):
                    chunk = ids[start:start + size]
                    if chunk:
                        slices.append({"start": start, "size": size,
                                       "decoded": tok.decode(chunk)})
            with_specials = tok.encode(text, add_special_tokens=True).ids
            cases.append({
                "text": text,
                "ids": list(ids),
                "decoded": tok.decode(ids),
                "slices": slices,
                "with_specials": {
                    "ids": list(with_specials),
                    # NOT "skipped": a lone `skipped` key is this script's marker for a family it
                    # could not regenerate, and naming a data field that way once made this very
                    # family look unregenerable.
                    "skipping_specials": tok.decode(with_specials),
                    "kept": tok.decode(with_specials, skip_special_tokens=False),
                },
            })
        out["by"][name] = {
            # What this tokenizer IS, so a fixture recorded against a different checkpoint cannot
            # be mistaken for a matching one -- the same guard tokenizer_ids applies.
            "decoder": _decoder_shape(tokenizer_json),
            "cases": cases,
        }
    out["mixed_token"] = _mixed_token_scope()
    out["synthetic_decoders"] = _synthetic_decoders()
    return out


def _synthetic_decoders():
    """What the crate does with the decoder shapes no shipped checkpoint declares.

    `TokenDecoder` implements six decoders because a checkpoint may declare any of them, but the
    two shipped ones between them use four: english is `ByteLevel`, multilingual is
    `Sequence[Replace, ByteFallback, Fuse]`. So `Strip` shipped with no coverage at all -- a 44
    line decoder with a documented start/stop semantic that an adversarial mutant could gut while
    all 191 recorded cases passed -- and the `hex` helper's refusal of lowercase byte tokens had
    nothing exercising it either.

    Rather than assert what the implementation was written to do, each shape below is a real
    tokenizer the crate is asked to decode, exactly as `mixed_token` is. `byte_fallback` is set on
    the model so the port's loader guard, which refuses a vocabulary that could drop a character
    without trace, is satisfied by something other than a full byte-level alphabet.
    """
    import io
    import os
    import tempfile

    try:
        from tokenizers import Tokenizer
    except ImportError:
        return {"skipped": "the tokenizers package is not installed"}

    def spec_for(decoder, vocab):
        spec = {
            "version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [], "normalizer": None, "pre_tokenizer": None,
            "post_processor": None,
            "model": {"type": "BPE", "dropout": None, "unk_token": None,
                      "continuing_subword_prefix": None, "end_of_word_suffix": None,
                      "fuse_unk": False, "byte_fallback": True, "ignore_merges": True,
                      "vocab": vocab, "merges": []},
        }
        # `decoder=None` means the key is ABSENT, which is its own code path -- not a decoder
        # whose type is null.
        if decoder is not None:
            spec["decoder"] = decoder
        return spec

    shapes = {
        # Strip removes up to `start` leading and up to `stop` trailing copies of `content`, per
        # TOKEN. The crate serialises the Python `Strip(content, left, right)` as start/stop,
        # which is why those are the field names the port reads.
        "strip": {
            "why": "Strip, which no shipped checkpoint declares",
            "spec": spec_for(
                {"type": "Strip", "content": " ", "start": 2, "stop": 1},
                {"  hello ": 0, " x": 1, "  ": 2, "no": 3}),
            "ids": [[0], [1], [3], [0, 3]],
            # The crate PANICS on these, so there is no reference answer to record: stripping 2
            # leading spaces from the two-space token leaves an empty slice and `stop` then
            # indexes backwards through it -- "slice index starts at 2 but ends at 1", a Rust
            # panic that crosses the FFI boundary as pyo3_runtime.PanicException. The port guards
            # the same arithmetic (`to > from`) and returns "". That difference is deliberate and
            # is asserted on the Java side: a decoder that aborts the process on a token that is
            # all `content` is not behaviour worth reproducing.
            "crate_panics": [[2], [2, 2]],
        },
        # Which `<0x..>` spellings the crate reads as a BYTE. Its predicate is a length-6 token
        # starting "<0x" and ending ">", parsed by Rust's `u8::from_str_radix(.., 16)` -- which is
        # case-INSENSITIVE and accepts a leading "+". This port had it backwards, rejecting
        # lowercase and "+" with a comment asserting the crate wrote these uppercase only.
        #
        # The "+" case needs its own token: with only the upper/lower pair here, a mutant that
        # re-broke "+" passed all 195 cases. "<0x 5>" and "<0XFF>" record the other side of the
        # boundary -- a space is not a digit, and the "<0x" prefix IS case-sensitive -- so the
        # rule is pinned from both directions rather than only where it says yes.
        "bytefallback_hex_case": {
            "why": "which byte-token spellings the crate reads as a byte",
            "spec": spec_for(
                {"type": "ByteFallback"},
                {"<0xFF>": 0, "<0xff>": 1, "<0x41>": 2, "<0x61>": 3, "a": 4,
                 "<0x+5>": 5, "<0x 5>": 6, "<0XFF>": 7, "<0xfF>": 8,
                 "<0xF3>": 9, "<0xB0>": 10, "<0x80>": 11}),
            # The last four are a TRUNCATED multi-byte run, which is the only shape that can tell
            # lossy decoding from one replacement per byte: F3 B0 is a single maximal subpart
            # (lossy gives ONE U+FFFD) and two unusable bytes (the crate gives TWO). Every other
            # invalid run in this vocabulary is a single byte, where both rules agree -- so a
            # mutant that made this decoder lossy passed all 195 cases.
            "ids": [[2], [3], [1], [0], [2, 3], [0, 0], [1, 4],
                    [5], [6], [7], [8], [5, 4], [6, 4], [7, 4],
                    [9, 10], [9, 10, 11], [9, 10, 11, 11], [4, 9, 10, 4]],
        },
        # No `decoder` key at all. The crate's fallback is `tokens.join(" ")` -- a SPACE -- and
        # this port joined with nothing, which was wrong for every multi-token decode. Neither
        # shipped checkpoint can reach it, because both declare a decoder.
        "no_decoder": {
            "why": "no decoder section: the separator the crate falls back to",
            "spec": spec_for(None, {"ab": 0, "cd": 1, "ef": 2, " ": 3}),
            "ids": [[0], [0, 1], [0, 1, 2], [0, 3, 1], [3, 3]],
        },
        # A Sequence inside a Sequence. The crate chains `decode_chain` and returns a LIST, so the
        # outer sequence's next step sees the inner one's pieces; re-joining them collapsed the
        # boundaries. Strip is the step that can tell, because it acts per piece.
        "nested_sequence": {
            "why": "a Sequence nested in a Sequence keeps its pieces",
            "spec": spec_for(
                {"type": "Sequence", "decoders": [
                    {"type": "Sequence", "decoders": [
                        {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
                        {"type": "ByteFallback"}]},
                    {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
                {"\u2581Hello": 0, "\u2581world": 1, "\u2581": 2, "\u2581a": 3}),
            "ids": [[0, 1], [2, 2, 0], [2, 3, 2], [0], [3]],
        },
        # Whether a failed ByteFallback run is one piece of N replacement characters or N pieces
        # of one. Only a per-piece step after it can tell, so Replace looks for the PAIR: it
        # matches if the two characters ended up in the same piece.
        "bytefallback_piece_boundaries": {
            "why": "a failed byte run is one piece per byte, not one piece of N",
            "spec": spec_for(
                {"type": "Sequence", "decoders": [
                    {"type": "ByteFallback"},
                    {"type": "Replace", "pattern": {"String": "\ufffd\ufffd"},
                     "content": "X"}]},
                {"<0xF3>": 0, "<0xB0>": 1, "<0x80>": 2, "hi": 3}),
            "ids": [[0, 1], [0, 1, 2], [3, 0, 1, 3], [0, 1, 2, 2], [3]],
        },
    }

    for name, shape in shapes.items():
        path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
        io.open(path, "w", encoding="utf-8").write(json.dumps(shape["spec"]))
        tok = Tokenizer.from_file(path)
        shape["cases"] = [{"ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)}
                          for ids in shape.pop("ids")]
        # A shape whose cases all decoded to the same thing would pin nothing, and these are
        # hand-chosen rather than swept, so the check is worth its one line.
        distinct = {case["decoded"] for case in shape["cases"]}
        if len(distinct) < 2:
            raise AssertionError(
                "the %s shape's cases all decode to %r, so they cannot tell any rule apart"
                % (name, distinct.pop()))
    return shapes


def _mixed_token_scope():
    """Whether the ByteLevel fallback is scoped to a TOKEN or to a CHARACTER, from the crate.

    Neither real checkpoint can answer this. Their byte-level tokens are entirely stand-in
    alphabet and their added tokens are entirely literal whitespace, so both rules produce the
    same bytes for every token in both vocabularies -- a mutant that swaps per-token for
    per-character passed all 190 recorded cases. The rules differ only for a token that MIXES
    the two, so this builds a vocabulary that has one and records what the crate does with it.

    The separating token is "Ġ ": U+0120 is the stand-in for byte 0x20, and the literal
    space after it is not in the alphabet at all.

      per TOKEN     -> every character contributes its own UTF-8 bytes: C4 A0 20 -> "Ġ "
      per CHARACTER -> the mapped one contributes its byte:             20 20     -> "  "

    The spec travels in the fixture rather than being written twice, so the Java test loads the
    same bytes the crate was asked about.
    """
    import io
    import os
    import tempfile

    try:
        from tokenizers import Tokenizer
    except ImportError:
        return {"skipped": "the tokenizers package is not installed"}

    mixed = "Ġ "

    # Every single-byte alphabet symbol, from the crate's own table rather than a transcription.
    # The port's loader refuses a vocabulary that cannot represent some text, and a vocabulary of
    # three tokens cannot -- so the three interesting tokens join a complete byte-level one. They
    # do not interact: decoding is driven by the ids this records, not by what else is present.
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre

    vocab = {}
    for symbol in sorted(ByteLevelPre.alphabet()):
        vocab[symbol] = len(vocab)
    interesting = {}
    for token in (mixed, "Ġhi", "   "):
        if token not in vocab:
            vocab[token] = len(vocab)
        interesting[token] = vocab[token]

    spec = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [], "normalizer": None,
        # The pre-tokenizer matches the decoder. It plays no part in decoding, but the port's
        # loader refuses a vocabulary where a character could be dropped without trace -- no
        # unk_token, no byte_fallback, no byte-level pre-tokenizer -- and that guard is right.
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                          "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                    "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": []},
    }

    path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
    io.open(path, "w", encoding="utf-8").write(json.dumps(spec))
    tok = Tokenizer.from_file(path)

    cases = [
        {"why": "the mixed token alone: the whole point",
         "ids": [interesting[mixed]]},
        {"why": "entirely in the alphabet", "ids": [interesting["Ġhi"]]},
        {"why": "entirely literal", "ids": [interesting["   "]]},
        {"why": "a mixed token beside a plain one",
         "ids": [interesting[mixed], interesting["Ġhi"]]},
    ]
    for case in cases:
        case["decoded"] = tok.decode(case["ids"], skip_special_tokens=False)

    scope = "token" if cases[0]["decoded"] == mixed else (
        "character" if cases[0]["decoded"] == "  " else "neither")
    if scope != "token":
        raise AssertionError(
            "the crate's ByteLevel fallback is scoped per %s, and the port assumes per token; "
            "decoding the mixed token gave %r" % (scope, cases[0]["decoded"]))
    return {"spec": spec, "separating_token": mixed, "scope": scope, "cases": cases}


def _decoder_shape(tokenizer_json):
    """The decoder section's type chain, flattened, so the fixture says what it was recorded on."""
    import io
    import json

    document = json.load(io.open(tokenizer_json, encoding="utf-8"))
    node = document.get("decoder")
    if node is None:
        return None

    def shape(n):
        kind = n.get("type")
        if kind == "Sequence":
            return {"type": "Sequence",
                    "decoders": [shape(step) for step in n.get("decoders", [])]}
        return {"type": kind}

    return shape(node)


FAMILIES = {
    "lang_tables.json": lang_tables,
    "lang_detect.json": lang_detect,
    "presets.json": presets,
    "router.json": router,
    "shortlist.json": shortlist,
    "tokenizer_ids.json": tokenizer_ids,
    "decode_text.json": decode_text,
    "sequences.json": sequences,
    "decode.json": decode_answers,
    "python_json.json": python_json,
    "email.json": email_clean,
    "predict.json": predict_golden,
}


def render(payload):
    """One canonical serialisation, so `--check` compares content and never formatting.

    `sort_keys` is deliberately OFF. Key order is not formatting here, it is data: a `choice`
    question's options are rendered in the criteria map's INSERTION order, so sorting the keys on
    the way into the fixture asks a different question than the one that was measured, and a state
    dict re-serialised in sorted order tokenizes differently. Sorting them made the committed
    fixture disagree with the library it was generated from, and a port that matched the fixture
    was wrong in exactly the way the fixture existed to prevent. Python dicts preserve insertion
    order and this generator builds them deterministically, so the output is still canonical.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def unverifiable(payload):
    """True when regeneration produced nothing but skip markers.

    A family that needs a checkpoint or an exported graph records `{"skipped": ...}` when it cannot
    reach one. That must never be mistaken for data:

    * in write mode it would REPLACE a good committed fixture with a skip marker, so running this
      script on a machine without the checkpoints would silently delete the parity expectations;
    * in `--check` mode it would be reported as drift against the committed file, so the gate would
      fail for the one reason that is not a problem.

    Both are refused below. `--strict` turns "could not verify" into an error, which is what a CI
    lane that DID download the checkpoints should pass -- there, a skip means the download failed.
    """
    if not isinstance(payload, dict):
        return False
    if "skipped" in payload:
        return True
    # ANY section, not all of them. `tokenizer_ids.json` carries a `corpus_text` section that is
    # always present, so requiring every section to be skipped let a payload through whose two
    # per-checkpoint halves were both skip markers -- and the write replaced a 107 KB fixture with
    # a 9 KB one. Partial regeneration is still unsafe to write.
    #
    # At ANY depth, not just one level down. This looked only at the payload's immediate values,
    # and `decode_text.json` groups its per-checkpoint sections under a `by` key -- so the skip
    # markers sat two levels down, the payload was judged regenerable, and both halves of this
    # docstring came true at once: on a machine without the checkpoints, `--check` reported the
    # committed fixture as stale, and a plain run replaced 75 KB of parity expectations with 7 KB
    # of skip markers. Measured, not reasoned about. A family should not have to keep its sections
    # at the top level to be seen, so the search descends instead.
    return _mentions_skip(payload)


def _mentions_skip(node):
    """True when a skip MARKER appears anywhere in the payload.

    A marker is a dict of exactly one `skipped` key, which is what every builder above writes.
    Merely containing a `skipped` key is not enough, and that is not a hypothetical distinction:
    `decode_text.json` records, per text, what the reference returns with special tokens skipped
    and without, and the first version of that family named the first of those two fields
    `skipped`. The recursion below read it as a marker, so a fully recorded family reported itself
    unregenerable -- with the checkpoints present it was left alone instead of checked, and
    `--strict` in the parity lane would have failed on a family that had just been recorded
    correctly.

    That field is now `skipping_specials`, so the collision no longer exists in this script and
    nothing here depends on the exact-key rule to avoid it. The rule stays because it is the
    correct test for a marker, and because the next family to record a `skipped` field should not
    have to discover this interaction the way that one did.
    """
    if isinstance(node, dict):
        if set(node) == {"skipped"}:
            return True
        return any(_mentions_skip(value) for value in node.values())
    if isinstance(node, list):
        return any(_mentions_skip(item) for item in node)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if a committed fixture would change")
    parser.add_argument("--only", action="append", metavar="FILE",
                        help="restrict to one fixture file (repeatable)")
    parser.add_argument("--strict", action="store_true",
                        help="treat a family that cannot be regenerated as an error, for a lane "
                             "that is supposed to have the checkpoints")
    args = parser.parse_args(argv)

    # A --only naming no family used to exit 0 and print "fixtures match laya", so one typo --
    # a hyphen for an underscore, a renamed file, a family dropped from FAMILIES -- turned the
    # whole gate into a no-op that actively asserted the opposite.
    if args.only:
        unknown = sorted(set(args.only) - set(FAMILIES))
        if unknown:
            print("gen_fixtures: no such fixture family: %s\n  known families: %s"
                  % (", ".join(unknown), ", ".join(sorted(FAMILIES))), file=sys.stderr)
            return 2

    if not args.check:
        os.makedirs(FIXTURES, exist_ok=True)
    stale, written, unverified = [], [], []
    examined = 0
    for filename, build in sorted(FAMILIES.items()):
        if args.only and filename not in args.only:
            continue
        examined += 1
        path = os.path.join(FIXTURES, filename)
        payload = build()
        fresh = render(payload)
        if unverifiable(payload) and os.path.exists(path):
            unverified.append(filename)
            continue
        if args.check:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    current = handle.read()
            except OSError:
                stale.append("%s is missing" % filename)
                continue
            if current != fresh:
                stale.append("%s is stale" % filename)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(fresh)
            written.append("%s (%d bytes)" % (filename, len(fresh)))

    for filename in unverified:
        print("gen_fixtures: %s left alone: it needs a checkpoint or an exported graph this run "
              "could not reach" % filename, file=sys.stderr)
    if unverified and args.strict:
        print("gen_fixtures: --strict was asked for, so a family this run could not regenerate is "
              "an error: the checkpoints were expected to be present", file=sys.stderr)
        return 1

    if args.check:
        for line in stale:
            print("gen_fixtures: " + line, file=sys.stderr)
        if stale:
            print("gen_fixtures: run laya-java/scripts/gen_fixtures.py and commit the result",
                  file=sys.stderr)
            return 1
        if examined == 0:
            # Belt and braces with the --only validation above: reporting a match after
            # comparing nothing is the one outcome this gate must never produce.
            print("gen_fixtures: no fixture family was examined, so nothing was checked",
                  file=sys.stderr)
            return 2
        print("gen_fixtures: %d family/families match laya%s"
              % (examined,
                 " (%d unverified)" % len(unverified) if unverified else ""))
        return 0
    for line in written:
        print("wrote " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
