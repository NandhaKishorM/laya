#!/usr/bin/env python3
"""Generate `laya.lang`'s word lists and script ranges as Java source.

Compiled in rather than read from `fixtures/lang_tables.json`, because a runtime library must not
depend on a test fixture being on disk. The fixture keeps its job: `LanguageTablesTest` asserts the
compiled copy equals it, so a table that drifts from Python fails a test rather than changing a
routing decision quietly.

TWO ORDERINGS ARE LOAD-BEARING and are preserved deliberately:

  * `_SCRIPT_RANGES` is scanned in order and the FIRST matching range wins, so a reordering
    changes which script a code point is counted under.
  * `_STOP` is iterated to build the per-language scores, and `max` returns the FIRST of equal
    values, so its insertion order decides ties between languages with the same score.

    python laya-java/scripts/gen_language_tables.py            # write
    python laya-java/scripts/gen_language_tables.py --check     # fail if the committed copy differs
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TARGET = os.path.join(
    HERE, "..", "laya-java", "src", "main", "java", "com", "convaiinnovations", "laya", "lang",
    "LanguageTables.java")


# The control characters Java spells with a letter escape; everything else below U+0020 gets an
# octal escape, because a backslash-u escape is translated before the lexer sees it.
_JAVA_LETTER_ESCAPES = {
    "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r",
}


def quote(text):
    """A Java string literal, with every non-ASCII character escaped so the file stays ASCII."""
    out = ['"']
    for ch in text:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch in _JAVA_LETTER_ESCAPES:
            out.append(_JAVA_LETTER_ESCAPES[ch])
        elif ch < " ":
            # Below U+0020 a backslash-u escape is NOT safe: javac translates those before the
            # file is lexed, so a newline or carriage return would terminate the string literal
            # and the generated file would fail to compile with "unclosed string literal" rather
            # than anything that names the cause. An octal escape is handled by the lexer, not
            # the pre-pass, so it survives. Latent today -- no stopword or script name holds a
            # control character -- and a build break the first time one does.
            out.append("\\%03o" % ord(ch))
        elif " " <= ch <= "~":
            out.append(ch)
        else:
            encoded = ch.encode("utf-16-be")
            for i in range(0, len(encoded), 2):
                out.append("\\u%02X%02X" % (encoded[i], encoded[i + 1]))
    out.append('"')
    return "".join(out)


# A `new int[] {...}` or `Set.of(...)` literal compiles to instructions inside whatever method
# holds it, and a JVM method body may not exceed 65,535 bytes. Everything this generator emits
# lands in one `<clinit>`, measured at about 7,800 bytes today -- so there is room, but no signal
# if that changes. The sibling generator guards its own tables for exactly this reason; the
# failure mode without a guard is "code too large" from javac, which names no cause.
MAX_LITERALS = 20000


def _check_budget(name, count):
    if count > MAX_LITERALS:
        raise SystemExit(
            "gen_language_tables: %s would emit %d literals into the class initialiser, over the "
            "%d this generator will allow. Split it into its own factory method before "
            "regenerating." % (name, count, MAX_LITERALS))


def wrap_items(items, indent="            ", width=104):
    """Comma-separated items wrapped to a readable width, as Java array/argument contents."""
    lines, current = [], indent
    for item in items:
        piece = item + ", "
        if len(current) + len(piece) > width and current.strip():
            lines.append(current.rstrip())
            current = indent
        current += piece
    if current.strip():
        lines.append(current.rstrip().rstrip(","))
    return "\n" + "\n".join(lines) + "\n    "


def words_block(name, words, doc, sort=True):
    items = sorted(words) if sort else list(words)
    _check_budget(name, len(items))
    lines, current = [], "            "
    for word in items:
        piece = quote(word) + ", "
        if len(current) + len(piece) > 104:
            lines.append(current.rstrip())
            current = "            "
        current += piece
    if current.strip():
        lines.append(current.rstrip().rstrip(","))
    return ('    /** %s */\n    public static final Set<String> %s = Set.of(\n%s);\n'
            % (doc, name, "\n".join(lines)))


def render():
    sys.path.insert(0, REPO)
    from laya import lang

    script_rows = []
    for name, ranges in lang._SCRIPT_RANGES:
        pairs = ", ".join("%d, %d" % (lo, hi) for lo, hi in ranges)
        script_rows.append('            new Script(%s, new int[] {%s})' % (quote(name), pairs))

    # The flat lookup emitted below replaces the ordered scan with a binary search. The two are
    # equivalent only while no two ranges overlap: where they do, the first-match rule decides and
    # a table sorted by code point cannot see it. Checked here rather than trusted, so an edit to
    # `_SCRIPT_RANGES` that introduces an overlap fails this generator instead of quietly changing
    # which script a code point is counted under.
    flat = sorted((lo, hi, name) for name, ranges in lang._SCRIPT_RANGES for lo, hi in ranges)
    for (a_lo, a_hi, a_name), (b_lo, b_hi, b_name) in zip(flat, flat[1:]):
        if a_hi >= b_lo:
            raise SystemExit(
                "gen_language_tables: script ranges %s [%04X-%04X] and %s [%04X-%04X] overlap, so "
                "the flat lookup would lose the first-match rule. Drop SCRIPT_LOOKUP back to an "
                "ordered scan, or remove the overlap."
                % (a_name, a_lo, a_hi, b_name, b_lo, b_hi))
    lookup_pairs = wrap_items(["%d, %d" % (lo, hi) for lo, hi, _ in flat])
    distinct = [name for name, _ in lang._SCRIPT_RANGES]
    index_of = {name: n for n, name in enumerate(distinct)}
    lookup_index = wrap_items([str(index_of[name]) for _, _, name in flat])
    script_names = wrap_items([quote(name) for name in distinct])
    script_max = max(hi for _, hi, _ in flat)

    stop_rows = []
    for code, words in lang._STOP.items():
        items = ", ".join(quote(w) for w in sorted(words))
        stop_rows.append('        STOP.put(%s, Set.of(%s));' % (quote(code), items))

    header = '''// GENERATED by laya-java/scripts/gen_language_tables.py -- do not edit.
//
// Regenerate with:
//     python laya-java/scripts/gen_language_tables.py
// and `--check` fails when the committed copy no longer matches laya/lang.py.
package com.convaiinnovations.laya.lang;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The word lists, script ranges and thresholds {@code laya/lang.py} detects with.
 *
 * <p>Compiled in rather than read from {@code fixtures/lang_tables.json}: a runtime library must
 * not need a test fixture on disk. The fixture keeps its job -- a test asserts this copy equals it,
 * so a table that drifts from Python fails a test rather than changing a routing decision quietly.
 *
 * <p><b>Two orderings are load-bearing.</b> {@link #SCRIPT_RANGES} is scanned in order and the
 * first matching range wins, so a reordering changes which script a code point counts under. And
 * {@link #STOP_WORDS} is iterated to score languages, where the winner is the first of equal
 * scores, so its order decides ties.
 */
public final class LanguageTables {

    private LanguageTables() {
    }

    /**
     * One named non-Latin script and the code-point ranges that belong to it.
     *
     * <p>The array is copied in and out, so a table a caller holds cannot be edited under
     * detection. Nothing on the hot path reads it -- {@link #lookupScript} does -- so the copy
     * costs nothing that matters.
     */
    public record Script(String name, int[] ranges) {
        public Script {
            ranges = ranges.clone();
        }

        @Override
        public int[] ranges() {
            return ranges.clone();
        }
    }

'''
    body = (
        '    /**\n'
        '     * Named scripts, in scan order. The first range that contains a code point wins, so\n'
        '     * this is a sequence and not a set.\n'
        '     */\n'
        '    public static final List<Script> SCRIPT_RANGES = List.of(\n'
        + ",\n".join(script_rows) + ");\n\n"
        + ('    /**\n'
           '     * {@link #SCRIPT_RANGES} flattened and sorted by code point, as inclusive\n'
           '     * [lo, hi] pairs, with {@link #SCRIPT_LOOKUP_NAMES} naming each one.\n'
           '     *\n'
           '     * <p>Detection resolves a script with a binary search over this instead of\n'
           '     * walking the %d scripts in order the way the reference does. The two agree\n'
           '     * exactly because no two ranges overlap -- the generator refuses to emit this\n'
           '     * table otherwise, since the first-match rule would then be unrepresentable in a\n'
           '     * sorted table -- and a test sweeps every code point to prove it.\n'
           '     */\n'
           '    private static final int[] SCRIPT_LOOKUP = {%s};\n\n'
           '    /**\n'
           '     * The script each pair of {@link #SCRIPT_LOOKUP} belongs to, as an index into\n'
           '     * {@link #SCRIPT_NAMES}.\n'
           '     *\n'
           '     * <p>An index rather than the name, so that counting letters by script needs no\n'
           '     * boxing, no hashing and no string comparison per character: the tally is an\n'
           '     * {@code int[]} and the names are attached once at the end. The reference does a\n'
           '     * dictionary update per non-Latin letter instead.\n'
           '     */\n'
           '    private static final int[] SCRIPT_LOOKUP_INDEX = {%s};\n\n'
           '    /** The named scripts, in the order the reference declares them. */\n'
           '    private static final String[] SCRIPT_NAMES = {%s};\n\n'
           '    /** How many named scripts there are. */\n'
           '    public static final int SCRIPT_COUNT = SCRIPT_NAMES.length;\n\n'
           '    /**\n'
           '     * The largest code point any named script claims.\n'
           '     *\n'
           '     * <p>An early exit worth having: every letter above it -- the CJK extension\n'
           '     * planes, the kana supplement, and every astral script -- is counted under\n'
           '     * "other", and that is the text most likely to be long.\n'
           '     */\n'
           '    public static final int SCRIPT_MAX_CODE_POINT = 0x%04X;\n\n'
           '    /**\n'
           '     * The named script claiming {@code codePoint}, or null when none does.\n'
           '     *\n'
           '     * <p>Equivalent to scanning {@link #SCRIPT_RANGES} in order and taking the first\n'
           '     * match, in O(log n) comparisons rather than O(n) range tests.\n'
           '     */\n'
           '    public static String lookupScript(int codePoint) {\n'
           '        int index = lookupScriptIndex(codePoint);\n'
           '        return index < 0 ? null : SCRIPT_NAMES[index];\n'
           '    }\n\n'
           '    /**\n'
           '     * The index into {@link #SCRIPT_NAMES} of the script claiming {@code codePoint},\n'
           '     * or -1 when none does.\n'
           '     */\n'
           '    public static int lookupScriptIndex(int codePoint) {\n'
           '        if (codePoint > SCRIPT_MAX_CODE_POINT) {\n'
           '            return -1;\n'
           '        }\n'
           '        int low = 0;\n'
           '        int high = SCRIPT_LOOKUP_INDEX.length - 1;\n'
           '        while (low <= high) {\n'
           '            int mid = (low + high) >>> 1;\n'
           '            if (codePoint < SCRIPT_LOOKUP[mid * 2]) {\n'
           '                high = mid - 1;\n'
           '            } else if (codePoint > SCRIPT_LOOKUP[mid * 2 + 1]) {\n'
           '                low = mid + 1;\n'
           '            } else {\n'
           '                return SCRIPT_LOOKUP_INDEX[mid];\n'
           '            }\n'
           '        }\n'
           '        return -1;\n'
           '    }\n\n'
           '    /** The name of the script at {@code index}. */\n'
           '    public static String scriptName(int index) {\n'
           '        return SCRIPT_NAMES[index];\n'
           '    }\n\n')
        % (len(lang._SCRIPT_RANGES), lookup_pairs, lookup_index, script_names, script_max)
        + '    /**\n'
        '     * Function words per language, in the order Python declares them.\n'
        '     *\n'
        '     * <p>A {@link LinkedHashMap}, because the language that wins a tied score is the\n'
        '     * first one iterated.\n'
        '     */\n'
        '    public static final Map<String, Set<String>> STOP_WORDS;\n\n'
        '    static {\n'
        '        Map<String, Set<String>> STOP = new LinkedHashMap<>();\n'
        + "\n".join(stop_rows) + "\n"
        '        STOP_WORDS = Collections.unmodifiableMap(STOP);\n'
        '    }\n\n'
        + words_block("SHORT_SWEDISH_WORDS", lang._SHORT_SWEDISH_WORDS,
                      "Distinctive Swedish words that name the language from a short input.")
        + "\n"
        + words_block("NON_EN_DIACRITICS", lang._NON_EN_DIACRITICS,
                      "Letters whose presence is evidence the text is not English.")
        + "\n"
        + words_block("SHARED_WORDS", lang._SHARED_WORDS,
                      "Function words several lists hold, which therefore name no language.")
        + "\n"
        + words_block("NORDIC_OVERLAP_WORDS", lang._NORDIC_OVERLAP_WORDS,
                      "Markers shared by Swedish and Danish: evidence of non-English, not of which.")
        + "\n"
        + words_block("EN_ONLY_WORDS", lang._EN_ONLY_WORDS,
                      "English function words no other list holds.")
        + "\n"
        + words_block("EN_COLLISION_WORDS", lang._EN_COLLISION_WORDS,
                      "Words that collide with English, counted once however often they repeat.")
        + '''
    /**
     * A diacritic rate at or above this is evidence the text is not English, even with no stopword
     * match. Measured over every character, so one accented loanword in a short sentence clears it
     * -- which is what the English rescue exists to answer.
     */
    public static final double NON_EN_DIACRITIC_RATE = %r;

    /** Above this rate, English function words no longer rescue the text. */
    public static final double ENGLISH_RESCUE_DIACRITIC_RATE = %r;

    /** Non-Latin share at which non-Latin text is not for the English checkpoint. */
    public static final double NON_LATIN_FRACTION = %r;

    /** The lower share that counts when there are enough letters to dilute it. */
    public static final double NON_LATIN_MIN_FRACTION = %r;

    /** How many non-Latin letters make the lower share count. */
    public static final int NON_LATIN_MIN_LETTERS = %d;
}
''' % (lang.NON_EN_DIACRITIC_RATE, lang.ENGLISH_RESCUE_DIACRITIC_RATE,
       lang.NON_LATIN_FRACTION, lang.NON_LATIN_MIN_FRACTION, lang.NON_LATIN_MIN_LETTERS)
    )
    return header + body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the committed copy would change")
    args = parser.parse_args(argv)
    fresh = render()
    target = os.path.normpath(TARGET)
    if args.check:
        try:
            with open(target, encoding="utf-8") as handle:
                current = handle.read()
        except OSError:
            print("gen_language_tables: %s is missing" % target, file=sys.stderr)
            return 1
        if current != fresh:
            print("gen_language_tables: %s is stale; run "
                  "laya-java/scripts/gen_language_tables.py and commit the result" % target,
                  file=sys.stderr)
            return 1
        print("gen_language_tables: LanguageTables.java matches laya/lang.py")
        return 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(fresh)
    print("wrote %s (%d bytes)" % (os.path.relpath(target, REPO), len(fresh)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
