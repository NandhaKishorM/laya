#!/usr/bin/env python3
r"""Record the GPT-2 pre-tokenizer's letter and number classes FROM THE REFERENCE, as Java source.

WHY THIS EXISTS. The GPT-2 pre-tokenizer pattern is

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

and `\\p{L}`/`\\p{N}` are Unicode-version dependent. Three versions are in play and no two agree:

  * java.util.regex follows the JDK's. Corretto 17 ships Unicode 13.0.
  * CPython 3.12 ships 15.0, so the tables this repo already records do not match either.
  * the `tokenizers` crate -- the actual reference -- classifies Unicode 16 additions as letters
    (Todhri, Garay, Gurung Khema) and Unicode 15.1 CJK Ext-I too.

Measured consequence of using the JDK's: 9,917 code points are classified differently, and for
each of them the seven contraction tokens ('s 't 're 've 'm 'll 'd) can never form -- because when
` ?\\p{L}+` fails, ` ?[^\\s\\p{L}\\p{N}]+` swallows the apostrophe into the same piece. "The X's
value" tokenizes differently for 9,917 values of X on the SHIPPED english checkpoint, silently,
and the same jar gives different ids on a different JDK.

So the classes are recorded here from the reference and compiled in, which is what every other
table in this port does. The probe is behavioural rather than a property lookup, because the
property is not exposed. A code point is a LETTER if "a<cp>" stays ONE piece and a NUMBER if
"1<cp>" stays one piece: the only way the first match can reach past the "a" is ` ?\p{L}+`
consuming the code point, and past the "1" is ` ?\p{N}+`, so one piece means membership and two
means non-membership, with no third reading.

An earlier probe asked instead whether "<cp>'s" leaves "'s" as its own piece. That is NOT a test
of membership: it is true for every code point that fails to absorb the apostrophe, which includes
every whitespace code point and every digit. It marked TAB, LF, CR, NBSP, U+2007 and all 1,911
digits as letters, so a letter run swallowed the inner whitespace of "a  b\t\tc" and the scanner
returned 11 ids where the reference returns 14.

    python laya-java/scripts/gen_pretokenizer_tables.py --tokenizer <path/to/tokenizer.json>
    python laya-java/scripts/gen_pretokenizer_tables.py --check --tokenizer <path>
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(
    HERE, "..", "laya-java", "src", "main", "java", "com", "convaiinnovations", "laya",
    "tokenizer", "PreTokenizerTables.java")
MAX = 0x110000
SURROGATES = range(0xD800, 0xE000)
MAX_ELEMENTS_PER_TABLE = 4000


def classes_from_reference(tokenizer_path):
    """(letters, numbers) as sorted inclusive ranges, probed from the reference."""
    from tokenizers import Tokenizer

    pre = Tokenizer.from_file(tokenizer_path).pre_tokenizer
    letters, numbers = [], []
    for name, probe, judge in (
        ("letter", lambda ch: "a" + ch, lambda pieces: len(pieces) == 1),
        ("number", lambda ch: "1" + ch, lambda pieces: len(pieces) == 1),
    ):
        runs, start = [], None
        for cp in range(MAX):
            hit = False
            if cp not in SURROGATES:
                ch = chr(cp)
                # Both probes are anchored by a character of the class being tested, so a
                # second piece means the code point is outside it. Nothing else can join the
                # anchor, which is what makes the judge a membership test rather than a
                # statement about the apostrophe.
                try:
                    hit = judge(pre.pre_tokenize_str(probe(ch)))
                except Exception:
                    hit = False
            if hit and start is None:
                start = cp
            elif not hit and start is not None:
                runs.append((start, cp - 1))
                start = None
        if start is not None:
            runs.append((start, MAX - 1))
        (letters if name == "letter" else numbers).extend(runs)
    return letters, numbers


def table(name, rows, doc):
    values = [value for row in rows for value in row]
    if len(values) > MAX_ELEMENTS_PER_TABLE:
        raise SystemExit(
            "gen_pretokenizer_tables: %s has %d elements, over the %d a single factory method "
            "can hold" % (name, len(values), MAX_ELEMENTS_PER_TABLE))
    lines, current = [], "                "
    for value in values:
        piece = str(value) + ", "
        if len(current) + len(piece) > 100:
            lines.append(current.rstrip())
            current = "                "
        current += piece
    if current.strip():
        lines.append(current.rstrip().rstrip(","))
    factory = name.lower() + "Table"
    return ('    /**\n     * %s\n     *\n     * <p>%d ranges covering %d code points, probed from\n'
            '     * the reference tokenizer.\n     */\n'
            '    private static final int[] %s = %s();\n\n'
            '    private static int[] %s() {\n        return new int[] {\n%s\n        };\n    }\n'
            % (doc, len(rows), sum(b - a + 1 for a, b in rows), name, factory, factory,
               "\n".join(lines)))


def render(tokenizer_path):
    letters, numbers = classes_from_reference(tokenizer_path)
    import tokenizers

    header = '''// GENERATED by laya-java/scripts/gen_pretokenizer_tables.py -- do not edit.
//
// Regenerate with:
//     python laya-java/scripts/gen_pretokenizer_tables.py --tokenizer <path/to/tokenizer.json>
// and `--check` fails when the committed copy no longer matches the reference.
package com.convaiinnovations.laya.tokenizer;

/**
 * The letter and number classes the GPT-2 pre-tokenizer pattern splits on, recorded from the
 * reference tokenizer (tokenizers %s).
 *
 * <p>Compiled in rather than taken from {@code java.util.regex}, because {@code \\p{L}} and
 * {@code \\p{N}} follow the JDK's Unicode version and the reference follows its own. Corretto 17
 * ships Unicode 13.0; the reference classifies Unicode 15.1 and 16.0 additions as letters. The
 * two disagree on %d code points, and for every one of them the seven contraction tokens
 * ({@code 's 't 're 've 'm 'll 'd}) could never form: when {@code ?\\p{L}+} fails, the
 * {@code ?[^\\s\\p{L}\\p{N}]+} branch swallows the apostrophe into the same piece. "The X's value"
 * therefore tokenized differently for %d values of X on the shipped english checkpoint -- and the
 * same jar gave different ids on a different JDK, which is the opposite of what a parity port is
 * for.
 */
public final class PreTokenizerTables {

    /** The reference these classes were probed from. */
    public static final String REFERENCE = "tokenizers %s";

    private PreTokenizerTables() {
    }

''' % (tokenizers.__version__, 9917, 9917, tokenizers.__version__)

    body = (
        table("LETTER", letters,
              "The pattern's {@code \\\\p{L}}: what the reference treats as a letter.")
        + "\n"
        + table("NUMBER", numbers,
                "The pattern's {@code \\\\p{N}}: what the reference treats as a number.")
    )

    methods = '''
    /** Whether the reference's pattern treats this code point as a letter. */
    public static boolean isLetter(int codePoint) {
        return contains(LETTER, codePoint);
    }

    /** Whether the reference's pattern treats this code point as a number. */
    public static boolean isNumber(int codePoint) {
        return contains(NUMBER, codePoint);
    }

    /** Binary search over a flat array of inclusive [lo, hi] pairs. */
    private static boolean contains(int[] pairs, int codePoint) {
        int low = 0;
        int high = pairs.length / 2 - 1;
        while (low <= high) {
            int mid = (low + high) >>> 1;
            if (codePoint < pairs[mid * 2]) {
                high = mid - 1;
            } else if (codePoint > pairs[mid * 2 + 1]) {
                low = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }
}
'''
    return header + body + methods


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the committed copy would change")
    parser.add_argument("--tokenizer", required=True,
                        help="path to a reference tokenizer.json carrying the GPT-2 pre-tokenizer")
    args = parser.parse_args(argv)
    fresh = render(args.tokenizer)
    target = os.path.normpath(TARGET)
    if args.check:
        try:
            with open(target, encoding="utf-8") as handle:
                current = handle.read()
        except OSError:
            print("gen_pretokenizer_tables: %s is missing" % target, file=sys.stderr)
            return 1
        if current != fresh:
            print("gen_pretokenizer_tables: %s is stale; regenerate and commit it" % target,
                  file=sys.stderr)
            return 1
        print("gen_pretokenizer_tables: PreTokenizerTables.java matches the reference")
        return 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(fresh)
    print("wrote %s (%d bytes)" % (os.path.relpath(target), len(fresh)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
