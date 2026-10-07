#!/usr/bin/env python3
"""Generate the Unicode tables `laya.lang`'s character predicates need, as Java source.

WHY A GENERATED TABLE RATHER THAN THE JDK'S OWN PREDICATES. `laya/lang.py` leans on three
character properties, and the JDK disagrees with CPython about all three:

  * `str.isalpha()` is categories Lu/Ll/Lt/Lm/Lo, which IS what `Character.isLetter` means -- but
    JDK 17 ships Unicode 13.0 and CPython 3.12 ships 15.0, so the JDK does not know 4,863 of the
    code points Python calls letters (U+0870 Arabic Extended-B, U+11F00 Kawi, ...). It reports them
    UNASSIGNED, so text in a recently-added script would be counted under no script at all and
    routed to the English checkpoint.
  * Python's `\\w` minus `\\d` minus `_` is letters plus the Nl and No numeric categories, which has
    the same version gap.
  * `unicodedata.combining(ch) != 0` is the canonical combining CLASS, which the JDK does not
    expose at all. Approximating it with the Mn and Mc categories -- which is what the .NET port
    does -- is wrong on 1,460 code points: 1,410 it would call combining that Python does not
    (U+034F COMBINING GRAPHEME JOINER among them, class 0) and 50 it would miss.
  * `str.isspace()` is the set `str.strip()` and `str.split()` work on, and the JDK has no
    predicate for it: `Character.isWhitespace` rejects four of its members (U+0085, U+00A0,
    U+2007, U+202F -- the non-breaking spaces and the C1 NEXT LINE) and `Character.isSpaceChar`
    is wrong the other way, accepting those while rejecting tab, newline and every control.

So the tables are recorded from CPython here and compiled in. That also makes the predicates
independent of the JDK a consumer happens to run, which matters more than it sounds: a JDK upgrade
must not silently change which script a request is detected as.

    python laya-java/scripts/gen_unicode_tables.py            # write
    python laya-java/scripts/gen_unicode_tables.py --check     # fail if the committed copy differs
"""
import argparse
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(
    HERE, "..", "laya-java", "src", "main", "java", "com", "convaiinnovations", "laya", "lang",
    "UnicodeTables.java")

WORD = re.compile(r"[^\W\d_]", re.UNICODE)
DIGIT = re.compile(r"\d", re.UNICODE)
MAX = 0x110000
SURROGATES = range(0xD800, 0xE000)


def ranges(predicate):
    """Inclusive [lo, hi] runs where `predicate` holds. Surrogates never hold."""
    out, start = [], None
    for cp in range(MAX):
        hit = False if cp in SURROGATES else predicate(cp)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAX - 1))
    return out


# A `new int[] {...}` literal compiles to one store instruction per element inside whatever
# method holds it, and a JVM method body may not exceed 65,535 bytes. With every table in the
# field initialisers they share one `<clinit>` and the class stopped compiling with "code too
# large" the moment the lowercase map was added. So each table gets its OWN factory method, which
# gives every one of them the whole budget, and the limit below refuses to emit a table that would
# overflow its own method rather than letting the build break later.
#
# Measured at roughly 9 bytes of bytecode per element (dup, index push, value push, iastore, with
# wider pushes above 32,767), so the cap is set well under 65,535/9 and still leaves room for a
# table to grow by half.
MAX_ELEMENTS_PER_TABLE = 4000


def _elements(name, values):
    """The wrapped element list for a table, refusing one that would not fit its method."""
    if len(values) > MAX_ELEMENTS_PER_TABLE:
        raise SystemExit(
            "gen_unicode_tables: %s has %d elements, over the %d a single factory method can "
            "hold. Split it across two methods (return a concatenation) before regenerating; a "
            "literal that overflows compiles to 'code too large' and not to a clear error."
            % (name, len(values), MAX_ELEMENTS_PER_TABLE))
    lines, current = [], "                "
    for value in values:
        piece = str(value) + ", "
        if len(current) + len(piece) > 100:
            lines.append(current.rstrip())
            current = "                "
        current += piece
    if current.strip():
        lines.append(current.rstrip().rstrip(","))
    return "\n".join(lines)


def ascii_mask(name, predicate):
    """The predicate over U+0000..U+007F as two 64-bit masks, plus the accessor that reads them.

    Most text a router sees is ASCII, and every ASCII character was paying a binary search over
    hundreds of ranges -- ten-odd comparisons to answer a question two shifts can. The masks are
    computed from the SAME predicate as the range table below, never written by hand, so the fast
    path cannot disagree with the slow one; and the digest tests compare both against CPython over
    every code point regardless.
    """
    low = sum(1 << cp for cp in range(0, 64) if predicate(cp))
    high = sum(1 << (cp - 64) for cp in range(64, 128) if predicate(cp))
    return ('    private static final long %s_ASCII_LOW = %sL;\n'
            '    private static final long %s_ASCII_HIGH = %sL;\n\n'
            % (name, _signed(low), name, _signed(high)))


def _signed(mask):
    """A 64-bit mask as a Java long literal, which has no unsigned form."""
    return str(mask - (1 << 64) if mask >= (1 << 63) else mask)


def ascii_branch(name):
    """The two-shift lookup that answers an ASCII code point without touching the range table."""
    return ('        if (codePoint < 0x80) {\n'
            '            long mask = codePoint < 64 ? %s_ASCII_LOW >>> codePoint\n'
            '                    : %s_ASCII_HIGH >>> (codePoint - 64);\n'
            '            return (mask & 1L) != 0L;\n'
            '        }\n' % (name, name))


def table(name, rows, doc):
    """One flat `int[]` of lo, hi pairs, built by its own factory method."""
    values = [value for row in rows for value in row]
    return ('    /**\n     * %s\n     *\n     * <p>%d ranges covering %d code points, recorded\n'
            '     * from CPython %s.\n     */\n'
            '    private static final int[] %s = %s();\n\n'
            '    private static int[] %s() {\n        return new int[] {\n%s\n        };\n'
            '    }\n'
            % (doc, len(rows), sum(b - a + 1 for a, b in rows),
               unicodedata.unidata_version, name, _factory(name), _factory(name),
               _elements(name, values)))


def _factory(name):
    """`ALPHA` -> `alphaTable`, so the factory reads as a method and not as a constant."""
    head, _, tail = name.partition("_")
    return head.lower() + "".join(part.capitalize() for part in ([tail] if tail else [])) + "Table"


def flat_table(name, values, doc):
    """One flat `int[]` of plain values, built by its own factory method."""
    return ('    /**\n     * %s\n     *\n     * <p>%d entries, recorded from CPython %s.\n'
            '     */\n'
            '    private static final int[] %s = %s();\n\n'
            '    private static int[] %s() {\n        return new int[] {\n%s\n        };\n'
            '    }\n'
            % (doc, len(values), unicodedata.unidata_version, name, _factory(name),
               _factory(name), _elements(name, values)))


def render():
    alpha = ranges(lambda cp: chr(cp).isalpha())
    word = ranges(lambda cp: WORD.fullmatch(chr(cp)) is not None)
    combining = ranges(lambda cp: unicodedata.combining(chr(cp)) != 0)
    digit = ranges(lambda cp: DIGIT.fullmatch(chr(cp)) is not None)
    upper = ranges(lambda cp: chr(cp).isupper())
    lower = ranges(lambda cp: chr(cp).islower())
    space = ranges(lambda cp: chr(cp).isspace())
    printable = ranges(lambda cp: chr(cp).isprintable())
    # `laya.email` needs two properties none of the tables above supply.
    #
    # MARK is the Mn/Mc/Me categories, which `_drop_marks` spells `unicodedata.category(ch)
    # .startswith("M")` and the TS port spells `\p{M}`. COMBINING is NOT a substitute: it is the
    # canonical combining CLASS, and 1,528 code points are category M with class zero --
    # U+034F COMBINING GRAPHEME JOINER, the Cyrillic Me numerals, the Thaana vowels. Using it
    # would leave those marks in the tail that `_is_english_signoff` matches structurally, so a
    # name carrying one would stop reading as a name.
    #
    # INITIAL is the Lu/Lt/Lo categories, the test `_is_english_signoff` applies to each token's
    # first letter. UPPER is not a substitute either: `str.isupper()` is the Uppercase property,
    # true of 1,951 code points against these 133,474, and the two sets are not nested -- it
    # holds circled and squared letters that are category So, and rejects every caseless script,
    # so `Obrigado, 山田` would stop being a sign-off and `Thanks, Ⓐ` would start being one.
    mark = ranges(lambda cp: unicodedata.category(chr(cp)).startswith("M"))
    initial = ranges(lambda cp: unicodedata.category(chr(cp)) in ("Lu", "Lt", "Lo"))

    # Every code point whose lowercase is more than one code point. In Unicode 15 there is
    # exactly one, and `laya/lang.py` depends on it: it replaces U+0130 before lowering, because
    # otherwise the line-length reasoning in `_leaf_non_english` would be wrong.
    multi = []
    simple = []
    for cp in range(MAX):
        if cp in SURROGATES:
            continue
        low = chr(cp).lower()
        if len(low) > 1:
            multi.append((cp, [ord(c) for c in low]))
        elif low != chr(cp):
            simple.append((cp, ord(low)))

    header = '''// GENERATED by laya-java/scripts/gen_unicode_tables.py -- do not edit.
//
// Regenerate with:
//     python laya-java/scripts/gen_unicode_tables.py
// and `--check` fails when the committed copy no longer matches CPython.
package com.convaiinnovations.laya.lang;

import java.util.ArrayList;
import java.util.List;

/**
 * The Unicode properties {@code laya.lang} depends on, recorded from CPython %s.
 *
 * <p>Compiled in rather than read from the JDK, because the JDK disagrees with CPython about every
 * one of them. {@code Character.isLetter} means the same categories as {@code str.isalpha()},
 * but JDK 17 ships Unicode 13.0 against CPython's 15.0 and therefore does not know %d of the code
 * points Python calls letters -- it reports them unassigned, which would count text in a
 * recently-added script under no script at all and route it to the English checkpoint. The
 * canonical combining class is not exposed by the JDK at all, and the Mn/Mc categories are not a
 * substitute: they differ from it on 1,460 code points, including U+034F COMBINING GRAPHEME
 * JOINER, whose class is zero. {@code str.isspace()} has no JDK equivalent either:
 * {@code Character.isWhitespace} rejects the non-breaking spaces this set holds, and
 * {@code Character.isSpaceChar} rejects tab, newline and every control.
 *
 * <p>A table also makes detection independent of the JDK a consumer runs, which matters more than
 * it sounds: a JDK upgrade must not silently change which script a request is detected as.
 */
public final class UnicodeTables {

    /** The CPython release these tables were recorded from. */
    public static final String UNICODE_VERSION = "%s";

    private UnicodeTables() {
    }

''' % (unicodedata.unidata_version, 4863, unicodedata.unidata_version)

    body = (
        table("ALPHA", alpha,
              "Python's {@code str.isalpha()}: categories Lu, Ll, Lt, Lm and Lo.")
        + "\n"
        + table("WORD", word,
                "Python's {@code [^\\\\W\\\\d_]}: letters plus the Nl and No numeric categories, "
                "which is\n     * what {@code laya.lang}'s word regex matches.")
        + "\n"
        + table("COMBINING", combining,
                "Code points with a non-zero canonical combining class.")
        + "\n"
        + table("DIGIT", digit,
                "Python's {@code \\\\d}: the Nd category. Needed because Python's {@code \\\\w} is\n"
                "     * letters plus every numeric category, and the regexes subtract {@code \\\\d} from it.")
        + "\n"
        + table("UPPER", upper, "Python's {@code str.isupper()} for one character.")
        + "\n"
        + table("LOWER", lower, "Python's {@code str.islower()} for one character.")
        + "\n"
        + table("SPACE", space,
                "Python's {@code str.isspace()}, which is the set {@code str.strip()} and\n"
                "     * {@code str.split()} separate on.")
        + "\n"
        + flat_table("LOWER_FROM", [cp for cp, _ in simple],
                     "Every code point whose Python lowercase is one DIFFERENT code point,\n"
                     "     * sorted. {@link #LOWER_TO} holds what each one maps to.")
        + "\n"
        + "\n"
        + table("MARK", mark,
                "The Mn/Mc/Me categories: what {@code laya.email} drops before matching a\n"
                "sign-off's tail, and what the other ports spell {@code \\p{M}}.")
        + table("INITIAL", initial,
                "The Lu/Lt/Lo categories: the letters a name may begin with, in any script.")
        + table("PRINTABLE", printable,
                "Python's {@code str.isprintable()}, which is what {@code repr} leaves\n"
                "     * unescaped. Everything else it spells as a numeric escape.")
        + "\n"
        + flat_table("LOWER_TO", [low for _, low in simple],
                     "The lowercase of each entry of {@link #LOWER_FROM}, in the same order.")
        + "\n"
        + ('    /*\n'
           '     * ASCII fast paths. Each pair of masks is computed from the same predicate as\n'
           '     * the range table above, so the fast path cannot disagree with the slow one, and\n'
           '     * both are compared against CPython over every code point by UnicodeTablesTest.\n'
           '     */\n')
        + "".join(ascii_mask(name, predicate) for name, predicate in (
            ("ALPHA", lambda cp: chr(cp).isalpha()),
            ("WORD", lambda cp: WORD.fullmatch(chr(cp)) is not None),
            ("COMBINING", lambda cp: unicodedata.combining(chr(cp)) != 0),
            ("DIGIT", lambda cp: DIGIT.fullmatch(chr(cp)) is not None),
            ("UPPER", lambda cp: chr(cp).isupper()),
            ("LOWER", lambda cp: chr(cp).islower()),
            ("SPACE", lambda cp: chr(cp).isspace()),
            ("PRINTABLE", lambda cp: chr(cp).isprintable()),
            ("MARK", lambda cp: unicodedata.category(chr(cp)).startswith("M")),
            ("INITIAL", lambda cp: unicodedata.category(chr(cp)) in ("Lu", "Lt", "Lo")),
        ))
    )

    multi_cases = "\n".join(
        '            case 0x%04X:\n                return "%s";'
        % (cp, "".join("\\u%04X" % c for c in low)) for cp, low in multi)
    # Measured in this repository on Corretto 17 (Unicode 13.0) against CPython 15.0; see
    # UnicodeTablesTest, which fails if the gap ever changes.
    jdk_gap = 40
    methods = '''
    /** Python's {@code str.isalpha()} for one code point. */
    public static boolean isAlpha(int codePoint) {
        return contains(ALPHA, codePoint);
    }

    /** Python's {@code [^\\\\W\\\\d_]} for one code point: a word character that is not a digit. */
    public static boolean isWordChar(int codePoint) {
        return contains(WORD, codePoint);
    }

    /** Whether {@code unicodedata.combining} would return non-zero for this code point. */
    public static boolean isCombining(int codePoint) {
        return contains(COMBINING, codePoint);
    }

    /** Python's {@code \\d}: the Nd category. The JDK misses 30 of these at Unicode 13. */
    public static boolean isDigit(int codePoint) {
        return contains(DIGIT, codePoint);
    }

    /**
     * Python's {@code \\w} for a str: alphanumeric or underscore.
     *
     * <p>Spelled out because the regexes in {@code laya/lang.py} subtract from it --
     * {@code [^\\W\\d_]} is this minus digits and underscore, {@code [^\\W_]} is this minus
     * underscore -- and Java's own {@code \\w} is a different set again (it admits combining
     * marks and join controls, which Python's does not).
     */
    public static boolean isPythonWordChar(int codePoint) {
        return codePoint == '_' || isWordChar(codePoint) || isDigit(codePoint);
    }

    /** Python's {@code [^\\W_]}: a word character that is not the underscore. */
    public static boolean isWordOrDigit(int codePoint) {
        return isWordChar(codePoint) || isDigit(codePoint);
    }

    /**
     * Python's {@code str.isupper()} for one code point.
     *
     * <p>A table, because {@code Character.isUpperCase} misses 40 of these and
     * {@code Character.isLowerCase} misses 200 at Unicode 13 -- U+10FC, a Georgian modifier
     * letter Python calls lowercase through Other_Lowercase, among them. Detection reads the case
     * of the first letter of a non-Latin run to tell a proper noun from a request, so a letter
     * whose case the JDK does not know would change that decision.
     */
    public static boolean isUpper(int codePoint) {
        return contains(UPPER, codePoint);
    }

    /** Python's {@code str.islower()} for one code point. */
    public static boolean isLower(int codePoint) {
        return contains(LOWER, codePoint);
    }

    /** Whether a code point has case at all, which is what {@code str.isupper()} requires one of. */
    public static boolean isCased(int codePoint) {
        return isUpper(codePoint) || isLower(codePoint)
                || Character.getType(codePoint) == Character.TITLECASE_LETTER;
    }

    /**
     * Python's {@code str.isupper()} for a whole string: at least one cased code point, and every
     * cased code point uppercase.
     */
    public static boolean isUpperString(String text) {
        boolean sawCased = false;
        int i = 0;
        while (i < text.length()) {
            int codePoint = text.codePointAt(i);
            i += Character.charCount(codePoint);
            if (!isCased(codePoint)) {
                continue;
            }
            if (!isUpper(codePoint)) {
                return false;
            }
            sawCased = true;
        }
        return sawCased;
    }

    /** Whether any code point of {@code text} is lowercase, as Python's {@code islower} sees it. */
    public static boolean hasLower(String text) {
        int i = 0;
        while (i < text.length()) {
            int codePoint = text.codePointAt(i);
            i += Character.charCount(codePoint);
            if (isLower(codePoint)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Python's {@code str.isprintable()} for one code point.
     *
     * <p>What {@code repr} leaves alone. Everything else it spells as a numeric escape, and that
     * includes code points no ASCII-only check catches: U+00A0 NO-BREAK SPACE, U+00AD SOFT
     * HYPHEN, U+200B ZERO WIDTH SPACE, U+FEFF, and the private-use and unassigned planes. A
     * pasted ticket is full of the first one, and it reaches a route's reason string through the
     * mixed segment, so an escape check that stopped at U+007F would put a raw control character
     * into an API response.
     */
    public static boolean isPrintable(int codePoint) {
        return contains(PRINTABLE, codePoint);
    }

    /**
     * Python's {@code str.isspace()} for one code point.
     *
     * <p>A table, because neither JDK predicate is this set. {@code Character.isWhitespace}
     * rejects four of its members -- U+0085 NEXT LINE, U+00A0 NO-BREAK SPACE, U+2007 FIGURE SPACE
     * and U+202F NARROW NO-BREAK SPACE -- and {@code Character.isSpaceChar} is wrong the other
     * way, accepting those while rejecting tab, newline and every control. Detection strips and
     * splits on this set, so a no-break space pasted into a ticket would otherwise fuse two words
     * into one token that no stopword list holds.
     */
    public static boolean isSpace(int codePoint) {
        return contains(SPACE, codePoint);
    }

    /**
     * Whether this code point is a combining mark: the Mn, Mc and Me categories.
     *
     * <p>Not {@link #isCombining}, which is the canonical combining CLASS. They disagree on 1,528
     * code points that are category M with class zero, so the two are not interchangeable here.
     */
    public static boolean isMark(int codePoint) {
        if (codePoint < 0x80) {
            long mask = codePoint < 64 ? MARK_ASCII_LOW >>> codePoint
                    : MARK_ASCII_HIGH >>> (codePoint - 64);
            return (mask & 1L) != 0L;
        }
        return contains(MARK, codePoint);
    }

    /**
     * Whether a name may begin with this code point: the Lu, Lt and Lo categories.
     *
     * <p>Not {@link #isUpper}, which is Python's {@code str.isupper()} -- the Uppercase property,
     * which holds caseless symbols this does not and misses every caseless script this does.
     */
    public static boolean isInitial(int codePoint) {
        if (codePoint < 0x80) {
            long mask = codePoint < 64 ? INITIAL_ASCII_LOW >>> codePoint
                    : INITIAL_ASCII_HIGH >>> (codePoint - 64);
            return (mask & 1L) != 0L;
        }
        return contains(INITIAL, codePoint);
    }

    /** Python's {@code str.strip()} with no argument. */
    public static String strip(String text) {
        int start = 0;
        int end = text.length();
        while (start < end) {
            int codePoint = text.codePointAt(start);
            if (!isSpace(codePoint)) {
                break;
            }
            start += Character.charCount(codePoint);
        }
        while (end > start) {
            int codePoint = text.codePointBefore(end);
            if (!isSpace(codePoint)) {
                break;
            }
            end -= Character.charCount(codePoint);
        }
        return text.substring(start, end);
    }

    /** Whether {@code text} is empty or entirely whitespace, which is Python's {@code not s.strip()}. */
    public static boolean isBlank(String text) {
        int i = 0;
        while (i < text.length()) {
            int codePoint = text.codePointAt(i);
            if (!isSpace(codePoint)) {
                return false;
            }
            i += Character.charCount(codePoint);
        }
        return true;
    }

    /**
     * Python's {@code str.split()} with no argument: runs of whitespace separate, and leading or
     * trailing whitespace yields no empty token.
     */
    public static List<String> splitOnWhitespace(String text) {
        List<String> out = new ArrayList<>();
        int i = 0;
        int length = text.length();
        while (i < length) {
            int codePoint = text.codePointAt(i);
            if (isSpace(codePoint)) {
                i += Character.charCount(codePoint);
                continue;
            }
            int start = i;
            while (i < length) {
                int inner = text.codePointAt(i);
                if (isSpace(inner)) {
                    break;
                }
                i += Character.charCount(inner);
            }
            out.add(text.substring(start, i));
        }
        return out;
    }

    /**
     * Python's {@code str.lower()}, which is not Java's.
     *
     * <p>Two things make this not {@link String#toLowerCase}. It applies the contextual rules,
     * so the Greek capital sigma at the end of a word lowercases to a FINAL sigma where Python
     * gives an ordinary one -- mapping one code point at a time removes the context and therefore
     * the rule. And {@link Character#toLowerCase} follows the JDK's Unicode version: at 13.0 it
     * does not know the lowercase of %d code points CPython 15.0 maps, among them all of
     * Vithkuqi, and returns them unchanged. So the mapping is a table here as well, which is what
     * makes a routing decision the same on every JDK. A test compares a digest over all
     * 1,114,112 code points against one recorded from CPython.
     */
    public static String pythonLower(String text) {
        StringBuilder out = new StringBuilder(text.length());
        int i = 0;
        while (i < text.length()) {
            int codePoint = text.codePointAt(i);
            i += Character.charCount(codePoint);
            String expanded = multiCharLower(codePoint);
            if (expanded != null) {
                out.append(expanded);
            } else {
                out.appendCodePoint(lowerCodePoint(codePoint));
            }
        }
        return out.toString();
    }

    /**
     * Python's {@code str.lower()} for one code point, from the table rather than from the JDK.
     *
     * <p>ASCII is answered without the search, which is most text: CPython lowercases exactly
     * A-Z below U+0080, and the digest test over every code point proves this shortcut agrees
     * with the table.
     */
    public static int lowerCodePoint(int codePoint) {
        if (codePoint < 0x80) {
            return codePoint >= 'A' && codePoint <= 'Z' ? codePoint + 32 : codePoint;
        }
        int low = 0;
        int high = LOWER_FROM.length - 1;
        while (low <= high) {
            int mid = (low + high) >>> 1;
            int key = LOWER_FROM[mid];
            if (codePoint < key) {
                high = mid - 1;
            } else if (codePoint > key) {
                low = mid + 1;
            } else {
                return LOWER_TO[mid];
            }
        }
        return codePoint;
    }

    /**
     * The one code point whose Python lowercase is longer than one code point.
     *
     * <p>{@code laya.lang} depends on it: it replaces U+0130 before lowering, because the
     * line-length reasoning in its leaf scan counts code points on the RAW line and this mapping
     * would otherwise make a four-character line reach four word tokens.
     */
    private static String multiCharLower(int codePoint) {
        switch (codePoint) {
%s
            default:
                return null;
        }
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
''' % (jdk_gap, multi_cases)

    # Splice the ASCII fast path into each accessor. Done here rather than in the template so the
    # template stays readable, and asserted so a renamed accessor cannot silently lose its fast
    # path and quietly become ten times slower on ordinary Latin text.
    for name in ("ALPHA", "WORD", "COMBINING", "DIGIT", "UPPER", "LOWER", "SPACE",
                 "PRINTABLE"):
        marker = "        return contains(%s, codePoint);\n" % name
        if methods.count(marker) != 1:
            raise SystemExit(
                "gen_unicode_tables: expected exactly one `contains(%s, codePoint)` accessor to "
                "attach the ASCII fast path to, found %d" % (name, methods.count(marker)))
        methods = methods.replace(marker, ascii_branch(name) + marker)
    return header + body + methods


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
            print("gen_unicode_tables: %s is missing" % target, file=sys.stderr)
            return 1
        if current != fresh:
            print("gen_unicode_tables: %s is stale; run "
                  "laya-java/scripts/gen_unicode_tables.py and commit the result" % target,
                  file=sys.stderr)
            return 1
        print("gen_unicode_tables: UnicodeTables.java matches CPython %s"
              % unicodedata.unidata_version)
        return 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(fresh)
    print("wrote %s (%d bytes)" % (os.path.relpath(target, os.path.join(HERE, "..", "..")),
                                   len(fresh)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
