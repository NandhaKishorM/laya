package com.convaiinnovations.laya.json;

import com.convaiinnovations.laya.lang.UnicodeTables;
import java.math.BigInteger;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Serialises a value the way CPython's {@code json.dumps(..., ensure_ascii=False)} does.
 *
 * <p>This is not a convenience. laya feeds the result straight into the tokenizer:
 * {@code common.serialize_state} JSON-encodes a non-string state and {@code render_criterion}
 * JSON-encodes a structured criterion, so <b>every character this produces becomes token ids</b>.
 * A formatter that agrees with Python on ordinary values and disagrees on one float produces a
 * different sequence, a different answer, and no test above the tokenizer can localise it.
 *
 * <p>Java's own spelling is not Python's, and the gap is not small:
 *
 * <table border="1">
 *   <caption>measured against CPython 3.12</caption>
 *   <tr><th>value</th><th>CPython</th><th>{@code Double.toString}</th></tr>
 *   <tr><td>1e16</td><td>{@code 1e+16}</td><td>{@code 1.0E16}</td></tr>
 *   <tr><td>1e-5</td><td>{@code 1e-05}</td><td>{@code 1.0E-5}</td></tr>
 *   <tr><td>0.0001</td><td>{@code 0.0001}</td><td>{@code 1.0E-4}</td></tr>
 *   <tr><td>1e23</td><td>{@code 1e+23}</td><td>{@code 1.0E23}</td></tr>
 * </table>
 *
 * <p>Python switches to scientific notation when the decimal exponent leaves
 * {@code [-4, 16)}; Java switches outside {@code [1e-3, 1e7)}, writes a mantissa of {@code 1.0}
 * where Python writes {@code 1}, uses {@code E} for {@code e}, and omits the exponent's sign and
 * zero padding. JDK 17's {@code Double.toString} is additionally not guaranteed to be the shortest
 * round-tripping form, so the digits themselves are recovered here rather than taken from it.
 */
public final class PythonJson {

    private PythonJson() {
    }

    /** The compact form: {@code ", "} between items and {@code ": "} after a key. */
    public static String dumps(Object value) {
        StringBuilder out = new StringBuilder();
        // The identity set tracks the containers currently being written, so a cycle is refused
        // rather than recursed into. The state is caller-supplied and a cycle used to arrive as a
        // StackOverflowError -- an Error, which a server's `catch (Exception)` does not contain.
        write(out, value, java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>()));
        return out.toString();
    }

    /**
     * How deep {@code dumps} will descend.
     *
     * <p>The identity set closed the CYCLIC case; a deep ACYCLIC caller-supplied state still
     * threw {@link StackOverflowError} -- the same {@link Error} the cycle fix exists to stop.
     */
    private static final int MAX_DEPTH = 512;

    private static void write(StringBuilder out, Object value, Set<Object> open) {
        write(out, value, open, 0);
    }

    private static void write(StringBuilder out, Object value, Set<Object> open, int depth) {
        if (depth > MAX_DEPTH) {
            throw new Json.JsonException("value nested deeper than " + MAX_DEPTH + " levels");
        }
        writeValue(out, value, open, depth);
    }

    private static void writeValue(StringBuilder out, Object value, Set<Object> open, int depth) {
        if (value == null) {
            out.append("null");
        } else if (value instanceof String) {
            writeString(out, (String) value);
        } else if (value instanceof Boolean) {
            out.append(((Boolean) value) ? "true" : "false");
        } else if (value instanceof Double || value instanceof Float) {
            out.append(repr(((Number) value).doubleValue()));
        } else if (value instanceof Integer || value instanceof Long
                || value instanceof Short || value instanceof Byte
                || value instanceof BigInteger) {
            // Python integers are arbitrary precision and `json.dumps` writes every digit, so an
            // integer criterion must not be narrowed on the way through.
            out.append(value.toString());
        } else if (value instanceof Map) {
            enter(open, value);
            writeObject(out, (Map<?, ?>) value, open, depth);
            open.remove(value);
        } else if (value instanceof List) {
            enter(open, value);
            writeArray(out, (List<?>) value, open, depth);
            open.remove(value);
        } else {
            throw new Json.JsonException(
                    "cannot serialise " + value.getClass().getName() + " the way Python's json "
                    + "does; pass a String, Boolean, Long, BigInteger, Double, List or Map");
        }
    }

    /** Refuses a container that is already being written, which is what a cycle looks like. */
    private static void enter(Set<Object> open, Object container) {
        if (!open.add(container)) {
            // CPython's wording, because a caller who hits this is reading both messages.
            throw new Json.JsonException("Circular reference detected");
        }
    }

    private static void writeObject(StringBuilder out, Map<?, ?> map, Set<Object> open,
            int depth) {
        out.append('{');
        boolean first = true;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!first) {
                out.append(", ");
            }
            first = false;
            writeString(out, key(entry.getKey()));
            out.append(": ");
            write(out, entry.getValue(), open, depth + 1);
        }
        out.append('}');
    }

    /**
     * A mapping key, coerced the way CPython coerces one.
     *
     * <p>CPython writes a non-string key as text rather than refusing it: {@code {1: "x"}} becomes
     * {@code {"1": "x"}}, {@code True} becomes {@code "true"} and {@code None} becomes
     * {@code "null"}. This used to throw instead, on the stated grounds that laya never produces
     * such a key -- which is true of laya and not of the <b>caller-supplied</b> state that reaches
     * here, so a {@code Map<Integer, ?>} state was refused where Python answers.
     *
     * <p>A key CPython itself refuses (a list, a map, an arbitrary object) is still refused here.
     */
    private static String key(Object key) {
        if (key instanceof String) {
            return (String) key;
        }
        if (key == null) {
            return "null";
        }
        if (key instanceof Boolean) {
            return ((Boolean) key) ? "true" : "false";
        }
        if (key instanceof Double || key instanceof Float) {
            return repr(((Number) key).doubleValue());
        }
        if (key instanceof Integer || key instanceof Long || key instanceof Short
                || key instanceof Byte || key instanceof BigInteger) {
            return key.toString();
        }
        throw new Json.JsonException(
                "keys must be a string, a number, a boolean or null, as CPython requires; got "
                + key.getClass().getName());
    }

    private static void writeArray(StringBuilder out, List<?> list, Set<Object> open,
            int depth) {
        out.append('[');
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) {
                out.append(", ");
            }
            write(out, list.get(i), open, depth + 1);
        }
        out.append(']');
    }

    /**
     * {@code ensure_ascii=False} escaping: the two structural characters, the five short control
     * escapes, {@code \\uXXXX} for the remaining control characters, and nothing else.
     *
     * <p>Non-ASCII passes through unescaped -- that is what {@code ensure_ascii=False} means, and
     * it matters for every CJK or Arabic state. {@code /} is <b>not</b> escaped, and neither are
     * the line separators {@code U+2028}/{@code U+2029}, which some JavaScript-oriented writers
     * escape and Python does not.
     */
    private static void writeString(StringBuilder out, String text) {
        out.append('"');
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                case '\b': out.append("\\b"); break;
                case '\f': out.append("\\f"); break;
                default:
                    if (c < 0x20) {
                        out.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
            }
        }
        out.append('"');
    }

    /**
     * A double spelled as CPython's {@code repr} spells it.
     *
     * <p>{@code json.dumps} writes a float with {@code float.__repr__}, which is the shortest
     * decimal string that round-trips, rendered in fixed notation while the decimal exponent is in
     * {@code [-4, 16)} and in scientific notation outside it, with a sign and at least two digits
     * on the exponent.
     *
     * <p>NaN and the infinities are written as the bare words CPython writes -- which is not legal
     * JSON, and is deliberately what CPython does by default, so a state carrying one produces the
     * same tokens here as there rather than a different error.
     */
    /**
     * CPython's {@code repr} for a string, which is what a {@code %r} in a message interpolates.
     *
     * <p>Needed because the router's reason strings are built with {@code %r} and one of them
     * interpolates the mixed segment -- a slice of the caller's own text. So this sees arbitrary
     * input, not a short language code, and every rule below is reachable from a pasted ticket.
     *
     * <p>The rules, in CPython's order:
     *
     * <ul>
     *   <li>The quote is a single quote, unless the string holds a single quote and no double
     *       quote, in which case the whole thing is double-quoted and nothing needs escaping.
     *       A string holding both is single-quoted with its single quotes escaped.
     *   <li>A backslash and the chosen quote are backslash-escaped; newline, carriage return and
     *       tab get their letter escapes.
     *   <li>Anything {@link UnicodeTables#isPrintable} calls unprintable becomes a numeric escape
     *       -- backslash-x and two hex digits below U+0100, backslash-u and four below U+10000,
     *       backslash-U and eight above it -- in LOWERCASE hex. (Spelled out rather than shown:
     *       a backslash-u sequence in a Java comment is translated before the file is lexed, so
     *       writing the escape here would stop this file compiling.)
     *   <li>Everything else passes through as itself, including every non-ASCII printable
     *       character: CPython 3 does not escape those.
     * </ul>
     *
     * <p>That last pair is where an ASCII-only escape check goes wrong, and it is not a corner:
     * U+00A0 NO-BREAK SPACE is unprintable to CPython and arrives in pasted text constantly, so
     * stopping at U+007F would write a raw control character into an API response.
     */
    public static String repr(String value) {
        if (value == null) {
            return "None";
        }
        char quote = value.indexOf('\'') >= 0 && value.indexOf('"') < 0 ? '"' : '\'';
        StringBuilder out = new StringBuilder(value.length() + 2);
        out.append(quote);
        int i = 0;
        while (i < value.length()) {
            int cp = value.codePointAt(i);
            i += Character.charCount(cp);
            if (cp == quote || cp == '\\') {
                out.append('\\').appendCodePoint(cp);
            } else if (cp == '\n') {
                out.append("\\n");
            } else if (cp == '\r') {
                out.append("\\r");
            } else if (cp == '\t') {
                out.append("\\t");
            } else if (UnicodeTables.isPrintable(cp)) {
                out.appendCodePoint(cp);
            } else if (cp < 0x100) {
                out.append(String.format(Locale.ROOT, "\\x%02x", cp));
            } else if (cp < 0x10000) {
                out.append(String.format(Locale.ROOT, "\\u%04x", cp));
            } else {
                out.append(String.format(Locale.ROOT, "\\U%08x", cp));
            }
        }
        return out.append(quote).toString();
    }

    /**
     * CPython's {@code %.0f}: the nearest integer, halves to EVEN.
     *
     * <p>{@code String.format("%.0f", v)} rounds halves UP, so the two disagree on every halfway
     * value -- {@code 12.5} is {@code "12"} in CPython and {@code "13"} in Java. The router
     * reports a percentage of letters with this, and a share of one in eight letters is exactly
     * 12.5, so the disagreement is reachable rather than theoretical.
     */
    public static String percent0(double fraction) {
        double scaled = 100.0 * fraction;
        if (Double.isNaN(scaled)) {
            return "nan";
        }
        if (Double.isInfinite(scaled)) {
            // CPython's %-formatting spells these in lower case, which is NOT how `repr` spells
            // them: `'%.0f' % float('nan')` is "nan" where `repr` gives "NaN". Latent today --
            // both call sites are finite by construction -- and wrong the moment one is not.
            return scaled > 0 ? "inf" : "-inf";
        }
        // Math.rint is IEEE ties-to-even, which is the rule; a long keeps a share above 2^31.
        return Long.toString((long) Math.rint(scaled));
    }

    public static String repr(double value) {
        if (Double.isNaN(value)) {
            return "NaN";
        }
        if (Double.isInfinite(value)) {
            return value > 0 ? "Infinity" : "-Infinity";
        }
        // Negative zero must keep its sign: `-0.0` is what CPython writes, and the sign is lost by
        // every arithmetic test (`value < 0` is false for it).
        boolean negative = (Double.doubleToRawLongBits(value) & Long.MIN_VALUE) != 0;
        double magnitude = Math.abs(value);
        String body = magnitude == 0.0 ? "0.0" : positiveRepr(magnitude);
        return negative ? "-" + body : body;
    }

    private static String positiveRepr(double value) {
        String digits = shortestDigits(value);
        int exponent = decimalExponent(value, digits);
        if (exponent >= -4 && exponent < 16) {
            return fixed(digits, exponent);
        }
        StringBuilder out = new StringBuilder();
        out.append(digits.charAt(0));
        if (digits.length() > 1) {
            out.append('.').append(digits, 1, digits.length());
        }
        out.append('e').append(exponent < 0 ? '-' : '+');
        String magnitude = Integer.toString(Math.abs(exponent));
        if (magnitude.length() < 2) {
            out.append('0');
        }
        return out.append(magnitude).toString();
    }

    /**
     * The significant digits of the shortest decimal string that round-trips to {@code value}.
     *
     * <p>Found by trying increasing precision rather than read off {@code Double.toString}: on
     * JDK 17 that method may emit more digits than necessary, which would make this print
     * {@code 0.30000000000000004} where CPython prints {@code 0.3}. Seventeen significant digits
     * always round-trip a double, so the loop terminates.
     */
    private static String shortestDigits(double value) {
        for (int precision = 1; precision <= 17; precision++) {
            String candidate = String.format(Locale.ROOT, "%." + (precision - 1) + "e", value);
            if (Double.parseDouble(candidate) == value) {
                String mantissa = candidate.substring(0, candidate.indexOf('e')).replace(".", "");
                // strip trailing zeros the formatter padded, but never the last digit
                int end = mantissa.length();
                while (end > 1 && mantissa.charAt(end - 1) == '0') {
                    end--;
                }
                return mantissa.substring(0, end);
            }
        }
        throw new IllegalStateException("no 17-digit decimal round-trips " + value);
    }

    /** The power of ten of {@code value}'s leading digit. */
    private static int decimalExponent(double value, String digits) {
        String formatted = String.format(Locale.ROOT, "%." + (digits.length() - 1) + "e", value);
        return Integer.parseInt(formatted.substring(formatted.indexOf('e') + 1));
    }

    /** Fixed notation, always with a decimal point and at least one digit after it. */
    private static String fixed(String digits, int exponent) {
        StringBuilder out = new StringBuilder();
        if (exponent < 0) {
            out.append("0.");
            for (int i = 0; i < -exponent - 1; i++) {
                out.append('0');
            }
            out.append(digits);
            return out.toString();
        }
        int whole = exponent + 1;
        if (digits.length() <= whole) {
            out.append(digits);
            for (int i = digits.length(); i < whole; i++) {
                out.append('0');
            }
            out.append(".0");
            return out.toString();
        }
        return out.append(digits, 0, whole).append('.').append(digits, whole, digits.length())
                .toString();
    }
}
