package com.convaiinnovations.laya.tokenizer;

/**
 * The two character predicates the reference tokenizer uses, which Java's built-ins do not match.
 *
 * <p>The reference implementation is Rust, and {@code char::is_whitespace} there is the Unicode
 * <b>White_Space</b> property. {@link Character#isWhitespace} is not that property: it excludes the
 * non-breaking spaces {@code U+00A0}, {@code U+2007} and {@code U+202F} and the next-line
 * {@code U+0085}, and it includes the file/group/record/unit separators {@code U+001C}-{@code
 * U+001F} which White_Space does not.
 *
 * <p>Eight characters disagree, and they are reachable: an added token with {@code lstrip} absorbs
 * whitespace to its left, so {@code "a" + U+00A0 + "[MASK]"} tokenized to three ids here where the
 * reference gives two, and {@code "a" + U+001C + "[MASK]"} to two where the reference gives three.
 */
final class Unicode {

    private Unicode() {
    }

    /**
     * The Unicode {@code White_Space} property, which is what the reference tokenizer strips.
     *
     * <p>Spelled out rather than derived from a Java predicate, because no Java predicate is this
     * set: {@link Character#isSpaceChar} is the Z categories only (it misses the tab and newline
     * range) and {@link Character#isWhitespace} differs as described above. The set is small,
     * fixed by the Unicode standard, and this way it can be read against the standard.
     */
    /**
     * The same property for a full code point.
     *
     * <p>No code point above the BMP carries {@code White_Space}, so this narrows and delegates
     * -- but it narrows EXPLICITLY, because casting a surrogate pair's code point to {@code char}
     * silently asks about a different character.
     */
    static boolean isWhiteSpace(int codePoint) {
        return codePoint <= Character.MAX_VALUE && isWhiteSpace((char) codePoint);
    }

    static boolean isWhiteSpace(char c) {
        switch (c) {
            case '\t':          // U+0009 .. U+000D
            case '\n':
            case 0x000B:
            case '\f':
            case '\r':
            case ' ':           // U+0020
            case 0x0085:        // NEXT LINE
            case 0x00A0:        // NO-BREAK SPACE
            case 0x1680:        // OGHAM SPACE MARK
            case 0x2028:        // LINE SEPARATOR
            case 0x2029:        // PARAGRAPH SEPARATOR
            case 0x202F:        // NARROW NO-BREAK SPACE
            case 0x205F:        // MEDIUM MATHEMATICAL SPACE
            case 0x3000:        // IDEOGRAPHIC SPACE
                return true;
            default:
                return c >= 0x2000 && c <= 0x200A;   // EN QUAD .. HAIR SPACE
        }
    }

    /**
     * Rust's {@code char::is_alphanumeric}: the Unicode {@code Alphabetic} property, or a number.
     *
     * <p>{@link Character#isLetterOrDigit} is narrower -- it misses the other-numeric category, so
     * a superscript or a fraction counts as a word character to the reference and not to Java.
     * This only affects an added token with {@code single_word}, which neither shipped checkpoint
     * sets, so it is correctness kept rather than a bug fixed.
     */
    static boolean isAlphanumeric(char c) {
        if (Character.isAlphabetic(c)) {
            return true;        // L*, Nl and Other_Alphabetic
        }
        int type = Character.getType(c);
        return type == Character.DECIMAL_DIGIT_NUMBER || type == Character.OTHER_NUMBER;
    }
}
