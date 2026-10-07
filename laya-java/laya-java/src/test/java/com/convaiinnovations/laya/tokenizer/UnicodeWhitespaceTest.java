package com.convaiinnovations.laya.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The whitespace predicate an added token's {@code lstrip} uses must be Unicode
 * {@code White_Space}, which is what the reference tokenizer strips -- not Java's
 * {@link Character#isWhitespace}.
 *
 * <p>Eight characters disagree, and the disagreement is observable: with {@code lstrip} set, as
 * both shipped checkpoints set it on their mask token, {@code "a" + U+00A0 + "[MASK]"} tokenized
 * to three ids where the reference gives two.
 */
final class UnicodeWhitespaceTest {

    /** The characters where Java's own predicate and the Unicode property disagree. */
    private static final char[] JAVA_SAYS_SPACE_UNICODE_DOES_NOT =
            {0x001C, 0x001D, 0x001E, 0x001F};
    private static final char[] UNICODE_SAYS_SPACE_JAVA_DOES_NOT =
            {0x0085, 0x00A0, 0x2007, 0x202F};

    @Test
    @DisplayName("the property is the Unicode one, not Java's")
    void disagreesWithJavaExactlyWhereItShould() {
        for (char c : JAVA_SAYS_SPACE_UNICODE_DOES_NOT) {
            assertTrue(Character.isWhitespace(c), "Java calls U+" + hex(c) + " whitespace");
            assertFalse(Unicode.isWhiteSpace(c),
                    "U+" + hex(c) + " is not Unicode White_Space and must not be stripped");
        }
        for (char c : UNICODE_SAYS_SPACE_JAVA_DOES_NOT) {
            assertFalse(Character.isWhitespace(c), "Java does not call U+" + hex(c) + " whitespace");
            assertTrue(Unicode.isWhiteSpace(c),
                    "U+" + hex(c) + " IS Unicode White_Space and must be stripped");
        }
    }

    @Test
    @DisplayName("the whole White_Space set, and nothing outside it")
    void matchesTheProperty() {
        // The property is 25 code points below U+10000, fixed by the standard.
        int[] expected = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
                0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
                0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000};
        java.util.Set<Integer> wanted = new java.util.HashSet<>();
        for (int c : expected) {
            wanted.add(c);
        }
        int found = 0;
        for (int c = 0; c <= 0xFFFF; c++) {
            boolean is = Unicode.isWhiteSpace((char) c);
            assertEquals(wanted.contains(c), is, "U+" + hex((char) c));
            if (is) {
                found++;
            }
        }
        assertEquals(expected.length, found);
    }

    @Test
    @DisplayName("alphanumeric follows Rust's rule, which includes the other-number category")
    void alphanumericIncludesOtherNumbers() {
        assertTrue(Unicode.isAlphanumeric('a'));
        assertTrue(Unicode.isAlphanumeric('7'));
        assertTrue(Unicode.isAlphanumeric('½'), "VULGAR FRACTION ONE HALF is No");
        assertTrue(Unicode.isAlphanumeric('²'), "SUPERSCRIPT TWO is No");
        assertTrue(Unicode.isAlphanumeric('中'));
        assertFalse(Unicode.isAlphanumeric(' '));
        assertFalse(Unicode.isAlphanumeric('-'));
    }

    private static String hex(char c) {
        return String.format("%04X", (int) c);
    }
}
