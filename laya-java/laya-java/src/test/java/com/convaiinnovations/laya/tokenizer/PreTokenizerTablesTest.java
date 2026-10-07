package com.convaiinnovations.laya.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The letter and number classes are recorded from the reference by a behavioural probe, because
 * the property is not exposed. A probe can therefore be wrong in a way a round-trip through the
 * generator cannot catch -- regenerating agrees with itself whatever it asks -- so the classes are
 * checked here against properties the GPT-2 pattern's three alternatives imply.
 *
 * <p>These exist because a probe WAS wrong. It asked whether {@code "<cp>'s"} leaves {@code "'s"}
 * as its own piece, which is true of every code point that fails to absorb the apostrophe: every
 * whitespace code point and all 1,911 digits were recorded as letters. A letter run then swallowed
 * the inner whitespace of a state, and only a corpus fixture needing a checkpoint caught it.
 */
final class PreTokenizerTablesTest {

    /** The shipped english checkpoint's own pre_tokenizer node, verbatim. */
    private static PreTokenizer byteLevel() {
        Map<String, Object> node = new LinkedHashMap<>();
        node.put("type", "ByteLevel");
        node.put("add_prefix_space", Boolean.FALSE);
        node.put("trim_offsets", Boolean.TRUE);
        node.put("use_regex", Boolean.TRUE);
        return PreTokenizer.from(node);
    }

    @Test
    @DisplayName("no whitespace code point is a letter or a number")
    void whitespaceIsOutsideBothClasses() {
        // The alternatives ` ?\p{L}+` and ` ?\p{N}+` are reached before `\s+`, so a whitespace
        // code point inside either class is consumed by a word run and never becomes its own
        // piece -- which is what produced 11 ids where the reference produces 14.
        int examined = 0;
        for (int cp = 0; cp < 0x110000; cp++) {
            final int at = cp;
            if (Unicode.isWhiteSpace(cp)) {
                examined++;
                assertFalse(PreTokenizerTables.isLetter(cp),
                        () -> String.format("U+%04X is whitespace and must not be a letter", at));
                assertFalse(PreTokenizerTables.isNumber(cp),
                        () -> String.format("U+%04X is whitespace and must not be a number", at));
            }
        }
        assertEquals(25, examined, "the Unicode White_Space set is 25 code points");
    }

    @Test
    @DisplayName("the two classes are disjoint, and ASCII digits are only numbers")
    void classesAreDisjoint() {
        for (int cp = 0; cp < 0x110000; cp++) {
            final int at = cp;
            if (PreTokenizerTables.isLetter(cp)) {
                assertFalse(PreTokenizerTables.isNumber(cp),
                        () -> String.format("U+%04X is in both classes", at));
            }
        }
        for (int cp = '0'; cp <= '9'; cp++) {
            assertTrue(PreTokenizerTables.isNumber(cp), "an ASCII digit is a number");
            assertFalse(PreTokenizerTables.isLetter(cp), "an ASCII digit is not a letter");
        }
        for (int cp = 'a'; cp <= 'z'; cp++) {
            assertTrue(PreTokenizerTables.isLetter(cp), "an ASCII lowercase letter is a letter");
            assertFalse(PreTokenizerTables.isNumber(cp), "an ASCII letter is not a number");
        }
    }

    @Test
    @DisplayName("the tables carry the Unicode the JDK does not")
    void carriesTheReferencesUnicode() {
        // Each is a letter to the reference tokenizer and not to Corretto 17's Unicode 13.0.
        // Written as ranges rather than single values so the assertion does not pass on a table
        // that happens to include one code point of a block it is missing.
        int[][] laterThanTheJdk = {
            {0x2EBF0, 0x2EE5D},   // CJK Unified Ideographs Extension I (Unicode 15.1)
            {0x105C0, 0x105F3},   // Todhri (16.0)
            {0x10D4A, 0x10D65},   // Garay (16.0)
            {0x113A0, 0x113B2},   // Tulu-Tigalari (16.0)
            {0x16100, 0x1611D},   // Gurung Khema (16.0)
        };
        for (int[] block : laterThanTheJdk) {
            for (int cp = block[0]; cp <= block[1]; cp++) {
                final int at = cp;
                assertTrue(PreTokenizerTables.isLetter(cp),
                        () -> String.format("U+%04X is a letter to the reference", at));
            }
        }
    }

    @Test
    @DisplayName("an inner whitespace run is cut the way the reference cuts it")
    void innerWhitespaceRunsAreNotSwallowed() {
        // Recorded from the reference's own pre_tokenize_str. The two interesting rules are both
        // here: a space run before a word leaves its LAST space to the word, because ` ?\p{L}+`
        // is tried before `\s+(?!\S)`; and a TAB or newline run does not, because the optional
        // leading space of that alternative is a literal U+0020. So "  b" is two pieces and
        // "\t\tc" is three.
        PreTokenizer byteLevel = byteLevel();
        assertEquals(
                List.of("a",
                        "Ġ",
                        "Ġb",
                        "ĉ",
                        "ĉ",
                        "c",
                        "Ċ",
                        "Ċ",
                        "d",
                        "č",
                        "Ċ",
                        "e",
                        "ĠĠ",
                        "Ġf"),
                byteLevel.preTokenize("a  b\t\tc\n\nd\r\ne   f"));
    }

    @Test
    @DisplayName("a digit run stays its own piece next to a word")
    void digitRunsAreNotSwallowedByLetterRuns() {
        // The same bad probe put every digit in the letter class. Because ` ?\p{L}+` is tried
        // first, "abc123" became one piece instead of the reference's two.
        PreTokenizer byteLevel = byteLevel();
        assertEquals(List.of("abc", "123"), byteLevel.preTokenize("abc123"));
        assertEquals(List.of("123", "abc"), byteLevel.preTokenize("123abc"));
        assertEquals(List.of("a", "Ġ1", "Ġb"), byteLevel.preTokenize("a 1 b"));
    }
}
