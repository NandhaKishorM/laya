package com.convaiinnovations.laya.lang;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Fixtures;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The compiled tables against the fixture recorded from {@code laya/lang.py}.
 *
 * <p>{@link LanguageTables} is generated, so this is not a test of code: it is the gate that keeps
 * a generated file honest. It is worth having because the tables are <em>derived</em> in Python --
 * {@code _SHARED_WORDS}, {@code _EN_ONLY_WORDS} and {@code _EN_COLLISION_WORDS} are computed at
 * import time from the stopword lists -- so a hand-maintained Java copy would drift silently, and
 * the drift would show up as a misrouted request rather than as a failure.
 *
 * <p>Two orderings get their own tests, because they are data and not formatting: the stopword
 * order decides which language wins a tied score, and the script order decides which script claims
 * a code point two ranges both cover.
 */
class LanguageTablesTest {

    private static Map<String, Object> tables() {
        return Fixtures.load("lang_tables.json");
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        return (List<Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    private static Set<String> strings(Object value) {
        Set<String> out = new TreeSet<>();
        for (Object item : list(value)) {
            out.add((String) item);
        }
        return out;
    }

    private void assertWords(String key, Set<String> compiled) {
        assertEquals(strings(tables().get(key)), new TreeSet<>(compiled),
                key + " drifted from laya/lang.py; regenerate with"
                + " scripts/gen_language_tables.py");
    }

    @Test
    @DisplayName("every word list equals the one recorded from Python")
    void wordLists() {
        assertWords("short_swedish_words", LanguageTables.SHORT_SWEDISH_WORDS);
        assertWords("non_en_diacritics", LanguageTables.NON_EN_DIACRITICS);
        assertWords("shared_words", LanguageTables.SHARED_WORDS);
        assertWords("nordic_overlap_words", LanguageTables.NORDIC_OVERLAP_WORDS);
        assertWords("en_only_words", LanguageTables.EN_ONLY_WORDS);
        assertWords("en_collision_words", LanguageTables.EN_COLLISION_WORDS);
    }

    @Test
    @DisplayName("the stopword lists equal Python's, language by language")
    void stopWords() {
        Map<String, Object> recorded = map(tables().get("stop_words"));
        assertEquals(recorded.keySet(), new TreeSet<>(LanguageTables.STOP_WORDS.keySet()),
                "a language was added to or dropped from the stopword lists");
        for (Map.Entry<String, Object> entry : recorded.entrySet()) {
            assertEquals(strings(entry.getValue()),
                    new TreeSet<>(LanguageTables.STOP_WORDS.get(entry.getKey())),
                    "the " + entry.getKey() + " stopword list drifted from laya/lang.py");
        }
    }

    @Test
    @DisplayName("the stopword ORDER is Python's, because it decides a tied score")
    void stopWordOrder() {
        List<String> recorded = new ArrayList<>();
        for (Object code : list(tables().get("stop_word_order"))) {
            recorded.add((String) code);
        }
        assertEquals(recorded, new ArrayList<>(LanguageTables.STOP_WORDS.keySet()),
                "STOP_WORDS is iterated to score languages and the winner is the FIRST of equal"
                + " scores, so this order is data. A Map.of() or a sorted map here would change"
                + " which language a tie resolves to.");
    }

    @Test
    @DisplayName("the script ranges equal Python's, in Python's order")
    void scriptRanges() {
        List<Object> recorded = list(tables().get("script_ranges"));
        assertEquals(recorded.size(), LanguageTables.SCRIPT_RANGES.size());
        for (int i = 0; i < recorded.size(); i++) {
            Map<String, Object> row = map(recorded.get(i));
            LanguageTables.Script script = LanguageTables.SCRIPT_RANGES.get(i);
            assertEquals(row.get("name"), script.name(), "script " + i + " is out of order");
            List<Object> ranges = list(row.get("ranges"));
            int[] compiled = script.ranges();
            assertEquals(ranges.size() * 2, compiled.length,
                    script.name() + " has a different number of ranges");
            for (int r = 0; r < ranges.size(); r++) {
                List<Object> pair = list(ranges.get(r));
                assertEquals(((Number) pair.get(0)).intValue(), compiled[r * 2],
                        script.name() + " range " + r + " low bound");
                assertEquals(((Number) pair.get(1)).intValue(), compiled[r * 2 + 1],
                        script.name() + " range " + r + " high bound");
            }
        }
    }

    @Test
    @DisplayName("the thresholds equal Python's, to the bit")
    void thresholds() {
        Map<String, Object> recorded = map(tables().get("thresholds"));
        assertEquals(((Number) recorded.get("NON_EN_DIACRITIC_RATE")).doubleValue(),
                LanguageTables.NON_EN_DIACRITIC_RATE);
        assertEquals(((Number) recorded.get("ENGLISH_RESCUE_DIACRITIC_RATE")).doubleValue(),
                LanguageTables.ENGLISH_RESCUE_DIACRITIC_RATE);
        assertEquals(((Number) recorded.get("NON_LATIN_FRACTION")).doubleValue(),
                LanguageTables.NON_LATIN_FRACTION);
        assertEquals(((Number) recorded.get("NON_LATIN_MIN_FRACTION")).doubleValue(),
                LanguageTables.NON_LATIN_MIN_FRACTION);
        assertEquals(((Number) recorded.get("NON_LATIN_MIN_LETTERS")).intValue(),
                LanguageTables.NON_LATIN_MIN_LETTERS);
    }

    @Test
    @DisplayName("the flat lookup agrees with the ordered scan on every code point")
    void flatLookupMatchesTheOrderedScan() {
        // The reference walks the 25 scripts in order and takes the first range that matches.
        // Detection does a binary search over one sorted table instead, which is only the same
        // function while no two ranges overlap. The generator refuses to emit the table otherwise
        // -- and this sweeps all 1,114,112 code points to prove the two have not come apart.
        List<LanguageTables.Script> scripts = LanguageTables.SCRIPT_RANGES;
        int[][] ranges = new int[scripts.size()][];
        for (int i = 0; i < scripts.size(); i++) {
            ranges[i] = scripts.get(i).ranges();
        }
        int disagreements = 0;
        int firstDisagreement = -1;
        for (int cp = 0; cp < 0x110000; cp++) {
            String scanned = null;
            outer:
            for (int i = 0; i < ranges.length; i++) {
                for (int r = 0; r < ranges[i].length; r += 2) {
                    if (cp >= ranges[i][r] && cp <= ranges[i][r + 1]) {
                        scanned = scripts.get(i).name();
                        break outer;
                    }
                }
            }
            String searched = LanguageTables.lookupScript(cp);
            if (!java.util.Objects.equals(scanned, searched)) {
                disagreements++;
                if (firstDisagreement < 0) {
                    firstDisagreement = cp;
                }
            }
        }
        assertEquals(0, disagreements,
                "the binary search and the ordered scan disagree on " + disagreements
                + " code points, first at U+"
                + (firstDisagreement < 0 ? "none" : Integer.toHexString(firstDisagreement)));
    }

    @Test
    @DisplayName("scriptOf over every code point matches the digest recorded from Python")
    void scriptOfDigest() {
        @SuppressWarnings("unchecked")
        Map<String, Object> digests =
                (Map<String, Object>) Fixtures.load("lang_detect.json").get("unicode_digests");
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is required of every JVM", impossible);
        }
        for (int cp = 0; cp < 0x110000; cp++) {
            String name = LanguageDetection.scriptOf(cp);
            digest.update(((name == null ? "" : name) + ";").getBytes(
                    java.nio.charset.StandardCharsets.US_ASCII));
        }
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest.digest()) {
            hex.append(Character.forDigit((value >> 4) & 0xF, 16));
            hex.append(Character.forDigit(value & 0xF, 16));
        }
        assertEquals(digests.get("script_of"), hex.toString(),
                "scriptOf disagrees with laya.lang._script_of somewhere in Unicode. Note that its"
                + " Latin cut (U+0250) is deliberately LOWER than the one the script counter uses"
                + " (U+02B0): the IPA extensions are counted as Latin but name no script.");
    }

    @Test
    @DisplayName("the early exit above the last named script is where the table says it is")
    void scriptMaxCodePoint() {
        int highest = -1;
        for (LanguageTables.Script script : LanguageTables.SCRIPT_RANGES) {
            int[] ranges = script.ranges();
            for (int r = 1; r < ranges.length; r += 2) {
                highest = Math.max(highest, ranges[r]);
            }
        }
        assertEquals(highest, LanguageTables.SCRIPT_MAX_CODE_POINT,
                "lookupScript skips the search above SCRIPT_MAX_CODE_POINT, so a script range"
                + " beyond it would become invisible");
        assertNull(LanguageTables.lookupScript(LanguageTables.SCRIPT_MAX_CODE_POINT + 1));
        assertEquals(-1, LanguageTables.lookupScriptIndex(0x110000 - 1));
    }

    @Test
    @DisplayName("a Script hands out copies, so a caller cannot edit the table under detection")
    void scriptRangesAreCopied() {
        LanguageTables.Script script = LanguageTables.SCRIPT_RANGES.get(0);
        int[] first = script.ranges();
        int[] second = script.ranges();
        assertTrue(first != second, "ranges() must not hand out the backing array");
        first[0] = -1;
        assertEquals(second[0], script.ranges()[0], "editing the copy changed the table");
    }

    @Test
    @DisplayName("SCRIPT_COUNT is the number of named scripts, which sizes the tally")
    void scriptCount() {
        assertEquals(LanguageTables.SCRIPT_RANGES.size(), LanguageTables.SCRIPT_COUNT);
        assertSame(LanguageTables.SCRIPT_RANGES.get(0).name(),
                LanguageTables.scriptName(0),
                "scriptName must index the same declaration order the tally slots use");
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> LanguageTables.scriptName(LanguageTables.SCRIPT_COUNT));
    }
}
