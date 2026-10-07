package com.convaiinnovations.laya.lang;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Fixtures;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link LanguageDetection} against {@code laya/lang.py}, field for field.
 *
 * <p>Every expectation here was recorded by running the reference, never written by hand. That
 * matters more for this port than for the others: the reference leans on four regexes, two string
 * methods and a rounding rule that Java spells differently, and each difference is silent. A
 * hand-written expectation would have encoded what the port does rather than what the reference
 * does, which is the failure mode the fixtures exist to prevent.
 *
 * <p>Each case asserts the whole output and, for a plain string, every intermediate step too, so a
 * failure names the stage that broke -- the identifier substitution, the word scan, the acronym
 * blanking, the lowercase mapping -- instead of only reporting a different verdict.
 */
class LanguageDetectionParityTest {

    private static List<Map<String, Object>> cases() {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> rows =
                (List<Map<String, Object>>) Fixtures.load("lang_detect.json").get("cases");
        return rows;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        return (List<Object>) value;
    }

    private static List<String> strings(Object value) {
        List<String> out = new ArrayList<>();
        for (Object item : list(value)) {
            out.add((String) item);
        }
        return out;
    }

    /** The state as the reference saw it, rebuilt from the tagged fixture value. */
    private static Object state(Map<String, Object> row) {
        Map<String, Object> encoded = map(row.get("state"));
        String kind = (String) encoded.get("kind");
        if ("string".equals(kind)) {
            return encoded.get("value");
        }
        if ("bytes".equals(kind)) {
            List<Object> values = list(encoded.get("value"));
            byte[] out = new byte[values.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = (byte) ((Number) values.get(i)).intValue();
            }
            return out;
        }
        return encoded.get("value");
    }

    private static void assertProfile(String where, Object recorded, Map<String, Double> actual) {
        Map<String, Object> expected = map(recorded);
        // The ORDER is asserted as well as the contents: the reference builds this mapping with
        // Latin first and then every other script in first-encounter order, and callers see it.
        // A HashMap here would pass a contents-only check and still be wrong.
        assertEquals(new ArrayList<>(expected.keySet()), new ArrayList<>(actual.keySet()),
                where + ": script profile keys are in a different order");
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            assertEquals(((Number) entry.getValue()).doubleValue(), actual.get(entry.getKey()),
                    where + ": share of " + entry.getKey());
        }
    }

    @TestFactory
    @DisplayName("every recorded case matches the reference in every field")
    List<DynamicTest> parity() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Map<String, Object> row : cases()) {
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> assertCase(name, row)));
        }
        return tests;
    }

    private void assertCase(String name, Map<String, Object> row) {
        Object state = state(row);
        String text = (String) row.get("state_text");

        assertEquals(text, LanguageDetection.stateText(state), name + ": stateText");
        assertEquals(row.get("detect_script"), LanguageDetection.detectScript(text),
                name + ": detectScript");
        assertProfile(name, row.get("script_profile"), LanguageDetection.scriptProfile(text));

        Map<String, Object> expectedLatin = map(row.get("latin_profile"));
        LanguageDetection.LatinProfile latin = LanguageDetection.latinProfile(text);
        assertEquals(expectedLatin.get("language"), latin.language(),
                name + ": latinProfile language");
        assertEquals(((Number) expectedLatin.get("english_hits")).intValue(), latin.englishHits(),
                name + ": latinProfile englishHits");
        assertEquals(((Number) expectedLatin.get("diacritic_rate")).doubleValue(),
                latin.diacriticRate(), name + ": latinProfile diacriticRate");
        assertEquals(expectedLatin.get("looks_non_english"), latin.looksNonEnglish(),
                name + ": latinProfile looksNonEnglish");
        assertEquals(latin.language(), LanguageDetection.guessLatinLanguage(text),
                name + ": guessLatinLanguage must be latinProfile's language");

        Map<String, Object> expected = map(row.get("analyse"));
        LanguageDetection.Analysis analysis = LanguageDetection.analyse(state);
        assertEquals(expected.get("script"), analysis.script(), name + ": analyse script");
        assertProfile(name + " (analyse)", expected.get("script_profile"),
                analysis.scriptProfile());
        assertEquals(expected.get("language"), analysis.language(), name + ": analyse language");
        assertEquals(expected.get("is_english"), analysis.english(), name + ": analyse isEnglish");
        assertEquals(expected.get("language_undecided"), analysis.languageUndecided(),
                name + ": analyse languageUndecided");
        assertEquals(((Number) expected.get("diacritic_rate")).doubleValue(),
                analysis.diacriticRate(), name + ": analyse diacriticRate");
        assertEquals(((Number) expected.get("non_latin_fraction")).doubleValue(),
                analysis.nonLatinFraction(), name + ": analyse nonLatinFraction");
        assertEquals(expected.get("mixed_segment"), analysis.mixedSegment(),
                name + ": analyse mixedSegment");
        assertEquals(row.get("is_english"), LanguageDetection.isEnglish(state),
                name + ": isEnglish");

        if (row.containsKey("helpers")) {
            assertHelpers(name, (String) state, map(row.get("helpers")), latin);
        }
    }

    private void assertHelpers(String name, String text, Map<String, Object> helpers,
            LanguageDetection.LatinProfile latin) {
        assertEquals(strings(helpers.get("words")), LanguageDetection.words(text),
                name + ": the word scan");
        assertEquals(helpers.get("substitute_identifiers"),
                LanguageDetection.substituteIdentifiers(text),
                name + ": the identifier substitution");
        assertEquals(strings(helpers.get("non_latin_words")),
                LanguageDetection.nonLatinWords(text), name + ": the non-Latin word runs");
        assertEquals(helpers.get("named_prose_language"),
                LanguageDetection.namedProseLanguage(text), name + ": namedProseLanguage");
        assertEquals(helpers.get("has_code_line"), LanguageDetection.hasCodeLine(text),
                name + ": the code-line scan");
        assertEquals(helpers.get("blank_upper_runs"), LanguageDetection.blankUpperRuns(text),
                name + ": the acronym blanking");
        assertEquals(helpers.get("python_lower"), UnicodeTables.pythonLower(text),
                name + ": pythonLower");
        assertEquals(helpers.get("strip"), UnicodeTables.strip(text), name + ": strip");
        assertEquals(strings(helpers.get("split_whitespace")),
                UnicodeTables.splitOnWhitespace(text), name + ": splitOnWhitespace");
        assertEquals(strings(helpers.get("split_newline")), List.of(text.split("\n", -1)),
                name + ": splitting on newline must keep trailing empty fields");
        assertEquals(((Number) helpers.get("count_alpha")).intValue(),
                LanguageDetection.countAlpha(text), name + ": the letter count");
        assertEquals(((Number) helpers.get("code_point_length")).intValue(),
                text.codePointCount(0, text.length()), name + ": the code-point length");

        List<Boolean> joined = new ArrayList<>();
        for (String token : UnicodeTables.splitOnWhitespace(text)) {
            joined.add(LanguageDetection.hasJoined(token));
        }
        List<Boolean> expectedJoined = new ArrayList<>();
        for (Object value : list(helpers.get("has_joined_tokens"))) {
            expectedJoined.add((Boolean) value);
        }
        assertEquals(expectedJoined, joined, name + ": the compound-name scan, token by token");

        List<String> words = LanguageDetection.words(UnicodeTables.pythonLower(
                LanguageDetection.substituteIdentifiers(text).replace("İ", "i")));
        assertEquals(helpers.get("english_rescued_by_words"),
                LanguageDetection.englishRescuedByWords(words, latin.diacriticRate()),
                name + ": the English rescue");
    }

    @Test
    @DisplayName("the corpus covers the branches it was built for")
    void coverage() {
        // A parity suite that silently stopped exercising a branch would still be green, so the
        // corpus itself is asserted. These counts are the reason each group of cases was added;
        // if one drops to zero the fixture has lost its point and needs a case back.
        int mixedSegments = 0;
        int nonEnglish = 0;
        int undecided = 0;
        int nonLatinScripts = 0;
        int namedProse = 0;
        int rescued = 0;
        int identifiersRemoved = 0;
        int acronymsBlanked = 0;
        for (Map<String, Object> row : cases()) {
            Map<String, Object> analysis = map(row.get("analyse"));
            if (analysis.get("mixed_segment") != null) {
                mixedSegments++;
            }
            if (Boolean.FALSE.equals(analysis.get("is_english"))) {
                nonEnglish++;
            }
            if (Boolean.TRUE.equals(analysis.get("language_undecided"))) {
                undecided++;
            }
            String script = (String) analysis.get("script");
            if (!"latin".equals(script) && !"unknown".equals(script)) {
                nonLatinScripts++;
            }
            if (row.containsKey("helpers")) {
                Map<String, Object> helpers = map(row.get("helpers"));
                String raw = (String) map(row.get("state")).get("value");
                if (helpers.get("named_prose_language") != null) {
                    namedProse++;
                }
                if (Boolean.TRUE.equals(helpers.get("english_rescued_by_words"))) {
                    rescued++;
                }
                if (!raw.equals(helpers.get("substitute_identifiers"))) {
                    identifiersRemoved++;
                }
                if (!raw.equals(helpers.get("blank_upper_runs"))) {
                    acronymsBlanked++;
                }
            }
        }
        assertTrue(cases().size() >= 100, "the corpus shrank to " + cases().size() + " cases");
        assertTrue(mixedSegments >= 5, "only " + mixedSegments + " cases reach the segment scan,"
                + " which is the branch where a foreign line is found inside English prose");
        assertTrue(nonEnglish >= 40, "only " + nonEnglish + " cases route away from English");
        assertTrue(undecided >= 30, "only " + undecided + " cases are undecided, and undecided is"
                + " the answer that is neither English nor a named language");
        assertTrue(nonLatinScripts >= 14, "only " + nonLatinScripts + " non-Latin scripts");
        assertTrue(namedProse >= 15, "only " + namedProse + " cases name a prose language");
        assertTrue(rescued >= 10, "only " + rescued + " cases exercise the English rescue");
        assertTrue(identifiersRemoved >= 5, "only " + identifiersRemoved + " cases hold an"
                + " identifier for the substitution to remove");
        assertTrue(acronymsBlanked >= 5, "only " + acronymsBlanked + " cases hold an acronym for"
                + " the blanking to remove");
    }

    @Test
    @DisplayName("the identifier scan is linear, not quadratic, on a long unbroken token")
    void identifierScanIsLinear() {
        // The reference's pattern needs a lookbehind to stay linear: without it the greedy run is
        // retried at every offset inside a run of word characters and rescans it before failing,
        // which cost 205 ms on 4,000 characters against 0.45 ms for ordinary prose. A state is
        // user input, so this is a denial-of-service shape and not a tuning question.
        //
        // Asserted structurally rather than with a timer: 200,000 characters is 4x10^10 steps if
        // the scan is quadratic, so a green result here cannot be a fast machine.
        String run = "a".repeat(200_000);
        assertEquals(run, LanguageDetection.substituteIdentifiers(run),
                "a run with no joiner must come back untouched");
        assertEquals(" ", LanguageDetection.substituteIdentifiers(run + "." + run),
                "one joiner makes the whole thing a single identifier");
        assertEquals(1, LanguageDetection.words(run).size());
        assertFalse(LanguageDetection.hasJoined(run));
    }

    @Test
    @DisplayName("the integer round is half-to-even, pinned against CPython")
    void roundToIntIsHalfToEven() {
        // Pinned here rather than through the corpus on purpose. The rounded value feeds one
        // comparison -- nonLatinLetters >= NON_LATIN_MIN_LETTERS -- and half-to-even and half-up
        // differ only at an exact k+0.5 with k even, which at an EVEN threshold both land on the
        // same side of. A mutant that swapped the mode therefore passed every recorded case. The
        // rule is still matched, so that changing that table cannot introduce a divergence, and
        // this is what checks it.
        int halfwayCases = 0;
        int disagreements = 0;
        for (Object entry : list(Fixtures.load("lang_detect.json").get("round_to_int"))) {
            List<Object> pair = list(entry);
            double value = ((Number) pair.get(0)).doubleValue();
            int expected = ((Number) pair.get(1)).intValue();
            assertEquals(expected, LanguageDetection.roundHalfEven(value),
                    "round(" + value + ") disagrees with CPython");
            if (value == Math.floor(value) + 0.5) {
                halfwayCases++;
                if ((long) Math.floor(value + 0.5) != expected) {
                    disagreements++;
                }
            }
        }
        assertTrue(halfwayCases >= 20,
                "only " + halfwayCases + " halfway values, which are the only ones that can tell"
                + " the two modes apart");
        assertTrue(disagreements >= 10, "only " + disagreements + " of the halfway values actually"
                + " distinguish half-to-even from half-up, so this test would not notice the"
                + " difference");
    }

    @Test
    @DisplayName("the mode is unobservable at an even threshold and observable at an odd one")
    void theRoundingModeIsUnobservableAtTheCurrentThreshold() {
        // The measurement behind the comment on roundHalfEven, kept as a test so the claim is
        // checked. If NON_LATIN_MIN_LETTERS ever becomes odd this fails, and the surviving mutant
        // stops being explained by a property of the table and starts being a real gap.
        int threshold = LanguageTables.NON_LATIN_MIN_LETTERS;
        int distinguishing = 0;
        for (int k = 0; k <= 4000; k++) {
            double halfway = k + 0.5;
            boolean even = LanguageDetection.roundHalfEven(halfway) >= threshold;
            boolean up = (long) Math.floor(halfway + 0.5) >= threshold;
            if (even != up) {
                distinguishing++;
            }
        }
        assertEquals(threshold % 2, 0,
                "NON_LATIN_MIN_LETTERS is now odd, so the rounding mode IS observable through"
                + " analyse and the corpus must grow a case for it");
        assertEquals(0, distinguishing,
                "at an even threshold no halfway product can distinguish the two modes, yet "
                + distinguishing + " did");
        // ...and it really would be observable one lower, which is what makes the above a fact
        // about the threshold rather than about the rounding.
        int observableAtNine = 0;
        for (int k = 0; k <= 40; k++) {
            double halfway = k + 0.5;
            if ((LanguageDetection.roundHalfEven(halfway) >= threshold - 1)
                    != ((long) Math.floor(halfway + 0.5) >= threshold - 1)) {
                observableAtNine++;
            }
        }
        assertEquals(1, observableAtNine,
                "at a threshold of " + (threshold - 1) + " exactly one halfway product"
                + " distinguishes the modes, namely " + (threshold - 2) + ".5");
    }

    @Test
    @DisplayName("the two Latin cuts are equivalent only because no script reaches below U+0370")
    void theTwoLatinCutsAreInertForNow() {
        // scriptOf cuts Latin at U+0250 and the script counter at U+02B0, so they differ over the
        // IPA extensions. A mutant that unified them survived -- and this is why: no named script
        // range reaches that low, so no code point in the gap can name a script either way. The
        // faithful pair of constants is kept, and this test fails if a script range is ever added
        // below U+02B0, at which point the difference stops being inert.
        int lowest = Integer.MAX_VALUE;
        for (LanguageTables.Script script : LanguageTables.SCRIPT_RANGES) {
            int[] ranges = script.ranges();
            for (int r = 0; r < ranges.length; r += 2) {
                lowest = Math.min(lowest, ranges[r]);
            }
        }
        assertTrue(lowest > 0x02AF,
                "a script range now starts at U+" + Integer.toHexString(lowest) + ", inside the"
                + " U+0250..U+02AF gap between the two Latin cuts. They are no longer"
                + " interchangeable and the corpus needs a case in that range.");
        for (int cp = 0x0250; cp <= 0x02AF; cp++) {
            assertEquals(null, LanguageDetection.scriptOf(cp),
                    "U+" + Integer.toHexString(cp) + " names a script, which the gap forbids");
        }
    }

    @Test
    @DisplayName("no code point whose case the JDK disputes can start a non-Latin run")
    void theCaseDisputeCannotReachARunStart() {
        // A mutant that read the first letter of a non-Latin run with Character.isUpperCase
        // instead of the table survived. The reason is structural: the 40 code points CPython and
        // JDK 17 disagree about are all outside every script range, so none of them can be the
        // first character of a run. The table is still used -- it is the same predicate
        // everywhere else -- and this test fails if a disputed code point ever lands inside a
        // range, which would make the mutant a real defect.
        int reachable = 0;
        StringBuilder found = new StringBuilder();
        for (int cp = 0; cp < 0x110000; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) {
                continue;
            }
            if (UnicodeTables.isUpper(cp) == Character.isUpperCase(cp)) {
                continue;
            }
            if (LanguageDetection.scriptOf(cp) != null) {
                reachable++;
                if (found.length() < 60) {
                    found.append("U+").append(Integer.toHexString(cp)).append(' ');
                }
            }
        }
        assertEquals(0, reachable,
                "the JDK disputes the case of " + reachable + " code points that CAN start a"
                + " non-Latin run (" + found + "), so the corpus needs a case for one of them");
    }

    @Test
    @DisplayName("trailing newlines change nothing, which is why the keep-empties split is inert")
    void trailingEmptyLinesAreInert() {
        // The reference splits on newline Python-style, keeping trailing empty fields; Java's
        // String.split drops them, so the port passes -1. A mutant that dropped the -1 survived,
        // and this is the reason: both scans skip any line shorter than seven characters, so an
        // empty line can never be selected. The faithful spelling is kept and the equivalence is
        // asserted rather than assumed.
        String body = "The customer opened this ticket yesterday and we asked for a screenshot.\n"
                + "Nao consigo entrar na minha conta e a senha nao funciona de jeito nenhum";
        LanguageDetection.Analysis plain = LanguageDetection.analyse(body);
        for (String suffix : new String[] {"\n", "\n\n", "\n\n\n", "\n \n"}) {
            assertEquals(plain, LanguageDetection.analyse(body + suffix),
                    "a trailing newline changed the verdict");
        }
        assertEquals(3, body.concat("\n\n").split("\n", -1).length - 1,
                "the split itself does keep the empty fields");
    }

    @Test
    @DisplayName("a state nested past the depth limit contributes nothing")
    void depthLimit() {
        Object deep = "Ich kann mich nicht in mein Konto einloggen und das Passwort";
        for (int i = 0; i < 6; i++) {
            deep = List.of(deep);
        }
        assertEquals("Ich kann mich nicht in mein Konto einloggen und das Passwort",
                LanguageDetection.stateText(deep), "six levels of nesting are still read");
        assertEquals("", LanguageDetection.stateText(List.of(deep)),
                "the seventh level is not, which is where the reference stops");
    }

    @Test
    @DisplayName("an undecodable byte leaf contributes nothing rather than mojibake")
    void undecodableBytes() {
        byte[] invalid = {(byte) 0xFF, (byte) 0xFE, 'h', 'i'};
        assertEquals("", LanguageDetection.stateText(invalid));
        assertTrue(LanguageDetection.isEnglish(invalid),
                "a state with no readable text is left to the English checkpoint");
    }

    @Test
    @DisplayName("a Set state is not descended into, because its order is unspecified")
    void setStateIsIgnored() {
        // Deliberate: Set.of randomises iteration order per JVM run, so reading one would make
        // detection non-deterministic. The reference accepts str, bytes, Mapping, list and tuple,
        // and a Java Set is none of those.
        assertEquals("", LanguageDetection.stateText(java.util.Set.of("a", "b")));
    }
}
