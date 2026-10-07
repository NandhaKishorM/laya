package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Option rendering is the text the model reads, so it is a character-for-character contract.
 *
 * <p>Everything downstream is token ids derived from these strings; a rendering difference changes
 * the answer and nothing in the answer looks wrong.
 */
final class QuestionRenderingTest {

    private static Map<String, Object> criteria(Object... pairs) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            out.put((String) pairs[i], pairs[i + 1]);
        }
        return out;
    }

    @Test
    @DisplayName("a described choice renders 'label: description', an undescribed one its label alone")
    void choiceRendering() {
        assertIterableEquals(List.of("refund: money back", "other"),
                Question.choice("i", criteria("refund", "money back", "other", null)).renderOptions());
        assertIterableEquals(List.of("a"),
                Question.choice("i", criteria("a", "")).renderOptions());
    }

    @Test
    @DisplayName("only null and the empty string mean 'no description': 0 and false are values")
    void falsyCriteriaAreValues() {
        // Python's rule is `v is None or v == ""`, so a zero and a false are legitimate criteria
        // and must render. A port using a truthiness test loses both.
        assertIterableEquals(List.of("zero: 0", "no: false", "bare"),
                Question.choice("i", criteria("zero", 0L, "no", Boolean.FALSE, "bare", null))
                        .renderOptions());
    }

    @Test
    @DisplayName("a structured criterion renders as the JSON the model was shown")
    void structuredCriteria() {
        List<String> rendered = Question.choice("i",
                criteria("a", Map.of("d", 1L), "b", List.of(1L, 2L))).renderOptions();
        // Python spells these with `", "` and `": "` separators, which `PythonJson` reproduces.
        assertEquals(2, rendered.size());
        org.junit.jupiter.api.Assertions.assertTrue(rendered.contains("a: {\"d\": 1}"), rendered.toString());
        org.junit.jupiter.api.Assertions.assertTrue(rendered.contains("b: [1, 2]"), rendered.toString());
    }

    @Test
    @DisplayName("choice options follow the criteria map's INSERTION order, not a sorted one")
    void choiceOrderIsInsertionOrder() {
        // Positional: two orders are two different questions, and the answer's probabilities are
        // indexed by position. A HashMap passed in must not decide what the model was asked.
        assertIterableEquals(List.of("zebra: z", "apple: a"),
                Question.choice("i", criteria("zebra", "z", "apple", "a")).renderOptions());
        List<String> copied = new ArrayList<>(
                Question.choice("i", criteria("zebra", "z", "apple", "a")).labels());
        assertIterableEquals(List.of("zebra", "apple"), copied);
    }

    @Test
    @DisplayName("score levels render with their index and a legend keyed by index")
    void scoreRendering() {
        Question q = Question.score("i", List.of("bad", "ok"));
        assertIterableEquals(List.of("level 0: bad", "level 1: ok"), q.renderOptions());
        assertEquals(Map.of("0", "bad", "1", "ok"), q.legend());
        assertIterableEquals(List.of("0", "1"), q.labels());
    }

    @Test
    @DisplayName("a numeric score level comes back as text, so the answer's shape is stable")
    void scoreLegendStringifiesNumbers() {
        // A scale passed as [1, 2, 3] must not make the legend's value types depend on the caller.
        assertEquals(Map.of("0", "1", "1", "2"), Question.score("i", List.of(1L, 2L)).legend());
    }

    @Test
    @DisplayName("noul is always [false, true] with the documented defaults")
    void noulDefaults() {
        assertIterableEquals(
                List.of("false: no, the statement does not hold", "true: yes, the statement holds"),
                Question.noul("i").renderOptions());
    }

    @Test
    @DisplayName("noul labels replace the label but not the default descriptions")
    void noulLabels() {
        Question q = Question.noul("i", null, null, Map.of("false", " nope ", "true", "yep"));
        assertIterableEquals(
                List.of("nope: no, the statement does not hold", "yep: yes, the statement holds"),
                q.renderOptions());
        assertIterableEquals(List.of("nope", "yep"), q.labels());
    }

    @Test
    @DisplayName("noul labels are validated: both present, non-blank and distinct")
    void noulLabelsValidated() {
        assertThrows(IllegalArgumentException.class,
                () -> Question.noul("i", null, null, Map.of("false", "a")));
        assertThrows(IllegalArgumentException.class,
                () -> Question.noul("i", null, null, Map.of("false", "  ", "true", "b")));
        assertThrows(IllegalArgumentException.class,
                () -> Question.noul("i", null, null, Map.of("false", "same", "true", "same")));
    }

    @Test
    @DisplayName("a legend is a score question's alone")
    void legendIsScoreOnly() {
        assertThrows(IllegalStateException.class, () -> Question.noul("i").legend());
        assertThrows(IllegalStateException.class,
                () -> Question.choice("i", criteria("a", "b")).legend());
    }

    @Test
    @DisplayName("an empty question is refused at construction, not at inference")
    void emptyQuestionsRefused() {
        assertThrows(IllegalArgumentException.class, () -> Question.choice("i", Map.of()));
        assertThrows(IllegalArgumentException.class, () -> Question.score("i", List.of()));
        assertThrows(IllegalArgumentException.class, () -> Question.choice(null, criteria("a", "b")));
    }

    @Test
    @DisplayName("blank instructions are refused, not just null ones")
    void blankInstructionsRefused() {
        // An empty instruction produces the text "choice question: " and asks the model nothing,
        // which it answers anyway with a confident-looking distribution. Python refuses it.
        for (String blank : new String[] {"", "   ", "\t\n"}) {
            assertThrows(IllegalArgumentException.class,
                    () -> Question.choice(blank, criteria("a", "b")), "choice with " + blank.length()
                            + " blank chars");
            assertThrows(IllegalArgumentException.class, () -> Question.noul(blank));
            assertThrows(IllegalArgumentException.class, () -> Question.score(blank, List.of("x")));
        }
    }

    @Test
    @DisplayName("a null score level is refused rather than rendered as the text 'null'")
    void nullScoreLevelRefused() {
        List<Object> levels = new ArrayList<>();
        levels.add("fine");
        levels.add(null);
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> Question.score("i", levels));
        // The index is named, because that is what the caller has to go and fix.
        org.junit.jupiter.api.Assertions.assertTrue(refused.getMessage().contains("level 1"),
                refused.getMessage());
    }

    @Test
    @DisplayName("the question type's integer code is a model input and must not drift")
    void typeCodes() {
        assertEquals(0, Question.Type.CHOICE.code());
        assertEquals(1, Question.Type.SCORE.code());
        assertEquals(2, Question.Type.NOUL.code());
        assertEquals("choice", Question.Type.CHOICE.wireName());
    }

    @org.junit.jupiter.api.Test
    @DisplayName("blankness follows Python's whitespace set, not Java's")
    void blanknessUsesPythonsWhitespace() {
        // The reference refuses an instruction whose strip() is empty, and Python's strip treats
        // U+00A0, U+0085, U+2007 and U+202F as whitespace where Character.isWhitespace -- and so
        // String.isBlank and String.trim -- do not. An instruction of a single no-break space is
        // therefore refused by the reference, and was accepted here: the model would have been
        // shown "noul question: " and answered it anyway with a confident-looking distribution.
        int[] blanks = {0x20, 0xA0, 0x85, 0x2007, 0x202F};
        for (int cp : blanks) {
            String blank = new String(Character.toChars(cp));
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                    () -> Question.noul(blank),
                    "an instruction of U+" + Integer.toHexString(cp) + " must be refused");
        }
        // U+200B is NOT whitespace to Python either, so it stays a legitimate instruction: the
        // rule is Python's set, not "anything invisible".
        String zeroWidth = new String(Character.toChars(0x200B));
        assertEquals(zeroWidth, Question.noul(zeroWidth).instructions());
    }

    @org.junit.jupiter.api.Test
    @DisplayName("noul labels are stripped with Python's whitespace set")
    void noulLabelsUsePythonsStrip() {
        // Same root cause on the label path. trim leaves a no-break space in place, so the label
        // passes the non-empty check and renders whitespace into the model's input -- a different
        // token sequence from the reference, which refuses the label.
        String nbsp = new String(Character.toChars(0xA0));
        String narrow = new String(Character.toChars(0x202F));
        java.util.Map<String, String> blankLabel = new LinkedHashMap<>();
        blankLabel.put("false", nbsp);
        blankLabel.put("true", "yes");
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> Question.noul("Is this urgent?", null, null, blankLabel),
                "a label of one no-break space must be refused");

        java.util.Map<String, String> padded = new LinkedHashMap<>();
        padded.put("false", nbsp + "no");
        padded.put("true", "yes" + narrow);
        assertEquals(java.util.List.of("no", "yes"),
                Question.noul("Is this urgent?", null, null, padded).labels());
    }
}
