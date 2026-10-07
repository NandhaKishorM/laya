package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link Presets} against the question sets recorded from {@code laya.presets}.
 *
 * <p>Word for word, because one changed word is a different question put to the model and
 * therefore a different answer. The comparison is made on what the model is actually shown -- the
 * instruction, the rendered options in order, the labels and the legend -- rather than on the
 * Java object, so a difference that survives rendering is the only kind that can pass.
 */
class PresetsTest {

    private static Map<String, Object> fixture() {
        return Fixtures.load("presets.json");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        return (List<Object>) value;
    }

    /** The recorded spec rebuilt as a {@link Question}, so the two can be compared as rendered. */
    private static Question questionOf(Map<String, Object> spec) {
        String type = (String) spec.get("type");
        String instructions = (String) spec.get("instructions");
        Object criteria = spec.get("criteria");
        if ("choice".equals(type)) {
            return Question.choice(instructions, map(criteria));
        }
        if ("score".equals(type)) {
            return Question.score(instructions, list(criteria));
        }
        Map<String, Object> sides = criteria instanceof Map ? map(criteria) : Map.of();
        return Question.noul(instructions, sides.get("false"), sides.get("true"), null);
    }

    private static void assertSameQuestion(String where, Question expected, Question actual) {
        assertEquals(expected.type(), actual.type(), where + ": type");
        assertEquals(expected.instructions(), actual.instructions(), where + ": instructions");
        // Rendered options carry both the criterion text and its ORDER, which for a choice is
        // part of the question: the answer's probabilities are positional.
        assertEquals(expected.renderOptions(), actual.renderOptions(), where + ": rendered options");
        assertEquals(expected.labels(), actual.labels(), where + ": labels");
        // Only a score question has a legend; asking a choice or noul for one is an error by
        // design, because there is no level to describe.
        if (expected.type() == Question.Type.SCORE) {
            assertEquals(expected.legend(), actual.legend(), where + ": legend");
            assertEquals(new ArrayList<>(expected.legend().keySet()),
                    new ArrayList<>(actual.legend().keySet()), where + ": legend order");
        }
    }

    private static void assertSameSet(String name, Map<String, Object> recorded,
            Map<String, Question> built) {
        assertEquals(new ArrayList<>(recorded.keySet()), new ArrayList<>(built.keySet()),
                name + ": the questions, or their order, differ from the reference");
        for (Map.Entry<String, Object> entry : recorded.entrySet()) {
            assertSameQuestion(name + "." + entry.getKey(),
                    questionOf(map(entry.getValue())), built.get(entry.getKey()));
        }
    }

    @TestFactory
    @DisplayName("every preset matches the reference, question for question and word for word")
    List<DynamicTest> presetsMatch() {
        Map<String, Object> recorded = fixture();
        Map<String, Map<String, Question>> built = Presets.all();
        List<DynamicTest> tests = new ArrayList<>();
        for (Map.Entry<String, Map<String, Question>> entry : built.entrySet()) {
            String name = entry.getKey();
            Map<String, Object> expected = map(recorded.get(name));
            tests.add(DynamicTest.dynamicTest(name,
                    () -> assertSameSet(name, expected, entry.getValue())));
        }
        return tests;
    }

    @Test
    @DisplayName("the port carries every preset the reference has, and no invented one")
    void everyPresetIsCovered() {
        List<String> recorded = new ArrayList<>();
        for (String key : fixture().keySet()) {
            if (key.endsWith("_questions")) {
                recorded.add(key);
            }
        }
        List<String> built = new ArrayList<>(Presets.all().keySet());
        recorded.sort(null);
        List<String> sortedBuilt = new ArrayList<>(built);
        sortedBuilt.sort(null);
        assertEquals(recorded, sortedBuilt,
                "the reference's presets and the port's do not line up; a preset was added"
                + " upstream or invented here");
        assertTrue(built.size() >= 5, "only " + built.size() + " presets");
    }

    @Test
    @DisplayName("stateField names the one field each preset reads")
    void stateFieldPerPreset() {
        Map<String, Object> recorded = map(fixture().get("state_field"));
        Map<String, Map<String, Question>> built = Presets.all();
        for (Map.Entry<String, Object> entry : recorded.entrySet()) {
            Map<String, Question> questions = built.get(entry.getKey());
            assertTrue(questions != null, "no port preset named " + entry.getKey());
            assertEquals(entry.getValue(), Presets.stateField(questions),
                    entry.getKey() + ": the field named in backticks");
        }
        assertTrue(recorded.size() >= 5, "only " + recorded.size() + " presets checked");
    }

    @TestFactory
    @DisplayName("stateField answers the hostile cases the way the reference does")
    List<DynamicTest> stateFieldHostile() {
        Map<String, Object> expected = map(fixture().get("state_field_hostile"));
        Map<String, Object> specs = map(fixture().get("hostile_specs"));
        List<DynamicTest> tests = new ArrayList<>();
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            String name = entry.getKey();
            Map<String, Object> spec = map(specs.get(name));
            // Two of the recorded cases give an instruction of null or leave it out entirely.
            // They are not expressible here: Question refuses a blank instruction at
            // construction, which is strictly earlier than the reference, and that refusal is
            // asserted below instead of being skipped silently.
            boolean instructionMissing = false;
            for (Object value : spec.values()) {
                Map<String, Object> question = map(value);
                if (question.get("instructions") == null) {
                    instructionMissing = true;
                }
            }
            if (instructionMissing) {
                tests.add(DynamicTest.dynamicTest(name + " (refused at construction)",
                        () -> assertThrows(IllegalArgumentException.class,
                                () -> questionOf(map(spec.values().iterator().next())),
                                "a question with no instruction must be refused, not accepted"
                                + " and then reported as naming no field")));
                continue;
            }
            tests.add(DynamicTest.dynamicTest(name, () -> {
                Map<String, Question> questions = new LinkedHashMap<>();
                for (Map.Entry<String, Object> q : spec.entrySet()) {
                    questions.put(q.getKey(), questionOf(map(q.getValue())));
                }
                assertEquals(entry.getValue(), Presets.stateField(questions), name);
            }));
        }
        return tests;
    }

    @Test
    @DisplayName("custom email categories replace the defaults, in the caller's order")
    void emailCategoriesReplaceTheDefaults() {
        Map<String, String> categories = new LinkedHashMap<>();
        categories.put("ops", "incidents and deploys");
        categories.put("legal", "contracts and compliance");
        assertSameSet("email_questions_custom", map(fixture().get("email_questions_custom")),
                Presets.email(categories));
    }

    @Test
    @DisplayName("an EMPTY category map asks for the defaults, as the reference's truth test does")
    void emptyEmailCategoriesMeanTheDefaults() {
        // Reads like a way to ask for no categories, and is not: the reference tests the argument
        // for truthiness and an empty dict is falsy, so it falls through to the defaults. Pinned
        // because a port that checked only for null would build a choice question with no options
        // and throw instead.
        assertSameSet("email_questions_empty_categories",
                map(fixture().get("email_questions_empty_categories")),
                Presets.email(new LinkedHashMap<>()));
        assertEquals(Presets.email().get("category").renderOptions(),
                Presets.email(new LinkedHashMap<>()).get("category").renderOptions());
    }

    @Test
    @DisplayName("stateField reads the instruction only, never a criterion")
    void criteriaAreNotFields() {
        // Criteria quote user-facing labels, so a backtick in one is punctuation and not a field
        // name. A port that scanned the whole rendered question would find `message` here and
        // report a field this question does not read.
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("q", Question.choice("Pick one.",
                new LinkedHashMap<>(Map.of("x", "look at `message`"))));
        assertNull(Presets.stateField(questions));
    }

    @Test
    @DisplayName("the backtick scan resumes after the closing tick, as findall does")
    void backtickScanIsNonOverlapping() {
        // In "`a`b`c`" the fields are a and c. A scanner that resumed inside its own match would
        // also offer b, which would turn a one-field preset into a three-field one and make
        // stateField answer null.
        Map<String, Question> two = new LinkedHashMap<>();
        two.put("q", Question.noul("Read `a`b`c` please"));
        assertNull(Presets.stateField(two), "two fields named, so no single field");
        Map<String, Question> one = new LinkedHashMap<>();
        one.put("q", Question.noul("Read `a`b`a` please"));
        assertEquals("a", Presets.stateField(one), "the same field twice is still one field");
        Map<String, Question> doubled = new LinkedHashMap<>();
        doubled.put("q", Question.noul("Read ``a`` please"));
        assertEquals("a", Presets.stateField(doubled), "a doubled tick still names the field");
        Map<String, Question> none = new LinkedHashMap<>();
        none.put("q", Question.noul("Read `a-b` and `c d` please"));
        assertNull(Presets.stateField(none), "neither is a word, so neither is a field");
    }

    @Test
    @DisplayName("null and an empty question set name no field rather than throwing")
    void stateFieldHandlesNothing() {
        assertNull(Presets.stateField(null));
        assertNull(Presets.stateField(new LinkedHashMap<>()));
    }

    @Test
    @DisplayName("a null question is refused, not quietly skipped")
    void stateFieldRefusesANullQuestion() {
        // Skipping it would answer with the field the REMAINING questions name, which looks like
        // a correct answer and is not. The reference raises here, and a caller who dropped a
        // question by setting it to null should hear about it rather than get a plausible field
        // name back.
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("intent", Question.noul("Does `message` ask for a refund?"));
        questions.put("dropped", null);
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> Presets.stateField(questions));
        assertTrue(failure.getMessage().contains("dropped"),
                "the message must name the offending question: " + failure.getMessage());
    }

    @Test
    @DisplayName("each call returns a fresh mutable map, so a caller can edit one safely")
    void presetsAreFreshAndMutable() {
        // A caller is expected to drop a question it does not want. Returning a shared or
        // immutable map would make that either impossible or visible to the next caller.
        Map<String, Question> first = Presets.triage();
        Map<String, Question> second = Presets.triage();
        assertNotSame(first, second);
        first.remove("churn_risk");
        assertTrue(second.containsKey("churn_risk"),
                "editing one caller's preset changed the next caller's");
        assertEquals(5, second.size());
        second.put("extra", Question.noul("Is this a test?"));
        assertEquals(5, Presets.triage().size(), "an added question leaked into the preset");
    }

    @Test
    @DisplayName("every preset is accepted by the question renderer")
    void everyPresetRenders() {
        // The presets are data, and data can be wrong in a way that only shows when the model is
        // asked: a score level that is null, a choice with no options. Rendering every one of them
        // here means a typo in this file fails a test rather than a prediction.
        for (Map.Entry<String, Map<String, Question>> preset : Presets.all().entrySet()) {
            for (Map.Entry<String, Question> entry : preset.getValue().entrySet()) {
                String where = preset.getKey() + "." + entry.getKey();
                Question question = entry.getValue();
                List<String> options = question.renderOptions();
                assertTrue(options.size() >= 2, where + " offers " + options.size() + " options");
                for (String option : options) {
                    assertTrue(option != null && !option.isBlank(),
                            where + " renders a blank option");
                }
                assertTrue(!question.instructions().isBlank(), where + " has no instruction");
                assertEquals(options.size(), question.labels().size(),
                        where + ": one label per option");
            }
        }
    }
}
