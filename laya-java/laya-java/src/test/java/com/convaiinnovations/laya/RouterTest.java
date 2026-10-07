package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.json.PythonJson;
import com.convaiinnovations.laya.lang.LanguageDetection;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link Router} against the decisions recorded from {@code laya.router}.
 *
 * <p>The model and the reason are both asserted. The reason is not commentary: it goes into API
 * responses, and it is built from two Python formats Java spells differently -- {@code repr}, and
 * a {@code %.0f} that rounds halves to even. One of the reasons interpolates the mixed segment, a
 * slice of the caller's own text, so both see arbitrary input rather than a short language code.
 *
 * <p>A wrong route is not a degraded answer, it is a confident wrong one: the English checkpoint
 * scores 0.100 on 20-option Hindi intent, against 0.050 for random, at ECE 0.855. That is the
 * reason every precedence level and every detection branch has a recorded case here.
 */
class RouterTest {

    private static Map<String, Object> fixture() {
        return Fixtures.load("router.json");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        return (List<Object>) value;
    }

    /** The state as the reference saw it, rebuilt from the tagged fixture value. */
    private static Object state(Map<String, Object> encoded) {
        String kind = (String) encoded.get("kind");
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

    /** A question set with these ids, which is all the workflow match reads. */
    private static Map<String, Question> questionsWithIds(List<Object> ids) {
        Map<String, Question> out = new LinkedHashMap<>();
        for (Object id : ids) {
            out.put((String) id, Question.noul("Is `message` about " + id + "?"));
        }
        return out;
    }

    // ------------------------------------------------------------------ the registry

    @Test
    @DisplayName("the registry and the alias table are the reference's")
    void registry() {
        Map<String, Object> recorded = fixture();
        assertEquals(recorded.get("bundle_repo"), Router.BUNDLE_REPO);

        Map<String, Object> aliases = map(recorded.get("aliases"));
        assertEquals(aliases.keySet(), Router.aliases(), "the accepted aliases differ");
        for (Map.Entry<String, Object> entry : aliases.entrySet()) {
            assertEquals(entry.getValue(),
                    Router.normaliseName(entry.getKey()).wireName(),
                    "alias " + entry.getKey());
        }

        Map<String, Object> repos = map(recorded.get("repo_strings"));
        Router bundled = Router.withDefaults();
        for (Map.Entry<String, Object> entry : repos.entrySet()) {
            Router.Checkpoint key = Router.normaliseName(entry.getKey());
            assertEquals(entry.getValue(), bundled.models().get(key).repoString(),
                    "bundled repo id for " + entry.getKey());
        }

        Map<String, Object> standalone = map(recorded.get("standalone_repo_strings"));
        Router alone = Router.builder().standaloneRepos(true).build();
        for (Map.Entry<String, Object> entry : standalone.entrySet()) {
            Router.Checkpoint key = Router.normaliseName(entry.getKey());
            assertEquals(entry.getValue(), alone.models().get(key).repoString(),
                    "standalone repo id for " + entry.getKey());
        }
    }

    @TestFactory
    @DisplayName("normaliseName answers, or refuses with the reference's message")
    List<DynamicTest> normaliseName() {
        Map<String, Object> recorded = map(fixture().get("normalise_name"));
        List<DynamicTest> tests = new ArrayList<>();
        for (Map.Entry<String, Object> entry : recorded.entrySet()) {
            String name = entry.getKey();
            Map<String, Object> expected = map(entry.getValue());
            tests.add(DynamicTest.dynamicTest("name=" + PythonJson.repr(name), () -> {
                if (expected.containsKey("ok")) {
                    assertEquals(expected.get("ok"), Router.normaliseName(name).wireName());
                } else {
                    IllegalArgumentException failure = assertThrows(
                            IllegalArgumentException.class, () -> Router.normaliseName(name));
                    // The message lists both sets and is what a caller sees in a stack trace, so
                    // it is compared whole rather than merely for being non-empty.
                    assertEquals(expected.get("error"), failure.getMessage());
                }
            }));
        }
        assertTrue(tests.size() >= 20, "only " + tests.size() + " names checked");
        return tests;
    }

    @Test
    @DisplayName("resolveModelSpec answers for a registry name and null for anything else")
    void resolveModelSpec() {
        Map<String, Object> recorded = map(fixture().get("resolve_model_spec"));
        int nulls = 0;
        for (Map.Entry<String, Object> entry : recorded.entrySet()) {
            Router.ModelSpec spec = Router.resolveModelSpec(entry.getKey());
            if (entry.getValue() == null) {
                assertNull(spec, entry.getKey() + " is not a registry name");
                nulls++;
                continue;
            }
            List<Object> pair = list(entry.getValue());
            assertTrue(spec != null, entry.getKey() + " should resolve");
            assertEquals(pair.get(0), spec.repo(), entry.getKey() + ": repo");
            assertEquals(pair.get(1), spec.subfolder(), entry.getKey() + ": subfolder");
        }
        assertTrue(nulls >= 5, "only " + nulls + " cases exercise the non-throwing null path,"
                + " which is the whole difference from normaliseName");
    }

    @TestFactory
    @DisplayName("matchTypedDecisionsWorkflow needs an exact id set")
    List<DynamicTest> matchWorkflow() {
        List<DynamicTest> tests = new ArrayList<>();
        int matched = 0;
        int refused = 0;
        for (Object entry : list(fixture().get("match_workflow"))) {
            List<Object> pair = list(entry);
            List<Object> ids = list(pair.get(0));
            Object expected = pair.get(1);
            if (expected == null) {
                refused++;
            } else {
                matched++;
            }
            tests.add(DynamicTest.dynamicTest(expected == null ? "no match: " + ids
                    : (String) expected,
                    () -> assertEquals(expected,
                            Router.matchTypedDecisionsWorkflow(questionsWithIds(ids)))));
        }
        assertTrue(matched >= 4, "all four workflows must be covered, got " + matched);
        assertTrue(refused >= 4, "only " + refused + " near-misses; a superset and a subset must"
                + " both be refused or the match is a subset test");
        return tests;
    }

    @Test
    @DisplayName("the four workflow signatures are the reference's")
    void workflowSignatures() {
        Map<String, Object> recorded = map(fixture().get("typed_decision_workflows"));
        Map<String, Set<String>> built = Router.typedDecisionWorkflows();
        assertEquals(new TreeSet<>(recorded.keySet()), new TreeSet<>(built.keySet()));
        for (Map.Entry<String, Object> entry : recorded.entrySet()) {
            Set<String> expected = new TreeSet<>();
            for (Object id : list(entry.getValue())) {
                expected.add((String) id);
            }
            assertEquals(expected, new TreeSet<>(built.get(entry.getKey())),
                    "the " + entry.getKey() + " signature");
        }
    }

    @Test
    @DisplayName("matchTypedDecisionsWorkflow tolerates null, empty, and a null question id")
    void matchWorkflowHandlesNothing() {
        assertNull(Router.matchTypedDecisionsWorkflow(null));
        assertNull(Router.matchTypedDecisionsWorkflow(new LinkedHashMap<>()));

        // A HashMap permits a null key, and `Set.of(...).containsAll` throws on one. With five
        // keys the cardinality check passes first, so the whole route would abort on input the
        // reference routes normally -- it simply fails to match and carries on to detection.
        Map<String, Question> withNullId = new java.util.HashMap<>();
        for (String id : List.of("action", "needs_review", "outcome", "risk")) {
            withNullId.put(id, Question.noul("Is this " + id + "?"));
        }
        withNullId.put(null, Question.noul("Is this urgent?"));
        assertEquals(5, withNullId.size(), "the cardinality must match a real workflow");
        assertNull(Router.matchTypedDecisionsWorkflow(withNullId),
                "a null id matches no workflow, and must not throw");

        // and the route completes rather than aborting
        Router.RouteDecision decided = Router.builder().autoTaskDetection(true).build()
                .route("I cannot log in to my account at all today", withNullId);
        assertEquals(Router.Checkpoint.ENGLISH, decided.model());
        assertNull(decided.workflow());
    }

    @TestFactory
    @DisplayName("englishFromCode answers one bit, or abstains")
    List<DynamicTest> englishFromCode() {
        List<DynamicTest> tests = new ArrayList<>();
        int abstained = 0;
        int englishes = 0;
        int others = 0;
        for (Object entry : list(fixture().get("english_from_code"))) {
            List<Object> pair = list(entry);
            Object code = pair.get(0);
            Object expected = pair.get(1);
            if (expected == null) {
                abstained++;
            } else if (Boolean.TRUE.equals(expected)) {
                englishes++;
            } else {
                others++;
            }
            // A blank display name aborts the whole factory, and an aborted container is not a
            // failed test -- the suite stayed green while these 38 cases never ran. Hence the
            // quoted, never-blank label.
            tests.add(DynamicTest.dynamicTest("code=" + PythonJson.repr(
                    code == null ? null : String.valueOf(code)),
                    () -> assertEquals(expected, Router.englishFromCode(code))));
        }
        assertTrue(abstained >= 8, "only " + abstained + " codes abstain, and abstaining is the"
                + " behaviour that keeps LANG=C from pinning every request to one checkpoint");
        assertTrue(englishes >= 5 && others >= 5,
                englishes + " English and " + others + " non-English codes");
        return tests;
    }

    @Test
    @DisplayName("the schema signature matches the reference for every expressible schema")
    void questionSchema() {
        int compared = 0;
        for (Object entry : list(fixture().get("question_schema"))) {
            List<Object> pair = list(entry);
            Map<String, Object> specs = map(pair.get(0));
            String expected = (String) pair.get(1);
            Map<String, Question> questions = new LinkedHashMap<>();
            for (Map.Entry<String, Object> spec : specs.entrySet()) {
                questions.put(spec.getKey(), questionOf(map(spec.getValue())));
            }
            assertEquals(expected, Router.questionSchema(questions),
                    "the signature for " + expected);
            compared++;
        }
        assertTrue(compared >= 6, "only " + compared + " schemas compared");
    }

    @Test
    @DisplayName("option ORDER changes the signature, which is why it is order-sensitive")
    void schemaOrderIsSignificant() {
        // The property that matters, independent of the recorded strings: a choice between a and b
        // is a different question from a choice between b and a, because the answer's
        // probabilities are positional. Grouping them would hand one state the other's answer.
        Map<String, Object> forward = new LinkedHashMap<>();
        forward.put("a", "1");
        forward.put("b", "2");
        Map<String, Object> backward = new LinkedHashMap<>();
        backward.put("b", "2");
        backward.put("a", "1");
        Map<String, Question> first = new LinkedHashMap<>();
        first.put("q", Question.choice("x", forward));
        Map<String, Question> second = new LinkedHashMap<>();
        second.put("q", Question.choice("x", backward));
        assertNotEquals(Router.questionSchema(first), Router.questionSchema(second),
                "two option orders must not share a forward pass");

        // and the question ids' own order counts too
        Map<String, Question> ab = new LinkedHashMap<>();
        ab.put("a", Question.noul("x"));
        ab.put("b", Question.noul("y"));
        Map<String, Question> ba = new LinkedHashMap<>();
        ba.put("b", Question.noul("y"));
        ba.put("a", Question.noul("x"));
        assertNotEquals(Router.questionSchema(ab), Router.questionSchema(ba));

        // identical questions do share one, which is the point of the signature
        assertEquals(Router.questionSchema(first), Router.questionSchema(
                new LinkedHashMap<>(Map.of("q", Question.choice("x", forward)))));
        assertEquals("{}", Router.questionSchema(null));
    }

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

    // ------------------------------------------------------------------ routing

    @TestFactory
    @DisplayName("every recorded route matches, model and reason")
    List<DynamicTest> routes() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : list(fixture().get("routes"))) {
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> assertRoute(name, row)));
        }
        return tests;
    }

    private void assertRoute(String name, Map<String, Object> row) {
        Map<String, Object> options = map(row.get("options"));
        Object state = state(map(row.get("state")));
        Map<String, Object> expected = map(row.get("decision"));

        Router.Builder builder = Router.builder();
        if (options.get("default") != null) {
            builder.defaultCheckpoint(Router.normaliseName((String) options.get("default")));
        }
        if (Boolean.TRUE.equals(options.get("auto_task_detection"))) {
            builder.autoTaskDetection(true);
        }
        if (options.get("router_lang_guess") != null) {
            builder.langGuess(Router.LanguageHint.of((String) options.get("router_lang_guess")));
        }
        Router router = builder.build();

        Router.RouteOptions call = Router.RouteOptions.none();
        if (options.get("model") != null) {
            call = call.model((String) options.get("model"));
        }
        if (options.get("task") != null) {
            call = call.task((String) options.get("task"));
        }
        if (options.get("lang") != null) {
            call = call.lang((String) options.get("lang"));
        }
        if (options.get("lang_guess") != null) {
            call = call.langGuess(Router.LanguageHint.of((String) options.get("lang_guess")));
        }

        Map<String, Question> questions = null;
        if (options.get("questions") != null) {
            questions = new LinkedHashMap<>();
            for (Map.Entry<String, Object> q : map(options.get("questions")).entrySet()) {
                questions.put(q.getKey(), questionOf(map(q.getValue())));
            }
        }

        Router.RouteDecision got = router.route(state, questions, call);
        assertEquals(expected.get("model"), got.model().wireName(), name + ": model");
        assertEquals(expected.get("repo"), got.repo(), name + ": repo");
        assertEquals(expected.get("reason"), got.reason(), name + ": reason");
        assertEquals(expected.get("workflow"), got.workflow(), name + ": workflow");

        Object recordedDetection = expected.get("detection");
        if (recordedDetection == null) {
            assertNull(got.detection(), name + ": an explicit argument decided, so there is no"
                    + " detection to report");
            return;
        }
        Map<String, Object> detection = map(recordedDetection);
        LanguageDetection.Analysis actual = got.detection();
        assertTrue(actual != null, name + ": detection was expected");
        assertEquals(detection.get("script"), actual.script(), name + ": detection script");
        assertEquals(detection.get("language"), actual.language(), name + ": detection language");
        assertEquals(detection.get("is_english"), actual.english(), name + ": detection isEnglish");
        assertEquals(detection.get("language_undecided"), actual.languageUndecided(),
                name + ": detection languageUndecided");
        assertEquals(((Number) detection.get("non_latin_fraction")).doubleValue(),
                actual.nonLatinFraction(), name + ": detection nonLatinFraction");
        assertEquals(((Number) detection.get("diacritic_rate")).doubleValue(),
                actual.diacriticRate(), name + ": detection diacriticRate");
        assertEquals(detection.get("mixed_segment"), actual.mixedSegment(),
                name + ": detection mixedSegment");
    }

    @Test
    @DisplayName("the route corpus covers every precedence level and detection branch")
    void routeCoverage() {
        // A route suite that silently stopped exercising a branch would still be green, so the
        // corpus itself is asserted. Each count below is why a group of cases exists.
        int explicitModel = 0;
        int explicitTask = 0;
        int workflow = 0;
        int explicitLang = 0;
        int hint = 0;
        int detected = 0;
        int halfwayPercent = 0;
        int reprEscaped = 0;
        int doubleQuoted = 0;
        for (Object entry : list(fixture().get("routes"))) {
            Map<String, Object> decision = map(map(entry).get("decision"));
            String reason = (String) decision.get("reason");
            if (reason.startsWith("explicit model=")) {
                explicitModel++;
            } else if (reason.startsWith("explicit task=")) {
                explicitTask++;
            } else if (reason.startsWith("question ids match")) {
                workflow++;
            } else if (reason.startsWith("explicit lang=")) {
                explicitLang++;
            } else if (reason.contains("the caller identified this as")) {
                hint++;
            } else {
                detected++;
            }
            if (reason.contains("12% ")) {
                halfwayPercent++;
            }
            if (reason.contains("\\x") || reason.contains("\\u")) {
                reprEscaped++;
            }
            if (reason.contains("(\"")) {
                doubleQuoted++;
            }
        }
        assertTrue(explicitModel >= 4, "only " + explicitModel + " explicit-model routes");
        assertTrue(explicitTask >= 3, "only " + explicitTask + " explicit-task routes");
        assertTrue(workflow >= 4, "all four workflows must route, got " + workflow);
        assertTrue(explicitLang >= 4, "only " + explicitLang + " explicit-lang routes");
        assertTrue(hint >= 4, "only " + hint + " hint routes");
        assertTrue(detected >= 12, "only " + detected + " routes reach detection");
        // The two reasons that carry a percentage, at a share where half-even and half-up differ.
        // Without one of these the router could misreport a percentage shown to a user and every
        // other case would still pass -- which is exactly what happened before they were added.
        assertTrue(halfwayPercent >= 2, "only " + halfwayPercent + " routes report a percentage"
                + " at a halfway share, so the rounding mode is not gated through the router");
        // And the reason that interpolates the caller's own text, with something to escape in it.
        assertTrue(reprEscaped >= 1, "no route's reason escapes a non-printable character, so"
                + " repr's escape rule is not gated through the router");
        assertTrue(doubleQuoted >= 1, "no route's reason holds a segment with a single quote in"
                + " it, so repr's quote selection is not gated through the router");
    }

    @Test
    @DisplayName("an explicit model beats a detected workflow and an explicit language")
    void precedenceIsStrict() {
        Map<String, Question> workflow = questionsWithIds(
                List.of("action", "category", "churn_risk", "needs_human", "urgency"));
        Router router = Router.builder().autoTaskDetection(true)
                .langGuess(Router.LanguageHint.of("pt")).build();
        Router.RouteDecision pinned = router.route("plain english here", workflow,
                Router.RouteOptions.none().model("multi").task("typed").lang("fr"));
        assertEquals(Router.Checkpoint.MULTILINGUAL, pinned.model());
        assertEquals("explicit model='multi'", pinned.reason());
        assertNull(pinned.workflow(), "the model branch returns before the workflow is looked up");

        Router.RouteDecision byTask = router.route("plain english here", workflow,
                Router.RouteOptions.none().task("typed").lang("fr"));
        assertEquals(Router.Checkpoint.TYPED_DECISIONS, byTask.model());
        assertEquals("explicit task='typed'", byTask.reason());
    }

    @Test
    @DisplayName("the task branch's typed_decisions rewrite is redundant, and provably so")
    void taskSpellingRewriteIsRedundant() {
        // The reference rewrites a task spelled `typed_decisions` to `typed-decisions` before
        // resolving it. That rewrite cannot change an answer, and a mutant that removed it
        // survived the whole suite -- correctly. The proof is short: the rewrite fires only when
        // the task lowercased, with hyphens turned into underscores, equals "typed_decisions", so
        // the lowercased task is one of exactly two strings; and name resolution lowercases
        // first and accepts both of them, one as the canonical name and one as an alias.
        //
        // The faithful branch is kept -- the reference's control flow is the contract -- and this
        // is what says the surviving mutant is a property of the alias table rather than a gap in
        // the corpus. It fails if either spelling is ever dropped from that table.
        for (String lowered : new String[] {"typed-decisions", "typed_decisions"}) {
            assertEquals("typed_decisions", lowered.replace('-', '_'),
                    lowered + " is one of the two forms the rewrite can see");
            assertEquals(Router.Checkpoint.TYPED_DECISIONS, Router.normaliseName(lowered),
                    lowered + " must resolve without the rewrite, or the rewrite is load-bearing");
        }
        // and the spellings a caller actually types all land on the same checkpoint
        Router router = Router.withDefaults();
        for (String spelling : new String[] {"typed_decisions", "typed-decisions",
                                             "TYPED_DECISIONS", "Typed-Decisions",
                                             "  typed_decisions  "}) {
            Router.RouteDecision decided = router.route("plain english here", null,
                    Router.RouteOptions.none().task(spelling));
            assertEquals(Router.Checkpoint.TYPED_DECISIONS, decided.model(), "task=" + spelling);
            // the reason reports what the caller wrote, untouched
            assertEquals("explicit task=" + PythonJson.repr(spelling), decided.reason());
        }
    }

    @Test
    @DisplayName("a hint that abstains falls through to detection rather than deciding")
    void abstainingHintFallsThrough() {
        // The behaviour that lets a language-identification model say "I do not know" without
        // pinning the request. A hint returning a code that names no language does the same, which
        // is what LANG=C in a minimal container gives.
        Router router = Router.builder().langGuess(state -> null).build();
        Router.RouteDecision decided = router.route(
                "मैं अपने खाते");
        assertEquals(Router.Checkpoint.MULTILINGUAL, decided.model());
        assertTrue(decided.reason().startsWith("non-Latin script"), decided.reason());
        assertTrue(decided.detection() != null, "detection ran, so it is reported");

        Router agnostic = Router.builder().langGuess(Router.LanguageHint.of("C.UTF-8")).build();
        assertTrue(agnostic.route("plain english words here now").reason()
                .startsWith("Latin script"), "a code naming no language must not decide");
    }

    @Test
    @DisplayName("a hint sees the state, which is what makes a callable one useful")
    void hintReceivesTheState() {
        List<Object> seen = new ArrayList<>();
        Router router = Router.builder().langGuess(state -> {
            seen.add(state);
            return state instanceof String && ((String) state).startsWith("por") ? "pt" : "en";
        }).build();
        assertEquals(Router.Checkpoint.MULTILINGUAL, router.route("portuguese-ish").model());
        assertEquals(Router.Checkpoint.ENGLISH, router.route("english-ish").model());
        assertEquals(List.of("portuguese-ish", "english-ish"), seen,
                "the hint must be given the state it is hinting about");
    }

    @Test
    @DisplayName("routing reads no disk and runs no model")
    void routingIsPure() {
        // Worth pinning: this is what makes it safe to call on every request, and to test without
        // a checkpoint. A router configured to point at paths that do not exist still routes.
        Router router = Router.builder()
                .model(Router.Checkpoint.ENGLISH, "/nonexistent/english", null)
                .model(Router.Checkpoint.MULTILINGUAL, "/nonexistent/multi", "sub")
                .build();
        Router.RouteDecision decided = router.route("I cannot log in to my account at all today");
        assertEquals(Router.Checkpoint.ENGLISH, decided.model());
        assertEquals("/nonexistent/english", decided.repo());
        assertEquals("English Latin text", decided.reason());
        assertEquals("/nonexistent/multi/sub",
                router.route("계정에 로그인할 수 없어요")
                        .repo());
    }

    @Test
    @DisplayName("the default checkpoint is the only knob for unidentified Latin text")
    void defaultCheckpointDecidesUndecided() {
        String undecided = "Quero cancelar";
        assertEquals(Router.Checkpoint.ENGLISH, Router.withDefaults().route(undecided).model());
        assertEquals(Router.Checkpoint.MULTILINGUAL,
                Router.builder().defaultCheckpoint(Router.Checkpoint.MULTILINGUAL).build()
                        .route(undecided).model());
        assertEquals(Router.Checkpoint.MULTILINGUAL,
                Router.builder().defaultCheckpoint(Router.Checkpoint.MULTILINGUAL).build()
                        .route("12345 !!!").model());
    }
}
