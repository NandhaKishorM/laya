package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/**
 * End to end on a real graph: the Java runtime must answer what Python's {@code ONNXAgent} answered.
 *
 * <p>The only check that covers the whole stack at once -- tokenizer, sequence budgets, collation,
 * graph I/O, decode, rounding and usage -- which the fixture families cannot do individually.
 *
 * <p><b>Tolerance.</b> Structure is compared exactly: the answer type, the chosen label, the
 * question ids and their order, the legend, and every integer in {@link Usage}. Probabilities are
 * compared within {@value #PROBABILITY_TOLERANCE}, two units in the last reported decimal, because
 * the encoder's own arithmetic is platform-sensitive and the golden is recorded on one machine. A
 * chosen label is only asserted when the top two probabilities are clear of that tolerance; when
 * they are not, the argmax is genuinely a coin toss and asserting it would be asserting noise.
 */
final class PredictParityTest {

    /** Two units in the last decimal an answer reports. */
    private static final double PROBABILITY_TOLERANCE = 2e-4;

    /**
     * One agent for the whole class, closed when the class is done.
     *
     * <p>Opening it per factory loaded the 1.2 GB graph twice and doubled the parity lane's
     * wall-clock for no coverage. {@code @AutoClose} ties it to the class lifecycle rather than
     * leaking the native session for the rest of the JVM.
     */
    private static Agent shared;

    @AfterAll
    static void closeAgent() {
        if (shared != null) {
            shared.close();
            shared = null;
        }
    }

    @TestFactory
    @DisplayName("single-state predictions match Python")
    @SuppressWarnings("unchecked")
    List<DynamicTest> singleMatches() throws IOException {
        Map<String, Object> golden = requireGolden();
        Map<String, Object> singles = (Map<String, Object>) golden.get("single");
        List<DynamicTest> tests = new ArrayList<>();
        Agent agent = agent(golden);
        for (Map.Entry<String, Object> entry : singles.entrySet()) {
            Map<String, Object> c = (Map<String, Object>) entry.getValue();
            tests.add(DynamicTest.dynamicTest(entry.getKey(), () -> {
                Prediction got = agent.predict(c.get("state"),
                        questionsOf((Map<String, Object>) c.get("questions")),
                        (String) c.get("lang"));
                assertEquals(c.get("model"), got.model());
                assertAnswers((Map<String, Object>) c.get("answers"), got.answers());
                assertUsage((Map<String, Object>) c.get("usage"), got.usage());
            }));
        }
        assertTrue(tests.size() >= 5, "expected the golden's cases, got " + tests.size());
        return tests;
    }

    @TestFactory
    @DisplayName("batched predictions match Python at every batch size and sort order")
    @SuppressWarnings("unchecked")
    List<DynamicTest> batchMatches() throws IOException {
        Map<String, Object> golden = requireGolden();
        Agent agent = agent(golden);
        List<DynamicTest> tests = new ArrayList<>();
        for (Object configObject : (List<Object>) golden.get("batch")) {
            Map<String, Object> config = (Map<String, Object>) configObject;
            Object size = config.get("batch_size");
            boolean sort = Boolean.TRUE.equals(config.get("sort_by_length"));
            tests.add(DynamicTest.dynamicTest(
                    "batch_size=" + (size == null ? "all" : size) + " sort=" + sort, () -> {
                        List<Prediction> got = agent.predictBatch(
                                (List<Object>) config.get("states"),
                                questionsOf((Map<String, Object>) config.get("questions")),
                                null, size == null ? 0 : ((Number) size).intValue(), sort);
                        List<Object> want = (List<Object>) config.get("results");
                        assertEquals(want.size(), got.size(), "one prediction per state");
                        for (int i = 0; i < want.size(); i++) {
                            Map<String, Object> w = (Map<String, Object>) want.get(i);
                            // Order matters: grouping is an implementation choice and the results
                            // must come back in the caller's order whatever the grouping was.
                            assertAnswers((Map<String, Object>) w.get("answers"),
                                    got.get(i).answers());
                            assertUsage((Map<String, Object>) w.get("usage"), got.get(i).usage());
                        }
                    }));
        }
        return tests;
    }

    @SuppressWarnings("unchecked")
    private static void assertAnswers(Map<String, Object> want, Map<String, Answer> got) {
        assertIterableEquals(want.keySet(), got.keySet(), "question ids and their order");
        for (Map.Entry<String, Object> entry : want.entrySet()) {
            Map<String, Object> w = (Map<String, Object>) entry.getValue();
            Answer g = got.get(entry.getKey());
            String at = entry.getKey();
            assertEquals(w.get("type"), g.type(), at + ".type");
            assertEquals(number(w.get("answer_confidence")), g.answerConfidence(),
                    PROBABILITY_TOLERANCE, at + ".answer_confidence");
            assertEquals(number(w.get("confidence")), g.confidence(),
                    PROBABILITY_TOLERANCE, at + ".confidence");
            assertEquals(number(((Map<String, Object>) w.get("action")).get("act_probability")),
                    g.actProbability(), PROBABILITY_TOLERANCE, at + ".act_probability");
            if (g instanceof Answer.Choice choice) {
                Map<String, Object> probabilities = (Map<String, Object>) w.get("probabilities");
                assertProbabilities(at, probabilities, choice.probabilities());
                if (decisive(probabilities)) {
                    assertEquals(w.get("choice"), choice.choice(), at + ".choice");
                }
            } else if (g instanceof Answer.Score score) {
                assertEquals(number(w.get("score")), score.score(), PROBABILITY_TOLERANCE * 4,
                        at + ".score");
                assertProbabilities(at, (Map<String, Object>) w.get("probabilities"),
                        score.probabilities());
                Map<String, Object> legend = (Map<String, Object>) w.get("legend");
                assertIterableEquals(legend.keySet(), score.legend().keySet(), at + ".legend order");
                for (Map.Entry<String, Object> level : legend.entrySet()) {
                    assertEquals(level.getValue(), score.legend().get(level.getKey()));
                }
            } else if (g instanceof Answer.Noul noul) {
                assertEquals(number(w.get("noul")), noul.noul(), PROBABILITY_TOLERANCE, at + ".noul");
            }
        }
    }

    /** Whether the top two probabilities are far enough apart for the argmax to be meaningful. */
    private static boolean decisive(Map<String, Object> probabilities) {
        double best = -1.0;
        double second = -1.0;
        for (Object value : probabilities.values()) {
            double p = number(value);
            if (p > best) {
                second = best;
                best = p;
            } else if (p > second) {
                second = p;
            }
        }
        return best - second > PROBABILITY_TOLERANCE * 5;
    }

    private static void assertProbabilities(String at, Map<String, Object> want,
                                            Map<String, Double> got) {
        assertIterableEquals(want.keySet(), got.keySet(), at + ".probability key order");
        for (Map.Entry<String, Object> entry : want.entrySet()) {
            assertEquals(number(entry.getValue()), got.get(entry.getKey()),
                    PROBABILITY_TOLERANCE, at + ".p[" + entry.getKey() + "]");
        }
    }

    @SuppressWarnings("unchecked")
    private static void assertUsage(Map<String, Object> want, Usage got) {
        // Exact: these are counts, and a count that drifts is a bug, not float noise.
        assertEquals(((Number) want.get("input_tokens")).intValue(), got.inputTokens(),
                "usage.input_tokens");
        assertEquals(((Number) want.get("output_tokens")).intValue(), got.outputTokens());
        assertEquals(((Number) want.get("state_tokens")).intValue(), got.stateTokens(),
                "usage.state_tokens");
        assertEquals(((Number) want.get("state_tokens_dropped")).intValue(),
                got.stateTokensDropped(), "usage.state_tokens_dropped");
        assertEquals(want.get("truncated"), got.truncated(), "usage.truncated");
        assertIterableEquals((List<Object>) want.get("truncated_questions"),
                got.truncatedQuestions(), "usage.truncated_questions");
        Map<String, Object> options = (Map<String, Object>) want.get("options");
        if (options == null) {
            assertTrue(got.collapsedOptions().isEmpty(),
                    "no collapse expected, got " + got.collapsedOptions().keySet());
            return;
        }
        assertEquals(options.keySet(), got.collapsedOptions().keySet(), "usage.options");
        for (Map.Entry<String, Object> entry : options.entrySet()) {
            Map<String, Object> w = (Map<String, Object>) entry.getValue();
            Usage.CollapsedOptions g = got.collapsedOptions().get(entry.getKey());
            assertEquals(((Number) w.get("total")).intValue(), g.total());
            assertEquals(((Number) w.get("distinct")).intValue(), g.distinct());
        }
    }

    private static double number(Object value) {
        return ((Number) value).doubleValue();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Question> questionsOf(Map<String, Object> defs) {
        Map<String, Question> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : defs.entrySet()) {
            Map<String, Object> q = (Map<String, Object>) entry.getValue();
            String type = (String) q.get("type");
            String instructions = String.valueOf(q.get("instructions"));
            Object criteria = q.get("criteria");
            if ("choice".equals(type)) {
                out.put(entry.getKey(), Question.choice(instructions, (Map<String, Object>) criteria));
            } else if ("score".equals(type)) {
                out.put(entry.getKey(), Question.score(instructions, (List<Object>) criteria));
            } else {
                Map<String, Object> sides = criteria instanceof Map
                        ? (Map<String, Object>) criteria : Map.of();
                out.put(entry.getKey(), Question.noul(instructions, sides.get("false"),
                        sides.get("true"), null));
            }
        }
        return out;
    }

    private static Map<String, Object> requireGolden() {
        Map<String, Object> golden = Fixtures.load("predict.json");
        Assumptions.assumeFalse(golden.containsKey("skipped"),
                "predict.json was recorded without a graph: set " + Fixtures.GRAPH_ENV
                + " and re-run scripts/gen_fixtures.py");
        return golden;
    }

    /** The shared agent, opened on first use so the assumptions still skip cleanly. */
    private static synchronized Agent agent(Map<String, Object> golden) throws IOException {
        if (shared == null) {
            shared = openAgent(golden);
        }
        return shared;
    }

    private static Agent openAgent(Map<String, Object> golden) throws IOException {
        String checkpoint = (String) golden.getOrDefault("checkpoint", "multilingual");
        Path model = Fixtures.checkpoint(checkpoint);
        Assumptions.assumeTrue(model != null, Fixtures.missingCheckpoint(checkpoint));
        String graph = System.getenv(Fixtures.GRAPH_ENV);
        Assumptions.assumeTrue(graph != null && !graph.isBlank(),
                "set " + Fixtures.GRAPH_ENV + " to the exported laya.onnx this golden was recorded from");
        Path graphPath = Paths.get(graph);
        Path graphDirectory = Files.isDirectory(graphPath) ? graphPath : graphPath.getParent();
        Assumptions.assumeTrue(graphDirectory != null && Files.isDirectory(graphDirectory),
                Fixtures.GRAPH_ENV + " does not name a graph: " + graph);
        return Agent.open(model, graphDirectory);
    }
}
