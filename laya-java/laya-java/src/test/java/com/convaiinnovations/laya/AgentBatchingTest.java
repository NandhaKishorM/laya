package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Batching, row offsets, usage accounting and result ordering, over a synthetic checkpoint.
 *
 * <p>These are the parts most likely to be wrong and least related to inference, so they are tested
 * without a model. The real-model end-to-end run covers the same ground against Python, but only
 * where a 1.2 GB graph is available.
 */
final class AgentBatchingTest {

    private static Map<String, Question> twoQuestions() {
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("intent", Question.choice("What next?",
                TinyCheckpoint.ordered("refund", "money back", "escalate", "a human")));
        questions.put("urgent", Question.noul("Needs a human."));
        return questions;
    }

    @Test
    @DisplayName("every batch size and sort order gives the same answers, in the caller's order")
    void batchingDoesNotChangeAnswersOrOrder(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        List<Object> states = List.of("alpha", "a much longer state than the others here",
                "beta", "", "gamma gamma gamma", "d");
        Map<String, Question> questions = twoQuestions();

        List<String> reference = null;
        for (int batchSize : new int[] {0, 1, 2, 3, 4, 7}) {
            for (boolean sort : new boolean[] {false, true}) {
                try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
                     Agent agent = TinyCheckpoint.agent(root, session)) {
                    List<Prediction> got =
                            agent.predictBatch(states, questions, null, batchSize, sort);
                    assertEquals(states.size(), got.size(),
                            "one prediction per state at batchSize=" + batchSize);
                    List<String> rendered = new ArrayList<>();
                    for (Prediction prediction : got) {
                        assertIterableEquals(questions.keySet(), prediction.answers().keySet(),
                                "answers keyed by question id, in order");
                        rendered.add(prediction.answers().toString() + prediction.usage());
                    }
                    if (reference == null) {
                        reference = rendered;
                    } else {
                        // Grouping is a performance choice; it must not be observable in a result.
                        assertEquals(reference, rendered,
                                "batchSize=" + batchSize + " sort=" + sort + " changed a result");
                    }
                }
            }
        }
    }

    @Test
    @DisplayName("a batch size bounds the rows per graph call")
    void batchSizeBoundsRowsPerCall(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        List<Object> states = List.of("a", "b", "c", "d", "e");
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            agent.predictBatch(states, twoQuestions(), null, 2, false);
            // 5 states x 2 questions = 10 rows, at most 2 states (4 rows) per call
            assertEquals(3, session.batches.size(), "ceil(5 / 2) calls");
            for (var batch : session.batches) {
                assertTrue(batch.rows() <= 4, "rows per call: " + batch.rows());
            }
            int total = session.batches.stream().mapToInt(b -> b.rows()).sum();
            assertEquals(10, total);
        }
    }

    @Test
    @DisplayName("usage counts the state once per question, because each row carries it")
    void usageCountsStatePerRow(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 512, 128);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Prediction one = agent.predict("hello", twoQuestions());
            Usage usage = one.usage();
            assertEquals(0, usage.outputTokens(), "laya is not a generator");
            assertEquals(5, usage.stateTokens(), "one token per UTF-8 byte of \"hello\"");
            assertFalse(usage.truncated());
            assertTrue(usage.truncatedQuestions().isEmpty());
            assertTrue(usage.collapsedOptions().isEmpty());
            // Both rows embed the state, so input_tokens exceeds the state's own length.
            assertTrue(usage.inputTokens() > usage.stateTokens(),
                    "inputTokens=" + usage.inputTokens() + " stateTokens=" + usage.stateTokens());
            int rows = session.batches.get(0).rows();
            assertEquals(2, rows);
        }
    }

    @Test
    @DisplayName("a state that does not fit reports what was dropped, and which questions dropped it")
    void truncationIsReported(@TempDir Path root) throws IOException {
        // A tiny budget so the state cannot fit: the report is the only way a caller can tell.
        TinyCheckpoint.write(root, 40, 24);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Prediction one = agent.predict("x".repeat(500), twoQuestions());
            Usage usage = one.usage();
            assertEquals(500, usage.stateTokens());
            assertTrue(usage.truncated());
            assertTrue(usage.stateTokensDropped() > 400, "dropped " + usage.stateTokensDropped());
            assertIterableEquals(List.of("intent", "urgent"), usage.truncatedQuestions());
        }
    }

    @Test
    @DisplayName("options collapsed by the budget are reported with the count the question DEFINES")
    void collapsedOptionsAreReported(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 256, 32);
        Map<String, Question> questions = new LinkedHashMap<>();
        // Options identical for far longer than the per-option cap, so the cap collapses them.
        Map<String, Object> criteria = new LinkedHashMap<>();
        criteria.put("prefix ".repeat(40) + "one", "d");
        criteria.put("prefix ".repeat(40) + "two", "d");
        questions.put("pick", Question.choice("Pick.", criteria));
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Usage usage = agent.predict("s", questions).usage();
            Usage.CollapsedOptions collapsed = usage.collapsedOptions().get("pick");
            org.junit.jupiter.api.Assertions.assertNotNull(collapsed,
                    "a collapse must be reported: " + usage.collapsedOptions());
            assertEquals(2, collapsed.total(), "the count the question DEFINES");
            assertEquals(1, collapsed.distinct(), "both options became the same token span");
        }
    }

    @Test
    @DisplayName("the row decoded for a question is that question's own row")
    void rowOffsetsLineUp(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        // The stub's logits encode the row index, and they rise with the slot, so the argmax is
        // the LAST option of each question. A row offset that slipped would change the answer.
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("a", Question.choice("A?", TinyCheckpoint.ordered("x", "1", "y", "2")));
        questions.put("b", Question.choice("B?", TinyCheckpoint.ordered("p", "1", "q", "2", "r", "3")));
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Map<String, Answer> answers = agent.predict("s", questions).answers();
            assertEquals("y", ((Answer.Choice) answers.get("a")).choice());
            assertEquals("r", ((Answer.Choice) answers.get("b")).choice());
            // act_probability comes from the row's own act logits, which differ per row
            assertTrue(answers.get("a").actProbability() != answers.get("b").actProbability(),
                    "each question must read its own action row");
        }
    }

    @Test
    @DisplayName("no questions is an empty answer per state, and no states is an empty list")
    void degenerateInputs(@TempDir Path root) throws IOException {
        // This asserted that no questions is REFUSED. The reference returns empty answers and
        // zero usage with no tokenization and no forward pass -- so throwing meant the two
        // runtimes could not be swapped under a question set derived from a filter, where an
        // empty schema or a fully disabled rule set is a well-formed request.
        TinyCheckpoint.write(root, 128, 48);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            Prediction one = agent.predict("s", new LinkedHashMap<>());
            assertTrue(one.answers().isEmpty(), "no questions, no answers");
            assertEquals(0, one.usage().inputTokens());
            assertEquals(0, one.usage().outputTokens());

            List<Prediction> many = agent.predictBatch(List.of("a", "b"), new LinkedHashMap<>());
            assertEquals(2, many.size(), "one empty result per state");
            for (Prediction prediction : many) {
                assertTrue(prediction.answers().isEmpty());
                assertEquals(0, prediction.usage().inputTokens());
            }

            assertTrue(agent.predictBatch(List.of(), twoQuestions()).isEmpty());
            assertTrue(agent.predictBatch(List.of(), new LinkedHashMap<>()).isEmpty());
            assertTrue(session.batches.isEmpty(),
                    "no questions means no tokenization and no forward pass");
        }
    }

    @Test
    @DisplayName("a LIST state truncates from the left, so the newest turns survive")
    void listStateTruncatesLeft(@TempDir Path root) throws IOException {
        // The reference's rule is `truncate_left = isinstance(state, list)`, and a conversation is
        // serialised newest-last. Cutting from the right threw away the current turn and answered
        // about the opening of the conversation -- measured on a 120-turn history as one option's
        // probability moving by 3.8x, while `usage` stayed byte-identical so nothing a caller
        // could read revealed it.
        TinyCheckpoint.write(root, 96, 32);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            List<Object> history = new java.util.ArrayList<>();
            for (int turn = 0; turn < 40; turn++) {
                Map<String, Object> message = new LinkedHashMap<>();
                message.put("role", turn % 2 == 0 ? "user" : "agent");
                message.put("text", "turn " + turn);
                history.add(message);
            }
            Map<String, Question> one = new LinkedHashMap<>();
            one.put("urgent", Question.noul("Is this urgent?"));

            Prediction listed = agent.predict(history, one);
            assertTrue(listed.usage().stateTokensDropped() > 0,
                    "the fixture must actually truncate, or this proves nothing");

            // The surviving window is the TAIL. The tiny checkpoint is byte-level, so the kept
            // ids decode back to text: the last turn must be present and the first must not.
            long[] kept = session.batches.get(0).inputIds()[0];
            StringBuilder decoded = new StringBuilder();
            for (long id : kept) {
                if (id < 243) {
                    decoded.append((char) id);
                }
            }
            String window = decoded.toString();
            assertTrue(window.contains("turn 39"),
                    "the newest turn must survive, got: " + window);
            assertFalse(window.contains("turn 0\""),
                    "the oldest turn must be the one dropped, got: " + window);
        }
    }

    @Test
    @DisplayName("a question whose markers do not all fit is refused, not answered partially")
    void markerOverflowIsRefused(@TempDir Path root) throws IOException {
        // Markers sit at absolute positions and the sequence is cut to max_len, so with enough
        // options the later ones fall off. Answering anyway returns a distribution over the
        // survivors, with the rest absent from the answer and unchoosable -- and usage.truncated
        // would blame the state. Python refuses the request.
        TinyCheckpoint.write(root, 48, 200);
        Map<String, Object> criteria = new LinkedHashMap<>();
        for (int i = 0; i < 30; i++) {
            criteria.put("label" + i, "description number " + i);
        }
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("topic", Question.choice("Pick one.", criteria));
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> agent.predict("s", questions));
            String message = refused.getMessage();
            assertTrue(message.contains("option markers fit"), message);
            assertTrue(message.contains("of its 30 option markers"), message);
            // Both knobs are named: LOWERING head_max_len can fix it, raising it makes it worse.
            assertTrue(message.contains("head_max_len") && message.contains("max_len"), message);
            assertTrue(session.batches.isEmpty(), "the graph must not be called at all");
        }
    }

    @Test
    @DisplayName("mutating the questions map during a call cannot relabel an answer")
    void questionsAreSnapshotted(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("intent", Question.choice("What next?",
                TinyCheckpoint.ordered("refund", "money back", "escalate", "a human")));
        // A session that swaps the caller's map while the graph is "running": the sequences are
        // already built, and the answers are labelled afterwards. Reading the map twice let one
        // question's labels land on another question's logits, with no exception.
        InferenceSwappingSession session = new InferenceSwappingSession(questions);
        try (Agent agent = TinyCheckpoint.agent(root, session)) {
            Map<String, Answer> answers = agent.predict("s", questions).answers();
            Answer.Choice choice = (Answer.Choice) answers.get("intent");
            assertIterableEquals(List.of("refund", "escalate"), choice.probabilities().keySet(),
                    "the answer must carry the labels of the question that was actually encoded");
        }
    }

    /** Swaps the caller's questions map mid-call, to prove the runtime snapshotted it. */
    private static final class InferenceSwappingSession extends TinyCheckpoint.RecordingSession {
        private final Map<String, Question> victim;

        InferenceSwappingSession(Map<String, Question> victim) {
            this.victim = victim;
        }

        @Override
        public Output run(com.convaiinnovations.laya.sequence.Collator.Batch batch) {
            victim.put("intent", Question.choice("Different question.",
                    TinyCheckpoint.ordered("APPROVE_WIRE", "x", "HOLD_WIRE", "y")));
            return super.run(batch);
        }
    }

    @Test
    @DisplayName("using an agent after closing it is an API error, not a backend one")
    void useAfterCloseIsAnApiError(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
        Agent agent = TinyCheckpoint.agent(root, session);
        agent.close();
        IllegalStateException refused = assertThrows(IllegalStateException.class,
                () -> agent.predict("s", twoQuestions()));
        assertTrue(refused.getMessage().contains("closed"), refused.getMessage());
    }

    @Test
    @DisplayName("a null state is REFUSED, as Python refuses it")
    void nullStateIsRefused(@TempDir Path root) throws IOException {
        // This test asserted the opposite, with a comment claiming Python answers a null state
        // because serialize_state writes "null". Both Python backends raise
        // `TypeError("state must not be None; pass a string, dict, or list")` -- and the
        // reference's own comment gives that very serialisation as the REASON: answering it would
        // be a confident decision about the four characters "null", byte-identical to passing the
        // string. A caller whose state field is absent must hear about it.
        TinyCheckpoint.write(root, 128, 48);
        try (TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
             Agent agent = TinyCheckpoint.agent(root, session)) {
            assertThrows(IllegalArgumentException.class,
                    () -> agent.predict(null, twoQuestions()));
            assertThrows(IllegalArgumentException.class,
                    () -> agent.predictBatch(java.util.Collections.singletonList(null),
                            twoQuestions()));
            // and one null among several is refused by index, not silently answered
            IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                    () -> agent.predictBatch(java.util.Arrays.asList("a", null),
                            twoQuestions()));
            assertTrue(failure.getMessage().contains("index 1"), failure.getMessage());
            assertTrue(session.batches.isEmpty(), "nothing may reach the graph");
        }
    }

    @Test
    @DisplayName("closing the agent closes the session, and closing twice is safe")
    void closeIsIdempotent(@TempDir Path root) throws IOException {
        TinyCheckpoint.write(root, 128, 48);
        TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
        Agent agent = TinyCheckpoint.agent(root, session);
        agent.close();
        assertTrue(session.closed);
        agent.close();
    }
}
