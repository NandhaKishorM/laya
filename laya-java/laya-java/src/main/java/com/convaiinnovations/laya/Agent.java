package com.convaiinnovations.laya;

import com.convaiinnovations.laya.config.AgentConfig;
import com.convaiinnovations.laya.decode.Decoder;
import com.convaiinnovations.laya.infer.InferenceSession;
import com.convaiinnovations.laya.onnx.LayaSession;
import com.convaiinnovations.laya.sequence.Collator;
import com.convaiinnovations.laya.sequence.SequenceBuilder;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The runtime: ask several typed questions about one piece of evidence, in one forward pass.
 *
 * <p>laya is a "System 1" decision model -- a bidirectional encoder with a typed head, not a
 * generator. Every question about the same state becomes one row of a single batch, so four
 * questions about a document cost one batched encode rather than four round trips.
 *
 * <pre>{@code
 * try (Agent agent = Agent.open(modelDir, graphDir)) {
 *     Map<String, Question> questions = new LinkedHashMap<>();
 *     questions.put("intent", Question.choice("What does the customer want?", criteria));
 *     questions.put("urgent", Question.noul("This needs a human today."));
 *     Prediction p = agent.predict(email, questions, "en");
 *     if (p.usage().truncated()) {
 *         // the state did not fit: usage says how much was dropped, and from which questions
 *     }
 * }
 * }</pre>
 *
 * <p>Pass questions in a {@link LinkedHashMap}: a choice's options are positional, so iteration
 * order is part of what is asked.
 *
 * <p>One session is not safe for concurrent {@code predict} calls unless ONNX Runtime is
 * configured for it; hold one {@code Agent} per worker, or serialise access.
 */
public final class Agent implements AutoCloseable, Predictor {

    private final Tokenizer tokenizer;
    private final AgentConfig config;
    private final InferenceSession session;
    private final int padId;
    /**
     * This agent's own lifecycle, not the session's.
     *
     * <p>Tracked here rather than asked of the session: "is this agent usable" is the agent's
     * question, and putting it on {@link InferenceSession} would widen a seam that exists to be
     * narrow -- a stub would have to answer it with something, and a default answer of "open"
     * would be a lie the moment it was not.
     */
    private volatile boolean closed;

    private Agent(Tokenizer tokenizer, AgentConfig config, InferenceSession session, int padId) {
        this.tokenizer = tokenizer;
        this.config = config;
        this.session = session;
        this.padId = padId;
    }

    /**
     * Opens a checkpoint.
     *
     * @param modelDirectory holds {@code rl_agent_config.json} and {@code tokenizer/}
     * @param graphDirectory holds {@code laya.onnx}, or {@code encoder.onnx} and {@code head.onnx}
     */
    public static Agent open(Path modelDirectory, Path graphDirectory) throws IOException {
        // 0 means "let ONNX Runtime choose", which is what the Python runtime does. Pinning one
        // thread here ran the forward pass on a single core and cost about 3x.
        return open(modelDirectory, graphDirectory, 0);
    }

    /** Opens a checkpoint with an explicit intra-op thread count. */
    public static Agent open(Path modelDirectory, Path graphDirectory, int threads)
            throws IOException {
        return using(Tokenizer.fromModelDirectory(modelDirectory),
                AgentConfig.fromModelDirectory(modelDirectory),
                LayaSession.open(graphDirectory, threads));
    }

    /**
     * Assembles an agent from parts, for a caller that already holds them -- a router sharing one
     * tokenizer across checkpoints, or a test driving the batching and usage accounting through a
     * stub {@link InferenceSession} instead of a 1.2 GB graph.
     */
    public static Agent using(Tokenizer tokenizer, AgentConfig config, InferenceSession session) {
        // Padding is masked out of attention but still embedded, so it has to be a real id.
        // No silent fallback to the SEP id. The reference hands `tok.pad_token_id` straight to
        // the collator, so a checkpoint naming no pad_token fails the request outright -- and a
        // mis-exported checkpoint that fails loudly on one runtime must not answer quietly on the
        // other. (The graph masks padding, so the substituted id did not itself move the numbers;
        // the divergence was that one runtime accepted a configuration the other rejects.)
        int padId = tokenizer.padId().orElseThrow(() -> new IllegalStateException(
                "this checkpoint names no pad_token, so a batch cannot be padded; the reference "
                + "refuses the same checkpoint"));
        return new Agent(tokenizer, config, session, padId);
    }

    /** The checkpoint's tokenizer, for callers that want to measure a state's token cost. */
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    /** The checkpoint's budgets and temperatures. */
    public AgentConfig config() {
        return config;
    }

    /** Asks every question about one state, with no language override. */
    public Prediction predict(Object state, Map<String, Question> questions) {
        return predict(state, questions, null);
    }

    /**
     * Asks every question about one state in one forward pass.
     *
     * @param language a tag whose prefix before {@code -} may select a temperature override, or null
     */
    public Prediction predict(Object state, Map<String, Question> questions, String language) {
        // `singletonList`, not `List.of`, so that a null state reaches predictBatch's own check
        // and is refused there with a message rather than by a bare NullPointerException from
        // the list factory. The comment here used to claim a null state was legitimate because
        // "Python serialises None to the text null" -- which is backwards: the reference refuses
        // it for exactly that reason, since answering it would be a confident decision about the
        // four characters "null", byte-identical to passing the string.
        return predictBatch(Collections.singletonList(state), questions, language, 0, false).get(0);
    }

    /**
     * Asks every question about every state, one row per (state, question).
     *
     * <p>Length sorting is off, which is the Python runtime's default too. It never changes an
     * answer -- it only cuts padding -- but leaving it off keeps the batched path's grouping
     * predictable, and a caller who wants the throughput can ask for it.
     */
    public List<Prediction> predictBatch(List<?> states, Map<String, Question> questions) {
        return predictBatch(states, questions, null, 0, false);
    }

    /**
     * Asks every question about every state.
     *
     * @param batchSize    rows' worth of states per graph call, or 0 for all of them at once.
     *                     This bounds peak memory, not the answers
     * @param sortByLength group states of similar length into the same call, which cuts padding.
     *                     It changes nothing about the answers: every row is sliced back to its
     *                     own option count and the results are returned in the caller's order
     * @return one prediction per state, in the order the states were given
     */
    public List<Prediction> predictBatch(List<?> states, Map<String, Question> questions,
                                         String language, int batchSize, boolean sortByLength) {
        if (closed) {
            throw new IllegalStateException(
                    "this agent is closed; open a new one rather than reusing it");
        }
        for (int i = 0; i < states.size(); i++) {
            // Refused, as both Python backends refuse it: `serialize_state(None)` is
            // `json.dumps(None)`, so a missing state would otherwise be answered as a decision
            // about the literal text "null" -- byte-identical to passing the string, and at full
            // confidence. A caller whose state field is absent should hear about it.
            if (states.get(i) == null) {
                throw new IllegalArgumentException(
                        "state at index " + i + " is null; pass a string, a map or a list");
            }
        }
        if (states.isEmpty()) {
            return List.of();
        }
        if (questions.isEmpty()) {
            // Empty answers, zero usage, no tokenization and no forward pass -- which is what
            // the reference returns. Throwing here meant the two runtimes could not be swapped
            // under a question set derived from a filter: an empty schema or a disabled rule set
            // is a well-formed request with a well-formed empty answer.
            List<Prediction> empty = new ArrayList<>(states.size());
            for (int i = 0; i < states.size(); i++) {
                empty.add(new Prediction(Prediction.MODEL, Map.of(),
                        new Usage(0, 0, 0, 0, false, List.of(), Map.of())));
            }
            return List.copyOf(empty);
        }
        // Snapshotted once, at entry. The map is the caller's, and the sequences are built before
        // the graph runs while the answers are labelled after it: reading it twice let a mutation
        // in between attach one question's labels to another question's logits, with no exception
        // and nothing in `usage` to show it. Python snapshots for the same reason.
        Map<String, Question> asked = new LinkedHashMap<>(questions);
        for (Map.Entry<String, Question> entry : asked.entrySet()) {
            if (entry.getValue() == null) {
                throw new IllegalArgumentException("question " + entry.getKey() + " is null");
            }
        }
        List<String> questionIds = new ArrayList<>(asked.keySet());
        int chunk = batchSize > 0 ? batchSize : states.size();
        // Sorting only pays off when it can actually reorder across more than one call, which
        // mirrors the Python runtime's own guard.
        boolean reorder = sortByLength && chunk > 1 && chunk < states.size();
        // Bound the tokenized lookahead independently of the input size: sort within windows of
        // eight calls rather than over the whole input, so a million states do not have to be
        // encoded before the first one runs.
        int window = reorder ? chunk * 8 : chunk;

        Prediction[] out = new Prediction[states.size()];
        for (int start = 0; start < states.size(); start += window) {
            int end = Math.min(states.size(), start + window);
            List<List<SequenceBuilder.Sequence>> encoded = new ArrayList<>(end - start);
            for (int i = start; i < end; i++) {
                encoded.add(encodeState(states.get(i), questionIds, asked));
            }
            List<Integer> order = new ArrayList<>(encoded.size());
            for (int i = 0; i < encoded.size(); i++) {
                order.add(i);
            }
            if (reorder) {
                order.sort((a, b) -> Integer.compare(longestRow(encoded.get(a)),
                        longestRow(encoded.get(b))));
            }
            for (int offset = 0; offset < order.size(); offset += chunk) {
                List<Integer> indices = order.subList(offset, Math.min(order.size(), offset + chunk));
                List<Collator.Item> rows = new ArrayList<>();
                for (int index : indices) {
                    List<SequenceBuilder.Sequence> built = encoded.get(index);
                    for (int q = 0; q < questionIds.size(); q++) {
                        rows.add(new Collator.Item(built.get(q).ids(), built.get(q).markers(),
                                asked.get(questionIds.get(q)).type().code()));
                    }
                }
                InferenceSession.Output output = session.run(Collator.collate(rows, padId));
                int row = 0;
                for (int index : indices) {
                    List<SequenceBuilder.Sequence> built = encoded.get(index);
                    out[start + index] = assemble(built, questionIds, asked, output, row,
                            language);
                    row += questionIds.size();
                }
            }
        }
        return List.of(out);
    }

    /**
     * Builds every question's sequence for one state, tokenizing the state once.
     *
     * <p>Once, not once per question: that is what {@code build_sequence}'s pre-tokenized state
     * parameter exists for, and a document is usually far longer than the question asked about it.
     */
    private List<SequenceBuilder.Sequence> encodeState(Object state, List<String> questionIds,
                                                       Map<String, Question> questions) {
        int[] stateIds = tokenizer.encode(
                SequenceBuilder.serializeState(state).replace(maskToken(), " "), -1);
        List<SequenceBuilder.Sequence> built = new ArrayList<>(questionIds.size());
        for (String id : questionIds) {
            Question question = questions.get(id);
            // truncateLeft for a LIST state, which is the reference's rule
            // (`truncate_left = isinstance(state, list)`). A conversation is serialised
            // newest-last, so cutting from the right throws away the current turn and answers
            // about the opening of the conversation instead. Measured on a 120-turn history: the
            // kept window started at message 0 where the reference started mid-message-95, and
            // the answers moved -- one option's probability by 3.8x -- while `usage` stayed
            // byte-identical, so nothing a caller can read revealed it.
            boolean truncateLeft = state instanceof List;
            SequenceBuilder.Sequence sequence = SequenceBuilder.build(tokenizer, state, question,
                    config.maxLen(), config.headMaxLen(), null, truncateLeft, stateIds);
            int defined = question.renderOptions().size();
            if (sequence.markers().length != defined) {
                // Markers sit at absolute positions and the sequence is then cut to `max_len`, so
                // the ones past it are dropped. Answering anyway returns a distribution over the
                // options that happened to survive, with the rest absent from the answer and
                // unchoosable, while `usage.truncated` blames the state instead. Python refuses the
                // request; so does this, naming both knobs because LOWERING `head_max_len` can fix
                // it while raising it makes it worse.
                throw new IllegalArgumentException(String.format(
                        "question %s: only %d of its %d option markers fit in max_len=%d with "
                        + "head_max_len=%d spent on the question; lower head_max_len, raise "
                        + "max_len, or use fewer options",
                        id, sequence.markers().length, defined, config.maxLen(),
                        config.headMaxLen()));
            }
            built.add(sequence);
        }
        return built;
    }

    private Prediction assemble(List<SequenceBuilder.Sequence> built, List<String> questionIds,
                                Map<String, Question> questions, InferenceSession.Output output,
                                int firstRow, String language) {
        Map<String, Answer> answers = new LinkedHashMap<>();
        int inputTokens = 0;
        int dropped = 0;
        List<String> truncatedQuestions = new ArrayList<>();
        Map<String, Usage.CollapsedOptions> collapsed = new LinkedHashMap<>();
        for (int q = 0; q < questionIds.size(); q++) {
            String id = questionIds.get(q);
            SequenceBuilder.Sequence sequence = built.get(q);
            int row = firstRow + q;
            // The FILTERED marker count: a head that overran `max_len` loses markers, and the
            // answer must be derived over the options that survived into the sequence.
            int optionCount = sequence.markers().length;
            float[] actionProbabilities = Decoder.actionProbabilities(output.actLogits()[row]);
            answers.put(id, Decoder.decode(questions.get(id), output.logits()[row], optionCount,
                    actionProbabilities, null, language, config));

            // Every row carries the state, so the state's tokens are counted once per question --
            // which is what the forward pass actually costs.
            inputTokens += sequence.ids().length;
            dropped = Math.max(dropped, sequence.truncation().stateTokensDropped());
            if (sequence.truncation().truncated()) {
                truncatedQuestions.add(id);
            }
            SequenceBuilder.Stats stats = sequence.stats();
            if (stats.optionsDistinct() < stats.options()) {
                collapsed.put(id, new Usage.CollapsedOptions(stats.options(),
                        stats.optionsDistinct(), stats.tokensPerOption()));
            }
        }
        Usage usage = new Usage(inputTokens, 0, built.get(0).truncation().stateTokens(), dropped,
                dropped > 0, List.copyOf(truncatedQuestions),
                Collections.unmodifiableMap(collapsed));
        return new Prediction(Prediction.MODEL, Collections.unmodifiableMap(answers), usage);
    }

    private static int longestRow(List<SequenceBuilder.Sequence> built) {
        int longest = 0;
        for (SequenceBuilder.Sequence sequence : built) {
            longest = Math.max(longest, sequence.ids().length);
        }
        return longest;
    }

    private String maskToken() {
        String mask = tokenizer.maskToken();
        if (mask == null) {
            throw new IllegalStateException(
                    "this checkpoint's tokenizer_config.json names no mask_token");
        }
        return mask;
    }

    /** Closes the session. Idempotent, as {@code AutoCloseable} requires. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        session.close();
    }
}
