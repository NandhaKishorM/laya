package com.convaiinnovations.laya.sequence;

import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.json.PythonJson;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Composes a question and its state into the one token sequence the encoder receives.
 *
 * <pre>
 * [CLS] &lt;type&gt; question: &lt;instructions&gt; [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
 * </pre>
 *
 * <p>Each option is preceded by a {@code [MASK]} token whose <b>position</b> is the option's
 * marker; the head reads the encoder's output at those positions to produce one logit per option.
 * A marker that is off by one still yields a plausible answer, which is why the markers are part
 * of the parity fixture rather than inferred from the ids.
 *
 * <p>The budget arithmetic is the subtle part and it is reproduced step for step from
 * {@code laya/common.py:196}. Two rules decide what survives when the question does not fit:
 * <b>options yield nothing and the instruction yields everything</b> (the instruction is clamped to
 * whatever the options leave, never the reverse), and the state takes only the room left after the
 * head, silently, which is why {@link Sequence#truncation()} reports that clamp -- a caller cannot
 * reconstruct it from outside, because the budget is in tokens and moves per checkpoint and per
 * question.
 */
public final class SequenceBuilder {

    /** The minimum head budget the options must leave before they are re-capped. */
    private static final int MIN_OPTION_BUDGET = 16;
    /** The floor on the instruction, which keeps a question from losing its verb entirely. */
    private static final int MIN_INSTRUCTION_TOKENS = 8;
    /** The per-option cap the tokenizer applies before any budget arithmetic. */
    private static final int OPTION_TOKEN_CAP = 48;
    /** The floor on a re-capped option: the marker plus three tokens. */
    private static final int MIN_TOKENS_PER_OPTION = 4;

    private SequenceBuilder() {
    }

    /** What the option budget did to the question. */
    public record Stats(int options, int optionsDistinct, Integer tokensPerOption) {
        /**
         * @param options         how many options the question defines
         * @param optionsDistinct how many still have a token span of their own. Fewer than
         *                        {@code options} means the cap collapsed two options to the same
         *                        tokens, so the model cannot tell them apart -- and nothing else
         *                        downstream can see it, because the marker count still matches
         *                        (laya issue #538)
         * @param tokensPerOption the cap applied to each option, or null when none was
         */
        public Stats {
        }
    }

    /** What the state budget dropped. */
    public record Truncation(int stateTokens, int stateTokensUsed, int stateTokensDropped,
                             boolean truncated) {
    }

    /** A built sequence: the ids, each option's marker position, and what the budgets did. */
    public record Sequence(int[] ids, int[] markers, Stats stats, Truncation truncation) {
    }

    /** The question half alone, with no state: what {@link #build} appends the state to. */
    public record Head(List<Integer> ids, List<Integer> markers, Stats stats) {
    }

    /**
     * Builds the full sequence.
     *
     * @param tok          the checkpoint's tokenizer; it must name a CLS, SEP and mask token
     * @param state        the evidence: a String, or any value {@link PythonJson} can serialise
     * @param question     the decision to put to the model
     * @param maxLen       the checkpoint's {@code max_len} (512 for english, 1024 for multilingual)
     * @param headMaxLen   the checkpoint's {@code head_max_len} (192 and 256 respectively)
     * @param optionOrder  the order to present options in, or null for label order
     * @param truncateLeft keep the END of an oversized state rather than its beginning
     * @param stateIds     a pre-tokenized state, so a shared document is encoded once per request
     *                     rather than once per question, or null to encode {@code state} here
     */
    public static Sequence build(Tokenizer tok, Object state, Question question,
                                 int maxLen, int headMaxLen, int[] optionOrder,
                                 boolean truncateLeft, int[] stateIds) {
        Head head = buildHead(tok, question, headMaxLen, optionOrder);
        List<Integer> ids = new ArrayList<>(head.ids());
        int sep = required(tok.sepId(), "sep_token", tok);

        // `room` already reserves the closing [SEP], so every state token it admits survives the
        // final clamp to `maxLen`.
        int room = Math.max(0, maxLen - ids.size() - 1);
        int[] stateTokens = stateIds;
        if (stateTokens == null) {
            stateTokens = tok.encode(scrub(serializeState(state), tok), -1);
        }
        int used = Math.min(room, stateTokens.length);
        int from = truncateLeft ? stateTokens.length - used : 0;
        for (int i = from; i < from + used; i++) {
            ids.add(stateTokens[i]);
        }
        ids.add(sep);

        int size = Math.min(ids.size(), maxLen);
        int[] finalIds = new int[size];
        for (int i = 0; i < size; i++) {
            finalIds[i] = ids.get(i);
        }
        List<Integer> kept = new ArrayList<>(head.markers().size());
        for (int marker : head.markers()) {
            if (marker < maxLen) {
                kept.add(marker);
            }
        }
        int[] markers = new int[kept.size()];
        for (int i = 0; i < markers.length; i++) {
            markers[i] = kept.get(i);
        }
        Truncation truncation = new Truncation(
                stateTokens.length, used, stateTokens.length - used, used < stateTokens.length);
        return new Sequence(finalIds, markers, head.stats(), truncation);
    }

    /**
     * The question half: {@code [CLS] <type> question: <instructions> [SEP] [MASK] opt0 ... [SEP]}.
     */
    public static Head buildHead(Tokenizer tok, Question question, int headMaxLen,
                                 int[] optionOrder) {
        int cls = required(tok.clsId(), "cls_token", tok);
        int sep = required(tok.sepId(), "sep_token", tok);
        int mask = required(tok.maskId(), "mask_token", tok);

        List<String> options = question.renderOptions();
        int[] order = optionOrder != null ? optionOrder : naturalOrder(options.size());
        if (order.length != options.size()) {
            throw new IllegalArgumentException(
                    "option_order has " + order.length + " entries for " + options.size()
                    + " options");
        }

        String instructions = scrub(question.instructions(), tok);
        int[] headIds = tok.encode(question.type().wireName() + " question: " + instructions, -1);

        List<int[]> optionIds = new ArrayList<>(order.length);
        for (int index : order) {
            if (index < 0 || index >= options.size()) {
                throw new IllegalArgumentException(
                        "option_order references option " + index + " of " + options.size());
            }
            // The leading space is part of the contract: a byte-level tokenizer gives " refund" and
            // "refund" different ids, so dropping it changes every option's tokens.
            int[] text = tok.encode(" " + scrub(options.get(index), tok), OPTION_TOKEN_CAP);
            int[] withMarker = new int[text.length + 1];
            withMarker[0] = mask;
            System.arraycopy(text, 0, withMarker, 1, text.length);
            optionIds.add(withMarker);
        }

        int optionBudget = headMaxLen - totalLength(optionIds);
        Integer tokensPerOption = null;
        if (optionBudget < MIN_OPTION_BUDGET) {
            // `Math.floorDiv`, not `/`: Python floors toward negative infinity and Java truncates
            // toward zero, so a head budget below 16 would disagree. Both land on the floor of 4
            // here, but the arithmetic should be the same arithmetic rather than accidentally so.
            int per = Math.max(MIN_TOKENS_PER_OPTION,
                    Math.floorDiv(headMaxLen - MIN_OPTION_BUDGET, Math.max(1, optionIds.size())));
            tokensPerOption = per;
            for (int i = 0; i < optionIds.size(); i++) {
                int[] option = optionIds.get(i);
                if (option.length > per) {
                    optionIds.set(i, Arrays.copyOf(option, per));
                }
            }
            optionBudget = headMaxLen - totalLength(optionIds);
        }

        // The instruction yields to the options, never the other way round.
        int instructionRoom = Math.max(MIN_INSTRUCTION_TOKENS, optionBudget);
        if (headIds.length > instructionRoom) {
            headIds = Arrays.copyOf(headIds, instructionRoom);
        }

        List<Integer> ids = new ArrayList<>(headIds.length + 2 + totalLength(optionIds) + 1);
        ids.add(cls);
        for (int id : headIds) {
            ids.add(id);
        }
        ids.add(sep);
        List<Integer> markers = new ArrayList<>(optionIds.size());
        for (int[] option : optionIds) {
            markers.add(ids.size());
            for (int id : option) {
                ids.add(id);
            }
        }
        ids.add(sep);

        // Counted on the CAPPED option ids, before assembly: re-slicing the finished sequence
        // cannot close the last option's span, because it runs on into the state.
        Set<String> distinct = new HashSet<>();
        for (int[] option : optionIds) {
            distinct.add(Arrays.toString(option));
        }
        return new Head(ids, markers,
                new Stats(optionIds.size(), distinct.size(), tokensPerOption));
    }

    /**
     * The state as text: a String passes through, anything else becomes Python-compatible JSON.
     *
     * <p>A null state is not an error: Python's {@code serialize_state} hands {@code None} to
     * {@code json.dumps}, which writes the four characters {@code null}, and the model is asked
     * about that. Reproduced rather than refused, so the two runtimes answer the same question.
     */
    public static String serializeState(Object state) {
        return state instanceof String ? (String) state : PythonJson.dumps(state);
    }

    /**
     * Removes the mask token's text from caller-supplied text.
     *
     * <p>Applied to the instruction, to every option and to the serialised state. Without it a
     * caller can write the mask string into any of the three and inject option markers into the
     * sequence, so the head would read logits at positions that are not options.
     */
    private static String scrub(String text, Tokenizer tok) {
        String mask = tok.maskToken();
        if (mask == null) {
            throw new IllegalStateException(
                    "this checkpoint's tokenizer_config.json names no mask_token, so option "
                    + "markers cannot be placed and caller text cannot be made safe");
        }
        return text.replace(mask, " ");
    }

    private static int required(java.util.OptionalInt id, String role, Tokenizer tok) {
        if (id.isEmpty()) {
            throw new IllegalStateException(
                    "this checkpoint's tokenizer_config.json names no " + role
                    + ", which a sequence cannot be built without");
        }
        return id.getAsInt();
    }

    private static int totalLength(List<int[]> arrays) {
        int total = 0;
        for (int[] array : arrays) {
            total += array.length;
        }
        return total;
    }

    private static int[] naturalOrder(int count) {
        int[] order = new int[count];
        for (int i = 0; i < count; i++) {
            order[i] = i;
        }
        return order;
    }
}
