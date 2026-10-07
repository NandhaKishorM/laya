package com.convaiinnovations.laya;

import java.util.List;
import java.util.Map;

/**
 * What one prediction cost, and what it silently lost.
 *
 * <p>The second half is the point. A state longer than the room left after the question's head is
 * truncated without an error, and a caller cannot reconstruct that from outside: the budget is in
 * tokens, not characters, and the room left depends on {@code max_len}, {@code head_max_len}, the
 * instruction and the rendered options, so it moves per checkpoint and per question. A caller
 * guessing with a fixed character threshold is wrong in both directions -- it reports truncation
 * that did not happen and stays silent while evidence is being dropped.
 *
 * @param inputTokens         non-padding tokens fed to the encoder across every row of this state.
 *                            The state is re-encoded per question, so a shared document is counted
 *                            once per question -- which is what it actually costs
 * @param outputTokens        always zero: laya is not a generator
 * @param stateTokens         tokens the state serialised to, before any budget was applied
 * @param stateTokensDropped  the worst case across this state's questions, since they share a
 *                            state but not a head budget
 * @param truncated           whether any question dropped state tokens
 * @param truncatedQuestions  the ids of the questions that did
 * @param collapsedOptions    questions whose options no longer have a token span each, keyed by
 *                            question id. Empty unless the head budget collapsed some
 */
public record Usage(int inputTokens, int outputTokens, int stateTokens, int stateTokensDropped,
                    boolean truncated, List<String> truncatedQuestions,
                    Map<String, Usage.CollapsedOptions> collapsedOptions) {

    /**
     * A question whose options the budget collapsed.
     *
     * @param total           how many options the question DEFINES, not how many markers reached
     *                        the sequence. Counting from markers would report "43 of 58" about a
     *                        request where 28 options never entered the input at all
     * @param distinct        how many still have a token span of their own
     * @param tokensPerOption the cap applied, or null when the collapse came from the 48-token
     *                        per-option cap rather than the head budget
     */
    public record CollapsedOptions(int total, int distinct, Integer tokensPerOption) {
    }
}
