package com.convaiinnovations.laya;

import java.util.Map;

/**
 * One state's answers, with what they cost.
 *
 * @param model   the runtime that produced this, matching what Python's ONNX backend reports
 * @param answers the answers, keyed by question id, in the order the questions were asked
 * @param usage   token cost and what the budgets dropped; see {@link Usage}
 */
public record Prediction(String model, Map<String, Answer> answers, Usage usage) {

    /** The identifier the Python ONNX backend reports, so a consumer cannot tell them apart. */
    public static final String MODEL = "laya-rl-agent-onnx";

    /** Shorthand for one answer, or null when no question had that id. */
    public Answer answer(String questionId) {
        return answers.get(questionId);
    }
}
