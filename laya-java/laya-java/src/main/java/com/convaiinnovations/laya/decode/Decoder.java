package com.convaiinnovations.laya.decode;

import com.convaiinnovations.laya.Answer;
import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.config.AgentConfig;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Turns a question's raw logit row into a typed answer.
 *
 * <p>The steps, in order, are {@code agent.py:1316}'s: slice the row to the question's option
 * count, divide by the temperature its bucket names, softmax with the maximum subtracted, put the
 * row back into option order, then derive the type's fields and round everything to four decimals.
 *
 * <p><b>Precision.</b> The logits arrive from ONNX as float32 and the softmax is computed here in
 * {@code double}, which is what {@code laya-ts} and {@code laya-dotnet} both do. Python computes it
 * in float32, so the two are not bit-identical: measured over 200,000 random logit rows, the
 * four-decimal probabilities differ on <b>88</b> of them, by at most one unit in the last reported
 * decimal. Matching float32 exactly would mean reproducing NumPy's own float32 {@code exp} -- which
 * differs from rounding a double {@code exp} on 0.65% of values -- and its pairwise summation
 * order, which differs from a sequential sum on 27% of rows. Neither is a contract NumPy publishes,
 * so this port holds the same line the other two SDKs hold rather than a line it cannot keep.
 */
public final class Decoder {

    private Decoder() {
    }

    /**
     * Decodes one question's answer.
     *
     * @param question     the question that was asked; its type and labels shape the answer
     * @param logitRow     the head's output row, which may be <b>wider</b> than the option count
     * @param optionCount  how many options the question had markers for; the row is sliced to it
     * @param actProbabilities the head's second output for this row, <b>already softmaxed</b>
     *                         -- see {@link #actionProbabilities}. Its first element is reported
     * @param optionOrder  the order options were presented in, or null for label order
     * @param language     the request's language tag, which may select an override table
     * @param config       the checkpoint's temperatures
     */
    public static Answer decode(Question question, float[] logitRow, int optionCount,
                                float[] actProbabilities, int[] optionOrder, String language,
                                AgentConfig config) {
        if (optionCount < 1 || optionCount > logitRow.length) {
            throw new IllegalArgumentException(
                    "option count " + optionCount + " does not fit a logit row of "
                    + logitRow.length);
        }
        double temperature =
                config.temperatureFor(question.type(), optionCount, language);

        // Sliced to the option count, not the row width: the head's output is padded out to the
        // widest question in the batch, and the padding columns are large finite numbers rather
        // than -inf, so a softmax over the whole row would hand almost all the mass to padding.
        double[] z = new double[optionCount];
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < optionCount; i++) {
            z[i] = logitRow[i] / temperature;
            max = Math.max(max, z[i]);
        }
        double[] p = new double[optionCount];
        double total = 0.0;
        for (int i = 0; i < optionCount; i++) {
            // The maximum is subtracted first so a logit of 700 does not overflow `exp`.
            p[i] = Math.exp(z[i] - max);
            total += p[i];
        }
        for (int i = 0; i < optionCount; i++) {
            p[i] /= total;
        }
        p = unpermute(p, optionOrder);

        double answerConfidence = Rounding.round4(answerConfidence(p, optionCount));
        double actProbability = Rounding.round4(actProbabilities[0]);

        switch (question.type()) {
            case CHOICE: {
                List<String> labels = question.labels();
                Map<String, Double> probabilities = new LinkedHashMap<>();
                int best = argmax(p);
                for (int i = 0; i < labels.size() && i < p.length; i++) {
                    probabilities.put(labels.get(i), Rounding.round4(p[i]));
                }
                // `unmodifiableMap` over a LinkedHashMap, not `Map.copyOf`: the latter returns
                // a map whose iteration order is unspecified, and a choice's probabilities are
                // positional -- they must come back in option order, as Python's dict does.
                return new Answer.Choice(labels.get(best),
                        Collections.unmodifiableMap(probabilities),
                        Rounding.round4(entropyConfidence(p, optionCount)),
                        answerConfidence, actProbability);
            }
            case SCORE: {
                double expected = 0.0;
                for (int i = 0; i < optionCount; i++) {
                    expected += i * p[i];
                }
                Map<String, Double> probabilities = new LinkedHashMap<>();
                for (int i = 0; i < p.length; i++) {
                    probabilities.put(Integer.toString(i), Rounding.round4(p[i]));
                }
                return new Answer.Score(Rounding.round4(expected),
                        question.legend(), Collections.unmodifiableMap(probabilities),
                        Rounding.round4(entropyConfidence(p, optionCount)),
                        answerConfidence, actProbability);
            }
            default: {
                double trueProbability = p[1];
                return new Answer.Noul(Rounding.round4(trueProbability),
                        Rounding.round4(Math.max(trueProbability, 1.0 - trueProbability)),
                        answerConfidence, actProbability);
            }
        }
    }

    /**
     * The action head's output as probabilities: a softmax over one row of {@code act_logits}.
     *
     * <p>Separate from {@link #decode} because the runtime softmaxes the whole {@code act_logits}
     * block row-wise before decoding, and the decoder is given the result. Reporting the raw logit
     * instead would publish an unbounded real number as a probability -- it happens to look
     * plausible for a well-trained head, which is what makes it the wrong kind of bug.
     */
    public static float[] actionProbabilities(float[] actLogits) {
        if (actLogits.length == 0) {
            // Published as NaN before this. The reference indexes act[r, 0] and raises; an
            // escalation signal that reads NaN is worse than one that refuses.
            throw new IllegalArgumentException(
                    "the graph produced no action logits, so there is no action probability");
        }
        // The width the graph actually produced. Padding a single column up to two turned a
        // probability of 1.0 into sigmoid(logit) -- a plausible-looking number for a value the
        // model never produced, in the field callers gate escalation on.
        int width = actLogits.length;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < width; i++) {
            double value = i < actLogits.length ? actLogits[i] : 0.0;
            max = Math.max(max, value);
        }
        double[] exponentials = new double[width];
        double total = 0.0;
        for (int i = 0; i < width; i++) {
            double value = i < actLogits.length ? actLogits[i] : 0.0;
            exponentials[i] = Math.exp(value - max);
            total += exponentials[i];
        }
        float[] out = new float[width];
        for (int i = 0; i < width; i++) {
            out[i] = (float) (exponentials[i] / total);
        }
        return out;
    }

    /**
     * Puts a slot-ordered row back into the caller's option order.
     *
     * <p>A sequence puts option {@code order[slot]} in {@code slot}, so the model's row comes back
     * indexed by slot while everything downstream indexes by option -- the labels for a choice, the
     * level index for a score, {@code p[1]} for a noul. Without this inversion the probabilities
     * attach to the wrong options, and nothing about the answer looks wrong.
     *
     * <p>A missing or mismatched order returns the row untouched, which is what leaves the
     * canonical path unaffected.
     */
    static double[] unpermute(double[] p, int[] order) {
        if (order == null || order.length != p.length) {
            return p;
        }
        double[] canonical = new double[p.length];
        for (int slot = 0; slot < order.length; slot++) {
            canonical[order[slot]] = p[slot];
        }
        return canonical;
    }

    /** {@code max(p)}, clipped: the probability mass on the answer being reported. */
    static double answerConfidence(double[] p, int optionCount) {
        if (optionCount < 1) {
            return 1.0;
        }
        double best = p[0];
        for (int i = 1; i < optionCount; i++) {
            best = Math.max(best, p[i]);
        }
        return Math.min(1.0, Math.max(0.0, best));
    }

    /** Normalised Shannon entropy: {@code 1 - H(p) / log(k)}, clipped to {@code [0, 1]}. */
    static double entropyConfidence(double[] p, int optionCount) {
        if (optionCount < 2) {
            return 1.0;
        }
        double entropy = 0.0;
        for (int i = 0; i < optionCount; i++) {
            // Clipped at 1e-12 before the log, so a zero probability contributes zero rather than
            // negative infinity.
            entropy -= p[i] * Math.log(Math.min(1.0, Math.max(1e-12, p[i])));
        }
        double normalised = 1.0 - entropy / Math.log(optionCount);
        return Math.min(1.0, Math.max(0.0, normalised));
    }

    /** The first index holding the maximum, matching {@code numpy.argmax}'s tie-breaking. */
    static int argmax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }
}
