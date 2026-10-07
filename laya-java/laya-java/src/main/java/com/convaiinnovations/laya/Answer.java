package com.convaiinnovations.laya;

import java.util.Map;

/**
 * One typed answer. Sealed, because the three shapes are not interchangeable and a caller should
 * not have to test a string to find out which fields are present.
 *
 * <p><b>Two confidences, deliberately.</b> {@link #answerConfidence()} is the probability mass on
 * the reported answer -- {@code max(p)} -- and it is the quantity temperature scaling fits and
 * every calibration figure in laya is computed on. {@link #confidence()} means something different
 * per type: normalised entropy for a choice or a score, and {@code max(p, 1-p)} for a noul. They
 * are on different scales and carry different guarantees, so <b>they must not be compared against
 * the same threshold</b>.
 *
 * <p>The calibration guarantee on {@code answerConfidence} is conditional and the condition is not
 * met by default: it holds only after the temperatures have been fitted and validated on held-out
 * data for that checkpoint and option count. The shipped checkpoints are over-confident.
 *
 * <p>Every number is rounded to four decimals, as Python rounds them.
 */
public sealed interface Answer {

    /** The wire name of this answer's type: {@code choice}, {@code score} or {@code noul}. */
    String type();

    /** The calibrated confidence: the probability mass on the answer being reported. */
    double answerConfidence();

    /** The per-type confidence. See the class note: not comparable across types. */
    double confidence();

    /** The head's second output: the probability that this decision should be escalated. */
    double actProbability();

    /** A pick among labelled options, with the probability of each. */
    record Choice(String choice, Map<String, Double> probabilities, double confidence,
                  double answerConfidence, double actProbability) implements Answer {
        @Override
        public String type() {
            return "choice";
        }
    }

    /**
     * A rating on an ordered scale.
     *
     * @param score  the expected value over level indices, not the most likely level
     * @param legend each level index mapped to the text the model was shown for it
     */
    record Score(double score, Map<String, String> legend, Map<String, Double> probabilities,
                 double confidence, double answerConfidence, double actProbability)
            implements Answer {
        @Override
        public String type() {
            return "score";
        }
    }

    /** Whether a statement holds: {@code noul} is the probability that it does. */
    record Noul(double noul, double confidence, double answerConfidence, double actProbability)
            implements Answer {
        @Override
        public String type() {
            return "noul";
        }
    }
}
