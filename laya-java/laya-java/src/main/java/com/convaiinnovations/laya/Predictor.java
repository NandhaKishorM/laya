package com.convaiinnovations.laya;

import java.util.Map;

/**
 * Anything that can answer a set of questions about a state.
 *
 * <p>Implemented by both {@link Agent}, which asks one checkpoint, and {@link Router}, which picks
 * the checkpoint first. The distinction matters to a caller choosing between them and not at all
 * to code that merely needs an answer -- {@link Shortlist} takes this rather than one or the
 * other, so shortlisting works the same whether the questions go to a fixed checkpoint or a routed
 * one.
 *
 * <p>Deliberately the narrow overload. {@code Agent} also takes a language and {@code Router} also
 * takes routing arguments, and neither belongs here: a {@code Router} chooses the language itself
 * from what it detected, so an interface carrying one would invite a caller to override the very
 * thing routing had just worked out.
 */
public interface Predictor {

    /** Ask every question about one state. */
    Prediction predict(Object state, Map<String, Question> questions);
}
