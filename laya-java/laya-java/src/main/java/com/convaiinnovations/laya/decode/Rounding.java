package com.convaiinnovations.laya.decode;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * The four-decimal rounding every number in an answer goes through.
 *
 * <p>laya rounds with Python's {@code round(value, 4)}, which rounds half to <b>even</b> on the
 * exact binary value of the double. Two spellings were considered:
 *
 * <ul>
 *   <li>{@code Math.round(v * 1e4) / 1e4}, which is what {@code laya-ts} uses
 *       ({@code agent.ts:380}). It rounds half <b>up</b>, and the multiply by {@code 1e4} is itself
 *       inexact, so it can disagree with Python in both directions.
 *   <li>{@link BigDecimal} at scale 4 with {@link RoundingMode#HALF_EVEN}, which is Python's rule
 *       applied to Python's operand -- {@code new BigDecimal(double)} takes the exact binary value,
 *       not a decimal approximation of it.
 * </ul>
 *
 * <p>The second is used, and it is a deliberate departure from {@code laya-ts}: the contract this
 * port is held to is parity with the Python reference, and matching it exactly removes a
 * "happens not to diverge" caveat rather than inheriting one. Measured over 209,272 values --
 * probabilities, real softmax rows, and exact {@code .00005} steps engineered to land on a halfway
 * case -- against {@code round(v, 4)} from CPython 3.12.
 */
public final class Rounding {

    private Rounding() {
    }

    /** {@code value} rounded to four decimals the way Python's {@code round(value, 4)} rounds. */
    public static double round4(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            // Python returns NaN / inf unchanged rather than raising.
            return value;
        }
        double rounded = new BigDecimal(value).setScale(4, RoundingMode.HALF_EVEN).doubleValue();
        // BigDecimal has no signed zero, so -1e-5 came back as +0.0 where CPython's round gives
        // -0.0. The class claims exact parity, and a record's equals() distinguishes the two, so
        // a serialized-answer comparison would disagree. Restored from the operand's sign.
        if (rounded == 0.0 && (Double.doubleToRawLongBits(value) & Long.MIN_VALUE) != 0) {
            return -0.0;
        }
        return rounded;
    }

    /**
     * The {@code laya-ts} spelling, kept only so the two can be compared in a test.
     *
     * <p>Not used by the runtime. It exists so the claim that the two agree on softmax output, and
     * disagree on engineered halfway cases, is a measurement in this repository rather than an
     * assertion in a comment.
     */
    public static double round4HalfUp(double value) {
        return Math.round(value * 1e4) / 1e4;
    }
}
