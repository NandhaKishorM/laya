package com.convaiinnovations.laya.decode;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

final class DecoderTest {

    @Test
    @DisplayName("empty action logits produce neutral probabilities")
    void emptyActionLogitsAreNeutral() {
        assertArrayEquals(new float[]{0.5f, 0.5f},
                Decoder.actionProbabilities(new float[0]), 0.0f);
    }
}
