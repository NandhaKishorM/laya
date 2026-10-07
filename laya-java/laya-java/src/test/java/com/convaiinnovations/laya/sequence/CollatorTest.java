package com.convaiinnovations.laya.sequence;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Padding a ragged batch into the rectangular tensors the graph takes. */
final class CollatorTest {

    @Test
    @DisplayName("rows pad to the longest sequence and markers to the most options")
    void padsToTheWidest() {
        Collator.Batch batch = Collator.collate(List.of(
                new Collator.Item(new int[] {1, 2, 3}, new int[] {1}, 0),
                new Collator.Item(new int[] {4}, new int[] {0, 1, 2}, 2)), 99);
        assertEquals(2, batch.rows());
        assertEquals(3, batch.length());
        assertEquals(3, batch.markers());
        assertArrayEquals(new long[] {1, 2, 3}, batch.inputIds()[0]);
        // The pad id fills the tail, and the attention mask is what marks it as absent.
        assertArrayEquals(new long[] {4, 99, 99}, batch.inputIds()[1]);
        assertArrayEquals(new long[] {1, 1, 1}, batch.attentionMask()[0]);
        assertArrayEquals(new long[] {1, 0, 0}, batch.attentionMask()[1]);
        assertArrayEquals(new long[] {1, 0, 0}, batch.markerPos()[0]);
        assertArrayEquals(new boolean[] {true, false, false}, batch.markerMask()[0]);
        assertArrayEquals(new boolean[] {true, true, true}, batch.markerMask()[1]);
        assertArrayEquals(new long[] {0, 2}, batch.qtype());
    }

    @Test
    @DisplayName("an empty batch is refused rather than sent to the graph")
    void emptyRefused() {
        assertThrows(IllegalArgumentException.class, () -> Collator.collate(List.of(), 0));
    }

    @Test
    @DisplayName("a marker position of zero is still a live marker, not padding")
    void zeroIsALiveMarker() {
        // `markerPos` is zero-filled, so only `markerMask` can distinguish "position 0" from
        // "no marker". A port that inferred liveness from the position would mask out option 0.
        Collator.Batch batch = Collator.collate(List.of(
                new Collator.Item(new int[] {7}, new int[] {0}, 1)), 0);
        assertArrayEquals(new long[] {0}, batch.markerPos()[0]);
        assertArrayEquals(new boolean[] {true}, batch.markerMask()[0]);
    }
}
