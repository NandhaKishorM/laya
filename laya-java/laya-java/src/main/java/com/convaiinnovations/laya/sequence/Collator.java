package com.convaiinnovations.laya.sequence;

import java.util.List;

/**
 * Pads a batch of built sequences into the rectangular tensors the graph takes.
 *
 * <p>One row per question, not per state: a request asking four questions about one document is
 * four rows that share a state. Rows are padded to the longest sequence in the batch and markers to
 * the most options, with {@code attentionMask} and {@code markerMask} marking what is real.
 */
public final class Collator {

    /** The five inputs the graph takes, flattened row-major. */
    public record Batch(long[][] inputIds, long[][] attentionMask, long[][] markerPos,
                        boolean[][] markerMask, long[] qtype, int rows, int length, int markers) {
    }

    /** One row to collate: a built sequence and the question type that produced it. */
    public record Item(int[] ids, int[] markers, int qtype) {
    }

    private Collator() {
    }

    /**
     * @param items the rows, in the order their answers will be read back
     * @param padId the tokenizer's padding id. Padding is masked out, but it is still fed to the
     *              encoder, so an id outside the vocabulary would be an out-of-range embedding
     *              lookup rather than a harmless filler
     */
    public static Batch collate(List<Item> items, int padId) {
        if (items.isEmpty()) {
            throw new IllegalArgumentException("nothing to collate");
        }
        int rows = items.size();
        int length = 0;
        int markerWidth = 0;
        for (Item item : items) {
            length = Math.max(length, item.ids().length);
            markerWidth = Math.max(markerWidth, item.markers().length);
        }
        long[][] inputIds = new long[rows][length];
        long[][] attentionMask = new long[rows][length];
        long[][] markerPos = new long[rows][markerWidth];
        boolean[][] markerMask = new boolean[rows][markerWidth];
        long[] qtype = new long[rows];
        for (int r = 0; r < rows; r++) {
            Item item = items.get(r);
            for (int i = 0; i < length; i++) {
                inputIds[r][i] = i < item.ids().length ? item.ids()[i] : padId;
                attentionMask[r][i] = i < item.ids().length ? 1L : 0L;
            }
            for (int i = 0; i < item.markers().length; i++) {
                markerPos[r][i] = item.markers()[i];
                markerMask[r][i] = true;
            }
            qtype[r] = item.qtype();
        }
        return new Batch(inputIds, attentionMask, markerPos, markerMask, qtype,
                rows, length, markerWidth);
    }
}
