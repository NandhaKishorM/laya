package com.convaiinnovations.laya.infer;

import com.convaiinnovations.laya.sequence.Collator;

/**
 * The seam between the runtime and whatever actually evaluates the graph.
 *
 * <p>An interface rather than the ONNX session directly, for two reasons that both paid for
 * themselves:
 *
 * <ul>
 *   <li><b>The batching and usage accounting become testable.</b> Those are the parts most likely
 *       to be wrong -- row offsets across states, the filtered marker count, which question gets
 *       blamed for a truncation -- and they have nothing to do with ONNX. A stub implementation
 *       exercises them in milliseconds, where the real graph is 1.2 GB and would keep them out of
 *       any CI that does not download a checkpoint.
 *   <li><b>The backend stops leaking.</b> {@code OrtException} is a checked exception from one
 *       library; without this seam it appears in the signature of every public {@code predict}
 *       method and in every caller's try/catch.
 * </ul>
 *
 * <p>Implementations are not required to be safe for concurrent {@link #run} calls.
 */
public interface InferenceSession extends AutoCloseable {

    /** One batch's head outputs, row-aligned with the batch that produced them. */
    record Output(float[][] logits, float[][] actLogits) {
    }

    /**
     * Evaluates one collated batch.
     *
     * @throws InferenceException if the backend fails
     */
    Output run(Collator.Batch batch);

    @Override
    void close();
}
