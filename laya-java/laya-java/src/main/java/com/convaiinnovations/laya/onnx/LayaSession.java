package com.convaiinnovations.laya.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.convaiinnovations.laya.infer.InferenceException;
import com.convaiinnovations.laya.infer.InferenceSession;
import com.convaiinnovations.laya.sequence.Collator;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The ONNX graph, in either of the two forms laya exports.
 *
 * <ul>
 *   <li><b>Fused</b> {@code laya.onnx}: one graph taking {@code input_ids}, {@code attention_mask},
 *       {@code marker_pos}, {@code marker_mask}, {@code qtype} and returning {@code logits} and
 *       {@code act_logits}. This is what {@code ONNXAgent} loads by default.
 *   <li><b>Split</b> {@code encoder.onnx} + {@code head.onnx}: the encoder returns
 *       {@code last_hidden_state}, which the head consumes with the marker inputs.
 * </ul>
 *
 * <p>Both are supported because both exist in the wild -- {@code laya-dotnet} carries the same pair
 * of paths -- and which one a directory holds is discoverable, so there is no reason to make the
 * caller say. Instances are not thread-safe for concurrent {@code run} calls on one session unless
 * ONNX Runtime is configured for it; hold one per worker or serialise access.
 */
public final class LayaSession implements InferenceSession {

    /**
     * The outputs to ask the graph for.
     *
     * <p>The set only selects WHICH outputs are computed; nothing may depend on its order, and
     * {@link #floats} resolves by name for that reason.
     */
    private static final Set<String> REQUESTED = Set.of("logits", "act_logits");

    private final OrtEnvironment environment;
    private final OrtSession fused;
    private final OrtSession encoder;
    private final OrtSession head;
    private boolean closed;

    private LayaSession(OrtEnvironment environment, OrtSession fused, OrtSession encoder,
                        OrtSession head) {
        this.environment = environment;
        this.fused = fused;
        this.encoder = encoder;
        this.head = head;
    }

    /**
     * Opens whichever graph form {@code directory} holds, preferring the fused one.
     *
     * @param threads intra-op threads, or <b>0 to let ONNX Runtime decide</b>, which is what the
     *                Python runtime does by not setting it. Pinning this to 1 made a single
     *                prediction about 3x slower than Python on the same graph -- the forward pass
     *                is the whole cost and it was running on one core. Pass 1 deliberately for a
     *                request-per-thread server, where the parallelism is already in the requests
     */
    public static LayaSession open(Path directory, int threads) throws IOException {
        Path fusedPath = directory.resolve("laya.onnx");
        Path encoderPath = directory.resolve("encoder.onnx");
        Path headPath = directory.resolve("head.onnx");
        boolean fusedPresent = Files.isRegularFile(fusedPath);
        boolean splitPresent = Files.isRegularFile(encoderPath) && Files.isRegularFile(headPath);
        if (!fusedPresent && !splitPresent) {
            // An IOException, because this one IS the caller's problem: they pointed at the wrong
            // directory. A backend failure further down is an InferenceException instead.
            throw new IOException(
                    "no laya graph in " + directory + ": expected laya.onnx, or encoder.onnx and "
                    + "head.onnx");
        }
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        try {
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            if (threads > 0) {
                options.setIntraOpNumThreads(threads);
            }
            // Operator fusion and constant folding. Functionally neutral and free at inference
            // time; without it ONNX Runtime runs the unoptimised graph. The Python runtime sets
            // the same level and records ~1.45x on CPU for it, so omitting it here was a 1.45x
            // regression against the reference for no reason.
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            if (fusedPresent) {
                return new LayaSession(environment,
                        environment.createSession(fusedPath.toString(), options), null, null);
            }
            return new LayaSession(environment, null,
                    environment.createSession(encoderPath.toString(), options),
                    environment.createSession(headPath.toString(), options));
        } catch (OrtException failure) {
            throw new InferenceException("could not load the laya graph in " + directory, failure);
        }
    }

    /** Whether this session is the fused single-graph form. */
    public boolean isFused() {
        return fused != null;
    }

    /** Runs one collated batch. */
    @Override
    public Output run(Collator.Batch batch) {
        if (closed) {
            // The guard isClosed()'s javadoc promises ("so a caller gets an API error, not a
            // backend one") and which did not exist: a use-after-close surfaced ORT's own
            // IllegalStateException, outside both the OrtException catch below and
            // InferenceException -- precisely the one error class this API says it never emits.
            throw new InferenceException("this laya session is closed");
        }
        List<OnnxTensor> owned = new ArrayList<>();
        try {
            OnnxTensor inputIds = track(owned, OnnxTensor.createTensor(environment, batch.inputIds()));
            OnnxTensor attention = track(owned, OnnxTensor.createTensor(environment, batch.attentionMask()));
            OnnxTensor markerPos = track(owned, OnnxTensor.createTensor(environment, batch.markerPos()));
            OnnxTensor markerMask = track(owned, OnnxTensor.createTensor(environment, batch.markerMask()));
            OnnxTensor qtype = track(owned, OnnxTensor.createTensor(environment, batch.qtype()));
            if (fused != null) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputIds);
                inputs.put("attention_mask", attention);
                inputs.put("marker_pos", markerPos);
                inputs.put("marker_mask", markerMask);
                inputs.put("qtype", qtype);
                try (OrtSession.Result result = fused.run(inputs, REQUESTED)) {
                    return new Output(floats(result, "logits"), floats(result, "act_logits"));
                }
            }
            Map<String, OnnxTensor> encoderInputs = new HashMap<>();
            encoderInputs.put("input_ids", inputIds);
            encoderInputs.put("attention_mask", attention);
            try (OrtSession.Result encoded = encoder.run(encoderInputs,
                    Set.of("last_hidden_state"))) {
                OnnxTensor hidden = (OnnxTensor) encoded.get(0);
                Map<String, OnnxTensor> headInputs = new HashMap<>();
                headInputs.put("hidden_states", hidden);
                headInputs.put("marker_pos", markerPos);
                headInputs.put("marker_mask", markerMask);
                // [B, 1], not [B]. The exporter declares the head's qtype input with a
                // trailing dimension of 1 (it squeezes it back off internally), so a rank-1
                // tensor is refused outright: "Invalid rank for input: qtype Got: 1 Expected: 2".
                // Every split export the only exporter produces therefore failed, and the error
                // arrived as a per-batch backend failure that reads like a bad model rather than
                // a bad client. laya-ts passes [n, 1] here for the same reason.
                long[] flat = batch.qtype();
                long[][] column = new long[flat.length][1];
                for (int row = 0; row < flat.length; row++) {
                    column[row][0] = flat[row];
                }
                OnnxTensor qtypeColumn = track(owned,
                        OnnxTensor.createTensor(environment, column));
                headInputs.put("qtype", qtypeColumn);
                headInputs.put("attention_mask", attention);
                try (OrtSession.Result result = head.run(headInputs, REQUESTED)) {
                    return new Output(floats(result, "logits"), floats(result, "act_logits"));
                }
            }
        } catch (OrtException failure) {
            throw new InferenceException(
                    "the laya graph failed on a batch of " + batch.rows() + " rows x "
                    + batch.length() + " tokens", failure);
        } finally {
            for (OnnxTensor tensor : owned) {
                tensor.close();
            }
        }
    }

    private static OnnxTensor track(List<OnnxTensor> owned, OnnxTensor tensor) {
        owned.add(tensor);
        return tensor;
    }

    /**
     * One named output, resolved BY NAME.
     *
     * <p>Not by position. {@code OrtSession.Result}'s positional index follows the order of the
     * set of names that was <i>requested</i>, and {@code Set.of} with two or more elements
     * randomises its iteration order per JVM process (its salt is seeded from
     * {@code System.nanoTime()}). Reading {@code get(0)} and {@code get(1)} therefore returned
     * {@code logits} and {@code act_logits} in one process and swapped them in the next --
     * measured at 4 and 6 out of 10 fresh JVMs.
     *
     * <p>That was silent wherever every question in the batch had exactly two options, which is
     * every {@code noul} and any two-option choice: both outputs are then two columns wide, so no
     * shape check could notice, and the answer simply inverted. On the real graph it turned "the
     * customer was charged more than once" from true at 0.9972 into false at confidence 1.0000.
     * The Python runtime is immune because it passes an ordered list, not a set.
     */
    private static float[][] floats(OrtSession.Result result, String name) throws OrtException {
        OnnxValue value = result.get(name).orElseThrow(() -> new InferenceException(
                "the laya graph produced no output named " + name));
        if (!(value instanceof OnnxTensor)) {
            throw new InferenceException(
                    "the laya graph's " + name + " output is not a tensor but a "
                    + value.getClass().getSimpleName());
        }
        Object raw;
        try {
            raw = ((OnnxTensor) value).getValue();
        } catch (OrtException failure) {
            throw new InferenceException(
                    "the laya graph's " + name + " output could not be read", failure);
        }
        if (!(raw instanceof float[][])) {
            // Checked rather than cast. An fp16/fp64 export, or a re-export that changes the rank,
            // produced a ClassCastException naming neither the graph nor the output -- through a
            // method whose javadoc promises the backend never leaks, so a caller's
            // catch (InferenceException) did not catch it.
            throw new InferenceException(String.format(
                    "the laya graph's %s output must be a 2-D float tensor, got %s;"
                    + " re-export the graph with float32 logits of shape [batch, markers]",
                    name, raw == null ? "null" : raw.getClass().getSimpleName()));
        }
        return (float[][]) raw;
    }

    /** Whether this session has been closed, so a caller gets an API error, not a backend one. */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Closes every session this holds.
     *
     * <p>Idempotent, because {@code AutoCloseable} says so and because try-with-resources plus an
     * explicit {@code close()} is ordinary code -- ONNX Runtime itself throws
     * "Trying to close an already closed OrtSession" on the second call.
     *
     * <p>Every session is closed even when one of them throws: in the split form a failure closing
     * the encoder used to leave the head open, leaking native memory for the life of the process.
     * The first failure is reported and the rest are attached to it.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        InferenceException failure = null;
        for (OrtSession session : new OrtSession[] {fused, encoder, head}) {
            if (session == null) {
                continue;
            }
            try {
                session.close();
            } catch (OrtException | RuntimeException problem) {
                if (failure == null) {
                    failure = new InferenceException("closing the laya graph failed", problem);
                } else {
                    failure.addSuppressed(problem);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }
}
