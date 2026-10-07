package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.Json;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

/**
 * Locates the generated parity fixtures and the optional real checkpoints.
 *
 * <p>The fixtures are committed and are the gate that needs nothing but this repository. The
 * checkpoints are not: a test that needs one announces that and is skipped, rather than passing
 * vacuously when the model is absent. A skipped parity test is not a green parity test, which is
 * why CI asserts a minimum passed count as well as a zero failure count.
 */
public final class Fixtures {

    /** Set to the directory holding {@code english/} and {@code multilingual/} to enable those tests. */
    public static final String CHECKPOINTS_ENV = "LAYA_CHECKPOINTS";

    /** Set to the directory holding {@code laya.onnx} (or {@code encoder.onnx} + {@code head.onnx}). */
    public static final String GRAPH_ENV = "LAYA_ONNX_GRAPH";

    /** Set to a {@code predict.json} recorded from Python with a graph present. */
    public static final String PREDICT_GOLDEN_ENV = "LAYA_PREDICT_GOLDEN";

    private Fixtures() {
    }

    /** The {@code laya-java/fixtures} directory, found by walking up from the working directory. */
    public static Path directory() {
        Path here = Paths.get("").toAbsolutePath();
        for (Path candidate = here; candidate != null; candidate = candidate.getParent()) {
            Path fixtures = candidate.resolve("fixtures");
            if (Files.isDirectory(fixtures) && Files.isRegularFile(fixtures.resolve("decode.json"))) {
                return fixtures;
            }
        }
        throw new IllegalStateException(
                "no laya-java/fixtures directory above " + here
                + "; run `python scripts/gen_fixtures.py` from laya-java");
    }

    /** One fixture family, parsed. */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> load(String name) {
        Path path = directory().resolve(name);
        try (Reader reader = new BufferedReader(
                Files.newBufferedReader(path, StandardCharsets.UTF_8), 1 << 16)) {
            return (Map<String, Object>) Json.parse(reader);
        } catch (IOException failure) {
            throw new UncheckedIOException("cannot read fixture " + path, failure);
        }
    }

    /** The checkpoint root, or null when {@link #CHECKPOINTS_ENV} is unset or wrong. */
    public static Path checkpoints() {
        String configured = System.getenv(CHECKPOINTS_ENV);
        if (configured == null || configured.isBlank()) {
            return null;
        }
        Path root = Paths.get(configured);
        return Files.isDirectory(root) ? root : null;
    }

    /** Whether this checkpoint is present under the configured root. */
    public static Path checkpoint(String name) {
        Path root = checkpoints();
        if (root == null) {
            return null;
        }
        Path model = root.resolve(name);
        return Files.isRegularFile(model.resolve("rl_agent_config.json")) ? model : null;
    }

    /** The message a skipped checkpoint test prints, so the reason is actionable. */
    public static String missingCheckpoint(String name) {
        return "needs the " + name + " checkpoint: set " + CHECKPOINTS_ENV
                + " to the directory holding english/ and multilingual/";
    }

    /** Helper: a list of longs from a fixture array. */
    @SuppressWarnings("unchecked")
    public static int[] ints(Object value) {
        List<Object> items = (List<Object>) value;
        int[] out = new int[items.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) items.get(i)).intValue();
        }
        return out;
    }
}
