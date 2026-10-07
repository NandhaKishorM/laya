package com.convaiinnovations.laya;

import com.convaiinnovations.laya.config.AgentConfig;
import com.convaiinnovations.laya.infer.InferenceSession;
import com.convaiinnovations.laya.sequence.Collator;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A synthetic checkpoint, so the parts that have nothing to do with a model can be tested without
 * one.
 *
 * <p>Batching, row offsets, usage accounting and result ordering are the most likely things to be
 * wrong and the least related to inference. Requiring a 1.2 GB graph to exercise them would keep
 * them out of any CI that does not download a checkpoint, which is exactly where a regression
 * would then live.
 *
 * <p>The tokenizer is real, not a mock: a byte-level model whose vocabulary is the 243 single-byte
 * symbols text can reach, so every input encodes and the ids are predictable (one id per UTF-8
 * byte, with no merges).
 */
public final class TinyCheckpoint {

    /** Ids assigned to the four special tokens, above the byte alphabet. */
    public static final int CLS = 243;
    public static final int SEP = 244;
    public static final int MASK = 245;
    public static final int PAD = 246;

    private TinyCheckpoint() {
    }

    /** Writes {@code tokenizer/} and {@code rl_agent_config.json} under {@code root}. */
    public static void write(Path root, int maxLen, int headMaxLen) throws IOException {
        Path tokenizerDir = root.resolve("tokenizer");
        Files.createDirectories(tokenizerDir);
        Files.writeString(tokenizerDir.resolve("tokenizer.json"), tokenizerJson(),
                StandardCharsets.UTF_8);
        Files.writeString(tokenizerDir.resolve("tokenizer_config.json"), """
                {"cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]",
                 "pad_token": "[PAD]"}
                """, StandardCharsets.UTF_8);
        Files.writeString(root.resolve("rl_agent_config.json"),
                "{\"max_len\": " + maxLen + ", \"head_max_len\": " + headMaxLen
                + ", \"temperature\": [1.0, 1.0, 1.0], \"temperature_by_options\": {}}",
                StandardCharsets.UTF_8);
    }

    /** Opens an agent over a synthetic checkpoint and the given session. */
    public static Agent agent(Path root, InferenceSession session) throws IOException {
        return Agent.using(Tokenizer.fromModelDirectory(root),
                AgentConfig.fromModelDirectory(root), session);
    }

    /**
     * A byte-level BPE tokenizer with no merges, so one UTF-8 byte is one token.
     *
     * <p>No merges is what makes the expected ids computable by hand in a test: a sequence's length
     * is the byte length of its text plus its special tokens.
     */
    private static String tokenizerJson() {
        char[] byteToChar = byteLevelAlphabet();
        StringBuilder vocab = new StringBuilder();
        int next = 0;
        for (int b = 0; b < 256; b++) {
            if (b == 0xC0 || b == 0xC1 || b >= 0xF5) {
                continue;   // unreachable from valid UTF-8; see PreTokenizer.ByteLevel
            }
            if (next > 0) {
                vocab.append(", ");
            }
            vocab.append('"').append(escape(byteToChar[b])).append("\": ").append(next++);
        }
        // Specials live in added_tokens, as they do in the english checkpoint, where 88 of them are
        // absent from model.vocab entirely.
        String added = """
                [{"id": 243, "content": "[CLS]", "special": true, "lstrip": false,
                  "rstrip": false, "single_word": false, "normalized": false},
                 {"id": 244, "content": "[SEP]", "special": true, "lstrip": false,
                  "rstrip": false, "single_word": false, "normalized": false},
                 {"id": 245, "content": "[MASK]", "special": true, "lstrip": false,
                  "rstrip": false, "single_word": false, "normalized": false},
                 {"id": 246, "content": "[PAD]", "special": true, "lstrip": false,
                  "rstrip": false, "single_word": false, "normalized": false}]""";
        return "{\"version\": \"1.0\", \"added_tokens\": " + added
                + ", \"normalizer\": null"
                + ", \"pre_tokenizer\": {\"type\": \"ByteLevel\", \"add_prefix_space\": false,"
                + " \"trim_offsets\": true, \"use_regex\": false}"
                + ", \"model\": {\"type\": \"BPE\", \"vocab\": {" + vocab + "}, \"merges\": [],"
                + " \"unk_token\": null, \"fuse_unk\": false, \"byte_fallback\": false,"
                + " \"ignore_merges\": false}}";
    }

    private static char[] byteLevelAlphabet() {
        boolean[] printable = new boolean[256];
        for (int b = '!'; b <= '~'; b++) {
            printable[b] = true;
        }
        for (int b = 0xA1; b <= 0xAC; b++) {
            printable[b] = true;
        }
        for (int b = 0xAE; b <= 0xFF; b++) {
            printable[b] = true;
        }
        char[] table = new char[256];
        int lifted = 0;
        for (int b = 0; b < 256; b++) {
            table[b] = printable[b] ? (char) b : (char) (256 + lifted++);
        }
        return table;
    }

    private static String escape(char c) {
        if (c == '"') {
            return "\\\"";
        }
        if (c == '\\') {
            return "\\\\";
        }
        return c < 0x20 ? String.format("\\u%04x", (int) c) : String.valueOf(c);
    }

    /**
     * A session whose output is derived from each row's CONTENT, and which records every batch.
     *
     * <p>Content, not the row's index in the batch. That distinction is the whole point: a real
     * model's output for a row depends on the row, not on which other rows it was padded next to,
     * so a stub keyed on the index would make "the answers are the same at every batch size"
     * trivially false -- it was, and the test caught it. Keyed on content, the same assertion
     * actually tests the batching.
     *
     * <p>The value still varies per row (different questions produce different ids), and rises
     * with the option slot, so an off-by-one in the row offsets changes an answer and a test can
     * see it.
     */
    public static class RecordingSession implements InferenceSession {
        // Not final: a test subclasses it to perturb state mid-call, which is how the
        // questions-map snapshot is proved rather than asserted.

        public final List<Collator.Batch> batches = new ArrayList<>();
        public boolean closed;

        @Override
        public Output run(Collator.Batch batch) {
            batches.add(batch);
            float[][] logits = new float[batch.rows()][Math.max(1, batch.markers())];
            float[][] act = new float[batch.rows()][2];
            for (int r = 0; r < batch.rows(); r++) {
                int markers = 0;
                for (boolean live : batch.markerMask()[r]) {
                    if (live) {
                        markers++;
                    }
                }
                float base = fingerprint(batch, r);
                for (int m = 0; m < logits[r].length; m++) {
                    // Monotone in the slot so the argmax is the last live option, and offset by the
                    // row's own fingerprint so two questions never share a row's output.
                    logits[r][m] = m < markers ? base + m : 99.0f;
                }
                act[r][0] = base;
                act[r][1] = 0.0f;
            }
            return new Output(logits, act);
        }

        /**
         * A small, stable number derived from the row's live tokens and markers.
         *
         * <p>Padding is excluded deliberately: it is the one part of a row that DOES change with
         * batch composition, and including it would reintroduce exactly the dependence this exists
         * to avoid.
         */
        private static float fingerprint(Collator.Batch batch, int row) {
            long hash = 0x811C9DC5L;
            for (int i = 0; i < batch.length(); i++) {
                if (batch.attentionMask()[row][i] == 0L) {
                    continue;
                }
                hash = (hash ^ batch.inputIds()[row][i]) * 0x01000193L & 0xFFFFFFFFL;
            }
            for (int i = 0; i < batch.markers(); i++) {
                if (batch.markerMask()[row][i]) {
                    hash = (hash ^ (batch.markerPos()[row][i] + 1)) * 0x01000193L & 0xFFFFFFFFL;
                }
            }
            hash = (hash ^ batch.qtype()[row]) * 0x01000193L & 0xFFFFFFFFL;
            // A narrow range keeps the softmax away from saturation, so probabilities stay
            // informative rather than collapsing to 1.0 and 0.0.
            return (hash % 1000) / 1000.0f;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    /** A map literal that keeps insertion order, which a choice question requires. */
    public static Map<String, Object> ordered(Object... pairs) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            out.put((String) pairs[i], pairs[i + 1]);
        }
        return out;
    }
}
