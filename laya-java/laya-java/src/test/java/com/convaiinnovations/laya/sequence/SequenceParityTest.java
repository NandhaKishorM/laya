package com.convaiinnovations.laya.sequence;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Fixtures;
import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/**
 * Built sequences against {@code laya.common.build_sequence}: ids, markers, option stats and the
 * truncation report.
 *
 * <p>Markers are asserted, not inferred: a marker off by one still produces a plausible answer.
 * The stats and the truncation report are asserted for the same reason -- {@code options_distinct}
 * and {@code state_tokens_dropped} are invisible from the ids alone, and they are what tells a
 * caller that evidence was silently discarded.
 */
final class SequenceParityTest {

    @TestFactory
    @DisplayName("every budget and rendering branch matches Python, on both checkpoints")
    @SuppressWarnings("unchecked")
    List<DynamicTest> sequencesMatch() {
        Assumptions.assumeTrue(Fixtures.checkpoints() != null,
                Fixtures.missingCheckpoint("english and multilingual"));
        Map<String, Object> fixture = Fixtures.load("sequences.json");
        List<DynamicTest> tests = new ArrayList<>();
        for (String name : List.of("english", "multilingual")) {
            Path model = Fixtures.checkpoint(name);
            Map<String, Object> side = (Map<String, Object>) fixture.get(name);
            if (model == null || side == null || side.containsKey("skipped")) {
                continue;
            }
            Tokenizer tokenizer;
            try {
                tokenizer = Tokenizer.fromModelDirectory(model);
            } catch (IOException failure) {
                throw new IllegalStateException("cannot load " + name, failure);
            }
            Map<String, Object> cases = (Map<String, Object>) side.get("cases");
            for (Map.Entry<String, Object> entry : cases.entrySet()) {
                Map<String, Object> c = (Map<String, Object>) entry.getValue();
                tests.add(DynamicTest.dynamicTest(name + "/" + entry.getKey(),
                        () -> check(tokenizer, c)));
            }
        }
        assertTrue(tests.size() > 40, "expected both checkpoints' cases, got " + tests.size());
        return tests;
    }

    @SuppressWarnings("unchecked")
    private static void check(Tokenizer tokenizer, Map<String, Object> c) {
        Question question = questionOf((Map<String, Object>) c.get("question"));
        int[] order = c.get("option_order") == null ? null : Fixtures.ints(c.get("option_order"));
        SequenceBuilder.Sequence built = SequenceBuilder.build(tokenizer, c.get("state"), question,
                ((Number) c.get("max_len")).intValue(),
                ((Number) c.get("head_max_len")).intValue(),
                order, Boolean.TRUE.equals(c.get("truncate_left")), null);

        assertArrayEquals(Fixtures.ints(c.get("ids")), built.ids(), "ids");
        assertArrayEquals(Fixtures.ints(c.get("markers")), built.markers(), "markers");

        Map<String, Object> stats = (Map<String, Object>) c.get("stats");
        assertEquals(((Number) stats.get("options")).intValue(), built.stats().options());
        assertEquals(((Number) stats.get("options_distinct")).intValue(),
                built.stats().optionsDistinct(), "options_distinct");
        Object perOption = stats.get("tokens_per_option");
        assertEquals(perOption == null ? null : ((Number) perOption).intValue(),
                built.stats().tokensPerOption(), "tokens_per_option");

        Map<String, Object> truncation = (Map<String, Object>) c.get("truncation");
        assertEquals(((Number) truncation.get("state_tokens")).intValue(),
                built.truncation().stateTokens(), "state_tokens");
        assertEquals(((Number) truncation.get("state_tokens_used")).intValue(),
                built.truncation().stateTokensUsed(), "state_tokens_used");
        assertEquals(((Number) truncation.get("state_tokens_dropped")).intValue(),
                built.truncation().stateTokensDropped(), "state_tokens_dropped");
        assertEquals(truncation.get("truncated"), built.truncation().truncated(), "truncated");
    }

    @SuppressWarnings("unchecked")
    private static Question questionOf(Map<String, Object> q) {
        String type = (String) q.get("t");
        String instructions = (String) q.get("ins");
        Object criteria = q.get("crit");
        if ("choice".equals(type)) {
            return Question.choice(instructions, (Map<String, Object>) criteria);
        }
        if ("score".equals(type)) {
            return Question.score(instructions, (List<Object>) criteria);
        }
        Map<String, Object> sides = criteria instanceof Map ? (Map<String, Object>) criteria : Map.of();
        Map<String, Object> labels = (Map<String, Object>) q.get("labels");
        Map<String, String> resolved = null;
        if (labels != null) {
            resolved = new LinkedHashMap<>();
            for (Map.Entry<String, Object> label : labels.entrySet()) {
                resolved.put(label.getKey(), (String) label.getValue());
            }
        }
        return Question.noul(instructions, sides.get("false"), sides.get("true"), resolved);
    }
}
