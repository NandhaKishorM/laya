package com.convaiinnovations.laya.decode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Answer;
import com.convaiinnovations.laya.Fixtures;
import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.config.AgentConfig;
import com.convaiinnovations.laya.json.PythonJson;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/**
 * Answer decoding against expectations recorded from laya's own {@code Agent._decode_answers}.
 *
 * <p>Needs no checkpoint and no graph -- the logit rows are synthesised -- so this is the parity
 * gate CI can run on every push. The rows are chosen to hit branches rather than to look
 * realistic: every temperature bucket, a tie at the top, an exactly uniform softmax, magnitudes of
 * +/-700 that overflow a naive {@code exp}, a logit block wider than the option count, permuted
 * option orders, and the per-language override.
 */
final class DecoderParityTest {

    @TestFactory
    @DisplayName("decoded answers match Python, field by field")
    @SuppressWarnings("unchecked")
    List<DynamicTest> matchesPython() {
        Map<String, Object> fixture = Fixtures.load("decode.json");
        List<DynamicTest> tests = new ArrayList<>();
        for (String model : List.of("english", "multilingual")) {
            Map<String, Object> side = (Map<String, Object>) fixture.get(model);
            if (side == null || side.containsKey("skipped")) {
                continue;
            }
            AgentConfig config = configOf((Map<String, Object>) side.get("config"));
            Map<String, Object> cases = (Map<String, Object>) side.get("cases");
            for (Map.Entry<String, Object> entry : cases.entrySet()) {
                Map<String, Object> c = (Map<String, Object>) entry.getValue();
                tests.add(DynamicTest.dynamicTest(model + "/" + entry.getKey(),
                        () -> check(c, config)));
            }
        }
        assertTrue(tests.size() >= 20, "expected both checkpoints' decode cases, got " + tests.size());
        return tests;
    }

    @SuppressWarnings("unchecked")
    private static void check(Map<String, Object> c, AgentConfig config) {
        Question question = questionOf((Map<String, Object>) c.get("question"));
        List<Object> logits = (List<Object>) c.get("logits");
        int optionCount = logits.size();
        int width = ((Number) c.get("logits_block_width")).intValue();
        // Padded out past the option count on purpose: the head's real output is padded to the
        // widest question in the batch with large FINITE numbers, so a port that forgets to slice
        // hands almost all the probability mass to padding.
        float[] row = new float[width];
        for (int i = 0; i < width; i++) {
            row[i] = i < optionCount ? ((Number) logits.get(i)).floatValue() : 99.0f;
        }
        List<Object> act = (List<Object>) c.get("act");
        float[] actionProbabilities = {((Number) act.get(0)).floatValue(),
                ((Number) act.get(1)).floatValue()};
        int[] order = c.get("option_order") == null ? null
                : Fixtures.ints(c.get("option_order"));

        Answer got = Decoder.decode(question, row, optionCount, actionProbabilities, order,
                (String) c.get("lang"), config);
        Map<String, Object> want = (Map<String, Object>) c.get("answer");

        assertEquals(want.get("type"), got.type());
        assertEquals(number(want.get("answer_confidence")), got.answerConfidence(), 0.0,
                "answer_confidence");
        assertEquals(number(want.get("confidence")), got.confidence(), 0.0, "confidence");
        assertEquals(number(((Map<String, Object>) want.get("action")).get("act_probability")),
                got.actProbability(), 0.0, "act_probability");
        if (got instanceof Answer.Choice choice) {
            assertEquals(want.get("choice"), choice.choice());
            assertProbabilities((Map<String, Object>) want.get("probabilities"), choice.probabilities());
        } else if (got instanceof Answer.Score score) {
            assertEquals(number(want.get("score")), score.score(), 0.0, "score");
            assertProbabilities((Map<String, Object>) want.get("probabilities"), score.probabilities());
            Map<String, Object> legend = (Map<String, Object>) want.get("legend");
            assertIterableEquals(legend.keySet(), score.legend().keySet(), "legend key order");
            for (Map.Entry<String, Object> level : legend.entrySet()) {
                assertEquals(level.getValue(), score.legend().get(level.getKey()));
            }
        } else if (got instanceof Answer.Noul noul) {
            assertEquals(number(want.get("noul")), noul.noul(), 0.0, "noul");
        }
    }

    private static void assertProbabilities(Map<String, Object> want, Map<String, Double> got) {
        // Key ORDER, not just the mapping: a choice's probabilities are positional, and
        // `Map.copyOf` would have passed a set comparison while losing the order.
        assertIterableEquals(want.keySet(), got.keySet(), "probability key order");
        for (Map.Entry<String, Object> entry : want.entrySet()) {
            assertEquals(number(entry.getValue()), got.get(entry.getKey()), 0.0,
                    "p[" + entry.getKey() + "]");
        }
    }

    private static double number(Object value) {
        return ((Number) value).doubleValue();
    }

    @SuppressWarnings("unchecked")
    private static AgentConfig configOf(Map<String, Object> node) {
        Map<String, Object> document = new LinkedHashMap<>();
        document.put("temperature", node.get("temperature"));
        document.put("temperature_by_options", node.get("temperature_by_options"));
        document.put("lang_temperatures", node.get("lang_temperatures_raw"));
        return AgentConfig.fromJson(PythonJson.dumps(document));
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
        Map<String, Object> sides = criteria instanceof Map
                ? (Map<String, Object>) criteria : Map.of();
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
