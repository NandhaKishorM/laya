package com.convaiinnovations.laya.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Fixtures;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/**
 * Token ids against expectations recorded from the real HuggingFace tokenizer.
 *
 * <p>Ids, not answers: a tokenizer divergence that changes an answer is almost impossible to
 * localise from the answer and trivial from the ids.
 *
 * <p>Needs the checkpoints, so it is skipped without them -- and a skip is not a pass, which is why
 * CI asserts a minimum passed count for this class rather than only a zero failure count.
 */
final class TokenizerParityTest {

    @TestFactory
    @DisplayName("every corpus entry is byte-identical, on both checkpoints")
    @SuppressWarnings("unchecked")
    List<DynamicTest> corpusMatches() {
        Assumptions.assumeTrue(Fixtures.checkpoints() != null,
                Fixtures.missingCheckpoint("english and multilingual"));
        Map<String, Object> fixture = Fixtures.load("tokenizer_ids.json");
        Map<String, Object> corpusText = (Map<String, Object>) fixture.get("corpus_text");
        List<DynamicTest> tests = new ArrayList<>();
        for (String name : List.of("english", "multilingual")) {
            Path model = Fixtures.checkpoint(name);
            if (model == null) {
                continue;
            }
            Map<String, Object> side = (Map<String, Object>) fixture.get(name);
            if (side == null || side.containsKey("skipped")) {
                continue;
            }
            Tokenizer tokenizer;
            try {
                tokenizer = Tokenizer.fromModelDirectory(model);
            } catch (IOException failure) {
                throw new IllegalStateException("cannot load " + name, failure);
            }
            tests.add(DynamicTest.dynamicTest(name + "/identity", () -> {
                Map<String, Object> identity = (Map<String, Object>) side.get("identity");
                assertEquals("BPE", identity.get("model_type"));
                // The fixture records what the checkpoint IS, so a fixture generated against a
                // different checkpoint cannot be mistaken for a matching one.
                assertEquals(((Number) identity.get("vocab_size")).intValue()
                                + countAdded(side), tokenizer.vocabSize(),
                        "vocab plus added tokens");
            }));
            tests.add(DynamicTest.dynamicTest(name + "/specials", () -> {
                Map<String, Object> specials = (Map<String, Object>) side.get("special_ids");
                assertEquals(((Number) specials.get("cls")).intValue(), tokenizer.clsId().orElseThrow());
                assertEquals(((Number) specials.get("sep")).intValue(), tokenizer.sepId().orElseThrow());
                assertEquals(((Number) specials.get("mask")).intValue(), tokenizer.maskId().orElseThrow());
                assertEquals(((Number) specials.get("pad")).intValue(), tokenizer.padId().orElseThrow());
                assertEquals(specials.get("mask_token"), tokenizer.maskToken());
            }));
            Map<String, Object> corpus = (Map<String, Object>) side.get("corpus");
            for (Map.Entry<String, Object> entry : corpus.entrySet()) {
                String text = (String) corpusText.get(entry.getKey());
                int[] want = Fixtures.ints(entry.getValue());
                tests.add(DynamicTest.dynamicTest(name + "/" + entry.getKey(),
                        () -> assertArrayEquals(want, tokenizer.encode(text),
                                "ids for corpus entry " + entry.getKey())));
            }
            Map<String, Object> truncation = (Map<String, Object>) side.get("option_truncation");
            tests.add(DynamicTest.dynamicTest(name + "/option-truncation-48", () -> {
                String text = " " + truncation.get("text");
                assertArrayEquals(Fixtures.ints(truncation.get("uncapped")),
                        tokenizer.encode(text, -1), "uncapped");
                // The cap is applied AT the tokenizer, which must equal slicing the full encoding.
                assertArrayEquals(Fixtures.ints(truncation.get("capped_48")),
                        tokenizer.encode(text, 48), "capped at 48");
            }));
        }
        assertTrue(tests.size() > 40, "expected the full corpus, got " + tests.size());
        return tests;
    }

    @SuppressWarnings("unchecked")
    private static int countAdded(Map<String, Object> side) {
        // The fixture's vocab_size counts model.vocab only; the english checkpoint keeps 88 of its
        // added tokens outside it, so the tokenizer's own size is the union.
        Map<String, Object> identity = (Map<String, Object>) side.get("identity");
        Object added = identity.get("added_tokens_outside_vocab");
        return added == null ? 0 : ((Number) added).intValue();
    }
}
