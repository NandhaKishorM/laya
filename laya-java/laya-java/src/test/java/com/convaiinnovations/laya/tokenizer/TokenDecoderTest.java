package com.convaiinnovations.laya.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.convaiinnovations.laya.Fixtures;
import com.convaiinnovations.laya.json.Json;
import com.convaiinnovations.laya.json.PythonJson;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@code Tokenizer.decode}, against expectations recorded from the reference tokenizer.
 *
 * <p>Decoding has no Python reference to port -- it lives in the Rust {@code tokenizers} crate --
 * so {@code fixtures/decode_text.json} is the only specification this port has, and the three
 * rules it pins were each wrong on the first attempt. The counts in the comments below are what
 * each wrong guess cost against the reference, so a later simplification has a number to argue
 * with rather than a preference.
 *
 * <p>The SLICE cases carry the weight. {@code predictLong} cuts a tokenized state into windows at
 * arbitrary offsets and decodes each one, so a boundary lands inside a multi-byte character
 * routinely -- and every whole-text decode agreed while 629 of 201,279 slices did not. A fixture
 * of whole texts would have shipped the bug.
 */
final class TokenDecoderTest {

    @SuppressWarnings("unchecked")
    private static Map<String, Object> family() {
        return (Map<String, Object>) (Map<?, ?>) Fixtures.load("decode_text.json");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> forCheckpoint(String name) {
        Map<String, Object> by = (Map<String, Object>) family().get("by");
        Map<String, Object> entry = (Map<String, Object>) by.get(name);
        Assumptions.assumeTrue(entry != null && !entry.containsKey("skipped"),
                "the " + name + " checkpoint was not present when the fixture was recorded");
        return entry;
    }

    private static Tokenizer open(String name) {
        Path root = Fixtures.checkpoint(name);
        Assumptions.assumeTrue(root != null, "set LAYA_CHECKPOINTS to run the decode parity tests");
        try {
            return Tokenizer.fromFile(root.resolve("tokenizer").resolve("tokenizer.json"));
        } catch (java.io.IOException problem) {
            throw new IllegalStateException("cannot open the " + name + " tokenizer", problem);
        }
    }

    @TestFactory
    @DisplayName("every recorded decode matches, whole and sliced, on both checkpoints")
    @SuppressWarnings("unchecked")
    List<DynamicTest> recordedDecodesMatch() {
        List<DynamicTest> tests = new ArrayList<>();
        int sliceCases = 0;
        for (String name : List.of("english", "multilingual")) {
            Map<String, Object> entry = forCheckpoint(name);
            List<Object> cases = (List<Object>) entry.get("cases");
            assertTrue(cases.size() >= 23,
                    () -> "the decode corpus lost cases: " + cases.size() + " of at least 23");
            Tokenizer tok = open(name);
            for (Object raw : cases) {
                Map<String, Object> row = (Map<String, Object>) raw;
                String text = (String) row.get("text");
                List<Object> wantIds = (List<Object>) row.get("ids");
                String label = name + " " + describe(text);

                tests.add(dynamicTest(label + " ids", () -> {
                    int[] ids = tok.encode(text);
                    assertEquals(wantIds.size(), ids.length, "id count for " + describe(text));
                    for (int i = 0; i < ids.length; i++) {
                        assertEquals(((Number) wantIds.get(i)).intValue(), ids[i],
                                "id " + i + " for " + describe(text));
                    }
                }));

                tests.add(dynamicTest(label + " whole", () ->
                        assertEquals(row.get("decoded"), tok.decode(ids(wantIds)),
                                "decode diverged on " + describe(text))));

                List<Object> slices = (List<Object>) row.get("slices");
                sliceCases += slices.size();
                if (slices.isEmpty()) {
                    // No slices means no ids, which is true of exactly one corpus text. Asserting
                    // THAT is worth a test; registering a loop over nothing is not -- it passed
                    // unconditionally and still counted toward the suite's floor.
                    tests.add(dynamicTest(label + " has no ids to slice", () -> {
                        assertTrue(text.isEmpty(),
                                () -> "only the empty text should tokenize to nothing, but "
                                        + describe(text) + " did");
                        assertEquals(0, ids(wantIds).length, "and it must have no ids");
                    }));
                    continue;
                }
                tests.add(dynamicTest(label + " slices (" + slices.size() + ")", () -> {
                    int[] ids = ids(wantIds);
                    for (Object s : slices) {
                        Map<String, Object> slice = (Map<String, Object>) s;
                        int start = ((Number) slice.get("start")).intValue();
                        int size = ((Number) slice.get("size")).intValue();
                        int end = Math.min(start + size, ids.length);
                        int[] chunk = new int[end - start];
                        System.arraycopy(ids, start, chunk, 0, chunk.length);
                        assertEquals(slice.get("decoded"), tok.decode(chunk),
                                () -> String.format("slice [%d,%d) of %s", start, end,
                                        describe(text)));
                    }
                }));

                Map<String, Object> specials = (Map<String, Object>) row.get("with_specials");
                tests.add(dynamicTest(label + " specials", () -> {
                    int[] ids = ids((List<Object>) specials.get("ids"));
                    assertEquals(specials.get("skipping_specials"), tok.decode(ids),
                            "decode must skip special tokens by default");
                    assertEquals(specials.get("kept"), tok.decode(ids, false),
                            "decode(ids, false) must keep them");
                }));
            }
        }
        // A floor on the slices specifically, because they are the cases that caught the
        // ByteFallback rule and a corpus that lost them would still look healthy.
        final int total = sliceCases;
        assertTrue(total >= 400,
                () -> "the slice cases are the point of this family and there are only " + total);
        return tests;
    }

    @Test
    @DisplayName("the fixture was recorded against the decoders this port implements")
    @SuppressWarnings("unchecked")
    void fixtureNamesItsDecoders() {
        // Without this, a fixture regenerated against a checkpoint whose decoder changed would be
        // asserted against as though nothing had moved -- the same guard tokenizer_ids applies.
        Map<String, Object> en = (Map<String, Object>) forCheckpoint("english").get("decoder");
        assertEquals("ByteLevel", en.get("type"), "the english decoder changed");

        Map<String, Object> ml = (Map<String, Object>) forCheckpoint("multilingual").get("decoder");
        assertEquals("Sequence", ml.get("type"), "the multilingual decoder changed");
        List<Object> steps = (List<Object>) ml.get("decoders");
        List<String> types = new ArrayList<>();
        for (Object step : steps) {
            types.add((String) ((Map<String, Object>) step).get("type"));
        }
        assertEquals(List.of("Replace", "ByteFallback", "Fuse"), types,
                "the multilingual decoder chain changed");
    }

    // ------------------------------------------------- the three rules, each stated on its own

    @Test
    @DisplayName("a character outside the stand-in alphabet is kept, not dropped")
    void unmappedCharactersSurvive() {
        // Dropping them was the first guess and it got 414 of 5,312 texts wrong. This vocabulary
        // mixes byte-level tokens with 23 added tokens whose content is LITERAL whitespace -- id
        // 50275 is three real spaces, and 0x20 is lifted to U+0120 so a real space is not in the
        // alphabet at all.
        Tokenizer tok = open("english");
        assertEquals("   ", tok.decode(new int[] {50275}),
                "the literal-whitespace added token must survive decoding");
        assertEquals(" ", tok.decode(tok.encode(" ")), "and a byte-level space still works");
        assertEquals("   ", tok.decode(tok.encode("   ")));
    }

    /** Loads a synthetic spec recorded in the fixture, as the crate was asked about it. */
    private static Tokenizer fromSpec(Object spec) throws IOException {
        Path dir = Files.createTempDirectory("laya-synthetic");
        Path file = dir.resolve("tokenizer.json");
        Files.write(file, PythonJson.dumps(spec).getBytes(StandardCharsets.UTF_8));
        try {
            return Tokenizer.fromFile(file);
        } finally {
            Files.deleteIfExists(file);
            Files.deleteIfExists(dir);
        }
    }

    @TestFactory
    @DisplayName("the decoder shapes no shipped checkpoint declares")
    @SuppressWarnings("unchecked")
    List<DynamicTest> syntheticDecoderShapes() {
        // `TokenDecoder` implements six decoders, and the two shipped checkpoints between them use
        // four: english is ByteLevel, multilingual is Sequence[Replace, ByteFallback, Fuse]. So
        // the recorded corpus cannot reach Strip at all, cannot reach the no-decoder fallback, and
        // cannot tell a nested Sequence or a split byte run from a joined one. An adversarial
        // review gutted Strip entirely -- `apply` replaced by `return "MUTANT"` -- and all 191
        // cases passed.
        //
        // Four real divergences were hiding in that gap, each confirmed against the crate before
        // it was fixed: the no-decoder fallback joins with a SPACE (this joined with nothing), the
        // hex parse is case-insensitive and takes a leading "+" (this rejected both), a nested
        // Sequence keeps its pieces (this re-joined them), and a failed byte run is one piece PER
        // BYTE (this made it one piece of N). Every expectation below comes from the crate.
        Map<String, Object> shapes = (Map<String, Object>) family().get("synthetic_decoders");
        Assumptions.assumeTrue(shapes != null && !shapes.containsKey("skipped"),
                "the synthetic decoder shapes were not recorded");

        List<DynamicTest> tests = new ArrayList<>();
        for (Map.Entry<String, Object> entry : shapes.entrySet()) {
            String name = entry.getKey();
            Map<String, Object> shape = (Map<String, Object>) entry.getValue();
            tests.add(dynamicTest(name, () -> {
                Tokenizer tok = fromSpec(shape.get("spec"));
                List<Object> cases = (List<Object>) shape.get("cases");
                assertFalse(cases.isEmpty(), () -> name + " recorded no cases");
                for (Object raw : cases) {
                    Map<String, Object> one = (Map<String, Object>) raw;
                    int[] ids = ids((List<Object>) one.get("ids"));
                    assertEquals(one.get("decoded"), tok.decode(ids, false),
                            () -> name + " diverged on ids " + one.get("ids"));
                }

                // Where the crate PANICS there is no reference answer, so what is asserted is
                // that this port stays total: a Rust panic crossing the FFI boundary is not
                // behaviour worth reproducing, and silently returning something is not either.
                List<Object> panics = (List<Object>) shape.get("crate_panics");
                if (panics != null) {
                    for (Object raw : panics) {
                        int[] ids = ids((List<Object>) raw);
                        assertEquals("", tok.decode(ids, false),
                                () -> name + ": the crate panics on " + raw
                                        + " and this must return empty text, not throw");
                    }
                }
            }));
        }
        assertTrue(tests.size() >= 5,
                () -> "five shapes were recorded and only " + tests.size() + " are here");
        return tests;
    }

    @Test
    @DisplayName("an unsupported or malformed decoder is refused, not guessed at")
    void refusalsFire() {
        // Every one of these threw nowhere in the suite before this test: an adversarial review
        // replaced all four refusal paths at once -- the unknown-type throw by `return new
        // Join()`, the Regex-pattern throw and the Strip content-length throw by `if (false)` --
        // and all 191 cases passed. They are the guards whose whole stated purpose is to stop a
        // future checkpoint being silently mis-decoded, so "nothing proves they fire" is the one
        // thing they cannot afford. None of this needs a checkpoint, so it runs in every lane.
        assertThrows(Json.JsonException.class,
                () -> TokenDecoder.from(Map.of("type", "Metaspace")),
                "a decoder this port does not implement must be refused, not approximated");
        assertThrows(Json.JsonException.class,
                () -> TokenDecoder.from(new LinkedHashMap<>()),
                "a decoder node with no type must be refused");
        // The MESSAGE, not just the type. Disabling the Regex check leaves `literal` null,
        // so the NEXT guard throws Json.JsonException anyway -- and a mutant that removed
        // the Regex refusal outright passed this assertion while it checked only the
        // exception class. The two refusals say different things ("this port does not do
        // regex" versus "your pattern is missing") and have to be told apart.
        Json.JsonException regex = assertThrows(Json.JsonException.class,
                () -> TokenDecoder.from(Map.of("type", "Replace",
                        "pattern", Map.of("Regex", "\\s+"), "content", " ")),
                "a Regex Replace needs the crate's regex dialect and must be refused");
        assertTrue(regex.getMessage().contains("Regex"),
                () -> "the refusal must name the reason, got: " + regex.getMessage());
        assertThrows(Json.JsonException.class,
                () -> TokenDecoder.from(Map.of("type", "Replace", "content", " ")),
                "a Replace with no pattern must be refused");
        assertThrows(Json.JsonException.class,
                () -> TokenDecoder.from(Map.of("type", "Sequence")),
                "a Sequence with no decoders array must be refused");
        assertThrows(Json.JsonException.class,
                () -> TokenDecoder.from(Map.of("type", "Sequence",
                        "decoders", List.of("ByteLevel"))),
                "a Sequence step that is not an object must be refused");
        assertThrows(Json.JsonException.class,
                () -> TokenDecoder.from(Map.of("type", "Strip", "content", "ab")),
                "a Strip whose content is not one character must be refused");

        // And the one branch that is deliberately permissive. Its separator is a space, which is
        // what the crate falls back to; the recorded `no_decoder` shape is the parity check and
        // this states that `from(null)` reaches the same code.
        assertEquals("ab cd", TokenDecoder.from(null).decode(List.of("ab", "cd")),
                "no decoder node means join with a space, as the crate does");

        assertThrows(IllegalArgumentException.class, () -> PreTokenizer.charForByte(256),
                "a byte is 0 to 255, and an out-of-range index must say so");
        assertThrows(IllegalArgumentException.class, () -> PreTokenizer.charForByte(-1),
                "including below the range");
    }

    @Test
    @DisplayName("the ByteLevel fallback is scoped to a token, not to a character")
    @SuppressWarnings("unchecked")
    void byteLevelFallsBackPerToken() throws IOException {
        // Neither real checkpoint can tell these rules apart. Their byte-level tokens are entirely
        // stand-in alphabet and their added tokens are entirely literal whitespace, so a mutant
        // that swapped per-token for per-character passed all 190 recorded cases -- the rule was
        // correct and unpinned, which is the same as unverified.
        //
        // The rules differ only for a token that MIXES the two, so the generator builds a
        // vocabulary containing one and records what the crate does. "Ġ " separates them:
        // U+0120 is the stand-in for byte 0x20 and the literal space is not in the alphabet.
        //   per TOKEN     -> every character's own UTF-8 bytes: C4 A0 20 -> "Ġ "
        //   per CHARACTER -> the mapped one's byte:             20 20    -> "  "
        Map<String, Object> scope = (Map<String, Object>) family().get("mixed_token");
        Assumptions.assumeTrue(scope != null && !scope.containsKey("skipped"),
                "the mixed-token scope was not recorded");
        // Documentation, not a gate: the generator RAISES when the probed scope is not "token",
        // so no other value can reach the fixture and this line can only fail on a hand-edited
        // one. The real guard is that `raise` -- verified by changing the separating token so the
        // crate answered differently, which failed the regeneration step rather than this test.
        // The parity assertions below are what pin the behaviour.
        assertEquals("token", scope.get("scope"),
                "the crate's fallback scope changed and this port assumes per token");

        // The spec travels in the fixture, so this loads the same tokenizer the crate was asked
        // about instead of a second hand-written copy of it.
        Path dir = Files.createTempDirectory("laya-mixed-token");
        Path file = dir.resolve("tokenizer.json");
        try {
            Files.write(file, PythonJson.dumps(scope.get("spec"))
                    .getBytes(StandardCharsets.UTF_8));
            Tokenizer tok = Tokenizer.fromFile(file);
            for (Object raw : (List<Object>) scope.get("cases")) {
                Map<String, Object> one = (Map<String, Object>) raw;
                int[] ids = ids((List<Object>) one.get("ids"));
                assertEquals(one.get("decoded"), tok.decode(ids, false),
                        (String) one.get("why"));
            }
        } finally {
            Files.deleteIfExists(file);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    @DisplayName("ByteFallback emits one replacement character per byte, not per subpart")
    void byteFallbackReplacesPerByte() {
        // The rule that only the slices could see. U+F0000 is four ByteFallback tokens; cutting
        // after two leaves F3 B0, which lossy decoding calls one maximal subpart and reports as
        // ONE replacement character. The crate reports two -- one per byte it could not use.
        Tokenizer tok = open("multilingual");
        int[] ids = tok.encode("󰀀");                 // U+F0000
        assertTrue(ids.length >= 3,
                () -> "expected a Metaspace marker plus a byte run, got " + ids.length + " ids");
        int[] firstThree = {ids[0], ids[1], ids[2]};
        assertEquals(" ��", tok.decode(firstThree),
                "two bytes of a four-byte character must give two replacement characters");
        // and the whole thing still decodes to the character itself
        assertEquals(" 󰀀", tok.decode(ids),
                "the complete byte run must decode to the character");
    }

    @Test
    @DisplayName("decoding is lossy on the multilingual checkpoint, and that is the contract")
    void metaspaceAddsALeadingSpace() {
        // A port that "fixed" this would re-tokenize every predictLong window differently from the
        // reference and move every answer downstream. The fixture records the loss; this states it
        // where a reader will see it.
        Tokenizer ml = open("multilingual");
        assertEquals(" Hello world", ml.decode(ml.encode("Hello world")),
                "Metaspace prepends U+2581 and the decoder turns it into a space");
        Tokenizer en = open("english");
        assertEquals("Hello world", en.decode(en.encode("Hello world")),
                "the english checkpoint happens to round-trip");
    }

    @Test
    @DisplayName("an unknown id and a null argument are handled, not thrown at")
    void edges() {
        Tokenizer tok = open("english");
        assertEquals("", tok.decode(new int[0]), "no ids is empty text");
        assertEquals("", tok.decode(new int[] {Integer.MAX_VALUE}),
                "an id outside the vocabulary is skipped, as the crate skips it");
        // A NEGATIVE id is not crate parity -- the crate's ids are u32, so the Python binding
        // raises OverflowError rather than skipping it. Treating it like any other id outside the
        // vocabulary is this port's own choice, made because `int[]` can hold one and a caller's
        // off-by-one should not become an exception from a method whose job is to be lossy. Stated
        // here rather than filed under "as the crate skips it", which is what it used to say.
        assertEquals("a", tok.decode(new int[] {Integer.MAX_VALUE, tok.encode("a")[0], -1}),
                "a negative id is skipped like any other unknown one, by this port's choice");
        assertTrue(assertThrowsIllegalArgument(() -> tok.decode(null)),
                "a null id array is a caller error, not a silent empty string");
    }

    private static boolean assertThrowsIllegalArgument(Runnable work) {
        try {
            work.run();
            return false;
        } catch (IllegalArgumentException expected) {
            return true;
        }
    }

    @Test
    @DisplayName("the byte table and its inverse agree on all 256 values")
    void byteTableRoundTrips() {
        // What this catches, precisely: a table that is not INJECTIVE. If two bytes were given
        // the same stand-in character, the inverse would lose one of them and the round trip
        // would fail here. Measured, because the comment used to claim more than that: a mutant
        // that SWAPS two entries (0x41 and 0x42) is still injective and passes this test and all
        // 191 recorded cases. The table's actual values are pinned on the encode side instead,
        // by the recorded tokenizer fixtures -- the same swap fails 10 cases in
        // SequenceParityTest -- and none of the 23 texts in this family contains an "A" or a "B",
        // which is why the decode side could not see it.
        for (int value = 0; value < 256; value++) {
            final int b = value;
            char stand = PreTokenizer.charForByte(b);
            assertEquals(b, PreTokenizer.byteForChar(stand),
                    () -> String.format("byte 0x%02X does not survive the round trip", b));
        }
        assertFalse(PreTokenizer.byteForChar(' ') == 0x20,
                "a literal space must NOT be in the alphabet: 0x20 is lifted to U+0120, and the "
                + "whole literal-token rule depends on it being absent");
    }

    private static int[] ids(List<Object> recorded) {
        int[] out = new int[recorded.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) recorded.get(i)).intValue();
        }
        return out;
    }

    /** A readable label that cannot contain a line break or a lone surrogate. */
    private static String describe(String text) {
        if (text.isEmpty()) {
            return "<empty>";
        }
        StringBuilder out = new StringBuilder();
        text.codePoints().limit(18).forEach(cp -> {
            if (cp < 0x20 || cp > 0x7E) {
                out.append(String.format("U+%04X", cp));
            } else {
                out.appendCodePoint(cp);
            }
        });
        return out.toString();
    }
}
