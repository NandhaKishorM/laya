package com.convaiinnovations.laya.json;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Fixtures;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * {@link PythonJson} must spell a value exactly as CPython's
 * {@code json.dumps(..., ensure_ascii=False)} spells it.
 *
 * <p>Not a nicety: {@code serialize_state} and {@code render_criterion} feed their output straight
 * to the tokenizer, so every character becomes token ids. A float spelled {@code 1.0E16} where
 * Python writes {@code 1e+16} is a different sequence and a different answer.
 */
final class PythonJsonTest {

    @Test
    @DisplayName("every double is spelled as CPython spells it")
    @SuppressWarnings("unchecked")
    void doublesMatchCPython() {
        List<Object> cases = (List<Object>) Fixtures.load("python_json.json").get("doubles");
        int checked = 0;
        java.util.Set<Double> covered = new java.util.HashSet<>();
        for (Object entry : cases) {
            Map<String, Object> row = (Map<String, Object>) entry;
            double value = Double.longBitsToDouble(((Number) row.get("bits")).longValue());
            covered.add(value);
            // The operand is carried as a bit pattern, not as decimal text: the thing under test is
            // the SPELLING of a double, and a decimal literal would lose it on the way in.
            assertEquals((String) row.get("repr"), PythonJson.repr(value),
                    "repr of the double with bits " + row.get("bits"));
            checked++;
        }
        org.junit.jupiter.api.Assertions.assertTrue(checked > 400,
                "the doubles fixture should be substantial, found " + checked);
        // A size floor alone would survive a fixture that dropped the cases that discriminate, so
        // name the ones that do: each of these is spelled differently by Java than by CPython.
        for (double discriminating : new double[] {1e16, 1e-5, 0.0001, 1e23, -0.0, 5e-324, 2.0}) {
            org.junit.jupiter.api.Assertions.assertTrue(covered.contains(discriminating)
                            || (discriminating == 0.0 && covered.contains(-0.0)),
                    "the fixture must still cover " + discriminating);
        }
    }

    @Test
    @DisplayName("structured values match, including separators, escaping and key order")
    @SuppressWarnings("unchecked")
    void structuredMatchesCPython() {
        List<Object> cases = (List<Object>) Fixtures.load("python_json.json").get("structured");
        for (Object entry : cases) {
            Map<String, Object> row = (Map<String, Object>) entry;
            assertEquals((String) row.get("dumps"), PythonJson.dumps(row.get("value")));
        }
    }

    @Test
    @DisplayName("a non-finite double is written as CPython writes it, not as invalid JSON we invent")
    void nonFinite() {
        assertEquals("NaN", PythonJson.repr(Double.NaN));
        assertEquals("Infinity", PythonJson.repr(Double.POSITIVE_INFINITY));
        assertEquals("-Infinity", PythonJson.repr(Double.NEGATIVE_INFINITY));
    }

    @Test
    @DisplayName("negative zero keeps its sign, which no arithmetic test would catch")
    void negativeZero() {
        assertEquals("-0.0", PythonJson.repr(-0.0));
        assertEquals("0.0", PythonJson.repr(0.0));
    }

    @Test
    @DisplayName("an unsupported type is refused rather than stringified into the model's input")
    void unsupportedTypeIsRefused() {
        org.junit.jupiter.api.Assertions.assertThrows(Json.JsonException.class,
                () -> PythonJson.dumps(new Object()));
    }

    @Test
    @DisplayName("a non-string key is coerced the way CPython coerces it")
    void nonStringKeysAreCoerced() {
        // CPython writes `{1: "x"}` as `{"1": "x"}` rather than refusing it. This used to throw,
        // on the stated grounds that laya never produces such a key -- true of laya, and not of
        // the caller-supplied Object that reaches here as a state.
        assertEquals("{\"1\": \"x\"}", PythonJson.dumps(new java.util.LinkedHashMap<>(
                Map.of(1L, "x"))));
        assertEquals("{\"true\": 1}", PythonJson.dumps(new java.util.LinkedHashMap<>(
                Map.of(Boolean.TRUE, 1L))));
        java.util.Map<Object, Object> withNull = new java.util.LinkedHashMap<>();
        withNull.put(null, "x");
        assertEquals("{\"null\": \"x\"}", PythonJson.dumps(withNull));
        assertEquals("{\"2.5\": 1}", PythonJson.dumps(new java.util.LinkedHashMap<>(
                Map.of(2.5, 1L))));
    }

    @Test
    @DisplayName("a key type CPython itself refuses is still refused")
    void unsupportedKeyTypeIsRefused() {
        org.junit.jupiter.api.Assertions.assertThrows(Json.JsonException.class,
                () -> PythonJson.dumps(Map.of(List.of(1L), "x")));
    }

    @Test
    @DisplayName("a cyclic value is refused, not recursed into")
    void cyclicValueIsRefused() {
        // A StackOverflowError is an Error, so a server's `catch (Exception)` does not contain it,
        // and the state is caller-supplied. CPython raises ValueError("Circular reference
        // detected"); this raises with the same wording.
        List<Object> list = new java.util.ArrayList<>();
        list.add("a");
        list.add(list);
        Json.JsonException cyclic = org.junit.jupiter.api.Assertions.assertThrows(
                Json.JsonException.class, () -> PythonJson.dumps(list));
        assertEquals("Circular reference detected", cyclic.getMessage());

        java.util.Map<String, Object> map = new java.util.LinkedHashMap<>();
        map.put("self", map);
        org.junit.jupiter.api.Assertions.assertThrows(Json.JsonException.class,
                () -> PythonJson.dumps(map));
    }

    @Test
    @DisplayName("repr spells every recorded string the way CPython does")
    void reprMatchesCpython() {
        @SuppressWarnings("unchecked")
        List<Object> cases = (List<Object>) Fixtures.load("python_json.json").get("repr_strings");
        int escaped = 0;
        int doubleQuoted = 0;
        for (Object entry : cases) {
            @SuppressWarnings("unchecked")
            List<Object> pair = (List<Object>) entry;
            String input = (String) pair.get(0);
            String expected = (String) pair.get(1);
            assertEquals(expected, PythonJson.repr(input),
                    "repr of " + expected + " (" + input.length() + " chars)");
            if (expected.indexOf('\\') >= 0) {
                escaped++;
            }
            if (expected.startsWith("\"")) {
                doubleQuoted++;
            }
        }
        assertTrue(cases.size() >= 30, "only " + cases.size() + " repr cases");
        assertTrue(escaped >= 15, "only " + escaped + " cases actually escape something");
        assertTrue(doubleQuoted >= 1, "no case exercises the double-quoted form, which is the"
                + " branch for a string holding a single quote and no double quote");
    }

    @Test
    @DisplayName("repr escapes every unprintable code point in Unicode, and no printable one")
    void reprDigestMatchesCpython() {
        // The listed cases cover the rules a reader can name; this covers the boundary nobody
        // enumerated. 148,998 code points are printable and the rest are not, and an ASCII-only
        // escape check -- which is what the .NET port does -- passes every named case above while
        // leaving U+00A0 and U+200B raw in an API response.
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is required of every JVM", impossible);
        }
        for (int cp = 0; cp < 0x110000; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) {
                continue;
            }
            String value = "a" + new String(Character.toChars(cp)) + "b";
            digest.update(PythonJson.repr(value).getBytes(StandardCharsets.UTF_8));
        }
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest.digest()) {
            hex.append(Character.forDigit((value >> 4) & 0xF, 16));
            hex.append(Character.forDigit(value & 0xF, 16));
        }
        assertEquals(Fixtures.load("python_json.json").get("repr_digest"), hex.toString(),
                "repr disagrees with CPython somewhere in Unicode");
    }

    @Test
    @DisplayName("percent0 rounds halves to even, as CPython's %.0f does")
    void percent0MatchesCpython() {
        @SuppressWarnings("unchecked")
        List<Object> cases = (List<Object>) Fixtures.load("python_json.json").get("percent0");
        int distinguishing = 0;
        for (Object entry : cases) {
            @SuppressWarnings("unchecked")
            List<Object> pair = (List<Object>) entry;
            double fraction = ((Number) pair.get(0)).doubleValue();
            String expected = (String) pair.get(1);
            assertEquals(expected, PythonJson.percent0(fraction), "percent0 of " + fraction);
            // Java's own formatter rounds halves up, so count how many of these would be wrong
            // with it -- a count of zero would mean this test proves nothing.
            if (!expected.equals(String.format(java.util.Locale.ROOT, "%.0f", 100.0 * fraction))) {
                distinguishing++;
            }
        }
        assertTrue(cases.size() >= 25, "only " + cases.size() + " percent cases");
        assertTrue(distinguishing >= 3,
                "only " + distinguishing + " of these differ from String.format(\"%.0f\"), so"
                + " this test would not notice half-up rounding");
    }

    @Test
    @DisplayName("repr of null is None, as a %r of None would print")
    void reprOfNull() {
        assertEquals("None", PythonJson.repr((String) null));
    }

    @Test
    @DisplayName("the same container appearing twice is NOT a cycle")
    void repeatedContainerIsNotACycle() {
        // Only a container currently being written is a cycle. A shared sub-object is legitimate
        // and CPython writes it twice, so tracking by identity alone would reject valid input.
        List<Object> shared = List.of(1L, 2L);
        assertEquals("[[1, 2], [1, 2]]", PythonJson.dumps(List.of(shared, shared)));
    }
}
