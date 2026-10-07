package com.convaiinnovations.laya.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Question;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Budgets, temperature clamping, bucket selection and the language override. */
final class AgentConfigTest {

    @Test
    @DisplayName("budgets come from the checkpoint, never from a library default")
    void budgetsAreRead() {
        AgentConfig config = AgentConfig.fromJson(
                "{\"max_len\": 1024, \"head_max_len\": 256, \"temperature\": [1, 1, 1]}");
        assertEquals(1024, config.maxLen());
        assertEquals(256, config.headMaxLen());
    }

    @Test
    @DisplayName("a sharpening temperature is clamped: the shipped choice:11+ is 0.1006")
    void sharpeningIsClamped() {
        // Applied raw, 0.1006 multiplies logits ~10x and publishes a 0.24 top probability as 0.99.
        AgentConfig config = AgentConfig.fromJson(
                "{\"temperature\": [1.0, 1.0, 1.0],"
                + " \"temperature_by_options\": {\"choice:11+\": 0.100583}}");
        assertEquals(AgentConfig.TEMP_MIN,
                config.temperatureFor(Question.Type.CHOICE, 20, null), 0.0);
    }

    @Test
    @DisplayName("clampTemperature falls back exactly where the reference falls back")
    void nonNumbersFallBack() {
        // A bool used to float to 1.0/0.0 and read as a fitted or sharpening temperature.
        assertEquals(1.0, AgentConfig.clampTemperature(Boolean.TRUE), 0.0);
        assertEquals(1.0, AgentConfig.clampTemperature(Boolean.FALSE), 0.0);
        assertEquals(1.0, AgentConfig.clampTemperature(null), 0.0);
        assertEquals(1.0, AgentConfig.clampTemperature(Double.NaN), 0.0);
        assertEquals(1.0, AgentConfig.clampTemperature(Double.POSITIVE_INFINITY), 0.0);
        assertEquals(AgentConfig.TEMP_MAX, AgentConfig.clampTemperature(99.0), 0.0);

        // A NUMERIC STRING IS PARSED. The reference does `float(t)` in a try/except, so "2.0"
        // is 2.0 -- this assertion previously demanded 1.0, which is what the port did and the
        // reference does not. Discarding it defaulted the temperature silently and overstated
        // the published answer_confidence while the class's stated purpose is that these are
        // never defaulted.
        assertEquals(2.0, AgentConfig.clampTemperature("2.0"), 0.0);
        assertEquals(2.5, AgentConfig.clampTemperature("2.5"), 0.0);
        assertEquals(2.5, AgentConfig.clampTemperature(" 2.5 "), 0.0);
        // and a string that is not a number still falls back, as the except branch does
        assertEquals(1.0, AgentConfig.clampTemperature("abc"), 0.0);
        assertEquals(1.0, AgentConfig.clampTemperature(""), 0.0);
        // the clamp still applies to a parsed string
        assertEquals(AgentConfig.TEMP_MIN, AgentConfig.clampTemperature("0.1"), 0.0);
        assertEquals(AgentConfig.TEMP_MAX, AgentConfig.clampTemperature("99.0"), 0.0);
    }

    @Test
    @DisplayName("a malformed temperature refuses the checkpoint instead of defaulting it")
    void malformedTemperatureIsRefused() {
        // The reference raises "Incompatible model" for a non-list temperature. Defaulting it to
        // 1.0 let a corrupt or hand-edited checkpoint load and then answer with an uncalibrated
        // temperature while reporting confidence as though it were fitted.
        for (String document : new String[] {
            "{\"temperature\": 2.0}",
            "{\"temperature\": \"2.0\"}",
            "{\"temperature\": {\"0\": 2.0, \"1\": 2.0, \"2\": 2.0}}",
        }) {
            org.junit.jupiter.api.Assertions.assertThrows(
                    com.convaiinnovations.laya.json.Json.JsonException.class,
                    () -> AgentConfig.fromReader(new java.io.StringReader(document)),
                    "a non-list temperature must refuse the checkpoint: " + document);
        }
        org.junit.jupiter.api.Assertions.assertThrows(
                com.convaiinnovations.laya.json.Json.JsonException.class,
                () -> AgentConfig.fromReader(new java.io.StringReader(
                        "{\"temperature\": [2.0, 2.0, 2.0], \"temperature_by_options\": [1, 2]}")),
                "a non-mapping temperature_by_options must refuse the checkpoint too");
    }

    @Test
    @DisplayName("bucket boundaries are at 2, 5 and 10 options")
    void bucketBoundaries() {
        assertEquals("choice:2", AgentConfig.temperatureBucket(Question.Type.CHOICE, 1));
        assertEquals("choice:2", AgentConfig.temperatureBucket(Question.Type.CHOICE, 2));
        assertEquals("choice:3-5", AgentConfig.temperatureBucket(Question.Type.CHOICE, 3));
        assertEquals("choice:3-5", AgentConfig.temperatureBucket(Question.Type.CHOICE, 5));
        assertEquals("choice:6-10", AgentConfig.temperatureBucket(Question.Type.CHOICE, 6));
        assertEquals("choice:6-10", AgentConfig.temperatureBucket(Question.Type.CHOICE, 10));
        assertEquals("choice:11+", AgentConfig.temperatureBucket(Question.Type.CHOICE, 11));
        assertEquals("score:3-5", AgentConfig.temperatureBucket(Question.Type.SCORE, 4));
        assertEquals("noul:2", AgentConfig.temperatureBucket(Question.Type.NOUL, 2));
    }

    @Test
    @DisplayName("a bucket beats the per-type default, and a language beats both")
    void overridePrecedence() {
        AgentConfig config = AgentConfig.fromJson(
                "{\"temperature\": [1.5, 2.0, 2.5],"
                + " \"temperature_by_options\": {\"choice:2\": 1.75},"
                + " \"lang_temperatures\": {\"zh\": {\"temperature\": [3.0, 3.0, 3.0],"
                + " \"temperature_by_options\": {\"choice:2\": 4.0}}}}");
        assertEquals(1.75, config.temperatureFor(Question.Type.CHOICE, 2, null), 0.0);
        assertEquals(1.5, config.temperatureFor(Question.Type.CHOICE, 7, null), 0.0);
        assertEquals(4.0, config.temperatureFor(Question.Type.CHOICE, 2, "zh"), 0.0);
        assertEquals(3.0, config.temperatureFor(Question.Type.CHOICE, 7, "zh"), 0.0);
    }

    @Test
    @DisplayName("a language tag matches on its prefix, case-insensitively")
    void languagePrefixMatching() {
        AgentConfig config = AgentConfig.fromJson(
                "{\"temperature\": [1.0, 1.0, 1.0],"
                + " \"lang_temperatures\": {\"zh-Hans\": {\"temperature\": [3.0, 3.0, 3.0]}}}");
        assertTrue(config.hasLanguageOverride("zh"));
        assertTrue(config.hasLanguageOverride("ZH-Hant"));
        assertTrue(config.hasLanguageOverride("zh-Hans"));
        assertFalse(config.hasLanguageOverride("zho"));
        assertFalse(config.hasLanguageOverride(""));
        assertFalse(config.hasLanguageOverride(null));
        assertEquals(3.0, config.temperatureFor(Question.Type.CHOICE, 2, "ZH"), 0.0);
    }

    @Test
    @DisplayName("a null language entry inherits the checkpoint's own temperatures")
    void nullEntryInherits() {
        AgentConfig config = AgentConfig.fromJson(
                "{\"temperature\": [2.5, 1.0, 1.0], \"lang_temperatures\": {\"de\": null}}");
        assertTrue(config.hasLanguageOverride("de"));
        assertEquals(2.5, config.temperatureFor(Question.Type.CHOICE, 2, "de"), 0.0);
    }

    @Test
    @DisplayName("a malformed language override is refused at load, not at the first request")
    void malformedOverrideRefused() {
        assertThrows(IllegalArgumentException.class, () -> AgentConfig.fromJson(
                "{\"temperature\": [1,1,1], \"lang_temperatures\": {\"de\": {\"temperature\": 2}}}"));
        assertThrows(IllegalArgumentException.class, () -> AgentConfig.fromJson(
                "{\"temperature\": [1,1,1], \"lang_temperatures\": {\"de\": \"nope\"}}"));
        assertThrows(IllegalArgumentException.class, () -> AgentConfig.fromJson(
                "{\"temperature\": [1,1,1],"
                + " \"lang_temperatures\": {\"de\": {\"temperature_by_options\": 5}}}"));
    }

    @Test
    @DisplayName("the temperature array must have one entry per question type")
    void temperatureArityChecked() {
        assertThrows(RuntimeException.class,
                () -> AgentConfig.fromJson("{\"temperature\": [1.0, 2.0]}"));
    }

    @Test
    @DisplayName("the temperature array is copied, so a caller cannot mutate the config")
    void temperatureIsCopied() {
        AgentConfig config = AgentConfig.fromJson("{\"temperature\": [1.5, 1.5, 1.5]}");
        config.temperature()[0] = 99.0;
        assertEquals(1.5, config.temperatureFor(Question.Type.CHOICE, 7, null), 0.0);
    }
}
