package com.convaiinnovations.laya.config;

import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.json.Json;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A checkpoint's {@code rl_agent_config.json}: sequence budgets and the fitted temperatures.
 *
 * <p>These must be read from the checkpoint and never defaulted. The two shipped checkpoints
 * disagree on every one of them: english is {@code max_len 512 / head_max_len 192} with fitted
 * per-bucket temperatures, multilingual is {@code 1024/256} with none at all. A port that
 * hard-coded the library's own defaults would halve a multilingual state's budget.
 */
public final class AgentConfig {

    /**
     * A fitted temperature below 1 sharpens logits instead of softening them, and is refused.
     *
     * <p>Not a hypothetical: the english checkpoint ships {@code choice:11+} at <b>0.1006</b>, a
     * ~10x sharpener that publishes a 0.24 top probability as 0.99. A caller gating on confidence
     * would be told a coin flip is a certainty, so the runtime confines every temperature to
     * {@code [0.5, 5.0]}.
     */
    public static final double TEMP_MIN = 0.5;
    public static final double TEMP_MAX = 5.0;

    /** One language's override table. */
    public record LangTemperatures(double[] temperature, Map<String, Double> byOptions) {
    }

    private final int maxLen;
    private final int headMaxLen;
    private final double[] temperature;
    private final Map<String, Double> temperatureByOptions;
    private final Map<String, LangTemperatures> langTemperatures;
    private final String encoder;

    private AgentConfig(int maxLen, int headMaxLen, double[] temperature,
                        Map<String, Double> temperatureByOptions,
                        Map<String, LangTemperatures> langTemperatures, String encoder) {
        this.maxLen = maxLen;
        this.headMaxLen = headMaxLen;
        this.temperature = temperature;
        this.temperatureByOptions = temperatureByOptions;
        this.langTemperatures = langTemperatures;
        this.encoder = encoder;
    }

    /** Loads {@code <modelDir>/rl_agent_config.json}. */
    public static AgentConfig fromModelDirectory(Path modelDirectory) throws IOException {
        try (Reader reader = new BufferedReader(Files.newBufferedReader(
                modelDirectory.resolve("rl_agent_config.json"), StandardCharsets.UTF_8))) {
            return fromReader(reader);
        }
    }

    /** Loads a config document from a reader. */
    public static AgentConfig fromReader(Reader reader) throws IOException {
        Object parsed = Json.parse(reader);
        if (!(parsed instanceof Map)) {
            throw new Json.JsonException("an rl_agent_config.json must be a JSON object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> document = (Map<String, Object>) parsed;
        return of(document);
    }

    /** Loads a config document from a string. For tests. */
    public static AgentConfig fromJson(String json) {
        try {
            return fromReader(new StringReader(json));
        } catch (IOException impossible) {
            throw new IllegalStateException("StringReader cannot fail", impossible);
        }
    }

    private static AgentConfig of(Map<String, Object> document) {
        double[] temperature = {1.0, 1.0, 1.0};
        Object raw = document.get("temperature");
        if (raw != null && !(raw instanceof List)) {
            // Refused, not defaulted. The reference raises "Incompatible model" for a
            // non-list temperature; defaulting it to 1.0 let a corrupt or hand-edited checkpoint
            // load and then answer with an UNCALIBRATED temperature while reporting confidence as
            // though it were fitted. The wrong length was already refused here; only the wrong
            // type slipped through.
            throw new Json.JsonException(
                    "temperature must be a list of three values, one per question type, got "
                    + raw.getClass().getSimpleName());
        }
        if (raw instanceof List) {
            List<?> values = (List<?>) raw;
            if (values.size() != 3) {
                throw new Json.JsonException(
                        "temperature must be three values, one per question type, got "
                        + values.size());
            }
            for (int i = 0; i < 3; i++) {
                temperature[i] = clampTemperature(values.get(i));
            }
        }
        Map<String, Double> byOptions = new LinkedHashMap<>();
        Object tbo = document.get("temperature_by_options");
        if (tbo != null && !(tbo instanceof Map)) {
            // The reference calls .items() on it, so a non-mapping raises there too.
            throw new Json.JsonException(
                    "temperature_by_options must be a mapping, got "
                    + tbo.getClass().getSimpleName());
        }
        if (tbo instanceof Map) {
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) tbo).entrySet()) {
                byOptions.put(String.valueOf(entry.getKey()), clampTemperature(entry.getValue()));
            }
        }
        Map<String, LangTemperatures> languages =
                resolveLangTemperatures(document.get("lang_temperatures"), temperature);
        return new AgentConfig(
                Json.integer(document, "max_len", 512),
                Json.integer(document, "head_max_len", 192),
                temperature, Map.copyOf(byOptions), languages,
                Json.string(document, "encoder", null));
    }

    /**
     * Parses the {@code lang_temperatures} option into {@code language -> table}.
     *
     * <p>A {@code null} entry or a {@code null} temperature both mean "inherit the checkpoint's
     * own", which is how the Python resolver and the {@code laya-ts} port read the same option.
     * Keys are normalised to the part before a {@code -}, lower-cased, so {@code zh-Hans} and
     * {@code ZH} reach the same table.
     */
    public static Map<String, LangTemperatures> resolveLangTemperatures(
            Object raw, double[] baseTemperature) {
        Map<String, LangTemperatures> resolved = new LinkedHashMap<>();
        if (!(raw instanceof Map)) {
            return Map.copyOf(resolved);
        }
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) raw).entrySet()) {
            if (!(entry.getKey() instanceof String)) {
                throw new IllegalArgumentException(
                        "language override keys must be strings, got " + entry.getKey());
            }
            String language = normaliseLanguage((String) entry.getKey());
            Object value = entry.getValue();
            Map<?, ?> table = value == null ? Map.of()
                    : value instanceof Map ? (Map<?, ?>) value : null;
            if (table == null) {
                throw new IllegalArgumentException(
                        "language override " + entry.getKey() + " must be a mapping, got "
                        + value.getClass().getSimpleName());
            }
            Object temperatureRaw = table.get("temperature");
            double[] temperature = new double[3];
            if (temperatureRaw == null) {
                System.arraycopy(baseTemperature, 0, temperature, 0, 3);
            } else if (temperatureRaw instanceof List && ((List<?>) temperatureRaw).size() == 3) {
                List<?> values = (List<?>) temperatureRaw;
                for (int i = 0; i < 3; i++) {
                    temperature[i] = clampTemperature(values.get(i));
                }
            } else {
                throw new IllegalArgumentException(
                        "language override " + entry.getKey()
                        + " temperature must be a list of 3 values, got " + temperatureRaw);
            }
            Map<String, Double> byOptions = new LinkedHashMap<>();
            Object tbo = table.get("temperature_by_options");
            if (tbo instanceof Map) {
                for (Map.Entry<?, ?> option : ((Map<?, ?>) tbo).entrySet()) {
                    byOptions.put(String.valueOf(option.getKey()),
                            clampTemperature(option.getValue()));
                }
            } else if (tbo != null) {
                throw new IllegalArgumentException(
                        "language override " + entry.getKey()
                        + " temperature_by_options must be a mapping");
            }
            resolved.put(language, new LangTemperatures(temperature, Map.copyOf(byOptions)));
        }
        return Map.copyOf(resolved);
    }

    /**
     * A usable temperature: confined to {@code [TEMP_MIN, TEMP_MAX]}, falling back to 1.0.
     *
     * <p>A boolean is not a number either. {@code true}/{@code false} used to float to 1.0/0.0 in
     * Python and read as fitted and sharpening temperatures, so both are refused here.
     */
    public static double clampTemperature(Object value) {
        if (value instanceof Boolean) {
            return 1.0;
        }
        double t;
        if (value instanceof Number) {
            t = ((Number) value).doubleValue();
        } else if (value instanceof String) {
            // The reference does `float(t)` in a try/except, so "2.5" IS 2.5. Discarding it
            // silently defaulted the temperature to 1.0 and overstated the published
            // answer_confidence -- by 19 points on the case that found this -- while the class's
            // whole stated purpose is that these are never defaulted.
            try {
                t = Double.parseDouble(((String) value).trim());
            } catch (NumberFormatException notANumber) {
                return 1.0;
            }
        } else {
            return 1.0;
        }
        if (Double.isNaN(t) || Double.isInfinite(t)) {
            return 1.0;
        }
        return Math.min(TEMP_MAX, Math.max(TEMP_MIN, t));
    }

    /** The bucket a question of this type and option count keys its temperature by. */
    public static String temperatureBucket(Question.Type type, int options) {
        String size = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+";
        return type.wireName() + ":" + size;
    }

    /**
     * The temperature to divide a question's logits by.
     *
     * <p>The bucket table wins over the per-type default, and a language override wins over both --
     * but only when that language has a table, and only by its prefix before a {@code -}.
     */
    public double temperatureFor(Question.Type type, int options, String language) {
        String bucket = temperatureBucket(type, options);
        Double bucketed = temperatureByOptions.get(bucket);
        double resolved = bucketed != null ? bucketed : temperature[type.code()];
        if (language != null && !language.isEmpty()) {
            LangTemperatures table = langTemperatures.get(normaliseLanguage(language));
            if (table != null) {
                Double override = table.byOptions().get(bucket);
                resolved = override != null ? override : table.temperature()[type.code()];
            }
        }
        return resolved;
    }

    /** Whether a language override applies, which also suppresses a fitted binning map. */
    public boolean hasLanguageOverride(String language) {
        return language != null && !language.isEmpty()
                && langTemperatures.containsKey(normaliseLanguage(language));
    }

    private static String normaliseLanguage(String language) {
        int dash = language.indexOf('-');
        return (dash < 0 ? language : language.substring(0, dash)).toLowerCase(Locale.ROOT);
    }

    public int maxLen() {
        return maxLen;
    }

    public int headMaxLen() {
        return headMaxLen;
    }

    public double[] temperature() {
        return temperature.clone();
    }

    public Map<String, Double> temperatureByOptions() {
        return temperatureByOptions;
    }

    public Map<String, LangTemperatures> langTemperatures() {
        return langTemperatures;
    }

    public String encoder() {
        return encoder;
    }
}
