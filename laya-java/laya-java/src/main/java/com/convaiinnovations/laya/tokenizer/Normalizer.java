package com.convaiinnovations.laya.tokenizer;

import com.convaiinnovations.laya.json.Json;
import java.text.Normalizer.Form;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * The normalizer stage: the text transform a checkpoint applies before pre-tokenization.
 *
 * <p>The two laya checkpoints use different ones, and neither is a no-op:
 * the english checkpoint declares {@code {"type": "NFC"}}, and the multilingual one declares
 * {@code {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}}. Skipping this stage
 * would mis-tokenize every input containing a space on one model and every input containing a
 * decomposable character on the other.
 *
 * <p>Unrecognised normalizer types are a hard failure rather than a skip. A normalizer this port
 * does not implement produces token ids that are wrong in a way no answer-level assertion can
 * localise, so a checkpoint using one must be rejected at load time, loudly, naming the type.
 */
abstract class Normalizer {

    /** Applies this normalizer. */
    abstract String normalize(String text);

    /** The identity normalizer, for a checkpoint declaring {@code "normalizer": null}. */
    static final Normalizer NONE = new Normalizer() {
        @Override
        String normalize(String text) {
            return text;
        }
    };

    /**
     * Builds a normalizer from a checkpoint's {@code normalizer} node, which may be null.
     */
    static Normalizer from(Map<String, Object> node) {
        if (node == null) {
            return NONE;
        }
        String type = Json.string(node, "type", null);
        if (type == null) {
            throw new Json.JsonException("a normalizer must declare a type");
        }
        switch (type) {
            case "NFC":
                return unicode(Form.NFC);
            case "NFD":
                return unicode(Form.NFD);
            case "NFKC":
                return unicode(Form.NFKC);
            case "NFKD":
                return unicode(Form.NFKD);
            case "Replace":
                return replace(node);
            case "Lowercase":
                // Locale-independent on purpose: a Turkish default locale would otherwise map
                // 'I' to a dotless i and silently change ids for every consumer in that locale.
                return of(text -> text.toLowerCase(java.util.Locale.ROOT));
            case "Prepend":
                String prefix = Json.string(node, "prepend", "");
                return of(text -> prefix + text);
            case "Strip":
                boolean left = Json.bool(node, "strip_left", true);
                boolean right = Json.bool(node, "strip_right", true);
                return of(text -> strip(text, left, right));
            case "Sequence":
                List<Normalizer> steps = new ArrayList<>();
                for (Object child : Json.array(node, "normalizers")) {
                    steps.add(from(asObject(child, "a normalizer in a Sequence")));
                }
                return of(text -> {
                    String out = text;
                    for (Normalizer step : steps) {
                        out = step.normalize(out);
                    }
                    return out;
                });
            default:
                throw new Json.JsonException(
                        "this checkpoint uses the normalizer \"" + type + "\", which laya-java does "
                        + "not implement. Refusing to load rather than produce token ids that are "
                        + "wrong in a way no answer-level test would localise.");
        }
    }

    private static Normalizer unicode(Form form) {
        return of(text -> java.text.Normalizer.isNormalized(text, form)
                ? text
                : java.text.Normalizer.normalize(text, form));
    }

    /**
     * {@code Replace} takes either a literal string or a regex, spelled as a single-key object
     * ({@code {"String": " "}} or {@code {"Regex": "\\s+"}}). The literal form is the common one and
     * must not be compiled as a pattern: the multilingual checkpoint replaces a space, and a space
     * is a valid pattern meaning something else the moment the literal contains a metacharacter.
     */
    private static Normalizer replace(Map<String, Object> node) {
        Map<String, Object> pattern = Json.object(node, "pattern");
        String content = Json.string(node, "content", null);
        if (content == null) {
            throw new Json.JsonException("a Replace normalizer must declare content");
        }
        String literal = Json.string(pattern, "String", null);
        if (literal != null) {
            if (literal.isEmpty()) {
                throw new Json.JsonException("a Replace normalizer cannot match the empty string");
            }
            return of(text -> text.replace(literal, content));
        }
        String regex = Json.string(pattern, "Regex", null);
        if (regex == null) {
            throw new Json.JsonException(
                    "a Replace normalizer's pattern must be {\"String\": ...} or {\"Regex\": ...}");
        }
        Pattern compiled = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        String replacement = Pattern.quote(content).equals(content) ? content
                : java.util.regex.Matcher.quoteReplacement(content);
        return of(text -> compiled.matcher(text).replaceAll(replacement));
    }

    private static String strip(String text, boolean left, boolean right) {
        int from = 0;
        int to = text.length();
        while (left && from < to && Unicode.isWhiteSpace(text.charAt(from))) {
            from++;
        }
        while (right && to > from && Unicode.isWhiteSpace(text.charAt(to - 1))) {
            to--;
        }
        return text.substring(from, to);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asObject(Object value, String what) {
        if (!(value instanceof Map)) {
            throw new Json.JsonException(what + " must be an object");
        }
        return (Map<String, Object>) value;
    }

    private interface Transform {
        String apply(String text);
    }

    private static Normalizer of(Transform transform) {
        return new Normalizer() {
            @Override
            String normalize(String text) {
                return transform.apply(text);
            }
        };
    }
}
