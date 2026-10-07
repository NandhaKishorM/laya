package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.PythonJson;
import com.convaiinnovations.laya.lang.UnicodeTables;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One decision to put to the model: its type, its instruction, and the options it chooses between.
 *
 * <p>Three types, and they are not interchangeable -- the type name is the first token of the
 * sequence and an integer input to the head ({@code choice=0, score=1, noul=2}).
 *
 * <p>The factories take {@code Map<String, ?>} and {@code List<?>} rather than
 * {@code Map<String, Object>} and {@code List<Object>}. That is not cosmetic: Java generics are
 * invariant, so a parameter of {@code List<Object>} rejects {@code List.of("none", "minor")} --
 * writing the README is what surfaced it, because the first example in it did not compile.
 *
 * <p><b>Option order is part of the question.</b> For a {@code choice}, options are rendered in the
 * criteria map's <i>insertion</i> order, so two orders are two different questions and the answer's
 * probabilities are positional. That is why the criteria are held in a {@link LinkedHashMap} and
 * why {@link #choice} copies the caller's map in iteration order: accepting a {@code HashMap} and
 * keeping the reference would let the JVM's hash order decide what the model was asked.
 */
public final class Question {

    /** The question type, whose ordinal is the integer the head receives. */
    public enum Type {
        /** Pick one labelled option. */
        CHOICE("choice"),
        /** Rate on an ordered scale; the answer is the expected value over levels. */
        SCORE("score"),
        /** Does the statement hold? The answer is the probability that it does. */
        NOUL("noul");

        private final String wire;

        Type(String wire) {
            this.wire = wire;
        }

        /** The name that appears in the sequence text and keys the temperature tables. */
        public String wireName() {
            return wire;
        }

        /** The integer the head receives: {@code choice=0, score=1, noul=2}. */
        public int code() {
            return ordinal();
        }
    }

    private static final String DEFAULT_FALSE = "no, the statement does not hold";
    private static final String DEFAULT_TRUE = "yes, the statement holds";

    private final Type type;
    private final String instructions;
    private final Map<String, Object> criteriaMap;      // choice, noul
    private final List<Object> criteriaList;            // score
    private final String falseLabel;
    private final String trueLabel;

    private Question(Type type, String instructions, Map<String, Object> criteriaMap,
                     List<Object> criteriaList, String falseLabel, String trueLabel) {
        this.type = type;
        this.instructions = instructions;
        this.criteriaMap = criteriaMap;
        this.criteriaList = criteriaList;
        this.falseLabel = falseLabel;
        this.trueLabel = trueLabel;
    }

    /**
     * A choice between labelled options, rendered in the map's iteration order.
     *
     * <p>A criterion of {@code null} or {@code ""} means "no description", and the option renders
     * as its label alone. Only those two: {@code 0} and {@code false} are legitimate criterion
     * values and render as {@code "label: 0"} and {@code "label: false"}.
     */
    public static Question choice(String instructions, Map<String, ?> criteria) {
        requireInstructions(instructions);
        require(criteria != null && !criteria.isEmpty(), "a choice question needs criteria");
        return new Question(Type.CHOICE, instructions, new LinkedHashMap<>(criteria), null,
                null, null);
    }

    /** A score over ordered levels, rendered as {@code "level <i>: <criterion>"}. */
    public static Question score(String instructions, List<?> levels) {
        requireInstructions(instructions);
        require(levels != null && !levels.isEmpty(), "a score question needs levels");
        for (int i = 0; i < levels.size(); i++) {
            // A null level rendered as the literal text "level 0: null" into the model's input and
            // came back in the legend as a null value, so the question silently asked about a word
            // the caller never wrote. Python refuses it; so does this.
            require(levels.get(i) != null,
                    "score level " + i + " is null; give every level a description, index 0 first");
        }
        return new Question(Type.SCORE, instructions, null, new ArrayList<>(levels), null, null);
    }

    /** Does the statement hold? Options are always {@code [false, true]}, in that order. */
    public static Question noul(String instructions) {
        return noul(instructions, null, null, null);
    }

    /**
     * A noul with optional per-side criteria and optional labels.
     *
     * <p>{@code labels} is accepted <b>only</b> here -- a choice or score that supplies labels is
     * rejected in Python and is not expressible here. Both labels must be non-blank after trimming
     * and must differ.
     */
    public static Question noul(String instructions, Object falseCriterion, Object trueCriterion,
                                Map<String, String> labels) {
        requireInstructions(instructions);
        String falseLabel = "false";
        String trueLabel = "true";
        if (labels != null) {
            require(labels.size() == 2 && labels.containsKey("false") && labels.containsKey("true"),
                    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
            falseLabel = labels.get("false");
            trueLabel = labels.get("true");
            require(falseLabel != null && trueLabel != null,
                    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
            // Python's strip, not Java's trim. `trim` removes only code points at or below
            // U+0020, so a label of a single no-break space would survive it, pass the
            // non-empty check below, and render into the model's input as whitespace -- a
            // different token sequence from the one the reference produces, which refuses the
            // label outright.
            falseLabel = UnicodeTables.strip(falseLabel);
            trueLabel = UnicodeTables.strip(trueLabel);
            require(!falseLabel.isEmpty() && !trueLabel.isEmpty() && !falseLabel.equals(trueLabel),
                    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
        }
        Map<String, Object> criteria = new LinkedHashMap<>();
        criteria.put("false", falseCriterion);
        criteria.put("true", trueCriterion);
        return new Question(Type.NOUL, instructions, criteria, null, falseLabel, trueLabel);
    }

    public Type type() {
        return type;
    }

    /**
     * The question as the reference's own mapping: {@code type}, {@code instructions}, and the
     * {@code criteria} it has.
     *
     * <p>What a schema signature serialises. Two details are not decoration. A {@code noul} holds
     * a criteria map internally even when neither side was described, and the reference's mapping
     * simply has no {@code criteria} key then -- emitting an empty one would change the signature
     * of every plain noul. And the key order here is the order the reference writes, because the
     * signature is order-sensitive on purpose.
     *
     * <p><b>What this cannot carry.</b> The reference's questions are free-form dictionaries, so a
     * caller may put extra keys in one, or write {@code criteria} in an order of their own. A
     * typed {@link Question} has nowhere to keep either. That costs nothing for what the signature
     * is for -- grouping states whose questions render identically, so they can share a forward
     * pass -- because identical questions still produce identical signatures and any difference
     * that reaches the model still changes them. It does mean the signature is an internal
     * grouping key and not a cross-language wire format.
     */
    public Map<String, Object> spec() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("type", type.wireName());
        out.put("instructions", instructions);
        if (type == Type.SCORE) {
            out.put("criteria", new ArrayList<>(criteriaList));
        } else if (type == Type.CHOICE) {
            out.put("criteria", new LinkedHashMap<>(criteriaMap));
        } else {
            Object described = criteriaMap.get("true");
            Object negated = criteriaMap.get("false");
            if (described != null || negated != null) {
                Map<String, Object> criteria = new LinkedHashMap<>();
                criteria.put("true", described);
                criteria.put("false", negated);
                out.put("criteria", criteria);
            }
            if (!"false".equals(falseLabel) || !"true".equals(trueLabel)) {
                Map<String, Object> labels = new LinkedHashMap<>();
                labels.put("false", falseLabel);
                labels.put("true", trueLabel);
                out.put("labels", labels);
            }
        }
        return java.util.Collections.unmodifiableMap(out);
    }

    public String instructions() {
        return instructions;
    }

    /**
     * The option texts, in label-index order.
     *
     * <p>This is the text the model actually reads, so it is the contract a port has to match
     * character for character: everything downstream is token ids derived from it.
     */
    public List<String> renderOptions() {
        List<String> out = new ArrayList<>();
        switch (type) {
            case CHOICE:
                for (Map.Entry<String, Object> entry : criteriaMap.entrySet()) {
                    Object value = entry.getValue();
                    // `null` and `""` only. A label with no description renders as itself.
                    out.add(value == null || "".equals(value)
                            ? entry.getKey()
                            : entry.getKey() + ": " + renderCriterion(value));
                }
                return out;
            case SCORE:
                for (int i = 0; i < criteriaList.size(); i++) {
                    out.add("level " + i + ": " + renderCriterion(criteriaList.get(i)));
                }
                return out;
            default:
                Object no = criteriaMap.get("false");
                Object yes = criteriaMap.get("true");
                out.add(falseLabel + ": "
                        + (no == null || "".equals(no) ? DEFAULT_FALSE : renderCriterion(no)));
                out.add(trueLabel + ": "
                        + (yes == null || "".equals(yes) ? DEFAULT_TRUE : renderCriterion(yes)));
                return out;
        }
    }

    /** The labels an answer is keyed by: the criteria keys, or the level indices, or false/true. */
    public List<String> labels() {
        List<String> out = new ArrayList<>();
        switch (type) {
            case CHOICE:
                out.addAll(criteriaMap.keySet());
                return out;
            case SCORE:
                for (int i = 0; i < criteriaList.size(); i++) {
                    out.add(Integer.toString(i));
                }
                return out;
            default:
                out.add(falseLabel);
                out.add(trueLabel);
                return out;
        }
    }

    /**
     * A score question's legend: each level index mapped to the text the model was shown.
     *
     * <p>{@code renderCriterion} rather than a plain {@code toString}: a structured level comes
     * back as the same JSON text the model read, where stringifying it would return a Java object's
     * {@code toString}. A numeric scale passed as {@code [1, 2, 3]} must come back as
     * {@code {"0": "1", "1": "2", "2": "3"}}, so the answer's shape does not depend on what the
     * caller happened to pass.
     *
     * @throws IllegalStateException for a choice or noul question, which have no legend
     */
    public Map<String, String> legend() {
        if (type != Type.SCORE) {
            throw new IllegalStateException("only a score question has a legend, not " + type);
        }
        Map<String, String> out = new LinkedHashMap<>();
        for (int i = 0; i < criteriaList.size(); i++) {
            out.put(Integer.toString(i), renderCriterion(criteriaList.get(i)));
        }
        // Level order, so `Map.copyOf` is not an option: its iteration order is unspecified.
        return java.util.Collections.unmodifiableMap(out);
    }

    /**
     * One criterion as text: a string passes through, anything else becomes compact JSON.
     *
     * <p>Python spells that JSON with {@code json.dumps(..., ensure_ascii=False,
     * separators=(", ", ": "))}, which is what {@link PythonJson} reproduces -- including float
     * formatting, because this text is tokenized and {@code 1e+16} and {@code 1.0E16} are different
     * token sequences.
     *
     * <p>Python passes {@code default=str}, so an object it cannot serialise becomes its
     * {@code str()}. Here an unsupported type raises instead: inventing text from a Java object's
     * {@code toString()} would put a class name and an identity hash into the model's input.
     */
    public static String renderCriterion(Object value) {
        return value instanceof String ? (String) value : PythonJson.dumps(value);
    }

    /**
     * Instructions must be present and not blank.
     *
     * <p>Blank is refused, not just null: an empty or whitespace-only instruction produces the
     * text {@code "choice question: "} and the model is asked nothing, which it answers anyway with
     * a confident-looking distribution over the options. Python refuses the same input.
     */
    private static void requireInstructions(String instructions) {
        // UnicodeTables.isBlank, not String.isBlank: the reference tests `not ins.strip()`, and
        // Python's strip treats U+00A0, U+0085, U+2007 and U+202F as whitespace where
        // Character.isWhitespace does not. An instruction of a single no-break space is
        // therefore refused by the reference and would have been accepted here.
        require(instructions != null && !UnicodeTables.isBlank(instructions),
                "instructions must not be empty; add the text the model should answer");
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
