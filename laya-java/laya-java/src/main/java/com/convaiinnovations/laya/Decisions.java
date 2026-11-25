package com.convaiinnovations.laya;

import com.convaiinnovations.laya.decode.Rounding;
import com.convaiinnovations.laya.json.PythonJson;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Set;

/**
 * Schema-driven decisions: a JSON schema becomes Laya questions, and the answers come back as
 * the schema's own values.
 *
 * <p>The mapping is a documented subset, not a JSON-schema implementation: an object of
 * properties, each one an enum choice, a boolean, or a bounded integer scale. Anything that
 * cannot be answered from a fixed option set — a free string, an array, a nested object — is
 * refused with a {@link SchemaException} that names the path, because a caller holding a
 * 32-property schema cannot act on "this schema is not supported".
 *
 * <pre>{@code
 * Map<String, Object> schema = (Map<String, Object>) Json.parse("""
 *     {"type": "object", "properties": {
 *        "department": {"enum": ["billing", "support", "sales"]},
 *        "urgency":    {"type": "integer", "minimum": 1, "maximum": 5},
 *        "needs_human":{"type": "boolean"}}}""");
 * Map<String, Object> values = Decisions.decide(agent, ticket, schema).values();
 * }</pre>
 *
 * <p><b>Four rules a port gets wrong by default</b>, each pinned by
 * {@code fixtures/structured.json}:
 *
 * <ul>
 *   <li>A {@code choice} projects to the schema <b>value</b>, not the label the model answered
 *       with. For {@code enum: ["billing", "support"]} the two coincide; for
 *       {@code enum: [10, 20, 30]} a port returning the label hands back the string
 *       {@code "20"} where the schema says the integer {@code 20}.</li>
 *   <li>A {@code score} is {@code minimum + argmax(probabilities)}. The answer's own
 *       {@code score} field is the probability-weighted <b>level index</b>, not a field value, so
 *       the no-probabilities fallback rounds it and adds {@code minimum} once. Subtracting
 *       {@code minimum} first and adding it back cancels out and silently drops it from every
 *       scale that does not start at zero. The rounding is Python's, half to <b>even</b>;
 *       {@code Math.round} is half up and disagrees at every {@code k + 0.5}.</li>
 *   <li>A {@code noul} is {@code >= 0.5}, so exactly {@code 0.5} is <b>true</b>.</li>
 *   <li>A low-confidence answer projects to <b>null</b>; an answer that is <b>absent</b> leaves
 *       the field out of the map entirely. "not believed" and "not asked" are different
 *       outcomes, and a caller that cannot tell them apart reports a decision nobody made.</li>
 * </ul>
 *
 * <p><b>Where this differs from the reference, deliberately.</b>
 *
 * <ul>
 *   <li>Only the JSON-schema path is ported. There is no pydantic on the JVM, and the reference's
 *       {@code questions_from_pydantic} / {@code answer_to_pydantic} are a model-to-schema step in
 *       front of exactly this code. Mapping Java records would be a new design, not a port.</li>
 *   <li>{@code return_details} is gone: {@link #decide} always returns a {@link Decision}, whose
 *       {@link Decision#values()} is what the plain form returned. One return type instead of two
 *       selected by a boolean.</li>
 *   <li>The reference forwards {@code min_confidence} down to {@code runner.predict} and falls
 *       back to gating the result itself when the runner does not take the keyword. Here there is
 *       no such keyword on {@link Predictor}, so the second path is the only path — which is also
 *       the one the reference's own {@code decide_batch} always takes.</li>
 *   <li>{@code routing} is not reported. {@link Prediction} carries no routing block, so there is
 *       nothing to pass through; {@link Router} reports its decision at the point it makes it.</li>
 *   <li>A batched decision on a {@link Router} takes no routing keywords. The reference's
 *       {@code decide_batch} copies {@code model}, {@code task}, {@code lang},
 *       {@code lang_guess} and the token budgets onto every request; here a router routes each
 *       state on its own. A pin is a lambda runner,
 *       {@code (states, asked) -> router.predictBatch(states, asked, options)}; a budget is a
 *       list of {@link Router.Request}s.</li>
 *   <li>An {@code enum} must be an <b>array</b>. The reference measures and iterates whatever
 *       is there, so {@code "enum": "abc"} becomes a three-option choice over {@code "a"},
 *       {@code "b"}, {@code "c"} — a schema nobody wrote, built from a typo — and
 *       {@code "enum": 5} is a {@code TypeError} from inside {@code len()} rather than a
 *       refusal naming the property.</li>
 *   <li>A score's {@code minimum} and {@code maximum} must be integers, and a <b>boolean</b> is
 *       not one. Python's {@code bool} subclasses {@code int}, so the reference accepts
 *       {@code {"minimum": false, "maximum": true}} and asks "Score {@code x} from 0 to 1".</li>
 *   <li>A {@code description} must be a <b>string</b>. The reference puts whatever is there into
 *       the question's {@code instructions} — {@code 5}, or {@code ['a']} — which is not text a
 *       model can be asked.</li>
 *   <li>A {@code score} answer whose {@code score} is not finite is refused. CPython's
 *       {@code int()} refuses it too — {@code ValueError} for a NaN, {@code OverflowError} for
 *       an infinity — so the verdict matches and only the wording differs: this one names the
 *       field, as the rest of the answer-shape refusals do.</li>
 *   <li>An {@code anyOf}/{@code oneOf} that is neither falsy nor iterable — a number, or
 *       {@code true} — is a {@link SchemaException} naming the key, where the reference raises
 *       {@code TypeError: 'int' object is not iterable} from inside its own comprehension. The
 *       truthiness itself is reproduced: a falsy {@code anyOf} falls through to {@code oneOf}
 *       and a truthy non-array reaches the same "got 0" refusal the reference raises. This is
 *       the <b>one</b> exception to the rule that a refusal message matches character for
 *       character.</li>
 *   <li>A top level that is not an object is a <b>compile error</b>, not a refusal:
 *       {@link #plan} takes a {@link Map}, where the reference raises
 *       {@code expected a JSON schema object, got list}. A caller who reaches it through an
 *       unchecked cast — {@code (Map<String, Object>) Json.parse(text)}, which is the README's
 *       own idiom — gets a {@link ClassCastException} from the cast instead. The one case the
 *       static type does not cover, a null schema, is refused with the reference's message.</li>
 * </ul>
 */
public final class Decisions {

    /** The most properties one schema may have. Beyond this the states stop fitting a window. */
    public static final int MAX_PROPERTIES = 32;

    /** The most options one {@code enum} may have. */
    public static final int MAX_OPTIONS = 32;

    /** The most levels one numeric range may span before it should be an enum. */
    public static final int MAX_SCORE_LEVELS = 10;

    private Decisions() {
    }

    /**
     * One option of a choice: the label the model is shown, and the value the schema means.
     *
     * <p>Two fields and not one, because they are not the same thing for anything but a string
     * enum — {@code 20} is shown as {@code "20"} — and the projection has to go back the other
     * way.
     */
    public record Option(String label, Object value) {
    }

    /**
     * One planned property: the question to ask, and what is needed to read the answer back.
     *
     * @param kind     which of the three question types this property became
     * @param options  the choice's options in schema order; empty for a score or a noul
     * @param minimum  the value of level 0 for a score, null otherwise. Without it a scale that
     *     does not start at zero projects to the wrong number and nothing about the answer says so
     */
    public record Field(String name, Question.Type kind, Question question, List<Option> options,
                        Long minimum) {

        public Field {
            options = List.copyOf(options);
        }
    }

    /**
     * A validated schema: one field per property, in the schema's own order.
     *
     * <p>Held as a value because both halves of a decision need it and they happen at different
     * times — {@link #questions()} before the model runs, {@link #project} after. Re-deriving it
     * from the schema for the projection is what lets the two disagree.
     */
    public record Plan(List<Field> fields) {

        public Plan {
            fields = List.copyOf(fields);
        }

        /** The questions to ask, keyed by property name, in the schema's property order. */
        public Map<String, Question> questions() {
            Map<String, Question> out = new LinkedHashMap<>();
            for (Field field : fields) {
                out.put(field.name(), field.question());
            }
            return Collections.unmodifiableMap(out);
        }

        /** The answers projected onto the schema's values, with no gate applied. */
        public Map<String, Object> project(Map<String, Answer> answers) {
            return project(answers, null);
        }

        /**
         * The answers projected onto the schema's values.
         *
         * @param gate the {@link ConfidenceGate} report, or null when no gate ran. A field whose
         *     verdict is {@link ConfidenceGate.Gated#lowConfidence()} projects to <b>null</b>.
         *     The flag is read from the report rather than recomputed here, so the value a caller
         *     acts on and the state the gate reports cannot disagree
         * @return the values, in the schema's property order. A field with no answer is absent
         *     rather than null; nulls are legitimate values here, which is why the map is an
         *     unmodifiable {@link LinkedHashMap} and not {@code Map.copyOf}
         */
        public Map<String, Object> project(Map<String, Answer> answers,
                                           Map<String, ConfidenceGate.Gated> gate) {
            Map<String, Object> values = new LinkedHashMap<>();
            for (Field field : fields) {
                Answer answer = answers.get(field.name());
                if (answer == null) {
                    // Absent, not null. The reference skips the field entirely and the
                    // distinction is the point: a caller cannot tell "the model abstained" from
                    // "the question was never asked" if both arrive as null.
                    continue;
                }
                ConfidenceGate.Gated verdict = gate == null ? null : gate.get(field.name());
                if (verdict != null && verdict.lowConfidence()) {
                    values.put(field.name(), null);
                    continue;
                }
                values.put(field.name(), value(field, answer));
            }
            return Collections.unmodifiableMap(values);
        }

        private static Object value(Field field, Answer answer) {
            switch (field.kind()) {
                case NOUL: {
                    // `>=`, so exactly 0.5 is true. Spelled `>` this flips on that one value and
                    // on no other, which is why the fixture carries 0.5 and the double just
                    // below it.
                    return expect(field, answer, Answer.Noul.class).noul() >= 0.5;
                }
                case SCORE: {
                    Answer.Score score = expect(field, answer, Answer.Score.class);
                    long level;
                    if (!score.probabilities().isEmpty()) {
                        level = argmax(score.probabilities());
                    } else {
                        // Refused rather than narrowed. `(long)` on a non-finite double is silent
                        // -- NaN becomes 0 and +Infinity becomes Long.MAX_VALUE -- so
                        // `minimum + level` would project 1 for a NaN and
                        // -9223372036854775808 for an infinity onto a field the schema declared
                        // as 1..3, with nothing in the Decision saying so. CPython's `int()`
                        // refuses both (ValueError for a NaN, OverflowError for an infinity),
                        // and this is the one arithmetic path `expect` does not cover.
                        if (!Double.isFinite(score.score())) {
                            throw new IllegalArgumentException(String.format(
                                    "%s is a score field, so its score must be finite and is %s",
                                    field.name(), pythonStr(score.score())));
                        }
                        // `score` is already the 0-based LEVEL INDEX, so it is rounded on its own
                        // with no `minimum` subtracted first. Math.rint is Python's round: half to
                        // even. Math.round is half up and gives 3 for 2.5.
                        level = (long) Math.rint(score.score());
                    }
                    return (field.minimum() == null ? 0L : field.minimum()) + level;
                }
                default: {
                    String label = expect(field, answer, Answer.Choice.class).choice();
                    for (Option option : field.options()) {
                        if (option.label().equals(label)) {
                            return option.value();
                        }
                    }
                    // A label no option carries comes back as the label. The model is constrained
                    // to the options, so this only reaches a hand-built answer -- and raising
                    // would turn a caller's own test double into a crash.
                    return label;
                }
            }
        }

        /**
         * The index of the largest probability, keyed {@code "0"}, {@code "1"}, ...
         *
         * <p>Strictly greater, so a tie takes the <b>earliest</b> level, which is what Python's
         * {@code max} over a range does. A missing index counts as 0.0 rather than being skipped,
         * so the index space stays the level space.
         */
        private static long argmax(Map<String, Double> probabilities) {
            long best = 0;
            double highest = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < probabilities.size(); i++) {
                Double probability = probabilities.get(Integer.toString(i));
                double value = probability == null ? 0.0 : probability;
                if (value > highest) {
                    highest = value;
                    best = i;
                }
            }
            return best;
        }

        private static <T extends Answer> T expect(Field field, Answer answer, Class<T> wanted) {
            if (!wanted.isInstance(answer)) {
                // The reference reads a dict and would silently take the missing key's default,
                // projecting `minimum + 0` or `false` for an answer of the wrong shape. A typed
                // answer can say so instead, and this only reaches a caller who built the
                // answers by hand.
                throw new IllegalArgumentException(String.format(
                        "%s is a %s field, so its answer must be a %s and is a %s",
                        field.name(), field.kind().wireName(), wanted.getSimpleName(),
                        answer.getClass().getSimpleName()));
            }
            return wanted.cast(answer);
        }
    }

    /**
     * One decided state.
     *
     * <p>{@code values} is the schema-shaped output. The other maps are keyed by field, so a
     * caller filtering on confidence does not have to re-derive which answer belonged to which
     * property.
     *
     * @param answerConfidence {@code max(p)} per field — the probability mass on the answer being
     *     reported, which is the quantity {@code minConfidence} compares against and the one every
     *     calibration figure in laya is computed on. That is the reason to report it: a caller
     *     filtering this artifact to decide what to automate has to filter on the number the gate
     *     actually used. It is <b>not</b> a claim that the number is right — "about c of the
     *     answers returned at c are correct" holds only after temperatures have been fitted and
     *     validated on held-out data for that checkpoint and question shape, and
     *     {@code laya-multilingual} ships with no fitted temperatures at all. Empty for a field
     *     that reported no usable number, which is not the same as a reported 0.0
     * @param confidence the normalised-entropy value, a <b>different</b> quantity on a scale that
     *     depends on the label count. Kept under its own name for that reason
     * @param gate the {@link ConfidenceGate} report, or empty when no gate was configured. The
     *     presence of the report is how a caller tells "no gate ran" from "everything passed"
     */
    public record Decision(Map<String, Object> values, Map<String, Double> confidence,
                           Map<String, OptionalDouble> answerConfidence,
                           Map<String, Map<String, Double>> probabilities,
                           Map<String, Answer> answers, Usage usage,
                           Optional<Map<String, ConfidenceGate.Gated>> gate) {

        public Decision {
            // `values` legitimately holds nulls -- a gated field is null -- and `Map.copyOf`
            // rejects them, so the unmodifiable wrapper is the only option that keeps both the
            // nulls and the schema's property order.
            values = Collections.unmodifiableMap(new LinkedHashMap<>(values));
            confidence = Map.copyOf(confidence);
            answerConfidence = Map.copyOf(answerConfidence);
            probabilities = Map.copyOf(probabilities);
            answers = Map.copyOf(answers);
        }
    }

    // ---------------------------------------------------------------- schema -> questions

    /** Validates a JSON schema and returns one planned field per property. */
    public static Plan plan(Map<String, Object> schema) {
        if (schema == null) {
            // The reference's `isinstance(schema, dict)` check has one reachable case here and
            // this is it: `Map` is the guard for a list, a string or a number, and null is
            // assignable to it. Without this the refusal is an NPE from `schema.get`, which
            // names neither the argument nor what was wrong with it.
            throw new SchemaException("expected a JSON schema object, got NoneType");
        }
        Object type = schema.get("type");
        if (!(type == null || "object".equals(type)) || !schema.containsKey("properties")) {
            throw new SchemaException("the top level must be an object with 'properties'");
        }
        Object raw = schema.get("properties");
        if (!(raw instanceof Map) || ((Map<?, ?>) raw).isEmpty()) {
            throw new SchemaException("'properties' must be a non-empty object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> properties = (Map<String, Object>) raw;
        if (properties.size() > MAX_PROPERTIES) {
            throw new SchemaException(properties.size() + " properties exceeds MAX_PROPERTIES="
                    + MAX_PROPERTIES);
        }
        List<Field> fields = new ArrayList<>(properties.size());
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            fields.add(field("properties." + entry.getKey(), entry.getKey(), entry.getValue(),
                    schema, List.of()));
        }
        return new Plan(fields);
    }

    /** Turns a JSON schema into Laya questions, in the schema's property order. */
    public static Map<String, Question> questions(Map<String, Object> schema) {
        return plan(schema).questions();
    }

    /** Projects answers onto a schema's values: the choice value, the integer level, the boolean. */
    public static Map<String, Object> answersToJson(Map<String, Answer> answers,
                                                    Map<String, Object> schema) {
        return plan(schema).project(answers);
    }

    @SuppressWarnings("unchecked")
    private static Field field(String path, String name, Object raw, Map<String, Object> root,
                               List<String> seen) {
        if (!(raw instanceof Map)) {
            throw new SchemaException(path + ": property must be an object, got "
                    + pythonType(raw));
        }
        Map<String, Object> prop = (Map<String, Object>) raw;
        while (prop.containsKey("$ref")) {
            Object ref = prop.get("$ref");
            String spelled = pythonRepr(ref);
            if (seen.contains(spelled)) {
                throw new SchemaException(path + ": $ref " + spelled
                        + " is recursive; flatten the schema");
            }
            Map<String, Object> target = definition(root, ref);
            if (target == null) {
                throw new SchemaException(path + ": $ref " + spelled + " does not resolve to an"
                        + " entry of this schema's '$defs' or 'definitions'");
            }
            // The definition's own `description` is DROPPED. pydantic fills it from the enum's
            // docstring -- v1 writes "An enumeration." when there is none -- which describes the
            // type rather than asking about this field, so the wording stays the field's and an
            // `Enum` asks what the same `Literal` would.
            Map<String, Object> merged = new LinkedHashMap<>();
            for (Map.Entry<String, Object> entry : target.entrySet()) {
                if (!"description".equals(entry.getKey())) {
                    merged.put(entry.getKey(), entry.getValue());
                }
            }
            for (Map.Entry<String, Object> entry : prop.entrySet()) {
                if (!"$ref".equals(entry.getKey())) {
                    merged.put(entry.getKey(), entry.getValue());
                }
            }
            List<String> followed = new ArrayList<>(seen);
            followed.add(spelled);
            // A `$ref` the definition itself carries is an alias, followed by this same loop.
            prop = merged;
            seen = followed;
        }

        String description = describe(path, prop.get("description"));
        if (!prop.containsKey("const") && !prop.containsKey("enum") && !prop.containsKey("type")) {
            // pydantic v1 wraps a described `$ref` as {"allOf": [{"$ref": ...}], "description":
            // ...}; a one-item `allOf` IS that schema, so it is unwrapped with the outer keys on
            // top.
            Object wrapped = prop.get("allOf");
            if (wrapped instanceof List && ((List<?>) wrapped).size() == 1
                    && ((List<?>) wrapped).get(0) instanceof Map) {
                Map<String, Object> branch =
                        new LinkedHashMap<>((Map<String, Object>) ((List<?>) wrapped).get(0));
                for (Map.Entry<String, Object> entry : prop.entrySet()) {
                    if (!"allOf".equals(entry.getKey())) {
                        branch.put(entry.getKey(), entry.getValue());
                    }
                }
                return field(path, name, branch, root, seen);
            }
            // pydantic v2 renders Optional[X] as {"anyOf": [<X>, {"type": "null"}]} with no
            // top-level type -- the same nullable shape the list form `type: ["string", "null"]`
            // handles below. The single non-null branch is unwrapped so Optional[Literal[...]]
            // and friends map; a union of two real types is genuinely ambiguous and refused.
            //
            // A FALSY `anyOf` falls through to `oneOf`, because the reference spells this
            // `prop.get("anyOf") or prop.get("oneOf")`. Falsy is Python's, not "not a list":
            // an empty list, an empty object, an empty string, zero and false are all falsy
            // there and all fall through. What survives is handled only when it is an array: the
            // reference guards with `isinstance(union, list)`, so a truthy `{"anyOf": "abc"}`, a
            // truthy `{"anyOf": {...}}`, a `{"anyOf": 5}` or a `{"anyOf": true}` is neither
            // iterated nor unwrapped -- it falls past this block to the terminal "unsupported
            // schema" below. Iterating them used to reach the `got 0` refusal, which blamed a
            // union arity the request never had, and a non-iterable leaked a TypeError.
            Object union = prop.get("anyOf");
            if (!truthy(union)) {
                union = prop.get("oneOf");
            }
            if (union instanceof List) {
                List<Map<String, Object>> branches = new ArrayList<>();
                for (Object candidate : (List<?>) union) {
                    if (candidate instanceof Map
                            && !"null".equals(((Map<?, ?>) candidate).get("type"))) {
                        branches.add((Map<String, Object>) candidate);
                    }
                }
                if (branches.size() != 1) {
                    throw new SchemaException(path + ": only 'Optional[...]' unions (one non-null"
                            + " branch) are supported, got " + branches.size());
                }
                Map<String, Object> branch = new LinkedHashMap<>(branches.get(0));
                // The outer description is carried onto the branch only when the branch has none.
                // `containsKey`, not a null test: a branch that spells out `"description": null`
                // has one, and the reference's `setdefault` leaves it alone.
                if (!branch.containsKey("description")) {
                    branch.put("description", prop.get("description"));
                }
                return field(path, name, branch, root, seen);
            }
        }

        // `containsKey`, because `{"const": null}` is a const of null and not an absent key. A
        // port testing `get("const") != null` drops that field into "unsupported schema".
        if (prop.containsKey("const")) {
            List<Object> single = new ArrayList<>(1);
            single.add(prop.get("const"));
            return enumField(path, name, single, description);
        }
        if (prop.containsKey("enum")) {
            Object values = prop.get("enum");
            if (!(values instanceof List)) {
                // Python would take `len()` of whatever this is and iterate it -- a string enum
                // of "abc" becomes three one-character options. Refused here rather than
                // reproduced: the reference's behaviour there is an artifact of duck typing, not
                // a contract, and no JSON schema spells it.
                throw new SchemaException(path + ": 'enum' must be an array, got "
                        + pythonType(values));
            }
            return enumField(path, name, (List<Object>) values, description);
        }

        Object jtype = prop.get("type");
        if (jtype instanceof List) {               // nullable: ["string", "null"]
            List<Object> nonNull = new ArrayList<>();
            for (Object candidate : (List<?>) jtype) {
                if (!"null".equals(candidate)) {
                    nonNull.add(candidate);
                }
            }
            if (nonNull.size() > 1) {
                throw new SchemaException(path
                        + ": 'type' has multiple non-null types; unions are not supported");
            }
            jtype = nonNull.isEmpty() ? null : nonNull.get(0);
        }
        if ("boolean".equals(jtype)) {
            return noulField(name, description);
        }
        if ("string".equals(jtype)) {
            throw new SchemaException(path
                    + ": a free string cannot be a fixed option set; use 'enum' or a boolean");
        }
        if ("integer".equals(jtype) || "number".equals(jtype)) {
            return scoreField(path, name, prop, description);
        }
        if ("array".equals(jtype)) {
            throw new SchemaException(path
                    + ": arrays are not supported; ask one field per element");
        }
        if ("object".equals(jtype)) {
            throw new SchemaException(path
                    + ": nested objects are not supported; flatten the schema");
        }
        throw new SchemaException(path + ": unsupported schema " + pythonRepr(prop));
    }

    private static Field enumField(String path, String name, List<Object> values,
                                   String description) {
        if (values.size() > MAX_OPTIONS) {
            throw new SchemaException(path + ": " + values.size() + " options exceeds MAX_OPTIONS="
                    + MAX_OPTIONS);
        }
        if (values.isEmpty()) {
            throw new SchemaException(path + ": 'enum' must not be empty");
        }
        boolean allBoolean = true;
        for (Object value : values) {
            allBoolean &= value instanceof Boolean;
        }
        if (allBoolean) {
            // An enum of nothing but booleans is a noul, not a two-option choice: the labels
            // would be "True"/"False" and the head would be asked to pick between them.
            return noulField(name, description);
        }
        List<Option> options = new ArrayList<>(values.size());
        Set<String> labels = new LinkedHashSet<>();
        for (Object value : values) {
            String label = value == null ? "null" : pythonStr(value);
            options.add(new Option(label, value));
            labels.add(label);
        }
        if (labels.size() != options.size()) {
            throw new SchemaException(path + ": enum values produce duplicate choice labels");
        }
        Map<String, Object> criteria = new LinkedHashMap<>();
        for (Option option : options) {
            criteria.put(option.label(), null);
        }
        Question question = Question.choice(
                description != null ? description : "What is `" + name + "`?", criteria);
        return new Field(name, Question.Type.CHOICE, question, options, null);
    }

    private static Field noulField(String name, String description) {
        Question question = Question.noul(
                description != null ? description : "Is `" + name + "` true?");
        return new Field(name, Question.Type.NOUL, question, List.of(), null);
    }

    private static Field scoreField(String path, String name, Map<String, Object> prop,
                                    String description) {
        Object lo = prop.get("minimum");
        Object hi = prop.get("maximum");
        if (!isInteger(lo) || !isInteger(hi)) {
            throw new SchemaException(path + ": a numeric field needs integer 'minimum' and"
                    + " 'maximum' to become a score");
        }
        long low = ((Number) lo).longValue();
        long high = ((Number) hi).longValue();
        if (high < low) {
            throw new SchemaException(String.format("%s: 'maximum' %d is below 'minimum' %d",
                    path, high, low));
        }
        long span = high - low + 1;
        if (span > MAX_SCORE_LEVELS) {
            throw new SchemaException(String.format(
                    "%s: %d levels exceeds MAX_SCORE_LEVELS=%d; narrow the range or use an enum",
                    path, span, MAX_SCORE_LEVELS));
        }
        List<String> levels = new ArrayList<>((int) span);
        for (long level = low; level <= high; level++) {
            levels.add(Long.toString(level));
        }
        Question question = Question.score(
                description != null ? description
                        : String.format("Score `%s` from %d to %d", name, low, high),
                levels);
        return new Field(name, Question.Type.SCORE, question, List.of(), low);
    }

    /**
     * Python's truth value of a parsed-JSON value, which is what {@code a or b} tests.
     *
     * <p>Empty is false: {@code []}, <code>{}</code>, {@code ""}, {@code 0}, {@code 0.0} and
     * {@code false} are all falsy, which is why a schema may spell any of them and still mean
     * "look at the other key". A Java truth test on the same value would be a null check and
     * would differ on every one of them.
     */
    private static boolean truthy(Object value) {
        if (value == null) {
            return false;
        }
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof String) {
            return !((String) value).isEmpty();
        }
        if (value instanceof List) {
            return !((List<?>) value).isEmpty();
        }
        if (value instanceof Map) {
            return !((Map<?, ?>) value).isEmpty();
        }
        if (value instanceof BigInteger) {
            return ((BigInteger) value).signum() != 0;
        }
        if (value instanceof Number) {
            // `0.0`, `-0.0` and `0` are all falsy; a NaN is truthy, as every non-zero float is.
            return ((Number) value).doubleValue() != 0.0;
        }
        return true;
    }

    /**
     * Whether a bound counts as an integer.
     *
     * <p>A {@code Boolean} does not, where Python's {@code isinstance(x, int)} says it does
     * because {@code bool} subclasses {@code int} there. The reference would accept
     * {@code {"minimum": false, "maximum": true}} and ask "Score `x` from 0 to 1"; a JSON schema
     * that spells a bound as a boolean is a mistake, and refusing it is the more useful answer.
     */
    private static boolean isInteger(Object value) {
        return (value instanceof Long || value instanceof Integer || value instanceof BigInteger)
                && !(value instanceof Boolean);
    }

    /**
     * The instruction text a {@code description} supplies, or null to generate one.
     *
     * <p>Empty means "generate one", matching the reference's {@code description or ...}. A
     * description that is not text is refused rather than stringified: the reference would put
     * the value itself into the question's {@code instructions}, which is not text the model can
     * be asked.
     */
    private static String describe(String path, Object raw) {
        if (raw == null) {
            return null;
        }
        if (raw instanceof String) {
            return ((String) raw).isEmpty() ? null : (String) raw;
        }
        throw new SchemaException(path + ": 'description' must be a string, got "
                + pythonType(raw));
    }

    /** The {@code $defs} or {@code definitions} entry a local {@code $ref} names, or null. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> definition(Map<String, Object> root, Object ref) {
        if (!(ref instanceof String)) {
            return null;
        }
        Map<String, Object> found = null;
        // Only this schema's own definitions are looked up; anything else -- a remote ref, a
        // pointer into another document -- is refused rather than fetched.
        String[][] locals = {{"#/$defs/", "$defs"}, {"#/definitions/", "definitions"}};
        for (String[] local : locals) {
            Object defs = root.get(local[1]);
            if (((String) ref).startsWith(local[0]) && defs instanceof Map) {
                Object target = ((Map<String, Object>) defs)
                        .get(((String) ref).substring(local[0].length()));
                if (target instanceof Map) {
                    found = (Map<String, Object>) target;
                }
            }
        }
        return found;
    }

    // ---------------------------------------------------------------- deciding

    /** Decides one state against a schema, ungated. */
    public static Decision decide(Predictor runner, Object state, Map<String, Object> schema) {
        return decide(runner, state, schema, null);
    }

    /**
     * Decides one state, ungated: pass exactly <b>one</b> of {@code schema} or
     * {@code questions}.
     *
     * <p>The ungated spelling of the five-argument form, and the reason it exists is that the
     * five-argument form is <b>ambiguous</b> on a bare {@code null} -- both the {@link Double}
     * and the {@code Map} threshold overload match, so "no gate" could only be written
     * {@code (Double) null}. A cast is not an argument, and an explicit-questions caller who
     * wants no gate should not have to pick one of two threshold types to say so.
     *
     * @throws IllegalArgumentException if both or neither of {@code schema} and
     *     {@code questions} is given
     */
    public static Decision decide(Predictor runner, Object state, Map<String, Object> schema,
                                  Map<String, Question> questions) {
        return decide(runner, state, schema, questions, (Double) null);
    }

    /**
     * Decides one state.
     *
     * <p>Pass exactly <b>one</b> of {@code schema} or {@code questions}. With a schema the values
     * follow the schema; with explicit questions they are the raw answers, as the reference
     * returns them.
     *
     * @param minConfidence the abstention threshold, or null for no gate. A field whose answer
     *     falls below it comes back as null, with the answer itself kept in the decision
     * @throws IllegalArgumentException if both or neither of {@code schema} and {@code questions}
     *     is given
     */
    public static Decision decide(Predictor runner, Object state, Map<String, Object> schema,
                                  Map<String, Question> questions, Double minConfidence) {
        return decide(runner, state, schema, questions,
                answers -> ConfidenceGate.apply(answers, minConfidence));
    }

    /** Decides one state, gating each answer at its own option-count bucket's threshold. */
    public static Decision decide(Predictor runner, Object state, Map<String, Object> schema,
                                  Map<String, Question> questions,
                                  Map<String, ? extends Number> minConfidence) {
        return decide(runner, state, schema, questions,
                answers -> ConfidenceGate.apply(answers, minConfidence));
    }

    /** Decides many states against one schema in one batched call, ungated. */
    public static List<Decision> decideBatch(BatchPredictor runner, List<?> states,
                                             Map<String, Object> schema) {
        return decideBatch(runner, states, schema, null);
    }

    /**
     * Decides many states in one batched call, ungated: pass exactly <b>one</b> of
     * {@code schema} or {@code questions}.
     *
     * <p>The batched counterpart of {@link #decide(Predictor, Object, Map, Map)}, and there for
     * the same reason: the five-argument form cannot express "no gate" without a cast.
     *
     * @throws IllegalArgumentException if both or neither of {@code schema} and
     *     {@code questions} is given
     */
    public static List<Decision> decideBatch(BatchPredictor runner, List<?> states,
                                             Map<String, Object> schema,
                                             Map<String, Question> questions) {
        return decideBatch(runner, states, schema, questions, (Double) null);
    }

    /**
     * Decides many states against one schema in one batched call, in input order.
     *
     * <p>The throughput form of {@link #decide}: the schema is planned once and its questions are
     * evaluated over every state through {@link BatchPredictor#predictBatch} — the same shared
     * forward pass {@link Agent#predictBatch} runs — then each state's answers are projected
     * exactly as {@code decide} projects them.
     *
     * <p>A runner that cannot batch is a compile error here rather than the reference's
     * {@code TypeError}, because looping {@code decide} N times behind the caller's back is the
     * one thing neither wants to do silently.
     */
    public static List<Decision> decideBatch(BatchPredictor runner, List<?> states,
                                             Map<String, Object> schema,
                                             Map<String, Question> questions,
                                             Double minConfidence) {
        return decideBatch(runner, states, schema, questions,
                answers -> ConfidenceGate.apply(answers, minConfidence));
    }

    /** Decides many states, gating each answer at its own option-count bucket's threshold. */
    public static List<Decision> decideBatch(BatchPredictor runner, List<?> states,
                                             Map<String, Object> schema,
                                             Map<String, Question> questions,
                                             Map<String, ? extends Number> minConfidence) {
        return decideBatch(runner, states, schema, questions,
                answers -> ConfidenceGate.apply(answers, minConfidence));
    }

    /** How a decision's answers are gated: one of {@link ConfidenceGate}'s two {@code apply}s. */
    private interface Gating {
        Optional<Map<String, ConfidenceGate.Gated>> of(Map<String, Answer> answers);
    }

    private static Decision decide(Predictor runner, Object state, Map<String, Object> schema,
                                   Map<String, Question> questions, Gating gating) {
        Plan plan = planFor(schema, questions);
        Map<String, Question> asked = plan == null ? questions : plan.questions();
        return decided(plan, runner.predict(state, asked), gating);
    }

    private static List<Decision> decideBatch(BatchPredictor runner, List<?> states,
                                              Map<String, Object> schema,
                                              Map<String, Question> questions, Gating gating) {
        Plan plan = planFor(schema, questions);
        Map<String, Question> asked = plan == null ? questions : plan.questions();
        List<Prediction> predictions = runner.predictBatch(states, asked);
        List<Decision> out = new ArrayList<>(predictions.size());
        for (Prediction prediction : predictions) {
            out.add(decided(plan, prediction, gating));
        }
        return List.copyOf(out);
    }

    private static Plan planFor(Map<String, Object> schema, Map<String, Question> questions) {
        if ((schema == null) == (questions == null)) {
            throw new IllegalArgumentException("pass exactly one of schema= or questions=");
        }
        return schema == null ? null : plan(schema);
    }

    private static Decision decided(Plan plan, Prediction prediction, Gating gating) {
        Map<String, Answer> answers = prediction.answers();
        Optional<Map<String, ConfidenceGate.Gated>> gate = gating.of(answers);
        Map<String, Object> values;
        if (plan == null) {
            // The explicit-questions path returns the raw answers as the values, which is what
            // the reference's `dict(answers)` does. There is no schema to project onto, so the
            // gate's verdict is in `gate` and nothing is nulled.
            values = new LinkedHashMap<>(answers);
        } else {
            values = plan.project(answers, gate.orElse(null));
        }
        Map<String, Double> confidence = new LinkedHashMap<>();
        Map<String, OptionalDouble> answerConfidence = new LinkedHashMap<>();
        Map<String, Map<String, Double>> probabilities = new LinkedHashMap<>();
        for (Map.Entry<String, Answer> entry : answers.entrySet()) {
            Answer answer = entry.getValue();
            confidence.put(entry.getKey(), answer.confidence());
            // Read through the gate's own accessor, so this and the threshold it was gated at
            // cannot disagree about which quantity is being reported.
            answerConfidence.put(entry.getKey(), ConfidenceGate.answerConfidence(answer));
            if (answer instanceof Answer.Noul) {
                // A noul has no probability map of its own; the reference reports the two-sided
                // distribution it implies, rounded as every other number in an answer is.
                double p = ((Answer.Noul) answer).noul();
                Map<String, Double> sides = new LinkedHashMap<>();
                sides.put("false", Rounding.round4(1.0 - p));
                sides.put("true", Rounding.round4(p));
                probabilities.put(entry.getKey(), Collections.unmodifiableMap(sides));
            } else if (answer instanceof Answer.Choice) {
                probabilities.put(entry.getKey(), ((Answer.Choice) answer).probabilities());
            } else {
                probabilities.put(entry.getKey(), ((Answer.Score) answer).probabilities());
            }
        }
        return new Decision(values, confidence, answerConfidence, probabilities, answers,
                prediction.usage(), gate);
    }

    // ---------------------------------------------------------------- Python spellings

    /**
     * {@code str(value)} for an option label.
     *
     * <p>The label is what the model is shown and what the answer comes back keyed by, so it has
     * to be CPython's spelling and not Java's: {@code true} is {@code True} there, and a float is
     * whatever {@link PythonJson} reproduces — {@code 1e+16}, not {@code 1.0E16}.
     */
    private static String pythonStr(Object value) {
        if (value instanceof String) {
            return (String) value;
        }
        return pythonRepr(value);
    }

    /**
     * {@code repr(value)} for a parsed-JSON value, which is what the refusal messages interpolate.
     *
     * <p>The message is the contract — the fixtures assert it character for character — and
     * {@code "unsupported schema %r"} puts a whole schema fragment into one.
     */
    private static String pythonRepr(Object value) {
        if (value == null) {
            return "None";
        }
        if (value instanceof Boolean) {
            return ((Boolean) value) ? "True" : "False";
        }
        if (value instanceof String) {
            return PythonJson.repr((String) value);
        }
        if (value instanceof Double || value instanceof Float) {
            double number = ((Number) value).doubleValue();
            // CPython's `repr` spells the non-finite floats in LOWER CASE -- `nan`, `inf`,
            // `-inf` -- where `json.dumps` writes `NaN`, `Infinity`, `-Infinity`.
            // PythonJson.repr(double) reproduces the `dumps` spelling, which is right for
            // `dumps` and wrong here: this is `%r` and `str`, so a refusal message quotes the
            // fragment the way CPython would, and a choice LABEL -- what the model is shown and
            // what the answer comes back keyed by -- is the same string on both sides. Not
            // reachable through Json.parse, which rejects a bare NaN, and fully reachable from
            // the hand-built Map the public `questions(Map)` signature takes.
            if (Double.isNaN(number)) {
                return "nan";
            }
            if (Double.isInfinite(number)) {
                return number > 0 ? "inf" : "-inf";
            }
            return PythonJson.repr(number);
        }
        if (value instanceof Number) {
            return value.toString();
        }
        if (value instanceof List) {
            StringBuilder out = new StringBuilder("[");
            for (Object item : (List<?>) value) {
                if (out.length() > 1) {
                    out.append(", ");
                }
                out.append(pythonRepr(item));
            }
            return out.append("]").toString();
        }
        if (value instanceof Map) {
            StringBuilder out = new StringBuilder("{");
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                if (out.length() > 1) {
                    out.append(", ");
                }
                out.append(pythonRepr(entry.getKey())).append(": ")
                        .append(pythonRepr(entry.getValue()));
            }
            return out.append("}").toString();
        }
        throw new IllegalArgumentException(
                "not a parsed-JSON value: " + value.getClass().getName());
    }

    /** {@code type(value).__name__}, which two of the refusal messages name. */
    private static String pythonType(Object value) {
        if (value == null) {
            return "NoneType";
        }
        if (value instanceof String) {
            return "str";
        }
        if (value instanceof Boolean) {
            return "bool";
        }
        if (value instanceof Double || value instanceof Float) {
            return "float";
        }
        if (value instanceof Number) {
            return "int";
        }
        if (value instanceof List) {
            return "list";
        }
        if (value instanceof Map) {
            return "dict";
        }
        return value.getClass().getSimpleName();
    }
}
