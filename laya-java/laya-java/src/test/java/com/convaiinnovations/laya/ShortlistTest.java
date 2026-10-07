package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@link Shortlist} against the rankings recorded from {@code laya.shortlist}.
 *
 * <p>A real bi-encoder cannot go in a fixture, and it is not what needs gating: the embedder is
 * the caller's, and what the module decides is the ranking around it. So both sides embed with the
 * same deterministic stand-in -- FNV-1a over the UTF-8 bytes of {@code text:component}, mapped into
 * [-1, 1) -- which is reproducible in any language with 64-bit integers, with no floating point,
 * no library and no seeded generator whose stream differs between runtimes.
 *
 * <p>What that lets the fixture pin is every rule that decides which labels survive: the stable
 * tie order, the signed cosine, the zero-norm query, the non-finite cleaning, and the passthrough
 * that must not call the embedder at all.
 */
class ShortlistTest {

    private static final int EMBED_DIM = 8;
    private static final long FNV_OFFSET = 0xCBF29CE484222325L;
    private static final long FNV_PRIME = 0x100000001B3L;

    private static Map<String, Object> fixture() {
        return Fixtures.load("shortlist.json");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object value) {
        return (List<Object>) value;
    }

    /**
     * The stand-in embedder, byte for byte what the generator used.
     *
     * <p>{@code Long.remainderUnsigned} rather than {@code %}: the hash is a 64-bit unsigned value
     * and Java's remainder is signed, so half the components would come out negative and every
     * ranking would differ.
     */
    private static double[][] stubEmbed(List<String> texts) {
        double[][] rows = new double[texts.size()][EMBED_DIM];
        for (int row = 0; row < texts.size(); row++) {
            String text = texts.get(row) == null ? "" : texts.get(row);
            for (int component = 0; component < EMBED_DIM; component++) {
                byte[] key = (text + ":" + component).getBytes(StandardCharsets.UTF_8);
                long hash = FNV_OFFSET;
                for (byte value : key) {
                    hash = (hash ^ (value & 0xFFL)) * FNV_PRIME;
                }
                rows[row][component] = Long.remainderUnsigned(hash, 2000L) / 1000.0 - 1.0;
            }
        }
        return rows;
    }

    /**
     * A recorded cosine against a computed one, to the precision the fixture records.
     *
     * <p>The one tolerance in this suite, and it is not a convenience. The reference computes its
     * scores with a BLAS matrix-vector product, and how BLAS blocks that accumulation differs
     * between implementations -- measured on this project: with Apple Accelerate, one row of a
     * four-row product differs in the last bit from the same row computed alone, and CI's
     * OpenBLAS blocks it differently again. So the reference's own score is not reproducible
     * across machines and no port can match it bit for bit.
     *
     * <p>The fixture therefore records nine decimals, which is roughly three orders of magnitude
     * tighter than any porting error this repository has seen and seven orders looser than the
     * 1e-16 BLAS spread. What is still compared exactly is the label ORDER -- the only thing a
     * consumer observes, and the thing a ranking bug actually moves.
     */
    private static void assertScore(double expected, double actual, String what) {
        double tolerance = 5e-10;
        assertTrue(Math.abs(expected - actual) <= tolerance,
                what + ": expected " + expected + " +/- " + tolerance + ", got " + actual
                + " (difference " + Math.abs(expected - actual) + ")");
    }

    /** A double from the fixture, where a non-finite value is tagged as a string. */
    private static double number(Object value) {
        if (value instanceof String) {
            switch ((String) value) {
                case "nan":
                    return Double.NaN;
                case "inf":
                    return Double.POSITIVE_INFINITY;
                case "-inf":
                    return Double.NEGATIVE_INFINITY;
                default:
                    throw new IllegalArgumentException("not a tagged number: " + value);
            }
        }
        return ((Number) value).doubleValue();
    }

    private static Object state(Map<String, Object> encoded) {
        return encoded.get("value");
    }

    /**
     * The recorded criteria as a choice question.
     *
     * <p>The reference accepts a bare list of labels as criteria; a typed {@link Question} always
     * has a map, so a list becomes a map with null descriptions -- which is exactly what the
     * reference does with it internally, and the recorded option texts prove it.
     */
    private static Map<String, Object> criteriaOf(Object criteria) {
        Map<String, Object> options = new LinkedHashMap<>();
        if (criteria instanceof List) {
            for (Object label : list(criteria)) {
                options.put((String) label, null);
            }
        } else {
            options.putAll(map(criteria));
        }
        return options;
    }

    @Test
    @DisplayName("the stand-in embedder matches the one the fixture was recorded with")
    void embedderMatchesTheGenerator() {
        // Checked first and on its own: if this drifts, every ranking below fails for a reason
        // that has nothing to do with the ranking, and the failures would be unreadable.
        Map<String, Object> probe = map(fixture().get("embed_probe"));
        assertEquals(EMBED_DIM, ((Number) fixture().get("embed_dim")).intValue());
        for (Map.Entry<String, Object> entry : probe.entrySet()) {
            double[] expected = new double[EMBED_DIM];
            List<Object> recorded = list(entry.getValue());
            for (int i = 0; i < EMBED_DIM; i++) {
                expected[i] = number(recorded.get(i));
            }
            double[] actual = stubEmbed(List.of(entry.getKey()))[0];
            for (int i = 0; i < EMBED_DIM; i++) {
                assertEquals(expected[i], actual[i],
                        "component " + i + " of " + PythonJsonQuote.of(entry.getKey()));
            }
        }
        assertTrue(probe.size() >= 3, "only " + probe.size() + " probe texts");
    }

    @TestFactory
    @DisplayName("every recorded ranking matches, label order and score")
    List<DynamicTest> rankings() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : list(fixture().get("cases"))) {
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> assertCase(name, row)));
        }
        return tests;
    }

    private void assertCase(String name, Map<String, Object> row) {
        Object recordedState = state(map(row.get("state")));
        String instructions = (String) row.get("instructions");
        Map<String, Object> criteria = criteriaOf(row.get("criteria"));
        int k = ((Number) row.get("k")).intValue();

        // The two inputs to the ranking, asserted before the ranking itself, so a failure says
        // which of the three stages broke rather than only that the answer differs.
        assertEquals(row.get("query_text"),
                Shortlist.queryText(recordedState,
                        instructions == null || instructions.isEmpty() ? null : instructions),
                name + ": the query text");
        List<String> expectedOptions = new ArrayList<>();
        for (Object text : list(row.get("option_texts"))) {
            expectedOptions.add((String) text);
        }
        assertEquals(expectedOptions, Question.choice("placeholder", criteria).renderOptions(),
                name + ": the option texts");

        AtomicInteger calls = new AtomicInteger();
        Shortlist.Embedder counting = texts -> {
            calls.incrementAndGet();
            return stubEmbed(texts);
        };
        // The criteria overload, because a recorded case may have had no instruction at all --
        // which is the reference's default and which the Question overload cannot express.
        Shortlist.Ranking ranking = Shortlist.rank(recordedState, criteria, counting, k,
                instructions);

        List<String> expectedLabels = new ArrayList<>();
        for (Object label : list(row.get("labels"))) {
            expectedLabels.add((String) label);
        }
        assertEquals(expectedLabels, ranking.labels(), name + ": the kept labels, in rank order");
        assertEquals(row.get("passthrough"), ranking.passthrough(), name + ": passthrough");
        assertEquals(((Number) row.get("n")).intValue(), ranking.total(), name + ": label count");

        if (Boolean.TRUE.equals(row.get("passthrough"))) {
            assertNull(ranking.scores(), name + ": a passthrough reports no scores");
            assertEquals(0, calls.get(),
                    name + ": a passthrough must NOT call the embedder, and called it "
                    + calls.get() + " times");
        } else {
            List<Object> recordedScores = list(row.get("scores"));
            double[] scores = ranking.scores();
            assertNotNull(scores, name + ": scores");
            assertEquals(recordedScores.size(), scores.length, name + ": score count");
            for (int i = 0; i < scores.length; i++) {
                assertScore(number(recordedScores.get(i)), scores[i],
                        name + ": score " + i + " (" + ranking.labels().get(i) + ")");
            }
            assertEquals(1, calls.get(), name + ": the embedder is called exactly once");
            // The reduced criteria the model will be shown: the kept labels, in rank order. The
            // reference records that mapping separately, so this checks the order survives the
            // subsetting and is not merely the criteria order again.
            // The reference subsets a dict to a dict and a list to a list, so the recorded shape
            // follows the criteria it was given. Both carry the same thing: the kept labels in
            // rank order.
            Object recordedSubset = row.get("subset_criteria");
            List<String> subsetLabels = new ArrayList<>();
            if (recordedSubset instanceof Map) {
                for (Object label : map(recordedSubset).keySet()) {
                    subsetLabels.add((String) label);
                }
            } else {
                for (Object label : list(recordedSubset)) {
                    subsetLabels.add((String) label);
                }
            }
            assertEquals(subsetLabels, ranking.labels(),
                    name + ": the reduced criteria are the kept labels in rank order");
        }
    }

    @TestFactory
    @DisplayName("the cosine matches on every vector shape, including the ones that overflow")
    List<DynamicTest> cosine() {
        List<DynamicTest> tests = new ArrayList<>();
        int nonFinite = 0;
        for (Object entry : list(fixture().get("cosine"))) {
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            List<Object> recordedSims = list(row.get("sims"));
            for (Object value : recordedSims) {
                if (value instanceof String) {
                    nonFinite++;
                }
            }
            tests.add(DynamicTest.dynamicTest(name, () -> {
                List<Object> queryValues = list(row.get("query"));
                double[] query = new double[queryValues.size()];
                for (int i = 0; i < query.length; i++) {
                    query[i] = number(queryValues.get(i));
                }
                List<Object> docRows = list(row.get("docs"));
                double[][] matrix = new double[docRows.size() + 1][];
                matrix[0] = query;
                for (int r = 0; r < docRows.size(); r++) {
                    List<Object> values = list(docRows.get(r));
                    matrix[r + 1] = new double[values.size()];
                    for (int c = 0; c < values.size(); c++) {
                        matrix[r + 1][c] = number(values.get(c));
                    }
                }
                double[] sims = Shortlist.cosine(query, matrix, 1);
                assertEquals(recordedSims.size(), sims.length, name + ": count");
                for (int i = 0; i < sims.length; i++) {
                    double expected = number(recordedSims.get(i));
                    if (Double.isNaN(expected)) {
                        assertTrue(Double.isNaN(sims[i]),
                                name + ": sim " + i + " must be NaN, got " + sims[i]);
                    } else {
                        assertScore(expected, sims[i], name + ": sim " + i);
                    }
                }
            }));
        }
        assertTrue(nonFinite >= 1, "no recorded cosine is non-finite, so the case where a norm"
                + " overflows to infinity is not covered -- and clamping that to a number would"
                + " invent a ranking the reference does not have");
        return tests;
    }

    @TestFactory
    @DisplayName("a non-finite component becomes zero rather than spreading or raising")
    List<DynamicTest> nonFiniteCleaning() {
        List<DynamicTest> tests = new ArrayList<>();
        java.util.Set<String> kinds = new java.util.LinkedHashSet<>();
        for (Object entry : list(fixture().get("nonfinite"))) {
            kinds.add((String) map(entry).get("name"));
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> {
                List<Object> rawRows = list(row.get("rows"));
                double[][] raw = new double[rawRows.size()][];
                for (int r = 0; r < rawRows.size(); r++) {
                    List<Object> values = list(rawRows.get(r));
                    raw[r] = new double[values.size()];
                    for (int c = 0; c < values.size(); c++) {
                        raw[r][c] = number(values.get(c));
                    }
                }
                double[][] cleaned = Shortlist.embeddings(texts -> raw, List.of("q", "d"));
                List<Object> expectedRows = list(row.get("cleaned"));
                // Shape first: the inner loops below walk the EXPECTED rows, so an extra or
                // missing produced row would otherwise be invisible.
                assertEquals(expectedRows.size(), cleaned.length, name + ": row count");
                for (int r = 0; r < expectedRows.size(); r++) {
                    assertEquals(list(expectedRows.get(r)).size(), cleaned[r].length,
                            name + ": width of row " + r);
                }
                for (int r = 0; r < expectedRows.size(); r++) {
                    List<Object> values = list(expectedRows.get(r));
                    for (int c = 0; c < values.size(); c++) {
                        assertEquals(number(values.get(c)), cleaned[r][c],
                                name + ": row " + r + " component " + c);
                    }
                }
            }));
        }
        // A floor, with the reason the rows exist: an empty fixture key would otherwise generate
        // no tests and the suite would stay green. Each kind is a different non-finite value, and
        // each has to become zero rather than propagate.
        assertTrue(kinds.containsAll(List.of("nan", "posinf", "neginf", "all-nan")),
                "every non-finite kind must be covered, got " + kinds);
        return tests;
    }

    @TestFactory
    @DisplayName("what predict hands the model is the reduced question set")
    List<DynamicTest> predictAsksTheReducedSet() {
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : list(fixture().get("predicts"))) {
            Map<String, Object> row = map(entry);
            String name = (String) row.get("name");
            tests.add(DynamicTest.dynamicTest(name, () -> {
                int k = ((Number) row.get("k")).intValue();
                Map<String, Object> declared = map(row.get("questions"));
                Map<String, Question> questions = new LinkedHashMap<>();
                for (Map.Entry<String, Object> q : declared.entrySet()) {
                    questions.put(q.getKey(), fromSpec(map(q.getValue())));
                }
                Map<String, Question> original = new LinkedHashMap<>(questions);

                List<Map<String, Question>> asked = new ArrayList<>();
                Predictor recorder = (s, qs) -> {
                    asked.add(new LinkedHashMap<>(qs));
                    return new Prediction("stub", Map.of(),
                            new Usage(0, 0, 0, 0, false, List.of(), Map.of()));
                };
                Shortlist.Shortlisted result = Shortlist.predict(recorder,
                        "My card was charged twice for the same order last Tuesday.",
                        questions, ShortlistTest::stubEmbed, k);

                assertEquals(1, asked.size(), name + ": exactly one prediction, never one per"
                        + " question -- shortlisting must not add a second pass");
                Map<String, Question> seen = asked.get(0);
                Map<String, Object> expectedAsked = map(row.get("asked"));
                assertEquals(new ArrayList<>(expectedAsked.keySet()), new ArrayList<>(seen.keySet()),
                        name + ": the questions asked, and their order");
                for (Map.Entry<String, Object> expected : expectedAsked.entrySet()) {
                    Map<String, Object> spec = map(expected.getValue());
                    Question actual = seen.get(expected.getKey());
                    assertEquals(spec.get("type"), actual.type().wireName(),
                            name + "." + expected.getKey() + ": type");
                    Object criteria = spec.get("criteria");
                    if (criteria instanceof Map) {
                        assertEquals(new ArrayList<>(map(criteria).keySet()),
                                new ArrayList<>(map(actual.spec().get("criteria")).keySet()),
                                name + "." + expected.getKey() + ": the criteria kept, in order");
                    }
                }
                Map<String, Object> expectedMeta = map(row.get("shortlist"));
                assertEquals(expectedMeta.keySet(), result.shortlist().keySet(),
                        name + ": which questions were shortlisted");
                for (Map.Entry<String, Object> expected : expectedMeta.entrySet()) {
                    Map<String, Object> meta = map(expected.getValue());
                    Shortlist.Ranking ranking = result.shortlist().get(expected.getKey());
                    assertEquals(((Number) meta.get("n")).intValue(), ranking.total());
                    assertEquals(meta.get("passthrough"), ranking.passthrough());
                    List<String> labels = new ArrayList<>();
                    for (Object label : list(meta.get("labels"))) {
                        labels.add((String) label);
                    }
                    assertEquals(labels, ranking.labels(),
                            name + "." + expected.getKey() + ": shortlist labels");
                }
                assertEquals(original, questions, name + ": the caller's map was modified");
            }));
        }
        // Floors, with the reason: one reduced choice proves the subsetting, one passthrough
        // proves the embedder is skipped, and without them an empty key generates no tests at all.
        int reduced = 0;
        int passthrough = 0;
        for (Object entry : list(fixture().get("predicts"))) {
            for (Object ranking : map(map(entry).get("shortlist")).values()) {
                if (Boolean.TRUE.equals(map(ranking).get("passthrough"))) {
                    passthrough++;
                } else {
                    reduced++;
                }
            }
        }
        assertTrue(reduced >= 1, "no case actually reduces a choice");
        assertTrue(passthrough >= 1, "no case exercises the passthrough");
        return tests;
    }

    private static Question fromSpec(Map<String, Object> spec) {
        String type = (String) spec.get("type");
        String instructions = (String) spec.get("instructions");
        Object criteria = spec.get("criteria");
        if ("choice".equals(type)) {
            return Question.choice(instructions, map(criteria));
        }
        if ("score".equals(type)) {
            return Question.score(instructions, list(criteria));
        }
        return Question.noul(instructions);
    }

    @Test
    @DisplayName("the inputs the reference refuses are refused here too, or cannot be written")
    void refusals() {
        Map<String, Object> small = new LinkedHashMap<>();
        small.put("refund", "money back");
        small.put("technical", "a bug");
        small.put("billing", "an invoice");
        Question question = Question.choice("What?", small);

        // Refused at run time, as the reference refuses them.
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", question, ShortlistTest::stubEmbed, 0), "k = 0");
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", question, ShortlistTest::stubEmbed, -1), "k = -1");
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", question, null, 2), "a null embedder");
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", question, texts -> new double[1][1], 2),
                "too few rows");
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", question, texts -> new double[texts.size()][0], 2),
                "zero-width rows");
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", question, texts -> null, 2), "a null result");
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", Question.noul("Is this urgent?"),
                        ShortlistTest::stubEmbed, 2),
                "a noul has no labels to shortlist");
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.cached(ShortlistTest::stubEmbed, 0), "a cache bound of zero");

        // Refused by ragged rows, which the reference cannot even see: numpy would have made the
        // array object-dtype and failed the shape check instead.
        assertThrows(IllegalArgumentException.class,
                () -> Shortlist.rank("s", question, texts -> {
                    double[][] rows = new double[texts.size()][];
                    for (int i = 0; i < rows.length; i++) {
                        rows[i] = new double[i + 1];
                    }
                    return rows;
                }, 2), "rows of different widths");

        // And the ones the type system refuses before a test can run: a bool k, criteria that are
        // neither a map nor a list, a duplicated label (a Map cannot hold one), a one-dimensional
        // embedding, and questions that are not a map. Each is a compile error here, which is
        // strictly earlier than the reference's run-time check.
    }

    @Test
    @DisplayName("a zero-norm query keeps the first k labels in criteria order")
    void zeroNormQueryKeepsCriteriaOrder() {
        // Every score ties at zero, so the only thing deciding which labels survive is the
        // stability of the sort. An unstable sort passes every other test in this class and fails
        // this one.
        Map<String, Object> criteria = new LinkedHashMap<>();
        for (String label : List.of("a", "b", "c", "d", "e", "f")) {
            criteria.put(label, label + " description");
        }
        Question question = Question.choice("What?", criteria);
        Shortlist.Ranking ranking = Shortlist.rank("state", question,
                texts -> new double[texts.size()][EMBED_DIM], 3);
        assertEquals(List.of("a", "b", "c"), ranking.labels());
        assertEquals(3, ranking.scores().length, "three kept labels, three scores -- asserted"
                + " before the loop, or an empty array would make it vacuous");
        for (double score : ranking.scores()) {
            assertEquals(0.0, score);
        }
    }

    @Test
    @DisplayName("a query whose norm underflows scores exactly zero, not almost zero")
    void underflowingQueryScoresExactlyZero() {
        // Asserted exactly, and separately from the recorded scores, because the rounded
        // comparison cannot see it: without the zero-norm guard the score is 1.4e-200, which
        // agrees with 0 to nine decimals and to any tolerance worth having. A norm of zero does
        // NOT imply a dot product of zero once components underflow, which is the trap.
        double[] query = {1e-200, 1e-200};
        double[][] matrix = {query, {1.0, 1.0}};
        assertEquals(0.0, Shortlist.cosine(query, matrix, 1)[0],
                "an underflowing query must score zero outright");
    }

    @Test
    @DisplayName("a cosine computing above one is clamped to exactly one")
    void cosineIsClampedToOne() {
        // The self-cosine of this vector computes to 1.0000000000000002 with sequential
        // arithmetic, so the clamp is observable -- and again only exactly: the unclamped value
        // agrees with 1.0 to nine decimals. Asserted on an exact equality for that reason.
        double[] vector = {1.0 / 3.0, 1.0 / 3.0, 8.0 / 3.0};
        double[][] matrix = {vector, vector};
        double[] sims = Shortlist.cosine(vector, matrix, 1);
        assertEquals(1.0, sims[0], "a cosine above one must be clamped, not reported");
        // and the other side of the clamp
        double[] opposed = {-vector[0], -vector[1], -vector[2]};
        assertEquals(-1.0, Shortlist.cosine(vector, new double[][] {vector, opposed}, 1)[0],
                "and a cosine below minus one must be clamped too");
    }

    @Test
    @DisplayName("a label scoring zero outranks one scoring negative")
    void zeroOutranksNegative() {
        // The signed-cosine rule. A similarity floor that clamped negatives to zero would tie
        // these and then keep the earlier label, which is the opposite answer.
        Map<String, Object> criteria = new LinkedHashMap<>();
        criteria.put("opposite", "x");
        criteria.put("orthogonal", "y");
        Question question = Question.choice("What?", criteria);
        Shortlist.Ranking ranking = Shortlist.rank("state", question, texts -> new double[][] {
            {1.0, 0.0},            // the query
            {-1.0, 0.0},           // opposite: cosine -1
            {0.0, 0.0},            // orthogonal: zero norm, so zero
        }, 1);
        assertEquals(List.of("orthogonal"), ranking.labels());
        assertEquals(0.0, ranking.scores()[0]);
    }

    @Test
    @DisplayName("the cache embeds each text once and leaves the first call unchanged")
    void cacheEmbedsEachTextOnce() {
        AtomicInteger calls = new AtomicInteger();
        List<Integer> batchSizes = Collections.synchronizedList(new ArrayList<>());
        Shortlist.Embedder counting = texts -> {
            calls.incrementAndGet();
            batchSizes.add(texts.size());
            return stubEmbed(texts);
        };
        Shortlist.CachedEmbedder cached = Shortlist.cached(counting, 64);

        double[][] first = cached.embed(List.of("q1", "a", "b", "c"));
        assertEquals(1, calls.get(), "a cold cache costs one call, as the unwrapped embedder does");
        assertEquals(4, batchSizes.get(0));

        double[][] second = cached.embed(List.of("q2", "a", "b", "c"));
        assertEquals(2, calls.get());
        assertEquals(1, batchSizes.get(1),
                "a repeat must embed the new query alone, not the options again");
        for (int row = 1; row < 4; row++) {
            // Equal contents, but NOT the same array. The reference stacks its rows into a fresh
            // array on every return, so its cache cannot be reached through what it hands back;
            // an earlier version of this test asserted identity and was asserting the defect.
            assertArrayEquals(first[row], second[row], "the cached vector must come back");
            assertNotSame(first[row], second[row],
                    "the cache must hand out a copy, not its own row");
        }

        Map<String, Long> info = cached.cacheInfo();
        assertEquals(5L, info.get("size"), "q1, a, b, c and q2 -- the second query is cached too");
        assertEquals(64L, info.get("maxsize"));
        assertEquals(3L, info.get("hits"));
        assertEquals(5L, info.get("misses"));

        cached.cacheClear();
        assertEquals(0L, cached.cacheInfo().get("size"));
        assertEquals(0L, cached.cacheInfo().get("hits"));
    }

    @Test
    @DisplayName("the cache deduplicates within one call")
    void cacheDeduplicatesWithinOneCall() {
        List<Integer> batchSizes = new ArrayList<>();
        Shortlist.CachedEmbedder cached = Shortlist.cached(texts -> {
            batchSizes.add(texts.size());
            return stubEmbed(texts);
        }, 64);
        double[][] rows = cached.embed(List.of("a", "a", "a", "b"));
        assertEquals(List.of(2), batchSizes, "four texts, two distinct, one embedding call of two");
        assertArrayEquals(rows[0], rows[1], "the repeats carry the same vector");
        assertArrayEquals(rows[0], rows[2]);
        assertNotSame(rows[0], rows[1], "but not the same array: each row is its own storage,"
                + " as numpy's stack gives");
        assertFalse(java.util.Arrays.equals(rows[0], rows[3]), "and b is a different vector");
    }

    @Test
    @DisplayName("a NaN score ranks LAST, as numpy's argsort puts it, not first")
    void nanRanksLast() {
        // Java's Double.compare ranks NaN as the LARGEST double, so a descending comparator puts
        // it FIRST -- the opposite end from numpy. That is not a cosmetic difference: the kept
        // labels change, so the reduced criteria handed to the model are a different question.
        // A score is NaN whenever a vector's norm overflows to infinity.
        Map<String, Object> recorded = map(fixture().get("nan_ranking"));
        Map<String, Object> criteria = new LinkedHashMap<>(map(recorded.get("criteria")));
        int k = ((Number) recorded.get("k")).intValue();

        Shortlist.Embedder nan = texts -> {
            double[][] rows = new double[texts.size()][];
            for (int i = 0; i < texts.size(); i++) {
                String text = texts.get(i);
                if (text.startsWith("b:")) {
                    rows[i] = new double[] {1.0, 0.0};
                } else if (text.startsWith("c:")) {
                    rows[i] = new double[] {0.0, 1.0};
                } else {
                    rows[i] = new double[] {1e200, 1e200};
                }
            }
            return rows;
        };
        Shortlist.Ranking ranking = Shortlist.rank("state", criteria, nan, k);
        List<String> expected = new ArrayList<>();
        for (Object label : list(recorded.get("labels"))) {
            expected.add((String) label);
        }
        assertEquals(expected, ranking.labels(),
                "the NaN-scored label must be dropped first, not kept first");
        List<Object> recordedScores = list(recorded.get("scores"));
        for (int i = 0; i < recordedScores.size(); i++) {
            assertScore(number(recordedScores.get(i)), ranking.scores()[i], "score " + i);
        }
        assertEquals(k, ranking.scores().length, "one score per kept label -- asserted before"
                + " the loop, or an empty array would make it vacuous");
        for (double score : ranking.scores()) {
            assertFalse(Double.isNaN(score), "no NaN may survive the cut here");
        }
    }

    @Test
    @DisplayName("the cache refuses a changed embedding width and stays usable")
    void cacheRefusesAWidthChange() {
        // Rows of one width cannot be stacked against rows of another, and rank reads the result
        // as one matrix. The reference REFUSES the call before writing anything, and the ordering
        // is the point: caching the new width first leaves a cache that fails on every later call
        // touching both widths until someone clears it -- a refused call turned into a
        // permanently broken cache.
        int[] width = {3};
        Shortlist.CachedEmbedder cached = Shortlist.cached(texts -> {
            double[][] rows = new double[texts.size()][width[0]];
            for (int r = 0; r < rows.length; r++) {
                rows[r][0] = 1.0;
            }
            return rows;
        }, 64);
        assertEquals(3, cached.embed(List.of("a"))[0].length);

        width[0] = 5;
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> cached.embed(List.of("b")));
        assertTrue(failure.getMessage().contains("dim 5"), failure.getMessage());
        assertTrue(failure.getMessage().contains("cacheClear"),
                "the message must say how to recover: " + failure.getMessage());

        assertEquals(1L, cached.cacheInfo().get("size"),
                "the refused row must not have been cached");
        assertEquals(3, cached.embed(List.of("a"))[0].length,
                "and the cache must still serve what it already held");
        cached.cacheClear();
        assertEquals(5, cached.embed(List.of("b"))[0].length, "after a clear, the new width works");
    }

    @TestFactory
    @DisplayName("the cache counters match the reference, call by call")
    List<DynamicTest> cacheCountersMatchTheReference() {
        // hits and misses are both per OCCURRENCE in the reference, so hits + misses equals the
        // number of texts looked up. Counting misses per DISTINCT text -- the obvious reading of
        // "embed each text once" -- breaks that identity on the first repeated text, and anyone
        // sizing a cache from these numbers would be reading two different bases against
        // each other.
        List<DynamicTest> tests = new ArrayList<>();
        Shortlist.CachedEmbedder cached = Shortlist.cached(ShortlistTest::stubEmbed, 64);
        int looked = 0;
        for (Object entry : list(fixture().get("cache_counters"))) {
            Map<String, Object> call = map(entry);
            List<String> texts = new ArrayList<>();
            for (Object text : list(call.get("texts"))) {
                texts.add((String) text);
            }
            looked += texts.size();
            int expectedLooked = looked;
            tests.add(DynamicTest.dynamicTest(texts.toString(), () -> {
                cached.embed(texts);
                Map<String, Long> info = cached.cacheInfo();
                assertEquals(((Number) call.get("size")).longValue(), info.get("size"), "size");
                assertEquals(((Number) call.get("maxsize")).longValue(), info.get("maxsize"));
                assertEquals(((Number) call.get("hits")).longValue(), info.get("hits"), "hits");
                assertEquals(((Number) call.get("misses")).longValue(), info.get("misses"),
                        "misses must count occurrences, not distinct texts");
                assertEquals(expectedLooked, info.get("hits") + info.get("misses"),
                        "hits + misses must equal the texts looked up so far");
            }));
        }
        // A repeated text within one call is the whole point of per-occurrence counting, so it is
        // the floor. An empty key would otherwise generate nothing.
        boolean hasRepeat = false;
        for (Object entry : list(fixture().get("cache_counters"))) {
            List<Object> texts = list(map(entry).get("texts"));
            if (new java.util.HashSet<>(texts).size() < texts.size()) {
                hasRepeat = true;
            }
        }
        assertTrue(hasRepeat, "no recorded call repeats a text, so per-occurrence counting is"
                + " indistinguishable from per-distinct counting");
        assertTrue(tests.size() >= 3, "only " + tests.size() + " recorded calls");
        return tests;
    }

    @Test
    @DisplayName("editing what the cache returned cannot corrupt the cache")
    void returnedRowsAreIsolated() {
        Shortlist.CachedEmbedder cached = Shortlist.cached(ShortlistTest::stubEmbed, 64);
        double[][] first = cached.embed(List.of("a"));
        double original = first[0][0];
        first[0][0] = 99.0;
        assertEquals(original, cached.embed(List.of("a"))[0][0],
                "a caller's write must not reach the cached vector");
    }

    @Test
    @DisplayName("the cache evicts least-recently-used past its bound")
    void cacheEvictsLru() {
        AtomicInteger calls = new AtomicInteger();
        Shortlist.CachedEmbedder cached = Shortlist.cached(texts -> {
            calls.incrementAndGet();
            return stubEmbed(texts);
        }, 2);
        cached.embed(List.of("a"));
        cached.embed(List.of("b"));
        cached.embed(List.of("a"));          // a is now the most recent
        cached.embed(List.of("c"));          // evicts b
        assertEquals(2L, cached.cacheInfo().get("size"));
        int before = calls.get();
        cached.embed(List.of("a"));
        assertEquals(before, calls.get(), "a was kept, so this is a hit");
        cached.embed(List.of("b"));
        assertEquals(before + 1, calls.get(), "b was evicted, so this is a miss");
    }

    @Test
    @DisplayName("a failing embedder caches nothing, so a transient failure cannot poison it")
    void aFailingEmbedderCachesNothing() {
        AtomicInteger attempts = new AtomicInteger();
        Shortlist.CachedEmbedder cached = Shortlist.cached(texts -> {
            if (attempts.incrementAndGet() == 1) {
                throw new IllegalStateException("transient");
            }
            return stubEmbed(texts);
        }, 64);
        assertThrows(IllegalStateException.class, () -> cached.embed(List.of("a")));
        assertEquals(0L, cached.cacheInfo().get("size"), "nothing may be cached from a failure");
        double[][] rows = cached.embed(List.of("a"));
        assertEquals(EMBED_DIM, rows[0].length, "and the retry succeeds");
        assertEquals(1L, cached.cacheInfo().get("size"));
    }

    @Test
    @DisplayName("the cache is safe to share between threads")
    void cacheIsThreadSafe() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        Shortlist.CachedEmbedder cached = Shortlist.cached(texts -> {
            calls.incrementAndGet();
            return stubEmbed(texts);
        }, 1024);
        int threads = 8;
        CountDownLatch go = new CountDownLatch(1);
        List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
        List<Thread> workers = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            Thread worker = new Thread(() -> {
                try {
                    assertTrue(go.await(10, TimeUnit.SECONDS));
                    for (int i = 0; i < 50; i++) {
                        double[][] rows = cached.embed(List.of("shared-" + (i % 10)));
                        assertEquals(EMBED_DIM, rows[0].length);
                    }
                } catch (Throwable failure) {
                    failures.add(failure);
                }
            }, "cache-" + t);
            worker.start();
            workers.add(worker);
        }
        go.countDown();
        for (Thread worker : workers) {
            worker.join(TimeUnit.SECONDS.toMillis(20));
            assertFalse(worker.isAlive(), worker.getName() + " never finished");
        }
        assertEquals(List.of(), failures);
        assertEquals(10L, cached.cacheInfo().get("size"), "ten distinct texts, however they raced");
    }

    @Test
    @DisplayName("shortlisting works against a Router as well as an Agent")
    void worksAgainstEitherPredictor() {
        // The reason Predictor exists. The reference accepts anything with predict or system_one;
        // here one interface covers both, so a caller routing between checkpoints shortlists the
        // same way as one asking a fixed checkpoint.
        assertTrue(Predictor.class.isAssignableFrom(Agent.class),
                "Agent must be usable as a Predictor");
        assertTrue(Predictor.class.isAssignableFrom(Router.class),
                "Router must be usable as a Predictor");
    }

    @Test
    @DisplayName("the default k is the reference's")
    void defaultK() {
        assertEquals(((Number) fixture().get("default_k")).intValue(), Shortlist.DEFAULT_K);
    }

    @Test
    @DisplayName("the recorded scores are rounded far below the tolerance they are checked at")
    void scorePrecisionIsHonest() {
        // Guards the one tolerance in this suite from quietly widening. The fixture rounds to
        // nine decimals and the comparison allows 5e-10 -- half of the last recorded digit --
        // so the tolerance admits nothing the rounding did not already discard. If either number
        // moves without the other, this fails rather than letting a looser check pass unnoticed.
        int decimals = ((Number) fixture().get("score_decimals")).intValue();
        assertEquals(9, decimals, "the fixture's recorded precision changed");
        double halfOfLastDigit = 0.5 * Math.pow(10, -decimals);
        assertEquals(5e-10, halfOfLastDigit, 1e-15,
                "the tolerance in assertScore must stay half of the last recorded digit");
        // and the recorded values really are rounded, not full precision stored at a width that
        // happens to look rounded
        int checked = 0;
        for (Object entry : list(fixture().get("cases"))) {
            Object scores = map(entry).get("scores");
            if (scores == null) {
                continue;
            }
            for (Object score : list(scores)) {
                if (score instanceof String) {
                    continue;
                }
                // Via the shortest decimal that round-trips, not by scaling: multiplying a
                // 9-decimal value by 1e9 is not exact in binary, and the first version of this
                // check failed on its own arithmetic rather than on the data.
                double value = ((Number) score).doubleValue();
                int scale = new java.math.BigDecimal(Double.toString(value)).stripTrailingZeros()
                        .scale();
                assertTrue(scale <= decimals,
                        "a recorded score carries " + scale + " decimals, more than " + decimals
                        + ": " + value);
                checked++;
            }
        }
        assertTrue(checked >= 40, "only " + checked + " recorded scores checked");
    }

    /** Quotes a string for a failure message without pulling in the JSON writer's rules. */
    private static final class PythonJsonQuote {
        private PythonJsonQuote() {
        }

        static String of(String value) {
            return "\"" + value.replace("\"", "\\\"") + "\"";
        }
    }
}
