package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.PythonJson;
import com.convaiinnovations.laya.sequence.SequenceBuilder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Opt-in embedding shortlist for a choice question with many labels. A port of
 * {@code laya.shortlist}.
 *
 * <p>Every option of a choice shares one {@code head_max_len} budget, so a large label set leaves
 * only a few tokens per label and the model is shown truncated options. Shortlisting embeds the
 * state and each option with a caller-supplied embedder, keeps the top {@code k}, and then runs a
 * single {@link Predictor#predict} over that reduced set. It adds no second decision pass: the
 * model still scores every criterion it is handed, there are just fewer of them.
 *
 * <p>The coarse-to-fine pattern is the one the project's README recommends. Ranking is cosine
 * similarity on whatever vectors the embedder returns; the quality of the shortlist is the
 * embedder's business, and nothing here measures it.
 *
 * <p><b>Four rules decide which labels survive</b>, and each is a place a reimplementation goes
 * wrong quietly rather than loudly:
 *
 * <ul>
 *   <li>The sort is <b>stable</b> and descending, so a tie keeps the earlier label. An unstable
 *       sort reorders ties and silently changes which labels survive the cut.
 *   <li>The score is a <b>signed</b> cosine, not a similarity floor. A label scoring zero -- no
 *       signal, or a non-finite vector treated as none -- outranks one scoring negative, and
 *       {@code k} drops the negatives first.
 *   <li>A zero-norm query scores everything zero, so the kept set is the first {@code k} in
 *       criteria order rather than an arbitrary one.
 *   <li>{@code k} at or above the label count is a <b>passthrough</b>: the labels come back in
 *       criteria order, there are no scores, and <b>the embedder is never called</b>. A caller
 *       paying per embedding call cares about that last part.
 * </ul>
 *
 * <p>{@code embed_fn_from_agent} is deliberately not ported. It mean-pools the checkpoint's own
 * encoder, which needs the encoder's hidden states; the ONNX graph this runtime loads exposes
 * logits, not hidden states. A dedicated bi-encoder passed as an {@link Embedder} shortlists
 * better anyway, which is what the reference says of its own helper.
 */
public final class Shortlist {

    /** The reference's default: keep twenty labels. */
    public static final int DEFAULT_K = 20;

    private Shortlist() {
    }

    /**
     * Maps texts to vectors, one row per text, every row the same width.
     *
     * <p>Called once per shortlisted question, with the query text first and then one string per
     * option in criteria order.
     */
    @FunctionalInterface
    public interface Embedder {

        /** Embed these texts. The result must be {@code texts.size()} rows of equal width. */
        double[][] embed(List<String> texts);
    }

    /**
     * What a ranking decided.
     *
     * @param labels      the kept labels in rank order, or every label in criteria order on a
     *                    passthrough
     * @param scores      the signed cosine of each kept label in that order, or null when nothing
     *                    was dropped -- negative values included, never clamped to zero
     * @param passthrough whether {@code k} covered every label, in which case nothing was embedded
     * @param total       how many labels the question had
     */
    public record Ranking(List<String> labels, double[] scores, boolean passthrough, int total) {

        public Ranking {
            labels = List.copyOf(labels);
            scores = scores == null ? null : scores.clone();
            // Validated, because this is public API and the components can contradict each other:
            // two labels with one score, or a total below the number kept, give any consumer
            // zipping the two an out-of-bounds or a silently misattributed score.
            if (total < labels.size()) {
                throw new IllegalArgumentException(String.format(
                        "total is %d but %d labels were kept", total, labels.size()));
            }
            if (passthrough && scores != null) {
                throw new IllegalArgumentException("a passthrough ranking has no scores");
            }
            if (!passthrough && (scores == null || scores.length != labels.size())) {
                throw new IllegalArgumentException(String.format(
                        "%d labels need %d scores, got %s", labels.size(), labels.size(),
                        scores == null ? "none" : Integer.toString(scores.length)));
            }
        }

        /** The scores, copied, or null when nothing was dropped. */
        @Override
        public double[] scores() {
            return scores == null ? null : scores.clone();
        }
    }

    /** A prediction over shortlisted questions, and what each shortlist decided. */
    public record Shortlisted(Prediction prediction, Map<String, Ranking> shortlist) {

        public Shortlisted {
            if (prediction == null) {
                throw new IllegalArgumentException("prediction must not be null");
            }
            if (shortlist == null) {
                throw new IllegalArgumentException("shortlist must not be null");
            }
            shortlist = Collections.unmodifiableMap(new LinkedHashMap<>(shortlist));
        }
    }

    /** The top {@code k} labels of a choice question for this state. */
    public static List<String> choice(Object state, Question question, Embedder embedder, int k) {
        return rank(state, question, embedder, k).labels();
    }

    /** The top {@code k} of these labels for this state, with no instruction in the query. */
    public static List<String> choice(Object state, Map<String, ?> criteria, Embedder embedder,
            int k) {
        return rank(state, criteria, embedder, k, null).labels();
    }

    /**
     * Rank a choice question's labels for this state, keeping the best {@code k}.
     *
     * <p>The question's instruction goes into the query text, which is what the reference does
     * when one is supplied.
     *
     * @throws IllegalArgumentException if {@code k} is below one, or the question is not a choice
     */
    public static Ranking rank(Object state, Question question, Embedder embedder, int k) {
        requireChoice(question);
        return rank(state, question.instructions(), criteriaOf(question), question.renderOptions(),
                embedder, k);
    }

    /** Rank these labels for this state, with no instruction in the query. */
    public static Ranking rank(Object state, Map<String, ?> criteria, Embedder embedder, int k) {
        return rank(state, criteria, embedder, k, null);
    }

    /**
     * Rank these labels for this state, optionally prefixing the query with an instruction.
     *
     * <p>The entry point that takes criteria rather than a {@link Question}, because the reference
     * takes the two separately and its instruction is OPTIONAL -- its default is to embed the
     * state alone. A typed {@code Question} always carries a non-blank instruction, so the
     * {@code Question} overload cannot express that case, and shortlisting a label set a caller
     * has not yet turned into a question is a reasonable thing to want.
     *
     * @param instructions prefixed to the query text when present; null or empty embeds the state
     *                     alone, which is the reference's default
     */
    public static Ranking rank(Object state, Map<String, ?> criteria, Embedder embedder, int k,
            String instructions) {
        Map<String, Object> options = new LinkedHashMap<>();
        if (criteria == null) {
            throw new IllegalArgumentException("criteria must not be null");
        }
        options.putAll(criteria);
        if (options.isEmpty()) {
            throw new IllegalArgumentException("choice criteria must contain at least one option");
        }
        // Rendered through a Question rather than by a second copy of the same rule: the option
        // text is what gets embedded, so a divergence here would silently rank against different
        // strings than the model is shown. The instruction below is not used for rendering.
        List<String> optionTexts = Question.choice("placeholder", options).renderOptions();
        return rank(state, instructions, options, optionTexts, embedder, k);
    }

    /**
     * Shortlist every choice question, then ask all of them in one prediction.
     *
     * <p>A non-choice question is forwarded unchanged, and so is a choice whose label count is at
     * or below {@code k} -- which also means its embedder is never called. The caller's map is not
     * modified.
     *
     * <p>Probabilities on a shortlisted choice are over the kept labels only. That is the point
     * and worth stating: the model was never shown the dropped ones, so its distribution cannot
     * include them.
     */
    public static Shortlisted predict(Predictor predictor, Object state,
            Map<String, Question> questions, Embedder embedder, int k) {
        if (predictor == null) {
            throw new IllegalArgumentException("predictor must not be null");
        }
        if (questions == null) {
            throw new IllegalArgumentException("questions must not be null");
        }
        int checked = requirePositive(k, "k");
        Map<String, Question> reduced = new LinkedHashMap<>();
        Map<String, Ranking> meta = new LinkedHashMap<>();
        for (Map.Entry<String, Question> entry : questions.entrySet()) {
            String id = entry.getKey();
            Question question = entry.getValue();
            if (question == null || question.type() != Question.Type.CHOICE) {
                reduced.put(id, question);
                continue;
            }
            Map<String, Object> criteria = criteriaOf(question);
            Ranking ranking = rank(state, question.instructions(), criteria,
                    question.renderOptions(), embedder, checked);
            meta.put(id, ranking);
            if (ranking.passthrough()) {
                reduced.put(id, question);
                continue;
            }
            Map<String, Object> subset = new LinkedHashMap<>();
            for (String label : ranking.labels()) {
                subset.put(label, criteria.get(label));
            }
            reduced.put(id, Question.choice(question.instructions(), subset));
        }
        return new Shortlisted(predictor.predict(state, reduced), meta);
    }

    /** Shortlist with the reference's default {@code k}. */
    public static Shortlisted predict(Predictor predictor, Object state,
            Map<String, Question> questions, Embedder embedder) {
        return predict(predictor, state, questions, embedder, DEFAULT_K);
    }

    // ------------------------------------------------------------------ the ranking

    /** The one implementation. {@code instructions} may be null, which embeds the state alone. */
    private static Ranking rank(Object state, String instructions, Map<String, Object> criteria,
            List<String> optionTexts, Embedder embedder, int k) {
        int checked = requirePositive(k, "k");
        List<String> labels = new ArrayList<>(criteria.keySet());
        int total = labels.size();
        if (total == 0) {
            throw new IllegalArgumentException("choice criteria must contain at least one option");
        }
        if (checked >= total) {
            // Passthrough. The embedder is NOT called, which a caller paying per call relies on.
            return new Ranking(labels, null, true, total);
        }
        List<String> texts = new ArrayList<>(total + 1);
        texts.add(queryText(state, instructions));
        texts.addAll(optionTexts);
        double[][] matrix = embeddings(embedder, texts);
        double[] sims = cosine(matrix[0], matrix, 1);

        // A stable descending sort, with NaN LAST.
        //
        // Both halves of that are load-bearing. Stability makes a tie keep the earlier label,
        // which decides which labels survive whenever two score the same -- and with a zero-norm
        // query every label scores the same. And NaN has to be pushed to the end by hand, because
        // `Double.compare` ranks it as the LARGEST double, which in a descending comparator puts
        // it FIRST: the opposite end from `np.argsort`, which sorts NaN last. A score is NaN when
        // a vector's norm overflows to infinity, and getting its position wrong does not merely
        // misreport a number -- it changes WHICH LABELS survive the cut, so the model is asked a
        // different question.
        Integer[] order = new Integer[total];
        for (int i = 0; i < total; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (left, right) -> {
            double a = sims[left];
            double b = sims[right];
            boolean leftIsNaN = Double.isNaN(a);
            boolean rightIsNaN = Double.isNaN(b);
            if (leftIsNaN || rightIsNaN) {
                // Equal when both are NaN, so the stable sort keeps their original order.
                return leftIsNaN && rightIsNaN ? 0 : (leftIsNaN ? 1 : -1);
            }
            return Double.compare(b, a);
        });

        List<String> kept = new ArrayList<>(checked);
        double[] scores = new double[checked];
        for (int i = 0; i < checked; i++) {
            int index = order[i];
            kept.add(labels.get(index));
            scores[i] = sims[index];
        }
        return new Ranking(kept, scores, false, total);
    }

    /** The text the query vector is taken from: the instruction, then the serialised state. */
    static String queryText(Object state, String instructions) {
        String body = SequenceBuilder.serializeState(state);
        if (instructions == null || instructions.isEmpty()) {
            return body;
        }
        return instructions + "\n" + body;
    }

    /**
     * The embedder's output, validated and made finite.
     *
     * <p>A non-finite component becomes zero rather than propagating into every score or raising:
     * one bad row must cost that label its ranking, not the whole request.
     */
    static double[][] embeddings(Embedder embedder, List<String> texts) {
        if (embedder == null) {
            throw new IllegalArgumentException("embedder must not be null");
        }
        double[][] raw = embedder.embed(List.copyOf(texts));
        if (raw == null || raw.length != texts.size()) {
            throw new IllegalArgumentException(String.format(
                    "the embedder must return %d rows, got %s",
                    texts.size(), raw == null ? "null" : Integer.toString(raw.length)));
        }
        int width = -1;
        double[][] out = new double[raw.length][];
        for (int row = 0; row < raw.length; row++) {
            double[] values = raw[row];
            if (values == null || values.length < 1) {
                throw new IllegalArgumentException(
                        "the embedder returned an empty row at index " + row);
            }
            if (width < 0) {
                width = values.length;
            } else if (values.length != width) {
                throw new IllegalArgumentException(String.format(
                        "the embedder returned rows of different widths: %d then %d at index %d",
                        width, values.length, row));
            }
            out[row] = new double[width];
            for (int i = 0; i < width; i++) {
                double value = values[i];
                out[row][i] = Double.isFinite(value) ? value : 0.0;
            }
        }
        return out;
    }

    /**
     * Signed cosine of {@code query} against each row of {@code docs} from {@code from} onwards.
     *
     * <p>Zero-norm on either side scores zero -- not an error and not a skip, because every label
     * has to get a comparable number. Clamped to [-1, 1] because floating-point division can step
     * just outside it.
     */
    static double[] cosine(double[] query, double[][] docs, int from) {
        int count = docs.length - from;
        double[] sims = new double[Math.max(0, count)];
        double queryNorm = norm(query);
        if (queryNorm == 0.0 || count <= 0) {
            return sims;
        }
        for (int i = 0; i < count; i++) {
            double[] doc = docs[from + i];
            double denominator = norm(doc) * queryNorm;
            if (!(denominator > 0.0)) {
                continue;
            }
            double dot = 0.0;
            for (int j = 0; j < Math.min(query.length, doc.length); j++) {
                dot += doc[j] * query[j];
            }
            double value = dot / denominator;
            // NaN survives on purpose: it is what the reference produces when a norm overflows to
            // infinity, and clamping it to a number would invent a ranking the reference does not
            // have. Math.min/max would do exactly that, so the comparison is explicit.
            if (Double.isNaN(value)) {
                sims[i] = value;
            } else {
                sims[i] = value < -1.0 ? -1.0 : (value > 1.0 ? 1.0 : value);
            }
        }
        return sims;
    }

    private static double norm(double[] vector) {
        double total = 0.0;
        for (double value : vector) {
            total += value * value;
        }
        return Math.sqrt(total);
    }

    private static void requireChoice(Question question) {
        if (question == null) {
            throw new IllegalArgumentException("question must not be null");
        }
        if (question.type() != Question.Type.CHOICE) {
            throw new IllegalArgumentException(
                    "only a choice question has labels to shortlist, not " + question.type());
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> criteriaOf(Question question) {
        Object criteria = question.spec().get("criteria");
        if (!(criteria instanceof Map)) {
            throw new IllegalArgumentException("this choice question has no criteria to shortlist");
        }
        return (Map<String, Object>) criteria;
    }

    private static int requirePositive(int value, String what) {
        if (value < 1) {
            throw new IllegalArgumentException(what + " must be at least 1, got " + value);
        }
        return value;
    }

    // ------------------------------------------------------------------ caching

    /**
     * An {@link Embedder} that remembers each text it has embedded, under an LRU bound.
     *
     * <p>Shortlisting embeds the query plus every option on every call. When the option set is
     * fixed -- a label list, an intent taxonomy -- those rows never change between requests and
     * are re-embedded every time. Wrapping the embedder once leaves the first call unchanged and
     * reduces each repeat to embedding the new query alone.
     *
     * <p>A cold cache costs the same number of embedder calls as the unwrapped function: the
     * misses are deduplicated and embedded in one call. Nothing is cached when the embedder
     * throws or returns a bad shape, so a transient failure cannot poison the cache.
     *
     * <p>Safe to share between threads. The lock covers cache reads and writes only, never the
     * embedding call, so one slow embed does not block every other caller's cache hits.
     */
    public static final class CachedEmbedder implements Embedder {

        private final Embedder delegate;
        private final int maxSize;
        private final LinkedHashMap<String, double[]> rows;
        private final ReentrantLock lock = new ReentrantLock();
        private long hits;
        private long misses;

        private CachedEmbedder(Embedder delegate, int maxSize) {
            this.delegate = delegate;
            this.maxSize = maxSize;
            // accessOrder = true makes this an LRU: a get moves the entry to the end.
            this.rows = new LinkedHashMap<>(16, 0.75f, true);
        }

        @Override
        public double[][] embed(List<String> texts) {
            List<String> keys = new ArrayList<>(texts.size());
            for (String text : texts) {
                keys.add(text == null ? "" : text);
            }
            double[][] out = new double[keys.size()][];
            Set<String> wanted = new LinkedHashSet<>();
            int hitsThisCall = 0;
            lock.lock();
            try {
                for (int i = 0; i < keys.size(); i++) {
                    double[] row = rows.get(keys.get(i));
                    if (row != null) {
                        out[i] = row;
                        hits++;
                        hitsThisCall++;
                    } else {
                        wanted.add(keys.get(i));
                    }
                }
                // Per OCCURRENCE, not per distinct text, which is how the reference counts it.
                // `hits` is already per occurrence, so counting misses per distinct text would
                // put the two counters on different bases and `hits + misses` would stop equalling
                // the number of texts looked up -- which is exactly what anyone sizing a cache
                // from these numbers assumes.
                misses += keys.size() - hitsThisCall;
            } finally {
                lock.unlock();
            }
            if (wanted.isEmpty()) {
                return copyRows(out);
            }
            // Deduplicated and embedded in ONE call, outside the lock, so a cold cache costs what
            // the unwrapped embedder costs and a slow embed does not block other callers' hits.
            List<String> toEmbed = new ArrayList<>(wanted);
            double[][] fresh = embeddings(delegate, toEmbed);

            // Rows of one width cannot be stacked against rows of another, and `rank` reads the
            // result as one matrix. A changed embedder -- a swapped or reloaded bi-encoder -- is
            // the way this happens. The reference REFUSES the call here, before writing anything,
            // and that ordering is the point: caching the new width first would leave a cache
            // that fails on every later call touching both widths until someone clears it, which
            // turns a refused call into a permanently broken cache.
            int width = fresh.length == 0 ? -1 : fresh[0].length;
            lock.lock();
            try {
                if (width > 0 && !rows.isEmpty()) {
                    int held = rows.values().iterator().next().length;
                    if (width != held) {
                        throw new IllegalArgumentException(String.format(
                                "the embedder returned dim %d, but the cache holds dim %d;"
                                + " call cacheClear() if the model behind it changed",
                                width, held));
                    }
                }
                for (int i = 0; i < toEmbed.size(); i++) {
                    rows.put(toEmbed.get(i), fresh[i]);
                    while (rows.size() > maxSize) {
                        rows.remove(rows.keySet().iterator().next());
                    }
                }
            } finally {
                lock.unlock();
            }
            Map<String, double[]> byText = new LinkedHashMap<>();
            for (int i = 0; i < toEmbed.size(); i++) {
                byText.put(toEmbed.get(i), fresh[i]);
            }
            for (int i = 0; i < keys.size(); i++) {
                if (out[i] == null) {
                    out[i] = byText.get(keys.get(i));
                }
            }
            return copyRows(out);
        }

        /**
         * A fresh row per entry, so a caller cannot reach into the cache.
         *
         * <p>The reference stacks its rows into a new array on every return, so its cache is
         * immune to a caller editing what it handed back. Returning the stored arrays directly
         * would let one caller's write corrupt every later lookup of that text -- and alias the
         * same array into several rows when a text repeats within one call.
         */
        private static double[][] copyRows(double[][] rows) {
            double[][] out = new double[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                out[i] = rows[i] == null ? null : rows[i].clone();
            }
            return out;
        }

        /** How many texts are cached, the bound, and the hit and miss counts. */
        public Map<String, Long> cacheInfo() {
            lock.lock();
            try {
                Map<String, Long> info = new LinkedHashMap<>();
                info.put("size", (long) rows.size());
                info.put("maxsize", (long) maxSize);
                info.put("hits", hits);
                info.put("misses", misses);
                return Collections.unmodifiableMap(info);
            } finally {
                lock.unlock();
            }
        }

        /** Forget everything. Call this if the weights behind the embedder change. */
        public void cacheClear() {
            lock.lock();
            try {
                rows.clear();
                hits = 0;
                misses = 0;
            } finally {
                lock.unlock();
            }
        }
    }

    /** Wrap an embedder so each text is embedded once, under the reference's default bound. */
    public static CachedEmbedder cached(Embedder embedder) {
        return cached(embedder, 4096);
    }

    /** Wrap an embedder so each text is embedded once, keeping at most {@code maxSize} of them. */
    public static CachedEmbedder cached(Embedder embedder, int maxSize) {
        if (embedder == null) {
            throw new IllegalArgumentException("embedder must not be null");
        }
        return new CachedEmbedder(embedder, requirePositive(maxSize, "maxSize"));
    }

    /** The query text for a question, exposed so a caller can see what is being embedded. */
    public static String queryTextFor(Object state, Question question) {
        return queryText(state, question == null ? null : question.instructions());
    }

    /** What the reference writes for a non-string criterion, used when rendering an option. */
    static String renderCriterion(Object value) {
        return value instanceof String ? (String) value : PythonJson.dumps(value);
    }
}
