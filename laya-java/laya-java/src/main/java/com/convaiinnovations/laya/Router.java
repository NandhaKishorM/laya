package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.PythonJson;
import com.convaiinnovations.laya.lang.LanguageDetection;
import com.convaiinnovations.laya.lang.UnicodeTables;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Decides which laya checkpoint a request should go to. A port of {@code laya.router}.
 *
 * <p>Three checkpoints, and the choice between them is not a tuning question. The English
 * checkpoint does not degrade gently off English, it collapses: on 20-option MASSIVE intent it
 * scores 0.100 on Hindi and 0.103 on Korean against 0.050 for random guessing, and it reports high
 * confidence while doing so (ECE 0.855 on Hindi). The multilingual checkpoint gains 21 points on
 * non-English XNLI and loses about 6 on English suites. So script is the primary signal, and a
 * wrong route is a wrong answer delivered confidently.
 *
 * <p>{@code typed-decisions} is never chosen automatically unless you ask for it, with
 * {@link Builder#autoTaskDetection} or an explicit task: it is fine-tuned on four specific
 * workflows and should not be a silent default.
 *
 * <p><b>Precedence</b>, highest first: an explicit model, an explicit task, a detected
 * typed-decisions workflow (opt-in), an explicit language, a caller's language hint, the built-in
 * script and language detection, and finally the configured default.
 *
 * <p>This class decides; it does not load. {@link #route} runs no model and touches no disk, so it
 * is safe to call on every request and to test without a checkpoint.
 */
public final class Router implements AutoCloseable, Predictor {

    /** The hub repository that bundles all three checkpoints. */
    public static final String BUNDLE_REPO = "convaiinnovations/laya";

    /** One of the three checkpoints laya ships. */
    public enum Checkpoint {
        /** ModernBERT-large, 512 tokens, English only. */
        ENGLISH("english"),
        /** mmBERT-base, 1024 tokens, 100+ languages. */
        MULTILINGUAL("multilingual"),
        /** ModernBERT-large fine-tuned on the four typed-decisions workflows. */
        TYPED_DECISIONS("typed-decisions");

        private final String wire;

        Checkpoint(String wire) {
            this.wire = wire;
        }

        /** The name the reference uses, which is what appears in a reason string. */
        public String wireName() {
            return wire;
        }

        @Override
        public String toString() {
            return wire;
        }
    }

    /** Where a checkpoint lives: a repository, and a subfolder within it when it has one. */
    public record ModelSpec(String repo, String subfolder) {

        /** The readable id: {@code repo}, or {@code repo/subfolder} when there is one. */
        public String repoString() {
            return subfolder == null || subfolder.isEmpty() ? repo : repo + "/" + subfolder;
        }
    }

    /**
     * The routing outcome: which checkpoint, why, and what was detected.
     *
     * @param model      the checkpoint to use
     * @param repo       its readable repository id
     * @param reason     why, in the reference's words; this reaches API responses
     * @param detection  what the detector saw, or null when an explicit argument decided
     * @param workflow   the typed-decisions workflow the question ids match, or null
     */
    public record RouteDecision(Checkpoint model, String repo, String reason,
                                LanguageDetection.Analysis detection, String workflow) {
    }

    /**
     * A caller's hint about whether the English checkpoint can read a state.
     *
     * <p>The reference accepts either a language code or a callable taking the state; this one
     * interface covers both -- {@link #of} for a fixed code, a lambda for anything that has to
     * look at the state, such as a language-identification model.
     *
     * <p>Returning null abstains, and abstaining is a real answer: it falls through to detection
     * rather than pinning a checkpoint on no evidence. So does a code that names no language,
     * which is what {@code LANG=C} in a minimal container gives.
     */
    @FunctionalInterface
    public interface LanguageHint {

        /** The language code for this state, or null to abstain. */
        String codeFor(Object state);

        /** A hint that always reports the same code. */
        static LanguageHint of(String code) {
            return state -> code;
        }
    }

    /** Per-call routing arguments. Every field may be null, which means "not specified". */
    public record RouteOptions(String model, String task, String lang, LanguageHint langGuess) {

        /** No per-call arguments: route on the questions and the state alone. */
        public static RouteOptions none() {
            return new RouteOptions(null, null, null, null);
        }

        /** Pin the checkpoint by name or alias. Beats everything else. */
        public RouteOptions model(String value) {
            return new RouteOptions(value, task, lang, langGuess);
        }

        /** Pin by task name. Beats everything but an explicit model. */
        public RouteOptions task(String value) {
            return new RouteOptions(model, value, lang, langGuess);
        }

        /** The language of the state, if the caller knows it. */
        public RouteOptions lang(String value) {
            return new RouteOptions(model, task, value, langGuess);
        }

        /** A hint consulted after an explicit language and before detection. */
        public RouteOptions langGuess(LanguageHint value) {
            return new RouteOptions(model, task, lang, value);
        }
    }

    /**
     * Builds the agent for a checkpoint.
     *
     * <p>The reference downloads from the hub. This port takes local paths, so the mapping from a
     * checkpoint to an {@link Agent} belongs to the caller -- see
     * {@link Builder#checkpointsRoot(Path)} for the common layout.
     */
    @FunctionalInterface
    public interface AgentFactory {

        /** Build the agent for {@code checkpoint}. */
        Agent create(Checkpoint checkpoint) throws IOException;
    }

    /**
     * A borrowed agent, held open for as long as the lease is.
     *
     * <p>This exists because of a difference between the languages, not as ceremony. In the
     * reference an evicted agent stays alive as long as a caller still refers to it: dropping the
     * router's reference is all eviction does, and CPython's refcounting takes care of the rest.
     * A JVM has no equivalent for a native ONNX session -- the garbage collector will not free it,
     * so the router has to close it, and closing one that a caller is mid-prediction with would
     * crash that prediction.
     *
     * <p>So eviction marks a slot retired and closes it when the last lease is released.
     * {@link #predict} leases internally, which makes the ordinary path safe without a caller
     * thinking about it; this is for code that needs the agent directly.
     *
     * <pre>{@code
     * try (Router.Lease lease = router.lease("multilingual")) {
     *     Prediction answer = lease.agent().predict(state, questions);
     * }
     * }</pre>
     */
    public final class Lease implements AutoCloseable {

        private final Checkpoint checkpoint;
        private final Slot slot;
        /** Guarded by {@link Router#lock}: see {@link #close()} for why a plain flag is not enough. */
        private boolean released;

        private Lease(Checkpoint checkpoint, Slot slot) {
            this.checkpoint = checkpoint;
            this.slot = slot;
        }

        /** The agent, valid until this lease is closed. */
        public Agent agent() {
            lock.lock();
            try {
                if (released) {
                    throw new IllegalStateException("this lease on " + checkpoint.wireName()
                            + " has been released");
                }
                return slot.agent;
            } finally {
                lock.unlock();
            }
        }

        /** Which checkpoint this is. */
        public Checkpoint checkpoint() {
            return checkpoint;
        }

        /**
         * Release it. Idempotent, so a try-with-resources around an early return is safe.
         *
         * <p>The flag and the lease count move together under {@link Router#lock}. A plain
         * check-then-set is not enough: two threads closing the SAME lease both read
         * {@code released == false}, both decrement, and the count reaches zero while another
         * lease is still open -- so the agent is closed under a caller that is still using it,
         * which is the single failure this whole mechanism exists to prevent.
         */
        @Override
        public void close() {
            Agent toClose = null;
            lock.lock();
            try {
                if (released) {
                    return;
                }
                released = true;
                slot.leases--;
                if (slot.leases <= 0 && slot.retired && slot.ownsClosing) {
                    toClose = slot.agent;
                }
            } finally {
                lock.unlock();
            }
            if (toClose != null) {
                toClose.close();
            }
        }
    }

    /** One resident agent and what the router is allowed to do with it. */
    private static final class Slot {

        private final Agent agent;
        /**
         * False for an agent handed in through {@link Router#attach}: the caller kept ownership,
         * so the router must never close it. Eviction still drops it, which is what the reference
         * does too.
         */
        private final boolean ownsClosing;
        private int leases;
        private boolean retired;

        private Slot(Agent agent, boolean ownsClosing) {
            this.agent = agent;
            this.ownsClosing = ownsClosing;
        }
    }

    /**
     * One build in progress, which every other caller for that checkpoint waits on.
     *
     * <p>Worth being precise about what this earns, because {@link Router#buildLock} looks like it
     * covers the same ground. On the SUCCESS path it does: a caller that takes the build lock and
     * then finds the checkpoint resident returns it instead of building a second copy, so one
     * build per checkpoint holds with or without this latch.
     *
     * <p>The failure path is where they differ, and it is the expensive one. A build that fails
     * leaves nothing resident, so without this latch each waiter in turn takes the build lock,
     * finds no checkpoint, and re-enters the factory to discover the same failure for itself. For
     * a load that fails by timing out a download, that is six slow failures instead of one.
     * With it, the builder's error is handed to everyone waiting. Measured both ways: a mutant
     * that removes the latch leaves one build on success and six attempts on failure.
     */
    private static final class InFlight {

        private final CountDownLatch done = new CountDownLatch(1);
        private volatile RuntimeException error;
    }

    // Names people are likely to type. Insertion order is not significant here, but the sorted
    // order is: an unknown name's error message lists these, and that message is recorded.
    private static final Map<String, Checkpoint> ALIASES;

    static {
        Map<String, Checkpoint> aliases = new LinkedHashMap<>();
        aliases.put("en", Checkpoint.ENGLISH);
        aliases.put("laya", Checkpoint.ENGLISH);
        aliases.put("default", Checkpoint.ENGLISH);
        aliases.put("multi", Checkpoint.MULTILINGUAL);
        aliases.put("ml", Checkpoint.MULTILINGUAL);
        aliases.put("laya-multilingual", Checkpoint.MULTILINGUAL);
        aliases.put("typed", Checkpoint.TYPED_DECISIONS);
        aliases.put("typed_decisions", Checkpoint.TYPED_DECISIONS);
        aliases.put("laya-typed-decisions", Checkpoint.TYPED_DECISIONS);
        aliases.put("decisions", Checkpoint.TYPED_DECISIONS);
        ALIASES = Collections.unmodifiableMap(aliases);
    }

    /** The bundle: one repository, the checkpoint in a subfolder. Only that subfolder downloads. */
    private static final Map<Checkpoint, ModelSpec> BUNDLED;

    /** The same checkpoints in their own repositories, for anyone who prefers them. */
    private static final Map<Checkpoint, ModelSpec> STANDALONE;

    static {
        Map<Checkpoint, ModelSpec> bundled = new LinkedHashMap<>();
        bundled.put(Checkpoint.ENGLISH, new ModelSpec(BUNDLE_REPO, null));
        bundled.put(Checkpoint.MULTILINGUAL, new ModelSpec(BUNDLE_REPO, "multilingual"));
        bundled.put(Checkpoint.TYPED_DECISIONS, new ModelSpec(BUNDLE_REPO, "typed-decisions"));
        BUNDLED = Collections.unmodifiableMap(bundled);

        Map<Checkpoint, ModelSpec> standalone = new LinkedHashMap<>();
        standalone.put(Checkpoint.ENGLISH, new ModelSpec("convaiinnovations/laya", null));
        standalone.put(Checkpoint.MULTILINGUAL,
                new ModelSpec("convaiinnovations/laya-multilingual", null));
        standalone.put(Checkpoint.TYPED_DECISIONS,
                new ModelSpec("convaiinnovations/laya-typed-decisions", null));
        STANDALONE = Collections.unmodifiableMap(standalone);
    }

    /**
     * The question-id signatures of the four typed-decisions workflows.
     *
     * <p>Matched exactly, never as a subset, so an unrelated schema that happens to contain
     * {@code urgency} is not captured.
     */
    private static final Map<String, Set<String>> TYPED_DECISION_WORKFLOWS;

    static {
        Map<String, Set<String>> workflows = new LinkedHashMap<>();
        workflows.put("agent_trace_observability",
                Set.of("action", "needs_review", "outcome", "risk", "urgency"));
        workflows.put("customer_service",
                Set.of("action", "category", "churn_risk", "needs_human", "urgency"));
        workflows.put("invoice_processing", Set.of("discrepancy_severity", "disposition",
                "duplicate", "matches_order", "urgency"));
        workflows.put("security_incidents", Set.of("credential_compromise", "disposition",
                "severity", "true_positive", "urgency"));
        TYPED_DECISION_WORKFLOWS = Collections.unmodifiableMap(workflows);
    }

    /** Subtags that mean "the English checkpoint can read this". */
    private static final Set<String> ENGLISH_SUBTAGS = Set.of("en", "eng", "english");

    /**
     * Codes that are valid {@code $LANG} values but name no language, so they answer nothing.
     *
     * <p>{@code C}, {@code POSIX} and {@code C.UTF-8} are what minimal images ship --
     * {@code C.UTF-8} is the default in the official Python image -- and the ISO 639-2 special
     * codes say the same thing in the standard's own vocabulary: {@code und} undetermined,
     * {@code zxx} no linguistic content, {@code mul} multiple languages. They abstain rather than
     * forcing the multilingual checkpoint on English text.
     */
    private static final Set<String> LANGUAGE_AGNOSTIC_CODES =
            Set.of("c", "posix", "und", "zxx", "mul");

    /** How many checkpoints stay resident unless asked otherwise, as in the reference. */
    public static final int DEFAULT_MAX_LOADED = 2;

    private final Map<Checkpoint, ModelSpec> models;
    private final Checkpoint defaultCheckpoint;
    private final boolean autoTaskDetection;
    private final LanguageHint langGuess;
    private final AgentFactory agents;

    private final ReentrantLock lock = new ReentrantLock();
    /** Serialises builds, so two cold loads do not hold two checkpoints in flight at once. */
    private final ReentrantLock buildLock = new ReentrantLock();
    private final Map<Checkpoint, Slot> slots = new LinkedHashMap<>();
    /** Least recently used first, which is the end eviction takes from. */
    private final List<Checkpoint> order = new ArrayList<>();
    private final Map<Checkpoint, InFlight> loading = new LinkedHashMap<>();
    private int maxLoaded;
    private boolean closed;

    private Router(Builder builder) {
        Map<Checkpoint, ModelSpec> resolved =
                new LinkedHashMap<>(builder.standaloneRepos ? STANDALONE : BUNDLED);
        resolved.putAll(builder.overrides);
        this.models = Collections.unmodifiableMap(resolved);
        this.defaultCheckpoint = builder.defaultCheckpoint;
        this.autoTaskDetection = builder.autoTaskDetection;
        this.langGuess = builder.langGuess;
        this.agents = builder.agents;
        this.maxLoaded = builder.maxLoaded;
    }

    /** A router with the reference's defaults: the bundle, English as default, no auto-detection. */
    public static Router withDefaults() {
        return builder().build();
    }

    /** A router to configure. */
    public static Builder builder() {
        return new Builder();
    }

    /** Configures a {@link Router}. */
    public static final class Builder {

        private final Map<Checkpoint, ModelSpec> overrides = new LinkedHashMap<>();
        private Checkpoint defaultCheckpoint = Checkpoint.ENGLISH;
        private boolean autoTaskDetection;
        private boolean standaloneRepos;
        private LanguageHint langGuess;
        private AgentFactory agents;
        private int maxLoaded = DEFAULT_MAX_LOADED;

        private Builder() {
        }

        /**
         * Where to send a state nothing identifies.
         *
         * <p>English by default, which is the reference's choice. A deployment whose traffic is
         * mostly not English should set this to {@link Checkpoint#MULTILINGUAL}: an unidentified
         * Latin-script state is no evidence of English, and this is the only knob that says so.
         */
        public Builder defaultCheckpoint(Checkpoint value) {
            this.defaultCheckpoint = requireNonNull(value, "defaultCheckpoint");
            return this;
        }

        /**
         * Allow question ids alone to select {@code typed-decisions}. Off by default.
         *
         * <p>Off because that checkpoint is fine-tuned on four specific synthetic workflows, and
         * a schema whose ids happen to match one of them should not silently change model.
         */
        public Builder autoTaskDetection(boolean value) {
            this.autoTaskDetection = value;
            return this;
        }

        /** Use the standalone repositories instead of the bundle. */
        public Builder standaloneRepos(boolean value) {
            this.standaloneRepos = value;
            return this;
        }

        /** A hint applied to every request, consulted after a per-call one. */
        public Builder langGuess(LanguageHint value) {
            this.langGuess = value;
            return this;
        }

        /**
         * How many checkpoints may be resident at once. Two by default, as in the reference.
         *
         * <p>Each is hundreds of megabytes, so this is a memory ceiling and not a cache size
         * hint. The least recently used is evicted past it.
         */
        public Builder maxLoaded(int value) {
            if (value < 1) {
                throw new IllegalArgumentException("maxLoaded must be at least 1, got " + value);
            }
            this.maxLoaded = value;
            return this;
        }

        /**
         * How to build an agent for a checkpoint.
         *
         * <p>The reference downloads from the hub; this port takes local paths, so the mapping
         * from a checkpoint to an {@link Agent} is the caller's. A seam rather than a hard-coded
         * directory layout, for the same reason {@code InferenceSession} is one: it makes the
         * eviction order, the lease counting and the attach-versus-build distinction testable in
         * milliseconds, where a real checkpoint is hundreds of megabytes and would keep all of it
         * out of any CI that does not download one.
         */
        public Builder agents(AgentFactory factory) {
            this.agents = requireNonNull(factory, "factory");
            return this;
        }

        /**
         * Load each checkpoint from {@code <root>/<name>}, graph included.
         *
         * <p>The layout {@code Agent.open} expects, with the subdirectory named as the reference
         * names the checkpoint: {@code english}, {@code multilingual}, {@code typed-decisions}.
         */
        public Builder checkpointsRoot(Path root) {
            return checkpointsRoot(root, root);
        }

        /** As {@link #checkpointsRoot(Path)}, with the ONNX graphs under a separate root. */
        public Builder checkpointsRoot(Path models, Path graphs) {
            requireNonNull(models, "models");
            requireNonNull(graphs, "graphs");
            return agents(checkpoint -> Agent.open(models.resolve(checkpoint.wireName()),
                    graphs.resolve(checkpoint.wireName())));
        }

        /** Point one checkpoint somewhere else -- a mirror, or a local export. */
        public Builder model(Checkpoint checkpoint, String repo, String subfolder) {
            requireNonNull(checkpoint, "checkpoint");
            this.overrides.put(checkpoint, new ModelSpec(requireNonNull(repo, "repo"), subfolder));
            return this;
        }

        /** Build it. */
        public Router build() {
            return new Router(this);
        }
    }

    /** Where this router expects each checkpoint to live. */
    public Map<Checkpoint, ModelSpec> models() {
        return models;
    }

    /** Where a state nothing identifies goes. */
    public Checkpoint defaultCheckpoint() {
        return defaultCheckpoint;
    }

    /** Whether question ids alone may select {@code typed-decisions}. */
    public boolean autoTaskDetection() {
        return autoTaskDetection;
    }

    // ------------------------------------------------------------------ the registry

    /**
     * The checkpoint a name or alias means.
     *
     * <p>Case and surrounding whitespace are ignored, because these names are typed by hand and
     * read out of configuration files.
     *
     * @throws IllegalArgumentException for anything that is not a checkpoint or an alias, with the
     *     reference's message listing both sets
     */
    public static Checkpoint normaliseName(String name) {
        Checkpoint resolved = lookupName(name);
        if (resolved == null) {
            throw new IllegalArgumentException(String.format(
                    "unknown model %s; choose one of %s (or an alias: %s)",
                    PythonJson.repr(name), pythonList(checkpointNames()),
                    pythonList(new TreeSet<>(ALIASES.keySet()))));
        }
        return resolved;
    }

    /**
     * The registry spec for a checkpoint name or alias, or null when it is not one.
     *
     * <p>The non-throwing sibling of {@link #normaliseName}, for a caller that also accepts things
     * the registry knows nothing about -- a hub repository id, a local directory, an ONNX export.
     * Those are not errors there; they simply are not registry names.
     */
    public static ModelSpec resolveModelSpec(String name) {
        Checkpoint resolved = lookupName(name);
        return resolved == null ? null : BUNDLED.get(resolved);
    }

    private static Checkpoint lookupName(String name) {
        if (name == null) {
            return null;
        }
        // UnicodeTables.strip, not String.trim: trim only removes code points at or below
        // U+0020, so a name padded with a no-break space or an ideographic space -- which a
        // config file pasted from a browser or produced by a CJK input method carries routinely
        // -- would fail to resolve where the reference resolves it.
        String key = UnicodeTables.pythonLower(UnicodeTables.strip(name));
        Checkpoint alias = ALIASES.get(key);
        if (alias != null) {
            return alias;
        }
        for (Checkpoint candidate : Checkpoint.values()) {
            if (candidate.wireName().equals(key)) {
                return candidate;
            }
        }
        return null;
    }

    private static List<String> checkpointNames() {
        List<String> names = new ArrayList<>();
        for (Checkpoint candidate : Checkpoint.values()) {
            names.add(candidate.wireName());
        }
        Collections.sort(names);
        return names;
    }

    /** Python's {@code repr} of a list of strings, which is what its error message interpolates. */
    private static String pythonList(Collection<String> values) {
        StringBuilder out = new StringBuilder("[");
        boolean first = true;
        for (String value : values) {
            if (!first) {
                out.append(", ");
            }
            out.append(PythonJson.repr(value));
            first = false;
        }
        return out.append(']').toString();
    }

    /**
     * The typed-decisions workflow whose question ids these are, or null.
     *
     * <p>An exact id-set match. A superset is not a match either: a support schema that adds one
     * field of its own is not the synthetic workflow the checkpoint was tuned on, and capturing it
     * would silently change model.
     */
    public static String matchTypedDecisionsWorkflow(Map<String, ?> questions) {
        if (questions == null || questions.isEmpty()) {
            return null;
        }
        // Copied into a set that tolerates null, and tested member by member. `Set.of(...)`
        // throws on a null argument to `containsAll`, so a question map with a null id -- which
        // a HashMap allows -- would abort the whole route where the reference merely fails to
        // match and carries on to detection.
        Set<String> ids = new LinkedHashSet<>(questions.keySet());
        for (Map.Entry<String, Set<String>> entry : TYPED_DECISION_WORKFLOWS.entrySet()) {
            Set<String> signature = entry.getValue();
            if (signature.size() != ids.size()) {
                continue;
            }
            boolean matches = true;
            for (String id : ids) {
                if (id == null || !signature.contains(id)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return entry.getKey();
            }
        }
        return null;
    }

    /** The four workflow signatures, by name. */
    public static Map<String, Set<String>> typedDecisionWorkflows() {
        return TYPED_DECISION_WORKFLOWS;
    }

    /**
     * An order-sensitive signature for a question schema, so states asking the same thing can
     * share a forward pass.
     *
     * <p>Order-sensitive at every level, because option order is positional: a choice between
     * {@code {a, b}} and one between {@code {b, a}} are different questions whose answers mean
     * different things, and they must not be grouped. See {@link Question#spec()} for what a
     * typed question can and cannot carry into this.
     */
    public static String questionSchema(Map<String, Question> questions) {
        Map<String, Object> specs = new LinkedHashMap<>();
        if (questions != null) {
            for (Map.Entry<String, Question> entry : questions.entrySet()) {
                specs.put(entry.getKey(),
                        entry.getValue() == null ? null : entry.getValue().spec());
            }
        }
        return PythonJson.dumps(specs);
    }

    /**
     * True or false for a language code, or null when the code identifies nothing.
     *
     * <p>Accepts the forms a caller has to hand: {@code en}, {@code EN}, {@code en-US}, the POSIX
     * {@code en_US} that {@code $LANG} holds, and {@code en_US.UTF-8}.
     *
     * <p>Null means "no usable hint", which is what lets a language-identification model abstain
     * -- and it is also what a code naming no language returns, so {@code LANG=C} falls through to
     * detection instead of pinning every request to one checkpoint.
     *
     * <p>Routing needs one bit, not a language id: is this English Latin text, or something the
     * English checkpoint cannot read. So every code that names some other language answers false.
     */
    public static Boolean englishFromCode(Object value) {
        if (value == null) {
            return null;
        }
        // Python's strip and Python's lower, for the reason given on normaliseName. Getting this
        // wrong is worse here than there: a `lang` of "en" with a trailing ideographic space
        // would route to the multilingual checkpoint while the reason string still reported that
        // the caller asked for English, and a padded "C " would pin every request to
        // multilingual instead of falling through to detection.
        String code = UnicodeTables.pythonLower(UnicodeTables.strip(String.valueOf(value)));
        if (code.isEmpty()) {
            return null;
        }
        int dot = code.indexOf('.');
        if (dot >= 0) {
            code = code.substring(0, dot);                 // en_US.UTF-8 -> en_US
        }
        String primary = code.replace('_', '-');
        int dash = primary.indexOf('-');
        if (dash >= 0) {
            primary = primary.substring(0, dash);          // en_US -> en
        }
        if (primary.isEmpty() || LANGUAGE_AGNOSTIC_CODES.contains(primary)) {
            return null;
        }
        return ENGLISH_SUBTAGS.contains(primary);
    }

    // ------------------------------------------------------------------ routing

    /** Decide where this state goes. */
    public RouteDecision route(Object state) {
        return route(state, null, RouteOptions.none());
    }

    /** Decide where this state goes, with the questions available for workflow detection. */
    public RouteDecision route(Object state, Map<String, Question> questions) {
        return route(state, questions, RouteOptions.none());
    }

    /**
     * Decide where this state goes.
     *
     * <p>Runs no model and reads no disk. Precedence, highest first: an explicit model, an
     * explicit task, a detected workflow when {@link Builder#autoTaskDetection} is on, an explicit
     * language, a per-call hint, the router's installed hint, detection, then the default.
     */
    public RouteDecision route(Object state, Map<String, Question> questions,
            RouteOptions options) {
        RouteOptions settings = options == null ? RouteOptions.none() : options;

        if (settings.model() != null) {
            Checkpoint key = normaliseName(settings.model());
            return decision(key, "explicit model=" + PythonJson.repr(settings.model()), null,
                    null);
        }

        if (settings.task() != null) {
            String task = settings.task();
            // The reference accepts the task under either spelling, and reports back whichever
            // the caller wrote.
            Checkpoint key = "typed_decisions".equals(
                    task.toLowerCase(Locale.ROOT).replace('-', '_'))
                    ? Checkpoint.TYPED_DECISIONS
                    : normaliseName(task);
            return decision(key, "explicit task=" + PythonJson.repr(task), null, null);
        }

        // Computed here, and reported from here on even when a later branch decides: a caller
        // that pinned a language still wants to know its schema was a known workflow.
        String workflow = matchTypedDecisionsWorkflow(questions);
        if (workflow != null && autoTaskDetection) {
            return decision(Checkpoint.TYPED_DECISIONS,
                    "question ids match the " + PythonJson.repr(workflow)
                    + " typed-decisions workflow", null, workflow);
        }

        if (settings.lang() != null) {
            // Decisive only when the code names a language. Blank or whitespace is no usable
            // hint, so it falls through exactly as an abstaining hint does; a real code routes.
            Boolean resolved = englishFromCode(settings.lang());
            if (resolved != null) {
                Checkpoint key = resolved ? Checkpoint.ENGLISH : Checkpoint.MULTILINGUAL;
                return decision(key, "explicit lang=" + PythonJson.repr(settings.lang()), null,
                        workflow);
            }
        }

        // The caller's hint: the per-call one first, then the one installed on the router. Only a
        // hint that actually answers routes here; anything else falls through to detection.
        RouteDecision hinted = fromHint("lang_guess", settings.langGuess(), state, workflow);
        if (hinted != null) {
            return hinted;
        }
        hinted = fromHint("Router(lang_guess=...)", langGuess, state, workflow);
        if (hinted != null) {
            return hinted;
        }

        LanguageDetection.Analysis detection = LanguageDetection.analyse(state);
        Checkpoint key;
        String reason;
        if ("unknown".equals(detection.script())) {
            key = defaultCheckpoint;
            reason = "no letters detected in state; using default (" + key.wireName() + ")";
        } else if (!"latin".equals(detection.script())) {
            key = Checkpoint.MULTILINGUAL;
            reason = "non-Latin script (" + detection.script() + ", "
                    + PythonJson.percent0(detection.nonLatinFraction())
                    + "% of letters); the English checkpoint cannot read it";
        } else if (!detection.english()) {
            key = Checkpoint.MULTILINGUAL;
            if (detection.mixedSegment() != null && !detection.mixedSegment().isEmpty()) {
                reason = "Latin script, mostly English, but a line or field reads as "
                        + PythonJson.repr(detection.language()) + " ("
                        + PythonJson.repr(headCodePoints(detection.mixedSegment(), 60))
                        + "); the English checkpoint cannot read it";
            } else if (detection.language() != null) {
                reason = "Latin script but language looks like "
                        + PythonJson.repr(detection.language()) + ", not English";
            } else {
                // An unidentified Latin-script language, routed on the non-English letters alone,
                // because no stopword list here covers it.
                reason = "Latin script, language not identified but "
                        + PythonJson.percent0(detection.diacriticRate())
                        + "% non-English letters; not safe for the English checkpoint";
            }
        } else if (detection.languageUndecided()) {
            // Nothing identifies the language: too short, or only content words. That is no
            // evidence of English either, so it takes the same default as a state with no letters.
            key = defaultCheckpoint;
            reason = "Latin script, language not identified and no non-English letters; "
                    + "using default (" + key.wireName() + ")";
        } else {
            key = Checkpoint.ENGLISH;
            reason = "English Latin text";
        }
        return decision(key, reason, detection, workflow);
    }

    private RouteDecision fromHint(String source, LanguageHint hint, Object state,
            String workflow) {
        if (hint == null) {
            return null;
        }
        Boolean resolved = englishFromCode(hint.codeFor(state));
        if (resolved == null) {
            return null;
        }
        Checkpoint key = resolved ? Checkpoint.ENGLISH : Checkpoint.MULTILINGUAL;
        return decision(key, source + ": the caller identified this as "
                + (resolved ? "English" : "non-English") + " text", null, workflow);
    }

    private RouteDecision decision(Checkpoint key, String reason,
            LanguageDetection.Analysis detection, String workflow) {
        return new RouteDecision(key, models.get(key).repoString(), reason, detection, workflow);
    }

    // ------------------------------------------------------------------ the lifecycle

    /**
     * Borrow the agent for a checkpoint, building it on first use.
     *
     * <p>Concurrent callers asking for the same checkpoint share one build rather than each
     * paying for a copy. The lease keeps the agent open; see {@link Lease} for why that matters
     * here and not in the reference.
     */
    public Lease lease(String name) {
        return lease(normaliseName(name));
    }

    /** Borrow the agent for a checkpoint. */
    public Lease lease(Checkpoint checkpoint) {
        requireNonNull(checkpoint, "checkpoint");
        Slot slot = acquire(checkpoint);
        return new Lease(checkpoint, slot);
    }

    /**
     * The agent for a checkpoint, building it on first use.
     *
     * <p><b>Valid only while it stays resident.</b> Eviction and {@link #unload} close an agent
     * the router built, so a reference held across either is a closed session. That is the price
     * of a native resource on a JVM -- the reference can rely on refcounting here and this cannot.
     * Prefer {@link #lease} or {@link #predict}, which hold the agent open for exactly as long as
     * they use it; this exists for a caller doing its own lifetime management, and for
     * {@link #preload}.
     */
    public Agent load(String name) {
        Checkpoint checkpoint = normaliseName(name);
        Slot slot = acquire(checkpoint);
        release(slot);
        return slot.agent;
    }

    /** The slot for a checkpoint, with one lease taken. */
    private Slot acquire(Checkpoint checkpoint) {
        while (true) {
            InFlight waitFor = null;
            InFlight mine = null;
            lock.lock();
            try {
                requireOpen();
                // The fast path, and not merely an optimisation: it returns a resident checkpoint
                // without touching `buildLock`, so a request for a loaded model is not stalled
                // behind a cold load of a different one. Remove it and every cached call queues
                // on the build lock, which in a server means one checkpoint loading blocks all
                // the traffic the other checkpoints could have served.
                Slot resident = slots.get(checkpoint);
                if (resident != null) {
                    touch(checkpoint);
                    resident.leases++;
                    return resident;
                }
                waitFor = loading.get(checkpoint);
                if (waitFor == null) {
                    mine = new InFlight();
                    loading.put(checkpoint, mine);
                }
            } finally {
                lock.unlock();
            }
            if (mine != null) {
                // This caller owns the build, and `mine` is the handle the others are waiting on.
                // It has to be the one registered above: creating a second one here would leave
                // every waiter blocked on a latch nobody ever counts down.
                return build(checkpoint, mine);
            }
            // Someone else is building this one. Wait for them rather than build a second copy.
            await(waitFor);
            RuntimeException failure = waitFor.error;
            if (failure != null) {
                // Wrapped per waiter, with the builder's error as the cause. Rethrowing the one
                // instance gave every waiter a stack trace of frames it never executed -- the
                // builder's -- and let concurrent handlers mutate one Throwable's suppressed
                // list, which Throwable does not support.
                throw new IllegalStateException("loading the " + checkpoint.wireName()
                        + " checkpoint failed on the thread that was building it", failure);
            }
        }
    }

    private Slot build(Checkpoint checkpoint, InFlight inflight) {
        List<Agent> toClose = new ArrayList<>();
        Slot built;
        try {
            // The build itself runs outside `lock`, so routing and eviction are not stalled for
            // the seconds a cold checkpoint takes, and serialised by `buildLock` so two cold
            // loads do not hold two checkpoints in flight at once.
            buildLock.lock();
            try {
                lock.lock();
                try {
                    Slot resident = slots.get(checkpoint);
                    if (resident != null) {
                        // Built or attached while this caller waited for the build lock.
                        touch(checkpoint);
                        resident.leases++;
                        finish(checkpoint, inflight);
                        return resident;
                    }
                } finally {
                    lock.unlock();
                }
                Agent agent = create(checkpoint);
                boolean duplicate = false;
                lock.lock();
                try {
                    Slot attached = slots.get(checkpoint);
                    if (attached != null) {
                        // Attached while it was building: keep that one and close the duplicate,
                        // which the reference leaves to refcounting.
                        touch(checkpoint);
                        attached.leases++;
                        built = attached;
                        duplicate = true;
                    } else {
                        built = new Slot(agent, true);
                        built.leases++;
                        slots.put(checkpoint, built);
                        order.add(checkpoint);
                        evictLocked(toClose);
                    }
                } finally {
                    // ALWAYS, even if eviction threw. Skipping it left the new slot published
                    // with a lease that nothing could release -- so the agent could never be
                    // retired or closed -- and left every waiter parked on a latch for a
                    // checkpoint that had in fact loaded.
                    finish(checkpoint, inflight);
                    lock.unlock();
                }
                if (duplicate) {
                    toClose.add(agent);
                }
            } finally {
                buildLock.unlock();
            }
        } catch (RuntimeException | Error failure) {
            lock.lock();
            try {
                inflight.error = failure instanceof RuntimeException
                        ? (RuntimeException) failure
                        : new IllegalStateException("building " + checkpoint.wireName()
                                + " failed", failure);
                loading.remove(checkpoint, inflight);
                inflight.done.countDown();
            } finally {
                lock.unlock();
            }
            throw failure;
        }
        // Outside every lock. Closing a native session can be slow, and doing it under `lock`
        // stalled callers asking for an ALREADY-RESIDENT checkpoint -- which is the one thing the
        // fast path in `acquire` exists to keep fast. Measured at 1.3 s for an unrelated lease
        // behind one slow close.
        try {
            closeAll(toClose);
        } catch (RuntimeException | Error failure) {
            // The new slot is already published and holds the lease this call took. Throwing
            // without releasing it would strand that lease forever: nothing else can reach it,
            // so the agent could never be retired or closed -- an unreachable, unclosable native
            // session. The caller is about to get an exception instead of the lease, so giving it
            // back here is exactly right.
            release(built);
            throw failure;
        }
        return built;
    }

    private Agent create(Checkpoint checkpoint) {
        if (agents == null) {
            throw new IllegalStateException(
                    "this Router can route but not load: no agent factory was configured. Use"
                    + " Router.builder().checkpointsRoot(path) or .agents(factory) to give it one,"
                    + " or call route(...) and load the checkpoint yourself.");
        }
        try {
            Agent agent = agents.create(checkpoint);
            if (agent == null) {
                throw new IllegalStateException("the agent factory returned null for "
                        + checkpoint.wireName());
            }
            return agent;
        } catch (IOException failure) {
            throw new UncheckedIOException("cannot load the " + checkpoint.wireName()
                    + " checkpoint from " + models.get(checkpoint).repoString(), failure);
        }
    }

    private void finish(Checkpoint checkpoint, InFlight inflight) {
        loading.remove(checkpoint, inflight);
        inflight.done.countDown();
    }

    private static void await(InFlight inflight) {
        try {
            inflight.done.await();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while waiting for a checkpoint to load",
                    interrupted);
        }
    }

    /** Release one lease, closing the agent when it was retired and this was the last one. */
    private void release(Slot slot) {
        Agent toClose = null;
        lock.lock();
        try {
            slot.leases--;
            if (slot.leases <= 0 && slot.retired && slot.ownsClosing) {
                toClose = slot.agent;
            }
        } finally {
            lock.unlock();
        }
        if (toClose != null) {
            toClose.close();
        }
    }

    /** Move a checkpoint to the most-recently-used end. Caller holds {@link #lock}. */
    private void touch(Checkpoint checkpoint) {
        order.remove(checkpoint);
        order.add(checkpoint);
    }

    /**
     * Drop least-recently-used agents until {@link #maxLoaded} holds. Caller holds the lock.
     *
     * @return the checkpoints that left
     */
    private List<Checkpoint> evictLocked(List<Agent> toClose) {
        List<Checkpoint> evicted = new ArrayList<>();
        while (order.size() > maxLoaded) {
            Checkpoint victim = order.remove(0);
            Slot slot = slots.remove(victim);
            if (slot != null) {
                evicted.add(victim);
                retire(slot, toClose);
            }
        }
        // Keep the two views consistent, as the reference does: a slot with no place in the order
        // is not reachable and would otherwise be held for the life of the router.
        if (order.size() < slots.size()) {
            for (Checkpoint stray : new ArrayList<>(slots.keySet())) {
                if (!order.contains(stray)) {
                    Slot slot = slots.remove(stray);
                    if (slot != null) {
                        evicted.add(stray);
                        retire(slot, toClose);
                    }
                }
            }
        }
        return evicted;
    }

    /** Mark a slot gone, and collect its agent for closing when nothing is using it. */
    private void retire(Slot slot, List<Agent> toClose) {
        slot.retired = true;
        if (slot.leases <= 0 && slot.ownsClosing) {
            toClose.add(slot.agent);
        }
    }

    /**
     * Register an already-built agent instead of loading a second copy.
     *
     * <p>Useful when the process holds a checkpoint for other reasons: a service that already
     * built the English one can hand it over rather than pay for -- and keep resident -- a
     * duplicate 421M parameters.
     *
     * <p>The router never closes an attached agent: the caller keeps ownership. It also raises
     * {@link #maxLoaded} to fit whatever is now resident, so attaching never immediately evicts
     * what it just attached.
     */
    public Agent attach(String name, Agent agent) {
        requireNonNull(agent, "agent");
        Checkpoint checkpoint = normaliseName(name);
        List<Agent> toClose = new ArrayList<>();
        lock.lock();
        try {
            requireOpen();
            Slot previous = slots.put(checkpoint, new Slot(agent, false));
            // Not when it is the SAME agent. `attach`ing the instance `load` just returned is
            // exactly what the javadoc invites, and retiring the previous slot would close the
            // very agent the caller is being told the router will never close -- leaving a live
            // slot holding a closed session that nothing can ever replace.
            if (previous != null && previous.agent != agent) {
                retire(previous, toClose);
            }
            touch(checkpoint);
            maxLoaded = Math.max(maxLoaded, slots.size());
        } finally {
            lock.unlock();
        }
        closeAll(toClose);
        return agent;
    }

    /** Build every checkpoint up front, so no request pays a cold load. */
    public Router preload() {
        return preload(null);
    }

    /**
     * Build these checkpoints up front, so no request pays a cold load.
     *
     * <p>A cold load costs seconds; routing costs microseconds. With the checkpoints resident,
     * routing is effectively free, which is what a server wants. {@link #maxLoaded} is raised to
     * fit both what is asked for and whatever is already resident, so preloading in stages does
     * not evict either.
     *
     * @param names checkpoint names or aliases, or null for all of them
     */
    public Router preload(Collection<String> names) {
        List<Checkpoint> wanted = new ArrayList<>();
        if (names == null) {
            wanted.addAll(models.keySet());
        } else {
            for (String name : names) {
                wanted.add(normaliseName(name));
            }
        }
        lock.lock();
        try {
            requireOpen();
            Set<Checkpoint> union = new LinkedHashSet<>(wanted);
            union.addAll(slots.keySet());
            maxLoaded = Math.max(maxLoaded, union.size());
        } finally {
            lock.unlock();
        }
        for (Checkpoint checkpoint : wanted) {
            lock.lock();
            boolean already;
            try {
                already = slots.containsKey(checkpoint);
            } finally {
                lock.unlock();
            }
            if (!already) {
                release(acquire(checkpoint));
            }
        }
        return this;
    }

    /**
     * Free one checkpoint.
     *
     * <p>Waits for an in-flight build of that checkpoint only, so an unrelated cold load does not
     * stall this call.
     *
     * @return the checkpoints this router let go of, which is empty when it was not resident.
     *     "Let go of" and "closed" are not the same thing when a lease is outstanding: the slot
     *     is dropped immediately and its agent closes when the last lease is released.
     */
    public List<Checkpoint> unload(String name) {
        Checkpoint checkpoint = normaliseName(name);
        while (true) {
            InFlight inflight;
            List<Agent> toClose = new ArrayList<>();
            List<Checkpoint> freed = new ArrayList<>();
            lock.lock();
            try {
                inflight = loading.get(checkpoint);
                if (inflight == null) {
                    Slot slot = slots.remove(checkpoint);
                    order.remove(checkpoint);
                    if (slot != null) {
                        freed.add(checkpoint);
                        retire(slot, toClose);
                    }
                }
            } finally {
                lock.unlock();
            }
            if (inflight == null) {
                closeAll(toClose);
                return freed;
            }
            await(inflight);
        }
    }

    /** Free every checkpoint, waiting for any in-flight build first. */
    public List<Checkpoint> unloadAll() {
        while (true) {
            List<InFlight> inflights;
            List<Agent> toClose = new ArrayList<>();
            List<Checkpoint> freed = new ArrayList<>();
            lock.lock();
            try {
                inflights = new ArrayList<>(loading.values());
                if (inflights.isEmpty()) {
                    freed.addAll(order);
                    for (Slot slot : slots.values()) {
                        retire(slot, toClose);
                    }
                    slots.clear();
                    order.clear();
                }
            } finally {
                lock.unlock();
            }
            if (inflights.isEmpty()) {
                closeAll(toClose);
                return freed;
            }
            for (InFlight inflight : inflights) {
                await(inflight);
            }
        }
    }

    /** Which checkpoints are resident, least recently used first. */
    public List<Checkpoint> loaded() {
        lock.lock();
        try {
            return List.copyOf(order);
        } finally {
            lock.unlock();
        }
    }

    /**
     * How many checkpoints may be resident at once.
     *
     * <p>A high-water mark, not a window: {@link #attach} and {@link #preload} raise it to fit
     * what they make resident and nothing lowers it again, so detaching later leaves the raised
     * ceiling in place. That is the reference's behaviour and is kept deliberately; a caller who
     * needs the original ceiling back should build a new router.
     */
    public int maxLoaded() {
        lock.lock();
        try {
            return maxLoaded;
        } finally {
            lock.unlock();
        }
    }

    /** Route this state and answer its questions on whichever checkpoint wins. */
    public Prediction predict(Object state, Map<String, Question> questions) {
        return predict(state, questions, RouteOptions.none());
    }

    /** Route this state and answer its questions, with per-call routing arguments. */
    public Prediction predict(Object state, Map<String, Question> questions,
            RouteOptions options) {
        return predict(route(state, questions, options), state, questions,
                options == null ? null : options.lang());
    }

    /**
     * Answer these questions on an already-made decision.
     *
     * <p>For a caller that wants the reason as well as the answer: route once, show or log the
     * decision, then predict with it, rather than routing twice.
     */
    public Prediction predict(RouteDecision decision, Object state,
            Map<String, Question> questions) {
        return predict(decision, state, questions, null);
    }

    private Prediction predict(RouteDecision decision, Object state,
            Map<String, Question> questions, String lang) {
        requireNonNull(decision, "decision");
        // The language the checkpoint is told is the caller's if they gave one, else whatever
        // routing detected -- which is the reference's rule, and it matters: the multilingual
        // checkpoint takes a language and the detected one is the best available answer.
        String language = lang;
        if (language == null && decision.detection() != null) {
            language = decision.detection().language();
        }
        try (Lease lease = lease(decision.model())) {
            return lease.agent().predict(state, questions, language);
        }
    }

    /**
     * Close the router, freeing every checkpoint it built.
     *
     * <p>An attached agent is left alone: its owner is the caller. An agent still under lease is
     * closed when that lease is released, so closing the router does not break a prediction
     * already in flight.
     */
    @Override
    public void close() {
        lock.lock();
        try {
            closed = true;
        } finally {
            lock.unlock();
        }
        unloadAll();
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("this Router is closed");
        }
    }

    private static void closeAll(List<Agent> agents) {
        RuntimeException first = null;
        for (Agent agent : agents) {
            try {
                agent.close();
            } catch (RuntimeException failure) {
                if (first == null) {
                    first = failure;
                } else {
                    first.addSuppressed(failure);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    /** The first {@code count} code points, which is what the reference's slice takes. */
    private static String headCodePoints(String text, int count) {
        if (text.codePointCount(0, text.length()) <= count) {
            return text;
        }
        return text.substring(0, text.offsetByCodePoints(0, count));
    }

    /** The checkpoints this router knows, in declaration order. */
    public static List<Checkpoint> checkpoints() {
        return List.of(Checkpoint.values());
    }

    /** The alias table, for a caller that wants to show the accepted names. */
    public static Set<String> aliases() {
        return new LinkedHashSet<>(ALIASES.keySet());
    }

    private static <T> T requireNonNull(T value, String what) {
        if (value == null) {
            throw new IllegalArgumentException(what + " must not be null");
        }
        return value;
    }
}
