namespace Laya;

/// <summary>
/// The routing outcome: which checkpoint, why, and what was detected. Ports Python's
/// <c>RouteDecision</c> (a <c>dict</c> subclass there; a plain record here, since the .NET SDK
/// has no need to serialise it as a raw dictionary).
/// </summary>
/// <param name="Model">The checkpoint selected.</param>
/// <param name="Reason">
/// Human-readable explanation, byte-identical to Python's (including <c>%r</c>-style quoting and
/// half-even <c>%.0f%%</c> percentage formatting).
/// </param>
/// <param name="Detection">
/// The language/script detection behind the decision, or <see langword="null"/> when the model
/// was picked by an explicit <c>model</c>/<c>task</c>/<c>lang</c> argument instead.
/// </param>
/// <param name="Workflow">
/// The typed-decisions workflow the question ids matched, if any — reported even when
/// <see cref="LayaRouterOptions.AutoTaskDetection"/> is off and so played no part in the decision.
/// </param>
public sealed record RouteDecision(LayaCheckpoint Model, string Reason, LanguageAnalysis? Detection, string? Workflow);

/// <summary>Options for a <see cref="LayaRouter"/>.</summary>
public sealed class LayaRouterOptions
{
    /// <summary>
    /// Template used to build each checkpoint's <see cref="LayaEngine"/>: execution provider,
    /// thread counts, <see cref="LayaOptions.AllowDownload"/>, cache directory and HuggingFace
    /// token/repo are all taken from this template. <see cref="LayaOptions.Checkpoint"/> and
    /// <see cref="LayaOptions.ModelDirectory"/> on the template are ignored — the router always
    /// resolves each checkpoint through its own <see cref="LayaOptions.Checkpoint"/>-aware path
    /// (<c>LAYA_ONNX_ROOT</c>, cache, or download), the same way <see cref="LayaEngine.Create"/> does.
    /// </summary>
    public LayaOptions EngineOptions { get; set; } = new();

    /// <summary>
    /// How many checkpoints stay resident at once (least-recently-used is evicted). Default 2,
    /// matching Python's <c>Router(max_loaded=2)</c>, so alternating two languages does not reload
    /// a checkpoint on every request. Raised automatically by <see cref="LayaRouter.Attach"/> and
    /// <see cref="LayaRouter.Preload"/> to fit what they load.
    /// </summary>
    public int MaxLoaded { get; set; } = 2;

    /// <summary>
    /// Checkpoint used when a state has no letters or its Latin language is undetermined without
    /// non-English evidence. Default
    /// <see cref="LayaCheckpoint.Multilingual"/>, matching Python's default.
    /// </summary>
    // Follows the reference, which moved this to multilingual: undecided Latin text is no
    // evidence of English, and multilingual leads on 50 of the 51 languages swept.
    public LayaCheckpoint Default { get; set; } = LayaCheckpoint.Multilingual;

    /// <summary>
    /// Whether <see cref="LayaRouter.Route(object?, QuestionSet?, string?, string?, string?)"/> may pick <see cref="LayaCheckpoint.TypedDecisions"/>
    /// on its own when the question ids match one of the four typed-decisions workflow
    /// signatures. Default <see langword="false"/>: that checkpoint is fine-tuned on specific
    /// synthetic workflows and must never be a silent default.
    /// </summary>
    public bool AutoTaskDetection { get; set; }

    /// <summary>Whether every checkpoint should be built immediately when the router is constructed.</summary>
    public bool Preload { get; set; }

    /// <summary>
    /// A language hint applied to every request, Python's <c>Router(lang_guess=...)</c>: a language
    /// code or a callback taking the state. Checked after an explicit <c>lang</c> and a per-call hint,
    /// and before the built-in detection; a hint that names no language falls through to detection.
    /// See <see cref="Laya.LangGuess"/>.
    /// </summary>
    public LangGuess? LangGuess { get; init; }
}

/// <summary>
/// Lazily loads Laya checkpoints and sends each request to the right one. Ports Python's
/// <c>Router</c>.
/// </summary>
/// <remarks>
/// <para>
/// <code>
/// using var router = new LayaRouter();
/// router.Predict(new { message = "Mein Konto wurde zweimal belastet" }, questions);  // -&gt; multilingual
/// router.Predict(new { message = "I was charged twice" }, questions);                // -&gt; english
/// router.Predict(state, questions, model: "typed-decisions");                        // explicit
/// </code>
/// </para>
/// <para>
/// Checkpoints are downloaded/built on first use. <see cref="LayaRouterOptions.MaxLoaded"/> caps
/// how many stay resident (least-recently-used is evicted). For a server or a demo, prefer
/// <see cref="LayaRouterOptions.Preload"/> or <see cref="Preload"/>: a cold load costs seconds,
/// while detection costs microseconds, so anything that alternates languages at
/// <c>MaxLoaded=1</c> reloads on every request.
/// </para>
/// <para>
/// <b>Lifetime.</b> Unlike Python, which relies on the garbage collector to keep an evicted
/// checkpoint alive until nothing references it, .NET must dispose the underlying
/// <see cref="LayaEngine"/> explicitly. The router protects itself: a checkpoint evicted or
/// unloaded while a router-driven <see cref="Predict(object, QuestionSet)"/> call is still
/// running on it is disposed only once that call returns, never mid-inference. <see cref="Load"/>,
/// however, hands back a bare reference with no such protection — if you keep using it after the
/// router has evicted it (e.g. by loading more checkpoints than <c>MaxLoaded</c> allows), it may
/// already be disposed. Route requests through <see cref="Predict(object, QuestionSet)"/>, or use
/// <see cref="Attach"/> to hand the router an instance you own and will dispose yourself.
/// </para>
/// </remarks>
public sealed class LayaRouter : ILayaLongPredictor, IDisposable
{
    // Checkpoint keys people are likely to type, mapped to their canonical name. Matches
    // Python's _ALIASES exactly.
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.Ordinal)
    {
        ["en"] = "english", ["laya"] = "english", ["default"] = "english",
        ["multi"] = "multilingual", ["ml"] = "multilingual", ["laya-multilingual"] = "multilingual",
        ["typed"] = "typed-decisions", ["typed_decisions"] = "typed-decisions",
        ["laya-typed-decisions"] = "typed-decisions", ["decisions"] = "typed-decisions",
    };

    // Question-id signatures of the four typed-decisions workflows, used only when
    // AutoTaskDetection is enabled. Order matches Python's _TYPED_DECISION_WORKFLOWS.
    private static readonly (string Workflow, string[] Ids)[] TypedDecisionWorkflows =
    [
        ("agent_trace_observability", ["action", "needs_review", "outcome", "risk", "urgency"]),
        ("customer_service", ["action", "category", "churn_risk", "needs_human", "urgency"]),
        ("invoice_processing", ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"]),
        ("security_incidents", ["credential_compromise", "disposition", "severity", "true_positive", "urgency"]),
    ];

    private static readonly LayaCheckpoint[] AllCheckpoints =
        [LayaCheckpoint.English, LayaCheckpoint.Multilingual, LayaCheckpoint.TypedDecisions];

    private sealed class Slot
    {
        public required ILayaPredictor Predictor { get; init; }
        public int Leases;
        public bool Retired;
        // False for an instance handed in via Attach: the router does not own its lifetime and
        // must never dispose it.
        public required bool OwnsDisposal { get; init; }
    }

    private readonly Func<LayaCheckpoint, ILayaPredictor> _engineFactory;
    private readonly Lock _lock = new();
    private readonly Dictionary<LayaCheckpoint, Slot> _slots = new();
    private readonly List<LayaCheckpoint> _order = []; // least-recently-used first
    private int _maxLoaded;
    private bool _disposed;

    /// <summary>Checkpoint used when a state has no letters at all.</summary>
    public LayaCheckpoint Default { get; }

    /// <summary>
    /// Whether <see cref="Route(object?, QuestionSet?, string?, string?, string?)"/> may pick <see cref="LayaCheckpoint.TypedDecisions"/> on its
    /// own when the question ids match a typed-decisions workflow signature.
    /// </summary>
    public bool AutoTaskDetection { get; }

    /// <summary>
    /// The language hint installed with <see cref="LayaRouterOptions.LangGuess"/>, applied to every
    /// request after a per-call hint, or <see langword="null"/>.
    /// </summary>
    public LangGuess? LangGuess { get; }

    /// <summary>Create a router that builds real <see cref="LayaEngine"/> instances on demand.</summary>
    public LayaRouter(LayaRouterOptions? options = null)
        : this(options, BuildDefaultFactory((options ?? new LayaRouterOptions()).EngineOptions))
    {
    }

    /// <summary>
    /// Test-only seam: builds checkpoints via <paramref name="engineFactory"/> instead of
    /// <see cref="LayaEngine.Create"/>, so LRU/attach/preload/unload behaviour can be exercised
    /// with fake predictors and no ONNX weights.
    /// </summary>
    internal LayaRouter(LayaRouterOptions? options, Func<LayaCheckpoint, ILayaPredictor> engineFactory)
    {
        ArgumentNullException.ThrowIfNull(engineFactory);
        options ??= new LayaRouterOptions();
        _engineFactory = engineFactory;
        _maxLoaded = Math.Max(1, options.MaxLoaded);
        Default = options.Default;
        AutoTaskDetection = options.AutoTaskDetection;
        LangGuess = options.LangGuess;
        if (options.Preload) Preload();
    }

    private static Func<LayaCheckpoint, ILayaPredictor> BuildDefaultFactory(LayaOptions template) => checkpoint =>
        LayaEngine.Create(new LayaOptions
        {
            Checkpoint = checkpoint,
            ExecutionProvider = template.ExecutionProvider,
            IntraOpThreads = template.IntraOpThreads,
            InterOpThreads = template.InterOpThreads,
            AllowDownload = template.AllowDownload,
            HuggingFaceRepo = template.HuggingFaceRepo,
            HuggingFaceToken = template.HuggingFaceToken,
            CacheDirectory = template.CacheDirectory,
            DownloadProgress = template.DownloadProgress,
        });

    // ── name resolution ──────────────────────────────────────────────────────

    /// <summary>
    /// Resolve a checkpoint name or alias (case-insensitive) to its <see cref="LayaCheckpoint"/>.
    /// </summary>
    /// <exception cref="ArgumentException">If <paramref name="name"/> matches no checkpoint or alias.</exception>
    public static LayaCheckpoint NormaliseName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        var key = name.Trim().ToLowerInvariant();
        if (Aliases.TryGetValue(key, out var mapped)) key = mapped;
        return key switch
        {
            "english" => LayaCheckpoint.English,
            "multilingual" => LayaCheckpoint.Multilingual,
            "typed-decisions" => LayaCheckpoint.TypedDecisions,
            // Python's text, list reprs included (router.py:187): sorted(DEFAULT_MODELS), sorted(_ALIASES).
            _ => throw new ArgumentException(
                $"unknown model {PythonStr.Repr(name)}; choose one of {PyList(["english", "multilingual", "typed-decisions"])} "
                + $"(or an alias: {PyList(Aliases.Keys)})"),
        };
    }

    private static string PyList(IEnumerable<string> items) =>
        "[" + string.Join(", ", items.OrderBy(k => k, StringComparer.Ordinal).Select(PythonStr.Repr)) + "]";

    /// <summary>
    /// Name of the typed-decisions workflow whose question ids these are, else
    /// <see langword="null"/>. Requires an exact id-set match, so an unrelated schema that
    /// happens to contain <c>"urgency"</c> is never captured.
    /// </summary>
    public static string? MatchTypedDecisionsWorkflow(QuestionSet? questions)
    {
        var ids = new HashSet<string>(questions?.Ids ?? [], StringComparer.Ordinal);
        foreach (var (workflow, signature) in TypedDecisionWorkflows)
            if (ids.SetEquals(signature)) return workflow;
        return null;
    }

    // ── routing ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Decide which checkpoint to use, without loading or running anything.
    /// </summary>
    /// <remarks>
    /// Precedence: explicit <paramref name="model"/> &gt; explicit <paramref name="task"/> &gt;
    /// detected workflow (opt-in via <see cref="AutoTaskDetection"/>) &gt; explicit
    /// <paramref name="lang"/> &gt; the hint installed with <see cref="LayaRouterOptions.LangGuess"/>
    /// &gt; detected script/language &gt; <see cref="Default"/>. An explicit <paramref name="lang"/>
    /// decides only when its code names a language (blank, <c>C</c>, <c>POSIX</c>, <c>und</c> ... fall through).
    /// </remarks>
    public RouteDecision Route(
        object? state, QuestionSet? questions = null,
        string? model = null, string? task = null, string? lang = null)
        => Route(state, questions, model, task, lang, null);

    /// <summary>
    /// <see cref="Route(object?, QuestionSet?, string?, string?, string?)"/> with Python's per-call
    /// <c>lang_guess</c> hint: checked after an explicit <paramref name="lang"/> and before the
    /// built-in detection, ahead of the hint installed with <see cref="LayaRouterOptions.LangGuess"/>.
    /// </summary>
    public RouteDecision Route(
        object? state, QuestionSet? questions,
        string? model, string? task, string? lang, LangGuess? langGuess)
    {
        if (model is not null)
            return new RouteDecision(NormaliseName(model), $"explicit model={PythonStr.Repr(model)}", null, null);

        if (task is not null)
        {
            var normalisedTaskSpelling = task.ToLowerInvariant().Replace("-", "_");
            var key = NormaliseName(normalisedTaskSpelling == "typed_decisions" ? "typed-decisions" : task);
            return new RouteDecision(key, $"explicit task={PythonStr.Repr(task)}", null, null);
        }

        var workflow = MatchTypedDecisionsWorkflow(questions);
        if (workflow is not null && AutoTaskDetection)
            return new RouteDecision(LayaCheckpoint.TypedDecisions,
                $"question ids match the {PythonStr.Repr(workflow)} typed-decisions workflow", null, workflow);

        if (lang is not null)
        {
            // An explicit `lang` is decisive only when the code names a language. Blank or whitespace
            // resolves to no usable hint, so it falls through to lang_guess/detection (router.py:1242).
            var resolved = LangGuess.EnglishFromCode(lang);
            if (resolved is { } isEnglish)
                return new RouteDecision(isEnglish ? LayaCheckpoint.English : LayaCheckpoint.Multilingual,
                    $"explicit lang={PythonStr.Repr(lang)}", null, workflow);
        }

        // Caller-supplied hint, per-call first then the one installed on the router (router.py:1254).
        foreach (var (source, hint) in new[] { ("lang_guess", langGuess), ("Router(lang_guess=...)", LangGuess) })
        {
            if (hint?.Resolve(state) is { } guessEnglish)
                return new RouteDecision(guessEnglish ? LayaCheckpoint.English : LayaCheckpoint.Multilingual,
                    $"{source}: the caller identified this as {(guessEnglish ? "English" : "non-English")} text",
                    null, workflow);
        }

        var det = LanguageDetection.Analyse(state);
        LayaCheckpoint model2;
        string reason;
        if (det.Script == "unknown")
        {
            model2 = Default;
            reason = $"no letters detected in state; using default ({CheckpointName(model2)})";
        }
        else if (det.Script != "latin")
        {
            model2 = LayaCheckpoint.Multilingual;
            reason = $"non-Latin script ({det.Script}, {FormatPercent0(det.NonLatinFraction)}% of letters); "
                    + "the English checkpoint cannot read it";
        }
        else if (!det.IsEnglish)
        {
            model2 = LayaCheckpoint.Multilingual;
            reason = det.MixedSegment is { Length: > 0 } mixed
                ? $"Latin script, mostly English, but a line or field reads as {PythonStr.Repr(det.Language!)} "
                  + $"({PythonStr.Repr(LanguageDetection.SliceCodePoints(mixed, 60))}); the English checkpoint cannot read it"
                : det.Language is { } identified
                ? $"Latin script but language looks like {PythonStr.Repr(identified)}, not English"
                // Unidentified Latin-script language: routed on the non-English letters alone,
                // because no stopword list here covers it.
                : $"Latin script, language not identified but {FormatPercent0(det.DiacriticRate)}% "
                  + "non-English letters; not safe for the English checkpoint";
        }
        else if (det.LanguageUndecided)
        {
            model2 = Default;
            reason = $"Latin script, language not identified and no non-English letters; using default ({CheckpointName(model2)})";
        }
        else
        {
            model2 = LayaCheckpoint.English;
            reason = "English Latin text";
        }
        return new RouteDecision(model2, reason, det, workflow);
    }

    private static string CheckpointName(LayaCheckpoint c) => LayaOptions.CheckpointSubfolder(c);

    /// <summary>Python's <c>%.0f%%</c>: half-even rounding to the nearest integer percent.</summary>
    private static string FormatPercent0(double fraction) =>
        Math.Round(100.0 * fraction, MidpointRounding.ToEven).ToString("0", System.Globalization.CultureInfo.InvariantCulture);

    // ── loading ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Return the predictor for <paramref name="name"/>, downloading and building it on first
    /// use. Concurrent callers share a single build instead of racing duplicates.
    /// </summary>
    /// <remarks>See the lifetime note on <see cref="LayaRouter"/>: the returned reference is not leased.</remarks>
    public ILayaPredictor Load(string name)
    {
        var key = NormaliseName(name);
        lock (_lock)
        {
            ThrowIfDisposed();
            var slot = FindOrBuild(key);
            Evict();
            return slot.Predictor;
        }
    }

    // Must be called with _lock held.
    private Slot FindOrBuild(LayaCheckpoint key)
    {
        if (_slots.TryGetValue(key, out var existing))
        {
            Touch(key);
            return existing;
        }
        var predictor = _engineFactory(key);
        var slot = new Slot { Predictor = predictor, OwnsDisposal = true };
        _slots[key] = slot;
        _order.Add(key);
        return slot;
    }

    // Must be called with _lock held.
    private void Touch(LayaCheckpoint key)
    {
        _order.Remove(key);
        _order.Add(key);
    }

    // Must be called with _lock held. Evicts down to _maxLoaded; a victim still leased (an
    // in-flight router-driven Predict) is marked retired instead of disposed immediately.
    private void Evict()
    {
        while (_order.Count > _maxLoaded)
        {
            var victim = _order[0];
            _order.RemoveAt(0);
            if (_slots.Remove(victim, out var slot)) RetireOrDispose(slot);
        }
    }

    // Must be called with _lock held.
    private static void RetireOrDispose(Slot slot)
    {
        slot.Retired = true;
        if (slot.Leases == 0 && slot.OwnsDisposal && slot.Predictor is IDisposable disposable) disposable.Dispose();
    }

    /// <summary>
    /// Register an already-built predictor under <paramref name="name"/> instead of loading a
    /// second copy. Useful when the process has a checkpoint loaded for other reasons.
    /// </summary>
    /// <remarks>
    /// The router never disposes an attached predictor — the caller keeps ownership. Raises
    /// <see cref="LayaRouterOptions.MaxLoaded"/> to fit whatever is now resident, so it is not
    /// immediately evicted.
    /// </remarks>
    public ILayaPredictor Attach(string name, ILayaPredictor predictor)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        var key = NormaliseName(name);
        lock (_lock)
        {
            ThrowIfDisposed();
            if (_slots.TryGetValue(key, out var old)) RetireOrDispose(old);
            _slots[key] = new Slot { Predictor = predictor, OwnsDisposal = false };
            Touch(key);
            _maxLoaded = Math.Max(_maxLoaded, _slots.Count);
        }
        return predictor;
    }

    /// <summary>
    /// Download and build checkpoints up front so no request ever pays a model load.
    /// <see cref="LayaRouterOptions.MaxLoaded"/> is raised to fit whatever is preloaded.
    /// Omit <paramref name="names"/> to preload every checkpoint.
    /// </summary>
    public LayaRouter Preload(IEnumerable<string>? names = null)
    {
        var keys = names is null
            ? AllCheckpoints
            : names.Select(NormaliseName).ToArray();
        lock (_lock)
        {
            ThrowIfDisposed();
            _maxLoaded = Math.Max(_maxLoaded, Math.Max(keys.Length, _slots.Count));
            foreach (var key in keys)
            {
                if (_slots.ContainsKey(key)) continue; // an attached predictor is already built
                FindOrBuild(key);
                Evict();
            }
        }
        return this;
    }

    /// <summary>Free one checkpoint, or all of them when <paramref name="name"/> is omitted.</summary>
    public void Unload(string? name = null)
    {
        lock (_lock)
        {
            ThrowIfDisposed();
            if (name is null)
            {
                foreach (var slot in _slots.Values) RetireOrDispose(slot);
                _slots.Clear();
                _order.Clear();
            }
            else
            {
                var key = NormaliseName(name);
                if (_slots.Remove(key, out var slot)) RetireOrDispose(slot);
                _order.Remove(key);
            }
        }
    }

    /// <summary>Checkpoints currently resident, least-recently-used first.</summary>
    public IReadOnlyList<LayaCheckpoint> Loaded
    {
        get { lock (_lock) return [.. _order]; }
    }

    // ── running ───────────────────────────────────────────────────────────────

    /// <summary>Route with no explicit override, then answer every question in one forward pass.</summary>
    public LayaResult Predict(object? state, QuestionSet questions) => Predict(state, questions, null, null, null);

    /// <summary>
    /// Route, then answer every question in one forward pass on the chosen checkpoint. The
    /// result is the checkpoint's usual answer set, plus <see cref="LayaResult.Routing"/>
    /// recording the decision.
    /// </summary>
    public LayaResult Predict(
        object? state, QuestionSet questions,
        string? model = null, string? task = null, string? lang = null)
        => Predict(state, questions, model, task, lang, null);

    /// <summary>
    /// <see cref="Predict(object?, QuestionSet, string?, string?, string?)"/> with Python's per-call
    /// <c>lang_guess</c> hint; see <see cref="Route(object?, QuestionSet?, string?, string?, string?, LangGuess?)"/>.
    /// </summary>
    public LayaResult Predict(
        object? state, QuestionSet questions,
        string? model, string? task, string? lang, LangGuess? langGuess)
        => PredictRouted(state, questions, model, task, lang, langGuess,
            predictor => predictor.Predict(state, questions));

    /// <summary>
    /// Routes as <see cref="Predict(object?, QuestionSet)"/> does (no model, task, language or hint overrides), with
    /// per-call options (a <see langword="null"/> <paramref name="options"/> gates nothing, so a literal <c>Predict(s, q, null)</c> still means what it meant); see <see cref="Predict(object?, QuestionSet, string?, string?, string?, LangGuess?, LayaPredictOptions?)"/>.
    /// </summary>
    public LayaResult Predict(object? state, QuestionSet questions, LayaPredictOptions? options)
        => Predict(state, questions, null, null, null, null, options);

    /// <summary>
    /// <see cref="Predict(object?, QuestionSet, string?, string?, string?, LangGuess?)"/> with per-call options.
    /// Ports the <c>min_confidence</c> argument of Python's <c>Router.predict</c>: the threshold is validated after the
    /// state and questions are checked and before anything is routed or loaded, and the gate is applied to the
    /// answers whichever checkpoint produced them. <c>PredictLong</c> takes none, as in Python.
    /// </summary>
    /// <exception cref="ArgumentException">The threshold is invalid, with Python's <c>ValueError</c> text.</exception>
    public LayaResult Predict(
        object? state, QuestionSet questions, string? model, string? task, string? lang, LangGuess? langGuess,
        LayaPredictOptions? options)
        => PredictRouted(state, questions, model, task, lang, langGuess,
            predictor => predictor.Predict(state, questions), options?.MinConfidence);

    LayaResult ILayaLongPredictor.PredictLong(
        object state, QuestionSet questions, int? window, int? stride, string aggregate, int? batchSize) =>
        PredictLong(state, questions, window, stride, aggregate, batchSize);

    /// <summary>
    /// Route using the usual model/task/lang hints, then scan the entire state. An attached predictor
    /// must implement ILayaLongPredictor; unsupported predictors are never silently truncated.
    /// </summary>
    public LayaResult PredictLong(
        object state, QuestionSet questions, int? window = null, int? stride = null,
        string aggregate = "auto", int? batchSize = null,
        string? model = null, string? task = null, string? lang = null)
        => PredictLong(state, questions, window, stride, aggregate, batchSize, model, task, lang, null);

    /// <summary>
    /// <see cref="PredictLong(object, QuestionSet, int?, int?, string, int?, string?, string?, string?)"/>
    /// with Python's per-call <c>lang_guess</c> hint; see
    /// <see cref="Route(object?, QuestionSet?, string?, string?, string?, LangGuess?)"/>.
    /// </summary>
    public LayaResult PredictLong(
        object state, QuestionSet questions, int? window, int? stride,
        string aggregate, int? batchSize,
        string? model, string? task, string? lang, LangGuess? langGuess)
    {
        StateGuard.NotNull(state);
        return PredictRouted(state, questions, model, task, lang, langGuess, predictor =>
            predictor is ILayaLongPredictor scanner
                ? scanner.PredictLong(state, questions, window, stride, aggregate, batchSize)
                : throw new NotSupportedException(
                    $"{predictor.GetType().Name} has no PredictLong, so a state longer than its window cannot be scanned"));
    }

    private LayaResult PredictRouted(
        object? state, QuestionSet questions, string? model, string? task, string? lang, LangGuess? langGuess,
        Func<ILayaPredictor, LayaResult> predict, MinConfidence? minConfidence = null)
    {
        StateGuard.NotNull(state);
        ArgumentNullException.ThrowIfNull(questions);
        // router.py:1338: validated after the state and questions, before routing or loading anything.
        var gate = minConfidence?.Check();
        var decision = Route(state, questions, model, task, lang, langGuess);
        // Inference runs outside the lock, exactly as Python's predict() calls
        // agent.system_one after load() has already returned and released the lock — so
        // concurrent predictions on a shared checkpoint are not serialised by the router.
        var result = WithLease(decision.Model, predict);
        return (gate is { } g ? ConfidenceGate.Apply(result, g) : result).WithRouting(decision);
    }

    // Loads (or finds) the checkpoint, holds a lease so eviction cannot dispose it mid-call, and
    // runs `run` on it outside the lock.
    private T WithLease<T>(LayaCheckpoint checkpoint, Func<ILayaPredictor, T> run)
    {
        Slot slot;
        lock (_lock)
        {
            ThrowIfDisposed();
            slot = FindOrBuild(checkpoint);
            slot.Leases++;
            Evict();
        }
        try
        {
            return run(slot.Predictor);
        }
        finally
        {
            lock (_lock)
            {
                slot.Leases--;
                if (slot.Retired && slot.Leases == 0 && slot.OwnsDisposal && slot.Predictor is IDisposable disposable)
                    disposable.Dispose();
            }
        }
    }

    // ── batching ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Route a heterogeneous request batch without loading any checkpoint. Ports Python
    /// <c>Router.route_batch</c> (router.py:1538): each request is validated and routed exactly as
    /// <see cref="Route(object?, QuestionSet?, string?, string?, string?, LangGuess?)"/> does with its own
    /// <c>Model</c>, <c>Task</c>, <c>Lang</c> and <c>LangGuess</c>, and the decisions keep input order.
    /// </summary>
    /// <exception cref="ArgumentException">A request is null, has a null state or null questions, or names an unknown model or task.</exception>
    public IReadOnlyList<RouteDecision> RouteBatch(IReadOnlyList<LayaBatchRequest> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        var decisions = new List<RouteDecision>(requests.Count);
        for (var i = 0; i < requests.Count; i++)
        {
            var request = requests[i]
                ?? throw new ArgumentException($"request {i} must be a dict, got NoneType");
            if (request.State is null)
                throw new ArgumentNullException(null, $"request {i} 'state' must not be None; pass a string, dict, or list");
            if (request.Questions is null)
                throw new ArgumentException($"request {i} 'questions' must be a dict, got NoneType");
            decisions.Add(Route(request.State, request.Questions, request.Model, request.Task,
                                request.Lang, request.LangGuess));
        }
        return decisions;
    }

    /// <summary>
    /// Route and answer a heterogeneous request batch with minimal checkpoint churn. Ports Python
    /// <c>Router.predict_batch</c> (router.py:1603): requests are routed first
    /// (<see cref="RouteBatch"/>), grouped by routed checkpoint in first-seen order, and within a
    /// checkpoint split again by question schema (order-sensitive, router.py:221) so states sharing
    /// a schema share forward passes via <see cref="ILayaBatchPredictor.PredictBatch"/>. Results are
    /// returned in input order, each carrying its <see cref="LayaResult.Routing"/>.
    /// </summary>
    /// <param name="requests">The requests; an empty list returns an empty list and loads nothing.</param>
    /// <param name="options">Passed to every checkpoint batch; <see cref="LayaBatchOptions.SortByLength"/>
    /// acts per question group, as in Python. <see cref="LayaBatchOptions.MinConfidence"/> is batch-wide: it is
    /// applied by the router to every result, is not passed to the checkpoints, and is not part of the grouping.</param>
    /// <exception cref="NotSupportedException">A routed checkpoint (an attached predictor) has no batch support.</exception>
    public IReadOnlyList<LayaResult> PredictBatch(
        IReadOnlyList<LayaBatchRequest> requests, LayaBatchOptions? options = null)
    {
        // router.py:1669: validated before any request is looked at.
        var gate = options?.MinConfidence?.Check();
        // router.py:1789 calls the agent without min_confidence and gates the results itself.
        var inner = options is { MinConfidence: not null }
            ? new LayaBatchOptions { BatchSize = options.BatchSize, SortByLength = options.SortByLength }
            : options;
        var decisions = RouteBatch(requests);
        if (decisions.Count == 0) return [];

        // First-seen order, so cache effects stay deterministic and an interleaved workload costs at
        // most one load per routed checkpoint (router.py:1674).
        var checkpoints = new List<LayaCheckpoint>();
        var byCheckpoint = new Dictionary<LayaCheckpoint, List<int>>();
        for (var i = 0; i < decisions.Count; i++)
        {
            if (!byCheckpoint.TryGetValue(decisions[i].Model, out var indices))
            {
                byCheckpoint[decisions[i].Model] = indices = [];
                checkpoints.Add(decisions[i].Model);
            }
            indices.Add(i);
        }

        var results = new LayaResult[requests.Count];
        foreach (var checkpoint in checkpoints)
        {
            var indices = byCheckpoint[checkpoint];
            WithLease(checkpoint, predictor =>
            {
                if (predictor is not ILayaBatchPredictor batcher)
                    throw new NotSupportedException(
                        $"{predictor.GetType().Name} has no PredictBatch, so requests cannot share forward passes");
                foreach (var (questions, group) in GroupBySchema(requests, indices))
                {
                    var batch = batcher.PredictBatch([.. group.Select(i => requests[i].State)], questions, inner);
                    if (batch.Count != group.Count)
                        throw new InvalidOperationException(
                            $"internal error: Agent.predict_batch returned {batch.Count} results for {group.Count} states");
                    for (var k = 0; k < batch.Count; k++)
                        results[group[k]] = (gate is { } g ? ConfidenceGate.Apply(batch[k], g) : batch[k])
                            .WithRouting(decisions[group[k]]);
                }
                return 0;
            });
        }
        return results;
    }

    // Splits one checkpoint's requests by question schema, first-seen order (router.py:1726-1776). A
    // later per-request token budget or lang joins the grouping key here.
    private static List<(QuestionSet Questions, List<int> Indices)> GroupBySchema(
        IReadOnlyList<LayaBatchRequest> requests, List<int> indices)
    {
        var groups = new List<(string Schema, QuestionSet Questions, List<int> Indices)>();
        QuestionSet? lastQuestions = null;
        string? lastSchema = null;
        foreach (var i in indices)
        {
            var questions = requests[i].Questions;
            string schema;
            if (ReferenceEquals(questions, lastQuestions)) schema = lastSchema!;
            else
            {
                schema = QuestionSchema(questions);
                lastQuestions = questions;
                lastSchema = schema;
            }
            var at = groups.FindIndex(g => g.Schema == schema);
            if (at < 0) groups.Add((schema, questions, [i]));
            else groups[at].Indices.Add(i);
        }
        return [.. groups.Select(g => (g.Questions, g.Indices))];
    }

    /// <summary>
    /// Order-sensitive signature of a question set, for sharing forward passes (router.py:221): ids, types,
    /// instructions, labels and rendered options all count, in declaration order.
    /// </summary>
    internal static string QuestionSchema(QuestionSet questions)
    {
        // json.dumps(questions, sort_keys=False, ensure_ascii=False, default=str): insertion order at
        // every level, ", " / ": " separators, Python float repr (so 1 and 1.0 differ). A question
        // carries no memory of how its dict was spelled, so each is written in canonical key order
        // (type, instructions, criteria) and a noul only lists the wordings it was given.
        var all = new List<KeyValuePair<string, object?>>();
        foreach (var (id, question) in questions)
        {
            var entry = new List<KeyValuePair<string, object?>>
            {
                new("type", SequenceBuilder.TypeName(question.Type)),
                new("instructions", question.Instructions),
            };
            switch (question)
            {
                case ChoiceQuestion choice:
                    entry.Add(new("criteria", choice.Options.ToList()));
                    break;
                case ScoreQuestion score:
                    entry.Add(new("criteria", score.Levels));
                    break;
                case NoulQuestion { IfFalse: null, IfTrue: null }:
                    break;
                case NoulQuestion noul:
                    var wording = new List<KeyValuePair<string, object?>>();
                    if (noul.IfFalse is not null) wording.Add(new("false", noul.IfFalse));
                    if (noul.IfTrue is not null) wording.Add(new("true", noul.IfTrue));
                    entry.Add(new("criteria", wording));
                    break;
            }
            all.Add(new(id, entry));
        }
        return PythonJson.Dumps(all, escapeNonAscii: false, fallbackToStr: true);
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var slot in _slots.Values)
            {
                if (slot.Leases == 0 && slot.OwnsDisposal && slot.Predictor is IDisposable disposable)
                    disposable.Dispose();
                else
                    slot.Retired = true; // an in-flight Predict disposes it on release
            }
            _slots.Clear();
            _order.Clear();
        }
    }
}
