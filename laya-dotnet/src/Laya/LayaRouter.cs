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
    /// How many checkpoints stay resident at once (least-recently-used is evicted). Default 1,
    /// matching Python. Raised automatically by <see cref="LayaRouter.Attach"/> and
    /// <see cref="LayaRouter.Preload"/> to fit what they load.
    /// </summary>
    public int MaxLoaded { get; set; } = 1;

    /// <summary>
    /// Checkpoint used when a state has no letters or its Latin language is undetermined without
    /// non-English evidence. Default
    /// <see cref="LayaCheckpoint.English"/>, matching Python's default.
    /// </summary>
    public LayaCheckpoint Default { get; set; } = LayaCheckpoint.English;

    /// <summary>
    /// Whether <see cref="LayaRouter.Route"/> may pick <see cref="LayaCheckpoint.TypedDecisions"/>
    /// on its own when the question ids match one of the four typed-decisions workflow
    /// signatures. Default <see langword="false"/>: that checkpoint is fine-tuned on specific
    /// synthetic workflows and must never be a silent default.
    /// </summary>
    public bool AutoTaskDetection { get; set; }

    /// <summary>Whether every checkpoint should be built immediately when the router is constructed.</summary>
    public bool Preload { get; set; }
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
public sealed class LayaRouter : ILayaPredictor, IDisposable
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
    /// Whether <see cref="Route"/> may pick <see cref="LayaCheckpoint.TypedDecisions"/> on its
    /// own when the question ids match a typed-decisions workflow signature.
    /// </summary>
    public bool AutoTaskDetection { get; }

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
            _ => throw new ArgumentException(
                $"unknown model '{name}'; choose one of english, multilingual, typed-decisions "
                + $"(or an alias: {string.Join(", ", Aliases.Keys.OrderBy(k => k, StringComparer.Ordinal))})",
                nameof(name)),
        };
    }

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
    /// <paramref name="lang"/> &gt; detected script/language &gt; <see cref="Default"/>.
    /// </remarks>
    public RouteDecision Route(
        object? state, QuestionSet? questions = null,
        string? model = null, string? task = null, string? lang = null)
    {
        if (model is not null)
            return new RouteDecision(NormaliseName(model), $"explicit model={PyRepr(model)}", null, null);

        if (task is not null)
        {
            var normalisedTaskSpelling = task.ToLowerInvariant().Replace("-", "_");
            var key = NormaliseName(normalisedTaskSpelling == "typed_decisions" ? "typed-decisions" : task);
            return new RouteDecision(key, $"explicit task={PyRepr(task)}", null, null);
        }

        var workflow = MatchTypedDecisionsWorkflow(questions);
        if (workflow is not null && AutoTaskDetection)
            return new RouteDecision(LayaCheckpoint.TypedDecisions,
                $"question ids match the {PyRepr(workflow)} typed-decisions workflow", null, workflow);

        if (lang is not null)
        {
            var primary = lang.ToLowerInvariant().Split('-')[0];
            var key = primary is "en" or "eng" or "english" ? LayaCheckpoint.English : LayaCheckpoint.Multilingual;
            return new RouteDecision(key, $"explicit lang={PyRepr(lang)}", null, workflow);
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
                ? $"Latin script, mostly English, but a line or field reads as {PyRepr(det.Language!)} "
                  + $"({PyRepr(LanguageDetection.SliceCodePoints(mixed, 60))}); the English checkpoint cannot read it"
                : det.Language is { } identified
                ? $"Latin script but language looks like {PyRepr(identified)}, not English"
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

    /// <summary>Python's <c>%r</c> for a <see cref="string"/>: a quoted, escaped repr.</summary>
    private static string PyRepr(string s)
    {
        var hasSingle = s.Contains('\'');
        var hasDouble = s.Contains('"');
        var quote = hasSingle && !hasDouble ? '"' : '\'';
        var sb = new System.Text.StringBuilder(s.Length + 2);
        sb.Append(quote);
        foreach (var rune in s.EnumerateRunes())
        {
            if (rune.Value == quote) { sb.Append('\\').Append(quote); continue; }
            switch (rune.Value)
            {
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (rune.Value < 0x20 || rune.Value == 0x7f)
                        sb.Append("\\x").Append(rune.Value.ToString("x2"));
                    else
                        sb.Append(rune.ToString());
                    break;
            }
        }
        sb.Append(quote);
        return sb.ToString();
    }

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
    {
        ArgumentNullException.ThrowIfNull(questions);
        var decision = Route(state, questions, model, task, lang);

        Slot slot;
        lock (_lock)
        {
            ThrowIfDisposed();
            slot = FindOrBuild(decision.Model);
            slot.Leases++;
            Evict();
        }
        try
        {
            // Inference runs outside the lock, exactly as Python's predict() calls
            // agent.system_one after load() has already returned and released the lock — so
            // concurrent predictions on a shared checkpoint are not serialised by the router.
            var result = slot.Predictor.Predict(state, questions);
            return result.WithRouting(decision);
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
