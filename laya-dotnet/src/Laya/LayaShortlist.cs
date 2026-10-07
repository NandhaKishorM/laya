namespace Laya;

/// <summary>
/// Embeds a batch of texts into vectors. Called once per shortlist, with the query text first,
/// followed by one text per option in criteria order.
/// </summary>
/// <remarks>
/// Mirrors Python's <c>embed_fn: Callable[[Sequence[str]], Any]</c>. Python's return value is
/// duck-typed (a NumPy array, a list of lists, or a torch tensor via <c>.detach()</c>); this port
/// asks for a plain <c>double[][]</c> instead, of shape <c>(texts.Count, dim)</c>, so a caller can
/// implement <see cref="LayaEmbedFunction"/> without taking a dependency on this SDK's own tensor
/// or NumPy-equivalent types — any bi-encoder wrapper that already returns row vectors can be
/// adapted with a one-line lambda. Non-finite components are zeroed by
/// <see cref="LayaShortlist"/> before use, matching Python's <c>np.nan_to_num</c>.
/// </remarks>
/// <param name="texts">The texts to embed, in order.</param>
/// <returns>One row per input text, all rows the same length.</returns>
public delegate double[][] LayaEmbedFunction(IReadOnlyList<string> texts);

/// <summary>
/// Shortlist metadata for one choice question: which labels were kept, their similarity scores
/// (when ranking actually ran), and the <c>k</c>/<c>n</c> it ran under. Mirrors Python's
/// <c>shortlist[qid]</c> entry.
/// </summary>
public sealed class ShortlistInfo
{
    internal ShortlistInfo(IReadOnlyList<string> labels, IReadOnlyList<double>? scores, int k, int n, bool passthrough)
    {
        Labels = labels;
        Scores = scores;
        K = k;
        N = n;
        Passthrough = passthrough;
    }

    /// <summary>The kept labels, in rank order (descending cosine similarity; ties keep the earlier label).</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>
    /// Cosine similarity per kept label, aligned with <see cref="Labels"/>, or
    /// <see langword="null"/> when nothing was dropped (<see cref="Passthrough"/>).
    /// </summary>
    public IReadOnlyList<double>? Scores { get; }

    /// <summary>The shortlist budget this question was ranked under.</summary>
    public int K { get; }

    /// <summary>How many labels the original criteria had.</summary>
    public int N { get; }

    /// <summary>
    /// Whether every label was kept without calling <see cref="LayaEmbedFunction"/> at all
    /// (<c>K &gt;= N</c>).
    /// </summary>
    public bool Passthrough { get; }
}

/// <summary>
/// Opt-in embedding shortlist for high-cardinality choice questions. Ports <c>laya/shortlist.py</c>.
/// </summary>
/// <remarks>
/// <para>
/// Choice options share one token budget, so a large label set leaves only a few tokens per
/// label. <see cref="Predict"/> embeds the state and each option with a caller-supplied
/// <see cref="LayaEmbedFunction"/>, keeps the top <c>k</c>, and runs a single
/// <see cref="ILayaPredictor.Predict"/> on that reduced criteria set.
/// </para>
/// <para>
/// This does not change the decision model's forward pass and does not add a second model call
/// for ranking: ranking is pure cosine similarity on whatever vectors the caller's embedder
/// returns. Porting Python's <c>embed_fn_from_agent</c> (mean-pooling the checkpoint's own
/// encoder) is out of scope for this SDK; pass a real bi-encoder, or the deterministic
/// hashing embedder used in this SDK's tests and samples for a dependency-free demo.
/// </para>
/// </remarks>
public static class LayaShortlist
{
    /// <summary>Default shortlist size, matching Python's <c>DEFAULT_SHORTLIST_K</c>.</summary>
    public const int DefaultShortlistK = 20;

    /// <summary>
    /// Return the top-<paramref name="k"/> choice labels for <paramref name="state"/>, ranked by
    /// cosine similarity between <paramref name="state"/> (plus optional
    /// <paramref name="instructions"/>) and each option's rendered text.
    /// </summary>
    /// <remarks>
    /// <see cref="LayaEmbedFunction"/> is called once, with the query text first and then one
    /// string per option in criteria order. When <paramref name="k"/> is at least the number of
    /// labels, every label is returned in its original order and the embedder is not called.
    /// Ties keep the earlier label; a zero vector scores 0 and does not outrank a label that came
    /// before it.
    /// </remarks>
    public static IReadOnlyList<string> ShortlistChoice(
        object? state,
        IEnumerable<KeyValuePair<string, object?>> criteria,
        LayaEmbedFunction embedFn,
        int k = DefaultShortlistK,
        object? instructions = null)
    {
        ArgumentNullException.ThrowIfNull(criteria);
        ArgumentNullException.ThrowIfNull(embedFn);
        var checkedK = CheckK(k);
        // "" is a valid placeholder: RenderOptions never reads a choice question's Instructions.
        var question = Question.Choice("", criteria);
        var (labels, _, _, _) = Rank(state, question, embedFn, checkedK, instructions);
        return labels;
    }

    /// <summary>
    /// <see cref="ShortlistChoice(object, IEnumerable{KeyValuePair{string, object}}, LayaEmbedFunction, int, object)"/>
    /// over bare labels, with no descriptions.
    /// </summary>
    public static IReadOnlyList<string> ShortlistChoice(
        object? state,
        IEnumerable<string> labels,
        LayaEmbedFunction embedFn,
        int k = DefaultShortlistK,
        object? instructions = null)
    {
        ArgumentNullException.ThrowIfNull(labels);
        return ShortlistChoice(
            state, labels.Select(l => new KeyValuePair<string, object?>(l, null)), embedFn, k, instructions);
    }

    /// <summary>
    /// Shortlist each choice question in <paramref name="questions"/>, then call
    /// <paramref name="predictor"/> once over the reduced set.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Non-choice questions are forwarded unchanged. A choice whose label count is <c>&lt;= k</c>
    /// is forwarded unchanged and does not call <paramref name="embedFn"/>. <paramref name="questions"/>
    /// itself is never mutated.
    /// </para>
    /// <para>
    /// The returned <see cref="LayaResult"/> is whatever <paramref name="predictor"/> returned,
    /// plus <see cref="LayaResult.Shortlist"/>. Probabilities on a shortlisted choice are over the
    /// kept labels only. When <paramref name="predictor"/> is a <see cref="LayaRouter"/>, the
    /// result carries both <see cref="LayaResult.Routing"/> and <see cref="LayaResult.Shortlist"/>.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// If a choice question in <paramref name="questions"/> is missing <see cref="ChoiceQuestion"/>
    /// options, or <paramref name="embedFn"/> returns a malformed matrix.
    /// </exception>
    public static LayaResult Predict(
        ILayaPredictor predictor,
        object? state,
        QuestionSet questions,
        LayaEmbedFunction embedFn,
        int k = DefaultShortlistK)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        ArgumentNullException.ThrowIfNull(questions);
        ArgumentNullException.ThrowIfNull(embedFn);
        var checkedK = CheckK(k);

        var reduced = new QuestionSet();
        var meta = new Dictionary<string, ShortlistInfo>(StringComparer.Ordinal);

        foreach (var (qid, question) in questions)
        {
            if (question is not ChoiceQuestion choice)
            {
                reduced[qid] = question;
                continue;
            }

            var (labels, scores, passthrough, n) = Rank(state, choice, embedFn, checkedK, choice.Instructions);
            meta[qid] = new ShortlistInfo(labels, scores, checkedK, n, passthrough);
            if (passthrough)
            {
                reduced[qid] = question;
                continue;
            }
            reduced[qid] = Question.Choice(choice.Instructions, SubsetOptions(choice.Options, labels));
        }

        var result = predictor.Predict(state, reduced);
        return result.WithShortlist(meta);
    }

    // ── ranking ───────────────────────────────────────────────────────────────

    private static (List<string> Labels, List<double>? Scores, bool Passthrough, int N) Rank(
        object? state, ChoiceQuestion question, LayaEmbedFunction embedFn, int k, object? instructions)
    {
        var keys = question.Options.Select(o => o.Key).ToList();
        var n = keys.Count;
        if (k >= n) return (keys, null, true, n);

        var query = QueryText(state, instructions);
        var optionTexts = SequenceBuilder.RenderOptions(question);
        var texts = new List<string>(n + 1) { query };
        texts.AddRange(optionTexts);

        var matrix = Embeddings(embedFn, texts);
        var docs = new double[n][];
        Array.Copy(matrix, 1, docs, 0, n);
        var sims = Cosine(matrix[0], docs);

        // Stable descending sort: ties keep the earlier label, matching NumPy's
        // np.argsort(-sims, kind="mergesort") — LINQ's OrderBy family is documented stable.
        var order = Enumerable.Range(0, n).OrderByDescending(i => sims[i]).Take(k).ToList();
        var labels = order.Select(i => keys[i]).ToList();
        var scores = order.Select(i => sims[i]).ToList();
        return (labels, scores, false, n);
    }

    private static IEnumerable<KeyValuePair<string, object?>> SubsetOptions(
        IReadOnlyList<KeyValuePair<string, object?>> options, IReadOnlyList<string> labels)
    {
        var byLabel = options.ToDictionary(o => o.Key, o => o.Value, StringComparer.Ordinal);
        foreach (var label in labels) yield return new(label, byLabel[label]);
    }

    /// <summary>
    /// Query text: <c>instructions + "\n" + state</c> when instructions is present and non-empty,
    /// else just the serialized state. A non-string instructions value is dumped via
    /// <see cref="PythonJson.State"/> (Python's <c>json.dumps(..., ensure_ascii=False)</c>, no
    /// <c>default=str</c> fallback) — the same dialect <c>PythonJson.State</c> already uses for a
    /// non-string state, so no separate dialect was needed for this.
    /// </summary>
    private static string QueryText(object? state, object? instructions)
    {
        var body = PythonJson.State(state);
        if (instructions is null) return body;
        if (instructions is string { Length: 0 }) return body;
        return PythonJson.State(instructions) + "\n" + body;
    }

    private static int CheckK(int k)
    {
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k), k, "k must be a positive integer");
        return k;
    }

    // ── embedding + cosine ───────────────────────────────────────────────────

    private static double[][] Embeddings(LayaEmbedFunction embedFn, IReadOnlyList<string> texts)
    {
        var raw = embedFn(texts)
            ?? throw new ArgumentException(
                $"embed_fn must return an array of shape ({texts.Count}, dim), got null", nameof(embedFn));
        if (raw.Length != texts.Count)
            throw new ArgumentException(
                $"embed_fn must return an array of shape ({texts.Count}, dim), got ({raw.Length}, ...)",
                nameof(embedFn));

        var dim = raw.Length > 0 ? raw[0]?.Length ?? -1 : -1;
        if (dim < 1)
            throw new ArgumentException(
                $"embed_fn must return an array of shape ({texts.Count}, dim) with dim >= 1", nameof(embedFn));

        var cleaned = new double[raw.Length][];
        for (var i = 0; i < raw.Length; i++)
        {
            var row = raw[i]
                ?? throw new ArgumentException($"embed_fn returned a null row at index {i}", nameof(embedFn));
            if (row.Length != dim)
                throw new ArgumentException(
                    $"embed_fn must return an array of shape ({texts.Count}, dim); row {i} has length "
                    + $"{row.Length}, expected {dim}", nameof(embedFn));

            var outRow = new double[dim];
            for (var j = 0; j < dim; j++)
            {
                var v = row[j];
                // np.nan_to_num: only non-finite COMPONENTS are zeroed, not the whole vector.
                outRow[j] = double.IsNaN(v) || double.IsInfinity(v) ? 0.0 : v;
            }
            cleaned[i] = outRow;
        }
        return cleaned;
    }

    private static double[] Cosine(double[] query, double[][] docs)
    {
        var sims = new double[docs.Length];
        var qn = Norm(query);
        if (qn == 0.0) return sims;

        for (var i = 0; i < docs.Length; i++)
        {
            var denom = Norm(docs[i]) * qn;
            if (denom > 0.0) sims[i] = Dot(docs[i], query) / denom;
        }
        return sims;
    }

    private static double Norm(double[] v)
    {
        var sum = 0.0;
        foreach (var x in v) sum += x * x;
        return Math.Sqrt(sum);
    }

    private static double Dot(double[] a, double[] b)
    {
        var sum = 0.0;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }
}
