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
/// Tournament metadata for one choice question: the finalists that reached the final call and how wide the field
/// was. Mirrors Python's <c>tournament[qid]</c> entry.
/// </summary>
public sealed class TournamentInfo
{
    internal TournamentInfo(IReadOnlyList<string> labels, int n, int rounds)
    {
        Labels = labels;
        N = n;
        Rounds = rounds;
    }

    /// <summary>The finalists, in criteria order (every label when no round was needed).</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>
    /// How many labels the caller's question had before the first round: how wide the field was, not how many
    /// finalists came out of it, so on a choice that played a round it is larger than <c>Labels.Count</c>.
    /// </summary>
    public int N { get; }

    /// <summary>How many elimination rounds this question played (0 when it needed none).</summary>
    public int Rounds { get; }
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

    /// <summary>Default tournament group size, matching Python's <c>DEFAULT_TOURNAMENT_GROUP</c>.</summary>
    public const int DefaultTournamentGroup = 16;

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
            var narrowed = Question.Choice(choice.Instructions, SubsetOptions(choice.Options, labels));
            // shortlist.py:425-448: an order over the caller's label set no longer fits the kept labels, so it
            // is dropped (not remapped) with one RuntimeWarning naming the question.
            if (choice.OptionOrder is { } order)
                WarnOnce.Warn(
                    $"laya.shortlist: question {PythonStr.Repr(qid)}: 'option_order' ({order.Count} slots) was dropped because the label "
                    + $"set was narrowed to {labels.Count}; the order no longer fits the kept options and remapping "
                    + "it onto them would move options between slots.");
            reduced[qid] = narrowed;
        }

        var result = predictor.Predict(state, reduced);
        return result.WithShortlist(meta);
    }

    /// <summary>
    /// Narrow each large choice question by elimination, then call <paramref name="predictor"/> once more.
    /// Ports Python's <c>predict_tournament</c> (<c>shortlist.py:139-227</c>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A choice with more than <paramref name="groupSize"/> labels is cut, in criteria order, into groups of
    /// near-equal size and at most <paramref name="groupSize"/>. One <c>Predict</c> call answers every group of
    /// every such question (the groups go in as separate questions with the ids <c>"0"</c>, <c>"1"</c>, ...,
    /// so they share one forward pass) and each group's answer goes through to the next round. Rounds repeat
    /// until no choice has more than <paramref name="groupSize"/> labels left. Unlike
    /// <see cref="Predict"/> this needs no embedder. The final call answers every question of the request, with
    /// each choice that went through a round cut to its finalists; a request that needs no round is the only
    /// call. <paramref name="questions"/> is not mutated.
    /// </para>
    /// <para>
    /// A choice that went through a round loses an <see cref="Question.OptionOrder"/> it carried, with one
    /// warning naming the question (once per question per call, and once per process for identical text, as
    /// Python's default warnings filter shows it). A choice that needed no round keeps its order.
    /// </para>
    /// <para>
    /// The result is the final call's result (so its probabilities and <see cref="LayaResult.Usage"/> are the
    /// final call's alone) plus <see cref="LayaResult.Tournament"/>, one entry per choice question.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// <paramref name="groupSize"/> is below 2, with Python's <c>ValueError</c> text.
    /// </exception>
    /// <exception cref="KeyNotFoundException">
    /// A round's reply lacks the answer of a group, or the answer is not a choice, or a winner is not a label of
    /// its question: Python's <c>KeyError</c> text (the quoted key).
    /// </exception>
    public static LayaResult PredictTournament(
        ILayaPredictor predictor,
        object? state,
        QuestionSet questions,
        int groupSize = DefaultTournamentGroup)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        return PredictTournament(predictor.Predict, state, questions, groupSize);
    }

    /// <summary>
    /// <see cref="PredictTournament(ILayaPredictor, object, QuestionSet, int)"/> over a predict function, so a
    /// caller can forward per-call options to every round and to the final call, as Python's
    /// <c>**predict_kwargs</c> does: <c>(s, q) =&gt; engine.Predict(s, q, options)</c>.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// <paramref name="predict"/> returned <see langword="null"/>, with Python's <c>TypeError</c> text for a
    /// reply that is not a dict.
    /// </exception>
    public static LayaResult PredictTournament(
        Func<object?, QuestionSet, LayaResult> predict,
        object? state,
        QuestionSet questions,
        int groupSize = DefaultTournamentGroup)
    {
        ArgumentNullException.ThrowIfNull(predict);
        ArgumentNullException.ThrowIfNull(questions);
        if (groupSize < 2)
            throw new ArgumentException(
                $"group_size must be an integer of at least 2, got {groupSize.ToString(System.Globalization.CultureInfo.InvariantCulture)}");

        var entries = new List<TournamentEntry>();
        foreach (var (qid, question) in questions)
        {
            if (question is not ChoiceQuestion choice) continue;
            var labels = choice.Options.Select(o => o.Key).ToList();
            entries.Add(new TournamentEntry(qid, labels, labels.Count));
        }
        var warned = new HashSet<string>(StringComparer.Ordinal);

        while (true)
        {
            var groups = new List<(TournamentEntry Entry, List<string> Labels)>();
            foreach (var entry in entries)
            {
                var n = entry.Labels.Count;
                var parts = (int)(((long)n + groupSize - 1) / groupSize);     // -(-n // g) for positive n and g
                if (parts <= 1) continue;
                for (var i = 0; i < parts; i++)
                {
                    var lo = (int)((long)i * n / parts);
                    var hi = (int)((long)(i + 1) * n / parts);
                    groups.Add((entry, entry.Labels.GetRange(lo, hi - lo)));
                }
            }
            if (groups.Count == 0) break;

            var round = new QuestionSet();
            for (var i = 0; i < groups.Count; i++)
                round[GroupId(i)] = CutQuestion(questions, groups[i].Entry.Qid, groups[i].Labels, warned);
            var reply = CallPredict(predict, state, round);

            var winners = new Dictionary<TournamentEntry, List<string>>();
            for (var i = 0; i < groups.Count; i++)
            {
                if (!reply.TryGetAnswer(GroupId(i), out var answer) || answer is null)
                    throw new KeyNotFoundException(PythonStr.Repr(GroupId(i)));
                if (answer is not ChoiceAnswer picked)
                    throw new KeyNotFoundException(PythonStr.Repr("choice"));
                if (!winners.TryGetValue(groups[i].Entry, out var list)) winners[groups[i].Entry] = list = [];
                list.Add(picked.Choice);
            }
            foreach (var (entry, labels) in winners)
            {
                entry.Labels = labels;
                entry.Rounds++;
            }
        }

        var final = new QuestionSet(questions);
        foreach (var entry in entries)
            if (entry.Rounds > 0)
                final[entry.Qid] = CutQuestion(questions, entry.Qid, entry.Labels, warned);
        var result = CallPredict(predict, state, final);

        var meta = new Dictionary<string, TournamentInfo>(StringComparer.Ordinal);
        foreach (var entry in entries)
            meta[entry.Qid] = new TournamentInfo(entry.Labels, entry.N, entry.Rounds);
        return result.WithTournament(meta);
    }

    private sealed class TournamentEntry(string qid, List<string> labels, int n)
    {
        public string Qid { get; } = qid;
        public List<string> Labels { get; set; } = labels;
        public int N { get; } = n;
        public int Rounds { get; set; }
    }

    private static string GroupId(int i) => i.ToString(System.Globalization.CultureInfo.InvariantCulture);

    private static LayaResult CallPredict(Func<object?, QuestionSet, LayaResult> predict, object? state, QuestionSet questions) =>
        predict(state, questions)
        ?? throw new InvalidOperationException("predict/system_one must return a dict, got NoneType");

    /// <summary>
    /// Python's <c>cut</c> in <c>predict_tournament</c>: the question narrowed to <paramref name="labels"/>
    /// without an <c>option_order</c> that no longer fits it (<c>_drop_stale_order</c>, shortlist.py:425-448).
    /// The one warning site of the tournament, as Python's <c>stacklevel=2</c> attributes every such warning to
    /// the <c>cut</c> line, so it must not be inlined into either of its callers.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static ChoiceQuestion CutQuestion(QuestionSet questions, string qid, IReadOnlyList<string> labels, HashSet<string> warned)
    {
        var choice = (ChoiceQuestion)questions[qid];
        var byLabel = choice.Options.ToDictionary(o => o.Key, o => o.Value, StringComparer.Ordinal);
        var kept = new List<KeyValuePair<string, object?>>(labels.Count);
        foreach (var label in labels)
        {
            // Python's _subset_criteria indexes the criteria dict: an unknown winner is a KeyError(label).
            if (!byLabel.TryGetValue(label, out var description))
                throw new KeyNotFoundException(PythonStr.Repr(label));
            kept.Add(new(label, description));
        }
        var narrowed = Question.Choice(choice.Instructions, kept);
        if (choice.OptionOrder is { } order && warned.Add(qid))
            WarnOnce.Warn(
                $"laya.shortlist: question {PythonStr.Repr(qid)}: 'option_order' ({order.Count} slots) was dropped because the label "
                + $"set was narrowed to {labels.Count}; the order no longer fits the kept options and remapping "
                + "it onto them would move options between slots.");
        return narrowed;
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
            if (denom > 0.0)
            {
                // Python clips (`np.clip(dot / denom, -1.0, 1.0)`, laya/shortlist.py::_Cosine) and
                // laya-go ports that with TestCosineIsClipped. The quotient rounds outside [-1, 1]
                // for vectors that are parallel but not exactly representable -- a self-cosine of
                // [34, 15.285714285714286, 70.85714285714286] is 1.0000000000000002 -- so an
                // unclipped value breaks the range ShortlistInfo.Scores documents. Math.Clamp
                // returns NaN unchanged for a NaN input, as the Go port also leaves it alone.
                sims[i] = Math.Clamp(Dot(docs[i], query) / denom, -1.0, 1.0);
            }
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
