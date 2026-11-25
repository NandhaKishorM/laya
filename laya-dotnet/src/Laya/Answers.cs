namespace Laya;

/// <summary>Token accounting for one <c>Predict</c> call. Laya emits no tokens, so output is 0.</summary>
/// <param name="InputTokens">Non-padding tokens across every question in the batch.</param>
/// <param name="OutputTokens">Always 0: a System 1 pass generates nothing.</param>
public readonly record struct Usage(int InputTokens, int OutputTokens)
{
    /// <summary>Windows scored by PredictLong; null for a plain Predict call.</summary>
    public int? Windows { get; init; }

    /// <summary>State tokens read across scan windows, before per-question truncation.</summary>
    public int? StateTokens { get; init; }

    /// <summary>Sum of each window's worst per-question dropped-state count.</summary>
    public int? StateTokensDropped { get; init; }

    /// <summary>Number of scan windows with any truncated question (0 or 1 for a single window).</summary>
    public int? Truncated { get; init; }

    /// <summary>The last window's truncated question ids, matching Python's scan aggregation.</summary>
    public IReadOnlyList<string>? TruncatedQuestions { get; init; }

    /// <summary>Collapsed option spans, merged by question id across scan windows; null if none.</summary>
    public IReadOnlyDictionary<string, OptionUsage>? Options { get; init; }

    /// <summary>Value equality, including the contents of <see cref="TruncatedQuestions"/> and <see cref="Options"/>.</summary>
    public bool Equals(Usage other) =>
        InputTokens == other.InputTokens && OutputTokens == other.OutputTokens
        && Windows == other.Windows && StateTokens == other.StateTokens
        && StateTokensDropped == other.StateTokensDropped && Truncated == other.Truncated
        && SameList(TruncatedQuestions, other.TruncatedQuestions) && SameMap(Options, other.Options);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(InputTokens);
        hash.Add(OutputTokens);
        hash.Add(Windows);
        hash.Add(StateTokens);
        hash.Add(StateTokensDropped);
        hash.Add(Truncated);
        if (TruncatedQuestions is { } ids)
            foreach (var id in ids) hash.Add(id, StringComparer.Ordinal);
        // Order-independent, so equal maps hash equally whatever their enumeration order.
        if (Options is { } options)
            hash.Add(options.Aggregate(options.Count,
                (acc, pair) => acc ^ HashCode.Combine(StringComparer.Ordinal.GetHashCode(pair.Key), pair.Value)));
        return hash.ToHashCode();
    }

    private static bool SameList(IReadOnlyList<string>? a, IReadOnlyList<string>? b) =>
        ReferenceEquals(a, b) || (a is not null && b is not null && a.SequenceEqual(b, StringComparer.Ordinal));

    // Keys compare ordinally whatever comparer either map was built with, so equality stays
    // symmetric and agrees with GetHashCode's ordinal key hashes.
    private static bool SameMap(IReadOnlyDictionary<string, OptionUsage>? a, IReadOnlyDictionary<string, OptionUsage>? b) =>
        ReferenceEquals(a, b) || (a is not null && b is not null && a.Count == b.Count
            && a.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .Zip(b.OrderBy(pair => pair.Key, StringComparer.Ordinal))
                .All(pairs => string.Equals(pairs.First.Key, pairs.Second.Key, StringComparison.Ordinal)
                    && pairs.First.Value == pairs.Second.Value));
}

/// <summary>Option-span diagnostics from Python's common.collapsed_options.</summary>
/// <param name="Total">Number of declared options.</param>
/// <param name="Distinct">Number of distinct capped token spans.</param>
/// <param name="TokensPerOption">Cap including the marker, or null when no head-budget cap was needed.</param>
public readonly record struct OptionUsage(int Total, int Distinct, int? TokensPerOption);

/// <summary>The deciding window in the tokenized state; offsets are exclusive-end token indices.</summary>
/// <param name="Index">Zero-based deciding window index.</param>
/// <param name="TokenStart">First state token in the window.</param>
/// <param name="TokenEnd">One past the last state token in the window.</param>
/// <param name="Count">Total windows scored.</param>
public readonly record struct WindowInfo(int Index, int TokenStart, int TokenEnd, int Count);

/// <summary>
/// The action head's read on whether the answer can be acted on directly.
/// </summary>
/// <param name="ActProbability">
/// Probability of acting rather than escalating. Corresponds to Python's
/// <c>action.act_probability</c>: softmax over the action logits, first component.
/// </param>
public readonly record struct ActionInfo(double ActProbability);

/// <summary>One answer. Cast with <see cref="AsChoice"/>, <see cref="AsScore"/> or <see cref="AsNoul"/>.</summary>
public abstract class Answer
{
    internal Answer(QuestionType type, double confidence, double answerConfidence, ActionInfo action)
    {
        Type = type;
        Confidence = confidence;
        AnswerConfidence = answerConfidence;
        Action = action;
    }

    /// <summary>Which question type produced this answer.</summary>
    public QuestionType Type { get; }

    /// <summary>
    /// Calibrated confidence in [0, 1]. For choice and score this is normalized Shannon entropy
    /// (<c>1 - H(p)/log k</c>); for noul it is <c>max(p, 1 - p)</c>. Rounded to 4 decimals, as Python does.
    /// </summary>
    public double Confidence { get; }

    /// <summary>
    /// The calibrated <c>max(p)</c> confidence, in [0, 1], added alongside <see cref="Confidence"/>
    /// in Python 0.3.21 (<c>answer_confidence</c> in <c>agent.py</c>'s <c>_decode_answers</c>,
    /// computed by the new <c>laya/confidence.py</c>-adjacent <c>common.answer_confidence</c>).
    /// It means the same thing on every question type — the probability mass on the reported
    /// answer — unlike <see cref="Confidence"/>, whose formula differs between noul and the other
    /// two types, so it is the field to gate abstention on across mixed question types. Rounded to
    /// 4 decimals, as Python does. <see cref="Confidence"/> itself is unchanged.
    /// </summary>
    public double AnswerConfidence { get; }

    /// <summary>The action head's output.</summary>
    public ActionInfo Action { get; }

    /// <summary>
    /// Deciding scan window, or null for Predict and the single-window PredictLong fast path.
    /// Probabilities describe this window, not a calibrated whole-document probability.
    /// </summary>
    public WindowInfo? Window { get; private set; }

    internal Answer WithWindow(WindowInfo window)
    {
        var answer = (Answer)MemberwiseClone();
        answer.Window = window;
        return answer;
    }

    /// <summary>
    /// True only when a <see cref="MinConfidence"/> was set for the call and this answer's confidence fell
    /// below its threshold (Python's <c>low_confidence: True</c>, which is absent otherwise). False for an
    /// ungated call and for an answer that cleared the gate or could not be evaluated.
    /// </summary>
    public bool LowConfidence { get; private set; }

    /// <summary>
    /// What the abstention gate decided, one of <see cref="GateStates"/> (<c>passed</c>, <c>abstained</c>,
    /// <c>unevaluated</c>), or <see langword="null"/> when the call set no <see cref="MinConfidence"/>
    /// (Python writes no <c>abstention</c> key then).
    /// </summary>
    public string? Abstention { get; private set; }

    /// <summary>
    /// The threshold this answer was gated at: the scalar, or the map entry for its option-count bucket
    /// (else the map's <c>default</c>, else 0.0). <see langword="null"/> when the call set no
    /// <see cref="MinConfidence"/> (Python's <c>abstention_threshold</c>).
    /// </summary>
    public double? AbstentionThreshold { get; private set; }

    /// <summary>
    /// The <c>confidence</c> Jev would report for this answer's (rounded) probabilities, which is a different
    /// formula from <see cref="Confidence"/> (Python's <c>jev_confidence</c>): for choice
    /// <c>(max(p) - 1/n) / (1 - 1/n)</c>, for score <c>max(0, 1 - sum(p_i * |i - mode|) / D)</c>, both 1.0 for a
    /// single option. <see langword="null"/> for a noul answer and for malformed probabilities (non-finite or negative).
    /// </summary>
    public double? JevConfidence() => this switch
    {
        ChoiceAnswer choice => ConfidenceGate.Jev(false, [.. choice.Probabilities.Select(kv => kv.Value)]),
        ScoreAnswer score => ConfidenceGate.Jev(true, score.Probabilities),
        _ => null,
    };

    /// <summary>A copy carrying the gate's verdict; the original stays as it was, so a reused answer never keeps a stale flag.</summary>
    internal Answer WithGate(bool lowConfidence, string abstention, double threshold)
    {
        var answer = (Answer)MemberwiseClone();
        answer.LowConfidence = lowConfidence;
        answer.Abstention = abstention;
        answer.AbstentionThreshold = threshold;
        return answer;
    }

    /// <summary>This answer as a choice answer.</summary>
    /// <exception cref="InvalidOperationException">If the question was not a choice question.</exception>
    public ChoiceAnswer AsChoice() => this as ChoiceAnswer
        ?? throw new InvalidOperationException($"answer is {Type}, not Choice");

    /// <summary>This answer as a score answer.</summary>
    /// <exception cref="InvalidOperationException">If the question was not a score question.</exception>
    public ScoreAnswer AsScore() => this as ScoreAnswer
        ?? throw new InvalidOperationException($"answer is {Type}, not Score");

    /// <summary>This answer as a noul answer.</summary>
    /// <exception cref="InvalidOperationException">If the question was not a noul question.</exception>
    public NoulAnswer AsNoul() => this as NoulAnswer
        ?? throw new InvalidOperationException($"answer is {Type}, not Noul");
}

/// <summary>The answer to a <see cref="QuestionType.Choice"/> question.</summary>
public sealed class ChoiceAnswer : Answer
{
    internal ChoiceAnswer(string choice, IReadOnlyList<KeyValuePair<string, double>> probabilities,
                          double confidence, double answerConfidence, ActionInfo action)
        : base(QuestionType.Choice, confidence, answerConfidence, action)
    {
        Choice = choice;
        Probabilities = probabilities;
    }

    /// <summary>The highest-probability label.</summary>
    public string Choice { get; }

    /// <summary>Probability per label, in the order the options were declared. Rounded to 4 decimals.</summary>
    public IReadOnlyList<KeyValuePair<string, double>> Probabilities { get; }

    /// <summary>The probability assigned to one label.</summary>
    /// <exception cref="KeyNotFoundException">If no option carries that label.</exception>
    public double this[string label]
    {
        get
        {
            foreach (var kv in Probabilities)
                if (string.Equals(kv.Key, label, StringComparison.Ordinal)) return kv.Value;
            throw new KeyNotFoundException($"no option labelled '{label}'");
        }
    }

    /// <inheritdoc/>
    public override string ToString() => $"{Choice} ({Confidence:0.###} confidence)";
}

/// <summary>The answer to a <see cref="QuestionType.Score"/> question.</summary>
public sealed class ScoreAnswer : Answer
{
    internal ScoreAnswer(double score, IReadOnlyList<object?> legend, IReadOnlyList<double> probabilities,
                         double confidence, double answerConfidence, ActionInfo action)
        : base(QuestionType.Score, confidence, answerConfidence, action)
    {
        Score = score;
        Legend = legend;
        Probabilities = probabilities;
    }

    /// <summary>
    /// Expected level: the probability-weighted mean of the level indices, so a fractional value
    /// between 0 and <c>Legend.Count - 1</c>. Rounded to 4 decimals.
    /// </summary>
    public double Score { get; }

    /// <summary>Rendered text for each level index, including JSON text for structured criteria.</summary>
    public IReadOnlyList<object?> Legend { get; }

    /// <summary>Probability per level index. Rounded to 4 decimals.</summary>
    public IReadOnlyList<double> Probabilities { get; }

    /// <summary>The single most likely level index, as opposed to the expected value.</summary>
    public int MostLikelyLevel
    {
        get
        {
            var best = 0;
            for (var i = 1; i < Probabilities.Count; i++)
                if (Probabilities[i] > Probabilities[best]) best = i;
            return best;
        }
    }

    /// <inheritdoc/>
    public override string ToString() => $"{Score:0.####} of 0..{Legend.Count - 1}";
}

/// <summary>The answer to a <see cref="QuestionType.Noul"/> question.</summary>
public sealed class NoulAnswer : Answer
{
    internal NoulAnswer(double probability, double confidence, double answerConfidence, ActionInfo action)
        : base(QuestionType.Noul, confidence, answerConfidence, action) => Probability = probability;

    /// <summary>Probability that the statement holds, in [0, 1]. Rounded to 4 decimals.</summary>
    public double Probability { get; }

    /// <summary>Whether the statement more likely holds than not.</summary>
    public bool Value => Probability >= 0.5;

    /// <inheritdoc/>
    public override string ToString() => $"{Value} (p={Probability:0.####})";
}

/// <summary>The result of one <c>Predict</c> call: one answer per question, plus token usage.</summary>
public sealed class LayaResult
{
    private readonly Dictionary<string, Answer> _byId;
    private readonly List<string> _order;

    internal LayaResult(string model, List<string> order, Dictionary<string, Answer> byId, Usage usage,
                        RouteDecision? routing = null, IReadOnlyDictionary<string, ShortlistInfo>? shortlist = null)
    {
        Model = model;
        _order = order;
        _byId = byId;
        Usage = usage;
        Routing = routing;
        Shortlist = shortlist;
    }

    /// <summary>The model identifier, matching Python's <c>"laya-rl-agent"</c>.</summary>
    public string Model { get; }

    /// <summary>Token accounting for the call.</summary>
    public Usage Usage { get; }

    /// <summary>
    /// The routing decision that picked <see cref="Model"/>, when this result came from
    /// <see cref="LayaRouter.Predict(object, QuestionSet)"/>. <see langword="null"/> for a
    /// result answered directly by a <see cref="LayaEngine"/>. Mirrors Python's <c>result["routing"]</c>.
    /// </summary>
    public RouteDecision? Routing { get; }

    /// <summary>
    /// Per-question shortlist metadata, when this result came from
    /// <see cref="LayaShortlist.Predict(ILayaPredictor, object, QuestionSet, LayaEmbedFunction, int)"/>.
    /// <see langword="null"/> otherwise. Mirrors Python's <c>result["shortlist"]</c>. A result can
    /// carry both <see cref="Routing"/> and <see cref="Shortlist"/> when a shortlist ran over a
    /// <see cref="LayaRouter"/>.
    /// </summary>
    public IReadOnlyDictionary<string, ShortlistInfo>? Shortlist { get; }

    /// <summary>A copy of this result with <see cref="Routing"/> set. Used internally by <see cref="LayaRouter"/>.</summary>
    internal LayaResult WithRouting(RouteDecision routing) => new(Model, _order, _byId, Usage, routing, Shortlist);

    /// <summary>A copy of this result with <see cref="Shortlist"/> set. Used internally by <see cref="LayaShortlist"/>.</summary>
    internal LayaResult WithShortlist(IReadOnlyDictionary<string, ShortlistInfo> shortlist) =>
        new(Model, _order, _byId, Usage, Routing, shortlist);

    /// <summary>A copy of this result with its answers replaced (same ids). Used internally by the abstention gate.</summary>
    internal LayaResult WithAnswers(Dictionary<string, Answer> answers) =>
        new(Model, _order, answers, Usage, Routing, Shortlist);

    /// <summary>The question ids, in the order they were asked.</summary>
    public IReadOnlyList<string> Ids => _order;

    /// <summary>How many answers there are.</summary>
    public int Count => _order.Count;

    /// <summary>The answer to one question.</summary>
    /// <exception cref="KeyNotFoundException">If no question carried that id.</exception>
    public Answer this[string id] => _byId.TryGetValue(id, out var a)
        ? a
        : throw new KeyNotFoundException($"no answer for question '{id}'");

    /// <summary>Try to get the answer to one question.</summary>
    public bool TryGetAnswer(string id, out Answer? answer) => _byId.TryGetValue(id, out answer);

    /// <summary>Every answer, in the order the questions were asked.</summary>
    public IEnumerable<KeyValuePair<string, Answer>> Answers
    {
        get
        {
            foreach (var id in _order) yield return new(id, _byId[id]);
        }
    }
}
