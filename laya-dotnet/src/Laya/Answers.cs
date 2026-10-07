namespace Laya;

/// <summary>Token accounting for one <c>Predict</c> call. Laya emits no tokens, so output is 0.</summary>
/// <param name="InputTokens">Non-padding tokens across every question in the batch.</param>
/// <param name="OutputTokens">Always 0: a System 1 pass generates nothing.</param>
public readonly record struct Usage(int InputTokens, int OutputTokens);

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
