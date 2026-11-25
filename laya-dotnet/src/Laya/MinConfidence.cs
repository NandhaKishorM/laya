using System.Collections.ObjectModel;

namespace Laya;

/// <summary>
/// The three states of the abstention gate, spelled as Python's <c>GATE_STATES</c>
/// (<c>laya/confidence.py</c>). <see cref="Answer.Abstention"/> carries one of them whenever a
/// <see cref="MinConfidence"/> was set for the call, and is <see langword="null"/> otherwise: there is
/// deliberately no "no gate" member.
/// </summary>
public static class GateStates
{
    /// <summary>A gate ran and this answer's confidence cleared it.</summary>
    public const string Passed = "passed";

    /// <summary>A gate ran and this answer's confidence fell below it.</summary>
    public const string Abstained = "abstained";

    /// <summary>A gate ran and this answer carried no usable confidence.</summary>
    public const string Unevaluated = "unevaluated";

    /// <summary>Every state, in Python's order (<c>GATE_STATES</c>).</summary>
    public static IReadOnlyList<string> All { get; } = new ReadOnlyCollection<string>([Passed, Abstained, Unevaluated]);
}

/// <summary>
/// An abstention threshold: one number for every answer, or a map from option-count bucket to
/// number. Ports Python's <c>min_confidence</c> argument (<c>check_min_confidence</c> and
/// <c>check_min_confidence_map</c> in <c>laya/confidence.py</c>). An answer whose confidence falls below the
/// threshold is flagged (<see cref="Answer.LowConfidence"/>).
/// </summary>
/// <remarks>
/// Like Python, the value is validated when a prediction call receives it, not when it is built, so
/// a call that would fail for another reason first (a null state) still fails for that reason when
/// the order in Python says so. The failure is an <see cref="ArgumentException"/> carrying Python's
/// <c>ValueError</c> text.
/// </remarks>
public sealed class MinConfidence
{
    /// <summary>The map key for every bucket the map does not name.</summary>
    public const string DefaultBucket = "default";

    private static readonly string[] QuestionKinds = ["choice", "score", "noul"];
    private static readonly string[] Sizes = ["2", "3-5", "6-10", "11+"];

    /// <summary>
    /// Every option-count bucket a map can name: the answer shapes Python's <c>_option_bucket</c> produces
    /// (<c>choice:2</c>, <c>choice:3-5</c>, ... <c>noul:11+</c>). <see cref="DefaultBucket"/> is not among them.
    /// </summary>
    public static IReadOnlyList<string> BucketKeys { get; } = new ReadOnlyCollection<string>(
        [.. QuestionKinds.SelectMany(kind => Sizes.Select(size => $"{kind}:{size}"))]);

    private readonly double _scalar;
    private readonly List<KeyValuePair<string?, double>>? _map;

    private MinConfidence(double scalar, List<KeyValuePair<string?, double>>? map)
    {
        _scalar = scalar;
        _map = map;
    }

    /// <summary>One threshold for every answer. Must be a finite number in [0.0, 1.0] when the call validates it.</summary>
    public static MinConfidence Of(double threshold) => new(threshold, null);

    /// <summary>
    /// A threshold per option-count bucket (<see cref="BucketKeys"/>), plus an optional
    /// <see cref="DefaultBucket"/> for any bucket the map does not name; a bucket that is neither named
    /// nor covered by the default is gated at 0.0, that is, not at all. Insertion order is kept for error text.
    /// </summary>
    public static MinConfidence ForBuckets(IEnumerable<KeyValuePair<string, double>> thresholds)
    {
        ArgumentNullException.ThrowIfNull(thresholds);
        var map = new List<KeyValuePair<string?, double>>();
        foreach (var (key, value) in thresholds)
        {
            // A Python dict keeps one entry per key, at the key's first position, holding its last value.
            var at = map.FindIndex(e => string.Equals(e.Key, key, StringComparison.Ordinal));
            if (at < 0) map.Add(new(key, value));
            else map[at] = new(key, value);
        }
        return new(0.0, map);
    }

    /// <summary>Whether this is a per-bucket map rather than a single threshold.</summary>
    public bool IsMap => _map is not null;

    /// <summary>The validated thresholds, or Python's <c>ValueError</c> text as an <see cref="ArgumentException"/>.</summary>
    internal Resolved Check()
    {
        if (_map is null) return new Resolved(CheckOne(_scalar), null);
        if (_map.Count == 0)
            throw new ArgumentException("a min_confidence map must be a non-empty dict of bucket -> float, got {}");
        var allowed = new HashSet<string>(BucketKeys, StringComparer.Ordinal) { DefaultBucket };
        var checkedMap = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (key, value) in _map)
        {
            if (key is null || !allowed.Contains(key))
                throw new ArgumentException(
                    "min_confidence map keys must be a bucket like 'choice:3-5' or 'default', got "
                    + PythonFloat.Repr(key));
            checkedMap[key] = CheckOne(value);
        }
        return new Resolved(0.0, checkedMap);
    }

    private static double CheckOne(double value)
    {
        if (!double.IsFinite(value) || value < 0.0 || value > 1.0)
            throw new ArgumentException(
                "min_confidence must be a float in [0.0, 1.0], got " + PythonFloat.Repr(value));
        return value;
    }

    /// <inheritdoc/>
    public override string ToString() => _map is null
        ? PythonFloat.Repr(_scalar)
        : "{" + string.Join(", ", _map.Select(e => PythonFloat.Repr(e.Key) + ": " + PythonFloat.Repr(e.Value))) + "}";

    /// <summary>A validated threshold: the scalar, or the per-bucket map.</summary>
    internal readonly record struct Resolved(double Scalar, IReadOnlyDictionary<string, double>? Map);
}

/// <summary>
/// Per-call knobs for <see cref="LayaEngine.Predict(object?, QuestionSet, LayaPredictOptions?)"/> and
/// <see cref="LayaRouter"/>'s matching overload. A class with init-only properties so further per-call
/// options can be added without changing a signature. <c>PredictLong</c> takes none of these (it has
/// <see cref="LayaLongOptions"/>), as Python's <c>predict_long</c> takes no <c>min_confidence</c>, <c>max_len</c>
/// or <c>head_max_len</c>.
/// </summary>
public sealed class LayaPredictOptions
{
    /// <summary>
    /// Language of this state (Python <c>lang</c>): selects the entry of <see cref="LayaOptions.LangTemperatures"/>
    /// whose key is this code's part before the first <c>-</c>, lower-cased, in place of the checkpoint's own
    /// temperatures. <see langword="null"/>, an empty string, or a language the table does not hold changes nothing.
    /// On a <see cref="LayaRouter"/> use the <c>lang</c> argument instead (it also routes); setting this there is refused.
    /// </summary>
    public string? Lang { get; init; }

    /// <summary>
    /// Total sequence budget for this call, in place of the checkpoint's <c>max_len</c> (Python <c>max_len</c>);
    /// <see langword="null"/> keeps the checkpoint's. Python checks nothing, so neither does this: <c>0</c> or a small
    /// value leaves no room for the options' markers and the call is refused with the "option markers fit" error, and so
    /// is any negative value (Python's slice then counts from the end and no marker survives).
    /// </summary>
    public int? MaxLen { get; init; }

    /// <summary>
    /// Token budget for the question head (instructions plus options) for this call, in place of the checkpoint's
    /// <c>head_max_len</c> (Python <c>head_max_len</c>); <see langword="null"/> keeps the checkpoint's. Unchecked, as
    /// <see cref="MaxLen"/>: <c>0</c> and negative values squeeze each option to 4 tokens and the instruction to 8.
    /// </summary>
    public int? HeadMaxLen { get; init; }

    /// <summary>
    /// Opt-in abstention threshold (Python <c>min_confidence</c>). <see langword="null"/> writes nothing onto
    /// the answers; otherwise every answer reports <see cref="Answer.Abstention"/> and
    /// <see cref="Answer.AbstentionThreshold"/>, and <see cref="Answer.LowConfidence"/> is set on those below it.
    /// </summary>
    public MinConfidence? MinConfidence { get; init; }
}

/// <summary>
/// The abstention gate of Python's <c>laya/confidence.py</c> (<c>apply_confidence_gate</c>,
/// <c>flag_low_confidence</c>, <c>resolve_min_confidence</c>) and <c>jev_confidence</c>, over typed answers.
/// Answers are immutable, so applying a gate builds new answers instead of editing the old ones.
/// </summary>
internal static class ConfidenceGate
{
    /// <summary>The number the gate compares: <c>answer_confidence</c> first, then <c>confidence</c>, else none.</summary>
    internal static double? GateConfidence(Answer answer)
    {
        if (double.IsFinite(answer.AnswerConfidence)) return answer.AnswerConfidence;
        if (double.IsFinite(answer.Confidence)) return answer.Confidence;
        return null;
    }

    /// <summary>Python's <c>_option_bucket</c>.</summary>
    internal static string? OptionBucket(Answer answer)
    {
        var count = OptionCount(answer);
        if (count == 0 && answer.Type != QuestionType.Noul) return null;
        var k = count > 0 ? count : 2;
        var size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
        return $"{SequenceBuilder.TypeName(answer.Type)}:{size}";
    }

    private static int OptionCount(Answer answer) => answer switch
    {
        ChoiceAnswer choice => choice.Probabilities.Count,
        ScoreAnswer score => score.Probabilities.Count,
        _ => 0,
    };

    /// <summary>Python's <c>resolve_min_confidence</c>: the bucket's entry, else <c>default</c>, else 0.0.</summary>
    internal static double Resolve(Answer answer, IReadOnlyDictionary<string, double> thresholds)
    {
        var key = OptionBucket(answer);
        if (key is not null && thresholds.TryGetValue(key, out var own)) return own;
        return thresholds.TryGetValue(MinConfidence.DefaultBucket, out var fallback) ? fallback : 0.0;
    }

    /// <summary>One answer, gated: a copy carrying the abstention fields.</summary>
    internal static Answer Apply(Answer answer, MinConfidence.Resolved gate)
    {
        var threshold = gate.Map is null ? gate.Scalar : Resolve(answer, gate.Map);
        var confidence = GateConfidence(answer);
        var low = confidence is { } value && value < threshold;
        var state = low ? GateStates.Abstained : confidence is null ? GateStates.Unevaluated : GateStates.Passed;
        return answer.WithGate(low, state, threshold);
    }

    /// <summary>A result whose answers are gated; the original is untouched.</summary>
    internal static LayaResult Apply(LayaResult result, MinConfidence.Resolved gate)
    {
        var answers = new Dictionary<string, Answer>(result.Count, StringComparer.Ordinal);
        foreach (var id in result.Ids) answers[id] = Apply(result[id], gate);
        return result.WithAnswers(answers);
    }

    /// <summary>
    /// Python's <c>jev_confidence</c> over the probabilities of a choice or score answer, in option order
    /// (a score's by level index). Null for anything malformed: a non-finite or negative value.
    /// </summary>
    internal static double? Jev(bool isScore, IReadOnlyList<double> probabilities)
    {
        if (probabilities.Count == 0) return null;
        foreach (var v in probabilities)
            if (!double.IsFinite(v) || v < 0.0) return null;
        var n = probabilities.Count;
        if (n == 1) return 1.0;
        var total = PythonSum.Sum(probabilities);
        var p = new double[n];
        for (var i = 0; i < n; i++) p[i] = total != 0.0 ? probabilities[i] / total : 1.0 / n;
        if (!isScore)
        {
            var top = p.Max();
            return (top - 1.0 / n) / (1.0 - 1.0 / n);
        }
        var mode = 0;
        for (var i = 1; i < n; i++)
            if (p[i] > p[mode]) mode = i;
        var spread = PythonSum.Sum(p.Select((v, i) => v * Math.Abs(i - mode)));
        var center = (n - 1) / 2.0;
        var uniform = PythonSum.Sum(Enumerable.Range(0, n).Select(i => Math.Abs(i - center))) / n;
        return Math.Max(0.0, 1.0 - spread / uniform);
    }
}
