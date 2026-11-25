namespace Laya;

/// <summary>Optional scan capability, mirroring Python Router's predict_long capability check.</summary>
public interface ILayaLongPredictor : ILayaPredictor
{
    /// <summary>Scan overlapping state-token windows and select one answer per question.</summary>
    LayaResult PredictLong(object state, QuestionSet questions, int? window = null, int? stride = null,
                           string aggregate = "auto", int? batchSize = null);
}

/// <summary>
/// Per-call knobs for <see cref="LayaEngine.PredictLong(object, QuestionSet, LayaLongOptions?)"/> and the router's
/// <c>PredictLong</c>: the arguments of <see cref="ILayaLongPredictor.PredictLong"/> plus <see cref="Lang"/>.
/// <c>max_len</c> and <c>head_max_len</c> are deliberately absent: Python's <c>predict_long</c> takes neither, the
/// windows are sized from <see cref="Window"/> and the checkpoint's own budget.
/// </summary>
public sealed class LayaLongOptions
{
    /// <summary>State tokens per window; nonpositive or <see langword="null"/> uses the checkpoint default.</summary>
    public int? Window { get; init; }

    /// <summary>Token step; nonpositive or <see langword="null"/> uses half the effective window.</summary>
    public int? Stride { get; init; }

    /// <summary>Only <c>"auto"</c> is supported.</summary>
    public string Aggregate { get; init; } = "auto";

    /// <summary>Maximum windows per forward pass; nonpositive or <see langword="null"/> uses Python's automatic cap.</summary>
    public int? BatchSize { get; init; }

    /// <summary>
    /// Language of the state (Python <c>lang</c>), applied to every window; see <see cref="LayaPredictOptions.Lang"/>.
    /// </summary>
    public string? Lang { get; init; }
}

/// <summary>
/// A scanner that takes the per-call <see cref="LayaLongOptions.Lang"/>: a <see cref="LayaEngine"/>. A
/// <see cref="LayaRouter"/> scanning with one that does not implement this runs the scan without the language and
/// warns, as Python does for a <c>predict_long</c> with no <c>lang</c> parameter.
/// </summary>
public interface ILayaLongLangPredictor : ILayaLongPredictor
{
    /// <summary><see cref="ILayaLongPredictor.PredictLong"/> with <paramref name="options"/>.</summary>
    LayaResult PredictLong(object state, QuestionSet questions, LayaLongOptions? options);
}
