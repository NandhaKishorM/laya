namespace Laya;

/// <summary>
/// Anything that can answer a <see cref="QuestionSet"/> about a state in one call: a single
/// <see cref="LayaEngine"/>, or a <see cref="LayaRouter"/> that picks one first. Mirrors Python's
/// duck-typed <c>predict</c>/<c>system_one</c> (<c>shortlist._call_predict</c>), so
/// <see cref="LayaShortlist.Predict"/> works over either.
/// </summary>
public interface ILayaPredictor
{
    /// <summary>Answer every question in <paramref name="questions"/> about <paramref name="state"/>.</summary>
    LayaResult Predict(object? state, QuestionSet questions);
}

/// <summary>
/// A predictor that also takes the per-call <see cref="LayaPredictOptions.Lang"/>, <see cref="LayaPredictOptions.MaxLen"/>
/// and <see cref="LayaPredictOptions.HeadMaxLen"/>: a <see cref="LayaEngine"/>. A <see cref="LayaRouter"/> forwards them
/// to the checkpoint it routed to; one that does not implement this is called without them (and refused when a token
/// budget was asked for, as Python refuses a keyword the agent does not accept).
/// </summary>
public interface ILayaCallPredictor : ILayaPredictor
{
    /// <summary>
    /// <see cref="ILayaPredictor.Predict"/> with the per-call language and token budgets of
    /// <paramref name="options"/>. <see cref="LayaPredictOptions.MinConfidence"/> is the caller's to apply.
    /// </summary>
    LayaResult Predict(object? state, QuestionSet questions, LayaPredictOptions? options);
}

/// <summary>
/// Whether a predictor carries per-language temperatures (Python <c>getattr(agent, "lang_temperatures", None)</c> being
/// non-empty). A <see cref="LayaRouter"/> only splits a batch by language for a checkpoint that does.
/// </summary>
public interface ILayaLangTemperatures
{
    /// <summary>Whether <see cref="LayaOptions.LangTemperatures"/> named at least one language.</summary>
    bool HasLangTemperatures { get; }
}
