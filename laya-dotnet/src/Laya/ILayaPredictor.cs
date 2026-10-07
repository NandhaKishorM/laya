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
