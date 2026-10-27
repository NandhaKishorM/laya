namespace Laya;

/// <summary>A predictor that knows the pinned Hub revision it was loaded from.</summary>
internal interface IRevisioned
{
    string? Revision { get; }
}
