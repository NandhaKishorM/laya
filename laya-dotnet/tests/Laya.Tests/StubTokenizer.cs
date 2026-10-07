using Laya.Tokenization;

namespace Laya.Tests;

/// <summary>
/// A deterministic tokenizer with no model behind it, for testing sequence structure.
/// </summary>
/// <remarks>
/// It emits one id per whitespace-separated word, so a sequence's shape — marker positions, the
/// order of head, options and state, where truncation lands — can be asserted by counting words.
/// It says nothing about real token ids; that is what the golden vectors in
/// <see cref="TokenizerParityTests"/> are for.
/// </remarks>
public sealed class StubTokenizer : ILayaTokenizer
{
    /// <summary>Ids start here, above every special id, so a word is never mistaken for one.</summary>
    public const int FirstWordId = 100;

    /// <inheritdoc/>
    public int PadId => 0;

    /// <inheritdoc/>
    public int SepId => 1;

    /// <inheritdoc/>
    public int ClsId => 2;

    /// <inheritdoc/>
    public int UnkId => 3;

    /// <inheritdoc/>
    public int MaskId => 4;

    /// <inheritdoc/>
    public string MaskToken => "<mask>";

    /// <summary>How many times <see cref="Encode"/> has been called.</summary>
    public int EncodeCalls { get; private set; }

    /// <summary>Every text passed to <see cref="Encode"/>, in order.</summary>
    public List<string> Encoded { get; } = [];

    /// <inheritdoc/>
    public int[] Encode(string text)
    {
        EncodeCalls++;
        Encoded.Add(text);
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        // A stable, collision-tolerant id per word: the test only needs determinism, and using the
        // hash keeps identical words mapping to identical ids across calls.
        return [.. words.Select(w => FirstWordId + (Math.Abs(w.GetHashCode(StringComparison.Ordinal)) % 10_000))];
    }
}
