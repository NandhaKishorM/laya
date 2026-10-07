using System.Text;

namespace Laya.Routing.Sample;

/// <summary>
/// Deterministic, dependency-free demo embedder — <b>demo only, replace with a real embedding
/// model</b>. It is a straight copy of <c>laya-dotnet/tools/hashing_embedder.py</c> (also reimplemented in
/// <c>laya-dotnet/tests/Laya.Tests/HashingEmbedder.cs</c>), kept here so this sample does not take
/// a test-project dependency. It exists solely so the shortlist section produces a fixed,
/// reproducible result without a real bi-encoder or a model download: a hashed
/// character-trigram bag-of-features vector, chosen because it is a pure function of the input
/// text and is trivial to reimplement bit-for-bit in C# with nothing but a string,
/// UTF-8 encoding and 32-bit integer arithmetic.
/// </summary>
/// <remarks>
/// Algorithm (mirrors the Python docstring precisely):
/// <list type="number">
/// <item>Lowercase with full Unicode case mapping — <see cref="LanguageDetection.PythonLower"/>,
/// not <see cref="string.ToLowerInvariant()"/> directly, because this project builds with
/// <c>InvariantGlobalization</c> enabled, under which plain <c>ToLowerInvariant</c> leaves U+0130
/// (Turkish capital I with dot above) untouched instead of expanding it to "i" + combining dot,
/// as Python's <c>str.lower()</c> does.</item>
/// <item>Pad with two ASCII spaces on each side.</item>
/// <item>Slide a window of exactly 3 Unicode code points (via <see cref="System.Text.Rune"/>,
/// never UTF-16 code units) across the padded text, one code point at a time.</item>
/// <item>UTF-8-encode each window's own 3 code points (re-encoded fresh each time, not sliced
/// from the whole padded text's byte buffer).</item>
/// <item>Hash those bytes with 32-bit FNV-1a (unsigned, wrapping multiply).</item>
/// <item><c>bucket = hash % 64</c>.</item>
/// <item><c>sign = +1</c> when the hash's top bit is 0, else <c>-1</c>.</item>
/// <item>Accumulate <c>sign</c> into <c>vector[bucket]</c> for every window (float64 throughout).</item>
/// <item>No normalisation.</item>
/// </list>
/// </remarks>
internal static class DemoHashingEmbedder
{
    /// <summary>Vector width, matching Python's <c>DIM</c>.</summary>
    public const int Dimension = 64;

    private const uint FnvOffsetBasis = 0x811c9dc5;
    private const uint FnvPrime = 0x01000193;

    /// <summary>Embed a batch of texts, one <see cref="Dimension"/>-length vector per input, in order.</summary>
    public static double[][] Embed(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var result = new double[texts.Count][];
        for (var i = 0; i < texts.Count; i++) result[i] = EmbedOne(texts[i]);
        return result;
    }

    /// <summary>Embed a single text into a <see cref="Dimension"/>-length vector.</summary>
    public static double[] EmbedOne(string? text)
    {
        var vector = new double[Dimension];
        var padded = "  " + PythonLower(text ?? string.Empty) + "  ";
        var runes = padded.EnumerateRunes().ToArray();
        var windowCount = runes.Length - 2; // always >= 2: padding guarantees at least 4 code points

        for (var i = 0; i < windowCount; i++)
        {
            var window = string.Concat(runes[i].ToString(), runes[i + 1].ToString(), runes[i + 2].ToString());
            var hash = Fnv1a32(Encoding.UTF8.GetBytes(window));
            var bucket = (int)(hash % Dimension);
            var sign = (hash >> 31 & 1) == 0 ? 1.0 : -1.0;
            vector[bucket] += sign;
        }
        return vector;
    }

    /// <summary>32-bit FNV-1a over raw bytes, wrapping like an unsigned 32-bit integer.</summary>
    private static uint Fnv1a32(byte[] data)
    {
        var hash = FnvOffsetBasis;
        foreach (var b in data)
        {
            hash ^= b;
            hash *= FnvPrime; // uint * uint wraps silently (unchecked) by default in C#
        }
        return hash;
    }

    /// <summary>
    /// Python's <c>str.lower()</c>: full Unicode lowercasing, including the one unconditional
    /// 1-to-2-code-point special case, U+0130 → U+0069 U+0307. A local copy of
    /// <c>LanguageDetection.PythonLower</c> (internal to the Laya assembly, not visible here) —
    /// see that method's remarks for why plain <see cref="string.ToLowerInvariant()"/> alone is
    /// not enough under <c>InvariantGlobalization</c>.
    /// </summary>
    private static string PythonLower(string text)
    {
        var lowered = text.ToLowerInvariant();
        return lowered.Contains('İ') ? lowered.Replace("İ", "i̇") : lowered;
    }
}
