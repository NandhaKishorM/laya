namespace Laya.Tests;

using System.Text.Json;

/// <summary>
/// Verifies that <see cref="PythonJson"/> reproduces Python's <c>json.dumps</c> byte-for-byte
/// by asserting every entry in <c>golden/multilingual/json_probe.json</c>.
/// </summary>
public sealed class PythonJsonTests
{
    // Loaded once per test class; GoldenData.Directory walks the source tree on first access.
    private static readonly JsonDocument s_probe =
        GoldenData.Load("json_probe.json");

    /// <summary>
    /// Every entry in json_probe.json must match all three dialect outputs.
    /// The theory data includes the 0-based index so failures name the failing case.
    /// </summary>
    public static IEnumerable<object[]> ProbeRows()
    {
        var i = 0;
        foreach (var entry in s_probe.RootElement.EnumerateArray())
            yield return [i++, entry.Clone()];
    }

    [Theory]
    [MemberData(nameof(ProbeRows))]
    public void AllDialectsMatchGolden(int index, JsonElement entry)
    {
        var value = GoldenData.ToClr(entry.GetProperty("value"));

        // criterion_dialect is always present (never null).
        AssertDialect(index, "criterion", entry.GetProperty("criterion_dialect").GetString(),
                      PythonJson.Criterion(value));

        // instructions_dialect is always present (strings pass through; others are serialized).
        AssertDialect(index, "instructions", entry.GetProperty("instructions_dialect").GetString(),
                      PythonJson.Instructions(value));

        // state_dialect is always a string now: serialize_state accepts every JSON-serializable
        // type at runtime, so no probe value is outside its domain.
        AssertDialect(index, "state",
                      entry.GetProperty("state_dialect").GetString(),
                      PythonJson.State(value));
    }

    /// <summary>
    /// Compare one dialect's output, naming the probe entry on failure.
    /// </summary>
    /// <remarks>
    /// Written by hand rather than as <c>Assert.Equal</c> so the index reaches the message: the probe
    /// has 39 entries and several differ only in whitespace or escaping, which is hard to place from a
    /// bare string diff.
    /// </remarks>
    private static void AssertDialect(int index, string dialect, string? expected, string? actual)
    {
        if (expected == actual) return;
        // Built with string.Join so the message carries no escape sequences of its own: several
        // probe entries differ only in escaping, and an escape in the message would obscure that.
        Assert.Fail(string.Join(Environment.NewLine,
            $"json_probe[{index}] {dialect} dialect:",
            $"  python: {expected ?? NullText}",
            $"  dotnet: {actual ?? NullText}"));
    }

    /// <summary>Stands in for a null dialect output in a failure message.</summary>
    private const string NullText = "<null>";

    // ── State(null) ───────────────────────────────────────────────────────

    [Fact]
    public void StateWithNullReturnsJsonNull()
    {
        // Python: serialize_state(None) → json.dumps(None) → "null".
        // Returning C# null here caused an NRE in SequenceBuilder when a caller passed null state.
        Assert.Equal("null", PythonJson.State(null));
    }

    // ── Repr-specific unit tests ──────────────────────────────────────────

    [Theory]
    [InlineData(double.NaN, "NaN")]
    [InlineData(double.PositiveInfinity, "Infinity")]
    [InlineData(double.NegativeInfinity, "-Infinity")]
    [InlineData(0.0, "0.0")]
    [InlineData(1.0, "1.0")]
    [InlineData(2.0, "2.0")]
    [InlineData(-0.125, "-0.125")]
    [InlineData(0.5, "0.5")]
    [InlineData(1e21, "1e+21")]
    [InlineData(1e-7, "1e-07")]
    [InlineData(0.3333333333333333, "0.3333333333333333")]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(1e16, "1e+16")]
    [InlineData(1e17, "1e+17")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(1e-5, "1e-05")]
    [InlineData(123456789012345.6, "123456789012345.6")]
    [InlineData(1.7976931348623157e308, "1.7976931348623157e+308")]
    [InlineData(5e-324, "5e-324")]
    public void ReprMatchesPython(double input, string expected)
        => Assert.Equal(expected, PythonJson.Repr(input));

    // -0.0 needs a separate test because InlineData cannot distinguish it from 0.0.
    [Fact]
    public void ReprNegativeZero() => Assert.Equal("-0.0", PythonJson.Repr(-0.0));

    // ── Escaping unit tests ───────────────────────────────────────────────

    [Fact]
    public void EnsureAsciiEscapesNonAscii()
    {
        // ü = U+00FC; 😀 = U+1F600 (surrogate pair D83D DE00). Dumps(escapeNonAscii: true) still
        // exists and is exercised directly — 0.3.21 no longer routes Instructions through it, but
        // the escaping dialect itself is not deleted (0.3.6 golden fixtures and other callers may
        // still need it).
        string result = PythonJson.Dumps(new System.Collections.Generic.Dictionary<string, object?>
        {
            ["a"] = "naïve",
            ["b"] = "😀",
        }, escapeNonAscii: true);
        Assert.Equal("{\"a\": \"na\\u00efve\", \"b\": \"\\ud83d\\ude00\"}", result);
    }

    [Fact]
    public void EnsureAsciiFalseKeepsLiteralUnicode()
    {
        string result = PythonJson.Criterion(new System.Collections.Generic.Dictionary<string, object?>
        {
            ["a"] = "naïve",
        });
        Assert.Equal("{\"a\": \"naïve\"}", result);
    }

    [Fact]
    public void InstructionsNowMatchesEnsureAsciiFalse()
    {
        // 0.3.21's `_to_internal` switched to json.dumps(ins, ensure_ascii=False): real Unicode
        // glyphs, not \uXXXX escapes, reach the tokenizer. Same dialect as State/Criterion now.
        string result = PythonJson.Instructions(new System.Collections.Generic.Dictionary<string, object?>
        {
            ["a"] = "naïve",
            ["b"] = "😀",
        });
        Assert.Equal("{\"a\": \"naïve\", \"b\": \"😀\"}", result);
    }

    [Fact]
    public void HtmlCharsNotEscaped()
    {
        string result = PythonJson.Criterion("<tag> & 'quoted'");
        // Strings pass through verbatim.
        Assert.Equal("<tag> & 'quoted'", result);
    }

    [Fact]
    public void SlashNotEscaped()
    {
        // Python does not escape / — System.Text.Json's default encoder does.
        string result = PythonJson.Criterion(new System.Collections.Generic.Dictionary<string, object?>
        {
            ["url"] = "a/b",
        });
        Assert.Equal("{\"url\": \"a/b\"}", result);
    }

    [Fact]
    public void ControlCharsEscaped()
    {
        string result = PythonJson.Criterion(new System.Collections.Generic.Dictionary<string, object?>
        {
            ["x"] = "\x01\x1f",
        });
        Assert.Equal("{\"x\": \"\\u0001\\u001f\"}", result);
    }

    [Fact]
    public void CircularReferenceThrows()
    {
        var list = new System.Collections.Generic.List<object?>();
        list.Add(list);
        Assert.Throws<JsonException>(() => PythonJson.Criterion(list));
    }

    [Fact]
    public void UnknownTypeWithoutFallbackThrows()
    {
        var obj = new System.DateTime(2024, 1, 1);
        Assert.Throws<JsonException>(() => PythonJson.Dumps(obj));
    }

    [Fact]
    public void UnknownTypeWithFallbackUsesToString()
    {
        // PythonJson.Criterion uses fallbackToStr:true for non-string values.
        // Wrap in a dict so the object reaches the serializer.
        var result = PythonJson.Dumps(new System.DateTime(2024, 1, 1),
            escapeNonAscii: false, fallbackToStr: true);
        Assert.StartsWith("\"", result);
        Assert.EndsWith("\"", result);
    }
}
