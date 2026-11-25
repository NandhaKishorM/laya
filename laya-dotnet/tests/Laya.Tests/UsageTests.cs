namespace Laya.Tests;

public class UsageTests
{
    private static Usage Full(
        IReadOnlyList<string>? truncated = null, IReadOnlyDictionary<string, OptionUsage>? options = null) =>
        new(10, 0)
        {
            Windows = 2, StateTokens = 100, StateTokensDropped = 5, Truncated = 1,
            TruncatedQuestions = truncated, Options = options,
        };

    [Fact]
    public void Plain_Values_EqualAndHashEqual()
    {
        var a = new Usage(10, 0);
        var b = new Usage(10, 0);
        Assert.True(a.Equals(b));
        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.True(a == b);
        Assert.False(a != b);
    }

    [Fact]
    public void Plain_Different_NotEqual_OperatorsAgree()
    {
        var a = new Usage(10, 0);
        var b = new Usage(11, 0);
        Assert.False(a.Equals(b));
        Assert.False(a == b);
        Assert.True(a != b);
        Assert.NotEqual(new Usage(10, 0), new Usage(10, 1));
    }

    [Fact]
    public void EqualContent_DistinctLists_EqualAndHashEqual()
    {
        var a = Full(truncated: new List<string> { "q1", "q2" });
        var b = Full(truncated: new[] { "q1", "q2" });
        Assert.True(a.Equals(b));
        Assert.True(a == b);
        Assert.False(a != b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void EqualContent_DistinctDictionaries_EqualAndHashEqual()
    {
        var a = Full(options: new Dictionary<string, OptionUsage> { ["q1"] = new(4, 3, 8), ["q2"] = new(2, 2, null) });
        var b = Full(options: new Dictionary<string, OptionUsage> { ["q1"] = new(4, 3, 8), ["q2"] = new(2, 2, null) });
        Assert.True(a.Equals(b));
        Assert.True(a == b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void Dictionaries_DifferentInsertionOrder_EqualAndHashEqual()
    {
        var a = Full(options: new Dictionary<string, OptionUsage>
        {
            ["a"] = new(1, 1, null), ["b"] = new(2, 2, 5), ["c"] = new(3, 1, 7),
        });
        var b = Full(options: new Dictionary<string, OptionUsage>
        {
            ["c"] = new(3, 1, 7), ["a"] = new(1, 1, null), ["b"] = new(2, 2, 5),
        });
        Assert.True(a.Equals(b));
        Assert.True(a == b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void TruncatedQuestions_NullVsEmpty_NotEqual()
    {
        var a = Full(truncated: null);
        var b = Full(truncated: Array.Empty<string>());
        Assert.False(a.Equals(b));
        Assert.False(b.Equals(a));
        Assert.True(a != b);
    }

    [Fact]
    public void TruncatedQuestions_BothNull_Equal()
    {
        Assert.True(Full(truncated: null) == Full(truncated: null));
    }

    [Fact]
    public void TruncatedQuestions_DifferentOrder_NotEqual()
    {
        var a = Full(truncated: new[] { "q1", "q2" });
        var b = Full(truncated: new[] { "q2", "q1" });
        Assert.False(a.Equals(b));
        Assert.True(a != b);
    }

    [Fact]
    public void TruncatedQuestions_DifferentContentOrLength_NotEqual()
    {
        var a = Full(truncated: new[] { "q1", "q2" });
        Assert.NotEqual(a, Full(truncated: new[] { "q1", "q3" }));
        Assert.NotEqual(a, Full(truncated: new[] { "q1" }));
        Assert.NotEqual(a, Full(truncated: new[] { "Q1", "q2" })); // ordinal, case-sensitive
    }

    [Fact]
    public void Options_NullVsEmpty_NotEqual()
    {
        var a = Full(options: null);
        var b = Full(options: new Dictionary<string, OptionUsage>());
        Assert.False(a.Equals(b));
        Assert.False(b.Equals(a));
        Assert.True(a != b);
    }

    [Fact]
    public void Options_BothNullOrBothEmpty_Equal()
    {
        Assert.True(Full(options: null) == Full(options: null));
        Assert.True(
            Full(options: new Dictionary<string, OptionUsage>())
            == Full(options: new Dictionary<string, OptionUsage>()));
    }

    [Fact]
    public void Options_DifferentValue_NotEqual()
    {
        var a = Full(options: new Dictionary<string, OptionUsage> { ["q1"] = new(4, 3, 8) });
        Assert.NotEqual(a, Full(options: new Dictionary<string, OptionUsage> { ["q1"] = new(4, 2, 8) }));
        Assert.NotEqual(a, Full(options: new Dictionary<string, OptionUsage> { ["q1"] = new(4, 3, null) }));
    }

    [Fact]
    public void Options_ExtraKeyOrDifferentKey_NotEqual()
    {
        var a = Full(options: new Dictionary<string, OptionUsage> { ["q1"] = new(4, 3, 8) });
        var extra = Full(options: new Dictionary<string, OptionUsage>
        {
            ["q1"] = new(4, 3, 8), ["q2"] = new(4, 3, 8),
        });
        var renamed = Full(options: new Dictionary<string, OptionUsage> { ["q2"] = new(4, 3, 8) });
        Assert.NotEqual(a, extra);
        Assert.NotEqual(extra, a);
        Assert.NotEqual(a, renamed);
        Assert.True(a != extra);
    }

    [Fact]
    public void ScalarFields_EachDifference_NotEqual()
    {
        var a = Full();
        Assert.NotEqual(a, a with { Windows = 3 });
        Assert.NotEqual(a, a with { StateTokens = 101 });
        Assert.NotEqual(a, a with { StateTokensDropped = 6 });
        Assert.NotEqual(a, a with { Truncated = 0 });
        Assert.NotEqual(a, a with { Windows = null });
        Assert.NotEqual(a, a with { StateTokens = null });
        Assert.NotEqual(a, a with { StateTokensDropped = null });
        Assert.NotEqual(a, a with { Truncated = null });
    }

    [Fact]
    public void WithCopies_StayEqual()
    {
        var a = Full(
            truncated: new List<string> { "q1" },
            options: new Dictionary<string, OptionUsage> { ["q1"] = new(4, 3, 8) });
        var copy = a with { };
        Assert.True(a == copy);
        Assert.Equal(a.GetHashCode(), copy.GetHashCode());

        var sameBack = (a with { Windows = 9 }) with { Windows = 2 };
        Assert.True(a.Equals(sameBack));
        Assert.Equal(a.GetHashCode(), sameBack.GetHashCode());

        Assert.NotEqual(a, a with { InputTokens = 11 });
    }

    [Fact]
    public void Options_KeysCompareOrdinallyWhateverEitherMapsComparer()
    {
        var ordinal = Full(options: new Dictionary<string, OptionUsage>(StringComparer.Ordinal)
            { ["q1"] = new OptionUsage(5, 3, 9) });
        var ignoreCase = Full(options: new Dictionary<string, OptionUsage>(StringComparer.OrdinalIgnoreCase)
            { ["Q1"] = new OptionUsage(5, 3, 9) });
        Assert.False(ordinal.Equals(ignoreCase));
        Assert.False(ignoreCase.Equals(ordinal));

        var sameKeyIgnoreCase = Full(options: new Dictionary<string, OptionUsage>(StringComparer.OrdinalIgnoreCase)
            { ["q1"] = new OptionUsage(5, 3, 9) });
        Assert.True(ordinal.Equals(sameKeyIgnoreCase));
        Assert.True(sameKeyIgnoreCase.Equals(ordinal));
        Assert.Equal(ordinal.GetHashCode(), sameKeyIgnoreCase.GetHashCode());
    }
}
