using System.Numerics;

namespace Laya.Tests;

/// <summary>
/// Parity checks added after review: boxed numeric types in <c>clamp_temperature</c> / <c>temperature_rejected</c>,
/// and left truncation of a state only for a list. Expectations come from running the Python reference
/// (<c>float(x)</c> for each numeric type, <c>build_sequence(..., truncate_left=isinstance(state, list))</c>).
/// </summary>
public sealed class ReviewParityTests
{
    public static TheoryData<string, string, string> Numbers() => new()
    {
        { "byte", "0", "0.0" },
        { "byte", "255", "255.0" },
        { "sbyte", "-128", "-128.0" },
        { "sbyte", "127", "127.0" },
        { "short", "-32768", "-32768.0" },
        { "short", "32767", "32767.0" },
        { "ushort", "0", "0.0" },
        { "ushort", "65535", "65535.0" },
        { "int", "-2147483648", "-2147483648.0" },
        { "int", "2147483647", "2147483647.0" },
        { "int", "16777217", "16777217.0" },
        { "uint", "4294967295", "4294967295.0" },
        { "long", "9223372036854775807", "9.223372036854776e+18" },
        { "long", "-9223372036854775808", "-9.223372036854776e+18" },
        { "long", "9007199254740993", "9007199254740992.0" },
        { "long", "9007199254740995", "9007199254740996.0" },
        { "ulong", "18446744073709551615", "1.8446744073709552e+19" },
        { "ulong", "9007199254740993", "9007199254740992.0" },
        { "ulong", "9007199254740995", "9007199254740996.0" },
        { "ulong", "18446744073709550592", "1.8446744073709552e+19" },
        { "BigInteger", "9007199254740993", "9007199254740992.0" },
        { "BigInteger", "123456789012345678901234567890", "1.2345678901234568e+29" },
        { "BigInteger", "-9007199254740993", "-9007199254740992.0" },
        { "Int128", "170141183460469231731687303715884105727", "1.7014118346046923e+38" },
        { "Int128", "9007199254740993", "9007199254740992.0" },
        { "decimal", "0.1", "0.1" },
        { "decimal", "1.0000000000000001", "1.0" },
        { "decimal", "79228162514264337593543950335", "7.922816251426434e+28" },
        { "decimal", "-0.5", "-0.5" },
        { "decimal", "0.30000000000000004441", "0.30000000000000004" },
        { "decimal", "2.5", "2.5" },
        { "decimal", "4.35", "4.35" },
        { "decimal", "0.0000000000000000000000000001", "1e-28" },
        { "float", "0.1", "0.10000000149011612" },
        { "float", "2.5", "2.5" },
        { "float", "3.4028235e38", "3.4028234663852886e+38" },
        { "float", "1.17549435e-38", "1.1754943508222875e-38" },
        { "float", "-0.7", "-0.699999988079071" },
        { "Half", "0.1", "0.0999755859375" },
        { "Half", "2.5", "2.5" },
        { "Half", "65504", "65504.0" },
        { "Half", "-0.7", "-0.7001953125" },
    };

    private static object Box(string kind, string text) => kind switch
    {
        "byte" => byte.Parse(text), "sbyte" => sbyte.Parse(text), "short" => short.Parse(text), "ushort" => ushort.Parse(text),
        "int" => int.Parse(text), "uint" => uint.Parse(text), "long" => long.Parse(text), "ulong" => ulong.Parse(text),
        "BigInteger" => BigInteger.Parse(text), "Int128" => Int128.Parse(text), "decimal" => decimal.Parse(text, System.Globalization.CultureInfo.InvariantCulture),
        "float" => float.Parse(text, System.Globalization.CultureInfo.InvariantCulture),
        "Half" => Half.Parse(text, System.Globalization.CultureInfo.InvariantCulture),
        _ => throw new ArgumentException(kind),
    };

    [Theory]
    [MemberData(nameof(Numbers))]
    public void EveryBoxedNumberConvertsAsPythonsFloatDoes(string kind, string text, string repr)
    {
        var boxed = Box(kind, text);
        Assert.True(PythonFloat.TryConvert(boxed, out var value));
        Assert.Equal(repr, PythonFloat.Repr(value));
        Assert.Equal(Calibration.ClampTemperature(value), Calibration.ClampTemperature(boxed));
        Assert.Equal(value != Calibration.ClampTemperature(value), Calibration.TemperatureRejected(boxed, Calibration.ClampTemperature(value)));
    }

    private sealed class DigitsTokenizer : Laya.Tokenization.ILayaTokenizer
    {
        public int PadId => 0;
        public int SepId => 1;
        public int ClsId => 2;
        public int UnkId => 3;
        public int MaskId => 4;
        public string MaskToken => "[MASK]";
        public int[] Encode(string text) =>
            [.. System.Text.RegularExpressions.Regex.Matches(text, "[0-9]+").Select(m => int.Parse(m.Value))];
    }

    // The room for the state is four tokens, so eight numbers are over budget.
    private static int[] Built(object state)
    {
        var question = Question.Choice("pick", [new KeyValuePair<string, object?>("a", "x"), new KeyValuePair<string, object?>("b", "y")]);
        return SequenceBuilder.Build(new DigitsTokenizer(), state, question, 10, 10, PythonJson.IsListState(state)).Ids;
    }

    [Fact]
    public void AListOrArrayKeepsTheEndOfAnOverBudgetState()
    {
        Assert.Equal(new[] { 2, 1, 4, 4, 1, 1004, 1005, 1006, 1007, 1 }, Built(new List<int> { 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007 }));
        Assert.Equal(new[] { 2, 1, 4, 4, 1, 1004, 1005, 1006, 1007, 1 }, Built(new[] { 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007 }));
    }

    [Fact]
    public void ATupleKeepsTheStartOfAnOverBudgetStateAsPythonDoes() =>
        Assert.Equal(new[] { 2, 1, 4, 4, 1, 1000, 1001, 1002, 1003, 1 }, Built((1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007)));

    [Fact]
    public void ATupleIsNotAListState()
    {
        Assert.False(PythonJson.IsListState(("a", "b")));
        Assert.True(PythonJson.IsListState(new[] { "a" }));
        Assert.True(PythonJson.IsListState(new List<object?> { "a" }));
        Assert.False(PythonJson.IsListState(new Dictionary<string, object?>()));
    }
}
