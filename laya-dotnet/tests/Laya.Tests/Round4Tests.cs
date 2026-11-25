using System.Globalization;

namespace Laya.Tests;

/// <summary>
/// Tier 1: <see cref="Calibration.Round4"/> against Python's <c>round(x, 4)</c>. Every expected value
/// below was printed by CPython (<c>repr(round(x, 4))</c>), not computed here.
/// </summary>
/// <remarks>
/// <c>golden/round4/round4_python.txt</c> holds 20,000 seeded inputs and CPython's results as
/// <c>float.hex</c> pairs; regenerate it with <c>laya-dotnet/tools/dump_round4_golden.py</c>, never by hand.
/// </remarks>
public sealed class Round4Tests
{
    // Exact binary ties (k / 32, k / 64, k / 128, either sign) round half to even on the exact value.
    private static readonly (double Input, double Expected)[] Ties =
    [
        (0.03125, 0.0312), (-0.03125, -0.0312), (0.09375, 0.0938),
        (-0.09375, -0.0938), (0.15625, 0.1562), (-0.15625, -0.1562),
        (0.21875, 0.2188), (-0.21875, -0.2188), (0.28125, 0.2812),
        (-0.28125, -0.2812), (0.34375, 0.3438), (-0.34375, -0.3438),
        (0.40625, 0.4062), (-0.40625, -0.4062), (0.46875, 0.4688),
        (-0.46875, -0.4688), (0.53125, 0.5312), (-0.53125, -0.5312),
        (0.96875, 0.9688), (-0.96875, -0.9688), (1.03125, 1.0312),
        (-1.03125, -1.0312), (1.96875, 1.9688), (-1.96875, -1.9688),
        (2.03125, 2.0312), (-2.03125, -2.0312), (3.96875, 3.9688),
        (-3.96875, -3.9688), (4.03125, 4.0312), (-4.03125, -4.0312),
        (7.96875, 7.9688), (-7.96875, -7.9688), (8.03125, 8.0312),
        (-8.03125, -8.0312), (15.96875, 15.9688), (-15.96875, -15.9688),
        (0.015625, 0.0156), (-0.015625, -0.0156), (0.046875, 0.0469),
        (-0.046875, -0.0469), (0.078125, 0.0781), (-0.078125, -0.0781),
        (0.109375, 0.1094), (-0.109375, -0.1094), (0.140625, 0.1406),
        (-0.140625, -0.1406), (0.171875, 0.1719), (-0.171875, -0.1719),
        (0.203125, 0.2031), (-0.203125, -0.2031), (0.234375, 0.2344),
        (-0.234375, -0.2344), (0.265625, 0.2656), (-0.265625, -0.2656),
        (0.484375, 0.4844), (-0.484375, -0.4844), (0.515625, 0.5156),
        (-0.515625, -0.5156), (0.984375, 0.9844), (-0.984375, -0.9844),
        (1.015625, 1.0156), (-1.015625, -1.0156), (1.984375, 1.9844),
        (-1.984375, -1.9844), (2.015625, 2.0156), (-2.015625, -2.0156),
        (3.984375, 3.9844), (-3.984375, -3.9844), (4.015625, 4.0156),
        (-4.015625, -4.0156), (7.984375, 7.9844), (-7.984375, -7.9844),
        (0.0078125, 0.0078), (-0.0078125, -0.0078), (0.0234375, 0.0234),
        (-0.0234375, -0.0234), (0.0390625, 0.0391), (-0.0390625, -0.0391),
        (0.0546875, 0.0547), (-0.0546875, -0.0547), (0.0703125, 0.0703),
        (-0.0703125, -0.0703), (0.0859375, 0.0859), (-0.0859375, -0.0859),
        (0.1015625, 0.1016), (-0.1015625, -0.1016), (0.1171875, 0.1172),
        (-0.1171875, -0.1172), (0.1328125, 0.1328), (-0.1328125, -0.1328),
        (0.2421875, 0.2422), (-0.2421875, -0.2422), (0.2578125, 0.2578),
        (-0.2578125, -0.2578), (0.4921875, 0.4922), (-0.4921875, -0.4922),
        (0.5078125, 0.5078), (-0.5078125, -0.5078), (0.9921875, 0.9922),
        (-0.9921875, -0.9922), (1.0078125, 1.0078), (-1.0078125, -1.0078),
        (1.9921875, 1.9922), (-1.9921875, -1.9922), (2.0078125, 2.0078),
        (-2.0078125, -2.0078), (3.9921875, 3.9922), (-3.9921875, -3.9922),
        (0.00625, 0.0063), (0.0078125, 0.0078), (0.00390625, 0.0039)
    ];

    // k / 160 for k = 0..160: decimal near-ties whose binary value sits just either side of the tie.
    private static readonly (double Input, double Expected)[] NearTies =
    [
        (0.0, 0.0), (0.00625, 0.0063), (0.0125, 0.0125),
        (0.01875, 0.0187), (0.025, 0.025), (0.03125, 0.0312),
        (0.0375, 0.0375), (0.04375, 0.0437), (0.05, 0.05),
        (0.05625, 0.0563), (0.0625, 0.0625), (0.06875, 0.0688),
        (0.075, 0.075), (0.08125, 0.0813), (0.0875, 0.0875),
        (0.09375, 0.0938), (0.1, 0.1), (0.10625, 0.1062),
        (0.1125, 0.1125), (0.11875, 0.1187), (0.125, 0.125),
        (0.13125, 0.1313), (0.1375, 0.1375), (0.14375, 0.1437),
        (0.15, 0.15), (0.15625, 0.1562), (0.1625, 0.1625),
        (0.16875, 0.1688), (0.175, 0.175), (0.18125, 0.1812),
        (0.1875, 0.1875), (0.19375, 0.1938), (0.2, 0.2),
        (0.20625, 0.2062), (0.2125, 0.2125), (0.21875, 0.2188),
        (0.225, 0.225), (0.23125, 0.2313), (0.2375, 0.2375),
        (0.24375, 0.2437), (0.25, 0.25), (0.25625, 0.2562),
        (0.2625, 0.2625), (0.26875, 0.2687), (0.275, 0.275),
        (0.28125, 0.2812), (0.2875, 0.2875), (0.29375, 0.2938),
        (0.3, 0.3), (0.30625, 0.3063), (0.3125, 0.3125),
        (0.31875, 0.3187), (0.325, 0.325), (0.33125, 0.3312),
        (0.3375, 0.3375), (0.34375, 0.3438), (0.35, 0.35),
        (0.35625, 0.3563), (0.3625, 0.3625), (0.36875, 0.3688),
        (0.375, 0.375), (0.38125, 0.3812), (0.3875, 0.3875),
        (0.39375, 0.3937), (0.4, 0.4), (0.40625, 0.4062),
        (0.4125, 0.4125), (0.41875, 0.4188), (0.425, 0.425),
        (0.43125, 0.4313), (0.4375, 0.4375), (0.44375, 0.4437),
        (0.45, 0.45), (0.45625, 0.4562), (0.4625, 0.4625),
        (0.46875, 0.4688), (0.475, 0.475), (0.48125, 0.4813),
        (0.4875, 0.4875), (0.49375, 0.4938), (0.5, 0.5),
        (0.50625, 0.5062), (0.5125, 0.5125), (0.51875, 0.5188),
        (0.525, 0.525), (0.53125, 0.5312), (0.5375, 0.5375),
        (0.54375, 0.5437), (0.55, 0.55), (0.55625, 0.5563),
        (0.5625, 0.5625), (0.56875, 0.5687), (0.575, 0.575),
        (0.58125, 0.5813), (0.5875, 0.5875), (0.59375, 0.5938),
        (0.6, 0.6), (0.60625, 0.6062), (0.6125, 0.6125),
        (0.61875, 0.6188), (0.625, 0.625), (0.63125, 0.6312),
        (0.6375, 0.6375), (0.64375, 0.6438), (0.65, 0.65),
        (0.65625, 0.6562), (0.6625, 0.6625), (0.66875, 0.6687),
        (0.675, 0.675), (0.68125, 0.6813), (0.6875, 0.6875),
        (0.69375, 0.6937), (0.7, 0.7), (0.70625, 0.7063),
        (0.7125, 0.7125), (0.71875, 0.7188), (0.725, 0.725),
        (0.73125, 0.7312), (0.7375, 0.7375), (0.74375, 0.7438),
        (0.75, 0.75), (0.75625, 0.7562), (0.7625, 0.7625),
        (0.76875, 0.7688), (0.775, 0.775), (0.78125, 0.7812),
        (0.7875, 0.7875), (0.79375, 0.7937), (0.8, 0.8),
        (0.80625, 0.8063), (0.8125, 0.8125), (0.81875, 0.8187),
        (0.825, 0.825), (0.83125, 0.8313), (0.8375, 0.8375),
        (0.84375, 0.8438), (0.85, 0.85), (0.85625, 0.8562),
        (0.8625, 0.8625), (0.86875, 0.8688), (0.875, 0.875),
        (0.88125, 0.8812), (0.8875, 0.8875), (0.89375, 0.8938),
        (0.9, 0.9), (0.90625, 0.9062), (0.9125, 0.9125),
        (0.91875, 0.9187), (0.925, 0.925), (0.93125, 0.9313),
        (0.9375, 0.9375), (0.94375, 0.9437), (0.95, 0.95),
        (0.95625, 0.9563), (0.9625, 0.9625), (0.96875, 0.9688),
        (0.975, 0.975), (0.98125, 0.9812), (0.9875, 0.9875),
        (0.99375, 0.9938), (1.0, 1.0)
    ];

    // Values above 1 (expected scores reach 10) and magnitudes where the double is coarser than 1e-4.
    private static readonly (double Input, double Expected)[] Large =
    [
        (2.00005, 2.0), (9.99995, 10.0), (3.03125, 3.0312),
        (2.5, 2.5), (9.9999, 9.9999), (5.00005, 5.0),
        (7.12345, 7.1235), (1099511627776.5, 1099511627776.5), (500000000000.00006, 500000000000.0001),
        (1000000000000.0, 1000000000000.0), (1e+300, 1e+300)
    ];

    // Values that round to zero (Python keeps the sign of the input) and the extremes of the range.
    private static readonly (double Input, double Expected)[] Tiny =
    [
        (1e-05, 0.0), (4.9999e-05, 0.0), (5e-05, 0.0001),
        (-5e-05, -0.0001), (5e-05, 0.0001), (1e-300, 0.0),
        (-1e-300, -0.0), (5e-324, 0.0), (-5e-324, -0.0),
        (-1e-05, -0.0), (-4.9999e-05, -0.0), (0.0, 0.0),
        (-0.0, -0.0), (2.2250738585072014e-308, 0.0), (1.7976931348623157e+308, 1.7976931348623157e+308)
    ];

    private static readonly (double Input, double Expected)[] NonFinite =
    [
        (double.NaN, double.NaN), (double.PositiveInfinity, double.PositiveInfinity), (double.NegativeInfinity, double.NegativeInfinity)
    ];

    private static void AssertSame(double input, double expected, double actual)
    {
        // Compare bits so that -0.0 differs from 0.0; every NaN is the same result.
        var same = double.IsNaN(expected)
            ? double.IsNaN(actual)
            : BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual);
        Assert.True(same, $"Round4({input:R}) = {actual:R}, Python round({input:R}, 4) = {expected:R}");
    }

    [Fact]
    public void ExactTiesMatchPython()
    {
        foreach (var (input, expected) in Ties) AssertSame(input, expected, Calibration.Round4(input));
    }

    [Fact]
    public void DecimalNearTiesMatchPython()
    {
        foreach (var (input, expected) in NearTies) AssertSame(input, expected, Calibration.Round4(input));
    }

    [Fact]
    public void LargeValuesMatchPython()
    {
        foreach (var (input, expected) in Large) AssertSame(input, expected, Calibration.Round4(input));
    }

    [Fact]
    public void TinyValuesKeepPythonsSign()
    {
        foreach (var (input, expected) in Tiny) AssertSame(input, expected, Calibration.Round4(input));
    }

    [Fact]
    public void NonFiniteValuesPassThrough()
    {
        foreach (var (input, expected) in NonFinite) AssertSame(input, expected, Calibration.Round4(input));
    }

    [Fact]
    public void SeededRandomValuesMatchPython()
    {
        var path = Locate();
        var lines = File.ReadAllLines(path).Where(l => l.Length > 0).ToArray();
        Assert.Equal(20_000, lines.Length);
        foreach (var line in lines)
        {
            var parts = line.Trim().Split(' ');
            var input = ParseHex(parts[0]);
            AssertSame(input, ParseHex(parts[1]), Calibration.Round4(input));
        }
    }

    // Parses Python's float.hex() form, e.g. "-0x1.8p-3" or "0x0.0p+0".
    private static double ParseHex(string s)
    {
        var negative = s.StartsWith('-');
        if (negative) s = s[1..];
        s = s[2..]; // "0x"
        var p = s.IndexOf('p');
        var exponent = int.Parse(s[(p + 1)..], CultureInfo.InvariantCulture);
        var mantissa = s[..p];
        var dot = mantissa.IndexOf('.');
        var digits = mantissa.Remove(dot, 1);
        var bits = Convert.ToUInt64(digits, 16);
        var fractionNibbles = mantissa.Length - dot - 1;
        var value = Math.ScaleB((double)bits, exponent - 4 * fractionNibbles);
        return negative ? -value : value;
    }

    private static string Locate()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "golden", "round4", "round4_python.txt");
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException(
            "golden/round4/round4_python.txt not found; run `python laya-dotnet/tools/dump_round4_golden.py`.");
    }
}
