using System.Diagnostics;

namespace Laya.Tests;

/// <summary>
/// <c>clamp_temperature</c> / <c>temperature_rejected</c> on the JSON values a checkpoint's
/// <c>rl_agent_config.json</c> can hold. Every expectation below was produced by running the Python
/// functions in <c>laya/common.py</c> (CPython 3.12) on <c>json.loads(value)</c>, with the entry text
/// built as <c>onnx_agent.py</c> builds it: <c>"%s=%r -> %g"</c>.
/// </summary>
[Collection(WarningStateCollection.Name)]
public sealed class TemperatureConfigTests
{
    [Theory]
    [InlineData("2.5", 2.5, false, "temperature[0]=2.5 -> 2.5")]
    [InlineData("\"2.5\"", 2.5, false, "temperature[0]='2.5' -> 2.5")]
    [InlineData("\" 2.5 \"", 2.5, false, "temperature[0]=' 2.5 ' -> 2.5")]
    [InlineData("\"2_5\"", 5.0, true, "temperature[0]='2_5' -> 5")]
    [InlineData("\"1e1\"", 5.0, true, "temperature[0]='1e1' -> 5")]
    [InlineData("\"0x1\"", 1.0, true, "temperature[0]='0x1' -> 1")]
    [InlineData("\"\\uff12.\\uff15\"", 2.5, false, "temperature[0]='２.５' -> 2.5")]
    [InlineData("true", 1.0, true, "temperature[0]=True -> 1")]
    [InlineData("false", 1.0, true, "temperature[0]=False -> 1")]
    [InlineData("null", 1.0, true, "temperature[0]=None -> 1")]
    [InlineData("[1]", 1.0, true, "temperature[0]=[1] -> 1")]
    [InlineData("{\"a\": 1}", 1.0, true, "temperature[0]={'a': 1} -> 1")]
    [InlineData("\"nan\"", 1.0, true, "temperature[0]='nan' -> 1")]
    [InlineData("\"-Infinity\"", 1.0, true, "temperature[0]='-Infinity' -> 1")]
    [InlineData("\"5.\"", 5.0, false, "temperature[0]='5.' -> 5")]
    [InlineData("\".5\"", 0.5, false, "temperature[0]='.5' -> 0.5")]
    [InlineData("\"1__0\"", 1.0, true, "temperature[0]='1__0' -> 1")]
    [InlineData("\"_1\"", 1.0, true, "temperature[0]='_1' -> 1")]
    [InlineData("\"\"", 1.0, true, "temperature[0]='' -> 1")]
    [InlineData("\"abc\"", 1.0, true, "temperature[0]='abc' -> 1")]
    [InlineData("7", 5.0, true, "temperature[0]=7 -> 5")]
    [InlineData("0.1006", 0.5, true, "temperature[0]=0.1006 -> 0.5")]
    [InlineData("1e999", 1.0, true, "temperature[0]=inf -> 1")]
    [InlineData("\"\\u001f2.5\\u001f\"", 1.0, true, "temperature[0]='\\x1f2.5\\x1f' -> 1")]
    [InlineData("\"+3\"", 3.0, false, "temperature[0]='+3' -> 3")]
    [InlineData("\"1_e5\"", 1.0, true, "temperature[0]='1_e5' -> 1")]
    [InlineData("\"1e1_0\"", 5.0, true, "temperature[0]='1e1_0' -> 5")]
    [InlineData("\"\\u0663\"", 3.0, false, "temperature[0]='٣' -> 3")]
    [InlineData("\"1_000.0_1\"", 5.0, true, "temperature[0]='1_000.0_1' -> 5")]
    [InlineData("\"1.5e+0\"", 1.5, false, "temperature[0]='1.5e+0' -> 1.5")]
    [InlineData("\"  \\n3\\t\"", 3.0, false, "temperature[0]='  \\n3\\t' -> 3")]
    [InlineData("\"- 3\"", 1.0, true, "temperature[0]='- 3' -> 1")]
    [InlineData("\"inf\"", 1.0, true, "temperature[0]='inf' -> 1")]
    [InlineData("\"NaN\"", 1.0, true, "temperature[0]='NaN' -> 1")]
    [InlineData("\"1,5\"", 1.0, true, "temperature[0]='1,5' -> 1")]
    [InlineData("0", 0.5, true, "temperature[0]=0 -> 0.5")]
    [InlineData("-2", 0.5, true, "temperature[0]=-2 -> 0.5")]
    [InlineData("\"0.2\"", 0.5, true, "temperature[0]='0.2' -> 0.5")]
    [InlineData("10.0", 5.0, true, "temperature[0]=10.0 -> 5")]
    [InlineData("[\"a\", 1]", 1.0, true, "temperature[0]=['a', 1] -> 1")]
    [InlineData("{\"k\": [true, null]}", 1.0, true, "temperature[0]={'k': [True, None]} -> 1")]
    [InlineData("\"it's\"", 1.0, true, "temperature[0]=\"it's\" -> 1")]
    public void ConfigEntryMatchesPython(string json, double applied, bool rejected, string entry)
    {
        var config = LayaConfig.Parse("{\"temperature\": [" + json + ", 1, 1]}");
        Assert.Equal(applied, config.Temperature[0]);
        Assert.Equal(applied, Calibration.ClampTemperature(config.TemperatureRawValues[0]));
        Assert.Equal(rejected, Calibration.TemperatureRejected(config.TemperatureRawValues[0], applied));
        Assert.Equal<IEnumerable<string>>(rejected ? new[] { entry } : [], config.ClampedTemperatures);
    }

    private sealed class Capture : TraceListener
    {
        public List<string> Messages { get; } = [];
        public override void Write(string? message) => Messages.Add(message ?? "");
        public override void WriteLine(string? message) => Messages.Add(message ?? "");
        public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType, int id, string? format, params object?[]? args) =>
            Messages.Add(args is null ? format ?? "" : string.Format(format ?? "", args));
    }

    [Fact]
    public void WarnsOnceWithPythonsTextListingEveryRejectedEntry()
    {
        // Expected text from onnx_agent.py:254-262 run on the same document.
        const string json = "{\"temperature\": [true, \"2.5\", 0.1, 9, null, \"x\"], "
            + "\"temperature_by_options\": {\"choice:11+\": 0.1006, \"noul:2\": \"3\"}}";
        const string warning = "laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort confidence; "
            + "clamping choice:11+=0.1006 -> 0.5, temperature[0]=True -> 1, temperature[2]=0.1 -> 0.5, temperature[3]=9 -> 5, "
            + "temperature[4]=None -> 1, temperature[5]='x' -> 1. Treat confidence from the affected buckets as uncalibrated.";

        var capture = new Capture();
        Trace.Listeners.Add(capture);
        LayaConfig config;
        try { config = LayaConfig.Parse(json); }
        finally { Trace.Listeners.Remove(capture); }

        Assert.Equal<IEnumerable<double>>(new[] { 1.0, 2.5, 0.5, 5.0, 1.0, 1.0 }, config.Temperature);
        Assert.Equal(0.5, config.TemperatureByOptions["choice:11+"]);
        Assert.Equal(3.0, config.TemperatureByOptions["noul:2"]);
        // Other tests may warn through the same global listener meanwhile; this document warns exactly once.
        Assert.Single(capture.Messages, m => m == warning);
    }

    [Fact]
    public void AcceptedTemperaturesDoNotWarn()
    {
        var config = LayaConfig.Parse("{\"temperature\": [1.0, 2.5, \"3\"], \"temperature_by_options\": {\"choice:2\": 1.5}}");
        Assert.Empty(config.ClampedTemperatures);
        Assert.Equal<IEnumerable<double>>(new[] { 1.0, 2.5, 3.0 }, config.Temperature);
    }

    // Python: json.loads accepts the bare tokens NaN / Infinity / -Infinity (-> float), so the entry clamps to 1.0.
    [Fact]
    public void BareNaNAndInfinityTokensLoadAsPythonDoes()
    {
        var config = LayaConfig.Parse("{\"temperature\": [NaN, 1.2, 1], \"temperature_by_options\": {\"a\": Infinity, \"b\": -Infinity, \"c\": \"NaN\"}}");
        Assert.Equal<IEnumerable<double>>(new[] { 1.0, 1.2, 1.0 }, config.Temperature);
        Assert.Equal<IEnumerable<string>>(
            ["a=inf -> 1", "b=-inf -> 1", "c='NaN' -> 1", "temperature[0]=nan -> 1"], config.ClampedTemperatures);
    }

    // Python 0.4.20: float literals that overflow read as inf; 10**308 is an int that still converts, and an int too wide
    // for a double (10**309, 2**1024) is read by clamp_temperature/temperature_rejected as unreadable: 1.0, named in full.
    [Fact]
    public void OverflowingFloatLiteralIsInfinityAndAHugeIntegerFallsBackToOne()
    {
        var ok = LayaConfig.Parse("{\"temperature\": [1, 1e400, 1" + new string('0', 308) + "]}");
        Assert.Equal<IEnumerable<double>>(new[] { 1.0, 1.0, 5.0 }, ok.Temperature);
        Assert.Equal("temperature[1]=inf -> 1", ok.ClampedTemperatures[0]);
        Assert.Equal("temperature[2]=1" + new string('0', 308) + " -> 5", ok.ClampedTemperatures[1]);

        var wide = System.Numerics.BigInteger.Pow(2, 1024).ToString(System.Globalization.CultureInfo.InvariantCulture);
        foreach (var huge in new[] { "1" + new string('0', 309), "-1" + new string('0', 309), wide })
        {
            var config = LayaConfig.Parse(
                "{\"temperature\": [" + huge + ", 2.0, 3.0], \"temperature_by_options\": {\"choice:2\": " + huge + "}}");
            Assert.Equal<IEnumerable<double>>(new[] { 1.0, 2.0, 3.0 }, config.Temperature);
            Assert.Equal(1.0, config.TemperatureByOptions["choice:2"]);
            Assert.Equal<IEnumerable<string>>(
                ["choice:2=" + huge + " -> 1", "temperature[0]=" + huge + " -> 1"], config.ClampedTemperatures);
        }
    }
}
