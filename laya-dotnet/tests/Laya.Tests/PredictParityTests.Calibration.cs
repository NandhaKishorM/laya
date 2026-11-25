using System.Diagnostics;

namespace Laya.Tests;

// Where `LayaOptions.Calibration` is read in `LayaEngine.Create`: Python's ONNXAgent.__init__ loads the file last
// (onnx_agent.py:263), after the checkpoint and language-table warnings, the ONNX session and the tokenizer. That needs real
// artifacts, so it is pinned here, with the parity lanes; the file's own error texts are tested model-free in
// CalibrationFileTests.
public sealed partial class PredictParityTests
{
    private sealed class TraceCapture : TraceListener
    {
        public List<string> Messages { get; } = [];
        public override void Write(string? message) => Messages.Add(message ?? "");
        public override void WriteLine(string? message) => Messages.Add(message ?? "");
        public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType, int id, string? format, params object?[]? args) =>
            Messages.Add(args is null ? format ?? "" : string.Format(format ?? "", args));
    }

    public static IEnumerable<object[]> ParityCheckpointCases() => TestArtifacts.ParityCheckpoints.Select(c => new object[] { c });

    private string ArtifactDirectory(LayaCheckpoint checkpoint)
    {
        if (!fixture.Available(checkpoint)) fixture.Require(checkpoint);   // skips, with the usual reason
        return TestArtifacts.For(checkpoint).Directory;
    }

    [Theory]
    [MemberData(nameof(ParityCheckpointCases))]
    public void TheCalibrationClampWarningComesAfterTheCheckpointAndLanguageWarnings(LayaCheckpoint checkpoint)
    {
        var dir = ArtifactDirectory(checkpoint);
        // Names of this run's own, so the texts are not shared with a test that warned earlier or in parallel.
        var token = "calib" + Guid.NewGuid().ToString("N");
        var file = Path.Combine(Path.GetTempPath(), token + ".json");
        File.WriteAllText(file, $"{{\"temperature\": [9.0, 1.0, 1.0], \"temperature_by_options\": {{\"{token}\": 0.1}}}}");
        var capture = new TraceCapture();
        Trace.Listeners.Add(capture);
        try
        {
            using var engine = LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = dir,
                LangTemperatures = new Dictionary<string, LayaLangTemperature?> { [token] = new() { Temperature = [0.1, 1.0, 1.0] } },
                Calibration = file,
            });
        }
        finally
        {
            Trace.Listeners.Remove(capture);
            File.Delete(file);
        }

        var messages = capture.Messages;
        var ofLanguage = messages.FindIndex(m => m.StartsWith($"laya: language override '{token}'", StringComparison.Ordinal));
        var ofCalibration = messages.FindIndex(m => m.StartsWith(
            $"laya: calibration temperatures are invalid or outside [0.5, 5]; using temperature[0]=9.0 -> 5, {token}=0.1 -> 0.5.", StringComparison.Ordinal));
        Assert.True(ofLanguage >= 0 && ofCalibration > ofLanguage, string.Join("\n", messages));
        // A checkpoint that ships temperatures the clamp moves reports them after the language table and before the file.
        var ofCheckpoint = messages.FindIndex(m => m.StartsWith("laya ONNX: this checkpoint ships temperatures outside", StringComparison.Ordinal));
        if (ofCheckpoint >= 0)
            Assert.True(ofLanguage < ofCheckpoint && ofCheckpoint < ofCalibration, string.Join("\n", messages));
    }

    [Theory]
    [MemberData(nameof(ParityCheckpointCases))]
    public void ABrokenCheckpointFailsBeforeABadCalibrationFileIsRead(LayaCheckpoint checkpoint)
    {
        var dir = ArtifactDirectory(checkpoint);
        // The real graph with a tokenizer that cannot be read: Python builds the session, fails on the tokenizer, and never
        // reaches the calibration file. Linking the graph and its weights needs symbolic links, which Windows may refuse.
        var broken = Path.Combine(Path.GetTempPath(), "laya-broken-" + Guid.NewGuid().ToString("N"));
        var bad = Path.Combine(broken, "calibration.json");
        try
        {
            Directory.CreateDirectory(Path.Combine(broken, "tokenizer"));
            try
            {
                foreach (var name in new[] { "model.onnx", "model.onnx.data" })
                    File.CreateSymbolicLink(Path.Combine(broken, name), Path.Combine(dir, name));
            }
            catch (Exception e) when (e is UnauthorizedAccessException or IOException or PlatformNotSupportedException)
            {
                Assert.Skip($"cannot link the real graph into a scratch directory: {e.Message}");
            }
            File.Copy(Path.Combine(dir, "rl_agent_config.json"), Path.Combine(broken, "rl_agent_config.json"));
            File.WriteAllText(Path.Combine(broken, "tokenizer", "tokenizer.json"), "not a tokenizer");
            File.WriteAllText(bad, "[1, 2]");

            var alone = Assert.ThrowsAny<Exception>(() => LayaEngine.Create(new LayaOptions { ModelDirectory = broken }));
            var withFile = Assert.ThrowsAny<Exception>(() => LayaEngine.Create(new LayaOptions { ModelDirectory = broken, Calibration = bad }));
            Assert.IsNotType<InvalidDataException>(withFile);
            Assert.Equal(alone.GetType(), withFile.GetType());
            Assert.Equal(alone.Message, withFile.Message);
        }
        finally { Directory.Delete(broken, recursive: true); }
    }

    [Theory]
    [MemberData(nameof(ParityCheckpointCases))]
    public void ABadCalibrationFileRefusesTheLoadOfAGoodCheckpoint(LayaCheckpoint checkpoint)
    {
        var dir = ArtifactDirectory(checkpoint);
        var file = Path.Combine(Path.GetTempPath(), "calib" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(file, "[1, 2]");
        try
        {
            var error = Assert.Throws<InvalidDataException>(() => LayaEngine.Create(new LayaOptions { ModelDirectory = dir, Calibration = file }));
            Assert.Equal("ValueError: calibration JSON must be an object, got list", error.Message);
            Assert.Throws<FileNotFoundException>(() => LayaEngine.Create(new LayaOptions { ModelDirectory = dir, Calibration = file + ".absent" }));
            // `if calibration:` is false for "": no file is read.
            LayaEngine.Create(new LayaOptions { ModelDirectory = dir, Calibration = "" }).Dispose();
        }
        finally { File.Delete(file); }
    }
}
