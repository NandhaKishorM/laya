using System.Text.Json;
using System.Text.Json.Nodes;

namespace Laya.Tests;

/// <summary>
/// The hand-made calibration file the <c>calibration_*</c> golden cases were recorded with
/// (<c>laya-dotnet/tools/calibration_fixture.json</c>), which an engine under test loads through
/// <see cref="LayaOptions.Calibration"/>.
/// </summary>
internal static class CalibrationFixture
{
    internal const string FileName = "calibration_fixture.json";

    private static readonly Lazy<string> PathLazy = new(Locate);

    /// <summary>The fixture file, found by walking up from the test binary to the SDK's <c>tools</c> directory.</summary>
    internal static string Path => PathLazy.Value;

    /// <summary>Whether a recorded case was made with the calibration file.</summary>
    internal static bool Recorded(JsonElement root) => root.TryGetProperty("calibration", out var calibration) && calibration.ValueKind == JsonValueKind.Object;

    /// <summary>
    /// For a case recorded with a calibration file: the file it names is the fixture, and what it held then is what it holds now,
    /// so an engine loaded from the fixture is the engine that was recorded.
    /// </summary>
    internal static void AssertIsTheRecordedFile(JsonElement root)
    {
        if (!Recorded(root)) return;
        var recorded = root.GetProperty("calibration");
        Assert.Equal(FileName, recorded.GetProperty("file").GetString());
        Assert.True(
            JsonNode.DeepEquals(JsonNode.Parse(recorded.GetProperty("payload").GetRawText()), JsonNode.Parse(File.ReadAllText(Path))),
            $"{FileName} has changed since the golden was recorded; run tools/regen_golden.py");
    }

    private static string Locate()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = System.IO.Path.Combine(dir.FullName, "tools", FileName);
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException($"{FileName} not found in a tools directory above {AppContext.BaseDirectory}");
    }
}
