using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Serialises every test that changes process-wide environment variables (<c>LAYA_REVISION</c>,
/// <c>LAYA_SHA256_DIGESTS</c>, ...) and keeps the rest of the run from overlapping with it: engine,
/// artifact and router construction all read them.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ProcessEnvironmentCollection
{
    /// <summary>The collection name.</summary>
    public const string Name = "process-environment";
}

/// <summary>Sets environment variables for a scope and restores their previous values.</summary>
internal sealed class EnvScope : IDisposable
{
    private static readonly string[] Names =
        ["LAYA_REVISION", "LAYA_SHA256_DIGESTS", "LAYA_ONNX_DIR", "LAYA_ONNX_ROOT", "HF_TOKEN"];

    private readonly Dictionary<string, string?> _saved = [];

    /// <summary>Clears every variable this feature reads, then sets <paramref name="values"/>.</summary>
    public EnvScope(params (string Name, string? Value)[] values)
    {
        foreach (var name in Names)
        {
            _saved[name] = Environment.GetEnvironmentVariable(name);
            Environment.SetEnvironmentVariable(name, null);
        }
        foreach (var (name, value) in values) Environment.SetEnvironmentVariable(name, value);
    }

    public void Dispose()
    {
        foreach (var (name, value) in _saved) Environment.SetEnvironmentVariable(name, value);
    }
}

/// <summary>
/// Replays <c>golden/routing/revision_probe.json</c>, recorded from Python's own
/// <c>resolve_revision</c>, <c>verify_digests</c>, <c>_digests_from_env</c> and
/// <c>_merge_expected_digests</c> over the fixture files in <c>tools/revision_fixtures/</c>. No model
/// and no network.
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class RevisionProbeTests
{
    private static JsonElement Probe => RoutingGoldenData.Load("revision_probe.json").RootElement;

    private static string ToolsDirectory()
    {
        for (var dir = new DirectoryInfo(RoutingGoldenData.Directory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "tools");
            if (Directory.Exists(Path.Combine(candidate, "revision_fixtures"))) return candidate;
        }
        throw new DirectoryNotFoundException("tools/revision_fixtures not found above " + RoutingGoldenData.Directory);
    }

    private static string? Str(JsonElement e, string name) =>
        e.GetProperty(name) is { ValueKind: not JsonValueKind.Null } v ? v.GetString() : null;

    // Python's ValueError is an ArgumentException here and its FileNotFoundError a FileNotFoundException.
    private static void AssertOutcome(JsonElement result, Func<object?> run, string label, string? dir = null)
    {
        if (result.TryGetProperty("ok", out var ok))
        {
            var got = run();
            switch (ok.ValueKind)
            {
                case JsonValueKind.Null: Assert.True(got is null, label); break;
                case JsonValueKind.String: Assert.Equal(ok.GetString(), (string?)got); break;
                default: throw new InvalidOperationException(label + ": unsupported ok value");
            }
            return;
        }

        var type = result.GetProperty("error_type").GetString();
        var message = result.GetProperty("error_message").GetString()!;
        if (dir is not null) message = message.Replace("{dir}", dir);
        var ex = Assert.ThrowsAny<Exception>(() => run());
        Assert.True(type switch
        {
            "ValueError" => ex.GetType() == typeof(ArgumentException),
            "FileNotFoundError" => ex.GetType() == typeof(FileNotFoundException),
            _ => false,
        }, $"{label}: Python raised {type}, got {ex.GetType().Name}: {ex.Message}");
        Assert.Equal(message, ex.Message);
    }

    [Fact]
    public void PinnedTable_EqualsPythons()
    {
        var recorded = Probe.GetProperty("pinned_revisions").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetString()!);
        Assert.Equal(recorded.OrderBy(kv => kv.Key, StringComparer.Ordinal),
                     LayaRevisions.PinnedRevisions.OrderBy(kv => kv.Key, StringComparer.Ordinal));
        Assert.Equal(LayaRevisions.Reviewed, Probe.GetProperty("reviewed").GetString());
    }

    [Fact]
    public void ResolveRevision_MatchesPython()
    {
        var count = 0;
        foreach (var c in Probe.GetProperty("resolve").EnumerateArray())
        {
            var label = "resolve/" + c.GetProperty("label").GetString();
            using var env = new EnvScope(("LAYA_REVISION", Str(c, "env")));
            AssertOutcome(c.GetProperty("result"),
                () => LayaRevisions.ResolveRevision(c.GetProperty("model").GetString()!, Str(c, "revision")), label);
            count++;
        }
        Assert.True(count >= 30, "probe has too few resolve cases: " + count);
    }

    [Fact]
    public void VerifyDigests_MatchesPython()
    {
        var tools = ToolsDirectory();
        var count = 0;
        foreach (var c in Probe.GetProperty("verify").EnumerateArray())
        {
            var label = "verify/" + c.GetProperty("label").GetString();
            var dir = Path.Combine(tools, c.GetProperty("dir").GetString()!.Replace('/', Path.DirectorySeparatorChar));
            IReadOnlyDictionary<string, string>? expected = c.GetProperty("expected") is { ValueKind: JsonValueKind.Object } o
                ? new Dictionary<string, string>(o.EnumerateObject().Select(
                    p => new KeyValuePair<string, string>(p.Name, p.Value.GetString()!)))
                : null;
            var onnx = Str(c, "onnx_path");
            using var env = new EnvScope(("LAYA_SHA256_DIGESTS", Str(c, "env")));
            AssertOutcome(c.GetProperty("result"), () =>
            {
                LayaRevisions.VerifyDigests(dir, expected, onnx is null ? null : Path.Combine(dir, onnx));
                return null;
            }, label, dir);
            count++;
        }
        Assert.True(count >= 60, "probe has too few verify cases: " + count);
    }

    [Fact]
    public void DigestsFromEnv_MatchesPython()
    {
        var count = 0;
        foreach (var c in Probe.GetProperty("digests_from_env").EnumerateArray())
        {
            var label = "digests_from_env/" + c.GetProperty("label").GetString();
            using var env = new EnvScope(("LAYA_SHA256_DIGESTS", Str(c, "env")));
            var result = c.GetProperty("result");
            if (result.TryGetProperty("ok", out var ok))
            {
                var got = LayaRouter.DigestsFromEnv().ToDictionary(
                    kv => LayaOptions.CheckpointSubfolder(kv.Key), kv => kv.Value);
                var want = ok.EnumerateObject().ToDictionary(
                    p => p.Name,
                    p => p.Value.EnumerateObject().Select(f => (f.Name, f.Value.GetString()!)).ToArray());
                Assert.Equal(want.Keys.OrderBy(k => k, StringComparer.Ordinal), got.Keys.OrderBy(k => k, StringComparer.Ordinal));
                foreach (var (name, files) in want)
                    Assert.Equal(files, got[name].Select(f => (f.Key, f.Value)).ToArray());
            }
            else
            {
                AssertOutcome(result, () => LayaRouter.DigestsFromEnv(), label);
            }
            count++;
        }
        Assert.True(count >= 20, "probe has too few digests_from_env cases: " + count);
    }

    [Fact]
    public void MergeExpectedDigests_MatchesPython()
    {
        static Dictionary<string, string>? Map(JsonElement e) => e.ValueKind == JsonValueKind.Null
            ? null
            : e.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!);

        var count = 0;
        foreach (var c in Probe.GetProperty("merge").EnumerateArray())
        {
            var got = LayaRouter.MergeExpectedDigests(Map(c.GetProperty("shared")), Map(c.GetProperty("per_model")));
            var ok = c.GetProperty("result").GetProperty("ok");
            if (ok.ValueKind == JsonValueKind.Null)
                Assert.Null(got);
            else
                Assert.Equal(Map(ok)!.Select(kv => (kv.Key, kv.Value)).ToArray(),
                             got!.Select(kv => (kv.Key, kv.Value)).ToArray());
            count++;
        }
        Assert.True(count >= 10, "probe has too few merge cases: " + count);
    }
}
