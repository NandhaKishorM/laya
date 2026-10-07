using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Access to the checkpoint-independent routing/language/email/shortlist parity vectors in
/// <c>golden/routing/</c>, recorded by <c>laya-dotnet/tools/dump_routing_golden.py</c> running the real
/// Python <c>laya.lang</c>/<c>laya.router</c>/<c>laya.email</c>/<c>laya.shortlist</c> modules —
/// no ONNX weights are needed for any of these files.
/// </summary>
/// <remarks>
/// Read from the source tree rather than copied to the output directory, same rationale as
/// <see cref="CheckpointGoldenData"/>: they are the reference, and a stale copy beside the test
/// binary would let a real divergence pass.
/// </remarks>
public static class RoutingGoldenData
{
    private static readonly Lazy<string?> DirectoryLazy = new(Locate);
    private static readonly Lazy<JsonDocument> MetaDoc = new(() => Load("meta.json"));

    /// <summary>Whether the routing golden directory was found.</summary>
    public static bool Available => DirectoryOrNull is not null;

    /// <summary>The routing golden directory, or <see langword="null"/> if none was found.</summary>
    public static string? DirectoryOrNull => DirectoryLazy.Value;

    /// <summary>
    /// The routing golden directory. Throws <see cref="DirectoryNotFoundException"/> if not found.
    /// </summary>
    public static string Directory => DirectoryOrNull
        ?? throw new DirectoryNotFoundException(
            $"no golden/routing data found above {AppContext.BaseDirectory}. "
            + "Run `python laya-dotnet/tools/dump_routing_golden.py` to generate it.");

    /// <summary>The recording environment: generator, laya version, and file counts.</summary>
    public static JsonElement Meta => MetaDoc.Value.RootElement;

    /// <summary>Parse one file under <c>golden/routing/</c>. The caller owns the returned document.</summary>
    public static JsonDocument Load(string fileName) =>
        JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Directory, fileName)));

    /// <summary>A golden JSON value as the CLR object Python held when it recorded the case.</summary>
    public static object? ToClr(JsonElement element) => GoldenData.ToClr(element);

    private static string? Locate()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "golden", "routing");
            if (File.Exists(Path.Combine(candidate, "meta.json"))) return candidate;
        }
        return null;
    }
}
