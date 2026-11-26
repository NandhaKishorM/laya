using System.Security.Cryptography;
using System.Text.Json;

namespace Laya;

/// <summary>
/// Supply-chain integrity for published checkpoints: opt-in revision pins and SHA-256 digest checks.
/// Ports Python's <c>laya/revisions.py</c> (<c>PINNED_REVISIONS</c>, <c>resolve_revision</c>,
/// <c>verify_digests</c>) with the same semantics and, where an error has a Python text, the same text.
/// Python's <c>ValueError</c> is an <see cref="ArgumentException"/> here and its
/// <c>FileNotFoundError</c> a <see cref="FileNotFoundException"/>.
/// </summary>
public static class LayaRevisions
{
    /// <summary>
    /// Opt-in reviewed commit SHAs of the published checkpoints, mirroring Python's
    /// <c>PINNED_REVISIONS</c>. They are not applied implicitly; pass <see cref="Reviewed"/> as
    /// <see cref="LayaOptions.Revision"/> (or set <c>LAYA_REVISION=reviewed</c>) to pin a load.
    /// </summary>
    public static IReadOnlyDictionary<string, string> PinnedRevisions { get; } =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["convaiinnovations/laya"] = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
            ["convaiinnovations/laya-multilingual"] = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
            ["convaiinnovations/laya-typed-decisions"] = "1a793eb568e6718f15941d08f85432581df534e3",
        };

    /// <summary>
    /// The <c>LAYA_REVISION</c> value meaning "the reviewed SHA of the repository being loaded".
    /// </summary>
    public const string Reviewed = "reviewed";

    private static string? ReviewedPin(string modelIdOrPath)
    {
        if (PinnedRevisions.TryGetValue(modelIdOrPath, out var pinned)) return pinned;
        var wanted = PythonStr.Lower(modelIdOrPath);
        foreach (var (repo, sha) in PinnedRevisions)
            if (PythonStr.Lower(repo) == wanted) return sha;
        return null;
    }

    /// <summary>
    /// Picks the revision to download: <paramref name="revision"/> when it is non-empty, else
    /// <c>LAYA_REVISION</c>, stripped; empty means no pin (<see langword="null"/>). The word
    /// <c>reviewed</c> looks <paramref name="modelIdOrPath"/> up in <see cref="PinnedRevisions"/>,
    /// case-insensitively, and refuses a repository that has no entry. A non-empty but blank
    /// <paramref name="revision"/> wins and strips away to no pin, so it suppresses the environment
    /// variable, exactly as in Python.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// <c>reviewed</c> was asked for a repository with no reviewed SHA.
    /// </exception>
    public static string? ResolveRevision(string modelIdOrPath, string? revision = null)
    {
        ArgumentNullException.ThrowIfNull(modelIdOrPath);
        var value = PythonStr.Strip(!string.IsNullOrEmpty(revision)
            ? revision
            : Environment.GetEnvironmentVariable("LAYA_REVISION") ?? "");
        if (value.Length == 0) return null;
        if (value != Reviewed) return value;

        var pinned = ReviewedPin(modelIdOrPath);
        if (pinned is not null) return pinned;
        var repo = PythonStr.Repr(modelIdOrPath);
        if (!string.IsNullOrEmpty(revision))
            throw new ArgumentException(
                $"laya: revision= asked for 'reviewed', but {repo} has no reviewed SHA in "
                + "PINNED_REVISIONS; pass a commit, branch or tag in revision= instead");
        throw new ArgumentException(
            $"laya: LAYA_REVISION=reviewed, but {repo} has no reviewed SHA in "
            + "PINNED_REVISIONS; set LAYA_REVISION to a commit or unset it");
    }

    /// <summary>
    /// Verifies the SHA-256 digests of files under <paramref name="modelDir"/> against
    /// <c>{relative path: hex digest}</c>. <see langword="null"/> reads <c>LAYA_SHA256_DIGESTS</c>
    /// (a JSON object; unset or blank verifies nothing) and an empty map masks it. The keys
    /// <c>onnx</c> and <c>onnx_path</c> name <paramref name="onnxPath"/> when it is given.
    /// </summary>
    /// <exception cref="FileNotFoundException">A listed file is absent.</exception>
    /// <exception cref="ArgumentException">
    /// A digest mismatch, an unsafe (absolute or escaping) path, or a malformed environment value.
    /// </exception>
    public static void VerifyDigests(
        string modelDir, IReadOnlyDictionary<string, string>? expected = null, string? onnxPath = null)
        => Verify(modelDir, expected, onnxPath, onnxRefusal: null);

    // `onnxRefusal`, when set, is the message for an "onnx"/"onnx_path" key on a checkpoint that has
    // no single fused model.onnx (the split layout). Python has no such layout.
    internal static void Verify(
        string modelDir, IReadOnlyDictionary<string, string>? expected, string? onnxPath, string? onnxRefusal)
    {
        ArgumentNullException.ThrowIfNull(modelDir);
        if (expected is null)
        {
            var raw = PythonStr.Strip(Environment.GetEnvironmentVariable("LAYA_SHA256_DIGESTS") ?? "");
            if (raw.Length == 0) return;
            expected = ParseEnvMap(raw);
        }

        foreach (var (rel, want) in expected)
        {
            string path;
            var onnxKey = rel == "onnx" || rel == "onnx_path";
            if (onnxKey && onnxRefusal is not null)
                throw new ArgumentException(onnxRefusal);
            if (onnxKey && !string.IsNullOrEmpty(onnxPath))
            {
                path = onnxPath;
            }
            else
            {
                var rawRel = rel.Replace('\\', '/');
                if (IsAbsolute(rawRel))
                    throw new ArgumentException(
                        "laya: unsafe absolute path in expected digests: " + PythonStr.Repr(rel));
                var relNorm = rawRel.TrimStart('/');
                if (relNorm.Length == 0 || relNorm == ".." || relNorm.StartsWith("../", StringComparison.Ordinal)
                    || relNorm.Contains("/../", StringComparison.Ordinal))
                    throw new ArgumentException(
                        "laya: unsafe path in expected digests: " + PythonStr.Repr(rel));
                path = Path.Combine(modelDir, relNorm);
            }

            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"laya: cannot verify {PythonStr.Repr(rel)}: no such file under {modelDir}");

            string got;
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
                got = Convert.ToHexStringLower(SHA256.HashData(stream));
            if (got != PythonStr.Lower(PythonStr.Strip(want)))
                throw new ArgumentException(
                    $"laya: SHA-256 mismatch for {rel}: expected {want}, got {got}. The artifact does "
                    + "not match the reviewed digest; refusing to load it.");
        }
    }

    // posixpath.isabs / ntpath.isabs on the slash-normalised text: a leading "/" (which also covers
    // UNC and device prefixes) or any one character followed by ":/" (a drive and root).
    private static bool IsAbsolute(string rawRel)
    {
        if (rawRel.StartsWith('/')) return true;
        if (rawRel.Length == 0) return false;
        var first = char.IsHighSurrogate(rawRel[0]) && rawRel.Length > 1 ? 2 : 1;
        return string.CompareOrdinal(rawRel, first, ":/", 0, 2) == 0;
    }

    // json.loads(raw) as verify_digests reads LAYA_SHA256_DIGESTS: a JSON object (anything else is the
    // "must be a mapping" error), with each value taken as str(value), the only way it is ever read.
    private static IReadOnlyDictionary<string, string> ParseEnvMap(string raw)
    {
        object? data;
        try { data = PythonJsonReader.Parse(raw); }
        catch (JsonException)
        {
            throw new ArgumentException("LAYA_SHA256_DIGESTS must be a JSON object of artifact->sha256");
        }
        if (data is not IEnumerable<KeyValuePair<string, object?>> map)
            throw new ArgumentException("expected_sha256 must be a mapping of artifact paths to digests");
        var result = new OrderedDictionary<string, string>(StringComparer.Ordinal);
        foreach (var (key, value) in map) result[key] = PyStr(value);
        return result;
    }

    /// <summary>Python's <c>str()</c> of a JSON-derived value.</summary>
    internal static string PyStr(object? value) => value is string s ? s : PythonFloat.Repr(value);

    /// <summary>
    /// The folder a pinned download is cached under: the revision as one path segment. A branch such as
    /// <c>feature/x</c> is percent-encoded, and trailing dots (so a segment of only dots too) are escaped
    /// so it cannot name a parent directory or be trimmed by the file system.
    /// </summary>
    internal static string RevisionFolder(string revision)
    {
        var escaped = Uri.EscapeDataString(revision);
        var trailing = escaped.Length - escaped.TrimEnd('.').Length;
        return escaped[..^trailing] + string.Concat(Enumerable.Repeat("%2E", trailing));
    }
}
