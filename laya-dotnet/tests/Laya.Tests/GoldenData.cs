using System.Text.Json;

namespace Laya.Tests;

/// <summary>One entry of <c>index.json</c>.</summary>
/// <param name="Kind">
/// <see langword="null"/> for a standard parity case; <c>"shortlist"</c> for a case recorded
/// through <c>predict_shortlist</c> rather than <c>system_one</c> directly. Standard case
/// enumeration excludes non-null kinds, since their <c>result</c> shape and recording flow differ
/// from the plain parity cases <see cref="CheckpointGoldenData.SuccessCases"/> /
/// <see cref="CheckpointGoldenData.ErrorCases"/> are meant for.
/// </param>
public sealed record GoldenCaseInfo(string Name, string File, bool ExpectsError, string? Kind = null)
{
    /// <inheritdoc/>
    public override string ToString() => Name;
}

/// <summary>
/// Access to per-checkpoint parity vectors recorded by <c>laya-dotnet/tools/dump_golden.py</c> running the
/// real <c>laya.Agent.system_one</c> over an ONNX-backed forward.
/// </summary>
/// <remarks>
/// Golden files are read from the source tree rather than copied to the output directory: they are
/// the reference, and a stale copy beside the test binary would let a real divergence pass.
/// </remarks>
public sealed class CheckpointGoldenData
{
    private readonly LayaCheckpoint _checkpoint;
    private readonly Lazy<string?> _directoryLazy;
    private readonly Lazy<IReadOnlyList<GoldenCaseInfo>> _indexLazy;
    // The document is intentionally never disposed: it backs the Meta property's JsonElement for
    // the process lifetime, matching the same pattern used by the original GoldenData static class.
    private readonly Lazy<JsonDocument> _metaDoc;

    internal CheckpointGoldenData(LayaCheckpoint checkpoint)
    {
        _checkpoint = checkpoint;
        _directoryLazy = new(Locate);
        _indexLazy = new(LoadIndex);
        _metaDoc = new(() => Load("meta.json"));
    }

    // ── availability ──────────────────────────────────────────────────────────

    /// <summary>Whether any golden data was found for this checkpoint.</summary>
    public bool Available => DirectoryOrNull is not null;

    /// <summary>The golden directory, or <see langword="null"/> if none was found.</summary>
    public string? DirectoryOrNull => _directoryLazy.Value;

    /// <summary>
    /// The golden directory. Throws <see cref="DirectoryNotFoundException"/> if not found.
    /// Use <see cref="Available"/> to guard; this throws a clear error rather than a
    /// <see cref="NullReferenceException"/>.
    /// </summary>
    public string Directory => DirectoryOrNull
        ?? throw new DirectoryNotFoundException(
            $"no golden data found for '{_checkpoint}' above {AppContext.BaseDirectory}. "
            + "Run `python laya-dotnet/tools/dump_golden.py` to generate the golden vectors.");

    // ── data ──────────────────────────────────────────────────────────────────

    /// <summary>Every recorded case, in the order the dump script emitted them.</summary>
    public IReadOnlyList<GoldenCaseInfo> Index => _indexLazy.Value;

    /// <summary>The recording environment: versions, config, and the special token ids.</summary>
    public JsonElement Meta => _metaDoc.Value.RootElement;

    /// <summary>Standard (non-<see cref="GoldenCaseInfo.Kind"/>) cases that record a successful prediction.</summary>
    public IEnumerable<object[]> SuccessCases() =>
        Index.Where(c => c.Kind is null && !c.ExpectsError).Select(c => new object[] { c });

    /// <summary>Standard (non-<see cref="GoldenCaseInfo.Kind"/>) cases that record a Python exception.</summary>
    public IEnumerable<object[]> ErrorCases() =>
        Index.Where(c => c.Kind is null && c.ExpectsError).Select(c => new object[] { c });

    /// <summary>
    /// Cases recorded through <c>predict_shortlist</c> (<see cref="GoldenCaseInfo.Kind"/> <c>"shortlist"</c>).
    /// </summary>
    public IEnumerable<object[]> ShortlistCases() =>
        Index.Where(c => c.Kind == "shortlist").Select(c => new object[] { c });

    /// <summary>Parse one golden file. The caller owns the returned document.</summary>
    public JsonDocument Load(string fileName) =>
        JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Directory, fileName)));

    /// <summary>Parse the file for one case.</summary>
    public JsonDocument Load(GoldenCaseInfo info) => Load(info.File);

    // ── helpers ───────────────────────────────────────────────────────────────

    private IReadOnlyList<GoldenCaseInfo> LoadIndex()
    {
        using var doc = Load("index.json");
        return
        [
            .. doc.RootElement.EnumerateArray().Select(e => new GoldenCaseInfo(
                e.GetProperty("name").GetString()!,
                e.GetProperty("file").GetString()!,
                e.GetProperty("expects_error").GetBoolean(),
                e.TryGetProperty("kind", out var kind) ? kind.GetString() : null))
        ];
    }

    private string? Locate()
    {
        var subfolder = LayaOptions.CheckpointSubfolder(_checkpoint);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            // Layout: golden/<checkpoint>/index.json
            var candidate = Path.Combine(dir.FullName, "golden", subfolder);
            if (File.Exists(Path.Combine(candidate, "index.json"))) return candidate;
        }
        return null;
    }
}

/// <summary>
/// Access to the parity vectors in <c>golden/</c>, recorded by <c>laya-dotnet/tools/dump_golden.py</c> running
/// the real <c>laya.Agent.system_one</c> over an ONNX-backed forward.
/// </summary>
/// <remarks>
/// The goldens are read from the source tree rather than copied to the output directory: they are
/// the reference, and a stale copy beside the test binary would let a real divergence pass.
/// </remarks>
public static class GoldenData
{
    // One instance per LayaCheckpoint, indexed by the enum's integer value.
    private static readonly CheckpointGoldenData[] ByCheckpoint =
        Enum.GetValues<LayaCheckpoint>().Select(c => new CheckpointGoldenData(c)).ToArray();

    /// <summary>Returns per-checkpoint golden data for <paramref name="checkpoint"/>.</summary>
    public static CheckpointGoldenData For(LayaCheckpoint checkpoint) =>
        ByCheckpoint[(int)checkpoint];

    // ── backward-compat shims pointing at the multilingual checkpoint ─────────

    /// <summary>The multilingual golden directory in the source tree.</summary>
    public static string Directory => For(LayaCheckpoint.Multilingual).Directory;

    /// <summary>Every recorded multilingual case, in the order the dump script emitted them.</summary>
    public static IReadOnlyList<GoldenCaseInfo> Index => For(LayaCheckpoint.Multilingual).Index;

    /// <summary>The multilingual recording environment: versions, config, and special token ids.</summary>
    public static JsonElement Meta => For(LayaCheckpoint.Multilingual).Meta;

    /// <summary>Multilingual cases that record a successful prediction.</summary>
    public static IEnumerable<object[]> SuccessCases() =>
        For(LayaCheckpoint.Multilingual).SuccessCases();

    /// <summary>Multilingual cases that record a Python exception.</summary>
    public static IEnumerable<object[]> ErrorCases() =>
        For(LayaCheckpoint.Multilingual).ErrorCases();

    /// <summary>Multilingual cases recorded through <c>predict_shortlist</c>.</summary>
    public static IEnumerable<object[]> ShortlistCases() =>
        For(LayaCheckpoint.Multilingual).ShortlistCases();

    /// <summary>Parse one multilingual golden file. The caller owns the returned document.</summary>
    public static JsonDocument Load(string fileName) =>
        For(LayaCheckpoint.Multilingual).Load(fileName);

    /// <summary>Parse the multilingual file for one case.</summary>
    public static JsonDocument Load(GoldenCaseInfo info) =>
        For(LayaCheckpoint.Multilingual).Load(info);

    // ── pure utilities (checkpoint-independent) ────────────────────────────────

    /// <summary>
    /// A golden JSON value as the CLR object Python held when it recorded the case.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Objects become <see cref="OrderedDictionary{TKey, TValue}"/>, because key order survives a
    /// round trip through Python's <c>dict</c> and decides both the serialized state text and, for
    /// choice criteria, which label each logit is attached to.
    /// </para>
    /// <para>
    /// Numbers are split the way <c>json.load</c> splits them: a bare integer literal becomes
    /// <see cref="long"/> and anything with a fraction or exponent becomes <see cref="double"/>.
    /// The distinction is visible in the prompt, since <c>json.dumps</c> writes <c>2</c> for one and
    /// <c>2.0</c> for the other.
    /// </para>
    /// </remarks>
    public static object? ToClr(JsonElement element)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                var map = new OrderedDictionary<string, object?>(StringComparer.Ordinal);
                foreach (var p in element.EnumerateObject()) map[p.Name] = ToClr(p.Value);
                return map;

            case JsonValueKind.Array:
                var list = new List<object?>();
                foreach (var item in element.EnumerateArray()) list.Add(ToClr(item));
                return list;

            case JsonValueKind.String:
                return element.GetString();

            case JsonValueKind.Number:
                var raw = element.GetRawText();
                // The (object) cast is load-bearing: without it the conditional's type unifies to
                // double, so an integer would be widened before boxing and every int in a state or
                // criterion would serialize as "3.0" where Python writes "3" -- a different token
                // stream, and so a different answer, with nothing failing loudly.
                return raw.AsSpan().IndexOfAny('.', 'e', 'E') >= 0
                    ? element.GetDouble()
                    : (object)element.GetInt64();

            case JsonValueKind.True:
                return true;

            case JsonValueKind.False:
                return false;

            case JsonValueKind.Null or JsonValueKind.Undefined:
                return null;

            default:
                throw new ArgumentOutOfRangeException(nameof(element), element.ValueKind, "unexpected JSON kind");
        }
    }

    /// <summary>Rebuild a case's <see cref="QuestionSet"/> from its recorded definitions.</summary>
    public static QuestionSet BuildQuestions(JsonElement questions)
    {
        var set = new QuestionSet();
        foreach (var entry in questions.EnumerateObject())
        {
            var def = entry.Value;
            var type = def.GetProperty("type").GetString();
            var instructions = ToClr(def.GetProperty("instructions"))
                ?? throw new InvalidOperationException($"question '{entry.Name}' has null instructions");
            var hasCriteria = def.TryGetProperty("criteria", out var criteria);

            set[entry.Name] = type switch
            {
                "choice" => BuildChoice(entry.Name, instructions, criteria),
                "score" => Question.Score(instructions,
                    criteria.EnumerateArray().Select(ToClr)),
                "noul" => BuildNoul(instructions, hasCriteria ? criteria : default),
                _ => throw new InvalidOperationException($"unknown question type '{type}'"),
            };
        }
        return set;
    }

    private static Question BuildChoice(string id, object instructions, JsonElement criteria) => criteria.ValueKind switch
    {
        // Python normalizes a list of labels to {label: None} in Agent._to_internal.
        JsonValueKind.Array => Question.Choice(instructions,
            criteria.EnumerateArray().Select(e => new KeyValuePair<string, object?>(
                e.GetString() ?? throw new InvalidOperationException($"question '{id}' has a non-string label"),
                null))),
        JsonValueKind.Object => Question.Choice(instructions,
            criteria.EnumerateObject().Select(p => new KeyValuePair<string, object?>(p.Name, ToClr(p.Value)))),
        _ => throw new InvalidOperationException($"question '{id}' has unusable criteria"),
    };

    private static Question BuildNoul(object instructions, JsonElement criteria)
    {
        if (criteria.ValueKind != JsonValueKind.Object) return Question.Noul(instructions);
        var ifFalse = criteria.TryGetProperty("false", out var f) ? ToClr(f) : null;
        var ifTrue = criteria.TryGetProperty("true", out var t) ? ToClr(t) : null;
        return Question.Noul(instructions, ifFalse, ifTrue);
    }

    /// <summary>A recorded <c>[rows][cols]</c> float matrix, flattened row-major.</summary>
    public static float[] Matrix(JsonElement element, out int rows, out int cols)
    {
        rows = element.GetArrayLength();
        cols = rows == 0 ? 0 : element[0].GetArrayLength();
        var flat = new float[rows * cols];
        var i = 0;
        foreach (var row in element.EnumerateArray())
            foreach (var value in row.EnumerateArray())
                flat[i++] = value.GetSingle();
        return flat;
    }

    /// <summary>A recorded integer array.</summary>
    public static int[] Ints(JsonElement element) => [.. element.EnumerateArray().Select(e => e.GetInt32())];
}
