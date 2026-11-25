using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// The per-call options and the <c>lang_temperatures</c> table a recorded golden case carries
/// (<c>laya-dotnet/tools/golden_cases.py</c>), read into this SDK's types.
/// </summary>
internal static class CallOptions
{
    /// <summary>The language a recorded <c>options</c> object names, or null.</summary>
    internal static string? Lang(JsonElement options) =>
        options.TryGetProperty("lang", out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    /// <summary>
    /// A recorded token budget: a number, or the bool Python took for the int 0 or 1 (<c>True</c> is
    /// <c>1</c>), or null when the case sets none.
    /// </summary>
    internal static int? Budget(JsonElement options, string name) =>
        !options.TryGetProperty(name, out var value) ? null : value.ValueKind switch
        {
            JsonValueKind.True => 1,
            JsonValueKind.False => 0,
            JsonValueKind.Null => null,
            _ => value.GetInt32(),
        };

    /// <summary>The options a recorded <c>system_one</c> case passes to <see cref="LayaEngine.Predict(object?, QuestionSet, LayaPredictOptions?)"/>.</summary>
    internal static LayaPredictOptions Predict(JsonElement root)
    {
        if (!root.TryGetProperty("options", out var options)) return new LayaPredictOptions();
        return new LayaPredictOptions
        {
            MinConfidence = ConfidenceGateTests.MinConfidenceIn(options),
            Lang = Lang(options),
            MaxLen = Budget(options, "max_len"),
            HeadMaxLen = Budget(options, "head_max_len"),
        };
    }

    /// <summary>The <c>lang_temperatures</c> table of a recorded case, in the order it was written, or null if it has none.</summary>
    internal static IEnumerable<KeyValuePair<string, LayaLangTemperature?>>? LangTemperatures(JsonElement root)
    {
        if (!root.TryGetProperty("lang_temperatures", out var table) || table.ValueKind != JsonValueKind.Object) return null;
        return [.. table.EnumerateObject().Select(p => new KeyValuePair<string, LayaLangTemperature?>(p.Name, Entry(p.Value)))];
    }

    /// <summary>One recorded entry: <c>null</c> inherits, otherwise an object with optional <c>temperature</c> and <c>temperature_by_options</c>.</summary>
    internal static LayaLangTemperature? Entry(JsonElement entry)
    {
        if (entry.ValueKind != JsonValueKind.Object) return null;
        return new LayaLangTemperature
        {
            Temperature = entry.TryGetProperty("temperature", out var t) && t.ValueKind == JsonValueKind.Array
                ? [.. t.EnumerateArray().Select(Number)]
                : null,
            TemperatureByOptions = entry.TryGetProperty("temperature_by_options", out var b) && b.ValueKind == JsonValueKind.Object
                ? b.EnumerateObject().ToDictionary(p => p.Name, p => Number(p.Value), StringComparer.Ordinal)
                : null,
        };
    }

    /// <summary>A number, or one of the strings the dumper uses for the non-finite ones.</summary>
    internal static double Number(JsonElement value) => value.ValueKind == JsonValueKind.String
        ? value.GetString() switch
        {
            "nan" => double.NaN,
            "inf" => double.PositiveInfinity,
            "-inf" => double.NegativeInfinity,
            var other => throw new InvalidOperationException($"unexpected number text {other}"),
        }
        : value.GetDouble();
}
