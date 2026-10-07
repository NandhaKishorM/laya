using System.Text.Json;

namespace Laya;

/// <summary>
/// The contents of a checkpoint's <c>rl_agent_config.json</c>.
/// </summary>
/// <remarks>
/// Defaults match Python's <c>cfg.get(...)</c> fallbacks in <c>Agent.system_one</c>, not the
/// shipped multilingual values, so a config missing a key behaves the same in both runtimes.
/// </remarks>
public sealed class LayaConfig
{
    private LayaConfig(string? encoder, int maxLen, int headMaxLen,
                       IReadOnlyList<double> temperatureRaw,
                       IReadOnlyDictionary<string, double> temperatureByOptionsRaw)
    {
        Encoder = encoder;
        MaxLen = maxLen;
        HeadMaxLen = headMaxLen;
        TemperatureRaw = temperatureRaw;
        TemperatureByOptionsRaw = temperatureByOptionsRaw;

        Temperature = temperatureRaw.Select(Calibration.ClampTemperature).ToList();
        TemperatureByOptions = temperatureByOptionsRaw
            .ToDictionary(kv => kv.Key, kv => Calibration.ClampTemperature(kv.Value), StringComparer.Ordinal);

        var rejected = new List<string>();
        foreach (var kv in temperatureByOptionsRaw)
            if (Calibration.ClampTemperature(kv.Value) != kv.Value)
                rejected.Add($"{kv.Key}={kv.Value:G4}");
        for (var i = 0; i < temperatureRaw.Count; i++)
            if (Calibration.ClampTemperature(temperatureRaw[i]) != temperatureRaw[i])
                rejected.Add($"temperature[{i}]={temperatureRaw[i]:G4}");
        ClampedTemperatures = rejected;

        // Mirrors Python's RuntimeWarning. One warning per clamped value so a log scan finds
        // them without needing the property; use System.Diagnostics.Trace so the caller can
        // suppress them by removing trace listeners rather than needing a new dependency.
        foreach (var desc in rejected)
            System.Diagnostics.Trace.TraceWarning(
                "Laya: temperature {0} is outside [{1}, {2}] and was clamped — " +
                "confidence from that bucket is uncalibrated",
                desc, Calibration.TempMin, Calibration.TempMax);
    }

    /// <summary>The encoder the checkpoint was trained on, for information only.</summary>
    public string? Encoder { get; }

    /// <summary>Maximum total sequence length.</summary>
    public int MaxLen { get; }

    /// <summary>Token budget for the question head: instructions plus all option fragments.</summary>
    public int HeadMaxLen { get; }

    /// <summary>Per-question-type temperature, clamped to the usable range. Always 3 entries.</summary>
    public IReadOnlyList<double> Temperature { get; }

    /// <summary>Per <c>(type, option-count)</c> bucket temperature, clamped. Keys as <see cref="Calibration.TempBucket"/> builds them.</summary>
    public IReadOnlyDictionary<string, double> TemperatureByOptions { get; }

    /// <summary>What the checkpoint shipped, before clamping.</summary>
    public IReadOnlyList<double> TemperatureRaw { get; }

    /// <summary>What the checkpoint shipped, before clamping.</summary>
    public IReadOnlyDictionary<string, double> TemperatureByOptionsRaw { get; }

    /// <summary>
    /// Descriptions of every temperature that had to be clamped, empty when none were. A
    /// non-empty list means confidence from those buckets is uncalibrated and worth surfacing.
    /// Each entry also triggers a <c>System.Diagnostics.Trace.TraceWarning</c> at load
    /// time, mirroring Python's <c>RuntimeWarning</c> for the same condition.
    /// </summary>
    public IReadOnlyList<string> ClampedTemperatures { get; }

    /// <summary>Read a config from a <c>rl_agent_config.json</c> file.</summary>
    public static LayaConfig Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Parse a config from JSON text.</summary>
    public static LayaConfig Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var temperature = new List<double>();
        if (root.TryGetProperty("temperature", out var temp) && temp.ValueKind == JsonValueKind.Array)
            foreach (var t in temp.EnumerateArray())
                temperature.Add(t.ValueKind == JsonValueKind.Number ? t.GetDouble() : double.NaN);
        // Python indexes self.temperature[qtype] for qtype 0..2, so a short list would be an
        // IndexError there; pad to 3 with the neutral value rather than failing later per-question.
        while (temperature.Count < 3) temperature.Add(1.0);

        var byOptions = new Dictionary<string, double>(StringComparer.Ordinal);
        if (root.TryGetProperty("temperature_by_options", out var tbo) && tbo.ValueKind == JsonValueKind.Object)
            foreach (var p in tbo.EnumerateObject())
                byOptions[p.Name] = p.Value.ValueKind == JsonValueKind.Number ? p.Value.GetDouble() : double.NaN;

        return new LayaConfig(
            root.TryGetProperty("encoder", out var enc) && enc.ValueKind == JsonValueKind.String
                ? enc.GetString()
                : null,
            root.TryGetProperty("max_len", out var ml) && ml.ValueKind == JsonValueKind.Number ? ml.GetInt32() : 512,
            root.TryGetProperty("head_max_len", out var hml) && hml.ValueKind == JsonValueKind.Number
                ? hml.GetInt32()
                : 192,
            temperature,
            byOptions);
    }
}
