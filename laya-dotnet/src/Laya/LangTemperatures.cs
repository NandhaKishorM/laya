namespace Laya;

/// <summary>
/// One language's entry of <see cref="LayaOptions.LangTemperatures"/>: Python's
/// <c>{"temperature": [3 floats] | None, "temperature_by_options": {bucket: float}}</c> for one key of
/// <c>lang_temperatures</c>. A <see langword="null"/> entry in the table, or a <see langword="null"/>
/// <see cref="Temperature"/>, inherits the checkpoint's own temperatures.
/// </summary>
public sealed class LayaLangTemperature
{
    /// <summary>
    /// The temperature per question type, in <c>choice</c>, <c>score</c>, <c>noul</c> order: exactly three values.
    /// <see langword="null"/> inherits the checkpoint's own. Every value is clamped to the usable range
    /// (<see cref="Calibration.TempMin"/>..<see cref="Calibration.TempMax"/>), and a value the clamp moved is
    /// named in a warning, exactly as for the checkpoint's own temperatures.
    /// </summary>
    public IReadOnlyList<double>? Temperature { get; init; }

    /// <summary>
    /// Per <c>(type, option-count)</c> bucket temperature, keyed as <see cref="Calibration.TempBucket"/> builds
    /// them (<c>"choice:3-5"</c>); a bucket it does not name uses <see cref="Temperature"/>. Clamped like
    /// <see cref="Temperature"/>. <see langword="null"/> is the same as empty.
    /// </summary>
    public IReadOnlyDictionary<string, double>? TemperatureByOptions { get; init; }
}

/// <summary>A language's temperatures after <c>resolve_lang_temperatures</c>: clamped, and complete.</summary>
internal sealed record ResolvedLangTemperature(
    IReadOnlyList<double> Temperature, IReadOnlyDictionary<string, double> TemperatureByOptions);

/// <summary>
/// Python's <c>resolve_lang_temperatures</c> (<c>laya/common.py</c>) and the language lookup of
/// <c>_decode_answers</c> (<c>onnx_agent.py:728</c>, <c>:743</c>).
/// </summary>
internal static class LangTemperatures
{
    /// <summary><c>lang.split("-")[0].lower()</c>.</summary>
    internal static string Normalise(string lang) => PythonStr.Lower(lang.Split('-', 2)[0]);

    /// <summary>
    /// The override <paramref name="lang"/> selects, or <see langword="null"/>: <c>lang and lang.split("-")[0].lower() in
    /// self.lang_temperatures</c>, so an empty language never matches.
    /// </summary>
    internal static ResolvedLangTemperature? Find(
        IReadOnlyDictionary<string, ResolvedLangTemperature> table, string? lang) =>
        !string.IsNullOrEmpty(lang) && table.TryGetValue(Normalise(lang), out var found) ? found : null;

    /// <summary>
    /// Parse <paramref name="raw"/> into <c>{language: resolved}</c>. <paramref name="baseSource"/> is the
    /// checkpoint's <c>temperature</c> as the config held it, which an entry without its own inherits;
    /// <paramref name="baseTemperature"/> is the same, clamped.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Python's <c>ValueError</c>, with its text, for a language whose temperature is not three values (or, inherited,
    /// the checkpoint's is not a list of three). The other shapes Python refuses (a key that is not a string, an entry
    /// or <c>temperature_by_options</c> that is not a mapping) cannot be written in the typed table.
    /// </exception>
    internal static IReadOnlyDictionary<string, ResolvedLangTemperature> Resolve(
        IEnumerable<KeyValuePair<string, LayaLangTemperature?>>? raw,
        object? baseSource, IReadOnlyList<double> baseTemperature)
    {
        var resolved = new Dictionary<string, ResolvedLangTemperature>(StringComparer.Ordinal);
        if (raw is null) return resolved;

        foreach (var (lang, entry) in raw)
        {
            if (lang is null)
                throw new ArgumentException("Language override keys must be strings, got None");
            var norm = Normalise(lang);

            var tRaw = entry?.Temperature;
            var inherited = tRaw is null;
            // `t_raw = base_temperature` keeps the checkpoint's own list, whatever it holds, so the shape check and
            // its message read the config's value; an own list is a typed list of doubles.
            if (inherited)
            {
                if (baseSource is not List<object?> { Count: 3 })
                    throw new ArgumentException(
                        $"Language override {PythonStr.Repr(lang)} temperature must be a list of 3 floats, got "
                        + PythonFloat.Repr(baseSource));
            }
            else if (tRaw!.Count != 3)
                throw new ArgumentException(
                    $"Language override {PythonStr.Repr(lang)} temperature must be a list of 3 floats, got "
                    + "[" + string.Join(", ", tRaw.Select(v => PythonFloat.Repr(v))) + "]");

            var byOptionsRaw = entry?.TemperatureByOptions;
            List<double> temperature = inherited
                ? [.. baseTemperature]
                : [.. tRaw!.Select(Calibration.ClampTemperature)];
            var byOptions = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var (bucket, value) in byOptionsRaw ?? new Dictionary<string, double>())
                byOptions[bucket] = Calibration.ClampTemperature(value);

            // An override that omits `temperature` inherits the checkpoint's own list, and the loader that got here has
            // already reported the entries it replaced in it; repeating that line per language would bury the overrides.
            var rejected = new List<string>();
            if (!inherited)
                for (var i = 0; i < 3; i++)
                    if (Calibration.TemperatureRejected(tRaw![i], temperature[i]))
                        rejected.Add($"temperature[{i}]={PythonFloat.Repr(tRaw[i])} -> {PythonFloat.FormatG(temperature[i])}");
            foreach (var (bucket, value) in byOptionsRaw ?? new Dictionary<string, double>())
                if (Calibration.TemperatureRejected(value, byOptions[bucket]))
                    rejected.Add($"{bucket}={PythonFloat.Repr(value)} -> {PythonFloat.FormatG(byOptions[bucket])}");
            if (rejected.Count > 0)
                WarnOnce.Warn(
                    $"laya: language override {PythonStr.Repr(lang)} ships invalid temperatures or values outside "
                    + $"[{PythonFloat.FormatG(Calibration.TempMin)}, {PythonFloat.FormatG(Calibration.TempMax)}]; "
                    + $"using {string.Join(", ", rejected)}. Treat confidence from the affected entries as uncalibrated.");

            resolved[norm] = new ResolvedLangTemperature(temperature, byOptions);
        }
        return resolved;
    }
}
