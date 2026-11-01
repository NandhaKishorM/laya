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
                       IReadOnlyList<object?> temperatureValues,
                       IReadOnlyDictionary<string, object?> temperatureByOptionsValues,
                       object? temperatureSource, Action<LayaConfig>? beforeWarning)
    {
        TemperatureSource = temperatureSource;
        TemperatureRawValues = temperatureValues;
        TemperatureByOptionsRawValues = temperatureByOptionsValues;
        var temperatureRaw = temperatureValues.Select(AsDouble).ToList();
        var temperatureByOptionsRaw = temperatureByOptionsValues
            .ToDictionary(kv => kv.Key, kv => AsDouble(kv.Value), StringComparer.Ordinal);
        Encoder = encoder;
        MaxLen = maxLen;
        HeadMaxLen = headMaxLen;
        TemperatureRaw = temperatureRaw;
        TemperatureByOptionsRaw = temperatureByOptionsRaw;

        // clamp_temperature on every entry, with the JSON value as Python holds it (onnx_agent.py:249-252).
        Temperature = temperatureValues.Select(Calibration.ClampTemperature).ToList();
        TemperatureByOptions = temperatureByOptionsValues
            .ToDictionary(kv => kv.Key, kv => Calibration.ClampTemperature(kv.Value), StringComparer.Ordinal);

        // onnx_agent.py:254-262: "<name>=<repr of the raw value> -> <applied, %g>", by-options entries first.
        var rejected = new List<string>();
        foreach (var kv in temperatureByOptionsValues)
            if (Calibration.TemperatureRejected(kv.Value, TemperatureByOptions[kv.Key]))
                rejected.Add($"{kv.Key}={PythonFloat.Repr(kv.Value)} -> {PythonFloat.FormatG(TemperatureByOptions[kv.Key])}");
        for (var i = 0; i < temperatureValues.Count; i++)
            if (Calibration.TemperatureRejected(temperatureValues[i], Temperature[i]))
                rejected.Add($"temperature[{i}]={PythonFloat.Repr(temperatureValues[i])} -> {PythonFloat.FormatG(Temperature[i])}");
        ClampedTemperatures = rejected;

        // Python resolves `lang_temperatures` between reading these temperatures and warning about them
        // (onnx_agent.py:249 before :254), so an invalid table refuses the load before this warning is shown and
        // its own warnings come first.
        beforeWarning?.Invoke(this);

        // Mirrors Python's RuntimeWarning, one warning listing every rejected entry, shown once per process
        // for the same text (WarnOnce); it goes through System.Diagnostics.Trace so a caller can suppress
        // it by removing trace listeners.
        if (rejected.Count > 0)
            WarnOnce.Warn(
                $"laya ONNX: this checkpoint ships temperatures outside [{PythonFloat.FormatG(Calibration.TempMin)}, "
                + $"{PythonFloat.FormatG(Calibration.TempMax)}] which would distort "
                + $"confidence; clamping {string.Join(", ", rejected)}. Treat confidence from the affected buckets as uncalibrated.");
    }

    // The numeric view of a raw JSON value kept for TemperatureRaw: a number as itself, a numeric string
    // as float() reads it, anything else NaN.
    private static double AsDouble(object? value) => PythonFloat.TryConvert(value, out var d) && value is not bool ? d : double.NaN;

    /// <summary>The encoder the checkpoint was trained on, for information only.</summary>
    public string? Encoder { get; }

    /// <summary>Maximum total sequence length.</summary>
    public int MaxLen { get; }

    /// <summary>Token budget for the question head: instructions plus all option fragments.</summary>
    public int HeadMaxLen { get; }

    /// <summary>
    /// Per-question-type temperature, clamped to the usable range: the entries of the config's <c>temperature</c>, three
    /// for a shipped checkpoint. A shorter list is kept as it is, and a prediction that needs the missing entry fails as
    /// Python's does (<c>IndexError: list index out of range</c>).
    /// </summary>
    public IReadOnlyList<double> Temperature { get; }

    /// <summary>Per <c>(type, option-count)</c> bucket temperature, clamped. Keys as <see cref="Calibration.TempBucket"/> builds them.</summary>
    public IReadOnlyDictionary<string, double> TemperatureByOptions { get; }

    /// <summary>
    /// What the checkpoint shipped, before clamping, as numbers: an entry that is not a number
    /// (and not a numeric string <c>float()</c> accepts) reads <see cref="double.NaN"/>. See
    /// <see cref="TemperatureRawValues"/> for the values themselves.
    /// </summary>
    public IReadOnlyList<double> TemperatureRaw { get; }

    /// <summary>
    /// The <c>temperature</c> entries exactly as the JSON held them: a <see cref="long"/> or
    /// <see cref="System.Numerics.BigInteger"/> for an integer, a <see cref="double"/>, a
    /// <see cref="string"/>, a <see cref="bool"/>, <see langword="null"/>, or a list / key-value list.
    /// </summary>
    public IReadOnlyList<object?> TemperatureRawValues { get; }

    /// <summary>The <c>temperature_by_options</c> entries exactly as the JSON held them; see <see cref="TemperatureRawValues"/>.</summary>
    public IReadOnlyDictionary<string, object?> TemperatureByOptionsRawValues { get; }

    /// <summary>What the checkpoint shipped, before clamping.</summary>
    public IReadOnlyDictionary<string, double> TemperatureByOptionsRaw { get; }

    /// <summary>
    /// Descriptions of every temperature that had to be clamped, empty when none were. A
    /// non-empty list means confidence from those buckets is uncalibrated and worth surfacing.
    /// Each entry also triggers a <c>System.Diagnostics.Trace.TraceWarning</c> at load
    /// time, mirroring Python's <c>RuntimeWarning</c> for the same condition.
    /// </summary>
    public IReadOnlyList<string> ClampedTemperatures { get; }

    /// <summary>
    /// <c>cfg.get("temperature", [1.0, 1.0, 1.0])</c> as Python holds it, before it is iterated: a list, or the string or
    /// dictionary a malformed config held (<see cref="TemperatureRawValues"/> then lists its characters or keys).
    /// What an inheriting language override is checked against.
    /// </summary>
    internal object? TemperatureSource { get; }

    /// <summary>Read a config from a <c>rl_agent_config.json</c> file.</summary>
    public static LayaConfig Load(string path) => Parse(PythonJsonReader.ReadAllText(path));

    /// <summary><see cref="Load(string)"/> running <paramref name="beforeWarning"/> where Python resolves <c>lang_temperatures</c>.</summary>
    internal static LayaConfig Load(string path, Action<LayaConfig>? beforeWarning) =>
        Parse(PythonJsonReader.ReadAllText(path), beforeWarning);

    /// <summary>Parse a config from JSON text, read as Python's <c>json.loads</c> reads it.</summary>
    /// <exception cref="InvalidDataException">
    /// Where Python's loader raises: a config that is not an object, an unsupported <c>option_layout</c>, a
    /// <c>temperature</c> that is not iterable, a <c>temperature_by_options</c> that is not a mapping, or an
    /// integer temperature too large for a float. The message carries Python's exception type and text.
    /// </exception>
    public static LayaConfig Parse(string json) => Parse(json, null);

    internal static LayaConfig Parse(string json, Action<LayaConfig>? beforeWarning)
    {
        var parsed = PythonJsonReader.Parse(json);
        if (parsed is not IReadOnlyDictionary<string, object?> root)
            throw new InvalidDataException($"AttributeError: '{PythonFloat.TypeName(parsed)}' object has no attribute 'get'");

        // onnx_agent.py:193: only the sequential layout runs on the exported graph.
        if (root.TryGetValue("option_layout", out var layout) && layout is not "sequential")
            throw new InvalidDataException(
                $"ValueError: option_layout={PythonFloat.Repr(layout)} is not supported by the ONNX runtime; load this checkpoint with laya.Agent");

        // `cfg.get("temperature", [1.0, 1.0, 1.0])`, then `[clamp_temperature(t) for t in temperature_raw]`: whatever
        // the key holds is iterated, so a string yields its characters and a dict its keys, while a number, bool or
        // null raises TypeError (onnx_agent.py:247-250).
        var temperature = new List<object?>();
        object? temperatureSource;
        if (root.TryGetValue("temperature", out var temp)) { temperature.AddRange(Iterate(temp)); temperatureSource = temp; }
        else { temperature.AddRange([1.0, 1.0, 1.0]); temperatureSource = new List<object?>(temperature); }
        // A list shorter than three is kept as it is: Python indexes self.temperature[qtype] when a question is
        // answered and raises IndexError there, which Calibration.ResolveTemperature reproduces.

        // `{k: clamp_temperature(v) for k, v in temperature_by_options_raw.items()}`: a mapping or AttributeError.
        var byOptions = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (root.TryGetValue("temperature_by_options", out var tbo))
        {
            if (tbo is not IEnumerable<KeyValuePair<string, object?>> map)
                throw new InvalidDataException($"AttributeError: '{PythonFloat.TypeName(tbo)}' object has no attribute 'items'");
            foreach (var (key, value) in map)
                byOptions[key] = value;
        }

        var lengths = Lengths(root);
        return new LayaConfig(
            root.TryGetValue("encoder", out var enc) && enc is string encoder ? encoder : null,
            lengths.MaxLen,
            lengths.HeadMaxLen,
            temperature,
            byOptions,
            temperatureSource,
            beforeWarning);
    }

    // `for t in raw`: a list's items, a string's characters (code points), a dict's keys.
    private static IEnumerable<object?> Iterate(object? raw)
    {
        switch (raw)
        {
            case List<object?> list: return list;
            case string s: return CodePointStrings(s);
            case IEnumerable<KeyValuePair<string, object?>> map: return map.Select(kv => (object?)kv.Key).ToList();
            default: throw new InvalidDataException($"TypeError: '{PythonFloat.TypeName(raw)}' object is not iterable");
        }
    }

    private static List<object?> CodePointStrings(string s)
    {
        var items = new List<object?>();
        for (var i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) { items.Add(s.Substring(i, 2)); i++; }
            else items.Add(s[i].ToString());
        }
        return items;
    }

    // `cfg.get("max_len", 512)` and `cfg.get("head_max_len", 192)`. Python keeps whatever the JSON held and fails at
    // the first prediction that does arithmetic or slicing with it; this port refuses a value it cannot use when the
    // config is loaded, with the TypeError Python raises at that prediction (build_sequence meets head_max_len first,
    // then max_len). An int >= 0 or a bool (true is 1, false is 0, as in Python) is accepted. Anything else is refused:
    // a float, string, null, list or dict, and a negative int (Python slices from the end for it, which this port
    // does not support).
    private static (int MaxLen, int HeadMaxLen) Lengths(IReadOnlyDictionary<string, object?> root)
    {
        var maxLen = root.TryGetValue("max_len", out var maxRaw) ? maxRaw : 512L;
        var headMaxLen = root.TryGetValue("head_max_len", out var headRaw) ? headRaw : 192L;
        var left = PythonFloat.TypeName(maxLen);
        var right = PythonFloat.TypeName(headMaxLen);
        const string SliceError = "TypeError: slice indices must be integers or None or have an __index__ method";
        string Operand(string kind) => $"TypeError: unsupported operand type(s) for -: '{kind}' and 'int'";
        var leftBad = left is not ("int" or "bool");
        var rightBad = right is not ("int" or "bool");
        if (rightBad) throw new InvalidDataException(right == "float" ? SliceError : Operand(right));
        if (leftBad) throw new InvalidDataException(left == "float" ? SliceError : Operand(left));
        return (Length("max_len", maxLen), Length("head_max_len", headMaxLen));
    }

    private static int Length(string key, object? v) => v switch
    {
        bool b => b ? 1 : 0,
        long l when l < 0 => throw NegativeLength(key, l.ToString(System.Globalization.CultureInfo.InvariantCulture)),
        long l => (int)Math.Min(l, int.MaxValue),
        System.Numerics.BigInteger big when big.Sign < 0 => throw NegativeLength(key, big.ToString(System.Globalization.CultureInfo.InvariantCulture)),
        _ => int.MaxValue,   // an integer beyond 64 bits
    };

    private static InvalidDataException NegativeLength(string key, string literal) =>
        new($"{key}={literal}: must be a non-negative integer");
}
