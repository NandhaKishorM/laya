using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

namespace Laya;

/// <summary>One bucket of a histogram-binning map: <c>bins</c> equal-width bins over [0, 1] and one recalibrated confidence per bin.</summary>
internal sealed record BinningEntry(int Bins, IReadOnlyList<double> Values);

/// <summary>What <c>apply_calibration_payload</c> installs on an agent: the clamped temperatures and the binning map.</summary>
internal sealed record CalibrationPayload(
    IReadOnlyList<double> Temperature,
    IReadOnlyDictionary<string, double> TemperatureByOptions,
    IReadOnlyDictionary<string, BinningEntry>? BinningMap);

/// <summary>
/// Python's <c>ONNXAgent.load_calibration</c> (<c>onnx_agent.py:266</c>), <c>apply_calibration_payload</c>
/// (<c>calibrate.py:754</c>) and <c>apply_binning_map</c> (<c>calibrate.py:567</c>). Only the reading and applying side
/// is ported: the fitters and <c>save_calibration</c> are not.
/// </summary>
/// <remarks>
/// Every refusal is an <see cref="InvalidDataException"/> whose message is the Python exception's type and text
/// (<c>ValueError: ...</c>), as <see cref="LayaConfig"/> reports a malformed checkpoint config.
/// </remarks>
internal static partial class CalibrationFile
{
    // int(str): [sign] digits, underscores only between digits. The input is already reduced to ASCII.
    [GeneratedRegex(@"^[+-]?[0-9]+(?:_[0-9]+)*$")]
    private static partial Regex IntShape();

    /// <summary><c>open(path)</c>, <c>json.load</c>, then <see cref="Apply"/>.</summary>
    internal static CalibrationPayload Load(string path) =>
        Apply(PythonJsonReader.Parse(PythonJsonReader.ReadAllText(path)));

    /// <summary>
    /// <c>apply_calibration_payload(agent, payload)</c>: validate the shape of <paramref name="payload"/>, and return the temperatures (clamped, with the clamp warning)
    /// and the binning map (<see langword="null"/> when the payload holds none) to install. Warnings go to <paramref name="warn"/>,
    /// by default <see cref="WarnOnce"/>.
    /// </summary>
    internal static CalibrationPayload Apply(object? payload, Action<string>? warn = null)
    {
        warn ??= WarnOnce.Warn;
        if (payload is not IReadOnlyDictionary<string, object?> root)
            throw Refused($"calibration JSON must be an object, got {PythonFloat.TypeName(payload)}");

        // The version is validated, but a version 2 file's recorded checkpoint (`_warn_if_identity_mismatch`) is not compared:
        // Python's `model_id_or_path` is the string its caller passed, which this SDK has no equivalent of.
        _ = ReadVersion(root);

        if (!root.TryGetValue("temperature", out var temps) || temps is not List<object?> { Count: 3 } temperature)
            throw Refused("calibration JSON must contain temperature: [3 floats]");

        // An absent key and an explicit null are the same: a file from before bucket temperatures.
        IReadOnlyDictionary<string, object?> byOptions = new Dictionary<string, object?>();
        if (root.TryGetValue("temperature_by_options", out var rawByOptions) && rawByOptions is not null)
        {
            if (rawByOptions is not IReadOnlyDictionary<string, object?> map)
                throw Refused(
                    "calibration JSON temperature_by_options must be an object of bucket -> float, "
                    + $"got {PythonFloat.TypeName(rawByOptions)}");
            byOptions = map;
        }

        var binning = ReadBinningMap(root);


        var (applied, appliedByOptions) = InstallTemperatures(temperature, byOptions, warn);
        return new CalibrationPayload(applied, appliedByOptions, binning);
    }

    /// <summary>
    /// <c>apply_binning_map(confidence, bucket, binning_map)</c>: the recalibrated confidence for the bin <paramref name="confidence"/>
    /// falls in, or <paramref name="confidence"/> itself when the map holds no entry for <paramref name="bucket"/>.
    /// </summary>
    internal static double ApplyBinning(double confidence, string bucket, IReadOnlyDictionary<string, BinningEntry>? map)
    {
        if (map is null || !map.TryGetValue(bucket, out var entry)) return confidence;
        // `int(float(confidence) * bins)` truncates toward zero, and refuses a NaN or an infinity.
        var scaled = confidence * entry.Bins;
        if (double.IsNaN(scaled)) throw new InvalidDataException("ValueError: cannot convert float NaN to integer");
        if (double.IsInfinity(scaled)) throw new InvalidDataException("OverflowError: cannot convert float infinity to integer");
        var truncated = Math.Truncate(scaled);
        var bin = truncated <= 0 ? 0 : truncated >= entry.Bins - 1 ? entry.Bins - 1 : (int)truncated;
        return entry.Values[bin];
    }

    // ── version ──────────────────────────────────────────────────────────────

    private static BigInteger ReadVersion(IReadOnlyDictionary<string, object?> root)
    {
        // payload.get("version", 1), and an explicit null is 1 as well.
        var raw = root.TryGetValue("version", out var found) && found is not null ? found : 1L;
        var message = $"calibration JSON version must be an integer, got {PythonFloat.Repr(raw)}";
        switch (raw)
        {
            case long l: return l;
            case BigInteger big: return big;
            // `int()` truncates, which once read a 1.9 as version 1; a whole number written `2.0` still loads.
            case double d: return double.IsFinite(d) && d == Math.Floor(d) ? new BigInteger(d) : throw Refused(message);
            case string s: return TryParseInt(s, out var parsed) ? parsed : throw Refused(message);
            default: throw Refused(message);   // a bool, list or object
        }
    }

    /// <summary><c>int(text)</c> for a string: Unicode whitespace stripped, Unicode decimal digits read as their value.</summary>
    private static bool TryParseInt(string text, out BigInteger value)
    {
        value = BigInteger.Zero;
        var ascii = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            if (rune.Value < 128) ascii.Append((char)rune.Value);
            else if (PythonStr.IsSpace(rune.Value)) ascii.Append(' ');
            else
            {
                var digit = CharUnicodeInfo.GetDecimalDigitValue(rune.ToString(), 0);
                if (digit < 0 || Rune.GetUnicodeCategory(rune) != UnicodeCategory.DecimalDigitNumber) return false;
                ascii.Append((char)('0' + digit));
            }
        }
        var s = ascii.ToString().Trim(' ', '\t', '\n', '\v', '\f', '\r');
        if (!IntShape().IsMatch(s)) return false;
        var digits = s.TrimStart('+', '-').Replace("_", "");
        // Python 3.11+ refuses to turn more digits than sys.int_info.default_max_str_digits into an int.
        if (digits.Length > 4300) return false;
        value = BigInteger.Parse(digits, CultureInfo.InvariantCulture);
        if (s[0] == '-') value = -value;
        return true;
    }

    // ── binning map ──────────────────────────────────────────────────────────

    private static Dictionary<string, BinningEntry>? ReadBinningMap(IReadOnlyDictionary<string, object?> root)
    {
        if (!root.TryGetValue("binning_map", out var raw) || raw is null) return null;
        if (raw is not IReadOnlyDictionary<string, object?> map)
            throw Refused(
                "calibration JSON binning_map must be an object of bucket -> {bins, values}, "
                + $"got {PythonFloat.TypeName(raw)}");

        var result = new Dictionary<string, BinningEntry>(StringComparer.Ordinal);
        foreach (var (name, entryRaw) in map)
        {
            var at = $"calibration JSON binning_map[{PythonStr.Repr(name)}]";
            if (entryRaw is not IReadOnlyDictionary<string, object?> entry)
                throw Refused($"{at} must be an object with \"bins\" and \"values\", got {PythonFloat.TypeName(entryRaw)}");

            entry.TryGetValue("bins", out var binsRaw);
            entry.TryGetValue("values", out var valuesRaw);
            BigInteger bins = binsRaw switch { long l => l, BigInteger b => b, _ => BigInteger.Zero };   // a bool is not an int here
            if (binsRaw is not (long or BigInteger) || bins < 1)
                throw Refused($"{at} must have an integer \"bins\" >= 1, got {PythonFloat.Repr(binsRaw)}");
            if (valuesRaw is not List<object?> values || values.Count != bins)
                throw Refused($"{at} must have \"values\" of length \"bins\" ({bins.ToString(CultureInfo.InvariantCulture)})");

            var numbers = $"{at} values must be numbers, got {PythonFloat.Repr(values)}";
            // `[float(v) for v in values]` runs over every value before any type check: a value float() refuses is the
            // ValueError, and an integer too large for a float is an OverflowError that nothing catches.
            var parsed = new double[values.Count];
            for (var i = 0; i < values.Count; i++)
            {
                if (values[i] is BigInteger big && double.IsInfinity((double)big))
                    throw new InvalidDataException("OverflowError: int too large to convert to float");
                if (!PythonFloat.TryConvert(values[i], out parsed[i])) throw Refused(numbers);
            }
            foreach (var v in values)
                if (v is bool || v is not (long or BigInteger or double)) throw Refused(numbers);
            foreach (var v in parsed)
            {
                if (!double.IsFinite(v)) throw Refused($"{at} values must be finite, got {PythonFloat.Repr(values)}");
                if (!(0.0 <= v && v <= 1.0)) throw Refused($"{at} values must be in [0, 1], got {PythonFloat.Repr(values)}");
            }
            result[name] = new BinningEntry((int)bins, parsed);
        }
        return result;
    }

    // ── temperatures ─────────────────────────────────────────────────────────

    /// <summary><c>_install_temperatures</c>: clamp every entry, and name the ones the clamp had to replace.</summary>
    private static (double[] Temperature, Dictionary<string, double> ByOptions) InstallTemperatures(
        List<object?> raw, IReadOnlyDictionary<string, object?> rawByOptions, Action<string> warn)
    {
        // An integer too large for a float is an OverflowError inside `clamp_temperature` and `temperature_rejected`,
        // which both read it as "does not convert": 1.0, and named in the warning digit for digit. TryConvert reads
        // it as infinity, which the clamp answers with 1.0 and the rejection check reports.
        var applied = raw.Select(Calibration.ClampTemperature).ToArray();
        var appliedByOptions = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (key, value) in rawByOptions) appliedByOptions[key] = Calibration.ClampTemperature(value);

        var rejected = new List<string>();
        for (var i = 0; i < raw.Count; i++)
            if (Calibration.TemperatureRejected(raw[i], applied[i]))
                rejected.Add($"temperature[{i}]={PythonFloat.Repr(raw[i])} -> {PythonFloat.FormatG(applied[i])}");
        foreach (var (key, value) in rawByOptions)
            if (Calibration.TemperatureRejected(value, appliedByOptions[key]))
                rejected.Add($"{key}={PythonFloat.Repr(value)} -> {PythonFloat.FormatG(appliedByOptions[key])}");
        if (rejected.Count > 0)
            warn(
                $"laya: calibration temperatures are invalid or outside [{PythonFloat.FormatG(Calibration.TempMin)}, "
                + $"{PythonFloat.FormatG(Calibration.TempMax)}]; using {string.Join(", ", rejected)}. "
                + "Treat confidence from the affected entries as uncalibrated.");
        return (applied, appliedByOptions);
    }

    private static InvalidDataException Refused(string text) => new($"ValueError: {text}");
}
