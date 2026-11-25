using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

namespace Laya;

/// <summary>Python's <c>float(value)</c> and <c>repr(value)</c> for the values a JSON document can hold.</summary>
internal static partial class PythonFloat
{
    // float(str): [sign] (digits [. [digits]] | . digits) [e [sign] digits], underscores only between
    // digits, or inf / infinity / nan in any case. The input is already reduced to ASCII.
    [GeneratedRegex(@"^[+-]?(?:(?:[0-9]+(?:_[0-9]+)*)(?:\.(?:[0-9]+(?:_[0-9]+)*)?)?|\.[0-9]+(?:_[0-9]+)*)(?:[eE][+-]?[0-9]+(?:_[0-9]+)*)?$")]
    private static partial Regex NumberShape();

    /// <summary>
    /// <c>float(text)</c> for a <see cref="string"/>: <see langword="false"/> where Python raises
    /// <c>ValueError</c>. Unicode whitespace is stripped and Unicode decimal digits are read as
    /// their ASCII value, as <c>_PyUnicode_TransformDecimalAndSpaceToASCII</c> does; only the
    /// ASCII whitespace characters are stripped, so a control character such as U+001F is rejected.
    /// </summary>
    internal static bool TryParse(string text, out double value)
    {
        value = 0;
        var ascii = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            if (rune.Value < 128) ascii.Append((char)rune.Value);   // ASCII is left as it is, U+001C..U+001F included
            else if (PythonStr.IsSpace(rune.Value)) ascii.Append(' ');
            else
            {
                var digit = CharUnicodeInfo.GetDecimalDigitValue(rune.ToString(), 0);
                if (digit < 0 || Rune.GetUnicodeCategory(rune) != UnicodeCategory.DecimalDigitNumber) return false;
                ascii.Append((char)('0' + digit));
            }
        }
        var s = ascii.ToString().Trim(' ', '\t', '\n', '\v', '\f', '\r');
        if (s.Length == 0) return false;

        var body = s[0] is '+' or '-' ? s[1..] : s;
        var negative = s[0] == '-';
        if (body.Equals("inf", StringComparison.OrdinalIgnoreCase) || body.Equals("infinity", StringComparison.OrdinalIgnoreCase))
        {
            value = negative ? double.NegativeInfinity : double.PositiveInfinity;
            return true;
        }
        if (body.Equals("nan", StringComparison.OrdinalIgnoreCase))
        {
            value = double.NaN;
            return true;
        }
        if (!NumberShape().IsMatch(s)) return false;
        return double.TryParse(s.Replace("_", ""), NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    /// <summary>
    /// <c>float(raw)</c> for a JSON-derived value; <see langword="false"/> where Python raises
    /// <c>TypeError</c> or <c>ValueError</c> (<see langword="null"/>, lists, dicts, non-numeric strings).
    /// A <see cref="bool"/> is converted too, as <c>float(True)</c> is 1.0; callers that refuse it say so first.
    /// </summary>
    internal static bool TryConvert(object? raw, out double value)
    {
        switch (raw)
        {
            case bool b: value = b ? 1.0 : 0.0; return true;
            case double d: value = d; return true;
            case string s: return TryParse(s, out value);
            // Any number float() accepts. An integer rounds to nearest, ties to even, as Python's int -> float does;
            // double.Parse reads the exact decimal text and rounds the same way, which a cast from decimal or from
            // the 64-bit unsigned types does not promise.
            case nint n: value = n; return true;
            case long or int or short or sbyte or byte or ushort or uint:
                value = Convert.ToInt64(raw, CultureInfo.InvariantCulture);
                return true;
            case ulong or nuint or BigInteger or decimal or Int128 or UInt128:
                value = double.Parse(Convert.ToString(raw, CultureInfo.InvariantCulture)!, NumberStyles.Float, CultureInfo.InvariantCulture);
                return true;
            case float f: value = f; return true;
            case Half h: value = (double)h; return true;
            default: value = 0; return false;
        }
    }

    /// <summary>Python's <c>repr()</c> of a JSON-derived value (<c>None</c>, <c>True</c>, int, float, str, list, dict).</summary>
    internal static string Repr(object? value) => value switch
    {
        null => "None",
        bool b => b ? "True" : "False",
        long l => l.ToString(CultureInfo.InvariantCulture),
        BigInteger big => big.ToString(CultureInfo.InvariantCulture),
        double d => double.IsNaN(d) ? "nan" : double.IsPositiveInfinity(d) ? "inf" : double.IsNegativeInfinity(d) ? "-inf" : PythonJson.Repr(d),
        string s => PythonStr.Repr(s),
        IReadOnlyList<object?> list => "[" + string.Join(", ", list.Select(Repr)) + "]",
        IEnumerable<KeyValuePair<string, object?>> map => "{" + string.Join(", ", map.Select(kv => PythonStr.Repr(kv.Key) + ": " + Repr(kv.Value))) + "}",
        _ => value.ToString() ?? "None",
    };

    /// <summary>Python's <c>type(value).__name__</c> for a JSON-derived value.</summary>
    internal static string TypeName(object? value) => value switch
    {
        null => "NoneType",
        bool => "bool",
        long or BigInteger => "int",
        double => "float",
        string => "str",
        List<object?> => "list",
        _ => "dict",
    };

    /// <summary>Python's <c>"%g"</c> for a temperature already clamped into [0.5, 5].</summary>
    internal static string FormatG(double value) => value.ToString("G6", CultureInfo.InvariantCulture);
}
