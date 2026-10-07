namespace Laya;

using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.Json;

/// <summary>
/// JSON serializer that byte-for-byte reproduces Python's <c>json.dumps</c> output, including
/// Python-style float repr, ensure-ASCII escaping, and the three prompt call-site dialects.
/// </summary>
/// <remarks>
/// <para>
/// <c>System.Text.Json</c> defaults diverge from Python on every axis: no spaces, HTML-escaping,
/// and <c>2</c> instead of <c>2.0</c>. This class writes JSON manually to match Python exactly.
/// </para>
/// <para>
/// All three entry points differ only in <c>escapeNonAscii</c> and <c>fallbackToStr</c>, and all
/// three pass a <c>string</c> through verbatim without quoting — reproducing the <c>isinstance(v,
/// str)</c> fast-path in <c>serialize_state</c>, <c>render_criterion</c>, and <c>_to_internal</c>.
/// </para>
/// </remarks>
public static class PythonJson
{
    /// <summary>
    /// Matches Python's <c>serialize_state</c>: a <see cref="string"/> passes through verbatim;
    /// everything else — including <see langword="null"/> (<c>"null"</c>), booleans, numbers,
    /// dicts, and lists — is serialized with <c>ensure_ascii=False</c>.
    /// </summary>
    /// <remarks>
    /// Python's <c>serialize_state</c> is typed as <c>Union[str, dict, list]</c> but its
    /// implementation is <c>json.dumps(state)</c> for every non-string, which accepts any JSON-
    /// serializable value at runtime. This port mirrors that runtime behaviour exactly.
    /// </remarks>
    public static string State(object? value)
    {
        if (value is string s) return s;
        return Dumps(value, escapeNonAscii: false, fallbackToStr: false);
    }

    /// <summary>
    /// Matches Python's <c>render_criterion</c>: a <see cref="string"/> passes through verbatim;
    /// anything else is serialized with <c>ensure_ascii=False</c> and <c>default=str</c>.
    /// </summary>
    public static string Criterion(object? value)
    {
        if (value is string s) return s;
        return Dumps(value, escapeNonAscii: false, fallbackToStr: true);
    }

    /// <summary>
    /// Matches Python's <c>_to_internal</c> instructions path: a <see cref="string"/> passes
    /// through verbatim; anything else is serialized with <c>ensure_ascii=False</c>.
    /// </summary>
    /// <remarks>
    /// 0.3.6 used <c>ensure_ascii=True</c>, escaping non-ASCII to literal <c>\uXXXX</c> before the
    /// tokenizer ever saw it. 0.3.21's <c>agent.py</c> <c>_to_internal</c> switched to
    /// <c>json.dumps(ins, ensure_ascii=False)</c> — same dialect as <see cref="State"/> and
    /// <see cref="Criterion"/> — because the escaped form measurably changed answers (see the
    /// upstream comment: an English-checkpoint question answered differently for an escaped dict
    /// vs. the equivalent plain string). This port now matches 0.3.21.
    /// </remarks>
    public static string Instructions(object? value)
    {
        if (value is string s) return s;
        return Dumps(value, escapeNonAscii: false, fallbackToStr: false);
    }

    /// <summary>
    /// The underlying serializer, equivalent to <c>json.dumps(value, ensure_ascii=escapeNonAscii,
    /// separators=(", ", ": "), default=str if fallbackToStr)</c>.
    /// </summary>
    /// <param name="value">The value to serialize.</param>
    /// <param name="escapeNonAscii">
    /// When <see langword="true"/>, matches Python's <c>ensure_ascii=True</c>: every non-ASCII
    /// character is escaped as <c>\uXXXX</c> (or a surrogate pair for non-BMP characters).
    /// </param>
    /// <param name="fallbackToStr">
    /// When <see langword="true"/>, matches Python's <c>default=str</c>: values with no JSON
    /// equivalent are stringified via <c>ToString()</c>. When <see langword="false"/>, they
    /// throw <see cref="JsonException"/>, mirroring Python's <c>TypeError</c>.
    /// </param>
    /// <exception cref="JsonException">
    /// If <paramref name="value"/> contains a circular reference, or if a value has no JSON
    /// representation and <paramref name="fallbackToStr"/> is <see langword="false"/>.
    /// </exception>
    public static string Dumps(object? value, bool escapeNonAscii = false, bool fallbackToStr = false)
    {
        var sb = new StringBuilder();
        // Track containers to catch circular references — Python raises ValueError for those.
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        WriteValue(sb, value, escapeNonAscii, fallbackToStr, seen);
        return sb.ToString();
    }

    /// <summary>
    /// Formats a <see cref="double"/> exactly as Python's <c>repr(float)</c>: shortest round-trip
    /// decimal, always containing a <c>.</c> or <c>e</c>, exponent-notation threshold at ±16/−5.
    /// </summary>
    /// <remarks>
    /// C# <c>ToString("R")</c> gives the same digits but different formatting thresholds and
    /// uppercase <c>E</c>, so we normalise the output here.
    /// </remarks>
    public static string Repr(double value)
    {
        if (double.IsNaN(value)) return "NaN";
        if (double.IsPositiveInfinity(value)) return "Infinity";
        if (double.IsNegativeInfinity(value)) return "-Infinity";
        // Negative zero must remain -0.0 so the prompt matches Python exactly.
        if (value == 0.0) return double.IsNegative(value) ? "-0.0" : "0.0";

        string raw = value.ToString("R", CultureInfo.InvariantCulture);
        return ReformatAsPython(raw);
    }

    // ── private helpers ────────────────────────────────────────────────────

    private static bool IsDictLike(object? v) =>
        v is IDictionary or IEnumerable<KeyValuePair<string, object?>>;

    private static bool IsListLike(object? v) =>
        v is IEnumerable and not string && !IsDictLike(v);

    /// <summary>
    /// Whether <paramref name="value"/> is the .NET analogue of Python's <c>isinstance(state,
    /// list)</c>: any non-string, non-dict-like enumerable. There is no exact "list vs. tuple"
    /// distinction on this side, so anything <see cref="Dumps"/> would render as a JSON array
    /// counts, matching how <c>serialize_state</c> already treats the value.
    /// </summary>
    public static bool IsListState(object? value) => IsListLike(value);

    private static void WriteValue(
        StringBuilder sb, object? value,
        bool escapeNonAscii, bool fallbackToStr,
        HashSet<object> seen)
    {
        switch (value)
        {
            case null:
                sb.Append("null");
                break;

            case bool b:
                sb.Append(b ? "true" : "false");
                break;

            case string s:
                WriteString(sb, s, escapeNonAscii);
                break;

            // Integer types — render as plain decimal digits.
            case long l:
                sb.Append(l.ToString(CultureInfo.InvariantCulture));
                break;
            case int i:
                sb.Append(i.ToString(CultureInfo.InvariantCulture));
                break;
            case short sh:
                sb.Append(sh.ToString(CultureInfo.InvariantCulture));
                break;
            case byte by:
                sb.Append(by.ToString(CultureInfo.InvariantCulture));
                break;
            case ulong ul:
                sb.Append(ul.ToString(CultureInfo.InvariantCulture));
                break;
            case uint ui:
                sb.Append(ui.ToString(CultureInfo.InvariantCulture));
                break;
            case System.Numerics.BigInteger bi:
                sb.Append(bi.ToString(CultureInfo.InvariantCulture));
                break;

            // Float types — use Python repr rules.
            case double d:
                sb.Append(Repr(d));
                break;
            case float f:
                sb.Append(Repr(f));
                break;
            case decimal dec:
                sb.Append(Repr((double)dec));
                break;

            // Dictionaries: more-specific IEnumerable<KVP> before general IDictionary before general IEnumerable.
            case IEnumerable<KeyValuePair<string, object?>> kvpEnum:
                WriteKvpSequence(sb, kvpEnum, escapeNonAscii, fallbackToStr, seen);
                break;

            case IDictionary dict:
                WriteGenericDict(sb, dict, escapeNonAscii, fallbackToStr, seen);
                break;

            case IEnumerable seq:
                WriteArray(sb, seq, escapeNonAscii, fallbackToStr, seen);
                break;

            default:
                if (!fallbackToStr)
                    throw new JsonException(
                        $"Object of type {value!.GetType().FullName} is not JSON serializable");
                WriteString(sb, value is IFormattable fmt
                    ? fmt.ToString(null, CultureInfo.InvariantCulture)
                    : value.ToString() ?? string.Empty,
                    escapeNonAscii);
                break;
        }
    }

    private static void WriteKvpSequence(
        StringBuilder sb, IEnumerable<KeyValuePair<string, object?>> pairs,
        bool escapeNonAscii, bool fallbackToStr, HashSet<object> seen)
    {
        if (!seen.Add(pairs))
            throw new JsonException("Circular reference detected");
        try
        {
            sb.Append('{');
            bool first = true;
            foreach (var (k, v) in pairs)
            {
                if (!first) sb.Append(", ");
                first = false;
                WriteString(sb, k, escapeNonAscii);
                sb.Append(": ");
                WriteValue(sb, v, escapeNonAscii, fallbackToStr, seen);
            }
            sb.Append('}');
        }
        finally { seen.Remove(pairs); }
    }

    private static void WriteGenericDict(
        StringBuilder sb, IDictionary dict,
        bool escapeNonAscii, bool fallbackToStr, HashSet<object> seen)
    {
        if (!seen.Add(dict))
            throw new JsonException("Circular reference detected");
        try
        {
            sb.Append('{');
            bool first = true;
            foreach (DictionaryEntry entry in dict)
            {
                if (!first) sb.Append(", ");
                first = false;
                // Python coerces non-str dict keys through a repr-like conversion.
                string keyStr = entry.Key switch
                {
                    string s => s,
                    bool b => b ? "true" : "false",
                    double d => Repr(d),
                    float f => Repr(f),
                    null => "null",
                    var k => k.ToString() ?? string.Empty,
                };
                WriteString(sb, keyStr, escapeNonAscii);
                sb.Append(": ");
                WriteValue(sb, entry.Value, escapeNonAscii, fallbackToStr, seen);
            }
            sb.Append('}');
        }
        finally { seen.Remove(dict); }
    }

    private static void WriteArray(
        StringBuilder sb, IEnumerable seq,
        bool escapeNonAscii, bool fallbackToStr, HashSet<object> seen)
    {
        if (!seen.Add(seq))
            throw new JsonException("Circular reference detected");
        try
        {
            sb.Append('[');
            bool first = true;
            foreach (var item in seq)
            {
                if (!first) sb.Append(", ");
                first = false;
                WriteValue(sb, item, escapeNonAscii, fallbackToStr, seen);
            }
            sb.Append(']');
        }
        finally { seen.Remove(seq); }
    }

    private static void WriteString(StringBuilder sb, string s, bool escapeNonAscii)
    {
        sb.Append('"');
        // Iterate by char (UTF-16 code units). Surrogates are escaped individually when
        // escapeNonAscii is true, matching Python's \uXXXX surrogate-pair output.
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < 0x20)
                    {
                        // Other control characters: \u00XX with lowercase hex.
                        sb.Append("\\u");
                        sb.Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else if (escapeNonAscii && c > 0x7f)
                    {
                        // Non-ASCII: \uXXXX. Surrogates are escaped as-is (each half separately),
                        // reproducing Python's output for characters above U+FFFF.
                        sb.Append("\\u");
                        sb.Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(c);
                    }
                    break;
            }
        }
        sb.Append('"');
    }

    /// <summary>
    /// Converts the shortest-round-trip decimal string from <c>ToString("R")</c> into the form
    /// Python's <c>repr(float)</c> would produce.
    /// </summary>
    private static string ReformatAsPython(string raw)
    {
        bool neg = raw[0] == '-';
        // Index into raw where the unsigned mantissa starts.
        int start = neg ? 1 : 0;

        // Locate the exponent marker (C# uses uppercase E).
        int eIdx = raw.IndexOfAny(s_expChars, start);
        int baseExp = 0;
        string mantissa;
        if (eIdx >= 0)
        {
            baseExp = int.Parse(raw.AsSpan(eIdx + 1), NumberStyles.AllowLeadingSign,
                CultureInfo.InvariantCulture);
            mantissa = raw[start..eIdx];
        }
        else
        {
            mantissa = raw[start..];
        }

        // Split mantissa into integer and fractional parts.
        int dotIdx = mantissa.IndexOf('.');
        string intPart = dotIdx >= 0 ? mantissa[..dotIdx] : mantissa;
        string fracPart = dotIdx >= 0 ? mantissa[(dotIdx + 1)..] : string.Empty;

        // Derive the decimal exponent of the most significant digit and the significant-digit string.
        string digits;
        int exp10;

        bool intIsZero = intPart is "" or "0";
        if (intIsZero && baseExp == 0)
        {
            // Fixed form like "0.0001" — find the first non-zero fractional digit.
            int leadZeros = 0;
            while (leadZeros < fracPart.Length && fracPart[leadZeros] == '0') leadZeros++;
            digits = fracPart[leadZeros..];
            exp10 = -leadZeros - 1;
        }
        else
        {
            // Form with a non-zero integer part, e.g. "1.5" (from "1.5E+21") or "123.456".
            // exp10 must be computed from intPart.Length BEFORE any digit trimming.
            exp10 = baseExp + intPart.Length - 1;
            digits = intPart + fracPart;
        }

        // Trim trailing zeros that the "R" formatter may leave in the mantissa. Ryu does not
        // emit trailing zeros, but trimming defensively keeps exp10 unaffected (computed above).
        digits = digits.TrimEnd('0');
        if (digits.Length == 0) digits = "0";

        // Python uses exponent notation when the decimal exponent is >= 16 or <= -5.
        string prefix = neg ? "-" : string.Empty;
        if (exp10 >= 16 || exp10 <= -5)
        {
            string mantissaStr = digits.Length == 1
                ? digits
                : digits[..1] + "." + digits[1..];
            // Exponent is always signed, minimum 2 digits.
            string expStr = exp10 >= 0
                ? $"e+{exp10:D2}"
                : $"e-{Math.Abs(exp10):D2}";
            return prefix + mantissaStr + expStr;
        }
        else if (exp10 >= 0)
        {
            // Fixed notation, number >= 1.
            int intDigitCount = exp10 + 1;
            if (digits.Length <= intDigitCount)
            {
                // All digits are in the integer part; pad with zeros and append ".0".
                return prefix + digits.PadRight(intDigitCount, '0') + ".0";
            }
            return prefix + digits[..intDigitCount] + "." + digits[intDigitCount..];
        }
        else
        {
            // Fixed notation, number in (0, 1) — prefix with "0." and the required leading zeros.
            int leadingZeros = -exp10 - 1;
            return prefix + "0." + new string('0', leadingZeros) + digits;
        }
    }

    private static readonly char[] s_expChars = ['E', 'e'];
}
