using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace Laya;

/// <summary>
/// <c>json.loads</c> as CPython's C scanner implements it, producing the values Python would hold:
/// <see cref="long"/> / <see cref="BigInteger"/> for an integer literal, <see cref="double"/> for a float
/// literal (a literal that overflows is infinity), <see cref="string"/>, <see cref="bool"/>,
/// <see langword="null"/>, <c>List&lt;object?&gt;</c> and <c>OrderedDictionary&lt;string, object?&gt;</c>.
/// </summary>
/// <remarks>
/// <see cref="JsonDocument"/> is stricter than Python in exactly one place that matters here: Python also
/// accepts the bare tokens <c>NaN</c>, <c>Infinity</c> and <c>-Infinity</c> in value position. They are
/// parsed to doubles directly, so no marker text exists that a real string could collide with. Everything
/// else follows <c>json.decoder</c>: whitespace is space, tab, LF and CR only; <c>-?(0|[1-9][0-9]*)</c> with
/// an optional fraction and exponent, so <c>01</c>, <c>+1</c>, <c>.5</c> and <c>1.</c> are errors; strings
/// reject control characters but keep lone surrogates; no trailing commas, comments or single quotes; a
/// duplicate key keeps its first position and its last value. Errors are <see cref="JsonException"/>.
/// </remarks>
internal static class PythonJsonReader
{
    // Python 3.11+ refuses to turn more digits than this into an int (sys.int_info.default_max_str_digits).
    private const int MaxIntDigits = 4300;
    // The nesting at which CPython 3.12's C scanner raises RecursionError (measured: 2997 levels load).
    internal const int MaxDepth = 2997;

    /// <summary>
    /// Read a JSON file the way Python's <c>json.load(open(path))</c> does. The reference opens these files
    /// without an <c>encoding</c>, so it decodes with the locale's: UTF-8 on Linux and macOS (and with
    /// Python's UTF-8 mode), the ANSI code page on a default Windows install. This reads UTF-8, strictly: invalid bytes are an
    /// error (<c>UnicodeDecodeError</c>), and a leading BOM is kept as U+FEFF so <see cref="Parse"/> rejects it as
    /// <c>json.loads</c> does. The two decodings agree on every ASCII file.
    /// </summary>
    internal static string ReadAllText(string path)
    {
        string text;
        try { text = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true).GetString(File.ReadAllBytes(path)); }
        catch (DecoderFallbackException e) { throw new JsonException($"{path} is not valid UTF-8: {e.Message}", e); }
        // Text mode (universal newlines): "\r\n" and a lone "\r" read as "\n", which is what json.load sees and counts lines by.
        return text.Replace("\r\n", "\n").Replace('\r', '\n');
    }

    internal static object? Parse(string text)
    {
        var p = new State(text);
        if (text.StartsWith('\uFEFF')) throw p.Error("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0);
        var value = p.Value();
        p.SkipWhitespace();
        if (p.Pos != text.Length) throw p.Error("Extra data");
        return value;
    }

    /// <summary>
    /// Parse, then re-emit as a <see cref="JsonDocument"/> for code that reads through that API. NaN and
    /// the infinities, which JSON text cannot hold, become <c>null</c>.
    /// </summary>
    internal static JsonDocument ParseDocument(string text) => ParseDocument(text, out _);

    /// <summary>
    /// <see cref="ParseDocument(string)"/> that also hands back the Python-typed value tree, so a caller can tell
    /// when a value it reads was altered by the re-emission (NaN and infinities become <c>null</c>, a lone
    /// surrogate becomes U+FFFD). Nesting up to <see cref="MaxDepth"/> levels, which Python loads, is kept.
    /// </summary>
    internal static JsonDocument ParseDocument(string text, out object? tree)
    {
        tree = Parse(text);
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { MaxDepth = MaxDepth + 8 }))
            Write(writer, tree);
        return JsonDocument.Parse(stream.ToArray(), new JsonDocumentOptions { MaxDepth = MaxDepth + 8 });
    }

    private static bool HasLoneSurrogate(string s)
    {
        for (var i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) { i++; continue; }
            if (char.IsSurrogate(s[i])) return true;
        }
        return false;
    }

    /// <summary>Whether re-emitting <paramref name="value"/> as JSON text changes what a reader of it sees.</summary>
    /// <remarks>
    /// Walks with an explicit stack, for the same reason <see cref="Parse"/> and <see cref="Write"/> do: the
    /// value may be a document at <see cref="MaxDepth"/>, and a recursive walk would be the one remaining place
    /// a deep document could exhaust the caller's stack.
    /// </remarks>
    internal static bool IsAlteredByReEmission(object? value)
    {
        var pending = new Stack<object?>();
        pending.Push(value);
        while (pending.Count > 0)
        {
            switch (pending.Pop())
            {
                case double d when !double.IsFinite(d): return true;
                case string s when HasLoneSurrogate(s): return true;
                case List<object?> list:
                    foreach (var item in list) pending.Push(item);
                    break;
                case IEnumerable<KeyValuePair<string, object?>> map:
                    foreach (var kv in map)
                    {
                        if (HasLoneSurrogate(kv.Key)) return true;
                        pending.Push(kv.Value);
                    }
                    break;
            }
        }
        return false;
    }

    /// <summary>One open container while re-emitting, with the enumerator over its children.</summary>
    private struct WriteFrame
    {
        internal IEnumerator<object?>? Items;                          // an array: its elements
        internal IEnumerator<KeyValuePair<string, object?>>? Pairs;    // an object: its members
        internal bool IsObject;
    }

    /// <summary>Re-emit <paramref name="value"/> as JSON text.</summary>
    /// <remarks>
    /// Iterative: <see cref="Utf8JsonWriter"/> keeps its own depth counter and does not recurse, so this walk was
    /// the only thing that made a deep document need stack proportional to its depth. A
    /// <c>tokenizer_config.json</c> nested at CPython's limit is re-emitted through here
    /// (<see cref="ParseDocument(string, out object?)"/>), so it must not be the call that overflows. Scalar
    /// spellings, key order and lone-surrogate replacement are unchanged.
    /// </remarks>
    private static void Write(Utf8JsonWriter w, object? value)
    {
        if (!IsContainer(value, out var items, out var pairs)) { WriteScalar(w, value); return; }
        var frames = new List<WriteFrame>();
        Open(w, items, pairs, frames);
        while (frames.Count > 0)
        {
            var top = frames.Count - 1;
            var frame = frames[top];
            if (frame.IsObject)
            {
                if (frame.Pairs!.MoveNext())
                {
                    w.WritePropertyName(ReplaceLoneSurrogates(frame.Pairs.Current.Key));
                    OpenChild(w, frame.Pairs.Current.Value, frames);
                    continue;
                }
                w.WriteEndObject();
            }
            else
            {
                if (frame.Items!.MoveNext())
                {
                    OpenChild(w, frame.Items.Current, frames);
                    continue;
                }
                w.WriteEndArray();
            }
            frames.RemoveAt(top);
        }
    }

    /// <summary>Write one child: a scalar here, or a container's opening token and a frame to walk it.</summary>
    private static void OpenChild(Utf8JsonWriter w, object? value, List<WriteFrame> frames)
    {
        if (IsContainer(value, out var items, out var pairs)) Open(w, items, pairs, frames);
        else WriteScalar(w, value);
    }

    /// <summary>Write a container's opening token and push the frame that walks it.</summary>
    private static void Open(Utf8JsonWriter w, IEnumerator<object?>? items,
                             IEnumerator<KeyValuePair<string, object?>>? pairs, List<WriteFrame> frames)
    {
        if (pairs is not null)
        {
            w.WriteStartObject();
            frames.Add(new WriteFrame { IsObject = true, Pairs = pairs });
        }
        else
        {
            w.WriteStartArray();
            frames.Add(new WriteFrame { Items = items });
        }
    }

    private static bool IsContainer(object? value, out IEnumerator<object?>? items,
                                    out IEnumerator<KeyValuePair<string, object?>>? pairs)
    {
        items = value is List<object?> list ? list.GetEnumerator() : null;
        pairs = value is IEnumerable<KeyValuePair<string, object?>> map ? map.GetEnumerator() : null;
        return items is not null || pairs is not null;
    }

    /// <summary>The scalar half of the re-emission: the same spellings the recursive walk wrote.</summary>
    private static void WriteScalar(Utf8JsonWriter w, object? value)
    {
        switch (value)
        {
            case null: w.WriteNullValue(); break;
            case bool b: w.WriteBooleanValue(b); break;
            case long l: w.WriteNumberValue(l); break;
            case BigInteger big: w.WriteRawValue(big.ToString(CultureInfo.InvariantCulture)); break;
            case double d when double.IsFinite(d): w.WriteNumberValue(d); break;
            case double: w.WriteNullValue(); break;   // NaN and the infinities are not JSON
            case string s: w.WriteStringValue(ReplaceLoneSurrogates(s)); break;
        }
    }

    private static string ReplaceLoneSurrogates(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var rune in s.EnumerateRunes()) sb.Append(rune.ToString());
        return sb.ToString();
    }

    private sealed class State(string text)
    {
        internal int Pos;

        // json.decoder.JSONDecodeError: "<msg>: line L column C (char P)", the position counted in code points.
        internal JsonException Error(string message, int at = -1)
        {
            if (at < 0) at = Pos;
            var line = 1;
            var lastNewline = -1;
            var codePoints = 0;
            for (var i = 0; i < at && i < text.Length; i++)
            {
                if (char.IsLowSurrogate(text[i]) && i > 0 && char.IsHighSurrogate(text[i - 1])) continue;
                if (text[i] == '\n') { line++; lastNewline = codePoints; }
                codePoints++;
            }
            var column = lastNewline < 0 ? codePoints + 1 : codePoints - lastNewline;
            return new JsonException($"{message}: line {line} column {column} (char {codePoints})");
        }

        internal void SkipWhitespace()
        {
            while (Pos < text.Length && text[Pos] is ' ' or '\t' or '\n' or '\r') Pos++;
        }

        /// <summary>One open container while parsing: an array's list, or an object's map and pending key.</summary>
        private sealed class Frame
        {
            internal readonly List<object?>? Items;
            internal readonly OrderedDictionary<string, object?>? Map;
            internal string Key = "";

            internal Frame(List<object?> items) => Items = items;

            internal Frame(OrderedDictionary<string, object?> map, string key)
            {
                Map = map;
                Key = key;
            }

            internal bool IsObject => Map is not null;

            internal void Add(object? value)
            {
                if (Map is not null) Map[Key] = value;
                else Items!.Add(value);
            }
        }

        /// <summary>
        /// Parse one complete JSON value, leaving any trailing text for the caller (`Parse` reports "Extra data").
        /// </summary>
        /// <remarks>
        /// Iterative, for the reason <see cref="MaxDepth"/> exists at all: CPython loads documents nested nearly
        /// 3000 levels deep, and a recursive descent costs about two managed frames per level -- more than a
        /// thread with the platform-default 1 MB stack has. Reaching the limit that way is not an error the
        /// caller can catch: .NET cannot catch a stack overflow, so the process aborts instead, and a document
        /// CPython accepts (a downloaded <c>tokenizer_config.json</c>) kills the host. Containers are held in an
        /// explicit stack here, so depth costs heap and never stack.
        ///
        /// Everything observable is unchanged: tokens and their positions (the exact texts
        /// `PythonJsonErrorTests` pins), the scalar types, last-value-wins duplicate keys, and the depth limit,
        /// which counts the open containers -- the same number the recursive `depth` parameter carried.
        /// </remarks>
        internal object? Value()
        {
            var open = new Stack<Frame>();
            while (true)
            {
                // One value, in "expect a value" position. The whitespace before it belongs here rather than in
                // the container loops: the recursive version skipped it in `Object`, `Array` and `Parse` alike.
                SkipWhitespace();
                if (Pos >= text.Length) throw Error("Expecting value");
                object? value;
                switch (text[Pos])
                {
                    case '"':
                        Pos++;
                        value = String();
                        break;
                    case '{':
                        if (open.Count >= MaxDepth)
                            throw new JsonException("maximum recursion depth exceeded while decoding a JSON object from a unicode string");
                        Pos++;
                        var map = new OrderedDictionary<string, object?>(StringComparer.Ordinal);
                        SkipWhitespace();
                        if (Pos < text.Length && text[Pos] == '}')
                        {
                            Pos++;
                            value = map;                 // `{}` completes without ever opening a frame
                            break;
                        }
                        open.Push(new Frame(map, ReadKey()));
                        continue;                        // now parse that member's value
                    case '[':
                        if (open.Count >= MaxDepth)
                            throw new JsonException("maximum recursion depth exceeded while decoding a JSON array from a unicode string");
                        Pos++;
                        var list = new List<object?>();
                        SkipWhitespace();
                        if (Pos < text.Length && text[Pos] == ']')
                        {
                            Pos++;
                            value = list;                // `[]` completes without ever opening a frame
                            break;
                        }
                        open.Push(new Frame(list));
                        continue;                        // now parse the first element
                    case 'n' when Literal("null"):
                        value = null;
                        break;
                    case 't' when Literal("true"):
                        value = true;
                        break;
                    case 'f' when Literal("false"):
                        value = false;
                        break;
                    case 'N' when Literal("NaN"):
                        value = double.NaN;
                        break;
                    case 'I' when Literal("Infinity"):
                        value = double.PositiveInfinity;
                        break;
                    case '-' when string.CompareOrdinal(text, Pos, "-Infinity", 0, 9) == 0:
                        Pos += 9;
                        value = double.NegativeInfinity;
                        break;
                    default:
                        value = Number() ?? throw Error("Expecting value");
                        break;
                }

                // The value is complete: hand it to the innermost open container, then read the delimiter that
                // follows. A container that closes becomes a value for ITS parent, so closing walks outward
                // through this same loop instead of returning through stack frames.
                while (true)
                {
                    if (open.Count == 0) return value;
                    var frame = open.Peek();
                    frame.Add(value);
                    SkipWhitespace();
                    if (frame.IsObject)
                    {
                        if (Pos < text.Length && text[Pos] == '}')
                        {
                            Pos++;
                            open.Pop();
                            value = frame.Map;
                            continue;
                        }
                        if (Pos >= text.Length || text[Pos] != ',') throw Error("Expecting ',' delimiter");
                        Pos++;
                        SkipWhitespace();
                        frame.Key = ReadKey();
                        break;                          // parse the next member's value
                    }
                    if (Pos < text.Length && text[Pos] == ']')
                    {
                        Pos++;
                        open.Pop();
                        value = frame.Items;
                        continue;
                    }
                    if (Pos >= text.Length || text[Pos] != ',') throw Error("Expecting ',' delimiter");
                    Pos++;
                    break;                              // parse the next element
                }
            }
        }

        /// <summary>The <c>"key":</c> of one object member, with the whitespace around the colon consumed.</summary>
        private string ReadKey()
        {
            if (Pos >= text.Length || text[Pos] != '"') throw Error("Expecting property name enclosed in double quotes");
            Pos++;
            var key = String();
            SkipWhitespace();
            if (Pos >= text.Length || text[Pos] != ':') throw Error("Expecting ':' delimiter");
            Pos++;
            return key;
        }

        private bool Literal(string word)
        {
            if (string.CompareOrdinal(text, Pos, word, 0, word.Length) != 0) return false;
            Pos += word.Length;
            return true;
        }

        private static bool Digit(char c) => c is >= '0' and <= '9';

        // _match_number_unicode: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?
        private object? Number()
        {
            var start = Pos;
            var i = Pos;
            if (i < text.Length && text[i] == '-') i++;
            if (i >= text.Length || !Digit(text[i])) return null;
            if (text[i] == '0') i++;
            else while (i < text.Length && Digit(text[i])) i++;
            var integerEnd = i;
            var isFloat = false;
            if (i + 1 < text.Length && text[i] == '.' && Digit(text[i + 1]))
            {
                i += 2;
                while (i < text.Length && Digit(text[i])) i++;
                isFloat = true;
            }
            if (i < text.Length && text[i] is 'e' or 'E')
            {
                var j = i + 1;
                if (j < text.Length && text[j] is '+' or '-') j++;
                if (j < text.Length && Digit(text[j]))
                {
                    while (j < text.Length && Digit(text[j])) j++;
                    i = j;
                    isFloat = true;
                }
            }
            var literal = text[start..i];
            Pos = i;
            if (isFloat)
                return double.Parse(literal, NumberStyles.Float, CultureInfo.InvariantCulture);   // correctly rounded; overflow is infinity
            var digits = integerEnd - start - (text[start] == '-' ? 1 : 0);
            if (digits > MaxIntDigits)
                throw new JsonException($"Exceeds the limit ({MaxIntDigits} digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit");
            return long.TryParse(literal, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var l)
                ? (object)l   // not a bare `l`: the conditional would widen it to BigInteger
                : BigInteger.Parse(literal, CultureInfo.InvariantCulture);
        }

        // py_scanstring with strict=True: control characters below U+0020 are errors, lone surrogates stay.
        private string String()
        {
            var start = Pos - 1;   // the opening quote: where "Unterminated string" is reported
            var sb = new StringBuilder();
            while (true)
            {
                if (Pos >= text.Length) throw Error("Unterminated string starting at", start);
                var c = text[Pos++];
                if (c == '"') return sb.ToString();
                if (c < ' ') throw Error("Invalid control character at", Pos - 1);
                if (c != '\\') { sb.Append(c); continue; }
                if (Pos >= text.Length) throw Error("Unterminated string starting at", start);
                var e = text[Pos++];
                switch (e)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u': sb.Append(Hex4()); break;
                    default: throw Error("Invalid \\escape", Pos - 2);
                }
            }
        }

        private char Hex4()
        {
            // CPython needs one more character after the four digits (scanstring_unicode: end + 4 >= len), so an escape
            // that ends the text is reported at its 'u' even when the digits are fine.
            if (Pos + 4 >= text.Length) throw Error("Invalid \\uXXXX escape", Pos - 1);
            var value = 0;
            for (var k = 0; k < 4; k++)
            {
                var h = text[Pos + k];
                var digit = h is >= '0' and <= '9' ? h - '0' : h is >= 'a' and <= 'f' ? h - 'a' + 10 : h is >= 'A' and <= 'F' ? h - 'A' + 10 : -1;
                if (digit < 0) throw Error("Invalid \\uXXXX escape", Pos - 1);
                value = value * 16 + digit;
            }
            Pos += 4;
            return (char)value;   // a high surrogate followed by \uDC00.. joins in the UTF-16 string, as Python joins the pair
        }

    }
}
