using System.Numerics;
using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Differential parity: replays the generated corpus in <c>golden/routing/differential_probe.json</c>
/// (written by <c>tools/dump_differential_golden.py</c> from the real Python <c>laya.lang</c>,
/// <c>laya.router</c> and <c>laya.email</c>; the generator is the same file the Rust SDK uses) and requires
/// every answer to match. Floats are compared bit for bit. All mismatches are collected first; the failure
/// message lists the first 20.
/// </summary>
public sealed class DifferentialParityTests
{
    private sealed record Mismatch(string Label, string Input, string Python, string DotNet);

    private sealed class NoopPredictor : ILayaPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) =>
            new("noop", [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
    }

    private static string Show(string? s)
    {
        if (s is null) return "null";
        var runes = s.EnumerateRunes().Take(300).Select(r => r.ToString());
        return PythonStr.Repr(string.Concat(runes));
    }

    private static bool Bits(double a, double b) => BitConverter.DoubleToInt64Bits(a) == BitConverter.DoubleToInt64Bits(b);

    private static string? Str(JsonElement e) => e.ValueKind == JsonValueKind.Null ? null : e.GetString();

    private static void CheckLang(JsonElement cases, List<Mismatch> output)
    {
        foreach (var c in cases.EnumerateArray())
        {
            var label = c.GetProperty("label").GetString()!;
            var state = RoutingGoldenData.ToClr(c.GetProperty("state"));
            var input = Show(c.GetProperty("state").GetRawText());

            var text = LanguageDetection.StateText(state);
            var expectedText = c.GetProperty("state_text").GetString()!;
            if (text != expectedText)
                output.Add(new($"{label}: state_text", input, Show(expectedText), Show(text)));

            var a = c.GetProperty("analyse");
            var got = LanguageDetection.Analyse(state);
            var bad = new List<string>();
            if (got.Script != a.GetProperty("script").GetString()) bad.Add("script");
            if (got.Language != Str(a.GetProperty("language"))) bad.Add("language");
            if (got.IsEnglish != a.GetProperty("is_english").GetBoolean()) bad.Add("is_english");
            if (got.LanguageUndecided != a.GetProperty("language_undecided").GetBoolean()) bad.Add("language_undecided");
            if (!Bits(got.DiacriticRate, a.GetProperty("diacritic_rate").GetDouble())) bad.Add("diacritic_rate");
            if (!Bits(got.NonLatinFraction, a.GetProperty("non_latin_fraction").GetDouble())) bad.Add("non_latin_fraction");
            if (got.MixedSegment != Str(a.GetProperty("mixed_segment"))) bad.Add("mixed_segment");
            var profile = a.GetProperty("script_profile").EnumerateObject().ToList();
            var sameProfile = profile.Count == got.ScriptProfile.Count
                && profile.All(p => got.ScriptProfile.TryGetValue(p.Name, out var g) && Bits(g, p.Value.GetDouble()));
            if (!sameProfile) bad.Add("script_profile");
            if (bad.Count > 0)
                output.Add(new($"{label}: analyse [{string.Join(", ", bad)}]", input, a.GetRawText(),
                    $"script={got.Script} language={got.Language} is_english={got.IsEnglish} undecided={got.LanguageUndecided} "
                    + $"diacritic={got.DiacriticRate:R} non_latin={got.NonLatinFraction:R} mixed={Show(got.MixedSegment)} "
                    + $"profile={string.Join(",", got.ScriptProfile.Select(p => $"{p.Key}:{p.Value:R}"))}"));
        }
    }

    private static void CheckRoutes(JsonElement langCases, JsonElement cases, List<Mismatch> output)
    {
        var states = langCases.EnumerateArray().Select(c => c.GetProperty("state")).ToList();
        var routers = new Dictionary<string, LayaRouter>(StringComparer.Ordinal);
        LayaRouter RouterFor(string? code)
        {
            var key = code ?? "\0plain";
            if (!routers.TryGetValue(key, out var router))
                routers[key] = router = new LayaRouter(
                    new LayaRouterOptions { LangGuess = code is null ? null : LangGuess.FromCode(code) }, _ => new NoopPredictor());
            return router;
        }

        try
        {
            var n = 0;
            foreach (var c in cases.EnumerateArray())
            {
                var index = c.GetProperty("state").GetInt32();
                var kwargs = c.GetProperty("kwargs");
                string? Kw(string name) => kwargs.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
                var guess = Kw("lang_guess");
                var decision = RouterFor(Kw("router_lang_guess")).Route(
                    RoutingGoldenData.ToClr(states[index]), null, null, null, Kw("lang"),
                    guess is null ? null : LangGuess.FromCode(guess));
                var model = LayaOptions.CheckpointSubfolder(decision.Model);
                var expectedModel = c.GetProperty("model").GetString();
                var expectedReason = c.GetProperty("reason").GetString();
                if (model != expectedModel || decision.Reason != expectedReason)
                    output.Add(new($"route #{n} state {index} {kwargs.GetRawText()}", Show(states[index].GetRawText()),
                        $"{expectedModel} / {Show(expectedReason)}", $"{model} / {Show(decision.Reason)}"));
                n++;
            }
        }
        finally
        {
            foreach (var router in routers.Values) router.Dispose();
        }
    }

    private static void CheckEmailClean(List<long> budgets, JsonElement cases, List<Mismatch> output)
    {
        foreach (var c in cases.EnumerateArray())
        {
            var label = c.GetProperty("label").GetString()!;
            var body = c.GetProperty("body").GetString()!;
            var results = c.GetProperty("results").EnumerateArray().ToList();
            for (var i = 0; i < budgets.Count; i++)
            {
                var got = LayaEmail.CleanBody(body, checked((int)budgets[i]));
                var expected = results[i].GetString()!;
                if (got != expected)
                    output.Add(new($"{label} max_chars={budgets[i]}", Show(body), Show(expected), Show(got)));
            }
        }
    }

    // A JSON value against the CLR value the state holds: numbers by value, objects in order.
    private static bool Same(JsonElement expected, object? got)
    {
        switch (expected.ValueKind)
        {
            case JsonValueKind.Null: return got is null;
            case JsonValueKind.True: return got is true;
            case JsonValueKind.False: return got is false;
            case JsonValueKind.String: return got is string s && s == expected.GetString();
            case JsonValueKind.Number:
                return got switch
                {
                    long l => expected.TryGetInt64(out var e) && e == l,
                    int i => expected.TryGetInt64(out var e) && e == i,
                    BigInteger big => BigInteger.TryParse(expected.GetRawText(), out var e) && e == big,
                    double d => Bits(d, expected.GetDouble()),
                    _ => false,
                };
            case JsonValueKind.Array:
                return got is System.Collections.IEnumerable seq and not string
                    && Zip(expected.EnumerateArray().ToList(), seq.Cast<object?>().ToList());
            case JsonValueKind.Object:
                var props = expected.EnumerateObject().ToList();
                var pairs = got switch
                {
                    IEnumerable<KeyValuePair<string, object?>> kv => kv.ToList(),
                    _ => null,
                };
                return pairs is not null && pairs.Count == props.Count
                    && props.Zip(pairs).All(t => t.First.Name == t.Second.Key && Same(t.First.Value, t.Second.Value));
            default: return false;
        }
    }

    private static bool Zip(List<JsonElement> expected, List<object?> got) =>
        expected.Count == got.Count && expected.Zip(got).All(t => Same(t.First, t.Second));

    private static void CheckEmailState(JsonElement cases, List<Mismatch> output)
    {
        foreach (var c in cases.EnumerateArray())
        {
            var label = c.GetProperty("label").GetString()!;
            var maxChars = 3000;
            var extra = new List<KeyValuePair<string, object?>>();
            foreach (var p in c.GetProperty("extra").EnumerateObject())
            {
                if (p.Name == "max_chars") maxChars = p.Value.GetInt32();   // consumed, never a field
                else extra.Add(new(p.Name, RoutingGoldenData.ToClr(p.Value)));
            }
            var got = LayaEmail.State(Str(c.GetProperty("subject")), Str(c.GetProperty("body")), Str(c.GetProperty("sender")),
                c.GetProperty("clean").GetBoolean(), extra, maxChars);
            if (!Same(c.GetProperty("result"), got))
                output.Add(new(label, Show(c.GetRawText()), Show(c.GetProperty("result").GetRawText()),
                    Show(string.Join(", ", got.Select(p => $"{p.Key}={p.Value}")))));
        }
    }

    // str.lower(): final sigma, dotted capital I, Kelvin, sharp S, titlecase letters.
    private static void CheckLower(JsonElement cases, List<Mismatch> output)
    {
        foreach (var c in cases.EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var expected = c.GetProperty("lower").GetString()!;
            var got = LanguageDetection.PythonLower(text);
            if (got != expected) output.Add(new("lower", Show(text), Show(expected), Show(got)));
        }
    }

    // json.loads against the reader that backs the config parsers: the exact ValueError / JSONDecodeError text, or none.
    private static void CheckJsonErrors(JsonElement cases, List<Mismatch> output)
    {
        foreach (var c in cases.EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var expected = Str(c.GetProperty("error"));
            string? got = null;
            try { PythonJsonReader.Parse(text); }
            catch (JsonException e) { got = e.Message; }
            if (got != expected)
                output.Add(new("json error (reader)", Show(text), expected is null ? "valid" : Show(expected), got is null ? "valid" : Show(got)));
            if (expected is not null)
            {
                // The config parser reports the same text for a document that is not JSON.
                string? viaConfig = null;
                try { LayaConfig.Parse(text); }
                catch (JsonException e) { viaConfig = e.Message; }
                catch (Exception) { }
                if (viaConfig != expected)
                    output.Add(new("json error (LayaConfig.Parse)", Show(text), Show(expected), viaConfig is null ? "no JsonException" : Show(viaConfig)));
            }
        }
    }

    [Fact]
    public void GeneratedCorpusMatchesPython()
    {
        using var doc = RoutingGoldenData.Load("differential_probe.json");
        var root = doc.RootElement;
        var lang = root.GetProperty("lang");
        var budgets = root.GetProperty("email_budgets").EnumerateArray().Select(v => v.GetInt64()).ToList();

        var mismatches = new List<Mismatch>();
        CheckLang(lang, mismatches);
        var afterLang = mismatches.Count;
        CheckRoutes(lang, root.GetProperty("route"), mismatches);
        var afterRoutes = mismatches.Count;
        CheckEmailClean(budgets, root.GetProperty("email_clean"), mismatches);
        var afterClean = mismatches.Count;
        CheckEmailState(root.GetProperty("email_state"), mismatches);
        var afterState = mismatches.Count;
        CheckLower(root.GetProperty("lower"), mismatches);
        var afterLower = mismatches.Count;
        CheckJsonErrors(root.GetProperty("json_errors"), mismatches);

        var summary = $"lang {afterLang}, route {afterRoutes - afterLang}, email_clean {afterClean - afterRoutes}, "
            + $"email_state {afterState - afterClean}, lower {afterLower - afterState}, json_errors {mismatches.Count - afterLower}";
        Assert.True(mismatches.Count == 0,
            $"{mismatches.Count} mismatches against Python ({summary}); first {Math.Min(20, mismatches.Count)}:"
            + string.Concat(mismatches.Take(20).Select(m =>
                $"\n- {m.Label}\n    input:  {m.Input}\n    python: {m.Python}\n    .NET:   {m.DotNet}")));
    }
}
