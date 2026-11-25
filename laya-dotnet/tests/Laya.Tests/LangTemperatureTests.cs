using System.Diagnostics;
using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Model-free tests of per-language temperatures and the per-call <c>lang</c> / <c>max_len</c> /
/// <c>head_max_len</c>: <c>resolve_lang_temperatures</c> against <c>golden/routing/lang_temperatures_probe.json</c>,
/// what a router hands the engine against <c>golden/routing/router_forwarding_probe.json</c> (Python's own
/// <c>Router</c> over stub agents), and the token budgets against <c>build_sequence</c>. The decode itself, which
/// needs the weights, is pinned by the language cases of <see cref="PredictParityTests"/>.
/// </summary>
[Collection(WarningStateCollection.Name)]
public sealed class LangTemperatureTests
{
    private sealed class Capture : TraceListener
    {
        public List<string> Messages { get; } = [];
        public override void Write(string? message) => Messages.Add(message ?? "");
        public override void WriteLine(string? message) => Messages.Add(message ?? "");
        public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType, int id, string? format, params object?[]? args) =>
            Messages.Add(args is null ? format ?? "" : string.Format(format ?? "", args));
    }

    /// <summary>Runs <paramref name="action"/> with a fresh warning registry and returns the messages with the given prefix, in order.</summary>
    private static List<string> Warnings(string prefix, Action action)
    {
        var capture = new Capture();
        WarnOnce.Reset();
        Trace.Listeners.Add(capture);
        try { action(); }
        finally { Trace.Listeners.Remove(capture); }
        return [.. capture.Messages.Where(m => m.StartsWith(prefix, StringComparison.Ordinal))];
    }

    private static string Text(JsonElement e) => e.GetString()!;

    // ── resolve_lang_temperatures ─────────────────────────────────────────────

    private static readonly Lazy<JsonDocument> Probe = new(() => RoutingGoldenData.Load("lang_temperatures_probe.json"));

    private static JsonElement ProbeCases => Probe.Value.RootElement.GetProperty("cases");

    public static IEnumerable<object[]> TypedCases() =>
        Enumerable.Range(0, ProbeCases.GetArrayLength())
            .Where(i => ProbeCases[i].GetProperty("typed").GetBoolean())
            .Select(i => new object[] { i });

    [Fact]
    public void ProbeCoversEveryCaseASDKCanStateAndSomeItCannot()
    {
        Assert.True(ProbeCases.GetArrayLength() > 60);
        Assert.Contains(TypedCases(), c => ProbeCases[(int)c[0]].TryGetProperty("error_message", out _));
        Assert.Contains(TypedCases(), c => ProbeCases[(int)c[0]].GetProperty("warnings").GetArrayLength() > 0);
        Assert.Contains(Enumerable.Range(0, ProbeCases.GetArrayLength()), i => !ProbeCases[i].GetProperty("typed").GetBoolean());
    }

    [Theory]
    [MemberData(nameof(TypedCases))]
    public void ResolvesTheTableLikePython(int index)
    {
        var c = ProbeCases[index];
        var label = Text(c.GetProperty("label"));
        IEnumerable<KeyValuePair<string, LayaLangTemperature?>>? table = c.GetProperty("pairs").ValueKind == JsonValueKind.Null
            ? null
            : [.. c.GetProperty("pairs").EnumerateArray()
                .Select(pair => new KeyValuePair<string, LayaLangTemperature?>(Text(pair[0]), CallOptions.Entry(pair[1])))];
        var baseSource = RoutingGoldenData.ToClr(c.GetProperty("base"));
        // What the config clamps its own list to: the base an entry without a temperature inherits.
        IReadOnlyList<double> baseTemperature = baseSource is List<object?> list
            ? [.. list.Select(Calibration.ClampTemperature)]
            : [];

        IReadOnlyDictionary<string, ResolvedLangTemperature>? resolved = null;
        ArgumentException? error = null;
        var warnings = Warnings("laya: language override", () =>
        {
            try { resolved = LangTemperatures.Resolve(table, baseSource, baseTemperature); }
            catch (ArgumentException e) { error = e; }
        });

        Assert.Equal(c.GetProperty("warnings").EnumerateArray().Select(w => Text(w.GetProperty("message"))), warnings);
        if (c.TryGetProperty("error_message", out var message))
        {
            Assert.NotNull(error);
            Assert.Equal(Text(message), error.Message);
            Assert.Equal("ValueError", Text(c.GetProperty("error_type")));
            return;
        }
        Assert.Null(error);
        var want = c.GetProperty("result");
        Assert.Equal(want.EnumerateObject().Select(p => p.Name).OrderBy(k => k, StringComparer.Ordinal),
                     resolved!.Keys.OrderBy(k => k, StringComparer.Ordinal));
        foreach (var language in want.EnumerateObject())
        {
            var got = resolved[language.Name];
            Assert.Equal(language.Value.GetProperty("temperature").EnumerateArray().Select(CallOptions.Number),
                         got.Temperature);
            var buckets = language.Value.GetProperty("temperature_by_options");
            Assert.Equal(buckets.EnumerateObject().Select(p => p.Name), got.TemperatureByOptions.Keys);
            foreach (var bucket in buckets.EnumerateObject())
                Assert.Equal(CallOptions.Number(bucket.Value), got.TemperatureByOptions[bucket.Name]);
        }
        Assert.True(label.Length > 0);
    }

    [Theory]
    [InlineData("de", "de")]
    [InlineData("DE", "de")]
    [InlineData("de-AT", "de")]
    [InlineData("De-ch", "de")]
    [InlineData("zh-Hant-TW", "zh")]
    [InlineData("de_DE", "de_de")]
    [InlineData("-de", "")]
    [InlineData("-", "")]
    [InlineData("", "")]
    [InlineData("İ", "i̇")]
    [InlineData("ΣΣ", "σς")]
    public void NormalisesAKeyLikePythonsSplitAndLower(string lang, string expected) =>
        Assert.Equal(expected, LangTemperatures.Normalise(lang));

    [Fact]
    public void LooksALanguageUpAsDecodeDoes()
    {
        var table = LangTemperatures.Resolve(
            [new("de-AT", new LayaLangTemperature { Temperature = [2.0, 3.0, 1.5] }), new("", new LayaLangTemperature())],
            new List<object?> { 1.0, 1.0, 1.0 }, [1.0, 1.0, 1.0]);

        Assert.NotNull(LangTemperatures.Find(table, "de"));
        Assert.NotNull(LangTemperatures.Find(table, "DE-x_y"));
        Assert.Same(LangTemperatures.Find(table, "de"), LangTemperatures.Find(table, "De-ch"));
        Assert.Null(LangTemperatures.Find(table, "ja"));
        Assert.Null(LangTemperatures.Find(table, "de_DE"));
        // `lang and ...`: an empty language never selects, even though the table has an empty key.
        Assert.Null(LangTemperatures.Find(table, ""));
        Assert.Null(LangTemperatures.Find(table, null));
        // ... but a language that only normalises to it does.
        Assert.NotNull(LangTemperatures.Find(table, "-de"));
    }

    // ── engine creation, without a model ──────────────────────────────────────

    private static string FakeArtifacts(string config)
    {
        var dir = Path.Combine(Path.GetTempPath(), "laya-lang-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(dir, "tokenizer"));
        foreach (var name in new[] { "model.onnx", "model.onnx.data", Path.Combine("tokenizer", "tokenizer.json") })
            File.WriteAllBytes(Path.Combine(dir, name), []);
        File.WriteAllText(Path.Combine(dir, "rl_agent_config.json"), config);
        return dir;
    }

    [Fact]
    public void AnInvalidTableRefusesTheLoadWithPythonsTextBeforeAnySessionIsBuilt()
    {
        var dir = FakeArtifacts("{\"temperature\": [1.0, 1.0, 1.0]}");
        try
        {
            var error = Assert.Throws<ArgumentException>(() => LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = dir,
                LangTemperatures = new Dictionary<string, LayaLangTemperature?>
                {
                    ["de"] = new() { Temperature = [1.0, 2.0] },
                },
            }));
            Assert.Equal("Language override 'de' temperature must be a list of 3 floats, got [1.0, 2.0]", error.Message);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void AnInheritingEntryIsCheckedAgainstTheCheckpointsOwnList()
    {
        var dir = FakeArtifacts("{\"temperature\": [1.0, 2.0]}");
        try
        {
            var error = Assert.Throws<ArgumentException>(() => LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = dir,
                LangTemperatures = new Dictionary<string, LayaLangTemperature?> { ["fr"] = null },
            }));
            Assert.Equal("Language override 'fr' temperature must be a list of 3 floats, got [1.0, 2.0]", error.Message);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void ALanguageWarningComesBeforeTheCheckpointsOwnAndAnInvalidTableSuppressesIt()
    {
        const string config = "{\"temperature\": [8.5, 1.0, 1.0]}";
        var dir = FakeArtifacts(config);
        try
        {
            var clamped = new Dictionary<string, LayaLangTemperature?>
            {
                ["de"] = new() { Temperature = [0.1, 1.0, 1.0] },
            };
            // The empty model.onnx fails when the session is built; both warnings are out by then.
            var messages = Warnings("laya", () => Assert.ThrowsAny<Exception>(
                () => LayaEngine.Create(new LayaOptions { ModelDirectory = dir, LangTemperatures = clamped })));
            var ofLanguage = messages.FindIndex(m => m.StartsWith("laya: language override 'de'", StringComparison.Ordinal));
            var ofCheckpoint = messages.FindIndex(m => m.StartsWith("laya ONNX: this checkpoint ships", StringComparison.Ordinal) && m.Contains("temperature[0]=8.5", StringComparison.Ordinal));
            Assert.True(ofLanguage >= 0 && ofCheckpoint > ofLanguage, string.Join("\n", messages));

            // Other tests warn through the same global listener meanwhile; this checkpoint's text is its own.
            var refused = Warnings("laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort confidence; clamping temperature[0]=8.5", () => Assert.Throws<ArgumentException>(() => LayaEngine.Create(new LayaOptions
            {
                ModelDirectory = dir,
                LangTemperatures = new Dictionary<string, LayaLangTemperature?> { ["de"] = new() { Temperature = [1.0] } },
            })));
            Assert.Empty(refused);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // ── token budgets: build_sequence ─────────────────────────────────────────

    // Expected values are Python's `build_sequence(tok, state, q, max_len, head_max_len, return_stats=True,
    // return_truncation_stats=True)` over a tokenizer that makes one id per whitespace-separated word, for the choice
    // question and 40-word state below: the length of the ids, the markers, the state tokens dropped and the per-option cap.
    [Theory]
    [InlineData(64, 32, 64, new[] { 12, 17, 22 }, 5, 5)]
    [InlineData(0, 32, 0, new int[] { }, 40, 5)]
    [InlineData(-1, 32, 28, new int[] { }, 40, 5)]
    [InlineData(-7, 32, 22, new int[] { }, 40, 5)]
    [InlineData(10, 32, 10, new int[] { }, 40, 5)]
    [InlineData(30, 32, 30, new[] { 12, 17, 22 }, 39, 5)]
    [InlineData(64, 0, 64, new[] { 10, 14, 18 }, 0, 4)]
    [InlineData(64, -5, 64, new[] { 10, 14, 18 }, 0, 4)]
    [InlineData(64, -1000000, 64, new[] { 10, 14, 18 }, 0, 4)]
    [InlineData(-5, -5, 19, new int[] { }, 40, 4)]
    [InlineData(int.MinValue, 32, 0, new int[] { }, 40, 5)]
    [InlineData(64, int.MinValue, 64, new[] { 10, 14, 18 }, 0, 4)]
    [InlineData(int.MaxValue, 32, 69, new[] { 12, 17, 22 }, 0, 5)]
    [InlineData(64, int.MaxValue, 64, new[] { 12, 22, 32 }, 17, null)]
    [InlineData(int.MinValue, int.MinValue, 0, new int[] { }, 40, 4)]
    [InlineData(int.MaxValue, int.MaxValue, 81, new[] { 12, 22, 32 }, 0, null)]
    [InlineData(1, 1, 1, new int[] { }, 40, 4)]
    [InlineData(200, 16, 64, new[] { 10, 14, 18 }, 0, 4)]
    [InlineData(200, 15, 64, new[] { 10, 14, 18 }, 0, 4)]
    public void BudgetsFollowPythonsArithmeticWhateverTheValue(
        int maxLen, int headMaxLen, int length, int[] markers, int dropped, int? perOption)
    {
        var tok = new StubTokenizer();
        var question = Question.Choice("which of these teams should handle the request",
            ("billing", "invoices and payments and refunds for double charges"),
            ("tech", "bugs outages and integration errors in the product"),
            ("account", "logins seats and plan changes"));
        var state = string.Join(' ', Enumerable.Range(0, 40).Select(i => "w" + i));

        var built = SequenceBuilder.BuildDetailed(tok, state, question, maxLen, headMaxLen);

        Assert.Equal(length, built.Ids.Length);
        Assert.Equal(markers, built.Markers);
        Assert.Equal(dropped, built.StateTokensDropped);
        Assert.Equal(perOption, built.Options.TokensPerOption);
    }

    // ── what a router hands the engine ────────────────────────────────────────

    private sealed record Call(string Model, string Entry, string?[] States, string? Lang, int? MaxLen, int? HeadMaxLen);

    private sealed class Agent(string name, List<Call> log, bool hasLangTemperatures)
        : ILayaCallPredictor, ILayaBatchPredictor, ILayaLongLangPredictor, ILayaLangTemperatures
    {
        public bool HasLangTemperatures => hasLangTemperatures;

        private LayaResult Answer() =>
            new(name, [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));

        public LayaResult Predict(object? state, QuestionSet questions) => Predict(state, questions, null);

        public LayaResult Predict(object? state, QuestionSet questions, LayaPredictOptions? options)
        {
            log.Add(new(name, "system_one", [(string?)state], options?.Lang, options?.MaxLen, options?.HeadMaxLen));
            return Answer();
        }

        public IReadOnlyList<LayaResult> PredictBatch(
            IReadOnlyList<object?> states, QuestionSet questions, LayaBatchOptions? options = null)
        {
            log.Add(new(name, "predict_batch", [.. states.Cast<string?>()], options?.Lang, options?.MaxLen, options?.HeadMaxLen));
            return [.. states.Select(_ => Answer())];
        }

        public LayaResult PredictLong(object state, QuestionSet questions, int? window = null, int? stride = null,
                                      string aggregate = "auto", int? batchSize = null) =>
            PredictLong(state, questions, new LayaLongOptions());

        public LayaResult PredictLong(object state, QuestionSet questions, LayaLongOptions? options)
        {
            log.Add(new(name, "predict_long", [(string?)state], options?.Lang, null, null));
            return Answer();
        }
    }

    /// <summary>A scanner with no way to take a language, like a Python <c>predict_long</c> with no <c>lang</c> parameter.</summary>
    private sealed class ScanOnly(string name, List<Call> log) : ILayaLongPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) => throw new NotSupportedException();

        public LayaResult PredictLong(object state, QuestionSet questions, int? window = null, int? stride = null,
                                      string aggregate = "auto", int? batchSize = null)
        {
            log.Add(new(name, "predict_long", [(string?)state], null, null, null));
            return new(name, [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
        }
    }

    private static LayaRouter RouterOver(List<Call> log, bool english = false, bool multilingual = false, bool scanOnly = false) =>
        new(null, checkpoint =>
        {
            var name = LayaOptions.CheckpointSubfolder(checkpoint);
            if (scanOnly) return new ScanOnly(name, log);
            return new Agent(name, log, checkpoint switch
            {
                LayaCheckpoint.English => english,
                LayaCheckpoint.Multilingual => multilingual,
                _ => false,
            });
        });

    private static readonly Lazy<JsonDocument> Forwarding = new(() => RoutingGoldenData.Load("router_forwarding_probe.json"));

    private static JsonElement ForwardingSection(string name) => Forwarding.Value.RootElement.GetProperty(name);

    public static IEnumerable<object[]> ForwardingCases(string section) =>
        Enumerable.Range(0, ForwardingSection(section).GetArrayLength()).Select(i => new object[] { i });

    public static IEnumerable<object[]> PredictForwardingCases() => ForwardingCases("predict");
    public static IEnumerable<object[]> LongForwardingCases() => ForwardingCases("long");
    public static IEnumerable<object[]> BatchForwardingCases() => ForwardingCases("batch");

    private static string? Str(JsonElement args, string name) =>
        args.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    private static int? Int(JsonElement obj, string name) =>
        obj.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;

    private static LangGuess? Guess(JsonElement args) => Str(args, "lang_guess") is { } code ? LangGuess.FromCode(code) : null;

    private static readonly QuestionSet RouterQuestions =
        new() { ["q"] = Question.Noul("Is the customer asking for a refund?") };

    [Theory]
    [MemberData(nameof(PredictForwardingCases))]
    public void RouterPredictHandsTheEngineWhatPythonsRouterDoes(int index)
    {
        var c = ForwardingSection("predict")[index];
        var args = c.GetProperty("args");
        var received = c.GetProperty("received");
        var log = new List<Call>();
        using var router = RouterOver(log);

        var result = router.Predict(Str(args, "state"), RouterQuestions, Str(args, "model"), null, Str(args, "lang"), Guess(args),
            new LayaPredictOptions { MaxLen = Int(args, "max_len"), HeadMaxLen = Int(args, "head_max_len") });

        var call = Assert.Single(log);
        Assert.Equal(Text(c.GetProperty("agent")), call.Model);
        Assert.Equal(Text(c.GetProperty("model")), LayaOptions.CheckpointSubfolder(result.Routing!.Model));
        Assert.Equal(Str(received, "lang"), call.Lang);
        Assert.Equal(Int(received, "max_len"), call.MaxLen);
        Assert.Equal(Int(received, "head_max_len"), call.HeadMaxLen);
    }

    [Theory]
    [MemberData(nameof(LongForwardingCases))]
    public void RouterPredictLongHandsTheScanWhatPythonsRouterDoes(int index)
    {
        var c = ForwardingSection("long")[index];
        var args = c.GetProperty("args");
        var log = new List<Call>();
        var takesLang = !args.TryGetProperty("stub_takes_lang", out var flag) || flag.GetBoolean();
        using var router = RouterOver(log, scanOnly: !takesLang);

        LayaResult? result = null;
        var warnings = Warnings("laya: Router.predict_long", () =>
            result = router.PredictLong(Str(args, "state")!, RouterQuestions, null, null, "auto", null,
                                        Str(args, "model"), null, Str(args, "lang"), Guess(args)));

        var call = Assert.Single(log);
        Assert.Equal(Text(c.GetProperty("model")), LayaOptions.CheckpointSubfolder(result!.Routing!.Model));
        Assert.Equal(Str(c.GetProperty("received"), "lang"), call.Lang);
        // The warning names the agent's type; Python's stub is `_StubAgent`.
        Assert.Equal(c.GetProperty("warnings").EnumerateArray()
                         .Select(w => Text(w.GetProperty("message")).Replace("_StubAgent.predict_long", "ScanOnly.predict_long", StringComparison.Ordinal)),
                     warnings);
    }

    [Theory]
    [MemberData(nameof(BatchForwardingCases))]
    public void RouterPredictBatchGroupsAndForwardsWhatPythonsRouterDoes(int index)
    {
        var c = ForwardingSection("batch")[index];
        var has = c.GetProperty("has_lang_temperatures");
        var log = new List<Call>();
        using var router = RouterOver(log, english: has.GetProperty("english").GetBoolean(),
                                      multilingual: has.GetProperty("multilingual").GetBoolean());
        var requests = c.GetProperty("requests").EnumerateArray().Select(r => new LayaBatchRequest
        {
            State = Text(r.GetProperty("state")),
            Questions = GoldenData.BuildQuestions(r.GetProperty("questions")),
            Model = Str(r, "model"),
            Lang = Str(r, "lang"),
            LangGuess = Guess(r),
            MaxLen = Int(r, "max_len"),
            HeadMaxLen = Int(r, "head_max_len"),
        }).ToList();

        var results = router.PredictBatch(requests);

        Assert.Equal(c.GetProperty("models").EnumerateArray().Select(Text),
                     results.Select(r => LayaOptions.CheckpointSubfolder(r.Routing!.Model)));
        Assert.Equal(
            c.GetProperty("calls").EnumerateArray().Select(call => (Text(call.GetProperty("model")),
                call.GetProperty("states").EnumerateArray().Select(Text).ToArray(),
                Str(call.GetProperty("kwargs"), "lang"), Int(call.GetProperty("kwargs"), "max_len"),
                Int(call.GetProperty("kwargs"), "head_max_len"))),
            log.Select(call => (call.Model, call.States.Select(s => s!).ToArray(), call.Lang, call.MaxLen, call.HeadMaxLen)));
    }

    // ── guards that have no Python call to be tested against ───────────────────

    [Fact]
    public void ARouterRefusesAPerCallLanguageInTheOptionsItAlreadyTakesAsAnArgument()
    {
        using var router = RouterOver([]);
        var error = Assert.Throws<ArgumentException>(() =>
            router.Predict("hello", RouterQuestions, new LayaPredictOptions { Lang = "de" }));
        Assert.Contains("lang argument", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("lang")]
    [InlineData("max_len")]
    [InlineData("head_max_len")]
    public void ARouterTakesLanguageAndBudgetsPerRequestNotPerBatch(string name)
    {
        using var router = RouterOver([]);
        var options = name switch
        {
            "lang" => new LayaBatchOptions { Lang = "de" },
            "max_len" => new LayaBatchOptions { MaxLen = 10 },
            _ => new LayaBatchOptions { HeadMaxLen = 10 },
        };
        var request = new LayaBatchRequest { State = "hello", Questions = RouterQuestions };

        var error = Assert.Throws<ArgumentException>(() => router.PredictBatch([request], options));
        Assert.StartsWith($"Router.predict_batch() got an unexpected keyword argument '{name}'", error.Message, StringComparison.Ordinal);
    }

    private sealed class PlainPredictor : ILayaPredictor
    {
        public int Calls;
        public LayaResult Predict(object? state, QuestionSet questions)
        {
            Calls++;
            return new("plain", [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
        }
    }

    [Fact]
    public void AnAgentWithoutPerCallOptionsIsCalledPlainlyForALanguageAndRefusedAnyTokenBudget()
    {
        var plain = new PlainPredictor();
        using var router = new LayaRouter(null, _ => plain);

        // Python retries without `lang` when the agent does not take it.
        router.Predict("hello", RouterQuestions, null, null, "de", null);
        Assert.Equal(1, plain.Calls);

        var error = Assert.Throws<NotSupportedException>(() => router.Predict(
            "hello", RouterQuestions, null, null, null, null, new LayaPredictOptions { HeadMaxLen = 5 }));
        Assert.Equal("PlainPredictor.Predict() got an unexpected keyword argument 'head_max_len'", error.Message);
        Assert.Equal(1, plain.Calls);
    }
}
