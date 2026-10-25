using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Model-free tests of batched prediction: the chunk/window/sort rules of Python's
/// <c>ONNXAgent._infer_batch</c> (<see cref="BatchPlanning"/>), and
/// <see cref="LayaRouter.PredictBatch"/> / <see cref="LayaRouter.RouteBatch"/> over fake predictors.
/// The model-backed parity of the same rules is in <see cref="PredictParityTests"/>.
/// </summary>
public sealed class BatchPredictionTests
{
    // ── planning: onnx_agent.py:809-815 ───────────────────────────────────────

    [Theory]
    [InlineData(5, null, false, 5, false, 5)]
    [InlineData(5, 0, false, 5, false, 5)]
    [InlineData(5, -3, false, 5, false, 5)]
    [InlineData(5, 2, false, 2, false, 2)]
    [InlineData(5, 100, false, 100, false, 100)]
    // sort_by_length needs an explicit batch_size with 1 < batch_size < len(states).
    [InlineData(20, 2, true, 2, true, 16)]
    [InlineData(20, 3, true, 3, true, 20)]
    [InlineData(20, null, true, 20, false, 20)]
    [InlineData(20, 0, true, 20, false, 20)]
    [InlineData(20, -1, true, 20, false, 20)]
    [InlineData(20, 1, true, 1, false, 1)]
    [InlineData(20, 20, true, 20, false, 20)]
    [InlineData(20, 25, true, 25, false, 25)]
    [InlineData(1, 1, true, 1, false, 1)]
    public void PlanFollowsPythonsChunkAndWindowRules(
        int count, int? batchSize, bool sort, int chunk, bool reorder, int window)
    {
        Assert.Equal((chunk, reorder, window), BatchPlanning.Plan(count, batchSize, sort));
    }

    [Fact]
    public void LookaheadWindowDoesNotOverflowForHugeBatchSizes()
    {
        var (chunk, reorder, window) = BatchPlanning.Plan(int.MaxValue, int.MaxValue - 1, sortByLength: true);
        Assert.Equal(int.MaxValue - 1, chunk);
        Assert.True(reorder);
        Assert.Equal(int.MaxValue, window);
    }

    [Fact]
    public void SortedOrderIsStableAscendingByWidth()
    {
        Assert.Equal([1, 3, 0, 4, 2], BatchPlanning.SortedOrder([5, 1, 7, 1, 5]));
    }

    // States are their own widths; the fake "model" returns each state's width times ten, so a
    // result proves which state it answered.
    private static (IReadOnlyList<int> Results, List<int[]> Runs, List<int> Encoded) Execute(
        int[] widths, int? batchSize, bool sort)
    {
        var runs = new List<int[]>();
        var encoded = new List<int>();
        var results = BatchPlanning.Execute<int, int>(
            [.. widths.Cast<object?>()], batchSize, sort,
            part => { encoded.Add(part.Count); return [.. part.Select(p => (int)p!)]; },
            width => width,
            chunk => { runs.Add([.. chunk]); return [.. chunk.Select(w => w * 10)]; });
        return (results, runs, encoded);
    }

    [Fact]
    public void NoBatchSizeRunsEverythingOnceInInputOrder()
    {
        var (results, runs, encoded) = Execute([5, 1, 3], null, sort: true);
        Assert.Equal([50, 10, 30], results);
        Assert.Equal([[5, 1, 3]], runs);
        Assert.Equal([3], encoded);
    }

    [Fact]
    public void CapSplitsIntoChunksAndEncodesOneChunkAtATimeWithoutSort()
    {
        var (results, runs, encoded) = Execute([5, 1, 3, 2, 4], 2, sort: false);
        Assert.Equal([50, 10, 30, 20, 40], results);
        Assert.Equal([[5, 1], [3, 2], [4]], runs);
        Assert.Equal([2, 2, 1], encoded);
    }

    [Fact]
    public void SortGroupsByWidthInsideTheLookaheadWindowAndRestoresInputOrder()
    {
        var (results, runs, encoded) = Execute([5, 1, 3, 2, 4], 2, sort: true);
        Assert.Equal([50, 10, 30, 20, 40], results);
        Assert.Equal([[1, 2], [3, 4], [5]], runs);
        Assert.Equal([5], encoded);
    }

    [Fact]
    public void SortNeverReachesAcrossWindowsOfEightBatches()
    {
        // 20 states, batch 2: windows of 16 and 4. The smallest width in the second window runs
        // after the largest of the first even though it is smaller than all of them.
        int[] widths = [.. Enumerable.Range(0, 16).Select(i => 100 - i), 1, 2, 3, 4];
        var (results, runs, encoded) = Execute(widths, 2, sort: true);
        Assert.Equal(widths.Select(w => w * 10), results);
        Assert.Equal([16, 4], encoded);
        Assert.Equal([99, 100], runs[7]);
        Assert.Equal([1, 2], runs[8]);
        Assert.Equal(10, runs.Count);
    }

    [Theory]
    [InlineData(int.MaxValue, false)]
    [InlineData(int.MaxValue, true)]
    [InlineData(1_000_000_000, false)]
    [InlineData(1_000_000_000, true)]
    public void HugeBatchSizesNeverSizeBuffersByTheCap(int batchSize, bool sort)
    {
        var (results, runs, encoded) = Execute([5, 1, 3], batchSize, sort);
        Assert.Equal([50, 10, 30], results);
        Assert.Equal([[5, 1, 3]], runs);
        Assert.Equal([3], encoded);
    }

    [Fact]
    public void EqualWidthsKeepInputOrder()
    {
        var (_, runs, _) = Execute([3, 3, 3, 3], 2, sort: true);
        Assert.Equal([[3, 3], [3, 3]], runs);
        var (results, _, _) = Execute([4, 2, 4, 2, 4], 2, sort: true);
        Assert.Equal([40, 20, 40, 20, 40], results);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    [InlineData(-5)]
    public void EmptyStatesNeverEncodeOrRun(int batchSize)
    {
        var (results, runs, encoded) = Execute([], batchSize, sort: true);
        Assert.Empty(results);
        Assert.Empty(runs);
        Assert.Empty(encoded);
    }

    [Fact]
    public void ChunkResultCountMismatchIsAnInternalError()
    {
        var error = Assert.Throws<InvalidOperationException>(() => BatchPlanning.Execute<int, int>(
            [1, 2, 3], null, false, part => [.. part.Select(p => (int)p!)], w => w, _ => [1]));
        Assert.Contains("internal error", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void OptionsDefaultToNoCapAndNoSort()
    {
        var options = new LayaBatchOptions();
        Assert.Null(options.BatchSize);
        Assert.False(options.SortByLength);
    }

    // ── router over fake predictors ───────────────────────────────────────────

    private sealed class FakeBatchPredictor(string name) : ILayaBatchPredictor
    {
        public string Name { get; } = name;
        public List<(IReadOnlyList<object?> States, QuestionSet Questions, LayaBatchOptions? Options)> Calls { get; } = [];

        public LayaResult Predict(object? state, QuestionSet questions) => Answer(state);

        public IReadOnlyList<LayaResult> PredictBatch(
            IReadOnlyList<object?> states, QuestionSet questions, LayaBatchOptions? options = null)
        {
            Calls.Add(([.. states], questions, options));
            return [.. states.Select(Answer)];
        }

        // The state is the result's model, so each result says which request it answered.
        private LayaResult Answer(object? state) =>
            new($"{Name}:{state}", [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
    }

    private sealed class PlainPredictor : ILayaPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) =>
            new("plain", [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
    }

    private sealed class ShortChangePredictor : ILayaBatchPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) => throw new NotSupportedException();

        public IReadOnlyList<LayaResult> PredictBatch(
            IReadOnlyList<object?> states, QuestionSet questions, LayaBatchOptions? options = null) => [];
    }

    private sealed class Rig : IDisposable
    {
        public Dictionary<LayaCheckpoint, FakeBatchPredictor> Predictors { get; } = [];
        public List<LayaCheckpoint> Built { get; } = [];
        public LayaRouter Router { get; }

        public Rig(LayaRouterOptions? options = null) =>
            Router = new LayaRouter(options, checkpoint =>
            {
                Built.Add(checkpoint);
                return Predictors[checkpoint] = new FakeBatchPredictor(LayaOptions.CheckpointSubfolder(checkpoint));
            });

        public void Dispose() => Router.Dispose();
    }

    private static QuestionSet Choice(string instructions, params string[] labels) =>
        new() { ["dept"] = Question.Choice(instructions, labels) };

    private static LayaBatchRequest Request(
        object state, QuestionSet questions, string? model = null, string? task = null,
        string? lang = null, LangGuess? guess = null) =>
        new() { State = state, Questions = questions, Model = model, Task = task, Lang = lang, LangGuess = guess };

    private static readonly QuestionSet Q = Choice("Which team?", "billing", "tech");

    [Fact]
    public void RouteBatchLoadsNothingAndKeepsInputOrder()
    {
        using var rig = new Rig();
        var decisions = rig.Router.RouteBatch(
        [
            Request("I was charged twice, please refund.", Q),
            Request("Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung die nicht korrekt ist", Q),
            Request("hello", Q, model: "ml"),
            Request("hello", Q, task: "typed_decisions"),
            Request("hello", Q, lang: "de"),
            Request("Care este ora in Tokyo?", Q, guess: "ro"),
        ]);

        Assert.Equal(
            [LayaCheckpoint.English, LayaCheckpoint.Multilingual, LayaCheckpoint.Multilingual,
             LayaCheckpoint.TypedDecisions, LayaCheckpoint.Multilingual, LayaCheckpoint.Multilingual],
            decisions.Select(d => d.Model));
        Assert.Empty(rig.Built);
        Assert.Empty(rig.Router.Loaded);
    }

    [Fact]
    public void RouteBatchOfNothingIsEmpty()
    {
        using var rig = new Rig();
        Assert.Empty(rig.Router.RouteBatch([]));
        Assert.Empty(rig.Router.PredictBatch([]));
        Assert.Empty(rig.Built);
    }

    [Fact]
    public void RouteBatchValidatesEveryRequestLikePython()
    {
        using var rig = new Rig();
        var none = Assert.Throws<ArgumentNullException>(() =>
            rig.Router.RouteBatch([Request("ok", Q), new LayaBatchRequest { State = null, Questions = Q }]));
        Assert.Equal("request 1 'state' must not be None; pass a string, dict, or list", none.Message);

        var noQuestions = Assert.Throws<ArgumentException>(() =>
            rig.Router.RouteBatch([new LayaBatchRequest { State = "ok", Questions = null! }]));
        Assert.Equal("request 0 'questions' must be a dict, got NoneType", noQuestions.Message);

        Assert.Throws<ArgumentException>(() => rig.Router.RouteBatch([Request("ok", Q, model: "gpt")]));
        Assert.Throws<ArgumentNullException>(() => rig.Router.RouteBatch(null!));
        Assert.Empty(rig.Built);
    }

    [Fact]
    public void PredictBatchGroupsByCheckpointThenSchemaAndRestoresOrder()
    {
        using var rig = new Rig();
        var other = Choice("Which queue?", "a", "b");
        var results = rig.Router.PredictBatch(
        [
            Request("r0", Q, model: "english"),
            Request("r1", Q, model: "multilingual"),
            Request("r2", other, model: "english"),
            Request("r3", Q, model: "english"),
            Request("r4", Choice("Which team?", "billing", "tech"), model: "multilingual"),
            Request("r5", other, model: "english"),
        ]);

        Assert.Equal(
            ["english:r0", "multilingual:r1", "english:r2", "english:r3", "multilingual:r4", "english:r5"],
            results.Select(r => r.Model));
        // One load per routed checkpoint, in first-seen order.
        Assert.Equal([LayaCheckpoint.English, LayaCheckpoint.Multilingual], rig.Built);

        var english = rig.Predictors[LayaCheckpoint.English].Calls;
        Assert.Equal(2, english.Count);
        Assert.Equal(["r0", "r3"], english[0].States);
        Assert.Equal(["r2", "r5"], english[1].States);
        // Equal-looking but distinct QuestionSets share a forward pass; r1 and r4 are one group.
        var multilingual = rig.Predictors[LayaCheckpoint.Multilingual].Calls;
        Assert.Single(multilingual);
        Assert.Equal(["r1", "r4"], multilingual[0].States);
    }

    [Fact]
    public void SchemaGroupingIsOrderSensitive()
    {
        using var rig = new Rig();
        var results = rig.Router.PredictBatch(
        [
            Request("a", Choice("Which team?", "billing", "tech"), model: "english"),
            Request("b", Choice("Which team?", "tech", "billing"), model: "english"),
            Request("c", new QuestionSet { ["x"] = Question.Noul("n"), ["y"] = Question.Noul("n") }, model: "english"),
            Request("d", new QuestionSet { ["y"] = Question.Noul("n"), ["x"] = Question.Noul("n") }, model: "english"),
        ]);
        Assert.Equal(4, rig.Predictors[LayaCheckpoint.English].Calls.Count);
        Assert.Equal(["english:a", "english:b", "english:c", "english:d"], results.Select(r => r.Model));
    }

    [Fact]
    public void SchemaDistinguishesTypeInstructionsAndOptions()
    {
        var baseline = LayaRouter.QuestionSchema(Choice("Which team?", "billing", "tech"));
        Assert.Equal(baseline, LayaRouter.QuestionSchema(Choice("Which team?", "billing", "tech")));
        Assert.NotEqual(baseline, LayaRouter.QuestionSchema(Choice("Which squad?", "billing", "tech")));
        Assert.NotEqual(baseline, LayaRouter.QuestionSchema(Choice("Which team?", "billing", "tech", "other")));
        Assert.NotEqual(
            LayaRouter.QuestionSchema(new QuestionSet { ["dept"] = Question.Choice("q", ("a", "described")) }),
            LayaRouter.QuestionSchema(new QuestionSet { ["dept"] = Question.Choice("q", ("a", (object?)null)) }));
        Assert.NotEqual(
            LayaRouter.QuestionSchema(new QuestionSet { ["dept"] = Question.Noul("q") }),
            LayaRouter.QuestionSchema(new QuestionSet { ["dept"] = Question.Score("q", "a", "b") }));
    }

    [Fact]
    public void PredictBatchRecordsEachRequestsOwnRoutingDecision()
    {
        using var rig = new Rig();
        var results = rig.Router.PredictBatch(
        [
            Request("hello", Q, model: "english"),
            Request("hello", Q, lang: "de"),
            Request("hello", Q, guess: "ro"),
            Request("x", Q, task: "typed-decisions"),
        ]);
        Assert.Equal("explicit model='english'", results[0].Routing!.Reason);
        Assert.Equal("explicit lang='de'", results[1].Routing!.Reason);
        Assert.Equal(LayaCheckpoint.Multilingual, results[2].Routing!.Model);
        Assert.StartsWith("lang_guess:", results[2].Routing!.Reason, StringComparison.Ordinal);
        Assert.Equal(LayaCheckpoint.TypedDecisions, results[3].Routing!.Model);
    }

    [Fact]
    public void PerRequestLangGuessBeatsTheInstalledOne()
    {
        using var rig = new Rig(new LayaRouterOptions { LangGuess = "ro" });
        var results = rig.Router.PredictBatch(
        [
            Request("Care este ora in Tokyo?", Q),
            Request("Care este ora in Tokyo?", Q, guess: "en"),
            Request("Care este ora in Tokyo?", Q, guess: "C"),
        ]);
        Assert.Equal(
            [LayaCheckpoint.Multilingual, LayaCheckpoint.English, LayaCheckpoint.Multilingual],
            results.Select(r => r.Routing!.Model));
        Assert.StartsWith("Router(lang_guess=...)", results[0].Routing!.Reason, StringComparison.Ordinal);
        Assert.StartsWith("lang_guess:", results[1].Routing!.Reason, StringComparison.Ordinal);
        Assert.StartsWith("Router(lang_guess=...)", results[2].Routing!.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictBatchPassesTheOptionsToEveryGroup()
    {
        using var rig = new Rig();
        var options = new LayaBatchOptions { BatchSize = 4, SortByLength = true };
        rig.Router.PredictBatch(
            [Request("a", Q, model: "english"), Request("b", Choice("other", "x", "y"), model: "english"),
             Request("c", Q, model: "multilingual")], options);
        foreach (var predictor in rig.Predictors.Values)
            foreach (var call in predictor.Calls)
                Assert.Same(options, call.Options);
    }

    [Fact]
    public void PredictBatchNeedsBatchCapablePredictors()
    {
        using var rig = new Rig();
        rig.Router.Attach("english", new PlainPredictor());
        var error = Assert.Throws<NotSupportedException>(() =>
            rig.Router.PredictBatch([Request("a", Q, model: "english")]));
        Assert.Contains("PlainPredictor", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictBatchRefusesAPredictorThatReturnsTheWrongCount()
    {
        using var rig = new Rig();
        rig.Router.Attach("english", new ShortChangePredictor());
        var error = Assert.Throws<InvalidOperationException>(() =>
            rig.Router.PredictBatch([Request("a", Q, model: "english")]));
        Assert.Equal("internal error: Agent.predict_batch returned 0 results for 1 states", error.Message);
    }

    [Fact]
    public void PredictBatchRoutesAllRequestsBeforeLoadingAnything()
    {
        using var rig = new Rig();
        Assert.Throws<ArgumentException>(() => rig.Router.PredictBatch(
            [Request("a", Q, model: "english"), Request("b", Q, model: "gpt")]));
        Assert.Empty(rig.Built);
    }

    [Fact]
    public void AnAttachedBatchPredictorIsUsedWithoutLoading()
    {
        var attached = new FakeBatchPredictor("mine");
        using var rig = new Rig();
        rig.Router.Attach("english", attached);
        var results = rig.Router.PredictBatch([Request("a", Q, model: "english"), Request("b", Q, model: "english")]);
        Assert.Equal(["mine:a", "mine:b"], results.Select(r => r.Model));
        Assert.Single(attached.Calls);
        Assert.Empty(rig.Built);
    }


    // ── Python's refusals and texts ───────────────────────────────────────────

    private const string NullStateText = "state must not be None; pass a string, dict, or list";

    [Fact]
    public void RouterPredictRefusesANullStateBeforeAnythingElse()
    {
        using var rig = new Rig();
        var plain = Assert.Throws<ArgumentNullException>(() => rig.Router.Predict(null, null!));
        Assert.Equal(NullStateText, plain.Message);
        var scan = Assert.Throws<ArgumentNullException>(() => rig.Router.PredictLong(null!, Q));
        Assert.Equal(NullStateText, scan.Message);
        Assert.Empty(rig.Built);
    }

    [Fact]
    public void UnknownModelMessageIsPythons()
    {
        var error = Assert.Throws<ArgumentException>(() => LayaRouter.NormaliseName("gpt"));
        Assert.Equal(
            "unknown model 'gpt'; choose one of ['english', 'multilingual', 'typed-decisions'] "
            + "(or an alias: ['decisions', 'default', 'en', 'laya', 'laya-multilingual', "
            + "'laya-typed-decisions', 'ml', 'multi', 'typed', 'typed_decisions'])",
            error.Message);
    }

    [Fact]
    public void QuestionSchemaIsPythonsJsonDumpsOfTheQuestions()
    {
        var questions = new QuestionSet
        {
            ["dept"] = Question.Choice("Wh\u00e4t team?", ("billing", 1), ("tech", 1.0), ("other", (object?)null)),
            ["urgency"] = Question.Score("How urgent?", "low", 2.5, true),
            ["churn"] = Question.Noul("Cancel?"),
            ["flag"] = Question.Noul("Hold?", ifTrue: "yes"),
        };
        Assert.Equal(
            "{\"dept\": {\"type\": \"choice\", \"instructions\": \"Wh\u00e4t team?\", "
            + "\"criteria\": {\"billing\": 1, \"tech\": 1.0, \"other\": null}}, "
            + "\"urgency\": {\"type\": \"score\", \"instructions\": \"How urgent?\", "
            + "\"criteria\": [\"low\", 2.5, true]}, "
            + "\"churn\": {\"type\": \"noul\", \"instructions\": \"Cancel?\"}, "
            + "\"flag\": {\"type\": \"noul\", \"instructions\": \"Hold?\", \"criteria\": {\"true\": \"yes\"}}}",
            LayaRouter.QuestionSchema(questions));
    }

    [Fact]
    public void IntegerAndFloatCriteriaAreDifferentSchemas()
    {
        using var rig = new Rig();
        rig.Router.PredictBatch(
        [
            Request("a", new QuestionSet { ["q"] = Question.Choice("x", ("a", 1)) }, model: "english"),
            Request("b", new QuestionSet { ["q"] = Question.Choice("x", ("a", 1.0)) }, model: "english"),
            Request("c", new QuestionSet { ["q"] = Question.Choice("x", ("a", 1)) }, model: "english"),
        ]);
        var calls = rig.Predictors[LayaCheckpoint.English].Calls;
        Assert.Equal(2, calls.Count);
        Assert.Equal(["a", "c"], calls[0].States);
        Assert.Equal(["b"], calls[1].States);
    }

    // ── route_batch against the Python recording ──────────────────────────────

    public static IEnumerable<object[]> RouteBatchCases()
    {
        using var doc = RoutingGoldenData.Load("route_batch_probe.json");
        return [.. doc.RootElement.GetProperty("cases").EnumerateArray()
            .Select(e => new object[] { e.GetProperty("label").GetString()! })];
    }

    public static IEnumerable<object[]> RouteBatchErrorCases()
    {
        using var doc = RoutingGoldenData.Load("route_batch_probe.json");
        return [.. doc.RootElement.GetProperty("errors").EnumerateArray()
            .Select(e => new object[] { e.GetProperty("label").GetString()! })];
    }

    private static JsonElement ProbeEntry(string section, string label)
    {
        using var doc = RoutingGoldenData.Load("route_batch_probe.json");
        foreach (var e in doc.RootElement.GetProperty(section).EnumerateArray())
            if (e.GetProperty("label").GetString() == label)
                return JsonDocument.Parse(e.GetRawText()).RootElement;
        throw new InvalidOperationException($"no route_batch_probe.json {section} entry labelled '{label}'");
    }

    private static string? OptString(JsonElement obj, string prop) =>
        obj.TryGetProperty(prop, out var v) && v.ValueKind != JsonValueKind.Null ? v.GetString() : null;

    private static LayaRouterOptions RouterOptions(JsonElement kwargs)
    {
        var installed = OptString(kwargs, "lang_guess");
        var options = new LayaRouterOptions { LangGuess = installed is null ? null : LangGuess.FromCode(installed) };
        if (kwargs.TryGetProperty("auto_task_detection", out var atd)) options.AutoTaskDetection = atd.GetBoolean();
        if (kwargs.TryGetProperty("default", out var def)) options.Default = LayaRouter.NormaliseName(def.GetString()!);
        return options;
    }

    [Theory]
    [MemberData(nameof(RouteBatchCases))]
    public void RouteBatchReproducesPythonDecisions(string label)
    {
        var e = ProbeEntry("cases", label);
        var requests = e.GetProperty("requests").EnumerateArray().Select(r =>
        {
            var guess = OptString(r, "lang_guess");
            return new LayaBatchRequest
            {
                State = RoutingGoldenData.ToClr(r.GetProperty("state")),
                Questions = GoldenData.BuildQuestions(r.GetProperty("questions")),
                Model = OptString(r, "model"),
                Task = OptString(r, "task"),
                Lang = OptString(r, "lang"),
                LangGuess = guess is null ? null : LangGuess.FromCode(guess),
            };
        }).ToArray();

        using var rig = new Rig(RouterOptions(e.GetProperty("router_kwargs")));
        var got = rig.Router.RouteBatch(requests);

        var want = e.GetProperty("decisions").EnumerateArray().ToArray();
        Assert.Equal(want.Length, got.Count);
        for (var i = 0; i < want.Length; i++)
        {
            Assert.Equal(want[i].GetProperty("model").GetString(), LayaOptions.CheckpointSubfolder(got[i].Model));
            Assert.Equal(want[i].GetProperty("reason").GetString(), got[i].Reason);
            Assert.Equal(OptString(want[i], "workflow"), got[i].Workflow);
            Assert.Equal(want[i].GetProperty("detection").ValueKind == JsonValueKind.Null, got[i].Detection is null);
        }
        Assert.Empty(rig.Built);
    }

    [Theory]
    [MemberData(nameof(RouteBatchErrorCases))]
    public void RouteBatchRefusesWhatPythonRefuses(string label)
    {
        var e = ProbeEntry("errors", label);
        var requests = e.GetProperty("requests").EnumerateArray().ToArray();
        // A typed request cannot omit a required member, be a non-dict, or be absent; those Python
        // errors have no spelling here, so only the representable ones are replayed.
        if (requests.Any(r => r.ValueKind != JsonValueKind.Object
                              || !r.TryGetProperty("state", out _) || !r.TryGetProperty("questions", out _)))
            return;
        var typed = requests.Select(r => new LayaBatchRequest
        {
            State = RoutingGoldenData.ToClr(r.GetProperty("state")),
            Questions = r.GetProperty("questions").ValueKind == JsonValueKind.Null
                ? null!
                : GoldenData.BuildQuestions(r.GetProperty("questions")),
            Model = OptString(r, "model"),
        }).ToArray();

        using var rig = new Rig(RouterOptions(e.GetProperty("router_kwargs")));
        var error = Assert.ThrowsAny<ArgumentException>(() => rig.Router.RouteBatch(typed));
        Assert.Equal(e.GetProperty("error_message").GetString(), error.Message);
        Assert.Empty(rig.Built);
    }
}
