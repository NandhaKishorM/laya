using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Model-free tests of the abstention gate (<c>min_confidence</c>, <c>jev_confidence</c>): every case of
/// <c>golden/routing/confidence_probe.json</c> (Python's own <c>laya/confidence.py</c> over the synthetic
/// inputs of <c>tools/confidence_cases.py</c>), plus the call-level behaviour over fake predictors.
/// </summary>
public sealed class ConfidenceGateTests
{
    // ── probe plumbing ────────────────────────────────────────────────────────

    private static readonly Lazy<JsonDocument> Probe = new(() => RoutingGoldenData.Load("confidence_probe.json"));
    private static JsonElement Root => Probe.Value.RootElement;

    public static IEnumerable<object[]> Indices(string section) =>
        Enumerable.Range(0, Root.GetProperty(section).GetArrayLength()).Select(i => new object[] { i });

    public static IEnumerable<object[]> ScalarCases() => Indices("scalars");
    public static IEnumerable<object[]> MapCases() => Indices("maps");
    public static IEnumerable<object[]> BucketCases() => Indices("buckets");
    public static IEnumerable<object[]> ResolveCases() => Indices("resolve");
    public static IEnumerable<object[]> GateCases() => Indices("gate");
    public static IEnumerable<object[]> ReuseCases() => Indices("reuse");
    public static IEnumerable<object[]> JevCases() => Indices("jev");

    /// <summary>A number, or one of the strings the dumper uses for the non-finite ones; absent is NaN (no usable number).</summary>
    private static double Num(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.Number => value.GetDouble(),
        JsonValueKind.String => value.GetString() switch
        {
            "nan" => double.NaN,
            "inf" => double.PositiveInfinity,
            "-inf" => double.NegativeInfinity,
            var other => throw new InvalidOperationException($"unexpected number text {other}"),
        },
        _ => throw new InvalidOperationException($"unexpected {value.ValueKind}"),
    };

    private static double Field(JsonElement answer, string name) =>
        answer.TryGetProperty(name, out var value) ? Num(value) : double.NaN;

    /// <summary>The typed answer a recorded raw answer stands for.</summary>
    private static Answer ToAnswer(JsonElement raw)
    {
        var confidence = Field(raw, "confidence");
        var answerConfidence = Field(raw, "answer_confidence");
        var action = new ActionInfo(0.5);
        switch (raw.GetProperty("type").GetString())
        {
            case "choice":
            {
                var probs = raw.GetProperty("probabilities").EnumerateObject()
                    .Select(p => new KeyValuePair<string, double>(p.Name, Num(p.Value))).ToList();
                return new ChoiceAnswer(probs.Count == 0 ? "" : probs[0].Key, probs, confidence, answerConfidence, action);
            }
            case "score":
            {
                var probs = raw.GetProperty("probabilities").EnumerateObject()
                    .OrderBy(p => int.Parse(p.Name)).Select(p => Num(p.Value)).ToArray();
                return new ScoreAnswer(0.0, [.. probs.Select((_, i) => (object?)i.ToString())], probs,
                                       confidence, answerConfidence, action);
            }
            case "noul":
                return new NoulAnswer(Field(raw, "noul"), confidence, answerConfidence, action);
            default:
                throw new InvalidOperationException("not a typed answer");
        }
    }

    private static LayaResult ToResult(JsonElement answers)
    {
        var byId = new Dictionary<string, Answer>(StringComparer.Ordinal);
        var order = new List<string>();
        foreach (var entry in answers.EnumerateObject())
        {
            order.Add(entry.Name);
            byId[entry.Name] = ToAnswer(entry.Value);
        }
        return new LayaResult("m", order, byId, new Usage(0, 0));
    }

    private static MinConfidence? ToThreshold(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.Null => null,
        JsonValueKind.Object => MinConfidence.ForBuckets(
            value.EnumerateObject().Select(p => new KeyValuePair<string, double>(p.Name, Num(p.Value)))),
        _ => MinConfidence.Of(Num(value)),
    };

    private static void AssertGated(JsonElement expected, LayaResult got, string where)
    {
        Assert.Equal(expected.EnumerateObject().Select(p => p.Name), got.Ids);
        foreach (var entry in expected.EnumerateObject())
            AssertGateFields(entry.Value, got[entry.Name], $"{where}/{entry.Name}");
    }

    /// <summary>
    /// The three keys Python's gate writes, against an answer: absent in the recording means false/null here.
    /// Shared with the model-backed parity tests, whose recorded answers carry the same keys.
    /// </summary>
    internal static void AssertGateFields(JsonElement want, Answer answer, string at)
    {
        Assert.True((want.TryGetProperty("low_confidence", out var low) && low.GetBoolean()) == answer.LowConfidence,
                    at + " low_confidence");
        Assert.Equal(want.TryGetProperty("abstention", out var state) ? state.GetString() : null, answer.Abstention);
        Assert.Equal(want.TryGetProperty("abstention_threshold", out var thr) ? thr.GetDouble() : (double?)null,
                     answer.AbstentionThreshold);
    }

    /// <summary>The <c>min_confidence</c> a recorded case's <c>options</c> carry, or null if it has none.</summary>
    internal static MinConfidence? MinConfidenceOf(JsonElement root) =>
        root.TryGetProperty("options", out var options) ? MinConfidenceIn(options) : null;

    /// <summary>The <c>min_confidence</c> in a recorded <c>options</c> object, or null if it has none.</summary>
    internal static MinConfidence? MinConfidenceIn(JsonElement options) =>
        options.TryGetProperty("min_confidence", out var value) ? ToThreshold(value) : null;

    // ── check_min_confidence / check_min_confidence_map ───────────────────────

    [Theory]
    [MemberData(nameof(ScalarCases))]
    public void ScalarThresholdsAreValidatedLikePython(int index)
    {
        var want = Root.GetProperty("scalars")[index];
        var value = Num(want.GetProperty("value"));
        if (want.GetProperty("valid").GetBoolean())
        {
            Assert.Equal(Num(want.GetProperty("result")), MinConfidence.Of(value).Check().Scalar);
            return;
        }
        var error = Assert.Throws<ArgumentException>(() => MinConfidence.Of(value).Check());
        Assert.Equal(want.GetProperty("error_message").GetString(), error.Message);
    }

    [Theory]
    [MemberData(nameof(MapCases))]
    public void MapThresholdsAreValidatedLikePython(int index)
    {
        var want = Root.GetProperty("maps")[index];
        var pairs = want.GetProperty("pairs").EnumerateArray()
            .Select(p => new KeyValuePair<string, double>(p[0].GetString()!, Num(p[1]))).ToArray();
        var threshold = MinConfidence.ForBuckets(pairs);
        if (want.GetProperty("valid").GetBoolean())
        {
            var got = threshold.Check();
            var expected = want.GetProperty("result").EnumerateArray().ToDictionary(p => p[0].GetString()!, p => Num(p[1]));
            Assert.Equal(expected.OrderBy(kv => kv.Key), got.Map!.OrderBy(kv => kv.Key));
            return;
        }
        var error = Assert.Throws<ArgumentException>(() => threshold.Check());
        Assert.Equal(want.GetProperty("error_message").GetString(), error.Message);
    }

    [Fact]
    public void BucketKeysAndGateStatesAreThoseOfPython()
    {
        Assert.Equal(Root.GetProperty("bucket_keys").EnumerateArray().Select(k => k.GetString()!).Order(StringComparer.Ordinal),
                     MinConfidence.BucketKeys.Order(StringComparer.Ordinal));
        Assert.Equal(Root.GetProperty("gate_states").EnumerateArray().Select(k => k.GetString()!), GateStates.All);
    }

    // ── _option_bucket / resolve_min_confidence ───────────────────────────────

    [Theory]
    [MemberData(nameof(BucketCases))]
    public void OptionBucketsAreThoseOfPython(int index)
    {
        var want = Root.GetProperty("buckets")[index];
        var bucket = want.GetProperty("bucket");
        Assert.Equal(bucket.ValueKind == JsonValueKind.Null ? null : bucket.GetString(),
                     ConfidenceGate.OptionBucket(ToAnswer(want.GetProperty("answer"))));
    }

    [Theory]
    [MemberData(nameof(ResolveCases))]
    public void ThresholdResolutionFollowsPython(int index)
    {
        var want = Root.GetProperty("resolve")[index];
        var map = want.GetProperty("thresholds").EnumerateObject().ToDictionary(p => p.Name, p => Num(p.Value));
        Assert.Equal(Num(want.GetProperty("result")), ConfidenceGate.Resolve(ToAnswer(want.GetProperty("answer")), map));
    }

    // ── apply_confidence_gate ─────────────────────────────────────────────────

    [Theory]
    [MemberData(nameof(GateCases))]
    public void TheGateWritesWhatPythonWrites(int index)
    {
        var want = Root.GetProperty("gate")[index];
        var result = ToResult(Root.GetProperty("gate_answers"));
        var threshold = ToThreshold(want.GetProperty("threshold"));
        var gated = threshold is null ? result : ConfidenceGate.Apply(result, threshold.Check());
        AssertGated(want.GetProperty("expected"), gated, $"gate[{index}]");
    }

    // #910: Python re-gates one payload in place; here each step must depend only on its own threshold,
    // and the answers handed in must not change at all.
    [Theory]
    [MemberData(nameof(ReuseCases))]
    public void AReusedResultCarriesNoStaleVerdict(int index)
    {
        var result = ToResult(Root.GetProperty("gate_answers"));
        var steps = Root.GetProperty("reuse")[index].GetProperty("steps").EnumerateArray().ToArray();
        for (var s = 0; s < steps.Length; s++)
        {
            var gated = ConfidenceGate.Apply(result, ToThreshold(steps[s].GetProperty("threshold"))!.Check());
            AssertGated(steps[s].GetProperty("expected"), gated, $"reuse[{index}] step {s}");
            foreach (var id in result.Ids)
            {
                Assert.False(result[id].LowConfidence);
                Assert.Null(result[id].Abstention);
                Assert.Null(result[id].AbstentionThreshold);
            }
        }
    }

    [Fact]
    public void GatingTheSameAnswerFromManyThreadsNeverLeaksAVerdictAcrossCalls()
    {
        var result = ToResult(Root.GetProperty("gate_answers"));
        var answer = result["choice2"];
        Parallel.For(0, 400, i =>
        {
            var strict = i % 2 == 0;
            var gated = ConfidenceGate.Apply(result, MinConfidence.Of(strict ? 0.9 : 0.0).Check())["choice2"];
            Assert.Equal(strict, gated.LowConfidence);
            Assert.Equal(strict ? GateStates.Abstained : GateStates.Passed, gated.Abstention);
        });
        Assert.Null(answer.Abstention);
    }

    // ── jev_confidence ────────────────────────────────────────────────────────

    [Theory]
    [MemberData(nameof(JevCases))]
    public void JevConfidenceMatchesPython(int index)
    {
        var want = Root.GetProperty("jev")[index];
        if (!want.GetProperty("typed").GetBoolean()) return; // a shape a typed answer cannot hold
        var result = want.GetProperty("result");
        var got = ToAnswer(want.GetProperty("answer")).JevConfidence();
        if (result.ValueKind == JsonValueKind.Null) Assert.Null(got);
        else
        {
            Assert.NotNull(got);
            // Bit for bit: CPython 3.12's compensated float sum() is reproduced by PythonSum.
            Assert.Equal(result.GetDouble(), got.Value);
        }
    }

    [Fact]
    public void TheProbeCoversTypedAndUntypedJevCases()
    {
        var cases = Root.GetProperty("jev").EnumerateArray().ToArray();
        Assert.Contains(cases, c => !c.GetProperty("typed").GetBoolean());
        Assert.True(cases.Count(c => c.GetProperty("typed").GetBoolean()) >= 60);
        Assert.Equal(9, cases.Count(c => c.GetProperty("name").GetString()!.StartsWith("sum-differs-", StringComparison.Ordinal)));
        Assert.Contains(cases, c => c.GetProperty("result").ValueKind == JsonValueKind.Null && c.GetProperty("typed").GetBoolean());
    }

    // ── call level: an ungated call writes nothing ────────────────────────────

    private static ChoiceAnswer Choice(double answerConfidence, double confidence = 0.1) => new(
        "a", [new("a", answerConfidence), new("b", 1.0 - answerConfidence)], confidence, answerConfidence, new ActionInfo(0.5));

    private sealed class FakeBatchPredictor(string name, double answerConfidence) : ILayaBatchPredictor
    {
        public int Calls, BatchCalls;
        public LayaBatchOptions? LastOptions;

        private LayaResult One() => new(name, ["q"], new Dictionary<string, Answer>(StringComparer.Ordinal)
        {
            ["q"] = Choice(answerConfidence),
        }, new Usage(0, 0));

        public LayaResult Predict(object? state, QuestionSet questions)
        {
            Interlocked.Increment(ref Calls);
            return One();
        }

        public IReadOnlyList<LayaResult> PredictBatch(IReadOnlyList<object?> states, QuestionSet questions, LayaBatchOptions? options = null)
        {
            Interlocked.Increment(ref BatchCalls);
            LastOptions = options;
            return [.. states.Select(_ => One())];
        }
    }

    private sealed class Rig : IDisposable
    {
        public Dictionary<LayaCheckpoint, FakeBatchPredictor> Predictors { get; } = [];
        public int Builds;
        public LayaRouter Router { get; }

        public Rig(double answerConfidence = 0.6) =>
            Router = new LayaRouter(null, checkpoint =>
            {
                Interlocked.Increment(ref Builds);
                return Predictors[checkpoint] = new FakeBatchPredictor(LayaOptions.CheckpointSubfolder(checkpoint), answerConfidence);
            });

        public void Dispose() => Router.Dispose();
    }

    private static QuestionSet Questions() => new() { ["q"] = Question.Noul("x") };


    [Fact]
    public void AnUngatedRouterCallWritesNothing()
    {
        using var rig = new Rig();
        var answer = rig.Router.Predict("I was charged twice", Questions())["q"];
        Assert.False(answer.LowConfidence);
        Assert.Null(answer.Abstention);
        Assert.Null(answer.AbstentionThreshold);
        var viaOptions = rig.Router.Predict("I was charged twice", Questions(), null, null, null, null, new LayaPredictOptions())["q"];
        Assert.Null(viaOptions.Abstention);
    }

    [Fact]
    public void ARouterCallGatesTheAnswersAndKeepsTheRouting()
    {
        using var rig = new Rig(answerConfidence: 0.6);
        var options = new LayaPredictOptions { MinConfidence = MinConfidence.Of(0.7) };
        var result = rig.Router.Predict("I was charged twice", Questions(), null, null, null, null, options);
        Assert.NotNull(result.Routing);
        Assert.True(result["q"].LowConfidence);
        Assert.Equal(GateStates.Abstained, result["q"].Abstention);
        Assert.Equal(0.7, result["q"].AbstentionThreshold);

        var passes = rig.Router.Predict("I was charged twice", Questions(), null, null, null, null,
            new LayaPredictOptions { MinConfidence = MinConfidence.Of(0.0) });
        Assert.False(passes["q"].LowConfidence);
        Assert.Equal(GateStates.Passed, passes["q"].Abstention);
        Assert.Equal(0.0, passes["q"].AbstentionThreshold);
    }

    [Fact]
    public void TheThreeArgumentRouterOverloadRoutesLikeThePlainOneAndGates()
    {
        using var rig = new Rig(answerConfidence: 0.6);
        var gated = rig.Router.Predict("I was charged twice", Questions(),
            new LayaPredictOptions { MinConfidence = MinConfidence.Of(0.7) });
        Assert.Equal(GateStates.Abstained, gated["q"].Abstention);
        Assert.Equal(rig.Router.Predict("I was charged twice", Questions()).Routing!.Model, gated.Routing!.Model);
        Assert.Null(rig.Router.Predict("I was charged twice", Questions(), null)["q"].Abstention);
    }

    [Fact]
    public void ARouterMapThresholdIsResolvedPerAnswerBucket()
    {
        using var rig = new Rig(answerConfidence: 0.6);
        var map = MinConfidence.ForBuckets([new("choice:2", 0.65), new("default", 0.1)]);
        var result = rig.Router.Predict("I was charged twice", Questions(), null, null, null, null,
            new LayaPredictOptions { MinConfidence = map });
        Assert.Equal(0.65, result["q"].AbstentionThreshold);
        Assert.True(result["q"].LowConfidence);
    }

    [Fact]
    public void AnInvalidThresholdIsRefusedBeforeAnythingIsRoutedOrLoaded()
    {
        using var rig = new Rig();
        var bad = new LayaPredictOptions { MinConfidence = MinConfidence.Of(1.5) };
        var error = Assert.Throws<ArgumentException>(
            () => rig.Router.Predict("text", Questions(), null, null, null, null, bad));
        Assert.Equal("min_confidence must be a float in [0.0, 1.0], got 1.5", error.Message);
        Assert.Equal(0, rig.Builds);
        Assert.Empty(rig.Router.Loaded);
    }

    [Fact]
    public void TheRouterChecksStateAndQuestionsBeforeTheThreshold()
    {
        // router.py:1335-1338: a missing state is the TypeError even when the threshold is also wrong.
        using var rig = new Rig();
        var bad = new LayaPredictOptions { MinConfidence = MinConfidence.Of(double.NaN) };
        var none = Assert.Throws<ArgumentNullException>(
            () => rig.Router.Predict(null, Questions(), null, null, null, null, bad));
        Assert.Equal("state must not be None; pass a string, dict, or list", none.Message);
        Assert.Throws<ArgumentNullException>(() => rig.Router.Predict("text", null!, null, null, null, null, bad));
    }

    [Fact]
    public void ABatchThresholdIsCheckedBeforeTheRequests()
    {
        // router.py:1669: the threshold is the first thing predict_batch looks at.
        using var rig = new Rig();
        var error = Assert.Throws<ArgumentException>(() => rig.Router.PredictBatch(
            null!, new LayaBatchOptions { MinConfidence = MinConfidence.ForBuckets([new("bogus", 0.5)]) }));
        Assert.Equal(typeof(ArgumentException), error.GetType());
        Assert.Equal("min_confidence map keys must be a bucket like 'choice:3-5' or 'default', got 'bogus'", error.Message);
        Assert.Equal(0, rig.Builds);
    }

    [Fact]
    public void ABatchThresholdIsBatchWideAndNotPartOfTheGrouping()
    {
        using var rig = new Rig(answerConfidence: 0.6);
        var questions = Questions();
        LayaBatchRequest[] requests =
        [
            new() { State = "I was charged twice", Questions = questions },
            new() { State = "Refund my invoice", Questions = questions },
            new() { State = "Please cancel my account", Questions = questions },
        ];
        var plain = rig.Router.PredictBatch(requests);
        var plainCalls = rig.Predictors.Values.Sum(p => p.BatchCalls);
        var gated = rig.Router.PredictBatch(
            requests, new LayaBatchOptions { BatchSize = 2, SortByLength = true, MinConfidence = MinConfidence.Of(0.7) });
        // router.py:1789: the checkpoints get the batching knobs but not the threshold; the router gates.
        Assert.All(rig.Predictors.Values.Where(p => p.LastOptions is not null), p =>
        {
            Assert.Null(p.LastOptions!.MinConfidence);
            Assert.Equal(2, p.LastOptions.BatchSize);
            Assert.True(p.LastOptions.SortByLength);
        });
        Assert.Equal(plainCalls * 2, rig.Predictors.Values.Sum(p => p.BatchCalls));
        Assert.All(plain, r => Assert.Null(r["q"].Abstention));
        Assert.Equal(3, gated.Count);
        Assert.All(gated, r =>
        {
            Assert.NotNull(r.Routing);
            Assert.Equal(GateStates.Abstained, r["q"].Abstention);
            Assert.True(r["q"].LowConfidence);
            Assert.Equal(0.7, r["q"].AbstentionThreshold);
        });
    }

    [Fact]
    public void AnAttachedPredictorWithoutGateSupportIsStillGatedByTheRouter()
    {
        using var router = new LayaRouter(null, _ => throw new InvalidOperationException("must not be built"));
        router.Attach("english", new FakeBatchPredictor("mine", 0.4));
        var result = router.Predict("I was charged twice", Questions(), "english", null, null, null,
            new LayaPredictOptions { MinConfidence = MinConfidence.Of(0.5) });
        Assert.Equal(GateStates.Abstained, result["q"].Abstention);
    }

    // ── answers keep their gate through the copies the API makes ──────────────

    [Fact]
    public void AGatedAnswerStaysGatedWhenAWindowIsAttached()
    {
        var gated = ConfidenceGate.Apply(Choice(0.2), MinConfidence.Of(0.5).Check());
        var windowed = gated.WithWindow(new WindowInfo(0, 0, 10, 1));
        Assert.Equal(GateStates.Abstained, windowed.Abstention);
        Assert.True(windowed.LowConfidence);
        Assert.Null(Choice(0.2).Abstention);
    }

    // ── JevConfidence on an answer ────────────────────────────────────────────

    [Fact]
    public void PythonSumIsCompensatedLikeCPython312()
    {
        // A plain left-to-right sum of ten 0.1s is 0.9999999999999999; CPython 3.12 returns 1.0.
        Assert.Equal(1.0, PythonSum.Sum(Enumerable.Repeat(0.1, 10)));
        Assert.Equal(0.0, PythonSum.Sum([]));
        Assert.True(double.IsPositiveInfinity(PythonSum.Sum([double.MaxValue, double.MaxValue])));
        Assert.True(double.IsNaN(PythonSum.Sum([double.PositiveInfinity, double.NegativeInfinity])));
    }

    [Fact]
    public void JevConfidenceIsNullForNoul()
    {
        Assert.Null(new NoulAnswer(0.9, 0.9, 0.9, new ActionInfo(0.5)).JevConfidence());
    }
}
