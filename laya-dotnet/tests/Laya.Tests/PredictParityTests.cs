using System.Globalization;
using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Tier 2: full answers against the ones Python recorded, which needs the weights.
/// </summary>
/// <remarks>
/// The goldens come from the real <c>laya.Agent.system_one</c> over the same ONNX graph, so any gap
/// here is this port's, not the model's. Recorded values are rounded to four decimals on both sides,
/// which is why the tolerances sit a little above 1e-4 rather than at zero.
/// </remarks>
[Collection(AllEnginesCollection.Name)]
public sealed class PredictParityTests(AllEnginesFixture fixture)
{
    /// <summary>Probabilities and confidences: 4-dp values, so a rounding boundary is 1e-4 wide.</summary>
    private const double ProbTolerance = 2e-4;

    /// <summary>A score is a weighted sum over level indices, so it carries several of those errors.</summary>
    private const double ScoreTolerance = 2e-3;

    // ── cross-checkpoint theory data ──────────────────────────────────────────

    /// <summary>
    /// All success cases across every checkpoint whose golden data is available, each paired
    /// with its checkpoint. Cases for checkpoints without golden data are omitted.
    /// </summary>
    public static IEnumerable<object[]> AllSuccessCases() =>
        from c in TestArtifacts.ParityCheckpoints
        where GoldenData.For(c).Available
        // Kind is null for a standard parity case; a "shortlist" case's `result` comes from
        // predict_shortlist over a reduced criteria set, so it is not comparable to a plain
        // engine.Predict over the full (unreduced) criteria recorded in its `questions` — see
        // ShortlistEndToEndTests instead.
        from info in GoldenData.For(c).Index.Where(i => i.Kind is null && !i.ExpectsError)
        select new object[] { c, info };

    /// <summary>All error cases across every checkpoint whose golden data is available.</summary>
    public static IEnumerable<object[]> AllErrorCases() =>
        from c in TestArtifacts.ParityCheckpoints
        where GoldenData.For(c).Available
        from info in GoldenData.For(c).Index.Where(i => i.Kind is null && i.ExpectsError)
        select new object[] { c, info };

    /// <summary>
    /// The same cases as <see cref="AllSuccessCases"/>, for the split ONNX layout. The golden
    /// answers are recorded once per checkpoint (not once per layout), so the same expected
    /// values apply: the split graph is mathematically the fused graph cut in two, not a
    /// different model. Skips per-checkpoint at run time via <see cref="AllEnginesFixture.RequireSplit"/>
    /// when <c>&lt;repo&gt;/onnx-split/&lt;checkpoint&gt;</c> is absent, exactly like the fused
    /// theories below skip when <c>onnx/&lt;checkpoint&gt;</c> is absent.
    /// </summary>
    public static IEnumerable<object[]> AllSuccessCasesSplit() => AllSuccessCases();

    /// <summary>Structured score levels whose legend must retain the exact rendered JSON text.</summary>
    public static IEnumerable<object[]> StructuredScoreCases() =>
        AllSuccessCases().Where(values => ((GoldenCaseInfo)values[1]).Name == "criterion_shapes");

    // ── theory tests ──────────────────────────────────────────────────────────

    [Theory]
    [MemberData(nameof(StructuredScoreCases))]
    public void StructuredScoreLegendsAreRenderedText(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        ReproducesTheRecordedAnswers(checkpoint, info);
        ReproducesTheRecordedAnswersSplit(checkpoint, info);
    }

    [Theory]
    [MemberData(nameof(AllSuccessCases))]
    public void ReproducesTheRecordedAnswers(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        var engine = fixture.Require(checkpoint);
        var golden = GoldenData.For(checkpoint);
        using var doc = golden.Load(info);
        var root = doc.RootElement;

        var state     = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var result    = engine.Predict(state, questions);
        var expected  = root.GetProperty("result");

        Assert.Equal(expected.GetProperty("model").GetString(), result.Model);
        Assert.Equal(expected.GetProperty("usage").GetProperty("input_tokens").GetInt32(),
                     result.Usage.InputTokens);
        Assert.Equal(expected.GetProperty("usage").GetProperty("output_tokens").GetInt32(),
                     result.Usage.OutputTokens);

        var answers = expected.GetProperty("answers");
        // Ids in order: a result that answered the right questions in the wrong order would attach
        // every answer to the wrong question, and per-id comparison alone would not notice.
        Assert.Equal([.. answers.EnumerateObject().Select(p => p.Name)], result.Ids);

        foreach (var entry in answers.EnumerateObject())
        {
            var want  = entry.Value;
            var got   = result[entry.Name];
            var where = $"{info.Name}/{entry.Name}";

            Assert.Equal(want.GetProperty("type").GetString(), SequenceBuilder.TypeName(got.Type));
            AssertClose(want.GetProperty("confidence").GetDouble(), got.Confidence,
                        ProbTolerance, $"{where}.confidence");
            AssertClose(want.GetProperty("action").GetProperty("act_probability").GetDouble(),
                        got.Action.ActProbability, ProbTolerance, $"{where}.action.act_probability");

            switch (got.Type)
            {
                case QuestionType.Choice:
                    AssertChoice(want, got.AsChoice(), where);
                    break;
                case QuestionType.Score:
                    AssertScore(want, got.AsScore(), where);
                    break;
                case QuestionType.Noul:
                    AssertClose(want.GetProperty("noul").GetDouble(), got.AsNoul().Probability,
                                ProbTolerance, $"{where}.noul");
                    break;
                default:
                    throw new InvalidOperationException($"{where}: unexpected type {got.Type}");
            }
        }
    }

    /// <summary>
    /// Same assertions as <see cref="ReproducesTheRecordedAnswers"/>, but loading the checkpoint's
    /// split-layout export (<c>encoder.onnx</c> + <c>head.onnx</c>) instead of the fused one.
    /// Not run in CI or locally until an <c>onnx-split/</c> export exists — see the class remarks.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllSuccessCasesSplit))]
    public void ReproducesTheRecordedAnswersSplit(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        var engine = fixture.RequireSplit(checkpoint);
        var golden = GoldenData.For(checkpoint);
        using var doc = golden.Load(info);
        var root = doc.RootElement;

        var state     = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var result    = engine.Predict(state, questions);
        var expected  = root.GetProperty("result");

        Assert.Equal(expected.GetProperty("model").GetString(), result.Model);
        Assert.Equal(expected.GetProperty("usage").GetProperty("input_tokens").GetInt32(),
                     result.Usage.InputTokens);

        var answers = expected.GetProperty("answers");
        Assert.Equal([.. answers.EnumerateObject().Select(p => p.Name)], result.Ids);

        foreach (var entry in answers.EnumerateObject())
        {
            var want  = entry.Value;
            var got   = result[entry.Name];
            var where = $"split/{info.Name}/{entry.Name}";

            AssertClose(want.GetProperty("confidence").GetDouble(), got.Confidence,
                        ProbTolerance, $"{where}.confidence");

            switch (got.Type)
            {
                case QuestionType.Choice:
                    AssertChoice(want, got.AsChoice(), where);
                    break;
                case QuestionType.Score:
                    AssertScore(want, got.AsScore(), where);
                    break;
                case QuestionType.Noul:
                    AssertClose(want.GetProperty("noul").GetDouble(), got.AsNoul().Probability,
                                ProbTolerance, $"{where}.noul");
                    break;
                default:
                    throw new InvalidOperationException($"{where}: unexpected type {got.Type}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(AllErrorCases))]
    public void RejectsTheSameQuestionsPythonRejects(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        var engine = fixture.Require(checkpoint);
        var golden = GoldenData.For(checkpoint);
        using var doc = golden.Load(info);
        var root = doc.RootElement;

        var state     = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));

        var error = Assert.Throws<ArgumentException>(() => engine.Predict(state, questions));
        // ArgumentException appends the parameter name, so the recorded text is a prefix of the
        // message rather than all of it.
        Assert.StartsWith(root.GetProperty("error_message").GetString()!, error.Message, StringComparison.Ordinal);
    }

    // ── fact tests (multilingual checkpoint) ─────────────────────────────────

    [Fact]
    public void AnswersTheSameWayWhenQuestionsAreAskedOneAtATime()
    {
        // Every question rides in one batch, so a padding or masking mistake could let a question
        // be influenced by its neighbours. Asking them singly has to give the same answers.
        var engine = fixture.Require(LayaCheckpoint.Multilingual);
        using var doc = GoldenData.For(LayaCheckpoint.Multilingual).Load("case_quickstart.json");
        var root      = doc.RootElement;
        var state     = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));

        var batched = engine.Predict(state, questions);

        foreach (var id in questions.Ids)
        {
            var single = engine.Predict(state, new QuestionSet { [id] = questions[id] });
            AssertClose(batched[id].Confidence, single[id].Confidence, ProbTolerance, $"{id}.confidence");
        }
    }

    [Fact]
    public void IsSafeToCallFromSeveralThreadsAtOnce()
    {
        // The engine is documented as shareable and a caller will hold it as a singleton, so the
        // tokenizer gate and the session have to tolerate concurrent use.
        var engine = fixture.Require(LayaCheckpoint.Multilingual);
        using var doc = GoldenData.For(LayaCheckpoint.Multilingual).Load("case_quickstart.json");
        var root      = doc.RootElement;
        var state     = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var want      = engine.Predict(state, questions)["department"].AsChoice().Choice;

        var results = new string[8];
        Parallel.For(0, results.Length,
            i => results[i] = engine.Predict(state, questions)["department"].AsChoice().Choice);

        Assert.All(results, choice => Assert.Equal(want, choice));
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static void AssertChoice(JsonElement want, ChoiceAnswer got, string where)
    {
        var probabilities = want.GetProperty("probabilities");
        // Labels in order, because the entire marker layout rests on labels being positional.
        Assert.Equal([.. probabilities.EnumerateObject().Select(p => p.Name)],
                     got.Probabilities.Select(p => p.Key));

        foreach (var p in probabilities.EnumerateObject())
            AssertClose(p.Value.GetDouble(), got[p.Name], ProbTolerance, $"{where}.probabilities[{p.Name}]");

        // Checked after the distribution: when the top two are close the argmax is the fragile part,
        // and the distribution mismatch is the more informative failure to report.
        Assert.Equal(want.GetProperty("choice").GetString(), got.Choice);
        AssertClose(1.0, got.Probabilities.Sum(p => p.Value), 1e-3, $"{where}.probabilities sum");
    }

    private static void AssertScore(JsonElement want, ScoreAnswer got, string where)
    {
        // Python's JSON keys both the legend and the distribution by the stringified level index.
        var legend = want.GetProperty("legend");
        Assert.Equal(legend.GetPropertyCount(), got.Legend.Count);
        for (var i = 0; i < got.Legend.Count; i++)
        {
            var level = legend.GetProperty(i.ToString(CultureInfo.InvariantCulture));
            Assert.Equal(level.GetString(), Assert.IsType<string>(got.Legend[i]));
        }

        var probabilities = want.GetProperty("probabilities");
        Assert.Equal(probabilities.GetPropertyCount(), got.Probabilities.Count);
        for (var i = 0; i < got.Probabilities.Count; i++)
            AssertClose(probabilities.GetProperty(i.ToString(CultureInfo.InvariantCulture)).GetDouble(),
                        got.Probabilities[i], ProbTolerance, $"{where}.probabilities[{i}]");

        AssertClose(want.GetProperty("score").GetDouble(), got.Score, ScoreTolerance, $"{where}.score");
    }

    private static void AssertClose(double expected, double actual, double tolerance, string what) =>
        Assert.True(Math.Abs(expected - actual) <= tolerance,
            $"{what}: python {expected}, dotnet {actual} (tolerance {tolerance})");
}
