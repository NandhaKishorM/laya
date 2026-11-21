namespace Laya.Tests;

/// <summary>
/// Tier 2: the embedding shortlist over a real checkpoint end to end, for all three checkpoints.
/// Compares <see cref="LayaShortlist.Predict"/> (using <see cref="HashingEmbedder"/>, the same
/// deterministic demo embedder the golden vector was recorded with) against
/// <c>case_shortlist_many_options.json</c>'s recorded <c>predict_shortlist</c> result.
/// </summary>
/// <remarks>
/// Each test loads and disposes its own <see cref="LayaEngine"/> rather than sharing
/// <see cref="AllEnginesFixture"/> (which keeps every checkpoint it has touched resident for the
/// whole run): the ONNX artifacts are 1.3-1.7 GB apiece, and these tests must not hold more than
/// one at a time. The class still joins <see cref="AllEnginesCollection"/>, purely so it never
/// runs concurrently with another engine-loading test class — the native tokenizer binding is not
/// safe to construct from two threads at once for the same checkpoint.
/// </remarks>
[Collection(AllEnginesCollection.Name)]
public sealed class ShortlistEndToEndTests
{
    /// <summary>Tolerance for the shortlist's raw (unrounded) cosine scores.</summary>
    private const double ScoreTolerance = 1e-9;

    /// <summary>Tolerance for the model's own 4-dp probabilities/confidence, as in <see cref="PredictParityTests"/>.</summary>
    private const double ProbTolerance = 2e-4;

    public static TheoryData<LayaCheckpoint> Checkpoints => new(TestArtifacts.ParityCheckpoints);

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void ShortlistManyOptionsMatchesTheRecordedResult(LayaCheckpoint checkpoint)
    {
        var golden = GoldenData.For(checkpoint);
        if (!golden.Available)
            Assert.Skip($"no golden data for '{checkpoint}'; run laya-dotnet/tools/dump_golden.py");
        if (!TestArtifacts.For(checkpoint).HasModel || !TestArtifacts.For(checkpoint).HasTokenizer)
            Assert.Skip($"no ONNX artifacts for '{checkpoint}'; set LAYA_ONNX_ROOT to the parent of all checkpoints");

        using var doc = golden.Load("case_shortlist_many_options.json");
        var root = doc.RootElement;
        var state = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var k = root.GetProperty("shortlist_k").GetInt32();

        using var engine = LayaEngine.FromDirectory(TestArtifacts.For(checkpoint).Directory);
        var result = LayaShortlist.Predict(engine, state, questions, HashingEmbedder.Embed, k);

        var expected = root.GetProperty("result");
        Assert.Equal(expected.GetProperty("model").GetString(), result.Model);

        var wantShortlist = expected.GetProperty("shortlist").GetProperty("intent");
        var gotShortlist = result.Shortlist!["intent"];
        Assert.Equal(
            [.. wantShortlist.GetProperty("labels").EnumerateArray().Select(e => e.GetString()!)],
            gotShortlist.Labels);
        Assert.Equal(wantShortlist.GetProperty("k").GetInt32(), gotShortlist.K);
        Assert.Equal(wantShortlist.GetProperty("n").GetInt32(), gotShortlist.N);
        Assert.Equal(wantShortlist.GetProperty("passthrough").GetBoolean(), gotShortlist.Passthrough);

        var wantScores = wantShortlist.GetProperty("scores").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        Assert.NotNull(gotShortlist.Scores);
        Assert.Equal(wantScores.Length, gotShortlist.Scores!.Count);
        for (var i = 0; i < wantScores.Length; i++)
            Assert.Equal(wantScores[i], gotShortlist.Scores[i], ScoreTolerance);

        var wantAnswer = expected.GetProperty("answers").GetProperty("intent");
        var gotAnswer = result["intent"].AsChoice();
        Assert.Equal(wantAnswer.GetProperty("choice").GetString(), gotAnswer.Choice);
        foreach (var p in wantAnswer.GetProperty("probabilities").EnumerateObject())
            Assert.Equal(p.Value.GetDouble(), gotAnswer[p.Name], ProbTolerance);
        Assert.Equal(wantAnswer.GetProperty("confidence").GetDouble(), result["intent"].Confidence, ProbTolerance);
    }

    // The narrowed choice loses its option_order (with a warning); the passthrough choice, the score and
    // the noul keep theirs, so the recorded answers pin both halves.
    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void ShortlistOptionOrderMatchesTheRecordedResult(LayaCheckpoint checkpoint)
    {
        var golden = GoldenData.For(checkpoint);
        if (!golden.Available)
            Assert.Skip($"no golden data for '{checkpoint}'; run laya-dotnet/tools/dump_golden.py");
        if (!TestArtifacts.For(checkpoint).HasModel || !TestArtifacts.For(checkpoint).HasTokenizer)
            Assert.Skip($"no ONNX artifacts for '{checkpoint}'; set LAYA_ONNX_ROOT to the parent of all checkpoints");

        using var doc = golden.Load("case_shortlist_option_order.json");
        var root = doc.RootElement;
        var state = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var k = root.GetProperty("shortlist_k").GetInt32();

        using var engine = LayaEngine.FromDirectory(TestArtifacts.For(checkpoint).Directory);
        var result = LayaShortlist.Predict(engine, state, questions, HashingEmbedder.Embed, k);

        var expected = root.GetProperty("result");
        foreach (var entry in expected.GetProperty("shortlist").EnumerateObject())
        {
            var got = result.Shortlist![entry.Name];
            Assert.Equal([.. entry.Value.GetProperty("labels").EnumerateArray().Select(e => e.GetString()!)], got.Labels);
            Assert.Equal(entry.Value.GetProperty("n").GetInt32(), got.N);
            Assert.Equal(entry.Value.GetProperty("passthrough").GetBoolean(), got.Passthrough);
        }

        var answers = expected.GetProperty("answers");
        Assert.Equal([.. answers.EnumerateObject().Select(p => p.Name)], result.Ids);
        foreach (var entry in answers.EnumerateObject())
        {
            var want = entry.Value;
            var got = result[entry.Name];
            var where = $"{checkpoint}/{entry.Name}";
            switch (got.Type)
            {
                case QuestionType.Choice: PredictParityTests.AssertChoice(want, got.AsChoice(), where); break;
                case QuestionType.Score: PredictParityTests.AssertScore(want, got.AsScore(), where); break;
                default:
                    PredictParityTests.AssertClose(want.GetProperty("noul").GetDouble(), got.AsNoul().Probability,
                                                   PredictParityTests.ProbTolerance, where);
                    break;
            }
        }
    }
}
