namespace Laya.Tests;

/// <summary>
/// Tier 2: <see cref="LayaRouter"/> driving real checkpoints end to end. Needs the ONNX artifacts
/// for both the english and multilingual checkpoints (set <c>LAYA_ONNX_ROOT</c>); skipped
/// entirely otherwise, same convention as <see cref="PredictParityTests"/>.
/// </summary>
/// <remarks>
/// Joins <see cref="AllEnginesCollection"/> purely to run sequentially with the other
/// engine-loading test classes, never concurrently with them: the underlying native tokenizer
/// binding is not safe to construct from two threads at once for the same checkpoint, and this
/// class always builds its own fresh <see cref="LayaEngine"/> instances rather than sharing
/// <see cref="AllEnginesFixture"/>'s.
/// </remarks>
[Collection(AllEnginesCollection.Name)]
public sealed class RouterEndToEndTests
{
    private static void RequireEnglishAndMultilingual()
    {
        if (!TestArtifacts.For(LayaCheckpoint.English).HasModel
            || !TestArtifacts.For(LayaCheckpoint.Multilingual).HasModel)
            Assert.Skip("router end-to-end tests need both english and multilingual ONNX artifacts; "
                + "set LAYA_ONNX_ROOT to the parent of all checkpoints");
    }

    // Build each engine from the directory TestArtifacts found (the repo walk-up included), not
    // the router's default resolution, which only checks LAYA_ONNX_ROOT and the cache: otherwise the
    // skip check above can pass while the router still fails to find the artifacts.
    private static LayaRouter NewRouter() => new(new LayaRouterOptions { MaxLoaded = 1 }, checkpoint =>
        LayaEngine.Create(new LayaOptions
        {
            Checkpoint = checkpoint,
            ModelDirectory = TestArtifacts.For(checkpoint).Directory,
        }));

    [Fact]
    public void RoutesTheEnglishSampleToEnglishAndMatchesItsGolden()
    {
        RequireEnglishAndMultilingual();
        // MaxLoaded=1: this test only ever routes to one checkpoint, and each test method builds
        // its own router (disposed at the end of the method), so peak stays at one engine here.
        using var router = NewRouter();
        AssertRoutesAndMatchesGolden(router, LayaCheckpoint.English, "case_sample_app_english.json");
    }

    [Fact]
    public void RoutesTheHindiSampleToMultilingualAndMatchesItsGolden()
    {
        RequireEnglishAndMultilingual();
        using var router = NewRouter();
        AssertRoutesAndMatchesGolden(router, LayaCheckpoint.Multilingual, "case_sample_app_hindi.json");
    }

    private static void AssertRoutesAndMatchesGolden(LayaRouter router, LayaCheckpoint expectedCheckpoint, string file)
    {
        var golden = GoldenData.For(expectedCheckpoint);
        using var doc = golden.Load(file);
        var root = doc.RootElement;
        var state = GoldenData.ToClr(root.GetProperty("state"));
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));

        var result = router.Predict(state, questions);

        Assert.NotNull(result.Routing);
        Assert.Equal(expectedCheckpoint, result.Routing!.Model);
        Assert.Equal(root.GetProperty("result").GetProperty("model").GetString(), result.Model);

        AssertMatchesRecordedAnswers(root.GetProperty("result").GetProperty("answers"), result);
    }

    private static void AssertMatchesRecordedAnswers(System.Text.Json.JsonElement answers, LayaResult result)
    {
        // Tolerances mirror PredictParityTests: 4-dp recorded values, so a rounding boundary is
        // 1e-4 wide; a score sums several such errors.
        const double probTolerance = 2e-4;
        const double scoreTolerance = 2e-3;

        foreach (var entry in answers.EnumerateObject())
        {
            var got = result[entry.Name];
            switch (got.Type)
            {
                case QuestionType.Choice:
                    Assert.Equal(entry.Value.GetProperty("choice").GetString(), got.AsChoice().Choice);
                    break;
                case QuestionType.Score:
                    Assert.Equal(entry.Value.GetProperty("score").GetDouble(), got.AsScore().Score, scoreTolerance);
                    break;
                case QuestionType.Noul:
                    Assert.Equal(entry.Value.GetProperty("noul").GetDouble(), got.AsNoul().Probability, probTolerance);
                    break;
            }
            Assert.Equal(entry.Value.GetProperty("confidence").GetDouble(), got.Confidence, probTolerance);
        }
    }

    [Fact]
    public void EightThreadsAlternatingCheckpointsWithMaxLoaded1StayConsistent()
    {
        RequireEnglishAndMultilingual();

        using var englishDoc = GoldenData.For(LayaCheckpoint.English).Load("case_sample_app_english.json");
        using var hindiDoc = GoldenData.For(LayaCheckpoint.Multilingual).Load("case_sample_app_hindi.json");
        var englishState = GoldenData.ToClr(englishDoc.RootElement.GetProperty("state"));
        var englishQuestions = GoldenData.BuildQuestions(englishDoc.RootElement.GetProperty("questions"));
        var hindiState = GoldenData.ToClr(hindiDoc.RootElement.GetProperty("state"));
        var hindiQuestions = GoldenData.BuildQuestions(hindiDoc.RootElement.GetProperty("questions"));

        // Ground truth comes from the golden files (already verified bit-for-bit by
        // PredictParityTests), not a second live inference pass: an extra router here would
        // transiently double the resident engine count, and the recorded answer is exactly what
        // a correct single-threaded call would produce anyway.
        var wantEnglish = englishDoc.RootElement.GetProperty("result").GetProperty("answers")
            .GetProperty("department").GetProperty("choice").GetString();
        var wantHindi = hindiDoc.RootElement.GetProperty("result").GetProperty("answers")
            .GetProperty("department").GetProperty("choice").GetString();

        // MaxLoaded=1 forces the router to evict and reload the other checkpoint on every
        // alternation, so at most one engine (plus a transient second one mid-handoff) is
        // resident at a time.
        using var router = NewRouter();
        var results = new (string Department, LayaCheckpoint Model)[8];
        var errors = new System.Collections.Concurrent.ConcurrentBag<Exception>();

        Parallel.For(0, results.Length, i =>
        {
            try
            {
                var isEnglish = i % 2 == 0;
                var result = router.Predict(
                    isEnglish ? englishState : hindiState,
                    isEnglish ? englishQuestions : hindiQuestions);
                results[i] = (result["department"].AsChoice().Choice, result.Routing!.Model);
            }
            catch (Exception ex)
            {
                errors.Add(ex);
            }
        });

        Assert.Empty(errors); // no use of a disposed engine, no other failure under eviction pressure
        for (var i = 0; i < results.Length; i++)
        {
            var isEnglish = i % 2 == 0;
            Assert.Equal(isEnglish ? LayaCheckpoint.English : LayaCheckpoint.Multilingual, results[i].Model);
            Assert.Equal(isEnglish ? wantEnglish : wantHindi, results[i].Department);
        }
    }
}
