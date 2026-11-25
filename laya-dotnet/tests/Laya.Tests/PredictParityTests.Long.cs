using Laya.Tokenization;
using System.Text.Json;

namespace Laya.Tests;

public sealed partial class PredictParityTests
{
    public static IEnumerable<object[]> LongSuccessCases() =>
        from checkpoint in TestArtifacts.ParityCheckpoints
        where GoldenData.For(checkpoint).Available
        from info in GoldenData.For(checkpoint).Index.Where(i => i.Kind == "long" && !i.ExpectsError)
        from split in TestArtifacts.ParitySplitFlags
        select new object[] { checkpoint, info, split };

    public static IEnumerable<object[]> LongErrorCases() =>
        from checkpoint in TestArtifacts.ParityCheckpoints
        where GoldenData.For(checkpoint).Available
        from info in GoldenData.For(checkpoint).Index.Where(i => i.Kind == "long" && i.ExpectsError)
        select new object[] { checkpoint, info };

    [Theory]
    [Trait(TestArtifacts.LayoutTrait, TestArtifacts.AnyLayout)]
    [MemberData(nameof(LongSuccessCases))]
    public void ReproducesLongPredictions(LayaCheckpoint checkpoint, GoldenCaseInfo info, bool split)
    {
        using var doc = GoldenData.For(checkpoint).Load(info);
        var root = doc.RootElement;
        var engine = fixture.RequireFor(checkpoint, root, split);
        var state = GoldenData.ToClr(root.GetProperty("state"))!;
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var options = root.GetProperty("options");
        var lang = CallOptions.Lang(options);
        var got = engine.PredictLong(state, questions, new LayaLongOptions
        {
            Window = LongInt(options, "window"),
            Stride = LongInt(options, "stride"),
            Aggregate = options.TryGetProperty("aggregate", out var aggregate) ? aggregate.GetString()! : "auto",
            BatchSize = LongInt(options, "batch_size"),
            Lang = lang,
        });
        var expected = root.GetProperty("result");
        // The SDK keeps its existing model identifier instead of Python ONNXAgent's suffix.
        Assert.Equal("laya-rl-agent-onnx", expected.GetProperty("model").GetString());
        Assert.Equal("laya-rl-agent", got.Model);
        AssertLongUsage(expected.GetProperty("usage"), got.Usage);
        var answers = expected.GetProperty("answers");
        Assert.Equal(answers.EnumerateObject().Select(p => p.Name), got.Ids);
        foreach (var entry in answers.EnumerateObject())
        {
            var want = entry.Value;
            var answer = got[entry.Name];
            var where = $"{checkpoint}/{split}/{info.Name}/{entry.Name}";
            Assert.Equal(want.GetProperty("type").GetString(), SequenceBuilder.TypeName(answer.Type));
            AssertClose(want.GetProperty("confidence").GetDouble(), answer.Confidence, ProbTolerance, where + ".confidence");
            AssertClose(want.GetProperty("answer_confidence").GetDouble(), answer.AnswerConfidence,
                        ProbTolerance, where + ".answer_confidence");
            AssertClose(want.GetProperty("action").GetProperty("act_probability").GetDouble(),
                        answer.Action.ActProbability, ProbTolerance, where + ".action");
            switch (answer.Type)
            {
                case QuestionType.Choice: AssertChoice(want, answer.AsChoice(), where); break;
                case QuestionType.Score: AssertScore(want, answer.AsScore(), where); break;
                case QuestionType.Noul:
                    AssertClose(want.GetProperty("noul").GetDouble(), answer.AsNoul().Probability, ProbTolerance, where);
                    break;
            }
            if (want.TryGetProperty("window", out var window))
                Assert.Equal(new WindowInfo(window.GetProperty("index").GetInt32(),
                    window.GetProperty("token_start").GetInt32(), window.GetProperty("token_end").GetInt32(),
                    window.GetProperty("count").GetInt32()), answer.Window);
            else
                Assert.Null(answer.Window);
        }

        // These comparisons pin token offsets, decoder cleanup and special-token retention
        // independently of model probabilities (a dropped token could otherwise round away).
        using var tokenizer = new HfTokenizer(engine.Artifacts.TokenizerPath);
        var stateIds = tokenizer.Encode(PythonJson.State(state).Replace(tokenizer.MaskToken, " ", StringComparison.Ordinal));
        Assert.Equal(root.GetProperty("state_tokens").GetInt32(), stateIds.Length);
        var (budget, step, room) = LongPrediction.Budget(tokenizer, questions, engine.Config.MaxLen,
            engine.Config.HeadMaxLen, LongInt(options, "window"), LongInt(options, "stride"));
        Assert.Equal(root.GetProperty("effective_window").GetInt32(), budget);
        Assert.Equal(root.GetProperty("effective_stride").GetInt32(), step);
        Assert.Equal(root.GetProperty("effective_room").GetInt32(), room);
        foreach (var window in root.GetProperty("windows").EnumerateArray())
            Assert.Equal(window.GetProperty("text").GetString(), tokenizer.Decode(stateIds[
                window.GetProperty("token_start").GetInt32()..window.GetProperty("token_end").GetInt32()]));

        if (info.Name == "long_exact_limit")
        {
            Assert.Equal(budget, stateIds.Length);
            Assert.Equal(1, got.Usage.Windows);
        }
        if (info.Name == "long_room_fast_path")
        {
            Assert.True(stateIds.Length > budget);
            Assert.Equal(room, stateIds.Length);
            Assert.Equal(1, got.Usage.Windows);
        }
        if (got.Usage.Windows == 1 && questions.Count > 0)
        {
            var plain = engine.Predict(state, questions, new LayaPredictOptions { Lang = lang });
            Assert.Equal(plain.Usage.InputTokens, got.Usage.InputTokens);
            foreach (var id in questions.Ids)
            {
                Assert.Equal(plain[id].Confidence, got[id].Confidence);
                Assert.Equal(plain[id].AnswerConfidence, got[id].AnswerConfidence);
                Assert.Equal(plain[id].Action, got[id].Action);
            }
        }
    }

    [Theory]
    [MemberData(nameof(LongErrorCases))]
    public void ReproducesLongErrors(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        var engine = fixture.Require(checkpoint);
        using var doc = GoldenData.For(checkpoint).Load(info);
        var root = doc.RootElement;
        var options = root.GetProperty("options");
        var error = Assert.Throws<ArgumentException>(() => engine.PredictLong(
            GoldenData.ToClr(root.GetProperty("state"))!, GoldenData.BuildQuestions(root.GetProperty("questions")),
            LongInt(options, "window"), LongInt(options, "stride"),
            options.TryGetProperty("aggregate", out var aggregate) ? aggregate.GetString()! : "auto",
            LongInt(options, "batch_size")));
        Assert.StartsWith(root.GetProperty("error_message").GetString()!, error.Message, StringComparison.Ordinal);
    }

    private static int? LongInt(JsonElement options, string name) =>
        options.TryGetProperty(name, out var value) ? value.GetInt32() : null;

    private static void AssertLongUsage(JsonElement want, Usage got)
    {
        Assert.Equal(want.GetProperty("input_tokens").GetInt32(), got.InputTokens);
        Assert.Equal(want.GetProperty("output_tokens").GetInt32(), got.OutputTokens);
        Assert.Equal(want.GetProperty("windows").GetInt32(), got.Windows);
        Assert.Equal(LongInt(want, "state_tokens"), got.StateTokens);
        Assert.Equal(LongInt(want, "state_tokens_dropped"), got.StateTokensDropped);
        int? truncated = want.TryGetProperty("truncated", out var t)
            ? t.ValueKind == JsonValueKind.Number ? t.GetInt32() : t.GetBoolean() ? 1 : 0 : null;
        Assert.Equal(truncated, got.Truncated);
        if (want.TryGetProperty("truncated_questions", out var ids))
            Assert.Equal(ids.EnumerateArray().Select(id => id.GetString()!), got.TruncatedQuestions);
        else
            Assert.Null(got.TruncatedQuestions);
        if (want.TryGetProperty("options", out var options))
        {
            Assert.NotNull(got.Options);
            Assert.Equal(options.EnumerateObject().Select(p => p.Name), got.Options.Keys);
            foreach (var entry in options.EnumerateObject())
            {
                var tokens = entry.Value.GetProperty("tokens_per_option");
                Assert.Equal(new OptionUsage(entry.Value.GetProperty("total").GetInt32(),
                    entry.Value.GetProperty("distinct").GetInt32(),
                    tokens.ValueKind == JsonValueKind.Null ? null : tokens.GetInt32()), got.Options[entry.Name]);
            }
        }
        else
            Assert.Null(got.Options);
    }
}
