using Laya.Tokenization;
using System.Text.Json;

namespace Laya.Tests;

public sealed partial class PredictParityTests
{
    public static IEnumerable<object[]> BatchSuccessCases() =>
        from checkpoint in TestArtifacts.ParityCheckpoints
        where GoldenData.For(checkpoint).Available
        from info in GoldenData.For(checkpoint).Index.Where(i => i.Kind == "batch" && !i.ExpectsError)
        from split in TestArtifacts.ParitySplitFlags
        select new object[] { checkpoint, info, split };

    public static IEnumerable<object[]> BatchErrorCases() =>
        from checkpoint in TestArtifacts.ParityCheckpoints
        where GoldenData.For(checkpoint).Available
        from info in GoldenData.For(checkpoint).Index.Where(i => i.Kind == "batch" && i.ExpectsError)
        select new object[] { checkpoint, info };

    private static LayaBatchOptions BatchOptions(JsonElement options) => new()
    {
        BatchSize = options.TryGetProperty("batch_size", out var size) ? size.GetInt32() : null,
        SortByLength = options.TryGetProperty("sort_by_length", out var sort) && sort.GetBoolean(),
        MinConfidence = ConfidenceGateTests.MinConfidenceIn(options),
        Lang = CallOptions.Lang(options),
        MaxLen = CallOptions.Budget(options, "max_len"),
        HeadMaxLen = CallOptions.Budget(options, "head_max_len"),
    };

    private static List<object?> BatchStates(JsonElement root) =>
        [.. root.GetProperty("states").EnumerateArray().Select(GoldenData.ToClr)];

    [Theory]
    [Trait(TestArtifacts.LayoutTrait, TestArtifacts.AnyLayout)]
    [MemberData(nameof(BatchSuccessCases))]
    public void ReproducesBatchPredictions(LayaCheckpoint checkpoint, GoldenCaseInfo info, bool split)
    {
        using var doc = GoldenData.For(checkpoint).Load(info);
        var root = doc.RootElement;
        var engine = fixture.RequireFor(checkpoint, root, split);
        var states = BatchStates(root);
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));

        var got = engine.PredictBatch(states, questions, BatchOptions(root.GetProperty("options")));

        var expected = root.GetProperty("results").EnumerateArray().ToArray();
        Assert.Equal(expected.Length, got.Count);
        for (var s = 0; s < expected.Length; s++)
        {
            var where = $"{checkpoint}/{split}/{info.Name}[{s}]";
            Assert.Equal("laya-rl-agent-onnx", expected[s].GetProperty("model").GetString());
            Assert.Equal("laya-rl-agent", got[s].Model);
            AssertBatchUsage(expected[s].GetProperty("usage"), got[s].Usage);
            var answers = expected[s].GetProperty("answers");
            Assert.Equal(answers.EnumerateObject().Select(p => p.Name), got[s].Ids);
            foreach (var entry in answers.EnumerateObject())
            {
                var want = entry.Value;
                var answer = got[s][entry.Name];
                var at = $"{where}/{entry.Name}";
                Assert.Equal(want.GetProperty("type").GetString(), SequenceBuilder.TypeName(answer.Type));
                AssertClose(want.GetProperty("confidence").GetDouble(), answer.Confidence, ProbTolerance, at + ".confidence");
                ConfidenceGateTests.AssertGateFields(want, answer, at);
                AssertClose(want.GetProperty("answer_confidence").GetDouble(), answer.AnswerConfidence,
                            ProbTolerance, at + ".answer_confidence");
                AssertClose(want.GetProperty("action").GetProperty("act_probability").GetDouble(),
                            answer.Action.ActProbability, ProbTolerance, at + ".action");
                switch (answer.Type)
                {
                    case QuestionType.Choice: AssertChoice(want, answer.AsChoice(), at); break;
                    case QuestionType.Score: AssertScore(want, answer.AsScore(), at); break;
                    case QuestionType.Noul:
                        AssertClose(want.GetProperty("noul").GetDouble(), answer.AsNoul().Probability, ProbTolerance, at);
                        break;
                }
            }
        }
    }

    // Python's own session-run shapes against this port's chunk planning, over this port's own
    // token counts: each state's width (its longest question row) must equal Python's, and the
    // planned runs must be the ones Python made.
    [Theory]
    [MemberData(nameof(BatchSuccessCases))]
    public void PlansTheRecordedSessionRuns(LayaCheckpoint checkpoint, GoldenCaseInfo info, bool split)
    {
        if (split) return; // planning is layout-independent
        var engine = fixture.Require(checkpoint);
        using var doc = GoldenData.For(checkpoint).Load(info);
        var root = doc.RootElement;
        var states = BatchStates(root);
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        if (questions.Count == 0 || states.Count == 0) return;

        var options = BatchOptions(root.GetProperty("options"));
        using var tokenizer = new HfTokenizer(engine.Artifacts.TokenizerPath);
        var maxLen = options.MaxLen ?? engine.Config.MaxLen;
        var headMaxLen = options.HeadMaxLen ?? engine.Config.HeadMaxLen;
        var widths = states.Select(state => questions.Max(q =>
            SequenceBuilder.Build(tokenizer, state, q.Value, maxLen, headMaxLen,
                                  PythonJson.IsListState(state)).Ids.Length)).ToArray();
        Assert.Equal(root.GetProperty("state_widths").EnumerateArray().Select(w => w.GetInt32()), widths);

        var runs = new List<int[]>();
        BatchPlanning.Execute<int, int>(
            [.. Enumerable.Range(0, states.Count).Cast<object?>()], options.BatchSize, options.SortByLength,
            part => [.. part.Select(index => widths[(int)index!])],
            width => width,
            chunk =>
            {
                runs.Add([chunk.Count * questions.Count, chunk.Max()]);
                return [.. chunk];
            });
        Assert.Equal(
            root.GetProperty("runs").EnumerateArray().Select(r => r.EnumerateArray().Select(d => d.GetInt32()).ToArray()),
            runs);
    }

    // system_one is predict_batch([state]): the single-state path agrees with the batch on the
    // two edges that used to differ.
    [Theory]
    [InlineData(LayaCheckpoint.English)]
    [InlineData(LayaCheckpoint.Multilingual)]
    [InlineData(LayaCheckpoint.TypedDecisions)]
    public void PlainPredictIsAOneStateBatch(LayaCheckpoint checkpoint)
    {
        var engine = fixture.Require(checkpoint);
        var empty = engine.Predict("anything", new QuestionSet());
        Assert.Equal(0, empty.Count);
        Assert.Equal(new Usage(0, 0), empty.Usage);
        var none = Assert.Throws<ArgumentNullException>(() => engine.Predict(null, new QuestionSet()));
        Assert.Equal("state must not be None; pass a string, dict, or list", none.Message);
    }

    [Theory]
    [MemberData(nameof(BatchErrorCases))]
    public void ReproducesBatchErrors(LayaCheckpoint checkpoint, GoldenCaseInfo info)
    {
        var engine = fixture.Require(checkpoint);
        using var doc = GoldenData.For(checkpoint).Load(info);
        var root = doc.RootElement;
        var states = BatchStates(root);
        var questions = GoldenData.BuildQuestions(root.GetProperty("questions"));
        var error = Assert.ThrowsAny<ArgumentException>(
            () => engine.PredictBatch(states, questions, BatchOptions(root.GetProperty("options"))));
        Assert.StartsWith(root.GetProperty("error_message").GetString()!, error.Message, StringComparison.Ordinal);
    }

    private static void AssertBatchUsage(JsonElement want, Usage got)
    {
        Assert.Equal(want.GetProperty("input_tokens").GetInt32(), got.InputTokens);
        Assert.Equal(want.GetProperty("output_tokens").GetInt32(), got.OutputTokens);
        Assert.Null(got.Windows);
        Assert.Equal(OptInt(want, "state_tokens"), got.StateTokens);
        Assert.Equal(OptInt(want, "state_tokens_dropped"), got.StateTokensDropped);
        int? truncated = want.TryGetProperty("truncated", out var t) ? t.GetBoolean() ? 1 : 0 : null;
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

    private static int? OptInt(JsonElement obj, string name) =>
        obj.TryGetProperty(name, out var value) ? value.GetInt32() : null;
}
