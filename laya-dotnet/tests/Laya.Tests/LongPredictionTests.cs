namespace Laya.Tests;

public sealed class LongPredictionTests
{
    private static QuestionSet Questions() => new()
    {
        ["choice"] = Question.Choice("Pick a topic.", "a", "b"),
        ["score"] = Question.Score("How urgent?", "low", "high"),
        ["noul"] = Question.Noul("Customer cancels."),
    };

    [Theory]
    [InlineData(64, 64, 32, 1, 0, 64)]
    [InlineData(65, 64, 32, 2, 32, 65)]
    [InlineData(128, 64, 32, 3, 64, 128)]
    [InlineData(129, 64, 32, 4, 96, 129)]
    [InlineData(129, 64, 64, 3, 128, 129)]
    [InlineData(5, 1, 1, 5, 4, 5)]
    public void SpansCoverEveryTokenAndStopAtFirstCompleteWindow(
        int tokens, int window, int stride, int count, int lastStart, int lastEnd)
    {
        var spans = LongPrediction.Spans(tokens, window, stride);
        Assert.Equal(count, spans.Count);
        Assert.Equal(new WindowInfo(count - 1, lastStart, lastEnd, count), spans[^1]);
        Assert.Equal(0, spans[0].TokenStart);
        for (var i = 0; i < spans.Count; i++)
        {
            Assert.Equal(i, spans[i].Index);
            Assert.Equal(count, spans[i].Count);
            if (i > 0) Assert.True(spans[i].TokenStart <= spans[i - 1].TokenEnd);
        }
    }

    [Fact]
    public void SpansOfAnEmptyStateAreEmptyAndAShortStateIsOneSpan()
    {
        Assert.Empty(LongPrediction.Spans(0, 64, 32));                       // `while i < n` never enters
        Assert.Equal(new WindowInfo(0, 0, 10, 1), Assert.Single(LongPrediction.Spans(10, 64, 32)));
    }

    [Fact]
    public void SpansEndAtTheStateLengthNotPastIt()
    {
        // i + budget >= n breaks after the window that reaches the end; token_end = min(start + budget, n).
        var spans = LongPrediction.Spans(100, 64, 48);
        Assert.Equal([new WindowInfo(0, 0, 64, 2), new WindowInfo(1, 48, 100, 2)], spans);
    }

    [Theory]
    [InlineData(null, null, 312, 156)]
    [InlineData(0, -1, 312, 156)]
    [InlineData(-5, 0, 312, 156)]
    [InlineData(32, 32, 32, 32)]
    [InlineData(1, null, 1, 1)]
    public void BudgetUsesPythonDefaults(int? window, int? stride, int size, int step)
    {
        var (got, gotStep, _) = LongPrediction.Budget(new StubTokenizer(), Questions(), 512, 192, window, stride);
        Assert.Equal((size, step), (got, gotStep));
    }

    [Fact]
    public void BudgetCapsAtSmallestQuestionRoomAndAdjustsAValidRequestedStride()
    {
        var tok = new StubTokenizer();
        var questions = Questions();
        questions["many"] = Question.Choice("Pick.",
            Enumerable.Range(0, 25).Select(i => new KeyValuePair<string, object?>($"label{i}", "four word description here")));
        var room = questions.Ids.Min(id => SequenceBuilder.StateRoom(tok, questions[id], 160, 192));
        Assert.Equal((room, Math.Max(1, room / 2), room), LongPrediction.Budget(tok, questions, 160, 192, 1000, 500));
    }

    // window_budget returns `room = min(rooms) if rooms else size` (common.py:403); the single-pass
    // condition in predict_long reads it back (onnx_agent.py:553).
    [Fact]
    public void BudgetRoomIsTheSmallestQuestionRoomAndMayExceedTheDefaultWindow()
    {
        var tok = new StubTokenizer();
        var questions = Questions();
        var room = questions.Ids.Min(id => SequenceBuilder.StateRoom(tok, questions[id], 512, 192));
        var (size, _, got) = LongPrediction.Budget(tok, questions, 512, 192, null, null);
        Assert.Equal(room, got);
        Assert.Equal(312, size);
        Assert.True(got > size);
        // An explicit window below the room leaves the room untouched too.
        var (small, _, same) = LongPrediction.Budget(tok, questions, 512, 192, 100, null);
        Assert.Equal((100, room), (small, same));
    }

    [Theory]
    [InlineData(null, 312)]
    [InlineData(0, 312)]
    [InlineData(-3, 312)]
    [InlineData(40, 40)]
    [InlineData(2000, 2000)]
    public void BudgetWithNoQuestionsHasRoomEqualToTheRequestedWindow(int? window, int expected)
    {
        var (size, step, room) = LongPrediction.Budget(new StubTokenizer(), new QuestionSet(), 512, 192, window, null);
        Assert.Equal((expected, Math.Max(1, expected / 2), expected), (size, step, room));
    }

    [Fact]
    public void BudgetWithNoQuestionsUsesTheDefaultWindowFloorAndStillRefusesAStrideGap()
    {
        // max(64, max_len - head_max_len - 8): the 64 floor holds when head_max_len is widened.
        var (size, _, room) = LongPrediction.Budget(new StubTokenizer(), new QuestionSet(), 200, 192, null, null);
        Assert.Equal((64, 64), (size, room));
        // Nothing clamps the window, so a stride past it is the caller's error, not a library clamp.
        Assert.Throws<ArgumentException>(() =>
            LongPrediction.Budget(new StubTokenizer(), new QuestionSet(), 512, 192, 40, 41));
    }

    [Fact]
    public void BudgetRoomEqualsTheWindowWhenTheRoomCapsIt()
    {
        var tok = new StubTokenizer();
        var questions = Questions();
        questions["many"] = Question.Choice("Pick.",
            Enumerable.Range(0, 25).Select(i => new KeyValuePair<string, object?>($"label{i}", "four word description here")));
        var room = questions.Ids.Min(id => SequenceBuilder.StateRoom(tok, questions[id], 160, 192));
        Assert.InRange(room, 1, 311);
        foreach (int? window in new int?[] { null, 1000 })
        {
            var (size, _, got) = LongPrediction.Budget(tok, questions, 160, 192, window, null);
            Assert.Equal((room, room), (size, got));
        }
    }

    [Fact]
    public void BudgetRefusesAStrideThatOvershotTheRequestedWindowEvenWhenTheWindowWasClamped()
    {
        var tok = new StubTokenizer();
        var questions = Questions();
        questions["many"] = Question.Choice("Pick.",
            Enumerable.Range(0, 25).Select(i => new KeyValuePair<string, object?>($"label{i}", "four word description here")));
        // stride > requested window: the caller's mistake (common.py:441), reported against the clamped window.
        var error = Assert.Throws<ArgumentException>(() => LongPrediction.Budget(tok, questions, 160, 192, 1000, 1001));
        Assert.Contains("these questions leave room for", error.Message);
        // stride == requested window is valid for it, so it is clamped to half the effective window instead.
        var room = questions.Ids.Min(id => SequenceBuilder.StateRoom(tok, questions[id], 160, 192));
        Assert.Equal((room, Math.Max(1, room / 2), room), LongPrediction.Budget(tok, questions, 160, 192, 1000, 1000));
    }

    [Fact]
    public void BudgetRejectsGapsAndZeroStateRoom()
    {
        var error = Assert.Throws<ArgumentException>(() =>
            LongPrediction.Budget(new StubTokenizer(), Questions(), 512, 192, 32, 33));
        Assert.StartsWith("predict_long: stride=33 steps past the 32-token window", error.Message);
        Assert.Throws<ArgumentException>(() =>
            LongPrediction.Budget(new StubTokenizer(), Questions(), 10, 192, null, null));
        Assert.Throws<ArgumentException>(() =>
            LongPrediction.Budget(new StubTokenizer(), Questions(), 100, 192, 1000, 1001));
    }

    [Theory]
    [InlineData(100, 312, 312, null, null)]
    [InlineData(100, 200, 312, null, null)]
    [InlineData(100, 50, 312, null, 17)]
    [InlineData(100, 50, 312, 0, 17)]
    [InlineData(100, 50, 312, -1, 17)]
    [InlineData(100, 50, 312, 3, 3)]
    public void AutomaticBatchCapMatchesPython(int count, int window, int defaultBudget, int? batch, int? expected) =>
        Assert.Equal(expected, LongPrediction.BatchCap(count, window, defaultBudget, batch));

    // window_batch_cap: unclamped = ceil(n * max(1, w // 2) / max(1, budget // 2)); None when n <= 2 * unclamped.
    [Theory]
    [InlineData(100, 312, 312, 7, 7)]   // explicit batch_size is honoured even when the window is not capped
    [InlineData(100, 50, 1, null, null)] // window >= budget: not capped, and budget // 2 floors at 1
    [InlineData(1, 1, 312, null, null)]  // unclamped 1; 1 <= 2
    [InlineData(2, 50, 312, null, null)] // unclamped ceil(50/156) = 1; n == 2 * unclamped keeps the single pass
    [InlineData(3, 50, 312, null, 1)]    // one past the blow-up factor chunks at unclamped
    [InlineData(12, 50, 312, null, 2)]   // ceil(300/156) = 2; 12 > 4
    [InlineData(34, 50, 312, null, 6)]   // ceil(850/156) = 6; 34 > 12
    [InlineData(5, 200, 312, null, null)] // ceil(500/156) = 4; 5 <= 8: a mild cap
    [InlineData(100, 1, 312, null, 1)]   // window // 2 floors at 1: ceil(100/156) = 1
    public void BatchCapEdgeCasesMatchPython(int count, int window, int defaultBudget, int? batch, int? expected) =>
        Assert.Equal(expected, LongPrediction.BatchCap(count, window, defaultBudget, batch));

    private static LayaResult Result(double choiceConfidence, double scoreConfidence, double noul, int tokens,
                                     IReadOnlyDictionary<string, OptionUsage>? options = null) =>
        new("laya-rl-agent", ["choice", "score", "noul"], new(StringComparer.Ordinal)
        {
            ["choice"] = new ChoiceAnswer("a", [new("a", choiceConfidence), new("b", 1 - choiceConfidence)],
                0.1, choiceConfidence, new ActionInfo(choiceConfidence)),
            ["score"] = new ScoreAnswer(scoreConfidence, ["low", "high"], [1 - scoreConfidence, scoreConfidence],
                0.2, scoreConfidence, new ActionInfo(scoreConfidence)),
            ["noul"] = new NoulAnswer(noul, Math.Max(noul, 1 - noul), Math.Max(noul, 1 - noul), new ActionInfo(noul)),
        }, new Usage(tokens, 0)
        {
            StateTokens = tokens - 10, StateTokensDropped = 1, Truncated = 1,
            TruncatedQuestions = tokens == 20 ? ["score"] : [], Options = options,
        });

    [Fact]
    public void AggregationSelectsEvidenceNotEntropyAndPreservesFirstTiesAndWholeAnswer()
    {
        var first = Result(0.8, 0.6, 0.05, 20, new Dictionary<string, OptionUsage> { ["choice"] = new(2, 1, 4) });
        var second = Result(0.8, 0.9, 0.7, 30, new Dictionary<string, OptionUsage> { ["score"] = new(2, 1, 4) });
        var spans = LongPrediction.Spans(65, 64, 32);
        var got = LongPrediction.Aggregate(Questions(), [first, second], spans);
        Assert.Equal(new[] { "choice", "score", "noul" }, got.Ids);
        Assert.Equal(spans[0], got["choice"].Window);
        Assert.Equal(spans[1], got["score"].Window);
        Assert.Equal(spans[1], got["noul"].Window);
        Assert.Equal(first["choice"].Action, got["choice"].Action);
        Assert.Equal(second["score"].AsScore().Score, got["score"].AsScore().Score);
        Assert.Equal(0.7, got["noul"].AsNoul().Probability);
        Assert.Null(first["choice"].Window);
        Assert.Equal(50, got.Usage.InputTokens);
        Assert.Equal(30, got.Usage.StateTokens);
        Assert.Equal(2, got.Usage.StateTokensDropped);
        Assert.Equal(2, got.Usage.Truncated);
        Assert.Empty(got.Usage.TruncatedQuestions!);
        Assert.Equal(new[] { "choice", "score" }, got.Usage.Options!.Keys);
    }

    [Fact]
    public void AggregationKeepsTheFirstWindowOnTiesAndLetsALaterOptionsRecordWin()
    {
        var a = Result(0.8, 0.6, 0.7, 20, new Dictionary<string, OptionUsage> { ["choice"] = new(2, 1, 4) });
        var b = Result(0.8, 0.6, 0.7, 30, new Dictionary<string, OptionUsage> { ["choice"] = new(5, 3, 9) });
        var spans = LongPrediction.Spans(65, 64, 32);
        var got = LongPrediction.Aggregate(Questions(), [a, b], spans);
        foreach (var id in new[] { "choice", "score", "noul" }) Assert.Equal(spans[0], got[id].Window);
        // Python merges `{**prev, **val}`: the later window's record for the same id replaces the earlier.
        Assert.Equal(new OptionUsage(5, 3, 9), got.Usage.Options!["choice"]);
        Assert.Equal(2, got.Usage.Windows);
    }

    [Fact]
    public void AggregationOfOneWindowAndOfUsageWithoutTruncationFields()
    {
        var one = new LayaResult("laya-rl-agent", ["choice", "score", "noul"], Result(0.8, 0.6, 0.7, 20).Answers.ToDictionary(
            p => p.Key, p => p.Value, StringComparer.Ordinal), new Usage(20, 5));
        var got = LongPrediction.Aggregate(Questions(), [one], LongPrediction.Spans(10, 64, 32));
        Assert.Equal(new WindowInfo(0, 0, 10, 1), got["noul"].Window);
        Assert.Equal(20, got.Usage.InputTokens);
        Assert.Equal(0, got.Usage.OutputTokens);   // usage["output_tokens"] = 0 (onnx_agent.py:659)
        Assert.Equal(1, got.Usage.Windows);
        Assert.Null(got.Usage.StateTokens);
        Assert.Null(got.Usage.StateTokensDropped);
        Assert.Null(got.Usage.Truncated);
        Assert.Null(got.Usage.Options);
    }

    [Fact]
    public void AggregationCarriesTheLastWindowsTruncatedQuestions()
    {
        var spans = LongPrediction.Spans(65, 64, 32);
        var got = LongPrediction.Aggregate(Questions(), [Result(0.8, 0.6, 0.7, 20), Result(0.8, 0.6, 0.7, 30)], spans);
        Assert.Empty(got.Usage.TruncatedQuestions!);        // the last window's list, though the first named "score"
        Assert.Equal(2, got.Usage.Truncated);
    }

    private sealed class Scanner : ILayaLongPredictor
    {
        public bool Called;
        public object? State;
        public (int?, int?, string, int?) Options;
        public LayaResult Predict(object? state, QuestionSet questions) => throw new InvalidOperationException("plain path used");
        public LayaResult PredictLong(object state, QuestionSet questions, int? window = null, int? stride = null,
                                       string aggregate = "auto", int? batchSize = null)
        {
            Called = true;
            State = state;
            Options = (window, stride, aggregate, batchSize);
            return Result(0.7, 0.8, 0.6, 20);
        }
    }

    [Fact]
    public void RouterRoutesOriginalStateAndForwardsTheScanArguments()
    {
        var scanner = new Scanner();
        using var router = new LayaRouter(null, _ => scanner);
        var got = router.PredictLong("Mein Konto wurde zweimal belastet.", Questions(),
            model: null, window: 32, stride: 16, batchSize: 2);
        Assert.True(scanner.Called);
        Assert.Equal("Mein Konto wurde zweimal belastet.", scanner.State);
        Assert.Equal((32, 16, "auto", 2), scanner.Options);
        Assert.Equal(LayaCheckpoint.Multilingual, got.Routing!.Model);
    }

    private sealed class PlainPredictor : ILayaPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) => throw new InvalidOperationException("must not truncate");
    }

    [Fact]
    public void RouterRejectsPredictorsWithoutScanSupportAndNullState()
    {
        using var router = new LayaRouter(null, _ => new PlainPredictor());
        Assert.Throws<NotSupportedException>(() => router.PredictLong("hello", Questions()));
        Assert.Throws<ArgumentNullException>(() => router.PredictLong(null!, Questions()));
    }
}
