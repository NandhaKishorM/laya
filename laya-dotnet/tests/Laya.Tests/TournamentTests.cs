using System.Diagnostics;
using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Tier 1: <see cref="LayaShortlist.PredictTournament(ILayaPredictor, object, QuestionSet, int)"/> against
/// <c>golden/routing/tournament_probe.json</c>, which Python's own <c>predict_tournament</c> produced over a
/// scripted predictor (no model): every call it made, the warnings it raised, the result or the error.
/// </summary>
public sealed class TournamentTests
{
    private static JsonDocument Probe() => RoutingGoldenData.Load("tournament_probe.json");

    public static TheoryData<string> CaseNames()
    {
        using var doc = Probe();
        return new TheoryData<string>(doc.RootElement.EnumerateArray().Select(e => e.GetProperty("name").GetString()!));
    }

    private sealed class Capture : TraceListener
    {
        private readonly int _thread = Environment.CurrentManagedThreadId;
        private readonly List<string> _messages = [];
        public List<string> Messages { get { lock (_messages) return [.. _messages]; } }
        public override void Write(string? message) { }
        public override void WriteLine(string? message) { }
        public override void TraceEvent(TraceEventCache? eventCache, string source, TraceEventType eventType, int id, string? format, params object?[]? args)
        {
            if (Environment.CurrentManagedThreadId != _thread) return;
            lock (_messages) _messages.Add(args is null ? format ?? "" : string.Format(format ?? "", args));
        }
    }

    /// <summary>The scripted predictor of <c>_ScriptedAgent</c> in dump_routing_golden.py.</summary>
    private sealed class Scripted(string mode)
    {
        public List<QuestionSet> Calls { get; } = [];

        public LayaResult? Predict(object? state, QuestionSet questions)
        {
            Calls.Add(questions);
            var number = Calls.Count;
            if (mode == "none") return null;
            var order = new List<string>();
            var byId = new Dictionary<string, Answer>(StringComparer.Ordinal);
            foreach (var (qid, q) in questions)
            {
                order.Add(qid);
                byId[qid] = q switch
                {
                    ChoiceQuestion choice => Pick(choice, number),
                    ScoreQuestion => new ScoreAnswer(1.5, ["low"], [1.0], 1.0, 1.0, new ActionInfo(1.0)),
                    _ => new NoulAnswer(0.7, 1.0, 1.0, new ActionInfo(1.0)),
                };
            }
            if (mode == "omit_group" && number == 1) { order.Remove("1"); byId.Remove("1"); }
            if (mode == "score_group" && number == 1)
                byId["1"] = new ScoreAnswer(1.5, ["low"], [1.0], 1.0, 1.0, new ActionInfo(1.0));
            return new LayaResult("scripted", order, byId, new Usage(100 * number + questions.Count, 0));
        }

        private ChoiceAnswer Pick(ChoiceQuestion choice, int number)
        {
            var labels = choice.Labels;
            var pick = labels[labels.Count * 2 / 3];
            if (mode == "bad_winner" && number == 1) pick = "no-such-label";
            return new ChoiceAnswer(pick, [new(pick, 1.0)], 1.0, 1.0, new ActionInfo(1.0));
        }
    }

    // Everything a question shows the model, as a string: id, type, instructions, options, order.
    internal static string Describe(QuestionSet questions) => string.Join("\n", questions.Select(pair =>
    {
        var (id, q) = pair;
        var options = q switch
        {
            ChoiceQuestion c => string.Join(",", c.Options.Select(o => $"{o.Key}={o.Value}")),
            ScoreQuestion s => string.Join(",", s.Levels),
            NoulQuestion n => $"{n.IfFalse}|{n.IfTrue}",
            _ => "",
        };
        return $"{id}|{q.Type}|{q.Instructions}|{options}|{(q.OptionOrder is null ? "-" : string.Join(",", q.OptionOrder))}";
    }));

    /// <summary>
    /// The same set with every caller id suffixed (so a warning's text is unique to the test); the group
    /// questions of a round ("0", "1", ...) are the module's own ids and keep their names.
    /// </summary>
    private static QuestionSet Suffixed(QuestionSet questions, string suffix)
    {
        var set = new QuestionSet();
        foreach (var (id, q) in questions) set[id.All(char.IsAsciiDigit) ? id : id + suffix] = q;
        return set;
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void MatchesPythonsOwnPredictTournament(string name)
    {
        using var doc = Probe();
        var rec = doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        var suffix = "-" + Guid.NewGuid().ToString("N");
        var state = GoldenData.ToClr(rec.GetProperty("state"));
        var questions = Suffixed(GoldenData.BuildQuestions(rec.GetProperty("questions")), suffix);
        var mode = rec.GetProperty("mode").GetString()!;
        var scripted = new Scripted(mode);
        var forwarded = 0;
        Func<object?, QuestionSet, LayaResult> predict = (s, q) => { forwarded++; return scripted.Predict(s, q)!; };

        var capture = new Capture();
        Trace.Listeners.Add(capture);
        LayaResult? result = null;
        Exception? error = null;
        try
        {
            try
            {
                result = rec.TryGetProperty("group_size", out var g)
                    ? LayaShortlist.PredictTournament(predict, state, questions, g.GetInt32())
                    : LayaShortlist.PredictTournament(predict, state, questions);
            }
            catch (Exception e) { error = e; }
        }
        finally { Trace.Listeners.Remove(capture); }

        // The calls: the same questions, in the same order, as Python sent them.
        var calls = rec.GetProperty("calls").EnumerateArray().ToList();
        Assert.Equal(calls.Count, scripted.Calls.Count);
        Assert.Equal(calls.Count, forwarded);
        for (var i = 0; i < calls.Count; i++)
        {
            var want = Suffixed(GoldenData.BuildQuestions(calls[i].GetProperty("questions")), suffix);
            Assert.Equal(Describe(want), Describe(scripted.Calls[i]));
        }

        // The warnings: Python's text, in Python's order, with the test's ids.
        var wantWarnings = rec.GetProperty("warnings").EnumerateArray()
            .Select(w => Assert.IsType<string>(w.GetProperty("message").GetString()))
            .Select(m => SuffixQuestionIn(m, suffix)).ToList();
        Assert.Equal(wantWarnings, capture.Messages.Where(m => m.Contains(suffix, StringComparison.Ordinal)));

        var wantError = rec.GetProperty("error");
        if (wantError.ValueKind != JsonValueKind.Null)
        {
            Assert.NotNull(error);
            Assert.Null(result);
            var type = wantError.GetProperty("type").GetString();
            Assert.IsType(type switch
            {
                "ValueError" => typeof(ArgumentException),
                "KeyError" => typeof(KeyNotFoundException),
                "TypeError" => typeof(InvalidOperationException),
                _ => throw new InvalidOperationException(type),
            }, error);
            Assert.Equal(wantError.GetProperty("message").GetString(), error!.Message);
            return;
        }

        Assert.Null(error);
        var want_ = rec.GetProperty("result");
        Assert.Equal(want_.GetProperty("model").GetString(), result!.Model);
        Assert.Equal(want_.GetProperty("usage").GetProperty("input_tokens").GetInt32(), result.Usage.InputTokens);
        Assert.Equal([.. want_.GetProperty("answers").EnumerateObject().Select(p => p.Name + suffix)], result.Ids);
        foreach (var answer in want_.GetProperty("answers").EnumerateObject())
            if (answer.Value.TryGetProperty("choice", out var choice))
                Assert.Equal(choice.GetString(), result[answer.Name + suffix].AsChoice().Choice);

        var tournament = want_.GetProperty("tournament");
        Assert.NotNull(result.Tournament);
        Assert.Equal([.. tournament.EnumerateObject().Select(p => p.Name + suffix)], result.Tournament!.Keys);
        foreach (var entry in tournament.EnumerateObject())
        {
            var got = result.Tournament[entry.Name + suffix];
            Assert.Equal([.. entry.Value.GetProperty("labels").EnumerateArray().Select(e => e.GetString()!)], got.Labels);
            Assert.Equal(entry.Value.GetProperty("n").GetInt32(), got.N);
            Assert.Equal(entry.Value.GetProperty("rounds").GetInt32(), got.Rounds);
        }
    }

    private static string SuffixQuestionIn(string message, string suffix)
    {
        // "laya.shortlist: question 'intent': 'option_order' (...)": the question id is the first quoted word.
        const string head = "laya.shortlist: question '";
        Assert.StartsWith(head, message, StringComparison.Ordinal);
        var end = message.IndexOf('\'', head.Length);
        return message[..end] + suffix + message[end..];
    }

    // ── properties of the port that the probe does not name ───────────────────

    private static QuestionSet Banking() => new()
    {
        ["intent"] = Question.Choice("Which?", Enumerable.Range(0, 40).Select(i => new KeyValuePair<string, object?>($"l{i:00}", $"d{i}"))),
        ["urgency"] = Question.Score("How?", "lo", "hi"),
    };

    [Fact]
    public void TheCallersQuestionsAreNotMutated()
    {
        var questions = Banking();
        var before = Describe(questions);
        var scripted = new Scripted("normal");
        LayaShortlist.PredictTournament((s, q) => scripted.Predict(s, q)!, "state", questions, 4);
        Assert.Equal(before, Describe(questions));
        Assert.Equal(2, questions.Count);
    }

    [Fact]
    public void ThePredictorOverloadIsTheSameAlgorithm()
    {
        var viaFunc = new Scripted("normal");
        var viaInterface = new Scripted("normal");
        var a = LayaShortlist.PredictTournament((s, q) => viaFunc.Predict(s, q)!, "state", Banking(), 8);
        var b = LayaShortlist.PredictTournament(new InterfaceAdapter(viaInterface), "state", Banking(), 8);
        Assert.Equal(viaFunc.Calls.Select(Describe), viaInterface.Calls.Select(Describe));
        Assert.Equal(a.Tournament!["intent"].Labels, b.Tournament!["intent"].Labels);
        Assert.Equal(16, LayaShortlist.DefaultTournamentGroup);
    }

    private sealed class InterfaceAdapter(Scripted inner) : ILayaPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) => inner.Predict(state, questions)!;
    }

    [Fact]
    public void OnlyChoiceQuestionsGetTournamentMetadata()
    {
        var scripted = new Scripted("normal");
        var result = LayaShortlist.PredictTournament((s, q) => scripted.Predict(s, q)!, "state", Banking(), 8);
        Assert.Equal(["intent"], result.Tournament!.Keys);
        Assert.Null(result.Shortlist);
        Assert.Null(result.Routing);
        // The shortlist path stays untouched.
        var shortlisted = LayaShortlist.Predict(new InterfaceAdapter(new Scripted("normal")), "state", Banking(),
            texts => [.. texts.Select((_, i) => new[] { 1.0 - i * 0.01, i * 0.01 })], 5);
        Assert.Null(shortlisted.Tournament);
    }

    [Fact]
    public void AnOptionOrderWarnsOncePerProcessForIdenticalText()
    {
        var id = "intent-" + Guid.NewGuid().ToString("N");
        QuestionSet Make() => new()
        {
            [id] = Question.Choice("Which?", Enumerable.Range(0, 20).Select(i => new KeyValuePair<string, object?>($"l{i}", null)))
                .WithOptionOrder(Enumerable.Range(0, 20).Reverse()),
        };
        var capture = new Capture();
        Trace.Listeners.Add(capture);
        try
        {
            for (var run = 0; run < 2; run++)
            {
                var scripted = new Scripted("normal");
                LayaShortlist.PredictTournament((s, q) => scripted.Predict(s, q)!, "state", Make(), 4);
            }
        }
        finally { Trace.Listeners.Remove(capture); }

        // Two groups' worth of rounds and two calls, one warning: Python's `warned` set and its default filter.
        Assert.Equal(
            [$"laya.shortlist: question '{id}': 'option_order' (20 slots) was dropped because the label "
             + "set was narrowed to 4; the order no longer fits the kept options and remapping "
             + "it onto them would move options between slots."],
            capture.Messages.Where(m => m.Contains(id, StringComparison.Ordinal)));
    }

    [Theory]
    [InlineData(1, "group_size must be an integer of at least 2, got 1")]
    [InlineData(0, "group_size must be an integer of at least 2, got 0")]
    [InlineData(-7, "group_size must be an integer of at least 2, got -7")]
    public void AGroupSizeBelowTwoIsRefusedBeforeAnyCall(int groupSize, string message)
    {
        var scripted = new Scripted("normal");
        var e = Assert.Throws<ArgumentException>(
            () => LayaShortlist.PredictTournament((s, q) => scripted.Predict(s, q)!, "state", Banking(), groupSize));
        Assert.Equal(message, e.Message);
        Assert.Empty(scripted.Calls);
    }

    [Fact]
    public void ANullPredictorOrQuestionsIsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => LayaShortlist.PredictTournament((ILayaPredictor)null!, "s", Banking()));
        Assert.Throws<ArgumentNullException>(
            () => LayaShortlist.PredictTournament((Func<object?, QuestionSet, LayaResult>)null!, "s", Banking()));
        Assert.Throws<ArgumentNullException>(
            () => LayaShortlist.PredictTournament((s, q) => throw new InvalidOperationException(), "s", null!));
    }

    [Fact]
    public void AGroupSizeOfIntMaxIsOneCallAndTheOverflowDoesNotWrap()
    {
        var scripted = new Scripted("normal");
        var result = LayaShortlist.PredictTournament((s, q) => scripted.Predict(s, q)!, "state", Banking(), int.MaxValue);
        Assert.Single(scripted.Calls);
        Assert.Equal(0, result.Tournament!["intent"].Rounds);
    }
}
