using System.Diagnostics;

namespace Laya.Tests;

/// <summary>
/// Tier 1: the question-level <c>option_order</c> (agent.py:1089-1100, common.py:237-241 and :558-571,
/// shortlist.py:425-448), with no model behind it. The answers it produces are pinned by the
/// <c>option_order_*</c> goldens in the parity tests.
/// </summary>
public sealed class OptionOrderTests
{
    private readonly StubTokenizer _tok = new();

    // ── API ───────────────────────────────────────────────────────────────────

    [Fact]
    public void QuestionsCarryNoOrderUnlessGivenOne()
    {
        Assert.Null(Question.Choice("q", "a", "b").OptionOrder);
        Assert.Null(Question.Score("q", "x", "y").OptionOrder);
        Assert.Null(Question.Noul("q").OptionOrder);
    }

    [Fact]
    public void WithOptionOrderReturnsACopyOfTheSameKindAndKeepsTheContent()
    {
        var choice = Question.Choice("q", ("a", "first"), ("b", null), ("c", 3));
        var ordered = choice.WithOptionOrder(2, 0, 1);
        Assert.Equal([2, 0, 1], ordered.OptionOrder);
        Assert.Null(choice.OptionOrder);
        Assert.Equal(choice.Options, ordered.Options);
        Assert.Equal(choice.Instructions, ordered.Instructions);

        var score = Question.Score("q", "lo", "mid", "hi").WithOptionOrder([1, 2, 0]);
        Assert.Equal([1, 2, 0], score.OptionOrder);
        Assert.Equal(["lo", "mid", "hi"], score.Levels);

        var noul = Question.Noul("q", "no", "yes").WithOptionOrder(1, 0);
        Assert.Equal([1, 0], noul.OptionOrder);
        Assert.Equal(("no", "yes"), ((string)noul.IfFalse!, (string)noul.IfTrue!));

        Assert.Null(ordered.WithOptionOrder((IEnumerable<int>?)null).OptionOrder);
    }

    [Fact]
    public void TheOrderIsCopiedSoLaterChangesToTheCallersArrayDoNotReachTheQuestion()
    {
        int[] order = [1, 0];
        var noul = Question.Noul("q").WithOptionOrder(order);
        order[0] = 0;
        Assert.Equal([1, 0], noul.OptionOrder);
    }

    // ── validation: Python's text, question id included ───────────────────────

    private static string Message(string id, Question question) =>
        Assert.Throws<ArgumentException>(() => question.CheckOptionOrder(id)).Message;

    [Theory]
    [InlineData(new[] { 0, 1 }, "[0, 1]")]
    [InlineData(new[] { 0, 0, 1 }, "[0, 0, 1]")]
    [InlineData(new[] { 0, 1, 3 }, "[0, 1, 3]")]
    [InlineData(new[] { -1, 0, 1 }, "[-1, 0, 1]")]
    [InlineData(new int[0], "[]")]
    [InlineData(new[] { 0, 1, 2, 3 }, "[0, 1, 2, 3]")]
    public void ChoiceRejectsAnythingButAPermutation(int[] order, string repr)
    {
        var q = Question.Choice("q", "a", "b", "c").WithOptionOrder(order);
        Assert.Equal(
            "question 'dept': 'option_order' must be a permutation of range(3) -- one slot per "
            + $"option, each option once -- got {repr}",
            Message("dept", q));
    }

    [Fact]
    public void ScoreCountsItsLevelsAndNoulIsAlwaysTwo()
    {
        Assert.Equal(
            "question 'urgency': 'option_order' must be a permutation of range(4) -- one slot per "
            + "option, each option once -- got [2, 0, 1]",
            Message("urgency", Question.Score("q", "a", "b", "c", "d").WithOptionOrder(2, 0, 1)));
        Assert.Equal(
            "question 'churn': 'option_order' must be a permutation of range(2) -- one slot per "
            + "option, each option once -- got [0, 1, 2]",
            Message("churn", Question.Noul("q", "no", "yes").WithOptionOrder(0, 1, 2)));
    }

    [Fact]
    public void TheIdIsQuotedAsPythonsReprQuotesIt()
    {
        Assert.Equal(
            "question \"it's\": 'option_order' must be a permutation of range(2) -- one slot per "
            + "option, each option once -- got [1, 1]",
            Message("it's", Question.Noul("q").WithOptionOrder(1, 1)));
    }

    [Fact]
    public void EveryPermutationAndNoOrderPass()
    {
        Question.Noul("q").CheckOptionOrder("a");
        Question.Choice("q", "a", "b", "c").WithOptionOrder(0, 1, 2).CheckOptionOrder("a");
        foreach (var order in new[] { new[] { 2, 0, 1 }, [1, 2, 0], [2, 1, 0], [0, 2, 1] })
            Question.Choice("q", "a", "b", "c").WithOptionOrder(order).CheckOptionOrder("a");
        Question.Choice("q", "only").WithOptionOrder(0).CheckOptionOrder("a");
    }

    // ── decode: unpermute_probs ───────────────────────────────────────────────

    [Fact]
    public void UnpermuteScattersTheSlotRowBackToOptionOrder()
    {
        // Slot s shows option order[s], so canonical[order[s]] = p[s]: numpy gives [.5, .3, .2].
        Assert.Equal([0.5, 0.3, 0.2], Question.Unpermute([0.2, 0.5, 0.3], [2, 0, 1]));
        Assert.Equal([0.1, 0.9], Question.Unpermute([0.9, 0.1], [1, 0]));
        Assert.Equal([0.2, 0.5, 0.3], Question.Unpermute([0.2, 0.5, 0.3], [0, 1, 2]));
    }

    [Fact]
    public void UnpermuteLeavesTheRowAloneWithoutAnOrderOrOnALengthMismatch()
    {
        double[] p = [0.2, 0.5, 0.3];
        Assert.Same(p, Question.Unpermute(p, null));
        Assert.Same(p, Question.Unpermute(p, [1, 0]));
    }

    // ── sequence: slot order (common.py:237-241) ──────────────────────────────

    [Fact]
    public void ChoiceFragmentsAndMarkersFollowTheSlotsNotTheOptions()
    {
        var plain = Question.Choice("h", "c", "a", "b");
        var ordered = Question.Choice("h", "a", "b", "c").WithOptionOrder(2, 0, 1);

        var want = SequenceBuilder.Build(_tok, "state", plain, 64, 32);
        var got = SequenceBuilder.Build(_tok, "state", ordered, 64, 32);

        Assert.Equal(want.Ids, got.Ids);
        Assert.Equal(want.Markers, got.Markers);
        Assert.Equal(["a", "b", "c"], SequenceBuilder.RenderOptions(ordered));
    }

    [Fact]
    public void ScoreAndNoulFragmentsAreEncodedInSlotOrder()
    {
        var tok = new StubTokenizer();
        SequenceBuilder.Build(tok, "s", Question.Score("h", "x", "y", "z").WithOptionOrder(2, 0, 1), 64, 32);
        Assert.Equal([" level 2: z", " level 0: x", " level 1: y"], tok.Encoded.Skip(1).Take(3));

        var noul = new StubTokenizer();
        SequenceBuilder.Build(noul, "s", Question.Noul("h").WithOptionOrder(1, 0), 64, 32);
        Assert.Equal([" true: yes, the statement holds", " false: no, the statement does not hold"],
                     noul.Encoded.Skip(1).Take(2));
    }

    [Fact]
    public void TheIdentityOrderBuildsTheCanonicalSequence()
    {
        var plain = Question.Score("h", "x", "y", "z");
        var want = SequenceBuilder.Build(_tok, "s", plain, 64, 32);
        var got = SequenceBuilder.Build(_tok, "s", plain.WithOptionOrder(0, 1, 2), 64, 32);
        Assert.Equal(want.Ids, got.Ids);
        Assert.Equal(want.Markers, got.Markers);
    }

    [Fact]
    public void TheStateRoomIsMeasuredWithoutTheOrderLikePythonsStateRoom()
    {
        var plain = Question.Choice("h", ("a", "one two"), ("b", "three"), ("c", null));
        var ordered = plain.WithOptionOrder(2, 0, 1);
        Assert.Equal(SequenceBuilder.StateRoom(_tok, plain, 64, 32), SequenceBuilder.StateRoom(_tok, ordered, 64, 32));
    }

    // ── router: the order is part of the schema (router.py:221-228) ───────────

    [Fact]
    public void SchemaWritesTheOrderLastAsJsonDumpsWould()
    {
        var questions = new QuestionSet
        {
            ["q"] = Question.Choice("x", "a", "b").WithOptionOrder(1, 0),
            ["n"] = Question.Noul("y", ifTrue: "yes").WithOptionOrder(1, 0),
            ["m"] = Question.Noul("z").WithOptionOrder(0, 1),
            ["s"] = Question.Score("w", "lo", "hi"),
        };
        Assert.Equal(
            "{\"q\": {\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": {\"a\": null, \"b\": null}, "
            + "\"option_order\": [1, 0]}, "
            + "\"n\": {\"type\": \"noul\", \"instructions\": \"y\", \"criteria\": {\"true\": \"yes\"}, \"option_order\": [1, 0]}, "
            + "\"m\": {\"type\": \"noul\", \"instructions\": \"z\", \"option_order\": [0, 1]}, "
            + "\"s\": {\"type\": \"score\", \"instructions\": \"w\", \"criteria\": [\"lo\", \"hi\"]}}",
            LayaRouter.QuestionSchema(questions));
    }

    // ── shortlist: a narrowed choice drops its order with a warning (shortlist.py:425-448) ──

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

    private sealed class Recorder : ILayaPredictor
    {
        public QuestionSet? Seen { get; private set; }

        public LayaResult Predict(object? state, QuestionSet questions)
        {
            Seen = questions;
            return new LayaResult("fake", [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
        }
    }

    private static double[][] Vectors(IReadOnlyList<string> texts) =>
        [.. texts.Select((_, i) => i == 0 ? new[] { 1.0, 0.0 } : new[] { 1.0 - i * 0.1, i * 0.1 })];

    // A fresh id per test: WarnOnce is process-wide and other classes reset it, so these tests never do.
    private static QuestionSet Questions(string intent) => new()
    {
        [intent] = Question.Choice("Which?", "a", "b", "c", "d").WithOptionOrder(3, 2, 1, 0),
        ["small"] = Question.Choice("Which?", "x", "y").WithOptionOrder(1, 0),
        ["urgency"] = Question.Score("How?", "lo", "hi").WithOptionOrder(1, 0),
        ["churn"] = Question.Noul("Cancel?").WithOptionOrder(1, 0),
    };

    [Fact]
    public void ANarrowedChoiceLosesItsOrderAndWarnsOnceWithPythonsText()
    {
        var intent = "intent-" + Guid.NewGuid().ToString("N");
        var recorder = new Recorder();
        var questions = Questions(intent);
        var capture = new Capture();
        Trace.Listeners.Add(capture);
        try
        {
            LayaShortlist.Predict(recorder, "state", questions, Vectors, k: 2);
            LayaShortlist.Predict(recorder, "state", questions, Vectors, k: 2);
        }
        finally { Trace.Listeners.Remove(capture); }

        var seen = recorder.Seen!;
        Assert.Null(seen[intent].OptionOrder);
        Assert.Equal(2, ((ChoiceQuestion)seen[intent]).Labels.Count);
        // Not narrowed (2 labels <= k) and not a choice: forwarded untouched, order and all.
        Assert.Same(questions["small"], seen["small"]);
        Assert.Equal([1, 0], seen["urgency"].OptionOrder);
        Assert.Equal([1, 0], seen["churn"].OptionOrder);
        // The caller's question still has its order.
        Assert.Equal([3, 2, 1, 0], questions[intent].OptionOrder);

        Assert.Equal(
            [$"laya.shortlist: question '{intent}': 'option_order' (4 slots) was dropped because the label "
             + "set was narrowed to 2; the order no longer fits the kept options and remapping "
             + "it onto them would move options between slots."],
            capture.Messages.Where(m => m.StartsWith("laya.shortlist", StringComparison.Ordinal) && m.Contains(intent)));
    }

    [Fact]
    public void APassthroughChoiceKeepsItsOrderWithoutAWarning()
    {
        var intent = "intent-" + Guid.NewGuid().ToString("N");
        var recorder = new Recorder();
        var capture = new Capture();
        Trace.Listeners.Add(capture);
        try { LayaShortlist.Predict(recorder, "state", Questions(intent), Vectors, k: 4); }
        finally { Trace.Listeners.Remove(capture); }

        Assert.Equal([3, 2, 1, 0], recorder.Seen![intent].OptionOrder);
        Assert.DoesNotContain(capture.Messages, m => m.StartsWith("laya.shortlist", StringComparison.Ordinal));
    }
}
