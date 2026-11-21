namespace Laya.Tests;

/// <summary>
/// Tier 1: custom noul <c>labels</c> (common.py:114-125 and :147-153, agent.py:1101-1106), with no model
/// behind it. The answers they produce are pinned by the <c>noul_labels_*</c> goldens in the parity tests.
/// </summary>
public sealed class NoulLabelsTests
{
    private const string Invalid = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";

    // ── API ───────────────────────────────────────────────────────────────────

    [Fact]
    public void ANoulHasNoLabelsUnlessGivenSome()
    {
        Assert.Null(Question.Noul("q").Labels);
        Assert.Null(Question.Noul("q", "no", "yes").Labels);
    }

    [Fact]
    public void WithLabelsReturnsACopyAndKeepsTheRestIncludingTheOrder()
    {
        var plain = Question.Noul("q", "no", "yes").WithOptionOrder(1, 0);
        var labelled = plain.WithLabels(" B ", "A");

        Assert.Equal(new NoulLabels(" B ", "A"), labelled.Labels);
        Assert.Null(plain.Labels);
        Assert.Equal([1, 0], labelled.OptionOrder);
        Assert.Equal(("no", "yes"), ((string)labelled.IfFalse!, (string)labelled.IfTrue!));
        // And the other way round: an order set afterwards keeps the labels.
        Assert.Equal(new NoulLabels(" B ", "A"), labelled.WithOptionOrder(0, 1).Labels);
        Assert.Null(labelled.WithoutLabels().Labels);
        Assert.Equal([1, 0], labelled.WithoutLabels().OptionOrder);
    }

    // ── validation: Python's text, question id included, raised when answered ─

    private static string Message(string id, Question question) =>
        Assert.Throws<ArgumentException>(() => question.CheckLabels(id)).Message;

    [Theory]
    [InlineData("", "A")]
    [InlineData("B", "")]
    [InlineData("  ", "A")]
    [InlineData("B", " \t\n ")]
    [InlineData("same", "same")]
    [InlineData("yes", "  yes ")]
    [InlineData(" a ", "a")]
    [InlineData(null, "A")]
    [InlineData("B", null)]
    [InlineData(null, null)]
    public void BlankEqualAndNonStringLabelsAreRejectedWithPythonsText(string? f, string? t)
    {
        var q = Question.Noul("q").WithLabels(f, t);
        Assert.Equal($"question 'risk': {Invalid}", Message("risk", q));
    }

    [Fact]
    public void BuildingTheQuestionDoesNotValidateItPythonDoesSoWhenItIsAnswered()
    {
        var q = Question.Noul("q").WithLabels("x", "x");
        Assert.Equal(new NoulLabels("x", "x"), q.Labels);
    }

    [Fact]
    public void TheIdIsQuotedAsPythonsReprQuotesIt()
    {
        Assert.Equal($"question \"it's\": {Invalid}", Message("it's", Question.Noul("q").WithLabels("a", "a")));
    }

    [Fact]
    public void ValidLabelsAndNoLabelsPassAndOtherQuestionsCarryNone()
    {
        Question.Noul("q").CheckLabels("a");
        Question.Noul("q").WithLabels("B", "A").CheckLabels("a");
        Question.Noul("q").WithLabels(" 1 ", "0 x").CheckLabels("a");
        Question.Choice("q", "a", "b").CheckLabels("a");
        Question.Score("q", "lo", "hi").CheckLabels("a");
    }

    [Fact]
    public void QuestionsAreCheckedInDeclarationOrderAndOptionOrderComesFirstWithinOne()
    {
        var firstBad = new QuestionSet
        {
            ["ok"] = Question.Choice("q", "a", "b"),
            ["bad_labels"] = Question.Noul("q").WithLabels("x", "x"),
            ["bad_order"] = Question.Noul("q").WithOptionOrder(0, 0),
        };
        Assert.Equal($"question 'bad_labels': {Invalid}",
                     Assert.Throws<ArgumentException>(() => LayaEngine.CheckQuestions(firstBad)).Message);

        var both = new QuestionSet { ["n"] = Question.Noul("q").WithLabels("", "").WithOptionOrder(0, 0) };
        Assert.StartsWith("question 'n': 'option_order' must be a permutation",
                          Assert.Throws<ArgumentException>(() => LayaEngine.CheckQuestions(both)).Message);

        LayaEngine.CheckQuestions(new QuestionSet { ["n"] = Question.Noul("q").WithLabels("B", "A").WithOptionOrder(1, 0) });
    }

    // ── rendering (common.py:147-153) ─────────────────────────────────────────

    [Fact]
    public void OptionsRenderTheStrippedLabelBeforeTheCriterionOrTheDefaultText()
    {
        Assert.Equal(["B: no, the statement does not hold", "A: yes, the statement holds"],
                     SequenceBuilder.RenderOptions(Question.Noul("q").WithLabels("B", "A")));
        Assert.Equal(["low: a template", "high: a person"],
                     SequenceBuilder.RenderOptions(Question.Noul("q", "a template", "a person").WithLabels("  low ", "\thigh\n")));
        Assert.Equal(["low: no, the statement does not hold", "high: a person"],
                     SequenceBuilder.RenderOptions(Question.Noul("q", null, "a person").WithLabels("low", "high")));
        // Spelled like the defaults, they are the defaults.
        Assert.Equal(SequenceBuilder.RenderOptions(Question.Noul("q")),
                     SequenceBuilder.RenderOptions(Question.Noul("q").WithLabels("false", "true")));
    }

    [Fact]
    public void RenderingInvalidLabelsRaisesPythonsBareText()
    {
        var e = Assert.Throws<ArgumentException>(() => SequenceBuilder.RenderOptions(Question.Noul("q").WithLabels("a", "a")));
        Assert.Equal(Invalid, e.Message);
    }

    [Fact]
    public void TheEncodedFragmentsCarryTheLabelsInSlotOrder()
    {
        var tok = new StubTokenizer();
        SequenceBuilder.Build(tok, "s", Question.Noul("h").WithLabels("B", "A").WithOptionOrder(1, 0), 64, 32);
        Assert.Equal([" A: yes, the statement holds", " B: no, the statement does not hold"],
                     tok.Encoded.Skip(1).Take(2));
    }

    [Fact]
    public void LabelsSpelledLikeTheDefaultsBuildTheDefaultSequence()
    {
        var tok = new StubTokenizer();
        var plain = SequenceBuilder.Build(tok, "state", Question.Noul("h"), 64, 32);
        var labelled = SequenceBuilder.Build(tok, "state", Question.Noul("h").WithLabels("false", "true"), 64, 32);
        Assert.Equal(plain.Ids, labelled.Ids);
        Assert.Equal(plain.Markers, labelled.Markers);
    }

    [Fact]
    public void TheStateRoomIsMeasuredFromTheLabelledHead()
    {
        var tok = new StubTokenizer();
        var plain = Question.Noul("h");
        var longLabels = plain.WithLabels("a very long false label indeed", "a very long true label indeed");
        Assert.True(SequenceBuilder.StateRoom(tok, longLabels, 64, 32) <= SequenceBuilder.StateRoom(tok, plain, 64, 32));
    }

    // ── router: the labels are part of the schema (router.py:221-228) ─────────

    [Fact]
    public void SchemaWritesTheUnstrippedLabelsAfterTheCriteriaAndBeforeTheOrder()
    {
        var questions = new QuestionSet
        {
            ["a"] = Question.Noul("x").WithLabels(" B", "A"),
            ["b"] = Question.Noul("y", ifTrue: "yes").WithLabels("n", "y").WithOptionOrder(1, 0),
            ["c"] = Question.Noul("z"),
        };
        Assert.Equal(
            "{\"a\": {\"type\": \"noul\", \"instructions\": \"x\", \"labels\": {\"false\": \" B\", \"true\": \"A\"}}, "
            + "\"b\": {\"type\": \"noul\", \"instructions\": \"y\", \"criteria\": {\"true\": \"yes\"}, "
            + "\"labels\": {\"false\": \"n\", \"true\": \"y\"}, \"option_order\": [1, 0]}, "
            + "\"c\": {\"type\": \"noul\", \"instructions\": \"z\"}}",
            LayaRouter.QuestionSchema(questions));
    }
}
