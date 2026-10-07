using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Tier 1: the embedding shortlist, which needs no artifacts. A mock <see cref="ILayaPredictor"/>
/// stands in for the decision model, so these tests never construct one. Named cases mirror
/// <c>tests/test_shortlist.py</c> (excluding the torch mean-pool block, which ports Python's
/// <c>embed_fn_from_agent</c> — out of scope per the porting plan); <see cref="EveryShortlistProbeEntry"/>
/// consumes every entry of <c>golden/routing/shortlist_probe.json</c>.
/// </summary>
public sealed class LayaShortlistTests
{
    // Vectors: query [1, 0]. alpha and delta tie at cosine 1; gamma is 0.6; beta is 0.
    // Stable order must keep alpha ahead of delta.
    private static readonly Dictionary<string, object?> Criteria = new(StringComparer.Ordinal)
    {
        ["alpha"] = null, ["beta"] = "", ["gamma"] = "mid", ["delta"] = "same",
    };

    private static readonly Dictionary<string, double[]> OptionVectors = new(StringComparer.Ordinal)
    {
        ["alpha"] = [1.0, 0.0],
        ["beta"] = [0.0, 1.0],
        ["gamma: mid"] = [0.6, 0.8],
        ["delta: same"] = [1.0, 0.0],
    };

    private sealed class TableEmbed(IReadOnlyDictionary<string, double[]> vectors)
    {
        public List<List<string>> Calls { get; } = [];

        public double[][] Embed(IReadOnlyList<string> texts)
        {
            Calls.Add([.. texts]);
            var missing = texts.Where(t => !vectors.ContainsKey(t)).ToList();
            if (missing.Count > 0) throw new InvalidOperationException($"unexpected texts: {string.Join(", ", missing)}");
            return [.. texts.Select(t => vectors[t])];
        }
    }

    private static LayaEmbedFunction ForQuery(string queryText, IReadOnlyDictionary<string, double[]> optionVectors)
    {
        var full = new Dictionary<string, double[]>(optionVectors, StringComparer.Ordinal) { [queryText] = [1.0, 0.0] };
        return new TableEmbed(full).Embed;
    }

    // ── deterministic top-k ──────────────────────────────────────────────────

    [Fact]
    public void TopKKeepsTheCosineTieInInputOrder()
    {
        var embed = new TableEmbed(new Dictionary<string, double[]>(OptionVectors, StringComparer.Ordinal) { ["pay me"] = [1.0, 0.0] });
        var labels = LayaShortlist.ShortlistChoice("pay me", Criteria, embed.Embed, k: 2);
        Assert.Equal(["alpha", "delta"], labels);
        Assert.Single(embed.Calls);
        Assert.Equal("pay me", embed.Calls[0][0]);
        Assert.Equal(["alpha", "beta", "gamma: mid", "delta: same"], embed.Calls[0].Skip(1));
    }

    [Fact]
    public void K1IsTheEarliestMax()
    {
        var embed = ForQuery("pay me", OptionVectors);
        Assert.Equal(["alpha"], LayaShortlist.ShortlistChoice("pay me", Criteria, embed, k: 1));
    }

    [Fact]
    public void K3AppendsTheNextCosine()
    {
        var embed = ForQuery("pay me", OptionVectors);
        Assert.Equal(["alpha", "delta", "gamma"], LayaShortlist.ShortlistChoice("pay me", Criteria, embed, k: 3));
    }

    [Fact]
    public void ZeroQueryKeepsOriginalOrder()
    {
        // Every cosine is 0 when the query vector is 0, so the earliest labels win.
        var vectors = new Dictionary<string, double[]>(StringComparer.Ordinal)
        {
            ["pay me"] = [0.0, 0.0],
            ["alpha"] = [1.0, 0.0],
            ["beta"] = [0.0, 1.0],
            ["gamma: mid"] = [0.6, 0.8],
            ["delta: same"] = [3.0, 4.0],
        };
        var embed = new TableEmbed(vectors);
        Assert.Equal(["alpha", "beta"], LayaShortlist.ShortlistChoice("pay me", Criteria, embed.Embed, k: 2));
    }

    [Fact]
    public void NanVectorSortsBehindAFiniteMatch()
    {
        // A non-finite option vector is treated as 0 and loses to a real match.
        var vectors = new Dictionary<string, double[]>(StringComparer.Ordinal)
        {
            ["pay me"] = [1.0, 0.0],
            ["alpha"] = [double.NaN, double.NaN],
            ["beta"] = [1.0, 0.0],
        };
        var embed = new TableEmbed(vectors);
        var criteria = new Dictionary<string, object?> { ["alpha"] = null, ["beta"] = null };
        Assert.Equal(["beta"], LayaShortlist.ShortlistChoice("pay me", criteria, embed.Embed, k: 1));
    }

    // ── list criteria and instructions ───────────────────────────────────────

    [Fact]
    public void InstructionsChangeTheQueryAndTheWinner()
    {
        var vectors = new Dictionary<string, double[]>(StringComparer.Ordinal)
        {
            ["Classify\npay me"] = [0.0, 1.0],
            ["alpha"] = [1.0, 0.0],
            ["beta"] = [0.0, 1.0],
            ["gamma"] = [0.0, 0.2],
        };
        var embed = new TableEmbed(vectors);
        var labels = LayaShortlist.ShortlistChoice("pay me", ["alpha", "beta", "gamma"], embed.Embed, k: 2, instructions: "Classify");
        Assert.Equal(["beta", "gamma"], labels);
        Assert.Equal("Classify\npay me", embed.Calls[0][0]);
        Assert.Equal(["alpha", "beta", "gamma"], embed.Calls[0].Skip(1));
    }

    [Fact]
    public void DictStateIsSerialized()
    {
        var state = new Dictionary<string, object?> { ["text"] = "hi" };
        var vectors = new Dictionary<string, double[]>(StringComparer.Ordinal)
        {
            ["Classify\n{\"text\": \"hi\"}"] = [1.0, 0.0],
            ["alpha"] = [1.0, 0.0],
            ["beta"] = [0.0, 1.0],
        };
        var embed = new TableEmbed(vectors);
        var labels = LayaShortlist.ShortlistChoice(state, ["alpha", "beta"], embed.Embed, k: 1, instructions: "Classify");
        Assert.Equal(["alpha"], labels);
        Assert.Equal("Classify\n{\"text\": \"hi\"}", embed.Calls[0][0]);
    }

    [Fact]
    public void ZeroAndFalseCriterionValuesStayInTheOptionText()
    {
        // 0 and False are real criterion values, so they are part of the embedded text.
        var rich = new Dictionary<string, object?>
        {
            ["zero"] = 0L, ["no"] = false, ["bare"] = null,
            ["named"] = new Dictionary<string, object?> { ["desc"] = "payments" },
        };
        var richRendered = SequenceBuilder.RenderOptions(Question.Choice("", rich));
        var vectors = new Dictionary<string, double[]>(StringComparer.Ordinal) { ["pay me"] = [1.0, 0.0] };
        foreach (var t in richRendered) vectors[t] = [1.0, 0.0];
        var embed = new TableEmbed(vectors);
        LayaShortlist.ShortlistChoice("pay me", rich, embed.Embed, k: 1);
        Assert.Equal(richRendered, embed.Calls[0].Skip(1));
    }

    // ── k >= n pass-through ───────────────────────────────────────────────────

    [Fact]
    public void KEqualsNReturnsEveryLabelInOrderWithoutCallingEmbedFn()
    {
        LayaEmbedFunction boom = _ => throw new InvalidOperationException("embed_fn should not run when k >= n");
        Assert.Equal([.. Criteria.Keys], LayaShortlist.ShortlistChoice("pay me", Criteria, boom, k: 4));
    }

    [Fact]
    public void KGreaterThanNReturnsEveryLabelInOrderWithoutCallingEmbedFn()
    {
        LayaEmbedFunction boom = _ => throw new InvalidOperationException("embed_fn should not run when k >= n");
        Assert.Equal([.. Criteria.Keys], LayaShortlist.ShortlistChoice("pay me", Criteria, boom, k: 20));
    }

    // ── Predict: mock predictor sees only k criteria ─────────────────────────

    private sealed class Recorder : ILayaPredictor
    {
        public List<(object? State, QuestionSet Questions)> Calls { get; } = [];

        public LayaResult Predict(object? state, QuestionSet questions)
        {
            Calls.Add((state, questions));
            var order = new List<string>();
            var byId = new Dictionary<string, Answer>(StringComparer.Ordinal);
            foreach (var (qid, q) in questions)
            {
                order.Add(qid);
                if (q is ChoiceQuestion choice)
                {
                    var labels = choice.Labels;
                    var probs = labels.Select((l, i) => new KeyValuePair<string, double>(l, i == 0 ? 1.0 : 0.0)).ToList();
                    byId[qid] = new ChoiceAnswer(labels[0], probs, 1.0, 1.0, new ActionInfo(1.0));
                }
            }
            return new LayaResult("fake", order, byId, new Usage(0, 0));
        }
    }

    [Fact]
    public void PredictShortlistsChoiceCriteriaToTheTopK()
    {
        var sentinel = new Dictionary<string, object?> { ["desc"] = "payments" };
        var full = new Dictionary<string, object?> { ["billing"] = sentinel, ["tech"] = "bugs", ["sales"] = null, ["other"] = "misc" };
        var vectors = new Dictionary<string, double[]>(StringComparer.Ordinal)
        {
            ["Which desk?\nI was charged twice"] = [1.0, 0.0],
            ["billing: {\"desc\": \"payments\"}"] = [0.0, 1.0],
            ["tech: bugs"] = [1.0, 0.0],
            ["sales"] = [0.2, 0.2],
            ["other: misc"] = [0.0, 1.0],
        };
        var embed = new TableEmbed(vectors);
        var scoreQ = Question.Score("How urgent?", "low", "mid", "high", "now");
        var noulQ = Question.Noul("Is a refund requested?");
        var questions = new QuestionSet
        {
            ["intent"] = Question.Choice("Which desk?", full),
            ["urgency"] = scoreQ,
            ["refund"] = noulQ,
        };
        var recorder = new Recorder();

        var result = LayaShortlist.Predict(recorder, "I was charged twice", questions, embed.Embed, k: 2);

        Assert.Single(recorder.Calls);
        var gotQuestions = recorder.Calls[0].Questions;
        Assert.Same("I was charged twice", recorder.Calls[0].State);
        var reducedIntent = (ChoiceQuestion)gotQuestions["intent"];
        Assert.Equal(["tech", "sales"], reducedIntent.Labels);
        Assert.Equal("bugs", reducedIntent.Options.First(o => o.Key == "tech").Value);
        Assert.Same(scoreQ, gotQuestions["urgency"]);
        Assert.Same(noulQ, gotQuestions["refund"]);

        // The caller's original question and criteria are untouched.
        Assert.Equal(4, ((ChoiceQuestion)questions["intent"]).Options.Count);
        Assert.Equal(sentinel, full["billing"]);

        Assert.Equal("tech", result["intent"].AsChoice().Choice);
        Assert.Equal(["tech", "sales"], result.Shortlist!["intent"].Labels);
        Assert.True(result.Shortlist["intent"].Scores![0] > result.Shortlist["intent"].Scores![1]);
        Assert.True(result.Shortlist["intent"].Scores![1] > 0);
        Assert.Equal(2, result.Shortlist["intent"].K);
        Assert.Equal(4, result.Shortlist["intent"].N);
        Assert.False(result.Shortlist["intent"].Passthrough);
        Assert.False(result.Shortlist.ContainsKey("urgency"));
    }

    [Fact]
    public void PredictReturnsACopyRatherThanMutatingThePredictorsResult()
    {
        var holding = new HoldingPredictor();
        var criteria = new Dictionary<string, object?>(Criteria);
        var out1 = LayaShortlist.Predict(holding, "pay me", new QuestionSet { ["intent"] = Question.Choice("", criteria) },
            _ => throw new InvalidOperationException(), k: 4);
        Assert.NotNull(out1.Shortlist);
        Assert.Null(holding.Held?.Shortlist);
    }

    private sealed class HoldingPredictor : ILayaPredictor
    {
        public LayaResult? Held;
        public LayaResult Predict(object? state, QuestionSet questions) =>
            Held = new LayaResult("fake", [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
    }

    [Fact]
    public void KGreaterThanOrEqualNReachesPredictUnchanged()
    {
        var full = new Dictionary<string, object?> { ["billing"] = null, ["tech"] = "bugs", ["sales"] = null, ["other"] = "misc" };
        var originalQuestion = Question.Choice("Which desk?", full);
        var recorder = new Recorder();
        var out1 = LayaShortlist.Predict(recorder, "I was charged twice",
            new QuestionSet { ["intent"] = originalQuestion },
            _ => throw new InvalidOperationException("embed_fn should not run"), k: 4);

        Assert.Same(originalQuestion, recorder.Calls[0].Questions["intent"]);
        Assert.Equal([.. full.Keys], out1.Shortlist!["intent"].Labels);
        Assert.Null(out1.Shortlist["intent"].Scores);
        Assert.True(out1.Shortlist["intent"].Passthrough);
    }

    [Fact]
    public void ListCriteriaStayInRankOrderAsAList()
    {
        var vectors = new Dictionary<string, double[]>(StringComparer.Ordinal)
        {
            ["Which?\nhello"] = [1.0, 0.0],
            ["alpha"] = [0.0, 1.0],
            ["beta"] = [1.0, 0.0],
            ["gamma"] = [0.0, 0.0],
        };
        var embed = new TableEmbed(vectors);
        var recorder = new Recorder();
        var questions = new QuestionSet { ["intent"] = Question.Choice("Which?", "alpha", "beta", "gamma") };
        LayaShortlist.Predict(recorder, "hello", questions, embed.Embed, k: 2);

        var received = (ChoiceQuestion)recorder.Calls[0].Questions["intent"];
        Assert.Equal(["beta", "alpha"], received.Labels);
        // The caller's own question is untouched.
        Assert.Equal(["alpha", "beta", "gamma"], ((ChoiceQuestion)questions["intent"]).Labels);
    }

    // ── errors ────────────────────────────────────────────────────────────────

    [Fact]
    public void KZeroThrows() => Assert.Throws<ArgumentOutOfRangeException>(
        () => LayaShortlist.ShortlistChoice("pay me", Criteria, ForQuery("pay me", OptionVectors), k: 0));

    [Fact]
    public void KNegativeThrows() => Assert.Throws<ArgumentOutOfRangeException>(
        () => LayaShortlist.ShortlistChoice("pay me", Criteria, ForQuery("pay me", OptionVectors), k: -3));

    [Fact]
    public void EmptyDictCriteriaThrows() => Assert.Throws<ArgumentException>(
        () => LayaShortlist.ShortlistChoice("pay me", new Dictionary<string, object?>(), ForQuery("pay me", OptionVectors), k: 1));

    [Fact]
    public void EmptyListCriteriaThrows() => Assert.Throws<ArgumentException>(
        () => LayaShortlist.ShortlistChoice("pay me", Array.Empty<string>(), ForQuery("pay me", OptionVectors), k: 1));

    [Fact]
    public void DuplicateLabelThrows() => Assert.Throws<ArgumentException>(
        () => LayaShortlist.ShortlistChoice("pay me", ["alpha", "alpha"], ForQuery("pay me", OptionVectors), k: 1));

    [Fact]
    public void MissingCriteriaOnAChoiceQuestionThrows()
    {
        // A choice question with no ChoiceQuestion.Options cannot exist in the .NET model
        // (construction itself validates), so the analogous failure here is empty criteria.
        Assert.Throws<ArgumentException>(() => Question.Choice("x", new Dictionary<string, object?>()));
    }

    private static readonly double[][] BadShapeVectors = [[0.0, 0.0, 0.0, 0.0]];

    [Fact]
    public void BadEmbedShapeThrowsAndNeverCallsPredict()
    {
        var recorder = new Recorder();
        var full = new Dictionary<string, object?> { ["billing"] = null, ["tech"] = "bugs", ["sales"] = null, ["other"] = "misc" };
        var questions = new QuestionSet { ["intent"] = Question.Choice("Which desk?", full) };
        Assert.Throws<ArgumentException>(() =>
            LayaShortlist.Predict(recorder, "pay me", questions, _ => BadShapeVectors, k: 2));
        Assert.Empty(recorder.Calls);
    }

    // ── every recorded probe entry ────────────────────────────────────────────

    public static IEnumerable<object[]> ShortlistProbeCases()
    {
        using var doc = RoutingGoldenData.Load("shortlist_probe.json");
        return [.. doc.RootElement.EnumerateArray().Select(e => new object[] { e.GetProperty("label").GetString()! })];
    }

    private static JsonElement FindEntry(string label)
    {
        using var doc = RoutingGoldenData.Load("shortlist_probe.json");
        foreach (var e in doc.RootElement.EnumerateArray())
            if (e.GetProperty("label").GetString() == label)
                return JsonDocument.Parse(e.GetRawText()).RootElement;
        throw new InvalidOperationException($"no shortlist_probe.json entry labelled '{label}'");
    }

    [Theory]
    [MemberData(nameof(ShortlistProbeCases))]
    public void EveryShortlistProbeEntry(string label)
    {
        var e = FindEntry(label);
        var state = RoutingGoldenData.ToClr(e.GetProperty("state"));
        var k = e.GetProperty("k").GetInt32();
        var instructionsProp = e.GetProperty("instructions");
        var instructions = instructionsProp.ValueKind == JsonValueKind.Null ? null : RoutingGoldenData.ToClr(instructionsProp);

        var criteriaElement = e.GetProperty("criteria");
        var criteria = criteriaElement.ValueKind == JsonValueKind.Array
            ? criteriaElement.EnumerateArray().Select(x => new KeyValuePair<string, object?>(x.GetString()!, null)).ToList()
            : criteriaElement.EnumerateObject()
                .Select(p => new KeyValuePair<string, object?>(p.Name, RoutingGoldenData.ToClr(p.Value))).ToList();

        var passthrough = e.GetProperty("passthrough").GetBoolean();
        var wantLabels = e.GetProperty("labels").EnumerateArray().Select(x => x.GetString()!).ToArray();
        var n = e.GetProperty("n").GetInt32();

        var recorder = new Recorder();
        var question = Question.Choice("", criteria);
        var questions = new QuestionSet { ["q"] = question };

        LayaEmbedFunction embed = passthrough
            ? (_ => throw new InvalidOperationException($"{label}: embed_fn should not run when k >= n"))
            : texts => ReadVectorsMatrix(e.GetProperty("vectors"), texts.Count);

        var result = LayaShortlist.Predict(recorder, state, questions, embed, k);
        var info = result.Shortlist!["q"];

        Assert.Equal(wantLabels, info.Labels);
        Assert.Equal(k, info.K);
        Assert.Equal(n, info.N);
        Assert.Equal(passthrough, info.Passthrough);

        if (passthrough)
        {
            Assert.Null(info.Scores);
        }
        else
        {
            var wantScores = e.GetProperty("scores").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            Assert.NotNull(info.Scores);
            Assert.Equal(wantScores.Length, info.Scores!.Count);
            for (var i = 0; i < wantScores.Length; i++)
                Assert.Equal(wantScores[i], info.Scores[i], 1e-9);
        }
    }

    /// <summary>
    /// Parse a probe "vectors" matrix (first row is the query, remaining rows the options, in
    /// criteria order). Non-finite components are encoded as the JSON strings "nan"/"inf"/"-inf"
    /// per <c>meta.json</c>'s notes; <paramref name="expectedRows"/> is asserted against the
    /// embed_fn call's own text count, mirroring how <c>_rank</c> always calls embed_fn with
    /// exactly <c>1 + n</c> texts.
    /// </summary>
    private static double[][] ReadVectorsMatrix(JsonElement vectors, int expectedRows)
    {
        var rows = vectors.EnumerateArray().Select(row =>
            row.EnumerateArray().Select(ParseComponent).ToArray()).ToArray();
        Assert.Equal(expectedRows, rows.Length);
        return rows;
    }

    private static double ParseComponent(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.String => e.GetString() switch
        {
            "nan" => double.NaN,
            "inf" => double.PositiveInfinity,
            "-inf" => double.NegativeInfinity,
            var s => double.Parse(s!, System.Globalization.CultureInfo.InvariantCulture),
        },
        _ => e.GetDouble(),
    };
}
