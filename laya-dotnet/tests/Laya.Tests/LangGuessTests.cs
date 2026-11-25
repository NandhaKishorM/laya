using System.Text;
using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// The parts of the Python parity port the golden probes cannot carry: a <c>lang_guess</c> callback,
/// byte-string states, and <c>max_chars</c> beside <c>clean=False</c>. Expected values were produced by
/// running the Python reference (<c>Router._route</c>, <c>lang.state_text</c>, <c>lang.analyse</c>,
/// <c>email_state</c>) on the same inputs.
/// </summary>
public sealed class LangGuessTests
{
    private sealed class NoopPredictor : ILayaPredictor
    {
        public LayaResult Predict(object? state, QuestionSet questions) =>
            new("noop", [], new Dictionary<string, Answer>(StringComparer.Ordinal), new Usage(0, 0));
    }

    private static LayaRouter NewRouter(LangGuess? installed = null) =>
        new(new LayaRouterOptions { LangGuess = installed }, _ => new NoopPredictor());

    private const string NonEnglish = "lang_guess: the caller identified this as non-English text";
    private const string Undecided = "Latin script, language not identified and no non-English letters; using default (multilingual)";

    [Fact]
    public void CallbackAnswersWithACode()
    {
        using var router = NewRouter();
        var decision = router.Route("hello", null, null, null, null, LangGuess.FromCallback(_ => "pt"));
        Assert.Equal(LayaCheckpoint.Multilingual, decision.Model);
        Assert.Equal(NonEnglish, decision.Reason);
        Assert.Null(decision.Detection);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("C")]
    public void CallbackThatAbstainsFallsThroughToDetection(string? code)
    {
        using var router = NewRouter();
        var decision = router.Route("hello", null, null, null, null, LangGuess.FromCallback(_ => code));
        Assert.Equal(Undecided, decision.Reason);
        Assert.NotNull(decision.Detection);
    }

    [Fact]
    public void CallbackReceivesTheState()
    {
        object? seen = null;
        using var router = NewRouter();
        var state = new Dictionary<string, object?> { ["text"] = "hola" };
        router.Route(state, null, null, null, null, LangGuess.FromCallback(s => { seen = s; return null; }));
        Assert.Same(state, seen);
    }

    [Fact]
    public void RouterInstalledHintAppliesToEveryRequestAndPerCallOverridesIt()
    {
        using var router = NewRouter(LangGuess.FromCallback(s => s is string t && t.Contains("hola") ? "es" : "en"));
        Assert.Equal("Router(lang_guess=...): the caller identified this as non-English text", router.Route("hola amigo").Reason);
        Assert.Equal("Router(lang_guess=...): the caller identified this as English text",
            router.Route("Care este ora in Tokyo?").Reason);
        Assert.Equal(NonEnglish, router.Route("hola amigo", null, null, null, null, "pt").Reason);
    }

    [Fact]
    public void ExplicitLangBeatsAHintButABlankLangDoesNot()
    {
        using var router = NewRouter();
        Assert.Equal("explicit lang='en'", router.Route("hola", null, null, null, "en", "pt").Reason);
        Assert.Equal(NonEnglish, router.Route("hola", null, null, null, "  ", "pt").Reason);
    }

    [Fact]
    public void ReasonQuotesCallerTextLikePythonRepr() =>
        Assert.Equal("explicit lang='x\\u200by'", NewRouter().Route("x", lang: "x​y").Reason);

    [Fact]
    public void PredictAcceptsAPerCallHint()
    {
        using var router = NewRouter();
        var result = router.Predict("hello", new QuestionSet(), null, null, null, LangGuess.FromCode("fr"));
        Assert.Equal(NonEnglish, result.Routing!.Reason);
    }

    // ── bytes states (lang.py:253-255) ────────────────────────────────────────

    [Fact]
    public void ValidUtf8BytesAreReadAsText()
    {
        Assert.Equal("hola mundo", LanguageDetection.StateText(Encoding.UTF8.GetBytes("hola mundo")));
        var analysis = LanguageDetection.Analyse(Encoding.UTF8.GetBytes("Preciso de ajuda com a minha fatura, nao consigo pagar hoje"));
        Assert.Equal("latin", analysis.Script);
        Assert.Equal("pt", analysis.Language);
        Assert.False(analysis.IsEnglish);
    }

    [Fact]
    public void InvalidUtf8BytesAreNoText()
    {
        Assert.Equal("", LanguageDetection.StateText(new byte[] { 0x63, 0x61, 0x66, 0xC3, 0xA9, 0x20, 0xFF, 0x20, 0x6F, 0x6B }));
        Assert.Equal("unknown", LanguageDetection.Analyse(new byte[] { 0xFF, 0xFE }).Script);
    }

    [Fact]
    public void BytesNestedInAStateAreReadToo() =>
        Assert.Equal("ab c", LanguageDetection.StateText(new Dictionary<string, object?> { ["a"] = "ab"u8.ToArray(), ["b"] = "c" }));

    [Fact]
    public void BytesAreNotJsonSerializableSoPredictionRejectsThem() =>
        Assert.Throws<JsonException>(() => PythonJson.State(new byte[] { 1, 2 }));

    // ── email_state(max_chars=...) ────────────────────────────────────────────

    [Fact]
    public void MaxCharsIsIgnoredWhenNotCleaningAndIsNeverAField()
    {
        var state = LayaEmail.State("  Locked out ", "My account is locked and I need help right now.", null, false, null, 5);
        Assert.Equal("My account is locked and I need help right now.", state["body"]);
        Assert.Equal(["subject", "body"], state.Keys);
        Assert.Equal("Locked out", state["subject"]);
    }

    [Theory]
    [InlineData(10, "My account")]
    [InlineData(0, "")]
    [InlineData(-5, "My account is locked ")]
    public void MaxCharsBudgetCutsTheCleanedBody(int maxChars, string body)
    {
        // clean_email_body("My account is locked and I need help right now.", n): 0 -> "", -5 -> 5*5 characters off the end.
        var state = LayaEmail.State("s", "My account is locked and I need help right now.", null, true, null, maxChars);
        Assert.Equal(body, state["body"]);
    }
}
