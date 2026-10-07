using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Tier 1: email cleaning and state construction, which needs no artifacts. Named cases mirror
/// <c>tests/test_email.py</c>; <see cref="EveryCleanProbeEntry"/> and
/// <see cref="EveryStateProbeEntry"/> consume every entry of <c>golden/routing/email_probe.json</c>.
/// </summary>
public sealed class LayaEmailTests
{
    private const string Disclaimer = "This email is confidential and intended solely for the named addressee.";

    // ── the request survives the footer (regression: disclaimer used to delete whole paragraphs) ──

    [Fact]
    public void InlineFooterNoBlankLineKeepsTheRequest() =>
        Assert.Equal("My account is locked. Please unlock it.",
            LayaEmail.CleanBody($"My account is locked.\n{Disclaimer}\nPlease unlock it."));

    [Fact]
    public void InlineFooterNoTerminalPunctuationStillRecoversTheRequest() =>
        Assert.Equal("My account is locked Please unlock it.",
            LayaEmail.CleanBody($"My account is locked\n{Disclaimer}\nPlease unlock it."));

    [Fact]
    public void InlineFooterBodyIsNeverEmptied() =>
        Assert.Equal("My account is locked.", LayaEmail.CleanBody($"My account is locked. {Disclaimer}"));

    [Fact]
    public void InlineFooterIsNotEmpty() =>
        Assert.NotEqual("", LayaEmail.CleanBody($"My account is locked. {Disclaimer}").Trim());

    [Fact]
    public void EmailStateBodyKeepsTheRequest() =>
        Assert.Equal("My account is locked.",
            LayaEmail.State("Locked out", $"My account is locked. {Disclaimer}")["body"]);

    // ── a pure footer is still removed ───────────────────────────────────────

    [Fact]
    public void StandaloneFooterParagraphIsStillDropped() =>
        Assert.Equal("My account is locked.", LayaEmail.CleanBody($"My account is locked.\n\n{Disclaimer}"));

    [Fact]
    public void WrappedStandaloneFooterIsStillDropped() =>
        Assert.Equal("My account is locked.", LayaEmail.CleanBody(
            "My account is locked.\n\nThis email and any files transmitted with it are\n"
            + "confidential and intended solely for the named addressee."));

    [Fact]
    public void ReceivedInErrorFooterIsStillDropped() =>
        Assert.Equal("Please reopen ticket 4411.", LayaEmail.CleanBody(
            "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it."));

    // ── unrelated cleaning is unchanged ──────────────────────────────────────

    [Fact]
    public void QuotedHistoryIsStillRemoved() =>
        Assert.Equal("Thanks for the update.",
            LayaEmail.CleanBody("Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text"));

    [Fact]
    public void SignatureBlockIsStillRemoved() =>
        Assert.Equal("Hi team,\nCan you confirm the refund?",
            LayaEmail.CleanBody("Hi team,\nCan you confirm the refund?\nRegards,\nAlice"));

    // ── French device footers (regression: only matched with two spaces before the device) ──

    [Theory]
    [InlineData("Envoyé depuis mon iPhone")]
    [InlineData("Envoyé de mon iPad.")]
    [InlineData("Envoyé depuis iPhone")]
    [InlineData("Envoyé de iPad")]
    [InlineData("Envoyé depuis Outlook")]
    [InlineData("ENVOYÉ DEPUIS MON IPHONE")]
    public void FrenchDeviceFooterIsCut(string footer) =>
        Assert.Equal("Merci de rembourser la facture.",
            LayaEmail.CleanBody("Merci de rembourser la facture.\n\n" + footer));

    [Theory]
    [InlineData("Envoyé depuis mon iPhone par erreur.")]
    [InlineData("Envoyé de mon iPad hier.")]
    public void FrenchDeviceSentenceIsKept(string sentence)
    {
        var body = $"Bonjour,\n{sentence}\nMerci de rembourser la facture.";
        Assert.Equal(body, LayaEmail.CleanBody(body));
    }

    [Fact]
    public void EmptyBodyStaysEmpty() => Assert.Equal("", LayaEmail.CleanBody(""));

    [Fact]
    public void NullBodyIsTreatedAsEmpty() => Assert.Equal("", LayaEmail.CleanBody(null));

    // ── email_state ───────────────────────────────────────────────────────────

    [Fact]
    public void StateWithSenderAddsFromField()
    {
        var state = LayaEmail.State("Refund request", "Please refund invoice 4411.", sender: "user@example.com");
        Assert.Equal("user@example.com", state["from"]);
        // Insertion order: subject, body, from — matching Python's dict construction order.
        Assert.Equal(["subject", "body", "from"], state.Keys);
    }

    [Fact]
    public void StateSubjectIsWhitespaceStripped() =>
        Assert.Equal("Refund request", LayaEmail.State("   Refund request   ", "Please help.")["subject"]);

    [Fact]
    public void StateCleanFalseKeepsRawBody() =>
        Assert.Equal("My account is locked.\nThis is not cleaned.",
            LayaEmail.State("s", "My account is locked.\nThis is not cleaned.", clean: false)["body"]);

    [Fact]
    public void StateExtraNoneValuesAreDropped()
    {
        var state = LayaEmail.State("Ticket", "Body text.", extra:
        [
            new("nullable", null),
            new("kept", "value"),
        ]);
        Assert.False(state.ContainsKey("nullable"));
        Assert.Equal("value", state["kept"]);
    }

    // ── every recorded probe entry ────────────────────────────────────────────

    public static IEnumerable<object[]> CleanProbeCases() => ProbeLabels("clean");
    public static IEnumerable<object[]> StateProbeCases() => ProbeLabels("state");

    private static IEnumerable<object[]> ProbeLabels(string section)
    {
        using var doc = RoutingGoldenData.Load("email_probe.json");
        return
        [
            .. doc.RootElement.GetProperty(section).EnumerateArray()
                .Select(e => new object[] { e.GetProperty("label").GetString()! }),
        ];
    }

    private static JsonElement FindEntry(string section, string label)
    {
        using var doc = RoutingGoldenData.Load("email_probe.json");
        foreach (var e in doc.RootElement.GetProperty(section).EnumerateArray())
            if (e.GetProperty("label").GetString() == label)
                return JsonDocument.Parse(e.GetRawText()).RootElement;
        throw new InvalidOperationException($"no email_probe.json[{section}] entry labelled '{label}'");
    }

    [Theory]
    [MemberData(nameof(CleanProbeCases))]
    public void EveryCleanProbeEntry(string label)
    {
        var e = FindEntry("clean", label);
        var body = e.GetProperty("body").GetString();
        var maxChars = e.GetProperty("max_chars").GetInt32();
        Assert.Equal(e.GetProperty("result").GetString(), LayaEmail.CleanBody(body, maxChars));
    }

    [Theory]
    [MemberData(nameof(StateProbeCases))]
    public void EveryStateProbeEntry(string label)
    {
        var e = FindEntry("state", label);
        var subject = e.GetProperty("subject").GetString();
        var body = e.GetProperty("body").GetString();
        var sender = e.GetProperty("sender").ValueKind == JsonValueKind.Null ? null : e.GetProperty("sender").GetString();
        var clean = e.GetProperty("clean").GetBoolean();
        var extra = e.GetProperty("extra").EnumerateObject()
            .Select(p => new KeyValuePair<string, object?>(p.Name, RoutingGoldenData.ToClr(p.Value)));

        var got = LayaEmail.State(subject, body, sender, clean, extra);
        var want = e.GetProperty("result");

        Assert.Equal(want.EnumerateObject().Select(p => p.Name), got.Keys);
        foreach (var p in want.EnumerateObject())
            Assert.Equal(RoutingGoldenData.ToClr(p.Value), got[p.Name]);
    }
}
