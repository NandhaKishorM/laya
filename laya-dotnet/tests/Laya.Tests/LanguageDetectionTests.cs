using System.Text.Json;

namespace Laya.Tests;

/// <summary>
/// Tier 1: script and Latin-language detection, which needs no artifacts. Named cases mirror
/// <c>tests/test_router.py</c>'s language-detection sections; <see cref="EveryLangProbeEntry"/>
/// consumes every entry of <c>golden/routing/lang_probe.json</c>.
/// </summary>
public sealed class LanguageDetectionTests
{
    // ── script detection ─────────────────────────────────────────────────────

    [Theory]
    [InlineData("The customer was charged twice and wants a refund.", "latin")]
    [InlineData("Հայերեն", "armenian")]
    [InlineData("ՀԱՅԵՐԵՆ", "armenian")]
    [InlineData("։֊", "unknown")] // Armenian punctuation only: no letters
    [InlineData("Le client a été facturé deux fois et demande un remboursement.", "latin")]
    [InlineData("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari")]
    [InlineData("お客様は二重に請求されたため返金を希望しています。", "kana")]
    [InlineData("客户被重复扣款要求退款", "han")]
    [InlineData("고객이 두 번 청구되어 환불을 원합니다", "hangul")]
    [InlineData("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic")]
    [InlineData("வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil")]
    [InlineData("С клиента дважды сняли деньги и он хочет возврат", "cyrillic")]
    [InlineData("ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai")]
    [InlineData("Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek")]
    [InlineData("הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew")]
    [InlineData("", "unknown")]
    [InlineData("12345 6789", "unknown")]
    public void DetectScriptMatchesPython(string text, string expected) =>
        Assert.Equal(expected, LanguageDetection.DetectScript(text));

    [Fact]
    public void DetectScriptTieBreaksOnFirstEncounteredNonLatinScript()
    {
        // "latin" is folded into the tally only after the whole text is scanned, so it never wins
        // a tie against a non-Latin script actually present, whichever side of the text it's on.
        Assert.Equal("greek", LanguageDetection.DetectScript("ΑА")); // greek first
        Assert.Equal("cyrillic", LanguageDetection.DetectScript("АΑ")); // cyrillic first
        Assert.Equal("han", LanguageDetection.DetectScript("中 a"));
        Assert.Equal("han", LanguageDetection.DetectScript("a 中"));
        Assert.Equal("greek", LanguageDetection.DetectScript("ΑАԱ")); // three-way tie: greek first
    }

    // ── is_english / analyse ─────────────────────────────────────────────────

    [Theory]
    [InlineData("Please refund the duplicate charge on invoice 4411 today.", true)]
    [InlineData("Հայերեն", false)]
    [InlineData("refund me", true)]
    [InlineData("ग्राहक से दो बार शुल्क लिया गया", false)]
    [InlineData("お客様は二重に請求されました", false)]
    [InlineData("С клиента дважды сняли деньги", false)]
    [InlineData("Le client a été facturé deux fois et il demande un remboursement pour la "
        + "facture qui a été payée le mois dernier avec la carte de crédit", false)]
    [InlineData("Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
        + "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde", false)]
    // Latin-script languages with no stopword list of their own (#35): an unidentified language
    // must never be assumed English.
    [InlineData("Gătește-mi o rețetă de sarmale de post pentru mâine.", false)]
    [InlineData("Am fost taxat de două ori pentru factura din luna martie și vreau banii", false)]
    [InlineData("Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", false)]
    [InlineData("Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", false)]
    [InlineData("Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", false)]
    [InlineData("Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", false)]
    // English with the odd loanword must not tip over into the multilingual checkpoint.
    [InlineData("We visited a cafe in Zurich and the naive assumption about the "
        + "invoice was wrong, so please refund the duplicate charge", true)]
    public void IsEnglishMatchesPython(string text, bool expected) =>
        Assert.Equal(expected, LanguageDetection.IsEnglish(text));

    [Fact]
    public void UndecidedIsFlaggedRatherThanNamingALanguage()
    {
        // Undecided is reported as undecided rather than dressed up as a detection: a single
        // shared function word used to name a language ("para" in Turkish text was called Spanish).
        var a = LanguageDetection.Analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor");
        Assert.True(a.LanguageUndecided);
        Assert.Null(a.Language);

        var b = LanguageDetection.Analyse("Please refund the duplicate charge on the invoice");
        Assert.False(b.LanguageUndecided);
    }

    [Fact]
    public void DiacriticRateIsReportedAndZeroForPlainEnglish()
    {
        Assert.True(LanguageDetection.Analyse("Gătește-mi o rețetă de sarmale").DiacriticRate > 0.02);
        Assert.Equal(0.0, LanguageDetection.Analyse("Please refund the duplicate charge today").DiacriticRate);
    }

    [Fact]
    public void GuessLatinLanguageZeroTieInventsNothing() =>
        // A 0-0 tie between non-English stopword lists is no evidence for any of them.
        Assert.Null(LanguageDetection.GuessLatinLanguage("Cât e ora acum la Tokyo"));

    [Fact]
    public void KnownGapRomanianWithoutDiacriticsReadsAsEnglish() =>
        // Known limitation, kept visible on purpose (see laya/lang.py and #35): a real LID model
        // is the fix, not more stopwords.
        Assert.True(LanguageDetection.IsEnglish("Care este ora in Tokyo?"));

    // ── Latin language guess ─────────────────────────────────────────────────

    [Theory]
    [InlineData("The customer was charged twice and wants a refund for this invoice", "en")]
    [InlineData("Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr")]
    [InlineData("Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung", "de")]
    [InlineData("El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura", "es")]
    [InlineData("refund", null)]
    public void GuessLatinLanguageMatchesPython(string text, string? expected) =>
        Assert.Equal(expected, LanguageDetection.GuessLatinLanguage(text));

    [Fact]
    public void GuessLatinLanguageLongEnglishStaysEn() =>
        Assert.Equal("en", LanguageDetection.GuessLatinLanguage(
            "Please refund the duplicate charge on invoice 4411 today because "
            + "we have been waiting for three days and nobody has replied to us"));

    // ── state flattening ──────────────────────────────────────────────────────

    [Fact]
    public void StateTextFlattensDictListAndIgnoresKeys()
    {
        Assert.Contains("charged twice", LanguageDetection.StateText(
            new Dictionary<string, object?> { ["body"] = "charged twice", ["n"] = 3L }));
        Assert.Contains("deep", LanguageDetection.StateText(
            new Dictionary<string, object?> { ["a"] = new Dictionary<string, object?> { ["b"] = new List<object?> { "deep" } } }));
        Assert.Contains("x", LanguageDetection.StateText(
            new List<object?> { "x", new Dictionary<string, object?> { ["y"] = "z" } }));
        Assert.Equal("", LanguageDetection.StateText(null));

        // English keys around Hindi content must not drive detection toward English.
        var keyed = new Dictionary<string, object?> { ["subject"] = "नमस्ते", ["body"] = "ग्राहक से दो बार शुल्क लिया गया" };
        Assert.False(LanguageDetection.Analyse(keyed).IsEnglish);
    }

    // ── script profile ────────────────────────────────────────────────────────

    [Fact]
    public void ScriptProfileReportsFractions()
    {
        var armenian = LanguageDetection.Analyse("Հայերեն").ScriptProfile;
        Assert.Equal(1.0, armenian["armenian"]);

        Assert.Equal(0.7, LanguageDetection.Analyse("Հայերեն abc").NonLatinFraction);
    }

    // ── plain-ASCII Romance (#172): accents stripped must still route non-English ──────────────

    [Theory]
    [InlineData("es", "El pedido llego roto y nadie responde cuando escribo al soporte")]
    [InlineData("es", "Quiero cancelar mi plan y pedir un reembolso")]
    [InlineData("es", "La factura tiene un error en el importe total")]
    [InlineData("es", "Necesito que me devuelvan el dinero de la compra duplicada")]
    [InlineData("it", "Il cliente e stato addebitato due volte e vuole un rimborso")]
    [InlineData("it", "Voglio cancellare il mio abbonamento e chiedere un rimborso")]
    [InlineData("it", "La fattura contiene un errore nell importo totale")]
    [InlineData("pt", "O cliente foi cobrado duas vezes e quer o dinheiro de volta")]
    [InlineData("fr", "Le client a ete facture deux fois et demande un remboursement")]
    [InlineData("fr", "Je ne peux pas acceder a mon compte et j ai besoin d aide")]
    public void PlainAsciiRomanceIsIdentifiedAndNotEnglish(string expectedLang, string text)
    {
        Assert.Equal(expectedLang, LanguageDetection.GuessLatinLanguage(text));
        Assert.False(LanguageDetection.IsEnglish(text));
    }

    [Theory]
    [InlineData("La facturación tiene un error y necesito una corrección urgente")]
    [InlineData("La fattura è sbagliata, devo avere un rimborso per il pagamento")]
    [InlineData("La commande est arrivée cassée et personne ne répond au support")]
    public void AccentedRomanceRoutesOnDiacritics(string text) =>
        Assert.False(LanguageDetection.IsEnglish(text));

    [Theory]
    [InlineData("The customer was charged twice and wants a refund for this invoice")]
    [InlineData("Please cancel my subscription and refund the duplicate charge today")]
    [InlineData("The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio")]
    [InlineData("Our MI5 and UN contacts discussed the DOS attack in LA last month")]
    [InlineData("No refund was issued, so I am writing to you again about invoice 4411")]
    [InlineData("no refund no reply")]
    [InlineData("The son of the director filed a complaint about the duplicate invoice")]
    public void EnglishWithRomanceLoanwordsStaysEnglish(string text) =>
        Assert.True(LanguageDetection.IsEnglish(text));

    [Fact]
    public void SharedWordsAloneNameNoLanguageButStillRouteNonEnglish()
    {
        // "la" and "e" alone say "not English" without saying which language; the Romanian
        // diacritic rate is what actually routes it, not a guessed language.
        Assert.Null(LanguageDetection.Analyse("Cât e ora acum la Tokyo").Language);
        Assert.Equal("it", LanguageDetection.GuessLatinLanguage("La fattura contiene un errore nell importo totale"));
        Assert.Equal("es", LanguageDetection.GuessLatinLanguage("La factura tiene un error en el importe total"));
    }

    // ── every recorded probe entry ───────────────────────────────────────────

    public static IEnumerable<object[]> LangProbeCases()
    {
        using var doc = LoadLangProbe();
        return [.. doc.RootElement.EnumerateArray().Select(e => new object[] { e.GetProperty("label").GetString()! })];
    }

    private static JsonElement FindLangProbe(string label)
    {
        using var doc = LoadLangProbe();
        foreach (var e in doc.RootElement.EnumerateArray())
            if (e.GetProperty("label").GetString() == label)
                return JsonDocument.Parse(e.GetRawText()).RootElement;
        throw new InvalidOperationException($"no lang_probe.json entry labelled '{label}'");
    }

    private static JsonDocument LoadLangProbe() => RoutingGoldenData.Load("lang_probe.json");

    [Theory]
    [MemberData(nameof(LangProbeCases))]
    public void EveryLangProbeEntry(string label)
    {
        var e = FindLangProbe(label);
        var stateObj = RoutingGoldenData.ToClr(e.GetProperty("state"));

        var stateText = LanguageDetection.StateText(stateObj);
        Assert.Equal(e.GetProperty("state_text").GetString(), stateText);

        Assert.Equal(e.GetProperty("detect_script").GetString(), LanguageDetection.DetectScript(stateText));

        AssertScriptProfile(e.GetProperty("script_profile"), LanguageDetection.ScriptProfile(stateText));

        var latinProfile = LanguageDetection.LatinProfile(stateText);
        var wantLatin = e.GetProperty("latin_profile");
        Assert.Equal(GetNullableString(wantLatin, "language"), latinProfile.Language);
        Assert.Equal(wantLatin.GetProperty("english_hits").GetInt32(), latinProfile.EnglishHits);
        Assert.Equal(wantLatin.GetProperty("diacritic_rate").GetDouble(), latinProfile.DiacriticRate, 1e-9);
        Assert.Equal(wantLatin.GetProperty("looks_non_english").GetBoolean(), latinProfile.LooksNonEnglish);

        var analysed = LanguageDetection.Analyse(stateObj);
        var wantAnalyse = e.GetProperty("analyse");
        Assert.Equal(wantAnalyse.GetProperty("script").GetString(), analysed.Script);
        AssertScriptProfile(wantAnalyse.GetProperty("script_profile"), analysed.ScriptProfile);
        Assert.Equal(GetNullableString(wantAnalyse, "language"), analysed.Language);
        Assert.Equal(wantAnalyse.GetProperty("is_english").GetBoolean(), analysed.IsEnglish);
        Assert.Equal(wantAnalyse.GetProperty("language_undecided").GetBoolean(), analysed.LanguageUndecided);
        Assert.Equal(wantAnalyse.GetProperty("diacritic_rate").GetDouble(), analysed.DiacriticRate, 1e-9);
        Assert.Equal(wantAnalyse.GetProperty("non_latin_fraction").GetDouble(), analysed.NonLatinFraction, 1e-9);
        Assert.Equal(GetNullableString(wantAnalyse, "mixed_segment"), analysed.MixedSegment);
    }

    private static void AssertScriptProfile(JsonElement want, IReadOnlyDictionary<string, double> got)
    {
        var wantMap = want.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetDouble(), StringComparer.Ordinal);
        Assert.Equal(wantMap.Count, got.Count);
        foreach (var (k, v) in wantMap)
        {
            Assert.True(got.TryGetValue(k, out var actual), $"missing script '{k}' in {string.Join(",", got.Keys)}");
            Assert.Equal(v, actual, 1e-9);
        }
    }

    private static string? GetNullableString(JsonElement obj, string prop) =>
        obj.TryGetProperty(prop, out var v) && v.ValueKind != JsonValueKind.Null ? v.GetString() : null;
}
