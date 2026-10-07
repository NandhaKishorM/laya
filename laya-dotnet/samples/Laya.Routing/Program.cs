using System.Globalization;
using System.Diagnostics;
using Laya;
using Laya.Routing.Sample;

// Combined sample for LanguageDetection, LayaRouter, LayaEmail and LayaShortlist.
//
// OUTPUT CONTRACT (see README.md for the full spec — its stdout is checked against the
// Python-recorded goldens (tools/routing_cases.py), apart from timings, which only ever go to stderr):
//
//   --model-root <dir>   Sets LAYA_ONNX_ROOT for this process (default: the LAYA_ONNX_ROOT
//                         already in the environment). Each checkpoint is then resolved as
//                         <dir>/<subfolder>, same as LayaRouter always does.
//   --download           Allow the router to fetch a missing checkpoint from Hugging Face.
//   --preload             Build every checkpoint up front instead of lazily.
//   -h, --help            Print usage and exit 0.
//
// Exit codes: 0 ok, 1 runtime/artifact error, 2 bad usage.

const string Usage = """
    Usage: dotnet run --project samples/Laya.Routing [options]

      --model-root <dir>   Root directory holding one subdirectory per checkpoint
                            (english/, multilingual/, typed-decisions/). Sets LAYA_ONNX_ROOT
                            for this process; overrides any LAYA_ONNX_ROOT already set.
      --download           Allow fetching a missing checkpoint from Hugging Face.
      --preload             Build every checkpoint up front instead of lazily.
      -h, --help            Print this message.

    With no --model-root the artifact root is resolved from LAYA_ONNX_ROOT.
    """;

var allowDownload = false;
var preload = false;
string? modelRoot = null;

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model-root":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("--model-root needs a directory.");
                Console.Error.WriteLine(Usage);
                return 2;
            }
            modelRoot = args[++i];
            break;

        case "--download":
            allowDownload = true;
            break;

        case "--preload":
            preload = true;
            break;

        case "-h":
        case "--help":
            Console.WriteLine(Usage);
            return 0;

        default:
            Console.Error.WriteLine($"Unrecognised argument: {args[i]}");
            Console.Error.WriteLine(Usage);
            return 2;
    }
}

// LayaOptions has no "model root" concept of its own -- LayaRouter always resolves each
// checkpoint independently via LAYA_ONNX_ROOT/<subfolder>, cache, or download (see
// ModelArtifacts.ResolveAsync). Setting the environment variable for this process is therefore
// the direct equivalent of --model-dir on Laya.Sample for a router that manages several
// checkpoints rather than one fixed directory.
if (modelRoot is not null)
    Environment.SetEnvironmentVariable("LAYA_ONNX_ROOT", modelRoot);

// ── Section 1: language detection ──────────────────────────────────────────

Console.WriteLine("== 1. Language detection ==");
foreach (var (name, state) in DetectionStates())
{
    var analysis = LanguageDetection.Analyse(state);
    var languageDisplay = analysis.Language ?? "-";
    var isEnglish = analysis.IsEnglish ? "true" : "false";
    Console.WriteLine(
        $"{name,-22} script={analysis.Script,-10} language={languageDisplay,-4} is_english={isEnglish}");
}

Console.WriteLine();

// ── Section 2: routing decisions (Route only, nothing loaded) ──────────────

Console.WriteLine("== 2. Routing decisions ==");
using (var plainRouter = new LayaRouter())
using (var autoTaskRouter = new LayaRouter(new LayaRouterOptions { AutoTaskDetection = true }))
{
    RouteAndPrint(plainRouter, "auto_english",
        "Please refund the duplicate charge on invoice 4411 today.");
    RouteAndPrint(plainRouter, "auto_hindi",
        "मुझसे दो बार शुल्क लिया गया, कृपया रिफंड करें।");
    RouteAndPrint(plainRouter, "auto_unknown_latin_romanian",
        "Gătește-mi o rețetă de sarmale de post pentru mâine.");
    RouteAndPrint(plainRouter, "explicit_model_multilingual",
        "anything at all", model: "multilingual");
    RouteAndPrint(plainRouter, "explicit_lang_de", "hello there", lang: "de");

    var typedDecisionQuestions = TypedDecisionStubQuestions();
    RouteAndPrint(autoTaskRouter, "workflow_opt_in_customer_service",
        new Dictionary<string, object?> { ["body"] = "I was charged twice" }, typedDecisionQuestions);
    RouteAndPrint(plainRouter, "workflow_opt_in_off_by_default",
        new Dictionary<string, object?> { ["body"] = "I was charged twice" }, typedDecisionQuestions);
}

Console.WriteLine();

// ── Sections 3-5: everything that needs a real checkpoint ───────────────────

var routerOptions = new LayaRouterOptions
{
    MaxLoaded = 2,
    Preload = preload,
    EngineOptions = new LayaOptions
    {
        AllowDownload = allowDownload,
        DownloadProgress = allowDownload
            ? new Progress<ArtifactDownloadProgress>(p => Console.Error.WriteLine($"  {p}"))
            : null,
    },
};

LayaRouter router;
try
{
    var buildClock = Stopwatch.StartNew();
    router = new LayaRouter(routerOptions);
    buildClock.Stop();
    Console.Error.WriteLine($"router construction: {buildClock.Elapsed.TotalMilliseconds:F0} ms");
}
catch (Exception error) when (error is DirectoryNotFoundException or FileNotFoundException)
{
    Console.Error.WriteLine(error.Message);
    if (!allowDownload)
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine("Point --model-root or LAYA_ONNX_ROOT at a local ONNX export tree.");
    }
    return 1;
}

try
{
    using (router)
    {
        // ── Section 3: router predict over an English and a Hindi support email ─────

        Console.WriteLine("== 3. Router predict ==");
        var supportQuestions = SupportQuestions();

        var englishEmail = new Dictionary<string, object?>
        {
            ["from"] = "user@acme.com",
            ["subject"] = "Duplicate charge on invoice #4411",
            ["body"] = "Hi, we were billed twice for March. Please refund the duplicate today "
                     + "or we will cancel our plan.",
        };
        PredictAndPrint(router, "en", englishEmail, supportQuestions);

        var hindiEmail = new Dictionary<string, object?>
        {
            ["from"] = "user@acme.com",
            ["subject"] = "चालान में दोहरा शुल्क",
            ["body"] = "मार्च का बिल दो बार लिया गया, कृपया रिफंड करें।",
        };
        PredictAndPrint(router, "hi", hindiEmail, supportQuestions);

        Console.WriteLine();

        // ── Section 4: email cleaning ────────────────────────────────────────────────

        Console.WriteLine("== 4. Email cleaning ==");
        const string rawBody =
            "Hi team,\n\n"
            + "I was charged twice for the Pro plan this month and the second charge still has "
            + "not been refunded. This is the third time I have written about it -- please "
            + "resolve this today or I will need to cancel the account.\n\n"
            + "Best regards,\n"
            + "Priya Sharma\n"
            + "Sent from my iPhone\n\n"
            + "This email and any files transmitted with it are confidential and intended "
            + "solely for the use of the individual or entity to whom they are addressed. If "
            + "you have received this email in error please notify the sender.\n\n"
            + "On Tue, Sep 22, 2026 at 9:14 AM, Support <support@example.com> wrote:\n"
            + "> Hi Priya, thanks for reaching out, we are looking into it.\n"
            + "> - Support Team\n";

        var cleaned = LayaEmail.CleanBody(rawBody);
        Console.WriteLine("--- cleaned ---");
        Console.WriteLine(cleaned);
        Console.WriteLine("--- end ---");

        var emailState = LayaEmail.State(
            subject: "Re: Refund for invoice 4411", body: rawBody, sender: "priya.sharma@example.com");
        var emailQuestions = LayaPresets.Email();
        PredictAndPrint(router, "en", emailState, emailQuestions);

        Console.WriteLine();

        // ── Section 5: shortlist ─────────────────────────────────────────────────────

        Console.WriteLine("== 5. Shortlist ==");
        const string bankingState =
            "I tried to withdraw cash from the ATM but it kept the card and gave me no money.";
        const string bankingInstructions = "What is the customer's banking intent?";

        var shortlistQuestions = new QuestionSet
        {
            ["intent"] = Question.Choice(bankingInstructions, BankingCriteria()),
        };

        var shortlistClock = Stopwatch.StartNew();
        var shortlistResult = LayaShortlist.Predict(
            router, bankingState, shortlistQuestions, DemoHashingEmbedder.Embed);
        shortlistClock.Stop();
        Console.Error.WriteLine($"shortlist predict: {shortlistClock.Elapsed.TotalMilliseconds:F0} ms");

        var info = shortlistResult.Shortlist!["intent"];
        Console.WriteLine($"kept {info.K} of {info.N}:");
        for (var i = 0; i < info.Labels.Count; i++)
        {
            var score = info.Scores![i];
            Console.WriteLine($"  {info.Labels[i],-28} {score.ToString("0.0000", CultureInfo.InvariantCulture)}");
        }

        var intentAnswer = shortlistResult["intent"].AsChoice();
        var probability = intentAnswer[intentAnswer.Choice];
        Console.WriteLine(
            $"answer: {intentAnswer.Choice} ({probability.ToString("0.0000", CultureInfo.InvariantCulture)})");
    }
}
catch (Exception error) when (error is DirectoryNotFoundException or FileNotFoundException)
{
    Console.Error.WriteLine(error.Message);
    if (!allowDownload)
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine("Point --model-root or LAYA_ONNX_ROOT at a local ONNX export tree.");
    }
    return 1;
}

return 0;

// ── helpers ──────────────────────────────────────────────────────────────────

static IEnumerable<(string Name, object State)> DetectionStates()
{
    yield return ("english", "The customer was charged twice this month and wants a refund.");
    yield return ("spanish_accent_stripped",
        "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero");
    yield return ("french", "Le client a été facturé deux fois et demande un remboursement.");
    yield return ("romanian", "Am fost taxat de două ori pentru factura din luna martie și vreau banii");
    yield return ("hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।");
    yield return ("arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال");
    yield return ("japanese", "お客様は二重に請求されたため返金を希望しています。");
    yield return ("dict_state", new Dictionary<string, object?>
    {
        ["subject"] = "Double charge",
        ["body"] = "मुझसे दो बार शुल्क लिया गया",
    });
}

static void RouteAndPrint(
    LayaRouter router, string name, object? state, QuestionSet? questions = null,
    string? model = null, string? lang = null)
{
    var decision = router.Route(state, questions, model: model, lang: lang);
    Console.WriteLine($"{name,-26} -> {Subdir(decision.Model)}  ({decision.Reason})");
}

static QuestionSet TypedDecisionStubQuestions()
{
    var questions = new QuestionSet();
    foreach (var id in new[] { "action", "category", "churn_risk", "needs_human", "urgency" })
        questions[id] = Question.Noul("x");
    return questions;
}

static QuestionSet SupportQuestions() => new()
{
    ["department"] = Question.Choice(
        "Which department should handle this request?",
        ("billing", "invoices, payments, refunds"),
        ("technical", "bugs, outages, system errors"),
        ("sales", "pricing, new contracts"),
        ("other", "everything else")),
    ["urgency"] = Question.Score(
        "How urgent is this request?",
        "not urgent", "soon", "critical deadline or blocking issue"),
    ["churn_risk"] = Question.Noul("Does the user threaten to cancel or leave?"),
    ["refund_requested"] = Question.Noul("Does the user explicitly request a refund?"),
};

static void PredictAndPrint(LayaRouter router, string label, object? state, QuestionSet questions)
{
    var clock = Stopwatch.StartNew();
    var result = router.Predict(state, questions);
    clock.Stop();
    Console.Error.WriteLine($"{label} predict: {clock.Elapsed.TotalMilliseconds:F0} ms");

    var decision = result.Routing!;
    Console.WriteLine($"[{label}] routed to {Subdir(decision.Model)}: {decision.Reason}");
    PrintAnswers(result, questions);
}

static void PrintAnswers(LayaResult result, QuestionSet questions)
{
    foreach (var id in result.Ids)
    {
        var answer = result[id];
        var value = answer switch
        {
            ChoiceAnswer choice => choice.Choice,
            ScoreAnswer score => score.Score.ToString("0.0000", CultureInfo.InvariantCulture),
            NoulAnswer noul => noul.Probability.ToString("0.0000", CultureInfo.InvariantCulture),
            _ => "?",
        };
        Console.WriteLine($"  {id,-18} {value}");
    }
}

static string Subdir(LayaCheckpoint checkpoint) => checkpoint switch
{
    LayaCheckpoint.English => "english",
    LayaCheckpoint.Multilingual => "multilingual",
    LayaCheckpoint.TypedDecisions => "typed-decisions",
    _ => throw new ArgumentOutOfRangeException(nameof(checkpoint), checkpoint, null),
};

static IEnumerable<KeyValuePair<string, object?>> BankingCriteria()
{
    (string, string)[] entries =
    [
        ("activate_my_card", "turn on a newly received card"),
        ("age_limit", "minimum or maximum age to use the service"),
        ("apple_pay_or_google_pay", "adding the card to a mobile wallet"),
        ("atm_support", "which ATMs the card works at"),
        ("automatic_top_up", "automatically topping up the balance"),
        ("balance_not_updated_after_bank_transfer", "a bank transfer is missing from the balance"),
        ("balance_not_updated_after_cheque_or_cash_deposit", "a deposit is missing from the balance"),
        ("beneficiary_not_allowed", "cannot add a payment recipient"),
        ("cancel_transfer", "wants to cancel a transfer already sent"),
        ("card_about_to_expire", "the card is close to its expiry date"),
        ("card_acceptance", "where the card is accepted"),
        ("card_arrival", "asking when a new card will arrive"),
        ("card_delivery_estimate", "asking how long delivery will take"),
        ("card_linking", "linking a card to the account"),
        ("card_not_working", "the card is declined or not working"),
        ("card_payment_fee_charged", "an unexpected fee was charged on a card payment"),
        ("card_payment_not_recognised", "a card payment the customer does not recognise"),
        ("card_payment_wrong_exchange_rate", "the exchange rate used was wrong"),
        ("card_swallowed", "the ATM kept the card"),
        ("cash_withdrawal_charge", "an unexpected charge for a cash withdrawal"),
        ("cash_withdrawal_not_recognised", "a cash withdrawal the customer does not recognise"),
        ("change_pin", "wants to change the PIN"),
        ("compromised_card", "the card may have been compromised"),
        ("contactless_not_working", "contactless payment is not working"),
        ("country_support", "which countries the service supports"),
        ("declined_card_payment", "a card payment was declined"),
        ("declined_cash_withdrawal", "a cash withdrawal was declined"),
        ("declined_transfer", "a transfer was declined"),
        ("direct_debit_payment_not_recognised", "a direct debit the customer does not recognise"),
        ("disposable_card_limits", "limits on a disposable virtual card"),
        ("edit_personal_details", "wants to edit name, address or other personal details"),
        ("exchange_charge", "a fee charged for currency exchange"),
        ("exchange_rate", "asking what the current exchange rate is"),
        ("exchange_via_app", "exchanging currency inside the app"),
        ("extra_charge_on_statement", "an unexplained extra charge on the statement"),
        ("failed_transfer", "a transfer failed"),
        ("fiat_currency_support", "which regular currencies are supported"),
        ("get_disposable_virtual_card", "wants a disposable virtual card"),
        ("get_physical_card", "wants a physical card"),
        ("getting_spare_card", "wants a spare or backup card"),
    ];
    foreach (var (label, description) in entries)
        yield return new KeyValuePair<string, object?>(label, description);
}
