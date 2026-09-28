//! Combined `LayaRouter` / language detection / email cleaning / shortlist sample. See the
//! README's contract section before touching output formatting: this file's stdout is checked
//! against the Python-recorded goldens generated from `tools/routing_cases.py` (apart from
//! timings, which go to stderr).
//!
//! Five sections, run in order:
//! 1. Language detection over a handful of states — `LanguageDetection::analyse`, no engine.
//! 2. Routing decisions — `LayaRouter::route`, no engine loaded.
//! 3. `LayaRouter::predict` on an English and a Hindi support email (`MaxLoaded=2`).
//! 4. `LayaEmail::clean_body` / `LayaEmail::state` on a raw email, then `LayaPresets::email()`
//!    through the same router.
//! 5. `LayaShortlist::predict` over a 40-intent banking choice question, with the demo hashing
//!    embedder (see `hashing_embedder.rs`) standing in for a real embedding model.
//!
//! All sample inputs are copied verbatim from `tools/routing_cases.py`'s `SAMPLE_*` constants —
//! no runtime JSON is read. See `tools/dump_routing_golden.py`'s `sample_inputs.json` for the
//! same values recorded from Python.

mod hashing_embedder;

use laya::{
    Answer, ArtifactDownloadProgress, LayaEmail, LayaError, LayaOptions, LayaPresets, LayaResult,
    LayaRouter, LayaRouterOptions, LayaShortlist, Question, QuestionSet, Result as LayaResultT,
    laya_shortlist::DEFAULT_SHORTLIST_K,
};
use serde_json::{Value, json};
use std::time::Instant;

const USAGE: &str = "\
Usage: cargo run --release -p laya-routing -- [options]

  --model-root <dir>          Root directory holding one subdirectory per checkpoint
                              (english/, multilingual/, typed-decisions/). Overrides
                              LAYA_ONNX_ROOT for this process.
  --download                  Fetch missing checkpoints from Hugging Face into the local cache.
  --preload                   Build and load every checkpoint up front instead of on first use.
  -h, --help                  Print this message.

With no options the artifact root is resolved from LAYA_ONNX_ROOT.";

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut model_root: Option<String> = None;
    let mut allow_download = false;
    let mut preload = false;

    // Parsed by hand, like laya-sample: an unrecognised argument is rejected with exit code 2
    // rather than silently ignored.
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-root" => {
                if i + 1 >= args.len() {
                    eprintln!("--model-root needs a directory.");
                    eprintln!("{USAGE}");
                    return 2;
                }
                i += 1;
                model_root = Some(args[i].clone());
            }
            "--download" => allow_download = true,
            "--preload" => preload = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return 0;
            }
            other => {
                eprintln!("Unrecognised argument: {other}");
                eprintln!("{USAGE}");
                return 2;
            }
        }
        i += 1;
    }

    // `LayaOptions` has no "root" field of its own (only a single-checkpoint `model_directory`),
    // so `--model-root` is applied by overriding the `LAYA_ONNX_ROOT` env var that
    // `ModelArtifacts::resolve` already reads per checkpoint. `set_var` is `unsafe` on this
    // edition because it is process-global and not thread-safe with a concurrent `getenv`; this
    // happens once, before any threads exist, so that hazard does not apply here.
    if let Some(root) = &model_root {
        unsafe {
            std::env::set_var("LAYA_ONNX_ROOT", root);
        }
    }

    let mut engine_options = LayaOptions::new();
    engine_options.allow_download = allow_download;
    if allow_download {
        engine_options.download_progress =
            Some(std::sync::Arc::new(|p: ArtifactDownloadProgress| {
                eprintln!(
                    "  {}: {}/{}",
                    p.file_name,
                    p.bytes_received,
                    p.total_bytes
                        .map(|b| b.to_string())
                        .unwrap_or_else(|| "?".to_string())
                );
            }));
    }

    let router = match LayaRouter::new(LayaRouterOptions {
        engine_options,
        max_loaded: 2,
        preload,
        ..LayaRouterOptions::default()
    }) {
        Ok(router) => router,
        Err(e) => {
            report_error(&e, allow_download);
            return 1;
        }
    };

    section1_language_detection();
    if let Err(e) = section2_routing_decisions() {
        report_error(&e, allow_download);
        return 1;
    }
    if let Err(e) = section3_router_predict(&router) {
        report_error(&e, allow_download);
        return 1;
    }
    if let Err(e) = section4_email_cleaning(&router) {
        report_error(&e, allow_download);
        return 1;
    }
    if let Err(e) = section5_shortlist(&router) {
        report_error(&e, allow_download);
        return 1;
    }

    0
}

/// A directory that exists but is missing a file, or does not exist at all, is a
/// misconfiguration like any other, and the resolver's message already explains it better than a
/// stack trace would (see laya-sample's `main.rs` for the same match).
fn report_error(e: &LayaError, allow_download: bool) {
    eprintln!("{e}");
    if matches!(
        e,
        LayaError::ArtifactsNotFound { .. }
            | LayaError::DirectoryNotFound { .. }
            | LayaError::MissingFile { .. }
    ) && !allow_download
    {
        eprintln!();
        eprintln!("Point --model-root or LAYA_ONNX_ROOT at a directory holding every checkpoint.");
    }
}

// ============================================================================ section 1

/// ~8 states covering English, accent-stripped Spanish, French, Romanian, Hindi, Arabic,
/// Japanese and a dict state. Copied verbatim from `tools/routing_cases.py`'s
/// `SAMPLE_DETECTION_STATES`.
fn section1_language_detection() {
    println!("== 1. Language detection ==");

    let states: Vec<(&str, Value)> = vec![
        (
            "english",
            json!("The customer was charged twice this month and wants a refund."),
        ),
        (
            "spanish_accent_stripped",
            json!("El cliente fue cobrado dos veces y quiere que le devuelvan el dinero"),
        ),
        (
            "french",
            json!("Le client a été facturé deux fois et demande un remboursement."),
        ),
        (
            "romanian",
            json!("Am fost taxat de două ori pentru factura din luna martie și vreau banii"),
        ),
        (
            "hindi",
            json!("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।"),
        ),
        (
            "arabic",
            json!("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال"),
        ),
        (
            "japanese",
            json!("お客様は二重に請求されたため返金を希望しています。"),
        ),
        (
            "dict_state",
            json!({"subject": "Double charge", "body": "मुझसे दो बार शुल्क लिया गया"}),
        ),
    ];

    for (name, state) in &states {
        let analysis = laya::LanguageDetection::analyse(state);
        let language = analysis.language.as_deref().unwrap_or("-");
        println!(
            "{name:<22} script={script:<10} language={language:<4} is_english={is_english}",
            script = analysis.script,
            is_english = analysis.is_english,
        );
    }
}

// ============================================================================ section 2

/// Question ids of the `customer_service` typed-decisions workflow, used only for the two
/// workflow-detection rows. Matches `laya::laya_router`'s private `TYPED_DECISION_WORKFLOWS` /
/// Python's `_TD_IDS["customer_service"]`.
const CUSTOMER_SERVICE_IDS: &[&str] =
    &["action", "category", "churn_risk", "needs_human", "urgency"];

fn workflow_questions() -> LayaResultT<QuestionSet> {
    let mut questions = QuestionSet::new();
    for id in CUSTOMER_SERVICE_IDS {
        questions.insert(*id, Question::noul("x", Value::Null, Value::Null))?;
    }
    Ok(questions)
}

/// Routing-decision examples (`Route` only, nothing loaded), including explicit model/lang and
/// opt-in workflow detection. Copied verbatim from `tools/routing_cases.py`'s
/// `SAMPLE_ROUTING_DECISIONS`.
fn section2_routing_decisions() -> LayaResultT<()> {
    println!();
    println!("== 2. Routing decisions ==");

    let workflow_ids = workflow_questions()?;

    struct Case<'a> {
        name: &'a str,
        auto_task_detection: bool,
        state: Value,
        questions: Option<&'a QuestionSet>,
        model: Option<&'a str>,
        lang: Option<&'a str>,
    }

    let cases = [
        Case {
            name: "auto_english",
            auto_task_detection: false,
            state: json!("Please refund the duplicate charge on invoice 4411 today."),
            questions: None,
            model: None,
            lang: None,
        },
        Case {
            name: "auto_hindi",
            auto_task_detection: false,
            state: json!("मुझसे दो बार शुल्क लिया गया, कृपया रिफंड करें।"),
            questions: None,
            model: None,
            lang: None,
        },
        Case {
            name: "auto_unknown_latin_romanian",
            auto_task_detection: false,
            state: json!("Gătește-mi o rețetă de sarmale de post pentru mâine."),
            questions: None,
            model: None,
            lang: None,
        },
        Case {
            name: "explicit_model_multilingual",
            auto_task_detection: false,
            state: json!("anything at all"),
            questions: None,
            model: Some("multilingual"),
            lang: None,
        },
        Case {
            name: "explicit_lang_de",
            auto_task_detection: false,
            state: json!("hello there"),
            questions: None,
            model: None,
            lang: Some("de"),
        },
        Case {
            name: "workflow_opt_in_customer_service",
            auto_task_detection: true,
            state: json!({"body": "I was charged twice"}),
            questions: Some(&workflow_ids),
            model: None,
            lang: None,
        },
        Case {
            name: "workflow_opt_in_off_by_default",
            auto_task_detection: false,
            state: json!({"body": "I was charged twice"}),
            questions: Some(&workflow_ids),
            model: None,
            lang: None,
        },
    ];

    for case in &cases {
        let router = LayaRouter::new(LayaRouterOptions {
            auto_task_detection: case.auto_task_detection,
            ..LayaRouterOptions::default()
        })?;
        let decision = router.route(&case.state, case.questions, case.model, None, case.lang)?;
        println!(
            "{name:<26} -> {subdir}  ({reason})",
            name = case.name,
            subdir = decision.model.subdir(),
            reason = decision.reason,
        );
    }

    Ok(())
}

// ============================================================================ section 3 / 4 shared

/// Prints one `LayaResult` from a router `predict` call in the shared per-answer format used by
/// sections 3 and 4: a routed-to header, then one indented line per question.
///
/// `tag` is a short caller-chosen label for the input's language (e.g. `"en"`, `"hi"`) — it is
/// not derived from `RouteDecision::detection`, because a non-Latin-script decision (Hindi, in
/// this sample) carries no `language` guess at all (script detection alone routes it), so there
/// would be nothing to print for that case. The sample uses the same fixed tags per input as
/// `tools/routing_cases.py`'s `SAMPLE_*` entries.
/// Runs `f`, printing its wall-clock time to stderr (never stdout — see the README's contract:
/// stdout must diff identically against the Python-recorded golden, and timings never will).
fn timed<T>(label: &str, f: impl FnOnce() -> LayaResultT<T>) -> LayaResultT<T> {
    let clock = Instant::now();
    let result = f();
    eprintln!(
        "[timing] {label}: {:.0} ms",
        clock.elapsed().as_secs_f64() * 1000.0
    );
    result
}

fn print_routed_result(tag: &str, result: &LayaResult) {
    let decision = result
        .routing()
        .expect("router predict always records a routing decision");
    println!(
        "[{tag}] routed to {subdir}: {reason}",
        subdir = decision.model.subdir(),
        reason = decision.reason,
    );
    for id in result.ids() {
        let answer = &result[id.as_str()];
        let value = match answer {
            Answer::Choice(choice) => choice.choice.clone(),
            Answer::Score(score) => format!("{:.4}", score.score),
            Answer::Noul(noul) => format!("{:.4}", noul.probability),
        };
        println!("  {id:<18} {value}");
    }
}

// ============================================================================ section 3

/// Copied verbatim from `tools/routing_cases.py`'s `SAMPLE_SUPPORT_QUESTIONS`.
fn support_questions() -> LayaResultT<QuestionSet> {
    QuestionSet::new()
        .with(
            "department",
            Question::choice(
                "Which department should handle this request?",
                [
                    ("billing", "invoices, payments, refunds"),
                    ("technical", "bugs, outages, system errors"),
                    ("sales", "pricing, new contracts"),
                    ("other", "everything else"),
                ],
            )?,
        )?
        .with(
            "urgency",
            Question::score(
                "How urgent is this request?",
                ["not urgent", "soon", "critical deadline or blocking issue"],
            )?,
        )?
        .with(
            "churn_risk",
            Question::noul(
                "Does the user threaten to cancel or leave?",
                Value::Null,
                Value::Null,
            ),
        )?
        .with(
            "refund_requested",
            Question::noul(
                "Does the user explicitly request a refund?",
                Value::Null,
                Value::Null,
            ),
        )
}

/// The same support questions on an English and a Hindi email, showing which checkpoint
/// answered and the answers (`MaxLoaded=2`, so both stay resident). Copied verbatim from
/// `tools/routing_cases.py`'s `SAMPLE_SUPPORT_EMAIL_EN` / `SAMPLE_SUPPORT_EMAIL_HI`.
fn section3_router_predict(router: &LayaRouter) -> LayaResultT<()> {
    println!();
    println!("== 3. Router predict ==");

    let questions = support_questions()?;

    let english = json!({
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today \
                 or we will cancel our plan.",
    });
    let result = timed("section 3 english", || {
        router.predict(english, &questions, None, None, None)
    })?;
    print_routed_result("en", &result);

    // Written as escapes deliberately (see `tools/routing_cases.py`'s note on the same pitfall):
    // a hand-retyped Devanagari string is an easy place to silently swap a similar-looking
    // character and produce a "parity bug" that is really a transcription error.
    let hindi = json!({
        "from": "user@acme.com",
        "subject": "चालान में दोहरा शुल्क",
        "body": "मार्च का बिल दो बार लिया गया, कृपया रिफंड करें।",
    });
    let result = timed("section 3 hindi", || {
        router.predict(hindi, &questions, None, None, None)
    })?;
    print_routed_result("hi", &result);

    Ok(())
}

// ============================================================================ section 4

/// A raw email with quoted history, signature and disclaimer, run through
/// `LayaEmail::clean_body` -> `LayaEmail::state` -> `LayaPresets::email()` through the router.
/// Copied verbatim from `tools/routing_cases.py`'s `SAMPLE_RAW_EMAIL_*`.
fn section4_email_cleaning(router: &LayaRouter) -> LayaResultT<()> {
    println!();
    println!("== 4. Email cleaning ==");

    let subject = "Re: Refund for invoice 4411";
    let sender = "priya.sharma@example.com";
    let body = "Hi team,\n\n\
        I was charged twice for the Pro plan this month and the second charge still has \
        not been refunded. This is the third time I have written about it -- please \
        resolve this today or I will need to cancel the account.\n\n\
        Best regards,\n\
        Priya Sharma\n\
        Sent from my iPhone\n\n\
        This email and any files transmitted with it are confidential and intended \
        solely for the use of the individual or entity to whom they are addressed. If \
        you have received this email in error please notify the sender.\n\n\
        On Tue, Sep 22, 2026 at 9:14 AM, Support <support@example.com> wrote:\n\
        > Hi Priya, thanks for reaching out, we are looking into it.\n\
        > - Support Team\n";

    let cleaned = LayaEmail::clean_body(body, laya::laya_email::DEFAULT_MAX_CHARS);
    println!("--- cleaned ---");
    println!("{cleaned}");
    println!("--- end ---");

    let state = LayaEmail::state(subject, body, Some(sender), true, std::iter::empty());
    let questions = LayaPresets::email(None::<[(&str, &str); 0]>);
    let result = timed("section 4 email", || {
        router.predict(state, &questions, None, None, None)
    })?;
    print_routed_result("en", &result);

    Ok(())
}

// ============================================================================ section 5

/// A 40-intent banking choice question through `LayaShortlist::predict` with the hashing
/// embedder (demo only — replace with a real embedding model). Copied verbatim from
/// `tools/routing_cases.py`'s `SAMPLE_BANKING_*`.
///
/// The question id (`"banking_intent"`) is not specified by the Python corpus (which only
/// records the state/instructions/criteria, not a question id); this sample uses
/// `"banking_intent"` as a fixed label for its per-answer output line.
fn section5_shortlist(router: &LayaRouter) -> LayaResultT<()> {
    println!();
    println!("== 5. Shortlist ==");

    const QUESTION_ID: &str = "banking_intent";
    const CRITERIA: &[(&str, &str)] = &[
        ("activate_my_card", "turn on a newly received card"),
        ("age_limit", "minimum or maximum age to use the service"),
        (
            "apple_pay_or_google_pay",
            "adding the card to a mobile wallet",
        ),
        ("atm_support", "which ATMs the card works at"),
        ("automatic_top_up", "automatically topping up the balance"),
        (
            "balance_not_updated_after_bank_transfer",
            "a bank transfer is missing from the balance",
        ),
        (
            "balance_not_updated_after_cheque_or_cash_deposit",
            "a deposit is missing from the balance",
        ),
        ("beneficiary_not_allowed", "cannot add a payment recipient"),
        ("cancel_transfer", "wants to cancel a transfer already sent"),
        (
            "card_about_to_expire",
            "the card is close to its expiry date",
        ),
        ("card_acceptance", "where the card is accepted"),
        ("card_arrival", "asking when a new card will arrive"),
        (
            "card_delivery_estimate",
            "asking how long delivery will take",
        ),
        ("card_linking", "linking a card to the account"),
        ("card_not_working", "the card is declined or not working"),
        (
            "card_payment_fee_charged",
            "an unexpected fee was charged on a card payment",
        ),
        (
            "card_payment_not_recognised",
            "a card payment the customer does not recognise",
        ),
        (
            "card_payment_wrong_exchange_rate",
            "the exchange rate used was wrong",
        ),
        ("card_swallowed", "the ATM kept the card"),
        (
            "cash_withdrawal_charge",
            "an unexpected charge for a cash withdrawal",
        ),
        (
            "cash_withdrawal_not_recognised",
            "a cash withdrawal the customer does not recognise",
        ),
        ("change_pin", "wants to change the PIN"),
        ("compromised_card", "the card may have been compromised"),
        (
            "contactless_not_working",
            "contactless payment is not working",
        ),
        ("country_support", "which countries the service supports"),
        ("declined_card_payment", "a card payment was declined"),
        ("declined_cash_withdrawal", "a cash withdrawal was declined"),
        ("declined_transfer", "a transfer was declined"),
        (
            "direct_debit_payment_not_recognised",
            "a direct debit the customer does not recognise",
        ),
        (
            "disposable_card_limits",
            "limits on a disposable virtual card",
        ),
        (
            "edit_personal_details",
            "wants to edit name, address or other personal details",
        ),
        ("exchange_charge", "a fee charged for currency exchange"),
        ("exchange_rate", "asking what the current exchange rate is"),
        ("exchange_via_app", "exchanging currency inside the app"),
        (
            "extra_charge_on_statement",
            "an unexplained extra charge on the statement",
        ),
        ("failed_transfer", "a transfer failed"),
        (
            "fiat_currency_support",
            "which regular currencies are supported",
        ),
        (
            "get_disposable_virtual_card",
            "wants a disposable virtual card",
        ),
        ("get_physical_card", "wants a physical card"),
        ("getting_spare_card", "wants a spare or backup card"),
    ];

    let questions = QuestionSet::new().with(
        QUESTION_ID,
        Question::choice(
            "What is the customer's banking intent?",
            CRITERIA.iter().map(|&(label, desc)| (label, desc)),
        )?,
    )?;

    let state =
        json!("I tried to withdraw cash from the ATM but it kept the card and gave me no money.");

    let result = timed("section 5 shortlist", || {
        LayaShortlist::predict(
            router,
            state,
            &questions,
            hashing_embedder::embed,
            DEFAULT_SHORTLIST_K,
        )
    })?;

    let info = result
        .shortlist()
        .and_then(|m| m.get(QUESTION_ID))
        .expect("banking_intent was shortlisted");
    println!("kept {} of {}:", info.k, info.n);
    let scores = info
        .scores
        .as_ref()
        .expect("40 options > k=20, so this question was ranked, not passed through");
    for (label, score) in info.labels.iter().zip(scores) {
        println!("  {label:<28} {score:.4}");
    }

    let answer = result[QUESTION_ID]
        .as_choice()
        .expect("banking_intent is a choice question");
    let probability = answer.probability(&answer.choice).unwrap_or(0.0);
    println!("answer: {} ({:.4})", answer.choice, probability);

    Ok(())
}
