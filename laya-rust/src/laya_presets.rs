//! Ready-to-use question sets for common production decision workflows. Ports `laya/presets.py`;
//! wording must be byte-identical to it.
//!
//! Every preset is fixed, known-good data, so the `.expect(...)` calls below can never actually
//! fail — they exist only because [`Question::choice`] and [`Question::score`] return `Result` for
//! the sake of *caller*-supplied data. A failure here would mean a bug in this file, not bad input.

use crate::questions::{Question, QuestionSet};
use serde_json::Value;

const INFALLIBLE: &str = "preset question data is fixed and always valid";

/// Namespace for the built-in presets. See the module docs for the source this ports.
pub struct LayaPresets;

impl LayaPresets {
    /// The routing categories [`Self::email`] uses when none are supplied, in offer order.
    /// Python's default is a `dict`, whose iteration order the language guarantees; this ordered
    /// slice is the equivalent for a caller who cares about order (which option order always does
    /// — see `sequence-construction.md`'s "Positional choice labels").
    pub const DEFAULT_EMAIL_CATEGORIES: &'static [(&'static str, &'static str)] = &[
        ("billing", "invoices, payments, refunds"),
        ("technical", "bugs, outages, integrations"),
        ("sales", "pricing, demos, new purchases"),
        ("security", "phishing, scams, account compromise"),
        ("hr", "hiring, leave, payroll"),
        ("other", "none of the above"),
    ];

    /// Preset questions for customer support ticket triage. Ports `triage_questions()`.
    pub fn triage() -> QuestionSet {
        QuestionSet::new()
            .with(
                "intent",
                Question::choice(
                    "What does the customer want in `message`?",
                    [
                        ("refund", "money returned or a duplicate charge reversed"),
                        ("technical_help", "a bug, outage or integration problem"),
                        (
                            "billing_question",
                            "a question about an invoice, plan or payment method",
                        ),
                        ("information", "general information, pricing or how-to"),
                        ("cancellation", "wants to cancel or downgrade"),
                        ("other", "none of the other options fits"),
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
            .with(
                "is_urgent",
                Question::noul(
                    "Does `message` communicate time pressure or a deadline?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "frustration",
                Question::score(
                    "How frustrated does the customer sound in `message`?",
                    [
                        "calm and neutral",
                        "concerned but civil",
                        "clearly annoyed",
                        "very angry or using strong language",
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
            .with(
                "refund_requested",
                Question::noul(
                    "Does the customer ask for money back?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "churn_risk",
                Question::noul(
                    "Does `message` suggest the customer may leave for a competitor or cancel?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
    }

    /// Preset questions for inbound email triage and threat filtering. Ports `email_questions()`.
    ///
    /// `categories` are routing categories as ordered `(label, description)` pairs; option order
    /// decides which label each logit is attached to, so pass an explicitly ordered sequence
    /// rather than an unordered map. `None` uses [`Self::DEFAULT_EMAIL_CATEGORIES`].
    pub fn email<'a, I>(categories: Option<I>) -> QuestionSet
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let category_question = match categories {
            Some(cats) => Question::choice(
                "Which team should handle the email in `body`?",
                cats.into_iter()
                    .map(|(label, desc)| (label.to_string(), desc.to_string())),
            ),
            None => Question::choice(
                "Which team should handle the email in `body`?",
                Self::DEFAULT_EMAIL_CATEGORIES
                    .iter()
                    .map(|&(l, d)| (l.to_string(), d.to_string())),
            ),
        }
        .expect(INFALLIBLE);

        QuestionSet::new()
            .with("category", category_question)
            .expect(INFALLIBLE)
            .with(
                "is_spam",
                Question::noul(
                    "Is this email unsolicited spam or bulk marketing?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "is_phishing",
                Question::noul(
                    "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                    "a legitimate email",
                    "phishing, scam, or fraud",
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "urgency",
                Question::score(
                    "How urgent is the request in `body`?",
                    [
                        "no time pressure",
                        "needs attention soon",
                        "blocking issue or hard deadline",
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
            .with(
                "needs_reply",
                Question::noul("Does the sender expect a reply?", Value::Null, Value::Null),
            )
            .expect(INFALLIBLE)
    }

    /// Preset questions for real-time LLM input guardrails. Ports `guard_questions()`.
    pub fn guard() -> QuestionSet {
        QuestionSet::new()
            .with(
                "jailbreak",
                Question::noul(
                    "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "prompt_injection",
                Question::noul(
                    "Does `prompt` contain instructions aimed at the AI system rather than a genuine user request?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "sensitive_data",
                Question::noul(
                    "Does `prompt` contain credentials, personal data or other sensitive information?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "harm_severity",
                Question::score(
                    "How much harm would complying with `prompt` cause?",
                    [
                        "none: ordinary request",
                        "minor: mildly inappropriate",
                        "serious: unsafe advice or abuse",
                        "severe: dangerous or illegal",
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
            // Python uses None for all descriptions here — bare labels only.
            .with(
                "topic",
                Question::choice_labels(
                    "What is `prompt` about?",
                    [
                        "product_support",
                        "coding",
                        "general_knowledge",
                        "personal_advice",
                        "security_testing",
                        "other",
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
    }

    /// Preset questions for content safety and moderation. Ports `moderation_questions()`.
    pub fn moderation() -> QuestionSet {
        QuestionSet::new()
            .with(
                "toxic",
                Question::noul(
                    "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "harassment",
                Question::noul(
                    "Does `post` target or harass a specific person?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "threat",
                Question::noul(
                    "Does `post` threaten violence, harm or intimidation?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "spam",
                Question::noul("Is `post` spam or advertising?", Value::Null, Value::Null),
            )
            .expect(INFALLIBLE)
            .with(
                "severity",
                Question::score(
                    "How severe is any rule-breaking in `post`?",
                    [
                        "no rule-breaking: ordinary on-topic post",
                        "mild: rude tone or off-topic, no target",
                        "clear violation: insults, harassment or spam aimed at someone",
                        "severe: threats, hate speech or calls for violence",
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
    }

    /// Preset questions for intelligent model routing. Ports `router_questions()`.
    pub fn router() -> QuestionSet {
        QuestionSet::new()
            .with(
                "difficulty",
                Question::score(
                    "How hard is `request` for a language model?",
                    [
                        "trivial: a lookup or one-liner",
                        "easy: short answer, no reasoning",
                        "moderate: several steps",
                        "hard: long multi-step reasoning or specialist knowledge",
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
            .with(
                "domain",
                Question::choice(
                    "What domain does `request` belong to?",
                    [
                        (
                            "code",
                            "software engineering, programming, refactoring, architecture, debugging",
                        ),
                        ("math_or_logic", "mathematics, logic puzzles, proofs, complex calculation"),
                        ("writing", "creative writing, essays, emails, blog posts, copywriting"),
                        ("factual_lookup", "facts, definitions, trivia, history"),
                        ("data_analysis", "statistics, SQL, data manipulation, metrics"),
                        ("chitchat", "casual conversation, greetings, small talk"),
                    ],
                )
                .expect(INFALLIBLE),
            )
            .expect(INFALLIBLE)
            .with(
                "needs_tools",
                Question::noul(
                    "Does answering `request` require external tools, search or private data?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
            .with(
                "is_sensitive",
                Question::noul(
                    "Does `request` involve money, legal, medical or safety consequences?",
                    Value::Null,
                    Value::Null,
                ),
            )
            .expect(INFALLIBLE)
    }
}
