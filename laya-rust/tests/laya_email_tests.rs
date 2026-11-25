//! Tier 1: email cleaning. Ports every check from `tests/test_email.py` plus every entry of the
//! routing golden's `email_probe.json` (`clean` and `state` sections).

mod common;

use common::routing_golden;
use laya::laya_email::LayaEmail;
use serde_json::Value;

const DISCLAIMER: &str = "This email is confidential and intended solely for the named addressee.";

// ---------------------------------------------------------------------- named cases (test_email.py)

#[test]
fn inline_footer_keeps_the_request() {
    assert_eq!(
        LayaEmail::clean_body(
            &format!("My account is locked.\n{DISCLAIMER}\nPlease unlock it."),
            3000
        ),
        "My account is locked. Please unlock it."
    );
    assert_eq!(
        LayaEmail::clean_body(
            &format!("My account is locked\n{DISCLAIMER}\nPlease unlock it."),
            3000
        ),
        "My account is locked Please unlock it."
    );
    assert_eq!(
        LayaEmail::clean_body(&format!("My account is locked. {DISCLAIMER}"), 3000),
        "My account is locked."
    );
    assert!(
        !LayaEmail::clean_body(&format!("My account is locked. {DISCLAIMER}"), 3000)
            .trim()
            .is_empty()
    );
}

#[test]
fn email_state_body_keeps_the_request() {
    let state = LayaEmail::state(
        "Locked out",
        &format!("My account is locked. {DISCLAIMER}"),
        None,
        true,
        std::iter::empty(),
    );
    assert_eq!(state["body"].as_str().unwrap(), "My account is locked.");
}

#[test]
fn pure_footer_paragraph_is_still_removed() {
    assert_eq!(
        LayaEmail::clean_body(&format!("My account is locked.\n\n{DISCLAIMER}"), 3000),
        "My account is locked."
    );
    assert_eq!(
        LayaEmail::clean_body(
            "My account is locked.\n\nThis email and any files transmitted with it are\n\
             confidential and intended solely for the named addressee.",
            3000
        ),
        "My account is locked."
    );
    assert_eq!(
        LayaEmail::clean_body(
            "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.",
            3000
        ),
        "Please reopen ticket 4411."
    );
}

#[test]
fn unrelated_cleaning_is_unchanged() {
    assert_eq!(
        LayaEmail::clean_body(
            "Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text",
            3000
        ),
        "Thanks for the update."
    );
    assert_eq!(
        LayaEmail::clean_body(
            "Hi team,\nCan you confirm the refund?\nRegards,\nAlice",
            3000
        ),
        "Hi team,\nCan you confirm the refund?"
    );
    assert_eq!(LayaEmail::clean_body("", 3000), "");
}

// ---------------------------------------------------------------------- email_probe.json

#[test]
fn email_probe_clean_matches_every_entry() {
    let Some(root) = routing_golden::load("email_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };
    let cases = root["clean"].as_array().expect("'clean' must be an array");
    for case in cases {
        let label = case["label"].as_str().unwrap();
        let body = case["body"].as_str().unwrap();
        // Signed on purpose: zero and negative budgets are specified behaviour.
        let max_chars = case["max_chars"].as_i64().unwrap();
        let expected = case["result"].as_str().unwrap();
        assert_eq!(
            LayaEmail::clean_body_signed(body, max_chars),
            expected,
            "email_probe/clean/{label}"
        );
    }
    eprintln!(
        "email_probe_clean_matches_every_entry: checked {} cases",
        cases.len()
    );
}

#[test]
fn email_probe_state_matches_every_entry() {
    let Some(root) = routing_golden::load("email_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };
    let cases = root["state"].as_array().expect("'state' must be an array");
    for case in cases {
        let label = case["label"].as_str().unwrap();
        let subject = case["subject"].as_str().unwrap_or("");
        let body = case["body"].as_str().unwrap_or("");
        let sender = case["sender"].as_str();
        let clean = case["clean"].as_bool().unwrap();
        // `max_chars` is a named parameter of `email_state`, not a state field: it is taken out
        // of `extra` the way Python binds it.
        let max_chars = case["extra"]["max_chars"].as_i64().unwrap_or(3000);
        let extra: Vec<(String, Value)> = case["extra"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter(|(k, _)| k.as_str() != "max_chars")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let actual =
            LayaEmail::state_with_max_chars(subject, body, sender, clean, max_chars, extra);
        let expected = &case["result"];
        assert_eq!(&actual, expected, "email_probe/state/{label}");
    }
    eprintln!(
        "email_probe_state_matches_every_entry: checked {} cases",
        cases.len()
    );
}

// ---------------------------------------------------------------------- max_chars budget

#[test]
fn unsigned_clean_body_is_the_signed_one() {
    let body = "Please refund the duplicate charge on invoice 4411 today.";
    for budget in [0usize, 1, 10, 3000] {
        assert_eq!(
            LayaEmail::clean_body(body, budget),
            LayaEmail::clean_body_signed(body, budget as i64)
        );
    }
}

#[test]
fn budget_zero_and_negative_follow_python_slicing() {
    // `text[:0]` is empty; a negative budget trims `5 * |n|` from the end (the `4 * |n|` pre-cut,
    // then the final `[:n]` slice).
    assert_eq!(LayaEmail::clean_body_signed("Please refund it.", 0), "");
    assert_eq!(LayaEmail::clean_body_signed("", 0), "");
    let body = "abcdefghij abcdefghij abcdefghij";
    assert_eq!(body.chars().count(), 32);
    assert_eq!(
        LayaEmail::clean_body_signed(body, -1),
        "abcdefghij abcdefghij abcde"
    );
    assert_eq!(LayaEmail::clean_body_signed(body, -7), "");
    assert_eq!(LayaEmail::clean_body_signed(body, i64::MIN), "");
    assert_eq!(LayaEmail::clean_body_signed(body, i64::MAX), body);
}

#[test]
fn state_with_max_chars_budgets_the_body_only_when_cleaning() {
    let body = "My account is locked and I need help right now.";
    let cleaned = LayaEmail::state_with_max_chars("Locked", body, None, true, 10, []);
    assert_eq!(cleaned["body"], "My account");
    let raw = LayaEmail::state_with_max_chars("Locked", body, None, false, 10, []);
    assert_eq!(raw["body"], body);
    let zero = LayaEmail::state_with_max_chars("Locked", body, None, true, 0, []);
    assert_eq!(zero["body"], "");
    assert_eq!(zero["subject"], "Locked");
    // the default entry point keeps the 3000-character budget
    let long = "word ".repeat(1000);
    let state = LayaEmail::state("s", &long, None, true, []);
    assert_eq!(state["body"].as_str().unwrap().chars().count(), 3000);
}

#[test]
fn dated_attribution_needs_a_digit_but_undated_ones_stay_body() {
    let undated = "Preciso de ajuda.\nEm resposta ao que voce escreveu:\nMais detalhes.";
    assert_eq!(
        LayaEmail::clean_body(undated, 3000),
        "Preciso de ajuda.\nEm resposta ao que voce escreveu:\nMais detalhes."
    );
    let dated = "Preciso de ajuda.\nEm 10/09 Joao escreveu:\nOld request.";
    assert_eq!(LayaEmail::clean_body(dated, 3000), "Preciso de ajuda.");
}

#[test]
fn wrapped_attribution_takes_its_dated_head_with_it() {
    let body = "Please refund it.\nOn 10/09 Jane Smith\njane@example.com> wrote:\nOld request.";
    assert_eq!(LayaEmail::clean_body(body, 3000), "Please refund it.");
    // an undated head is ordinary body text: only the tail line is cut
    let body = "Please refund it.\nOn the other hand\njane@example.com> wrote:\nOld request.";
    assert_eq!(
        LayaEmail::clean_body(body, 3000),
        "Please refund it.\nOn the other hand"
    );
}

#[test]
fn signoff_name_rule_is_judged_per_token() {
    for (line, cut) in [
        ("Regards, Łukasz", true),
        ("Thanks, Anna Maria Lopez", true),
        ("Thanks, Anna Maria Lopez Garcia", false),
        ("Thanks, żaneta needs help", false),
        ("Thanks for the quick reply", false),
        ("Best regards, D'Arcy-Smith", true),
        ("Regards, Jose\u{301}", true),
    ] {
        let body = format!("Please refund it.\n{line}");
        let expected = if cut {
            "Please refund it.".to_string()
        } else {
            body.clone()
        };
        assert_eq!(LayaEmail::clean_body(&body, 3000), expected, "{line:?}");
    }
}
