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
        "Please unlock it."
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
        let max_chars = case["max_chars"].as_u64().unwrap() as usize;
        let expected = case["result"].as_str().unwrap();
        assert_eq!(
            LayaEmail::clean_body(body, max_chars),
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
        let extra: Vec<(String, Value)> = case["extra"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let actual = LayaEmail::state(subject, body, sender, clean, extra);
        let expected = &case["result"];
        assert_eq!(&actual, expected, "email_probe/state/{label}");
    }
    eprintln!(
        "email_probe_state_matches_every_entry: checked {} cases",
        cases.len()
    );
}
