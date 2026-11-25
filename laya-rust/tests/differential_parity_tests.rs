//! Differential parity: replays the generated corpus in `differential_probe.json` (written by
//! `tools/dump_differential_golden.py` from the real Python `laya.lang`, `laya.router` and
//! `laya.email`) through the Rust port and requires every answer to match. All mismatches are
//! collected first; the failure message lists the first 20.

mod common;

use common::routing_golden;
use laya::language_detection::LanguageDetection;
use laya::laya_email::LayaEmail;
use laya::laya_router::{LangGuess, LayaRouter, LayaRouterOptions};
use serde_json::Value;

struct Mismatch {
    label: String,
    input: String,
    python: String,
    rust: String,
}

/// Floats are compared bit for bit: both sides compute the same IEEE operations.
fn close(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn show(s: &str) -> String {
    let shown: String = s.chars().take(300).collect();
    format!("{shown:?}")
}

fn record(out: &mut Vec<Mismatch>, label: &str, input: String, python: String, rust: String) {
    out.push(Mismatch {
        label: label.to_string(),
        input,
        python,
        rust,
    });
}

fn check_lang(cases: &[Value], out: &mut Vec<Mismatch>) {
    for case in cases {
        let label = case["label"].as_str().unwrap();
        let state = &case["state"];
        let input = show(&state.to_string());
        let text = LanguageDetection::state_text_default(state);
        let expected_text = case["state_text"].as_str().unwrap();
        if text != expected_text {
            record(
                out,
                &format!("{label}: state_text"),
                input.clone(),
                show(expected_text),
                show(&text),
            );
        }
        let a = &case["analyse"];
        let got = LanguageDetection::analyse(state);
        let mut bad = Vec::new();
        if got.script != a["script"].as_str().unwrap() {
            bad.push("script");
        }
        if got.language.as_deref() != a["language"].as_str() {
            bad.push("language");
        }
        if got.is_english != a["is_english"].as_bool().unwrap() {
            bad.push("is_english");
        }
        if got.language_undecided != a["language_undecided"].as_bool().unwrap() {
            bad.push("language_undecided");
        }
        if !close(got.diacritic_rate, a["diacritic_rate"].as_f64().unwrap()) {
            bad.push("diacritic_rate");
        }
        if !close(
            got.non_latin_fraction,
            a["non_latin_fraction"].as_f64().unwrap(),
        ) {
            bad.push("non_latin_fraction");
        }
        if got.mixed_segment.as_deref() != a["mixed_segment"].as_str() {
            bad.push("mixed_segment");
        }
        let profile = a["script_profile"].as_object().unwrap();
        let same_profile = profile.len() == got.script_profile.len()
            && profile.iter().all(|(k, v)| {
                got.script_profile
                    .get(k)
                    .is_some_and(|g| close(*g, v.as_f64().unwrap()))
            });
        if !same_profile {
            bad.push("script_profile");
        }
        if !bad.is_empty() {
            record(
                out,
                &format!("{label}: analyse {bad:?}"),
                input,
                a.to_string(),
                format!(
                    "script={} language={:?} is_english={} undecided={} diacritic={} non_latin={} mixed={:?} profile={:?}",
                    got.script,
                    got.language,
                    got.is_english,
                    got.language_undecided,
                    got.diacritic_rate,
                    got.non_latin_fraction,
                    got.mixed_segment,
                    got.script_profile
                ),
            );
        }
    }
}

fn check_routes(lang_cases: &[Value], cases: &[Value], out: &mut Vec<Mismatch>) {
    let plain = LayaRouter::new(LayaRouterOptions::default()).unwrap();
    for (n, case) in cases.iter().enumerate() {
        let index = case["state"].as_u64().unwrap() as usize;
        let state = &lang_cases[index]["state"];
        let kwargs = &case["kwargs"];
        let lang = kwargs.get("lang").and_then(Value::as_str);
        let guess = kwargs
            .get("lang_guess")
            .and_then(Value::as_str)
            .map(LangGuess::from);
        let installed;
        let router = match kwargs.get("router_lang_guess").and_then(Value::as_str) {
            Some(code) => {
                installed = LayaRouter::new(LayaRouterOptions::default())
                    .unwrap()
                    .with_lang_guess(code);
                &installed
            }
            None => &plain,
        };
        let decision = router
            .route_with_lang_guess(state, None, None, None, lang, guess.as_ref())
            .unwrap();
        let model = decision.model.subdir();
        let label = format!("route #{n} state {index} {kwargs}");
        if model != case["model"].as_str().unwrap()
            || decision.reason != case["reason"].as_str().unwrap()
        {
            record(
                out,
                &label,
                show(&state.to_string()),
                format!("{} / {}", case["model"], case["reason"]),
                format!("{model:?} / {:?}", decision.reason),
            );
        }
    }
}

fn check_email_clean(budgets: &[i64], cases: &[Value], out: &mut Vec<Mismatch>) {
    for case in cases {
        let label = case["label"].as_str().unwrap();
        let body = case["body"].as_str().unwrap();
        let results = case["results"].as_array().unwrap();
        for (budget, expected) in budgets.iter().zip(results) {
            let got = LayaEmail::clean_body_signed(body, *budget);
            if got != expected.as_str().unwrap() {
                record(
                    out,
                    &format!("{label} max_chars={budget}"),
                    show(body),
                    show(expected.as_str().unwrap()),
                    show(&got),
                );
            }
        }
    }
}

fn check_email_state(cases: &[Value], out: &mut Vec<Mismatch>) {
    for case in cases {
        let label = case["label"].as_str().unwrap();
        let max_chars = case["extra"]["max_chars"].as_i64().unwrap_or(3000);
        let extra: Vec<(String, Value)> = case["extra"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, _)| k.as_str() != "max_chars")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let got = LayaEmail::state_with_max_chars(
            case["subject"].as_str().unwrap(),
            case["body"].as_str().unwrap(),
            case["sender"].as_str(),
            case["clean"].as_bool().unwrap(),
            max_chars,
            extra,
        );
        if got != case["result"] {
            record(
                out,
                label,
                show(&case.to_string()),
                show(&case["result"].to_string()),
                show(&got.to_string()),
            );
        }
    }
}

/// `str.lower()` against `str::to_lowercase`: final sigma, dotted capital I and the one-to-many cases.
fn check_lower(cases: &[Value], out: &mut Vec<Mismatch>) {
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let expected = case["lower"].as_str().unwrap();
        let got = text.to_lowercase();
        if got != expected {
            record(out, "lower", show(text), show(expected), show(&got));
        }
    }
}

/// The text of the error `json.loads` raised (or `None`) against what the checkpoint JSON
/// reader says for the same text.
fn check_json_errors(cases: &[Value], out: &mut Vec<Mismatch>) {
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let got = laya::laya_config::LayaConfig::parse(text)
            .err()
            .map(|e| e.to_string());
        let expected = case["error"].as_str().map(str::to_string);
        if got != expected {
            record(
                out,
                "json error",
                show(text),
                format!("{expected:?}"),
                format!("{got:?}"),
            );
        }
    }
}

#[test]
fn generated_corpus_matches_python() {
    let Some(root) = routing_golden::load("differential_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };
    let lang = root["lang"].as_array().unwrap();
    let routes = root["route"].as_array().unwrap();
    let clean = root["email_clean"].as_array().unwrap();
    let states = root["email_state"].as_array().unwrap();
    let budgets: Vec<i64> = root["email_budgets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    let mut mismatches = Vec::new();
    check_lang(lang, &mut mismatches);
    let after_lang = mismatches.len();
    check_routes(lang, routes, &mut mismatches);
    let after_routes = mismatches.len();
    check_email_clean(&budgets, clean, &mut mismatches);
    let after_clean = mismatches.len();
    check_email_state(states, &mut mismatches);
    let after_state = mismatches.len();
    check_lower(root["lower"].as_array().unwrap(), &mut mismatches);
    let after_lower = mismatches.len();
    check_json_errors(root["json_errors"].as_array().unwrap(), &mut mismatches);
    eprintln!(
        "differential: {} lower, {} json_errors; mismatches: lower {}, json_errors {}",
        root["lower"].as_array().unwrap().len(),
        root["json_errors"].as_array().unwrap().len(),
        after_lower - after_state,
        mismatches.len() - after_lower
    );

    eprintln!(
        "differential: {} lang, {} route, {} email_clean x {} budgets, {} email_state; mismatches: lang {}, route {}, email_clean {}, email_state {}",
        lang.len(),
        routes.len(),
        clean.len(),
        budgets.len(),
        states.len(),
        after_lang,
        after_routes - after_lang,
        after_clean - after_routes,
        mismatches.len() - after_clean
    );
    if !mismatches.is_empty() {
        let listing: String = mismatches
            .iter()
            .take(20)
            .map(|m| {
                format!(
                    "\n- {}\n    input:  {}\n    python: {}\n    rust:   {}",
                    m.label, m.input, m.python, m.rust
                )
            })
            .collect();
        panic!(
            "{} mismatches against Python; first {}:{listing}",
            mismatches.len(),
            mismatches.len().min(20)
        );
    }
}
