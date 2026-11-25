//! Tier 1: language/script detection. Ports the language-detection checks from
//! `tests/test_router.py` (no model weights needed) plus every entry of the routing golden's
//! `lang_probe.json`.

mod common;

use common::routing_golden;
use laya::language_detection::LanguageDetection;
use serde_json::json;
use std::collections::HashMap;

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
}

fn profile_from_json(v: &serde_json::Value) -> HashMap<String, f64> {
    v.as_object()
        .expect("script_profile must be an object")
        .iter()
        .map(|(k, val)| {
            (
                k.clone(),
                val.as_f64().expect("profile value must be a number"),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------- named cases (test_router.py)

#[test]
fn detect_script_matches_named_cases() {
    let cases: [(&str, &str, &str); 17] = [
        (
            "english",
            "The customer was charged twice and wants a refund.",
            "latin",
        ),
        ("armenian", "Հայերեն", "armenian"),
        ("armenian uppercase", "ՀԱՅԵՐԵՆ", "armenian"),
        ("armenian punctuation only", "։֊", "unknown"),
        (
            "french",
            "Le client a été facturé deux fois et demande un remboursement.",
            "latin",
        ),
        (
            "hindi",
            "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।",
            "devanagari",
        ),
        (
            "japanese",
            "お客様は二重に請求されたため返金を希望しています。",
            "kana",
        ),
        ("chinese", "客户被重复扣款要求退款", "han"),
        ("korean", "고객이 두 번 청구되어 환불을 원합니다", "hangul"),
        (
            "arabic",
            "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال",
            "arabic",
        ),
        ("tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
        (
            "russian",
            "С клиента дважды сняли деньги и он хочет возврат",
            "cyrillic",
        ),
        ("thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
        (
            "greek",
            "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων",
            "greek",
        ),
        ("hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
        ("empty", "", "unknown"),
        ("digits only", "12345 6789", "unknown"),
    ];
    for (label, text, want) in cases {
        assert_eq!(
            LanguageDetection::detect_script(text),
            want,
            "script/{label}"
        );
    }
}

#[test]
fn is_english_matches_named_cases() {
    let cases: [(&str, &str, bool); 16] = [
        (
            "plain english",
            "Please refund the duplicate charge on invoice 4411 today.",
            true,
        ),
        ("armenian", "Հայերեն", false),
        ("english short", "refund me", true),
        ("hindi", "ग्राहक से दो बार शुल्क लिया गया", false),
        ("japanese", "お客様は二重に請求されました", false),
        ("russian", "С клиента дважды сняли деньги", false),
        (
            "french long",
            "Le client a été facturé deux fois et il demande un remboursement pour la \
             facture qui a été payée le mois dernier avec la carte de crédit",
            false,
        ),
        (
            "german long",
            "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die \
             Rechnung die nicht korrekt ist und auch nicht bezahlt wurde",
            false,
        ),
        (
            "romanian",
            "Gătește-mi o rețetă de sarmale de post pentru mâine.",
            false,
        ),
        (
            "romanian invoice",
            "Am fost taxat de două ori pentru factura din luna martie și vreau banii",
            false,
        ),
        (
            "polish",
            "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
            false,
        ),
        (
            "czech",
            "Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz",
            false,
        ),
        (
            "turkish",
            "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım",
            false,
        ),
        (
            "vietnamese",
            "Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay",
            false,
        ),
        (
            "english with loanwords",
            "We visited a cafe in Zurich and the naive assumption about the \
             invoice was wrong, so please refund the duplicate charge",
            true,
        ),
        ("empty", "", true),
    ];
    for (label, text, want) in cases {
        let v = json!(text);
        assert_eq!(
            LanguageDetection::is_english(&v),
            want,
            "is_english/{label}"
        );
    }
}

#[test]
fn latin_profile_undecided_reporting() {
    let undecided = json!("Müşteriden iki kez ücret alındı ve para iadesi istiyor");
    let a = LanguageDetection::analyse(&undecided);
    assert!(a.language_undecided, "latin/undecided is flagged");
    assert_eq!(a.language, None, "latin/undecided names no language");

    let english = json!("Please refund the duplicate charge on the invoice");
    assert!(
        !LanguageDetection::analyse(&english).language_undecided,
        "latin/english is not undecided"
    );

    let romanian = json!("Gătește-mi o rețetă de sarmale");
    assert!(
        LanguageDetection::analyse(&romanian).diacritic_rate > 0.02,
        "latin/diacritic rate reported"
    );

    let plain_english = json!("Please refund the duplicate charge today");
    approx(
        LanguageDetection::analyse(&plain_english).diacritic_rate,
        0.0,
    );

    // A 0-0 tie between non-English stopword lists is no evidence for any of them.
    assert_eq!(
        LanguageDetection::guess_latin_language("Cât e ora acum la Tokyo"),
        None,
        "latin_lang/zero tie invents nothing"
    );

    // Known limitation, kept visible on purpose (see laya/lang.py and issue #35): short Romanian
    // with no diacritics and an English function word still reads as English.
    assert!(LanguageDetection::is_english(&json!(
        "Care este ora in Tokyo?"
    )));
}

#[test]
fn guess_latin_language_matches_named_cases() {
    let cases: [(&str, &str, Option<&str>); 5] = [
        (
            "english",
            "The customer was charged twice and wants a refund for this invoice",
            Some("en"),
        ),
        (
            "french",
            "Le client a ete facture deux fois et il demande un remboursement pour la facture",
            Some("fr"),
        ),
        (
            "german",
            "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung",
            Some("de"),
        ),
        (
            "spanish",
            "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura",
            Some("es"),
        ),
        ("too short", "refund", None),
    ];
    for (label, text, want) in cases {
        assert_eq!(
            LanguageDetection::guess_latin_language(text).as_deref(),
            want,
            "latin_lang/{label}"
        );
    }
    assert_eq!(
        LanguageDetection::guess_latin_language(
            "Please refund the duplicate charge on invoice 4411 today because \
             we have been waiting for three days and nobody has replied to us"
        )
        .as_deref(),
        Some("en"),
        "latin_lang/long english stays en"
    );
}

#[test]
fn state_text_spends_the_budget_like_python() {
    // Values from Python lang.state_text: a leaf that ends exactly at the budget gets no
    // trailing space; a leaf that crosses it is cut.
    let exact = json!(["a".repeat(3999), "b"]);
    assert_eq!(
        LanguageDetection::state_text_default(&exact),
        "a".repeat(3999)
    );
    let crossing = json!(["a".repeat(3998), "bb"]);
    assert_eq!(
        LanguageDetection::state_text_default(&crossing),
        format!("{} b", "a".repeat(3998))
    );
    let long_leaf = json!({"x": "a".repeat(4000), "y": "b"});
    assert_eq!(
        LanguageDetection::state_text_default(&long_leaf),
        "a".repeat(4000)
    );
}

#[test]
fn state_text_flattens_nested_state() {
    let dict_state = json!({"body": "charged twice", "n": 3});
    assert!(LanguageDetection::state_text_default(&dict_state).contains("charged twice"));

    let nested = json!({"a": {"b": ["deep"]}});
    assert!(LanguageDetection::state_text_default(&nested).contains("deep"));

    let list_state = json!(["x", {"y": "z"}]);
    assert!(LanguageDetection::state_text_default(&list_state).contains("x"));

    assert_eq!(LanguageDetection::state_text_default(&json!(null)), "");

    // Keys must not drive detection: English keys around Hindi content stay non-English.
    let keyed = json!({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"});
    assert!(
        !LanguageDetection::is_english(&keyed),
        "state_text/keys ignored"
    );
}

#[test]
fn script_profile_named_cases() {
    let armenian = LanguageDetection::analyse(&json!("Հայերեն"));
    assert_eq!(armenian.script_profile.len(), 1);
    approx(*armenian.script_profile.get("armenian").unwrap(), 1.0);

    let mixed = LanguageDetection::analyse(&json!("Հայերեն abc"));
    approx(mixed.non_latin_fraction, 0.7);
}

#[test]
fn plain_ascii_romance_named_cases() {
    // #172: ASCII-stripped Romance text carries no diacritic rate, so the function-word lists are
    // the only evidence left.
    let cases: [(&str, &str); 10] = [
        (
            "es",
            "El pedido llego roto y nadie responde cuando escribo al soporte",
        ),
        ("es", "Quiero cancelar mi plan y pedir un reembolso"),
        ("es", "La factura tiene un error en el importe total"),
        (
            "es",
            "Necesito que me devuelvan el dinero de la compra duplicada",
        ),
        (
            "it",
            "Il cliente e stato addebitato due volte e vuole un rimborso",
        ),
        (
            "it",
            "Voglio cancellare il mio abbonamento e chiedere un rimborso",
        ),
        ("it", "La fattura contiene un errore nell importo totale"),
        (
            "pt",
            "O cliente foi cobrado duas vezes e quer o dinheiro de volta",
        ),
        (
            "fr",
            "Le client a ete facture deux fois et demande un remboursement",
        ),
        (
            "fr",
            "Je ne peux pas acceder a mon compte et j ai besoin d aide",
        ),
    ];
    for (lang, text) in cases {
        assert_eq!(
            LanguageDetection::guess_latin_language(text).as_deref(),
            Some(lang),
            "latin_lang/plain ascii {lang} {text}"
        );
        assert!(
            !LanguageDetection::is_english(&json!(text)),
            "is_english/plain ascii {lang} {text}"
        );
    }

    for text in [
        "The customer was charged twice and wants a refund for this invoice",
        "Please cancel my subscription and refund the duplicate charge today",
        "The report by Smith et al. shows the de facto standard, e.g. the LA office and Rio",
        "Our MI5 and UN contacts discussed the DOS attack in LA last month",
        "No refund was issued, so I am writing to you again about invoice 4411",
        "no refund no reply",
        "The son of the director filed a complaint about the duplicate invoice",
    ] {
        assert!(
            LanguageDetection::is_english(&json!(text)),
            "is_english/romance control {text}"
        );
    }

    // A word several lists claim (`la`, `e`, `o`) says "not English" without saying *which*
    // language, so it may not name one on its own.
    assert_eq!(
        LanguageDetection::guess_latin_language("Cât e ora acum la Tokyo"),
        None,
        "latin_lang/shared words alone name nothing"
    );
    // ...but a distinctive word in the same state is enough to name the language it belongs to.
    assert_eq!(
        LanguageDetection::guess_latin_language(
            "La fattura contiene un errore nell importo totale"
        )
        .as_deref(),
        Some("it")
    );
    assert_eq!(
        LanguageDetection::guess_latin_language("La factura tiene un error en el importe total")
            .as_deref(),
        Some("es")
    );
}

// ---------------------------------------------------------------------- lang_probe.json

#[test]
fn lang_probe_matches_every_entry() {
    let Some(cases) = routing_golden::load_array("lang_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };

    for case in &cases {
        let label = case["label"].as_str().unwrap();
        let state = &case["state"];

        let expected_state_text = case["state_text"].as_str().unwrap();
        assert_eq!(
            LanguageDetection::state_text_default(state),
            expected_state_text,
            "{label}: state_text"
        );

        let text = LanguageDetection::state_text_default(state);
        assert_eq!(
            LanguageDetection::detect_script(&text),
            case["detect_script"].as_str().unwrap(),
            "{label}: detect_script"
        );

        let expected_profile = profile_from_json(&case["script_profile"]);
        let actual_profile = LanguageDetection::script_profile(&text);
        assert_eq!(
            actual_profile.len(),
            expected_profile.len(),
            "{label}: script_profile key count"
        );
        for (k, v) in &expected_profile {
            let got = actual_profile
                .get(k)
                .unwrap_or_else(|| panic!("{label}: script_profile missing key {k}"));
            approx(*got, *v);
        }

        let lp = &case["latin_profile"];
        let actual_lp = LanguageDetection::latin_profile(&text);
        assert_eq!(
            actual_lp.language.as_deref(),
            lp["language"].as_str(),
            "{label}: latin_profile.language"
        );
        assert_eq!(
            actual_lp.english_hits,
            lp["english_hits"].as_i64().unwrap(),
            "{label}: latin_profile.english_hits"
        );
        approx(
            actual_lp.diacritic_rate,
            lp["diacritic_rate"].as_f64().unwrap(),
        );
        assert_eq!(
            actual_lp.looks_non_english,
            lp["looks_non_english"].as_bool().unwrap(),
            "{label}: latin_profile.looks_non_english"
        );

        let a = &case["analyse"];
        let actual = LanguageDetection::analyse(state);
        assert_eq!(
            actual.script,
            a["script"].as_str().unwrap(),
            "{label}: analyse.script"
        );
        assert_eq!(
            actual.language.as_deref(),
            a["language"].as_str(),
            "{label}: analyse.language"
        );
        assert_eq!(
            actual.is_english,
            a["is_english"].as_bool().unwrap(),
            "{label}: analyse.is_english"
        );
        assert_eq!(
            actual.language_undecided,
            a["language_undecided"].as_bool().unwrap(),
            "{label}: analyse.language_undecided"
        );
        approx(actual.diacritic_rate, a["diacritic_rate"].as_f64().unwrap());
        approx(
            actual.non_latin_fraction,
            a["non_latin_fraction"].as_f64().unwrap(),
        );
        assert_eq!(
            actual.mixed_segment.as_deref(),
            a["mixed_segment"].as_str(),
            "{label}: analyse.mixed_segment"
        );
    }
    eprintln!(
        "lang_probe_matches_every_entry: checked {} cases",
        cases.len()
    );
}

// ---------------------------------------------------------------------- 0.4.11 behaviours

#[test]
fn mixed_segment_is_set_only_by_the_segment_scan() {
    let state = json!(
        "Please review the error and send me the details.\nEu nao consigo entrar na minha conta."
    );
    let det = LanguageDetection::analyse(&state);
    assert!(!det.is_english);
    assert_eq!(det.language.as_deref(), Some("pt"));
    assert_eq!(
        det.mixed_segment.as_deref(),
        Some("Eu nao consigo entrar na minha conta.")
    );
    // a value flipped by the per-value read keeps `mixed_segment` unset, as Python does
    let flipped = json!({
        "note": "Please review the error and send me the details. ".repeat(5),
        "x": "žluťoučký kůň úpěl ďábelské ódy",
    });
    let det = LanguageDetection::analyse(&flipped);
    assert!(!det.is_english);
    assert!(det.mixed_segment.is_none());
}

#[test]
fn shouted_acronym_lines_are_not_named_foreign() {
    for text in [
        "MON DES EST LA",
        "STORES LOS ANGELES LAS VEGAS EL PASO CLOSED",
    ] {
        let det = LanguageDetection::analyse(&json!(text));
        assert!(
            det.is_english,
            "{text}: an acronym line is not a foreign sentence"
        );
        assert!(det.language_undecided, "{text}");
    }
    // ...while shouted prose with a long matched stopword, or diacritics, still is
    for (text, lang) in [
        ("EU NAO CONSIGO ENTRAR NA MINHA CONTA", Some("pt")),
        ("ICH HABE MEIN PAKET NOCH NICHT BEKOMMEN", Some("de")),
    ] {
        let det = LanguageDetection::analyse(&json!(text));
        assert!(!det.is_english, "{text}");
        assert_eq!(det.language.as_deref(), lang, "{text}");
    }
    assert!(!LanguageDetection::is_english(&json!(
        "WIE SPÄT IST ES IN KÖLN"
    )));
}

#[test]
fn nordic_swedish_and_english_rescue_rules() {
    assert_eq!(
        LanguageDetection::guess_latin_language("glömt lösenord").as_deref(),
        Some("sv")
    );
    assert_eq!(
        LanguageDetection::guess_latin_language("kan inte logga in").as_deref(),
        Some("sv")
    );
    // a lone Swedish-Danish overlap word has no language but still is not English
    let overlap = LanguageDetection::latin_profile("hej min vän");
    assert_eq!(overlap.language, None);
    assert!(overlap.looks_non_english);
    // one accented loanword does not pull plain English off the English checkpoint
    assert!(LanguageDetection::is_english(&json!(
        "Please send me the café receipt today."
    )));
    assert!(!LanguageDetection::is_english(&json!(
        "ñandú ñandú ñandú the please"
    )));
}

#[test]
fn latin_cutoff_counts_ipa_and_fullwidth_as_latin_and_unclaimed_letters_as_other() {
    assert_eq!(LanguageDetection::detect_script("ＡＢＣＤ"), "latin");
    assert_eq!(LanguageDetection::detect_script("ɐɪɲ"), "latin");
    // modifier letters (U+02B0 and up) are alphabetic but claimed by no range
    assert_eq!(LanguageDetection::detect_script("\u{2b0}\u{2b2}"), "other");
    assert_eq!(LanguageDetection::detect_script("ㄅㄆㄇ"), "other");
    let profile = LanguageDetection::script_profile("ab ㄅㄆ");
    assert!((profile["latin"] - 0.5).abs() < 1e-12);
    assert!((profile["other"] - 0.5).abs() < 1e-12);
}
