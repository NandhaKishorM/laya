//! `bytes`/`bytearray` states: Python's `lang._iter_text` decodes them as strict UTF-8 and reads
//! invalid UTF-8 as no text (`lang.py:253-255`), so `state_text`, `analyse` and `Router.route`
//! accept them. Expectations were produced by running those Python functions on the same bytes.

use laya::error::Result;
use laya::language_detection::LanguageDetection;
use laya::laya_options::LayaCheckpoint;
use laya::laya_router::{LayaPredictor, LayaRouter, LayaRouterOptions};
use std::sync::Arc;

const GERMAN: &[u8] = b"ich habe mein paket noch nicht";
const INVALID: &[u8] = b"caf\xc3\xa9 \xff ok";

#[test]
fn state_text_decodes_valid_utf8_and_reads_invalid_as_empty() {
    assert_eq!(
        "hola mundo",
        LanguageDetection::state_text_bytes(b"hola mundo", 4000)
    );
    assert_eq!(
        "hola",
        LanguageDetection::state_text_bytes(b"hola mundo", 4)
    );
    assert_eq!(
        "\u{4f60}\u{597d}\u{4e16}\u{754c}",
        LanguageDetection::state_text_bytes("\u{4f60}\u{597d}\u{4e16}\u{754c}".as_bytes(), 4000)
    );
    assert_eq!("", LanguageDetection::state_text_bytes(INVALID, 4000));
    assert_eq!("", LanguageDetection::state_text_bytes(b"\xff", 4000));
    assert_eq!("", LanguageDetection::state_text_bytes(b"", 4000));
}

#[test]
fn analyse_bytes_matches_python() {
    let valid = LanguageDetection::analyse_bytes(b"please send the receipt");
    assert_eq!(("latin", Some("en"), true, false), summary(&valid));

    let german = LanguageDetection::analyse_bytes(GERMAN);
    assert_eq!(("latin", Some("de"), false, false), summary(&german));

    let han = LanguageDetection::analyse_bytes("\u{4f60}\u{597d}\u{4e16}\u{754c}".as_bytes());
    assert_eq!(("han", None, false, true), summary(&han));

    let undecided = LanguageDetection::analyse_bytes(b"hola mundo");
    assert_eq!(("latin", None, true, true), summary(&undecided));

    for empty_like in [INVALID, b"\xff".as_slice(), b"".as_slice()] {
        let analysis = LanguageDetection::analyse_bytes(empty_like);
        assert_eq!(("unknown", None, true, true), summary(&analysis));
        assert!(LanguageDetection::is_english_bytes(empty_like));
    }
    assert!(!LanguageDetection::is_english_bytes(GERMAN));
}

fn summary(a: &laya::language_detection::LanguageAnalysis) -> (&str, Option<&str>, bool, bool) {
    (
        a.script.as_str(),
        a.language.as_deref(),
        a.is_english,
        a.language_undecided,
    )
}

struct Never;
impl LayaPredictor for Never {
    fn predict(
        &self,
        _state: serde_json::Value,
        _questions: &laya::questions::QuestionSet,
    ) -> Result<laya::answers::LayaResult> {
        unreachable!("routing only")
    }
}

fn router() -> LayaRouter {
    LayaRouter::with_factory(LayaRouterOptions::default(), |_, _| {
        Ok(Arc::new(Never) as Arc<dyn LayaPredictor>)
    })
    .unwrap()
}

#[test]
fn route_bytes_matches_python() {
    let r = router();
    let cases: [(&[u8], LayaCheckpoint, &str); 5] = [
        (
            b"please send the receipt",
            LayaCheckpoint::English,
            "English Latin text",
        ),
        (
            GERMAN,
            LayaCheckpoint::Multilingual,
            "Latin script but language looks like 'de', not English",
        ),
        (
            "\u{4f60}\u{597d}\u{4e16}\u{754c}".as_bytes(),
            LayaCheckpoint::Multilingual,
            "non-Latin script (han, 100% of letters); the English checkpoint cannot read it",
        ),
        (
            INVALID,
            LayaCheckpoint::Multilingual,
            "no letters detected in state; using default (multilingual)",
        ),
        (
            b"",
            LayaCheckpoint::Multilingual,
            "no letters detected in state; using default (multilingual)",
        ),
    ];
    for (bytes, model, reason) in cases {
        let decision = r.route_bytes(bytes, None, None, None, None, None).unwrap();
        assert_eq!(model, decision.model, "{bytes:?}");
        assert_eq!(reason, decision.reason, "{bytes:?}");
    }
    let hola = r
        .route_bytes(b"hola mundo", None, None, None, None, None)
        .unwrap();
    assert_eq!(LayaCheckpoint::Multilingual, hola.model);
    assert_eq!(
        "Latin script, language not identified and no non-English letters; using default (multilingual)",
        hola.reason
    );
}

#[test]
fn lang_guess_callback_receives_the_original_bytes_as_python_does() {
    use laya::laya_router::{GuessInput, LangGuess};
    use std::sync::Mutex;

    // Python: `Router(lang_guess=cb).route(b'caf\xc3\xa9 \xff')` calls `cb(b'caf\xc3\xa9 \xff')`,
    // invalid UTF-8 included; a str or dict state reaches `cb` as that str or dict.
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let log = seen.clone();
    let hint = LangGuess::callback(move |state| {
        log.lock().unwrap().push(match state {
            GuessInput::Bytes(b) => format!("bytes:{b:?}"),
            GuessInput::Value(v) => format!("value:{v}"),
        });
        None
    });
    let r = router().with_lang_guess(hint.clone());
    r.route_bytes(INVALID, None, None, None, None, None)
        .unwrap();
    r.route_bytes(b"hola", None, None, None, None, Some(&hint))
        .unwrap();
    r.route(&serde_json::json!("txt"), None, None, None, None)
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(
        vec![
            format!("bytes:{INVALID:?}"),
            "bytes:[104, 111, 108, 97]".to_string(),
            "bytes:[104, 111, 108, 97]".to_string(),
            "value:\"txt\"".to_string(),
        ],
        *seen
    );
}
