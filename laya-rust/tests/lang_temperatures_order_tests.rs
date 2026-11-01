//! The order of `ONNXAgent.__init__`: the `lang_temperatures` table is resolved before the
//! checkpoint's own temperature warning, so a table's warnings come first and an invalid table
//! fails the load before that warning is logged. Records the `log` output of one process, so this
//! file holds a single test.

use laya::error::LayaError;
use laya::lang_temperatures::LangTemperature;
use laya::laya_config::LayaConfig;
use std::sync::Mutex;

static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
struct Capture;
impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        assert_eq!("laya", record.target());
        SEEN.lock().unwrap().push(record.args().to_string());
    }
    fn flush(&self) {}
}
static LOGGER: Capture = Capture;

fn take() -> Vec<String> {
    std::mem::take(&mut *SEEN.lock().unwrap())
}

#[test]
fn the_table_resolves_before_the_checkpoint_warning() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);

    // Both the checkpoint and the table ship an out-of-range value.
    let config = r#"{"temperature": [9.0, 1.0, 1.0]}"#;
    let table = vec![(
        "de".to_string(),
        Some(LangTemperature::new().temperature([1.0, 0.1, 1.0])),
    )];
    LayaConfig::parse_with_lang_temperatures(config, &table).unwrap();
    let seen = take();
    assert_eq!(2, seen.len(), "{seen:?}");
    assert!(
        seen[0].starts_with("laya: language override 'de' ships invalid"),
        "{seen:?}"
    );
    assert!(
        seen[1].starts_with("laya ONNX: this checkpoint ships temperatures"),
        "{seen:?}"
    );

    // An invalid table fails the load, and the checkpoint's warning is never logged.
    let bad = vec![(
        "fr".to_string(),
        Some(LangTemperature::new().temperature([1.0, 2.0])),
    )];
    match LayaConfig::parse_with_lang_temperatures(r#"{"temperature": [7.0, 1.0, 1.0]}"#, &bad) {
        Err(LayaError::InvalidConfig(text)) => assert_eq!(
            "Language override 'fr' temperature must be a list of 3 floats, got [1.0, 2.0]",
            text
        ),
        other => panic!("expected the ValueError, got {other:?}"),
    }
    assert!(
        take().is_empty(),
        "the checkpoint warning must not be logged"
    );

    // Python 0.4.20: a temperature too wide for a double is the neutral 1.0 and is named, digit for
    // digit, in the checkpoint's warning; an entry that inherits the list is not reported again,
    // and the load does not fail.
    let wide = format!("1{}", "0".repeat(400));
    let config = format!(r#"{{"temperature": [{wide}, 1.2, 1.1]}}"#);
    let inherit = vec![("de".to_string(), None)];
    let loaded = LayaConfig::parse_with_lang_temperatures(&config, &inherit).unwrap();
    assert_eq!(vec![1.0, 1.2, 1.1], loaded.temperature);
    let seen = take();
    assert_eq!(1, seen.len(), "{seen:?}");
    assert!(
        seen[0].contains(&format!("clamping temperature[0]={wide} -> 1.")),
        "{seen:?}"
    );
}
