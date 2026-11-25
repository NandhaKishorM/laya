//! The clamped-temperature warning of `ONNXAgent.__init__`, sent to the `log` facade with
//! Python's text. The expected string was produced by Python's own `clamp_temperature`,
//! `temperature_rejected` and the warning's format string on the same config.

use laya::laya_config::LayaConfig;
use std::sync::Mutex;

static SEEN: Mutex<Vec<(String, log::Level, String)>> = Mutex::new(Vec::new());
struct Capture;
impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        SEEN.lock().unwrap().push((
            record.target().to_string(),
            record.level(),
            record.args().to_string(),
        ));
    }
    fn flush(&self) {}
}
static LOGGER: Capture = Capture;

#[test]
fn clamped_temperatures_warn_with_pythons_text_and_accepted_ones_do_not() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);

    LayaConfig::parse(
        r#"{"temperature": [1.0, 2.5, 5.0],
            "temperature_by_options": {"choice:2": 0.5, "noul:2": 2.5}}"#,
    )
    .unwrap();
    assert!(
        SEEN.lock().unwrap().is_empty(),
        "accepted temperatures must not warn"
    );

    let config = LayaConfig::parse(
        r#"{"temperature": [1.0, 9, true, null],
            "temperature_by_options": {"score:3-5": "x", "choice:2": 0.2, "noul:2": 2.5,
                                       "choice:11+": 7.25, "a": 1e-7, "b": [1, "q"]}}"#,
    )
    .unwrap();
    assert!(!config.clamped_temperatures.is_empty());

    let seen = SEEN.lock().unwrap();
    assert_eq!(1, seen.len());
    assert_eq!("laya", seen[0].0);
    assert_eq!(log::Level::Warn, seen[0].1);
    assert_eq!(
        "laya ONNX: this checkpoint ships temperatures outside [0.5, 5] which would distort \
         confidence; clamping score:3-5='x' -> 1, choice:2=0.2 -> 0.5, choice:11+=7.25 -> 5, \
         a=1e-07 -> 0.5, b=[1, 'q'] -> 1, temperature[1]=9 -> 5, temperature[2]=True -> 1, \
         temperature[3]=None -> 1. Treat confidence from the affected buckets as uncalibrated.",
        seen[0].2
    );
}
