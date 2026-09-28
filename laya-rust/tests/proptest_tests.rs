//! A few cheap property-based checks that complement the golden-vector and example-based tests
//! above. Kept deliberately small and cheap.

use laya::calibration::Calibration;
use laya::python_json::PythonJson;
use proptest::prelude::*;

proptest! {
    /// `round4` never moves a value further than half a unit in the 4th decimal place, and is
    /// idempotent (rounding an already-rounded value changes nothing).
    #[test]
    fn round4_is_idempotent_and_close_to_input(x in -1.0e6f64..1.0e6f64) {
        let once = Calibration::round4(x);
        let twice = Calibration::round4(once);
        prop_assert_eq!(once, twice);
        prop_assert!((once - x).abs() <= 5e-5 + 1e-9);
    }

    /// `repr` always round-trips back to the same `f64` through the standard library parser —
    /// it must, since it claims to be the *shortest* round-tripping decimal.
    #[test]
    fn repr_round_trips_through_std_parse(x in any::<f64>().prop_filter("finite only", |x| x.is_finite())) {
        let rendered = PythonJson::repr(x);
        // Python's exponent form ("1e+21") uses an explicit '+' that Rust's f64::from_str also
        // accepts, so no reformatting is needed before parsing it back.
        let parsed: f64 = rendered.parse().unwrap_or_else(|e| panic!("{rendered:?} did not parse: {e}"));
        if x == 0.0 {
            // +0.0 and -0.0 compare equal with `==`, but repr must still distinguish them in text.
            prop_assert_eq!(x.is_sign_negative(), parsed.is_sign_negative());
        } else {
            prop_assert_eq!(x, parsed);
        }
    }

    /// `confidence_from_probs` is always a valid confidence value for any probability-like input,
    /// regardless of how skewed or uniform the distribution is.
    #[test]
    fn confidence_from_probs_is_always_in_unit_range(
        a in 0.0f64..1.0, b in 0.0f64..1.0, c in 0.0f64..1.0
    ) {
        let sum = a + b + c;
        prop_assume!(sum > 0.0);
        let p = [a / sum, b / sum, c / sum];
        let confidence = Calibration::confidence_from_probs(&p, 3);
        prop_assert!((0.0..=1.0).contains(&confidence));
    }

    /// The expected score of a probability distribution over `0..n` never falls outside that
    /// range, regardless of the distribution's shape.
    #[test]
    fn expected_score_stays_within_the_level_range(
        a in 0.0f64..1.0, b in 0.0f64..1.0, c in 0.0f64..1.0, d in 0.0f64..1.0
    ) {
        let sum = a + b + c + d;
        prop_assume!(sum > 0.0);
        let p = [a / sum, b / sum, c / sum, d / sum];
        let score = Calibration::expected_score(&p);
        prop_assert!((0.0..=3.0).contains(&score));
    }
}
