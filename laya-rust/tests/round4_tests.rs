//! `Calibration::round4` against Python's builtin `round(x, 4)`, bit for bit.
//!
//! The table below and `round4_python_cases.txt` (written by `tools/gen_round4_cases.py`) hold
//! Python's results as IEEE-754 bit patterns, so `-0.0` and one-ulp differences are caught.

use laya::calibration::Calibration;

/// `(input bits, Python's output bits)`; the comment gives `repr(input) -> repr(output)`.
const TABLE: &[(u64, u64)] = &[
    (0x3f7999999999999a, 0x3f79ce075f6fd220), // 0.00625 -> 0.0063
    (0x3fa0000000000000, 0x3f9ff2e48e8a71de), // 0.03125 -> 0.0312
    (0x3fb8000000000000, 0x3fb80346dc5d6388), // 0.09375 -> 0.0938
    (0xbfa0000000000000, 0xbf9ff2e48e8a71de), // -0.03125 -> -0.0312
    (0xbfb8000000000000, 0xbfb80346dc5d6388), // -0.09375 -> -0.0938
    (0x3fc4000000000000, 0x3fc3fe5c91d14e3c), // 0.15625 -> 0.1562
    (0x3fde000000000000, 0x3fde00d1b71758e2), // 0.46875 -> 0.4688
    (0x3f90000000000000, 0x3f8ff2e48e8a71de), // 0.015625 -> 0.0156
    (0x3fa8000000000000, 0x3fa80346dc5d6388), // 0.046875 -> 0.0469
    (0x3fb4000000000000, 0x3fb3fe5c91d14e3c), // 0.078125 -> 0.0781
    (0xbfbc000000000000, 0xbfbc01a36e2eb1c4), // -0.109375 -> -0.1094
    (0x3f80000000000000, 0x3f7ff2e48e8a71de), // 0.0078125 -> 0.0078
    (0x3f98000000000000, 0x3f97f62b6ae7d567), // 0.0234375 -> 0.0234
    (0xbfa4000000000000, 0xbfa404ea4a8c154d), // -0.0390625 -> -0.0391
    (0x3f8999999999999a, 0x3f8999999999999a), // 0.0125 -> 0.0125
    (0x3faccccccccccccd, 0x3facd35a858793de), // 0.05625 -> 0.0563
    (0x3fb999999999999a, 0x3fb999999999999a), // 0.1 -> 0.1
    (0x4000001a36e2eb1c, 0x4000000000000000), // 2.00005 -> 2.0
    (0x4023fff972474539, 0x4024000000000000), // 9.99995 -> 10.0
    (0x4008400000000000, 0x40083fe5c91d14e4), // 3.03125 -> 3.0312
    (0x3ee4f8b588e368f1, 0x0000000000000000), // 1e-05 -> 0.0
    (0x3f0a36c08f0472df, 0x0000000000000000), // 4.9999e-05 -> 0.0
    (0x3f0a36e2eb1c432d, 0x3f1a36e2eb1c432d), // 5e-05 -> 0.0001
    (0xbf0a36e2eb1c432d, 0xbf1a36e2eb1c432d), // -5e-05 -> -0.0001
    (0xbee4f8b588e368f1, 0x8000000000000000), // -1e-05 -> -0.0
    (0x01a56e1fc2f8f359, 0x0000000000000000), // 1e-300 -> 0.0
    (0x81a56e1fc2f8f359, 0x8000000000000000), // -1e-300 -> -0.0
    (0x0000000000000001, 0x0000000000000000), // 5e-324 -> 0.0
    (0x8000000000000001, 0x8000000000000000), // -5e-324 -> -0.0
    (0x8000000000000000, 0x8000000000000000), // -0.0 -> -0.0
    (0x0000000000000000, 0x0000000000000000), // 0.0 -> 0.0
    (0x430c6bf526340002, 0x430c6bf526340002), // 1000000000000000.2 -> 1000000000000000.2
    (0x423cbe991a14587e, 0x423cbe991a145880), // 123456789012.34567 -> 123456789012.3457
    (0x4330000000000000, 0x4330000000000000), // 4503599627370496.0 -> 4503599627370496.0
    (0x7e37e43c8800759c, 0x7e37e43c8800759c), // 1e+300 -> 1e+300
    (0x3fefff972474538f, 0x3ff0000000000000), // 0.99995 -> 1.0
    (0x3f23a92a30553261, 0x3f1a36e2eb1c432d), // 0.00015 -> 0.0001
    (0x3f36f0068db8bac7, 0x3f33a92a30553261), // 0.00035 -> 0.0003
];

#[test]
fn round4_matches_python_table() {
    for &(input, expected) in TABLE {
        let x = f64::from_bits(input);
        let got = Calibration::round4(x);
        assert_eq!(
            expected,
            got.to_bits(),
            "round4({x:?}): expected {:?}, got {got:?}",
            f64::from_bits(expected)
        );
    }
}

#[test]
fn round4_leaves_nan_and_infinities_unchanged() {
    assert!(Calibration::round4(f64::NAN).is_nan());
    assert_eq!(f64::INFINITY, Calibration::round4(f64::INFINITY));
    assert_eq!(f64::NEG_INFINITY, Calibration::round4(f64::NEG_INFINITY));
}

#[test]
fn round4_matches_python_on_20000_seeded_values() {
    let text = include_str!("round4_python_cases.txt");
    let mut checked = 0;
    for line in text.lines() {
        let mut parts = line.split(' ');
        let input = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
        let expected = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
        let x = f64::from_bits(input);
        let got = Calibration::round4(x);
        assert_eq!(
            expected,
            got.to_bits(),
            "round4({x:?}): expected {:?}, got {got:?}",
            f64::from_bits(expected)
        );
        checked += 1;
    }
    assert_eq!(20_000, checked);
}
