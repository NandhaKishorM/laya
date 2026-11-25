//! JSON floats must be Python's `float()` bit for bit. Without serde_json's `float_roundtrip`
//! feature its fast path is off by one ulp on long literals (5,542 of 20,000 random 17..30-digit
//! literals differed); these three are among them. Expected bits come from Python's `float()`.

#[test]
fn json_floats_equal_python_float_bit_for_bit() {
    for (literal, bits) in [
        ("4.5807302157368193036426212e8", 0x41bb4da3bd92dcd2_u64),
        ("0.642792877574476216e14", 0x42cd3b180e7d43d0),
        ("4.529078454748554565275582823e10", 0x422517150e46f899),
    ] {
        let value: serde_json::Value = serde_json::from_str(literal).unwrap();
        assert_eq!(bits, value.as_f64().unwrap().to_bits(), "{literal}");
    }
}
