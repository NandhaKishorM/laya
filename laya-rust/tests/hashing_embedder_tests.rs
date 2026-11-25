//! Tier 1: the deterministic hashing embedder (`tests/common/hashing_embedder.rs`) against every
//! entry of the routing golden's `hashing_embedder_probe.json`. This is a test-helper probe, not
//! a check on `laya` itself — the embedder is reimplemented per-SDK, per the plan, so the Rust
//! test helper must reproduce `tools/hashing_embedder.py` bit-for-bit.

mod common;

use common::{hashing_embedder, routing_golden};

#[test]
fn hashing_embedder_matches_every_probe_entry() {
    let Some(cases) = routing_golden::load_array("hashing_embedder_probe.json") else {
        eprintln!("skip: routing golden not found (tests/golden/routing)");
        return;
    };

    for case in &cases {
        let text = case["text"]
            .as_str()
            .expect("probe entry needs a 'text' string");
        let expected: Vec<f64> = case["vector"]
            .as_array()
            .expect("probe entry needs a 'vector' array")
            .iter()
            .map(|v| v.as_f64().expect("vector component must be a number"))
            .collect();

        let got = hashing_embedder::embed(&[text.to_string()]).expect("embed never fails");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].len(), expected.len(), "text {text:?}: vector length");
        for (i, (&g, &e)) in got[0].iter().zip(expected.iter()).enumerate() {
            assert_eq!(g, e, "text {text:?}: vector[{i}]");
        }
    }
    eprintln!(
        "hashing_embedder_matches_every_probe_entry: checked {} texts",
        cases.len()
    );
}

#[test]
fn hashing_embedder_is_order_preserving_and_batches_independently() {
    let a = hashing_embedder::embed(&["alpha".to_string()]).unwrap();
    let b = hashing_embedder::embed(&["beta".to_string(), "alpha".to_string()]).unwrap();
    // Each text is embedded independently of its neighbours in the batch.
    assert_eq!(a[0], b[1]);
}
