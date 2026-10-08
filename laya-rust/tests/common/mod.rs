//! Shared test helpers. Not every test binary in `tests/` uses every item here, so dead-code
//! analysis is turned off for the whole module rather than annotating individual functions.
#![allow(dead_code)]

pub mod engines;
pub mod golden_data;
pub mod hashing_embedder;
pub mod result_compare;
pub mod routing_golden;
pub mod stub_tokenizer;
pub mod test_artifacts;
