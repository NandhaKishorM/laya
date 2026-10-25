//! Per-call options for `LayaEngine::predict_with_options` and `LayaRouter::predict_with_options`.
//!
//! Python's `system_one` and `Router.predict` take `min_confidence`, `lang`, `max_len`,
//! `head_max_len` and per-call hooks. Only `min_confidence` is ported so far; [`PredictOptions`] is
//! `#[non_exhaustive]` so the others can be added as fields without a signature change.

use crate::confidence::MinConfidence;

/// Options for one `predict` call. `Default` gives Python's defaults (nothing set). Build one with
/// the chained setters.
///
/// `lang`, `max_len`, `head_max_len` and the per-call hooks, which Python's `system_one` also
/// accepts, are not part of this port yet. `predict_long` takes no `min_confidence`, as in Python.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct PredictOptions {
    /// Opt-in abstention threshold on `answer_confidence` (#361). With it set, every answer
    /// reports `Answer::abstention` and `Answer::abstention_threshold`, and `Answer::low_confidence`
    /// is true for those below it. `None` writes nothing, so an ungated call is unchanged.
    pub min_confidence: Option<MinConfidence>,
}

impl PredictOptions {
    /// Python's defaults: no gate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`Self::min_confidence`]. Build the value with [`MinConfidence::scalar`] or
    /// [`MinConfidence::map`].
    #[must_use]
    pub fn min_confidence(mut self, min_confidence: MinConfidence) -> Self {
        self.min_confidence = Some(min_confidence);
        self
    }
}
