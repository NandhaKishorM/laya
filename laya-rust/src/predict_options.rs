//! Per-call options for `LayaEngine::predict_with_options` and `LayaRouter::predict_with_options`.
//!
//! Python's `system_one` and `Router.predict` take `min_confidence`, `lang`, `max_len`,
//! `head_max_len` and per-call hooks. All but the hooks are ported; [`PredictOptions`] is
//! `#[non_exhaustive]` so a field can be added without a signature change.

use crate::confidence::MinConfidence;

/// Options for one `predict` call. `Default` gives Python's defaults (nothing set). Build one with
/// the chained setters.
///
/// The per-call hooks, which Python's `system_one` also accepts, are not part of this port.
/// `predict_long` takes no `min_confidence`, as in Python.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct PredictOptions {
    /// Opt-in abstention threshold on `answer_confidence` (#361). With it set, every answer
    /// reports `Answer::abstention` and `Answer::abstention_threshold`, and `Answer::low_confidence`
    /// is true for those below it. `None` writes nothing, so an ungated call is unchanged.
    pub min_confidence: Option<MinConfidence>,
    /// Selects an entry of [`crate::LayaOptions::lang_temperatures`] for this call: the code is
    /// reduced to `code.split('-')[0].lower()` and, if the table has that language, its
    /// temperatures replace the checkpoint's. An unknown or empty code changes nothing. On
    /// `LayaRouter` the explicit `lang` argument plays this part (it also routes), falling back to
    /// the language the router detected, and this field is not read.
    pub lang: Option<String>,
    /// Override the checkpoint's `max_len` (total sequence tokens) for this call. `None` keeps it.
    /// Signed, because Python does not validate it: `0` and negative values flow into the same
    /// arithmetic and slices, and a question that then loses an option marker is
    /// [`crate::LayaError::NoMarkers`], naming the value as written.
    pub max_len: Option<i64>,
    /// Override the checkpoint's `head_max_len` (the question head's token budget) for this call.
    /// `None` keeps it. Signed like [`Self::max_len`].
    pub head_max_len: Option<i64>,
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

    /// Set [`Self::lang`].
    #[must_use]
    pub fn lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = Some(lang.into());
        self
    }

    /// Set [`Self::max_len`].
    #[must_use]
    pub fn max_len(mut self, max_len: i64) -> Self {
        self.max_len = Some(max_len);
        self
    }

    /// Set [`Self::head_max_len`].
    #[must_use]
    pub fn head_max_len(mut self, head_max_len: i64) -> Self {
        self.head_max_len = Some(head_max_len);
        self
    }
}
