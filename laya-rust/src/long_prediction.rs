//! The window plan behind `LayaEngine::predict_long`: how wide the scan windows are, where they
//! start, how many go through one forward pass, and how the per-window answers collapse into one.
//! Ports `window_budget`, `window_batch_cap` and `state_room` from `laya/common.py` and the
//! aggregation in `ONNXAgent.predict_long` (`laya/onnx_agent.py`), `aggregate="auto"` only.
//!
//! Everything here is pure arithmetic over token counts and finished results, so it is testable
//! without a model. The engine supplies the tokenizer-dependent inputs (the per-question state
//! room and the state's token ids) and runs the windows.
//!
//! Python also emits `warnings.warn` when it caps the window or reduces the stride. Those warnings
//! are ported with Python's exact text and sent through the `log` facade at `warn` level, target
//! `laya`; install any `log` backend (for example `env_logger`) to see them.

use crate::answers::{Answer, LayaResult, OptionUsage, ScanUsage, StateUsage, Usage, WindowInfo};
use crate::error::{LayaError, Result};
use crate::questions::{QuestionSet, QuestionType};
use crate::sequence_builder::SequenceBuilder;
use crate::tokenization::LayaTokenizer;
use indexmap::IndexMap;
use std::collections::HashMap;

/// Options for `LayaEngine::predict_long`. Every field is optional; `Default` gives Python's
/// defaults. Build one with the chained setters.
///
/// Per-call hooks and the confidence gate that Python's `predict_long` also accepts are not part
/// of this port. It takes no `max_len` or `head_max_len`, as in Python.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LongOptions {
    /// State tokens per window. `None` or `0` uses the checkpoint default,
    /// `max(64, max_len - head_max_len - 8)`. Capped at the smallest state room the questions
    /// leave.
    pub window: Option<usize>,
    /// Token step between windows. `None` or `0` uses half the effective window. A step past the
    /// requested window is an error; a valid step past a window the questions' room clamped is
    /// reduced.
    pub stride: Option<usize>,
    /// Only `"auto"` (the default when `None`) is supported.
    pub aggregate: Option<String>,
    /// Maximum windows per forward pass. `None` or `0` uses Python's automatic cap.
    pub batch_size: Option<usize>,
    /// Selects an entry of [`crate::LayaOptions::lang_temperatures`] for every window; see
    /// [`crate::PredictOptions::lang`]. On `LayaRouter::predict_long` the explicit `lang`
    /// argument plays this part, falling back to the detected language, and this field is not
    /// read.
    pub lang: Option<String>,
}

impl LongOptions {
    /// Python's defaults: checkpoint window, half-window stride, `"auto"`, automatic batch cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`Self::window`].
    pub fn window(mut self, window: usize) -> Self {
        self.window = Some(window);
        self
    }

    /// Set [`Self::stride`].
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = Some(stride);
        self
    }

    /// Set [`Self::aggregate`]. Anything but `"auto"` makes `predict_long` return
    /// [`LayaError::UnsupportedAggregate`].
    pub fn aggregate(mut self, aggregate: impl Into<String>) -> Self {
        self.aggregate = Some(aggregate.into());
        self
    }

    /// Set [`Self::batch_size`].
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set [`Self::lang`].
    pub fn lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = Some(lang.into());
        self
    }

    pub(crate) fn check_aggregate(&self) -> Result<()> {
        match self.aggregate.as_deref() {
            None | Some("auto") => Ok(()),
            Some(other) => Err(LayaError::UnsupportedAggregate(other.to_string())),
        }
    }
}

/// The resolved scan geometry, plus the room that decides the one-pass shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowBudget {
    /// Tokens per window, after clamping to `room`.
    pub window: usize,
    /// Tokens between window starts.
    pub stride: usize,
    /// The smallest state room any question leaves.
    pub room: usize,
}

/// The checkpoint's default window: `max(64, max_len - head_max_len - 8)`, the 64 being a floor.
pub(crate) fn default_window(max_len: usize, head_max_len: usize) -> usize {
    max_len
        .saturating_sub(head_max_len)
        .saturating_sub(8)
        .max(64)
}

/// `window_budget` with the per-question rooms already measured. An empty `rooms` (no questions)
/// leaves the requested window as the room. The `RuntimeWarning`s Python raises are dropped.
#[cfg(test)]
pub(crate) fn plan_budget(
    rooms: &[usize],
    max_len: usize,
    head_max_len: usize,
    window: usize,
    stride: usize,
) -> Result<WindowBudget> {
    plan_budget_warned(rooms, max_len, head_max_len, window, stride).map(|(budget, _)| budget)
}

/// [`plan_budget`], also returning the text of every `warnings.warn` Python's `window_budget`
/// would emit for the same call, in order, exactly as Python words it.
pub(crate) fn plan_budget_warned(
    rooms: &[usize],
    max_len: usize,
    head_max_len: usize,
    window: usize,
    stride: usize,
) -> Result<(WindowBudget, Vec<String>)> {
    let mut warnings = Vec::new();
    let requested = if window > 0 {
        window
    } else {
        default_window(max_len, head_max_len)
    };
    let room = rooms.iter().copied().min().unwrap_or(requested);
    if room == 0 {
        return Err(LayaError::InvalidWindow(format!(
            "predict_long: the questions' options fill the whole sequence (max_len={max_len}, \
             head_max_len={head_max_len}), leaving no room for the state; no window can carry any \
             of it. A label set this large is what laya.shortlist.predict_shortlist is for"
        )));
    }
    let size = requested.min(room);
    if requested > room {
        if window > 0 {
            warnings.push(format!(
                "laya: predict_long: window={requested} is wider than the {room} state tokens \
                    these questions leave inside max_len={max_len}, so every window would be \
                    truncated to {room} on the way to the model; scanning with window={room} instead"
            ));
        } else if requested >= room.saturating_mul(WINDOW_CLAMP_WARN_RATIO) {
            warnings.push(format!(
                "laya: predict_long: these questions leave only {room} of max_len={max_len} for \
                    the state (their heads take the rest), so the scan window is capped {requested} \
                    -> {room} and roughly {:.1}x as many windows -- each a full forward pass -- are \
                    needed to read the document. Fewer or shorter options, a larger max_len, or \
                    laya.shortlist.predict_shortlist for a large label set will all cost less than \
                    scanning at this width",
                requested as f64 / room as f64
            ));
        }
    }
    let mut step = if stride > 0 {
        stride
    } else {
        (size / 2).max(1)
    };
    if step > size {
        if size < requested && stride <= requested {
            let reduced = (size / 2).max(1);
            warnings.push(format!(
                "laya: predict_long: stride={step} was a 50% step for the window={requested} you \
                    asked for, but the window was reduced to {size} to fit the room these questions \
                    leave; scanning with stride={reduced} instead"
            ));
            step = reduced;
        } else {
            return Err(LayaError::InvalidWindow(format!(
                "predict_long: stride={step} steps past the {size}-token window{}, so {} tokens \
                 between every pair of windows would be read by no window at all; pass stride <= \
                 {size}",
                if size < requested {
                    " these questions leave room for"
                } else {
                    ""
                },
                step - size
            )));
        }
    }
    Ok((
        WindowBudget {
            window: size,
            stride: step,
            room,
        },
        warnings,
    ))
}

/// Python's `_WINDOW_CLAMP_WARN_RATIO`: a default window cut to half its size or less is warned.
const WINDOW_CLAMP_WARN_RATIO: usize = 2;

/// Sends each warning to the `log` facade (`warn`, target `laya`).
///
/// Python's default warnings filter shows an identical warning once per location, so a text that
/// has already been emitted in this process is not emitted again.
///
/// Python keys that on the message text and the file and line of the frame the warning is
/// attributed to (`stacklevel`): the caller's line, not the library's. `at` is that place, taken
/// from `#[track_caller]` on the public entry points. A call made through the router comes from
/// the router's own fixed line, as it does in Python.
pub(crate) fn emit_warnings(warnings: &[String], at: &std::panic::Location<'_>) {
    type Shown = std::collections::HashSet<(String, String, u32)>;
    static SHOWN: std::sync::Mutex<Option<Shown>> = std::sync::Mutex::new(None);
    for message in warnings {
        let first = SHOWN
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .insert((message.clone(), at.file().to_string(), at.line()));
        if first {
            log::warn!(target: "laya", "{message}");
        }
    }
}

/// `window_budget`: measures each question's state room, then plans.
///
/// Public only so the parity tests, a separate crate, can compare the effective window, stride and
/// room with Python's; hidden from the docs like `LayaOutput`.
#[doc(hidden)]
pub fn window_budget(
    tokenizer: &dyn LayaTokenizer,
    questions: &QuestionSet,
    max_len: usize,
    head_max_len: usize,
    window: usize,
    stride: usize,
) -> Result<WindowBudget> {
    window_budget_warned(tokenizer, questions, max_len, head_max_len, window, stride)
        .map(|(budget, _)| budget)
}

/// [`window_budget`] for the engine: the budget plus Python's warnings, not yet emitted.
pub(crate) fn window_budget_warned(
    tokenizer: &dyn LayaTokenizer,
    questions: &QuestionSet,
    max_len: usize,
    head_max_len: usize,
    window: usize,
    stride: usize,
) -> Result<(WindowBudget, Vec<String>)> {
    let mut rooms = Vec::with_capacity(questions.len());
    for id in questions.ids() {
        rooms.push(SequenceBuilder::state_room(
            tokenizer,
            &questions[id],
            max_len,
            head_max_len,
        )?);
    }
    plan_budget_warned(&rooms, max_len, head_max_len, window, stride)
}

/// The overlapping windows over `tokens` state tokens. A window that reaches the end is the last.
pub(crate) fn spans(tokens: usize, window: usize, stride: usize) -> Vec<WindowInfo> {
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < tokens {
        bounds.push((start, (start + window).min(tokens)));
        if start + window >= tokens {
            break;
        }
        start += stride;
    }
    let count = bounds.len();
    bounds
        .into_iter()
        .enumerate()
        .map(|(index, (token_start, token_end))| WindowInfo {
            index,
            token_start,
            token_end,
            count,
        })
        .collect()
}

/// `window_batch_cap`: windows per forward pass, or `None` for all of them in one pass.
pub(crate) fn batch_cap(
    count: usize,
    window: usize,
    config_budget: usize,
    batch_size: usize,
) -> Option<usize> {
    if batch_size > 0 {
        return Some(batch_size);
    }
    if window >= config_budget {
        return None;
    }
    let step = (window / 2).max(1);
    let config_step = (config_budget / 2).max(1);
    let unclamped = (count * step).div_ceil(config_step).max(1);
    // Capping that multiplies the window count by 2 or less keeps the shared pass.
    (count > unclamped * 2).then_some(unclamped)
}

/// Collapse per-window results into one: for each question the window with the greatest evidence
/// (P(true) for noul, `answer_confidence` for choice and score) wins, the first window on ties.
pub(crate) fn aggregate(
    model: &str,
    questions: &QuestionSet,
    results: &[LayaResult],
    spans: &[WindowInfo],
) -> LayaResult {
    let ids: Vec<String> = questions.ids().map(str::to_string).collect();
    let mut answers: HashMap<String, Answer> = HashMap::with_capacity(ids.len());
    for id in &ids {
        let evidence = |index: usize| -> f64 {
            let answer = &results[index][id.as_str()];
            match (questions[id.as_str()].question_type(), answer.as_noul()) {
                (QuestionType::Noul, Some(noul)) => noul.probability,
                _ => answer.answer_confidence(),
            }
        };
        let mut best = 0;
        for index in 1..results.len() {
            if evidence(index) > evidence(best) {
                best = index;
            }
        }
        answers.insert(
            id.clone(),
            results[best][id.as_str()].clone().with_window(spans[best]),
        );
    }

    let usage = Usage {
        input_tokens: results.iter().map(|r| r.usage().input_tokens).sum(),
        output_tokens: 0,
    };
    let result = LayaResult::new(model, ids, answers, usage);
    let diagnostics: Vec<&StateUsage> = results.iter().filter_map(|r| r.state_usage()).collect();
    let truncated_windows = diagnostics.iter().filter(|u| u.truncated).count();
    let scan = ScanUsage {
        windows: results.len(),
        truncated_windows,
    };
    if diagnostics.is_empty() {
        return result.with_scan_usage(scan);
    }
    let mut options: IndexMap<String, OptionUsage> = IndexMap::new();
    for usage in &diagnostics {
        for (id, stats) in &usage.options {
            options.insert(id.clone(), *stats);
        }
    }
    result
        .with_state_usage(StateUsage {
            state_tokens: diagnostics.iter().map(|u| u.state_tokens).sum(),
            state_tokens_dropped: diagnostics.iter().map(|u| u.state_tokens_dropped).sum(),
            truncated: truncated_windows > 0,
            // The last window's list, as in Python's scan aggregation.
            truncated_questions: results
                .last()
                .and_then(|r| r.state_usage())
                .map(|u| u.truncated_questions.clone())
                .unwrap_or_default(),
            options,
        })
        .with_scan_usage(scan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answers::{ActionInfo, ChoiceAnswer, NoulAnswer, ScoreAnswer};
    use crate::questions::Question;
    use serde_json::Value;

    fn budget(rooms: &[usize], window: usize, stride: usize) -> Result<WindowBudget> {
        plan_budget(rooms, 512, 192, window, stride)
    }

    fn message(result: Result<WindowBudget>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn default_window_has_a_64_token_floor() {
        assert_eq!(default_window(512, 192), 312);
        assert_eq!(default_window(128, 120), 64);
        assert_eq!(default_window(64, 192), 64);
    }

    #[test]
    fn budget_defaults_to_the_checkpoint_window_and_half_stride() {
        let b = budget(&[478, 400], 0, 0).unwrap();
        assert_eq!((b.window, b.stride, b.room), (312, 156, 400));
    }

    #[test]
    fn budget_without_questions_uses_the_requested_window_as_room() {
        let b = budget(&[], 32, 0).unwrap();
        assert_eq!((b.window, b.stride, b.room), (32, 16, 32));
    }

    #[test]
    fn budget_clamps_the_window_to_the_room() {
        let b = budget(&[100], 4096, 0).unwrap();
        assert_eq!((b.window, b.stride, b.room), (100, 50, 100));
        let b = budget(&[100], 0, 0).unwrap();
        assert_eq!((b.window, b.stride), (100, 50));
    }

    #[test]
    fn budget_keeps_a_stride_inside_the_clamped_window() {
        let b = budget(&[100], 4096, 60).unwrap();
        assert_eq!(b.stride, 60);
    }

    #[test]
    fn budget_rejects_a_stride_past_the_window() {
        assert_eq!(
            message(budget(&[478], 32, 33)),
            "predict_long: stride=33 steps past the 32-token window, so 1 tokens between every \
             pair of windows would be read by no window at all; pass stride <= 32"
        );
    }

    #[test]
    fn budget_rejects_a_stride_past_the_requested_window_even_when_clamped() {
        assert_eq!(
            message(budget(&[100], 120, 130)),
            "predict_long: stride=130 steps past the 100-token window these questions leave room \
             for, so 30 tokens between every pair of windows would be read by no window at all; \
             pass stride <= 100"
        );
    }

    #[test]
    fn budget_reduces_a_stride_past_the_clamped_window_but_within_the_request() {
        let b = budget(&[100], 4096, 2048).unwrap();
        assert_eq!((b.window, b.stride), (100, 50));
    }

    #[test]
    fn budget_refuses_when_the_questions_leave_no_room() {
        assert_eq!(
            message(budget(&[300, 0], 0, 0)),
            "predict_long: the questions' options fill the whole sequence (max_len=512, \
             head_max_len=192), leaving no room for the state; no window can carry any of it. A \
             label set this large is what laya.shortlist.predict_shortlist is for"
        );
    }

    #[test]
    fn spans_overlap_and_stop_at_the_window_that_reaches_the_end() {
        let s = spans(100, 40, 20);
        let bounds: Vec<_> = s.iter().map(|w| (w.token_start, w.token_end)).collect();
        assert_eq!(bounds, [(0, 40), (20, 60), (40, 80), (60, 100)]);
        assert!(s.iter().all(|w| w.count == 4));
        assert_eq!(s[3].index, 3);
    }

    #[test]
    fn spans_clip_the_last_window_and_handle_an_exact_fit() {
        let s = spans(50, 40, 30);
        let bounds: Vec<_> = s.iter().map(|w| (w.token_start, w.token_end)).collect();
        assert_eq!(bounds, [(0, 40), (30, 50)]);
        assert_eq!(spans(40, 40, 20).len(), 1);
        assert!(spans(0, 40, 20).is_empty());
    }

    #[test]
    fn batch_cap_honours_an_explicit_size() {
        assert_eq!(batch_cap(10, 312, 312, 3), Some(3));
        assert_eq!(batch_cap(10, 23, 312, 3), Some(3));
    }

    #[test]
    fn batch_cap_is_off_unless_the_window_was_clamped_hard() {
        assert_eq!(batch_cap(29, 312, 312, 0), None);
        // A 64-token budget reduced to 43: under twice the windows, a mild cap.
        assert_eq!(batch_cap(20, 43, 64, 0), None);
        // 23 against 312: windows multiplied about 13x, bounded at the un-capped count.
        assert_eq!(batch_cap(413, 23, 312, 0), Some(30));
    }

    fn noul(p: f64) -> Answer {
        Answer::Noul(NoulAnswer::new(p, p, p, ActionInfo::default()))
    }

    fn choice(c: f64) -> Answer {
        let mut p = IndexMap::new();
        p.insert("a".to_string(), c);
        Answer::Choice(ChoiceAnswer::new("a", p, c, c, ActionInfo::default()))
    }

    fn window_result(
        noul_answer: Answer,
        choice_answer: Answer,
        input_tokens: u32,
        dropped: usize,
    ) -> LayaResult {
        let mut answers = HashMap::new();
        answers.insert("flag".to_string(), noul_answer);
        answers.insert("topic".to_string(), choice_answer);
        LayaResult::new(
            "m",
            vec!["flag".to_string(), "topic".to_string()],
            answers,
            Usage {
                input_tokens,
                output_tokens: 0,
            },
        )
        .with_state_usage(StateUsage {
            state_tokens: 40,
            state_tokens_dropped: dropped,
            truncated: dropped > 0,
            truncated_questions: if dropped > 0 {
                vec!["flag".to_string()]
            } else {
                Vec::new()
            },
            options: IndexMap::new(),
        })
    }

    fn question_set() -> QuestionSet {
        QuestionSet::new()
            .with("flag", Question::noul("flag?", Value::Null, Value::Null))
            .unwrap()
            .with(
                "topic",
                Question::choice_labels("topic?", ["a", "b"]).unwrap(),
            )
            .unwrap()
    }

    #[test]
    fn aggregate_takes_max_probability_for_noul_and_max_confidence_for_choice() {
        let results = vec![
            window_result(noul(0.2), choice(0.6), 10, 0),
            window_result(noul(0.9), choice(0.5), 11, 3),
            window_result(noul(0.4), choice(0.7), 12, 0),
        ];
        let all = spans(100, 40, 30);
        let out = aggregate("m", &question_set(), &results, &all[..3]);
        assert_eq!(out["flag"].window().unwrap().index, 1);
        assert_eq!(out["topic"].window().unwrap().index, 2);
        assert_eq!(out["flag"].as_noul().unwrap().probability, 0.9);
        assert_eq!(out.usage().input_tokens, 33);
        let state = out.state_usage().unwrap();
        assert_eq!(state.state_tokens, 120);
        assert_eq!(state.state_tokens_dropped, 3);
        assert!(state.truncated);
        // The last window's list, which lost nothing.
        assert!(state.truncated_questions.is_empty());
        assert_eq!(
            out.scan_usage(),
            Some(ScanUsage {
                windows: 3,
                truncated_windows: 1
            })
        );
    }

    #[test]
    fn aggregate_prefers_the_first_window_on_ties() {
        let results = vec![
            window_result(noul(0.7), choice(0.6), 10, 0),
            window_result(noul(0.7), choice(0.6), 10, 0),
        ];
        let all = spans(60, 40, 20);
        let out = aggregate("m", &question_set(), &results, &all);
        assert_eq!(out["flag"].window().unwrap().index, 0);
        assert_eq!(out["topic"].window().unwrap().index, 0);
    }

    #[test]
    fn aggregate_merges_options_with_later_windows_winning() {
        let usage = |distinct| OptionUsage {
            total: 3,
            distinct,
            tokens_per_option: Some(4),
        };
        let with_option = |distinct| {
            let result = window_result(noul(0.1), choice(0.5), 1, 0);
            let mut state = result.state_usage().unwrap().clone();
            state.options.insert("topic".into(), usage(distinct));
            result.with_state_usage(state)
        };
        let all = spans(60, 40, 20);
        let out = aggregate(
            "m",
            &question_set(),
            &[with_option(1), with_option(2)],
            &all,
        );
        assert_eq!(out.state_usage().unwrap().options["topic"], usage(2));
    }

    #[test]
    fn aggregate_without_questions_still_counts_windows() {
        let empty = || LayaResult::new("m", Vec::new(), HashMap::new(), Usage::default());
        let all = spans(60, 40, 20);
        let out = aggregate("m", &QuestionSet::new(), &[empty(), empty()], &all);
        assert!(out.is_empty());
        assert!(out.state_usage().is_none());
        assert_eq!(out.scan_usage().unwrap().windows, 2);
    }

    #[test]
    fn default_budget_without_questions_keeps_the_64_floor_for_a_widened_head() {
        let b = plan_budget(&[], 200, 192, 0, 0).unwrap();
        assert_eq!((b.window, b.stride, b.room), (64, 32, 64));
        let b = plan_budget(&[], 512, 192, 0, 0).unwrap();
        assert_eq!((b.window, b.stride, b.room), (312, 156, 312));
        // Nothing clamps the window, so a stride past it is the caller's error.
        let error = message(plan_budget(&[], 512, 192, 40, 41));
        assert!(
            error.contains("stride=41 steps past the 40-token"),
            "{error}"
        );
        assert!(!error.contains("these questions leave room for"), "{error}");
    }

    #[test]
    fn budget_room_may_exceed_the_default_window_and_an_explicit_one() {
        let b = budget(&[478], 0, 0).unwrap();
        assert_eq!((b.window, b.room), (312, 478));
        let b = budget(&[478], 100, 0).unwrap();
        assert_eq!((b.window, b.room), (100, 478));
    }

    #[test]
    fn budget_room_equals_the_window_when_the_room_caps_it() {
        for window in [0, 1000] {
            let b = budget(&[120, 200], window, 0).unwrap();
            assert_eq!((b.window, b.room), (120, 120));
        }
    }

    #[test]
    fn budget_stride_equal_to_the_requested_window_is_reduced_after_the_clamp() {
        let b = budget(&[100], 1000, 1000).unwrap();
        assert_eq!((b.window, b.stride, b.room), (100, 50, 100));
        let error = message(budget(&[100], 1000, 1001));
        assert!(error.contains("these questions leave room for"), "{error}");
    }

    #[test]
    fn budget_floors_the_default_stride_at_one_token() {
        let b = budget(&[], 1, 0).unwrap();
        assert_eq!((b.window, b.stride), (1, 1));
    }

    #[test]
    fn budget_with_zero_state_room_names_the_cause() {
        let error = message(plan_budget(&[0], 10, 192, 0, 0));
        assert!(
            error.starts_with(
                "predict_long: the questions' options fill the whole sequence (max_len=10, \
                 head_max_len=192), leaving no room for the state"
            ),
            "{error}"
        );
    }

    #[test]
    fn spans_cover_every_token_and_stop_at_the_first_complete_window() {
        // (tokens, window, stride, count, last start, last end)
        let cases = [
            (64, 64, 32, 1, 0, 64),
            (65, 64, 32, 2, 32, 65),
            (128, 64, 32, 3, 64, 128),
            (129, 64, 32, 4, 96, 129),
            (129, 64, 64, 3, 128, 129),
            (5, 1, 1, 5, 4, 5),
        ];
        for (tokens, window, stride, count, last_start, last_end) in cases {
            let all = spans(tokens, window, stride);
            assert_eq!(all.len(), count, "{tokens}/{window}/{stride}");
            let last = all[count - 1];
            assert_eq!(
                (last.index, last.token_start, last.token_end, last.count),
                (count - 1, last_start, last_end, count)
            );
            assert_eq!(all[0].token_start, 0);
            for (i, w) in all.iter().enumerate() {
                assert_eq!((w.index, w.count), (i, count));
                if i > 0 {
                    assert!(w.token_start <= all[i - 1].token_end);
                }
            }
        }
    }

    #[test]
    fn spans_of_a_short_state_are_one_span_and_end_at_the_length() {
        let one = spans(10, 64, 32);
        assert_eq!(one.len(), 1);
        assert_eq!(
            (
                one[0].index,
                one[0].token_start,
                one[0].token_end,
                one[0].count
            ),
            (0, 0, 10, 1)
        );
        let two = spans(100, 64, 48);
        let bounds: Vec<_> = two.iter().map(|w| (w.token_start, w.token_end)).collect();
        assert_eq!(bounds, [(0, 64), (48, 100)]);
        assert!(two.iter().all(|w| w.count == 2));
    }

    #[test]
    fn batch_cap_edge_cases_match_python() {
        // (count, window, budget, batch_size, expected)
        let cases = [
            (100, 312, 312, 7, Some(7)), // explicit size, even when the window is not capped
            (100, 50, 1, 0, None),       // window >= budget; budget / 2 floors at 1
            (1, 1, 312, 0, None),
            (2, 50, 312, 0, None), // n == 2 * unclamped keeps the single pass
            (3, 50, 312, 0, Some(1)), // one past it chunks at unclamped
            (12, 50, 312, 0, Some(2)), // ceil(300 / 156) = 2; 12 > 4
            (34, 50, 312, 0, Some(6)), // ceil(850 / 156) = 6; 34 > 12
            (5, 200, 312, 0, None), // ceil(500 / 156) = 4; 5 <= 8
            (100, 1, 312, 0, Some(1)), // window / 2 floors at 1
            (100, 312, 312, 0, None),
            (100, 200, 312, 0, None),
            (100, 50, 312, 0, Some(17)),
            (100, 50, 312, 3, Some(3)),
        ];
        for (count, window, budget, batch, expected) in cases {
            assert_eq!(
                batch_cap(count, window, budget, batch),
                expected,
                "{count}/{window}/{budget}/{batch}"
            );
        }
    }

    fn score(level: f64, c: f64) -> Answer {
        Answer::Score(ScoreAnswer::new(
            level,
            vec![Value::from("low"), Value::from("high")],
            vec![1.0 - c, c],
            0.1,
            c,
            ActionInfo::default(),
        ))
    }

    fn three_questions() -> QuestionSet {
        question_set()
            .with(
                "urgency",
                Question::score("urgent?", ["low", "high"]).unwrap(),
            )
            .unwrap()
    }

    fn triple(noul_answer: Answer, c: f64, s: f64, input: u32) -> LayaResult {
        let mut answers = HashMap::new();
        answers.insert("flag".to_string(), noul_answer);
        answers.insert("topic".to_string(), choice(c));
        answers.insert("urgency".to_string(), score(s, s));
        LayaResult::new(
            "m",
            vec!["flag".into(), "topic".into(), "urgency".into()],
            answers,
            Usage {
                input_tokens: input,
                output_tokens: 7,
            },
        )
    }

    #[test]
    fn aggregate_selects_evidence_not_entropy() {
        // Window 1's noul has the lower answer_confidence but the higher P(true): it must win.
        let low_p = Answer::Noul(NoulAnswer::new(0.05, 0.95, 0.95, ActionInfo::default()));
        let high_p = Answer::Noul(NoulAnswer::new(0.7, 0.7, 0.7, ActionInfo::default()));
        let all = spans(65, 64, 32);
        let out = aggregate(
            "m",
            &three_questions(),
            &[triple(low_p, 0.8, 0.6, 20), triple(high_p, 0.8, 0.9, 30)],
            &all,
        );
        assert_eq!(out["flag"].window(), Some(all[1]));
        assert_eq!(out["topic"].window(), Some(all[0])); // tie
        assert_eq!(out["urgency"].window(), Some(all[1]));
        assert_eq!(out["urgency"].as_score().unwrap().score, 0.9);
        assert_eq!(out.usage().input_tokens, 50);
        assert_eq!(out.usage().output_tokens, 0);
    }

    #[test]
    fn aggregate_ties_go_to_the_first_window_for_all_three_types() {
        let tie = || triple(noul(0.7), 0.8, 0.6, 20);
        let all = spans(65, 64, 32);
        let out = aggregate("m", &three_questions(), &[tie(), tie()], &all);
        for id in ["flag", "topic", "urgency"] {
            assert_eq!(out[id].window(), Some(all[0]), "{id}");
        }
        assert_eq!(out.scan_usage().unwrap().windows, 2);
    }

    #[test]
    fn aggregate_of_one_window_without_state_usage_leaves_the_fields_none() {
        let all = spans(10, 64, 32);
        let out = aggregate(
            "m",
            &three_questions(),
            &[triple(noul(0.7), 0.8, 0.6, 20)],
            &all,
        );
        let w = out["flag"].window().unwrap();
        assert_eq!(
            (w.index, w.token_start, w.token_end, w.count),
            (0, 0, 10, 1)
        );
        assert_eq!(out.usage().input_tokens, 20);
        assert_eq!(out.usage().output_tokens, 0);
        assert!(out.state_usage().is_none());
        assert_eq!(
            out.scan_usage(),
            Some(ScanUsage {
                windows: 1,
                truncated_windows: 0
            })
        );
    }

    #[test]
    fn aggregate_carries_the_last_windows_truncated_questions() {
        let truncated = |input, names: &[&str]| {
            triple(noul(0.7), 0.8, 0.6, input).with_state_usage(StateUsage {
                state_tokens: 10,
                state_tokens_dropped: 1,
                truncated: true,
                truncated_questions: names.iter().map(|n| n.to_string()).collect(),
                options: IndexMap::new(),
            })
        };
        let all = spans(65, 64, 32);
        let out = aggregate(
            "m",
            &three_questions(),
            &[truncated(20, &["urgency"]), truncated(30, &[])],
            &all,
        );
        let state = out.state_usage().unwrap();
        assert!(state.truncated_questions.is_empty());
        assert_eq!(state.state_tokens_dropped, 2);
        assert_eq!(out.scan_usage().unwrap().truncated_windows, 2);
    }

    #[test]
    fn long_options_default_to_python_and_check_aggregate() {
        let options = LongOptions::new();
        assert_eq!(
            (options.window, options.stride, options.batch_size),
            (None, None, None)
        );
        assert!(options.check_aggregate().is_ok());
        assert!(
            LongOptions::new()
                .aggregate("auto")
                .check_aggregate()
                .is_ok()
        );
        let error = LongOptions::new()
            .aggregate("max")
            .check_aggregate()
            .unwrap_err();
        assert!(
            matches!(error, LayaError::UnsupportedAggregate(_)),
            "{error:?}"
        );
    }

    const WARN_CAPPED_DEFAULT: &str = "laya: predict_long: these questions leave only 102 of \
        max_len=512 for the state (their heads take the rest), so the scan window is capped 312 \
        -> 102 and roughly 3.1x as many windows -- each a full forward pass -- are needed to read \
        the document. Fewer or shorter options, a larger max_len, or \
        laya.shortlist.predict_shortlist for a large label set will all cost less than scanning \
        at this width";
    const WARN_WINDOW: &str = "laya: predict_long: window=100 is wider than the 32 state tokens \
        these questions leave inside max_len=512, so every window would be truncated to 32 on the \
        way to the model; scanning with window=32 instead";
    const WARN_STRIDE: &str = "laya: predict_long: stride=64 was a 50% step for the window=100 \
        you asked for, but the window was reduced to 32 to fit the room these questions leave; \
        scanning with stride=16 instead";

    fn warned(rooms: &[usize], window: usize, stride: usize) -> Vec<String> {
        plan_budget_warned(rooms, 512, 192, window, stride)
            .unwrap()
            .1
    }

    // Expected texts and which calls warn were produced by running Python's
    // `common.window_budget` with `warnings.catch_warnings(record=True)`.
    #[test]
    fn warns_with_pythons_text_when_the_default_window_is_capped_hard() {
        assert_eq!(vec![WARN_CAPPED_DEFAULT], warned(&[102], 0, 0));
        assert!(warned(&[150], 0, 0)[0].contains("capped 312 -> 150 and roughly 2.1x as many"));
    }

    #[test]
    fn stays_silent_when_the_default_window_is_capped_mildly_or_not_at_all() {
        assert!(warned(&[200], 0, 0).is_empty());
        assert!(warned(&[400], 0, 0).is_empty());
        assert!(warned(&[312], 0, 0).is_empty());
    }

    // 312 / 96 = 3.25 is an exact tie: Python's `%.1f` rounds it half to even and prints 3.2.
    #[test]
    fn window_capped_ratio_tie_prints_as_python_does() {
        let text = warned(&[96], 0, 0).remove(0);
        assert!(
            text.contains("capped 312 -> 96 and roughly 3.2x as many windows"),
            "{text}"
        );
    }

    #[test]
    fn warns_with_pythons_text_when_an_explicit_window_is_capped() {
        assert_eq!(vec![WARN_WINDOW], warned(&[32], 100, 0));
        assert_eq!(2, warned(&[100], 300, 150).len());
        // A stride that already fits the capped window adds nothing.
        assert_eq!(1, warned(&[100], 300, 40).len());
    }

    #[test]
    fn warns_with_pythons_text_when_the_stride_is_reduced() {
        assert_eq!(vec![WARN_WINDOW, WARN_STRIDE], warned(&[32], 100, 64));
        assert_eq!(
            "laya: predict_long: stride=150 was a 50% step for the window=300 you asked for, but \
                the window was reduced to 100 to fit the room these questions leave; scanning with \
                stride=50 instead",
            warned(&[100], 300, 150)[1]
        );
    }

    #[test]
    fn emit_warnings_reaches_the_log_facade_at_warn_level_under_target_laya() {
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
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Warn);

        let at = std::panic::Location::caller();
        emit_warnings(&[WARN_WINDOW.to_string(), WARN_STRIDE.to_string()], at);
        // Python shows an identical warning once per process; a repeat is dropped.
        emit_warnings(&[WARN_WINDOW.to_string()], at);

        let seen = SEEN.lock().unwrap();
        let ours: Vec<_> = seen.iter().filter(|(t, ..)| t == "laya").collect();
        assert_eq!(2, ours.len());
        assert_eq!(log::Level::Warn, ours[0].1);
        assert_eq!(WARN_WINDOW, ours[0].2);
        assert_eq!(WARN_STRIDE, ours[1].2);
    }
}
