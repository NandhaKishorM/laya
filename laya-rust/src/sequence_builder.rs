//! Builds the typed token sequence one question is answered from. Ports `build_sequence`,
//! `render_options` and `render_criterion` from `laya/common.py`, step by step.
//!
//! The layout is `[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
//! state [SEP]`. The scorer reads the hidden state at each `[MASK]`, so a marker position that is
//! off by one token attaches an option's probability to the wrong place in the sequence and the
//! model answers a different question without anything failing. Every offset here is therefore
//! load-bearing.

use crate::error::Result;
use crate::python_json::PythonJson;
use crate::questions::Question;
use crate::tokenization::LayaTokenizer;
use serde_json::Value;

/// Namespace for sequence construction. See the module docs for the source this ports.
pub struct SequenceBuilder;

#[derive(Debug, Clone, Copy)]
pub(crate) struct StateTokenStats {
    pub tokens: usize,
    pub dropped: usize,
    pub options: Option<crate::answers::OptionUsage>,
}

impl SequenceBuilder {
    /// One option fragment is capped at this many tokens, before the marker is prepended.
    const MAX_OPTION_TOKENS: usize = 48;

    /// The head is never squeezed below this, even when the options have eaten the budget.
    const MIN_HEAD_TOKENS: usize = 8;

    /// The Python type name, which appears verbatim in the prompt text.
    pub fn type_name(qtype: crate::questions::QuestionType) -> &'static str {
        use crate::questions::QuestionType;
        match qtype {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }

    /// The instruction text as it reaches the tokenizer: a string verbatim, anything else through
    /// the `json.dumps` dialect that keeps non-ASCII characters literal (`ensure_ascii=False`).
    /// Ports `Agent._to_internal` (laya 0.3.21; 0.3.6 escaped non-ASCII here instead).
    pub fn render_instructions(question: &Question) -> String {
        match question.instructions() {
            Value::String(s) => s.clone(),
            other => PythonJson::instructions(other),
        }
    }

    /// The option texts in label-index order. Index N is the option the Nth logit scores, so this
    /// order is what binds a probability to a label.
    pub fn render_options(question: &Question) -> Vec<String> {
        match question {
            Question::Choice(c) => c
                .options()
                .iter()
                .map(|(label, description)| {
                    if Self::is_blank_criterion(description) {
                        label.clone()
                    } else {
                        format!("{label}: {}", PythonJson::criterion(description))
                    }
                })
                .collect(),
            Question::Score(s) => s
                .levels()
                .iter()
                .enumerate()
                .map(|(i, level)| format!("level {i}: {}", PythonJson::criterion(level)))
                .collect(),
            Question::Noul(n) => {
                let false_text = if Self::is_blank_criterion(n.if_false()) {
                    "no, the statement does not hold".to_string()
                } else {
                    PythonJson::criterion(n.if_false())
                };
                let true_text = if Self::is_blank_criterion(n.if_true()) {
                    "yes, the statement holds".to_string()
                } else {
                    PythonJson::criterion(n.if_true())
                };
                vec![format!("false: {false_text}"), format!("true: {true_text}")]
            }
        }
    }

    /// Whether a criterion value means "no description". Python tests `v is None or v == ""`, so
    /// only null and the empty string qualify: `0` and `false` are real descriptions and get
    /// rendered.
    fn is_blank_criterion(value: &Value) -> bool {
        matches!(value, Value::Null) || matches!(value, Value::String(s) if s.is_empty())
    }

    /// A literal mask token anywhere in user input would forge a marker the scorer then reads as
    /// an option, so every fragment has it replaced with a space before encoding.
    fn scrub(text: &str, mask_token: &str) -> String {
        if mask_token.is_empty() {
            text.to_string()
        } else {
            text.replace(mask_token, " ")
        }
    }

    /// Python's `//`: floor division, rounding toward negative infinity. `b` is always `>= 1` at
    /// every call site here, so this is equivalent to (and implemented with) `div_euclid`.
    fn floor_div(a: i64, b: i64) -> i64 {
        a.div_euclid(b)
    }

    /// Build the token sequence and marker positions for one question.
    ///
    /// `tokenizer` must encode without special tokens. `state` is the value being judged;
    /// `max_len` is the total sequence budget; `head_max_len` is the budget for instructions plus
    /// option fragments; `truncate_left` keeps the end of an over-long state rather than its
    /// start. `LayaEngine::predict_value` sets this to `true` exactly when `state` is a JSON
    /// array, matching `Agent._encode_state`'s `truncate_left = isinstance(state, list)` (laya
    /// 0.3.21): a conversation-turn list is serialized newest-last, so right-truncating it would
    /// drop the newest turn instead of the oldest one.
    ///
    /// Returns the token ids and one marker position per option. Markers landing at or past
    /// `max_len` are dropped — a short marker list means options did not fit, which the caller
    /// must treat as an error rather than answering a truncated question; this function itself
    /// never errors on that condition, matching `SequenceBuilder.Build` exactly.
    pub fn build(
        tokenizer: &dyn LayaTokenizer,
        state: &Value,
        question: &Question,
        max_len: usize,
        head_max_len: usize,
        truncate_left: bool,
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        Self::build_with_stats(tokenizer, state, question, max_len, head_max_len, truncate_left)
            .map(|(ids, markers, _)| (ids, markers))
    }

    pub(crate) fn build_with_stats(
        tokenizer: &dyn LayaTokenizer,
        state: &Value,
        question: &Question,
        max_len: usize,
        head_max_len: usize,
        truncate_left: bool,
    ) -> Result<(Vec<u32>, Vec<usize>, StateTokenStats)> {
        let special = tokenizer.special();
        let mask_token = special.mask_token.as_str();

        let options = Self::render_options(question);
        let instructions = Self::scrub(&Self::render_instructions(question), mask_token);
        let header_text = format!(
            "{} question: {instructions}",
            Self::type_name(question.question_type())
        );
        let mut header_ids: Vec<u32> = tokenizer.encode(&header_text)?;

        let mut option_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
        for option in &options {
            let encoded = tokenizer.encode(&format!(" {}", Self::scrub(option, mask_token)))?;
            let take = Self::MAX_OPTION_TOKENS.min(encoded.len());
            let mut fragment = Vec::with_capacity(take + 1);
            fragment.push(special.mask_id);
            fragment.extend_from_slice(&encoded[..take]);
            option_ids.push(fragment);
        }

        let option_len_sum =
            |opts: &[Vec<u32>]| -> i64 { opts.iter().map(|o| o.len() as i64).sum() };

        let mut opt_budget = head_max_len as i64 - option_len_sum(&option_ids);
        let mut tokens_per_option = None;
        if opt_budget < 16 {
            let per = 4i64.max(Self::floor_div(
                head_max_len as i64 - 16,
                1i64.max(option_ids.len() as i64),
            )) as usize;
            tokens_per_option = Some(per);
            for fragment in &mut option_ids {
                if fragment.len() > per {
                    fragment.truncate(per);
                }
            }
            opt_budget = head_max_len as i64 - option_len_sum(&option_ids);
        }

        let head_take = (Self::MIN_HEAD_TOKENS as i64).max(opt_budget) as usize;
        if header_ids.len() > head_take {
            header_ids.truncate(head_take);
        }

        let mut ids: Vec<u32> = Vec::with_capacity(max_len);
        ids.push(special.cls_id);
        ids.extend_from_slice(&header_ids);
        ids.push(special.sep_id);

        let mut markers = Vec::with_capacity(option_ids.len());
        for fragment in &option_ids {
            markers.push(ids.len());
            ids.extend_from_slice(fragment);
        }
        ids.push(special.sep_id);

        // -1 reserves the final SEP that closes the whole sequence, appended below.
        let room = (max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
        let state_text = Self::scrub(&PythonJson::state(state), mask_token);
        let state_ids = tokenizer.encode(&state_text)?;
        let sliced: &[u32] = if truncate_left {
            // Python's `state_ids[max(0, len(state_ids) - room):]` (laya 0.3.21): newest tokens
            // kept, and with no room left none of the state (0.3.6's `st[-room:]` kept all of
            // it). `LayaEngine::predict_value` reaches this branch for a JSON-array state (a
            // conversation-turn list).
            &state_ids[state_ids.len().saturating_sub(room)..]
        } else {
            &state_ids[..room.min(state_ids.len())]
        };
        ids.extend_from_slice(sliced);
        ids.push(special.sep_id);

        if ids.len() > max_len {
            ids.truncate(max_len);
        }
        markers.retain(|&m| m < max_len);

        let stats = StateTokenStats {
            tokens: state_ids.len(),
            dropped: state_ids.len() - sliced.len(),
            options: {
                let distinct = option_ids.iter().collect::<std::collections::HashSet<_>>().len();
                (distinct < option_ids.len()).then_some(crate::answers::OptionUsage {
                    total: option_ids.len(),
                    distinct,
                    tokens_per_option,
                })
            },
        };
        Ok((ids, markers, stats))
    }
}
