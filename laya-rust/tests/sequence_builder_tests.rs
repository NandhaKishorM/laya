//! Tier 1: sequence layout and the option budget, using [`StubTokenizer`] so no artifacts are
//! needed. Real token ids are a Phase 2 tokenizer-parity concern.

mod common;

use common::stub_tokenizer::StubTokenizer;
use laya::questions::{Question, QuestionType};
use laya::sequence_builder::SequenceBuilder;
use laya::tokenization::LayaTokenizer;
use serde_json::{Value, json};

#[test]
fn lays_out_cls_head_sep_options_sep_state_sep() {
    let tok = StubTokenizer::new();
    let q = Question::choice("pick one", [("a", Value::Null), ("b", Value::Null)]).unwrap();
    let (ids, markers) =
        SequenceBuilder::build(&tok, &json!("state text"), &q, 64, 32, false).unwrap();

    assert_eq!(tok.special().cls_id, ids[0]);
    assert_eq!(tok.special().sep_id, *ids.last().unwrap());
    // Each marker is a mask token, and the two options are adjacent fragments.
    assert_eq!(2, markers.len());
    for &m in &markers {
        assert_eq!(tok.special().mask_id, ids[m]);
    }
    assert!(markers[1] > markers[0]);
    // A SEP closes the head, immediately before the first marker.
    assert_eq!(tok.special().sep_id, ids[markers[0] - 1]);
}

#[test]
fn marker_positions_point_at_consecutive_option_fragments() {
    let tok = StubTokenizer::new();
    // One word per option keeps the arithmetic checkable: mask + 1 word = 2 ids per fragment.
    let q = Question::choice(
        "h",
        [
            ("alpha", Value::Null),
            ("beta", Value::Null),
            ("gamma", Value::Null),
        ],
    )
    .unwrap();
    let (ids, markers) = SequenceBuilder::build(&tok, &json!("s"), &q, 64, 32, false).unwrap();

    assert_eq!(3, markers.len());
    assert_eq!(markers[0] + 2, markers[1]);
    assert_eq!(markers[1] + 2, markers[2]);
    assert_eq!(tok.special().sep_id, ids[markers[2] + 2]);
}

#[test]
fn puts_the_type_name_in_the_head_text() {
    let tok = StubTokenizer::new();
    let q = Question::noul("does it hold", Value::Null, Value::Null);
    SequenceBuilder::build(&tok, &json!("s"), &q, 64, 32, false).unwrap();
    assert_eq!("noul question: does it hold", tok.encoded()[0]);
}

#[test]
fn type_names_match_python() {
    assert_eq!("choice", SequenceBuilder::type_name(QuestionType::Choice));
    assert_eq!("score", SequenceBuilder::type_name(QuestionType::Score));
    assert_eq!("noul", SequenceBuilder::type_name(QuestionType::Noul));
}

#[test]
fn option_fragments_are_encoded_with_a_leading_space() {
    let tok = StubTokenizer::new();
    let q = Question::choice("h", [("a", Value::Null)]).unwrap();
    SequenceBuilder::build(&tok, &json!("s"), &q, 64, 32, false).unwrap();
    // Python encodes " " + option, which the Metaspace pre-tokenizer turns into a word boundary.
    assert!(tok.encoded().iter().any(|s| s == " a"));
}

#[test]
fn renders_options_the_way_python_does() {
    assert_eq!(
        vec!["a: first".to_string(), "b".to_string()],
        SequenceBuilder::render_options(
            &Question::choice("h", [("a", json!("first")), ("b", Value::Null)]).unwrap()
        )
    );

    assert_eq!(
        vec!["level 0: low".to_string(), "level 1: high".to_string()],
        SequenceBuilder::render_options(&Question::score("h", ["low", "high"]).unwrap())
    );

    assert_eq!(
        vec![
            "false: no, the statement does not hold".to_string(),
            "true: yes, the statement holds".to_string()
        ],
        SequenceBuilder::render_options(&Question::noul("h", Value::Null, Value::Null))
    );

    assert_eq!(
        vec!["false: all fine".to_string(), "true: on fire".to_string()],
        SequenceBuilder::render_options(&Question::noul("h", "all fine", "on fire"))
    );
}

#[test]
fn treats_only_null_and_empty_string_as_a_missing_description() {
    // Python tests `v is None or v == ""`, so 0 and false are real descriptions.
    let q = Question::choice(
        "h",
        [
            ("zero", json!(0)),
            ("no", json!(false)),
            ("blank", json!("")),
            ("none", Value::Null),
        ],
    )
    .unwrap();
    assert_eq!(
        vec![
            "zero: 0".to_string(),
            "no: false".to_string(),
            "blank".to_string(),
            "none".to_string()
        ],
        SequenceBuilder::render_options(&q)
    );
}

#[test]
fn scrubs_a_mask_token_out_of_every_fragment() {
    let tok = StubTokenizer::new();
    let mask = &tok.special().mask_token;
    // A literal <mask> in user input would forge a marker the scorer then reads as an option.
    let q = Question::choice(
        format!("is this {mask} safe"),
        [(format!("a{mask}b"), Value::Null)],
    )
    .unwrap();
    SequenceBuilder::build(
        &tok,
        &json!(format!("state {mask} here")),
        &q,
        64,
        32,
        false,
    )
    .unwrap();

    for text in tok.encoded() {
        assert!(
            !text.contains(mask.as_str()),
            "encoded text still contains the mask token: {text:?}"
        );
    }
}

#[test]
fn truncates_long_options_to_forty_eight_tokens() {
    let tok = StubTokenizer::new();
    let words: Vec<String> = (0..120).map(|i| format!("w{i}")).collect();
    let q = Question::choice("h", [("opt", json!(words.join(" ")))]).unwrap();
    let (ids, markers) = SequenceBuilder::build(&tok, &json!("s"), &q, 512, 256, false).unwrap();

    // The only fragment is mask + at most 48 option tokens, then the closing SEP. "opt: " adds two
    // words, so 48 is the cap that bites rather than the option running out.
    assert_eq!(1, markers.len());
    let fragment_end = ids[markers[0]..]
        .iter()
        .position(|&id| id == tok.special().sep_id)
        .unwrap()
        + markers[0];
    assert_eq!(markers[0] + 49, fragment_end);
}

#[test]
fn shrinks_options_when_they_would_eat_the_whole_head_budget() {
    let tok = StubTokenizer::new();
    // 20 options of ~10 tokens each against a 96-token head budget forces the `opt_budget < 16`
    // branch, which caps every fragment at `(head_max_len - 16) // n`.
    let options: Vec<(String, Value)> = (0..20)
        .map(|i| {
            let desc = (0..10)
                .map(|j| format!("d{i}_{j}"))
                .collect::<Vec<_>>()
                .join(" ");
            (format!("opt{i}"), json!(desc))
        })
        .collect();
    let q = Question::choice("head words here", options).unwrap();
    let (ids, markers) = SequenceBuilder::build(&tok, &json!("s"), &q, 512, 96, false).unwrap();

    assert_eq!(20, markers.len());
    // Mirrors the production formula `max(4, (head_max_len - 16) // n)`; with these numbers it
    // happens to always pick the 4-token floor.
    #[allow(clippy::unnecessary_min_or_max)]
    let per = 4.max((96 - 16) / 20);
    for i in 1..markers.len() {
        let width = markers[i] - markers[i - 1];
        assert!(
            width <= per,
            "fragment {} is {width} tokens, above the {per}-token cap",
            i - 1
        );
    }
    assert_eq!(tok.special().cls_id, ids[0]);
}

#[test]
fn keeps_at_least_eight_head_tokens_when_options_dominate() {
    let tok = StubTokenizer::new();
    let options: Vec<(String, Value)> = (0..30)
        .map(|i| (format!("o{i}"), json!("a b c d e f g h")))
        .collect();
    let q = Question::choice("one two three four five six seven eight nine ten", options).unwrap();
    let (ids, markers) = SequenceBuilder::build(&tok, &json!("s"), &q, 512, 64, false).unwrap();

    // CLS + head + SEP, so the first marker cannot be earlier than position 2.
    assert!(markers[0] >= 2);
    assert_eq!(tok.special().sep_id, ids[markers[0] - 1]);
}

#[test]
fn truncates_the_state_to_fit_max_len() {
    let tok = StubTokenizer::new();
    let state: Vec<String> = (0..500).map(|i| format!("s{i}")).collect();
    let q = Question::noul("h", Value::Null, Value::Null);
    let (ids, markers) =
        SequenceBuilder::build(&tok, &json!(state.join(" ")), &q, 64, 32, false).unwrap();

    assert_eq!(64, ids.len());
    assert_eq!(tok.special().sep_id, *ids.last().unwrap());
    assert_eq!(2, markers.len());
}

#[test]
fn keeps_the_state_tail_when_truncating_left() {
    let tok = StubTokenizer::new();
    let state: Vec<String> = (0..200).map(|i| format!("s{i}")).collect();
    let state = json!(state.join(" "));
    let q = Question::noul("h", Value::Null, Value::Null);
    let (head_ids, head_markers) = SequenceBuilder::build(&tok, &state, &q, 64, 32, false).unwrap();
    let (tail_ids, tail_markers) = SequenceBuilder::build(&tok, &state, &q, 64, 32, true).unwrap();

    assert_eq!(head_ids.len(), tail_ids.len());
    assert_ne!(head_ids, tail_ids);
    // Both keep the same prefix up to the state, and differ from there on.
    assert_eq!(head_markers, tail_markers);
}

#[test]
fn left_truncation_with_no_room_drops_the_whole_state() {
    // laya 0.3.21 slices `state_ids[max(0, len - room):]`, so with room == 0 no state token
    // survives. 0.3.6's `st[-room:]` kept the whole state there (`x[-0:]` is all of `x`).
    let tok = StubTokenizer::new();
    let q = Question::noul("h", Value::Null, Value::Null);
    let (empty_ids, _) = SequenceBuilder::build(&tok, &json!(""), &q, 512, 32, false).unwrap();
    let max_len = empty_ids.len();

    let state: Vec<String> = (0..50).map(|i| format!("s{i}")).collect();
    let (ids, _) =
        SequenceBuilder::build(&tok, &json!(state.join(" ")), &q, max_len, 32, true).unwrap();

    assert_eq!(empty_ids, ids);
}

#[test]
fn drops_markers_that_would_land_past_max_len() {
    let tok = StubTokenizer::new();
    // 40 options cannot fit a 32-token sequence, so markers are lost. A later `predict` phase must
    // treat a short marker list as an error rather than answering a question missing options.
    let options: Vec<(String, Value)> = (0..40).map(|i| (format!("o{i}"), json!("desc"))).collect();
    let option_count = options.len();
    let q = Question::choice("h", options).unwrap();
    let (ids, markers) = SequenceBuilder::build(&tok, &json!("s"), &q, 32, 256, false).unwrap();

    assert_eq!(32, ids.len());
    assert!(markers.len() < option_count);
    for &m in &markers {
        assert!(m < 32);
    }
}

#[test]
fn renders_non_string_instructions_with_literal_unicode() {
    // laya 0.3.21: json.dumps(..., ensure_ascii=False), so non-ASCII stays literal rather than
    // coming out as \uXXXX (0.3.6's behavior, which the tokenizer then read back as escape text).
    let q = Question::noul(json!({"hint": "Müller"}), Value::Null, Value::Null);
    let rendered = SequenceBuilder::render_instructions(&q);

    assert!(rendered.contains('ü'));
    assert!(!rendered.contains("\\u"));
}

#[test]
fn passes_string_instructions_through_untouched() {
    let q = Question::noul("Müller & co <urgent>", Value::Null, Value::Null);
    assert_eq!(
        "Müller & co <urgent>",
        SequenceBuilder::render_instructions(&q)
    );
}
