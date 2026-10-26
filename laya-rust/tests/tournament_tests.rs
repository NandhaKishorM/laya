//! Tier 1: `LayaShortlist::predict_tournament` and the cached embedder, without a model. Ports the
//! model-free checks of `tests/test_shortlist.py` (`predict_tournament`, `cached_embed_fn`); the
//! shared cases recorded from Python are in `tournament_golden_tests.rs`.

use indexmap::IndexMap;
use laya::answers::{ActionInfo, Answer, ChoiceAnswer, LayaResult, NoulAnswer, Usage};
use laya::error::{LayaError, Result};
use laya::laya_router::LayaPredictor;
use laya::laya_shortlist::{CachedEmbed, DEFAULT_CACHE_MAXSIZE, LayaShortlist, TournamentInfo};
use laya::questions::{Question, QuestionSet};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;

/// What the mock does with a round.
#[derive(Clone, Copy)]
enum Mode {
    /// The winner of every choice is the last label it was shown.
    Last,
    /// Leave out the answer to the question with this id.
    Omit(&'static str),
    /// Answer the question with this id with a noul answer.
    NotChoice(&'static str),
    /// Name a winner that was never a label.
    Stray,
}

/// Records every request; `input_tokens` of its usage is the number of the call, so a result's
/// usage says which call it came from.
struct Mock {
    calls: Mutex<Vec<QuestionSet>>,
    mode: Mode,
}

impl Mock {
    fn new(mode: Mode) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            mode,
        }
    }
    fn calls(&self) -> Vec<QuestionSet> {
        self.calls.lock().unwrap().clone()
    }
}

impl LayaPredictor for Mock {
    fn predict(&self, _state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        let number = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(questions.clone());
            calls.len() as u32
        };
        let order: Vec<String> = questions.ids().map(String::from).collect();
        let mut by_id = HashMap::new();
        for id in &order {
            let answer = match (self.mode, &questions[id.as_str()]) {
                (Mode::Omit(skip), _) if skip == id => continue,
                (Mode::NotChoice(only), _) if only == id => {
                    Answer::Noul(NoulAnswer::new(0.9, 1.0, 1.0, ActionInfo::default()))
                }
                (mode, Question::Choice(c)) => {
                    let pick = if matches!(mode, Mode::Stray) {
                        "zzz".to_string()
                    } else {
                        c.labels().last().unwrap().to_string()
                    };
                    let mut probabilities = IndexMap::new();
                    probabilities.insert(pick.clone(), 1.0);
                    Answer::Choice(ChoiceAnswer::new(
                        pick,
                        probabilities,
                        1.0,
                        1.0,
                        ActionInfo::default(),
                    ))
                }
                _ => continue,
            };
            by_id.insert(id.clone(), answer);
        }
        let usage = Usage {
            input_tokens: number,
            output_tokens: 0,
        };
        Ok(LayaResult::new("mock", order, by_id, usage))
    }
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i}")).collect()
}

fn choice(prefix: &str, n: usize) -> Question {
    Question::choice_labels("pick", names(prefix, n)).unwrap()
}

fn labels_of(set: &QuestionSet, id: &str) -> Vec<String> {
    set[id]
        .as_choice()
        .unwrap()
        .labels()
        .map(String::from)
        .collect()
}

fn info(labels: &[&str], n: usize, rounds: usize) -> TournamentInfo {
    TournamentInfo {
        labels: labels.iter().map(|s| s.to_string()).collect(),
        n,
        rounds,
    }
}

#[test]
fn one_round_cuts_contiguous_near_equal_groups_and_final_holds_the_winners() {
    let questions = QuestionSet::new().with("q", choice("a", 40)).unwrap();
    let mock = Mock::new(Mode::Last);
    let result = LayaShortlist::predict_tournament(&mock, "s", &questions, 16).unwrap();

    let calls = mock.calls();
    assert_eq!(2, calls.len());
    let ids: Vec<&str> = calls[0].ids().collect();
    assert_eq!(vec!["0", "1", "2"], ids);
    // 40 labels in ceil(40/16) = 3 parts: floor(i * 40 / 3) bounds 0, 13, 26, 40.
    assert_eq!(names("a", 13), labels_of(&calls[0], "0"));
    assert_eq!(names("a", 26)[13..].to_vec(), labels_of(&calls[0], "1"));
    assert_eq!(names("a", 40)[26..].to_vec(), labels_of(&calls[0], "2"));
    assert_eq!(vec!["a12", "a25", "a39"], labels_of(&calls[1], "q"));

    assert_eq!(
        Some(&info(&["a12", "a25", "a39"], 40, 1)),
        result.tournament().unwrap().get("q")
    );
    // The usage is the final call's, not a sum.
    assert_eq!(2, result.usage().input_tokens);
    assert!(result.shortlist().is_none());
}

#[test]
fn rounds_repeat_until_every_choice_fits_a_group() {
    let questions = QuestionSet::new().with("q", choice("a", 5)).unwrap();
    let mock = Mock::new(Mode::Last);
    let result = LayaShortlist::predict_tournament(&mock, "s", &questions, 2).unwrap();

    let calls = mock.calls();
    assert_eq!(3, calls.len());
    // 5 labels in 3 groups (1, 2, 2), then the 3 winners in 2 groups (1, 2).
    assert_eq!(vec!["0", "1", "2"], calls[0].ids().collect::<Vec<_>>());
    assert_eq!(vec!["0", "1"], calls[1].ids().collect::<Vec<_>>());
    assert_eq!(vec!["a0"], labels_of(&calls[1], "0"));
    assert_eq!(vec!["a2", "a4"], labels_of(&calls[1], "1"));
    assert_eq!(vec!["a0", "a4"], labels_of(&calls[2], "q"));
    assert_eq!(
        Some(&info(&["a0", "a4"], 5, 2)),
        result.tournament().unwrap().get("q")
    );
}

#[test]
fn several_choices_share_each_round_and_other_questions_pass_through() {
    let questions = QuestionSet::new()
        .with("big", choice("b", 5))
        .unwrap()
        .with("score", Question::score("rate", ["lo", "hi"]).unwrap())
        .unwrap()
        .with("small", choice("s", 2))
        .unwrap()
        .with("other", choice("o", 3))
        .unwrap();
    let mock = Mock::new(Mode::Last);
    let result = LayaShortlist::predict_tournament(&mock, "s", &questions, 2).unwrap();

    let calls = mock.calls();
    // "big" (3 groups) and "other" (2 groups) number their groups together, in question order.
    assert_eq!(
        vec!["0", "1", "2", "3", "4"],
        calls[0].ids().collect::<Vec<_>>()
    );
    assert_eq!(vec!["b0"], labels_of(&calls[0], "0"));
    assert_eq!(vec!["o0"], labels_of(&calls[0], "3"));
    assert_eq!(vec!["o1", "o2"], labels_of(&calls[0], "4"));

    let last = calls.last().unwrap();
    assert_eq!(
        vec!["big", "score", "small", "other"],
        last.ids().collect::<Vec<_>>()
    );
    assert!(last["score"].as_score().is_some());
    assert_eq!(vec!["s0", "s1"], labels_of(last, "small"));
    assert_eq!(vec!["o0", "o2"], labels_of(last, "other"));

    let meta = result.tournament().unwrap();
    assert_eq!(
        vec!["big", "small", "other"],
        meta.keys().map(String::as_str).collect::<Vec<_>>()
    );
    assert_eq!(Some(&info(&["s0", "s1"], 2, 0)), meta.get("small"));
    assert_eq!(Some(&info(&["o0", "o2"], 3, 1)), meta.get("other"));
    assert!(meta.get("score").is_none());
}

#[test]
fn nothing_to_narrow_is_one_unchanged_call_that_keeps_option_order() {
    let questions = QuestionSet::new()
        .with(
            "q",
            choice("a", 4)
                .with_option_order([3, 2, 1, 0])
                .with_labels(Value::Null),
        )
        .unwrap();
    let mock = Mock::new(Mode::Last);
    let result = LayaShortlist::predict_tournament(&mock, "s", &questions, 4).unwrap();

    let calls = mock.calls();
    assert_eq!(1, calls.len());
    assert_eq!(names("a", 4), labels_of(&calls[0], "q"));
    assert_eq!(Some(&[3, 2, 1, 0][..]), calls[0]["q"].option_order());
    assert_eq!(1, result.usage().input_tokens);
    assert_eq!(
        Some(&info(&["a0", "a1", "a2", "a3"], 4, 0)),
        result.tournament().unwrap().get("q")
    );
}

#[test]
fn criteria_descriptions_follow_their_labels_into_groups_and_the_final_call() {
    let described = Question::choice(
        "pick",
        (0..6).map(|i| (format!("a{i}"), Value::from(format!("about {i}")))),
    )
    .unwrap();
    let questions = QuestionSet::new().with("q", described).unwrap();
    let mock = Mock::new(Mode::Last);
    LayaShortlist::predict_tournament(&mock, "s", &questions, 3).unwrap();
    let calls = mock.calls();
    let options = calls[1]["q"].as_choice().unwrap().options().to_vec();
    assert_eq!(
        vec![
            ("a2".to_string(), Value::from("about 2")),
            ("a5".to_string(), Value::from("about 5"))
        ],
        options
    );
    // The caller's set is as it was.
    assert_eq!(6, questions["q"].as_choice().unwrap().options().len());
}

#[test]
fn the_closure_variant_runs_for_every_round_and_the_final_call() {
    let questions = QuestionSet::new().with("q", choice("a", 5)).unwrap();
    let mock = Mock::new(Mode::Last);
    let seen = Mutex::new(Vec::new());
    let result = LayaShortlist::predict_tournament_with(
        |state, set| {
            seen.lock().unwrap().push(state.clone());
            mock.predict(state, set)
        },
        "the state",
        &questions,
        2,
    )
    .unwrap();
    assert_eq!(3, seen.lock().unwrap().len());
    assert!(seen.lock().unwrap().iter().all(|s| s == "the state"));
    assert_eq!(3, result.usage().input_tokens);
}

#[test]
fn group_size_below_two_is_refused_with_pythons_text() {
    let questions = QuestionSet::new().with("q", choice("a", 3)).unwrap();
    let mock = Mock::new(Mode::Last);
    for n in [0usize, 1] {
        let error = LayaShortlist::predict_tournament(&mock, "s", &questions, n).unwrap_err();
        assert!(matches!(error, LayaError::InvalidQuestion(_)));
        assert_eq!(
            format!("group_size must be an integer of at least 2, got {n}"),
            error.to_string()
        );
    }
    assert!(mock.calls().is_empty());
}

#[test]
fn a_reply_missing_a_group_raises_the_plain_key_error_text() {
    let questions = QuestionSet::new().with("q", choice("a", 5)).unwrap();
    let mock = Mock::new(Mode::Omit("1"));
    let error = LayaShortlist::predict_tournament(&mock, "s", &questions, 2).unwrap_err();
    assert!(matches!(error, LayaError::InvalidQuestion(_)));
    assert_eq!("'1'", error.to_string());
    assert_eq!(1, mock.calls().len());
}

#[test]
fn a_group_answer_that_is_not_a_choice_raises_the_key_error_for_choice() {
    let questions = QuestionSet::new().with("q", choice("a", 5)).unwrap();
    let mock = Mock::new(Mode::NotChoice("0"));
    let error = LayaShortlist::predict_tournament(&mock, "s", &questions, 2).unwrap_err();
    assert_eq!("'choice'", error.to_string());
}

#[test]
fn a_winner_that_is_not_a_label_raises_the_key_error_for_it() {
    let questions = QuestionSet::new().with("q", choice("a", 5)).unwrap();
    let mock = Mock::new(Mode::Stray);
    let error = LayaShortlist::predict_tournament(&mock, "s", &questions, 2).unwrap_err();
    assert_eq!("'zzz'", error.to_string());
    assert_eq!(1, mock.calls().len());
}

#[test]
fn a_failing_predict_stops_the_tournament() {
    let questions = QuestionSet::new().with("q", choice("a", 5)).unwrap();
    let error = LayaShortlist::predict_tournament_with(
        |_, _| Err(LayaError::Onnx("boom".to_string())),
        "s",
        &questions,
        2,
    )
    .unwrap_err();
    assert_eq!("boom", error.to_string());
}

// ------------------------------------------------------------------------- option_order warning

struct Capture;
static SEEN: Mutex<Vec<(String, log::Level, String)>> = Mutex::new(Vec::new());
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

#[test]
fn a_narrowed_choice_loses_its_option_order_with_one_warning_per_question_per_call() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Warn);
    let questions = QuestionSet::new()
        .with("wide", choice("a", 40).with_option_order((0..40).rev()))
        .unwrap()
        .with("kept", choice("k", 2).with_option_order([1, 0]))
        .unwrap();
    let mock = Mock::new(Mode::Last);
    LayaShortlist::predict_tournament(&mock, "s", &questions, 16).unwrap();

    let calls = mock.calls();
    assert!(calls[0].iter().all(|(_, q)| q.option_order().is_none()));
    let last = calls.last().unwrap();
    assert_eq!(None, last["wide"].option_order());
    assert_eq!(Some(&[1, 0][..]), last["kept"].option_order());
    // The caller's set keeps it.
    assert_eq!(40, questions["wide"].option_order().unwrap().len());

    // Three groups and the final cut, one warning; the count is of the first group cut.
    let log = SEEN.lock().unwrap();
    assert_eq!(1, log.len());
    assert_eq!("laya", log[0].0);
    assert_eq!(log::Level::Warn, log[0].1);
    assert_eq!(
        "laya.shortlist: question 'wide': 'option_order' (40 slots) was dropped because the label \
         set was narrowed to 13; the order no longer fits the kept options and remapping it onto \
         them would move options between slots.",
        log[0].2
    );
}

// ------------------------------------------------------------------------------ cached embedder

type Calls = Mutex<Vec<Vec<String>>>;

/// An embedder whose row for a text is `[len, 1]`, which records each call.
fn recording(calls: &Calls) -> impl Fn(&[String]) -> Result<Vec<Vec<f64>>> + '_ {
    move |texts| {
        calls.lock().unwrap().push(texts.to_vec());
        Ok(texts.iter().map(|t| vec![t.len() as f64, 1.0]).collect())
    }
}

fn strings(texts: &[&str]) -> Vec<String> {
    texts.iter().map(|s| s.to_string()).collect()
}

#[test]
fn misses_are_deduplicated_into_one_call_and_repeats_hit() {
    let calls: Calls = Mutex::new(Vec::new());
    let cache = CachedEmbed::new(recording(&calls));
    assert_eq!(DEFAULT_CACHE_MAXSIZE, cache.cache_info().maxsize);

    let rows = cache.embed(&strings(&["bb", "a", "bb", "ccc"])).unwrap();
    assert_eq!(
        vec![
            vec![2.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![3.0, 1.0]
        ],
        rows
    );
    assert_eq!(vec![strings(&["bb", "a", "ccc"])], *calls.lock().unwrap());
    let c = cache.cache_info();
    assert_eq!((3, 0, 4), (c.size, c.hits, c.misses));

    cache.embed(&strings(&["a", "new", "a"])).unwrap();
    assert_eq!(strings(&["new"]), calls.lock().unwrap()[1]);
    let c = cache.cache_info();
    assert_eq!((4, 2, 5), (c.size, c.hits, c.misses));

    // An all-hit call does not embed, and an empty one touches nothing.
    cache.embed(&strings(&["a"])).unwrap();
    assert_eq!(2, calls.lock().unwrap().len());
    assert!(cache.embed(&[]).unwrap().is_empty());
    let c = cache.cache_info();
    assert_eq!((4, 3, 5), (c.size, c.hits, c.misses));
}

#[test]
fn keys_are_exact_text() {
    let calls: Calls = Mutex::new(Vec::new());
    let cache = CachedEmbed::new(recording(&calls));
    cache.embed(&strings(&["a", "A", "a "])).unwrap();
    assert_eq!(strings(&["a", "A", "a "]), calls.lock().unwrap()[0]);
    assert_eq!(3, cache.cache_info().size);
}

#[test]
fn the_least_recently_used_text_is_evicted() {
    let calls: Calls = Mutex::new(Vec::new());
    let cache = LayaShortlist::cached_embed_fn(recording(&calls), 2).unwrap();
    cache.embed(&strings(&["a", "b"])).unwrap();
    cache.embed(&strings(&["a"])).unwrap(); // a is now the newest
    cache.embed(&strings(&["c"])).unwrap(); // evicts b
    assert_eq!(2, cache.cache_info().size);
    cache.embed(&strings(&["a"])).unwrap(); // still cached
    assert_eq!(2, calls.lock().unwrap().len());
    cache.embed(&strings(&["b"])).unwrap(); // embedded again
    assert_eq!(strings(&["b"]), *calls.lock().unwrap().last().unwrap());
}

#[test]
fn a_call_with_more_misses_than_maxsize_still_answers_every_text() {
    let calls: Calls = Mutex::new(Vec::new());
    let cache = LayaShortlist::cached_embed_fn(recording(&calls), 2).unwrap();
    let rows = cache.embed(&strings(&["a", "bb", "ccc"])).unwrap();
    assert_eq!(3, rows.len());
    assert_eq!(2, cache.cache_info().size);
    // Only the last two stay.
    cache.embed(&strings(&["a"])).unwrap();
    assert_eq!(2, calls.lock().unwrap().len());
}

#[test]
fn rows_are_stored_as_f32() {
    let cache = CachedEmbed::new(|texts: &[String]| Ok(vec![vec![0.1, 1e300]; texts.len()]));
    let rows = cache.embed(&strings(&["x"])).unwrap();
    // 0.1 as f32, widened; 1e300 overflows f32 to inf, which is then zeroed.
    assert_eq!(vec![vec![f64::from(0.1f32), 0.0]], rows);
    assert_ne!(0.1, rows[0][0]);
}

#[test]
fn non_finite_values_become_zero_and_the_repaired_row_is_cached() {
    let calls: Calls = Mutex::new(Vec::new());
    let cache = CachedEmbed::new(|texts: &[String]| {
        calls.lock().unwrap().push(texts.to_vec());
        Ok(texts
            .iter()
            .map(|_| vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0])
            .collect())
    });
    let rows = cache.embed(&strings(&["x"])).unwrap();
    assert_eq!(vec![vec![0.0, 0.0, 0.0, 2.0]], rows);
    assert_eq!(rows, cache.embed(&strings(&["x"])).unwrap());
    assert_eq!(1, calls.lock().unwrap().len());
}

#[test]
fn a_bad_shape_is_refused_with_pythons_text_and_nothing_is_cached() {
    let wrong_rows = CachedEmbed::new(|_: &[String]| Ok(vec![vec![1.0]]));
    let error = wrong_rows.embed(&strings(&["a", "b"])).unwrap_err();
    assert_eq!(
        "embed_fn must return an array of shape (2, dim), got (1, 1)",
        error.to_string()
    );
    let empty = CachedEmbed::new(|_: &[String]| Ok(Vec::new()));
    assert_eq!(
        "embed_fn must return an array of shape (2, dim), got (0,)",
        empty.embed(&strings(&["a", "b"])).unwrap_err().to_string()
    );
    let no_dim = CachedEmbed::new(|texts: &[String]| Ok(vec![Vec::new(); texts.len()]));
    assert_eq!(
        "embed_fn must return an array of shape (1, dim), got (1, 0)",
        no_dim.embed(&strings(&["a"])).unwrap_err().to_string()
    );
    for cache_size in [
        wrong_rows.cache_info().size,
        empty.cache_info().size,
        no_dim.cache_info().size,
    ] {
        assert_eq!(0, cache_size);
    }
    // The misses were counted before the call, as in Python.
    assert_eq!(2, wrong_rows.cache_info().misses);
}

#[test]
fn an_embed_error_is_passed_on_and_nothing_is_cached() {
    let cache = CachedEmbed::new(|_: &[String]| Err(LayaError::Onnx("down".to_string())));
    assert_eq!(
        "down",
        cache.embed(&strings(&["a"])).unwrap_err().to_string()
    );
    assert_eq!(0, cache.cache_info().size);
}

#[test]
fn a_changed_dimension_is_refused_until_the_cache_is_cleared() {
    let width = Mutex::new(2usize);
    let cache = CachedEmbed::new(|texts: &[String]| {
        let w = *width.lock().unwrap();
        Ok(vec![vec![1.0; w]; texts.len()])
    });
    cache.embed(&strings(&["a"])).unwrap();
    *width.lock().unwrap() = 3;
    let error = cache.embed(&strings(&["b"])).unwrap_err();
    assert_eq!(
        "embed_fn returned dim 3, but the cache holds dim 2; call cache_clear() if the model \
         behind embed_fn changed",
        error.to_string()
    );
    assert_eq!(1, cache.cache_info().size);
    cache.cache_clear();
    assert_eq!(3, cache.embed(&strings(&["b"])).unwrap()[0].len());
}

#[test]
fn cache_clear_empties_the_cache_and_resets_the_counters() {
    let calls: Calls = Mutex::new(Vec::new());
    let cache = CachedEmbed::new(recording(&calls));
    cache.embed(&strings(&["a", "a"])).unwrap();
    cache.embed(&strings(&["a"])).unwrap();
    cache.cache_clear();
    let c = cache.cache_info();
    assert_eq!((0, 4096, 0, 0), (c.size, c.maxsize, c.hits, c.misses));
    cache.embed(&strings(&["a"])).unwrap();
    assert_eq!(2, calls.lock().unwrap().len());
}

#[test]
fn maxsize_below_one_is_refused() {
    let error = LayaShortlist::cached_embed_fn(|_: &[String]| Ok(Vec::new()), 0)
        .err()
        .unwrap();
    assert!(matches!(error, LayaError::InvalidQuestion(_)));
    assert_eq!(
        "maxsize must be a positive integer, got 0",
        error.to_string()
    );
}

#[test]
fn the_cache_is_shared_between_threads_and_embeds_outside_its_lock() {
    let embedded = Mutex::new(0usize);
    let cache = CachedEmbed::new(|texts: &[String]| {
        *embedded.lock().unwrap() += texts.len();
        Ok(texts.iter().map(|t| vec![t.len() as f64]).collect())
    });
    let texts = strings(&["a", "bb", "ccc"]);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..50 {
                    let rows = cache.embed(&texts).unwrap();
                    assert_eq!(vec![vec![1.0], vec![2.0], vec![3.0]], rows);
                }
            });
        }
    });
    let c = cache.cache_info();
    assert_eq!(3, c.size);
    assert_eq!(8 * 50 * 3, (c.hits + c.misses) as usize);
    assert!(*embedded.lock().unwrap() >= 3);
}
