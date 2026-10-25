//! Routes a request to the Laya checkpoint best suited to it. Ports `laya/router.py`.
//!
//! Three checkpoints, measured on a shared benchmark (17,416 questions, one T4, identical
//! questions per model):
//!
//! ```text
//! english          convaiinnovations/laya                421M  ModernBERT-large, 512 tokens
//! multilingual     convaiinnovations/laya-multilingual   322M  mmBERT-base, 1024 tokens, 100+ langs
//! typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
//!                                                                fine-tuned on the typed-decisions
//!                                                                workflows
//! ```
//!
//! The English checkpoint does not gently degrade off English, it collapses: on 20-option
//! MASSIVE intent it scores 0.100 on Hindi and 0.103 on Korean, against 0.050 for random
//! guessing -- and it reports high confidence while doing so. Script detection is therefore the
//! primary routing signal; see [`crate::language_detection`].
//!
//! `typed-decisions` is never selected automatically unless the caller opts in with
//! [`LayaRouterOptions::auto_task_detection`] or passes `task = Some("typed_decisions")`: it is
//! fine-tuned on four specific synthetic workflows and should not be a silent default.
//!
//! ## Eviction safety
//!
//! Python relies on garbage collection: an evicted `Agent` is simply dropped from `self._agents`
//! and freed once nothing else references it, including an in-flight `system_one` call. Rust has
//! no GC, so every loaded engine here is held as `Arc<dyn LayaPredictor>`: [`Self::predict`]
//! clones the `Arc` out from under the router lock before running inference, so an eviction that
//! happens mid-predict only drops the router's own reference — the engine itself is freed only
//! once that in-flight predict's clone is also dropped. Inference therefore never runs while
//! holding the router's lock, matching Python's `Agent.system_one` being called outside
//! `self._lock`.
//!
//! Building an engine (unlike inference) *does* happen under the lock, exactly as Python's
//! `load()` holds `self._lock` for the whole `Agent(...)` construction: this is what makes
//! concurrent `Load` of the same checkpoint build the engine exactly once (every later caller
//! blocks on the lock, then sees the checkpoint already resident).

use crate::answers::LayaResult;
use crate::batch_prediction::BatchOptions;
use crate::confidence::apply_confidence_gate;
use crate::error::{LayaError, Result};
use crate::language_detection::{LanguageAnalysis, LanguageDetection, prefix_chars, py_strip};
use crate::laya_engine::{LayaEngine, refuse_null_states};
use crate::laya_options::{LayaCheckpoint, LayaOptions};
use crate::long_prediction::LongOptions;
use crate::predict_options::PredictOptions;
use crate::questions::QuestionSet;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Answers typed questions about a piece of state in one forward pass. Implemented by
/// [`LayaEngine`] and [`LayaRouter`] so `laya_shortlist` (and any other caller) can work over
/// either without knowing which one it has — the Rust equivalent of Python's `_call_predict`
/// duck-typing (`getattr(agent, "predict", None) or getattr(agent, "system_one", None)`).
///
/// Takes `state: Value` rather than `impl Into<Value>` (unlike [`LayaEngine::predict`]) so the
/// trait stays object-safe: `LayaRouter` stores its loaded checkpoints as `Arc<dyn
/// LayaPredictor>`, and a generic method cannot be part of a trait object's vtable. Callers with
/// an `impl Into<Value>` state can convert with `.into()` at the call site.
pub trait LayaPredictor: Send + Sync {
    /// Answer every question in `questions` about `state`.
    fn predict(&self, state: Value, questions: &QuestionSet) -> Result<LayaResult>;

    /// This predictor's scan capability, if it has one. The default is `None`, so implementing
    /// [`LayaPredictor`] elsewhere keeps compiling; [`LayaEngine`] and [`LayaRouter`] return
    /// themselves. [`LayaRouter::predict_long`] needs it: Rust has no run-time interface check on
    /// a `dyn LayaPredictor`, and a predictor without it must fail loudly rather than silently
    /// truncate a long state.
    fn as_long_predictor(&self) -> Option<&dyn LayaLongPredictor> {
        None
    }

    /// This predictor's batch capability, if it has one. The default is `None`, so implementing
    /// [`LayaPredictor`] elsewhere keeps compiling; [`LayaEngine`] and [`LayaRouter`] return
    /// themselves. [`LayaRouter::predict_batch`] needs it.
    fn as_batch_predictor(&self) -> Option<&dyn LayaBatchPredictor> {
        None
    }
}

/// A [`LayaPredictor`] that can score many states against the same questions in shared forward
/// passes. The Rust counterpart of Python's router calling `agent.predict_batch`.
pub trait LayaBatchPredictor: LayaPredictor {
    /// Answer `questions` about every state, in input order. See [`LayaEngine::predict_batch`].
    fn predict_batch(
        &self,
        states: &[Value],
        questions: &QuestionSet,
        options: &BatchOptions,
    ) -> Result<Vec<LayaResult>>;
}

/// A [`LayaPredictor`] that can also scan a state longer than one sequence. The Rust counterpart
/// of Python's router checking `getattr(agent, "predict_long", None)`.
pub trait LayaLongPredictor: LayaPredictor {
    /// Scan overlapping state-token windows and select one answer per question. See
    /// [`LayaEngine::predict_long`].
    fn predict_long(
        &self,
        state: Value,
        questions: &QuestionSet,
        options: &LongOptions,
    ) -> Result<LayaResult>;
}

impl LayaPredictor for LayaEngine {
    fn predict(&self, state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        LayaEngine::predict(self, state, questions)
    }

    fn as_long_predictor(&self) -> Option<&dyn LayaLongPredictor> {
        Some(self)
    }

    fn as_batch_predictor(&self) -> Option<&dyn LayaBatchPredictor> {
        Some(self)
    }
}

impl LayaBatchPredictor for LayaEngine {
    fn predict_batch(
        &self,
        states: &[Value],
        questions: &QuestionSet,
        options: &BatchOptions,
    ) -> Result<Vec<LayaResult>> {
        LayaEngine::predict_batch(self, states, questions, options)
    }
}

impl LayaLongPredictor for LayaEngine {
    fn predict_long(
        &self,
        state: Value,
        questions: &QuestionSet,
        options: &LongOptions,
    ) -> Result<LayaResult> {
        LayaEngine::predict_long(self, state, questions, options)
    }
}

/// Question-id signatures of the four typed-decisions workflows, used only when
/// [`LayaRouterOptions::auto_task_detection`] is enabled.
const TYPED_DECISION_WORKFLOWS: &[(&str, &[&str])] = &[
    (
        "agent_trace_observability",
        &["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        &["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        &[
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        &[
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

/// The state a [`LangGuess`] callback is handed, exactly as the caller gave it to the router: a
/// JSON value from `route`, `predict` and `predict_long`, the original bytes from `route_bytes`
/// (Python passes the `bytes` object itself, invalid UTF-8 included).
#[derive(Debug, Clone, Copy)]
pub enum GuessInput<'a> {
    /// A JSON state.
    Value(&'a Value),
    /// A `bytes` state, untouched.
    Bytes(&'a [u8]),
}

/// The callback form of [`LangGuess`]: reads the state, answers with a language code or `None`.
type GuessFn = dyn Fn(GuessInput<'_>) -> Option<String> + Send + Sync;

/// A caller-supplied answer to "can the English checkpoint read this state?", Python's
/// `lang_guess`: a language code, or a callback taking the state. Any code that is not English
/// routes to the multilingual checkpoint; a code that names no language (blank, `C`, `POSIX`,
/// `C.UTF-8`, `und`, `zxx`, `mul`) or a callback returning `None` abstains, and routing falls
/// through to detection.
#[derive(Clone)]
#[non_exhaustive]
pub enum LangGuess {
    /// A language code such as `"en"`, `"en-US"`, `"en_US.UTF-8"` or `"pt-BR"`.
    Code(String),
    /// A function of the state that returns a language code, or `None` to abstain (for example
    /// a language-identification model that is not confident).
    Callback(Arc<GuessFn>),
}

impl LangGuess {
    /// A hint that reads the state and answers with a code, or `None` to abstain.
    pub fn callback(f: impl Fn(GuessInput<'_>) -> Option<String> + Send + Sync + 'static) -> Self {
        Self::Callback(Arc::new(f))
    }

    /// `_resolve_hint`: `Some(true)` when the hint says English, `Some(false)` when it names
    /// another language, `None` when it identifies nothing.
    fn resolve(&self, state: GuessInput<'_>) -> Option<bool> {
        match self {
            Self::Code(code) => english_from_code(code),
            Self::Callback(f) => f(state).and_then(|code| english_from_code(&code)),
        }
    }
}

impl std::fmt::Debug for LangGuess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Code(code) => f.debug_tuple("Code").field(code).finish(),
            Self::Callback(_) => f.write_str("Callback(..)"),
        }
    }
}

impl From<&str> for LangGuess {
    fn from(code: &str) -> Self {
        Self::Code(code.to_string())
    }
}

impl From<String> for LangGuess {
    fn from(code: String) -> Self {
        Self::Code(code)
    }
}

/// Codes that are valid `$LANG` values but name no language, so they answer nothing about the
/// state: `C`, `POSIX` and `C.UTF-8` (what minimal images ship) and the ISO 639-2 special codes
/// `und` (undetermined), `zxx` (no linguistic content) and `mul` (multiple languages).
const LANGUAGE_AGNOSTIC_CODES: [&str; 5] = ["c", "posix", "und", "zxx", "mul"];

/// `_english_from_code`: `Some(true)` for an English code, `Some(false)` for any other language
/// code, `None` when the code identifies nothing.
///
/// Accepts the forms a caller is likely to have to hand: `"en"`, `"EN"`, `"en-US"`, the POSIX
/// `"en_US"` (which `$LANG` holds), and `"en_US.UTF-8"`.
pub(crate) fn english_from_code(value: &str) -> Option<bool> {
    let code = py_strip(value).to_lowercase();
    if code.is_empty() {
        return None;
    }
    let code = code.split('.').next().unwrap_or("");
    let primary = code.replace('_', "-");
    let primary = primary.split('-').next().unwrap_or("");
    if primary.is_empty() || LANGUAGE_AGNOSTIC_CODES.contains(&primary) {
        return None;
    }
    Some(matches!(primary, "en" | "eng" | "english"))
}

/// The routing outcome: which checkpoint, why, and what was detected. Serializes into
/// `LayaResult::routing` (`answers.rs`). Python's `RouteDecision` also carries a `repo` field
/// (the resolved HuggingFace repo/subfolder string); it is intentionally not ported, since this
/// crate's [`LayaRouterOptions`] already resolves artifacts per checkpoint through the same
/// [`crate::model_artifacts::ModelArtifacts`] path every other engine uses, so there is no
/// separate "which repo string" question to answer here.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteDecision {
    /// The chosen checkpoint.
    pub model: LayaCheckpoint,
    /// Human-readable reason, byte-identical to Python's (including `%r`-style quoting and
    /// `%.0f%%` percentages) so the two are directly comparable in logs.
    pub reason: String,
    /// The script/language detection this decision was based on, or `None` when the model was
    /// chosen without looking at the state at all (an explicit `model`/`task`, or an opted-in
    /// typed-decisions workflow match).
    pub detection: Option<LanguageAnalysis>,
    /// The typed-decisions workflow the question ids matched, if any — populated whenever a
    /// match exists, even if [`LayaRouterOptions::auto_task_detection`] was off and the match
    /// therefore did not decide the routing.
    pub workflow: Option<String>,
}

impl std::fmt::Display for RouteDecision {
    /// Matches Python's `RouteDecision.__repr__`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RouteDecision(model={}, reason={})",
            python_repr_str(self.model.subdir()),
            python_repr_str(&self.reason)
        )
    }
}

/// One request of [`LayaRouter::route_batch`] / [`LayaRouter::predict_batch`]: a state, its
/// questions and the same optional routing overrides [`LayaRouter::route_with_lang_guess`] takes.
/// Python passes these as a request dict with the keys `state`, `questions`, `model`, `task`,
/// `lang` and `lang_guess`.
///
/// Per-request `max_len` and `head_max_len` are not part of this port yet; the type is
/// `#[non_exhaustive]` so they can be added as fields later. Build one with [`Self::new`] and the
/// chained setters.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BatchRequest {
    /// What to decide about. A `null` state is refused, as Python's `TypeError`.
    pub state: Value,
    /// The questions to answer. Shared by `Arc`, so requests that ask the same questions need not
    /// each hold a copy; requests with equal question schemas share a forward pass either way.
    pub questions: Arc<QuestionSet>,
    /// Route to this checkpoint (a name or alias), skipping detection.
    pub model: Option<String>,
    /// Route by task, such as `"typed_decisions"`.
    pub task: Option<String>,
    /// A language code that decides English or not, ahead of detection.
    pub lang: Option<String>,
    /// A per-request language hint; see [`LayaRouter::route_with_lang_guess`].
    pub lang_guess: Option<LangGuess>,
}

impl BatchRequest {
    /// A request with no routing overrides.
    pub fn new(state: impl Into<Value>, questions: impl Into<Arc<QuestionSet>>) -> Self {
        Self {
            state: state.into(),
            questions: questions.into(),
            model: None,
            task: None,
            lang: None,
            lang_guess: None,
        }
    }

    /// Set [`Self::model`].
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set [`Self::task`].
    #[must_use]
    pub fn task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    /// Set [`Self::lang`].
    #[must_use]
    pub fn lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = Some(lang.into());
        self
    }

    /// Set [`Self::lang_guess`].
    #[must_use]
    pub fn lang_guess(mut self, lang_guess: impl Into<LangGuess>) -> Self {
        self.lang_guess = Some(lang_guess.into());
        self
    }
}

/// Configuration for [`LayaRouter`]. `engine_options` is the template every loaded checkpoint's
/// [`LayaOptions`] is built from (provider, threads, download settings, cache, HF token); its own
/// `checkpoint` field is ignored, since each load overrides it with the checkpoint being built.
pub struct LayaRouterOptions {
    /// Template engine options shared by every checkpoint the router loads. Only the
    /// `checkpoint` field varies per load; every other field (execution provider, thread counts,
    /// `allow_download`, cache directory, HF token) is copied as-is.
    pub engine_options: LayaOptions,
    /// How many checkpoints may stay resident at once before the least-recently-used one is
    /// evicted. Always at least 1. Defaults to 2, as Python's `Router(max_loaded=2)` does, so
    /// alternating two languages does not reload a checkpoint on every request.
    pub max_loaded: usize,
    /// Checkpoint used when [`LayaRouter::route`] cannot find any letters in the state at all.
    /// Also used for undecided Latin text. Multilingual by default, matching Python 0.4.0;
    /// mostly-English deployments may explicitly select English.
    pub default: LayaCheckpoint,
    /// Whether a question-id set that exactly matches one of the four typed-decisions workflow
    /// signatures should automatically route to `TypedDecisions`. Off by default: that checkpoint
    /// is fine-tuned on four specific synthetic workflows and should never be a silent default.
    pub auto_task_detection: bool,
    /// Build and load every checkpoint immediately, so no request ever pays a cold load. See
    /// [`LayaRouter::preload`].
    pub preload: bool,
}

impl Default for LayaRouterOptions {
    fn default() -> Self {
        Self {
            engine_options: LayaOptions::default(),
            max_loaded: 2,
            default: LayaCheckpoint::Multilingual,
            auto_task_detection: false,
            preload: false,
        }
    }
}

/// Builds the [`LayaPredictor`] for one checkpoint. The production path
/// ([`LayaRouter::new`]) always builds a real [`LayaEngine`]; tests substitute a factory that
/// returns a fake predictor with no ONNX weights via [`LayaRouter::with_factory`].
type Factory =
    Box<dyn Fn(LayaCheckpoint, &LayaOptions) -> Result<Arc<dyn LayaPredictor>> + Send + Sync>;

struct RouterState {
    loaded: HashMap<LayaCheckpoint, Arc<dyn LayaPredictor>>,
    /// Least-recently-used first.
    order: Vec<LayaCheckpoint>,
    max_loaded: usize,
}

/// Lazily loads Laya checkpoints and sends each request to the right one. See the module docs.
///
/// ```no_run
/// use laya::laya_router::{LayaRouter, LayaRouterOptions};
///
/// let router = LayaRouter::new(LayaRouterOptions::default())?;
/// // router.predict(state, &questions, None, None, None)?;  // routes, then answers
/// # Ok::<(), laya::error::LayaError>(())
/// ```
pub struct LayaRouter {
    engine_options: LayaOptions,
    default: LayaCheckpoint,
    auto_task_detection: bool,
    lang_guess: Option<LangGuess>,
    factory: Factory,
    state: Mutex<RouterState>,
}

impl LayaRouter {
    /// Build a router whose checkpoints are real [`LayaEngine`]s, built with
    /// [`LayaEngine::create`] on first use.
    pub fn new(options: LayaRouterOptions) -> Result<Self> {
        Self::with_factory(options, |checkpoint, opts| {
            let mut opts = opts.clone();
            opts.checkpoint = checkpoint;
            let engine = LayaEngine::create(opts)?;
            Ok(Arc::new(engine) as Arc<dyn LayaPredictor>)
        })
    }

    /// A test-only engine-factory seam: builds a router whose checkpoints come from `factory`
    /// instead of a real [`LayaEngine`], so LRU/attach/preload/unload tests can run with fake
    /// predictors and no ~1.3-1.7 GB weights. `factory` receives the checkpoint being loaded and
    /// the (already checkpoint-substituted) engine options template.
    ///
    /// `#[doc(hidden)]` rather than crate-private: Rust has no visibility scoped to "this crate's
    /// own integration tests", and the integration test suite is a separate crate that links
    /// against this one, so the seam must be `pub` to reach it.
    #[doc(hidden)]
    pub fn with_factory(
        options: LayaRouterOptions,
        factory: impl Fn(LayaCheckpoint, &LayaOptions) -> Result<Arc<dyn LayaPredictor>>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self> {
        let router = Self {
            engine_options: options.engine_options,
            default: options.default,
            auto_task_detection: options.auto_task_detection,
            lang_guess: None,
            factory: Box::new(factory),
            state: Mutex::new(RouterState {
                loaded: HashMap::new(),
                order: Vec::new(),
                max_loaded: options.max_loaded.max(1),
            }),
        };
        if options.preload {
            router.preload(None)?;
        }
        Ok(router)
    }

    /// Install a language hint that applies to every request, as Python's
    /// `Router(lang_guess=...)` does: checked after an explicit `lang` and a per-call hint, and
    /// before the built-in detection. See [`Self::route_with_lang_guess`].
    #[must_use]
    pub fn with_lang_guess(mut self, lang_guess: impl Into<LangGuess>) -> Self {
        self.lang_guess = Some(lang_guess.into());
        self
    }

    // ------------------------------------------------------------------ name resolution

    /// Aliases people are likely to type, resolved to the canonical checkpoint name before
    /// matching. Mirrors Python's `_ALIASES`.
    fn alias(key: &str) -> Option<&'static str> {
        match key {
            "en" | "laya" | "default" => Some("english"),
            "multi" | "ml" | "laya-multilingual" => Some("multilingual"),
            "typed" | "typed_decisions" | "laya-typed-decisions" | "decisions" => {
                Some("typed-decisions")
            }
            _ => None,
        }
    }

    /// Normalise a model name (case-insensitive, whitespace-trimmed, with aliases resolved) to a
    /// [`LayaCheckpoint`]. Ports `normalise_name`.
    pub fn normalise_name(name: &str) -> Result<LayaCheckpoint> {
        let key = name.trim().to_lowercase();
        let key = Self::alias(&key).unwrap_or(key.as_str());
        match key {
            "english" => Ok(LayaCheckpoint::English),
            "multilingual" => Ok(LayaCheckpoint::Multilingual),
            "typed-decisions" => Ok(LayaCheckpoint::TypedDecisions),
            _ => Err(LayaError::InvalidConfig(format!(
                "unknown model {}; choose one of ['english', 'multilingual', 'typed-decisions'] \
                 (or an alias: ['decisions', 'default', 'en', 'laya', 'laya-multilingual', \
                 'laya-typed-decisions', 'ml', 'multi', 'typed', 'typed_decisions'])",
                python_repr_str(name)
            ))),
        }
    }

    /// Name of the typed-decisions workflow whose question ids these are, else `None`. Requires
    /// an exact id-set match, so an unrelated schema that happens to contain `"urgency"` is never
    /// captured. Ports `match_typed_decisions_workflow`.
    pub fn match_typed_decisions_workflow(questions: &QuestionSet) -> Option<String> {
        let ids: HashSet<&str> = questions.ids().collect();
        for (workflow, signature) in TYPED_DECISION_WORKFLOWS {
            let sig: HashSet<&str> = signature.iter().copied().collect();
            if ids == sig {
                return Some((*workflow).to_string());
            }
        }
        None
    }

    // ------------------------------------------------------------------ loading

    /// Return the predictor for `name`, downloading and building it on first use. Concurrent
    /// callers requesting the same checkpoint share a single build (see the module docs).
    pub fn load(&self, name: &str) -> Result<Arc<dyn LayaPredictor>> {
        let key = Self::normalise_name(name)?;
        self.load_checkpoint(key)
    }

    fn load_checkpoint(&self, key: LayaCheckpoint) -> Result<Arc<dyn LayaPredictor>> {
        let mut state = self.state.lock().expect("router state mutex poisoned");
        if let Some(engine) = state.loaded.get(&key) {
            let engine = Arc::clone(engine);
            Self::touch(&mut state, key);
            return Ok(engine);
        }
        // The lock is held across the build itself (see the module docs): a second caller asking
        // for the same checkpoint blocks here rather than racing a duplicate build.
        let engine = (self.factory)(key, &self.engine_options)?;
        state.loaded.insert(key, Arc::clone(&engine));
        state.order.push(key);
        Self::evict(&mut state);
        Ok(engine)
    }

    fn touch(state: &mut RouterState, key: LayaCheckpoint) {
        state.order.retain(|&k| k != key);
        state.order.push(key);
    }

    fn evict(state: &mut RouterState) {
        while state.order.len() > state.max_loaded {
            let victim = state.order.remove(0);
            state.loaded.remove(&victim);
        }
    }

    /// Register an already-built predictor under `name` instead of loading a second copy. Raises
    /// `max_loaded` to fit it if needed, so attaching never immediately evicts what was just
    /// attached.
    pub fn attach(
        &self,
        name: &str,
        predictor: Arc<dyn LayaPredictor>,
    ) -> Result<Arc<dyn LayaPredictor>> {
        let key = Self::normalise_name(name)?;
        let mut state = self.state.lock().expect("router state mutex poisoned");
        state.loaded.insert(key, Arc::clone(&predictor));
        Self::touch(&mut state, key);
        state.max_loaded = state.max_loaded.max(state.loaded.len());
        Ok(predictor)
    }

    /// Download and build checkpoints up front so no request ever pays a model load. `names`
    /// defaults to every checkpoint (`english`, `multilingual`, `typed-decisions`, in that
    /// order). `max_loaded` is raised first, before any of the loads run, so the loop's own
    /// incremental loads never trigger a premature eviction of what preload is about to build.
    ///
    /// Unlike Python (whose single re-entrant lock covers the whole preload call), each
    /// checkpoint's load here takes and releases the router's lock separately — `std::sync::
    /// Mutex` is not re-entrant, so holding it across a loop of `self.load_checkpoint` calls
    /// (which each need the lock themselves) would deadlock. This does not change the
    /// observable end state: `load_checkpoint` already treats an already-resident checkpoint as
    /// a touch rather than a rebuild, so a preload racing another thread's load/attach still
    /// converges to the same result, just interleaved more finely than Python's.
    pub fn preload(&self, names: Option<&[&str]>) -> Result<()> {
        let keys: Vec<LayaCheckpoint> = match names {
            Some(names) => names
                .iter()
                .map(|n| Self::normalise_name(n))
                .collect::<Result<_>>()?,
            None => vec![
                LayaCheckpoint::English,
                LayaCheckpoint::Multilingual,
                LayaCheckpoint::TypedDecisions,
            ],
        };
        {
            let mut state = self.state.lock().expect("router state mutex poisoned");
            state.max_loaded = state.max_loaded.max(keys.len()).max(state.loaded.len());
        }
        for key in keys {
            self.load_checkpoint(key)?;
        }
        Ok(())
    }

    /// Free one checkpoint, or every checkpoint when `name` is `None`.
    pub fn unload(&self, name: Option<&str>) -> Result<()> {
        let mut state = self.state.lock().expect("router state mutex poisoned");
        match name {
            None => {
                state.loaded.clear();
                state.order.clear();
            }
            Some(name) => {
                let key = Self::normalise_name(name)?;
                state.loaded.remove(&key);
                state.order.retain(|&k| k != key);
            }
        }
        Ok(())
    }

    /// Currently-resident checkpoints, least-recently-used first.
    pub fn loaded(&self) -> Vec<LayaCheckpoint> {
        self.state
            .lock()
            .expect("router state mutex poisoned")
            .order
            .clone()
    }

    // ------------------------------------------------------------------ routing

    /// Decide which checkpoint to use, without loading or running anything.
    ///
    /// Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) > explicit
    /// `lang` > `lang_guess` (see [`Self::route_with_lang_guess`]) > detected script/language >
    /// [`LayaRouterOptions::default`].
    pub fn route(
        &self,
        state: &Value,
        questions: Option<&QuestionSet>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
    ) -> Result<RouteDecision> {
        self.route_with_lang_guess(state, questions, model, task, lang, None)
    }

    /// [`Self::route_with_lang_guess`] for a `bytes` state, which Python's `lang.analyse` accepts:
    /// the bytes are read as strict UTF-8 and invalid UTF-8 counts as no text. A `lang_guess`
    /// callback receives the original bytes as [`GuessInput::Bytes`], as Python's receives the
    /// `bytes` object.
    pub fn route_bytes(
        &self,
        state: &[u8],
        questions: Option<&QuestionSet>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&LangGuess>,
    ) -> Result<RouteDecision> {
        let text = Value::String(std::str::from_utf8(state).unwrap_or("").to_string());
        self.route_inner(
            GuessInput::Bytes(state),
            &text,
            questions,
            model,
            task,
            lang,
            lang_guess,
        )
    }

    /// [`Self::route`] with Python's per-call `lang_guess` hint.
    ///
    /// `lang_guess` is an opt-in hint -- a language code or a callback taking the state --
    /// checked after an explicit `lang` and before the built-in detection. It only answers "can
    /// the English checkpoint read this?", so any non-English code routes to the multilingual
    /// checkpoint. A hint that resolves to nothing (a blank code, `C`, `POSIX`, `und`, ... or a
    /// callback returning `None`) falls through to detection, which lets a language-identification
    /// model abstain. A hint installed with [`Self::with_lang_guess`] applies to every request,
    /// after this per-call one.
    pub fn route_with_lang_guess(
        &self,
        state: &Value,
        questions: Option<&QuestionSet>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&LangGuess>,
    ) -> Result<RouteDecision> {
        self.route_inner(
            GuessInput::Value(state),
            state,
            questions,
            model,
            task,
            lang,
            lang_guess,
        )
    }

    /// `Router._route`: `given` is the state as the caller passed it (what a hint callback sees),
    /// `state` the JSON text detection reads.
    #[allow(clippy::too_many_arguments)]
    fn route_inner(
        &self,
        given: GuessInput<'_>,
        state: &Value,
        questions: Option<&QuestionSet>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&LangGuess>,
    ) -> Result<RouteDecision> {
        if let Some(model) = model {
            let key = Self::normalise_name(model)?;
            return Ok(RouteDecision {
                model: key,
                reason: format!("explicit model={}", python_repr_str(model)),
                detection: None,
                workflow: None,
            });
        }

        if let Some(task) = task {
            let normalized = task.to_lowercase().replace('-', "_");
            let target = if normalized == "typed_decisions" {
                "typed-decisions"
            } else {
                task
            };
            let key = Self::normalise_name(target)?;
            return Ok(RouteDecision {
                model: key,
                reason: format!("explicit task={}", python_repr_str(task)),
                detection: None,
                workflow: None,
            });
        }

        let workflow = questions.and_then(Self::match_typed_decisions_workflow);
        if let Some(wf) = &workflow
            && self.auto_task_detection
        {
            return Ok(RouteDecision {
                model: LayaCheckpoint::TypedDecisions,
                reason: format!(
                    "question ids match the {} typed-decisions workflow",
                    python_repr_str(wf)
                ),
                detection: None,
                workflow: Some(wf.clone()),
            });
        }

        if let Some(lang) = lang {
            // An explicit `lang` is decisive only when the code names a language. Blank or
            // whitespace, and `C`/`POSIX`/`und`/`zxx`/`mul`, resolve to no usable hint, so they
            // fall through to `lang_guess`/detection exactly as an abstaining hint does.
            if let Some(english) = english_from_code(lang) {
                return Ok(RouteDecision {
                    model: if english {
                        LayaCheckpoint::English
                    } else {
                        LayaCheckpoint::Multilingual
                    },
                    reason: format!("explicit lang={}", python_repr_str(lang)),
                    detection: None,
                    workflow,
                });
            }
        }

        // Caller-supplied hint, per-call first, then the one installed on the router. Only a hint
        // that actually answers the question routes here; anything else falls through.
        let hints = [
            ("lang_guess", lang_guess),
            ("Router(lang_guess=...)", self.lang_guess.as_ref()),
        ];
        for (source, hint) in hints {
            if let Some(english) = hint.and_then(|h| h.resolve(given)) {
                return Ok(RouteDecision {
                    model: if english {
                        LayaCheckpoint::English
                    } else {
                        LayaCheckpoint::Multilingual
                    },
                    reason: format!(
                        "{source}: the caller identified this as {} text",
                        if english { "English" } else { "non-English" }
                    ),
                    detection: None,
                    workflow,
                });
            }
        }

        let det = LanguageDetection::analyse(state);
        let (key, reason) = if det.script == "unknown" {
            (
                self.default,
                format!(
                    "no letters detected in state; using default ({})",
                    self.default.subdir()
                ),
            )
        } else if det.script != "latin" {
            (
                LayaCheckpoint::Multilingual,
                format!(
                    "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                    det.script,
                    100.0 * det.non_latin_fraction
                ),
            )
        } else if !det.is_english {
            if let (Some(lang), Some(mixed)) = (&det.language, det.mixed_segment.as_deref())
                && !mixed.is_empty()
            {
                (
                    LayaCheckpoint::Multilingual,
                    format!(
                        "Latin script, mostly English, but a line or field reads as {} ({}); \
                         the English checkpoint cannot read it",
                        python_repr_str(lang),
                        python_repr_str(prefix_chars(mixed, 60))
                    ),
                )
            } else if let Some(lang) = &det.language {
                (
                    LayaCheckpoint::Multilingual,
                    format!(
                        "Latin script but language looks like {}, not English",
                        python_repr_str(lang)
                    ),
                )
            } else {
                (
                    LayaCheckpoint::Multilingual,
                    format!(
                        "Latin script, language not identified but {:.0}% non-English letters; \
                         not safe for the English checkpoint",
                        100.0 * det.diacritic_rate
                    ),
                )
            }
        } else if det.language_undecided {
            (
                self.default,
                format!(
                    "Latin script, language not identified and no non-English letters; \
                     using default ({})",
                    self.default.subdir()
                ),
            )
        } else {
            (LayaCheckpoint::English, "English Latin text".to_string())
        };

        Ok(RouteDecision {
            model: key,
            reason,
            detection: Some(det),
            workflow,
        })
    }

    // ------------------------------------------------------------------ running

    /// Route, then answer every question in one forward pass on the chosen checkpoint. The
    /// result's [`LayaResult::routing`] records the decision.
    pub fn predict(
        &self,
        state: impl Into<Value>,
        questions: &QuestionSet,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
    ) -> Result<LayaResult> {
        self.predict_with_lang_guess(state, questions, model, task, lang, None)
    }

    /// [`Self::predict`] with a per-call `lang_guess` hint; see [`Self::route_with_lang_guess`].
    pub fn predict_with_lang_guess(
        &self,
        state: impl Into<Value>,
        questions: &QuestionSet,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&LangGuess>,
    ) -> Result<LayaResult> {
        self.predict_with_options(
            state,
            questions,
            model,
            task,
            lang,
            lang_guess,
            &PredictOptions::default(),
        )
    }

    /// [`Self::predict_with_lang_guess`] with per-call [`PredictOptions`]. Ports
    /// `Router.predict(min_confidence=...)`: a `null` state is refused first, then the threshold is
    /// validated ([`LayaError::InvalidMinConfidence`]), both before anything is routed or loaded.
    /// The gate runs on the router's own result, so it also covers an attached predictor that
    /// knows nothing of it.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_with_options(
        &self,
        state: impl Into<Value>,
        questions: &QuestionSet,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&LangGuess>,
        options: &PredictOptions,
    ) -> Result<LayaResult> {
        let state = state.into();
        // Python's `Router.predict` refuses `None` before it routes or loads anything.
        refuse_null_states(std::slice::from_ref(&state))?;
        let min_confidence = options.min_confidence.as_ref();
        if let Some(min_confidence) = min_confidence {
            min_confidence.validate()?;
        }
        let decision =
            self.route_with_lang_guess(&state, Some(questions), model, task, lang, lang_guess)?;
        let engine = self.load_checkpoint(decision.model)?;
        // Inference deliberately runs after the `Arc<dyn LayaPredictor>` clone above and outside
        // any lock: see the module docs on eviction safety.
        let mut results = [engine.predict(state, questions)?];
        apply_confidence_gate(&mut results, min_confidence);
        let [result] = results;
        Ok(result.with_routing(decision))
    }

    /// Route a heterogeneous request batch without loading any checkpoint. Ports
    /// `Router.route_batch`: each request is routed as [`Self::route_with_lang_guess`] routes it
    /// from its own `model`, `task`, `lang` and `lang_guess`, and the decisions come back in
    /// input order.
    ///
    /// A request whose state is `null` is refused with [`LayaError::InvalidState`], naming the
    /// request's index as Python does; the first failing request fails the call.
    pub fn route_batch(&self, requests: &[BatchRequest]) -> Result<Vec<RouteDecision>> {
        requests
            .iter()
            .enumerate()
            .map(|(index, request)| {
                if request.state.is_null() {
                    return Err(LayaError::InvalidState(format!(
                        "request {index} 'state' must not be None; pass a string, dict, or list"
                    )));
                }
                self.route_with_lang_guess(
                    &request.state,
                    Some(&request.questions),
                    request.model.as_deref(),
                    request.task.as_deref(),
                    request.lang.as_deref(),
                    request.lang_guess.as_ref(),
                )
            })
            .collect()
    }

    /// Route and answer a heterogeneous request batch with minimal model churn. Ports
    /// `Router.predict_batch`: requests are routed first (nothing loads if routing fails), grouped
    /// by checkpoint in order of first appearance, and within a checkpoint by question schema
    /// (order-sensitive: ids, types, instructions and options), so the states of one group share
    /// forward passes through [`LayaEngine::predict_batch`]. Each result carries its request's
    /// [`LayaResult::routing`] and the results come back in input order.
    ///
    /// `options` goes to every group: `batch_size` caps states per forward pass and
    /// `sort_by_length` reorders within a group only when that group clears the engine's own
    /// `1 < batch_size < group size` gate. No requests give an empty list. The first failing group
    /// fails the call. Returns [`LayaError::NotSupported`] when a chosen predictor (for example
    /// one registered with [`Self::attach`]) has no batch capability.
    ///
    /// `options.min_confidence` is validated before anything is routed
    /// ([`LayaError::InvalidMinConfidence`]) and gates every result of the call, as one batch-wide
    /// threshold: it takes no part in the grouping, and each result is gated here, by the router,
    /// so an attached predictor needs no support for it.
    pub fn predict_batch(
        &self,
        requests: &[BatchRequest],
        options: &BatchOptions,
    ) -> Result<Vec<LayaResult>> {
        let min_confidence = options.min_confidence.as_ref();
        if let Some(min_confidence) = min_confidence {
            min_confidence.validate()?;
        }
        let decisions = self.route_batch(requests)?;
        if decisions.is_empty() {
            return Ok(Vec::new());
        }

        // Insertion order keeps cache effects deterministic and collapses an interleaved
        // workload to one load per routed checkpoint for this call.
        let mut groups: Vec<(LayaCheckpoint, Vec<usize>)> = Vec::new();
        for (index, decision) in decisions.iter().enumerate() {
            match groups
                .iter_mut()
                .find(|(model, _)| *model == decision.model)
            {
                Some((_, indices)) => indices.push(index),
                None => groups.push((decision.model, vec![index])),
            }
        }

        // The gate is the router's: the predictors score ungated, whatever the caller asked for.
        let ungated = BatchOptions {
            min_confidence: None,
            ..options.clone()
        };
        let mut results: Vec<Option<LayaResult>> = Vec::with_capacity(requests.len());
        results.resize_with(requests.len(), || None);
        for (model, indices) in groups {
            let predictor = self.load_checkpoint(model)?;
            let batcher = predictor.as_batch_predictor().ok_or_else(|| {
                LayaError::NotSupported(format!(
                    "the predictor for '{}' has no predict_batch, so a batch cannot share \
                     forward passes",
                    model.subdir()
                ))
            })?;

            // Order-sensitive at every level: options are positional, so two equal schemas with
            // different orders must not share a group.
            let mut question_groups: Vec<(String, Vec<usize>)> = Vec::new();
            let mut last: Option<(&Arc<QuestionSet>, String)> = None;
            for &index in &indices {
                let questions = &requests[index].questions;
                let key = match &last {
                    Some((seen, key)) if Arc::ptr_eq(seen, questions) => key.clone(),
                    _ => questions.schema_key(),
                };
                last = Some((questions, key.clone()));
                match question_groups.iter_mut().find(|(known, _)| *known == key) {
                    Some((_, items)) => items.push(index),
                    None => question_groups.push((key, vec![index])),
                }
            }

            for (_, items) in question_groups {
                let states: Vec<Value> = items
                    .iter()
                    .map(|&index| requests[index].state.clone())
                    .collect();
                let batch =
                    batcher.predict_batch(&states, &requests[items[0]].questions, &ungated)?;
                if batch.len() != items.len() {
                    return Err(LayaError::Onnx(format!(
                        "internal error: predict_batch returned {} results for {} states",
                        batch.len(),
                        items.len()
                    )));
                }
                for (&index, result) in items.iter().zip(batch) {
                    results[index] = Some(result.with_routing(decisions[index].clone()));
                }
            }
        }

        let mut results = results
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| {
                    LayaError::Onnx(
                        "internal error: batch execution did not produce every result".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        apply_confidence_gate(&mut results, min_confidence);
        Ok(results)
    }

    /// Route once, on the original state, then scan the whole state on the chosen checkpoint with
    /// [`LayaEngine::predict_long`]. The result's [`LayaResult::routing`] records the decision.
    ///
    /// Returns [`LayaError::NotSupported`] when the chosen predictor (for example one registered
    /// with [`Self::attach`]) has no scan capability, rather than truncating the state.
    pub fn predict_long(
        &self,
        state: impl Into<Value>,
        questions: &QuestionSet,
        options: &LongOptions,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
    ) -> Result<LayaResult> {
        self.predict_long_with_lang_guess(state, questions, options, model, task, lang, None)
    }

    /// [`Self::predict_long`] with a per-call `lang_guess` hint; see
    /// [`Self::route_with_lang_guess`].
    #[allow(clippy::too_many_arguments)]
    pub fn predict_long_with_lang_guess(
        &self,
        state: impl Into<Value>,
        questions: &QuestionSet,
        options: &LongOptions,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
        lang_guess: Option<&LangGuess>,
    ) -> Result<LayaResult> {
        let state = state.into();
        // Python's `Router.predict` refuses `None` before it routes or loads anything.
        refuse_null_states(std::slice::from_ref(&state))?;
        let decision =
            self.route_with_lang_guess(&state, Some(questions), model, task, lang, lang_guess)?;
        let engine = self.load_checkpoint(decision.model)?;
        let scanner = engine.as_long_predictor().ok_or_else(|| {
            LayaError::NotSupported(format!(
                "the predictor for '{}' has no predict_long, so a state longer than its window \
                 cannot be scanned",
                decision.model.subdir()
            ))
        })?;
        let result = scanner.predict_long(state, questions, options)?;
        Ok(result.with_routing(decision))
    }
}

impl LayaPredictor for LayaRouter {
    /// [`LayaRouter::predict`] with every override left at its default (no explicit model, task
    /// or lang) — the entry point `laya_shortlist`'s generic `&dyn LayaPredictor` path uses.
    fn predict(&self, state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        LayaRouter::predict(self, state, questions, None, None, None)
    }

    fn as_long_predictor(&self) -> Option<&dyn LayaLongPredictor> {
        Some(self)
    }

    fn as_batch_predictor(&self) -> Option<&dyn LayaBatchPredictor> {
        Some(self)
    }
}

impl LayaBatchPredictor for LayaRouter {
    /// [`LayaRouter::predict_batch`] with every override left at its default: each state is
    /// routed on its own, as [`LayaRouter::predict`] routes one.
    fn predict_batch(
        &self,
        states: &[Value],
        questions: &QuestionSet,
        options: &BatchOptions,
    ) -> Result<Vec<LayaResult>> {
        let questions = Arc::new(questions.clone());
        let requests: Vec<BatchRequest> = states
            .iter()
            .map(|state| BatchRequest::new(state.clone(), Arc::clone(&questions)))
            .collect();
        LayaRouter::predict_batch(self, &requests, options)
    }
}

impl LayaLongPredictor for LayaRouter {
    /// [`LayaRouter::predict_long`] with every override left at its default.
    fn predict_long(
        &self,
        state: Value,
        questions: &QuestionSet,
        options: &LongOptions,
    ) -> Result<LayaResult> {
        LayaRouter::predict_long(self, state, questions, options, None, None, None)
    }
}

/// Python `repr()` for a string: single-quoted unless the string itself contains a `'` and no
/// `"`, in which case double quotes are used instead; backslash, the chosen quote character, and
/// the control characters are escaped, as is every character Python does not consider printable
/// (`\xNN`, `\uNNNN`, `\UNNNNNNNN`). The routing reasons embed caller text in this form (the
/// `lang` code and the mixed-language line), so the quoting has to agree with Python's.
pub(crate) fn python_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if !python_is_printable(c) => {
                let cp = c as u32;
                if cp < 0x100 {
                    out.push_str(&format!("\\x{cp:02x}"));
                } else if cp < 0x10000 {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    out.push_str(&format!("\\U{cp:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `str.isprintable()` for one character: everything except the "Other" and "Separator"
/// categories (control, format, surrogate, private use, unassigned, line/paragraph/space
/// separators), with the ASCII space as the one printable separator.
fn python_is_printable(c: char) -> bool {
    use unicode_general_category::{GeneralCategory as G, get_general_category};
    c == ' '
        || !matches!(
            get_general_category(c),
            G::Control
                | G::Format
                | G::Surrogate
                | G::PrivateUse
                | G::Unassigned
                | G::LineSeparator
                | G::ParagraphSeparator
                | G::SpaceSeparator
        )
}
