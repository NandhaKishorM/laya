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
use crate::error::{LayaError, Result};
use crate::language_detection::{LanguageAnalysis, LanguageDetection};
use crate::laya_engine::LayaEngine;
use crate::laya_options::{LayaCheckpoint, LayaOptions};
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
}

impl LayaPredictor for LayaEngine {
    fn predict(&self, state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        LayaEngine::predict(self, state, questions)
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

/// Configuration for [`LayaRouter`]. `engine_options` is the template every loaded checkpoint's
/// [`LayaOptions`] is built from (provider, threads, download settings, cache, HF token); its own
/// `checkpoint` field is ignored, since each load overrides it with the checkpoint being built.
pub struct LayaRouterOptions {
    /// Template engine options shared by every checkpoint the router loads. Only the
    /// `checkpoint` field varies per load; every other field (execution provider, thread counts,
    /// `allow_download`, cache directory, HF token) is copied as-is.
    pub engine_options: LayaOptions,
    /// How many checkpoints may stay resident at once before the least-recently-used one is
    /// evicted. Always at least 1.
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
            max_loaded: 1,
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
                "unknown model {}; choose one of \"english\", \"multilingual\", \"typed-decisions\" \
                 (or an alias: \"en\", \"laya\", \"default\", \"multi\", \"ml\", \"laya-multilingual\", \
                 \"typed\", \"typed_decisions\", \"laya-typed-decisions\", \"decisions\")",
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
    /// `lang` > detected script/language > [`LayaRouterOptions::default`].
    pub fn route(
        &self,
        state: &Value,
        questions: Option<&QuestionSet>,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
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
            let lowered = lang.to_lowercase();
            let head = lowered.split('-').next().unwrap_or("");
            let key = if matches!(head, "en" | "eng" | "english") {
                LayaCheckpoint::English
            } else {
                LayaCheckpoint::Multilingual
            };
            return Ok(RouteDecision {
                model: key,
                reason: format!("explicit lang={}", python_repr_str(lang)),
                detection: None,
                workflow,
            });
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
            if let Some(lang) = &det.language {
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
        let state = state.into();
        let decision = self.route(&state, Some(questions), model, task, lang)?;
        let engine = self.load_checkpoint(decision.model)?;
        // Inference deliberately runs after the `Arc<dyn LayaPredictor>` clone above and outside
        // any lock: see the module docs on eviction safety.
        let result = engine.predict(state, questions)?;
        Ok(result.with_routing(decision))
    }
}

impl LayaPredictor for LayaRouter {
    /// [`LayaRouter::predict`] with every override left at its default (no explicit model, task
    /// or lang) — the entry point `laya_shortlist`'s generic `&dyn LayaPredictor` path uses.
    fn predict(&self, state: Value, questions: &QuestionSet) -> Result<LayaResult> {
        LayaRouter::predict(self, state, questions, None, None, None)
    }
}

/// Python `repr()` for a string: single-quoted unless the string itself contains a `'` and no
/// `"`, in which case double quotes are used instead; backslash, the chosen quote character, and
/// the common control characters are escaped. Every model/task/lang name recorded in the routing
/// golden vectors is plain ASCII, so this covers the cases that matter without attempting a full
/// `repr()` (which would also need to classify every Unicode character as printable or not).
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
            c if (c as u32) < 0x20 || (c as u32) == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}
