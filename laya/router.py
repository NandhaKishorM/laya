"""Route a request to the Laya checkpoint best suited to it.

Three checkpoints, measured on a shared benchmark (17,416 questions, one T4, identical questions
per model -- see the repository's benchmark notebook):

  english          convaiinnovations/laya                421M  ModernBERT-large, 512 tokens
  multilingual     convaiinnovations/laya-multilingual   322M  mmBERT-base, 1024 tokens, 100+ langs
  typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
                                                                fine-tuned on the typed-decisions
                                                                workflows

Why routing is worth it -- accuracy by language family:

                      english   multilingual
  MASSIVE intent  en    0.783       0.657        <- English checkpoint wins
  MASSIVE intent  non-en 0.306      0.451
  XNLI            en    0.860       0.843
  XNLI            non-en 0.521      0.731        <- +21 points for multilingual
  English suites        0.684       0.619

The English checkpoint does not gently degrade off English, it collapses: on 20-option MASSIVE
intent it scores 0.100 on Hindi and 0.103 on Korean, against 0.050 for random guessing -- and it
reports high confidence while doing so (ECE 0.855 on Hindi). Script detection is therefore the
primary routing signal.

`typed-decisions` is never selected automatically unless you opt in with
`auto_task_detection=True` or pass `task="typed_decisions"`: it is fine-tuned on four specific
synthetic workflows and should not be a silent default.
"""
import gc
import json
import os
import threading
import time
from collections.abc import Sequence as SequenceABC
from typing import Any, Dict, List, Optional, Sequence, Union

from .hooks import (
    HookRegistry, PredictContext, aggregate_usage, compose_hooks, dispatch, normalise_hooks,
    validate_timeout,
)
from .hooks import _SKIP_DEFAULTS
from .lang import analyse

# The hub repo bundles all three checkpoints; only the requested subfolder is downloaded.
BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}

# The same checkpoints also live in their own repos, for anyone who prefers them.
STANDALONE_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}


def _repo_str(spec):
    """Human-readable id for a model spec: 'repo' or 'repo/subfolder'."""
    repo, sub = _split(spec)
    return "%s/%s" % (repo, sub) if sub else repo


def _split(spec):
    """Normalise a model spec to (repo_or_path, subfolder)."""
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None

# Aliases people are likely to type.
_ALIASES = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

# Question-id signatures of the four typed-decisions workflows, used only when
# auto_task_detection is enabled.
_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}


class RouteDecision(dict):
    """The routing outcome: which model, why, and what was detected.

    Behaves as a dict so it serialises straight into an API response.
    """

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError("unknown model %r; choose one of %s (or an alias: %s)"
                         % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES)))
    return key


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """Name of the typed-decisions workflow whose question ids these are, else None.

    Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
    is never captured.
    """
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


def _question_schema(questions: Dict[str, Any]) -> str:
    """Order-sensitive signature of a question schema, for sharing forward passes.

    sort_keys=False keeps insertion order significant at every nesting level, because
    option order is positional in render_options. default=str matches render_criterion's
    tolerance, so schemas that render identically still share a group.
    """
    return json.dumps(questions, sort_keys=False, ensure_ascii=False, default=str)


# Subtags that mean "the English checkpoint can read this". Routing needs one bit -- is this
# English Latin text, or something the English checkpoint cannot read -- not a language id, so
# every other code that names a language resolves to the multilingual checkpoint.
_ENGLISH_SUBTAGS = ("en", "eng", "english")

# Codes that are valid `$LANG` values but name no language, so they answer nothing about the
# state. `C`, `POSIX` and `C.UTF-8` are what minimal images ship -- `C.UTF-8` is the default
# `LANG` in the official Python image, which is where `laya-serve` runs -- and the ISO 639-2
# special codes say the same thing in the standard's own vocabulary: `und` undetermined,
# `zxx` no linguistic content, `mul` multiple languages. They abstain, which is what the blank
# case below already does, rather than forcing the multilingual checkpoint on English text.
_LANGUAGE_AGNOSTIC_CODES = ("c", "posix", "und", "zxx", "mul")


def _english_from_code(value: Any) -> Optional[bool]:
    """True/False for a language code, or None when the code identifies nothing.

    Accepts the forms a caller is likely to have to hand: `"en"`, `"EN"`, `"en-US"`, the
    POSIX `"en_US"` (which `$LANG` holds), and `"en_US.UTF-8"`. `None` here means "no usable
    hint", which is what lets a language-identification model abstain -- and it is also what a
    code that names no language returns, so `LANG=C` falls through to detection instead of
    pinning every request to one checkpoint.
    """
    if value is None:
        return None
    code = str(value).strip().lower()
    if not code:
        return None
    code = code.split(".", 1)[0]                       # en_US.UTF-8 -> en_US
    primary = code.replace("_", "-").split("-", 1)[0]  # en_US -> en
    if not primary or primary in _LANGUAGE_AGNOSTIC_CODES:
        return None
    return primary in _ENGLISH_SUBTAGS


class Router(HookRegistry):
    """Lazily loads Laya checkpoints and sends each request to the right one.

        from laya import Router

        r = Router()
        r.predict({"message": "Mein Konto wurde zweimal belastet"}, questions)   # -> multilingual
        r.predict({"message": "I was charged twice"}, questions)                 # -> english
        r.predict(state, questions, model="typed-decisions")                     # explicit

    Models are downloaded and built on first use. `max_loaded` caps how many stay resident
    (least-recently-used is evicted), because all three together are ~1.16B parameters.

    The default is 2, because automatic routing only ever chooses between `english` and
    `multilingual`: a cap of one rebuilds the checkpoint it just evicted on every script switch,
    which is seconds per request on exactly the traffic the Router exists for. Traffic that only
    ever sees one language never builds the second checkpoint, so the default costs it nothing.
    Lower it to 1 for a memory-constrained host, and raise it to 3 (or preload) when
    `auto_task_detection`, an explicit `model=` or an explicit `task=` can reach
    `typed-decisions` as well.

    For a server or a demo, preload instead: a cold load costs seconds, while detection costs
    microseconds, so even the default still pays a load the first time a language appears.

        r = Router(preload=True)                    # all three resident, routing is free
        r = Router(preload=True, device="cuda")
        r.preload(["english", "multilingual"])      # or just the two you serve

    Hub revisions are opt-in. `revision` applies one commit to every model;
    `revisions={"english": "...", "multilingual": "..."}` overrides that per model,
    which is useful when standalone repositories were reviewed at different commits.
    Without either, huggingface_hub's normal default and existing offline cache are used.

    Hooks are opt-in and run at the Router level: `on_route` sees the routing decision,
    `on_load` / `on_evict` see model lifecycle, and `on_predict_start` / `on_predict_end`
    wrap the whole route+infer call. See `laya.hooks`.
    """

    # Opt-in defaults so a hand-built instance (`Router.__new__` in tests) works unset.
    # `hooks`/`_hooks_mutex` come from HookRegistry.
    hooks_raise = True
    hooks_concurrent = True
    hooks_timeout = None
    _hooks_lock = None

    def __init__(
        self,
        models: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        revisions: Optional[Dict[str, Optional[str]]] = None,
        max_loaded: int = 2,
        default: str = "english",
        auto_task_detection: bool = False,
        standalone_repos: bool = False,
        preload: bool = False,
        lang_guess: Optional[Any] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: bool = True,
        hooks_concurrent: bool = True,
        hooks_timeout: Optional[float] = None,
    ):
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_concurrent = bool(hooks_concurrent)
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._hooks_lock = threading.RLock() if not hooks_concurrent else None
        self._hooks_mutex = threading.Lock()
        self.models = dict(STANDALONE_MODELS if standalone_repos else DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(k): v for k, v in models.items()})
        self.device = device
        self.token = token or os.environ.get("HF_TOKEN")
        # Optional Hub revision (commit SHA/branch/tag) applied to every checkpoint load.
        # `revisions` overrides it per normalized model name, for standalone repos whose
        # reviewed commits differ.
        self.revision = revision
        self.revisions: Dict[str, Optional[str]] = {
            normalise_name(k): v for k, v in (revisions or {}).items()
        }
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        # An opt-in language hint installed for every request: a code, or a callable taking the
        # state and returning one (or None to abstain). Checked before the built-in detection,
        # never before an explicit `model`, `task` or `lang`. The default path is unchanged, so
        # the heuristic stays dependency-free; this is the seam for a real LID model.
        self.lang_guess = lang_guess
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []          # least-recently-used first
        # Re-entrant lock guarding model lifecycle (load/unload/attach/preload) and the
        # LRU bookkeeping. RLock so the public methods can call the private `_touch`/`_evict`
        # helpers without deadlocking. Inference (`Agent.system_one`) is deliberately left
        # outside the lock so concurrent predictions share a checkpoint without serialising.
        self._lock = threading.RLock()
        if preload:
            self.preload()

    # ------------------------------------------------------------------ loading
    def load(self, name: str):
        """Return the Agent for `name`, downloading and building it on first use.

        Concurrent callers share a single Agent instead of building duplicates.
        """
        key = normalise_name(name)
        with self._lock:
            if key in self._agents:
                self._touch(key)
                return self._agents[key]
            from .agent import Agent
            repo, sub = _split(self.models[key])
            kwargs = {"device": self.device, "token": self.token, "subfolder": sub}
            model_revision = self.revisions.get(key, self.revision)
            if model_revision is not None:
                kwargs["revision"] = model_revision
            agent = Agent(repo, **kwargs)
            self._agents[key] = agent
            self._order.append(key)
            evicted = self._evict_locked()
        # Lifecycle hooks fire after the lock is released, so a hook can safely call the Router.
        self._dispatch_lifecycle("on_evict", evicted)
        dispatch(compose_hooks(self.hooks), "on_load",
                 PredictContext(states=[], questions={}, model=key, agent=agent, router=self),
                 raise_errors=self.hooks_raise, lock=self._hooks_lock, timeout=self.hooks_timeout)
        return agent

    def _touch(self, key: str):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict_locked(self) -> List[str]:
        """Drop least-recently-used agents until `max_loaded` holds. Returns evicted names."""
        evicted: List[str] = []
        while len(self._order) > self.max_loaded:
            victim = self._order.pop(0)
            agent = self._agents.pop(victim, None)
            if agent is not None:
                evicted.append(victim)
                del agent
        if len(self._order) < len(self._agents):     # keep the two views consistent
            for k in list(self._agents):
                if k not in self._order:
                    agent = self._agents.pop(k, None)
                    if agent is not None:
                        evicted.append(k)
                        del agent
        if evicted:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return evicted

    def _evict(self) -> List[str]:
        with self._lock:
            return self._evict_locked()

    def _dispatch_lifecycle(self, event: str, names: List[str]) -> None:
        for name in names:
            dispatch(compose_hooks(self.hooks), event,
                     PredictContext(states=[], questions={}, model=name, router=self),
                     raise_errors=self.hooks_raise, lock=self._hooks_lock, timeout=self.hooks_timeout)

    def attach(self, name: str, agent: Any):
        """Register an already-built Agent under `name` instead of loading a second copy.

        Useful when the process has a checkpoint loaded for other reasons: a demo that already
        built `convaiinnovations/laya` can hand it to the router rather than pay for -- and hold
        in memory -- a duplicate 421M parameters.
        """
        key = normalise_name(name)
        with self._lock:
            self._agents[key] = agent
            self._touch(key)
            self.max_loaded = max(self.max_loaded, len(self._agents))
        return agent

    def preload(self, names: Optional[List[str]] = None):
        """Download and build checkpoints up front so no request ever pays a model load.

        A cold load costs seconds; language detection costs microseconds. With every
        checkpoint resident, routing is effectively free -- which is what you want in a
        server or a demo. `max_loaded` is raised to fit both the requested checkpoints and
        all already-resident agents, so incremental preloading does not evict either.
        """
        names = [normalise_name(n) for n in (list(self.models) if names is None else names)]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(set(names) | set(self._agents)))
        for n in names:
            with self._lock:
                already = n in self._agents    # an attached agent is already built
            if not already:
                # load() dispatches on_load outside the lock; do not hold it across the call.
                self.load(n)
        return self

    def unload(self, name: Optional[str] = None):
        """Free one model, or all of them."""
        with self._lock:
            if name is None:
                freed = list(self._order)
                self._agents.clear()
                self._order.clear()
            else:
                key = normalise_name(name)
                agent = self._agents.pop(key, None)
                if key in self._order:
                    self._order.remove(key)
                freed = [key] if agent is not None else []
                del agent
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            except Exception:
                pass
        self._dispatch_lifecycle("on_evict", freed)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    @property
    def loaded_revisions(self) -> Dict[str, Optional[str]]:
        """Commit SHA each resident agent was loaded from (None for local paths)."""
        with self._lock:
            return {name: getattr(agent, "revision", None) for name, agent in self._agents.items()}

    def _resolve_hint(self, hint: Any, state: Union[str, dict, list, None]) -> Optional[bool]:
        """True/False for a hint about whether the English checkpoint can read `state`.

        `hint` is either a language code or a callable taking the state. Anything the hint
        cannot answer returns None, which makes `route` fall through to detection rather than
        picking a checkpoint on no evidence.
        """
        if hint is None:
            return None
        if callable(hint):
            hint = hint(state)
        return _english_from_code(hint)

    # ------------------------------------------------------------------ routing
    def route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, then let `on_route` hooks observe or replace it.

        `ctx.decision` is the `RouteDecision`; a hook may replace it (for example to pin a
        checkpoint) and the replacement is what gets returned and used. `hooks` are per-call
        hooks, appended after any installed on the Router.
        """
        decision = self._route(state, questions, model=model, task=task, lang=lang, lang_guess=lang_guess)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        active = compose_hooks(self.hooks, hooks)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        ctx = PredictContext(states=[state], questions=questions or {}, decision=decision, router=self)
        dispatch(active, "on_route", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
        return ctx.decision

    def _route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, without loading or running anything.

        Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
        explicit `lang` > `lang_guess` > detected script/language > default.

        `lang_guess` is an opt-in hint -- a language code or a callable taking the state --
        checked after an explicit `lang` and before the built-in detection. It only answers
        "can the English checkpoint read this?", so any non-English code routes to the
        multilingual checkpoint. A hint that resolves to nothing falls through to detection,
        which lets a language-identification model abstain. Pass one here, or set
        `Router(lang_guess=...)` to apply it to every request.
        """
        if model is not None:
            key = normalise_name(model)
            return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason="explicit model=%r" % model,
                                 detection=None, workflow=None)

        if task is not None:
            key = normalise_name("typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task)
            return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason="explicit task=%r" % task,
                                 detection=None, workflow=None)

        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(model="typed-decisions", repo=_repo_str(self.models["typed-decisions"]),
                                 reason="question ids match the %r typed-decisions workflow" % workflow,
                                 detection=None, workflow=workflow)

        if lang is not None:
            # An explicit `lang` is decisive only when the code names a language. Blank or
            # whitespace resolves to no usable hint, so it falls through to lang_guess/detection
            # exactly as an abstaining hint does; real English/non-English codes still route now.
            resolved = _english_from_code(lang)
            if resolved is not None:
                key = "english" if resolved else "multilingual"
                return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason="explicit lang=%r" % lang,
                                     detection=None, workflow=workflow)

        # Caller-supplied hint, per-call first then the one installed on the Router. Only a hint
        # that actually answers the question routes here; anything else falls through.
        for source, hint in (("lang_guess", lang_guess), ("Router(lang_guess=...)", self.lang_guess)):
            resolved = self._resolve_hint(hint, state)
            if resolved is not None:
                key = "english" if resolved else "multilingual"
                return RouteDecision(
                    model=key, repo=_repo_str(self.models[key]),
                    reason="%s: the caller identified this as %s text" % (
                        source, "English" if resolved else "non-English"),
                    detection=None, workflow=workflow)

        det = analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it" % (
                det["script"], 100 * float(det["non_latin_fraction"]))
        elif not det["is_english"]:
            key = "multilingual"
            if det.get("mixed_segment"):
                reason = ("Latin script, mostly English, but a line or field reads as %r (%r); "
                          "the English checkpoint cannot read it" % (det["language"], det["mixed_segment"][:60]))
            elif det["language"]:
                reason = "Latin script but language looks like %r, not English" % det["language"]
            else:
                # Unidentified Latin-script language: routed on the non-English letters alone,
                # because no stopword list here covers it.
                reason = ("Latin script, language not identified but %.0f%% non-English letters; "
                          "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"])))
        elif det["language_undecided"]:
            # Nothing identifies the language: too short, or only content words ("Quero cancelar",
            # "Esqueci minha senha"). That is no evidence of English either, so it takes the same
            # `default` as a state with no letters. A deployment that serves mostly non-English
            # traffic sets `Router(default="multilingual")`; the stock default keeps it English.
            key = self.default
            reason = ("Latin script, language not identified and no non-English letters; "
                      "using default (%s)" % key)
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason=reason,
                             detection=det, workflow=workflow)

    # ------------------------------------------------------------------ running
    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        hooks_timeout: Optional[float] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Route, then answer every question in one forward pass on the chosen checkpoint.

        The result is the usual `system_one` payload plus a `routing` key recording the decision.
        Router-level `on_predict_start` / `on_predict_end` hooks wrap the whole route+infer call
        and see `ctx.decision`; see `laya.hooks`. `max_len` / `head_max_len` override the agent
        token budget for this call (a start hook may set `ctx.max_len` / `ctx.head_max_len`).
        """
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)

        # Per-call hooks apply to the whole call, including on_route inside route().
        decision = self.route(state, questions, model=model, task=task, lang=lang,
                              lang_guess=lang_guess, hooks=hooks, hooks_raise=hooks_raise,
                              hooks_timeout=hooks_timeout)
        agent = self.load(decision["model"])
        effective_lang = lang
        if effective_lang is None and decision.get("detection") and decision["detection"].get("language"):
            effective_lang = decision["detection"]["language"]

        ctx = PredictContext(states=[state], questions=questions, decision=dict(decision),
                             model=decision["model"], agent=agent, router=self,
                             max_len=max_len, head_max_len=head_max_len)
        try:
            dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            if ctx.results is None:
                # Pass token-budget overrides only when set, so any Agent-like object that does
                # not accept them still works on the default path.
                overrides = {}
                if ctx.max_len is not None:
                    overrides["max_len"] = ctx.max_len
                if ctx.head_max_len is not None:
                    overrides["head_max_len"] = ctx.head_max_len
                
                skip = _SKIP_DEFAULTS.set(True)
                try:
                    result = agent.system_one(ctx.states[0], ctx.questions, lang=effective_lang, **overrides)
                except TypeError as e:
                    if "unexpected keyword argument 'lang'" in str(e):
                        result = agent.system_one(ctx.states[0], ctx.questions, **overrides)
                    else:
                        raise
                finally:
                    _SKIP_DEFAULTS.reset(skip)
                result["routing"] = dict(decision)
                ctx.results = [result]
            else:
                # A cache hit short-circuits inference, but Router.predict still promises a
                # `routing` key. Add it without overwriting a routing the cached payload has.
                for result in ctx.results:
                    if isinstance(result, dict):
                        result.setdefault("routing", dict(decision))
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(active, "on_error", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors, lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results[0]

    def decide(self, state: Union[str, dict, list], schema: Any = None, *,
               questions: Optional[Dict[str, Any]] = None, return_details: bool = False,
               **predict_kwargs) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        (for example `model=`, `task=`, `hooks=`) are forwarded to `predict`.
        """
        from .structured import decide as _decide
        return _decide(self, state, schema, questions=questions,
                       return_details=return_details, **predict_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False

    system_one = predict

    def route_batch(self, requests: Sequence[Dict[str, Any]]) -> List[RouteDecision]:
        """Route a heterogeneous request batch without loading any checkpoints.

        Each request is a mapping with ``state`` and ``questions`` plus the same optional
        routing overrides accepted by :meth:`route`: ``model``, ``task``, ``lang`` and
        ``lang_guess``. The returned decisions preserve input order.

        This is intentionally separate from inference so callers can inspect or aggregate
        routing decisions before paying model-load cost.
        """
        if not isinstance(requests, SequenceABC) or isinstance(requests, (str, bytes)):
            raise TypeError("requests must be a sequence of request dictionaries")

        decisions: List[RouteDecision] = []
        for i, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError("request %d must be a dict, got %s" % (i, type(request).__name__))
            if "state" not in request:
                raise ValueError("request %d is missing required key 'state'" % i)
            if "questions" not in request:
                raise ValueError("request %d is missing required key 'questions'" % i)

            questions = request["questions"]
            if not isinstance(questions, dict):
                raise TypeError(
                    "request %d 'questions' must be a dict, got %s"
                    % (i, type(questions).__name__)
                )

            decisions.append(
                self.route(
                    request["state"],
                    questions,
                    model=request.get("model"),
                    task=request.get("task"),
                    lang=request.get("lang"),
                    lang_guess=request.get("lang_guess"),
                )
            )

        return decisions

    def predict_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        batch_size: Optional[int] = None,
        hooks_timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Route and execute a heterogeneous request batch with minimal model churn.

        Requests are routed first and grouped by checkpoint. Within each checkpoint,
        requests that share the same question schema are passed to
        ``Agent.predict_batch`` so their states can share forward passes. Results are
        then restored to the original request order.

        Requests may independently specify ``model``, ``task``, ``lang`` or
        ``lang_guess`` and may use different question schemas.

        Router-level predict hooks run per request, as ``predict`` runs them: each request
        gets its own ``PredictContext``, so ``on_predict_start`` can replace that request's
        state, questions or token budget, or ``ctx.skip(...)`` it, and ``on_predict_end``
        sees and may replace its result. Requests are grouped for the forward pass after
        their start hooks have run. If the batch fails, every request whose start hook ran
        and that has no result gets ``on_error``, then every started request gets
        ``on_predict_end``, before the exception propagates.

        Args:
            requests: Sequence of request dictionaries. Every item requires ``state`` and
                ``questions`` and may include ``model``, ``task``, ``lang`` or
                ``lang_guess`` overrides.
            batch_size: Optional maximum number of states per Agent forward-pass batch.
            hooks_timeout: Override the Router's ``hooks_timeout`` for this call.

        Returns:
            One normal Router prediction result per request, in the same order as the input.
        """
        decisions = self.route_batch(requests)
        if not decisions:
            return []

        # Dict insertion order preserves the order in which model groups first appear. This
        # keeps cache effects deterministic while collapsing an arbitrarily interleaved
        # workload to at most one load per routed checkpoint for this call.
        groups: Dict[str, List[int]] = {}
        for i, decision in enumerate(decisions):
            # Indexed, not `.model`: an on_route hook may replace the decision with a plain dict,
            # which `predict` accepts too.
            groups.setdefault(decision["model"], []).append(i)

        results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
        # `compose_hooks`, not `list(self.hooks)`: this is the composition `predict` uses at its
        # own dispatch site, and it is what merges in `set_default_hooks`. Reading the instance
        # list alone silently dropped every process-wide default from the batched path while
        # keeping them on `predict`, so a default audit or metrics hook saw no Router-level event
        # for a request that arrived through `predict_batch`.
        active = compose_hooks(self.hooks)
        raise_errors = self.hooks_raise
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)

        for model_name, indices in groups.items():
            agent = self.load(model_name)
            started: List[PredictContext] = []
            try:
                # One context per request, built and started the way `predict` does it, so a
                # start hook sees -- and can redact, rewrite or skip -- each request before it
                # joins a shared forward pass.
                for i in indices:
                    ctx = PredictContext(states=[requests[i]["state"]], questions=requests[i]["questions"],
                                         decision=dict(decisions[i]), model=model_name, agent=agent,
                                         router=self)
                    started.append(ctx)
                    dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors,
                             lock=self._hooks_lock, timeout=timeout)

                # Agent.predict_batch evaluates one shared question schema and token budget over
                # many states. Preserve Router's heterogeneous-request API by splitting each
                # checkpoint group again on what the start hooks left, so a rewritten question
                # set or `ctx.max_len` only applies to its own request.
                question_groups: List[Dict[str, Any]] = []
                for i, ctx in zip(indices, started):
                    if ctx.results is not None:
                        # A cache hit short-circuits inference; keep the `routing` key predict adds.
                        for result in ctx.results:
                            if isinstance(result, dict):
                                result.setdefault("routing", dict(decisions[i]))
                        continue
                    # Pass token-budget overrides only when set, as `predict` does, so an
                    # Agent-like object that does not accept them still works.
                    overrides = {key: value for key, value in (("max_len", ctx.max_len),
                                                               ("head_max_len", ctx.head_max_len))
                                 if value is not None}
                    # `predict` forwards the language of the request so the agent can apply its
                    # per-language temperatures; the batched path forwarded only the token
                    # budgets, so the same request scored differently depending on the entry
                    # point. Only computed for an agent that actually carries them: `lang` is
                    # otherwise unused, and adding it to the group key would split a group that
                    # shares one forward pass today.
                    lang_key = None
                    if getattr(agent, "lang_temperatures", None):
                        lang_key = requests[i].get("lang")
                        if lang_key is None:
                            detection = decisions[i].get("detection") or {}
                            lang_key = detection.get("language")
                    # Order-sensitive at every nesting level (#166): options are positional, so two
                    # equal schemas with different key orders must not share a group.
                    schema = _question_schema(ctx.questions)
                    for group in question_groups:
                        if (group["schema"] == schema and group["overrides"] == overrides
                                and group["lang"] == lang_key):
                            group["items"].append((i, ctx))
                            break
                    else:
                        question_groups.append({
                            "questions": ctx.questions,
                            "schema": schema,
                            "overrides": overrides,
                            "lang": lang_key,
                            "items": [(i, ctx)],
                        })

                for group in question_groups:
                    items = group["items"]
                    batch_kwargs = dict(group["overrides"])
                    if group["lang"] is not None:
                        batch_kwargs["lang"] = group["lang"]
                    skip = _SKIP_DEFAULTS.set(True)
                    try:
                        batch_results = agent.predict_batch(
                            [ctx.states[0] for _, ctx in items],
                            group["questions"],
                            batch_size=batch_size,
                            **batch_kwargs,
                        )
                    except TypeError as e:
                        # Same tolerance `predict` has for an Agent-like object whose
                        # `predict_batch` predates the `lang` argument.
                        if batch_kwargs.get("lang") is not None and "unexpected keyword argument 'lang'" in str(e):
                            batch_kwargs.pop("lang")
                            batch_results = agent.predict_batch(
                                [ctx.states[0] for _, ctx in items],
                                group["questions"],
                                batch_size=batch_size,
                                **batch_kwargs,
                            )
                        else:
                            raise
                    finally:
                        _SKIP_DEFAULTS.reset(skip)

                    if len(batch_results) != len(items):
                        raise RuntimeError(
                            "internal error: Agent.predict_batch returned %d results for %d states"
                            % (len(batch_results), len(items))
                        )

                    for (i, ctx), result in zip(items, batch_results):
                        result["routing"] = dict(decisions[i])
                        ctx.results = [result]
            except BaseException as exc:
                # Every started request is ended, so a hook that opens something in start (a
                # span, an in-flight count) always sees the matching end. A request that already
                # has its result keeps it; the rest failed with the batch.
                for ctx in started:
                    if ctx.results is None:
                        ctx.error = exc
                        try:
                            dispatch(active, "on_error", ctx, raise_errors=raise_errors,
                                     lock=self._hooks_lock, timeout=timeout)
                        except BaseException as hook_exc:
                            exc.__context__ = hook_exc
                try:
                    self._end_contexts(active, started, raise_errors, timeout)
                except BaseException as hook_exc:
                    exc.__context__ = hook_exc
                raise

            self._end_contexts(active, started, raise_errors, timeout)
            for i, ctx in zip(indices, started):
                results[i] = ctx.results[0]

        # Every input index is assigned exactly once by construction. Keep this assertion local
        # so a future refactor cannot silently return a partially-filled batch.
        if any(result is None for result in results):
            raise RuntimeError("internal error: batch execution did not produce every result")

        return [result for result in results if result is not None]

    predict_many = predict_batch

    def _end_contexts(self, active: List[Any], contexts: List[PredictContext], raise_errors: bool,
                      timeout: Optional[float] = None) -> None:
        """Finish each request of a batch the way `predict`'s `finally` finishes one.

        Every context gets its `on_predict_end` even if an earlier one's end hook raises; the
        first such failure is raised afterwards. On a context that already failed, a raising
        end hook is chained onto its error instead, as in `predict`. Timing and usage are set on
        all of them first, so one request's `elapsed_ms` never includes another's end hooks.
        """
        now = time.perf_counter()
        for ctx in contexts:
            ctx.elapsed_ms = (now - ctx.started_at) * 1000.0
            if ctx.results is not None:
                ctx.usage = aggregate_usage(ctx.results)
        first_error: Optional[BaseException] = None
        for ctx in contexts:
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors,
                         lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                elif first_error is None:
                    first_error = hook_exc
        if first_error is not None:
            raise first_error

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (self.loaded, self.max_loaded, self.default)
