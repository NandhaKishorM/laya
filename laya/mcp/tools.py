"""Thin tool layer over laya. Lay-a calls go through injectable hooks for tests."""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol, Sequence

from ..presets import state_field
from .device import agent_device, device_report, router_agent


class ToolError(Exception):
    """Raised for user-facing tool failures. Message is safe to return to the LLM."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AgentLike(Protocol):
    def predict(self, state: dict, questions: dict) -> dict: ...


PRESETS: dict[str, str] = {
    "guard": "guard_questions",
    "moderation": "moderation_questions",
    "triage": "triage_questions",
    "model_router": "router_questions",
    "email": "email_questions",
}

# `router` is what the CLI calls this preset, and it is the word a caller reaches for first;
# `model_router` stayed because clients already have it in their prompts. Both name one preset,
# and the canonical key is what comes back in the result.
PRESET_ALIASES: dict[str, str] = {"router": "model_router"}


def get_available_presets() -> dict[str, dict[str, Any]]:
    """The built-in presets a ``laya_preset`` call can name, with what each one reads.

    The state field comes from the questions themselves rather than from a table beside them, so
    this cannot drift from the presets: a caller that has to hand-build a state (and the MCP
    ``tools/list`` description, which is built from here) sees the field each preset actually asks
    about. Builders are resolved through ``laya`` at call time, the way ``laya.mcp.server`` resolves
    them, so a preset added in a newer version shows up here as soon as it is importable.
    """
    import laya

    aliases: dict[str, list[str]] = {}
    for alias, target in PRESET_ALIASES.items():
        aliases.setdefault(target, []).append(alias)

    out: dict[str, dict[str, Any]] = {}
    for name, attr in sorted(PRESETS.items()):
        entry: dict[str, Any] = {"questions": attr}
        if name in aliases:
            entry["aliases"] = sorted(aliases[name])
        builder = getattr(laya, attr, None)
        if builder is not None:
            questions = builder()
            entry["n_questions"] = len(questions)
            field = state_field(questions)
            if field is not None:
                entry["state_field"] = field
        out[name] = entry
    return out


VALID_TYPES = {"choice", "score", "noul"}
# `auto` is this layer's own sentinel -- "route it, do not pin a checkpoint". Every other value is a
# checkpoint name, and the registry of those (names, aliases, casing) is core's: `laya.router` runs
# every model argument through `normalise_name` before it loads anything, which is the same call the
# tools below end up making through `router.predict(model=...)`. A second list here could only ever
# be narrower than that one, so it is not repeated.
AUTO = "auto"


def validate_questions(questions: Any) -> dict:
    if not isinstance(questions, dict) or not questions:
        raise ToolError(
            "invalid_questions",
            "questions must be a non-empty JSON object keyed by question name",
        )
    cleaned: dict[str, Any] = {}
    for name, spec in questions.items():
        if not isinstance(name, str) or not name:
            raise ToolError("invalid_questions", f"question name must be a non-empty string: {name!r}")
        if not isinstance(spec, dict):
            raise ToolError("invalid_questions", f"questions[{name}] must be an object")
        qtype = spec.get("type")
        if qtype not in VALID_TYPES:
            raise ToolError(
                "invalid_questions",
                f"questions[{name}].type must be one of {sorted(VALID_TYPES)}, got {qtype!r}",
            )
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ToolError(
                "invalid_questions",
                f"questions[{name}].instructions must be a non-empty string",
            )
        entry: dict[str, Any] = {"type": qtype, "instructions": instructions}
        criteria = spec.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].criteria must be a non-empty object of label -> description",
                )
            entry["criteria"] = {str(k): v for k, v in criteria.items()}
        elif qtype == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].criteria must be a non-empty list of rubric levels",
                )
            entry["criteria"] = list(criteria)
        else:  # noul
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise ToolError(
                        "invalid_questions",
                        f"questions[{name}].criteria must be an object when present (noul)",
                    )
                entry["criteria"] = {str(k): v for k, v in criteria.items()}
            if "labels" in spec:
                entry["labels"] = spec["labels"]
        cleaned[name] = entry
    return cleaned


def validate_state(state: Any) -> dict:
    if not isinstance(state, dict) or not state:
        raise ToolError("invalid_state", "state must be a non-empty JSON object")
    return dict(state)


def validate_preset(preset: Any) -> str:
    """Canonical :data:`PRESETS` key for a preset argument.

    The one spelling a caller reaches for that is not a table key is ``router`` -- what the CLI
    calls ``model_router``. Aliases resolve here so both names work, and the canonical key is what
    comes back, so the preset a caller reads is not the one it happened to type.
    """
    name = PRESET_ALIASES.get(preset, preset) if isinstance(preset, str) else preset
    if not isinstance(name, str) or name not in PRESETS:
        # The arguments arrive from a model, so a JSON list or object is a real possibility and
        # belongs in the same invalid_preset as an unknown name.
        raise ToolError(
            "invalid_preset",
            f"preset must be one of {sorted(PRESETS)}, got {preset!r}",
        )
    return name


def validate_model(model: Any) -> str:
    """Canonical checkpoint name for a tool argument, or ``"auto"``.

    Core's ``normalise_name`` is what decides whether something names a checkpoint: it trims,
    lowercases and resolves ``laya.router._ALIASES``. Running the argument through it here means
    this layer cannot reject a name that ``router.predict(model=...)`` would have accepted a few
    lines later, and an alias comes back canonical so the ``routing.model`` a caller reads does not
    depend on how the checkpoint was spelled. Deferred import: nothing else in this module pulls
    torch in, and importing this file is how an MCP client starts the server.
    """
    if model is None:
        return AUTO
    if isinstance(model, str) and model.strip().lower() == AUTO:
        return AUTO
    from laya.router import normalise_name

    try:
        return normalise_name(model)
    except ValueError as error:
        raise ToolError("invalid_model", "%s, or %r" % (error, AUTO)) from None


def validate_task(task: Any) -> str | None:
    """Core's ``task`` override, checked the way core checks it, or ``None`` for "not set".

    ``Router._route`` sends ``task`` through the same ``normalise_name`` as ``model`` and then looks
    the result up in its registry, so a task that names nothing raises ``KeyError`` from inside the
    router -- and over MCP a ``KeyError`` is reported as ``internal_error``, which tells a caller
    nothing about what it typed. Validating here turns that into ``invalid_task`` with the list of
    names that do resolve. The caller's own spelling is what gets forwarded, so ``routing.reason``
    still reads as the request that was made.
    """
    if task is None:
        return None
    from laya.router import normalise_name

    # The remap is core's own expression, copied rather than relied upon through the alias table:
    # `Router._route` turns task="typed_decisions" (the underscore form the CLI and the question ids
    # use) into the hyphenated checkpoint name before normalising it. `_ALIASES` happens to hold the
    # same mapping today, so this costs nothing and keeps the two paths agreeing if it ever goes.
    name = "typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task
    try:
        normalise_name(name)
    except ValueError as error:
        raise ToolError("invalid_task", "%s, or %r" % (error, "typed_decisions")) from None
    return task


def validate_lang(lang: Any) -> str | None:
    """A language code for core's ``lang`` override, or ``None`` for "not set".

    Passed through verbatim: what a code means is core's business -- on the router it decides which
    checkpoint can read the state, on a checkpoint it selects the per-language temperature table --
    and a blank or unknown code falls through to the built-in detection rather than failing. Only
    the type is checked, because a truthy list or object would reach the temperature lookup and
    raise ``AttributeError`` on ``.split`` deep inside a forward pass.
    """
    if lang is None:
        return None
    if not isinstance(lang, str):
        raise ToolError("invalid_lang", "lang must be a language code like 'en' or 'de', got %r" % (lang,))
    return lang


def validate_budget(value: Any, name: str) -> int | None:
    """One of the two per-call token budgets (``max_len``, ``head_max_len``), or ``None``.

    Core slices and compares against these, so a float, a numeric string or a zero does not fail
    cleanly -- it truncates the sequence to nothing or raises ``TypeError`` mid-forward. ``None``
    means "use what the checkpoint was trained with", which is the same thing the CLI's flags mean.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ToolError("invalid_%s" % name, "%s must be a positive integer, got %r" % (name, value))
    return value


def _overrides(task: Any, lang: Any, max_len: Any, head_max_len: Any) -> dict:
    """The per-call controls as keyword arguments, with the unset ones left out.

    Only what the caller actually set is forwarded: an injectable router or agent is not required to
    accept a keyword it was never asked about, and dropping the unset ones keeps every call that
    passes no controls byte-identical to the call it made before they existed.
    """
    values = {"task": task, "lang": lang, "max_len": max_len, "head_max_len": head_max_len}
    return {name: value for name, value in values.items() if value is not None}


def _normalize_answers(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ToolError("internal_error", "predict returned non-object answers")
    return raw


def laya_predict(
    state: Any,
    questions: Any,
    model: Any = "auto",
    *,
    task: Any = None,
    lang: Any = None,
    max_len: Any = None,
    head_max_len: Any = None,
    router: Any = None,
    agent: Any = None,
) -> dict:
    """Typed questions, one forward pass.

    ``router`` is used when model == "auto"; ``agent`` for a direct checkpoint.

    ``task``/``lang`` are the router's own overrides (an explicit ``model`` outranks an explicit
    ``task``, which outranks an explicit ``lang``); ``max_len``/``head_max_len`` override the token
    budget the answering checkpoint was configured with. None of them is forwarded unless set, so a
    router or agent that predates those keywords keeps working.
    """
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)
    budget = _overrides(
        validate_task(task),
        validate_lang(lang),
        validate_budget(max_len, "max_len"),
        validate_budget(head_max_len, "head_max_len"),
    )
    # `task` picks a checkpoint by saying what the work is, so it means nothing once one is pinned:
    # `Router._route` checks an explicit `model` first and never reaches the task, and
    # `Agent.system_one` does not accept the keyword at all. Both would answer as if it were unset,
    # so it is refused here instead.
    if model_name != AUTO and "task" in budget:
        raise ToolError(
            "invalid_task",
            "task routes between checkpoints; with a pinned model there is nothing to route",
        )

    def _run() -> Any:
        if model_name == "auto":
            if router is None:
                raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
            return router.predict(state_d, questions_d, **budget)
        if agent is not None:
            return agent.predict(state_d, questions_d, **budget)
        if router is None:
            raise ToolError("models_not_ready", "no agent/router loaded")
        return router.predict(state_d, questions_d, model=model_name, **budget)

    started = time.perf_counter()
    result = _run()
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not isinstance(result, dict):
        raise ToolError("internal_error", "predict returned non-object")
    # Router.predict and Agent.system_one both return the system_one payload,
    # which always carries an "answers" object (empty for empty questions).
    answers = _normalize_answers(result["answers"])
    routing = result.get("routing") or {"model": model_name, "repo": None, "reason": "explicit model"}
    # Real device of the checkpoint that answered: Agent.device reflects a
    # silent GPU -> CPU fallback. Omitted when it cannot be read, rather than
    # guessed.
    device = None
    if model_name != "auto" and agent is not None:
        device = agent_device(agent)
    else:
        model_used = routing.get("model")
        if isinstance(model_used, str) and model_used:
            device = agent_device(router_agent(router, model_used))
    out: dict[str, Any] = {
        "answers": answers,
        "routing": routing,
        "latency_ms": round(latency_ms, 3),
    }
    if device:
        out["device"] = device
    return out


def laya_route(
    state: Any,
    questions: Any,
    *,
    model: Any = None,
    task: Any = None,
    lang: Any = None,
    router: Any = None,
) -> dict:
    """Routing decision only: no forward pass.

    Takes the three routing overrides :meth:`laya_predict` takes -- ``model``, ``task``, ``lang`` --
    so "which checkpoint would this go to?" can be asked under a pin without running anything.
    Passing a predict call's controls here reproduces the ``routing`` block it returned, which is
    what makes a route a cheap explanation of a decision rather than a different decision.
    """
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    # `auto` is this layer's word for "do not pin"; core has no such name, so it becomes an absent
    # override rather than a ValueError from normalise_name.
    model_name = validate_model(model)
    task_name = validate_task(task)
    # Same rule as the decision tools: `_route` checks an explicit model first and never reaches
    # the task, so a call that sets both is asking a question with two answers.
    if model_name != AUTO and task_name is not None:
        raise ToolError(
            "invalid_task",
            "task routes between checkpoints; with a pinned model there is nothing to route",
        )
    overrides: dict[str, Any] = {}
    if model_name != AUTO:
        overrides["model"] = model_name
    if task_name is not None:
        overrides["task"] = task_name
    lang_code = validate_lang(lang)
    if lang_code is not None:
        overrides["lang"] = lang_code
    # Everything the caller typed is checked before the server is asked for a checkpoint.
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "route"):
        raise ToolError("internal_error", "router has no route() method")
    decision = router.route(state_d, questions_d, **overrides)
    if isinstance(decision, dict):
        return {
            "model": decision.get("model"),
            "repo": decision.get("repo"),
            "reason": decision.get("reason"),
        }
    return {
        "model": getattr(decision, "model", None),
        "repo": getattr(decision, "repo", None),
        "reason": getattr(decision, "reason", None),
    }


def _resident_or_load(router: Any, name: str) -> Any:
    """The resident agent for checkpoint ``name``, loading on demand when possible.

    Read-only lookup first: ``router_agent`` never calls ``load()``. A miss
    falls back to ``Router.load``, the same on-demand build ``Router.predict``
    performs after routing, so a lazily preloaded server (LAYA_PRELOAD=0, or a
    checkpoint outside LAYA_MODELS) behaves exactly like ``laya_predict``.
    """
    resident = router_agent(router, name)
    if resident is not None:
        return resident
    load = getattr(router, "load", None)
    if load is None:
        raise ToolError("models_not_ready", f"checkpoint {name!r} is not loaded")
    return load(name)


def laya_shortlist(
    state: Any,
    questions: Any,
    model: Any = "auto",
    k: Any = None,
    *,
    task: Any = None,
    lang: Any = None,
    max_len: Any = None,
    head_max_len: Any = None,
    router: Any = None,
    agent: Any = None,
    embed_fn: Callable[[Sequence[str]], Any] | None = None,
) -> dict:
    """Shortlist many-option choice questions to ``k`` labels, then one predict.

    The shared guardrails tell clients not to run >20-option choice questions
    without shortlisting; this tool is that shortlisting (the in-process
    ``laya.shortlist.predict_shortlist`` pattern over MCP). Embeddings come
    from the answering checkpoint's own encoder (``embed_fn_from_agent``), so
    no extra model is downloaded; ``embed_fn`` is injectable for tests or for
    a dedicated bi-encoder.

    ``routing`` reports the real route decision in auto mode (the forward
    pass then runs with an explicit ``model=``, so routing happens once).

    ``head_max_len`` matters more here than anywhere else: shortlisting exists
    because a large label set shares that budget, and narrowing to ``k`` is only
    half of the fix. The budget override reaches the answering forward pass;
    ``task``/``lang`` reach the route that chose the checkpoint.
    """
    # Lazy: keeps numpy/shortlist out of module import for laya.mcp.tools.
    from laya.shortlist import DEFAULT_SHORTLIST_K, embed_fn_from_agent, predict_shortlist

    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)
    routing_overrides = _overrides(validate_task(task), validate_lang(lang), None, None)
    budget = _overrides(None, None, validate_budget(max_len, "max_len"),
                        validate_budget(head_max_len, "head_max_len"))
    # Same rule as `laya_predict`: a checkpoint is already pinned, so `task` has nothing left to
    # decide -- and it is a routing keyword, which the answering `system_one` does not accept.
    if model_name != AUTO and "task" in routing_overrides:
        raise ToolError(
            "invalid_task",
            "task routes between checkpoints; with a pinned model there is nothing to route",
        )
    # `lang` survives pinning because it means two things: it can route, and on the answering
    # checkpoint it selects the per-language temperature table.
    forward_overrides = {name: value for name, value in routing_overrides.items() if name != "task"}
    if k is None:
        k = DEFAULT_SHORTLIST_K
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ToolError("invalid_k", f"k must be a positive integer, got {k!r}")

    routing: dict[str, Any]
    if model_name == "auto":
        if router is None:
            raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
        if not hasattr(router, "route"):
            raise ToolError("internal_error", "router has no route() method")
        decision = router.route(state_d, questions_d, **routing_overrides)
        if isinstance(decision, dict):
            routed = decision.get("model")
            routing = {
                "model": routed,
                "repo": decision.get("repo"),
                "reason": decision.get("reason"),
            }
        else:
            routed = getattr(decision, "model", None)
            routing = {
                "model": routed,
                "repo": getattr(decision, "repo", None),
                "reason": getattr(decision, "reason", None),
            }
        if not isinstance(routed, str) or not routed:
            raise ToolError("internal_error", "router.route returned no model")
        predict_target = router
        # The route already happened here, so the forward pass pins that checkpoint instead of
        # routing a second time.
        predict_kwargs: dict[str, Any] = {"model": routed, **forward_overrides, **budget}
        embed_agent = _resident_or_load(router, routed)
    else:
        routing = {"model": model_name, "repo": None, "reason": "explicit model"}
        if agent is not None:
            predict_target = agent
            predict_kwargs = dict(forward_overrides)
            embed_agent = agent
        else:
            if router is None:
                raise ToolError("models_not_ready", "no agent/router loaded")
            predict_target = router
            predict_kwargs = {"model": model_name, **forward_overrides}
            embed_agent = _resident_or_load(router, model_name)
        predict_kwargs.update(budget)

    if embed_fn is None:
        try:
            embed_fn = embed_fn_from_agent(embed_agent)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ToolError(
                "models_not_ready",
                f"cannot build shortlist embeddings from checkpoint {routing['model']!r}: {exc}",
            ) from exc

    started = time.perf_counter()
    result = predict_shortlist(predict_target, state_d, questions_d, embed_fn, k=k, **predict_kwargs)
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not isinstance(result, dict):
        raise ToolError("internal_error", "predict returned non-object")
    answers = _normalize_answers(result["answers"])
    # The answering checkpoint is the embedding checkpoint in every branch.
    device = agent_device(embed_agent)
    out: dict[str, Any] = {
        "answers": answers,
        "routing": routing,
        "shortlist": result.get("shortlist") or {},
        "latency_ms": round(latency_ms, 3),
    }
    if device:
        out["device"] = device
    return out


def laya_preset(
    preset: Any,
    state: Any,
    *,
    task: Any = None,
    lang: Any = None,
    max_len: Any = None,
    head_max_len: Any = None,
    router: Any = None,
    agent: Any = None,
    preset_builder: Callable[[str], dict] | None = None,
) -> dict:
    """Run a built-in workflow preset (guard / moderation / triage / model_router / email).

    A preset's questions read one named field of the state -- ``guard`` asks about `` `prompt` ``,
    ``triage`` about `` `message` `` -- and a caller that hands over its text under any other key is
    asked to trust an answer about a field that is not there. So a state that is one string gets
    placed under the field the questions actually name; anything richer than that is the caller's
    shape and is passed through untouched.

    The preset fixes the questions, not the route or the budget, so the per-call controls a
    hand-written :func:`laya_predict` takes are available here too -- most usefully ``lang``, since
    a preset's instructions are English text whatever state they read.
    """
    preset_name = validate_preset(preset)
    state_d = validate_state(state)
    if preset_builder is None:
        raise ToolError("internal_error", "preset_builder is not configured")
    questions = preset_builder(PRESETS[preset_name])
    field = state_field(questions)
    if field is not None and field not in state_d and len(state_d) == 1:
        (key, value), = state_d.items()
        if isinstance(value, str):
            state_d = {field: value}
    return laya_predict(
        state_d,
        questions,
        model="auto",
        task=task,
        lang=lang,
        max_len=max_len,
        head_max_len=head_max_len,
        router=router,
        agent=agent,
    )


def laya_status(*, router: Any = None, loaded: list[str] | None = None, preload: bool = True) -> dict:
    report = device_report()
    versions: dict[str, str | None] = {
        "laya": None,
        "torch": report.get("torch_version"),
        "transformers": None,
    }
    for pkg in ("laya", "transformers"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = None

    if loaded is None and router is not None:
        try:
            loaded = list(getattr(router, "loaded", []) or [])
        except Exception:
            loaded = []

    # Real device of every loaded checkpoint (Agent.device reflects a silent
    # GPU -> CPU fallback). The top-level "device" is that fact when something
    # is loaded; before any load it is the configured preference (LAYA_DEVICE
    # or auto), which "device_is_preference" flags as such.
    checkpoint_devices: dict[str, str] = {}
    for name in (loaded or []):
        device = agent_device(router_agent(router, name))
        if device:
            checkpoint_devices[name] = device
    actual = next(iter(checkpoint_devices.values()), None)

    return {
        **report,
        "device": actual or report["device"],
        "device_is_preference": actual is None,
        "checkpoint_devices": checkpoint_devices,
        "loaded": list(loaded or []),
        "router_preload": bool(preload),
        "router_ready": router is not None,
        "package_versions": versions,
    }
