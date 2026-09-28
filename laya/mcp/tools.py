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


def _check_noul_labels(name: str, labels: Any) -> None:
    """`labels` renames a noul answer's two model-facing texts; mirror the agent's rule.

    `laya.common._resolve_noul_labels` is the contract: exactly `false` and `true`, each a
    distinct non-empty string. The agent raises `ValueError` on anything else, and the tool
    wrapper turns a `ValueError` into `internal_error` -- the code this layer uses for a tool
    that broke. So the check is repeated here rather than imported, both to keep the message
    in this layer's `questions[...]` voice and to keep `laya.mcp.tools` free of a torch import.
    """
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ToolError(
            "invalid_questions",
            f"questions[{name}].labels must map exactly 'false' and 'true' to distinct "
            f"non-empty strings",
        )
    values = [labels["false"], labels["true"]]
    # The agent requires a str here, and it is checked before the text is used: a number or a
    # bool is not a label, it is a type mistake. Stringifying first would quietly accept what
    # the agent rejects, and the rejection would then arrive too late to be reported as a
    # caller error.
    if not all(isinstance(value, str) for value in values):
        raise ToolError(
            "invalid_questions",
            f"questions[{name}].labels must give 'false' and 'true' string values, got "
            f"{[type(v).__name__ for v in values]}",
        )
    texts = [value.strip() for value in values]
    if not texts[0] or not texts[1] or texts[0] == texts[1]:
        raise ToolError(
            "invalid_questions",
            f"questions[{name}].labels must give 'false' and 'true' distinct non-empty strings",
        )


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
            # A null level is a hole in the rubric. The agent rejects it before encoding, so
            # accepting it here only moved the failure past the point where the tool could
            # still name the question.
            if None in criteria:
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].criteria has a null level at index "
                    f"{list(criteria).index(None)}; give every level a description",
                )
            entry["criteria"] = list(criteria)
        else:  # noul
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise ToolError(
                        "invalid_questions",
                        f"questions[{name}].criteria must be an object when present (noul)",
                    )
                # `render_options` reads these descriptions by name, so a dict keyed any other
                # way is not a noul description at all -- the agent rejects it rather than
                # silently falling back to the default pair.
                keys = {str(k).lower() for k in criteria}
                if not keys <= {"true", "false"}:
                    raise ToolError(
                        "invalid_questions",
                        f"questions[{name}].criteria must be keyed only 'true'/'false' "
                        f"(either or both, omitting it is fine), got {sorted(keys)}. To word "
                        f"the answer differently set 'labels' instead.",
                    )
                entry["criteria"] = {str(k): v for k, v in criteria.items()}
        if "labels" in spec:
            # noul-only, like the agent: on any other type it is a caller mistake.
            if qtype != "noul":
                raise ToolError(
                    "invalid_questions",
                    f"questions[{name}].labels is only supported for noul questions",
                )
            _check_noul_labels(name, spec["labels"])
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


def _normalize_result(result: Any) -> dict:
    if not isinstance(result, dict):
        raise ToolError("internal_error", "predict returned non-object")
    # Router.predict and Agent.system_one both return the system_one payload,
    # which always carries an "answers" object (empty for empty questions).
    try:
        answers = result["answers"]
    except KeyError:
        raise ToolError("internal_error", "predict result has no 'answers' object") from None
    return {"answers": _normalize_answers(answers), "routing": result.get("routing")}


def _reading_device(router: Any, agent: Any, model_name: str, routing: dict | None) -> str | None:
    """Real device of the checkpoint that answered: Agent.device reflects a
    silent GPU -> CPU fallback. Omitted when it cannot be read, rather than
    guessed.
    """
    if model_name != "auto" and agent is not None:
        return agent_device(agent)
    model_used = (routing or {}).get("model")
    if isinstance(model_used, str) and model_used:
        return agent_device(router_agent(router, model_used))
    return None


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
    auto_without_router = False
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
                if agent is None:
                    raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
                # An Agent answers but does not route: run it directly and report
                # a null model below, instead of echoing 'auto' (#444). `task` only
                # chooses between checkpoints, so it has nothing to do here.
                nonlocal auto_without_router
                auto_without_router = True
                return agent.predict(state_d, questions_d,
                                     **{k: v for k, v in budget.items() if k != "task"})
            return router.predict(state_d, questions_d, **budget)
        if agent is not None:
            return agent.predict(state_d, questions_d, **budget)
        if router is None:
            raise ToolError("models_not_ready", "no agent/router loaded")
        return router.predict(state_d, questions_d, model=model_name, **budget)

    started = time.perf_counter()
    result = _run()
    latency_ms = (time.perf_counter() - started) * 1000.0

    norm = _normalize_result(result)
    answers = norm["answers"]
    if auto_without_router:
        routing = {"model": None, "repo": None, "reason": "auto routing without router"}
    elif norm["routing"]:
        routing = norm["routing"]
    elif model_name == "auto":
        # Agent.predict() returns the system_one payload, which carries no 'routing' key
        # because the Agent itself does no routing. 'auto' is a routing directive, not a
        # checkpoint name, so it must never be echoed back as the answering model (#444).
        routing = {"model": None, "repo": None, "reason": "auto routing without router"}
    else:
        routing = {"model": model_name, "repo": None, "reason": "explicit model"}
    # With no routing decision there is no checkpoint to read the device from.
    device = None if auto_without_router else _reading_device(router, agent, model_name, norm["routing"])
    out: dict[str, Any] = {
        "answers": answers,
        "routing": routing,
        "latency_ms": round(latency_ms, 3),
    }
    if device:
        out["device"] = device
    return out


def _decision_to_dict(decision: Any) -> dict:
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
    return _decision_to_dict(router.route(state_d, questions_d, **overrides))


# Where the reusable shortlist embedder lives. Kept private and on the agent, so the
# cache's lifetime is the answering checkpoint's rather than this process's.
_EMBED_CACHE_ATTR = "_laya_mcp_shortlist_embed_fn"


def _detected_language(detection: Any) -> str | None:
    """The language a route decision detected, or None when it reported none.

    Accepts either shape ``Router.route`` returns -- a ``RouteDecision`` (a dict)
    or a plain object -- and tolerates a decision that carries no detection at all,
    an explicit-model decision being the normal case.
    """
    if detection is None:
        return None
    if isinstance(detection, dict):
        lang = detection.get("language")
    else:
        lang = getattr(detection, "language", None)
    if isinstance(lang, str) and lang:
        return lang
    return None


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


def _embed_fn_with_cache(agent: Any, embed_fn_from_agent: Any, cached_embed_fn: Any) -> Any:
    """An embedder for ``agent`` whose rows survive between shortlist calls.

    ``predict_shortlist`` embeds the query plus every option text on each call, so a
    fixed option list is re-encoded on every request even though it never changes.
    The ``cached_embed_fn`` wrapper from #405 fixes that, but only if the same
    wrapper is reused across calls.

    The wrapper is stored on the answering agent itself, so its lifetime is exactly
    that checkpoint's: another checkpoint, or the same checkpoint reloaded into a
    fresh ``Agent``, starts with an empty cache and can never reuse another
    checkpoint's rows. No module-level state is involved, and the cache stays bounded
    by ``cached_embed_fn``'s own LRU limit.
    """
    cached = getattr(agent, _EMBED_CACHE_ATTR, None)
    if cached is not None:
        return cached
    cached = cached_embed_fn(embed_fn_from_agent(agent))
    try:
        setattr(agent, _EMBED_CACHE_ATTR, cached)
    except (AttributeError, TypeError):
        # An agent that will not take attributes simply gets no reuse between calls,
        # which is the behaviour before this cache existed.
        pass
    return cached


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
    from laya.shortlist import (
        DEFAULT_SHORTLIST_K,
        cached_embed_fn,
        embed_fn_from_agent,
        predict_shortlist,
    )

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
        routing = _decision_to_dict(decision)
        routed = routing["model"]
        detection = (decision.get("detection") if isinstance(decision, dict)
                     else getattr(decision, "detection", None))
        if not isinstance(routed, str) or not routed:
            raise ToolError("internal_error", "router.route returned no model")
        predict_target = router
        # The route already happened here, so the forward pass pins that checkpoint instead of
        # routing a second time.
        predict_kwargs: dict[str, Any] = {"model": routed, **forward_overrides, **budget}
        # Answering with an explicit model= makes predict re-route, and an
        # explicit-model decision carries no detection, so the language detected
        # here would otherwise be lost and the checkpoint's per-language
        # temperatures would not apply the way they do in laya_predict. Forward
        # it when the route reported one and the caller did not name one.
        detected = _detected_language(detection)
        if detected is not None and "lang" not in predict_kwargs:
            predict_kwargs["lang"] = detected
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
            embed_fn = _embed_fn_with_cache(embed_agent, embed_fn_from_agent, cached_embed_fn)
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
    preset_builder: Callable[..., dict] | None = None,
) -> dict:
    """Run a built-in workflow preset (guard / moderation / triage / model_router / email).

    A preset's questions read one named field of the state -- ``guard`` asks about `` `prompt` ``,
    ``triage`` about `` `message` `` -- and a caller that hands over its text under any other key is
    asked to trust an answer about a field that is not there. So a state that is one string gets
    placed under the field the questions actually name; anything richer than that is the caller's
    shape and is passed through untouched.

    The preset fixes the questions, not the route or the budget, so the per-call controls a
    hand-written :func:`laya_predict` takes are available here too. For email and triage,
    ``lang="sv"`` also selects Swedish instructions and descriptions.
    """
    preset_name = validate_preset(preset)
    state_d = validate_state(state)
    if preset_builder is None:
        raise ToolError("internal_error", "preset_builder is not configured")
    builder_kwargs = {}
    if preset_name in ("email", "triage") and isinstance(lang, str):
        language = lang.strip().lower().replace("_", "-").split("-", 1)[0]
        if language == "sv":
            builder_kwargs["language"] = lang
    questions = preset_builder(PRESETS[preset_name], **builder_kwargs)
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


# --- batch tools ----------------------------------------------------------

def _validate_batch_model(value: Any, where: str) -> str | None:
    """``model`` as a Router routing override: "auto" (or absent) means None so
    Router.route resolves the checkpoint per request; a name pins it. Kept
    separate from ``validate_model``, whose "auto" means "answer via
    router.predict" on the single-request path.
    """
    if value is None:
        return None
    try:
        name = validate_model(value)
    except ToolError as error:
        raise ToolError("invalid_model", "%s['model']: %s" % (where, error)) from None
    return None if name == AUTO else name


def _validate_batch_str(value: Any, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ToolError("invalid_request", f"{where} must be a non-empty string, got {value!r}")
    return value


def _validate_batch_item(request: Any, i: int) -> dict:
    where = "requests[%d]" % i
    if not isinstance(request, dict):
        raise ToolError("invalid_request", "%s must be an object with 'state' and 'questions'" % where)
    item: dict[str, Any] = {
        "state": validate_state(request.get("state")),
        "questions": validate_questions(request.get("questions")),
    }
    if "model" in request:
        model = _validate_batch_model(request["model"], where)
        if model is not None:
            item["model"] = model
    for key in ("task", "lang"):
        if key in request:
            item[key] = _validate_batch_str(request[key], "%s[%r]" % (where, key))
    if "lang_guess" in request:
        item["lang_guess"] = request["lang_guess"]
    return item


def validate_batch_requests(requests: Any) -> list[dict]:
    """Validate a tool payload of many requests, preserving order.

    Same per-item validation as ``laya_predict``/``laya_route`` (state, questions,
    optional model/task/lang/lang_guess overrides), run before any model loads so
    one malformed item fails the whole call instead of a partial batch.
    """
    if not isinstance(requests, list) or not requests:
        raise ToolError(
            "invalid_request",
            "requests must be a non-empty array of {state, questions, model?, task?, lang?, lang_guess?} objects",
        )
    return [_validate_batch_item(request, i) for i, request in enumerate(requests)]


def _validate_batch_size(batch_size: Any) -> int | None:
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ToolError("invalid_batch_size", f"batch_size must be a positive integer, got {batch_size!r}")
    return batch_size


def laya_predict_batch(
    requests: Any,
    batch_size: Any = None,
    *,
    router: Any = None,
) -> dict:
    """Answer many requests in one call: ``Router.predict_batch`` over MCP.

    ``laya_predict`` scores one request per round trip; an agent with many
    tickets to triage pays a routing pass and a separate forward pass every
    time. This tool hands the whole list to ``Router.predict_batch``, which
    groups requests by routed checkpoint and shares forward passes between
    requests with the same question schema, and returns the answers in input
    order. Every item carries the same ``answers``/``routing``/``device``
    fields as ``laya_predict``. Requests are validated up front, so one
    malformed item errors before any model loads.
    """
    items = validate_batch_requests(requests)
    size = _validate_batch_size(batch_size)
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "predict_batch"):
        raise ToolError("internal_error", "router has no predict_batch() method")

    started = time.perf_counter()
    try:
        results = router.predict_batch(items, batch_size=size) if size is not None \
            else router.predict_batch(items)
    except TypeError as exc:
        # A Router without batch support raises at the call itself; anything
        # else is a real bug and must surface unchanged.
        raise ToolError(
            "internal_error",
            "router.predict_batch failed: %s: %s" % (type(exc).__name__, exc),
        ) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    if not isinstance(results, list) or len(results) != len(items):
        raise ToolError(
            "internal_error",
            "predict_batch returned %r results for %d requests" % (
                len(results) if isinstance(results, list) else type(results).__name__, len(items)),
        )

    out_requests: list[dict[str, Any]] = []
    for result in results:
        norm = _normalize_result(result)
        entry: dict[str, Any] = {"answers": norm["answers"], "routing": norm["routing"] or {}}
        device = _reading_device(router, None, "auto", norm["routing"])
        if device:
            entry["device"] = device
        out_requests.append(entry)

    model_counts: dict[str, int] = {}
    for entry in out_requests:
        name = entry["routing"].get("model")
        key = name if isinstance(name, str) and name else "unknown"
        model_counts[key] = model_counts.get(key, 0) + 1

    return {
        "requests": out_requests,
        "model_counts": model_counts,
        "total_latency_ms": round(latency_ms, 3),
        "per_request_latency_ms": round(latency_ms / len(items), 3),
    }


def laya_route_batch(requests: Any, *, router: Any = None) -> dict:
    """Routing decisions for many requests: no forward pass, no checkpoint loads.

    The batch form of ``laya_route``, mirroring ``Router.route_batch``: it
    reports which checkpoint each request *would* answer from so clients can
    inspect or aggregate a workload's routing before paying any load cost.
    """
    items = validate_batch_requests(requests)
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "route_batch"):
        raise ToolError("internal_error", "router has no route_batch() method")
    decisions = router.route_batch(items)
    if not isinstance(decisions, list) or len(decisions) != len(items):
        raise ToolError(
            "internal_error",
            "route_batch returned %r decisions for %d requests" % (
                len(decisions) if isinstance(decisions, list) else type(decisions).__name__, len(items)),
        )
    out = [_decision_to_dict(decision) for decision in decisions]
    model_counts: dict[str, int] = {}
    for entry in out:
        name = entry["model"]
        key = name if isinstance(name, str) and name else "unknown"
        model_counts[key] = model_counts.get(key, 0) + 1
    return {"decisions": out, "model_counts": model_counts}


def laya_decide(
    state: Any,
    schema: Any,
    model: Any = "auto",
    *,
    router: Any = None,
    agent: Any = None,
) -> dict:
    """Answer a JSON-schema-shaped decision and return the decided values.

    The MCP form of ``laya.decide`` (the schema-driven API: an object of enum /
    bounded-integer / boolean properties is turned into Laya questions, answered
    in one forward pass, and projected back onto the schema). MCP clients carry
    JSON, not pydantic classes, so ``schema`` is a JSON schema dictionary. The
    answer is ``values`` -- enum members, integer levels, booleans -- beside the
    per-field ``confidence``/``probabilities`` and the usual routing/device/
    latency metadata, so a client never parses an answer map by hand.
    """
    # Lazy: keeps laya.structured (pure Python, but a module import is still a
    # module import) out of this module's import-time surface.
    from laya.structured import SchemaError, decide

    state_d = validate_state(state)
    model_name = validate_model(model)
    try:
        # `decide` validates the schema itself (SchemaError names the offending
        # path) and projects the answers; calling it with return_details keeps
        # this tool on the core's exact semantics instead of a copied projection.
        started = time.perf_counter()
        if model_name == "auto":
            if router is None:
                raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
            details = decide(router, state_d, schema=schema, return_details=True)
        elif agent is not None:
            details = decide(agent, state_d, schema=schema, return_details=True)
        else:
            if router is None:
                raise ToolError("models_not_ready", "no agent/router loaded")
            details = decide(router, state_d, schema=schema, return_details=True,
                             model=model_name)
    except SchemaError as exc:
        raise ToolError("invalid_schema", str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    answers = _normalize_answers(details.answers)
    routing = details.routing or {"model": model_name, "repo": None, "reason": "explicit model"}
    device = _reading_device(router, agent, model_name, details.routing)
    out: dict[str, Any] = {
        "values": details.values,
        "confidence": details.confidence,
        "probabilities": details.probabilities,
        "routing": routing,
        "latency_ms": round(latency_ms, 3),
    }
    if details.usage:
        out["usage"] = details.usage
    if device:
        out["device"] = device
    return out
