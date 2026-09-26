"""Thin tool layer over laya. Lay-a calls go through injectable hooks for tests."""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol, Sequence

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
}

VALID_TYPES = {"choice", "score", "noul"}
VALID_MODELS = {"auto", "english", "multilingual", "typed-decisions"}


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
    if preset not in PRESETS:
        raise ToolError(
            "invalid_preset",
            f"preset must be one of {sorted(PRESETS)}, got {preset!r}",
        )
    return preset


def validate_model(model: Any) -> str:
    if model is None:
        return "auto"
    if model not in VALID_MODELS:
        raise ToolError(
            "invalid_model",
            f"model must be one of {sorted(VALID_MODELS)}, got {model!r}",
        )
    return model


def _normalize_answers(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ToolError("internal_error", "predict returned non-object answers")
    return raw


def laya_predict(
    state: Any,
    questions: Any,
    model: Any = "auto",
    *,
    router: Any = None,
    agent: Any = None,
) -> dict:
    """Typed questions, one forward pass.

    ``router`` is used when model == "auto"; ``agent`` for a direct checkpoint.
    """
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)

    def _run() -> Any:
        if model_name == "auto":
            if router is None:
                raise ToolError("models_not_ready", "Router is not loaded (auto mode)")
            return router.predict(state_d, questions_d)
        if agent is not None:
            return agent.predict(state_d, questions_d)
        if router is None:
            raise ToolError("models_not_ready", "no agent/router loaded")
        return router.predict(state_d, questions_d, model=model_name)

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


def laya_route(state: Any, questions: Any, *, router: Any = None) -> dict:
    """Routing decision only: no forward pass."""
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "route"):
        raise ToolError("internal_error", "router has no route() method")
    decision = router.route(state_d, questions_d)
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
    """
    # Lazy: keeps numpy/shortlist out of module import for laya.mcp.tools.
    from laya.shortlist import DEFAULT_SHORTLIST_K, embed_fn_from_agent, predict_shortlist

    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    model_name = validate_model(model)
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
        decision = router.route(state_d, questions_d)
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
        predict_kwargs: dict[str, Any] = {"model": routed}
        embed_agent = _resident_or_load(router, routed)
    elif agent is not None:
        routing = {"model": model_name, "repo": None, "reason": "explicit model"}
        predict_target = agent
        predict_kwargs = {}
        embed_agent = agent
    else:
        if router is None:
            raise ToolError("models_not_ready", "no agent/router loaded")
        routing = {"model": model_name, "repo": None, "reason": "explicit model"}
        predict_target = router
        predict_kwargs = {"model": model_name}
        embed_agent = _resident_or_load(router, model_name)

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
    router: Any = None,
    agent: Any = None,
    preset_builder: Callable[[str], dict] | None = None,
) -> dict:
    """Run a built-in workflow preset (guard / moderation / triage / model_router)."""
    preset_name = validate_preset(preset)
    state_d = validate_state(state)
    if preset_builder is None:
        raise ToolError("internal_error", "preset_builder is not configured")
    questions = preset_builder(PRESETS[preset_name])
    return laya_predict(
        state_d,
        questions,
        model="auto",
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
