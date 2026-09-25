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

    norm = _normalize_result(result)
    answers = norm["answers"]
    routing = norm["routing"] or {"model": model_name, "repo": None, "reason": "explicit model"}
    device = _reading_device(router, agent, model_name, norm["routing"])
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


def laya_route(state: Any, questions: Any, *, router: Any = None) -> dict:
    """Routing decision only: no forward pass."""
    state_d = validate_state(state)
    questions_d = validate_questions(questions)
    if router is None:
        raise ToolError("models_not_ready", "Router is not loaded")
    if not hasattr(router, "route"):
        raise ToolError("internal_error", "router has no route() method")
    return _decision_to_dict(router.route(state_d, questions_d))


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
        routing = _decision_to_dict(router.route(state_d, questions_d))
        routed = routing["model"]
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


# --- batch tools ----------------------------------------------------------

def _validate_batch_model(value: Any, where: str) -> str | None:
    """``model`` as a Router routing override: "auto" (or absent) means None so
    Router.route resolves the checkpoint per request; a name pins it. Kept
    separate from ``validate_model``, whose "auto" means "answer via
    router.predict" on the single-request path.
    """
    if value is None or value == "auto":
        return None
    if value not in VALID_MODELS:
        raise ToolError(
            "invalid_model",
            f"{where}['model'] must be one of {sorted(VALID_MODELS)}, got {value!r}",
        )
    return value


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
