"""HTTP server exposing Laya over TypeSafe Jev's ``/v1/systemone`` wire protocol.

Laya's ``predict()`` output is already schema-compatible with the Jev decision
API -- ``choice`` / ``score`` / ``noul`` answers and a ``{input_tokens,
output_tokens}`` usage block -- so a client written against Jev (for example the
`hs-jev` Haskell client) can point its ``baseUrl`` at this server and keep
working unchanged. All this module adds is the HTTP surface Laya itself does not
ship: a ``POST /v1/systemone`` route, an optional bearer check, and a health
probe.

Configuration is entirely via environment variables so the same entry point
serves a laptop dev run and a systemd unit:

=========================  ============================================  =========
env var                    meaning                                        default
=========================  ============================================  =========
``LAYA_HOST``              bind address                                   0.0.0.0
``LAYA_PORT``              bind port                                      8000
``LAYA_DEVICE``            torch device for every checkpoint              (auto)
``LAYA_PRELOAD``           build the checkpoints at startup, not lazily   1
``LAYA_MODELS``            comma list to preload (english,multilingual,   (all)
                           typed-decisions); empty = every checkpoint
``LAYA_THREADS``           cap torch intra-op threads (CPU inference).    (torch
                           Keep <= physical cores; oversubscribing the     default)
                           logical/hyperthread count is a large regression.
``LAYA_AUTO_TASK``         auto-route to the typed-decisions checkpoint   0
``LAYA_DEFAULT_MODEL``     fallback checkpoint when a state carries no   (english)
                           language evidence; aliases like ml work
``LAYA_MAX_LOADED``        checkpoints kept resident at once. Below what  2
                           routing can choose, one reloads per switch.
``LAYA_API_KEY``           if set, require ``Authorization: Bearer <it>``  (none)
``LAYA_ROOT_PATH``         public URL prefix behind a reverse proxy        (none)
``LAYA_LOG_LEVEL``         uvicorn log level                              info
``LAYA_MAX_CONCURRENT``    cap on requests past auth at once; excess      16
                           gets 503 (see below)
``LAYA_MAX_TOKEN_BUDGET``  cap on per-request max_len / head_max_len       8192
=========================  ============================================  =========

``LAYA_DEVICE`` is a preference, not a guarantee: an ``Agent`` that asks for a
GPU it cannot get falls back to CPU silently and keeps answering correctly.
``/health`` reports where a resident checkpoint actually computes, using the
same helpers the MCP status tool uses (``laya.mcp.device``).

Imports of heavy dependencies (fastapi, uvicorn, torch via Router) are all
deferred into the functions that need them, so ``import laya.serve`` stays cheap
and touches no GPU -- which is what keeps the Nix ``pythonImportsCheck`` honest.
"""
import hmac
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# A failed inference is reported to the client as a fixed 500 so nothing about paths,
# weights or memory state leaks, which leaves the server log as the only place the
# actual cause can appear. Uvicorn configures the root logger, so a module logger
# propagates there without this module setting up any handlers.
_log = logging.getLogger("laya.serve")

# Guardrails for unauthenticated remote input. The state is tokenized once per
# question and collated into one tensor, so an unbounded body can OOM the worker;
# the single-worker pool means one large request would also starve /health.
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50000
MAX_BATCH_STATES = 64
MAX_BODY_BYTES = 2 * 1024 * 1024
# HTTP-only amplification guard; the library keeps its head_max_len-aware budget.
MAX_CHOICE_OPTIONS = 100
MAX_SCORE_LEVELS = 32
MAX_TOTAL_OPTIONS = 512

# Cap on requests past auth at once. Each one can buffer up to MAX_BODY_BYTES
# before inference, so without a bound many concurrent near-cap requests OOM
# the worker even though every request is individually valid (#330).
DEFAULT_MAX_CONCURRENT = 16
# Server-side ceiling on per-request max_len/head_max_len token budget overrides.
DEFAULT_MAX_TOKEN_BUDGET = 8192

# Every control ``Router.predict`` takes beyond ``state`` and ``questions``, split by whether a
# JSON body can mean it. ``tests/test_serve.py`` pins the two lists against
# ``inspect.signature(Router.predict)``, so a control added to the core call has to be placed on
# one side of that line -- forwarded or refused -- before the suite goes green again. The line is
# not drawn by value type: a hook is a callable that runs in the server process, and
# ``hooks_raise`` / ``hooks_timeout`` govern how the hooks the *operator* installed execute, so
# none of those five has a meaning a request could give it. Which checkpoint answers, which
# language it reads, how many tokens it gets and where it abstains are all plain data.
BODY_CONTROLS = ("model", "max_len", "head_max_len", "task", "lang", "lang_guess", "min_confidence")
BODY_REFUSALS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")

# ``Router.predict_batch`` reads two different kinds of control: call-level keyword arguments that
# apply to the whole batch (chunking, padding, the abstention threshold) and per-request keys that
# it lifts off each item of ``requests`` (checkpoint, task, language hint, token budget). The batch
# endpoint keeps them in separate tuples so a caller-visible JSON body -- one set of controls that
# applies to every state it sent -- is turned into both. ``tests/test_serve.py`` pins the two lists
# against ``inspect.signature(Router.predict_batch)`` (call-level) and against the MCP tool's own
# item key list (per-request), so a control added to either side has to be placed on one side of
# this split -- forwarded, refused, or moved to the item tuple -- before the suite goes green.
# ``hooks_timeout`` is refused on the batch path the same way it is on the single path: it belongs
# to ``predict_batch``'s call-level args but governs how the *deployment's* hooks execute, so a
# caller cannot be allowed to shorten or lengthen that deadline from an HTTP body.
BATCH_BODY_CALL_CONTROLS = ("batch_size", "min_confidence", "sort_by_length")
BATCH_BODY_ITEM_CONTROLS = ("max_len", "head_max_len", "task", "lang", "lang_guess")


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _published_model_ids() -> Dict[str, str]:
    """Public Hugging Face ids accepted so a client can name a checkpoint.

    Inverted from `laya.router.STANDALONE_MODELS` at call time rather than typed out: the
    ids a client may send belong to that registry, and a copy here would silently auto-route
    a checkpoint the router already knows. The root bundle is deliberately absent -- the
    documented ``convaiinnovations/laya`` value means "let the Router choose", rather than
    pinning the English checkpoint. Imported here rather than at module scope, the way this
    module's other `laya.router` use does, so ``import laya.serve`` stays cheap.
    """
    from .router import BUNDLE_REPO, STANDALONE_MODELS

    return {repo: name for name, repo in STANDALONE_MODELS.items() if repo != BUNDLE_REPO}


def _resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a client's `model` field onto a Laya checkpoint, or None to auto-route."""
    if not model:
        return None
    published = _published_model_ids().get(str(model).strip().lower())
    if published is not None:
        return published
    from .router import normalise_name

    # normalise_name raises ValueError on anything that is not a known checkpoint
    # or alias, and returns a name from router.DEFAULT_MODELS when it does accept one --
    # so it is the only list of accepted names this needs. A Jev client's `model` field
    # (e.g. "jev-1") is expected to miss; treat that as "no explicit checkpoint" and let
    # the router auto-select.
    try:
        return normalise_name(model)
    except Exception:
        return None


def _resolve_max_concurrent() -> int:
    """Bound on requests past auth at once, from LAYA_MAX_CONCURRENT."""
    raw = os.environ.get("LAYA_MAX_CONCURRENT")
    if not raw:
        return DEFAULT_MAX_CONCURRENT
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT
    return n if n > 0 else DEFAULT_MAX_CONCURRENT


def _resolve_max_token_budget() -> int:
    """Server-side cap on per-request max_len from LAYA_MAX_TOKEN_BUDGET."""
    raw = os.environ.get("LAYA_MAX_TOKEN_BUDGET")
    if not raw:
        return DEFAULT_MAX_TOKEN_BUDGET
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        _log.warning("invalid LAYA_MAX_TOKEN_BUDGET %r; falling back to %d", raw, DEFAULT_MAX_TOKEN_BUDGET)
        return DEFAULT_MAX_TOKEN_BUDGET
    if n <= 0:
        _log.warning("LAYA_MAX_TOKEN_BUDGET must be positive (got %d); falling back to %d", n, DEFAULT_MAX_TOKEN_BUDGET)
        return DEFAULT_MAX_TOKEN_BUDGET
    return n


def _validate_budget_param(body: Dict[str, Any], key: str, max_cap: int) -> Optional[int]:
    """Validate optional integer budget parameters (max_len / head_max_len) from request body (422)."""
    from fastapi import HTTPException

    if key not in body:
        return None
    val = body[key]
    if val is None:
        return None
    if not isinstance(val, int) or isinstance(val, bool):
        raise HTTPException(status_code=422, detail="%s must be an integer" % key)
    if val <= 0:
        raise HTTPException(status_code=422, detail="%s must be a positive integer" % key)
    if val > max_cap:
        raise HTTPException(
            status_code=422,
            detail="%s exceeds server limit (%d > %d)" % (key, val, max_cap),
        )
    return val


def _refuse_body_refusals(body: Dict[str, Any]) -> None:
    """Reject a request that sends one of ``BODY_REFUSALS`` (422) instead of ignoring it.

    A hook is a callable that runs inside ``predict`` -- it can cache a decision, gate one or
    rewrite its state -- and ``laya-serve`` has no way to receive or run one, which is why
    ``laya.integrations.langchain`` already refuses them on a node with a ``base_url``
    (``_reject_remote_hooks``). ``hooks_raise`` and ``hooks_timeout`` belong with them: they say
    how the hooks the deployment installed execute, so honouring a caller's value would let a
    request change server-side behaviour. Today all five are read into the body and dropped, so a
    client that asks for one is told nothing.
    """
    from fastapi import HTTPException

    given = sorted(key for key in BODY_REFUSALS if key in body and body[key] is not None)
    if given:
        raise HTTPException(
            status_code=422,
            detail="%s run inside the server process and cannot be sent to this endpoint; "
                   "install them where laya-serve runs, or drop them" % ", ".join(given))


def _validate_language_param(body: Dict[str, Any], key: str) -> Optional[str]:
    """Return a ``lang`` / ``lang_guess`` value, requiring the code-string form (422).

    ``Router`` reads both through ``_english_from_code``, which stringifies its input, so a JSON
    ``true`` would become the code ``"true"`` -- a real, non-English one -- and decide the
    checkpoint. A callable hint, the other form core accepts, cannot cross an HTTP body. Explicit
    ``null`` stays "no hint", which is what lets a deployment's ``Router(lang_guess=...)`` answer.
    """
    from fastapi import HTTPException

    val = body.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise HTTPException(status_code=422,
                            detail='%s must be a language code string such as "de", or null' % key)
    return val


def _validate_min_confidence(body: Dict[str, Any]) -> Optional[float]:
    """Round-trip the body's ``min_confidence`` through core's validator, reporting its 422.

    Validation happens here rather than inside ``predict`` so a bad threshold is refused before the
    request takes an inference slot, and through ``laya.confidence`` rather than a copy of its
    bounds so the accepted range cannot drift from what the abstention gate itself enforces.
    """
    from fastapi import HTTPException

    if body.get("min_confidence") is None:
        return None
    from .confidence import check_min_confidence

    try:
        return check_min_confidence(body["min_confidence"])
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


def _validate_batch_size_param(body: Dict[str, Any]) -> Optional[int]:
    """Validate the optional ``batch_size`` call argument on the batch endpoint (422).

    ``Router.predict_batch`` reads it as the maximum number of states per Agent forward-pass batch,
    so a non-positive or fractional value would either mean "one state per pass" or crash torch.
    A boolean would be truthy and coerce to 1/0, which is not what the caller meant; refusing here
    matches what the MCP tool's own validator does at ``laya/mcp/tools.py:_validate_batch_size``,
    so the same typo is rejected the same way on both surfaces.
    """
    from fastapi import HTTPException

    if "batch_size" not in body:
        return None
    val = body["batch_size"]
    if val is None:
        return None
    if not isinstance(val, int) or isinstance(val, bool):
        raise HTTPException(status_code=422, detail="batch_size must be an integer")
    if val < 1:
        raise HTTPException(status_code=422,
                            detail="batch_size must be a positive integer, got %r" % (val,))
    return val


def _validate_sort_by_length_param(body: Dict[str, Any]) -> Optional[bool]:
    """Validate the optional ``sort_by_length`` call argument on the batch endpoint (422).

    ``Router.predict_batch`` forwards this verbatim to every ``Agent.predict_batch`` call, and an
    agent that predates the knob drops it. Only the boolean is a meaningful value here -- a string
    would be truthy and take the sort path silently. When the caller asks for ``False`` we still
    forward ``None`` rather than ``False``, because ``predict_batch``'s own default is ``False``:
    an absent argument means "the caller did not ask" and cannot override a deployment that built
    its Router with the knob already on.
    """
    from fastapi import HTTPException

    if "sort_by_length" not in body:
        return None
    val = body["sort_by_length"]
    if val is None:
        return None
    if not isinstance(val, bool):
        raise HTTPException(status_code=422, detail="sort_by_length must be a boolean")
    return val


def _resolve_max_loaded() -> Optional[int]:
    """Resident-checkpoint cap from ``LAYA_MAX_LOADED``; ``None`` leaves it to ``Router``.

    ``Router`` keeps two checkpoints resident, which is exactly the number automatic routing
    picks between. ``LAYA_AUTO_TASK`` adds `typed-decisions` as a third possibility, and a cap
    below the number of checkpoints in play unloads the least recently used one on every
    switch -- #172 measured 20-23 s per request rebuilding a checkpoint on CPU against
    49-136 ms with it resident. The value is left out of the constructor when unset rather
    than defaulted to a copy of ``Router``'s own default here, so the two cannot drift.
    Unparseable or non-positive input falls back the way ``_resolve_max_concurrent`` does: a
    typo in a deployment file should not stop the server from starting.

    Only checkpoints reached on demand are bounded by it: ``Router.preload()`` raises the cap
    to hold whatever it builds, so a lower value never evicts a preloaded checkpoint.
    """
    raw = os.environ.get("LAYA_MAX_LOADED")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _default_model_option() -> Dict[str, str]:
    """Routing fallback from ``LAYA_DEFAULT_MODEL``, as a ``Router`` keyword; unset sends nothing.

    ``Router.default`` answers the two states that carry no language evidence at all: no letters,
    and Latin script too short to identify ("Quero cancelar", "Esqueci minha senha"). The README
    tells a deployment whose traffic is mostly non-English to set ``Router(default="multilingual")``,
    and this is the only way such a deployment can say so without writing its own server. Left out
    of the constructor when unset, so the value cannot drift from ``Router``'s own default -- the
    same reasoning as ``_resolve_max_loaded`` above.

    The name goes through ``normalise_name``, so the accepted set and its aliases are core's and
    not a list restated here. Unlike the numeric knobs, a typo here has no harmless fallback: a
    silently-ignored value would keep routing the ambiguous states to the checkpoint the operator
    just said cannot read them, so this raises and the caller refuses to start rather than serve a
    configuration nobody asked for. ``laya.mcp.server`` turns the same error into a ``ToolError``,
    because a stdio server has no startup to refuse.
    """
    raw = os.environ.get("LAYA_DEFAULT_MODEL")
    if raw is None or not raw.strip():
        return {}
    from .router import normalise_name

    try:
        name = normalise_name(raw)
    except ValueError as error:
        raise ValueError("invalid LAYA_DEFAULT_MODEL %r: %s" % (raw.strip(), error)) from None
    return {"default": name}


def _resolve_port() -> int:
    """Port from LAYA_PORT, validated. Exits with a message instead of a traceback."""
    raw = os.environ.get("LAYA_PORT", "8000")
    try:
        port = int(str(raw).strip())
    except (TypeError, ValueError):
        raise SystemExit("invalid LAYA_PORT %r: must be an integer 1-65535" % (raw,))
    if not 1 <= port <= 65535:
        raise SystemExit("invalid LAYA_PORT %r: must be an integer 1-65535" % (raw,))
    return port


#: Values the walk in `_state_length_lower_bound_over` will examine before giving up. This bounds the
#: values *inspected*, not the total work: `stack.extend` pushes a container's elements without
#: charging them against the budget, so a 2 MiB body of many small values costs ~0.43 ms of walking
#: before falling through to the dump it was going to pay anyway (~6% on top). An ACCEPTED state is
#: capped at MAX_STATE_CHARS, where the walk is ~0.01 ms.
_STATE_PROBE_VALUES = 64


def _state_length_lower_bound_over(state: Any, cap: int) -> int:
    """A measured lower bound on the serialized length, if it already exceeds `cap`; else 0.

    Non-zero only when exceeding `cap` is certain, and 0 for "not proven" -- never the other way
    round, so a 0 sends the state on to the exact `json.dumps` measurement and no verdict changes.

    The value is returned rather than a bool so the 413 can quote a number something measured. It
    understates the true length (see below), so the message is a true statement about the state --
    unlike reporting `cap + 1`, which would answer "state too large (50001 > 50000 chars)" for a
    60 012-character state. That is the same fabricated-count flaw this gate replaces, and an
    earlier revision of this branch reintroduced it.

    One consequence worth knowing: the number in a 413 is therefore the exact serialized length when
    the encoder ran, and this understated bound when it did not. Both are truthful and neither
    overstates, but a client trimming exactly the difference the message names can still be refused.

    The bound: the raw lengths of the string values are a *lower* bound on the JSON length. Escaping
    maps each character to one or more characters and so never shortens; keys, separators, brackets
    and quotes only add; numbers, booleans and nulls contribute at least one character each and are
    counted as zero here. Every one of those makes this an under-estimate, which is the safe
    direction: under-estimating can only fail to refuse, and then the encoder decides.

    Why it is worth having. The dump runs before the gate refuses, so its cost is bounded by
    `MAX_BODY_BYTES` rather than by `cap`: a 2 MiB body -- the largest the streaming cap admits --
    cost 5.68 ms to serialize for a request that is then rejected, which is more than the 2.22 ms
    `json.loads` spent parsing it in the first place. Finding one 2 MiB string instead takes a single
    `len()`. Measured: 5.687 ms -> 0.0003 ms on that state, while every state that IS accepted pays
    at most 0.01 ms more (worst ratio 2.08x, on a 6 us deeply-nested case).
    """
    total = 0
    budget = _STATE_PROBE_VALUES
    stack = [state]
    while stack and budget > 0:
        budget -= 1
        item = stack.pop()
        if isinstance(item, str):
            total += len(item)
            if total > cap:
                return total
        elif type(item) is dict:
            # `.values()` and `extend` stay at C level; the keys are ignored, which keeps this a
            # lower bound. Iterating items in Python here made a 2000-key state 1.66x slower.
            stack.extend(item.values())
        elif type(item) is list or type(item) is tuple:
            stack.extend(item)
        else:
            # EXACT types only, deliberately. A `dict` subclass may override `values()` while
            # `json.dumps` reads the real items, which would let this "lower bound" exceed the true
            # length -- measured: a 13-character state refused as `60000 > 50000`. Anything else,
            # including a subclass, a set or a cycle, is handed to the encoder, which decides the
            # size or raises and becomes the 400 below. `json.loads` only ever builds exact types,
            # so no HTTP request takes this branch.
            return 0
    return 0


def _state_length(state: Any) -> int:
    r"""Length of the state text that will be tokenized; 400 if it has none.

    `laya.common.serialize_state(state)` is what gets tokenized -- the state itself for a string,
    `json.dumps(state, ensure_ascii=False)` for a dict or list, which the agents then strip mask
    tokens out of, and that can only shorten it. `MAX_STATE_CHARS` has to be measured on that
    text, not on `str()`, which is a different length in both directions:

    * `repr` quotes a value with `'` and leaves a `"` inside it one character, where JSON escapes
      it to the two characters `\"`. A `{"body": '"' * 49988}` state has a `len(str())` of exactly
      50000, so it passed the gate, and then serialized to 99988 -- the documented 50000-character
      limit admitting very nearly twice what it says.
    * `repr` renders a zero-width space as the six characters `\u200b` where `ensure_ascii=False`
      writes the one character it is, so a state that serializes to 8344 characters -- a sixth of
      the limit -- was refused with "state too large (50004 > 50000 chars)".

    `json.dumps` inline rather than importing `serialize_state`: `laya.common` imports torch at
    module level and `import laya.serve` must not (tests/test_lazy_import.py). The two are pinned
    to each other in tests/test_serve.py.

    Cost. At the gate's own ceiling the dump is 0.133 ms against 0.080 ms for the `str()` it
    replaces -- orders of magnitude under the forward pass that state then gets, so the serialized
    text is not worth threading through `predict()` to be encoded only once. A state large enough
    for that difference to matter never reaches the dump: `_state_length_lower_bound_over` refuses it
    first, which is what keeps an oversized body from costing more to reject than it did upstream.
    (Best of 200 after 20 warm-up iterations, CPython 3.12 on a 10-core arm64 laptop.)
    """
    from fastapi import HTTPException

    if isinstance(state, str):
        return len(state)  # `serialize_state` returns a string state unchanged
    over = _state_length_lower_bound_over(state, MAX_STATE_CHARS)
    if over:
        # Already past the cap on a lower bound, so serializing the rest cannot change the verdict.
        # The bound is what gets reported: it is a measured number and it understates, so the 413
        # never claims a length larger than the state really has.
        return over
    try:
        return len(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError):
        # A state `serialize_state` cannot render is not a size problem, and the 413 this replaces
        # reported a count nothing had measured: `str()` raises on an integer of over 4300 digits
        # (CPython's own int-to-str guard), and the old `except` answered "state too large (50001
        # > 50000 chars)" for a state of a few kilobytes. No HTTP request reaches that case --
        # `json.loads` builds only JSON types and refuses that integer itself, which
        # `_systemone_inner` already answers 400 -- but in-process callers get here, and a state
        # `json.dumps` refuses (a set, a datetime, a circular reference) used to pass the gate on
        # its `str()` and then fail inside `serialize_state`, reported as a 500 "inference failed".
        raise HTTPException(status_code=400, detail="'state' must be JSON-serializable")


def _check_request_limits(state: Any, questions: Any) -> None:
    """Reject absent or oversized inference requests before tokenization (400/413)."""
    from fastapi import HTTPException

    # `serialize_state(None)` is `json.dumps(None)` == the four characters `null`, so a body with
    # no `state` key, or `"state": null`, was answered as a decision about the literal text
    # "null" -- HTTP 200, and at ~0.94 confidence here, byte-identical to sending `"state":
    # "null"`. Nothing downstream can tell that apart from a real string, so the check has to
    # happen before serialization. The repo's other two surfaces already require a state:
    # examples/server.py declares it as a required field and mcp/tools.py rejects an empty one.
    if state is None:
        raise HTTPException(status_code=400, detail="'state' is required")
    if not isinstance(questions, dict):
        raise HTTPException(status_code=400, detail="'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise HTTPException(status_code=413,
                            detail="too many questions (%d > %d)" % (len(questions), MAX_QUESTIONS))

    total_options = 0
    for qid, question in questions.items():
        if not isinstance(question, dict):
            continue
        crit = question.get("criteria")
        qtype = question.get("type")
        if qtype == "choice" and isinstance(crit, (dict, list)):
            count = len(crit)
            total_options += count
            if count > MAX_CHOICE_OPTIONS:
                raise HTTPException(
                    status_code=413,
                    detail="too many choice options for %r (%d > %d)" % (qid, count, MAX_CHOICE_OPTIONS),
                )
        elif qtype == "score" and isinstance(crit, list):
            count = len(crit)
            total_options += count
            if count > MAX_SCORE_LEVELS:
                raise HTTPException(
                    status_code=413,
                    detail="too many score levels for %r (%d > %d)" % (qid, count, MAX_SCORE_LEVELS),
                )
    if total_options > MAX_TOTAL_OPTIONS:
        raise HTTPException(
            status_code=413,
            detail="too many answer options across questions (%d > %d)" % (total_options, MAX_TOTAL_OPTIONS),
        )

    state_len = _state_length(state)
    if state_len > MAX_STATE_CHARS:
        raise HTTPException(status_code=413,
                            detail="state too large (%d > %d chars)" % (state_len, MAX_STATE_CHARS))


def _check_batch_limits(states: Any, questions: Any) -> None:
    """Validate batch states and questions before inference (400/413)."""
    from fastapi import HTTPException

    if not isinstance(states, list) or len(states) == 0:
        raise HTTPException(status_code=400, detail="'states' must be a non-empty list")
    if len(states) > MAX_BATCH_STATES:
        raise HTTPException(
            status_code=413,
            detail="too many states in batch (%d > %d)" % (len(states), MAX_BATCH_STATES),
        )
    for state in states:
        _check_request_limits(state, questions)
# One surrogate code point in a Python string. Compiled once: a nested body can hold many
# strings, and `re.search` scans each at C speed, where the per-character Python loop this
# replaced cost ~170 ms on a near-cap body on the event loop.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _has_lone_surrogate(value: Any) -> bool:
    """True if any string in the parsed body contains an unpaired surrogate code point.

    `json.loads` accepts a `\\udXXX` escape and builds a `str` holding that code point, which
    cannot be encoded to UTF-8. A *pair* of escapes is combined into one ordinary astral
    character by the decoder, so it never appears here and an emoji still works.

    Walks with an explicit stack rather than recursing: `json.loads` accepts nesting far deeper
    than Python's default recursion limit, so a recursive walk turned a body the parser handles
    into a `RecursionError` -- one 500 replaced by another. The size checks run first, so the
    strings reached here are bounded by `MAX_STATE_CHARS` and `MAX_QUESTIONS`.
    """
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _LONE_SURROGATE.search(item):
                return True
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


async def _read_body_capped(request: Any) -> bytes:
    """Read the request body, refusing to buffer more than ``MAX_BODY_BYTES``.

    ``Content-Length`` cannot be the only gate. It is a value the client chooses,
    and under ``Transfer-Encoding: chunked`` it is absent altogether -- HTTP/2 and
    HTTP/3 have no such header at all -- so a request that simply omits it was
    read into memory in full, whatever its size. The body is streamed here and
    abandoned as soon as it exceeds the cap, so the limit holds for every framing
    rather than only for clients that announce their length honestly.
    """
    from fastapi import HTTPException

    total = 0
    chunks = []
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            # Stop reading rather than draining the rest: the peer is already over
            # the limit and nothing further can make the request acceptable.
            raise HTTPException(status_code=413, detail="request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_json_body(request: Any) -> Any:
    """Read the request body and decode JSON, enforcing size caps and parse error guards."""
    from fastapi import HTTPException

    if request.headers.get("content-length"):
        try:
            if int(request.headers["content-length"]) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError:
            pass
    raw = await _read_body_capped(request)
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        raise HTTPException(status_code=400, detail="request body must be valid JSON")


def _apply_thread_limit():
    """Honour LAYA_THREADS by capping torch's intra-op thread count for CPU
    inference. Returns the value applied, or None if unset/invalid. torch is
    imported only when a limit is actually requested."""
    raw = os.environ.get("LAYA_THREADS")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n <= 0:
        return None
    import torch

    torch.set_num_threads(n)
    return n


def build_router():
    """Build a Router from the environment, preloading unless told otherwise."""
    from .mcp.device import env_device
    from .router import Router

    _apply_thread_limit()
    # `env_device` rather than a raw read: it strips, the way `LAYA_MODELS` two lines
    # below is stripped. A value copied out of a Dockerfile or a `.env` file carries a
    # trailing newline, and torch refuses that as an invalid device string.
    device = env_device()
    models_env = os.environ.get("LAYA_MODELS", "").strip()
    preload_names = [m.strip() for m in models_env.split(",") if m.strip()] or None
    options = {"device": device, "auto_task_detection": _env_bool("LAYA_AUTO_TASK", False)}
    max_loaded = _resolve_max_loaded()
    if max_loaded is not None:
        options["max_loaded"] = max_loaded
    # Resolved before the Router is built: a name `Router` would reject is a configuration error,
    # and `_resolve_port`'s idiom applies -- exit with the message, not a traceback.
    try:
        options.update(_default_model_option())
    except ValueError as error:
        raise SystemExit(str(error)) from None
    router = Router(**options)
    if _env_bool("LAYA_PRELOAD", True):
        router.preload(preload_names)
    return router


def create_app(router: Optional[Any] = None):
    """Build the FastAPI app. Pass a Router to inject one (tests); otherwise one
    is built from the environment (and preloaded) at app-creation time."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, Header, HTTPException, Request

    from .mcp.device import agent_device, resolve_device, router_agent

    if router is None:
        router = build_router()
    api_key = os.environ.get("LAYA_API_KEY") or None

    # Inference is synchronous torch, and a CPU call takes hundreds of milliseconds to
    # seconds, so it must not run on the event loop: one request would stall every
    # other client, `GET /health` included. One worker, because one forward pass at a
    # time is what a single CPU or GPU Agent wants (the Router already guards checkpoint
    # lifecycle, and leaves `Agent.system_one` unguarded deliberately so concurrent
    # predictions can share a checkpoint -- a GPU-shaped choice this endpoint does not
    # rely on). `loop.run_in_executor` is the API the issue asked for.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")
    # Created on first request, not here: an `asyncio.Lock` binds to the loop that is
    # running when it is first awaited, and `create_app` may be called before that loop
    # exists (module scope, TestClient startup, a preload script).
    gate: Optional[asyncio.Lock] = None
    # Admission bound, same late-creation reason as the gate. Checked before any
    # body byte is read and held through inference, so the bodies buffered at
    # once stay bounded no matter how many clients connect (#330). The inference
    # gate is still joined only after the body is complete, so a slow client
    # holds an admission slot but never an inference slot.
    max_concurrent = _resolve_max_concurrent()
    admission: Optional[asyncio.Semaphore] = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            # TestClient, embedded ASGI apps, and process supervisors all need
            # the executor to drain when the app stops.
            pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="laya-serve",
        summary="Laya System-1 decisions over the TypeSafe Jev /v1/systemone protocol",
        root_path=os.environ.get("LAYA_ROOT_PATH", ""),
        lifespan=lifespan,
    )

    # Compared as bytes, not str. `hmac.compare_digest` raises TypeError when a str
    # operand holds a non-ASCII character, and Starlette decodes request headers as
    # latin-1 -- so `Authorization: Bearer s\xe9cret`, which is legal on the wire,
    # made the comparison itself raise. That surfaced as HTTP 500 plus a traceback
    # in the log, reachable by any unauthenticated client with one byte. Encoding
    # both sides first keeps the comparison constant-time and total: every header a
    # client can send now answers 401.
    expected_auth = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""

    def _check_auth(authorization: Optional[str]) -> None:
        if api_key is None:
            return
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(supplied, expected_auth):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        # `device` is where a resident checkpoint really computes, not what was asked for:
        # `Agent.device` reflects the silent GPU -> CPU fallback, so a container that asked
        # for a GPU it did not get says so. With nothing resident it is the configured
        # preference, and `device_is_preference` tells the reader which of the two it is
        # looking at. Same convention, and the same three keys, as `laya_status` over MCP.
        checkpoint_devices: Dict[str, str] = {}
        for name in (router.loaded or []):
            device = agent_device(router_agent(router, name))
            if device:
                checkpoint_devices[name] = device
        actual = next(iter(checkpoint_devices.values()), None)

        # Per-agent CPU-fallback counters, read through the same side-effect-free
        # accessor the MCP status tool uses -- never Router.load(), which reorders the
        # LRU and rebuilds evicted checkpoints. Every access is getattr-guarded so
        # injected routers predating the counters stay health-compatible, like `revisions`.
        # This counts the *scoped* fallback in _infer (a request that demotes to CPU and
        # restores the device); a checkpoint built on CPU because the GPU was never
        # available is reported by `device`, not here -- see #574.
        fallbacks = {}
        for name in router.loaded or []:
            agent = router_agent(router, name)
            fallbacks[name] = {"count": getattr(agent, "cpu_fallback_count", 0),
                               "last_reason": getattr(agent, "last_fallback_reason", None)}
        return {
            "status": "ok",
            "loaded": router.loaded,
            "revisions": getattr(router, "loaded_revisions", {}),
            "device": actual or resolve_device(),
            "device_is_preference": actual is None,
            "checkpoint_devices": checkpoint_devices,
            "cpu_fallbacks": fallbacks,
        }

    @asynccontextmanager
    async def _admit():
        nonlocal admission
        if admission is None:
            admission = asyncio.Semaphore(max_concurrent)
        if admission.locked():
            # Non-blocking: excess load is refused rather than queued, so the
            # buffered bodies stay within the bound above. Retry-After tells
            # well-behaved clients when a slot is likely free: admission turns
            # over at inference speed, so one second is the honest hint.
            raise HTTPException(status_code=503, detail="server busy, try again later",
                                headers={"Retry-After": "1"})
        await admission.acquire()
        try:
            yield
        finally:
            admission.release()

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        async with _admit():
            return await _systemone_inner(request)

    async def _systemone_inner(request: Request):
        nonlocal gate
        body = await _read_json_body(request)
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(status_code=400, detail="request body must be an object with a 'questions' field")
        state = body.get("state")
        questions = body["questions"]
        _check_request_limits(state, questions)
        _refuse_body_refusals(body)
        # After the size checks, so an oversized body is refused before anything walks it, and
        # `MAX_STATE_CHARS`/`MAX_QUESTIONS` bound what the walk can reach. A `\udXXX` escape with
        # no pair is legal JSON that cannot be UTF-8 encoded, so the tokenizer raised `TypeError`
        # from inside `build_sequence` and the client's own mistake came back as a 500
        # "inference failed". A *paired* surrogate is an ordinary astral character (an emoji) by
        # the time `json.loads` is done, so only lone ones are rejected here.
        if _has_lone_surrogate(body):
            raise HTTPException(status_code=400,
                                detail="request body contains an unpaired surrogate escape; "
                                       "those cannot be encoded as UTF-8")
        model = _resolve_model(body.get("model"))
        max_budget_cap = _resolve_max_token_budget()
        max_len = _validate_budget_param(body, "max_len", max_budget_cap)
        head_max_len = _validate_budget_param(body, "head_max_len", max_budget_cap)
        # Each control is sent only when the client sent it: core reads an absent argument as
        # "inherit what the Router was built with", so passing None would override a deployment's
        # own `Router(lang_guess=...)` or abstention threshold with the server's default.
        predict_kwargs = {}
        if max_len is not None:
            predict_kwargs["max_len"] = max_len
        if head_max_len is not None:
            predict_kwargs["head_max_len"] = head_max_len
        # An unknown task is left to `route`, which normalises it through `normalise_name` and
        # raises; the `except ValueError` below turns that into a 422 naming the task, so the
        # accepted set is core's and not a list restated here.
        if body.get("task") is not None:
            predict_kwargs["task"] = body["task"]
        for key in ("lang", "lang_guess"):
            value = _validate_language_param(body, key)
            if value is not None:
                predict_kwargs[key] = value
        min_confidence = _validate_min_confidence(body)
        if min_confidence is not None:
            predict_kwargs["min_confidence"] = min_confidence
        if gate is None:
            gate = asyncio.Lock()
        try:
            # Laya's result is already Jev-shaped: {model, answers, usage, routing}.
            # hs-jev decodes `answers` and `usage` and ignores the rest.
            async with gate:
                loop = asyncio.get_running_loop()
                t0 = time.perf_counter()
                if predict_kwargs:
                    result = await loop.run_in_executor(
                        pool, lambda: router.predict(state, questions, model=model, **predict_kwargs))
                else:
                    result = await loop.run_in_executor(
                        pool, lambda: router.predict(state, questions, model=model))
                infer_ms = (time.perf_counter() - t0) * 1000.0
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    content=result,
                    headers={
                        "Server-Timing": f"inference;dur={infer_ms:.2f}",
                        "X-Inference-Time-Ms": f"{infer_ms:.2f}"
                    }
                )
        except HTTPException:
            raise
        except ValueError as e:
            # Question validation errors name the question and what to fix: safe for clients.
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # noqa: BLE001 -- never leak paths/weights/OOM text to clients
            # The client still learns nothing, but the operator gets the traceback. Without
            # this the container logs show only the 500, so a deterministic failure such as a
            # missing C compiler for triton's JIT (#365) is invisible from the running server
            # and has to be reproduced in-process to be diagnosed at all.
            # `model` is caller-supplied text. `_resolve_model` has already reduced it to a
            # shipped checkpoint name or None, and the line breaks are replaced so a crafted
            # value cannot forge log entries (py/log-injection) if that ever stops holding.
            _log.exception("inference failed for model=%s",
                           str(model).replace("\n", "\\n").replace("\r", "\\r"))
            raise HTTPException(status_code=500, detail="inference failed")

    @app.post("/v1/systemone/batch")
    async def systemone_batch(request: Request, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        async with _admit():
            return await _systemone_batch_inner(request)

    async def _systemone_batch_inner(request: Request):
        nonlocal gate
        body = await _read_json_body(request)
        if not isinstance(body, dict) or "questions" not in body or "states" not in body:
            raise HTTPException(
                status_code=400,
                detail="request body must be an object with 'states' and 'questions' fields",
            )
        states = body["states"]
        questions = body["questions"]
        _check_batch_limits(states, questions)
        # Same refusal policy as the single endpoint: hooks and the two hook-execution knobs
        # belong to the deployment, not the caller. `predict_batch` does take `hooks_timeout`,
        # so without this call a batch body would silently hand a caller a shorter deadline for
        # the operator's own hooks.
        _refuse_body_refusals(body)
        model = _resolve_model(body.get("model"))
        max_budget_cap = _resolve_max_token_budget()
        max_len = _validate_budget_param(body, "max_len", max_budget_cap)
        head_max_len = _validate_budget_param(body, "head_max_len", max_budget_cap)
        # Every control is sent only when the caller sent it, matching `_systemone_inner`: an
        # absent argument means "inherit what the Router was built with", so passing None would
        # override a deployment's own `Router(lang_guess=...)` or abstention threshold.
        # ``Router.predict_batch`` reads task / lang / lang_guess / max_len / head_max_len off
        # each item of ``requests`` (see its docstring), so they go into every synthesized
        # request dict; the HTTP body has one control set for the whole batch, which the Router
        # still honours -- it groups the requests by these values for the forward pass.
        item_overrides: Dict[str, Any] = {}
        if max_len is not None:
            item_overrides["max_len"] = max_len
        if head_max_len is not None:
            item_overrides["head_max_len"] = head_max_len
        # An unknown task is left to `route_batch`, which normalises it through `normalise_name`
        # and raises; the `except ValueError` below turns that into a 422 naming the task, so the
        # accepted set is core's and not a list restated here.
        if body.get("task") is not None:
            item_overrides["task"] = body["task"]
        for key in ("lang", "lang_guess"):
            value = _validate_language_param(body, key)
            if value is not None:
                item_overrides[key] = value
        # Call-level: `predict_batch` takes min_confidence / batch_size / sort_by_length as kwargs.
        call_kwargs: Dict[str, Any] = {}
        min_confidence = _validate_min_confidence(body)
        if min_confidence is not None:
            call_kwargs["min_confidence"] = min_confidence
        batch_size = _validate_batch_size_param(body)
        if batch_size is not None:
            call_kwargs["batch_size"] = batch_size
        sort_by_length = _validate_sort_by_length_param(body)
        if sort_by_length:
            # Only `True` is forwarded: `predict_batch`'s own default is `False`, and an attached
            # agent whose `predict_batch` predates the knob (#294) silently drops it. The MCP
            # tool makes the same choice for the same reason (see ``laya/mcp/tools.py``).
            call_kwargs["sort_by_length"] = True
        # The predict() fallback shape: `Router.predict` reads all six controls as call kwargs,
        # so the item dict is flattened back and `min_confidence` joins it.
        predict_kwargs: Dict[str, Any] = dict(item_overrides)
        if min_confidence is not None:
            predict_kwargs["min_confidence"] = min_confidence
        if gate is None:
            gate = asyncio.Lock()
        try:
            async with gate:
                loop = asyncio.get_running_loop()

                def _do_batch():
                    if hasattr(router, "predict_batch"):
                        reqs = [dict(state=s, questions=questions, model=model, **item_overrides)
                                for s in states]
                        results = router.predict_batch(reqs, **call_kwargs)
                    elif predict_kwargs:
                        results = [router.predict(s, questions, model=model, **predict_kwargs)
                                   for s in states]
                    else:
                        results = [router.predict(s, questions, model=model) for s in states]
                    total_tokens = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
                    return {
                        "results": results,
                        "total_usage": {"input_tokens": total_tokens, "output_tokens": 0},
                    }

                t0 = time.perf_counter()
                batch_res = await loop.run_in_executor(pool, _do_batch)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    content=batch_res,
                    headers={
                        "Server-Timing": f"inference;dur={infer_ms:.2f}",
                        "X-Inference-Time-Ms": f"{infer_ms:.2f}"
                    }
                )
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # noqa: BLE001
            _log.exception("batch inference failed")
            raise HTTPException(status_code=500, detail="inference failed")

    return app


def main() -> None:
    import socket

    import uvicorn
    from uvicorn.protocols.http.auto import AutoHTTPProtocol

    class NoDelayHTTPProtocol(AutoHTTPProtocol):
        # asyncio only sets TCP_NODELAY when the accepted socket reports IPPROTO_TCP, and
        # on macOS and Windows it reports 0, so Nagle delays small responses (#620).
        def connection_made(self, transport):
            sock = transport.get_extra_info("socket")
            if sock is not None and sock.family in (socket.AF_INET, socket.AF_INET6):
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass  # a latency hint, never a reason to drop the connection
            super().connection_made(transport)

    uvicorn.run(
        create_app(),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
        http=NoDelayHTTPProtocol,
    )


if __name__ == "__main__":
    main()
