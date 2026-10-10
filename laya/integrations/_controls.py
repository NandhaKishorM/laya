"""Per-call decision controls shared by the framework integrations.

Every integration wrapper ends at one of two places: a local `runner.predict(state, questions,
**overrides)` call, or a POST to a laya-serve node. Which overrides those two accept is a single
rule, and three independent copies of it drift -- this module is the copy the LangChain, CrewAI
and LlamaIndex wrappers import.

The names are listed once, in `PREDICT_CONTROLS`, `HOOK_CONTROLS`, `DECISION_CONTROLS` and
`ROUTER_ONLY_CONTROLS`, and the signatures below are checked against them by
`tests/test_langchain.py`, `tests/test_crewai.py` and `tests/test_llamaindex.py`, which also assert
every wrapper's constructor accepts all of them. A control added here without reaching a wrapper's
`__init__` fails that wrapper's suite rather than being dropped silently.

The last list is the one that needs a rule rather than a splat: two of these controls are read by a
`Router` and by no `Agent`, and a wrapper's runner can be either, so `require_router_controls` asks
the signature of the call about to be made.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

# The token-budget overrides `Agent.predict` / `Router.predict` take per call, and that
# laya-serve accepts in the request body up to its `LAYA_MAX_TOKEN_BUDGET` ceiling.
PREDICT_CONTROLS = ("max_len", "head_max_len")

# The per-call language and abstention controls. Unlike `task` / `lang_guess` -- Router-only
# routing keywords a direct `Agent.predict` does not accept -- both `lang` and `min_confidence`
# are read by `Agent.predict`/`system_one` AND `Router.predict`, and laya-serve takes each in the
# request body, so the same kwargs are safe on the local and the remote path alike. `lang` routes
# non-English text and selects the answering checkpoint's per-language calibration;
# `min_confidence` is core's abstention gate (#361).
DECISION_CONTROLS = ("lang", "min_confidence")

# The per-call hook family. These are Python callables and flags that run inside `predict`, so
# they exist only on the local path -- see `reject_remote_hooks`.
HOOK_CONTROLS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")

# The routing hints. `Router.route` reads both -- `task` names the checkpoint that answers, the way
# `model` does, and `lang_guess` decides English vs multilingual before core's own script detection
# runs -- and `Agent.predict` reads neither, because an Agent has no routing step: it answers on the
# checkpoint it was built with. So these two go to a runner that reads them and are refused by name
# at one that does not -- see `require_router_controls`.
ROUTER_ONLY_CONTROLS = ("task", "lang_guess")


def budget_kwargs(max_len: Optional[int] = None,
                  head_max_len: Optional[int] = None) -> Dict[str, Any]:
    """The two token budgets, with the unset ones omitted.

    Absent rather than `None`: on the local path a `None` budget would override the checkpoint's
    own default with nothing, and on the remote path an older stand-in `_call_remote` -- including
    the ones these integrations' own tests install -- must keep accepting the call.
    """
    return {key: value for key, value in (("max_len", max_len), ("head_max_len", head_max_len))
            if value is not None}


def decision_kwargs(lang: Optional[str] = None,
                    min_confidence: Optional[float] = None) -> Dict[str, Any]:
    """The language and abstention overrides, with the unset ones omitted.

    Absent rather than `None`: a `None` `lang` would shadow the checkpoint's own detection and a
    `None` `min_confidence` would override the runner's abstention default with nothing. Both use
    the `is not None` test rather than truthiness -- an empty `lang` is core's documented "fall
    through to detection", and a `0.0` abstention gate is a real decision (abstain over nothing),
    not an absence.
    """
    return {key: value for key, value in (("lang", lang), ("min_confidence", min_confidence))
            if value is not None}


def predict_kwargs(model: Optional[str] = None, max_len: Optional[int] = None,
                   head_max_len: Optional[int] = None, lang: Optional[str] = None,
                   min_confidence: Optional[float] = None) -> Dict[str, Any]:
    """The per-request overrides a local runner accepts, with the unset ones omitted."""
    kwargs: Dict[str, Any] = {}
    if model:
        kwargs["model"] = model
    kwargs.update(budget_kwargs(max_len, head_max_len))
    kwargs.update(decision_kwargs(lang, min_confidence))
    return kwargs


def hook_kwargs(hooks: Optional[Any] = None, on_predict_start: Optional[Any] = None,
                on_predict_end: Optional[Any] = None, hooks_raise: Optional[bool] = None,
                hooks_timeout: Optional[float] = None) -> Dict[str, Any]:
    """The per-call hook overrides, with the unset ones omitted.

    Core reads `None` as "inherit whatever the runner was built with", so an unset hook has to be
    absent rather than passed as `None`. Note the `is not None` tests: `hooks=[]` means "no hooks
    for this call", and `hooks_raise=False` means "keep deciding after a hook fails" -- both are
    decisions a caller made, not absences.
    """
    given = {"hooks": hooks, "on_predict_start": on_predict_start, "on_predict_end": on_predict_end,
             "hooks_raise": hooks_raise, "hooks_timeout": hooks_timeout}
    return {k: v for k, v in given.items() if v is not None}


def reject_remote_hooks(given: Dict[str, Any], base_url: Optional[str]) -> None:
    """Refuse hooks on a remote node rather than dropping them silently.

    A hook is a Python callable that runs inside `predict` -- it can cache a decision, gate one or
    rewrite its state. `laya-serve` has no way to receive or run one, so a node with a `base_url`
    and hooks configured would report success while never calling them.
    """
    if base_url and given:
        raise ValueError(
            "%s run in the local runner and cannot be sent to a laya-serve endpoint; "
            "install them where serve runs, or drop them" % ", ".join(sorted(given))
        )


def router_kwargs(task: Optional[str] = None, lang_guess: Optional[Any] = None) -> Dict[str, Any]:
    """The routing hints, with the unset ones omitted.

    Absent rather than `None`, for the same reason as the budgets: `Router(task=...)` and
    `Router(lang_guess=...)` are hints a deployment installs for every request, and an explicit
    `None` here would replace the deployment's own hint with nothing for this one call. A blank
    `lang_guess` is core's documented "no usable hint, fall through to detection", so it is
    forwarded rather than treated as an absence.
    """
    return {key: value for key, value in (("task", task), ("lang_guess", lang_guess))
            if value is not None}


def _reads(target: Any, name: str) -> bool:
    """Whether the entry point about to be called can read `name` as a keyword argument."""
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        # Not introspectable (a C callable, a class with a hand-written __call__ signature): let
        # core decide. Refusing a call on a guess would block a runner this module cannot see.
        return True
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return True
    return name in params


def require_router_controls(runner: Any, entry_point: str, given: Dict[str, Any]) -> Dict[str, Any]:
    """Return `given` when the runner's entry point reads it, else raise naming what it does not.

    `task` and `lang_guess` route, and only a Router routes -- `Agent` takes neither, and its two
    batch entry points do not either. Forwarding them anyway is a `TypeError` from inside core on
    the way to the answer, and dropping them quietly is the silence this module exists to break: the
    caller asked for the typed-decisions checkpoint or for German text to be read as German, and got
    the deployment's defaults instead.

    Read from the signature of the call about to be made (`entry_point`) rather than from a list of
    runner classes, because `predict`, `predict_batch` and a stand-in each accept a different set,
    and a wrapper declared `**kwargs` takes everything. A runner with no such method at all -- a
    duck-typed stand-in that routes somewhere else -- is not refused either: this cannot tell, and
    guessing wrong would block a call that works.
    """
    if not given:
        return {}
    target = getattr(runner, entry_point, None)
    unread = [name for name in ROUTER_ONLY_CONTROLS if name in given and not _reads(target, name)]
    if unread:
        raise ValueError(
            "%s %s a routing hint, and %s.%s does not read it: this runner answers on the checkpoint "
            "it was built with. Pass a Router as `agent=`, or drop %s."
            % (", ".join(unread), "is" if len(unread) == 1 else "are",
               type(runner).__name__, entry_point,
               "it" if len(unread) == 1 else "them")
        )
    return given


def reject_remote_lang_guess(given: Dict[str, Any], base_url: Optional[str]) -> None:
    """Refuse a non-string `lang_guess` on a remote node rather than sending a body serve rejects.

    `Router` reads `lang_guess` as a language code *or* as a callable taking the state, and only the
    code has a wire form: `laya-serve` validates it as a string and answers 422 to anything else, so
    a callable would fail as an HTTP status after the request went out. Naming the argument here
    says what to do instead -- install the callable on the Router where serve runs, or pass a code.
    """
    value = given.get("lang_guess")
    if base_url and value is not None and not isinstance(value, str):
        raise ValueError(
            "lang_guess=%r cannot be sent to a laya-serve endpoint: only a language code string "
            "such as \"de\" crosses HTTP. Install the callable on the Router where serve runs, or "
            "pass a code" % (value,)
        )
