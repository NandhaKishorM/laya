"""LangChain and LangGraph integration for Laya System 1 decision engine.

Provides fast (~33 ms), non-autoregressive routing, real-time guardrails, and
state evaluation nodes for LangChain Expression Language (LCEL) and LangGraph.

Supports both local in-process models (`Agent` / `Router`) and remote HTTP
deployments (your own `laya-serve`) without requiring PyTorch on edge clients.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional, Sequence, Union

# Optional LangChain base class integration
try:
    from langchain_core.runnables import RunnableConfig, RunnableSerializable
    _RUNNABLE_AVAILABLE = True
except ImportError:
    _RUNNABLE_AVAILABLE = False
    RunnableSerializable = object  # type: ignore
    RunnableConfig = Any  # type: ignore


class LayaGuardrailError(ValueError):
    """Raised when an input violates a Laya guardrail policy."""

    def __init__(self, message: str, violations: Dict[str, Any], raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.violations = violations
        self.raw_decision = raw_decision


def _extract_text(input_val: Any, state_key: Optional[Union[str, Callable[[Any], Any]]] = None) -> Union[str, dict, list]:
    """Extract evaluatable text from arbitrary LangChain/LangGraph states or messages."""
    if state_key is not None:
        if callable(state_key):
            return state_key(input_val)
        if isinstance(input_val, dict) and state_key in input_val:
            return _extract_from_message_or_value(input_val[state_key])

    if isinstance(input_val, str):
        return input_val

    if isinstance(input_val, dict):
        for candidate in ("input", "text", "query", "prompt", "message", "body", "content"):
            if candidate in input_val:
                return _extract_from_message_or_value(input_val[candidate])
        if "messages" in input_val and isinstance(input_val["messages"], list):
            return _extract_from_messages_list(input_val["messages"])
        return input_val

    if isinstance(input_val, list):
        return _extract_from_messages_list(input_val)

    return str(input_val)


def _extract_from_message_or_value(val: Any) -> Any:
    if hasattr(val, "content"):
        return str(val.content)
    if isinstance(val, list):
        return _extract_from_messages_list(val)
    return val


def _extract_from_messages_list(msgs: Sequence[Any]) -> str:
    if not msgs:
        return ""
    # Search backwards for the most recent human/user message
    for m in reversed(msgs):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("human", "user"):
            return str(getattr(m, "content", m))
    last = msgs[-1]
    return str(getattr(last, "content", last))


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward bearer credentials across an origin or HTTPS downgrade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        old_port = old.port or (443 if old.scheme.lower() == "https" else 80)
        new_port = new.port or (443 if new.scheme.lower() == "https" else 80)
        if (
            old.scheme.lower() != new.scheme.lower()
            or (old.hostname or "").lower() != (new.hostname or "").lower()
            or old_port != new_port
        ):
            raise urllib.error.URLError("refusing cross-origin or HTTPS-downgrade redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _call_remote(
    base_url: str,
    state: Any,
    questions: Dict[str, Any],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Send decision request to a remote laya-serve HTTP instance using standard library urllib."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1/systemone"):
        url = f"{url}/v1/systemone"

    payload: Dict[str, Any] = {"state": state, "questions": questions}
    if model:
        payload["model"] = model

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(_SameOriginRedirectHandler())
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Laya server error {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Laya server at {url}: {e}") from e


_DEFAULT_ROUTER = None
_DEFAULT_ROUTER_LOCK = threading.Lock()


def _get_default_router():
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        with _DEFAULT_ROUTER_LOCK:
            if _DEFAULT_ROUTER is None:
                from ..router import Router
                _DEFAULT_ROUTER = Router()
    return _DEFAULT_ROUTER


def _execute_decision(
    state: Any,
    questions: Dict[str, Any],
    agent: Optional[Any] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if base_url:
        return _call_remote(base_url, state, questions, api_key=api_key, model=model)
    runner = agent if agent is not None else _get_default_router()
    kwargs = {"model": model} if model else {}
    return runner.predict(state, questions, **kwargs)


class LayaRouter(RunnableSerializable):
    """Zero-latency LangGraph conditional edge and LangChain LCEL routing runnable.

    Evaluates user input against typed criteria in ~33 ms without token generation.
    Supports confidence threshold gating and fallback routing.
    """

    criteria: Dict[str, str]
    instructions: str = "Which route should handle this request?"
    confidence_threshold: float = 0.0
    fallback: Optional[str] = None
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    question_id: str = "route"
    last_decision: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        criteria: Dict[str, str],
        instructions: str = "Which route should handle this request?",
        confidence_threshold: float = 0.0,
        fallback: Optional[str] = None,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                criteria=criteria,
                instructions=instructions,
                confidence_threshold=confidence_threshold,
                fallback=fallback,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.criteria = criteria
            self.instructions = instructions
            self.confidence_threshold = confidence_threshold
            self.fallback = fallback
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
        self.question_id = "route"
        self.last_decision: Optional[Dict[str, Any]] = None

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> str:
        """Route input to a destination branch label."""
        text = _extract_text(input, self.state_key)
        questions = {
            self.question_id: {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": self.criteria,
            }
        }
        res = _execute_decision(
            text,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        self.last_decision = res
        ans = res["answers"][self.question_id]
        choice = ans["choice"]
        confidence = ans.get("confidence", 1.0)

        if self.confidence_threshold > 0.0 and confidence < self.confidence_threshold:
            if self.fallback is not None:
                return self.fallback

        return choice

    def __call__(self, state: Any) -> str:
        """Callable protocol for direct use as a LangGraph conditional edge."""
        return self.invoke(state)


class LayaGuardrail(RunnableSerializable):
    """Sub-40ms inline guardrail for LangChain chains and LangGraph nodes.

    Screens for prompt injections, jailbreaks, sensitive data, or custom harm
    criteria before passing inputs downstream.
    """

    questions: Optional[Dict[str, Any]] = None
    action: str = "raise"  # "raise", "filter", or "annotate"
    rejection_message: str = "I cannot fulfill this request because it violates safety guidelines."
    threshold: float = 0.5
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        questions: Optional[Dict[str, Any]] = None,
        action: str = "raise",
        rejection_message: str = "I cannot fulfill this request because it violates safety guidelines.",
        threshold: float = 0.5,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                questions=questions,
                action=action,
                rejection_message=rejection_message,
                threshold=threshold,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.questions = questions
            self.action = action
            self.rejection_message = rejection_message
            self.threshold = threshold
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model

    def _default_questions(self) -> Dict[str, Any]:
        from ..presets import guard_questions
        return guard_questions()

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Screen input against guardrail questions."""
        text = _extract_text(input, self.state_key)
        qdefs = self.questions if self.questions is not None else self._default_questions()

        res = _execute_decision(
            text,
            qdefs,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        answers = res.get("answers", {})

        violations: Dict[str, Any] = {}
        for qid, ans in answers.items():
            t = ans.get("type")
            if t == "noul" and ans.get("noul", 0.0) >= self.threshold:
                violations[qid] = {
                    "probability": ans["noul"],
                    "confidence": ans.get("confidence", 0.0),
                }
            elif t == "score" and ans.get("score", 0.0) >= self.threshold:
                violations[qid] = {
                    "score": ans["score"],
                    "confidence": ans.get("confidence", 0.0),
                }

        is_safe = len(violations) == 0

        if not is_safe and self.action == "raise":
            raise LayaGuardrailError(
                f"Laya guardrail policy violation detected: {list(violations.keys())}",
                violations=violations,
                raw_decision=res,
            )

        if not is_safe and self.action == "filter":
            if isinstance(input, dict):
                filtered = dict(input)
                filtered["output"] = self.rejection_message
                return filtered
            return self.rejection_message

        if self.action == "annotate":
            if isinstance(input, dict):
                annotated = dict(input)
                annotated["guardrails"] = {
                    "passed": is_safe,
                    "violations": violations,
                    "answers": answers,
                }
                return annotated
            return {
                "input": input,
                "guardrails": {
                    "passed": is_safe,
                    "violations": violations,
                    "answers": answers,
                },
            }

        return input

    def __call__(self, state: Any) -> Any:
        return self.invoke(state)


class LayaTriage(RunnableSerializable):
    """Customer support ticket and incoming message triage node for LangGraph.

    Analyzes intent, urgency, customer frustration, and churn risk in one single
    forward pass and enriches the graph state dictionary.
    """

    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model

    def invoke(self, state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Triage the state and return enriched fields."""
        from ..presets import triage_questions

        text = _extract_text(state, self.state_key)
        res = _execute_decision(
            text,
            triage_questions(),
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        ans = res.get("answers", {})

        triage_info = {
            "intent": ans.get("intent", {}).get("choice"),
            "intent_confidence": ans.get("intent", {}).get("confidence"),
            "is_urgent": ans.get("is_urgent", {}).get("noul", 0.0) >= 0.5,
            "frustration_score": ans.get("frustration", {}).get("score"),
            "churn_risk": ans.get("churn_risk", {}).get("noul", 0.0) >= 0.5,
            "refund_requested": ans.get("refund_requested", {}).get("noul", 0.0) >= 0.5,
        }

        if isinstance(state, dict):
            updated = dict(state)
            updated["triage"] = triage_info
            return updated

        return {"input": state, "triage": triage_info}

    def __call__(self, state: Any) -> Dict[str, Any]:
        return self.invoke(state)


class LayaEvaluator(RunnableSerializable):
    """Rubric-based output grading and hallucination evaluation for LangChain.

    Evaluates LLM responses against criteria without generating text.
    """

    questions: Dict[str, Any]
    state_key: Optional[Union[str, Callable[[Any], Any]]] = None
    agent: Optional[Any] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        questions: Dict[str, Any],
        state_key: Optional[Union[str, Callable[[Any], Any]]] = None,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        if _RUNNABLE_AVAILABLE:
            super().__init__(
                questions=questions,
                state_key=state_key,
                agent=agent,
                base_url=base_url,
                api_key=api_key,
                model=model,
                **kwargs,
            )
        else:
            self.questions = questions
            self.state_key = state_key
            self.agent = agent
            self.base_url = base_url
            self.api_key = api_key
            self.model = model

    def evaluate_strings(self, *, prediction: str, input: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """LangChain standard string evaluation interface."""
        state = {"input": input, "prediction": prediction} if input else prediction
        res = _execute_decision(
            state,
            self.questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        return res.get("answers", {})

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        text = _extract_text(input, self.state_key)
        res = _execute_decision(
            text,
            self.questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        return res.get("answers", {})

    def __call__(self, state: Any) -> Dict[str, Any]:
        return self.invoke(state)
