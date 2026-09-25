"""CrewAI integration for Laya System 1 decision engine.

Provides sub-35ms, non-autoregressive task delegation routing and real-time task
guardrails for CrewAI multi-agent crews.

Replaces slow, token-generating LLM managers with calibrated, typed decisions
executed in a single forward pass without token generation costs.

Supports both local in-process models (`Agent` / `Router`) and remote HTTP
deployments (your own `laya-serve`) without requiring PyTorch on edge clients.
"""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

# Optional CrewAI base classes
try:
    from crewai import Agent as CrewAgent, Task as CrewTask
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

    @dataclass
    class CrewAgent:  # type: ignore
        """Lightweight shim when crewai is not installed."""
        role: str
        goal: str = ""
        backstory: str = ""

    @dataclass
    class CrewTask:  # type: ignore
        """Lightweight shim when crewai is not installed."""
        description: str
        expected_output: str = ""
        agent: Optional[Any] = None


class LayaTaskGuardError(ValueError):
    """Raised when a task or instruction violates a Laya task guard policy."""

    def __init__(self, message: str, violations: Dict[str, Any], raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.violations = violations
        self.raw_decision = raw_decision


class LayaLowConfidenceError(ValueError):
    """Raised when task routing confidence falls below the configured threshold and no fallback is set."""

    def __init__(self, message: str, confidence: float, threshold: float, raw_decision: Dict[str, Any]):
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold
        self.raw_decision = raw_decision


@dataclass
class CrewRouteDecision:
    """Result of a Laya sub-35ms task delegation decision."""

    agent: Any
    agent_index: int
    role: str
    confidence: float
    reason: str
    raw_decision: Dict[str, Any]


def _extract_task_str(task: Union[str, CrewTask, Dict[str, Any], Any]) -> str:
    """Extract descriptive text from a CrewAI task or string."""
    if isinstance(task, str):
        return task
    if hasattr(task, "description"):
        desc = str(task.description)
        expected = getattr(task, "expected_output", "")
        if expected:
            return f"{desc} (Expected output: {expected})"
        return desc
    if isinstance(task, dict):
        desc = task.get("description", task.get("task", task.get("prompt", str(task))))
        expected = task.get("expected_output", "")
        if expected:
            return f"{desc} (Expected output: {expected})"
        return str(desc)
    return str(task)


def _get_agent_role(agent_obj: Any, default_idx: int) -> str:
    """Extract role string from agent object, dict, or string."""
    if isinstance(agent_obj, str):
        return agent_obj
    if isinstance(agent_obj, dict):
        return str(agent_obj.get("role") or f"Agent {default_idx}")
    return str(getattr(agent_obj, "role", None) or f"Agent {default_idx}")


def _format_agent_criteria(agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]]) -> Dict[str, str]:
    """Format candidate CrewAI agents into Laya choice criteria."""
    criteria: Dict[str, str] = {}
    for i, agent in enumerate(agents):
        key = f"agent_{i}"
        role = _get_agent_role(agent, i)
        goal = getattr(agent, "goal", None)

        if isinstance(agent, dict):
            goal = agent.get("goal", goal)

        if goal:
            criteria[key] = f"{role}: {goal}"
        else:
            criteria[key] = str(role)

    return criteria


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


class LayaCrewRouter:
    """Sub-35ms task delegation router for CrewAI multi-agent crews.

    Evaluates tasks against crew agent roles and goals in a single non-autoregressive
    forward pass, eliminating 2-4 second LLM manager delegation latency.
    """

    def __init__(
        self,
        instructions: str = "Which agent in the crew is best qualified to execute this task?",
        confidence_threshold: float = 0.0,
        fallback_agent_index: Optional[int] = None,
        raise_on_low_confidence: bool = False,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.instructions = instructions
        self.confidence_threshold = confidence_threshold
        self.fallback_agent_index = fallback_agent_index
        self.raise_on_low_confidence = raise_on_low_confidence
        self.agent = agent
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.last_decision: Optional[Dict[str, Any]] = None

    def route(
        self,
        task: Union[str, CrewTask, Dict[str, Any], Any],
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> CrewRouteDecision:
        """Evaluate task and route to the best matching crew agent in ~33ms."""
        if not agents:
            raise ValueError("No agents provided to route task to.")

        task_str = _extract_task_str(task)
        criteria = _format_agent_criteria(agents)

        questions = {
            "delegation": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }
        }

        res = _execute_decision(
            task_str,
            questions,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        self.last_decision = res

        ans = res.get("answers", {}).get("delegation", {})
        chosen_key = ans.get("choice")
        conf = ans.get("answer_confidence", ans.get("confidence", 1.0))

        # Map agent_i back to integer index i
        chosen_idx: int = 0
        if chosen_key and chosen_key.startswith("agent_"):
            try:
                chosen_idx = int(chosen_key.split("_")[1])
            except (ValueError, IndexError):
                chosen_idx = 0
        elif chosen_key in criteria:
            chosen_idx = list(criteria.keys()).index(chosen_key)

        # Confidence gating
        if self.confidence_threshold > 0.0 and conf < self.confidence_threshold:
            if self.fallback_agent_index is not None:
                fallback_obj = agents[self.fallback_agent_index]
                fallback_role = _get_agent_role(fallback_obj, self.fallback_agent_index)
                reason = (
                    f"Delegated to fallback agent '{fallback_role}' because routing confidence "
                    f"({conf:.3f}) was below threshold ({self.confidence_threshold:.3f})."
                )
                return CrewRouteDecision(
                    agent=fallback_obj,
                    agent_index=self.fallback_agent_index,
                    role=fallback_role,
                    confidence=conf,
                    reason=reason,
                    raw_decision=res,
                )
            if self.raise_on_low_confidence:
                raise LayaLowConfidenceError(
                    f"Routing confidence {conf:.3f} below threshold {self.confidence_threshold:.3f} "
                    f"for task: {task_str!r}",
                    confidence=conf,
                    threshold=self.confidence_threshold,
                    raw_decision=res,
                )

        selected_agent = agents[chosen_idx]
        role_name = _get_agent_role(selected_agent, chosen_idx)
        reason = f"Delegated to '{role_name}' via Laya System 1 decision (confidence: {conf:.3f})."

        return CrewRouteDecision(
            agent=selected_agent,
            agent_index=chosen_idx,
            role=role_name,
            confidence=conf,
            reason=reason,
            raw_decision=res,
        )

    def delegate(
        self,
        task: Any,
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> Any:
        """Assign task.agent to the winning agent and return the selected agent."""
        decision = self.route(task, agents)
        if hasattr(task, "agent"):
            try:
                task.agent = decision.agent
            except Exception:
                pass
        return decision.agent

    async def aroute(
        self,
        task: Union[str, CrewTask, Dict[str, Any], Any],
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> CrewRouteDecision:
        """Asynchronously route task without blocking the asyncio event loop."""
        return await asyncio.to_thread(self.route, task, agents)

    async def adelegate(
        self,
        task: Any,
        agents: Sequence[Union[CrewAgent, Dict[str, Any], Any]],
    ) -> Any:
        """Asynchronously delegate task to the winning agent."""
        decision = await self.aroute(task, agents)
        if hasattr(task, "agent"):
            try:
                task.agent = decision.agent
            except Exception:
                pass
        return decision.agent


class LayaTaskGuard:
    """Sub-40ms inline guardrail for CrewAI tasks and agent inputs.

    Screens incoming user instructions and task specifications for jailbreaks,
    prompt injections, and policy violations before agents execute tools.
    """

    def __init__(
        self,
        questions: Optional[Dict[str, Any]] = None,
        action: str = "raise",  # "raise", "filter", or "annotate"
        rejection_message: str = "This task cannot be executed because it violates safety guidelines.",
        threshold: float = 0.5,
        agent: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.questions = questions
        self.action = action
        self.rejection_message = rejection_message
        self.threshold = threshold
        self.agent = agent
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.last_decision: Optional[Dict[str, Any]] = None

    def _default_questions(self) -> Dict[str, Any]:
        from ..presets import guard_questions
        return guard_questions()

    def screen(self, task: Union[str, CrewTask, Dict[str, Any], Any]) -> Any:
        """Screen task against safety policies."""
        task_str = _extract_task_str(task)
        qdefs = self.questions if self.questions is not None else self._default_questions()

        res = _execute_decision(
            task_str,
            qdefs,
            agent=self.agent,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )
        self.last_decision = res
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
            raise LayaTaskGuardError(
                f"Laya task guardrail policy violation detected: {list(violations.keys())}",
                violations=violations,
                raw_decision=res,
            )

        if not is_safe and self.action == "filter":
            if hasattr(task, "description"):
                task.description = self.rejection_message
                return task
            if isinstance(task, dict):
                filtered = dict(task)
                filtered["description"] = self.rejection_message
                return filtered
            return self.rejection_message

        if self.action == "annotate":
            guard_meta = {
                "passed": is_safe,
                "violations": violations,
                "answers": answers,
            }
            if hasattr(task, "description"):
                setattr(task, "guardrail", guard_meta)
                return task
            if isinstance(task, dict):
                annotated = dict(task)
                annotated["guardrail"] = guard_meta
                return annotated
            return {
                "task": task,
                "guardrail": guard_meta,
            }

        return task

    def __call__(self, task: Any) -> Any:
        return self.screen(task)

    async def ascreen(self, task: Union[str, CrewTask, Dict[str, Any], Any]) -> Any:
        """Asynchronously screen task without blocking the event loop."""
        return await asyncio.to_thread(self.screen, task)
