"""Client SDK for Laya: local and remote drop-in TypeSafe System One client."""

import os
from typing import Any, Dict, List, Optional, Union
import httpx


_default_agent = None


def _get_default_agent():
    global _default_agent
    if _default_agent is None:
        from .agent import load

        _default_agent = load("convaiinnovations/laya")
    return _default_agent


class LayaClient:
    """Client for evaluating System One decisions via Laya (local or remote HTTP)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local: bool = True,
        timeout: float = 30.0,
        model_id: str = "convaiinnovations/laya",
    ):
        self.api_key = api_key or os.environ.get("LAYA_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = base_url or os.environ.get("LAYA_BASE_URL")
        self.local = local and (self.base_url is None)
        self.timeout = timeout
        self.model_id = model_id
        self._agent = None
        if not self.local and not self.base_url:
            self.base_url = "http://localhost:8000"

    def _get_agent(self):
        if self._agent is None:
            from .agent import load

            self._agent = load(self.model_id)
        return self._agent

    def system_one(
        self,
        state: Any,
        questions: Dict[str, Any],
        model: str = "laya-latest",
    ) -> Dict[str, Any]:
        """Evaluate a state and questions dictionary."""
        if self.local:
            agent = self._get_agent()
            res = agent.predict(state, questions)
            resolved_model = "laya-0.1.6" if model in ("laya-latest", "laya-preview", "jev-latest", "jev-preview", None) else model
            res["model"] = resolved_model
            return res

        url = f"{self.base_url.rstrip('/')}/v1/systemone"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "state": state,
            "questions": questions,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()


class AsyncLayaClient:
    """Asynchronous client for executing Laya System One queries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local: bool = True,
        timeout: float = 30.0,
        model_id: str = "convaiinnovations/laya",
    ):
        self.api_key = api_key or os.environ.get("LAYA_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = base_url or os.environ.get("LAYA_BASE_URL")
        self.local = local and (self.base_url is None)
        self.timeout = timeout
        self.model_id = model_id
        self._agent = None
        if not self.local and not self.base_url:
            self.base_url = "http://localhost:8000"

    def _get_agent(self):
        if self._agent is None:
            from .agent import load

            self._agent = load(self.model_id)
        return self._agent

    async def system_one(
        self,
        state: Any,
        questions: Dict[str, Any],
        model: str = "laya-latest",
    ) -> Dict[str, Any]:
        """Evaluate a state and questions asynchronously."""
        if self.local:
            agent = self._get_agent()
            res = agent.predict(state, questions)
            resolved_model = "laya-0.1.6" if model in ("laya-latest", "laya-preview", "jev-latest", "jev-preview", None) else model
            res["model"] = resolved_model
            return res

        url = f"{self.base_url.rstrip('/')}/v1/systemone"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "state": state,
            "questions": questions,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()


_default_client: Optional[LayaClient] = None


def _get_client() -> LayaClient:
    global _default_client
    if _default_client is None:
        _default_client = LayaClient(local=True)
    return _default_client


def system_one(
    state: Any,
    questions: Dict[str, Any],
    model: str = "laya-latest",
    client: Optional[LayaClient] = None,
) -> Dict[str, Any]:
    """Evaluate state and questions using Laya."""
    cli = client or _get_client()
    return cli.system_one(state=state, questions=questions, model=model)


def decide(
    state: Any,
    choices: Union[List[str], Dict[str, Optional[str]]],
    instructions: str = "Which option best describes the state?",
    model: str = "laya-latest",
) -> Dict[str, Any]:
    """Make a fast discrete decision among options."""
    if isinstance(choices, list):
        if len(choices) != len(set(choices)):
            raise ValueError(f"Duplicate choices found in options list: {choices}")
        criteria = {c: None for c in choices}
    else:
        criteria = choices

    q = {"type": "choice", "instructions": instructions, "criteria": criteria}
    resp = system_one(state=state, questions={"decision": q}, model=model)
    return resp["answers"]["decision"]


def judge(
    state: Any,
    instructions: str,
    criteria: Optional[Dict[str, str]] = None,
    model: str = "laya-latest",
) -> float:
    """Evaluate a yes/no question and return the probability (0.0 to 1.0)."""
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    resp = system_one(state=state, questions={"judgment": q}, model=model)
    return resp["answers"]["judgment"]["noul"]


def rate(
    state: Any,
    criteria: List[Union[str, Dict[str, Any]]],
    instructions: str = "Rate where the state falls on this scale:",
    model: str = "laya-latest",
) -> Dict[str, Any]:
    """Evaluate a state on an ordered multi-level scale."""
    q = {"type": "score", "instructions": instructions, "criteria": criteria}
    resp = system_one(state=state, questions={"rating": q}, model=model)
    return resp["answers"]["rating"]
