"""FastAPI server for Laya: drop-in replacement for TypeSafe Jev API."""

import os
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .agent import load

app = FastAPI(
    title="Laya Decision Server",
    description="Drop-in TypeSafe System One compatible server powered by Laya 421M RLCD.",
    version="0.1.6",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        model_id = os.environ.get("LAYA_MODEL_ID", "convaiinnovations/laya")
        _agent = load(model_id)
    return _agent


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    expected_key = os.environ.get("LAYA_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not expected_key:
        return True
    if not credentials or credentials.credentials != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide Authorization: Bearer <LAYA_API_KEY>",
        )
    return True


class SystemOneRequest(BaseModel):
    model: Optional[str] = Field(default="laya-latest", description="Model identifier")
    state: Any = Field(..., description="Context string, JSON object, or list")
    questions: Dict[str, Any] = Field(..., description="Map of question definitions")


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "laya",
        "version": "0.1.6",
        "timestamp": time.time(),
    }


@app.get("/v1/models")
def list_models(auth: bool = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": "laya-latest", "object": "model", "owned_by": "convaiinnovations", "active": True},
            {"id": "laya-0.1.6", "object": "model", "owned_by": "convaiinnovations", "active": True},
            {"id": "laya-preview", "object": "model", "owned_by": "convaiinnovations", "active": True},
            {"id": "jev-latest", "object": "model", "owned_by": "typesafe-compatibility", "active": True},
            {"id": "jev-1.13.0", "object": "model", "owned_by": "typesafe-compatibility", "active": True},
        ],
    }


@app.post("/v1/systemone")
def system_one(
    payload: SystemOneRequest,
    auth: bool = Depends(verify_api_key),
):
    try:
        agent = get_agent()
        res = agent.predict(payload.state, payload.questions)
        model_name = payload.model or "laya-latest"
        resolved_model = "laya-0.1.6" if model_name in ("laya-latest", "laya-preview", "jev-latest", "jev-preview") else model_name
        res["model"] = resolved_model
        return res
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
