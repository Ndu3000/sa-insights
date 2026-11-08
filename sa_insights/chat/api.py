"""HTTP surface for the chatbot.

Deliberately a thin FastAPI app over ``agent.Agent``: the interesting logic is
already covered by Python tests, so this file only does transport — parse,
call, serialise — and cannot quietly become the place where business rules live.

Why FastAPI and not the AG-UI protocol directly
-----------------------------------------------
AG-UI is the right wire format for a CopilotKit frontend, and it is what the
frontend will speak. But the protocol adapter is a translation layer, and
translation layers get rewritten. Keeping the domain endpoint plain means the
backend stays useful to a Streamlit tab, a curl call and a future frontend
alike, and the AG-UI adapter sits on top of something stable.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import Agent, Session
from . import tools

# One process-wide agent, so the graph is built once rather than per request.
_AGENT = Agent()

# Sessions are keyed by a client-supplied id. In-memory and bounded: this is a
# public, unauthenticated demo endpoint, so durable conversation storage would be
# a liability rather than a feature.
_SESSIONS: dict[str, Session] = {}
_MAX_SESSIONS = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str = Field(default="anonymous", max_length=80)


class CitationModel(BaseModel):
    series_key: str
    label: str
    period: str
    value: float | None = None
    source: str
    detail: str = ""


class AskResponse(BaseModel):
    kind: str
    text: str
    confidence: float
    follow_up: str | None = None
    citations: list[CitationModel] = []
    unverified_numbers: list[float] = []
    evidence_series: list[str] = []


app = FastAPI(
    title="SA Insights chat",
    description="Ask questions about South African macroeconomic data.",
    version="1.0.0",
)

# The dashboard and the chat UI are served from different origins when deployed
# (Streamlit on one host, the chat frontend on another), so CORS is required for
# the browser to call this at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SA_CHAT_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _session(session_id: str) -> Session:
    if session_id not in _SESSIONS:
        # Evict the oldest when full. A demo endpoint accumulating sessions
        # forever is a slow memory leak; insertion order makes the oldest first.
        if len(_SESSIONS) >= _MAX_SESSIONS:
            _SESSIONS.pop(next(iter(_SESSIONS)), None)
        _SESSIONS[session_id] = Session()
    return _SESSIONS[session_id]


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus a real warehouse check.

    Reports whether the warehouse is readable, because a 200 from the process
    says nothing about whether answers can be produced — the same
    "configured versus working" distinction that made the MLflow chip lie.
    """
    try:
        summary = tools.data_freshness()
        totals = summary.get("totals") or {}
        return {
            "status": "ok",
            "series": totals.get("series"),
            "observations": totals.get("observations"),
            "last_updated": str(totals.get("last_updated")),
        }
    except Exception as exc:  # noqa: BLE001 - health must always answer
        return {"status": "degraded", "error": str(exc)}


@app.get("/series")
def series(category: str | None = None) -> dict[str, Any]:
    """The catalogue, so a UI can show what is askable."""
    return {
        "categories": tools.categories(),
        "series": [
            {
                "key": s.key,
                "label": s.label,
                "category": s.category,
                "unit": s.unit,
                "last_period": s.last_period,
            }
            for s in tools.list_series(category)
        ],
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question, or explain what is missing.

    Never raises for a bad question: "I cannot answer that" is a valid response
    and must reach the user as one, not as a 500.
    """
    answer = _AGENT.ask(request.question, _session(request.session_id))
    payload = answer.as_dict()
    return AskResponse(**payload)


@app.post("/reset")
def reset(request: AskRequest) -> dict[str, str]:
    """Forget the conversation, so a follow-up does not inherit old context."""
    _SESSIONS.pop(request.session_id, None)
    return {"status": "cleared"}
