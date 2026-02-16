from __future__ import annotations

from app.core.models import ChatMessage, GraphState


def create_initial_state(
    query: str,
    messages: list[ChatMessage] | None = None,
) -> GraphState:
    return GraphState(
        query=query,
        intent="",
        search_results=[],
        relevance_score=0.0,
        context="",
        answer="",
        messages=messages or [],
        token_usage={},
        error=None,
    )
