from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, PgPool, Redis
from app.api.schemas import ChatRequest
from app.chat.logger import log_query
from app.chat.memory import ConversationMemory
from app.core.models import ChatMessage
from app.graph.state import create_initial_state

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    user_id: CurrentUser,
    pg_pool: PgPool,
    redis_client: Redis,
) -> StreamingResponse:
    session_id = str(body.session_id) if body.session_id else str(uuid.uuid4())
    memory = ConversationMemory(redis_client)

    # Set session owner if new session
    owner = await memory.get_owner(session_id)
    if owner is None:
        await memory.set_owner(session_id, user_id)
    elif owner != user_id:
        async def _forbidden_stream() -> AsyncGenerator[str]:
            yield _sse_event("error", {"message": "Access denied to this session"})

        return StreamingResponse(_forbidden_stream(), media_type="text/event-stream")

    # Load conversation history
    history = await memory.get_history(session_id, limit=20)

    # Save user message
    user_message = ChatMessage(
        role="user",
        content=body.message,
        timestamp=datetime.now(UTC),
    )
    await memory.add_message(session_id, user_message)

    # Create initial state
    state = create_initial_state(query=body.message, messages=history)

    # Get graph and openai client from app state
    graph = request.app.state.graph
    openai_client = request.app.state.openai_client

    async def _stream() -> AsyncGenerator[str]:
        start = time.perf_counter()
        try:
            # Run the graph pipeline
            result = await graph.ainvoke(state)

            intent = result.get("intent", "unclear")
            answer = result.get("answer", "")
            search_results = result.get("search_results", [])
            token_usage = result.get("token_usage", {})

            # Send metadata event
            yield _sse_event("metadata", {
                "session_id": session_id,
                "intent": intent,
            })

            # Stream tokens from the answer
            # Since graph.ainvoke returns the full answer, we simulate token streaming
            # by yielding the answer in chunks
            for char in answer:
                yield _sse_event("token", {"content": char})

            # Send sources event
            sources = []
            for sr in search_results:
                sources.append({
                    "chunk_id": sr.chunk_id,
                    "document_id": sr.document_id,
                    "title": sr.source_title,
                    "content": sr.content,
                    "score": sr.score,
                    "created_at": sr.source_created_at.isoformat(),
                })
            yield _sse_event("sources", {"results": sources})

            # Send usage event
            if token_usage:
                yield _sse_event("usage", {
                    "prompt_tokens": token_usage.get("prompt", 0),
                    "completion_tokens": token_usage.get("completion", 0),
                    "model": token_usage.get("model", ""),
                })

            # Send done event
            yield _sse_event("done", {})

            # Save assistant message
            assistant_message = ChatMessage(
                role="assistant",
                content=answer,
                timestamp=datetime.now(UTC),
            )
            await memory.add_message(session_id, assistant_message)

            # Log query (fire-and-forget)
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_query(
                pg_pool,
                user_id=user_id,
                session_id=session_id,
                query=body.message,
                intent=intent,
                search_type="hybrid" if search_results else None,
                result_count=len(search_results),
                latency_ms=latency_ms,
                token_usage_prompt=token_usage.get("prompt"),
                token_usage_completion=token_usage.get("completion"),
                model=token_usage.get("model"),
            )

        except Exception as exc:
            logger.exception("chat_stream_error")
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(_stream(), media_type="text/event-stream")
