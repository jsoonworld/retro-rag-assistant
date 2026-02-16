from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Response

from app.api.deps import CurrentUser, Redis
from app.api.schemas import MessageItem, SessionResponse
from app.chat.memory import ConversationMemory

logger = structlog.get_logger()

router = APIRouter(tags=["sessions"])


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    user_id: CurrentUser,
    redis_client: Redis,
) -> SessionResponse:
    memory = ConversationMemory(redis_client)
    sid = str(session_id)

    if not await memory.session_exists(sid):
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify ownership
    owner = await memory.get_owner(sid)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    history = await memory.get_history(sid, limit=100)

    messages = [
        MessageItem(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
        )
        for msg in history
    ]

    created_at = messages[0].timestamp if messages else None
    last_active_at = messages[-1].timestamp if messages else None

    return SessionResponse(
        session_id=sid,
        messages=messages,
        message_count=len(messages),
        created_at=created_at,
        last_active_at=last_active_at,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    user_id: CurrentUser,
    redis_client: Redis,
) -> Response:
    memory = ConversationMemory(redis_client)
    sid = str(session_id)

    # Verify ownership if session exists
    owner = await memory.get_owner(sid)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    await memory.clear_session(sid)
    return Response(status_code=204)
