from __future__ import annotations

import json

import structlog
from redis.asyncio import Redis

from app.core.config import settings
from app.core.models import ChatMessage

logger = structlog.get_logger()

_SESSION_PREFIX = "retro-rag:session"


class ConversationMemory:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"{_SESSION_PREFIX}:{session_id}:messages"

    @staticmethod
    def _owner_key(session_id: str) -> str:
        return f"{_SESSION_PREFIX}:{session_id}:owner"

    def _ttl_seconds(self) -> int:
        return settings.session_ttl_hours * 3600

    async def add_message(self, session_id: str, message: ChatMessage) -> None:
        key = self._messages_key(session_id)
        data = json.dumps({
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
        })
        await self._redis.rpush(key, data)
        await self._redis.expire(key, self._ttl_seconds())

    async def get_history(self, session_id: str, limit: int = 20) -> list[ChatMessage]:
        key = self._messages_key(session_id)
        raw_messages = await self._redis.lrange(key, -limit, -1)
        messages: list[ChatMessage] = []
        for raw in raw_messages:
            obj = json.loads(raw)
            from datetime import datetime, timezone

            messages.append(ChatMessage(
                role=obj["role"],
                content=obj["content"],
                timestamp=datetime.fromisoformat(obj["timestamp"]).replace(
                    tzinfo=timezone.utc
                )
                if "+" not in obj["timestamp"] and "Z" not in obj["timestamp"]
                else datetime.fromisoformat(obj["timestamp"]),
            ))
        return messages

    async def clear_session(self, session_id: str) -> None:
        await self._redis.delete(
            self._messages_key(session_id),
            self._owner_key(session_id),
        )

    async def session_exists(self, session_id: str) -> bool:
        return await self._redis.exists(self._messages_key(session_id)) > 0

    async def get_owner(self, session_id: str) -> int | None:
        owner = await self._redis.get(self._owner_key(session_id))
        if owner is None:
            return None
        return int(owner)

    async def set_owner(self, session_id: str, user_id: int) -> None:
        await self._redis.set(
            self._owner_key(session_id),
            str(user_id),
            ex=self._ttl_seconds(),
        )
