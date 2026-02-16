from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.chat.memory import ConversationMemory
from app.core.models import ChatMessage


def _make_mock_redis() -> AsyncMock:
    """Create a mock Redis that behaves like a real Redis list store."""
    redis = AsyncMock()
    store: dict[str, list[str]] = {}
    ttls: dict[str, int] = {}

    async def rpush(key: str, value: str) -> int:
        if key not in store:
            store[key] = []
        store[key].append(value)
        return len(store[key])

    async def lrange(key: str, start: int, end: int) -> list[str]:
        if key not in store:
            return []
        items = store[key]
        if end == -1:
            end = len(items)
        else:
            end = end + 1
        return items[start:end]

    async def expire(key: str, seconds: int) -> bool:
        ttls[key] = seconds
        return True

    async def delete(*keys: str) -> int:
        count = 0
        for key in keys:
            if key in store:
                del store[key]
                count += 1
            if key in ttls:
                del ttls[key]
        return count

    async def exists(key: str) -> int:
        return 1 if key in store else 0

    async def get(key: str) -> str | None:
        # For owner keys, they're stored as simple values
        return store.get(key, [None])[0] if key in store else None

    async def set(key: str, value: str, ex: int | None = None) -> None:
        store[key] = [value]
        if ex is not None:
            ttls[key] = ex

    redis.rpush = rpush
    redis.lrange = lrange
    redis.expire = expire
    redis.delete = delete
    redis.exists = exists
    redis.get = get
    redis.set = set
    redis._store = store
    redis._ttls = ttls

    return redis


def _make_message(role: str = "user", content: str = "Hello") -> ChatMessage:
    return ChatMessage(role=role, content=content, timestamp=datetime.now(UTC))


class TestConversationMemory:
    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_add_and_get_history(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        await memory.add_message("session-1", _make_message("user", "Hello"))
        await memory.add_message("session-1", _make_message("assistant", "Hi there"))

        history = await memory.get_history("session-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there"

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_ttl_is_set_on_add(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        await memory.add_message("session-1", _make_message())
        key = "retro-rag:session:session-1:messages"
        assert redis._ttls.get(key) == 86400

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_ttl_renewed_on_add(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        await memory.add_message("session-1", _make_message("user", "msg1"))
        await memory.add_message("session-1", _make_message("user", "msg2"))

        key = "retro-rag:session:session-1:messages"
        assert redis._ttls.get(key) == 86400

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_limit_parameter(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        for i in range(30):
            await memory.add_message(
                "session-1", _make_message("user", f"Message {i}")
            )

        history = await memory.get_history("session-1", limit=5)
        assert len(history) == 5
        assert history[0].content == "Message 25"
        assert history[-1].content == "Message 29"

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_default_limit_20(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        for i in range(30):
            await memory.add_message(
                "session-1", _make_message("user", f"Message {i}")
            )

        history = await memory.get_history("session-1")
        assert len(history) == 20

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_clear_session(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        await memory.add_message("session-1", _make_message())
        assert await memory.session_exists("session-1")

        await memory.clear_session("session-1")
        assert not await memory.session_exists("session-1")

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_nonexistent_session_returns_empty(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        history = await memory.get_history("nonexistent")
        assert history == []

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_session_exists_false_for_new(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)
        assert not await memory.session_exists("nonexistent")

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_owner_set_and_get(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        await memory.set_owner("session-1", 42)
        owner = await memory.get_owner("session-1")
        assert owner == 42

    @pytest.mark.asyncio
    @patch("app.chat.memory.settings")
    async def test_owner_none_for_new_session(self, mock_settings) -> None:
        mock_settings.session_ttl_hours = 24
        redis = _make_mock_redis()
        memory = ConversationMemory(redis)

        owner = await memory.get_owner("nonexistent")
        assert owner is None
