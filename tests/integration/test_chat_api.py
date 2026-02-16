"""Integration tests for Chat, Search, and Sessions APIs.

Uses FastAPI dependency overrides to inject mocked dependencies.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import _pg_pool, _redis, get_current_user
from app.core.models import ChatMessage, SearchResult
from app.main import create_app

_TEST_SECRET = "test-secret-key-for-integration-32ch"


def _make_jwt(user_id: int = 1) -> str:
    return jwt.encode(
        {"sub": str(user_id), "exp": int(time.time()) + 3600},
        _TEST_SECRET,
        algorithm="HS256",
    )


def _make_search_result(
    chunk_id: str = "c1",
    score: float = 0.8,
    content: str = "Test content",
    title: str = "Sprint 10 회고",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="d1",
        content=content,
        score=score,
        source_title=title,
        source_created_at=datetime(2024, 1, 15, tzinfo=UTC),
        search_type="hybrid",
    )


def _make_mock_redis():
    """Mock Redis that supports the operations we need."""
    redis = AsyncMock()
    store: dict[str, list[str]] = {}

    async def rpush(key, value):
        if key not in store:
            store[key] = []
        store[key].append(value)
        return len(store[key])

    async def lrange(key, start, end):
        if key not in store:
            return []
        items = store[key]
        if end == -1:
            end = len(items)
        else:
            end = end + 1
        return items[start:end]

    async def expire(key, seconds):
        return True

    async def delete(*keys):
        count = 0
        for k in keys:
            if k in store:
                del store[k]
                count += 1
        return count

    async def exists(key):
        return 1 if key in store else 0

    async def get(key):
        return store.get(key, [None])[0] if key in store else None

    async def set_fn(key, value, ex=None):
        store[key] = [value]

    redis.rpush = rpush
    redis.lrange = lrange
    redis.expire = expire
    redis.delete = delete
    redis.exists = exists
    redis.get = get
    redis.set = set_fn
    redis._store = store
    return redis


def _make_mock_pg_pool():
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=0)

    # Make pool.acquire() work as async context manager
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=cm)
    return pool


@pytest.fixture
def mock_redis():
    return _make_mock_redis()


@pytest.fixture
def mock_pg_pool():
    return _make_mock_pg_pool()


@pytest.fixture
def app(mock_redis, mock_pg_pool):
    """Create app with dependency overrides."""
    application = create_app()

    # Mock graph
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "query": "test",
        "intent": "retrospective_search",
        "answer": "Test answer",
        "search_results": [_make_search_result()],
        "token_usage": {"prompt": 100, "completion": 50, "model": "gpt-4o"},
        "context": "",
        "relevance_score": 0.8,
        "messages": [],
        "error": None,
    })
    application.state.graph = mock_graph
    application.state.openai_client = MagicMock()
    application.state.openai_client.create_embeddings = AsyncMock(
        return_value=[[0.1] * 1536]
    )

    # Override dependencies
    async def override_pg_pool():
        return mock_pg_pool

    async def override_redis():
        return mock_redis

    async def override_get_current_user():
        return 1

    application.dependency_overrides[_pg_pool] = override_pg_pool
    application.dependency_overrides[_redis] = override_redis
    application.dependency_overrides[get_current_user] = override_get_current_user

    return application


@pytest.fixture
def app_no_auth(mock_redis, mock_pg_pool):
    """Create app WITHOUT auth override (to test auth failures)."""
    application = create_app()
    application.state.graph = AsyncMock()
    application.state.openai_client = MagicMock()

    async def override_pg_pool():
        return mock_pg_pool

    async def override_redis():
        return mock_redis

    application.dependency_overrides[_pg_pool] = override_pg_pool
    application.dependency_overrides[_redis] = override_redis
    # NOTE: no override for get_current_user

    return application


class TestChatAPINoAuth:
    @pytest.mark.asyncio
    async def test_missing_auth_returns_422(self, app_no_auth) -> None:
        """Missing Authorization header returns 422 (required header)."""
        transport = ASGITransport(app=app_no_auth)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, app_no_auth) -> None:
        transport = ASGITransport(app=app_no_auth)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
                headers={"Authorization": "Bearer invalid-token"},
            )
            assert response.status_code == 401


class TestChatAPIStream:
    @pytest.mark.asyncio
    async def test_chat_returns_sse_stream(self, app, mock_redis) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Test question"},
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            # Parse SSE events
            events = []
            event_type = None
            for line in response.text.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: ") and event_type:
                    events.append((event_type, json.loads(line[6:])))
                    event_type = None

            event_types = [e[0] for e in events]
            assert "metadata" in event_types
            assert "token" in event_types
            assert "sources" in event_types
            assert "done" in event_types

            # Check metadata event
            metadata = next(d for t, d in events if t == "metadata")
            assert "session_id" in metadata
            assert metadata["intent"] == "retrospective_search"

    @pytest.mark.asyncio
    async def test_chat_generates_session_id_when_missing(self, app, mock_redis) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Hello"},
            )
            assert response.status_code == 200

            event_type = None
            events = []
            for line in response.text.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: ") and event_type:
                    events.append((event_type, json.loads(line[6:])))
                    event_type = None

            metadata = next(d for t, d in events if t == "metadata")
            # Should be a valid UUID
            uuid.UUID(metadata["session_id"])

    @pytest.mark.asyncio
    async def test_chat_sources_event_has_results(self, app, mock_redis) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "Test"},
            )
            assert response.status_code == 200

            event_type = None
            events = []
            for line in response.text.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: ") and event_type:
                    events.append((event_type, json.loads(line[6:])))
                    event_type = None

            sources = next(d for t, d in events if t == "sources")
            assert len(sources["results"]) == 1
            assert sources["results"][0]["title"] == "Sprint 10 회고"


class TestSearchAPI:
    @pytest.mark.asyncio
    async def test_search_no_auth_returns_422(self, app_no_auth) -> None:
        transport = ASGITransport(app=app_no_auth)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search",
                json={"query": "test"},
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_returns_json(self, app, mock_pg_pool) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search",
                json={"query": "test query", "search_type": "hybrid", "top_k": 5},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test query"
            assert data["search_type"] == "hybrid"
            assert "results" in data
            assert "total_results" in data
            assert "latency_ms" in data


class TestSessionsAPI:
    @pytest.mark.asyncio
    async def test_get_nonexistent_session_returns_404(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            sid = str(uuid.uuid4())
            response = await client.get(f"/api/v1/sessions/{sid}")
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_returns_204(self, app) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            sid = str(uuid.uuid4())
            response = await client.delete(f"/api/v1/sessions/{sid}")
            assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_already_deleted_session_returns_204(self, app) -> None:
        """Idempotent: deleting a non-existent session still returns 204."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            sid = str(uuid.uuid4())
            await client.delete(f"/api/v1/sessions/{sid}")
            response = await client.delete(f"/api/v1/sessions/{sid}")
            assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_get_session_with_messages(self, app, mock_redis) -> None:
        from app.chat.memory import ConversationMemory

        # Pre-populate session
        memory = ConversationMemory(mock_redis)
        sid = str(uuid.uuid4())
        await memory.set_owner(sid, 1)
        await memory.add_message(
            sid,
            ChatMessage(role="user", content="Hello", timestamp=datetime.now(UTC)),
        )
        await memory.add_message(
            sid,
            ChatMessage(role="assistant", content="Hi there", timestamp=datetime.now(UTC)),
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/api/v1/sessions/{sid}")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == sid
            assert data["message_count"] == 2
            assert len(data["messages"]) == 2
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][1]["role"] == "assistant"
