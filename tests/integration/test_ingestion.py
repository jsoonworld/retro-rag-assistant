"""Integration tests for the ingestion pipeline.

These tests require running PostgreSQL, MySQL, and Redis instances.
They use mocked EmbeddingService to avoid OpenAI API costs.
Run with: pytest tests/integration/ -v
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.fetcher import RetrospectiveRow
from app.ingestion.lock import SyncAlreadyRunningError, distributed_lock
from app.ingestion.sync import SyncJob


def _make_row(
    source_id: int = 1,
    title: str = "Test Retrospective",
    content: str = "This is test content for the retrospective.",
    user_id: int = 1,
) -> RetrospectiveRow:
    now = datetime.now(UTC)
    return RetrospectiveRow(
        id=source_id,
        title=title,
        content=content,
        user_id=user_id,
        created_at=now,
        updated_at=now,
    )


def _mock_embedder() -> EmbeddingService:
    """Create an EmbeddingService that returns random vectors."""
    service = MagicMock(spec=EmbeddingService)

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [[random.random() for _ in range(1536)] for _ in texts]  # noqa: S311

    service.embed_texts = _embed
    return service


class TestSyncJobWithMocks:
    """Test SyncJob with mocked dependencies (no real DB needed)."""

    @pytest.mark.asyncio
    async def test_full_sync_processes_documents(self) -> None:
        rows = [_make_row(source_id=i, title=f"Retro {i}") for i in range(1, 4)]

        fetcher = MagicMock()
        fetcher.fetch_all = AsyncMock(return_value=rows)
        fetcher.fetch_all_ids = AsyncMock(return_value={1, 2, 3})

        chunker = TextChunker()
        embedder = _mock_embedder()

        # Mock PG pool - fetchval returns "fake-uuid" for every call
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No existing doc
        mock_conn.fetchval = AsyncMock(return_value="fake-uuid")
        mock_conn.execute = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])  # No existing source_ids

        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        mock_acquired = AsyncMock()
        mock_acquired.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquired.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquired)

        job = SyncJob(mock_pool, fetcher, chunker, embedder)
        result = await job.run_full()

        assert result.status == "completed"
        assert result.documents_processed == 3
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_incremental_sync_no_prior(self) -> None:
        """When no prior sync exists, incremental falls back to full."""
        rows = [_make_row(source_id=1)]

        fetcher = MagicMock()
        fetcher.fetch_all = AsyncMock(return_value=rows)
        fetcher.fetch_all_ids = AsyncMock(return_value={1})

        chunker = TextChunker()
        embedder = _mock_embedder()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        # First fetchval call returns None (no last_sync), rest return "fake-uuid"
        mock_conn.fetchval = AsyncMock(side_effect=[None, "fake-uuid", "fake-uuid"])
        mock_conn.execute = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_transaction)

        mock_acquired = AsyncMock()
        mock_acquired.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquired.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquired)

        job = SyncJob(mock_pool, fetcher, chunker, embedder)
        result = await job.run_incremental()

        assert result.status == "completed"
        # Falls back to full sync
        fetcher.fetch_all.assert_called_once()


class TestDistributedLock:
    @pytest.mark.asyncio
    async def test_lock_acquire_and_release(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.eval = AsyncMock(return_value=1)

        async with distributed_lock(mock_redis) as lock_value:
            assert lock_value  # Should be a UUID string
            mock_redis.set.assert_called_once()

        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_lock_already_held(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=False)  # Lock not acquired

        with pytest.raises(SyncAlreadyRunningError):
            async with distributed_lock(mock_redis):
                pass  # Should not reach here

    @pytest.mark.asyncio
    async def test_lock_released_on_error(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.eval = AsyncMock(return_value=1)

        with pytest.raises(ValueError, match="intentional"):
            async with distributed_lock(mock_redis):
                raise ValueError("intentional")

        # Lock should still be released
        mock_redis.eval.assert_called_once()
