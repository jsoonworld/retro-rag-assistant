from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.search.cache import CACHE_PREFIX, EmbeddingCache


class TestEmbeddingCache:
    @pytest.mark.asyncio
    async def test_cache_hit(self) -> None:
        embedding = [0.1] * 1536
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(embedding))

        cache = EmbeddingCache(mock_redis, ttl=3600)
        result = await cache.get("test query")

        assert result == embedding
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_miss(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        cache = EmbeddingCache(mock_redis, ttl=3600)
        result = await cache.get("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set(self) -> None:
        embedding = [0.1] * 1536
        mock_redis = AsyncMock()

        cache = EmbeddingCache(mock_redis, ttl=7200)
        await cache.set("test query", embedding)

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args.kwargs["ex"] == 7200

    @pytest.mark.asyncio
    async def test_get_or_compute_hit(self) -> None:
        embedding = [0.2] * 1536
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(embedding))

        compute_fn = AsyncMock()
        cache = EmbeddingCache(mock_redis, ttl=3600)
        result = await cache.get_or_compute("test", compute_fn)

        assert result == embedding
        compute_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_compute_miss(self) -> None:
        embedding = [0.3] * 1536
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        compute_fn = AsyncMock(return_value=[embedding])
        cache = EmbeddingCache(mock_redis, ttl=3600)
        result = await cache.get_or_compute("test", compute_fn)

        assert result == embedding
        compute_fn.assert_called_once_with(["test"])
        mock_redis.set.assert_called_once()

    def test_sha256_key_generation(self) -> None:
        text = "test query"
        expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        expected_key = f"{CACHE_PREFIX}{expected_sha}"

        key = EmbeddingCache._make_key(text)
        assert key == expected_key

    def test_different_texts_different_keys(self) -> None:
        key1 = EmbeddingCache._make_key("query one")
        key2 = EmbeddingCache._make_key("query two")
        assert key1 != key2
