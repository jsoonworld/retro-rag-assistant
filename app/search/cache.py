from __future__ import annotations

import hashlib
import json

import structlog
from redis.asyncio import Redis

from app.core.config import settings

logger = structlog.get_logger()

CACHE_PREFIX = "retro-rag:embed:"


class EmbeddingCache:
    def __init__(self, redis: Redis, ttl: int | None = None) -> None:
        self._redis = redis
        self._ttl = ttl or settings.embedding_cache_ttl

    @staticmethod
    def _make_key(text: str) -> str:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{CACHE_PREFIX}{sha}"

    async def get(self, text: str) -> list[float] | None:
        key = self._make_key(text)
        cached = await self._redis.get(key)
        if cached is not None:
            logger.debug("embedding_cache_hit", key=key)
            return json.loads(cached)
        logger.debug("embedding_cache_miss", key=key)
        return None

    async def set(self, text: str, embedding: list[float]) -> None:
        key = self._make_key(text)
        await self._redis.set(key, json.dumps(embedding), ex=self._ttl)

    async def get_or_compute(
        self,
        text: str,
        compute_fn,  # noqa: ANN001 — Callable[[list[str]], Awaitable[list[list[float]]]]
    ) -> list[float]:
        cached = await self.get(text)
        if cached is not None:
            return cached
        embeddings = await compute_fn([text])
        embedding = embeddings[0]
        await self.set(text, embedding)
        return embedding
