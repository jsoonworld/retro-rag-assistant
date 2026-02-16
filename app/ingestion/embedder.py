from __future__ import annotations

import asyncio

import openai
import structlog

from app.core.config import settings

logger = structlog.get_logger()

BATCH_SIZE = 100
MAX_RETRIES = 3
BASE_DELAY = 1.0
BATCH_DELAY = 0.5


class EmbeddingService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrency: int = 5,
    ) -> None:
        key = api_key or settings.openai_api_key.get_secret_value()
        self._client = openai.AsyncOpenAI(api_key=key)
        self._model = model or settings.embedding_model
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches with rate limit handling."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

            # Delay between batches for rate limit safety
            if i + BATCH_SIZE < len(texts):
                await asyncio.sleep(BATCH_DELAY)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry and exponential backoff."""
        async with self._semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    response = await self._client.embeddings.create(
                        input=texts,
                        model=self._model,
                    )
                    return [item.embedding for item in response.data]
                except openai.RateLimitError as exc:
                    delay = BASE_DELAY * (2**attempt)
                    retry_after = getattr(exc, "headers", {})
                    if retry_after and hasattr(retry_after, "get"):
                        ra = retry_after.get("retry-after")
                        if ra:
                            delay = float(ra)
                    logger.warning(
                        "embedding_rate_limited",
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                except openai.APIError as exc:
                    if attempt == MAX_RETRIES - 1:
                        logger.error("embedding_failed", error=str(exc))
                        raise
                    delay = BASE_DELAY * (2**attempt)
                    logger.warning(
                        "embedding_api_error",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

            raise RuntimeError("Embedding failed after max retries")
