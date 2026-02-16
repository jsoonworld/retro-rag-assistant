import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ingestion.embedder import BATCH_SIZE, EmbeddingService


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_single_text_embedding(
        self, embedding_service: EmbeddingService
    ) -> None:
        results = await embedding_service.embed_texts(["Hello world"])
        assert len(results) == 1
        assert len(results[0]) == 1536

    @pytest.mark.asyncio
    async def test_batch_embedding(
        self, embedding_service: EmbeddingService
    ) -> None:
        texts = [f"Text number {i}" for i in range(10)]
        results = await embedding_service.embed_texts(texts)
        assert len(results) == 10
        for emb in results:
            assert len(emb) == 1536

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(
        self, embedding_service: EmbeddingService
    ) -> None:
        results = await embedding_service.embed_texts([])
        assert results == []

    @pytest.mark.asyncio
    async def test_large_batch_splits(
        self, embedding_service: EmbeddingService
    ) -> None:
        texts = [f"Text {i}" for i in range(BATCH_SIZE + 50)]
        results = await embedding_service.embed_texts(texts)
        assert len(results) == BATCH_SIZE + 50

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self) -> None:
        import openai

        service = EmbeddingService.__new__(EmbeddingService)
        service._model = "text-embedding-3-small"
        service._semaphore = asyncio.Semaphore(5)

        call_count = 0

        async def _mock_create(input, model):  # noqa: A002
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.status_code = 429
                resp.headers = {}
                raise openai.RateLimitError(
                    message="rate limit",
                    response=resp,
                    body=None,
                )
            # Second call succeeds
            data = []
            for _ in input:
                item = MagicMock()
                item.embedding = [0.1] * 1536
                data.append(item)
            response = MagicMock()
            response.data = data
            return response

        client = AsyncMock()
        client.embeddings.create = _mock_create
        service._client = client

        results = await service.embed_texts(["Test"])
        assert len(results) == 1
        assert call_count == 2  # First call failed, second succeeded

    @pytest.mark.asyncio
    async def test_api_error_exhausts_retries(self) -> None:
        import openai

        service = EmbeddingService.__new__(EmbeddingService)
        service._model = "text-embedding-3-small"
        service._semaphore = asyncio.Semaphore(5)

        async def _mock_create(input, model):  # noqa: A002
            resp = MagicMock()
            resp.status_code = 500
            resp.headers = {}
            raise openai.APIError(
                message="server error",
                request=MagicMock(),
                body=None,
            )

        client = AsyncMock()
        client.embeddings.create = _mock_create
        service._client = client

        with pytest.raises(openai.APIError):
            await service.embed_texts(["Test"])
