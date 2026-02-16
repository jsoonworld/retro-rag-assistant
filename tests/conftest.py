from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingService


@pytest.fixture
def chunker() -> TextChunker:
    return TextChunker(chunk_size=512, chunk_overlap=64)


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client that returns random 1536-dim embeddings."""
    client = AsyncMock()

    async def _create(input: list[str], model: str):  # noqa: A002
        data = []
        for i, _ in enumerate(input):
            embedding = [random.random() for _ in range(1536)]  # noqa: S311
            item = MagicMock()
            item.embedding = embedding
            data.append(item)
        response = MagicMock()
        response.data = data
        return response

    client.embeddings.create = _create
    return client


@pytest.fixture
def embedding_service(mock_openai_client: MagicMock) -> EmbeddingService:
    """EmbeddingService with mocked OpenAI client."""
    service = EmbeddingService.__new__(EmbeddingService)
    service._client = mock_openai_client
    service._model = "text-embedding-3-small"
    service._semaphore = __import__("asyncio").Semaphore(5)
    return service
