from __future__ import annotations

import asyncpg
import structlog

from app.core.models import SearchResult

logger = structlog.get_logger()

_VECTOR_SQL = """
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    dc.content,
    1 - (dc.embedding <=> $1) AS score,
    d.title AS source_title,
    d.source_created_at
FROM document_chunks dc
JOIN documents d ON d.id = dc.document_id
WHERE dc.embedding IS NOT NULL
ORDER BY dc.embedding <=> $1
LIMIT $2
"""


class VectorSearcher:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_VECTOR_SQL, query_embedding, top_k)

        results: list[SearchResult] = []
        for row in rows:
            score = float(row["score"])
            if score < min_score:
                continue
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    score=score,
                    source_title=row["source_title"],
                    source_created_at=row["source_created_at"],
                    search_type="vector",
                )
            )
        return results
