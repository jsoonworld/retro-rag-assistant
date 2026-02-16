from __future__ import annotations

import asyncpg
import structlog

from app.core.models import SearchResult

logger = structlog.get_logger()

_KEYWORD_SQL = """
SELECT
    dc.id AS chunk_id,
    dc.document_id,
    dc.content,
    ts_rank(dc.tsv, query) AS score,
    d.title AS source_title,
    d.source_created_at
FROM document_chunks dc
JOIN documents d ON d.id = dc.document_id,
     plainto_tsquery('simple', $1) query
WHERE dc.tsv @@ query
ORDER BY score DESC
LIMIT $2
"""


class KeywordSearcher:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_KEYWORD_SQL, query_text, top_k)

        results: list[SearchResult] = []
        for row in rows:
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    score=float(row["score"]),
                    source_title=row["source_title"],
                    source_created_at=row["source_created_at"],
                    search_type="keyword",
                )
            )
        return results
