from __future__ import annotations

import asyncio
from collections import defaultdict

import structlog

from app.core.config import settings
from app.core.models import SearchResult
from app.search.keyword import KeywordSearcher
from app.search.vector import VectorSearcher

logger = structlog.get_logger()

RRF_K = 60  # Standard RRF constant


class HybridSearcher:
    def __init__(
        self,
        vector_searcher: VectorSearcher,
        keyword_searcher: KeywordSearcher,
        alpha: float | None = None,
    ) -> None:
        self._vector = vector_searcher
        self._keyword = keyword_searcher
        self._alpha = alpha if alpha is not None else settings.hybrid_search_alpha

    async def search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        vector_results, keyword_results = await asyncio.gather(
            self._vector.search(query_embedding, top_k=top_k, min_score=min_score),
            self._keyword.search(query_text, top_k=top_k),
        )

        return self._fuse(vector_results, keyword_results, top_k)

    def _fuse(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        # RRF score accumulation by chunk_id
        rrf_scores: dict[str, float] = defaultdict(float)
        result_map: dict[str, SearchResult] = {}

        alpha = self._alpha
        k = RRF_K

        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] += alpha * (1.0 / (k + rank + 1))
            result_map[result.chunk_id] = result

        for rank, result in enumerate(keyword_results):
            rrf_scores[result.chunk_id] += (1.0 - alpha) * (1.0 / (k + rank + 1))
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)  # type: ignore[arg-type]

        results: list[SearchResult] = []
        for chunk_id in sorted_ids[:top_k]:
            original = result_map[chunk_id]
            results.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    document_id=original.document_id,
                    content=original.content,
                    score=rrf_scores[chunk_id],
                    source_title=original.source_title,
                    source_created_at=original.source_created_at,
                    search_type="hybrid",
                )
            )
        return results
