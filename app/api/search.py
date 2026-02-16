from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Request

from app.api.deps import CurrentUser, PgPool, Redis
from app.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.chat.logger import log_query
from app.search.cache import EmbeddingCache

logger = structlog.get_logger()

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    request: Request,
    user_id: CurrentUser,
    pg_pool: PgPool,
    redis_client: Redis,
) -> SearchResponse:
    start = time.perf_counter()

    openai_client = request.app.state.openai_client
    embedding_cache = EmbeddingCache(redis_client)

    # Get embedding for vector/hybrid search
    query_embedding = await embedding_cache.get_or_compute(
        body.query, openai_client.create_embeddings
    )

    # Import searchers and get pg pool for them
    from app.search.hybrid import HybridSearcher
    from app.search.keyword import KeywordSearcher
    from app.search.vector import VectorSearcher

    vector_searcher = VectorSearcher(pg_pool)
    keyword_searcher = KeywordSearcher(pg_pool)

    if body.search_type == "vector":
        results = await vector_searcher.search(query_embedding, top_k=body.top_k)
    elif body.search_type == "keyword":
        results = await keyword_searcher.search(body.query, top_k=body.top_k)
    else:
        hybrid_searcher = HybridSearcher(vector_searcher, keyword_searcher)
        results = await hybrid_searcher.search(
            query_text=body.query,
            query_embedding=query_embedding,
            top_k=body.top_k,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            title=r.source_title,
            content=r.content,
            score=r.score,
            created_at=r.source_created_at,
        )
        for r in results
    ]

    # Log query (fire-and-forget)
    log_query(
        pg_pool,
        user_id=user_id,
        session_id=None,
        query=body.query,
        search_type=body.search_type,
        result_count=len(results),
        latency_ms=latency_ms,
    )

    return SearchResponse(
        query=body.query,
        search_type=body.search_type,
        results=items,
        total_results=len(items),
        latency_ms=latency_ms,
    )
