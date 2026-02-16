import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.ingestion import router as ingestion_router
from app.api.search import router as search_router
from app.api.sessions import router as sessions_router
from app.core.config import settings
from app.core.database import close_pg_pool, get_pg_pool, init_pg_pool
from app.core.logging import setup_logging
from app.core.mysql import close_mysql_pool, init_mysql_pool
from app.core.openai_client import OpenAIClient
from app.core.redis import close_redis, get_redis, init_redis
from app.graph.builder import compile_graph
from app.search.cache import EmbeddingCache
from app.search.hybrid import HybridSearcher
from app.search.keyword import KeywordSearcher
from app.search.vector import VectorSearcher

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    setup_logging()
    logger.info("app_starting")

    # Startup
    await init_pg_pool()
    await init_mysql_pool()
    await init_redis()

    # Compile LangGraph RAG pipeline
    pg_pool = get_pg_pool()
    redis_client = get_redis()
    openai_client = OpenAIClient()
    vector_searcher = VectorSearcher(pg_pool)
    keyword_searcher = KeywordSearcher(pg_pool)
    hybrid_searcher = HybridSearcher(vector_searcher, keyword_searcher)
    embedding_cache = EmbeddingCache(redis_client)
    _app.state.graph = compile_graph(openai_client, hybrid_searcher, embedding_cache)
    _app.state.openai_client = openai_client

    logger.info("app_started")
    yield

    # Shutdown
    await close_redis()
    await close_mysql_pool()
    await close_pg_pool()
    logger.info("app_stopped")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Retro RAG Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @application.middleware("http")
    async def logging_middleware(request: Request, call_next) -> Response:  # noqa: ANN001
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        import time

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # Routers
    application.include_router(health_router)
    application.include_router(ingestion_router, prefix="/api/v1")
    application.include_router(chat_router, prefix="/api/v1")
    application.include_router(search_router, prefix="/api/v1")
    application.include_router(sessions_router, prefix="/api/v1")

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
