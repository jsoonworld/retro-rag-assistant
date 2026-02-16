from __future__ import annotations

from enum import StrEnum

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import MysqlPool, PgPool, Redis
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.fetcher import MySQLFetcher
from app.ingestion.lock import SyncAlreadyRunningError, distributed_lock
from app.ingestion.sync import SyncJob

logger = structlog.get_logger()
router = APIRouter(tags=["ingestion"])


class SyncMode(StrEnum):
    incremental = "incremental"
    full = "full"


class SyncRequest(BaseModel):
    mode: SyncMode = SyncMode.incremental


class SyncResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    duration_ms: int


class SyncStatusResponse(BaseModel):
    last_sync_at: str | None
    total_documents: int
    total_chunks: int
    is_syncing: bool


@router.post("/ingestion/sync", response_model=SyncResponse)
async def sync(
    body: SyncRequest,
    pg_pool: PgPool,
    mysql_pool: MysqlPool,
    redis_client: Redis,
) -> SyncResponse:
    try:
        async with distributed_lock(redis_client):
            fetcher = MySQLFetcher(mysql_pool)
            chunker = TextChunker()
            embedder = EmbeddingService()
            job = SyncJob(pg_pool, fetcher, chunker, embedder)

            if body.mode == SyncMode.full:
                result = await job.run_full()
            else:
                result = await job.run_incremental()

            return SyncResponse(
                status=result.status,
                documents_processed=result.documents_processed,
                chunks_created=result.chunks_created,
                duration_ms=result.duration_ms,
            )
    except SyncAlreadyRunningError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "sync_already_running",
                "message": "Another sync job is in progress",
            },
        )
    except Exception as exc:
        logger.exception("sync_failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "sync_failed",
                "message": str(exc),
            },
        )


@router.get("/ingestion/status", response_model=SyncStatusResponse)
async def status(
    pg_pool: PgPool,
    redis_client: Redis,
) -> SyncStatusResponse:
    async with pg_pool.acquire() as conn:
        last_sync = await conn.fetchval(
            "SELECT MAX(synced_at) FROM documents"
        )
        total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents")
        total_chunks = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")

    # Check if a sync is currently running
    is_syncing = await redis_client.exists("retro-rag:ingestion:sync-lock") > 0

    return SyncStatusResponse(
        last_sync_at=last_sync.isoformat() if last_sync else None,
        total_documents=total_docs,
        total_chunks=total_chunks,
        is_syncing=is_syncing,
    )
