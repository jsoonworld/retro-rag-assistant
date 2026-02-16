from __future__ import annotations

import time
from dataclasses import dataclass

import asyncpg
import structlog

from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.fetcher import MySQLFetcher, RetrospectiveRow

logger = structlog.get_logger()


@dataclass
class SyncResult:
    status: str
    documents_processed: int
    chunks_created: int
    documents_deleted: int
    duration_ms: int


class SyncJob:
    def __init__(
        self,
        pg_pool: asyncpg.Pool,
        fetcher: MySQLFetcher,
        chunker: TextChunker,
        embedder: EmbeddingService,
    ) -> None:
        self._pg = pg_pool
        self._fetcher = fetcher
        self._chunker = chunker
        self._embedder = embedder

    async def run_full(self) -> SyncResult:
        """Full sync: fetch all, compare, upsert, delete stale."""
        start = time.perf_counter()
        docs_processed = 0
        chunks_created = 0
        docs_deleted = 0

        rows = await self._fetcher.fetch_all()
        mysql_ids = {r.id for r in rows}

        # Detect deleted documents
        existing_source_ids = await self._get_existing_source_ids()
        deleted_ids = existing_source_ids - mysql_ids
        if deleted_ids:
            docs_deleted = await self._delete_documents_by_source_ids(deleted_ids)

        # Process each document
        for row in rows:
            try:
                processed, created = await self._upsert_document(row)
                if processed:
                    docs_processed += 1
                    chunks_created += created
            except Exception:
                logger.exception("sync_document_failed", source_id=row.id)

        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "full_sync_completed",
            documents_processed=docs_processed,
            chunks_created=chunks_created,
            documents_deleted=docs_deleted,
            duration_ms=duration_ms,
        )
        return SyncResult(
            status="completed",
            documents_processed=docs_processed,
            chunks_created=chunks_created,
            documents_deleted=docs_deleted,
            duration_ms=duration_ms,
        )

    async def run_incremental(self) -> SyncResult:
        """Incremental sync: fetch only changed documents since last sync."""
        start = time.perf_counter()
        docs_processed = 0
        chunks_created = 0

        last_sync = await self._get_last_sync_time()
        if last_sync is None:
            # No prior sync, run full
            return await self.run_full()

        rows = await self._fetcher.fetch_incremental(last_sync)

        for row in rows:
            try:
                processed, created = await self._upsert_document(row)
                if processed:
                    docs_processed += 1
                    chunks_created += created
            except Exception:
                logger.exception("sync_document_failed", source_id=row.id)

        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "incremental_sync_completed",
            documents_processed=docs_processed,
            chunks_created=chunks_created,
            duration_ms=duration_ms,
        )
        return SyncResult(
            status="completed",
            documents_processed=docs_processed,
            chunks_created=chunks_created,
            documents_deleted=0,
            duration_ms=duration_ms,
        )

    async def _upsert_document(self, row: RetrospectiveRow) -> tuple[bool, int]:
        """Upsert a single document. Returns (was_processed, chunk_count)."""
        async with self._pg.acquire() as conn:
            async with conn.transaction():
                # Check if document already exists and is unchanged
                existing = await conn.fetchrow(
                    "SELECT id, source_updated_at FROM documents WHERE source_id = $1",
                    row.id,
                )

                if existing and existing["source_updated_at"] >= row.updated_at:
                    return False, 0

                # Delete existing chunks if updating
                if existing:
                    await conn.execute(
                        "DELETE FROM document_chunks WHERE document_id = $1",
                        existing["id"],
                    )
                    await conn.execute(
                        "DELETE FROM documents WHERE id = $1",
                        existing["id"],
                    )

                # Insert document
                doc_id = await conn.fetchval(
                    "INSERT INTO documents "
                    "(source_id, title, content, author_id, "
                    "source_created_at, source_updated_at) "
                    "VALUES ($1, $2, $3, $4, $5, $6) "
                    "RETURNING id",
                    row.id,
                    row.title,
                    row.content,
                    row.user_id,
                    row.created_at,
                    row.updated_at,
                )

                # Chunk the document
                full_text = f"{row.title}\n\n{row.content}"
                chunks = self._chunker.chunk(full_text)

                if not chunks:
                    return True, 0

                # Generate embeddings for all chunks
                chunk_texts = [c.content for c in chunks]
                embeddings = await self._embedder.embed_texts(chunk_texts)

                # Insert chunks with embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    await conn.execute(
                        "INSERT INTO document_chunks "
                        "(document_id, chunk_index, content, embedding, token_count) "
                        "VALUES ($1, $2, $3, $4, $5)",
                        doc_id,
                        chunk.chunk_index,
                        chunk.content,
                        embedding,
                        chunk.token_count,
                    )

                return True, len(chunks)

    async def _get_last_sync_time(self):  # noqa: ANN202
        async with self._pg.acquire() as conn:
            return await conn.fetchval(
                "SELECT MAX(source_updated_at) FROM documents"
            )

    async def _get_existing_source_ids(self) -> set[int]:
        async with self._pg.acquire() as conn:
            rows = await conn.fetch("SELECT source_id FROM documents")
            return {row["source_id"] for row in rows}

    async def _delete_documents_by_source_ids(self, source_ids: set[int]) -> int:
        async with self._pg.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE source_id = ANY($1::bigint[])",
                list(source_ids),
            )
            count = int(result.split()[-1])
            logger.info("documents_deleted", count=count)
            return count
