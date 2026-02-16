from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import aiomysql
import structlog

logger = structlog.get_logger()

FETCH_PAGE_SIZE = 500


@dataclass
class RetrospectiveRow:
    id: int
    title: str
    content: str
    user_id: int | None
    created_at: datetime
    updated_at: datetime


class MySQLFetcher:
    def __init__(self, pool: aiomysql.Pool) -> None:
        self._pool = pool

    async def fetch_all(self) -> list[RetrospectiveRow]:
        """Fetch all retrospectives using keyset pagination."""
        rows: list[RetrospectiveRow] = []
        last_id = 0

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                while True:
                    await cur.execute(
                        "SELECT id, title, content, user_id, created_at, updated_at "
                        "FROM retrospectives "
                        "WHERE id > %s AND content IS NOT NULL AND content != '' "
                        "ORDER BY id ASC LIMIT %s",
                        (last_id, FETCH_PAGE_SIZE),
                    )
                    batch = await cur.fetchall()
                    if not batch:
                        break

                    for row in batch:
                        rows.append(_to_row(row))
                        last_id = row["id"]

        logger.info("mysql_fetched_all", count=len(rows))
        return rows

    async def fetch_incremental(
        self, since: datetime
    ) -> list[RetrospectiveRow]:
        """Fetch retrospectives updated after the given timestamp."""
        rows: list[RetrospectiveRow] = []
        last_id = 0

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                while True:
                    await cur.execute(
                        "SELECT id, title, content, user_id, created_at, updated_at "
                        "FROM retrospectives "
                        "WHERE updated_at > %s AND id > %s "
                        "AND content IS NOT NULL AND content != '' "
                        "ORDER BY id ASC LIMIT %s",
                        (since, last_id, FETCH_PAGE_SIZE),
                    )
                    batch = await cur.fetchall()
                    if not batch:
                        break

                    for row in batch:
                        rows.append(_to_row(row))
                        last_id = row["id"]

        logger.info("mysql_fetched_incremental", since=since.isoformat(), count=len(rows))
        return rows

    async def fetch_all_ids(self) -> set[int]:
        """Fetch all retrospective IDs (for deletion detection)."""
        ids: set[int] = set()
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT id FROM retrospectives")
                result = await cur.fetchall()
                for row in result:
                    ids.add(row[0])
        return ids


def _to_row(row: dict) -> RetrospectiveRow:
    return RetrospectiveRow(
        id=row["id"],
        title=row["title"],
        content=row["content"],
        user_id=row.get("user_id"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
