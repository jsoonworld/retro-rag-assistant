from __future__ import annotations

import asyncio

import asyncpg
import structlog

logger = structlog.get_logger()

_INSERT_SQL = """
INSERT INTO query_log (
    user_id, session_id, query, intent, search_type,
    result_count, latency_ms, token_usage_prompt,
    token_usage_completion, model
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
"""


async def _do_log(
    pg_pool: asyncpg.Pool,
    user_id: int,
    session_id: str | None,
    query: str,
    intent: str | None,
    search_type: str | None,
    result_count: int | None,
    latency_ms: int | None,
    token_usage_prompt: int | None,
    token_usage_completion: int | None,
    model: str | None,
) -> None:
    try:
        async with pg_pool.acquire() as conn:
            await conn.execute(
                _INSERT_SQL,
                user_id,
                session_id,
                query,
                intent,
                search_type,
                result_count,
                latency_ms,
                token_usage_prompt,
                token_usage_completion,
                model,
            )
    except Exception as exc:
        logger.error("query_log_insert_failed", error=str(exc))


def log_query(
    pg_pool: asyncpg.Pool,
    user_id: int,
    session_id: str | None,
    query: str,
    intent: str | None = None,
    search_type: str | None = None,
    result_count: int | None = None,
    latency_ms: int | None = None,
    token_usage_prompt: int | None = None,
    token_usage_completion: int | None = None,
    model: str | None = None,
) -> None:
    """Fire-and-forget query logging."""
    asyncio.create_task(
        _do_log(
            pg_pool,
            user_id,
            session_id,
            query,
            intent,
            search_type,
            result_count,
            latency_ms,
            token_usage_prompt,
            token_usage_completion,
            model,
        )
    )
