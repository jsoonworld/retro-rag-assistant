import asyncpg
import structlog
from pgvector.asyncpg import register_vector

from app.core.config import settings

logger = structlog.get_logger()

_pool: asyncpg.Pool | None = None


async def init_pg_pool() -> asyncpg.Pool:
    global _pool
    logger.info("pg_pool_creating", dsn=settings.database_url)

    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=5,
        max_size=20,
        command_timeout=30,
        init=_init_connection,
    )
    # Ensure pgvector extension exists
    async with _pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    logger.info("pg_pool_created")
    return _pool


async def close_pg_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("pg_pool_closed")


def get_pg_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("PostgreSQL pool is not initialized")
    return _pool
