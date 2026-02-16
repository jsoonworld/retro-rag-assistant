from typing import Annotated

import asyncpg
import redis.asyncio as aioredis
from fastapi import Depends

from app.core.database import get_pg_pool
from app.core.mysql import get_mysql_pool
from app.core.redis import get_redis


async def _pg_pool() -> asyncpg.Pool:
    return get_pg_pool()


async def _mysql_pool():  # noqa: ANN202
    return get_mysql_pool()


async def _redis() -> aioredis.Redis:
    return get_redis()


PgPool = Annotated[asyncpg.Pool, Depends(_pg_pool)]
MysqlPool = Annotated[asyncpg.Pool, Depends(_mysql_pool)]
Redis = Annotated[aioredis.Redis, Depends(_redis)]
