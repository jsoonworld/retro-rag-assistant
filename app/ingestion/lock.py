from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger()

LOCK_KEY = "retro-rag:ingestion:sync-lock"
LOCK_TTL = 600  # 10 minutes

# Lua script for atomic unlock (owner verification)
UNLOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class SyncAlreadyRunningError(Exception):
    pass


@asynccontextmanager
async def distributed_lock(
    redis_client: aioredis.Redis,
    key: str = LOCK_KEY,
    ttl: int = LOCK_TTL,
) -> AsyncGenerator[str]:
    """Acquire a distributed lock using Redis SET NX EX.

    Yields the lock value (owner UUID) on success.
    Raises SyncAlreadyRunningError if the lock is already held.
    """
    lock_value = str(uuid.uuid4())
    acquired = await redis_client.set(key, lock_value, nx=True, ex=ttl)

    if not acquired:
        raise SyncAlreadyRunningError("Another sync job is already running")

    logger.info("lock_acquired", key=key, owner=lock_value)
    try:
        yield lock_value
    finally:
        # Atomic unlock with owner verification
        result = await redis_client.eval(UNLOCK_SCRIPT, 1, key, lock_value)
        if result:
            logger.info("lock_released", key=key, owner=lock_value)
        else:
            logger.warning("lock_release_skipped", key=key, owner=lock_value)
