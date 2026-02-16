import redis.asyncio as aioredis
import structlog

from app.core.config import settings

logger = structlog.get_logger()

_redis: aioredis.Redis | None = None


async def init_redis() -> aioredis.Redis:
    global _redis
    logger.info("redis_connecting", url=settings.redis_url)

    _redis = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
    )
    # Verify connection
    await _redis.ping()

    logger.info("redis_connected")
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
        logger.info("redis_closed")


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis is not initialized")
    return _redis
