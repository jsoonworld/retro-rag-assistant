import aiomysql
import structlog

from app.core.config import settings

logger = structlog.get_logger()

_pool: aiomysql.Pool | None = None


async def init_mysql_pool() -> aiomysql.Pool:
    global _pool
    logger.info(
        "mysql_pool_creating",
        host=settings.mysql_host,
        database=settings.mysql_database,
    )

    _pool = await aiomysql.create_pool(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password.get_secret_value(),
        db=settings.mysql_database,
        minsize=2,
        maxsize=10,
        autocommit=True,
        charset="utf8mb4",
    )

    logger.info("mysql_pool_created")
    return _pool


async def close_mysql_pool() -> None:
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("mysql_pool_closed")


def get_mysql_pool() -> aiomysql.Pool:
    if _pool is None:
        raise RuntimeError("MySQL pool is not initialized")
    return _pool
