from __future__ import annotations

from typing import Annotated

import asyncpg
import jwt
import redis.asyncio as aioredis
import structlog
from fastapi import Depends, Header, HTTPException

from app.core.config import settings
from app.core.database import get_pg_pool
from app.core.mysql import get_mysql_pool
from app.core.redis import get_redis

logger = structlog.get_logger()


async def _pg_pool() -> asyncpg.Pool:
    return get_pg_pool()


async def _mysql_pool():  # noqa: ANN202
    return get_mysql_pool()


async def _redis() -> aioredis.Redis:
    return get_redis()


async def get_current_user(authorization: str = Header(...)) -> int:
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "message": "Missing authorization header"},
        )
    token = authorization[7:]
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={"error": "token_expired", "message": "JWT token has expired"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "message": "Invalid JWT token"},
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "message": "Invalid JWT token"},
        )
    return int(user_id)


PgPool = Annotated[asyncpg.Pool, Depends(_pg_pool)]
MysqlPool = Annotated[asyncpg.Pool, Depends(_mysql_pool)]
Redis = Annotated[aioredis.Redis, Depends(_redis)]
CurrentUser = Annotated[int, Depends(get_current_user)]
