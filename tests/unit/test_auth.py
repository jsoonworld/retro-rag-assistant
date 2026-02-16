from __future__ import annotations

import time
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException

from app.api.deps import get_current_user

_TEST_SECRET = "test-secret-key"


def _make_token(
    sub: int = 1,
    exp: int | None = None,
    secret: str = _TEST_SECRET,
) -> str:
    payload: dict = {"sub": str(sub)}
    if exp is not None:
        payload["exp"] = exp
    else:
        payload["exp"] = int(time.time()) + 3600
    return jwt.encode(payload, secret, algorithm="HS256")


class TestGetCurrentUser:
    @pytest.mark.asyncio
    @patch("app.api.deps.settings")
    async def test_valid_token_extracts_user_id(self, mock_settings) -> None:
        mock_settings.jwt_secret.get_secret_value.return_value = _TEST_SECRET
        token = _make_token(sub=42)
        user_id = await get_current_user(f"Bearer {token}")
        assert user_id == 42

    @pytest.mark.asyncio
    @patch("app.api.deps.settings")
    async def test_expired_token_raises_401(self, mock_settings) -> None:
        mock_settings.jwt_secret.get_secret_value.return_value = _TEST_SECRET
        token = _make_token(exp=int(time.time()) - 100)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(f"Bearer {token}")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "token_expired"

    @pytest.mark.asyncio
    @patch("app.api.deps.settings")
    async def test_invalid_signature_raises_401(self, mock_settings) -> None:
        mock_settings.jwt_secret.get_secret_value.return_value = _TEST_SECRET
        token = _make_token(secret="wrong-secret")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(f"Bearer {token}")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "invalid_token"

    @pytest.mark.asyncio
    async def test_missing_bearer_prefix_raises_401(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("Token some-token")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "unauthorized"

    @pytest.mark.asyncio
    @patch("app.api.deps.settings")
    async def test_missing_sub_claim_raises_401(self, mock_settings) -> None:
        mock_settings.jwt_secret.get_secret_value.return_value = _TEST_SECRET
        payload = {"exp": int(time.time()) + 3600}
        token = jwt.encode(payload, _TEST_SECRET, algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(f"Bearer {token}")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"] == "invalid_token"

    @pytest.mark.asyncio
    @patch("app.api.deps.settings")
    async def test_malformed_token_raises_401(self, mock_settings) -> None:
        mock_settings.jwt_secret.get_secret_value.return_value = _TEST_SECRET
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("Bearer not.a.valid.token")
        assert exc_info.value.status_code == 401
