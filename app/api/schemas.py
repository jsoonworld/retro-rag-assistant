from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, field_validator


def _sanitize_text(value: str) -> str:
    """Strip, remove NULL bytes, remove HTML tags, normalize whitespace."""
    value = value.replace("\x00", "")
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


class ChatRequest(BaseModel):
    message: str
    session_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = _sanitize_text(v)
        if len(v) < 1:
            msg = "Message must not be empty"
            raise ValueError(msg)
        if len(v) > 2000:
            msg = "Message must be at most 2000 characters"
            raise ValueError(msg)
        return v


class SearchRequest(BaseModel):
    query: str
    search_type: Literal["vector", "keyword", "hybrid"] = "hybrid"
    top_k: int = 5

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = _sanitize_text(v)
        if len(v) < 1:
            msg = "Query must not be empty"
            raise ValueError(msg)
        if len(v) > 500:
            msg = "Query must be at most 500 characters"
            raise ValueError(msg)
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1 or v > 20:
            msg = "top_k must be between 1 and 20"
            raise ValueError(msg)
        return v


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    content: str
    score: float
    created_at: datetime


class SearchResponse(BaseModel):
    query: str
    search_type: str
    results: list[SearchResultItem]
    total_results: int
    latency_ms: int


class MessageItem(BaseModel):
    role: str
    content: str
    timestamp: datetime


class SessionResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]
    message_count: int
    created_at: datetime | None
    last_active_at: datetime | None


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: str | None = None
