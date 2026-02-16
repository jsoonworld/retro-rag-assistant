from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    source_title: str
    source_created_at: datetime
    search_type: str  # "vector", "keyword", "hybrid"


@dataclass
class ChatMessage:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime


class GraphState(TypedDict, total=False):
    query: str
    intent: str  # "retrospective_search", "general_question", "greeting", "unclear"
    search_results: list[SearchResult]
    relevance_score: float
    context: str
    answer: str
    messages: list[ChatMessage]
    token_usage: dict  # {"prompt": int, "completion": int, "model": str}
    error: str | None
