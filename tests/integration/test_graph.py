"""Integration tests for the full LangGraph RAG pipeline.

Uses mock OpenAI and mock DB to test the complete graph flow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.models import ChatMessage, SearchResult
from app.core.openai_client import OpenAIClient
from app.graph.builder import compile_graph
from app.graph.state import create_initial_state
from app.search.cache import EmbeddingCache
from app.search.hybrid import HybridSearcher
from app.search.keyword import KeywordSearcher
from app.search.vector import VectorSearcher


def _make_search_result(
    chunk_id: str = "c1",
    score: float = 0.8,
    content: str = "스프린트 회고에서 코드 리뷰 지연이 주요 문제로 지적되었습니다.",
    title: str = "Sprint 10 회고",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="d1",
        content=content,
        score=score,
        source_title=title,
        source_created_at=datetime(2024, 1, 15, tzinfo=UTC),
        search_type="hybrid",
    )


def _make_openai_client(
    classify_response: str = "retrospective_search",
    answer_response: str = "회고에서 코드 리뷰 지연이 주요 문제였습니다.",
    should_fail_primary: bool = False,
) -> OpenAIClient:
    client = MagicMock(spec=OpenAIClient)
    call_count = 0

    async def _chat_completion(messages, model=None, temperature=0.3, max_tokens=1024):
        nonlocal call_count
        call_count += 1
        # First call is classify_intent, second is generate_answer
        if call_count == 1:
            return {
                "content": classify_response,
                "token_usage": {"prompt": 50, "completion": 5, "model": model or "gpt-4o-mini"},
            }
        return {
            "content": answer_response,
            "token_usage": {"prompt": 200, "completion": 100, "model": model or "gpt-4o"},
        }

    client.chat_completion = _chat_completion
    client.create_embeddings = AsyncMock(return_value=[[0.1] * 1536])
    return client


def _make_hybrid_searcher(results: list[SearchResult] | None = None) -> HybridSearcher:
    searcher = MagicMock(spec=HybridSearcher)
    if results is None:
        results = [_make_search_result(score=0.8)]
    searcher.search = AsyncMock(return_value=results)
    return searcher


def _make_embedding_cache() -> EmbeddingCache:
    cache = MagicMock(spec=EmbeddingCache)
    cache.get_or_compute = AsyncMock(return_value=[0.1] * 1536)
    return cache


class TestGraphRetroSearch:
    """Scenario 1: retrospective search -> retrieve -> relevance OK -> answer."""

    @pytest.mark.asyncio
    async def test_full_flow_with_relevant_results(self) -> None:
        client = _make_openai_client(
            classify_response="retrospective_search",
            answer_response="코드 리뷰 지연이 주요 문제였습니다.",
        )
        hybrid = _make_hybrid_searcher([_make_search_result(score=0.8)])
        cache = _make_embedding_cache()

        graph = compile_graph(client, hybrid, cache)
        state = create_initial_state("지난주 회고에서 주요 문제점이 뭐였어?")
        result = await graph.ainvoke(state)

        assert result["intent"] == "retrospective_search"
        assert result["relevance_score"] >= 0.4
        assert "코드 리뷰" in result["answer"]


class TestGraphRetroSearchLowRelevance:
    """Scenario 2: retrospective search -> retrieve -> relevance low -> fallback."""

    @pytest.mark.asyncio
    async def test_low_relevance_triggers_fallback(self) -> None:
        client = _make_openai_client(classify_response="retrospective_search")
        # Low score results
        hybrid = _make_hybrid_searcher([_make_search_result(score=0.1)])
        cache = _make_embedding_cache()

        graph = compile_graph(client, hybrid, cache)
        state = create_initial_state("존재하지 않는 내용 검색")
        result = await graph.ainvoke(state)

        assert result["intent"] == "retrospective_search"
        assert result["relevance_score"] < 0.4
        assert "찾지 못했습니다" in result["answer"]


class TestGraphGreeting:
    """Scenario 3: greeting -> direct fallback."""

    @pytest.mark.asyncio
    async def test_greeting_returns_greeting_response(self) -> None:
        client = _make_openai_client(classify_response="greeting")
        hybrid = _make_hybrid_searcher()
        cache = _make_embedding_cache()

        graph = compile_graph(client, hybrid, cache)
        state = create_initial_state("안녕하세요!")
        result = await graph.ainvoke(state)

        assert result["intent"] == "greeting"
        assert "안녕하세요" in result["answer"]
        # Should NOT have called hybrid search
        hybrid.search.assert_not_called()


class TestGraphGeneralQuestion:
    """Scenario 4: general question -> LLM answer without context."""

    @pytest.mark.asyncio
    async def test_general_question_answers_without_context(self) -> None:
        client = _make_openai_client(
            classify_response="general_question",
            answer_response="Python은 범용 프로그래밍 언어입니다.",
        )
        hybrid = _make_hybrid_searcher()
        cache = _make_embedding_cache()

        graph = compile_graph(client, hybrid, cache)
        state = create_initial_state("Python이란 무엇인가?")
        result = await graph.ainvoke(state)

        assert result["intent"] == "general_question"
        assert "Python" in result["answer"]
        # Should NOT have called hybrid search
        hybrid.search.assert_not_called()


class TestGraphEmbeddingCache:
    """Test that embedding cache is used for repeated queries."""

    @pytest.mark.asyncio
    async def test_cache_is_called(self) -> None:
        client = _make_openai_client(classify_response="retrospective_search")
        hybrid = _make_hybrid_searcher()
        cache = _make_embedding_cache()

        graph = compile_graph(client, hybrid, cache)
        state = create_initial_state("회고 검색")
        await graph.ainvoke(state)

        cache.get_or_compute.assert_called_once()
