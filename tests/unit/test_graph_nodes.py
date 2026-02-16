from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.models import ChatMessage, GraphState, SearchResult
from app.graph.nodes import (
    check_relevance,
    classify_intent,
    generate_answer,
    handle_fallback,
    retrieve_context,
)
from app.graph.state import create_initial_state


def _make_search_result(
    score: float = 0.8,
    chunk_id: str = "chunk-1",
    content: str = "Test content",
    title: str = "Test Retro",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        score=score,
        source_title=title,
        source_created_at=datetime(2024, 1, 15, tzinfo=UTC),
        search_type="hybrid",
    )


class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_retrospective_search(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value={"content": "retrospective_search", "token_usage": {}}
        )
        state = create_initial_state("지난주 스프린트 회고 요약해줘")
        result = await classify_intent(state, openai_client=client)
        assert result["intent"] == "retrospective_search"

    @pytest.mark.asyncio
    async def test_greeting(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value={"content": "greeting", "token_usage": {}}
        )
        state = create_initial_state("안녕하세요")
        result = await classify_intent(state, openai_client=client)
        assert result["intent"] == "greeting"

    @pytest.mark.asyncio
    async def test_general_question(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value={"content": "general_question", "token_usage": {}}
        )
        state = create_initial_state("Python이란 무엇인가?")
        result = await classify_intent(state, openai_client=client)
        assert result["intent"] == "general_question"

    @pytest.mark.asyncio
    async def test_unclear(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value={"content": "unclear", "token_usage": {}}
        )
        state = create_initial_state("ㅎ")
        result = await classify_intent(state, openai_client=client)
        assert result["intent"] == "unclear"

    @pytest.mark.asyncio
    async def test_invalid_intent_defaults_to_unclear(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value={"content": "some_invalid_intent", "token_usage": {}}
        )
        state = create_initial_state("test")
        result = await classify_intent(state, openai_client=client)
        assert result["intent"] == "unclear"

    @pytest.mark.asyncio
    async def test_llm_error_defaults_to_unclear(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(side_effect=RuntimeError("API error"))
        state = create_initial_state("test query")
        result = await classify_intent(state, openai_client=client)
        assert result["intent"] == "unclear"


class TestRetrieveContext:
    @pytest.mark.asyncio
    async def test_builds_context_string(self) -> None:
        results = [
            _make_search_result(score=0.9, chunk_id="c1", title="Sprint 10 회고"),
            _make_search_result(score=0.7, chunk_id="c2", title="Sprint 9 회고"),
        ]

        hybrid = MagicMock()
        hybrid.search = AsyncMock(return_value=results)

        cache = MagicMock()
        cache.get_or_compute = AsyncMock(return_value=[0.1] * 1536)

        client = MagicMock()

        state = create_initial_state("회고 요약")
        result = await retrieve_context(
            state,
            hybrid_searcher=hybrid,
            embedding_cache=cache,
            openai_client=client,
        )

        assert len(result["search_results"]) == 2
        assert "[회고 1]" in result["context"]
        assert "[회고 2]" in result["context"]
        assert "Sprint 10 회고" in result["context"]

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        hybrid = MagicMock()
        hybrid.search = AsyncMock(return_value=[])

        cache = MagicMock()
        cache.get_or_compute = AsyncMock(return_value=[0.1] * 1536)

        client = MagicMock()

        state = create_initial_state("없는 내용")
        result = await retrieve_context(
            state,
            hybrid_searcher=hybrid,
            embedding_cache=cache,
            openai_client=client,
        )

        assert result["search_results"] == []
        assert result["context"] == ""


class TestCheckRelevance:
    @pytest.mark.asyncio
    async def test_high_relevance(self) -> None:
        state = create_initial_state("test")
        state["search_results"] = [
            _make_search_result(score=0.9),
            _make_search_result(score=0.8),
            _make_search_result(score=0.7),
        ]
        result = await check_relevance(state)
        # Weighted avg: (0.9*0.5 + 0.8*0.3 + 0.7*0.2) / 1.0 = 0.83
        assert result["relevance_score"] == pytest.approx(0.83, abs=0.01)

    @pytest.mark.asyncio
    async def test_low_relevance(self) -> None:
        state = create_initial_state("test")
        state["search_results"] = [
            _make_search_result(score=0.2),
            _make_search_result(score=0.1),
        ]
        result = await check_relevance(state)
        # Weighted avg: (0.2*0.5 + 0.1*0.3) / 0.8 = 0.1625
        assert result["relevance_score"] < 0.4

    @pytest.mark.asyncio
    async def test_no_results_zero_score(self) -> None:
        state = create_initial_state("test")
        state["search_results"] = []
        result = await check_relevance(state)
        assert result["relevance_score"] == 0.0

    @pytest.mark.asyncio
    async def test_single_result(self) -> None:
        state = create_initial_state("test")
        state["search_results"] = [_make_search_result(score=0.6)]
        result = await check_relevance(state)
        # Only one result, weight 0.5, total_weight 0.5: 0.6*0.5/0.5 = 0.6
        assert result["relevance_score"] == pytest.approx(0.6, abs=0.01)


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_generates_answer_with_context(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value={
                "content": "회고에서 자주 나온 문제점은 코드 리뷰 지연입니다.",
                "token_usage": {"prompt": 100, "completion": 50, "model": "gpt-4o"},
            }
        )
        state = create_initial_state("자주 나온 문제점?")
        state["context"] = "[회고 1] Sprint 10 (2024-01-15)\n코드 리뷰가 지연되었습니다."
        result = await generate_answer(state, openai_client=client)
        assert "코드 리뷰" in result["answer"]
        assert result["token_usage"]["prompt"] == 100

    @pytest.mark.asyncio
    async def test_error_returns_fallback_message(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(side_effect=RuntimeError("API down"))
        state = create_initial_state("test")
        result = await generate_answer(state, openai_client=client)
        assert "오류" in result["answer"]
        assert result["error"] is not None


class TestHandleFallback:
    @pytest.mark.asyncio
    async def test_greeting_response(self) -> None:
        client = MagicMock()
        state = create_initial_state("안녕")
        state["intent"] = "greeting"
        result = await handle_fallback(state, openai_client=client)
        assert "안녕하세요" in result["answer"]

    @pytest.mark.asyncio
    async def test_unclear_response(self) -> None:
        client = MagicMock()
        state = create_initial_state("ㅎ")
        state["intent"] = "unclear"
        result = await handle_fallback(state, openai_client=client)
        assert "관련 회고 데이터를 찾지 못했습니다" in result["answer"]

    @pytest.mark.asyncio
    async def test_general_question_calls_llm(self) -> None:
        client = MagicMock()
        client.chat_completion = AsyncMock(
            return_value={
                "content": "Python은 프로그래밍 언어입니다.",
                "token_usage": {"prompt": 50, "completion": 30, "model": "gpt-4o"},
            }
        )
        state = create_initial_state("Python이란?")
        state["intent"] = "general_question"
        result = await handle_fallback(state, openai_client=client)
        assert "Python" in result["answer"]
        client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_results_default_response(self) -> None:
        client = MagicMock()
        state = create_initial_state("없는 내용 검색")
        state["intent"] = "retrospective_search"
        result = await handle_fallback(state, openai_client=client)
        assert "찾지 못했습니다" in result["answer"]
