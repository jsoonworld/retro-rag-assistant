from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.search.hybrid import HybridSearcher, RRF_K
from app.search.keyword import KeywordSearcher
from app.search.vector import VectorSearcher
from app.core.models import SearchResult


def _make_pg_row(
    chunk_id: str = "c1",
    document_id: str = "d1",
    content: str = "test content",
    score: float = 0.8,
    title: str = "Retro 1",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": content,
        "score": score,
        "source_title": title,
        "source_created_at": datetime(2024, 1, 15, tzinfo=UTC),
    }


class TestVectorSearcher:
    @pytest.mark.asyncio
    async def test_returns_results_above_min_score(self) -> None:
        rows = [
            _make_pg_row(chunk_id="c1", score=0.9),
            _make_pg_row(chunk_id="c2", score=0.5),
            _make_pg_row(chunk_id="c3", score=0.2),  # Below min_score
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=rows)

        mock_acquired = AsyncMock()
        mock_acquired.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquired.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquired)

        searcher = VectorSearcher(mock_pool)
        results = await searcher.search([0.1] * 1536, top_k=10, min_score=0.3)

        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[1].score == 0.5
        assert all(r.search_type == "vector" for r in results)

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_acquired = AsyncMock()
        mock_acquired.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquired.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquired)

        searcher = VectorSearcher(mock_pool)
        results = await searcher.search([0.1] * 1536)
        assert results == []


class TestKeywordSearcher:
    @pytest.mark.asyncio
    async def test_returns_results(self) -> None:
        rows = [
            _make_pg_row(chunk_id="c1", score=0.5),
            _make_pg_row(chunk_id="c2", score=0.3),
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=rows)

        mock_acquired = AsyncMock()
        mock_acquired.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquired.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquired)

        searcher = KeywordSearcher(mock_pool)
        results = await searcher.search("코드 리뷰", top_k=10)

        assert len(results) == 2
        assert all(r.search_type == "keyword" for r in results)

    @pytest.mark.asyncio
    async def test_korean_query(self) -> None:
        rows = [_make_pg_row(chunk_id="c1", score=0.4, content="한국어 회고 내용")]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=rows)

        mock_acquired = AsyncMock()
        mock_acquired.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquired.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquired)

        searcher = KeywordSearcher(mock_pool)
        results = await searcher.search("회고")
        assert len(results) == 1
        assert results[0].content == "한국어 회고 내용"


class TestHybridSearcher:
    def _make_result(
        self,
        chunk_id: str,
        score: float,
        search_type: str = "vector",
    ) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            document_id="d1",
            content="content",
            score=score,
            source_title="title",
            source_created_at=datetime(2024, 1, 15, tzinfo=UTC),
            search_type=search_type,
        )

    @pytest.mark.asyncio
    async def test_rrf_fusion_correctness(self) -> None:
        """Verify RRF score calculation matches manual computation."""
        vector_results = [
            self._make_result("c1", 0.9, "vector"),
            self._make_result("c2", 0.7, "vector"),
        ]
        keyword_results = [
            self._make_result("c2", 0.5, "keyword"),
            self._make_result("c3", 0.3, "keyword"),
        ]

        vector_searcher = MagicMock()
        vector_searcher.search = AsyncMock(return_value=vector_results)

        keyword_searcher = MagicMock()
        keyword_searcher.search = AsyncMock(return_value=keyword_results)

        alpha = 0.7
        searcher = HybridSearcher(vector_searcher, keyword_searcher, alpha=alpha)
        results = await searcher.search("test", [0.1] * 1536, top_k=10)

        # Manual RRF calculation
        k = RRF_K
        # c1: vector rank 0 -> alpha * 1/(k+1) = 0.7 * 1/61
        c1_expected = alpha * (1.0 / (k + 1))
        # c2: vector rank 1 -> alpha * 1/(k+2) + (1-alpha) * 1/(k+1)
        c2_expected = alpha * (1.0 / (k + 2)) + (1 - alpha) * (1.0 / (k + 1))
        # c3: keyword rank 1 -> (1-alpha) * 1/(k+2)
        c3_expected = (1 - alpha) * (1.0 / (k + 2))

        result_map = {r.chunk_id: r.score for r in results}
        assert result_map["c1"] == pytest.approx(c1_expected, abs=1e-6)
        assert result_map["c2"] == pytest.approx(c2_expected, abs=1e-6)
        assert result_map["c3"] == pytest.approx(c3_expected, abs=1e-6)

    @pytest.mark.asyncio
    async def test_alpha_weight_adjustment(self) -> None:
        """Higher alpha should favor vector results."""
        vector_results = [self._make_result("c1", 0.9, "vector")]
        keyword_results = [self._make_result("c2", 0.5, "keyword")]

        vector_searcher = MagicMock()
        vector_searcher.search = AsyncMock(return_value=vector_results)
        keyword_searcher = MagicMock()
        keyword_searcher.search = AsyncMock(return_value=keyword_results)

        # High alpha (0.9) -> vector favored
        searcher_high = HybridSearcher(vector_searcher, keyword_searcher, alpha=0.9)
        results_high = await searcher_high.search("test", [0.1] * 1536)

        vector_searcher.search = AsyncMock(return_value=vector_results)
        keyword_searcher.search = AsyncMock(return_value=keyword_results)

        # Low alpha (0.1) -> keyword favored
        searcher_low = HybridSearcher(vector_searcher, keyword_searcher, alpha=0.1)
        results_low = await searcher_low.search("test", [0.1] * 1536)

        # With high alpha, c1 (vector) should score higher relative to c2
        high_map = {r.chunk_id: r.score for r in results_high}
        low_map = {r.chunk_id: r.score for r in results_low}

        assert high_map["c1"] > high_map["c2"]  # vector dominates
        assert low_map["c2"] > low_map["c1"]  # keyword dominates

    @pytest.mark.asyncio
    async def test_parallel_execution(self) -> None:
        """Both searchers should be called."""
        vector_searcher = MagicMock()
        vector_searcher.search = AsyncMock(return_value=[])
        keyword_searcher = MagicMock()
        keyword_searcher.search = AsyncMock(return_value=[])

        searcher = HybridSearcher(vector_searcher, keyword_searcher)
        await searcher.search("test", [0.1] * 1536)

        vector_searcher.search.assert_called_once()
        keyword_searcher.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_top_k_limit(self) -> None:
        vector_results = [self._make_result(f"c{i}", 0.9 - i * 0.1) for i in range(5)]
        keyword_results = [
            self._make_result(f"k{i}", 0.5 - i * 0.1, "keyword") for i in range(5)
        ]

        vector_searcher = MagicMock()
        vector_searcher.search = AsyncMock(return_value=vector_results)
        keyword_searcher = MagicMock()
        keyword_searcher.search = AsyncMock(return_value=keyword_results)

        searcher = HybridSearcher(vector_searcher, keyword_searcher)
        results = await searcher.search("test", [0.1] * 1536, top_k=3)
        assert len(results) <= 3
