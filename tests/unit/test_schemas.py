from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.api.schemas import ChatRequest, SearchRequest


class TestChatRequest:
    def test_valid_message(self) -> None:
        req = ChatRequest(message="Hello world")
        assert req.message == "Hello world"
        assert req.session_id is None

    def test_valid_message_with_session(self) -> None:
        sid = uuid.uuid4()
        req = ChatRequest(message="Hello", session_id=sid)
        assert req.session_id == sid

    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_whitespace_only_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="   ")

    def test_over_2000_chars_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="a" * 2001)

    def test_exactly_2000_chars(self) -> None:
        req = ChatRequest(message="a" * 2000)
        assert len(req.message) == 2000

    def test_html_tag_sanitization(self) -> None:
        req = ChatRequest(message="Hello <script>alert('xss')</script> world")
        assert "<script>" not in req.message
        assert "alert" in req.message
        assert req.message == "Hello alert('xss') world"

    def test_whitespace_normalization(self) -> None:
        req = ChatRequest(message="Hello   world   test")
        assert req.message == "Hello world test"

    def test_null_byte_removal(self) -> None:
        req = ChatRequest(message="Hello\x00World")
        assert "\x00" not in req.message

    def test_strips_leading_trailing_whitespace(self) -> None:
        req = ChatRequest(message="  Hello world  ")
        assert req.message == "Hello world"

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", session_id="not-a-uuid")


class TestSearchRequest:
    def test_valid_query(self) -> None:
        req = SearchRequest(query="test query")
        assert req.query == "test query"
        assert req.search_type == "hybrid"
        assert req.top_k == 5

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_over_500_chars_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="a" * 501)

    def test_invalid_search_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="test", search_type="invalid")

    def test_top_k_below_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)

    def test_top_k_above_20_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=21)

    def test_vector_search_type(self) -> None:
        req = SearchRequest(query="test", search_type="vector")
        assert req.search_type == "vector"

    def test_keyword_search_type(self) -> None:
        req = SearchRequest(query="test", search_type="keyword")
        assert req.search_type == "keyword"

    def test_html_sanitization(self) -> None:
        req = SearchRequest(query="<b>bold</b> query")
        assert "<b>" not in req.query
        assert req.query == "bold query"

    def test_whitespace_normalization(self) -> None:
        req = SearchRequest(query="test   query   here")
        assert req.query == "test query here"
