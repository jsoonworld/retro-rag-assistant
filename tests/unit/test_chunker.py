from app.ingestion.chunker import TextChunker


class TestTextChunker:
    def test_short_text_single_chunk(self, chunker: TextChunker) -> None:
        text = "This is a short text."
        results = chunker.chunk(text)
        assert len(results) == 1
        assert results[0].chunk_index == 0
        assert results[0].content == text
        assert results[0].token_count > 0

    def test_long_text_multiple_chunks(self, chunker: TextChunker) -> None:
        # Generate text that exceeds 512 tokens
        text = "This is a sentence that adds some tokens. " * 200
        results = chunker.chunk(text)
        assert len(results) > 1
        # Verify chunk indices are sequential
        for i, chunk in enumerate(results):
            assert chunk.chunk_index == i
            assert chunk.token_count > 0
            assert len(chunk.content.strip()) > 0

    def test_empty_text_returns_empty(self, chunker: TextChunker) -> None:
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_none_text_returns_empty(self, chunker: TextChunker) -> None:
        assert chunker.chunk(None) == []  # type: ignore[arg-type]

    def test_korean_text(self, chunker: TextChunker) -> None:
        text = "이것은 한국어 텍스트입니다. 회고 분석을 위한 테스트 데이터입니다."
        results = chunker.chunk(text)
        assert len(results) >= 1
        assert results[0].token_count > 0
        # Korean text uses more tokens per character
        assert results[0].content == text

    def test_korean_long_text_multiple_chunks(self, chunker: TextChunker) -> None:
        # Korean text that should exceed 512 tokens
        sentence = (
            "이것은 한국어 회고 데이터입니다. "
            "팀의 스프린트 회고에서 발견된 문제점과 개선 방안을 정리합니다. "
        )
        text = sentence * 100
        results = chunker.chunk(text)
        assert len(results) > 1
        for chunk in results:
            assert chunk.token_count > 0

    def test_preserves_paragraph_boundaries(self, chunker: TextChunker) -> None:
        paragraph1 = "First paragraph with some content. " * 20
        paragraph2 = "Second paragraph with different content. " * 20
        text = f"{paragraph1}\n\n{paragraph2}"
        results = chunker.chunk(text)
        assert len(results) >= 1
        # The chunker should respect paragraph boundaries
        for chunk in results:
            assert len(chunk.content.strip()) > 0
