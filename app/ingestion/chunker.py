from __future__ import annotations

from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkResult:
    content: str
    chunk_index: int
    token_count: int


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _token_length(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def chunk(self, text: str) -> list[ChunkResult]:
        if not text or not text.strip():
            return []

        splits = self._splitter.split_text(text)
        results: list[ChunkResult] = []

        for idx, chunk_text in enumerate(splits):
            # Skip empty/whitespace-only chunks
            if not chunk_text.strip():
                continue

            token_count = self._token_length(chunk_text)
            results.append(
                ChunkResult(
                    content=chunk_text,
                    chunk_index=idx,
                    token_count=token_count,
                )
            )

        return results
