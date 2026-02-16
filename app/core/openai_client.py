from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import openai
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Fallback trigger HTTP status codes
_FALLBACK_STATUS_CODES = {500, 502, 503}
_MAX_RATE_LIMIT_RETRIES = 3


class OpenAIClient:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.openai_api_key.get_secret_value()
        self._client = openai.AsyncOpenAI(api_key=key, timeout=30.0)

    async def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict:
        """Non-streaming chat completion with automatic fallback."""
        primary_model = model or settings.llm_model
        fallback_model = settings.llm_fallback_model

        try:
            return await self._call_chat(messages, primary_model, temperature, max_tokens)
        except (openai.APIStatusError, openai.APITimeoutError) as exc:
            if self._should_fallback(exc):
                logger.warning(
                    "llm_fallback_triggered",
                    primary_model=primary_model,
                    fallback_model=fallback_model,
                    error=str(exc),
                )
                await asyncio.sleep(1.0)
                return await self._call_chat(
                    messages, fallback_model, temperature, max_tokens
                )
            raise

    async def chat_completion_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str]:
        """Streaming chat completion with automatic fallback."""
        primary_model = model or settings.llm_model
        fallback_model = settings.llm_fallback_model

        try:
            async for chunk in self._call_chat_stream(
                messages, primary_model, temperature, max_tokens
            ):
                yield chunk
        except (openai.APIStatusError, openai.APITimeoutError) as exc:
            if self._should_fallback(exc):
                logger.warning(
                    "llm_fallback_triggered_stream",
                    primary_model=primary_model,
                    fallback_model=fallback_model,
                    error=str(exc),
                )
                await asyncio.sleep(1.0)
                async for chunk in self._call_chat_stream(
                    messages, fallback_model, temperature, max_tokens
                ):
                    yield chunk
            else:
                raise

    async def create_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Create embeddings for a list of texts."""
        if not texts:
            return []
        emb_model = model or settings.embedding_model
        response = await self._client.embeddings.create(input=texts, model=emb_model)
        return [item.embedding for item in response.data]

    async def _call_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "content": choice.message.content or "",
            "model": response.model,
            "token_usage": {
                "prompt": usage.prompt_tokens if usage else 0,
                "completion": usage.completion_tokens if usage else 0,
                "model": response.model,
            },
        }

    async def _call_chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def _should_fallback(exc: Exception) -> bool:
        if isinstance(exc, openai.APITimeoutError):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in _FALLBACK_STATUS_CODES or exc.status_code == 429
        return False
