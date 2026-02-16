from __future__ import annotations

import structlog
import tiktoken

from app.core.config import settings
from app.core.models import GraphState, SearchResult
from app.core.openai_client import OpenAIClient
from app.search.cache import EmbeddingCache
from app.search.hybrid import HybridSearcher

logger = structlog.get_logger()

_INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier. "
    "Classify the user's query into one of the following categories:\n"
    "- retrospective_search: The user wants to search or ask about "
    "team retrospective data, sprint reviews, action items, "
    "or past team discussions.\n"
    "- general_question: The user asks a general question not related "
    "to retrospective data (e.g., \"What is Python?\").\n"
    "- greeting: The user is greeting or making small talk "
    "(e.g., \"Hello\", \"Hi there\").\n"
    "- unclear: The query is ambiguous or too short to classify.\n\n"
    "Respond with ONLY the category name, nothing else.\n\n"
    "Examples:\n"
    "- \"지난주 스프린트 회고 요약해줘\" -> retrospective_search\n"
    "- \"회고에서 자주 나온 문제점이 뭐야?\" -> retrospective_search\n"
    "- \"안녕하세요\" -> greeting\n"
    "- \"Python이란 무엇인가?\" -> general_question\n"
    "- \"ㅎ\" -> unclear"
)

_ANSWER_SYSTEM_PROMPT = """당신은 팀 회고 데이터를 기반으로 질문에 답변하는 AI 어시스턴트입니다.
제공된 회고 컨텍스트만을 기반으로 정확하게 답변하세요.
컨텍스트에 없는 내용은 추측하지 마세요."""

_GENERAL_SYSTEM_PROMPT = (
    "당신은 도움이 되는 AI 어시스턴트입니다. "
    "사용자의 질문에 친절하고 정확하게 답변하세요."
)

_GREETING_RESPONSES = {
    "default": "안녕하세요! 팀 회고 데이터에 대해 궁금한 점이 있으시면 물어보세요.",
}

_NO_RESULTS_RESPONSE = (
    "죄송합니다. 관련 회고 데이터를 찾지 못했습니다. 다른 키워드로 검색해 보시겠어요?"
)

MAX_HISTORY_TURNS = 10
MAX_CONTEXT_TOKENS = 4000


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


async def classify_intent(
    state: GraphState,
    openai_client: OpenAIClient,
) -> GraphState:
    query = state["query"]
    messages = [
        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    try:
        result = await openai_client.chat_completion(
            messages=messages,
            model=settings.llm_fallback_model,  # Use cheaper model for classification
            temperature=0.0,
            max_tokens=20,
        )
        intent = result["content"].strip().lower()
        valid_intents = {"retrospective_search", "general_question", "greeting", "unclear"}
        if intent not in valid_intents:
            intent = "unclear"
    except Exception as exc:
        logger.error("classify_intent_failed", error=str(exc))
        intent = "unclear"

    return {**state, "intent": intent}


async def retrieve_context(
    state: GraphState,
    hybrid_searcher: HybridSearcher,
    embedding_cache: EmbeddingCache,
    openai_client: OpenAIClient,
) -> GraphState:
    query = state["query"]

    # Get embedding (with cache)
    query_embedding = await embedding_cache.get_or_compute(
        query, openai_client.create_embeddings
    )

    # Search
    search_results = await hybrid_searcher.search(
        query_text=query,
        query_embedding=query_embedding,
        top_k=10,
    )

    # Build context string with token limit
    context_parts: list[str] = []
    total_tokens = 0
    kept_results: list[SearchResult] = []

    for i, result in enumerate(search_results):
        date_str = result.source_created_at.strftime("%Y-%m-%d")
        part = f"[회고 {i + 1}] {result.source_title} ({date_str})\n{result.content}"
        part_tokens = _count_tokens(part)

        if total_tokens + part_tokens > MAX_CONTEXT_TOKENS:
            break
        context_parts.append(part)
        total_tokens += part_tokens
        kept_results.append(result)

    context = "\n\n".join(context_parts)

    return {
        **state,
        "search_results": kept_results,
        "context": context,
    }


async def check_relevance(state: GraphState) -> GraphState:
    search_results = state.get("search_results", [])
    if not search_results:
        return {**state, "relevance_score": 0.0}

    # Weighted average of top 3 scores
    top_results = search_results[:3]
    weights = [0.5, 0.3, 0.2][: len(top_results)]
    total_weight = sum(weights)
    relevance_score = sum(r.score * w for r, w in zip(top_results, weights)) / total_weight

    return {**state, "relevance_score": relevance_score}


async def generate_answer(
    state: GraphState,
    openai_client: OpenAIClient,
) -> GraphState:
    context = state.get("context", "")
    query = state["query"]
    chat_messages = state.get("messages", [])

    # Build messages for LLM
    messages: list[dict] = []

    if context:
        messages.append({
            "role": "system",
            "content": f"{_ANSWER_SYSTEM_PROMPT}\n\n--- 회고 컨텍스트 ---\n{context}",
        })
    else:
        messages.append({"role": "system", "content": _GENERAL_SYSTEM_PROMPT})

    # Add recent history (limited)
    recent = chat_messages[-MAX_HISTORY_TURNS * 2 :]
    for msg in recent:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": query})

    try:
        result = await openai_client.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return {
            **state,
            "answer": result["content"],
            "token_usage": result["token_usage"],
        }
    except Exception as exc:
        logger.error("generate_answer_failed", error=str(exc))
        return {
            **state,
            "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "error": str(exc),
        }


async def handle_fallback(
    state: GraphState,
    openai_client: OpenAIClient,
) -> GraphState:
    intent = state.get("intent", "unclear")

    if intent == "greeting":
        return {**state, "answer": _GREETING_RESPONSES["default"]}

    if intent == "general_question":
        # Answer without retrospective context
        return await generate_answer(
            {**state, "context": ""},
            openai_client,
        )

    # Default: no relevant results
    return {**state, "answer": _NO_RESULTS_RESPONSE}
