from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.core.models import GraphState
from app.core.openai_client import OpenAIClient
from app.graph.nodes import (
    check_relevance,
    classify_intent,
    generate_answer,
    handle_fallback,
    retrieve_context,
)
from app.search.cache import EmbeddingCache
from app.search.hybrid import HybridSearcher


def _route_after_classify(state: GraphState) -> str:
    intent = state.get("intent", "unclear")
    if intent == "retrospective_search":
        return "retrieve_context"
    if intent == "general_question":
        return "generate_answer"
    # greeting, unclear -> fallback
    return "handle_fallback"


def _route_after_relevance(state: GraphState) -> str:
    score = state.get("relevance_score", 0.0)
    if score >= settings.relevance_threshold:
        return "generate_answer"
    return "handle_fallback"


def build_graph(
    openai_client: OpenAIClient,
    hybrid_searcher: HybridSearcher,
    embedding_cache: EmbeddingCache,
) -> StateGraph:
    graph = StateGraph(GraphState)

    # Add nodes — each wraps the corresponding function with injected deps
    graph.add_node(
        "classify_intent",
        partial(classify_intent, openai_client=openai_client),
    )
    graph.add_node(
        "retrieve_context",
        partial(
            retrieve_context,
            hybrid_searcher=hybrid_searcher,
            embedding_cache=embedding_cache,
            openai_client=openai_client,
        ),
    )
    graph.add_node("check_relevance", check_relevance)
    graph.add_node(
        "generate_answer",
        partial(generate_answer, openai_client=openai_client),
    )
    graph.add_node(
        "handle_fallback",
        partial(handle_fallback, openai_client=openai_client),
    )

    # Edges
    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "retrieve_context": "retrieve_context",
            "generate_answer": "generate_answer",
            "handle_fallback": "handle_fallback",
        },
    )

    graph.add_edge("retrieve_context", "check_relevance")

    graph.add_conditional_edges(
        "check_relevance",
        _route_after_relevance,
        {
            "generate_answer": "generate_answer",
            "handle_fallback": "handle_fallback",
        },
    )

    graph.add_edge("generate_answer", END)
    graph.add_edge("handle_fallback", END)

    return graph


def compile_graph(
    openai_client: OpenAIClient,
    hybrid_searcher: HybridSearcher,
    embedding_cache: EmbeddingCache,
):  # noqa: ANN201
    """Build and compile the graph. Returns a Runnable."""
    graph = build_graph(openai_client, hybrid_searcher, embedding_cache)
    return graph.compile()
