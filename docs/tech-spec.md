# AI Retrospect Assistant - 기술 명세서

> **문서 버전**: 1.0.0
> **최종 수정**: 2026-02-16
> **기반 문서**: [1-pager.md](./1-pager.md)

---

## 목차

1. [개요](#1-개요)
2. [기술 스택 상세](#2-기술-스택-상세)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [Ingestion Pipeline](#4-ingestion-pipeline)
5. [LangGraph 워크플로우](#5-langgraph-워크플로우)
6. [검색 엔진 (Hybrid Search)](#6-검색-엔진-hybrid-search)
7. [데이터 모델](#7-데이터-모델)
8. [API 상세 설계](#8-api-상세-설계)
9. [인증/인가](#9-인증인가)
10. [에러 처리](#10-에러-처리)
11. [설정 관리](#11-설정-관리)
12. [모니터링](#12-모니터링)
13. [Docker](#13-docker)
14. [테스트 전략](#14-테스트-전략)
15. [구현 페이즈](#15-구현-페이즈)
16. [Open Questions / Future Work](#16-open-questions--future-work)

---

## 1. 개요

### 프로젝트 요약

AI Retrospect Assistant는 moalog-platform 생태계에 추가되는 **RAG(Retrieval-Augmented Generation) 기반 회고 분석 서비스**이다. 기존 moalog-server의 단건 AI 분석(하나의 회고 → OpenAI → 분석 결과)을 넘어, **다수의 과거 회고 데이터를 벡터화하여 종합적 패턴 분석, 맥락 기반 Q&A, 기간 비교** 등을 제공한다.

핵심 사용 시나리오:
- "최근 6개월간 반복되는 문제점이 뭐야?"
- "프론트엔드 팀의 개선 트렌드를 보여줘"
- "지난번 비슷한 문제 때 어떻게 해결했지?"

### Goals

| 목표 | 측정 지표 | 기준값 |
|------|----------|--------|
| 응답 속도 | 첫 토큰 도달 시간 (Time to First Token) | p95 < 2초 |
| 응답 속도 | 전체 응답 완료 시간 | p95 < 10초 |
| 데이터 신선도 | Embedding freshness (최신 회고 반영 지연) | < 5분 |
| 검색 정확도 | Retrieval relevance (top-5 precision) | > 0.8 |
| 가용성 | 서비스 가동률 | > 99.5% |
| 비용 효율 | 월간 OpenAI API 비용 | < $50 (1000 queries/day 기준) |

### Non-Goals

| 제외 항목 | 이유 |
|----------|------|
| 실시간 스트리밍 동기화 (CDC) | 5분 폴링이면 충분. Debezium 등은 과도한 인프라 |
| 자체 Embedding 모델 학습 | 데이터 규모(수천~수만 건)에서 fine-tuning ROI 낮음 |
| 다국어 지원 | 현재 한국어 전용 서비스. 향후 확장 고려 |
| 프론트엔드 구현 | 이 서비스는 API만 제공. 프론트엔드는 별도 |
| 사용자별 개인화 모델 | 팀 단위 회고 데이터가 대상. 개인 맞춤은 범위 밖 |

---

## 2. 기술 스택 상세

### 핵심 의존성

| 패키지 | 버전 | 용도 | 선택 이유 |
|--------|------|------|----------|
| `python` | 3.12 | Runtime | LangGraph/LangChain 생태계가 Python 중심, 3.12의 성능 개선(specializing adaptive interpreter) |
| `fastapi` | ^0.115 | Web framework | async 네이티브, SSE 지원, 자동 OpenAPI 생성, Pydantic v2 통합 |
| `uvicorn[standard]` | ^0.34 | ASGI server | FastAPI 공식 권장, HTTP/2 지원, 성능 우수 |
| `langgraph` | ^0.2 | AI workflow orchestration | 조건부 분기, 상태 관리, 스트리밍 — 멀티스텝 RAG에 최적 |
| `langchain-openai` | ^0.2 | OpenAI LangChain integration | LangGraph 노드에서 OpenAI 호출 통합 |
| `langchain-community` | ^0.3 | Community integrations | pgvector retriever, 기타 통합 도구 |
| `openai` | ^1.60 | OpenAI API client | Embedding 배치 호출, 직접 API 사용 시 |
| `asyncpg` | ^0.30 | PostgreSQL async driver | SQLAlchemy async 백엔드, pgvector 호환 |
| `pgvector` | ^0.3 | pgvector Python bindings | vector 타입 등록, SQLAlchemy 통합 |
| `redis[hiredis]` | ^5.2 | Redis async client (aioredis) | 대화 메모리, TTL 기반 세션 관리, hiredis로 성능 향상 |
| `sqlalchemy[asyncio]` | ^2.0 | Async ORM | asyncpg 기반 비동기 쿼리, pgvector 타입 매핑 |
| `pydantic` | ^2.10 | Data validation | FastAPI 통합, 설정 관리, 스키마 정의 |
| `pydantic-settings` | ^2.7 | Configuration | 환경 변수 → 타입 안전 설정 객체 |
| `sse-starlette` | ^2.2 | SSE support | FastAPI 호환 Server-Sent Events |
| `prometheus-client` | ^0.21 | Metrics | 토큰 사용량, 응답 시간 메트릭 노출 |
| `httpx` | ^0.28 | HTTP client | async HTTP 호출 (moalog-server 연동 등) |
| `aiomysql` | ^0.2 | MySQL async driver | moalog MySQL 읽기 전용 연결 |
| `tiktoken` | ^0.8 | Token counter | 컨텍스트 윈도우 토큰 예산 관리 |
| `python-jose[cryptography]` | ^3.3 | JWT handling | moalog-server와 동일 HS256 JWT 검증 |

### 개발/테스트 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `pytest` | ^8.0 | 테스트 프레임워크 |
| `pytest-asyncio` | ^0.24 | 비동기 테스트 지원 |
| `httpx` | ^0.28 | 테스트 클라이언트 (TestClient) |
| `ruff` | ^0.9 | Linter + Formatter |
| `mypy` | ^1.14 | 정적 타입 체크 |
| `coverage` | ^7.0 | 테스트 커버리지 |

---

## 3. 프로젝트 구조

```
retro-rag-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI 앱 팩토리, 라이프사이클, 미들웨어
│   ├── config.py                   # pydantic-settings 기반 설정
│   ├── dependencies.py             # FastAPI Depends (DI 컨테이너)
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── models.py               # 핵심 도메인 모델 (Pydantic)
│   │   └── schemas.py              # API Request/Response 스키마
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── sync_job.py             # MySQL → pgvector 동기화 잡
│   │   ├── chunker.py              # 회고 → 청크 변환
│   │   └── embedder.py             # OpenAI Embedding 호출
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_search.py        # pgvector 코사인 유사도 검색
│   │   ├── keyword_search.py       # ts_vector 풀텍스트 검색
│   │   └── hybrid.py               # RRF (Reciprocal Rank Fusion)
│   │
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── graph.py                # LangGraph StateGraph 정의
│   │   ├── nodes.py                # 그래프 노드 구현
│   │   └── state.py                # AssistantState TypedDict
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   └── redis_memory.py         # Redis 기반 대화 이력 관리
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py                 # POST /assistant/chat (SSE)
│   │   ├── search.py               # POST /assistant/search
│   │   ├── session.py              # GET/DELETE /assistant/sessions
│   │   ├── ingestion.py            # POST /ingestion/sync, GET /ingestion/status
│   │   └── health.py               # GET /health
│   │
│   └── infrastructure/
│       ├── __init__.py
│       ├── database.py             # AsyncSession 팩토리 (PostgreSQL)
│       ├── mysql_client.py         # MySQL 읽기 전용 연결
│       ├── redis_client.py         # Redis 연결 관리
│       └── openai_client.py        # OpenAI 클라이언트 래퍼
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # 공통 fixture (DB, Redis mock)
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_hybrid_search.py
│   │   └── test_query_analyzer.py
│   ├── integration/
│   │   ├── test_vector_search.py
│   │   ├── test_redis_memory.py
│   │   └── test_ingestion.py
│   └── e2e/
│       └── test_chat_flow.py
│
├── Dockerfile
├── pyproject.toml
├── .env.example
└── docs/
    ├── 1-pager.md
    └── tech-spec.md                # 이 문서
```

### 디렉토리 역할 규칙

| 디렉토리 | 역할 | 금지 사항 |
|----------|------|----------|
| `domain/` | 순수 데이터 모델, 비즈니스 규칙 | 외부 I/O 직접 호출 금지 |
| `ingestion/` | 데이터 수집/변환 파이프라인 | API 핸들러 로직 혼합 금지 |
| `retrieval/` | 검색 로직 (vector + keyword + fusion) | 생성(generation) 로직 혼합 금지 |
| `workflow/` | LangGraph 워크플로우 오케스트레이션 | 인프라 직접 참조 금지 (DI 사용) |
| `memory/` | 대화 상태 관리 | 검색/생성 로직 금지 |
| `api/` | HTTP 핸들러 (얇은 레이어) | 비즈니스 로직 직접 구현 금지 |
| `infrastructure/` | 외부 시스템 연결 (DB, Redis, OpenAI) | 도메인 로직 금지 |

---

## 4. Ingestion Pipeline

### 개요

moalog-server의 MySQL DB에서 회고 데이터를 읽어, 청크 단위로 분할 후 OpenAI Embedding을 생성하여 pgvector에 저장하는 파이프라인이다.

```
moalog MySQL (read-only)
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Sync Job    │───▶│  Chunker     │───▶│  Embedder    │───▶│  pgvector    │
│ (5분 폴링)   │    │ (회고→청크)   │    │ (OpenAI API) │    │  (UPSERT)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### MySQL 소스 스키마 (moalog-server)

moalog-server의 SeaORM 엔티티에서 확인한 테이블 구조:

```sql
-- retrospects 테이블
-- Fields: retrospect_id, title, insight, retrospect_method, created_at, updated_at,
--         start_time, retrospect_room_id

-- retro_room 테이블
-- Fields: retrospect_room_id, title, description, invition_url,
--         invite_code_created_at, created_at, updated_at

-- response 테이블
-- Fields: response_id, question, response (content), created_at, updated_at, retrospect_id

-- 회고 방식(RetrospectMethod): KPT, FOUR_L, FIVE_F, PMI, FREE
```

### 동기화 전략: Incremental Sync

```
last_sync_time = SELECT last_synced_at FROM sync_status WHERE source = 'retrospects'

new/updated = SELECT * FROM retrospects WHERE updated_at > last_sync_time
    JOIN retro_room ON retrospect_room_id
    JOIN response ON retrospect_id
```

- **폴링 주기**: 5분 (`asyncio.sleep(300)`)
- **증분 처리**: `updated_at > last_sync` 조건으로 새로 생성/수정된 회고만 처리
- **멱등성**: `ON CONFLICT (retrospect_id, chunk_type, category) DO UPDATE`로 중복 방지

### 청킹 전략

회고 1건 → N개 청크로 분할:

| 청크 타입 | 내용 구성 | 메타데이터 | 예시 |
|----------|----------|-----------|------|
| `summary` | `[{room_title}] {retro_title} ({method}) {date} 참가자 {count}명` | room_id, method, date | `[백엔드팀] Sprint 3 회고 (KPT) 2026-01-15 5명` |
| `response` | 질문(question)별로 그룹핑: `[{question}] {response_1}\n{response_2}...` | room_id, method, date, question | `[개선이 필요한 문제점은?] - 코드 리뷰 지연\n- 배포 병목` |
| `analysis` | `[AI 분석] {insight}` (insight 필드가 있는 경우에만) | room_id, method, date | `[AI 분석] 주요 이슈는 코드 리뷰 프로세스...` |

### chunker.py

```python
"""회고 데이터를 검색 가능한 청크 단위로 분할한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetrospectChunk:
    """Embedding 대상이 되는 청크 단위."""

    retrospect_id: int
    chunk_type: str          # summary | response | analysis
    category: str | None     # question text 또는 None
    content: str             # 임베딩할 텍스트
    metadata: dict           # room_id, method, date 등


@dataclass
class RetrospectData:
    """MySQL에서 가져온 회고 원본 데이터."""

    retrospect_id: int
    title: str
    insight: str | None
    method: str                          # KPT, FOUR_L, FIVE_F, PMI, FREE
    start_time: datetime
    room_id: int
    room_title: str
    participant_count: int
    responses: list[ResponseData]


@dataclass
class ResponseData:
    """회고 응답 데이터."""

    response_id: int
    question: str
    content: str


def chunk_retrospect(retro: RetrospectData) -> list[RetrospectChunk]:
    """회고 1건을 N개의 청크로 분할한다.

    청킹 전략:
    - summary: 방 이름 + 제목 + 방법 + 날짜 + 참가자 수
    - response: 질문(question)별로 응답을 그룹핑
    - analysis: AI 분석 결과 (insight가 있는 경우에만)
    """
    chunks: list[RetrospectChunk] = []
    base_metadata = {
        "room_id": retro.room_id,
        "room_title": retro.room_title,
        "method": retro.method,
        "date": retro.start_time.isoformat(),
        "title": retro.title,
    }

    # 1) Summary 청크
    date_str = retro.start_time.strftime("%Y-%m-%d")
    summary_text = (
        f"[{retro.room_title}] {retro.title} "
        f"({retro.method}) {date_str} "
        f"참가자 {retro.participant_count}명"
    )
    chunks.append(
        RetrospectChunk(
            retrospect_id=retro.retrospect_id,
            chunk_type="summary",
            category=None,
            content=summary_text,
            metadata=base_metadata,
        )
    )

    # 2) Response 청크 — 질문(question)별 그룹핑
    question_groups: dict[str, list[str]] = {}
    for resp in retro.responses:
        question_groups.setdefault(resp.question, []).append(resp.content)

    for question, answers in question_groups.items():
        response_text = f"[{question}]\n" + "\n".join(
            f"- {answer}" for answer in answers
        )
        chunks.append(
            RetrospectChunk(
                retrospect_id=retro.retrospect_id,
                chunk_type="response",
                category=question,
                content=response_text,
                metadata={**base_metadata, "question": question},
            )
        )

    # 3) Analysis 청크 — insight가 있는 경우에만
    if retro.insight:
        analysis_text = f"[AI 분석] {retro.insight}"
        chunks.append(
            RetrospectChunk(
                retrospect_id=retro.retrospect_id,
                chunk_type="analysis",
                category=None,
                content=analysis_text,
                metadata=base_metadata,
            )
        )

    return chunks
```

### sync_job.py

```python
"""MySQL → pgvector 증분 동기화 잡."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiomysql
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.ingestion.chunker import (
    RetrospectData,
    ResponseData,
    chunk_retrospect,
)
from app.ingestion.embedder import batch_embed

logger = logging.getLogger(__name__)

# --- MySQL 쿼리 -----------------------------------------------------------

FETCH_RETROSPECTS_SQL = """
    SELECT
        r.retrospect_id,
        r.title,
        r.insight,
        r.retrospect_method  AS method,
        r.start_time,
        r.retrospect_room_id AS room_id,
        rr.title             AS room_title,
        r.updated_at,
        (
            SELECT COUNT(DISTINCT mr.member_id)
            FROM member_retro mr
            WHERE mr.retrospect_id = r.retrospect_id
        ) AS participant_count
    FROM retrospects r
    JOIN retro_room rr ON r.retrospect_room_id = rr.retrospect_room_id
    WHERE r.updated_at > %s
    ORDER BY r.updated_at ASC
"""

FETCH_RESPONSES_SQL = """
    SELECT response_id, question, response AS content
    FROM response
    WHERE retrospect_id = %s
"""

# --- 동기화 로직 -------------------------------------------------------------

async def get_last_sync_time(pg_session: AsyncSession) -> datetime:
    """pgvector DB에서 마지막 동기화 시각을 조회한다."""
    result = await pg_session.execute(
        text(
            "SELECT last_synced_at FROM sync_status "
            "WHERE source = 'retrospects'"
        )
    )
    row = result.fetchone()
    if row is None:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    return row[0].replace(tzinfo=timezone.utc)


async def update_last_sync_time(
    pg_session: AsyncSession,
    synced_at: datetime,
) -> None:
    """마지막 동기화 시각을 갱신한다."""
    await pg_session.execute(
        text(
            "INSERT INTO sync_status (source, last_synced_at) "
            "VALUES ('retrospects', :ts) "
            "ON CONFLICT (source) DO UPDATE SET last_synced_at = :ts"
        ),
        {"ts": synced_at},
    )
    await pg_session.commit()


async def fetch_retrospects_from_mysql(
    mysql_pool: aiomysql.Pool,
    since: datetime,
) -> list[RetrospectData]:
    """MySQL에서 since 이후 변경된 회고 목록을 가져온다."""
    retrospects: list[RetrospectData] = []

    async with mysql_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(FETCH_RETROSPECTS_SQL, (since,))
            rows = await cur.fetchall()

            for row in rows:
                # 각 회고의 응답 데이터 조회
                await cur.execute(
                    FETCH_RESPONSES_SQL, (row["retrospect_id"],)
                )
                resp_rows = await cur.fetchall()
                responses = [
                    ResponseData(
                        response_id=r["response_id"],
                        question=r["question"],
                        content=r["content"],
                    )
                    for r in resp_rows
                ]

                retrospects.append(
                    RetrospectData(
                        retrospect_id=row["retrospect_id"],
                        title=row["title"],
                        insight=row.get("insight"),
                        method=row["method"],
                        start_time=row["start_time"],
                        room_id=row["room_id"],
                        room_title=row["room_title"],
                        participant_count=row["participant_count"],
                        responses=responses,
                    )
                )

    return retrospects


async def upsert_embeddings(
    pg_session: AsyncSession,
    chunks: list[dict],
) -> int:
    """pgvector에 임베딩을 UPSERT한다. 반환값은 처리된 행 수."""
    if not chunks:
        return 0

    upserted = 0
    for chunk in chunks:
        await pg_session.execute(
            text(
                """
                INSERT INTO retrospect_embeddings
                    (retrospect_id, chunk_type, category, content,
                     embedding, metadata, updated_at)
                VALUES
                    (:retrospect_id, :chunk_type, :category, :content,
                     :embedding, :metadata::jsonb, NOW())
                ON CONFLICT (retrospect_id, chunk_type, category)
                DO UPDATE SET
                    content    = EXCLUDED.content,
                    embedding  = EXCLUDED.embedding,
                    metadata   = EXCLUDED.metadata,
                    updated_at = NOW()
                """
            ),
            chunk,
        )
        upserted += 1

    await pg_session.commit()
    return upserted


async def run_sync_once(
    mysql_pool: aiomysql.Pool,
    pg_session: AsyncSession,
    settings: Settings,
) -> dict:
    """1회 동기화를 실행한다. 결과 통계를 반환."""
    last_sync = await get_last_sync_time(pg_session)
    logger.info("Sync starting — last_sync=%s", last_sync.isoformat())

    # 1) MySQL에서 변경된 회고 조회
    retrospects = await fetch_retrospects_from_mysql(mysql_pool, last_sync)
    if not retrospects:
        logger.info("No new retrospects since %s", last_sync.isoformat())
        return {"synced": 0, "chunks": 0}

    # 2) 청킹
    all_chunks = []
    for retro in retrospects:
        all_chunks.extend(chunk_retrospect(retro))

    # 3) 배치 임베딩 (최대 2048개씩)
    contents = [c.content for c in all_chunks]
    embeddings = await batch_embed(contents, settings)

    # 4) UPSERT용 데이터 구성
    import json

    upsert_data = []
    for chunk, embedding in zip(all_chunks, embeddings):
        upsert_data.append(
            {
                "retrospect_id": chunk.retrospect_id,
                "chunk_type": chunk.chunk_type,
                "category": chunk.category or "",
                "content": chunk.content,
                "embedding": str(embedding),
                "metadata": json.dumps(
                    chunk.metadata, ensure_ascii=False
                ),
            }
        )

    # 5) pgvector에 UPSERT
    upserted = await upsert_embeddings(pg_session, upsert_data)

    # 6) sync_status 갱신
    latest_updated = max(r.start_time for r in retrospects)
    await update_last_sync_time(
        pg_session,
        latest_updated.replace(tzinfo=timezone.utc),
    )

    result = {"synced": len(retrospects), "chunks": upserted}
    logger.info("Sync complete — %s", result)
    return result


async def sync_loop(
    mysql_pool: aiomysql.Pool,
    pg_session_factory,
    settings: Settings,
) -> None:
    """5분 간격으로 동기화를 반복 실행한다."""
    while True:
        try:
            async with pg_session_factory() as session:
                await run_sync_once(mysql_pool, session, settings)
        except Exception:
            logger.exception("Sync job failed — will retry next cycle")

        await asyncio.sleep(settings.sync_interval_seconds)
```

### embedder.py

```python
"""OpenAI Embedding 배치 처리."""

from __future__ import annotations

import logging
from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)

# OpenAI Embedding API 배치 제한
MAX_BATCH_SIZE = 2048


async def batch_embed(
    texts: list[str],
    settings: Settings,
) -> list[list[float]]:
    """텍스트 목록을 OpenAI text-embedding-3-small로 배치 임베딩한다.

    2048개씩 분할하여 요청하고, 결과를 순서대로 합친다.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        logger.info(
            "Embedded batch %d-%d (%d tokens used)",
            i,
            i + len(batch),
            response.usage.total_tokens,
        )

    return all_embeddings
```

---

## 5. LangGraph 워크플로우

### AssistantState 정의

```python
"""LangGraph 상태 정의."""

from __future__ import annotations

from typing import TypedDict


class Citation(TypedDict):
    """검색 결과에서 추출한 출처 정보."""

    retrospect_id: int
    title: str
    room_title: str
    date: str
    relevance: float


class RetrievedChunk(TypedDict):
    """검색으로 가져온 청크."""

    content: str
    metadata: dict
    score: float


class AssistantState(TypedDict, total=False):
    """LangGraph 워크플로우 전역 상태.

    각 노드는 이 상태를 읽고 수정한다.
    """

    # --- 입력 ---
    query: str                          # 사용자 질문 원문
    session_id: str                     # 세션 ID
    user_id: str                        # JWT에서 추출한 사용자 ID
    filters: dict                       # room_ids, date_from, date_to 등

    # --- Query Analyzer 출력 ---
    intent: str                         # pattern_analysis | search | compare | general
    search_queries: list[str]           # 검색에 사용할 재작성된 쿼리
    search_filters: dict                # 분석된 필터 (날짜, 방, 방법)

    # --- Retriever 출력 ---
    retrieved_chunks: list[RetrievedChunk]

    # --- Context Builder 출력 ---
    context: str                        # GPT-4o에 전달할 최적화된 컨텍스트
    token_count: int                    # 컨텍스트 토큰 수

    # --- Generator 출력 ---
    answer: str                         # 생성된 답변 전문
    answer_stream: object               # SSE 스트리밍용 async generator

    # --- Format 출력 ---
    citations: list[Citation]           # 출처 목록

    # --- 대화 이력 ---
    conversation_history: list[dict]    # Redis에서 로드한 이전 대화
```

### graph.py (LangGraph StateGraph)

```python
"""LangGraph StateGraph 정의 — 의도별 조건부 분기 RAG 워크플로우."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.workflow.nodes import (
    analyze_query,
    build_context,
    format_response,
    generate_answer,
    retrieve,
)
from app.workflow.state import AssistantState


def route_by_intent(state: AssistantState) -> str:
    """Query Analyzer가 분류한 intent에 따라 다음 노드를 결정한다.

    - pattern_analysis: 넓은 범위 검색 → 패턴 추출
    - search: 키워드+벡터 하이브리드 검색
    - compare: 기간별/팀별 분리 검색 후 비교
    - general: 검색 불필요, 바로 LLM 답변
    """
    intent = state.get("intent", "search")
    if intent == "general":
        return "generate_answer"
    return "retrieve"


def build_graph() -> StateGraph:
    """RAG 워크플로우 그래프를 생성한다."""
    workflow = StateGraph(AssistantState)

    # 노드 등록
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("build_context", build_context)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("format_response", format_response)

    # 엣지 설정
    workflow.set_entry_point("analyze_query")

    # 의도별 조건부 분기
    workflow.add_conditional_edges(
        "analyze_query",
        route_by_intent,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
        },
    )

    # 검색 → 컨텍스트 빌드 → 답변 생성 → 포맷
    workflow.add_edge("retrieve", "build_context")
    workflow.add_edge("build_context", "generate_answer")
    workflow.add_edge("generate_answer", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()


# 싱글턴 그래프 인스턴스
assistant_graph = build_graph()
```

### nodes.py (노드 구현)

```python
"""LangGraph 노드 구현 — 각 노드는 AssistantState를 받아 부분 업데이트를 반환한다."""

from __future__ import annotations

import json
import logging
from typing import Any

import tiktoken
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.retrieval.hybrid import hybrid_search
from app.workflow.state import AssistantState, Citation, RetrievedChunk

logger = logging.getLogger(__name__)

# --- 상수 ----------------------------------------------------------------

INTENT_CLASSIFICATION_PROMPT = """당신은 회고(retrospective) 데이터 분석 어시스턴트입니다.
사용자의 질문 의도를 다음 4가지 중 하나로 분류하세요.

의도 유형:
- pattern_analysis: 반복되는 패턴, 트렌드, 빈도 분석 (예: "반복되는 문제?", "개선 트렌드")
- search: 특정 주제/키워드 검색 (예: "Redis 관련 회고", "배포 관련")
- compare: 기간, 팀, 방법 비교 (예: "1월 vs 2월", "백엔드 vs 프론트")
- general: 회고 데이터 검색 불필요한 일반 질문 (예: "KPT가 뭐야?", "회고 방법 추천")

대화 이력:
{conversation_history}

사용자 질문: {query}

JSON으로만 응답하세요:
{{"intent": "...", "search_queries": ["재작성된 검색 쿼리1", ...], "filters": {{"date_from": null, "date_to": null, "room_ids": null, "method": null}}}}
"""

ANSWER_SYSTEM_PROMPT = """당신은 moalog 회고 분석 어시스턴트입니다.
제공된 회고 데이터(컨텍스트)를 기반으로 사용자의 질문에 답변하세요.

규칙:
1. 반드시 제공된 컨텍스트에 기반하여 답변하세요.
2. 컨텍스트에 없는 내용은 추측하지 마세요.
3. 출처(어떤 회고에서 왔는지)를 명시하세요.
4. 한국어로 답변하세요.
5. 패턴 분석 시 빈도와 구체적 사례를 함께 제시하세요.

컨텍스트:
{context}

대화 이력:
{conversation_history}
"""

MAX_CONTEXT_TOKENS = 6000
ENCODING = tiktoken.encoding_for_model("gpt-4o")


# --- 노드 구현 -----------------------------------------------------------

async def analyze_query(state: AssistantState) -> dict[str, Any]:
    """사용자 질문의 의도를 분류하고, 검색 쿼리를 재작성한다.

    GPT-4o-mini를 사용하여 빠르고 저렴하게 처리.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
        temperature=0,
    )

    # 대화 이력 포맷
    history = state.get("conversation_history", [])
    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-6:]
    ) or "(없음)"

    prompt = INTENT_CLASSIFICATION_PROMPT.format(
        query=state["query"],
        conversation_history=history_text,
    )

    response = await llm.ainvoke(prompt)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # 파싱 실패 시 기본값
        parsed = {
            "intent": "search",
            "search_queries": [state["query"]],
            "filters": {},
        }

    logger.info("Query analyzed — intent=%s", parsed.get("intent"))

    # 사용자가 명시적으로 전달한 filters와 LLM이 추출한 filters를 병합
    user_filters = state.get("filters", {})
    llm_filters = parsed.get("filters", {})
    merged_filters = {
        k: v
        for k, v in {**llm_filters, **user_filters}.items()
        if v is not None
    }

    return {
        "intent": parsed.get("intent", "search"),
        "search_queries": parsed.get("search_queries", [state["query"]]),
        "search_filters": merged_filters,
    }


async def retrieve(state: AssistantState) -> dict[str, Any]:
    """하이브리드 검색(vector + keyword + RRF)으로 관련 청크를 가져온다."""
    search_queries = state.get("search_queries", [state["query"]])
    filters = state.get("search_filters", {})
    intent = state.get("intent", "search")

    # intent에 따라 top_k 조절
    top_k_map = {
        "pattern_analysis": 20,
        "compare": 15,
        "search": 10,
    }
    top_k = top_k_map.get(intent, 10)

    all_chunks: list[RetrievedChunk] = []
    seen_ids: set[str] = set()

    for query in search_queries:
        results = await hybrid_search(
            query=query,
            filters=filters,
            top_k=top_k,
        )
        for r in results:
            # 중복 제거 (retrospect_id + chunk_type + category)
            chunk_id = (
                f"{r['metadata'].get('retrospect_id')}"
                f"_{r['metadata'].get('chunk_type')}"
                f"_{r['metadata'].get('category', '')}"
            )
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(r)

    logger.info("Retrieved %d unique chunks", len(all_chunks))
    return {"retrieved_chunks": all_chunks}


async def build_context(state: AssistantState) -> dict[str, Any]:
    """검색된 청크를 토큰 예산 내에서 최적화된 컨텍스트로 조합한다.

    tiktoken으로 토큰 수를 계산하여 MAX_CONTEXT_TOKENS 이내로 유지.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"context": "(관련 회고 데이터 없음)", "token_count": 0}

    # 점수 기준 정렬 (높은 순)
    sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)

    context_parts: list[str] = []
    total_tokens = 0

    for i, chunk in enumerate(sorted_chunks):
        part = f"[출처 {i + 1}] {chunk['content']}"
        part_tokens = len(ENCODING.encode(part))

        if total_tokens + part_tokens > MAX_CONTEXT_TOKENS:
            break

        context_parts.append(part)
        total_tokens += part_tokens

    context = "\n\n".join(context_parts)
    logger.info(
        "Context built — %d chunks, %d tokens",
        len(context_parts),
        total_tokens,
    )
    return {"context": context, "token_count": total_tokens}


async def generate_answer(state: AssistantState) -> dict[str, Any]:
    """GPT-4o를 사용하여 컨텍스트 기반 답변을 생성한다 (스트리밍)."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
        streaming=True,
    )

    # 대화 이력 포맷
    history = state.get("conversation_history", [])
    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-6:]
    ) or "(없음)"

    context = state.get("context", "")
    system_prompt = ANSWER_SYSTEM_PROMPT.format(
        context=context,
        conversation_history=history_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]},
    ]

    # 스트리밍 답변 생성
    answer_parts: list[str] = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            answer_parts.append(chunk.content)

    full_answer = "".join(answer_parts)

    return {"answer": full_answer}


async def format_response(state: AssistantState) -> dict[str, Any]:
    """답변에서 출처 정보를 추출하고 Citation 목록을 생성한다."""
    chunks = state.get("retrieved_chunks", [])

    citations: list[Citation] = []
    seen: set[int] = set()

    for chunk in chunks:
        retro_id = chunk["metadata"].get("retrospect_id")
        if retro_id and retro_id not in seen:
            seen.add(retro_id)
            citations.append(
                Citation(
                    retrospect_id=retro_id,
                    title=chunk["metadata"].get("title", ""),
                    room_title=chunk["metadata"].get("room_title", ""),
                    date=chunk["metadata"].get("date", ""),
                    relevance=round(chunk["score"], 4),
                )
            )

    # relevance 기준 정렬
    citations.sort(key=lambda c: c["relevance"], reverse=True)

    return {"citations": citations[:10]}  # 최대 10개 출처
```

### 워크플로우 흐름도

```
                    ┌──────────────────┐
                    │  analyze_query   │
                    │  (GPT-4o-mini)   │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │  route_by_intent │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        intent=general  intent=search  intent=pattern_analysis
              │          intent=compare
              │              │
              │     ┌────────▼────────┐
              │     │    retrieve     │
              │     │ (hybrid search) │
              │     └────────┬────────┘
              │              │
              │     ┌────────▼────────┐
              │     │ build_context   │
              │     │ (token budget)  │
              │     └────────┬────────┘
              │              │
              └──────┬───────┘
                     │
            ┌────────▼────────┐
            │ generate_answer  │
            │   (GPT-4o, SSE) │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ format_response  │
            │  (citations)     │
            └────────┬────────┘
                     │
                    END
```

---

## 6. 검색 엔진 (Hybrid Search)

### 아키텍처

```
사용자 쿼리
    │
    ├──────────────────────┐
    │                      │
    ▼                      ▼
┌──────────────┐   ┌──────────────┐
│ Vector Search│   │Keyword Search│
│ (pgvector)   │   │ (ts_vector)  │
│ cosine sim.  │   │ full-text    │
└──────┬───────┘   └──────┬───────┘
       │                  │
       └────────┬─────────┘
                │
       ┌────────▼────────┐
       │  RRF Fusion     │
       │  (k=60)         │
       └────────┬────────┘
                │
        최종 결과 (top_k)
```

### vector_search.py

```python
"""pgvector 코사인 유사도 검색."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database import get_session
from app.infrastructure.openai_client import get_query_embedding


async def vector_search(
    query: str,
    filters: dict | None = None,
    top_k: int = 10,
) -> list[dict]:
    """pgvector를 사용한 코사인 유사도 검색.

    Args:
        query: 검색 쿼리 텍스트
        filters: room_ids, date_from, date_to, method 필터
        top_k: 반환할 최대 결과 수

    Returns:
        [{"content": str, "metadata": dict, "score": float}, ...]
    """
    query_embedding = await get_query_embedding(query)
    filters = filters or {}

    # 동적 WHERE 절 구성
    where_clauses = ["1=1"]
    params: dict = {
        "embedding": str(query_embedding),
        "top_k": top_k,
    }

    if filters.get("room_ids"):
        where_clauses.append(
            "(metadata->>'room_id')::int = ANY(:room_ids)"
        )
        params["room_ids"] = filters["room_ids"]

    if filters.get("date_from"):
        where_clauses.append("metadata->>'date' >= :date_from")
        params["date_from"] = filters["date_from"]

    if filters.get("date_to"):
        where_clauses.append("metadata->>'date' <= :date_to")
        params["date_to"] = filters["date_to"]

    if filters.get("method"):
        where_clauses.append("metadata->>'method' = :method")
        params["method"] = filters["method"]

    where_sql = " AND ".join(where_clauses)

    sql = text(f"""
        SELECT
            id,
            retrospect_id,
            chunk_type,
            category,
            content,
            metadata,
            1 - (embedding <=> :embedding::vector) AS score
        FROM retrospect_embeddings
        WHERE {where_sql}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :top_k
    """)

    async with get_session() as session:
        result = await session.execute(sql, params)
        rows = result.fetchall()

    return [
        {
            "content": row.content,
            "metadata": {
                **row.metadata,
                "retrospect_id": row.retrospect_id,
                "chunk_type": row.chunk_type,
                "category": row.category,
            },
            "score": float(row.score),
        }
        for row in rows
    ]
```

### keyword_search.py

```python
"""PostgreSQL ts_vector 기반 풀텍스트 검색."""

from __future__ import annotations

from sqlalchemy import text

from app.infrastructure.database import get_session


async def keyword_search(
    query: str,
    filters: dict | None = None,
    top_k: int = 10,
) -> list[dict]:
    """PostgreSQL ts_vector를 사용한 키워드 검색.

    한국어 텍스트는 'simple' configuration을 사용하여 토큰화.
    공백/기호 기준 분리로 기본적인 한국어 검색을 지원한다.

    Args:
        query: 검색 쿼리 텍스트
        filters: room_ids, date_from, date_to, method 필터
        top_k: 반환할 최대 결과 수

    Returns:
        [{"content": str, "metadata": dict, "score": float}, ...]
    """
    filters = filters or {}

    # 쿼리를 tsquery 형식으로 변환 (공백 → &)
    ts_terms = " & ".join(query.split())

    where_clauses = [
        "content_tsv @@ to_tsquery('simple', :ts_query)"
    ]
    params: dict = {"ts_query": ts_terms, "top_k": top_k}

    if filters.get("room_ids"):
        where_clauses.append(
            "(metadata->>'room_id')::int = ANY(:room_ids)"
        )
        params["room_ids"] = filters["room_ids"]

    if filters.get("date_from"):
        where_clauses.append("metadata->>'date' >= :date_from")
        params["date_from"] = filters["date_from"]

    if filters.get("date_to"):
        where_clauses.append("metadata->>'date' <= :date_to")
        params["date_to"] = filters["date_to"]

    if filters.get("method"):
        where_clauses.append("metadata->>'method' = :method")
        params["method"] = filters["method"]

    where_sql = " AND ".join(where_clauses)

    sql = text(f"""
        SELECT
            id,
            retrospect_id,
            chunk_type,
            category,
            content,
            metadata,
            ts_rank(content_tsv, to_tsquery('simple', :ts_query)) AS score
        FROM retrospect_embeddings
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT :top_k
    """)

    async with get_session() as session:
        result = await session.execute(sql, params)
        rows = result.fetchall()

    return [
        {
            "content": row.content,
            "metadata": {
                **row.metadata,
                "retrospect_id": row.retrospect_id,
                "chunk_type": row.chunk_type,
                "category": row.category,
            },
            "score": float(row.score),
        }
        for row in rows
    ]
```

### hybrid.py (RRF Fusion)

```python
"""RRF (Reciprocal Rank Fusion) 기반 하이브리드 검색.

벡터 검색(의미적 유사도)과 키워드 검색(어휘적 매칭)의 결과를
RRF 알고리즘으로 결합하여 양쪽의 장점을 취한다.

- 벡터만: "Redis"를 "캐시"로도 찾지만, 정확한 키워드 매칭이 약함
- 키워드만: 유사어/동의어 검색 불가
- RRF 결합: 양쪽 장점을 상호 보완
"""

from __future__ import annotations

from app.retrieval.keyword_search import keyword_search
from app.retrieval.vector_search import vector_search

# RRF 상수 — Cormack et al. (2009) 논문의 권장값
RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 알고리즘.

    여러 검색 결과 리스트의 순위를 결합한다.
    RRF_score(d) = Σ  1 / (k + rank_i(d))

    Args:
        result_lists: 각 검색 엔진의 결과 리스트
        k: RRF 상수 (기본값 60). 높을수록 순위 차이 영향 감소.

    Returns:
        RRF 점수 기준 정렬된 결합 결과
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            # 문서 고유 키: retrospect_id + chunk_type + category
            doc_key = (
                f"{doc['metadata'].get('retrospect_id')}"
                f"_{doc['metadata'].get('chunk_type')}"
                f"_{doc['metadata'].get('category', '')}"
            )
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_key] = doc

    # RRF 점수 기준 정렬
    sorted_keys = sorted(scores, key=scores.get, reverse=True)

    return [
        {**doc_map[key], "score": scores[key]}
        for key in sorted_keys
    ]


async def hybrid_search(
    query: str,
    filters: dict | None = None,
    top_k: int = 10,
) -> list[dict]:
    """벡터 + 키워드 하이브리드 검색을 수행한다.

    1. pgvector 코사인 유사도 검색
    2. PostgreSQL ts_vector 키워드 검색
    3. RRF로 결합
    4. top_k개 반환

    Args:
        query: 검색 쿼리 텍스트
        filters: 검색 필터
        top_k: 최종 반환 결과 수

    Returns:
        RRF 점수 기준 정렬된 결과 리스트
    """
    # 병렬 실행 — 두 검색을 동시에 수행
    import asyncio

    vector_task = asyncio.create_task(
        vector_search(query, filters, top_k=top_k * 2)
    )
    keyword_task = asyncio.create_task(
        keyword_search(query, filters, top_k=top_k * 2)
    )

    vector_results, keyword_results = await asyncio.gather(
        vector_task, keyword_task
    )

    # RRF 결합
    fused = reciprocal_rank_fusion([vector_results, keyword_results])

    return fused[:top_k]
```

---

## 7. 데이터 모델

### DDL: PostgreSQL (pgvector DB)

```sql
-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- ========================================================================
-- retrospect_embeddings: 회고 청크 + 벡터 임베딩
-- ========================================================================
CREATE TABLE retrospect_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    retrospect_id   BIGINT NOT NULL,
    chunk_type      VARCHAR(20) NOT NULL,         -- summary | response | analysis
    category        VARCHAR(200) NOT NULL DEFAULT '',  -- question text or empty
    content         TEXT NOT NULL,                 -- 원본 텍스트
    embedding       vector(1536) NOT NULL,        -- text-embedding-3-small
    metadata        JSONB NOT NULL DEFAULT '{}',  -- room_id, method, date, etc.
    content_tsv     TSVECTOR GENERATED ALWAYS AS  -- 풀텍스트 검색용
                        (to_tsvector('simple', content)) STORED,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE (retrospect_id, chunk_type, category)
);

-- HNSW 인덱스 (코사인 유사도 검색 최적화)
-- m=16: 노드당 최대 연결 수 (높을수록 정확하지만 메모리 소비 증가)
-- ef_construction=64: 인덱스 빌드 시 탐색 범위 (높을수록 정확하지만 빌드 느림)
CREATE INDEX idx_embedding_hnsw
    ON retrospect_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ts_vector GIN 인덱스 (풀텍스트 검색 최적화)
CREATE INDEX idx_content_tsv
    ON retrospect_embeddings
    USING gin (content_tsv);

-- metadata JSONB GIN 인덱스 (필터링 최적화)
CREATE INDEX idx_metadata_gin
    ON retrospect_embeddings
    USING gin (metadata jsonb_path_ops);

-- ========================================================================
-- sync_status: 동기화 상태 추적
-- ========================================================================
CREATE TABLE sync_status (
    source          VARCHAR(50) PRIMARY KEY,        -- 'retrospects'
    last_synced_at  TIMESTAMP WITH TIME ZONE NOT NULL,
    total_synced    BIGINT NOT NULL DEFAULT 0,
    last_error      TEXT,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ========================================================================
-- query_log: 질의 로그 (분석/모니터링용)
-- ========================================================================
CREATE TABLE query_log (
    id              BIGSERIAL PRIMARY KEY,
    session_id      VARCHAR(100) NOT NULL,
    user_id         VARCHAR(100),
    query           TEXT NOT NULL,
    intent          VARCHAR(30),
    chunks_retrieved INT,
    tokens_used     INT,
    latency_ms      INT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_query_log_session
    ON query_log (session_id);

CREATE INDEX idx_query_log_created
    ON query_log (created_at DESC);
```

### Pydantic 도메인 모델

```python
"""도메인 모델 — 비즈니스 엔티티의 Python 표현."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RetrospectEmbedding(BaseModel):
    """pgvector에 저장된 임베딩 레코드."""

    id: int
    retrospect_id: int
    chunk_type: str
    category: str
    content: str
    embedding: list[float] = Field(exclude=True)
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SyncStatus(BaseModel):
    """동기화 상태."""

    source: str
    last_synced_at: datetime
    total_synced: int
    last_error: str | None = None


class ConversationTurn(BaseModel):
    """대화 한 턴 (질문 또는 답변)."""

    role: str       # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### API 스키마 (Request/Response)

```python
"""API Request/Response 스키마."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# --- Chat ---

class ChatRequest(BaseModel):
    """채팅 요청."""

    session_id: str = Field(..., description="세션 ID (클라이언트 생성)")
    message: str = Field(..., min_length=1, max_length=2000, description="사용자 질문")
    filters: ChatFilters | None = Field(None, description="검색 필터")


class ChatFilters(BaseModel):
    """채팅 검색 필터."""

    room_ids: list[int] | None = Field(None, description="회고방 ID 목록")
    date_from: str | None = Field(None, description="시작 날짜 (YYYY-MM-DD)")
    date_to: str | None = Field(None, description="종료 날짜 (YYYY-MM-DD)")
    method: str | None = Field(None, description="회고 방법 (KPT, FOUR_L, etc.)")


class ChatSSEEvent(BaseModel):
    """SSE 이벤트 페이로드."""

    type: str       # "text" | "citations" | "done" | "error"
    content: str | None = None
    data: list[dict] | None = None


# --- Search ---

class SearchRequest(BaseModel):
    """벡터 검색 요청 (RAG 없이 검색만)."""

    query: str = Field(..., min_length=1, max_length=500)
    filters: ChatFilters | None = None
    top_k: int = Field(10, ge=1, le=50)


class SearchResult(BaseModel):
    """검색 결과 항목."""

    retrospect_id: int
    chunk_type: str
    content: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    """검색 응답."""

    results: list[SearchResult]
    total: int
    query: str


# --- Session ---

class SessionHistory(BaseModel):
    """세션 대화 이력."""

    session_id: str
    messages: list[SessionMessage]
    created_at: datetime | None = None


class SessionMessage(BaseModel):
    """세션 내 메시지."""

    role: str
    content: str
    timestamp: str


# --- Ingestion ---

class SyncTriggerResponse(BaseModel):
    """수동 동기화 트리거 응답."""

    status: str     # "started" | "already_running"
    message: str


class SyncStatusResponse(BaseModel):
    """동기화 상태 응답."""

    source: str
    last_synced_at: datetime | None
    total_synced: int
    is_running: bool
    last_error: str | None = None


# --- Health ---

class HealthResponse(BaseModel):
    """헬스체크 응답."""

    status: str             # "healthy" | "degraded" | "unhealthy"
    version: str
    checks: dict[str, str]  # {"postgres": "up", "redis": "up", "openai": "up"}
```

---

## 8. API 상세 설계

### 8.1 POST /api/v1/assistant/chat

**채팅 (SSE 스트리밍)**

질문을 받아 RAG 파이프라인을 실행하고, 답변을 Server-Sent Events로 스트리밍한다.

**Request**:
```http
POST /api/v1/assistant/chat
Content-Type: application/json
Authorization: Bearer <jwt_token>

{
    "session_id": "sess-abc123",
    "message": "우리 팀의 반복되는 문제점이 뭐야?",
    "filters": {
        "room_ids": [1, 2],
        "date_from": "2025-07-01"
    }
}
```

**Response** (SSE stream):
```
data: {"type": "text", "content": "최근 6개월간"}

data: {"type": "text", "content": " 회고를 분석한 결과,"}

data: {"type": "text", "content": " 가장 반복되는 문제점은 다음과 같습니다..."}

data: {"type": "citations", "data": [{"retrospect_id": 42, "title": "Sprint 3 회고", "room_title": "백엔드팀", "date": "2026-01-15", "relevance": 0.89}, {"retrospect_id": 38, "title": "Sprint 2 회고", "room_title": "백엔드팀", "date": "2025-12-15", "relevance": 0.82}]}

data: {"type": "done"}
```

**Status Codes**:

| 코드 | 설명 |
|------|------|
| 200 | SSE 스트림 시작 |
| 401 | JWT 인증 실패 |
| 422 | 요청 형식 오류 |
| 429 | Rate limit 초과 |
| 500 | 서버 내부 오류 |

**curl 예시**:
```bash
curl -N -X POST http://localhost:8085/api/v1/assistant/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9..." \
  -d '{
    "session_id": "sess-test-001",
    "message": "최근 3개월 반복 문제점은?",
    "filters": {"date_from": "2025-11-01"}
  }'
```

### 8.2 GET /api/v1/assistant/sessions/{session_id}/history

**세션 대화 이력 조회**

**Request**:
```http
GET /api/v1/assistant/sessions/sess-abc123/history
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
    "session_id": "sess-abc123",
    "messages": [
        {
            "role": "user",
            "content": "반복되는 문제점이 뭐야?",
            "timestamp": "2026-02-16T10:30:00Z"
        },
        {
            "role": "assistant",
            "content": "최근 6개월간 회고를 분석한 결과...",
            "timestamp": "2026-02-16T10:30:05Z"
        }
    ],
    "created_at": "2026-02-16T10:30:00Z"
}
```

**Status Codes**: 200, 401, 404 (session not found)

### 8.3 DELETE /api/v1/assistant/sessions/{session_id}

**세션 삭제**

**Request**:
```http
DELETE /api/v1/assistant/sessions/sess-abc123
Authorization: Bearer <jwt_token>
```

**Response**: `204 No Content`

**Status Codes**: 204, 401, 404

### 8.4 POST /api/v1/assistant/search

**벡터 검색 (RAG 없이)**

LLM 생성 없이 순수 하이브리드 검색만 수행. 관리자 도구나 디버깅에 유용.

**Request**:
```http
POST /api/v1/assistant/search
Content-Type: application/json
Authorization: Bearer <jwt_token>

{
    "query": "코드 리뷰 지연",
    "filters": {"room_ids": [1]},
    "top_k": 5
}
```

**Response**:
```json
{
    "results": [
        {
            "retrospect_id": 42,
            "chunk_type": "response",
            "content": "[개선이 필요한 문제점은?]\n- 코드 리뷰가 2-3일 지연됨\n- PR 크기가 너무 큼",
            "score": 0.0312,
            "metadata": {
                "room_id": 1,
                "room_title": "백엔드팀",
                "method": "KPT",
                "date": "2026-01-15"
            }
        }
    ],
    "total": 1,
    "query": "코드 리뷰 지연"
}
```

**Status Codes**: 200, 401, 422

**curl 예시**:
```bash
curl -X POST http://localhost:8085/api/v1/assistant/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9..." \
  -d '{"query": "코드 리뷰 지연", "top_k": 5}'
```

### 8.5 POST /api/v1/ingestion/sync

**수동 동기화 트리거**

스케줄 외에 수동으로 즉시 동기화를 실행.

**Request**:
```http
POST /api/v1/ingestion/sync
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
    "status": "started",
    "message": "동기화가 시작되었습니다."
}
```

**Status Codes**: 200, 401, 409 (already running)

### 8.6 GET /api/v1/ingestion/status

**동기화 상태 조회**

**Request**:
```http
GET /api/v1/ingestion/status
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
    "source": "retrospects",
    "last_synced_at": "2026-02-16T10:25:00Z",
    "total_synced": 156,
    "is_running": false,
    "last_error": null
}
```

**Status Codes**: 200, 401

### 8.7 GET /health

**헬스체크** (인증 불필요)

**Response**:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "checks": {
        "postgres": "up",
        "redis": "up",
        "openai": "up"
    }
}
```

**Status Codes**: 200 (healthy), 503 (unhealthy)

---

## 9. 인증/인가

### JWT 검증

moalog-server와 동일한 JWT 시크릿(`JWT_SECRET`)을 사용하여 토큰을 검증한다. moalog-server의 JWT 구조:

```python
# moalog-server JWT Claims 구조 (Rust jsonwebtoken)
{
    "sub": "user_123",         # Subject (User ID)
    "iat": 1708000000,         # Issued At
    "exp": 1708003600,         # Expiration
    "jti": null,               # JWT ID (access token은 null)
    "token_type": "access",    # "access" | "refresh" | "signup"
    "email": null,
    "provider": null
}
```

### 인증 미들웨어

```python
"""JWT 인증 미들웨어 — moalog-server와 동일한 인증 로직."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from jose import JWTError, jwt

from app.config import get_settings


async def get_current_user(request: Request) -> str:
    """요청에서 JWT를 추출하고 검증하여 user_id를 반환한다.

    인증 순서 (moalog-server와 동일):
    1. Authorization: Bearer <token> 헤더
    2. access_token 쿠키 (폴백)
    """
    settings = get_settings()
    token = _extract_token(request)

    if not token:
        raise HTTPException(status_code=401, detail="인증 토큰이 없습니다.")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
        )
    except JWTError:
        raise HTTPException(
            status_code=401, detail="유효하지 않은 토큰입니다."
        )

    # token_type이 "access"인지 검증
    if payload.get("token_type") != "access":
        raise HTTPException(
            status_code=401, detail="유효하지 않은 액세스 토큰입니다."
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401, detail="토큰에 사용자 정보가 없습니다."
        )

    return user_id


def _extract_token(request: Request) -> str | None:
    """요청에서 JWT 토큰을 추출한다.

    1. Authorization: Bearer <token>
    2. Cookie: access_token=<token>
    """
    # 1) Authorization 헤더
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    # 2) 쿠키 (moalog-server 쿠키 이름: access_token)
    return request.cookies.get("access_token")
```

### 인증 적용 규칙

| 엔드포인트 | 인증 필요 | 비고 |
|-----------|----------|------|
| `GET /health` | No | 모니터링용 |
| `GET /metrics` | No | Prometheus 스크래핑 |
| `POST /api/v1/assistant/chat` | Yes | user_id로 대화 소유권 검증 |
| `GET /api/v1/assistant/sessions/*/history` | Yes | 자기 세션만 조회 가능 |
| `DELETE /api/v1/assistant/sessions/*` | Yes | 자기 세션만 삭제 가능 |
| `POST /api/v1/assistant/search` | Yes | 검색 로그에 user_id 기록 |
| `POST /api/v1/ingestion/sync` | Yes | 추후 admin 권한 체크 고려 |
| `GET /api/v1/ingestion/status` | Yes | 동기화 상태 확인 |

---

## 10. 에러 처리

### 에러 코드 테이블

| 에러 코드 | HTTP Status | 설명 | 대응 |
|----------|-------------|------|------|
| `AUTH_001` | 401 | JWT 토큰 없음 | 로그인 필요 |
| `AUTH_002` | 401 | JWT 만료/유효하지 않음 | 토큰 갱신 |
| `AUTH_003` | 403 | 다른 사용자의 세션 접근 | 권한 없음 |
| `VALIDATION_001` | 422 | 요청 형식 오류 | 요청 수정 |
| `RATE_LIMIT_001` | 429 | API 호출 제한 초과 | 잠시 후 재시도 |
| `OPENAI_001` | 502 | OpenAI API 호출 실패 | 자동 재시도 |
| `OPENAI_002` | 502 | OpenAI rate limit | 지수 백오프 후 재시도 |
| `OPENAI_003` | 502 | OpenAI 토큰 한도 초과 | 쿼리/컨텍스트 축소 |
| `DB_001` | 503 | PostgreSQL 연결 실패 | 서비스 상태 확인 |
| `DB_002` | 503 | Redis 연결 실패 | 대화 메모리 없이 동작 (degraded) |
| `SEARCH_001` | 500 | 검색 쿼리 실행 실패 | 서버 로그 확인 |
| `SYNC_001` | 503 | MySQL 연결 실패 | moalog-server DB 확인 |
| `INTERNAL_001` | 500 | 알 수 없는 내부 오류 | 서버 로그 확인 |

### OpenAI API 에러 핸들링

```python
"""OpenAI API 에러 핸들링 — 재시도/폴백 로직."""

from __future__ import annotations

import asyncio
import logging
from functools import wraps

from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0  # 초


def with_openai_retry(func):
    """OpenAI API 호출에 지수 백오프 재시도를 적용하는 데코레이터."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                last_error = e
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI rate limit (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
            except APITimeoutError as e:
                last_error = e
                logger.warning(
                    "OpenAI timeout (attempt %d/%d)",
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(BASE_DELAY)
            except APIConnectionError as e:
                last_error = e
                logger.error("OpenAI connection error: %s", e)
                break  # 연결 오류는 재시도 불필요

        raise last_error

    return wrapper
```

### Graceful Degradation

| 장애 상황 | 동작 |
|----------|------|
| OpenAI Embedding API 다운 | 동기화 잡 건너뜀. 기존 임베딩으로 검색 계속 가능 |
| OpenAI GPT-4o API 다운 | SSE로 에러 이벤트 전송: `{"type": "error", "content": "답변 생성 서비스 일시 장애"}` |
| Redis 다운 | 대화 메모리 없이 단건 Q&A로 동작 (degraded mode) |
| MySQL(moalog) 다운 | 동기화 중단. 기존 벡터 데이터로 서비스 유지 |
| pgvector DB 다운 | 서비스 불가 (503 반환) |

---

## 11. 설정 관리

### config.py (Pydantic Settings)

```python
"""애플리케이션 설정 — pydantic-settings 기반 환경 변수 매핑."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """모든 설정은 환경 변수 또는 .env 파일에서 로드된다."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "retro-rag-assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- JWT (moalog-server와 동일) ---
    jwt_secret: str = "local_dev_secret"

    # --- PostgreSQL (pgvector) ---
    database_url: str = "postgresql+asyncpg://fluxpay:fluxpay@localhost:5433/fluxpay"

    # --- MySQL (moalog-server, read-only) ---
    moalog_db_host: str = "localhost"
    moalog_db_port: int = 3307
    moalog_db_user: str = "root"
    moalog_db_password: str = "moalog_local"
    moalog_db_name: str = "retrospect"

    # --- Redis ---
    redis_url: str = "redis://:moalog_redis_local@localhost:6382/1"

    # --- OpenAI ---
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"

    # --- Sync ---
    sync_interval_seconds: int = 300    # 5분
    sync_enabled: bool = True

    # --- Search ---
    vector_search_top_k: int = 10
    rrf_k: int = 60
    max_context_tokens: int = 6000

    # --- Conversation Memory ---
    max_conversation_turns: int = 10
    session_ttl_seconds: int = 3600     # 1시간

    @property
    def moalog_db_url(self) -> str:
        """MySQL 연결 URL을 구성한다."""
        return (
            f"mysql://{self.moalog_db_user}:{self.moalog_db_password}"
            f"@{self.moalog_db_host}:{self.moalog_db_port}"
            f"/{self.moalog_db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    """설정 싱글턴을 반환한다."""
    return Settings()
```

### .env.example

```bash
# ─── App ──────────────────────────────────────────────
APP_NAME=retro-rag-assistant
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# ─── Server ───────────────────────────────────────────
HOST=0.0.0.0
PORT=8000

# ─── JWT (moalog-server와 동일한 시크릿) ─────────────
JWT_SECRET=local_dev_secret

# ─── PostgreSQL (pgvector) ────────────────────────────
DATABASE_URL=postgresql+asyncpg://fluxpay:fluxpay@fluxpay-postgres:5432/fluxpay

# ─── MySQL (moalog-server DB, read-only) ──────────────
MOALOG_DB_HOST=mysql
MOALOG_DB_PORT=3306
MOALOG_DB_USER=root
MOALOG_DB_PASSWORD=moalog_local
MOALOG_DB_NAME=retrospect

# ─── Redis ────────────────────────────────────────────
REDIS_URL=redis://:moalog_redis_local@redis:6379/1

# ─── OpenAI ───────────────────────────────────────────
OPENAI_API_KEY=sk-your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o

# ─── Sync ─────────────────────────────────────────────
SYNC_INTERVAL_SECONDS=300
SYNC_ENABLED=true

# ─── Search ───────────────────────────────────────────
VECTOR_SEARCH_TOP_K=10
RRF_K=60
MAX_CONTEXT_TOKENS=6000

# ─── Conversation Memory ─────────────────────────────
MAX_CONVERSATION_TURNS=10
SESSION_TTL_SECONDS=3600
```

### Docker 환경에서의 설정 차이

| 변수 | 로컬 개발 | Docker Compose |
|------|----------|----------------|
| `DATABASE_URL` | `...@localhost:5433/fluxpay` | `...@fluxpay-postgres:5432/fluxpay` |
| `MOALOG_DB_HOST` | `localhost` | `mysql` |
| `MOALOG_DB_PORT` | `3307` | `3306` |
| `REDIS_URL` | `redis://:...@localhost:6382/1` | `redis://:...@redis:6379/1` |

---

## 12. 모니터링

### Prometheus 메트릭

```python
"""Prometheus 메트릭 정의 및 미들웨어."""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# --- 메트릭 정의 ---

APP_INFO = Info("rag_assistant", "AI Retrospect Assistant metadata")

# 쿼리 카운터 (intent별)
RAG_QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total RAG queries processed",
    ["intent", "status"],
)

# 쿼리 레이턴시 (히스토그램)
RAG_QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query end-to-end latency",
    ["intent"],
    buckets=[0.5, 1, 2, 5, 10, 30],
)

# 임베딩 동기화 카운터
EMBEDDING_SYNC_TOTAL = Counter(
    "embedding_sync_total",
    "Total retrospects synced to pgvector",
    ["status"],  # success | error
)

# OpenAI 토큰 사용량
OPENAI_TOKEN_USAGE = Counter(
    "openai_token_usage_total",
    "Total OpenAI tokens consumed",
    ["model", "type"],  # type: embedding | completion | prompt
)

# 활성 SSE 커넥션 수
ACTIVE_SSE_CONNECTIONS = Gauge(
    "active_sse_connections",
    "Number of active SSE streaming connections",
)

# 검색 결과 수
RETRIEVAL_RESULTS = Histogram(
    "retrieval_results_count",
    "Number of chunks retrieved per query",
    buckets=[0, 1, 5, 10, 15, 20, 30],
)


# --- /metrics 엔드포인트 ---

async def metrics_endpoint(request: Request) -> Response:
    """Prometheus 스크래핑 엔드포인트."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
```

### 헬스체크 로직

```python
"""헬스체크 — PostgreSQL, Redis, OpenAI 연결 상태를 확인한다."""

from __future__ import annotations

from app.config import get_settings
from app.domain.schemas import HealthResponse


async def check_health() -> HealthResponse:
    """모든 외부 의존성의 연결 상태를 확인한다."""
    settings = get_settings()
    checks: dict[str, str] = {}
    overall = "healthy"

    # PostgreSQL
    try:
        from app.infrastructure.database import get_session

        async with get_session() as session:
            await session.execute("SELECT 1")
        checks["postgres"] = "up"
    except Exception:
        checks["postgres"] = "down"
        overall = "unhealthy"

    # Redis
    try:
        from app.infrastructure.redis_client import get_redis

        redis = await get_redis()
        await redis.ping()
        checks["redis"] = "up"
    except Exception:
        checks["redis"] = "down"
        if overall == "healthy":
            overall = "degraded"  # Redis 없이도 동작 가능

    # OpenAI (가벼운 모델 목록 호출)
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        await client.models.retrieve("text-embedding-3-small")
        checks["openai"] = "up"
    except Exception:
        checks["openai"] = "down"
        if overall == "healthy":
            overall = "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        checks=checks,
    )
```

### Prometheus 스크래핑 설정 (prometheus.yml 추가분)

```yaml
# monitoring/prometheus.yml에 추가
- job_name: 'ai-assistant'
  scrape_interval: 15s
  static_configs:
    - targets: ['ai-assistant:8000']
  metrics_path: /metrics
```

---

## 13. Docker

### Dockerfile

```dockerfile
# ── Build stage ─────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# 시스템 의존성 (빌드 시에만 필요)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

# ── Runtime stage ───────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# 런타임 의존성
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# non-root 사용자
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# Python 패키지 복사
COPY --from=builder /install /usr/local

# 애플리케이션 코드 복사
COPY app/ ./app/

# 소유권 변경
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose 서비스 정의

moalog-server의 `docker-compose.yaml`에 추가할 서비스 정의:

```yaml
  # ─── AI Retrospect Assistant ──────────────────────────
  ai-assistant:
    build:
      context: ../retro-rag-assistant
      dockerfile: Dockerfile
    container_name: moalog-ai-assistant
    depends_on:
      fluxpay-postgres:
        condition: service_healthy
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+asyncpg://fluxpay:${FLUXPAY_DB_PASSWORD:-fluxpay}@fluxpay-postgres:5432/fluxpay
      MOALOG_DB_HOST: mysql
      MOALOG_DB_PORT: 3306
      MOALOG_DB_USER: root
      MOALOG_DB_PASSWORD: ${MYSQL_ROOT_PASSWORD:-moalog_local}
      MOALOG_DB_NAME: ${MYSQL_DATABASE:-retrospect}
      REDIS_URL: redis://:${REDIS_PASSWORD:-moalog_redis_local}@redis:6379/1
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      JWT_SECRET: ${JWT_SECRET:-local_dev_secret}
      SYNC_ENABLED: "true"
      LOG_LEVEL: ${AI_ASSISTANT_LOG_LEVEL:-INFO}
    ports:
      - "${AI_ASSISTANT_PORT:-8085}:8000"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"]
      interval: 30s
      timeout: 10s
      start_period: 30s
      retries: 3
    restart: unless-stopped
```

### PostgreSQL 이미지 변경

기존 `fluxpay-postgres` 서비스의 이미지를 pgvector 지원 이미지로 변경:

```yaml
  # 변경 전
  fluxpay-postgres:
    image: postgres:16-alpine

  # 변경 후
  fluxpay-postgres:
    image: pgvector/pgvector:pg16
```

pgvector 확장은 이미지에 포함되어 있으며, init-script에서 `CREATE EXTENSION IF NOT EXISTS vector;`를 실행하여 활성화한다.

### 포트 배치

| 서비스 | 내부 포트 | 외부 포트 (기본값) |
|--------|----------|------------------|
| ai-assistant | 8000 | 8085 |
| moalog-server | 8080 | 8090 |
| rate-limiter | 8080 | 8082 |
| fluxpay-engine | 8080 | 8081 |

---

## 14. 테스트 전략

### Unit 테스트

외부 의존성 없이 순수 로직 검증:

| 테스트 대상 | 파일 | 핵심 케이스 |
|-----------|------|-----------|
| Chunker | `test_chunker.py` | KPT 회고 → 3+ 청크, insight 없으면 analysis 생략, 빈 응답 처리 |
| RRF Fusion | `test_hybrid_search.py` | 동일 문서 양쪽에 있을 때 합산, 한쪽만 있을 때 처리, 빈 리스트 |
| Query Analyzer | `test_query_analyzer.py` | intent 분류 정확성 (mock LLM), 필터 병합 로직 |
| Token Budget | `test_context_builder.py` | MAX_CONTEXT_TOKENS 초과 시 절단, 빈 청크 처리 |

**예시: test_chunker.py**

```python
"""Chunker 단위 테스트."""

from datetime import datetime

import pytest

from app.ingestion.chunker import (
    RetrospectData,
    ResponseData,
    chunk_retrospect,
)


@pytest.fixture
def sample_retrospect() -> RetrospectData:
    """KPT 방식의 테스트 회고 데이터."""
    return RetrospectData(
        retrospect_id=1,
        title="Sprint 3 회고",
        insight="코드 리뷰 프로세스 개선이 시급합니다.",
        method="KPT",
        start_time=datetime(2026, 1, 15),
        room_id=1,
        room_title="백엔드팀",
        participant_count=5,
        responses=[
            ResponseData(
                response_id=1,
                question="계속 유지하고 싶은 좋은 점은 무엇인가요?",
                content="데일리 스크럼이 효과적이었다",
            ),
            ResponseData(
                response_id=2,
                question="개선이 필요한 문제점은 무엇인가요?",
                content="코드 리뷰가 2-3일 지연됨",
            ),
            ResponseData(
                response_id=3,
                question="개선이 필요한 문제점은 무엇인가요?",
                content="PR 크기가 너무 큼",
            ),
        ],
    )


def test_chunk_produces_summary(sample_retrospect):
    """summary 청크가 올바르게 생성되는지 검증한다."""
    chunks = chunk_retrospect(sample_retrospect)
    summaries = [c for c in chunks if c.chunk_type == "summary"]

    assert len(summaries) == 1
    assert "백엔드팀" in summaries[0].content
    assert "Sprint 3 회고" in summaries[0].content
    assert "KPT" in summaries[0].content
    assert "5명" in summaries[0].content


def test_chunk_groups_responses_by_question(sample_retrospect):
    """response 청크가 질문별로 그룹핑되는지 검증한다."""
    chunks = chunk_retrospect(sample_retrospect)
    responses = [c for c in chunks if c.chunk_type == "response"]

    # 2개 질문 → 2개 response 청크
    assert len(responses) == 2

    # "문제점" 질문의 청크에는 2개 답변이 포함
    problem_chunk = [
        c for c in responses if "문제점" in c.category
    ][0]
    assert "코드 리뷰" in problem_chunk.content
    assert "PR 크기" in problem_chunk.content


def test_chunk_includes_analysis_when_insight_exists(sample_retrospect):
    """insight가 있을 때 analysis 청크가 생성되는지 검증한다."""
    chunks = chunk_retrospect(sample_retrospect)
    analyses = [c for c in chunks if c.chunk_type == "analysis"]

    assert len(analyses) == 1
    assert "코드 리뷰 프로세스" in analyses[0].content


def test_chunk_omits_analysis_when_no_insight(sample_retrospect):
    """insight가 없을 때 analysis 청크가 생략되는지 검증한다."""
    sample_retrospect.insight = None
    chunks = chunk_retrospect(sample_retrospect)
    analyses = [c for c in chunks if c.chunk_type == "analysis"]

    assert len(analyses) == 0


def test_chunk_metadata_contains_required_fields(sample_retrospect):
    """모든 청크의 metadata에 필수 필드가 포함되는지 검증한다."""
    chunks = chunk_retrospect(sample_retrospect)

    for chunk in chunks:
        assert "room_id" in chunk.metadata
        assert "method" in chunk.metadata
        assert "date" in chunk.metadata
        assert chunk.metadata["room_id"] == 1
        assert chunk.metadata["method"] == "KPT"
```

**예시: test_hybrid_search.py (RRF)**

```python
"""RRF Fusion 단위 테스트."""

import pytest

from app.retrieval.hybrid import reciprocal_rank_fusion


def _make_doc(retro_id: int, chunk_type: str = "response") -> dict:
    """테스트용 문서를 생성한다."""
    return {
        "content": f"test content {retro_id}",
        "metadata": {
            "retrospect_id": retro_id,
            "chunk_type": chunk_type,
            "category": "",
        },
        "score": 0.5,
    }


def test_rrf_merges_two_lists():
    """두 리스트의 결과가 RRF로 올바르게 합산되는지 검증한다."""
    list_a = [_make_doc(1), _make_doc(2), _make_doc(3)]
    list_b = [_make_doc(2), _make_doc(3), _make_doc(4)]

    result = reciprocal_rank_fusion([list_a, list_b], k=60)

    # doc 2, 3은 양쪽에 존재 → 더 높은 RRF 점수
    result_ids = [
        r["metadata"]["retrospect_id"] for r in result
    ]
    # doc 2가 list_a의 rank 2, list_b의 rank 1 → 최상위
    assert result_ids[0] in [2, 3]
    assert len(result) == 4  # 총 4개 고유 문서


def test_rrf_empty_lists():
    """빈 리스트 입력 시 빈 결과를 반환하는지 검증한다."""
    result = reciprocal_rank_fusion([[], []])
    assert result == []


def test_rrf_single_list():
    """리스트가 하나일 때도 정상 동작하는지 검증한다."""
    docs = [_make_doc(1), _make_doc(2)]
    result = reciprocal_rank_fusion([docs])

    assert len(result) == 2
    assert result[0]["score"] > result[1]["score"]
```

### Integration 테스트

실제 DB/Redis를 사용하여 검증:

| 테스트 대상 | 필요 인프라 | 핵심 케이스 |
|-----------|-----------|-----------|
| pgvector 검색 | PostgreSQL + pgvector | 테스트 임베딩 삽입 → 코사인 검색 → 올바른 결과 |
| Redis 메모리 | Redis | 대화 저장 → 조회 → TTL 만료 확인 |
| Ingestion | MySQL + PostgreSQL | 테스트 회고 삽입 → sync → pgvector에 청크 존재 확인 |

### E2E 테스트

전체 파이프라인 통합 검증:

```
1. MySQL에 테스트 회고 데이터 삽입
2. POST /api/v1/ingestion/sync → 동기화 실행
3. GET /api/v1/ingestion/status → total_synced 확인
4. POST /api/v1/assistant/chat → SSE 이벤트 수신
5. SSE "text" 이벤트에 관련 내용 포함 확인
6. SSE "citations" 이벤트에 출처 포함 확인
7. SSE "done" 이벤트 수신 확인
8. GET /sessions/{id}/history → 대화 이력에 기록 확인
```

---

## 15. 구현 페이즈

### Phase 1: 기반 설정 + Ingestion (3일)

| 작업 | 상세 |
|------|------|
| 프로젝트 스캐폴딩 | pyproject.toml, Dockerfile, 디렉토리 구조 |
| config.py | pydantic-settings, .env.example |
| infrastructure/ | database.py (asyncpg), mysql_client.py (aiomysql), redis_client.py |
| DDL 적용 | retrospect_embeddings, sync_status, query_log 테이블 |
| Ingestion 구현 | chunker.py, embedder.py, sync_job.py |
| Ingestion API | POST /ingestion/sync, GET /ingestion/status |
| 테스트 | chunker unit test, sync integration test |

**완료 기준**: `docker compose up` 후 MySQL의 회고 데이터가 pgvector에 임베딩으로 저장됨

### Phase 2: 검색 엔진 + LangGraph (3일)

| 작업 | 상세 |
|------|------|
| vector_search.py | pgvector 코사인 유사도 검색, 필터링 |
| keyword_search.py | ts_vector 풀텍스트 검색 |
| hybrid.py | RRF fusion |
| LangGraph workflow | state.py, graph.py, nodes.py |
| Query Analyzer | GPT-4o-mini intent 분류 |
| Context Builder | tiktoken 토큰 예산 관리 |
| 테스트 | RRF unit test, 검색 integration test |

**완료 기준**: 쿼리 입력 → 하이브리드 검색 → 컨텍스트 빌드 → GPT-4o 답변 생성이 동작

### Phase 3: Chat API + 스트리밍 + 메모리 (2일)

| 작업 | 상세 |
|------|------|
| redis_memory.py | Redis List 기반 대화 이력 관리 |
| Chat SSE endpoint | POST /api/v1/assistant/chat (StreamingResponse) |
| Session endpoints | GET history, DELETE session |
| Search endpoint | POST /api/v1/assistant/search |
| JWT 인증 | python-jose 미들웨어 |
| 테스트 | Redis memory test, E2E chat flow |

**완료 기준**: SSE 스트리밍 답변 + 멀티턴 대화 + JWT 인증 동작

### Phase 4: 모니터링 + Docker 통합 (2일)

| 작업 | 상세 |
|------|------|
| Prometheus 메트릭 | rag_query_total, latency, token_usage, sync 메트릭 |
| Health check | /health (PostgreSQL, Redis, OpenAI 상태) |
| docker-compose 추가 | ai-assistant 서비스, pgvector 이미지 변경 |
| Prometheus 설정 | scrape config 추가 |
| Grafana 대시보드 | RAG 전용 패널 (선택) |
| 에러 처리 강화 | OpenAI retry, graceful degradation |

**완료 기준**: 전체 모니터링 스택에서 ai-assistant 메트릭 확인 가능

### Phase 5: 최적화 + 문서화 (1일)

| 작업 | 상세 |
|------|------|
| 성능 최적화 | 임베딩 캐싱, 쿼리 최적화, connection pool 튜닝 |
| 로깅 정리 | 구조화 로깅 (JSON), 민감 정보 마스킹 |
| E2E 테스트 완성 | 전체 시나리오 자동화 |
| API 문서 | OpenAPI/Swagger 정리 |
| CLAUDE.md 작성 | retro-rag-assistant용 개발 가이드 |

**완료 기준**: p95 응답 시간 < 10초, 테스트 커버리지 > 80%

---

## 16. Open Questions / Future Work

### Open Questions

| 질문 | 맥락 | 잠정 결론 |
|------|------|----------|
| 한국어 형태소 분석기 필요한가? | PostgreSQL `simple` config는 공백 기반 토큰화만 지원 | Phase 1에서는 `simple`로 시작, 검색 품질 측정 후 결정 |
| 임베딩 캐시가 필요한가? | 동일 텍스트 재임베딩 방지 | content hash 기반 캐시 검토 |
| moalog-server에 webhook이 필요한가? | 현재 5분 폴링. 실시간성이 필요하면 webhook | 5분 폴링으로 시작, 사용 패턴 보고 결정 |

### Future Work

| 항목 | 설명 | 우선순위 |
|------|------|---------|
| **Elasticsearch 통합** | 한국어 nori 형태소 분석기 지원, ts_vector 대체 | 높음 |
| **Fine-tuned Embedding** | 회고 도메인 특화 임베딩 모델 학습 | 중간 |
| **Feedback Loop** | 사용자 피드백(좋아요/나빠요)으로 검색 품질 개선 | 중간 |
| **다국어 지원** | 영어/일본어 회고 데이터 처리 | 낮음 |
| **Evaluation Pipeline** | RAGAS 등으로 RAG 품질 자동 측정 | 높음 |
| **Streaming Context** | LangGraph 노드 단위 중간 상태 SSE 전달 | 낮음 |
| **벡터 DB 마이그레이션** | 데이터 100K+ 시 Pinecone/Weaviate 전환 검토 | 낮음 |
| **Admin 대시보드** | 동기화 상태, 토큰 사용량, 인기 쿼리 시각화 | 중간 |
