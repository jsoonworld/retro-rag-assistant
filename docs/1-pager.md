# AI Retrospect Assistant — RAG + LangGraph

## 한 줄 요약

OpenAI Embedding + pgvector + LangGraph를 활용한 **AI 회고 어시스턴트**. 과거 회고 데이터를 벡터화하여, "우리 팀의 반복되는 문제가 뭐야?" 같은 질문에 근거 기반 답변을 제공한다.

---

## 풀고자 하는 문제

현재 moalog-server의 AI 기능:

```
현재: 단일 회고 → OpenAI → 그 회고에 대한 분석 (단건)
     "이번 스프린트 회고를 분석해줘"

없는 것: 여러 회고를 종합한 패턴 분석, 맥락 기반 Q&A
     "최근 6개월간 반복되는 이슈가 뭐야?"        ← 불가능
     "프론트엔드 팀의 개선 트렌드를 보여줘"        ← 불가능
     "지난번 비슷한 문제 때 어떻게 해결했지?"      ← 불가능
```

**핵심 한계**: 단건 분석만 가능. 과거 회고를 **검색하고 종합하는 능력**이 없음.

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│               AI Retrospect Assistant                             │
│               Python · FastAPI · LangGraph                       │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ Ingestion Pipeline                                     │      │
│  │                                                        │      │
│  │ moalog MySQL → Sync Job → Chunking → OpenAI Embedding │      │
│  │                (주기적)   (회고 단위)  (1536차원)       │      │
│  │                               │                        │      │
│  │                               ▼                        │      │
│  │                        ┌──────────────┐                │      │
│  │                        │   pgvector   │                │      │
│  │                        │ (PostgreSQL)  │                │      │
│  │                        └──────────────┘                │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ LangGraph Workflow                                     │      │
│  │                                                        │      │
│  │ User Query                                             │      │
│  │   ▼                                                    │      │
│  │ ┌─────────────┐   ┌───────────┐   ┌─────────────┐     │      │
│  │ │Query Analyzer│──▶│ Retriever │──▶│Context Build│     │      │
│  │ │의도 분류     │   │pgvector   │   │토큰 최적화  │     │      │
│  │ │필터 추출     │   │하이브리드 │   │             │     │      │
│  │ └─────────────┘   └───────────┘   └──────┬──────┘     │      │
│  │                                          ▼            │      │
│  │                                   ┌─────────────┐     │      │
│  │                                   │ GPT-4o      │     │      │
│  │                                   │ 답변 생성   │     │      │
│  │                                   │ + 출처 인용 │     │      │
│  │                                   │ (SSE 스트림) │     │      │
│  │                                   └─────────────┘     │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ Conversation Memory (Redis)                             │      │
│  │ 세션별 대화 이력 (최근 10턴, TTL 1시간)                  │      │
│  │ → 후속 질문 "그중에서 가장 심각한 건?" 지원              │      │
│  └────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 카테고리 | 기술 | 선택 이유 |
|---------|------|----------|
| 언어 | Python 3.12 | LangGraph/LangChain 생태계가 Python 중심 |
| 웹 프레임워크 | FastAPI | async 네이티브, SSE 지원, OpenAPI 자동 생성 |
| AI 워크플로우 | LangGraph | 조건부 분기, 멀티스텝 추론, 상태 관리 |
| LLM | OpenAI GPT-4o | 한국어 품질 최상위 |
| Embedding | text-embedding-3-small | 1536차원, 비용 효율적 |
| Vector DB | pgvector (PostgreSQL 확장) | 기존 PostgreSQL 활용, 별도 인프라 불필요 |
| 대화 메모리 | Redis | TTL 기반 세션 관리 |
| 모니터링 | prometheus_client | 토큰 사용량, 응답 시간 |

---

## 핵심 기능 (5개)

### 1. Ingestion — 회고 데이터 벡터화

**청킹 전략** (회고 1건 → N개 청크):

| 청크 타입 | 내용 | 예시 |
|----------|------|------|
| summary | 제목 + 방법 + 날짜 + 참가자 수 | "[백엔드팀] Sprint 3 회고 (2026-01-15) KPT 5명" |
| response | 카테고리별 응답 묶음 | "[Problem] - 코드 리뷰 지연 - 배포 병목" |
| analysis | AI 분석 결과 (있으면) | "[AI 분석] 주요 이슈는..." |

**임베딩**: OpenAI `text-embedding-3-small` (1536차원, 배치 처리)

**동기화**: 주기적 폴링 (5분) — 새로 생성/수정된 회고만 증분 처리

**트레이드오프: pgvector vs Pinecone/Weaviate**

| | pgvector (선택) | Pinecone / Weaviate |
|---|---|---|
| 인프라 | 기존 PostgreSQL에 확장 | 별도 서비스 |
| 비용 | 0 (self-hosted) | SaaS 비용 |
| 성능 | ~100K 벡터 OK | 수백만 벡터 |
| SQL 결합 | WHERE + 벡터 검색 하나의 쿼리 | 별도 API |

> **판단**: 회고 데이터는 수천~수만 건. pgvector HNSW 인덱스로 충분. SQL 필터링(날짜, 방, 방법)과 벡터 검색을 결합할 수 있는 것이 강점.

---

### 2. LangGraph 워크플로우 — 의도별 분기

```python
workflow = StateGraph(AssistantState)

workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve", retrieve_from_pgvector)
workflow.add_node("build_context", build_context)
workflow.add_node("generate_answer", generate_with_gpt4o)
workflow.add_node("format_response", format_response)

# 의도별 분기
workflow.add_conditional_edges("analyze_query", route_by_intent, {
    "pattern_analysis": "retrieve",  # "반복되는 문제" → 넓은 검색
    "search": "retrieve",           # "Redis 관련 회고" → 타겟 검색
    "compare": "retrieve",          # "1월 vs 2월 비교" → 다중 검색
    "general": "generate_answer",   # "KPT가 뭐야?" → 바로 답변
})
```

| 의도 | 예시 | 검색 전략 | 생성 전략 |
|------|------|----------|----------|
| pattern_analysis | "반복되는 문제?" | 최근 N개월 전체, Problem 가중 | 패턴 추출 + 빈도 |
| search | "Redis 관련 회고" | 키워드+벡터 하이브리드 | 목록 + 요약 |
| compare | "1월 vs 2월 비교" | 기간별 분리 검색 | 차이점 비교 |
| general | "KPT가 뭐야?" | 검색 불필요 | LLM 지식 |

**트레이드오프: LangGraph vs 단순 체이닝** — 의도별 다른 처리가 필요하므로 LangGraph의 조건부 분기가 적합. 단순 RAG라면 체이닝으로 충분.

---

### 3. 하이브리드 검색 — Vector + Keyword

```python
async def hybrid_search(query, filters, top_k=10):
    # 1. 벡터 검색 (의미적 유사도)
    vector_results = await pgvector_search(query_embedding, filters, top_k)

    # 2. 키워드 검색 (PostgreSQL ts_vector)
    keyword_results = await keyword_search(query, filters, top_k)

    # 3. RRF (Reciprocal Rank Fusion)로 결합
    return reciprocal_rank_fusion(vector_results, keyword_results)
```

> 벡터만 → "Redis"를 "캐시"로도 찾음 but 키워드 매칭 약함. 키워드만 → 유사어 못 찾음. RRF로 양쪽 장점 결합.

---

### 4. 스트리밍 응답 (SSE)

```python
@app.post("/api/v1/assistant/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(stream_answer(request), media_type="text/event-stream")

# SSE 이벤트:
# data: {"type": "text", "content": "최근 6개월간"}
# data: {"type": "text", "content": " 회고를 분석한 결과..."}
# data: {"type": "citations", "data": [{retrospect_id, title, date, relevance}]}
# data: {"type": "done"}
```

---

### 5. 대화 메모리 (Redis)

```python
# Redis List: session:{session_id}:history (TTL 1시간)
# 최근 10턴(20개 메시지) 유지

# 멀티턴 예시:
# User:  "반복되는 문제점이 뭐야?"
# AI:    "1. 코드 리뷰 지연 (5회) 2. 배포 병목 (4회)..."
# User:  "코드 리뷰 문제를 어떻게 해결했어?"   ← 후속 질문
# AI:    "코드 리뷰에 대해 팀이 시도한 해결책: PR 크기 제한, 리뷰어 자동 배정..."
```

---

## DB 스키마

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE retrospect_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    retrospect_id   BIGINT NOT NULL,
    chunk_type      VARCHAR(20) NOT NULL,
    category        VARCHAR(50),
    content         TEXT NOT NULL,
    embedding       vector(1536) NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (retrospect_id, chunk_type, category)
);

CREATE INDEX idx_embedding_hnsw ON retrospect_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

---

## API 설계

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/assistant/chat` | 질의응답 (SSE 스트리밍) |
| GET | `/api/v1/assistant/sessions/{id}/history` | 대화 이력 |
| DELETE | `/api/v1/assistant/sessions/{id}` | 세션 삭제 |
| POST | `/api/v1/assistant/search` | 벡터 검색만 (RAG 없이) |
| POST | `/api/v1/ingestion/sync` | 수동 동기화 트리거 |
| GET | `/api/v1/ingestion/status` | 동기화 상태 |

**Chat 요청**:
```json
{
  "session_id": "sess-abc123",
  "message": "우리 팀의 반복되는 문제점이 뭐야?",
  "filters": { "room_ids": [1, 2], "date_from": "2025-07-01" }
}
```

---

## Docker Compose 추가

```yaml
ai-assistant:
  build: ../nori-search-engine
  ports:
    - "8085:8000"
  environment:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
    DATABASE_URL: postgresql://fluxpay:fluxpay@fluxpay-postgres:5432/fluxpay
    MOALOG_DB_URL: mysql://root:moalog_local@mysql:3306/retrospect
    REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/1
  depends_on: [fluxpay-postgres, mysql, redis]

# PostgreSQL 이미지 변경: postgres:16-alpine → pgvector/pgvector:pg16
```

---

## moalog 통합

```
1. 프론트엔드에 "AI 어시스턴트" 탭 추가
   → POST /api/v1/assistant/chat → SSE 스트리밍 렌더링

2. 회고 제출 시 자동 벡터화
   → webhook 또는 sync job이 새 회고 감지 → 임베딩

3. 기존 단건 분석과 공존
   → 단건: moalog-server 기존 OpenAI 호출 (유지)
   → 종합: 이 서비스의 RAG 기반 답변 (신규)
```

---

## 면접 키워드

- RAG: Retrieval Augmented Generation, 환각 감소, 근거 기반 답변
- Embedding: text-embedding-3-small, 코사인 유사도, 차원수 선택
- pgvector: HNSW 인덱스, IVFFlat vs HNSW, SQL+벡터 결합
- LangGraph: 상태 기반 워크플로우, 조건부 분기
- 하이브리드 검색: RRF, 벡터+키워드 결합
- 청킹 전략: 문서 크기, 메타데이터 보존
- 토큰 관리: 컨텍스트 윈도우 최적화, 비용 제어
- 스트리밍: SSE, 토큰 단위 전달
