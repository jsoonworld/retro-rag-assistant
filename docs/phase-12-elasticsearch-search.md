# Phase 12: Elasticsearch 검색 + CDC 파이프라인

## 한 줄 요약

MySQL의 LIKE 검색을 **Elasticsearch + Nori 한국어 형태소 분석기**로 대체하고, **Debezium CDC**로 실시간 동기화 파이프라인을 구축한다.

---

## 현재 상태 (AS-IS)

```
Client → GET /api/v1/retrospects/search?keyword=회고
              │
              └─ moalog-server
                    │
                    └─ MySQL: SELECT * FROM retrospects
                              WHERE title LIKE '%회고%'
                              ORDER BY start_time DESC
```

**문제점**:
1. **LIKE '%keyword%'** — 인덱스 사용 불가, 풀스캔
2. **형태소 분석 없음** — "회고했다"로 검색하면 "회고" 결과 못 찾음
3. **검색 범위** — title만 검색 (응답 내용, 댓글 검색 불가)
4. **검색 기능 부족** — 자동완성, 하이라이팅, 유사어 없음

---

## 목표 상태 (TO-BE)

```
                     ┌──────────────────────────────────────┐
                     │         Elasticsearch 8.x             │
                     │                                      │
  검색 요청 ────────▶│  Index: moalog-retrospects            │
                     │  ├─ title (Nori 분석)                 │
                     │  ├─ responses[].content (Nori 분석)   │
                     │  ├─ comments[].text (Nori 분석)       │
                     │  ├─ method (keyword)                  │
                     │  ├─ room_id (keyword)                 │
                     │  ├─ created_at (date)                 │
                     │  └─ member_ids[] (keyword)            │
                     └──────────────▲───────────────────────┘
                                    │
                                    │ 실시간 동기화
                                    │
                     ┌──────────────┴───────────────────────┐
                     │         Debezium (CDC)                │
                     │                                      │
                     │  MySQL binlog → Kafka → ES Sink      │
                     │                                      │
                     │  ┌─────────┐  ┌──────┐  ┌────────┐  │
                     │  │Debezium │─▶│Kafka │─▶│ES Sink │  │
                     │  │Connector│  │      │  │Connect.│  │
                     │  └─────────┘  └──────┘  └────────┘  │
                     └──────────────────────────────────────┘
                                    ▲
                                    │ binlog
                     ┌──────────────┴───────────────────────┐
                     │          MySQL 8.0                    │
                     │  retrospects, response, ...           │
                     └──────────────────────────────────────┘
```

---

## 선택지: 동기화 방식

### Option A: Debezium CDC (선택)

```
MySQL binlog → Debezium → Kafka → Elasticsearch Sink Connector
```

| 장점 | 단점 |
|------|------|
| DB 변경 자동 감지 (binlog) | Debezium + Kafka Connect 인프라 추가 |
| 애플리케이션 코드 수정 없음 | 초기 설정 복잡 |
| 삭제/수정도 자동 반영 | Kafka 의존성 (이미 있음) |
| 스키마 변경 자동 추적 | 디버깅 어려움 |

### Option B: Application-Level Sync

```
moalog-server → (CRUD 이후) → HTTP로 ES에 직접 색인
```

| 장점 | 단점 |
|------|------|
| 단순, 인프라 추가 없음 | 모든 쓰기 코드에 ES 호출 추가 |
| 디버깅 쉬움 | DB 커밋 ↔ ES 색인 불일치 가능 |
| Rust에서 직접 제어 | 삭제/벌크 수정 누락 위험 |

### Option C: Dual Write (DB + ES 동시 쓰기)

| 장점 | 단점 |
|------|------|
| 실시간 반영 | **분산 트랜잭션 문제** (DB OK + ES FAIL) |
| 코드 제어 가능 | 데이터 불일치 위험 높음 |

> **판단**: Kafka가 이미 있으므로 Debezium CDC가 가장 자연스러움. Application-Level은 Rust 코드 곳곳에 ES 호출을 심어야 하므로 침투적. Dual Write는 일관성 문제로 제외. CDC는 면접에서도 데이터 파이프라인 이해도를 보여주는 강력한 포인트.

---

## 구현 범위 (4개 기능)

### 기능 1: Elasticsearch 클러스터 + Nori 분석기

**인덱스 매핑**:

```json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "korean": {
          "type": "custom",
          "tokenizer": "nori_tokenizer",
          "filter": ["nori_readingform", "lowercase", "nori_part_of_speech"]
        },
        "korean_autocomplete": {
          "type": "custom",
          "tokenizer": "nori_tokenizer",
          "filter": ["nori_readingform", "lowercase", "edge_ngram_filter"]
        }
      },
      "filter": {
        "edge_ngram_filter": {
          "type": "edge_ngram",
          "min_gram": 1,
          "max_gram": 10
        },
        "nori_part_of_speech": {
          "type": "nori_part_of_speech",
          "stoptags": ["E", "J", "SC", "SE", "SF", "SP", "SSC", "SSO", "VCN", "VCP", "VSV", "XPN", "XSA", "XSN", "XSV"]
        }
      }
    },
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "retrospect_id":   { "type": "long" },
      "title":           { "type": "text", "analyzer": "korean", "fields": {
                            "autocomplete": { "type": "text", "analyzer": "korean_autocomplete" },
                            "keyword": { "type": "keyword" }
                          }},
      "method":          { "type": "keyword" },
      "room_id":         { "type": "long" },
      "room_name":       { "type": "keyword" },
      "start_time":      { "type": "date" },
      "member_ids":      { "type": "long" },
      "responses": {
        "type": "nested",
        "properties": {
          "response_id":   { "type": "long" },
          "content":       { "type": "text", "analyzer": "korean" },
          "category":      { "type": "keyword" },
          "member_id":     { "type": "long" }
        }
      },
      "comments": {
        "type": "nested",
        "properties": {
          "comment_id":    { "type": "long" },
          "text":          { "type": "text", "analyzer": "korean" },
          "member_id":     { "type": "long" }
        }
      },
      "response_count":  { "type": "integer" },
      "comment_count":   { "type": "integer" },
      "like_count":      { "type": "integer" },
      "updated_at":      { "type": "date" }
    }
  }
}
```

**Nori 분석기 효과**:

| 입력 | LIKE 검색 | Nori 검색 |
|------|----------|----------|
| "회고했다" → "회고" 검색 | ✗ (불일치) | ✓ (형태소: 회고+했다) |
| "팀 프로젝트" → "프로젝트" | ✓ | ✓ |
| "개선점을" → "개선점" | ✗ | ✓ (형태소: 개선점+을) |
| "retrospect" → "Retrospect" | ✗ (대소문자) | ✓ (lowercase) |

**트레이드오프: Nori vs Mecab**

| | Nori (선택) | Mecab (은전한닢) |
|---|---|---|
| 설치 | ES 플러그인 (기본 내장 8.x) | 사전 파일 별도 설치 |
| 사전 | 기본 사전 (충분) | 커스텀 사전 지원 강력 |
| 정확도 | 실용적 수준 | 약간 더 정확 |
| 유지보수 | ES 업그레이드 시 자동 | 사전 수동 관리 |

> **판단**: ES 8.x에 Nori 기본 내장. 회고 서비스의 도메인 특화 용어가 많지 않으므로 기본 사전으로 충분. Mecab은 쇼핑/의료 등 특수 도메인에서 유리.

---

### 기능 2: Debezium CDC 파이프라인

**아키텍처**:

```
MySQL (binlog)
    │
    ▼
Debezium Source Connector (Kafka Connect)
    │ 변경 이벤트 발행
    ▼
Kafka Topics:
    ├─ dbserver1.retrospect.retrospects
    ├─ dbserver1.retrospect.response
    ├─ dbserver1.retrospect.response_comment
    └─ dbserver1.retrospect.retro_room
    │
    ▼
Elasticsearch Sink Connector (Kafka Connect)
    │ 문서 색인/업데이트/삭제
    ▼
Elasticsearch Index: moalog-retrospects
```

**Debezium Source Connector 설정**:

```json
{
  "name": "moalog-mysql-source",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "mysql",
    "database.port": "3306",
    "database.user": "debezium",
    "database.password": "${DEBEZIUM_DB_PASSWORD}",
    "database.server.id": "1001",
    "topic.prefix": "dbserver1",
    "database.include.list": "retrospect",
    "table.include.list": "retrospect.retrospects,retrospect.response,retrospect.response_comment,retrospect.retro_room",
    "schema.history.internal.kafka.bootstrap.servers": "fluxpay-kafka:9092",
    "schema.history.internal.kafka.topic": "schema-changes.retrospect",
    "include.schema.changes": "false",
    "transforms": "unwrap",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "transforms.unwrap.delete.handling.mode": "rewrite"
  }
}
```

**MySQL 사전 준비**:
```sql
-- Debezium 전용 유저 (binlog 읽기 권한)
CREATE USER 'debezium'@'%' IDENTIFIED BY 'debezium_password';
GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'debezium'@'%';

-- binlog 활성화 확인 (Docker MySQL 8.0은 기본 활성)
SHOW VARIABLES LIKE 'log_bin';          -- ON
SHOW VARIABLES LIKE 'binlog_format';    -- ROW (필수)
```

**Elasticsearch Sink Connector 설정**:

```json
{
  "name": "moalog-es-sink",
  "config": {
    "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
    "connection.url": "http://elasticsearch:9200",
    "topics": "dbserver1.retrospect.retrospects",
    "type.name": "_doc",
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "key.converter.schemas.enable": "false",
    "value.converter.schemas.enable": "false",
    "transforms": "extractKey",
    "transforms.extractKey.type": "org.apache.kafka.connect.transforms.ExtractField$Key",
    "transforms.extractKey.field": "retrospect_id",
    "key.ignore": "false",
    "schema.ignore": "true",
    "behavior.on.null.values": "DELETE",
    "write.method": "UPSERT"
  }
}
```

**변경 파일**:
- `moalog-server/monitoring/debezium/` (신규 디렉토리)
  - `mysql-source.json` — Debezium Source 커넥터 설정
  - `es-sink.json` — Elasticsearch Sink 커넥터 설정
  - `register-connectors.sh` — 커넥터 등록 스크립트

**트레이드오프: Debezium 위치**

| | Kafka Connect (선택) | 독립 Debezium Server |
|---|---|---|
| 의존성 | Kafka Connect 클러스터 필요 | 단독 실행 가능 |
| Sink 지원 | Sink Connector로 ES 직접 연결 | 별도 Consumer 필요 |
| 운영 | Kafka Connect REST API로 관리 | 간단하지만 Sink 없음 |
| 확장성 | Connector 추가로 확장 | 제한적 |

> **판단**: 이미 Kafka가 있으므로 Kafka Connect가 자연스러움. Source(Debezium) + Sink(ES) 커넥터를 하나의 Connect 클러스터에서 관리. REST API로 커넥터 상태 모니터링 가능.

---

### 기능 3: 검색 API (moalog-server 수정)

**기존 API 수정**:

```
기존: GET /api/v1/retrospects/search?keyword=회고
      → MySQL LIKE 검색

변경: GET /api/v1/retrospects/search?q=회고&method=KPT&from=2026-01-01&sort=relevance
      → Elasticsearch 검색 (fallback: MySQL LIKE)
```

**새 쿼리 파라미터**:

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `q` | string | 검색어 (title + responses + comments) |
| `method` | enum | 회고 방법 필터 (KPT, 4L, 5F, PMI, FREE) |
| `room_id` | i64 | 방 필터 |
| `from` / `to` | date | 날짜 범위 |
| `sort` | enum | relevance (기본) / date / likes |
| `page` / `size` | int | 페이지네이션 |

**Elasticsearch 쿼리**:

```json
{
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "회고",
            "fields": ["title^3", "responses.content", "comments.text"],
            "type": "best_fields",
            "analyzer": "korean"
          }
        }
      ],
      "filter": [
        { "terms": { "room_id": [1, 2, 3] } },
        { "range": { "start_time": { "gte": "2026-01-01" } } }
      ]
    }
  },
  "highlight": {
    "fields": {
      "title": {},
      "responses.content": { "fragment_size": 150 }
    },
    "pre_tags": ["<mark>"],
    "post_tags": ["</mark>"]
  },
  "from": 0,
  "size": 20
}
```

**새 엔드포인트 (자동완성)**:

```
GET /api/v1/retrospects/suggest?q=프로

→ Response:
{
  "suggestions": ["프로젝트 회고", "프로덕트 리뷰", "프로세스 개선"]
}
```

**Elasticsearch Suggest 쿼리**:
```json
{
  "suggest": {
    "title-suggest": {
      "prefix": "프로",
      "completion": {
        "field": "title.autocomplete",
        "size": 5,
        "skip_duplicates": true
      }
    }
  }
}
```

**변경 파일**:
- `moalog-server/codes/server/Cargo.toml` — `elasticsearch` 크레이트 추가
- `moalog-server/codes/server/src/config/elasticsearch.rs` (신규 — ES 클라이언트)
- `moalog-server/codes/server/src/domain/retrospect/search_service.rs` (신규 — 검색 로직)
- `moalog-server/codes/server/src/domain/retrospect/handler.rs` — search 핸들러 수정
- `moalog-server/codes/server/src/domain/retrospect/dto.rs` — SearchQueryParams 확장
- `moalog-server/codes/server/src/main.rs` — suggest 라우트 추가

**Fallback 전략**:
```rust
async fn search(params: SearchParams) -> Result<Vec<SearchResult>> {
    // 1차: Elasticsearch 검색 시도
    match es_client.search(&params).await {
        Ok(results) => Ok(results),
        Err(e) => {
            // ES 장애 시 MySQL LIKE 폴백
            warn!("ES search failed, falling back to MySQL: {}", e);
            mysql_like_search(&params).await
        }
    }
}
```

> **트레이드오프**: ES 장애 시 MySQL LIKE로 폴백하면 검색 품질은 떨어지지만 서비스는 유지. Fail Open과 동일한 철학.

---

### 기능 4: 초기 데이터 마이그레이션

ES가 처음 시작될 때 기존 MySQL 데이터를 Elasticsearch에 벌크 색인:

```bash
# 초기 색인 스크립트
#!/bin/bash
# 1. ES 인덱스 생성 (매핑 적용)
curl -X PUT "localhost:9200/moalog-retrospects" -H 'Content-Type: application/json' -d @index-mapping.json

# 2. Debezium snapshot 모드로 초기 로드
# Debezium의 snapshot.mode=initial이 자동으로 기존 데이터를 읽어서 Kafka → ES로 전달

# 3. 이후부터는 binlog 기반 실시간 동기화
```

Debezium의 `snapshot.mode=initial` 설정이 초기 전체 로드를 자동 처리.

---

## Docker Compose 변경

```yaml
# 신규 서비스 3개 추가

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    container_name: moalog-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "${ES_PORT:-9200}:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 5

  kafka-connect:
    image: confluentinc/cp-kafka-connect:7.5.0
    container_name: moalog-kafka-connect
    depends_on:
      fluxpay-kafka:
        condition: service_started
      elasticsearch:
        condition: service_healthy
    environment:
      CONNECT_BOOTSTRAP_SERVERS: fluxpay-kafka:9092
      CONNECT_GROUP_ID: moalog-connect
      CONNECT_CONFIG_STORAGE_TOPIC: connect-configs
      CONNECT_OFFSET_STORAGE_TOPIC: connect-offsets
      CONNECT_STATUS_STORAGE_TOPIC: connect-status
      CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR: 1
      CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR: 1
      CONNECT_STATUS_STORAGE_REPLICATION_FACTOR: 1
      CONNECT_KEY_CONVERTER: org.apache.kafka.connect.json.JsonConverter
      CONNECT_VALUE_CONVERTER: org.apache.kafka.connect.json.JsonConverter
      CONNECT_REST_PORT: 8083
      CONNECT_PLUGIN_PATH: /usr/share/java,/usr/share/confluent-hub-components
    ports:
      - "${KAFKA_CONNECT_PORT:-8083}:8083"
    command:
      - bash
      - -c
      - |
        confluent-hub install --no-prompt debezium/debezium-connector-mysql:2.5.0
        confluent-hub install --no-prompt confluentinc/kafka-connect-elasticsearch:14.0.0
        /etc/confluent/docker/run

  # Kibana (선택 — ES 관리/디버깅용)
  kibana:
    image: docker.elastic.co/kibana/kibana:8.12.0
    container_name: moalog-kibana
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    ports:
      - "${KIBANA_PORT:-5601}:5601"
    depends_on:
      elasticsearch:
        condition: service_healthy
```

**새 포트**:
- Elasticsearch: 9200
- Kafka Connect REST API: 8083
- Kibana: 5601 (선택)

---

## Prometheus 모니터링

**ES 메트릭**: Elasticsearch Exporter 추가

```yaml
  elasticsearch-exporter:
    image: quay.io/prometheuscommunity/elasticsearch-exporter:v1.7.0
    container_name: moalog-es-exporter
    command: ["--es.uri=http://elasticsearch:9200"]
    ports:
      - "9114:9114"
```

**Prometheus 타겟 추가**:
```yaml
- job_name: 'elasticsearch'
  static_configs:
    - targets: ['elasticsearch-exporter:9114']
```

**주요 메트릭**:
- `elasticsearch_cluster_health_status`
- `elasticsearch_indices_docs_count`
- `elasticsearch_indices_search_query_time_seconds`
- `elasticsearch_indices_indexing_index_time_seconds`

---

## 검증 방법

```bash
# 1. ES 클러스터 헬스
curl http://localhost:9200/_cluster/health?pretty

# 2. 인덱스 매핑 확인
curl http://localhost:9200/moalog-retrospects/_mapping?pretty

# 3. Nori 분석기 테스트
curl -X POST "localhost:9200/moalog-retrospects/_analyze" \
  -H 'Content-Type: application/json' \
  -d '{"analyzer": "korean", "text": "팀 프로젝트 회고를 진행했습니다"}'
# → ["팀", "프로젝트", "회고", "진행"]

# 4. 검색 테스트
curl "localhost:8090/api/v1/retrospects/search?q=회고&sort=relevance"

# 5. CDC 동기화 확인
# MySQL에 INSERT → ES 인덱스에 자동 반영 확인 (지연 <2초)

# 6. Kafka Connect 커넥터 상태
curl http://localhost:8083/connectors/moalog-mysql-source/status
curl http://localhost:8083/connectors/moalog-es-sink/status
```

---

## K8s 매니페스트 추가

```
k8s/base/
├── elasticsearch/
│   ├── statefulset.yaml     # ES 단일 노드
│   ├── service.yaml
│   └── kustomization.yaml
├── kafka-connect/
│   ├── deployment.yaml      # Debezium + ES Sink
│   ├── service.yaml
│   ├── configmap.yaml       # 커넥터 설정
│   └── kustomization.yaml
└── monitoring/exporters/
    └── elasticsearch-exporter/
```

---

## 예상 작업량

| 기능 | 신규 파일 | 수정 파일 | 난이도 |
|------|----------|----------|--------|
| ES + Nori 인덱스 | 2 (매핑, 설정) | 0 | ★★☆ |
| Debezium CDC | 3 (커넥터 설정) | 1 (docker-compose) | ★★★ |
| 검색 API 수정 | 3 (ES 클라이언트, 검색 서비스, suggest) | 3 (handler, dto, routes) | ★★★ |
| Docker/K8s 인프라 | 6 (compose, k8s manifests) | 2 (prometheus, makefile) | ★★☆ |

**의존성**: ES 인프라 → CDC 파이프라인 → 검색 API (순차)

---

## 면접 키워드

- Elasticsearch: 역인덱스, 형태소 분석, TF-IDF/BM25 스코어링
- Nori: 한국어 형태소 분석, 품사 태깅, 스톱태그
- CDC: Change Data Capture, binlog, Debezium, offset 관리
- Kafka Connect: Source/Sink 패턴, SMT(Single Message Transform)
- 검색 품질: relevance, boosting, multi_match, highlight
- 동기화 전략: CDC vs Dual Write vs Application-Level Sync
- Fallback: ES 장애 시 MySQL 폴백
