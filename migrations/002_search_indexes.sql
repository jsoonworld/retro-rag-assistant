-- Full-text search GIN index (PostgreSQL tsvector)
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN (tsv);

-- Vector search index (HNSW - pgvector 0.5+)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Query log time-series index
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log (created_at DESC);
