CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS techqa_chunks (
    chunk_id text PRIMARY KEY,
    doc_id text NOT NULL,
    content text NOT NULL,
    embedding vector(384) NOT NULL
);