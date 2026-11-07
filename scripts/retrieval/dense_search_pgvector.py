from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

from scripts.db import connect_db

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
def retrieve_pgvector(question, model, top_k=5):
    query_vector = Vector(model.encode(question, normalize_embeddings=True))

    with connect_db() as conn:
        register_vector(conn)
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("""
                SELECT doc_id, chunk_id, content AS text,
                       1 - (embedding <=> %s) AS score
                FROM techqa_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_vector, query_vector, top_k))
            return cursor.fetchall()


def main():
    question = input("Nhập câu hỏi: ").strip()
    if not question:
        return

    model = SentenceTransformer(MODEL_NAME)
    for rank, chunk in enumerate(retrieve_pgvector(question, model), start=1):
        print(f"{rank}. {chunk['doc_id']} | {chunk['chunk_id']} | {chunk['score']:.4f}")
        print(chunk["text"][:250])


if __name__ == "__main__":
    main()