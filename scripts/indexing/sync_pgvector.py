import json
from pathlib import Path

import numpy as np
from pgvector import Vector
from pgvector.psycopg import register_vector

from scripts.db import connect_db


def main():
    folder = Path("data/processed")
    vectors = np.load(folder / "chunk_embeddings.npy", mmap_mode="r")

    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] != 384:
        raise ValueError(f"Ma trận embedding không hợp lệ: {vectors.shape}")

    with (folder / "chunk_metadata.jsonl").open(encoding="utf-8") as metadata:
        with connect_db() as conn:
            register_vector(conn)
            conn.execute("TRUNCATE TABLE techqa_chunks")

            with conn.cursor() as cursor:
                with cursor.copy("""
                    COPY techqa_chunks (chunk_id, doc_id, content, embedding)
                    FROM STDIN
                """) as copy:
                    for line, vector in zip(metadata, vectors, strict=True):
                        chunk = json.loads(line)
                        copy.write_row((
                            chunk["chunk_id"],
                            chunk["doc_id"],
                            chunk["text"],
                            Vector(vector),
                        ))

    print(f"Đã nạp {len(vectors)} chunk vào PostgreSQL.")


if __name__ == "__main__":
    main()