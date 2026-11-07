import json
from pathlib import Path

import bm25s

from scripts.retrieval.bm25_search import INDEX_DIR, tokenize_bm25


CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def main():
    """Tạo và lưu BM25 index từ bộ chunk của dự án."""
    with CHUNKS_FILE.open(encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file]

    if not chunks:
        raise ValueError("chunks.jsonl không có dữ liệu.")

    print("Số chunk cần lập chỉ mục:", len(chunks))

    # BM25 tính điểm từ văn bản của từng chunk.
    texts = [chunk["text"] for chunk in chunks]
    corpus_tokens = tokenize_bm25(texts, show_progress=True)

    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)

    # Metadata đi cùng index để khi tìm được chunk, mình biết nguồn của nó.
    retriever.save(str(INDEX_DIR), corpus=chunks)

    print("Đã lưu BM25 index tại:", INDEX_DIR)


if __name__ == "__main__":
    main()