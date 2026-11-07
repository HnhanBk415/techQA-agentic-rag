from pathlib import Path

import bm25s


INDEX_DIR = Path("data/processed/bm25_index")

TOKEN_PATTERN = r"(?u)\b(?:\d+(?:\.\d+)+|\w+)\b"

def tokenize_bm25(texts, show_progress=False):
    """Tách token theo cùng một quy tắc cho tài liệu và câu hỏi."""
    return bm25s.tokenize(
        texts,
        lower=True,
        token_pattern=TOKEN_PATTERN,
        stopwords="en",
        show_progress=show_progress,
    )


def load_bm25():
    """Mở index đã tạo và nạp thông tin chunk đi kèm."""
    if not INDEX_DIR.is_dir():
        raise FileNotFoundError(
            "Chưa có BM25 index. Hãy chạy build_bm25_index trước."
        )

    # mmap cho phép đọc index từ đĩa theo nhu cầu.
    return bm25s.BM25.load(str(INDEX_DIR), load_corpus=True, mmap=True)


def retrieve_bm25(question, retriever, top_k=5):
    """Trả về tối đa top_k chunk có điểm BM25 dương."""
    if not question.strip() or top_k <= 0:
        return []

    query_tokens = tokenize_bm25([question])
    documents, scores = retriever.retrieve(
        query_tokens,
        k=min(top_k, len(retriever.corpus)),
        show_progress=False,
    )

    results = []
    # Chỉ có một câu hỏi, nên kết quả của nó nằm ở hàng [0].
    for chunk, score in zip(documents[0], scores[0]):
        # Điểm 0 nghĩa là chunk không khớp từ khóa nào được tính điểm.
        if score > 0:
            results.append({**chunk, "score": float(score)})

    return results


def main():
    question = input("Nhập câu hỏi: ").strip()
    if not question:
        return

    retriever = load_bm25()
    results = retrieve_bm25(question, retriever, top_k=5)

    for rank, chunk in enumerate(results, start=1):
        print(f"{rank}. {chunk['doc_id']} | {chunk['chunk_id']}")
        print(f"   BM25: {chunk['score']:.4f}")
        print(chunk["text"][:250])

    if not results:
        print("Không tìm thấy chunk khớp từ khóa.")


if __name__ == "__main__":
    main()