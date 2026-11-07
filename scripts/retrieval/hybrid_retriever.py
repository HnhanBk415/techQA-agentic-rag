from sentence_transformers import SentenceTransformer

from scripts.retrieval.bm25_search import load_bm25, retrieve_bm25
from scripts.retrieval.dense_search_pgvector import MODEL_NAME, retrieve_pgvector


def fuse_rrf(dense_results, bm25_results, top_k=5, rrf_constant=60):
    """Gộp hai danh sách xếp hạng và trả về Top K chunk."""
    fused = {}

    for rank_field, results in (
        ("dense_rank", dense_results),
        ("bm25_rank", bm25_results),
    ):
        # Thứ hạng bắt đầu từ 1, giống thứ tự in trên terminal.
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]

            # Cùng một chunk có thể xuất hiện ở cả hai nhánh.
            # Chỉ tạo một bản ghi rồi cộng điểm vào bản ghi đó.
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }

            result = fused[chunk_id]
            result[rank_field] = rank
            result["score"] += 1 / (rrf_constant + rank)

    # Điểm RRF cao đứng trước. Nếu bằng điểm, dùng mã chunk để chốt thứ tự.
    ranked = sorted(
        fused.values(),
        key=lambda chunk: (-chunk["score"], chunk["chunk_id"]),
    )
    return ranked[:top_k]


def retrieve_hybrid(question, dense_model, bm25_index, candidate_k=20, top_k=5):
    """Tìm ứng viên bằng cả Dense và BM25, sau đó gộp bằng RRF."""
    dense_results = retrieve_pgvector(question, dense_model, top_k=candidate_k)
    bm25_results = retrieve_bm25(question, bm25_index, top_k=candidate_k)

    return fuse_rrf(dense_results, bm25_results, top_k=top_k)


def main():
    question = input("Nhập câu hỏi: ").strip()
    if not question:
        return

    # Tải một lần rồi truyền tài nguyên vào hàm tìm kiếm.
    dense_model = SentenceTransformer(MODEL_NAME)
    bm25_index = load_bm25()
    results = retrieve_hybrid(question, dense_model, bm25_index)

    for rank, chunk in enumerate(results, start=1):
        print(f"\n{rank}. {chunk['doc_id']} | {chunk['chunk_id']}")
        print(
            f"   RRF: {chunk['score']:.5f} | "
            f"Dense: {chunk['dense_rank']} | BM25: {chunk['bm25_rank']}"
        )
        print(chunk["text"][:250])


if __name__ == "__main__":
    main()