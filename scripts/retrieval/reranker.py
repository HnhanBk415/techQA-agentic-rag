from sentence_transformers import CrossEncoder, SentenceTransformer

from scripts.retrieval.bm25_search import load_bm25
from scripts.retrieval.dense_search_pgvector import MODEL_NAME
from scripts.retrieval.hybrid_retriever import retrieve_hybrid


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def rerank_chunks(question, candidates, reranker, top_k=5):
    """Chấm lại các chunk ứng viên và trả về tối đa top_k kết quả."""
    if not candidates:
        return []

    # Danh sách văn bản giữ đúng thứ tự RRF ban đầu.
    texts = [chunk["text"] for chunk in candidates]

    # rank() ghép câu hỏi với từng văn bản, chấm điểm rồi xếp hạng.
    ranked = reranker.rank(
        question,
        texts,
        top_k=min(top_k, len(candidates)),
        batch_size=16,
        show_progress_bar=False,
    )

    results = []
    for item in ranked:
        # corpus_id là vị trí trong danh sách texts, đếm từ 0.
        # Dùng vị trí đó để lấy lại chunk và metadata tương ứng.
        index = item["corpus_id"]
        chunk = candidates[index]

        results.append({
            **chunk,
            "rrf_rank": index + 1,
            "rrf_score": chunk["score"],
            "score": float(item["score"]),
        })

    return results


def main():
    question = input("Nhập câu hỏi: ").strip()
    if not question:
        return

    dense_model = SentenceTransformer(MODEL_NAME)
    bm25_index = load_bm25()
    reranker = CrossEncoder(RERANKER_MODEL, device="cuda")
    print("Thiết bị reranker:", reranker.device)

    # Lấy tối đa 20 kết quả mỗi nhánh. Giữ 20 chunk tốt nhất sau RRF.
    candidates = retrieve_hybrid(
        question, dense_model, bm25_index, candidate_k=20, top_k=20
    )
    print("Số chunk đưa vào reranker:", len(candidates))

    for rank, chunk in enumerate(rerank_chunks(question, candidates, reranker), 1):
        print(f"\n{rank}. {chunk['doc_id']} | {chunk['chunk_id']}")
        print(f"   Hạng trước rerank: {chunk['rrf_rank']}")
        print(f"   Điểm reranker: {chunk['score']:.4f}")
        print(chunk["text"][:250])


if __name__ == "__main__":
    main()