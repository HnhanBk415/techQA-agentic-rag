import argparse
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sentence_transformers import CrossEncoder, SentenceTransformer

from scripts.generation.gemini_generator import generate_answer
from scripts.retrieval.bm25_search import load_bm25
from scripts.retrieval.dense_search_pgvector import MODEL_NAME
from scripts.retrieval.hybrid_retriever import retrieve_hybrid
from scripts.retrieval.reranker import RERANKER_MODEL, rerank_chunks


CANDIDATE_K = 20
TOP_K = 5


class HybridRAGState(TypedDict):
    question: str
    contexts: list[dict]
    answer: str


def build_graph(use_reranker=False):
    """Tạo luồng Hybrid RAG, có thể bật thêm reranker."""
    dense_model = SentenceTransformer(MODEL_NAME)
    bm25_index = load_bm25()

    # Chỉ tải model reranker khi người dùng bật --rerank.
    reranker = None
    if use_reranker:
        reranker = CrossEncoder(RERANKER_MODEL, device="cuda")

    def retrieve_node(state: HybridRAGState):
        question = state["question"]

        # Hai chế độ dùng cùng danh sách ứng viên sau RRF.
        candidates = retrieve_hybrid(
            question,
            dense_model,
            bm25_index,
            candidate_k=CANDIDATE_K,
            top_k=CANDIDATE_K,
        )

        # Chọn 5 chunk theo RRF hoặc theo điểm reranker.
        if reranker is None:
            chunks = candidates[:TOP_K]
        else:
            chunks = rerank_chunks(question, candidates, reranker, top_k=TOP_K)

        return {"contexts": chunks}

    def generate_node(state: HybridRAGState):
        # Gemini nhận câu hỏi và 5 chunk cuối cùng.
        answer = generate_answer(state["question"], state["contexts"])
        return {"answer": answer}

    builder = StateGraph(HybridRAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    question = input("Nhập câu hỏi: ").strip()
    if not question:
        return

    graph = build_graph(use_reranker=args.rerank)
    result = graph.invoke({
        "question": question,
        "contexts": [],
        "answer": "",
    })

    print("\nCâu trả lời:")
    print(result["answer"])

    score_name = "Reranker" if args.rerank else "RRF"
    print("\nNguồn đã đưa cho Gemini:")
    for rank, chunk in enumerate(result["contexts"], start=1):
        print(
            f"[S{rank}] {chunk['doc_id']} | {chunk['chunk_id']} | "
            f"{score_name}: {chunk['score']:.5f}"
        )


if __name__ == "__main__":
    main()