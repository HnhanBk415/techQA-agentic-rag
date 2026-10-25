from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from scripts.generation.gemini_generator import generate_answer
from scripts.retrieval.search_dense import load_search_resources, retrieve


class DenseRAGState(TypedDict):
    # Câu hỏi ban đầu người dùng nhập.
    question: str

    # Các chunks mà node retrieve tìm được.
    contexts: list[dict]

    # Câu trả lời mà node generate tạo ra.
    answer: str


def build_graph():
    # Nạp model, vectors và metadata một lần khi khởi động graph.
    # Không nạp lại các tài nguyên này ở mỗi node.
    model, embeddings, chunks = load_search_resources()

    # Tìm Top-K chunks rồi ghi chúng vào state.
    def retrieve_node(state: DenseRAGState) -> dict:
        retrieved_chunks = retrieve(
            question=state["question"],
            model=model,
            embeddings=embeddings,
            chunks=chunks,
            top_k=5,
        )

        return {"contexts": retrieved_chunks}

    # Đưa câu hỏi và chunks trong state cho Gemini trả lời
    def generate_node(state: DenseRAGState) -> dict:
        
        answer = generate_answer(
            question=state["question"],
            contexts=state["contexts"],
        )

        return {"answer": answer}

    # Khai báo graph sẽ lưu ba trường trong DenseRAGState.
    builder = StateGraph(DenseRAGState)

    # Mỗi node thực hiện một việc riêng.
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    # Nối luồng: bắt đầu → tìm tài liệu → sinh câu trả lời → kết thúc.
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


def main():
    question = input("Nhập câu hỏi: ").strip()

    if not question:
        print("Bạn chưa nhập câu hỏi.")
        return

    graph = build_graph()

    # Đây là state ban đầu; hai node sẽ lần lượt bổ sung contexts và answer.
    initial_state: DenseRAGState = {
        "question": question,
        "contexts": [],
        "answer": "",
    }

    # stream_mode="values" cho xem state sau từng bước của graph.
    final_state = None
    for state_snapshot in graph.stream(
        initial_state,
        stream_mode="values",
    ):
        final_state = state_snapshot
        print(
            "Graph đã cập nhật state:",
            list(state_snapshot.keys()),
            "| số chunks:",
            len(state_snapshot.get("contexts", [])),
        )

    print("\nCâu trả lời:")
    print(final_state["answer"])

    print("\nCác nguồn đã đưa cho Gemini:")
    for index, chunk in enumerate(final_state["contexts"], start=1):
        print(
            f"[S{index}] {chunk['doc_id']} | "
            f"{chunk['chunk_id']} | "
            f"điểm={chunk['score']:.4f}"
        )


if __name__ == "__main__":
    main()