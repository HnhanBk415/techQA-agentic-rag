import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

EMBEDDINGS_FILE = Path("data/processed/chunk_embeddings.npy")
METADATA_FILE = Path("data/processed/chunk_metadata.jsonl")

TOP_K = 5

# Đọc thông tin các chunks theo thứ tự đã dùng khi lưu embeddings.
def load_metadata():
    chunks = []

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks

# Tải model, vectors và metadata một lần để dùng cho các câu hỏi.
def load_search_resources():
    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_metadata()

    # Mỗi vector phải khớp với một chunk cùng vị trí trong metadata.
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Số vectors ({len(embeddings)}) không khớp "
            f"số chunks ({len(chunks)})"
        )

    model = SentenceTransformer(MODEL_NAME)
    print("Thiết bị đang dùng:", model.device)

    return model, embeddings, chunks

# Tìm và trả về top-K chunks liên quan nhất với câu hỏi.
def retrieve(question, model, embeddings, chunks, top_k=TOP_K):

    # Câu hỏi phải được mã hóa bằng đúng model đã mã hóa tài liệu.
    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Vectors đã chuẩn hóa nên tích vô hướng tương đương cosine similarity.
    scores = embeddings @ question_vector

    # Lấy vị trí các chunks từ điểm cao xuống thấp.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        # Giữ metadata gốc rồi thêm điểm để node sau có thể sử dụng.
        chunk = dict(chunks[int(index)])
        chunk["score"] = float(scores[index])
        results.append(chunk)

    # Mỗi kết quả có doc_id, chunk_id, text và score.
    return results


def main():
    question = input("Nhập câu hỏi: ").strip()

    if not question:
        print("Bạn chưa nhập câu hỏi.")
        return

    # Nạp tài nguyên một lần, sau đó tìm chunks cho câu hỏi.
    model, embeddings, chunks = load_search_resources()
    results = retrieve(question, model, embeddings, chunks)

    print(f"\nTop {len(results)} chunks tìm được:\n")

    for rank, chunk in enumerate(results, start=1):
        print(f"--- Hạng {rank} ---")
        print("Tài liệu:", chunk["doc_id"])
        print("Chunk:", chunk["chunk_id"])
        print("Điểm tương đồng:", f"{chunk['score']:.4f}")
        print("Nội dung:", chunk["text"][:700].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()