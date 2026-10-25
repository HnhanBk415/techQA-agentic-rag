import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Dùng cùng model embedding đã dùng để chia token cho chunks.
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/processed/chunk_embeddings.npy")
METADATA_FILE = Path("data/processed/chunk_metadata.jsonl")


def load_chunks():
    """Đọc từng chunk đã lưu trong file JSONL."""
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("Không tìm thấy chunk nào trong chunks.jsonl")

    return chunks


def main():
    # Đọc 3.921 chunks đã chuẩn bị ở bước trước.
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print("Số chunks cần tạo embedding:", len(texts))

    # Model tự chọn GPU nếu máy có hỗ trợ; nếu không thì chạy bằng CPU.
    model = SentenceTransformer(MODEL_NAME)

    # Chuyển mỗi đoạn văn thành một vector số.
    # Chuẩn hóa vector để sau này so sánh bằng tích vô hướng.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Tạo thư mục đầu ra nếu chưa tồn tại.
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Lưu vector. Hàng thứ n tương ứng với chunk thứ n trong metadata.
    np.save(EMBEDDINGS_FILE, embeddings.astype(np.float32))

    # Lưu nội dung và ID của chunks theo đúng thứ tự vector.
    with METADATA_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Kích thước ma trận embedding:", embeddings.shape)
    print("Đã lưu vectors tại:", EMBEDDINGS_FILE)
    print("Đã lưu thông tin chunks tại:", METADATA_FILE)


if __name__ == "__main__":
    main()