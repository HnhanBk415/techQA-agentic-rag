# Mục đích:
# Tách các tài liệu (contexts) khỏi dataset câu hỏi-trả lời,
# gộp các tài liệu trùng tên, rồi lưu thành corpus.jsonl cho RAG.
# Đầu vào: dataset TechQA-RAG-Eval trên Hugging Face.
# Đầu ra: data/processed/corpus.jsonl
# Script này không sửa dataset gốc và không tạo embedding.

import json
from pathlib import Path

from dataset_utils import get_all_contexts, load_techqa_dataframe


# 1. Đọc dataset thành bảng.
# Hàm dùng chung cũng thêm cột source_split để phân biệt TRAIN/DEV.
dataframe = load_techqa_dataframe()

# Chỉ dùng tài liệu từ TRAIN để làm kho truy xuất.
# Các câu hỏi DEV sẽ dùng để đánh giá, nên không lấy context từ DEV vào corpus.
dataframe = dataframe[dataframe["source_split"] == "TRAIN"].copy()

# 2. Gom các context trong từng câu hỏi thành một danh sách.
# Mỗi context là một tài liệu có tên file (filename) và nội dung (text).
contexts = get_all_contexts(dataframe)

# 3. Chuẩn bị từ điển để chỉ lưu một bản cho mỗi tên file.
# Ví dụ: documents["swg123.txt"] sẽ là tài liệu có tên đó.
documents = {}

for context in contexts:
    filename = context["filename"]
    text = context["text"]

    # Nếu đã gặp tên file này, kiểm tra nội dung có giống bản đã lưu không.
    if filename in documents:
        if documents[filename]["text"] != text:
            # Dừng lại nếu cùng tên file nhưng nội dung khác,
            # để mình kiểm tra thay vì âm thầm bỏ mất một phiên bản.
            raise ValueError(
                f"Cùng tên file nhưng nội dung khác nhau: {filename}"
            )

        # Nội dung giống nhau rồi, bỏ qua bản lặp và chuyển sang context kế.
        continue

    # Lần đầu gặp tên file: lưu ID nguồn và nội dung tài liệu.
    documents[filename] = {
        "doc_id": filename,
        "text": text
    }

# 4. Chọn đường dẫn đầu ra.
# Đường dẫn tính từ thư mục gốc repo khi chạy lệnh.
output_file = Path("data/processed/corpus.jsonl")

# Tạo data/processed nếu chưa có; nếu đã có thì tiếp tục bình thường.
output_file.parent.mkdir(parents=True, exist_ok=True)

# 5. Ghi mỗi tài liệu thành một dòng JSON (JSON Lines).
# Chế độ "w" tạo mới hoặc ghi đè file corpus khi chạy lại.
with output_file.open("w", encoding="utf-8") as file:
    for document in documents.values():
        # ensure_ascii=False giữ nguyên chữ Unicode thay vì mã hóa thành \\u...
        json_line = json.dumps(document, ensure_ascii=False)
        file.write(json_line + "\n")

# 6. In tóm tắt để kiểm tra kết quả.
print("Lượt context ban đầu:", len(contexts))
print("Tài liệu sau khi gộp:", len(documents))
print("Đã lưu tại:", output_file)
