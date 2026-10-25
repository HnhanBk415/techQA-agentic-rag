import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# Chunk tối đa 200 token; hai chunk cạnh nhau dùng chung 40 token.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# Model này được huấn luyện với đoạn đến khoảng 250 word pieces.
TRAINING_TOKEN_LIMIT = 250

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def make_chunks(document):
    text = document["text"]
    doc_id = document["doc_id"]

    # Tokenizer cho biết mỗi token nằm ở vị trí ký tự nào trong văn bản gốc.
    encoded_text = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    offsets = encoded_text["offset_mapping"]

    chunks = []
    start_token = 0
    chunk_index = 0

    while start_token < len(offsets):
        end_token = min(start_token + CHUNK_SIZE, len(offsets))

        # Tìm vị trí bắt đầu/kết thúc chunk trong văn bản gốc.
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]

        # Giữ nguyên ranh giới do tokenizer xác định để chunk không vượt quá
        # CHUNK_SIZE. Chunk có thể bắt đầu hoặc kết thúc giữa một từ.
        chunk_text = text[char_start:char_end].strip()

        # Đếm token thật trong đoạn văn bản sẽ được đưa vào embedding model.
        token_count = len(
            tokenizer.encode(chunk_text, add_special_tokens=False)
        )

        chunks.append({
            "chunk_id": f"{doc_id}_{chunk_index}",
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "text": chunk_text
        })

        chunk_index += 1

        # Đã tới cuối tài liệu thì dừng.
        if end_token == len(offsets):
            break

        # Lùi lại 40 token để chunk sau có phần chồng lấn.
        start_token = end_token - CHUNK_OVERLAP

    return chunks


corpus_file = Path("data/processed/corpus.jsonl")
chunks_file = Path("data/processed/chunks.jsonl")
chunks_file.parent.mkdir(parents=True, exist_ok=True)

document_count = 0
chunk_count = 0
max_chunk_tokens = 0

with corpus_file.open("r", encoding="utf-8") as input_file:
    with chunks_file.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            document = json.loads(line)
            document_count += 1

            for chunk in make_chunks(document):
                output_file.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )
                chunk_count += 1
                max_chunk_tokens = max(
                    max_chunk_tokens,
                    chunk["token_count"]
                )

print("Tài liệu đã chia:", document_count)
print("Tổng số chunks:", chunk_count)
print("Chunk dài nhất:", max_chunk_tokens, "token")
print("Đã lưu tại:", chunks_file)

if max_chunk_tokens > TRAINING_TOKEN_LIMIT:
    print("Lưu ý: có chunk dài hơn mức token model đã được huấn luyện.")
