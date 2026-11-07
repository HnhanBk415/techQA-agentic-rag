# Mục đích:
# Tách các tài liệu (contexts) khỏi dataset câu hỏi-trả lời,
# gộp các tài liệu trùng tên, rồi lưu thành corpus.jsonl cho RAG.
# Đầu vào: dataset TechQA-RAG-Eval trên Hugging Face.
# Đầu ra: data/processed/corpus.jsonl
# Script này không sửa dataset gốc và không tạo embedding.

import json
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


ARCHIVE_FILE = Path("data/raw/corpus.zip")
DEV_FILE = Path("data/processed/dev_eval.jsonl")
CORPUS_FILE = Path("data/processed/corpus.jsonl")


def check_dev_coverage(doc_ids):
    with DEV_FILE.open(encoding="utf-8") as file:
        dev_rows = [json.loads(line) for line in file]

    answerable = [row for row in dev_rows if not row["is_impossible"]]
    missing = [
        row for row in answerable
        if row["gold_filename"] not in doc_ids
    ]

    print(f"DEV có tài liệu chuẩn trong kho: {len(answerable) - len(missing)}/{len(answerable)}")

    if missing:
        examples = ", ".join(row["gold_filename"] for row in missing[:5])
        raise ValueError(f"Thiếu tài liệu chuẩn cho {len(missing)} câu DEV. Ví dụ: {examples}")


def main():
    if not ARCHIVE_FILE.is_file():
        raise SystemExit(f"Chưa tìm thấy {ARCHIVE_FILE}.")
    if not DEV_FILE.is_file():
        raise SystemExit("Chạy prepare_eval.py trước để tạo dev_eval.jsonl.")

    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc_ids = set()
    temp_path = None

    try:
        with ZipFile(ARCHIVE_FILE) as archive, NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=CORPUS_FILE.parent, delete=False
        ) as output:
            temp_path = Path(output.name)

            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".txt"):
                    continue

                # corpus/swg24042191.txt → swg24042191.txt
                doc_id = PurePosixPath(member.filename).name
                text = archive.read(member).decode("utf-8-sig")

                if doc_id in doc_ids or not text.strip():
                    raise ValueError(f"Tài liệu trùng tên hoặc rỗng: {doc_id}")

                doc_ids.add(doc_id)
                output.write(json.dumps(
                    {"doc_id": doc_id, "text": text}, ensure_ascii=False
                ) + "\n")

        if not doc_ids:
            raise ValueError("Không tìm thấy tài liệu TXT trong ZIP.")

        check_dev_coverage(doc_ids)
        temp_path.replace(CORPUS_FILE)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print("Số tài liệu trong kho:", len(doc_ids))
    print("Đã lưu tại:", CORPUS_FILE)


if __name__ == "__main__":
    main()