import json
from pathlib import Path

from dataset_utils import load_techqa_dataframe


dataframe = load_techqa_dataframe()

# Ghi nhớ các câu hỏi thuộc TRAIN để tìm câu trùng trong DEV.
train_questions = []

for _, row in dataframe.iterrows():
    if row["source_split"] == "TRAIN":
        train_questions.append(row["question"])

dev_eval = []
duplicate_count = 0

# Chỉ lấy câu DEV; bỏ qua câu nào đã xuất hiện trong TRAIN.
for _, row in dataframe.iterrows():
    if row["source_split"] != "DEV":
        continue

    if row["question"] in train_questions:
        duplicate_count += 1
        continue

    # Lưu tên tài liệu đúng để sau này kiểm tra hệ thống có tìm ra không.
    if row["contexts"]:
        gold_filename = row["contexts"][0]["filename"]
    else:
        gold_filename = None

    dev_eval.append({
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "is_impossible": bool(row["is_impossible"]),
        "gold_filename": gold_filename
    })

# Lưu mỗi câu hỏi thành một dòng JSON.
output_file = Path("data/processed/dev_eval.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

with output_file.open("w", encoding="utf-8") as file:
    for question in dev_eval:
        json_line = json.dumps(question, ensure_ascii=False)
        file.write(json_line + "\n")

print("Câu DEV trùng TRAIN đã bỏ khỏi bản đánh giá:", duplicate_count)
print("Câu hỏi còn lại để đánh giá:", len(dev_eval))
print("Đã lưu tại:", output_file)