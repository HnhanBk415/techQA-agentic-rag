import json
from pathlib import Path
from statistics import mean, median


corpus_file = Path("data/processed/corpus.jsonl")
word_counts = []

# Đọc từng tài liệu trong corpus và đếm số từ gần đúng.
with corpus_file.open("r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line)
        word_count = len(document["text"].split())
        word_counts.append(word_count)

print("Số tài liệu:", len(word_counts))
print("Ít từ nhất:", min(word_counts))
print("Độ dài trung vị:", median(word_counts))
print("Độ dài trung bình:", round(mean(word_counts)))
print("Nhiều từ nhất:", max(word_counts))