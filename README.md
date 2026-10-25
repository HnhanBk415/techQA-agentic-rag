# TechQA Agentic RAG

Dự án xây dựng và khảo sát các phương pháp Retrieval-Augmented Generation (RAG) trên bộ dữ liệu **NVIDIA TechQA-RAG-Eval**. Mục tiêu là so sánh khả năng tìm tài liệu, tạo câu trả lời có căn cứ và xử lý câu hỏi không có đủ evidence.

## Mục tiêu

Dự án hướng tới so sánh ba cấu hình:

1. **Dense RAG** — truy xuất bằng embedding và cosine similarity.
2. **Hybrid RAG** — kết hợp dense retrieval với BM25, gộp kết quả bằng Reciprocal Rank Fusion (RRF), sau đó rerank.
3. **Agentic RAG** — dùng LangGraph để đánh giá evidence và cho phép tối đa một lần truy xuất hiệu chỉnh.

Các hệ thống sẽ được đánh giá trên cùng tập câu hỏi, cùng model tạo câu trả lời và cùng ngân sách context. Các tiêu chí dự kiến gồm tìm đúng tài liệu, câu trả lời có căn cứ, citation, abstention và độ trễ.


## Cài đặt

Các lệnh sau chạy từ thư mục gốc của repository, trong CMD trên Windows:

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cấu hình Gemini

Tạo file `.env` ở thư mục gốc repository:

```text
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=your-name-model
```


## Chạy Dense RAG với LangGraph và Gemini

```cmd
python -m scripts.retrieval.dense_rag_graph
```

Graph hiện tại có luồng:

```text
START → retrieve → generate → END
```
