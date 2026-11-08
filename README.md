# TechQA Agentic RAG

Dự án xây dựng và khảo sát các phương pháp Retrieval-Augmented Generation (RAG) trên bộ dữ liệu **NVIDIA TechQA-RAG-Eval**. Mục tiêu là so sánh khả năng tìm tài liệu, tạo câu trả lời có căn cứ và xử lý câu hỏi không có đủ evidence.

## Mục tiêu

Dự án hướng tới so sánh các cấu hình:

1. **Dense RAG** — truy xuất bằng embedding và cosine similarity.
2. **Hybrid RAG** — Kết hợp kết quả Dense và BM25 bằng Reciprocal Rank Fusion (RRF).
3. **Hybrid RAG + Reranker** — Dùng CrossEncoder xếp lại tối đa 20 ứng viên sau RRF, rồi chọn tối đa 5 đoạn
4. **Agentic RAG** — dùng LangGraph để đánh giá evidence và cho phép tối đa một lần truy xuất hiệu chỉnh.

Các hệ thống sẽ được đánh giá trên cùng tập câu hỏi, cùng model tạo câu trả lời và cùng ngân sách context. Các tiêu chí dự kiến gồm tìm đúng tài liệu, câu trả lời có căn cứ, citation, abstention và độ trễ.

## Tổng quan dự án

![Sơ đồ tổng quan TechQA Agentic RAG](./overview_pipeline.drawio.svg)

## Cài đặt

Các lệnh sau chạy từ thư mục gốc của repository, trong CMD trên Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cấu hình môi trường

Nếu chưa có .env, tạo từ file mẫu.

```cmd
copy .env.example .env
```

Cấu hình mẫu trong `.env` có dạng sau.

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
PGPASSWORD=your_password
DATABASE_URL=postgresql://techqa:your_password@127.0.0.1:5433/techqa_rag
```

## Chuẩn bị dữ liệu và tạo chỉ mục

Tải `corpus.zip` vào `data/raw/`. Script đọc trực tiếp file ZIP.

```cmd
if not exist data\raw mkdir data\raw
curl.exe -fL "https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/resolve/0b5bbc84b7f07d6d09d063130e90b716d8d4a32a/corpus.zip" -o data\raw\corpus.zip
```

Chạy theo thứ tự sau. prepare_eval.py tạo tập DEV để prepare_corpus.py kiểm tra kho tài liệu có chứa các tài liệu tham chiếu cần thiết. Dense và BM25 cùng sử dụng các đoạn văn từ chunks.jsonl.

```cmd
python scripts\data\prepare_eval.py
python scripts\data\prepare_corpus.py
python scripts\data\prepare_chunks.py
python -m scripts.indexing.build_dense_index
python -m scripts.indexing.build_bm25_index
```

Kho truy xuất có **28.481 tài liệu**, được chia thành **186.474 đoạn**. Tập **291 câu hỏi DEV** được giữ riêng để đánh giá.

## Khởi động PostgreSQL và nạp embedding

```cmd
docker compose up -d
docker compose exec -T postgres pg_isready -U techqa -d techqa_rag
```

Khi PostgreSQL báo `accepting connections`, tạo bảng và nạp embedding.

```cmd
docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U techqa -d techqa_rag < sql\init.sql
python -m scripts.indexing.sync_pgvector
```

Docker volume giữ lại dữ liệu PostgreSQL khi container dừng. Nếu đã tạo các chỉ mục và nạp embedding, lần sau chỉ cần khởi động PostgreSQL nếu container đang dừng rồi chạy lệnh hỏi đáp.

## Chạy hỏi đáp

Chọn một trong ba lệnh sau.

```cmd
:: Dense RAG
python -m scripts.retrieval.dense_rag_graph

:: Hybrid RAG
python -m scripts.retrieval.hybrid_rag_graph

:: Hybrid RAG + Reranker
python -m scripts.retrieval.hybrid_rag_graph --rerank
```

Khi chương trình hiện `Nhập câu hỏi:`, thử câu sau.

```text
In Netcool/Impact IPL, what happens when Exit() is called inside a User Defined Function?
```

