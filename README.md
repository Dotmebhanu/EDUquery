# EduQuery — AI Exam Prep Assistant

An AI-powered backend system for engineering students to upload study material and get **accurate, citation-based answers** using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 What It Does

- Upload your notes, textbooks, or past papers (PDF)
- Ask any question — get answers with **source citations** from your uploaded documents
- Uses **hybrid retrieval (BM25 + vector search) + Cohere reranking** for high-quality results
- Evaluates pipeline quality using **RAGAS metrics**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| RAG Framework | LangChain |
| Vector Database | Pinecone |
| Hybrid Retrieval | BM25 + Vector Search |
| Reranking | Cohere Rerank API |
| Evaluation | RAGAS |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
EDUquery/
└── backend/
    ├── main.py                    # FastAPI app entry point
    ├── config.py                  # Configuration and environment variables
    ├── requirements.txt           # Python dependencies
    │
    ├── ingestion/
    │   ├── chunker.py             # PDF chunking logic
    │   ├── embedder.py            # Embedding generation
    │   └── loader.py              # PDF loading and preprocessing
    │
    ├── retrieval/
    │   ├── vector_store.py        # Pinecone vector store operations
    │   ├── bm25_store.py          # BM25 keyword retrieval
    │   └── reranker.py            # Cohere reranking logic
    │
    ├── generation/
    │   └── answer.py              # LLM answer generation with citations
    │
    ├── routes/
    │   ├── upload.py              # /upload endpoint
    │   └── query.py               # /query endpoint
    │
    └── evaluation/
        ├── ragas_eval.py          # RAGAS evaluation pipeline
        ├── test_dataset.py        # Test dataset generation
        └── evaluation_results.txt # Sample evaluation output
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Dotmebhanu/EDUquery.git
cd EDUquery/backend
```

### 2. Create a virtual environment
```bash
python -m venv myvenv
source myvenv/bin/activate   # On Windows: myvenv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the `backend/` folder:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_environment
COHERE_API_KEY=your_cohere_key
```

### 5. Run the server
```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload a PDF document |
| POST | `/query` | Ask a question, get a cited answer |

---

## 🔍 How the RAG Pipeline Works

1. **Ingestion** — PDF is loaded, chunked, and embedded via `ingestion/`
2. **Storage** — Embeddings stored in Pinecone; BM25 index built in parallel
3. **Retrieval** — Hybrid search (vector + BM25) fetches top-k relevant chunks
4. **Reranking** — Cohere Rerank re-orders results for higher precision
5. **Generation** — LLM generates a grounded answer with source citations
6. **Evaluation** — RAGAS measures faithfulness, answer relevance, and context recall

---

## 📊 Evaluation (RAGAS)

The pipeline is evaluated using [RAGAS](https://github.com/explodinggradients/ragas):

- **Faithfulness** — Is the answer grounded in retrieved context?
- **Answer Relevance** — Does the answer address the question?
- **Context Recall** — Were the right chunks retrieved?

Results are logged in `evaluation/evaluation_results.txt`.

---

## 📌 Status

> ✅ Backend fully functional — PDF upload, hybrid retrieval, Cohere reranking, citation-based answers, and RAGAS evaluation are all working.  
> 🚧 Additional features (quiz generator, topic frequency analyzer) are in progress.

---

## 👨‍💻 Author

**Bhanuprakash Reddy**
- 📧 dotmebhanu@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/bhanuprakash-reddy22/))
- 🐙 [GitHub](https://github.com/Dotmebhanu)
