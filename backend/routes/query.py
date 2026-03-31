from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ingestion.embedder import embed_query
from retrieval.vector_store import search_vectors
from retrieval.bm25_store import search_bm25
from retrieval.reranker import rerank_chunks
from generation.answer import generate_answer

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Step 1: Embed query
        query_embedding = embed_query(request.question)

        # Step 2: Hybrid search
        vector_results = search_vectors(query_embedding, top_k=10)
        bm25_results = search_bm25(request.question, top_k=10)

        # Step 3: Merge results
        # Convert vector results to same format as bm25
        from langchain_core.documents import Document
        merged = []

        for match in vector_results:
            merged.append(Document(
                page_content=match.metadata["text"],
                metadata={
                    "filename": match.metadata.get("filename", "unknown"),
                    "page": match.metadata.get("page", 0)
                }
            ))

        merged.extend(bm25_results)

        # Deduplicate by text content
        seen = set()
        unique = []
        for doc in merged:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)

        # Step 4: Rerank
        reranked = rerank_chunks(request.question, unique, top_n=5)

        # Step 5: Generate answer
        result = generate_answer(request.question, reranked)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))