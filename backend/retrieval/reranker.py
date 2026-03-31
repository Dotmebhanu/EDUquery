import cohere
from config import COHERE_API_KEY

co = cohere.Client(COHERE_API_KEY)

def rerank_chunks(query: str, chunks, top_n: int = 5):
    if not chunks:
        return []

    documents = [chunk.page_content for chunk in chunks]

    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-english-v3.0"
    )

    # Return reranked chunks in order
    reranked = []
    for result in results.results:
        chunk = chunks[result.index]
        reranked.append({
            "text": chunk.page_content,
            "filename": chunk.metadata.get("filename", "unknown"),
            "page": chunk.metadata.get("page", 0),
            "relevance_score": result.relevance_score
        })

    return reranked