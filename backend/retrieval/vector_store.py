from pinecone import Pinecone
from config import PINECONE_API_KEY,PINECONE_INDEX_NAME
from ingestion.embedder import embed_texts
import uuid


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def store_chunks(chunks):
    vectors = []
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_texts(texts)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "filename": chunk.metadata.get("filename", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "chunk_index": chunk.metadata.get("chunk_index", i)
            }
        })

    # Upsert in batches of 100 (Pinecone limit). upserts only 100 vectors as batch 
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    return len(vectors)

def search_vectors(query_embedding: list[float], top_k: int = 10):
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches
