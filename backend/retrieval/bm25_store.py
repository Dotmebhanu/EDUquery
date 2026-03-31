from rank_bm25 import BM25Okapi
import pickle, os

BM25_PATH = "bm25_store.pkl"

# In-memory store for current session
bm25_data = {
    "bm25": None,
    "chunks": []
}

def index_chunks_bm25(chunks):
    texts = [chunk.page_content for chunk in chunks]
    tokenized = [text.lower().split() for text in texts]
    
    bm25_data["bm25"] = BM25Okapi(tokenized)
    bm25_data["chunks"] = chunks

    # Save to disk so it persists across restarts
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

def search_bm25(query: str, top_k: int = 10):
    # Load from disk if not in memory
    if bm25_data["bm25"] is None and os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            loaded = pickle.load(f)
            bm25_data["bm25"] = loaded["bm25"]
            bm25_data["chunks"] = loaded["chunks"]

    if bm25_data["bm25"] is None:
        return []  # No documents indexed yet

    tokenized_query = query.lower().split()
    scores = bm25_data["bm25"].get_scores(tokenized_query)
    
    # Get top_k indices sorted by score
    top_indices = sorted(
        range(len(scores)), 
        key=lambda i: scores[i], 
        reverse=True
    )[:top_k]

    return [bm25_data["chunks"][i] for i in top_indices]