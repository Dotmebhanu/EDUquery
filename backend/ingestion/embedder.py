from langchain_huggingface import HuggingFaceEmbeddings
embeddings= HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

"""embedding both uploaded doc  and query"""
def embed_texts(texts: list[str])->list[list[float]]:
    return embeddings.embed_documents(texts)

def embed_query(query: list[str])->list[float]:
    return embeddings.embed_query(query)
    
