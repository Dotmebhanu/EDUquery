from fastapi import APIRouter, UploadFile, File, HTTPException
from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from retrieval.vector_store import store_chunks
from retrieval.bm25_store import index_chunks_bm25

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type
    allowed = [".pdf", ".docx", ".ppt", ".pptx"]
    ext = "." + file.filename.split(".")[-1].lower()
    
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Use PDF, DOCX, or PPT."
        )

    try:
        file_bytes = await file.read()
        
        # Step 1: Load
        documents = load_document(file_bytes, file.filename)
        
        # Step 2: Chunk
        chunks = chunk_documents(documents)
        
        # Step 3: Store in Pinecone
        stored = store_chunks(chunks)
        
        # Step 4: Index for BM25
        index_chunks_bm25(chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_stored": stored
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))