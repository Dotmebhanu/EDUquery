from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router
from routes.query import router as query_router

app = FastAPI(
    title="EduQuery API",
    description="AI-powered exam prep assistant",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(upload_router)
app.include_router(query_router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "EduQuery API"}