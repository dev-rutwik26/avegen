from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import chromadb
from Ai_Service.AI_service import AI_Service
import chromaDB

ai_service = AI_Service()
app = FastAPI(title="HVAC Support Portal API")

# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "HVAC API is running",
    }

# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats")
def get_stats():
    """Returns basic usage statistics for the admin dashboard."""
    try:
        client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            tenant="avegen_assignment",
            database="knowledge_base"
        )
        collection = client.get_collection(name="hvac_manuals")
        count = collection.count()

        # Count unique PDFs from metadata
        results = collection.get(include=["metadatas"])
        pdf_names = set()
        if results and "metadatas" in results:
            for meta in results["metadatas"]:
                if meta and "pdf_name" in meta:
                    pdf_names.add(meta["pdf_name"])

        return {
            "total_chunks": count,
            "total_documents": len(pdf_names),
            "documents": list(pdf_names),
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "documents": [],
            "error": str(e)
        }

# ── File Upload + Ingest ──────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_types = ["application/pdf", "image/png", "image/jpeg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF, PNG, JPEG files allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = ""
    chunks = []
    embeddings = []

    if file.content_type == "application/pdf":
        extracted_text = ai_service.pdf_text_extractor(file_path)
        if extracted_text.strip():
            chunks = ai_service.pdf_chunker(extracted_text)
            embeddings = ai_service.embed_batch(chunks)
            chromaDB.store_in_database(chunks, embeddings, file.filename)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "content_type": file.content_type,
        "chunk_count": len(chunks),
        "embeddings_created": len(embeddings)
    }

# ── Chat (non-streaming) ──────────────────────────────────────────────────────

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    question_embedding = ai_service.embed_batch([request.question])[0]
    relevant_chunks = ai_service.query_database(question_embedding, n_results=3)

    if not relevant_chunks:
        return {"answer": "No relevant documents found. Please upload a manual first!", "sources": []}

    llm_response = ai_service.ask_llm(request.question, relevant_chunks)

    return {
        "question": request.question,
        "answer": llm_response,
        "sources": relevant_chunks
    }

# ── Chat (streaming) ──────────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """Streams the LLM response token-by-token as Server-Sent Events."""
    question_embedding = ai_service.embed_batch([request.question])[0]
    relevant_chunks = ai_service.query_database(question_embedding, n_results=3)

    if not relevant_chunks:
        async def no_docs():
            yield "data: No relevant documents found. Please upload a manual first!\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_docs(), media_type="text/event-stream")

    # Store chunks in a header so the client can read sources (clean newlines to avoid splitting SSE lines)
    sources_header = " ||| ".join([c[:100].replace("\n", " ").replace("\r", " ") for c in relevant_chunks])

    def event_stream():
        # First send metadata about sources
        yield f"data: __SOURCES__{sources_header}__END_SOURCES__\n\n"
        # Then stream LLM tokens
        for chunk in ai_service.ask_llm_stream(request.question, relevant_chunks):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
