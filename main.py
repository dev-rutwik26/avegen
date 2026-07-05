from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from Ai_Service.AI_service import AI_Service
import chromaDB
from pydantic import BaseModel
class QueryRequest(BaseModel):
    question: str
ai_service = AI_Service()
app = FastAPI(title="HVAC Support Portal API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "HVAC API is running",
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


@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    # 1. Turn the user's question into an embedding
    # (We wrap it in a list because our function expects a batch list, then take the 0th result)
    question_embedding = ai_service.embed_batch([request.question])[0]
    
    # 2. Search ChromaDB for the 3 most relevant text chunks
    relevant_chunks = ai_service.query_database(question_embedding, n_results=3)
    
    if not relevant_chunks:
        return {"answer": "No relevant documents found. Please upload a manual first!"}
    
    # 3. Send the chunks and the question to the LLM
    llm_response = ai_service.ask_llm(request.question, relevant_chunks)
    
    return {
        "question": request.question,
        "answer": llm_response,
        "sources": relevant_chunks # So the technician can see where the answer came from
    }


