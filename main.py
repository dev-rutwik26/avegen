from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from Ai_Service.AI_service import AI_Service

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
    
    # Process if the uploaded file is a PDF
    if file.content_type == "application/pdf":
        # 1. Extract text
        extracted_text = ai_service.pdf_text_extractor(file_path)
        
        # 2. Chunk text
        if extracted_text.strip():
            chunks = ai_service.pdf_chunker(extracted_text)
            
            # 3. Create embeddings for each chunk
            for chunk in chunks:
                embedding = ai_service.embed_text(chunk)
                embeddings.append(embedding)
                
            # TODO: Here is where you will add these chunks and embeddings to ChromaDB

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "content_type": file.content_type,
        "chunk_count": len(chunks),
        "embeddings_created": len(embeddings)
    }


