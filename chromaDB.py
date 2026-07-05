import chromadb
# Persistent client — saves data to ./chroma_db folder on disk
client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection (like a table in SQL)
collection = client.get_or_create_collection(
    name="hvac_docs",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

def store_in_database( chunks: list[str], embeddings: list[list[float]], pdf_name: str):        
    admin_client = chromadb.AdminClient(chromadb.config.Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_host="localhost",
        chroma_server_http_port="8000"
    ))
    
    try:
        admin_client.create_tenant("avegen_assignment")
    except Exception:
        pass # Tenant already exists
        
    try:
        admin_client.create_database("knowledge_base", tenant="avegen_assignment")
    except Exception:
        pass # Database already exists

    # 2. Connect to our specific tenant and database
    client = chromadb.HttpClient(
        host="localhost", 
        port=8000, 
        tenant="avegen_assignment", 
        database="knowledge_base"
    )

    # 3. Create or get the collection
    collection = client.get_or_create_collection(
        name="hvac_manuals",
        metadata={"hnsw:space": "cosine"} # Cosine similarity for semantic search
    )

    # 4. Prepare data and insert
    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"pdf_name": pdf_name, "text": chunk} for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    return True