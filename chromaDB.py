import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="hvac_docs",
    metadata={"hnsw:space": "cosine"}  
)
def store_in_database(chunks: list[str], embeddings: list[list[float]], pdf_name: str):        
    import chromadb
    
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

    client = chromadb.HttpClient(
        host="localhost", 
        port=8000, 
        tenant="avegen_assignment", 
        database="knowledge_base"
    )

    collection = client.get_or_create_collection(
        name="hvac_manuals",
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"pdf_name": pdf_name, "text": chunk} for chunk in chunks]

    # CLEAN EMBEDDINGS: Convert numpy floats to native Python floats
    clean_embeddings = [[float(val) for val in emb] for emb in embeddings]

    collection.add(
        ids=ids,
        embeddings=clean_embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    return True
