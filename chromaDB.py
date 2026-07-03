import chromadb
# Persistent client — saves data to ./chroma_db folder on disk
client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection (like a table in SQL)
collection = client.get_or_create_collection(
    name="hvac_docs",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)
