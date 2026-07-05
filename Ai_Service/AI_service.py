from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
from huggingface_hub import InferenceClient
import chromadb
from openai import OpenAI
load_dotenv()

class AI_Service:
    def __init__(self):
        # Configure text splitter: 500 char chunks with 50 overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=500,
            length_function=len
        )

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'(?m)^\s*-?\s*\d+\s*-?\s*$', '', text)
        
        return text.strip()

    def pdf_text_extractor(self, pdf_path: str) -> str:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        cleaned_text = self.clean_text(full_text)
        return cleaned_text


    def pdf_chunker(self, text: str) -> list[str]:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", ".", "?", "!", ",", " ", ""]
        )
        return self.text_splitter.split_text(text)

    hf_client = InferenceClient(
        provider="hf-inference",
        api_key=os.environ.get("HUGGINGFACE_API_KEY"),
    )
    client = OpenAI(
        base_url= "https://router.huggingface.co/v1",
        api_key=os.environ.get("HUGGINGFACE_API_KEY"),
    )
    def embed_batch(self, text_chunks: list[str]) -> list[list[float]]:
        embeddings = []
        for chunk in text_chunks:
            result = self.hf_client.feature_extraction(
                chunk,
                model="google/embeddinggemma-300m",
            )
            # Append the ENTIRE result (converted to standard floats), not just result[0]
            embeddings.append([float(val) for val in result])
        return embeddings


    def query_database(self,query_embedding: list[float], n_results: int = 3) -> list[str]:
        client = chromadb.HttpClient(
            host="localhost", 
            port=8000, 
            tenant="avegen_assignment", 
            database="knowledge_base"
        )
        
        collection = client.get_collection(name="hvac_manuals")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Return the text of the matched chunks
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        return []

    def ask_llm(self, user_query: str, context_chunks: list[str]) -> str:
            context_text = "\n\n---\n\n".join(context_chunks)
            
            prompt = f"""You are a helpful industrial HVAC technician assistant. 
    Answer the user's question using ONLY the provided context below. If the answer is not in the context, say "I don't have enough information to answer that."

    Context:
    {context_text}

    Question: {user_query}
    Answer:"""
            response = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3.2-Exp:novita", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000
            )

            return response.choices[0].message.content


