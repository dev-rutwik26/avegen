from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
import cohere

# Load variables from .env file into the environment
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

    co = cohere.ClientV2(
        api_key=os.environ.get("COHERE_API_KEY")
    ) 
    def embed_batch(self, text_chunks: list[str]) -> list[list[float]]:
        response = self.co.embed(
            texts=text_chunks,
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"]
        )
        
        return response.embeddings.float


    def query_database(self, query):
        """
        Left empty as requested.
        """
        pass
