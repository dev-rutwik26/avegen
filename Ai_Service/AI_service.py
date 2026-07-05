from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re

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
        """
        Removes unnecessary noise, extra whitespaces, and unprintable characters from text.
        """
        # Remove non-printable characters (keep standard ASCII and common punctuation)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove standalone page numbers (e.g., a line with just "12" or "- 12 -")
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
            length_function=len
        )
        return self.text_splitter.split_text(text)

    def embed_text(self, text_chunk: str) -> list[float]:
        genai.configure(api_key=os.environ.get("GOOGLE_GEMINI_API_KEY"))
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text_chunk,
            task_type="retrieval_document"
        )
        return result["embedding"]

    def query_database(self, query):
        """
        Left empty as requested.
        """
        pass
