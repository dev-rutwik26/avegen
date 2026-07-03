from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

class AI_Service:
    def __init__(self):
        # Configure text splitter: 500 char chunks with 50 overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

    def pdf_text_extractor(self, pdf_path: str) -> str:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        return full_text

    def pdf_chunker(self, text: str) -> list[str]:
        # Actually split the text and return the list of chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
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
