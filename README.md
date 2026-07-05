# HVAC Support Portal — Avegen Assignment

An intelligent retrieval-augmented generation (RAG) platform that enables administrators to index HVAC equipment manuals and facility technicians to ask questions about industrial heating/cooling units. The system answers queries using only the uploaded documentation, streaming the response token-by-token alongside source document citations.

---

## 🛠️ Architecture & Technology Stack

- **Frontend:** Streamlit (Clean, professional enterprise-white interface)
- **Backend API:** FastAPI (High performance, supports SSE streaming, file validation, and usage statistics)
- **Vector Database:** ChromaDB (Self-hosted HTTP client with tenant & database level separation)
- **Embeddings:** HuggingFace Serverless (`google/embeddinggemma-300m`)
- **LLM Engine:** HuggingFace Router running `deepseek-ai/DeepSeek-V3-0324` via OpenAI API interface
- **Text Chunking:** LangChain `RecursiveCharacterTextSplitter` configured with hierarchical separators

---

## 📋 Prerequisites

- **Python 3.10+** (Conda environment recommended)
- **Hugging Face Hub API Token** (To access serverless embedding and text-generation models)

---

## 🚀 Setup & Installation

### 1. Clone & Navigate to Repository
Open your terminal and enter the workspace directory:
```bash
cd avegen_assignment
```

### 2. Environment Setup
Create a virtual environment (using Conda or venv) and activate it:
```bash
# Using Conda
conda create -n avegen_assignment python=3.10 -y
conda activate avegen_assignment
```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root of the project:
```ini
HUGGINGFACE_API_KEY=your_hugging_face_token_here
```
*(Replace `your_hugging_face_token_here` with your actual Hugging Face Hub token.)*

---

## 🖥️ Running the Application

To run the entire system, you will need **3 separate terminal windows** active at the same time:

### Terminal 1: ChromaDB Server
Start the ChromaDB vector database locally on port `8000`:
```bash
chroma run --path ./chroma_db --port 8000
```

### Terminal 2: FastAPI Backend
Launch the backend server reload process on port `8000` (FastAPI automatically handles routing and streams LLM answers):
```bash
uvicorn main:app --reload
```

### Terminal 3: Streamlit Frontend
Start the Streamlit portal (it will open automatically in your browser at `http://localhost:8501`):
```bash
streamlit run app.py
```

---

## 👥 Portals & User Roles

### 👤 Customer (Technician View)
- Accessible by selecting **Customer** at the login screen (no login required).
- Enter queries in plain English (e.g., *"How do I resolve Error Code 41?"*).
- Answers stream back **live** token-by-token.
- Expand the **View Source Chunks** drawer to verify exactly where in the manuals the information was sourced.

### 🔐 Administrator View
- Select **Administrator** at the login page.
- **Login Credentials:** 
  - **Username:** `admin`
  - **Password:** *Any password (must not be empty)*
- **Features:**
  - View real-time database stats (Indexed Documents count, Knowledge Chunks count, Indexed Manual filenames).
  - Upload PDF equipment manuals or PNG/JPEG schematics directly into the ChromaDB knowledge base.
