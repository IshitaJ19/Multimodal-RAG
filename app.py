import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

from scripts.data_loader import *
from scripts.vector_db import *
from scripts.rag import *


ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
chroma_host = os.getenv("CHROMA_DB_HOST", "http://localhost:8000")


embeddings = OllamaEmbeddings(base_url=ollama_url, model="llama3.2:latest")

client = chromadb.HttpClient(
    host="host.docker.internal",  # e.g., "localhost" or "your-server.com"
    port=8000, 
    settings=Settings(allow_reset=True)
)

vector_store = Chroma(
    client=client,
    collection_name="rag",
    embedding_function=embeddings
)

model = OllamaLLM(base_url=ollama_url, model="gemma3:12b")


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)
print('file uploaded')

if uploaded_file:
    upload_pdf(uploaded_file)
    text = load_pdf(pdfs_dir / uploaded_file.name, figures_dir, model)
    chunked_texts = split_text(text)
    index_docs(chunked_texts, vector_store)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_docs = retrieve_docs(question, vector_store)
        answer = answer_question(question, related_docs, model)
        st.chat_message("assistant").write(answer)
