
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma

from scripts.data_loader import *
from scripts.vector_db import *
from scripts.rag import *


embeddings = OllamaEmbeddings(model="llama3:latest")
vector_store = Chroma(
    collection_name="data",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)

model = OllamaLLM(model="gemma3:latest")


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

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

