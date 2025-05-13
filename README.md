# Multimodal RAG PDF Chatbot

This project implements a Retrieval Augmented Generation (RAG) chatbot that can answer questions about PDF documents. It leverages multimodal capabilities to understand not only the text within PDFs but also information contained in images and tables. The application is built using Streamlit for the user interface, Ollama for running local Large Language Models (LLMs), and ChromaDB for vector storage.

## Features

-   **PDF Text Extraction**: Extracts textual content from PDF files.
-   **Image Understanding**: Uses an LLM (via Ollama) to describe images found within the PDF.
-   **Table Extraction**: Extracts structured data from tables within the PDF using `img2table` and Tesseract OCR, and converts it to JSON.
-   **RAG Pipeline**:
    -   Chunks extracted text and data.
    -   Generates embeddings using an Ollama model (e.g., `llama3.2`).
    -   Stores embeddings in ChromaDB.
    -   Retrieves relevant document chunks based on user queries.
    -   Generates answers using an Ollama LLM (e.g., `gemma3:12b`) and the retrieved context.
-   **Streamlit UI**: Provides a simple web interface for uploading PDFs and chatting with the RAG system.
-   **Docker Support**: Includes `Dockerfile.rag` and `docker-compose.yml` for easy setup and deployment using Docker.

## Project Structure

```
Multimodal-RAG/
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── data/
│   ├── figures/                # Stores images/tables extracted from PDFs
│   └── pdfs/                   # Stores uploaded PDFs
├── scripts/
│   ├── data_loader.py          # Handles PDF parsing, image/table extraction
│   ├── pull_models.sh          # Script to download necessary Ollama models
│   ├── rag.py                  # Core RAG logic (prompting, answer generation)
│   └── vector_db.py            # Manages vector store interactions (indexing, retrieval)
├── app.py                      # Main Streamlit application
├── Dockerfile.rag              # Dockerfile for the RAG application
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup and Installation

### Prerequisites

1.  **Docker and Docker Compose**: Ensure Docker and Docker Compose are installed on your system.
2.  **Ollama**: Install Ollama from [ollama.com](https://ollama.com/). Make sure it is running.

### Steps

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd Multimodal-RAG
    ```

2.  **Pull Ollama Models:**
    The application uses `llama3.2` for embeddings and `gemma3:12b` for answer generation by default. You can pull these models manually or use the provided script:
    ```bash
    sh scripts/pull_models.sh
    ```
    This script checks if the models exist and pulls them if necessary. Ensure Ollama is running before executing this.

3.  **Environment Variables (Optional but Recommended):**
    The application can be configured using environment variables. You can create a `.env` file in the project root or set them in your environment:
    *   `OLLAMA_BASE_URL`: Base URL for the Ollama service (default: `http://localhost:11434`).
    *   `CHROMA_DB_HOST`: Hostname for the ChromaDB service (default: `http://localhost:8000` when running locally, see `docker-compose.yml` for Docker setup).

    When using Docker Compose, `CHROMA_DB_HOST` is typically set to the service name (e.g., `http://chromadb:8000`). `OLLAMA_BASE_URL` for the application container would be `http://host.docker.internal:11434` if Ollama is running on your host machine.

4.  **Install Python Dependencies (if running without Docker for development):**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

## Running the Application

### Using Docker Compose (Recommended)

This is the easiest way to run the application and its dependencies (ChromaDB).

1.  **Start the services:**
    ```bash
    docker-compose up -d --build
    ```
    This will build the `rag_app` image and start the `rag_app` and `chromadb` services.

2.  **Access the application:**
    Open your web browser and go to `http://localhost:8501` (or the port Streamlit is configured to run on).

3.  **Stopping the services:**
    ```bash
    docker-compose down
    ```

### Running Locally (for Development)

1.  **Start ChromaDB:**
    You can run ChromaDB separately, either as a local process or in Docker:
    ```bash
    # Example using Docker
    docker run -d -p 8000:8000 chromadb/chroma
    ```
    Ensure your `CHROMA_DB_HOST` environment variable (or the default in `app.py`) points to this instance.

2.  **Start Ollama:**
    Ensure Ollama is running and accessible (e.g., `ollama serve` or the Ollama desktop application).

3.  **Run the Streamlit App:**
    Make sure you have installed dependencies and activated your virtual environment.
    ```bash
    streamlit run app.py
    ```
    The application should be accessible at `http://localhost:8501`.

## How it Works

1.  **PDF Upload**: The user uploads a PDF file through the Streamlit interface.
2.  **Data Extraction (`data_loader.py`):**
    *   The PDF is saved locally.
    *   `unstructured` library parses the PDF, extracting text, identifying images and tables.
    *   Images are processed by an Ollama LLM to generate textual descriptions.
    *   Tables are processed by `img2table` and Tesseract OCR to extract structured data, which is then converted to JSON format.
    *   All extracted text, image descriptions, and table data are combined.
3.  **Text Splitting (`data_loader.py`):** The combined text is split into smaller chunks suitable for RAG.
4.  **Indexing (`vector_db.py`):**
    *   Embeddings are generated for each chunk using `llama3.2` (or the configured embedding model) via Ollama.
    *   These embeddings and their corresponding text chunks are stored in ChromaDB.
5.  **Question Answering (`app.py`, `rag.py`, `vector_db.py`):**
    *   The user asks a question in the chat interface.
    *   The question is used to query ChromaDB for relevant document chunks (similarity search).
    *   The retrieved chunks (context) and the original question are passed to an Ollama LLM (`gemma3:12b` or the configured model) using a predefined prompt.
    *   The LLM generates an answer based on the provided context.
    *   The answer is displayed to the user.

## Key Technologies

-   **Streamlit**: For the web application UI.
-   **Langchain**: Framework for building LLM applications, used here for integrating components like LLMs, vector stores, and prompts.
-   **Ollama**: For running local LLMs (e.g., Llama 3.2, Gemma 3).
-   **ChromaDB**: Vector database for storing and retrieving text embeddings.
-   **unstructured**: Library for parsing PDF documents and extracting various elements.
-   **img2table**: For extracting tables from images using OCR.
-   **Tesseract OCR**: Optical Character Recognition engine used by `img2table`.
-   **Docker**: For containerization and simplified deployment.

