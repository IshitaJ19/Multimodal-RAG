# Multimodal-RAG

This project creates a simple RAG application for multimodal data, particularly PDFs that may contain text, tables and images.

## Models used
- Llama3.2
- Gemma3:4b

## How to run?

1. Clone the repository and cd into it
2. Download and install [Ollama](https://ollama.com/)
2. Pull the manifests for Ollama models required in this project by running in the terminal:
    - `ollama pull llama3.2`
    - `ollama pull gemma3`
2. Install dependencies:
    - poppler 
    - tesseract

    In MacOS, run `brew install poppler tesseract`

2. Install `uv` using `pip install uv`
2. Create a virtual environment using `uv venv`
3. Activate the virtual environment by running `source .venv/bin/activate`
4. Install the dependencies by running `pip install -r requirements.txt`
5. Run the app using `python app.py`

