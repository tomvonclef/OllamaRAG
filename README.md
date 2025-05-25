# RAG Explorer with ChromaDB

## Overview
This is a simple Retrieval-Augmented Generation (RAG) application that utilizes ChromaDB, HuggingFace embeddings, and the Ollama LLM (Qwen). The application allows users to upload documents, extract text, chunk the text, and perform queries to retrieve relevant information. It has been tested on MacOS but should work on other systems.

## Features
- **Document Ingestion**: Upload `.txt` or `.pdf` files to extract text.
- **Chunking**: Split the document into manageable chunks for processing.
- **Embedding & Vector Store**: Use ChromaDB to create a persistent vector store of document embeddings.
- **Interactive Visualization**: Visualize the embedding space using UMAP and Plotly.
- **Question Answering**: Ask questions based on the uploaded document and retrieve relevant chunks.
- **LLM Integration**: Generate answers using the Qwen model via the Ollama API.

## Requirements
- Ollama
- Python 3.x
- Streamlit
- ChromaDB
- PyMuPDF
- Requests
- UMAP
- Plotly
- Sentence Transformers
- Langchain
- Langchain-community

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   uv venv
   uv sync
   ```

3. Ensure that the Ollama API is running locally and the qwen model is installed.

## Usage
1. Run the Streamlit application:
   ```bash
   uv run streamlit run app.py
   ```

2. Upload a document and follow the on-screen instructions to interact with the RAG system.

## Configuration
- **OLLAMA_MODEL**: Set the Ollama model to use (default: `qwen3`).
- **OLLAMA_URL**: Set the URL for the Ollama API.
- **EMBED_MODEL**: Specify the embedding model to use (default: `all-MiniLM-L6-v2`).

## License
This project is licensed under the MIT License.