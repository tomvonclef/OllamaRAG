"""
Configuration settings for the OllamaRAG application.
"""

# Ollama API Configuration
OLLAMA_MODEL = "qwen3"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Embedding Model Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"

# UI Configuration
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SHOW_UMAP = True
