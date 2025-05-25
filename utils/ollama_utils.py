"""Ollama API utilities for OllamaRAG application."""
import requests
from config import OLLAMA_MODEL, OLLAMA_URL


def ollama_generate(prompt, model=OLLAMA_MODEL):
    """Send prompt to Ollama API and get LLM response."""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=150)  # Add timeout
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API({OLLAMA_URL}): {e}")
        raise