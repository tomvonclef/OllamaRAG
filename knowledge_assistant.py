# Full-featured RAG app using ChromaDB, HuggingFace embeddings, and Ollama LLM (Qwen)

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import requests
import umap
import plotly.express as px
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import shutil
import time
import gc
import warnings
import stat
import tempfile

warnings.filterwarnings(
    "ignore", message=".*force_all_finite.*", category=FutureWarning)

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

OLLAMA_MODEL = "qwen3"
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "all-MiniLM-L6-v2"
# Use a temporary directory that we know we can write to
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_store")

# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------


def extract_text_from_pdf(file):
    """Extract all pages as text from uploaded PDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text, len(doc)


def ollama_generate(prompt, model=OLLAMA_MODEL):
    """Send prompt to Ollama API and get LLM response."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]


def plot_embedding_space(embeddings, chunks, retrieved_indices=None):
    """Project embeddings to 2D and visualize with interactive Plotly scatter."""
    reducer = umap.UMAP(n_components=2)
    reduced = reducer.fit_transform(embeddings)

    df_plot = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "text": chunks,
        "retrieved": ["Yes" if i in retrieved_indices else "No" for i in range(len(chunks))]
    })

    fig = px.scatter(
        df_plot,
        x="x", y="y",
        color="retrieved",
        hover_data={"text": True},
        title="📊 Embedding Space (UMAP Projection)",
        labels={"retrieved": "Retrieved Chunk"}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig

# ------------------------------------------------------------------------------
# RAG ENGINE (CHROMA)
# ------------------------------------------------------------------------------


class ChromaRAG:
    def __init__(self, persist_dir=None):
        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        # Create a unique directory for this session to avoid conflicts
        if persist_dir is None:
            self.persist_dir = os.path.join(
                CHROMA_DIR, f"session_{int(time.time())}")
        else:
            self.persist_dir = persist_dir
        self.db = None
        self.chunks = []

    def fix_permissions(self, path):
        """Recursively fix permissions on a directory and its contents."""
        try:
            if os.path.exists(path):
                # Fix directory permissions
                if os.path.isdir(path):
                    os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    for root, dirs, files in os.walk(path):
                        for d in dirs:
                            dir_path = os.path.join(root, d)
                            os.chmod(dir_path, stat.S_IRWXU |
                                     stat.S_IRWXG | stat.S_IRWXO)
                        for f in files:
                            file_path = os.path.join(root, f)
                            os.chmod(file_path, stat.S_IRUSR |
                                     stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
                else:
                    # Fix file permissions
                    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR |
                             stat.S_IRGRP | stat.S_IWGRP)
        except Exception as e:
            print(f"[Warning] Could not fix permissions for {path}: {e}")

    def reset_store(self):
        """Forcefully release ChromaDB and delete persist directory."""
        if self.db:
            try:
                del self.db
                self.db = None
            except Exception as e:
                print(f"[Warning] Error while closing ChromaDB: {e}")

        gc.collect()
        time.sleep(0.5)

        if os.path.exists(self.persist_dir):
            try:
                # First try to fix permissions
                self.fix_permissions(self.persist_dir)

                # Then try to remove
                def handle_remove_readonly(func, path, exc):
                    """Error handler for removing read-only files."""
                    try:
                        os.chmod(path, stat.S_IRWXU |
                                 stat.S_IRWXG | stat.S_IRWXO)
                        func(path)
                    except Exception as e:
                        print(f"[Warning] Could not remove {path}: {e}")

                shutil.rmtree(self.persist_dir, onerror=handle_remove_readonly)
                print(f"[Info] Successfully removed {self.persist_dir}")

            except Exception as e:
                print(f"[Error] Failed to delete {self.persist_dir}: {e}")
                # If we can't delete, create a new unique directory
                self.persist_dir = os.path.join(
                    CHROMA_DIR, f"session_{int(time.time())}")

    def ensure_directory_writable(self):
        """Ensure the persist directory exists and is writable."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_dir, exist_ok=True)

            # Test write permissions
            test_file = os.path.join(self.persist_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)

            return True
        except Exception as e:
            print(f"[Error] Directory not writable: {e}")
            return False

    def build_db(self, chunks):
        """Build a persistent Chroma vector store from text chunks."""
        try:
            # Ensure directory is writable
            if not self.ensure_directory_writable():
                # Fall back to a new temporary directory
                self.persist_dir = os.path.join(
                    tempfile.gettempdir(), f"chroma_fallback_{int(time.time())}")
                os.makedirs(self.persist_dir, exist_ok=True)

            self.chunks = chunks
            documents = [Document(page_content=c, metadata={
                                  "chunk_id": i}) for i, c in enumerate(chunks)]

            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder,
                persist_directory=self.persist_dir
            )

            print(
                f"[Info] ChromaDB created successfully at {self.persist_dir}")

        except Exception as e:
            st.error(f"Failed to create ChromaDB: {e}")
            # Try one more time with a completely fresh directory
            try:
                fallback_dir = os.path.join(
                    tempfile.gettempdir(), f"chroma_emergency_{int(time.time())}")
                os.makedirs(fallback_dir, exist_ok=True)

                self.db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedder,
                    persist_directory=fallback_dir
                )
                self.persist_dir = fallback_dir
                st.success(
                    f"ChromaDB created in fallback location: {fallback_dir}")

            except Exception as e2:
                st.error(f"Complete failure to create ChromaDB: {e2}")
                raise

    def retrieve(self, query, k=4):
        if not self.db:
            raise ValueError(
                "Database not initialized. Call build_db() first.")
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return docs

# ------------------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------------------


st.set_page_config(layout="wide", page_title="🧠 RAG Explorer with ChromaDB")
st.title("🧠 Fantastic RAG Explorer (ChromaDB + Qwen via Ollama)")

st.sidebar.header("⚙️ RAG Settings")
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, step=10)
show_umap = st.sidebar.checkbox("📊 Show Embedding Visualization", value=True)
step_explain = st.sidebar.checkbox("🧩 Explain Steps", value=True)

uploaded_file = st.file_uploader(
    "📄 Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file:
    # STEP 1: Extract and preview text
    if uploaded_file.name.endswith(".pdf"):
        text, pages = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode()
        pages = 1

    st.header("📘 1. Document Ingestion")
    st.markdown(
        f"- **Pages**: `{pages}` | **Words**: `{len(text.split()):,}` | **Chars**: `{len(text):,}`")
    st.text_area("📄 Document Preview",
                 text[:1000] + "..." if len(text) > 1000 else text, height=200)

    # STEP 2: Chunking
    st.header("✂️ 2. Chunking")
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    st.success(f"Split into {len(chunks)} chunks.")
    st.code(chunks[0])

    # STEP 3: Build vector store with Chroma
    st.header("🧬 3. Embedding & Vector Store (ChromaDB)")

    if "rag" not in st.session_state:
        st.session_state.rag = ChromaRAG()

    rag = st.session_state.rag

    with st.spinner("Building ChromaDB vector store..."):
        rag.reset_store()
        rag.build_db(chunks)

    st.success("Chunks embedded and saved in ChromaDB.")

    # Optional: Embedding visual
    if show_umap:
        with st.spinner("Creating embedding visualization..."):
            embedder = SentenceTransformer(EMBED_MODEL)
            vectors = embedder.encode(chunks)
            fig = plot_embedding_space(vectors, chunks, [])
            st.plotly_chart(fig, use_container_width=True)

    # STEP 4: Ask a question
    st.header("🧠 4. Ask Your Question")
    query = st.text_input("Enter your question:")

    if query:
        # STEP 5: Retrieve
        st.subheader("🔍 5. Retrieval")
        try:
            docs = rag.retrieve(query)
            results = [doc.page_content for doc in docs]
            indices = [doc.metadata["chunk_id"] for doc in docs]

            df = pd.DataFrame({
                "Chunk #": indices,
                "Content": results
            })
            st.dataframe(df)

            if show_umap:
                vectors = SentenceTransformer(EMBED_MODEL).encode(chunks)
                fig = plot_embedding_space(vectors, chunks, indices)
                st.plotly_chart(fig, use_container_width=True)

            # STEP 6: Prompt
            st.subheader("📜 6. Prompt Construction")
            context = "\n\n".join(results)
            prompt = f"""Use the following context to answer the question:

Context:
{context}

Question:
{query}

Answer:"""
            st.code(prompt)

            # STEP 7: LLM Answer
            st.subheader("💬 7. LLM Response from Qwen (Ollama)")
            with st.spinner("Calling Qwen via Ollama..."):
                response = ollama_generate(prompt)
            st.success(response)

            if step_explain:
                st.info(
                    "✅ Completed full RAG pipeline using ChromaDB + Ollama Qwen.")

        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            st.info("Try uploading the document again or restart the app.")
else:
    st.info("📂 Please upload a document to begin.")
