# Full-featured RAG app using ChromaDB, HuggingFace embeddings, and Ollama LLM (DeepSeek-R1)

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import requests
import umap
import plotly.express as px
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import shutil
import time
import gc

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_store"

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

def ollama_generate(prompt, model="deepseek-r1:7b"):
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
    reducer = umap.UMAP(n_components=2, random_state=42)
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
    def __init__(self, persist_dir=CHROMA_DIR):
        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.persist_dir = persist_dir
        self.db = None
        self.chunks = []

    def reset_store(self):
        """Forcefully release ChromaDB and delete persist directory on Windows."""
        if self.db:
            try:
                self.db.persist()
                del self.db
                self.db = None
            except Exception as e:
                print(f"[Warning] Error while closing ChromaDB: {e}")

        gc.collect()
        time.sleep(0.5)

        if os.path.exists(self.persist_dir):
            def handle_remove_readonly(func, path, _):
                os.chmod(path, 0o666)
                func(path)
            try:
                shutil.rmtree(self.persist_dir, onerror=handle_remove_readonly)
            except Exception as e:
                print(f"[Error] Failed to delete {self.persist_dir}: {e}")

    def build_db(self, chunks):
        """Build a persistent Chroma vector store from text chunks."""
        self.chunks = chunks
        documents = [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(chunks)]

        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            persist_directory=self.persist_dir
        )
        self.db.persist()

    def retrieve(self, query, k=4):
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        return docs

# ------------------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="🧠 RAG Explorer with ChromaDB")
st.title("🧠 Fantastic RAG Explorer (ChromaDB + DeepSeek via Ollama)")

st.sidebar.header("⚙️ RAG Settings")
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, step=10)
show_umap = st.sidebar.checkbox("📊 Show Embedding Visualization", value=True)
step_explain = st.sidebar.checkbox("🧩 Explain Steps", value=True)

uploaded_file = st.file_uploader("📄 Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file:
    # STEP 1: Extract and preview text
    if uploaded_file.name.endswith(".pdf"):
        text, pages = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode()
        pages = 1

    st.header("📘 1. Document Ingestion")
    st.markdown(f"- **Pages**: `{pages}` | **Words**: `{len(text.split()):,}` | **Chars**: `{len(text):,}`")
    st.text_area("📄 Document Preview", text[:1000] + "..." if len(text) > 1000 else text, height=200)

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
    rag.reset_store()
    rag.build_db(chunks)
    st.success("Chunks embedded and saved in ChromaDB.")

    # Optional: Embedding visual
    if show_umap:
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
        st.subheader("💬 7. LLM Response from DeepSeek (Ollama)")
        with st.spinner("Calling DeepSeek-R1 via Ollama..."):
            response = ollama_generate(prompt)
        st.success(response)

        if step_explain:
            st.info("✅ Completed full RAG pipeline using ChromaDB + Ollama DeepSeek.")
else:
    st.info("📂 Please upload a document to begin.")

