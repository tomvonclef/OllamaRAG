# Full-featured RAG app using ChromaDB, HuggingFace embeddings, and Ollama LLM (Qwen)

import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import warnings
from config import EMBED_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_SHOW_UMAP
import sys

# Import utilities from separate files
from utils.pdf_utils import extract_text_from_pdf
from utils.ollama_utils import ollama_generate
from utils.visualization_utils import plot_embedding_space
from utils.rag_utils import ChromaRAG

# Kill some annoying warnings
if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']
warnings.filterwarnings(
    "ignore", message=".*force_all_finite.*", category=FutureWarning)

# ------------------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------------------


st.set_page_config(layout="wide", page_title="🧠 RAG Explorer with ChromaDB")
st.title("🧠 Fantastic RAG Explorer (ChromaDB + Qwen via Ollama)")

st.sidebar.header("⚙️ RAG Settings")
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, DEFAULT_CHUNK_SIZE, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, DEFAULT_CHUNK_OVERLAP, step=10)
show_umap = st.sidebar.checkbox("📊 Show Embedding Visualization", value=DEFAULT_SHOW_UMAP)
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
                 text[:20000] + "..." if len(text) > 20000 else text, height=200)

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

    with st.spinner("Building vector store..."):
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
