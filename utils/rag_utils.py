"""ChromaDB RAG utilities for OllamaRAG application."""
import os
import stat
import errno
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from config import EMBED_MODEL


def handle_remove_readonly(func, path, exc):
    """Error handler for removing read-only files."""
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        # Add write permission to the file
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # Retry the operation
        func(path)


class ChromaRAG:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.db = None
        self.chunks = []

    def build_db(self, chunks):
        """Build a persistent Chroma vector store from text chunks."""
        # Store chunks for later reference
        self.chunks = chunks

        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Store chunk index to help with visualization
            doc = Document(
                page_content=chunk,
                metadata={"chunk_id": i}
            )
            documents.append(doc)

        try:
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder
            )
        except Exception as e:
            print(f"Error building ChromaDB: {e}")

    def retrieve(self, query, k=4):
        """Retrieve relevant chunks for a query."""
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return docs

