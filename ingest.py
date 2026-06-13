"""
ingest.py — PDF Ingestion Pipeline
===================================
Loads a PDF document, splits it into chunks, generates embeddings
using HuggingFace (all-MiniLM-L6-v2), and stores them in a FAISS
vector index for fast retrieval.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Configuration ──────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embeddings():
    """Return a HuggingFace embedding model (runs locally, no API key needed)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_pdf(pdf_path: str):
    """Load a PDF file and return a list of Document objects (one per page)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s) from: {pdf_path}")
    return documents


def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunk(s)")
    return chunks


def create_vectorstore(pdf_path: str) -> FAISS:
    """
    Full ingestion pipeline:
    1. Load PDF
    2. Split into chunks
    3. Create embeddings & build FAISS index
    4. Save index to disk for reuse

    Returns the FAISS vectorstore.
    """
    # Step 1: Load
    documents = load_pdf(pdf_path)

    # Step 2: Split
    chunks = split_documents(documents)

    # Step 3 & 4: Embed + Store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"✅ FAISS index saved to: {VECTORSTORE_DIR}/")

    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a previously saved FAISS vectorstore from disk."""
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            "No vectorstore found. Please upload and process a PDF first."
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    print(f"✅ Loaded FAISS index from: {VECTORSTORE_DIR}/")
    return vectorstore


# ── Standalone usage ───────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    create_vectorstore(pdf_file)
    print("\n🎉 Ingestion complete! You can now run the chatbot.")
