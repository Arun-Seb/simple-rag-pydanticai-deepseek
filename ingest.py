"""
ingest.py — Build the knowledge base (run this ONCE per document set).

Pipeline:  PDF  ->  text  ->  chunks  ->  embeddings  ->  ChromaDB

This is the "write" side of the RAG system. It reads your documents, splits
them into small pieces, turns each piece into a vector, and stores everything
in a local ChromaDB database on disk. The query side (rag.py / app.py) only
ever *reads* that database, so you run this again only when your documents change.

Usage:
    python ingest.py data/my_document.pdf
    python ingest.py data/my_document.pdf --chunk-size 500 --overlap 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_pdf(path: str) -> str:
    """Stage 1 — read a PDF and return all its text as one string."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Stage 2 — split text into overlapping character windows.

    The overlap repeats the last `overlap` characters of one chunk at the start
    of the next, so a sentence split across a boundary still survives whole in
    at least one chunk.
    """
    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)  # guard against overlap >= chunk_size
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def build_index(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> None:
    source_name = Path(pdf_path).name

    print(f"[1/4] Loading {pdf_path} ...")
    text = load_pdf(pdf_path)
    print(f"      {len(text):,} characters read")

    print(f"[2/4] Chunking (size={chunk_size}, overlap={overlap}) ...")
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"      {len(chunks)} chunks")

    print(f"[3/4] Embedding with {EMBEDDING_MODEL} (first run downloads the model) ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vectors = [v.tolist() for v in embedder.encode(chunks, normalize_embeddings=True)]

    print("[4/4] Storing in ChromaDB ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Rebuild cleanly so re-running doesn't create duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": source_name} for _ in chunks],
    )
    print(f"Done. Stored {collection.count()} chunks in '{CHROMA_PATH}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a ChromaDB index from a PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF, e.g. data/my_document.pdf")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    build_index(args.pdf_path, args.chunk_size, args.overlap)
